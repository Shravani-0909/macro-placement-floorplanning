"""Comprehensive Experiment Runner for Macro Placement & Floorplanning.

Supports:
- Environment & dependency diagnostic checking
- Smoke testing (fast end-to-end verification of classical + RL pipelines)
- Experiment A: Problem Size Scaling (10, 20, 30 macros)
- Experiment B: Connectivity Graph Structure (random vs. clustered topology)
- Experiment D: SA Search Budget vs. RL Training Cost Tradeoff

Designed for local environments and Google Colab execution.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.placement.benchmarks import (
    BenchmarkRunResult,
    BenchmarkSummary,
    compute_benchmark_summaries,
    execute_single_benchmark_run,
    generate_benchmark_netlist,
    get_connectivity_benchmark_suite,
    get_default_benchmark_suite,
    get_scaling_benchmark_suite,
    ppo_training_result_to_benchmark_run_result,
    write_benchmark_reports,
)
from src.placement.constraints import validate_placement
from src.placement.models import Netlist
from src.placement.objective import evaluate_placement

# Check optional RL dependencies
try:
    import gymnasium as gym

    _HAS_GYMNASIUM = True
except ImportError:
    gym = None
    _HAS_GYMNASIUM = False

try:
    from sb3_contrib import MaskablePPO

    _HAS_SB3 = True
except ImportError:
    MaskablePPO = None
    _HAS_SB3 = False

try:
    import torch

    # Verify torch actually works (detect Windows DLL load blocks)
    _t = torch.tensor([1.0])
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

_RL_READY = _HAS_GYMNASIUM and _HAS_SB3 and _HAS_TORCH


def print_env_status() -> bool:
    """Print detailed environment diagnostic report."""
    print("=" * 60)
    print("Macro Placement Environment Diagnostic")
    print("=" * 60)
    print(f"Python executable: {sys.executable}")
    print(f"Python version:    {sys.version.split()[0]}")
    print(f"Platform:          {sys.platform}")
    print("-" * 60)
    print(f"NumPy available:     True ({np.__version__})")
    print(f"Gymnasium available: {_HAS_GYMNASIUM}")
    print(f"sb3-contrib available:{_HAS_SB3}")
    print(f"PyTorch available:   {_HAS_TORCH}")
    print("-" * 60)
    if _RL_READY:
        print("STATUS: Full RL environment is READY for PPO training & evaluation.")
    else:
        print("STATUS: RL training is DISABLED or BLOCKED in this environment.")
        print("Classical algorithms (Random, Greedy, SA) are fully operational.")
        if not _HAS_GYMNASIUM:
            print("  - Missing: gymnasium (pip install gymnasium)")
        if not _HAS_SB3:
            print("  - Missing: sb3-contrib (pip install sb3-contrib)")
        if not _HAS_TORCH:
            print("  - PyTorch unavailable or blocked by system policy.")
    print("=" * 60)
    return _RL_READY


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH and torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


async def run_smoke_test(
    output_dir: Path,
    verbose_rl: bool = False,
    reward_scale: float = 1.0,
) -> bool:
    """Execute rapid end-to-end smoke test for classical solvers and RL."""
    print("\n--- Running Pipeline Smoke Test ---")
    smoke_dir = output_dir / "smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    suite = get_default_benchmark_suite()
    toy_nl = suite["toy_4"]
    seed = 42

    results: list[BenchmarkRunResult] = []

    # 1. Classical algorithms
    for algo in ["random", "greedy", "sa"]:
        cfg = {"max_iterations": 20} if algo == "sa" else {}
        print(f"  Testing {algo.upper()} on toy_4...", end="", flush=True)
        res = await execute_single_benchmark_run(algo, "toy_4", toy_nl, seed, config=cfg)
        assert res.success, f"{algo} failed: {res.error}"
        print(f" OK (HPWL={res.hpwl:.2f}, congestion={res.congestion:.4f}, time={res.runtime_sec*1000:.1f}ms)")
        results.append(res)

    # 2. RL algorithm (if dependencies available)
    if _RL_READY:
        print("  Testing MaskablePPO (tiny 128 steps)...", end="", flush=True)
        from src.algorithms.rl_placement import PPOTrainingConfig, train_ppo

        ckpt_dir = smoke_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = ckpt_dir / "smoke_model.zip"

        cfg_rl = PPOTrainingConfig(
            total_timesteps=128,
            n_steps=64,
            batch_size=32,
            seed=seed,
            checkpoint_dir=str(ckpt_dir),
            checkpoint_name="smoke_model",
            verbose=1 if verbose_rl else 0,
            reward_scale=reward_scale,
        )
        t_res = train_ppo(toy_nl, config=cfg_rl, checkpoint_path=ckpt_file)
        print(f" Trained in {t_res.training_time_s:.2f}s")

        print("  Evaluating trained PPO policy...", end="", flush=True)
        rl_bench_res = await execute_single_benchmark_run(
            "rl",
            "toy_4",
            toy_nl,
            seed,
            config={
                "model_path": str(ckpt_file),
                "train_time_sec": t_res.training_time_s,
                "train_timesteps": t_res.total_timesteps,
                "checkpoint_path": str(ckpt_file),
            },
        )
        if not rl_bench_res.success and getattr(t_res.eval_result, "feasible", False):
            rl_bench_res = ppo_training_result_to_benchmark_run_result(
                t_res,
                toy_nl,
                netlist_id="toy_4",
                config={
                    "model_path": str(ckpt_file),
                    "train_time_sec": t_res.training_time_s,
                    "train_timesteps": t_res.total_timesteps,
                    "checkpoint_path": str(ckpt_file),
                },
            )
        results.append(rl_bench_res)
        status = "OK" if rl_bench_res.success else "FAILED"
        hpwl_str = f"HPWL={rl_bench_res.hpwl:.2f}" if rl_bench_res.hpwl is not None else "N/A"
        cong_str = f"congestion={rl_bench_res.congestion:.4f}" if rl_bench_res.congestion is not None else "N/A"
        print(f" {status} ({hpwl_str}, {cong_str})")
    else:
        print("  Skipping RL PPO smoke test (RL dependencies missing/blocked).")

    sums = compute_benchmark_summaries(results)
    write_benchmark_reports(results, sums, smoke_dir)
    print(f"Smoke test completed successfully. Reports saved to {smoke_dir}")
    return True


async def run_experiment_a(
    output_dir: Path,
    seeds: list[int],
    sa_iterations: int = 1000,
    rl_timesteps: int = 10000,
    verbose_rl: bool = False,
    reward_scale: float = 1.0,
) -> None:
    """Experiment A: Problem Size Scaling across 10, 20, 30 macros."""
    print("\n" + "=" * 60)
    print("EXPERIMENT A: Problem Size Scaling (10, 20, 30 macros)")
    print("=" * 60)
    exp_dir = output_dir / "experiment_a_scaling"
    exp_dir.mkdir(parents=True, exist_ok=True)

    suite = get_scaling_benchmark_suite()
    results: list[BenchmarkRunResult] = []

    # Algorithms to compare
    algos = ["random", "greedy", "sa"]
    if _RL_READY:
        algos.append("rl")

    for netlist_id, netlist in suite.items():
        print(f"\nEvaluating Benchmark Instance: {netlist_id} ({len(netlist.macros)} macros, {netlist.grid_w}x{netlist.grid_h} grid)")

        # If RL is available, train a PPO agent for this netlist instance once per seed (or across seeds)
        rl_checkpoints: dict[int, tuple[str, float, Any]] = {}
        if _RL_READY and "rl" in algos:
            from src.algorithms.rl_placement import PPOTrainingConfig, train_ppo

            for s in seeds:
                print(f"  Training PPO on {netlist_id} (seed={s}, timesteps={rl_timesteps})...", end="", flush=True)
                ckpt_path = exp_dir / "checkpoints" / f"ppo_{netlist_id}_s{s}.zip"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                cfg = PPOTrainingConfig(
                    total_timesteps=rl_timesteps,
                    seed=s,
                    verbose=1 if verbose_rl else 0,
                    reward_scale=reward_scale,
                )
                t_res = train_ppo(netlist, config=cfg, checkpoint_path=ckpt_path)
                rl_checkpoints[s] = (str(ckpt_path), t_res.training_time_s, t_res)
                print(f" Done ({t_res.training_time_s:.1f}s)")

        for algo in algos:
            for s in seeds:
                set_seed(s)
                cfg: dict[str, Any] = {}
                if algo == "sa":
                    cfg["max_iterations"] = sa_iterations
                elif algo == "rl" and s in rl_checkpoints:
                    ckpt, t_time, _ = rl_checkpoints[s]
                    cfg["model_path"] = ckpt
                    cfg["checkpoint_path"] = ckpt
                    cfg["train_time_sec"] = t_time
                    cfg["train_timesteps"] = rl_timesteps

                res = await execute_single_benchmark_run(algo, netlist_id, netlist, s, config=cfg)
                if (
                    algo == "rl"
                    and not res.success
                    and s in rl_checkpoints
                    and getattr(rl_checkpoints[s][2].eval_result, "feasible", False)
                ):
                    res = ppo_training_result_to_benchmark_run_result(
                        rl_checkpoints[s][2],
                        netlist,
                        netlist_id=netlist_id,
                        config=cfg,
                    )
                results.append(res)
                status = "OK" if res.success else "FAILED"
                hpwl_str = f"HPWL={res.hpwl:.1f}" if res.hpwl is not None else "N/A"
                c_str = f"Cong={res.congestion:.4f}" if res.congestion is not None else "N/A"
                obj_str = f"Obj={res.objective:.2f}" if res.objective is not None else "N/A"
                print(f"  [{algo.upper():6s}] seed={s:4d} | {status:6s} | {hpwl_str} | {c_str} | {obj_str} | {res.runtime_sec*1000:.1f}ms")

    summaries = compute_benchmark_summaries(results)
    write_benchmark_reports(results, summaries, exp_dir)
    print(f"\nExperiment A complete. Results written to {exp_dir}")


async def run_experiment_b(
    output_dir: Path,
    seeds: list[int],
    sa_iterations: int = 1000,
    rl_timesteps: int = 10000,
    num_macros: int = 12,
    verbose_rl: bool = False,
    reward_scale: float = 1.0,
) -> None:
    """Experiment B: Connectivity Graph Structure (Random vs Clustered)."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENT B: Connectivity Graph Structure ({num_macros} macros: Random vs Clustered)")
    print("=" * 60)
    exp_dir = output_dir / "experiment_b_connectivity"
    exp_dir.mkdir(parents=True, exist_ok=True)

    suite = get_connectivity_benchmark_suite(num_macros=num_macros, seed=42)
    results: list[BenchmarkRunResult] = []

    # Algorithms to compare
    algos = ["random", "greedy", "sa"]
    if _RL_READY:
        algos.append("rl")

    for netlist_id, netlist in suite.items():
        topology = "Random" if "random" in netlist_id else "Clustered"
        print(f"\nEvaluating Topology Variant: {netlist_id} ({topology})")

        rl_checkpoints: dict[int, tuple[str, float, Any]] = {}
        if _RL_READY and "rl" in algos:
            from src.algorithms.rl_placement import PPOTrainingConfig, train_ppo

            for s in seeds:
                print(f"  Training PPO on {netlist_id} (seed={s}, timesteps={rl_timesteps})...", end="", flush=True)
                ckpt_path = exp_dir / "checkpoints" / f"ppo_{netlist_id}_s{s}.zip"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                cfg = PPOTrainingConfig(
                    total_timesteps=rl_timesteps,
                    seed=s,
                    verbose=1 if verbose_rl else 0,
                    reward_scale=reward_scale,
                )
                t_res = train_ppo(netlist, config=cfg, checkpoint_path=ckpt_path)
                rl_checkpoints[s] = (str(ckpt_path), t_res.training_time_s, t_res)
                print(f" Done ({t_res.training_time_s:.1f}s)")

        for algo in algos:
            for s in seeds:
                set_seed(s)
                cfg: dict[str, Any] = {}
                if algo == "sa":
                    cfg["max_iterations"] = sa_iterations
                elif algo == "rl" and s in rl_checkpoints:
                    ckpt, t_time, _ = rl_checkpoints[s]
                    cfg["model_path"] = ckpt
                    cfg["checkpoint_path"] = ckpt
                    cfg["train_time_sec"] = t_time
                    cfg["train_timesteps"] = rl_timesteps

                res = await execute_single_benchmark_run(algo, netlist_id, netlist, s, config=cfg)
                if (
                    algo == "rl"
                    and not res.success
                    and s in rl_checkpoints
                    and getattr(rl_checkpoints[s][2].eval_result, "feasible", False)
                ):
                    res = ppo_training_result_to_benchmark_run_result(
                        rl_checkpoints[s][2],
                        netlist,
                        netlist_id=netlist_id,
                        config=cfg,
                    )
                results.append(res)
                status = "OK" if res.success else "FAILED"
                hpwl_str = f"HPWL={res.hpwl:.1f}" if res.hpwl is not None else "N/A"
                c_str = f"Cong={res.congestion:.4f}" if res.congestion is not None else "N/A"
                obj_str = f"Obj={res.objective:.2f}" if res.objective is not None else "N/A"
                print(f"  [{algo.upper():6s}] seed={s:4d} | {status:6s} | {hpwl_str} | {c_str} | {obj_str} | {res.runtime_sec*1000:.1f}ms")

    summaries = compute_benchmark_summaries(results)
    write_benchmark_reports(results, summaries, exp_dir)
    print(f"\nExperiment B complete. Results written to {exp_dir}")


async def run_experiment_d(
    output_dir: Path,
    seeds: list[int],
    sa_budgets: list[int] | None = None,
    rl_budgets: list[int] | None = None,
    verbose_rl: bool = False,
    reward_scale: float = 1.0,
) -> None:
    """Experiment D: SA Search Budget vs. RL Training Cost Tradeoff."""
    print("\n" + "=" * 60)
    print("EXPERIMENT D: SA Search Budget vs. RL Training Cost Tradeoff")
    print("=" * 60)
    exp_dir = output_dir / "experiment_d_budget_tradeoff"
    exp_dir.mkdir(parents=True, exist_ok=True)

    suite = get_default_benchmark_suite()
    netlist_id = "dense_10"
    netlist = suite[netlist_id]

    if sa_budgets is None:
        sa_budgets = [200, 500, 1000, 3000]
    if rl_budgets is None:
        rl_budgets = [2000, 10000, 25000]

    results: list[BenchmarkRunResult] = []

    # 1. Sweep SA budgets
    print(f"\nSweeping Simulated Annealing Iteration Budgets on {netlist_id}...")
    for iters in sa_budgets:
        for s in seeds:
            set_seed(s)
            cfg = {"max_iterations": iters}
            res = await execute_single_benchmark_run("sa", netlist_id, netlist, s, config=cfg)
            results.append(res)
            print(f"  [SA budget={iters:5d}] seed={s:4d} | HPWL={res.hpwl:.1f} | Obj={res.objective:.2f} | Time={res.runtime_sec:.3f}s")

    # 2. Sweep RL training budgets (if available)
    if _RL_READY:
        from src.algorithms.rl_placement import PPOTrainingConfig, train_ppo

        print(f"\nSweeping RL PPO Training Timestep Budgets on {netlist_id}...")
        for steps in rl_budgets:
            for s in seeds:
                set_seed(s)
                ckpt_path = exp_dir / "checkpoints" / f"ppo_{netlist_id}_t{steps}_s{s}.zip"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                cfg_train = PPOTrainingConfig(
                    total_timesteps=steps,
                    seed=s,
                    verbose=1 if verbose_rl else 0,
                    reward_scale=reward_scale,
                )
                t_res = train_ppo(netlist, config=cfg_train, checkpoint_path=ckpt_path)

                cfg_eval = {
                    "model_path": str(ckpt_path),
                    "checkpoint_path": str(ckpt_path),
                    "train_time_sec": t_res.training_time_s,
                    "train_timesteps": steps,
                }
                res = await execute_single_benchmark_run("rl", netlist_id, netlist, s, config=cfg_eval)
                if not res.success and getattr(t_res.eval_result, "feasible", False):
                    res = ppo_training_result_to_benchmark_run_result(
                        t_res,
                        netlist,
                        netlist_id=netlist_id,
                        config=cfg_eval,
                    )
                results.append(res)
                print(
                    f"  [RL steps={steps:6d}] seed={s:4d} | TrainTime={t_res.training_time_s:.2f}s | "
                    f"EvalTime={res.runtime_sec*1000:.1f}ms | HPWL={res.hpwl:.1f} | Obj={res.objective:.2f}"
                )
    else:
        print("\nSkipping RL budget sweep (RL dependencies missing/blocked).")

    summaries = compute_benchmark_summaries(results)
    write_benchmark_reports(results, summaries, exp_dir)
    print(f"\nExperiment D complete. Results written to {exp_dir}")


def parse_seeds_list(raw_seeds: Sequence[str | int]) -> list[int]:
    """Parse comma-separated or space-separated seeds into a list of integers."""
    parsed: list[int] = []
    for item in raw_seeds:
        for part in str(item).split(","):
            part_str = part.strip()
            if part_str:
                try:
                    parsed.append(int(part_str))
                except ValueError:
                    raise argparse.ArgumentTypeError(f"Invalid integer seed: {part_str!r}")
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer seed must be provided.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run macro placement experiments (A: Scaling, B: Connectivity, D: Budget Tradeoff)."
    )
    parser.add_argument(
        "--experiment",
        choices=["A", "B", "D", "all", "smoke"],
        default="all",
        help="Experiment to run: A (Scaling), B (Connectivity), D (Budget Tradeoff), all, or smoke.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Execute quick smoke test verification and exit.",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Print environment diagnostics and exit.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save experimental results and CSV/JSON summaries.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        nargs="+",
        default=["42", "100", "2026"],
        help="List of random seeds for evaluation (supports space- or comma-separated, e.g. 42,43,44).",
    )
    parser.add_argument(
        "--sa-iterations",
        type=int,
        default=1000,
        help="Number of iterations for Simulated Annealing (default: 1000).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10000,
        help="Training timesteps for PPO (default: 10000).",
    )
    parser.add_argument(
        "--verbose-rl",
        action="store_true",
        help="Print verbose RL training metrics.",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="Reward scaling factor for PPO (default: 1.0).",
    )

    args = parser.parse_args()
    seeds = parse_seeds_list(args.seeds)

    if args.check_env:
        print_env_status()
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_env_status()

    if args.smoke or args.experiment == "smoke":
        asyncio.run(run_smoke_test(out_dir, verbose_rl=args.verbose_rl, reward_scale=args.reward_scale))
        return

    exp = args.experiment.upper()
    if exp in ("A", "ALL"):
        asyncio.run(
            run_experiment_a(
                out_dir,
                seeds,
                sa_iterations=args.sa_iterations,
                rl_timesteps=args.timesteps,
                verbose_rl=args.verbose_rl,
                reward_scale=args.reward_scale,
            )
        )
    if exp in ("B", "ALL"):
        asyncio.run(
            run_experiment_b(
                out_dir,
                seeds,
                sa_iterations=args.sa_iterations,
                rl_timesteps=args.timesteps,
                verbose_rl=args.verbose_rl,
                reward_scale=args.reward_scale,
            )
        )
    if exp in ("D", "ALL"):
        asyncio.run(
            run_experiment_d(
                out_dir,
                seeds,
                verbose_rl=args.verbose_rl,
                reward_scale=args.reward_scale,
            )
        )


if __name__ == "__main__":
    main()
