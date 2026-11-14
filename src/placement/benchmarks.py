"""Benchmark suite and execution framework for evaluating placement algorithms."""

import asyncio
import csv
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .constraints import validate_placement
from .hpwl import calculate_hpwl
from .jobs import ALGORITHM_REGISTRY, JobProgress
from .models import Macro, Net, Netlist
from .objective import evaluate_placement


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Detailed record of a single benchmark execution run."""

    algorithm: str
    config: dict[str, Any]
    netlist_id: str
    grid_w: int
    grid_h: int
    seed: int
    success: bool
    is_legal: bool
    is_complete: bool
    hpwl: float | None
    runtime_sec: float
    error: str | None = None
    congestion: float | None = None
    objective: float | None = None
    eval_time_sec: float | None = None
    train_time_sec: float | None = None
    train_timesteps: int | None = None
    checkpoint_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to serializable dict."""
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregated statistical performance summary for an algorithm on a netlist."""

    algorithm: str
    netlist_id: str
    total_runs: int
    success_count: int
    success_rate: float
    mean_hpwl: float | None
    median_hpwl: float | None
    min_hpwl: float | None
    max_hpwl: float | None
    std_hpwl: float | None
    mean_runtime_sec: float
    median_runtime_sec: float
    mean_congestion: float | None = None
    median_congestion: float | None = None
    std_congestion: float | None = None
    mean_objective: float | None = None
    median_objective: float | None = None
    std_objective: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to serializable dict."""
        return asdict(self)


def get_default_benchmark_suite() -> dict[str, Netlist]:
    """Return standard reproducible synthetic benchmark netlists."""
    # 1. Small toy instance (4 macros, low density)
    toy_4 = Netlist(
        version="1.0",
        grid_w=15,
        grid_h=15,
        macros=(
            Macro(id=0, w=3, h=3),
            Macro(id=1, w=4, h=2),
            Macro(id=2, w=2, h=4),
            Macro(id=3, w=3, h=2),
        ),
        nets=(
            Net(id=0, macro_ids=(0, 1), weight=1.0),
            Net(id=1, macro_ids=(1, 2), weight=1.5),
            Net(id=2, macro_ids=(0, 2, 3), weight=2.0),
        ),
    )

    # 2. Medium instance (8 macros, moderate connectivity)
    medium_8 = Netlist(
        version="1.0",
        grid_w=30,
        grid_h=30,
        macros=(
            Macro(id=0, w=6, h=6),
            Macro(id=1, w=5, h=4),
            Macro(id=2, w=4, h=7),
            Macro(id=3, w=8, h=5),
            Macro(id=4, w=5, h=5),
            Macro(id=5, w=6, h=3),
            Macro(id=6, w=4, h=4),
            Macro(id=7, w=7, h=6),
        ),
        nets=(
            Net(id=0, macro_ids=(0, 1), weight=1.0),
            Net(id=1, macro_ids=(1, 2, 3), weight=2.0),
            Net(id=2, macro_ids=(0, 4), weight=1.0),
            Net(id=3, macro_ids=(4, 5, 6), weight=1.5),
            Net(id=4, macro_ids=(2, 6, 7), weight=2.5),
            Net(id=5, macro_ids=(3, 7), weight=1.0),
        ),
    )

    # 3. Dense instance (10 macros, high area utilization)
    dense_10 = Netlist(
        version="1.0",
        grid_w=24,
        grid_h=24,
        macros=(
            Macro(id=0, w=7, h=7),
            Macro(id=1, w=6, h=8),
            Macro(id=2, w=8, h=5),
            Macro(id=3, w=6, h=6),
            Macro(id=4, w=5, h=6),
            Macro(id=5, w=7, h=4),
            Macro(id=6, w=4, h=7),
            Macro(id=7, w=5, h=5),
            Macro(id=8, w=6, h=5),
            Macro(id=9, w=4, h=4),
        ),
        nets=(
            Net(id=0, macro_ids=(0, 1, 2), weight=2.0),
            Net(id=1, macro_ids=(1, 3), weight=1.0),
            Net(id=2, macro_ids=(2, 4, 5), weight=1.5),
            Net(id=3, macro_ids=(3, 6), weight=1.0),
            Net(id=4, macro_ids=(4, 7, 8), weight=2.0),
            Net(id=5, macro_ids=(5, 8, 9), weight=1.5),
            Net(id=6, macro_ids=(0, 6, 9), weight=2.5),
        ),
    )

    # 4. Scaling instance: 20 macros (36x36 grid)
    scaling_20 = Netlist(
        version="1.0",
        grid_w=36,
        grid_h=36,
        macros=(
            Macro(id=0, w=6, h=5),
            Macro(id=1, w=5, h=6),
            Macro(id=2, w=7, h=5),
            Macro(id=3, w=4, h=7),
            Macro(id=4, w=5, h=5),
            Macro(id=5, w=6, h=6),
            Macro(id=6, w=7, h=4),
            Macro(id=7, w=4, h=6),
            Macro(id=8, w=5, h=4),
            Macro(id=9, w=6, h=4),
            Macro(id=10, w=5, h=7),
            Macro(id=11, w=7, h=6),
            Macro(id=12, w=6, h=5),
            Macro(id=13, w=4, h=5),
            Macro(id=14, w=5, h=6),
            Macro(id=15, w=6, h=7),
            Macro(id=16, w=4, h=4),
            Macro(id=17, w=5, h=5),
            Macro(id=18, w=7, h=5),
            Macro(id=19, w=6, h=6),
        ),
        nets=(
            Net(id=0, macro_ids=(0, 1, 2), weight=2.0),
            Net(id=1, macro_ids=(1, 3), weight=1.0),
            Net(id=2, macro_ids=(2, 4, 5), weight=1.5),
            Net(id=3, macro_ids=(3, 6, 7), weight=1.0),
            Net(id=4, macro_ids=(4, 7, 8), weight=2.0),
            Net(id=5, macro_ids=(5, 8, 9), weight=1.5),
            Net(id=6, macro_ids=(6, 10, 11), weight=2.5),
            Net(id=7, macro_ids=(7, 11, 12), weight=1.0),
            Net(id=8, macro_ids=(8, 12, 13), weight=1.5),
            Net(id=9, macro_ids=(9, 13, 14), weight=2.0),
            Net(id=10, macro_ids=(10, 14, 15), weight=1.0),
            Net(id=11, macro_ids=(11, 15, 16), weight=2.0),
            Net(id=12, macro_ids=(12, 16, 17), weight=1.5),
            Net(id=13, macro_ids=(13, 17, 18), weight=1.0),
            Net(id=14, macro_ids=(14, 18, 19), weight=2.5),
            Net(id=15, macro_ids=(0, 5, 15), weight=1.5),
            Net(id=16, macro_ids=(2, 9, 19), weight=2.0),
            Net(id=17, macro_ids=(3, 10, 17), weight=1.0),
        ),
    )

    # 5. Scaling instance: 30 macros (48x48 grid)
    scaling_30 = Netlist(
        version="1.0",
        grid_w=48,
        grid_h=48,
        macros=(
            Macro(id=0, w=6, h=5),
            Macro(id=1, w=5, h=6),
            Macro(id=2, w=7, h=5),
            Macro(id=3, w=4, h=7),
            Macro(id=4, w=5, h=5),
            Macro(id=5, w=6, h=6),
            Macro(id=6, w=7, h=4),
            Macro(id=7, w=4, h=6),
            Macro(id=8, w=5, h=4),
            Macro(id=9, w=6, h=4),
            Macro(id=10, w=5, h=7),
            Macro(id=11, w=7, h=6),
            Macro(id=12, w=6, h=5),
            Macro(id=13, w=4, h=5),
            Macro(id=14, w=5, h=6),
            Macro(id=15, w=6, h=7),
            Macro(id=16, w=4, h=4),
            Macro(id=17, w=5, h=5),
            Macro(id=18, w=7, h=5),
            Macro(id=19, w=6, h=6),
            Macro(id=20, w=5, h=5),
            Macro(id=21, w=6, h=4),
            Macro(id=22, w=4, h=6),
            Macro(id=23, w=7, h=5),
            Macro(id=24, w=5, h=7),
            Macro(id=25, w=6, h=6),
            Macro(id=26, w=4, h=5),
            Macro(id=27, w=5, h=4),
            Macro(id=28, w=7, h=6),
            Macro(id=29, w=6, h=5),
        ),
        nets=(
            Net(id=0, macro_ids=(0, 1, 2), weight=2.0),
            Net(id=1, macro_ids=(1, 3), weight=1.0),
            Net(id=2, macro_ids=(2, 4, 5), weight=1.5),
            Net(id=3, macro_ids=(3, 6, 7), weight=1.0),
            Net(id=4, macro_ids=(4, 7, 8), weight=2.0),
            Net(id=5, macro_ids=(5, 8, 9), weight=1.5),
            Net(id=6, macro_ids=(6, 10, 11), weight=2.5),
            Net(id=7, macro_ids=(7, 11, 12), weight=1.0),
            Net(id=8, macro_ids=(8, 12, 13), weight=1.5),
            Net(id=9, macro_ids=(9, 13, 14), weight=2.0),
            Net(id=10, macro_ids=(10, 14, 15), weight=1.0),
            Net(id=11, macro_ids=(11, 15, 16), weight=2.0),
            Net(id=12, macro_ids=(12, 16, 17), weight=1.5),
            Net(id=13, macro_ids=(13, 17, 18), weight=1.0),
            Net(id=14, macro_ids=(14, 18, 19), weight=2.5),
            Net(id=15, macro_ids=(15, 19, 20), weight=1.5),
            Net(id=16, macro_ids=(16, 20, 21), weight=2.0),
            Net(id=17, macro_ids=(17, 21, 22), weight=1.0),
            Net(id=18, macro_ids=(18, 22, 23), weight=1.5),
            Net(id=19, macro_ids=(19, 23, 24), weight=2.0),
            Net(id=20, macro_ids=(20, 24, 25), weight=1.0),
            Net(id=21, macro_ids=(21, 25, 26), weight=2.5),
            Net(id=22, macro_ids=(22, 26, 27), weight=1.5),
            Net(id=23, macro_ids=(23, 27, 28), weight=2.0),
            Net(id=24, macro_ids=(24, 28, 29), weight=1.0),
            Net(id=25, macro_ids=(0, 10, 20), weight=2.0),
            Net(id=26, macro_ids=(5, 15, 25), weight=1.5),
            Net(id=27, macro_ids=(9, 19, 29), weight=2.5),
        ),
    )

    return {
        "toy_4": toy_4,
        "medium_8": medium_8,
        "dense_10": dense_10,
        "scaling_20": scaling_20,
        "scaling_30": scaling_30,
    }


def get_scaling_benchmark_suite() -> dict[str, Netlist]:
    """Return benchmark netlists across scaling problem sizes (10, 20, 30 macros) for Experiment A."""
    defaults = get_default_benchmark_suite()
    return {
        "dense_10": defaults["dense_10"],
        "scaling_20": defaults["scaling_20"],
        "scaling_30": defaults["scaling_30"],
    }


def generate_benchmark_netlist(
    num_macros: int = 12,
    grid_w: int = 30,
    grid_h: int = 30,
    connectivity: str = "random",
    num_nets: int | None = None,
    seed: int = 42,
    num_clusters: int = 3,
) -> Netlist:
    """Deterministically generate a synthetic benchmark netlist with controlled connectivity."""
    rng = random.Random(seed)
    n_nets = num_nets if num_nets is not None else int(1.2 * num_macros)
    macros: list[Macro] = []
    for i in range(num_macros):
        w = rng.randint(3, min(7, max(3, grid_w // 3)))
        h = rng.randint(3, min(7, max(3, grid_h // 3)))
        macros.append(Macro(id=i, w=w, h=h))

    nets: list[Net] = []
    if connectivity == "clustered":
        n_clusters = max(1, min(num_clusters, num_macros))
        clusters: list[list[int]] = [[] for _ in range(n_clusters)]
        for i in range(num_macros):
            clusters[i % n_clusters].append(i)
        for nid in range(n_nets):
            eligible = [c for c in clusters if len(c) >= 2]
            if rng.random() < 0.75 and eligible:
                c = rng.choice(eligible)
                k = min(len(c), rng.randint(2, 4))
                mids = tuple(sorted(rng.sample(c, k)))
            elif len(clusters) >= 2:
                c1, c2 = rng.sample(clusters, 2)
                mids = tuple(sorted([rng.choice(c1), rng.choice(c2)]))
            else:
                k = min(num_macros, rng.randint(2, max(2, min(4, num_macros))))
                mids = tuple(sorted(rng.sample(range(num_macros), k)))
            wt = rng.choice([1.0, 1.5, 2.0])
            nets.append(Net(id=nid, macro_ids=mids, weight=wt))
    else:
        for nid in range(n_nets):
            k = rng.randint(2, min(4, num_macros))
            mids = tuple(sorted(rng.sample(range(num_macros), k)))
            wt = rng.choice([1.0, 1.5, 2.0])
            nets.append(Net(id=nid, macro_ids=mids, weight=wt))

    return Netlist(
        version="1.0",
        grid_w=grid_w,
        grid_h=grid_h,
        macros=tuple(macros),
        nets=tuple(nets),
    )


def get_connectivity_benchmark_suite(
    num_macros: int = 12,
    seed: int = 42,
) -> dict[str, Netlist]:
    """Return paired random and clustered netlists with identical macro dimensions for Experiment B."""
    grid_size = max(24, int(math.ceil(math.sqrt(num_macros * 25 * 2.5))))
    nl_rand = generate_benchmark_netlist(
        num_macros=num_macros,
        grid_w=grid_size,
        grid_h=grid_size,
        connectivity="random",
        seed=seed,
    )
    nl_clust = generate_benchmark_netlist(
        num_macros=num_macros,
        grid_w=grid_size,
        grid_h=grid_size,
        connectivity="clustered",
        seed=seed,
    )
    return {
        f"exp_b_{num_macros}_random": nl_rand,
        f"exp_b_{num_macros}_clustered": nl_clust,
    }


def get_generalization_benchmark_suite(
    num_macros: int = 12,
    grid_size: int = 28,
) -> dict[str, Netlist]:
    """Return benchmark netlists for Experiment D (RL Generalization / Robustness).

    Includes:
    1. exp_d_train_random: Training instance (random connectivity, seed=42)
    2. exp_d_test_unseen_inst: In-distribution unseen instance (random connectivity, seed=101)
    3. exp_d_test_unseen_topo: Out-of-distribution unseen topology instance (clustered connectivity, seed=202)

    All three share identical dimensions (12 macros, 28x28 grid) to match frozen RL policy dimensions.
    """
    return {
        "exp_d_train_random": generate_benchmark_netlist(
            num_macros=num_macros,
            grid_w=grid_size,
            grid_h=grid_size,
            connectivity="random",
            seed=42,
        ),
        "exp_d_test_unseen_inst": generate_benchmark_netlist(
            num_macros=num_macros,
            grid_w=grid_size,
            grid_h=grid_size,
            connectivity="random",
            seed=101,
        ),
        "exp_d_test_unseen_topo": generate_benchmark_netlist(
            num_macros=num_macros,
            grid_w=grid_size,
            grid_h=grid_size,
            connectivity="clustered",
            seed=202,
        ),
    }


def ppo_training_result_to_benchmark_run_result(
    training_result: Any,
    netlist: Netlist,
    netlist_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> BenchmarkRunResult:
    """Convert a PPOTrainingResult or PPOEvalResult into a canonical BenchmarkRunResult."""
    cfg = dict(config or {})

    # Extract eval_result if PPOTrainingResult is passed, else assume eval_result directly
    eval_res = getattr(training_result, "eval_result", training_result)
    target_netlist_id = netlist_id or getattr(training_result, "netlist_id", getattr(netlist, "version", "custom"))

    train_time_sec = getattr(training_result, "training_time_s", cfg.get("train_time_sec", None))
    if train_time_sec is not None:
        train_time_sec = float(train_time_sec)

    train_timesteps = getattr(training_result, "total_timesteps", cfg.get("train_timesteps", None))
    if train_timesteps is not None:
        train_timesteps = int(train_timesteps)

    checkpoint_path = getattr(training_result, "checkpoint_path", cfg.get("checkpoint_path", None))
    if checkpoint_path is not None:
        checkpoint_path = str(checkpoint_path)

    eval_time_sec = getattr(eval_res, "runtime_sec", cfg.get("eval_time_sec", None))
    if eval_time_sec is not None:
        eval_time_sec = float(eval_time_sec)
    else:
        eval_time_sec = 0.0

    feasible = bool(getattr(eval_res, "feasible", False))
    hpwl = getattr(eval_res, "hpwl", None)
    congestion = getattr(eval_res, "congestion", None)
    objective = getattr(eval_res, "objective", None)
    error = getattr(eval_res, "error", None) if not feasible else None
    if not feasible and error is None:
        error = "RL placement was infeasible or reached dead end."

    return BenchmarkRunResult(
        algorithm=getattr(eval_res, "algorithm", "rl"),
        config=cfg,
        netlist_id=str(target_netlist_id),
        grid_w=netlist.grid_w,
        grid_h=netlist.grid_h,
        seed=int(getattr(eval_res, "seed", cfg.get("seed", 42))),
        success=feasible,
        is_legal=feasible,
        is_complete=feasible,
        hpwl=float(hpwl) if hpwl is not None else None,
        runtime_sec=eval_time_sec,
        error=error,
        congestion=float(congestion) if congestion is not None else None,
        objective=float(objective) if objective is not None else None,
        eval_time_sec=eval_time_sec,
        train_time_sec=train_time_sec,
        train_timesteps=train_timesteps,
        checkpoint_path=checkpoint_path,
    )


async def execute_single_benchmark_run(
    algorithm: str,
    netlist_id: str,
    netlist: Netlist,
    seed: int,
    config: dict[str, Any] | None = None,
) -> BenchmarkRunResult:
    """Execute a single deterministic algorithm run and evaluate placement legality and HPWL."""
    cfg = config or {}
    eval_time_sec_cfg = float(cfg["eval_time_sec"]) if "eval_time_sec" in cfg and cfg["eval_time_sec"] is not None else None
    train_time_sec = float(cfg["train_time_sec"]) if "train_time_sec" in cfg and cfg["train_time_sec"] is not None else None
    train_timesteps = int(cfg["train_timesteps"]) if "train_timesteps" in cfg and cfg["train_timesteps"] is not None else None
    checkpoint_path = str(cfg["checkpoint_path"]) if "checkpoint_path" in cfg and cfg["checkpoint_path"] is not None else None

    runner = ALGORITHM_REGISTRY.get(algorithm)

    if runner is None:
        return BenchmarkRunResult(
            algorithm=algorithm,
            config=cfg,
            netlist_id=netlist_id,
            grid_w=netlist.grid_w,
            grid_h=netlist.grid_h,
            seed=seed,
            success=False,
            is_legal=False,
            is_complete=False,
            hpwl=None,
            runtime_sec=0.0,
            error=f"Algorithm '{algorithm}' is not registered in ALGORITHM_REGISTRY.",
            eval_time_sec=eval_time_sec_cfg or 0.0,
            train_time_sec=train_time_sec,
            train_timesteps=train_timesteps,
            checkpoint_path=checkpoint_path,
        )

    # Dummy callback for progress
    async def noop_progress(p: JobProgress) -> None:
        pass

    start_time = time.perf_counter()
    try:
        placement = await runner(netlist, seed, cfg, noop_progress)
        elapsed = time.perf_counter() - start_time
        eval_time_sec = eval_time_sec_cfg if eval_time_sec_cfg is not None else elapsed

        report = validate_placement(netlist, placement)
        if not report.is_valid or not report.is_complete:
            return BenchmarkRunResult(
                algorithm=algorithm,
                config=cfg,
                netlist_id=netlist_id,
                grid_w=netlist.grid_w,
                grid_h=netlist.grid_h,
                seed=seed,
                success=False,
                is_legal=report.is_valid,
                is_complete=report.is_complete,
                hpwl=None,
                runtime_sec=elapsed,
                error=f"Illegal or incomplete placement: {list(report.errors)}",
                eval_time_sec=eval_time_sec,
                train_time_sec=train_time_sec,
                train_timesteps=train_timesteps,
                checkpoint_path=checkpoint_path,
            )

        alpha = float(cfg.get("alpha", 1.0))
        beta = float(cfg.get("beta", 1.0))
        eval_res = evaluate_placement(netlist, placement, alpha=alpha, beta=beta)
        hpwl = eval_res.hpwl
        congestion = eval_res.congestion
        objective = eval_res.objective

        return BenchmarkRunResult(
            algorithm=algorithm,
            config=cfg,
            netlist_id=netlist_id,
            grid_w=netlist.grid_w,
            grid_h=netlist.grid_h,
            seed=seed,
            success=True,
            is_legal=True,
            is_complete=True,
            hpwl=hpwl,
            runtime_sec=elapsed,
            error=None,
            congestion=congestion,
            objective=objective,
            eval_time_sec=eval_time_sec,
            train_time_sec=train_time_sec,
            train_timesteps=train_timesteps,
            checkpoint_path=checkpoint_path,
        )

    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start_time
        eval_time_sec = eval_time_sec_cfg if eval_time_sec_cfg is not None else elapsed
        return BenchmarkRunResult(
            algorithm=algorithm,
            config=cfg,
            netlist_id=netlist_id,
            grid_w=netlist.grid_w,
            grid_h=netlist.grid_h,
            seed=seed,
            success=False,
            is_legal=False,
            is_complete=False,
            hpwl=None,
            runtime_sec=elapsed,
            error=str(exc),
            eval_time_sec=eval_time_sec,
            train_time_sec=train_time_sec,
            train_timesteps=train_timesteps,
            checkpoint_path=checkpoint_path,
        )


def compute_benchmark_summaries(
    results: list[BenchmarkRunResult],
) -> list[BenchmarkSummary]:
    """Compute statistics grouped by (algorithm, netlist_id).

    Statistical conventions:
    - Success rate: success_count / total_runs (failed runs are retained in denominator).
    - HPWL statistics: Computed strictly over successful runs with non-null HPWL.
      Failed or incomplete runs with null HPWL are never treated as zero.
    - Standard deviation: Uses sample standard deviation with Bessel's correction
      (N - 1 degrees of freedom) via statistics.stdev.
    """
    grouped: dict[tuple[str, str], list[BenchmarkRunResult]] = {}
    for r in results:
        grouped.setdefault((r.algorithm, r.netlist_id), []).append(r)

    summaries: list[BenchmarkSummary] = []
    for (algo, netlist_id), runs in grouped.items():
        total_runs = len(runs)
        successful_runs = [r for r in runs if r.success and r.hpwl is not None]
        success_count = len(successful_runs)
        success_rate = float(success_count / total_runs) if total_runs > 0 else 0.0

        runtimes = [r.runtime_sec for r in runs]
        mean_runtime = float(statistics.mean(runtimes)) if runtimes else 0.0
        median_runtime = float(statistics.median(runtimes)) if runtimes else 0.0

        if successful_runs:
            hpwls = [float(r.hpwl) for r in successful_runs if r.hpwl is not None]
            congestions = [float(r.congestion) for r in successful_runs if r.congestion is not None]
            objectives = [float(r.objective) for r in successful_runs if r.objective is not None]

            mean_hpwl = float(statistics.mean(hpwls)) if hpwls else None
            median_hpwl = float(statistics.median(hpwls)) if hpwls else None
            min_hpwl = float(min(hpwls)) if hpwls else None
            max_hpwl = float(max(hpwls)) if hpwls else None
            std_hpwl = float(statistics.stdev(hpwls)) if len(hpwls) > 1 else 0.0

            mean_congestion = float(statistics.mean(congestions)) if congestions else None
            median_congestion = float(statistics.median(congestions)) if congestions else None
            std_congestion = float(statistics.stdev(congestions)) if len(congestions) > 1 else 0.0
            mean_objective = float(statistics.mean(objectives)) if objectives else None
            median_objective = float(statistics.median(objectives)) if objectives else None
            std_objective = float(statistics.stdev(objectives)) if len(objectives) > 1 else 0.0
        else:
            mean_hpwl = None
            median_hpwl = None
            min_hpwl = None
            max_hpwl = None
            std_hpwl = None
            mean_congestion = None
            median_congestion = None
            std_congestion = None
            mean_objective = None
            median_objective = None
            std_objective = None

        summaries.append(
            BenchmarkSummary(
                algorithm=algo,
                netlist_id=netlist_id,
                total_runs=total_runs,
                success_count=success_count,
                success_rate=success_rate,
                mean_hpwl=mean_hpwl,
                median_hpwl=median_hpwl,
                min_hpwl=min_hpwl,
                max_hpwl=max_hpwl,
                std_hpwl=std_hpwl,
                mean_runtime_sec=mean_runtime,
                median_runtime_sec=median_runtime,
                mean_congestion=mean_congestion,
                median_congestion=median_congestion,
                std_congestion=std_congestion,
                mean_objective=mean_objective,
                median_objective=median_objective,
                std_objective=std_objective,
            )
        )

    return summaries


def compute_benchmark_summaries_from_csv(
    raw_csv_path: Path | str,
) -> list[BenchmarkSummary]:
    """Compute aggregated statistical summaries directly from a benchmark_raw_results.csv file."""
    csv_p = Path(raw_csv_path)
    if not csv_p.exists():
        raise FileNotFoundError(f"Raw benchmark results CSV not found at {csv_p}")

    results: list[BenchmarkRunResult] = []
    with open(csv_p, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hpwl_val = float(row["hpwl"]) if row.get("hpwl") else None
            cong_val = float(row["congestion"]) if row.get("congestion") else None
            obj_val = float(row["objective"]) if row.get("objective") else None
            eval_t = float(row["eval_time_sec"]) if row.get("eval_time_sec") else None
            train_t = float(row["train_time_sec"]) if row.get("train_time_sec") else None
            train_steps = int(row["train_timesteps"]) if row.get("train_timesteps") else None

            results.append(
                BenchmarkRunResult(
                    algorithm=row["algorithm"],
                    config={},
                    netlist_id=row["netlist_id"],
                    grid_w=int(row["grid_w"]),
                    grid_h=int(row["grid_h"]),
                    seed=int(row["seed"]),
                    success=row["success"].strip().lower() == "true",
                    is_legal=row["is_legal"].strip().lower() == "true",
                    is_complete=row["is_complete"].strip().lower() == "true",
                    hpwl=hpwl_val,
                    runtime_sec=float(row.get("runtime_sec", 0.0)),
                    error=row.get("error") or None,
                    congestion=cong_val,
                    objective=obj_val,
                    eval_time_sec=eval_t,
                    train_time_sec=train_t,
                    train_timesteps=train_steps,
                    checkpoint_path=row.get("checkpoint_path") or None,
                )
            )

    return compute_benchmark_summaries(results)


def write_benchmark_reports(
    results: list[BenchmarkRunResult],
    summaries: list[BenchmarkSummary] | None,
    output_dir: Path | str,
) -> None:
    """Synchronously write benchmark raw records and statistical summaries to CSV and JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Write raw results JSON
    raw_json_path = out_path / "benchmark_raw_results.json"
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    # Write raw results CSV
    raw_csv_path = out_path / "benchmark_raw_results.csv"
    with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm", "netlist_id", "grid_w", "grid_h", "seed",
            "success", "is_legal", "is_complete", "hpwl", "congestion", "objective",
            "runtime_sec", "eval_time_sec", "train_time_sec", "train_timesteps",
            "checkpoint_path", "error",
        ])
        for r in results:
            writer.writerow([
                r.algorithm, r.netlist_id, r.grid_w, r.grid_h, r.seed,
                r.success, r.is_legal, r.is_complete,
                f"{r.hpwl:.4f}" if r.hpwl is not None else "",
                f"{r.congestion:.6f}" if r.congestion is not None else "",
                f"{r.objective:.4f}" if r.objective is not None else "",
                f"{r.runtime_sec:.6f}",
                f"{r.eval_time_sec:.6f}" if r.eval_time_sec is not None else "",
                f"{r.train_time_sec:.6f}" if r.train_time_sec is not None else "",
                str(r.train_timesteps) if r.train_timesteps is not None else "",
                r.checkpoint_path or "",
                r.error or "",
            ])

    # Always compute verified summaries directly from the persisted raw CSV file
    verified_summaries = compute_benchmark_summaries_from_csv(raw_csv_path)

    # Write summary JSON
    summary_json_path = out_path / "benchmark_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in verified_summaries], f, indent=2)

    # Write summary CSV
    summary_csv_path = out_path / "benchmark_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm", "netlist_id", "total_runs", "success_count", "success_rate",
            "mean_hpwl", "median_hpwl", "min_hpwl", "max_hpwl", "std_hpwl",
            "mean_congestion", "median_congestion", "std_congestion",
            "mean_objective", "median_objective", "std_objective",
            "mean_runtime_sec", "median_runtime_sec",
        ])
        for s in verified_summaries:
            writer.writerow([
                s.algorithm, s.netlist_id, s.total_runs, s.success_count, f"{s.success_rate:.2%}",
                f"{s.mean_hpwl:.4f}" if s.mean_hpwl is not None else "",
                f"{s.median_hpwl:.4f}" if s.median_hpwl is not None else "",
                f"{s.min_hpwl:.4f}" if s.min_hpwl is not None else "",
                f"{s.max_hpwl:.4f}" if s.max_hpwl is not None else "",
                f"{s.std_hpwl:.4f}" if s.std_hpwl is not None else "",
                f"{s.mean_congestion:.6f}" if s.mean_congestion is not None else "",
                f"{s.median_congestion:.6f}" if s.median_congestion is not None else "",
                f"{s.std_congestion:.6f}" if s.std_congestion is not None else "",
                f"{s.mean_objective:.4f}" if s.mean_objective is not None else "",
                f"{s.median_objective:.4f}" if s.median_objective is not None else "",
                f"{s.std_objective:.4f}" if s.std_objective is not None else "",
                f"{s.mean_runtime_sec:.6f}", f"{s.median_runtime_sec:.6f}",
            ])


async def run_benchmark_suite(
    netlists: dict[str, Netlist] | None = None,
    algorithms: list[str] | None = None,
    seeds: list[int] | None = None,
    config: dict[str, Any] | None = None,
    output_dir: Path | str | None = None,
) -> tuple[list[BenchmarkRunResult], list[BenchmarkSummary]]:
    """Run full benchmark evaluation matrix across netlists, algorithms, and seeds."""
    suite = netlists if netlists is not None else get_default_benchmark_suite()
    if algorithms is not None:
        algos = algorithms
    else:
        # Default canonical benchmark algorithms (avoiding duplicate runs of alias 'simulated_annealing')
        default_order = ["random", "greedy", "sa"]
        registered = ALGORITHM_REGISTRY.keys()
        algos = [a for a in default_order if a in registered]
        for a in registered:
            if a not in algos and a != "simulated_annealing":
                algos.append(a)
    seed_list = seeds if seeds is not None else [42, 100, 2026]

    raw_results: list[BenchmarkRunResult] = []

    for algo in algos:
        for netlist_id, netlist in suite.items():
            for s in seed_list:
                res = await execute_single_benchmark_run(
                    algorithm=algo,
                    netlist_id=netlist_id,
                    netlist=netlist,
                    seed=s,
                    config=config,
                )
                raw_results.append(res)

    summaries = compute_benchmark_summaries(raw_results)

    if output_dir is not None:
        write_benchmark_reports(raw_results, summaries, output_dir)

    return raw_results, summaries


if __name__ == "__main__":
    out_dir = Path("benchmarks/results")
    print(f"Executing macro placement benchmark suite... Output directory: {out_dir}")
    raw, sums = asyncio.run(run_benchmark_suite(output_dir=out_dir))
    print(f"Completed {len(raw)} benchmark runs across {len(sums)} summaries.")
    for s in sums:
        hpwl_str = f"mean_hpwl={s.mean_hpwl:.2f}" if s.mean_hpwl is not None else "failed"
        print(
            f"[{s.algorithm.upper()}] {s.netlist_id}: success={s.success_rate:.0%} "
            f"({s.success_count}/{s.total_runs}), {hpwl_str}, time={s.mean_runtime_sec*1000:.2f}ms"
        )
