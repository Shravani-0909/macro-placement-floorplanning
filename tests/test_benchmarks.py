"""Tests for benchmark suite execution, metrics calculation, and CSV/JSON reporting."""

import asyncio
import functools
import csv
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

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
    run_benchmark_suite,
    write_benchmark_reports,
)
from src.placement.models import Macro, Net, Netlist


def async_test(func: Callable[..., Any]) -> Callable[..., Any]:
    """Helper decorator running coroutine tests via asyncio.run without extra plugins."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(func(*args, **kwargs))

    return wrapper


def test_default_benchmark_suite_validity() -> None:
    """All default benchmark suite netlists must have valid geometry and connections."""
    suite = get_default_benchmark_suite()
    assert len(suite) >= 3
    assert "toy_4" in suite
    assert "medium_8" in suite
    assert "dense_10" in suite

    for netlist in suite.values():
        assert netlist.grid_w > 0
        assert netlist.grid_h > 0
        macro_ids = {m.id for m in netlist.macros}
        assert len(macro_ids) == len(netlist.macros)

        for macro in netlist.macros:
            assert macro.w <= netlist.grid_w
            assert macro.h <= netlist.grid_h

        for net in netlist.nets:
            assert len(net.macro_ids) >= 2
            for mid in net.macro_ids:
                assert mid in macro_ids


@async_test
async def test_execute_single_benchmark_run_success() -> None:
    """Random baseline completes successfully on a valid netlist."""
    suite = get_default_benchmark_suite()
    netlist = suite["toy_4"]

    result = await execute_single_benchmark_run(
        algorithm="random",
        netlist_id="toy_4",
        netlist=netlist,
        seed=42,
    )

    assert result.success is True
    assert result.is_legal is True
    assert result.is_complete is True
    assert result.hpwl is not None
    assert result.hpwl > 0.0
    assert result.runtime_sec > 0.0
    assert result.error is None


@async_test
async def test_execute_single_benchmark_run_unregistered_algo() -> None:
    """Unregistered algorithms report clear failure without raising unhandled errors."""
    suite = get_default_benchmark_suite()
    netlist = suite["toy_4"]

    result = await execute_single_benchmark_run(
        algorithm="unimplemented_ppo",
        netlist_id="toy_4",
        netlist=netlist,
        seed=42,
    )

    assert result.success is False
    assert result.is_legal is False
    assert result.is_complete is False
    assert result.hpwl is None
    assert "not registered" in (result.error or "")


def test_compute_benchmark_summaries_with_failures() -> None:
    """Summaries include failed runs in success rate and compute accurate stats."""
    runs = [
        BenchmarkRunResult(
            algorithm="test_algo",
            config={},
            netlist_id="net_a",
            grid_w=10,
            grid_h=10,
            seed=1,
            success=True,
            is_legal=True,
            is_complete=True,
            hpwl=20.0,
            runtime_sec=0.01,
        ),
        BenchmarkRunResult(
            algorithm="test_algo",
            config={},
            netlist_id="net_a",
            grid_w=10,
            grid_h=10,
            seed=2,
            success=True,
            is_legal=True,
            is_complete=True,
            hpwl=30.0,
            runtime_sec=0.03,
        ),
        BenchmarkRunResult(
            algorithm="test_algo",
            config={},
            netlist_id="net_a",
            grid_w=10,
            grid_h=10,
            seed=3,
            success=False,
            is_legal=False,
            is_complete=False,
            hpwl=None,
            runtime_sec=0.02,
            error="Failed to place macro",
        ),
    ]

    summaries = compute_benchmark_summaries(runs)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.algorithm == "test_algo"
    assert s.netlist_id == "net_a"
    assert s.total_runs == 3
    assert s.success_count == 2
    assert s.success_rate == pytest.approx(2.0 / 3.0)
    assert s.mean_hpwl == pytest.approx(25.0)
    assert s.median_hpwl == pytest.approx(25.0)
    assert s.min_hpwl == pytest.approx(20.0)
    assert s.max_hpwl == pytest.approx(30.0)
    assert s.mean_runtime_sec == pytest.approx(0.02)


@async_test
async def test_run_benchmark_suite_output_files(tmp_path: Path) -> None:
    """run_benchmark_suite exports CSV and JSON summaries into output directory."""
    mini_suite = {
        "tiny": Netlist(
            version="1.0",
            grid_w=10,
            grid_h=10,
            macros=(Macro(0, 2, 2), Macro(1, 2, 2)),
            nets=(Net(0, (0, 1)),),
        )
    }

    raw, summaries = await run_benchmark_suite(
        netlists=mini_suite,
        algorithms=["random"],
        seeds=[11, 22],
        output_dir=tmp_path,
    )

    assert len(raw) == 2
    assert len(summaries) == 1
    assert summaries[0].success_rate == 1.0

    # Verify generated files
    assert (tmp_path / "benchmark_raw_results.json").exists()
    assert (tmp_path / "benchmark_raw_results.csv").exists()
    assert (tmp_path / "benchmark_summary.json").exists()
    assert (tmp_path / "benchmark_summary.csv").exists()


@async_test
async def test_execute_single_benchmark_run_greedy() -> None:
    """Greedy benchmark execution resolves runner and produces valid result."""
    suite = get_default_benchmark_suite()
    netlist = suite["toy_4"]

    result = await execute_single_benchmark_run(
        algorithm="greedy",
        netlist_id="toy_4",
        netlist=netlist,
        seed=42,
    )

    assert result.success is True
    assert result.is_legal is True
    assert result.is_complete is True
    assert result.hpwl is not None
    assert result.hpwl > 0.0
    assert result.runtime_sec > 0.0
    assert result.error is None


@async_test
async def test_execute_single_benchmark_run_sa() -> None:
    """SA benchmark execution resolves runner and produces valid result."""
    suite = get_default_benchmark_suite()
    netlist = suite["toy_4"]

    result = await execute_single_benchmark_run(
        algorithm="sa",
        netlist_id="toy_4",
        netlist=netlist,
        seed=42,
        config={"max_iterations": 10, "cooling_rate": 0.8},
    )

    assert result.success is True
    assert result.is_legal is True
    assert result.is_complete is True
    assert result.hpwl is not None
    assert result.hpwl > 0.0
    assert result.runtime_sec > 0.0
    assert result.error is None


@async_test
async def test_run_benchmark_suite_all_classical(tmp_path: Path) -> None:
    """Full suite invocation runs random, greedy, and sa across benchmark instances."""
    mini_suite = {
        "tiny": Netlist(
            version="1.0",
            grid_w=10,
            grid_h=10,
            macros=(Macro(0, 2, 2), Macro(1, 2, 2)),
            nets=(Net(0, (0, 1)),),
        )
    }

    raw, summaries = await run_benchmark_suite(
        netlists=mini_suite,
        algorithms=["random", "greedy", "sa"],
        seeds=[42],
        output_dir=tmp_path,
    )

    assert len(raw) == 3
    assert len(summaries) == 3
    assert all(s.success_rate == 1.0 for s in summaries)
    algo_names = {s.algorithm for s in summaries}
    assert algo_names == {"random", "greedy", "sa"}


def test_scaling_benchmark_suite_validity() -> None:
    """Scaling suite must contain dense_10, scaling_20, and scaling_30 with valid geometry."""
    scaling_suite = get_scaling_benchmark_suite()
    assert "dense_10" in scaling_suite
    assert "scaling_20" in scaling_suite
    assert "scaling_30" in scaling_suite

    s20 = scaling_suite["scaling_20"]
    assert len(s20.macros) == 20
    assert s20.grid_w == 36
    assert s20.grid_h == 36
    assert len(s20.nets) == 18

    s30 = scaling_suite["scaling_30"]
    assert len(s30.macros) == 30
    assert s30.grid_w == 48
    assert s30.grid_h == 48
    assert len(s30.nets) == 28


def test_generate_benchmark_netlist_determinism() -> None:
    """generate_benchmark_netlist is deterministic given identical seeds."""
    nl1 = generate_benchmark_netlist(num_macros=10, grid_w=20, grid_h=20, seed=123)
    nl2 = generate_benchmark_netlist(num_macros=10, grid_w=20, grid_h=20, seed=123)
    nl3 = generate_benchmark_netlist(num_macros=10, grid_w=20, grid_h=20, seed=456)

    assert [(m.w, m.h) for m in nl1.macros] == [(m.w, m.h) for m in nl2.macros]
    assert [net.macro_ids for net in nl1.nets] == [net.macro_ids for net in nl2.nets]
    assert [(m.w, m.h) for m in nl1.macros] != [(m.w, m.h) for m in nl3.macros]


def test_connectivity_benchmark_suite_paired_geometry() -> None:
    """Experiment B paired benchmark suite maintains identical macro dimensions across topologies."""
    pair = get_connectivity_benchmark_suite(num_macros=12, seed=42)
    assert "exp_b_12_random" in pair
    assert "exp_b_12_clustered" in pair

    nl_rand = pair["exp_b_12_random"]
    nl_clust = pair["exp_b_12_clustered"]

    assert len(nl_rand.macros) == 12
    assert len(nl_clust.macros) == 12
    assert nl_rand.grid_w == nl_clust.grid_w
    assert nl_rand.grid_h == nl_clust.grid_h

    # Crucial property for Experiment B: identical macro dimensions
    rand_dims = [(m.id, m.w, m.h) for m in nl_rand.macros]
    clust_dims = [(m.id, m.w, m.h) for m in nl_clust.macros]
    assert rand_dims == clust_dims


@async_test
async def test_benchmark_metrics_congestion_and_objective() -> None:
    """Single benchmark execution records HPWL, congestion, objective, and timing metrics."""
    suite = get_default_benchmark_suite()
    netlist = suite["toy_4"]

    cfg = {
        "alpha": 1.0,
        "beta": 0.5,
        "train_time_sec": 12.34,
        "train_timesteps": 5000,
        "checkpoint_path": "models/test.zip",
    }
    result = await execute_single_benchmark_run(
        algorithm="greedy",
        netlist_id="toy_4",
        netlist=netlist,
        seed=42,
        config=cfg,
    )

    assert result.success is True
    assert result.congestion is not None
    assert result.congestion >= 0.0
    assert result.objective is not None
    assert result.objective == pytest.approx(result.hpwl + 0.5 * result.congestion)
    assert result.train_time_sec == pytest.approx(12.34)
    assert result.train_timesteps == 5000
    assert result.checkpoint_path == "models/test.zip"

    # Verify summary computation includes mean_congestion and mean_objective
    summaries = compute_benchmark_summaries([result])
    assert len(summaries) == 1
    s = summaries[0]
    assert s.mean_congestion == pytest.approx(result.congestion)
    assert s.mean_objective == pytest.approx(result.objective)


def test_ppo_training_result_to_benchmark_run_result_success() -> None:
    """Verify that a successful PPOTrainingResult maps all 9 metrics into BenchmarkRunResult."""
    suite = get_default_benchmark_suite()
    netlist = suite["toy_4"]

    mock_eval = SimpleNamespace(
        algorithm="rl",
        feasible=True,
        objective=23.456,
        hpwl=23.123,
        congestion=0.333,
        runtime_sec=0.045,
        seed=42,
    )
    mock_training = SimpleNamespace(
        algorithm="rl",
        netlist_id="toy_4",
        seed=42,
        total_timesteps=1000,
        training_time_s=15.2,
        checkpoint_path="checkpoints/ppo_toy_4.zip",
        eval_result=mock_eval,
    )

    result = ppo_training_result_to_benchmark_run_result(
        mock_training,
        netlist,
        netlist_id="toy_4",
    )

    assert result.algorithm == "rl"
    assert result.netlist_id == "toy_4"
    assert result.grid_w == 15
    assert result.grid_h == 15
    assert result.seed == 42
    assert result.success is True
    assert result.is_legal is True
    assert result.is_complete is True
    assert result.hpwl == pytest.approx(23.123)
    assert result.congestion == pytest.approx(0.333)
    assert result.objective == pytest.approx(23.456)
    assert result.eval_time_sec == pytest.approx(0.045)
    assert result.runtime_sec == pytest.approx(0.045)
    assert result.train_time_sec == pytest.approx(15.2)
    assert result.train_timesteps == 1000
    assert result.checkpoint_path == "checkpoints/ppo_toy_4.zip"
    assert result.error is None


def test_ppo_training_result_to_benchmark_run_result_failure() -> None:
    """Verify that an infeasible/dead-end PPO result preserves metrics and records failure."""
    suite = get_default_benchmark_suite()
    netlist = suite["toy_4"]

    mock_eval = SimpleNamespace(
        algorithm="rl",
        feasible=False,
        objective=None,
        hpwl=None,
        congestion=None,
        runtime_sec=0.012,
        seed=100,
    )
    mock_training = SimpleNamespace(
        algorithm="rl",
        netlist_id="toy_4",
        seed=100,
        total_timesteps=500,
        training_time_s=8.5,
        checkpoint_path="checkpoints/ppo_failed.zip",
        eval_result=mock_eval,
    )

    result = ppo_training_result_to_benchmark_run_result(
        mock_training,
        netlist,
        netlist_id="toy_4",
    )

    assert result.algorithm == "rl"
    assert result.success is False
    assert result.is_legal is False
    assert result.is_complete is False
    assert result.hpwl is None
    assert result.congestion is None
    assert result.objective is None
    assert result.train_time_sec == pytest.approx(8.5)
    assert result.train_timesteps == 500
    assert result.checkpoint_path == "checkpoints/ppo_failed.zip"
    assert result.eval_time_sec == pytest.approx(0.012)
    assert "infeasible" in (result.error or "").lower() or "dead end" in (result.error or "").lower()


def test_write_benchmark_reports_rl_persistence(tmp_path: Path) -> None:
    """write_benchmark_reports persists all RL metrics into CSV and JSON artifacts."""
    rl_run_success = BenchmarkRunResult(
        algorithm="rl",
        config={"alpha": 1.0, "beta": 1.0},
        netlist_id="toy_4",
        grid_w=15,
        grid_h=15,
        seed=42,
        success=True,
        is_legal=True,
        is_complete=True,
        hpwl=25.5,
        congestion=0.15,
        objective=25.65,
        runtime_sec=0.05,
        eval_time_sec=0.05,
        train_time_sec=12.34,
        train_timesteps=2000,
        checkpoint_path="checkpoints/model.zip",
        error=None,
    )
    rl_run_failed = BenchmarkRunResult(
        algorithm="rl",
        config={"alpha": 1.0, "beta": 1.0},
        netlist_id="toy_4",
        grid_w=15,
        grid_h=15,
        seed=99,
        success=False,
        is_legal=False,
        is_complete=False,
        hpwl=None,
        congestion=None,
        objective=None,
        runtime_sec=0.02,
        eval_time_sec=0.02,
        train_time_sec=12.34,
        train_timesteps=2000,
        checkpoint_path="checkpoints/model.zip",
        error="RL placement reached a dead end at step 2/4.",
    )
    classical_run = BenchmarkRunResult(
        algorithm="sa",
        config={},
        netlist_id="toy_4",
        grid_w=15,
        grid_h=15,
        seed=42,
        success=True,
        is_legal=True,
        is_complete=True,
        hpwl=30.0,
        congestion=0.20,
        objective=30.20,
        runtime_sec=0.10,
        eval_time_sec=0.10,
        train_time_sec=None,
        train_timesteps=None,
        checkpoint_path=None,
        error=None,
    )

    results = [classical_run, rl_run_success, rl_run_failed]
    summaries = compute_benchmark_summaries(results)
    write_benchmark_reports(results, summaries, tmp_path)

    # 1. Validate raw CSV
    raw_csv = tmp_path / "benchmark_raw_results.csv"
    assert raw_csv.exists()
    with open(raw_csv, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 3
        rl_rows = [row for row in reader if row["algorithm"] == "rl"]
        assert len(rl_rows) == 2

        # Check successful RL record
        row_s = rl_rows[0]
        assert row_s["success"] == "True"
        assert row_s["is_legal"] == "True"
        assert row_s["is_complete"] == "True"
        assert float(row_s["hpwl"]) == pytest.approx(25.5)
        assert float(row_s["congestion"]) == pytest.approx(0.15)
        assert float(row_s["objective"]) == pytest.approx(25.65)
        assert float(row_s["train_time_sec"]) == pytest.approx(12.34)
        assert int(row_s["train_timesteps"]) == 2000
        assert row_s["checkpoint_path"] == "checkpoints/model.zip"
        assert float(row_s["eval_time_sec"]) == pytest.approx(0.05)
        assert row_s["error"] == ""

        # Check failed RL record
        row_f = rl_rows[1]
        assert row_f["success"] == "False"
        assert row_f["is_legal"] == "False"
        assert row_f["is_complete"] == "False"
        assert row_f["hpwl"] == ""
        assert row_f["congestion"] == ""
        assert row_f["objective"] == ""
        assert float(row_f["train_time_sec"]) == pytest.approx(12.34)
        assert int(row_f["train_timesteps"]) == 2000
        assert row_f["checkpoint_path"] == "checkpoints/model.zip"
        assert "dead end" in row_f["error"]

    # 2. Validate raw JSON
    raw_json = tmp_path / "benchmark_raw_results.json"
    assert raw_json.exists()
    with open(raw_json, mode="r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data) == 3
        rl_data = [d for d in data if d["algorithm"] == "rl"]
        assert len(rl_data) == 2
        assert rl_data[0]["success"] is True
        assert rl_data[0]["train_timesteps"] == 2000
        assert rl_data[0]["train_time_sec"] == pytest.approx(12.34)
        assert rl_data[0]["checkpoint_path"] == "checkpoints/model.zip"
        assert rl_data[1]["success"] is False
        assert rl_data[1]["train_timesteps"] == 2000


@async_test
async def test_execute_single_benchmark_run_preserves_rl_metadata_on_failure() -> None:
    """execute_single_benchmark_run retains RL metadata even when the runner fails."""
    from src.placement.jobs import register_algorithm

    async def failing_runner(netlist, seed, params, progress_cb):
        raise ValueError("Simulated placement runner failure.")

    register_algorithm("failing_rl_test", failing_runner)

    suite = get_default_benchmark_suite()
    netlist = suite["toy_4"]
    cfg = {
        "train_time_sec": 19.85,
        "train_timesteps": 4096,
        "checkpoint_path": "checkpoints/failing.zip",
        "eval_time_sec": 0.03,
    }

    result = await execute_single_benchmark_run(
        algorithm="failing_rl_test",
        netlist_id="toy_4",
        netlist=netlist,
        seed=42,
        config=cfg,
    )

    assert result.success is False
    assert result.is_legal is False
    assert result.is_complete is False
    assert result.error == "Simulated placement runner failure."
    assert result.train_time_sec == pytest.approx(19.85)
    assert result.train_timesteps == 4096
    assert result.checkpoint_path == "checkpoints/failing.zip"
    assert result.eval_time_sec == pytest.approx(0.03)


def test_parse_seeds_list() -> None:
    """CLI seed parser handles comma-separated, space-separated, and single seed inputs."""
    import argparse
    from scripts.run_experiments import parse_seeds_list

    # Comma-separated
    assert parse_seeds_list(["42,43,44"]) == [42, 43, 44]
    # Space-separated
    assert parse_seeds_list(["42", "43", "44"]) == [42, 43, 44]
    # Single seed
    assert parse_seeds_list(["42"]) == [42]
    # Mixed whitespace and comma
    assert parse_seeds_list([" 42 , 43 ", "44"]) == [42, 43, 44]
    # Empty or invalid
    with pytest.raises(argparse.ArgumentTypeError):
        parse_seeds_list([""])
    with pytest.raises(argparse.ArgumentTypeError):
        parse_seeds_list(["not_a_number"])


@async_test
async def test_multi_seed_independent_execution_and_persistence(tmp_path: Path) -> None:
    """Multiple seeds execute independently without overwriting results in CSV and JSON."""
    suite = get_default_benchmark_suite()
    toy_nl = suite["toy_4"]

    raw, sums = await run_benchmark_suite(
        netlists={"toy_4": toy_nl},
        algorithms=["random"],
        seeds=[42, 43, 44],
        output_dir=tmp_path,
    )

    # 1. In-memory results check
    assert len(raw) == 3
    result_seeds = [r.seed for r in raw]
    assert result_seeds == [42, 43, 44]
    for r in raw:
        assert r.success is True
        assert r.is_legal is True
        assert r.is_complete is True
        assert r.netlist_id == "toy_4"
        assert r.algorithm == "random"

    # 2. Summary check
    assert len(sums) == 1
    assert sums[0].total_runs == 3
    assert sums[0].success_count == 3

    # 3. Raw CSV persistence check
    csv_file = tmp_path / "benchmark_raw_results.csv"
    assert csv_file.exists()
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 3
        csv_seeds = [int(row["seed"]) for row in reader]
        assert csv_seeds == [42, 43, 44]
        for row in reader:
            assert row["algorithm"] == "random"
            assert row["netlist_id"] == "toy_4"
            assert row["success"] == "True"
