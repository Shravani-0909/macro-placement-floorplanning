"""Tests for benchmark suite execution, metrics calculation, and CSV/JSON reporting."""

import asyncio
import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.placement.benchmarks import (
    BenchmarkRunResult,
    compute_benchmark_summaries,
    execute_single_benchmark_run,
    get_default_benchmark_suite,
    run_benchmark_suite,
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
