"""Benchmark suite and execution framework for evaluating placement algorithms."""

import asyncio
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .constraints import validate_placement
from .hpwl import calculate_hpwl
from .jobs import ALGORITHM_REGISTRY, JobProgress
from .models import Macro, Net, Netlist


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

    return {
        "toy_4": toy_4,
        "medium_8": medium_8,
        "dense_10": dense_10,
    }


async def execute_single_benchmark_run(
    algorithm: str,
    netlist_id: str,
    netlist: Netlist,
    seed: int,
    config: dict[str, Any] | None = None,
) -> BenchmarkRunResult:
    """Execute a single deterministic algorithm run and evaluate placement legality and HPWL."""
    cfg = config or {}
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
        )

    # Dummy callback for progress
    async def noop_progress(p: JobProgress) -> None:
        pass

    start_time = time.perf_counter()
    try:
        placement = await runner(netlist, seed, cfg, noop_progress)
        elapsed = time.perf_counter() - start_time

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
            )

        hpwl = calculate_hpwl(netlist, placement, allow_incomplete=False)
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
        )

    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start_time
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
            mean_hpwl = float(statistics.mean(hpwls))
            median_hpwl = float(statistics.median(hpwls))
            min_hpwl = float(min(hpwls))
            max_hpwl = float(max(hpwls))
            std_hpwl = float(statistics.stdev(hpwls)) if len(hpwls) > 1 else 0.0
        else:
            mean_hpwl = None
            median_hpwl = None
            min_hpwl = None
            max_hpwl = None
            std_hpwl = None

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
            )
        )

    return summaries


def write_benchmark_reports(
    results: list[BenchmarkRunResult],
    summaries: list[BenchmarkSummary],
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
            "success", "is_legal", "is_complete", "hpwl", "runtime_sec", "error",
        ])
        for r in results:
            writer.writerow([
                r.algorithm, r.netlist_id, r.grid_w, r.grid_h, r.seed,
                r.success, r.is_legal, r.is_complete,
                f"{r.hpwl:.4f}" if r.hpwl is not None else "",
                f"{r.runtime_sec:.6f}", r.error or "",
            ])

    # Write summary JSON
    summary_json_path = out_path / "benchmark_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in summaries], f, indent=2)

    # Write summary CSV
    summary_csv_path = out_path / "benchmark_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm", "netlist_id", "total_runs", "success_count", "success_rate",
            "mean_hpwl", "median_hpwl", "min_hpwl", "max_hpwl", "std_hpwl",
            "mean_runtime_sec", "median_runtime_sec",
        ])
        for s in summaries:
            writer.writerow([
                s.algorithm, s.netlist_id, s.total_runs, s.success_count, f"{s.success_rate:.2%}",
                f"{s.mean_hpwl:.4f}" if s.mean_hpwl is not None else "",
                f"{s.median_hpwl:.4f}" if s.median_hpwl is not None else "",
                f"{s.min_hpwl:.4f}" if s.min_hpwl is not None else "",
                f"{s.max_hpwl:.4f}" if s.max_hpwl is not None else "",
                f"{s.std_hpwl:.4f}" if s.std_hpwl is not None else "",
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
    algos = algorithms if algorithms is not None else list(ALGORITHM_REGISTRY.keys())
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
