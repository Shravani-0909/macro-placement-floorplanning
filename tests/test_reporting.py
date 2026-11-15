"""Comprehensive Automated Tests for Macro Placement Research Report Generator.

Tests data models, CSV loaders, normalization layer, statistical aggregations,
publication figure rendering, reproducibility provenance, and full report generation.
"""

from __future__ import annotations

import csv
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

from src.reporting.archival import (
    create_all_archival_results,
    create_experiment_b_archival_data,
    create_experiment_d_archival_data,
)
from src.reporting.loaders import (
    _parse_bool,
    _parse_float,
    _parse_int,
    discover_experiments,
    load_experiment_dir,
    load_raw_csv,
    load_summary_csv,
)
from src.reporting.models import (
    ExperimentData,
    GeneralizationGap,
    ProvenanceInfo,
    RunRecord,
    SummaryRecord,
    TopologySensitivity,
)
from src.reporting.plots import (
    generate_all_figures,
    plot_experiment_a_comparison,
    plot_experiment_a_scaling,
    plot_experiment_b_sensitivity,
    plot_experiment_d_generalization,
    plot_objective_decomposition,
)
from src.reporting.provenance import collect_provenance, get_git_commit, get_package_version
from src.reporting.report import (
    build_experiment_a_table,
    build_experiment_d_wide_table,
    format_stat_cell,
    generate_research_report,
)
from src.reporting.statistics import (
    compute_connectivity_sensitivities,
    compute_generalization_gaps,
    compute_stats,
    percentage_difference,
    summarize_records,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_run_records() -> list[RunRecord]:
    """Sample multi-algorithm, multi-seed run records across 3 benchmark instances."""
    records = []
    benchmarks = [("dense_10", 24, 10), ("scaling_20", 36, 20), ("scaling_30", 48, 30)]
    algos = ["random", "greedy", "sa", "rl"]
    seeds = [42, 43, 44]

    base_objectives = {
        ("random", "dense_10"): 260.0,
        ("greedy", "dense_10"): 155.0,
        ("sa", "dense_10"): 190.0,
        ("rl", "dense_10"): 210.0,
        ("random", "scaling_20"): 950.0,
        ("greedy", "scaling_20"): 550.0,
        ("sa", "scaling_20"): 600.0,
        ("rl", "scaling_20"): 720.0,
        ("random", "scaling_30"): 2050.0,
        ("greedy", "scaling_30"): 1140.0,
        ("sa", "scaling_30"): 1350.0,
        ("rl", "scaling_30"): 1600.0,
    }

    for b_id, g_dim, macros in benchmarks:
        for algo in algos:
            for s in seeds:
                base = base_objectives[(algo, b_id)]
                jitter = (s - 43) * 5.0
                hpwl = base * 0.99 + jitter
                cong = 0.5 + (s - 43) * 0.05
                obj = hpwl + cong
                records.append(
                    RunRecord(
                        algorithm=algo,
                        netlist_id=b_id,
                        grid_w=g_dim,
                        grid_h=g_dim,
                        seed=s,
                        success=True,
                        is_legal=True,
                        is_complete=True,
                        hpwl=hpwl,
                        congestion=cong,
                        objective=obj,
                        runtime_sec=0.1 * macros,
                        eval_time_sec=0.1 * macros,
                        train_time_sec=12.5 if algo == "rl" else None,
                        train_timesteps=10000 if algo == "rl" else None,
                        checkpoint_path=f"checkpoints/ppo_{b_id}_s{s}.zip" if algo == "rl" else None,
                    )
                )
    return records


@pytest.fixture
def sample_connectivity_records() -> list[RunRecord]:
    """Sample records for Experiment B comparing random and clustered topologies."""
    records = []
    topologies = ["exp_b_12_random", "exp_b_12_clustered"]
    algos = ["random", "greedy", "sa", "rl"]
    seeds = [42, 43, 44]

    # Clustered topology generally yields lower (better) HPWL and congestion
    multipliers = {"exp_b_12_random": 1.0, "exp_b_12_clustered": 0.82}

    for topo in topologies:
        mult = multipliers[topo]
        for algo in algos:
            base = {"random": 300.0, "greedy": 180.0, "sa": 210.0, "rl": 240.0}[algo] * mult
            for s in seeds:
                hpwl = base + (s - 43) * 3.0
                cong = 0.45 * mult
                records.append(
                    RunRecord(
                        algorithm=algo,
                        netlist_id=topo,
                        grid_w=28,
                        grid_h=28,
                        seed=s,
                        success=True,
                        is_legal=True,
                        is_complete=True,
                        hpwl=hpwl,
                        congestion=cong,
                        objective=hpwl + cong,
                        runtime_sec=0.5,
                    )
                )
    return records


@pytest.fixture
def sample_generalization_records() -> list[RunRecord]:
    """Sample records for Experiment D comparing train, unseen instance, and unseen topology."""
    records = []
    conditions = ["exp_d_train_random", "exp_d_test_unseen_inst", "exp_d_test_unseen_topo"]
    algos = ["random", "greedy", "sa", "rl"]
    seeds = [42, 43, 44]

    # RL has higher degradation on unseen instance and topology
    degradations = {
        "random": {"exp_d_train_random": 320.0, "exp_d_test_unseen_inst": 322.0, "exp_d_test_unseen_topo": 290.0},
        "greedy": {"exp_d_train_random": 190.0, "exp_d_test_unseen_inst": 192.0, "exp_d_test_unseen_topo": 170.0},
        "sa": {"exp_d_train_random": 220.0, "exp_d_test_unseen_inst": 224.0, "exp_d_test_unseen_topo": 185.0},
        "rl": {"exp_d_train_random": 250.0, "exp_d_test_unseen_inst": 285.0, "exp_d_test_unseen_topo": 275.0},
    }

    for cond in conditions:
        for algo in algos:
            base = degradations[algo][cond]
            for s in seeds:
                hpwl = base + (s - 43) * 2.0
                cong = 0.40
                records.append(
                    RunRecord(
                        algorithm=algo,
                        netlist_id=cond,
                        grid_w=28,
                        grid_h=28,
                        seed=s,
                        success=True,
                        is_legal=True,
                        is_complete=True,
                        hpwl=hpwl,
                        congestion=cong,
                        objective=hpwl + cong,
                        runtime_sec=0.5,
                    )
                )
    return records


# ---------------------------------------------------------------------------
# Unit Tests: Models & Parsing
# ---------------------------------------------------------------------------


def test_models_to_dict(sample_run_records: list[RunRecord]) -> None:
    """RunRecord, SummaryRecord, and GeneralizationGap serializations."""
    rec = sample_run_records[0]
    d = rec.to_dict()
    assert d["algorithm"] == "random"
    assert d["netlist_id"] == "dense_10"
    assert rec.macro_count == 10
    assert not rec.is_rl

    rl_rec = [r for r in sample_run_records if r.is_rl][0]
    assert rl_rec.is_rl
    assert rl_rec.train_timesteps == 10000

    gap = GeneralizationGap(
        algorithm="rl",
        train_netlist_id="exp_d_train_random",
        test_netlist_id="exp_d_test_unseen_inst",
        test_type="unseen_instance",
        train_mean_objective=250.0,
        test_mean_objective=285.0,
        gap_delta=35.0,
        gap_percentage=14.0,
    )
    assert "Degradation" in gap.interpretation


def test_parser_helpers() -> None:
    """Helper parsers handle booleans, floats, percentages, and integers safely."""
    assert _parse_bool("True") is True
    assert _parse_bool("false") is False
    assert _parse_bool("1") is True
    assert _parse_bool(None) is False

    assert _parse_float("123.45") == pytest.approx(123.45)
    assert _parse_float("85.5%") == pytest.approx(0.855)
    assert _parse_float("N/A") is None
    assert _parse_float("") is None
    assert _parse_float(None) is None

    assert _parse_int("42") == 42
    assert _parse_int("42.0") == 42
    assert _parse_int("") is None
    assert _parse_int(None) is None


# ---------------------------------------------------------------------------
# Unit Tests: Statistics
# ---------------------------------------------------------------------------


def test_compute_stats() -> None:
    """Statistical functions compute correct mean, std, median, min, max."""
    vals = [10.0, 20.0, 30.0]
    stats = compute_stats(vals)
    assert stats["mean"] == pytest.approx(20.0)
    assert stats["median"] == pytest.approx(20.0)
    assert stats["min"] == pytest.approx(10.0)
    assert stats["max"] == pytest.approx(30.0)
    assert stats["std"] == pytest.approx(10.0)
    assert stats["count"] == 3.0

    # Single value has std = 0.0
    single = compute_stats([42.0])
    assert single["std"] == 0.0
    assert single["mean"] == 42.0

    # Empty list
    empty = compute_stats([])
    assert empty["mean"] == 0.0
    assert empty["count"] == 0.0


def test_percentage_difference() -> None:
    """Percentage difference handles increases, decreases, and division by zero."""
    assert percentage_difference(150.0, 100.0) == pytest.approx(50.0)
    assert percentage_difference(80.0, 100.0) == pytest.approx(-20.0)
    assert percentage_difference(100.0, 100.0) == pytest.approx(0.0)

    # Zero denominator
    assert percentage_difference(0.0, 0.0) == 0.0
    assert math.isinf(percentage_difference(10.0, 0.0))
    assert math.isinf(percentage_difference(-10.0, 0.0))

    # None or NaN
    assert percentage_difference(None, 100.0) == 0.0
    assert percentage_difference(100.0, float("nan")) == 0.0


def test_summarize_records(sample_run_records: list[RunRecord]) -> None:
    """Records are correctly aggregated into summaries per algorithm and netlist."""
    summaries = summarize_records(sample_run_records)
    # 3 benchmarks * 4 algorithms = 12 summaries
    assert len(summaries) == 12

    sum_map = {(s.algorithm, s.netlist_id): s for s in summaries}
    greedy_dense = sum_map[("greedy", "dense_10")]
    assert greedy_dense.total_runs == 3
    assert greedy_dense.success_count == 3
    assert greedy_dense.success_rate == 1.0
    assert greedy_dense.mean_objective is not None
    assert greedy_dense.mean_objective < 200.0


def test_compute_connectivity_sensitivities(sample_connectivity_records: list[RunRecord]) -> None:
    """Connectivity sensitivity correctly calculates delta and percentage change."""
    sens = compute_connectivity_sensitivities(sample_connectivity_records)
    assert len(sens) == 4
    for s in sens:
        # All methods should improve (negative delta / lower objective) on clustered topology
        assert s.clustered_mean_obj < s.random_mean_obj
        assert s.obj_delta < 0.0
        assert s.obj_pct < 0.0


def test_compute_generalization_gaps(sample_generalization_records: list[RunRecord]) -> None:
    """Generalization gaps identify train-to-test degradation."""
    gaps = compute_generalization_gaps(sample_generalization_records)
    # 4 algorithms * 2 holdout conditions = 8 gaps
    assert len(gaps) == 8

    rl_inst_gap = [g for g in gaps if g.algorithm == "rl" and g.test_type == "unseen_instance"][0]
    assert rl_inst_gap.gap_delta > 0.0
    assert rl_inst_gap.gap_percentage > 0.0
    assert "Degradation" in rl_inst_gap.interpretation


# ---------------------------------------------------------------------------
# Unit Tests: Loaders & Discovery
# ---------------------------------------------------------------------------


def test_load_raw_csv(tmp_path: Path) -> None:
    """Loader reads CSV, normalizes types, and raises on missing file or malformed header."""
    csv_file = tmp_path / "benchmark_raw_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "netlist_id", "grid_w", "grid_h", "seed", "success", "is_legal", "is_complete", "hpwl", "congestion", "objective", "runtime_sec"])
        writer.writerow(["random", "dense_10", "24", "24", "42", "True", "True", "True", "250.5", "0.45", "250.95", "0.12"])

    records = load_raw_csv(csv_file)
    assert len(records) == 1
    r = records[0]
    assert r.algorithm == "random"
    assert r.netlist_id == "dense_10"
    assert r.grid_w == 24
    assert r.seed == 42
    assert r.success is True
    assert r.hpwl == pytest.approx(250.5)
    assert r.congestion == pytest.approx(0.45)
    assert r.objective == pytest.approx(250.95)

    # Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_raw_csv(tmp_path / "non_existent.csv")

    # Malformed header raises ValueError
    bad_csv = tmp_path / "bad.csv"
    with open(bad_csv, "w", encoding="utf-8") as f:
        f.write("foo,bar\n1,2\n")
    with pytest.raises(ValueError):
        load_raw_csv(bad_csv)


def test_load_summary_csv(tmp_path: Path) -> None:
    """Loader parses summary CSV."""
    sum_file = tmp_path / "benchmark_summary.csv"
    with open(sum_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "netlist_id", "total_runs", "success_count", "success_rate", "mean_hpwl", "mean_objective"])
        writer.writerow(["greedy", "dense_10", "3", "3", "100.0%", "150.0", "150.5"])

    sums = load_summary_csv(sum_file)
    assert len(sums) == 1
    s = sums[0]
    assert s.algorithm == "greedy"
    assert s.total_runs == 3
    assert s.success_rate == 1.0
    assert s.mean_hpwl == 150.0


def test_discover_experiments(tmp_path: Path) -> None:
    """Discovery scans subdirectories and aliases."""
    exp_a_dir = tmp_path / "experiment_a_scaling"
    exp_a_dir.mkdir()
    (exp_a_dir / "benchmark_raw_results.csv").write_text("algorithm,netlist_id\ngreedy,dense_10\n", encoding="utf-8")

    discovered = discover_experiments(tmp_path)
    assert "exp_a" in discovered
    assert discovered["exp_a"].name == "exp_a"


# ---------------------------------------------------------------------------
# Unit Tests: Plot Generation
# ---------------------------------------------------------------------------


def test_generate_all_figures(
    tmp_path: Path,
    sample_run_records: list[RunRecord],
    sample_connectivity_records: list[RunRecord],
    sample_generalization_records: list[RunRecord],
) -> None:
    """All 5 research figures are generated as valid, non-empty PNG files."""
    fig_dir = tmp_path / "figures"
    fig_paths = generate_all_figures(
        output_dir=fig_dir,
        records_a=sample_run_records,
        records_b=sample_connectivity_records,
        records_d=sample_generalization_records,
    )

    assert len(fig_paths) == 5
    for key, path in fig_paths.items():
        assert path.is_file(), f"Figure {key} not found at {path}"
        assert path.stat().st_size > 1000, f"Figure {key} at {path} is too small / empty"


# ---------------------------------------------------------------------------
# Unit Tests: Provenance
# ---------------------------------------------------------------------------


def test_collect_provenance(sample_run_records: list[RunRecord]) -> None:
    """Provenance collector inspects system environment and packages."""
    prov = collect_provenance(records=sample_run_records)
    assert prov.timestamp != ""
    assert prov.python_version != ""
    assert prov.platform != ""
    assert prov.numpy_version != "Not available"
    assert prov.matplotlib_version != "Not available"
    assert prov.jinja2_version != "Not available"
    assert prov.seeds == [42, 43, 44]
    assert prov.alpha == 1.0
    assert prov.beta == 1.0
    assert prov.reward_scale == 100.0


# ---------------------------------------------------------------------------
# Integration Tests: Full Report Generation
# ---------------------------------------------------------------------------


def test_generate_research_report_end_to_end(
    tmp_path: Path,
    sample_run_records: list[RunRecord],
    sample_connectivity_records: list[RunRecord],
    sample_generalization_records: list[RunRecord],
) -> None:
    """Complete end-to-end report generation with synthetic multi-experiment suite."""
    res_dir = tmp_path / "results"
    out_dir = tmp_path / "reports"

    # Write synthetic CSVs
    a_dir = res_dir / "experiment_a_scaling"
    b_dir = res_dir / "experiment_b_connectivity"
    d_dir = res_dir / "experiment_d_generalization"

    for d, recs in [(a_dir, sample_run_records), (b_dir, sample_connectivity_records), (d_dir, sample_generalization_records)]:
        d.mkdir(parents=True)
        raw_csv = d / "benchmark_raw_results.csv"
        with open(raw_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["algorithm", "netlist_id", "grid_w", "grid_h", "seed", "success", "is_legal", "is_complete", "hpwl", "congestion", "objective", "runtime_sec", "train_timesteps"])
            for r in recs:
                writer.writerow([r.algorithm, r.netlist_id, r.grid_w, r.grid_h, r.seed, r.success, r.is_legal, r.is_complete, r.hpwl, r.congestion, r.objective, r.runtime_sec, r.train_timesteps or ""])

    result = generate_research_report(
        results_dir=res_dir,
        output_dir=out_dir,
        generate_pdf=True,
        embed_images=True,
    )

    assert result["html_path"].is_file()
    assert result["num_records_a"] > 0
    assert result["num_records_b"] > 0
    assert result["num_records_d"] > 0
    assert len(result["figures"]) == 5

    # Check HTML content
    with open(result["html_path"], "r", encoding="utf-8") as f:
        html = f.read()

    # All 16 sections represented
    for sec in [
        "1. Title" if "1. Title" in html else "Title",
        "2. Abstract / Executive Summary",
        "3. Problem Statement",
        "4. Methodology",
        "5. Objective Function",
        "6. Algorithms",
        "7. Experimental Setup",
        "8. Experiment A",
        "9. Experiment B",
        "10. Experiment D",
        "11. Statistical Summary",
        "12. RL vs Classical Comparison",
        "13. Generalization Findings",
        "14. Limitations",
        "15. Conclusions",
        "16. Reproducibility / Provenance",
    ]:
        assert sec in html, f"Missing section heading: {sec}"

    # Verify no NaN or None token leaks in prose (excluding base64 blobs)
    prose = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", html)
    assert not re.search(r"\bNaN\b", prose)
    assert not re.search(r"\bNone\b", prose)


def test_cli_smoke_execution(tmp_path: Path) -> None:
    """CLI script executes smoothly against the repo's existing results directory."""
    cmd = [
        sys.executable,
        "scripts/generate_report.py",
        "--results-dir",
        "results",
        "--output-dir",
        str(tmp_path / "reports"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    assert "Loaded Experiment A:" in res.stdout
    assert "Generated:" in res.stdout
    assert (tmp_path / "reports" / "macro_placement_report.html").is_file()


# ---------------------------------------------------------------------------
# Archival Reconstruction Tests
# ---------------------------------------------------------------------------


def test_archival_reconstruction_utility(tmp_path: Path) -> None:
    """Archival utility reconstructs Experiment B and D without executing training/solvers."""
    import src.reporting.archival as archival_mod

    # Verify that archival module does NOT import torch, PPO, or experiment runners
    archival_source = Path(archival_mod.__file__).read_text(encoding="utf-8")
    assert "run_experiment" not in archival_source
    assert "train_ppo" not in archival_source
    assert "PlacementEngine" not in archival_source
    assert "execute_single_benchmark_run" not in archival_source

    paths = create_all_archival_results(results_dir=tmp_path)

    assert (paths["exp_b"] / "benchmark_raw_results.csv").is_file()
    assert (paths["exp_b"] / "benchmark_summary.csv").is_file()
    assert (paths["exp_b"] / "archival_metadata.json").is_file()

    assert (paths["exp_d"] / "benchmark_raw_results.csv").is_file()
    assert (paths["exp_d"] / "benchmark_summary.csv").is_file()
    assert (paths["exp_d"] / "archival_metadata.json").is_file()


def test_archival_b_data_loading(tmp_path: Path) -> None:
    """Archival B dataset loads 24 records, correct algorithms, verified means, and sensitivity."""
    b_dir = create_experiment_b_archival_data(output_dir=tmp_path)
    exp_b = load_experiment_dir(b_dir, exp_name="exp_b")

    assert exp_b.source_type == "verified_archival_reconstruction"
    assert exp_b.is_archival is True
    assert len(exp_b.raw_records) == 24
    assert len(exp_b.summaries) == 8

    # All records flagged as archival
    for r in exp_b.raw_records:
        assert r.is_archival is True
        assert r.grid_w == 28
        assert r.grid_h == 28
        assert r.hpwl is None  # Not fabricated
        assert r.congestion is None  # Not fabricated

    algos = set(exp_b.algorithms)
    assert algos == {"random", "greedy", "sa", "rl"}

    # Compute and verify sensitivity
    sens = compute_connectivity_sensitivities(exp_b.raw_records)
    assert len(sens) == 4
    sens_map = {s.algorithm: s for s in sens}

    # Verified sensitivity changes
    assert sens_map["random"].obj_pct == pytest.approx(-13.61, abs=0.05)
    assert sens_map["greedy"].obj_pct == pytest.approx(-15.79, abs=0.05)
    assert sens_map["sa"].obj_pct == pytest.approx(-29.34, abs=0.05)
    assert sens_map["rl"].obj_pct == pytest.approx(-15.15, abs=0.05)

    # SA has largest topology sensitivity
    assert sens_map["sa"].obj_pct < sens_map["greedy"].obj_pct
    assert sens_map["sa"].obj_pct < sens_map["random"].obj_pct

    # Ranking in both topologies: Greedy < SA < RL < Random
    summaries = summarize_records(exp_b.raw_records)
    sum_map = {(s.algorithm, s.netlist_id): s.mean_objective for s in summaries}

    for nid in ["exp_b_12_random", "exp_b_12_clustered"]:
        assert sum_map[("greedy", nid)] < sum_map[("sa", nid)] < sum_map[("rl", nid)] < sum_map[("random", nid)]


def test_archival_d_data_loading(tmp_path: Path) -> None:
    """Archival D dataset loads 36 records, correct algorithms, verified means, and generalization gaps."""
    d_dir = create_experiment_d_archival_data(output_dir=tmp_path)
    exp_d = load_experiment_dir(d_dir, exp_name="exp_d")

    assert exp_d.source_type == "verified_archival_reconstruction"
    assert exp_d.is_archival is True
    assert len(exp_d.raw_records) == 36
    assert len(exp_d.summaries) == 12

    for r in exp_d.raw_records:
        assert r.is_archival is True
        assert r.grid_w == 28
        assert r.grid_h == 28
        assert r.hpwl is None  # Unpreserved, not fabricated

    algos = set(exp_d.algorithms)
    assert algos == {"random", "greedy", "sa", "rl"}

    # Exact per-seed RL verification (canonical values)
    rl_records = [r for r in exp_d.raw_records if r.algorithm == "rl"]
    rl_seed_map = {(r.netlist_id, r.seed): r.objective for r in rl_records}

    # exp_d_train_random
    assert rl_seed_map[("exp_d_train_random", 42)] == pytest.approx(300.2118, abs=1e-4)
    assert rl_seed_map[("exp_d_train_random", 43)] == pytest.approx(356.9547, abs=1e-4)
    assert rl_seed_map[("exp_d_train_random", 44)] == pytest.approx(349.0801, abs=1e-4)

    # exp_d_test_unseen_inst
    assert rl_seed_map[("exp_d_test_unseen_inst", 42)] == pytest.approx(430.1819, abs=1e-4)
    assert rl_seed_map[("exp_d_test_unseen_inst", 43)] == pytest.approx(439.1122, abs=1e-4)
    assert rl_seed_map[("exp_d_test_unseen_inst", 44)] == pytest.approx(455.6179, abs=1e-4)

    # exp_d_test_unseen_topo
    assert rl_seed_map[("exp_d_test_unseen_topo", 42)] == pytest.approx(512.6232, abs=1e-4)
    assert rl_seed_map[("exp_d_test_unseen_topo", 43)] == pytest.approx(461.9027, abs=1e-4)
    assert rl_seed_map[("exp_d_test_unseen_topo", 44)] == pytest.approx(406.8334, abs=1e-4)

    # Verified means
    summaries = summarize_records(exp_d.raw_records)
    sum_map = {(s.algorithm, s.netlist_id): s.mean_objective for s in summaries}

    assert sum_map[("rl", "exp_d_train_random")] == pytest.approx(335.42, abs=0.05)
    assert sum_map[("rl", "exp_d_test_unseen_inst")] == pytest.approx(441.64, abs=0.05)
    assert sum_map[("rl", "exp_d_test_unseen_topo")] == pytest.approx(460.45, abs=0.05)

    assert sum_map[("random", "exp_d_train_random")] == pytest.approx(513.57, abs=0.05)
    assert sum_map[("greedy", "exp_d_train_random")] == pytest.approx(225.48, abs=0.05)
    assert sum_map[("sa", "exp_d_train_random")] == pytest.approx(305.04, abs=0.05)

    # Generalization gaps
    gaps = compute_generalization_gaps(exp_d.raw_records)
    assert len(gaps) == 8
    gap_map = {(g.algorithm, g.test_type): g for g in gaps}

    # RL gaps
    rl_inst = gap_map[("rl", "unseen_instance")]
    assert rl_inst.gap_delta == pytest.approx(106.22, abs=0.1)
    assert rl_inst.gap_percentage == pytest.approx(31.67, abs=0.1)

    rl_topo = gap_map[("rl", "unseen_topology")]
    assert rl_topo.gap_delta == pytest.approx(125.04, abs=0.1)
    assert rl_topo.gap_percentage == pytest.approx(37.28, abs=0.1)

    # Classical gaps
    greedy_inst = gap_map[("greedy", "unseen_instance")]
    assert greedy_inst.gap_delta == pytest.approx(80.28, abs=0.1)
    assert greedy_inst.gap_percentage == pytest.approx(35.60, abs=0.1)

    sa_topo = gap_map[("sa", "unseen_topology")]
    assert sa_topo.gap_delta == pytest.approx(-24.17, abs=0.1)
    assert sa_topo.gap_percentage == pytest.approx(-7.92, abs=0.1)

    sa_inst = gap_map[("sa", "unseen_instance")]
    assert sa_inst.gap_delta == pytest.approx(43.49, abs=0.1)
    assert sa_inst.gap_percentage == pytest.approx(14.26, abs=0.1)

    random_inst = gap_map[("random", "unseen_instance")]
    assert random_inst.gap_delta == pytest.approx(21.87, abs=0.1)
    assert random_inst.gap_percentage == pytest.approx(4.26, abs=0.1)

    random_topo = gap_map[("random", "unseen_topology")]
    assert random_topo.gap_delta == pytest.approx(55.29, abs=0.1)
    assert random_topo.gap_percentage == pytest.approx(10.77, abs=0.1)


def test_experiment_a_provenance_and_incomplete_labeling(tmp_path: Path) -> None:
    """Experiment A dataset with 9 rows is recognized as incomplete/legacy single-seed baseline."""
    from src.reporting.archival import VERIFIED_EXPERIMENT_A_SUMMARIES

    assert len(VERIFIED_EXPERIMENT_A_SUMMARIES) == 3
    bench_map = {s["netlist_id"]: s for s in VERIFIED_EXPERIMENT_A_SUMMARIES}
    assert bench_map["dense_10"]["rl"] == pytest.approx(183.87, abs=0.01)
    assert bench_map["dense_10"]["random"] == pytest.approx(258.79, abs=0.01)
    assert bench_map["scaling_20"]["rl"] == pytest.approx(862.23, abs=0.01)
    assert bench_map["scaling_30"]["rl"] == pytest.approx(1873.50, abs=0.01)

    res = generate_research_report(
        results_dir="results",
        output_dir=tmp_path / "reports",
        generate_pdf=False,
    )
    with open(res["html_path"], "r", encoding="utf-8") as f:
        html = f.read()

    assert "Incomplete / Legacy Baselines" in html
    assert "Table 8.1: Verified Final Multi-Seed Experiment A Comparison" in html
    assert "Table 8.2: Local Incomplete / Legacy Single-Seed Baseline Measurements" in html


def test_provenance_identifies_archival_vs_original(tmp_path: Path) -> None:
    """Provenance collector distinguishes original CSV from archival reconstruction."""
    # Create fake original A dir
    a_dir = tmp_path / "experiment_a_scaling"
    a_dir.mkdir()
    (a_dir / "benchmark_raw_results.csv").write_text("algorithm,netlist_id,objective\ngreedy,dense_10,150.0\n", encoding="utf-8")

    # Create archival B and D dirs
    create_all_archival_results(results_dir=tmp_path)

    res = generate_research_report(
        results_dir=tmp_path,
        output_dir=tmp_path / "reports",
        generate_pdf=False,
    )

    with open(res["html_path"], "r", encoding="utf-8") as f:
        html = f.read()

    assert "Original Experiment CSV" in html
    assert "Verified Archival Reconstruction" in html
    assert "Archival Data Notice" in html


def test_normal_original_csv_loading_unchanged(tmp_path: Path) -> None:
    """Standard raw CSVs without archival_metadata.json load as original_csv and is_archival=False."""
    csv_file = tmp_path / "benchmark_raw_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "netlist_id", "grid_w", "grid_h", "seed", "success", "is_legal", "is_complete", "hpwl", "congestion", "objective", "runtime_sec"])
        writer.writerow(["greedy", "toy_4", "15", "15", "42", "True", "True", "True", "25.0", "0.20", "25.20", "0.05"])

    exp = load_experiment_dir(tmp_path, exp_name="normal")
    assert exp.source_type == "original_csv"
    assert exp.is_archival is False
    assert len(exp.raw_records) == 1
    assert exp.raw_records[0].is_archival is False


def test_cli_smoke_execution_with_archival_data(tmp_path: Path) -> None:
    """CLI script outputs row counts for Experiment A, B, and D when archival data is present."""
    cmd = [
        sys.executable,
        "scripts/generate_report.py",
        "--results-dir",
        "results",
        "--output-dir",
        str(tmp_path / "reports"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    assert "Loaded Experiment A: 9 rows" in res.stdout
    assert "Loaded Experiment B: 24 rows" in res.stdout
    assert "Loaded Experiment D: 36 rows" in res.stdout
