"""Loaders and Normalization Layer for Macro Placement Results.

Parses raw benchmark CSV/JSON results and statistical summaries across
Experiment A (Scaling), Experiment B (Connectivity), and Experiment D (Generalization).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.reporting.models import ExperimentData, RunRecord, SummaryRecord


def _parse_bool(val: Any) -> bool:
    """Safely parse boolean from string or boolean value."""
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in ("true", "1", "yes", "t")


def _parse_float(val: Any) -> float | None:
    """Safely parse float from string or numeric value."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("none", "null", "nan", "n/a", ""):
        return None
    # Strip percentage sign if present
    if s.endswith("%"):
        s = s[:-1]
        try:
            return float(s) / 100.0
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_int(val: Any) -> int | None:
    """Safely parse integer from string or numeric value."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("none", "null", "nan", "n/a", ""):
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def load_raw_csv(csv_path: Path | str) -> list[RunRecord]:
    """Load raw benchmark run records from CSV.

    Args:
        csv_path: Path to benchmark_raw_results.csv.

    Returns:
        List of RunRecord objects.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If file is empty or missing essential fields.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw results CSV file not found: {path}")

    records: list[RunRecord] = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV file is empty or missing header: {path}")

        fieldnames = set(reader.fieldnames)
        if "algorithm" not in fieldnames or "netlist_id" not in fieldnames:
            raise ValueError(f"CSV missing mandatory columns ('algorithm', 'netlist_id'): {path}")

        for row_idx, row in enumerate(reader, start=2):
            algo = row.get("algorithm", "").strip()
            netlist = row.get("netlist_id", "").strip()
            if not algo or not netlist:
                continue

            grid_w = _parse_int(row.get("grid_w", 0)) or 0
            grid_h = _parse_int(row.get("grid_h", 0)) or 0
            seed = _parse_int(row.get("seed", 0)) or 0
            success = _parse_bool(row.get("success", False))
            is_legal = _parse_bool(row.get("is_legal", False))
            is_complete = _parse_bool(row.get("is_complete", False))

            hpwl = _parse_float(row.get("hpwl"))
            congestion = _parse_float(row.get("congestion"))
            objective = _parse_float(row.get("objective"))

            # Derive objective if missing but hpwl and congestion present (alpha=1.0, beta=1.0)
            if objective is None and hpwl is not None and congestion is not None:
                objective = hpwl + congestion

            runtime_sec = _parse_float(row.get("runtime_sec")) or 0.0
            eval_time_sec = _parse_float(row.get("eval_time_sec"))
            train_time_sec = _parse_float(row.get("train_time_sec"))
            train_timesteps = _parse_int(row.get("train_timesteps"))
            checkpoint_path = row.get("checkpoint_path") or None
            error = row.get("error") or None

            records.append(
                RunRecord(
                    algorithm=algo,
                    netlist_id=netlist,
                    grid_w=grid_w,
                    grid_h=grid_h,
                    seed=seed,
                    success=success,
                    is_legal=is_legal,
                    is_complete=is_complete,
                    hpwl=hpwl,
                    congestion=congestion,
                    objective=objective,
                    runtime_sec=runtime_sec,
                    eval_time_sec=eval_time_sec,
                    train_time_sec=train_time_sec,
                    train_timesteps=train_timesteps,
                    checkpoint_path=checkpoint_path,
                    error=error,
                )
            )

    return records


def load_summary_csv(csv_path: Path | str) -> list[SummaryRecord]:
    """Load benchmark summaries from CSV.

    Args:
        csv_path: Path to benchmark_summary.csv.

    Returns:
        List of SummaryRecord objects.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If file is empty or missing essential fields.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Summary CSV file not found: {path}")

    summaries: list[SummaryRecord] = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV file is empty or missing header: {path}")

        fieldnames = set(reader.fieldnames)
        if "algorithm" not in fieldnames or "netlist_id" not in fieldnames:
            raise ValueError(f"Summary CSV missing mandatory columns: {path}")

        for row in reader:
            algo = row.get("algorithm", "").strip()
            netlist = row.get("netlist_id", "").strip()
            if not algo or not netlist:
                continue

            total_runs = _parse_int(row.get("total_runs")) or 0
            success_count = _parse_int(row.get("success_count")) or 0
            success_rate = _parse_float(row.get("success_rate")) or 0.0

            mean_hpwl = _parse_float(row.get("mean_hpwl"))
            median_hpwl = _parse_float(row.get("median_hpwl"))
            min_hpwl = _parse_float(row.get("min_hpwl"))
            max_hpwl = _parse_float(row.get("max_hpwl"))
            std_hpwl = _parse_float(row.get("std_hpwl"))

            mean_congestion = _parse_float(row.get("mean_congestion"))
            median_congestion = _parse_float(row.get("median_congestion"))
            std_congestion = _parse_float(row.get("std_congestion"))

            mean_objective = _parse_float(row.get("mean_objective"))
            median_objective = _parse_float(row.get("median_objective"))
            std_objective = _parse_float(row.get("std_objective"))

            mean_runtime_sec = _parse_float(row.get("mean_runtime_sec")) or 0.0
            median_runtime_sec = _parse_float(row.get("median_runtime_sec")) or 0.0

            summaries.append(
                SummaryRecord(
                    algorithm=algo,
                    netlist_id=netlist,
                    total_runs=total_runs,
                    success_count=success_count,
                    success_rate=success_rate,
                    mean_hpwl=mean_hpwl,
                    median_hpwl=median_hpwl,
                    min_hpwl=min_hpwl,
                    max_hpwl=max_hpwl,
                    std_hpwl=std_hpwl,
                    mean_congestion=mean_congestion,
                    median_congestion=median_congestion,
                    std_congestion=std_congestion,
                    mean_objective=mean_objective,
                    median_objective=median_objective,
                    std_objective=std_objective,
                    mean_runtime_sec=mean_runtime_sec,
                    median_runtime_sec=median_runtime_sec,
                )
            )

    return summaries


def load_experiment_dir(dir_path: Path | str, exp_name: str | None = None) -> ExperimentData:
    """Load an experiment directory containing benchmark_raw_results and/or benchmark_summary.

    Args:
        dir_path: Directory path.
        exp_name: Optional experiment name override.

    Returns:
        ExperimentData object.
    """
    path = Path(dir_path)
    name = exp_name or path.name

    raw_records: list[RunRecord] = []
    summaries: list[SummaryRecord] = []
    source_type = "original_csv"
    metadata: dict[str, Any] = {}

    meta_path = path / "archival_metadata.json"
    if meta_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            source_type = metadata.get("source_type", "verified_archival_reconstruction")
        except Exception:
            source_type = "verified_archival_reconstruction"

    is_archival = source_type == "verified_archival_reconstruction"

    raw_csv = path / "benchmark_raw_results.csv"
    if raw_csv.is_file():
        raw_records = load_raw_csv(raw_csv)
        if is_archival:
            for r in raw_records:
                r.is_archival = True

    sum_csv = path / "benchmark_summary.csv"
    if sum_csv.is_file():
        summaries = load_summary_csv(sum_csv)

    return ExperimentData(
        name=name,
        dir_path=path,
        raw_records=raw_records,
        summaries=summaries,
        source_type=source_type,
        metadata=metadata,
    )


def discover_experiments(results_dir: Path | str) -> dict[str, ExperimentData]:
    """Scan results directory and return discovered experiments.

    Maps:
      - "exp_a": Experiment A Problem Size Scaling
      - "exp_b": Experiment B Connectivity Sensitivity
      - "exp_d": Experiment D Generalization / Robustness

    Also checks direct directory if results_dir contains CSV files directly.
    """
    base = Path(results_dir)
    discovered: dict[str, ExperimentData] = {}

    if not base.exists():
        return discovered

    # Canonical mapping checks
    candidates = {
        "exp_a": ["experiment_a_scaling", "exp_a_scaling", "scaling", "exp_a"],
        "exp_b": ["experiment_b_connectivity", "exp_b_connectivity", "connectivity", "exp_b"],
        "exp_d": ["experiment_d_generalization", "exp_d_generalization", "generalization", "exp_d"],
    }

    for key, aliases in candidates.items():
        for alias in aliases:
            target = base / alias
            if target.is_dir() and (target / "benchmark_raw_results.csv").is_file():
                discovered[key] = load_experiment_dir(target, exp_name=key)
                break

    # If not found via aliases, scan all immediate subdirectories
    if base.is_dir():
        for sub in base.iterdir():
            if sub.is_dir() and (sub / "benchmark_raw_results.csv").is_file():
                sub_name = sub.name.lower()
                matched_key = None
                for key, aliases in candidates.items():
                    if any(a in sub_name for a in aliases):
                        matched_key = key
                        break
                target_key = matched_key or sub.name
                if target_key not in discovered:
                    discovered[target_key] = load_experiment_dir(sub, exp_name=target_key)

    # Check if base itself is an experiment directory
    if (base / "benchmark_raw_results.csv").is_file() and not discovered:
        discovered["experiment"] = load_experiment_dir(base, exp_name=base.name)

    return discovered
