"""Statistical Analysis Layer for Macro Placement Experiments.

Computes mean, standard deviation, median, min, max, percentage differences,
topology sensitivities (Experiment B), and generalization gaps (Experiment D).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np

from src.reporting.models import GeneralizationGap, RunRecord, SummaryRecord, TopologySensitivity


def compute_stats(values: Sequence[float]) -> dict[str, float]:
    """Calculate descriptive statistics for a sequence of numeric values.

    Returns dict with keys: 'mean', 'median', 'std', 'min', 'max', 'count'.
    """
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0.0,
        }

    arr = np.array(valid, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": float(len(arr)),
    }


def percentage_difference(a: float | None, b: float | None) -> float:
    """Calculate percentage difference ((a - b) / b) * 100.

    Safe against None, NaN, and zero division.
    """
    if a is None or b is None:
        return 0.0
    if math.isnan(a) or math.isnan(b):
        return 0.0
    if abs(b) < 1e-9:
        if abs(a) < 1e-9:
            return 0.0
        return float("inf") if a > 0 else float("-inf")
    return float(((a - b) / b) * 100.0)


def summarize_records(records: Sequence[RunRecord]) -> list[SummaryRecord]:
    """Group run records by (algorithm, netlist_id) and compute summary metrics."""
    grouped: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for r in records:
        grouped[(r.algorithm, r.netlist_id)].append(r)

    summaries: list[SummaryRecord] = []
    for (algo, netlist_id), runs in grouped.items():
        total_runs = len(runs)
        successes = [r for r in runs if r.success and r.is_legal]
        success_count = len(successes)
        success_rate = (success_count / total_runs) if total_runs > 0 else 0.0

        # Compute metrics from successful runs
        hpwls = [r.hpwl for r in successes if r.hpwl is not None]
        congestions = [r.congestion for r in successes if r.congestion is not None]
        objectives = [r.objective for r in successes if r.objective is not None]
        runtimes = [r.runtime_sec for r in runs if r.runtime_sec is not None]

        hpwl_stats = compute_stats(hpwls) if hpwls else None
        cong_stats = compute_stats(congestions) if congestions else None
        obj_stats = compute_stats(objectives) if objectives else None
        rt_stats = compute_stats(runtimes) if runtimes else {"mean": 0.0, "median": 0.0}

        summaries.append(
            SummaryRecord(
                algorithm=algo,
                netlist_id=netlist_id,
                total_runs=total_runs,
                success_count=success_count,
                success_rate=success_rate,
                mean_hpwl=hpwl_stats["mean"] if hpwl_stats else None,
                median_hpwl=hpwl_stats["median"] if hpwl_stats else None,
                min_hpwl=hpwl_stats["min"] if hpwl_stats else None,
                max_hpwl=hpwl_stats["max"] if hpwl_stats else None,
                std_hpwl=hpwl_stats["std"] if hpwl_stats else None,
                mean_congestion=cong_stats["mean"] if cong_stats else None,
                median_congestion=cong_stats["median"] if cong_stats else None,
                std_congestion=cong_stats["std"] if cong_stats else None,
                mean_objective=obj_stats["mean"] if obj_stats else None,
                median_objective=obj_stats["median"] if obj_stats else None,
                std_objective=obj_stats["std"] if obj_stats else None,
                mean_runtime_sec=rt_stats["mean"],
                median_runtime_sec=rt_stats["median"],
            )
        )

    return summaries


def compute_connectivity_sensitivities(records: Sequence[RunRecord]) -> list[TopologySensitivity]:
    """Compute Experiment B random vs clustered topology sensitivity metrics."""
    summaries = summarize_records(records)
    summary_map = {(s.algorithm, s.netlist_id): s for s in summaries}

    algorithms = sorted(list({s.algorithm for s in summaries}))
    sensitivities: list[TopologySensitivity] = []

    for algo in algorithms:
        # Find random and clustered netlists
        rand_s = None
        clust_s = None
        for (a, nid), s in summary_map.items():
            if a == algo:
                nid_l = nid.lower()
                if "rand" in nid_l:
                    rand_s = s
                elif "clust" in nid_l:
                    clust_s = s

        if rand_s and clust_s and rand_s.mean_objective is not None and clust_s.mean_objective is not None:
            r_hpwl = rand_s.mean_hpwl or 0.0
            c_hpwl = clust_s.mean_hpwl or 0.0
            r_cong = rand_s.mean_congestion or 0.0
            c_cong = clust_s.mean_congestion or 0.0
            r_obj = rand_s.mean_objective or 0.0
            c_obj = clust_s.mean_objective or 0.0

            sensitivities.append(
                TopologySensitivity(
                    algorithm=algo,
                    random_mean_hpwl=r_hpwl,
                    clustered_mean_hpwl=c_hpwl,
                    hpwl_delta=c_hpwl - r_hpwl,
                    hpwl_pct=percentage_difference(c_hpwl, r_hpwl),
                    random_mean_congestion=r_cong,
                    clustered_mean_congestion=c_cong,
                    congestion_delta=c_cong - r_cong,
                    congestion_pct=percentage_difference(c_cong, r_cong),
                    random_mean_obj=r_obj,
                    clustered_mean_obj=c_obj,
                    obj_delta=c_obj - r_obj,
                    obj_pct=percentage_difference(c_obj, r_obj),
                )
            )

    return sensitivities


def compute_generalization_gaps(records: Sequence[RunRecord]) -> list[GeneralizationGap]:
    """Compute Experiment D generalization gaps between train and unseen instances/topologies."""
    summaries = summarize_records(records)
    summary_map = {(s.algorithm, s.netlist_id): s for s in summaries}

    algorithms = sorted(list({s.algorithm for s in summaries}))
    gaps: list[GeneralizationGap] = []

    # Expected netlist IDs
    train_id = "exp_d_train_random"
    unseen_inst_id = "exp_d_test_unseen_inst"
    unseen_topo_id = "exp_d_test_unseen_topo"

    for algo in algorithms:
        train_s = summary_map.get((algo, train_id))
        unseen_inst_s = summary_map.get((algo, unseen_inst_id))
        unseen_topo_s = summary_map.get((algo, unseen_topo_id))

        # Fallback partial matching if exact ID slightly differs
        if not train_s or not unseen_inst_s or not unseen_topo_s:
            for (a, nid), s in summary_map.items():
                if a == algo:
                    if "train" in nid.lower():
                        train_s = s
                    elif "unseen_inst" in nid.lower() or "holdout_inst" in nid.lower():
                        unseen_inst_s = s
                    elif "unseen_topo" in nid.lower() or "holdout_topo" in nid.lower():
                        unseen_topo_s = s

        if train_s and train_s.mean_objective is not None:
            train_obj = train_s.mean_objective

            if unseen_inst_s and unseen_inst_s.mean_objective is not None:
                inst_obj = unseen_inst_s.mean_objective
                gaps.append(
                    GeneralizationGap(
                        algorithm=algo,
                        train_netlist_id=train_s.netlist_id,
                        test_netlist_id=unseen_inst_s.netlist_id,
                        test_type="unseen_instance",
                        train_mean_objective=train_obj,
                        test_mean_objective=inst_obj,
                        gap_delta=inst_obj - train_obj,
                        gap_percentage=percentage_difference(inst_obj, train_obj),
                    )
                )

            if unseen_topo_s and unseen_topo_s.mean_objective is not None:
                topo_obj = unseen_topo_s.mean_objective
                gaps.append(
                    GeneralizationGap(
                        algorithm=algo,
                        train_netlist_id=train_s.netlist_id,
                        test_netlist_id=unseen_topo_s.netlist_id,
                        test_type="unseen_topology",
                        train_mean_objective=train_obj,
                        test_mean_objective=topo_obj,
                        gap_delta=topo_obj - train_obj,
                        gap_percentage=percentage_difference(topo_obj, train_obj),
                    )
                )

    return gaps
