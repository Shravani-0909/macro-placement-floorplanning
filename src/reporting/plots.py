"""Research-Grade Plot Generation for Macro Placement Floorplanning.

Generates Matplotlib publication figures:
- Fig 1: Exp A Objective Comparison (grouped bar chart with std error bars)
- Fig 2: Exp A Problem Size Scaling (macro count vs objective line chart)
- Fig 3: Exp B Connectivity Sensitivity (random vs clustered topology)
- Fig 4: Exp D Generalization Gap (train vs unseen instance vs unseen topology)
- Fig 5: Objective Decomposition (HPWL vs Congestion analysis)
"""

from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Sequence

# Ensure headless matplotlib backend and safe writable temp cache
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.reporting.models import GeneralizationGap, RunRecord, SummaryRecord
from src.reporting.statistics import summarize_records

# Consistent algorithm color palette
ALGO_COLORS = {
    "random": "#7f8c8d",  # Cool gray
    "greedy": "#2980b9",  # Steel blue
    "sa": "#27ae60",      # Emerald green
    "rl": "#e74c3c",      # Crimson red
}

ALGO_LABELS = {
    "random": "Random",
    "greedy": "Greedy",
    "sa": "Simulated Annealing",
    "rl": "PPO (RL)",
}


def _setup_figure_style() -> None:
    """Set publication style parameters."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def plot_experiment_a_comparison(records: Sequence[RunRecord], output_path: Path | str) -> Path:
    """Figure 1: Experiment A Objective Comparison (Grouped Bar Chart with error bars)."""
    _setup_figure_style()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    summaries = summarize_records(records)
    if not summaries:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Experiment A: No Data Available", ha="center", va="center")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out

    # Determine unique benchmarks and algorithms
    benchmarks = []
    for r in records:
        if r.netlist_id not in benchmarks:
            benchmarks.append(r.netlist_id)

    algorithms = []
    for r in records:
        if r.algorithm not in algorithms:
            algorithms.append(r.algorithm)

    summary_map = {(s.algorithm, s.netlist_id): s for s in summaries}

    x = np.arange(len(benchmarks))
    n_algos = len(algorithms)
    width = 0.8 / max(n_algos, 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, algo in enumerate(algorithms):
        means = []
        stds = []
        for b in benchmarks:
            s = summary_map.get((algo, b))
            if s and s.mean_objective is not None:
                means.append(s.mean_objective)
                stds.append(s.std_objective or 0.0)
            else:
                means.append(0.0)
                stds.append(0.0)

        offset = (idx - (n_algos - 1) / 2) * width
        color = ALGO_COLORS.get(algo.lower(), "#34495e")
        label = ALGO_LABELS.get(algo.lower(), algo.upper())
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=label,
            color=color,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.8,
        )

    ax.set_xlabel("Benchmark Instance")
    ax.set_ylabel("Placement Objective J(P) = HPWL + Congestion\n(Lower is Better)")
    ax.set_title("Experiment A: Placement Objective Comparison Across Benchmarks")
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", " ").title() for b in benchmarks])
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_experiment_a_scaling(records: Sequence[RunRecord], output_path: Path | str) -> Path:
    """Figure 2: Experiment A Problem Size Scaling (Macro Count vs Objective Line Chart)."""
    _setup_figure_style()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    summaries = summarize_records(records)
    if not summaries:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Experiment A Scaling: No Data Available", ha="center", va="center")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out

    # Group by algorithm and macro count
    algo_macro_obj: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.success and r.is_legal and r.objective is not None:
            algo_macro_obj[r.algorithm][r.macro_count].append(r.objective)

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"random": "o", "greedy": "s", "sa": "^", "rl": "D"}

    for algo, macro_dict in algo_macro_obj.items():
        macros = sorted(macro_dict.keys())
        means = [float(np.mean(macro_dict[m])) for m in macros]
        stds = [float(np.std(macro_dict[m], ddof=1)) if len(macro_dict[m]) > 1 else 0.0 for m in macros]

        color = ALGO_COLORS.get(algo.lower(), "#34495e")
        label = ALGO_LABELS.get(algo.lower(), algo.upper())
        marker = markers.get(algo.lower(), "o")

        ax.errorbar(
            macros,
            means,
            yerr=stds,
            marker=marker,
            markersize=7,
            linewidth=2,
            capsize=4,
            label=label,
            color=color,
        )

    ax.set_xlabel("Number of Macros (|M|)")
    ax.set_ylabel("Placement Objective J(P)\n(Lower is Better)")
    ax.set_title("Experiment A: Objective Scaling with Problem Size")
    ax.set_xticks([10, 20, 30] if any(m in [10, 20, 30] for m in [10, 20, 30]) else sorted(list(set(r.macro_count for r in records))))
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_experiment_b_sensitivity(records: Sequence[RunRecord], output_path: Path | str) -> Path:
    """Figure 3: Experiment B Connectivity Sensitivity (Random vs Clustered Topology)."""
    _setup_figure_style()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    summaries = summarize_records(records)
    if not summaries:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Experiment B: Connectivity Data Not Found in Results", ha="center", va="center")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out

    summary_map = {(s.algorithm, s.netlist_id): s for s in summaries}
    algorithms = sorted(list({s.algorithm for s in summaries}))

    rand_means = []
    clust_means = []
    algos_present = []

    for algo in algorithms:
        r_s = None
        c_s = None
        for (a, nid), s in summary_map.items():
            if a == algo:
                if "rand" in nid.lower():
                    r_s = s
                elif "clust" in nid.lower():
                    c_s = s
        if r_s and c_s and r_s.mean_objective is not None and c_s.mean_objective is not None:
            rand_means.append(r_s.mean_objective)
            clust_means.append(c_s.mean_objective)
            algos_present.append(algo)

    if not algos_present:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Experiment B: Topologies Incomplete", ha="center", va="center")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out

    x = np.arange(len(algos_present))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width / 2, rand_means, width, label="Random Topology", color="#e67e22", alpha=0.85, edgecolor="black")
    rects2 = ax.bar(x + width / 2, clust_means, width, label="Clustered Topology", color="#8e44ad", alpha=0.85, edgecolor="black")

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Placement Objective J(P)\n(Lower is Better)")
    ax.set_title("Experiment B: Objective Sensitivity to Netlist Topology (Random vs Clustered)")
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS.get(a.lower(), a.upper()) for a in algos_present])
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_experiment_d_generalization(
    records: Sequence[RunRecord],
    output_path: Path | str,
) -> Path:
    """Figure 4: Experiment D Generalization Gap (Train vs Unseen Instance vs Unseen Topology)."""
    _setup_figure_style()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    summaries = summarize_records(records)
    if not summaries:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Experiment D: Generalization Data Not Found in Results", ha="center", va="center")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out

    summary_map = {(s.algorithm, s.netlist_id): s for s in summaries}
    algorithms = sorted(list({s.algorithm for s in summaries}))

    train_vals = []
    unseen_inst_vals = []
    unseen_topo_vals = []
    algos_present = []

    for algo in algorithms:
        train_s = None
        inst_s = None
        topo_s = None
        for (a, nid), s in summary_map.items():
            if a == algo:
                if "train" in nid.lower():
                    train_s = s
                elif "unseen_inst" in nid.lower():
                    inst_s = s
                elif "unseen_topo" in nid.lower():
                    topo_s = s
        if train_s and inst_s and topo_s and train_s.mean_objective is not None:
            train_vals.append(train_s.mean_objective or 0.0)
            unseen_inst_vals.append(inst_s.mean_objective or 0.0)
            unseen_topo_vals.append(topo_s.mean_objective or 0.0)
            algos_present.append(algo)

    if not algos_present:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Experiment D: Generalization Conditions Incomplete", ha="center", va="center")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out

    x = np.arange(len(algos_present))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, train_vals, width, label="Training Instance (Random)", color="#27ae60", alpha=0.85, edgecolor="black")
    ax.bar(x, unseen_inst_vals, width, label="Unseen Instance (Holdout)", color="#2980b9", alpha=0.85, edgecolor="black")
    ax.bar(x + width, unseen_topo_vals, width, label="Unseen Topology (Clustered)", color="#e74c3c", alpha=0.85, edgecolor="black")

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Placement Objective J(P)\n(Lower is Better)")
    ax.set_title("Experiment D: Generalization Transfer to Unseen Conditions")
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS.get(a.lower(), a.upper()) for a in algos_present])
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_objective_decomposition(records: Sequence[RunRecord], output_path: Path | str) -> Path:
    """Figure 5: Objective Decomposition (HPWL vs Congestion across algorithms)."""
    _setup_figure_style()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    summaries = summarize_records(records)
    if not summaries:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Objective Decomposition: No Data Available", ha="center", va="center")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out

    # Check if HPWL/congestion metrics exist in the dataset
    has_decomp = any(r.hpwl is not None and r.congestion is not None for r in records)
    if not has_decomp:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "Objective Decomposition: HPWL and Congestion splits unavailable for archival datasets\n(Composite placement objective is preserved)",
            ha="center",
            va="center",
            fontsize=10,
            color="#555555",
        )
        ax.axis("off")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out

    # Filter to instances with non-zero metrics
    benchmarks = []
    for r in records:
        if r.netlist_id not in benchmarks:
            benchmarks.append(r.netlist_id)

    algorithms = []
    for r in records:
        if r.algorithm not in algorithms:
            algorithms.append(r.algorithm)

    summary_map = {(s.algorithm, s.netlist_id): s for s in summaries}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(benchmarks))
    n_algos = len(algorithms)
    width = 0.8 / max(n_algos, 1)

    for idx, algo in enumerate(algorithms):
        hpwl_means = []
        cong_means = []
        for b in benchmarks:
            s = summary_map.get((algo, b))
            hpwl_means.append(s.mean_hpwl if (s and s.mean_hpwl is not None) else 0.0)
            cong_means.append(s.mean_congestion if (s and s.mean_congestion is not None) else 0.0)

        offset = (idx - (n_algos - 1) / 2) * width
        color = ALGO_COLORS.get(algo.lower(), "#34495e")
        label = ALGO_LABELS.get(algo.lower(), algo.upper())

        ax1.bar(x + offset, hpwl_means, width, label=label, color=color, alpha=0.85, edgecolor="black")
        ax2.bar(x + offset, cong_means, width, label=label, color=color, alpha=0.85, edgecolor="black")

    ax1.set_title("HPWL Wirelength (α = 1.0)")
    ax1.set_xlabel("Benchmark")
    ax1.set_ylabel("Mean HPWL (Wirelength)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([b.replace("_", "\n") for b in benchmarks], fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8)

    ax2.set_title("RUDY Congestion (β = 1.0)")
    ax2.set_xlabel("Benchmark")
    ax2.set_ylabel("Mean Peak Congestion")
    ax2.set_xticks(x)
    ax2.set_xticklabels([b.replace("_", "\n") for b in benchmarks], fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8)

    fig.suptitle("Figure 5: Objective Component Decomposition (HPWL vs Congestion)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_all_figures(
    output_dir: Path | str,
    records_a: Sequence[RunRecord] | None = None,
    records_b: Sequence[RunRecord] | None = None,
    records_d: Sequence[RunRecord] | None = None,
) -> dict[str, Path]:
    """Generate all 5 research figures into output_dir.

    Returns dict mapping figure key to generated file path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_paths = {}

    # Fig 1 & Fig 2: Experiment A
    a_recs = records_a or []
    fig_paths["fig1"] = plot_experiment_a_comparison(a_recs, out_dir / "fig1_exp_a_objective_comparison.png")
    fig_paths["fig2"] = plot_experiment_a_scaling(a_recs, out_dir / "fig2_exp_a_scaling.png")

    # Fig 3: Experiment B
    b_recs = records_b or []
    fig_paths["fig3"] = plot_experiment_b_sensitivity(b_recs, out_dir / "fig3_exp_b_connectivity_sensitivity.png")

    # Fig 4: Experiment D
    d_recs = records_d or []
    fig_paths["fig4"] = plot_experiment_d_generalization(d_recs, out_dir / "fig4_exp_d_generalization.png")

    # Fig 5: Objective Decomposition (using Exp A or best available with valid HPWL/congestion)
    decomp_candidates = [r for r in (records_a or []) if r.hpwl is not None and r.congestion is not None]
    if not decomp_candidates:
        decomp_candidates = [r for r in ((records_b or []) + (records_d or [])) if r.hpwl is not None and r.congestion is not None]
    decomp_recs = decomp_candidates if decomp_candidates else (records_a or [])
    fig_paths["fig5"] = plot_objective_decomposition(decomp_recs, out_dir / "fig5_objective_decomposition.png")

    return fig_paths
