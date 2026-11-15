"""Data models for Macro Placement Research Reporting.

Defines typed dataclasses for raw benchmark records, summary metrics,
statistical comparisons, generalization gaps, and report datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    """Represents a single benchmark placement execution."""

    algorithm: str
    netlist_id: str
    grid_w: int
    grid_h: int
    seed: int
    success: bool
    is_legal: bool
    is_complete: bool
    hpwl: float | None = None
    congestion: float | None = None
    objective: float | None = None
    runtime_sec: float = 0.0
    eval_time_sec: float | None = None
    train_time_sec: float | None = None
    train_timesteps: int | None = None
    checkpoint_path: str | None = None
    error: str | None = None
    is_archival: bool = False

    @property
    def macro_count(self) -> int:
        """Infer macro count from netlist ID or naming conventions."""
        nid = self.netlist_id.lower()
        if "10" in nid:
            return 10
        if "20" in nid:
            return 20
        if "30" in nid:
            return 30
        if "12" in nid or "exp_b" in nid or "exp_d" in nid:
            return 12
        if "4" in nid:
            return 4
        if "8" in nid:
            return 8
        return 10

    @property
    def is_rl(self) -> bool:
        """Check if algorithm is RL/PPO."""
        return self.algorithm.lower() == "rl"

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "algorithm": self.algorithm,
            "netlist_id": self.netlist_id,
            "grid_w": self.grid_w,
            "grid_h": self.grid_h,
            "seed": self.seed,
            "success": self.success,
            "is_legal": self.is_legal,
            "is_complete": self.is_complete,
            "hpwl": self.hpwl,
            "congestion": self.congestion,
            "objective": self.objective,
            "runtime_sec": self.runtime_sec,
            "eval_time_sec": self.eval_time_sec,
            "train_time_sec": self.train_time_sec,
            "train_timesteps": self.train_timesteps,
            "checkpoint_path": self.checkpoint_path,
            "error": self.error,
            "is_archival": self.is_archival,
        }


@dataclass
class SummaryRecord:
    """Aggregated statistical summary for an algorithm on a benchmark instance."""

    algorithm: str
    netlist_id: str
    total_runs: int
    success_count: int
    success_rate: float
    mean_hpwl: float | None = None
    median_hpwl: float | None = None
    min_hpwl: float | None = None
    max_hpwl: float | None = None
    std_hpwl: float | None = None
    mean_congestion: float | None = None
    median_congestion: float | None = None
    std_congestion: float | None = None
    mean_objective: float | None = None
    median_objective: float | None = None
    std_objective: float | None = None
    mean_runtime_sec: float = 0.0
    median_runtime_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "algorithm": self.algorithm,
            "netlist_id": self.netlist_id,
            "total_runs": self.total_runs,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "mean_hpwl": self.mean_hpwl,
            "median_hpwl": self.median_hpwl,
            "min_hpwl": self.min_hpwl,
            "max_hpwl": self.max_hpwl,
            "std_hpwl": self.std_hpwl,
            "mean_congestion": self.mean_congestion,
            "median_congestion": self.median_congestion,
            "std_congestion": self.std_congestion,
            "mean_objective": self.mean_objective,
            "median_objective": self.median_objective,
            "std_objective": self.std_objective,
            "mean_runtime_sec": self.mean_runtime_sec,
            "median_runtime_sec": self.median_runtime_sec,
        }


@dataclass
class GeneralizationGap:
    """Experiment D generalization gap metrics for an algorithm."""

    algorithm: str
    train_netlist_id: str
    test_netlist_id: str
    test_type: str  # "unseen_instance" or "unseen_topology"
    train_mean_objective: float
    test_mean_objective: float
    gap_delta: float  # test - train (positive = worse objective)
    gap_percentage: float  # (test - train) / train * 100

    @property
    def interpretation(self) -> str:
        """Human-readable interpretation of generalization gap."""
        if abs(self.gap_percentage) < 2.0:
            return "Flat transfer"
        if self.gap_percentage > 0:
            return f"Degradation (+{self.gap_percentage:.1f}%)"
        return f"Improvement ({self.gap_percentage:.1f}%)"


@dataclass
class TopologySensitivity:
    """Experiment B connectivity topology sensitivity metrics for an algorithm."""

    algorithm: str
    random_mean_hpwl: float
    clustered_mean_hpwl: float
    hpwl_delta: float
    hpwl_pct: float
    random_mean_congestion: float
    clustered_mean_congestion: float
    congestion_delta: float
    congestion_pct: float
    random_mean_obj: float
    clustered_mean_obj: float
    obj_delta: float
    obj_pct: float


@dataclass
class ProvenanceInfo:
    """Reproducibility and environment provenance metadata."""

    timestamp: str
    git_commit: str
    python_version: str
    platform: str
    numpy_version: str = "Not available"
    gymnasium_version: str = "Not available"
    sb3_version: str = "Not available"
    torch_version: str = "Not available"
    matplotlib_version: str = "Not available"
    jinja2_version: str = "Not available"
    seeds: list[int] = field(default_factory=list)
    timesteps: int | None = 10000
    reward_scale: float | None = 100.0
    alpha: float = 1.0
    beta: float = 1.0
    source_dirs: list[str] = field(default_factory=list)
    data_sources: dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentData:
    """Container for raw and summarized data of a single experiment."""

    name: str
    dir_path: Path
    raw_records: list[RunRecord] = field(default_factory=list)
    summaries: list[SummaryRecord] = field(default_factory=list)
    source_type: str = "original_csv"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_archival(self) -> bool:
        """Check if experiment data was loaded from verified archival reconstruction."""
        return self.source_type == "verified_archival_reconstruction"

    @property
    def algorithms(self) -> list[str]:
        """Ordered list of unique algorithms present."""
        seen = []
        for r in self.raw_records:
            if r.algorithm not in seen:
                seen.append(r.algorithm)
        return seen

    @property
    def netlist_ids(self) -> list[str]:
        """Ordered list of unique netlist IDs present."""
        seen = []
        for r in self.raw_records:
            if r.netlist_id not in seen:
                seen.append(r.netlist_id)
        return seen


@dataclass
class ReportData:
    """Complete aggregated dataset for research report rendering."""

    exp_a: ExperimentData | None = None
    exp_b: ExperimentData | None = None
    exp_d: ExperimentData | None = None
    generalization_gaps: list[GeneralizationGap] = field(default_factory=list)
    topology_sensitivities: list[TopologySensitivity] = field(default_factory=list)
    provenance: ProvenanceInfo = field(default_factory=lambda: ProvenanceInfo(
        timestamp="", git_commit="Not available", python_version="", platform=""
    ))
    generated_at: str = ""
