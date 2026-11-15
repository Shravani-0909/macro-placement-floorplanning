"""Automated Research Report Generator for Macro Placement Floorplanning."""

from src.reporting.archival import (
    VERIFIED_EXPERIMENT_A_SUMMARIES,
    create_all_archival_results,
    create_experiment_b_archival_data,
    create_experiment_d_archival_data,
)
from src.reporting.loaders import discover_experiments, load_raw_csv, load_summary_csv
from src.reporting.models import (
    ExperimentData,
    GeneralizationGap,
    ProvenanceInfo,
    ReportData,
    RunRecord,
    SummaryRecord,
    TopologySensitivity,
)
from src.reporting.plots import generate_all_figures
from src.reporting.provenance import collect_provenance
from src.reporting.report import generate_research_report
from src.reporting.statistics import (
    compute_connectivity_sensitivities,
    compute_generalization_gaps,
    compute_stats,
    percentage_difference,
    summarize_records,
)

__all__ = [
    "RunRecord",
    "SummaryRecord",
    "GeneralizationGap",
    "TopologySensitivity",
    "ProvenanceInfo",
    "ExperimentData",
    "ReportData",
    "load_raw_csv",
    "load_summary_csv",
    "discover_experiments",
    "compute_stats",
    "percentage_difference",
    "summarize_records",
    "compute_connectivity_sensitivities",
    "compute_generalization_gaps",
    "generate_all_figures",
    "collect_provenance",
    "generate_research_report",
    "create_experiment_b_archival_data",
    "create_experiment_d_archival_data",
    "create_all_archival_results",
]
