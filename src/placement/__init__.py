"""Placement package containing data models, geometric constraints, engine, and HPWL."""

from .benchmarks import (
    BenchmarkRunResult,
    BenchmarkSummary,
    get_default_benchmark_suite,
    run_benchmark_suite,
)
from .constraints import (
    DuplicateMacroError,
    OutOfBoundsError,
    OverlapError,
    PlacementError,
    UnknownMacroError,
    ValidationReport,
    find_overlapping_macro,
    is_within_boundary,
    macros_overlap,
    validate_macro_boundary,
    validate_placement,
)
from .engine import (
    PlacementEngine,
    action_to_coord,
    coord_to_action,
)
from .hpwl import (
    IncompletePlacementError,
    calculate_hpwl,
    calculate_net_hpwls,
    get_macro_center,
)
from .jobs import (
    default_greedy_placement_runner,
    default_random_placement_runner,
    default_sa_placement_runner,
    get_registered_algorithms,
    register_algorithm,
)
from .models import (
    Macro,
    Net,
    Netlist,
    PlacedMacro,
    Placement,
    placement_to_dict,
)

__all__ = [
    "BenchmarkRunResult",
    "BenchmarkSummary",
    "DuplicateMacroError",
    "default_greedy_placement_runner",
    "default_random_placement_runner",
    "default_sa_placement_runner",
    "IncompletePlacementError",
    "Macro",
    "Net",
    "Netlist",
    "OutOfBoundsError",
    "OverlapError",
    "PlacedMacro",
    "Placement",
    "PlacementEngine",
    "PlacementError",
    "UnknownMacroError",
    "ValidationReport",
    "action_to_coord",
    "calculate_hpwl",
    "calculate_net_hpwls",
    "coord_to_action",
    "find_overlapping_macro",
    "get_default_benchmark_suite",
    "get_macro_center",
    "get_registered_algorithms",
    "is_within_boundary",
    "macros_overlap",
    "placement_to_dict",
    "register_algorithm",
    "run_benchmark_suite",
    "validate_macro_boundary",
    "validate_placement",
]
