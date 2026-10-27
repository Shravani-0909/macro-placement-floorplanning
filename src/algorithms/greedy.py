"""Greedy Placement baseline for macro placement and floorplanning.

This module provides the Greedy Placement algorithm baseline. It places macros
sequentially according to a deterministic order (largest area first, with
stable tie-breaking by macro ID). For each macro, it evaluates all legal candidate
anchors provided by Zoya's shared grid environment (`GridEnv.legal_mask`) and
selects the position minimizing incremental wirelength (`GridEnv.hpwl_delta_move`).

NOTE:
    Greedy placement is locally optimized for incremental HPWL at each step.
    It is a heuristic baseline and is not guaranteed to find a feasible completion
    for every netlist (it can reach a dead end if remaining macros cannot fit legally)
    nor is it guaranteed to minimize the combined objective J(P).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, NamedTuple, Sequence
import numpy as np

from src.placement.objective import EvaluationResult, evaluate_placement

# Dynamic import for Zoya's shared GridEnv once implemented
try:
    from src.placement.grid import GridEnv  # type: ignore[import-untyped]
except ImportError:
    GridEnv = None


class GreedyPlacementResult(NamedTuple):
    """Result of greedy macro placement.

    Attributes:
        placement: Placed macro positions mapping macro_id -> (x, y), or partial placement
            if placement reached a dead end, or None if initialization failed.
        success: True if all macros were legally placed, False otherwise.
        hpwl: Half-Perimeter Wirelength (float), or None if placement failed.
        congestion: RUDY congestion score proxy (float), or None if placement failed.
        objective: Combined placement objective J(P) = alpha * HPWL + beta * congestion,
            or None if placement failed.
    """

    placement: Any | None
    success: bool
    hpwl: float | None
    congestion: float | None
    objective: float | None


PlacementResult = GreedyPlacementResult


def _extract_macros(netlist: Any) -> dict[Any, tuple[int, int]]:
    """Extract mapping of macro_id -> (width, height) from netlist."""
    if isinstance(netlist, Mapping):
        raw_macros = netlist.get("macros", {})
    else:
        raw_macros = getattr(netlist, "macros", {})

    macro_dict: dict[Any, tuple[int, int]] = {}

    if isinstance(raw_macros, Mapping):
        for m_id, m_info in raw_macros.items():
            if isinstance(m_info, (tuple, list)) and len(m_info) == 2:
                w, h = m_info
            elif isinstance(m_info, Mapping):
                w = m_info.get("w", m_info.get("width"))
                h = m_info.get("h", m_info.get("height"))
            else:
                w = getattr(m_info, "w", getattr(m_info, "width", None))
                h = getattr(m_info, "h", getattr(m_info, "height", None))
            if w is not None and h is not None:
                macro_dict[m_id] = (int(w), int(h))
    elif isinstance(raw_macros, Sequence):
        for item in raw_macros:
            if isinstance(item, Mapping):
                m_id = item.get("id")
                w = item.get("w", item.get("width"))
                h = item.get("h", item.get("height"))
            else:
                m_id = getattr(item, "id", None)
                w = getattr(item, "w", getattr(item, "width", None))
                h = getattr(item, "h", getattr(item, "height", None))
            if m_id is not None and w is not None and h is not None:
                macro_dict[m_id] = (int(w), int(h))

    return macro_dict


def _extract_placed_ids(placement: Any) -> set[Any]:
    """Extract set of placed macro IDs from placement mapping or object."""
    if placement is None:
        return set()
    if hasattr(placement, "positions"):
        raw_pos = placement.positions
    elif isinstance(placement, Mapping):
        raw_pos = placement
    else:
        try:
            raw_pos = {k: placement[k] for k in placement}
        except (TypeError, AttributeError):
            raw_pos = {}

    if isinstance(raw_pos, Mapping):
        return {k for k, v in raw_pos.items() if v is not None}
    return set()


def _get_sorted_macro_ids(macros: dict[Any, tuple[int, int]]) -> list[Any]:
    """Determine deterministic macro placement ordering: largest area first, tie-break by ID."""
    return sorted(
        macros.keys(),
        key=lambda m_id: (-macros[m_id][0] * macros[m_id][1], str(m_id)),
    )


def greedy_placement(
    netlist: Any,
    alpha: float = 1.0,
    beta: float = 1.0,
    *,
    env_class: Callable[..., Any] | None = None,
) -> GreedyPlacementResult:
    """Execute Greedy Placement using incremental HPWL minimization on GridEnv.

    Algorithm:
    1. Instantiate the grid environment `GridEnv(netlist)`.
    2. Sort macros deterministically by largest area first, tie-breaking by macro ID.
    3. For each macro:
       a. Retrieve the legal anchor mask `mask = env.legal_mask(macro_id)` of shape (H, W).
       b. Enumerate legal positions in deterministic row-major order (y then x).
       c. Query `env.hpwl_delta_move(macro_id, x, y)` for each legal anchor.
       d. Select the anchor with minimum incremental HPWL (tie-break: smaller y, then smaller x).
       e. Commit the placement via `env.place(macro_id, x, y)`.
       f. If no legal position exists, terminate early with a documented failure result.
    4. Retrieve `placement = env.placement()`.
    5. Evaluate the combined objective J(P) = alpha * HPWL + beta * C_score.

    Parameters
    ----------
    netlist : Any
        Circuit netlist providing grid dimensions, macros, and nets.
    alpha : float, default=1.0
        Non-negative weight for HPWL in the combined objective.
    beta : float, default=1.0
        Non-negative weight for Congestion in the combined objective.
    env_class : Callable[..., Any] | None, optional
        Custom GridEnv class or factory for testing. If None, uses
        `src.placement.grid.GridEnv`.

    Returns
    -------
    GreedyPlacementResult
        NamedTuple containing:
            - placement: final placement mapping or partial placement if failed.
            - success: True if all macros were legally placed, False otherwise.
            - hpwl: Half-Perimeter Wirelength (float or None).
            - congestion: RUDY congestion score (float or None).
            - objective: Combined objective value J(P) (float or None).

    Raises
    ------
    NotImplementedError
        If Zoya's shared engine `src.placement.grid.GridEnv` is not available
        and no `env_class` is provided.
    ValueError
        If alpha or beta are invalid (negative, NaN, infinite).
    """
    for name, val in [("alpha", alpha), ("beta", beta)]:
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"Weight '{name}' must be numeric, got {type(val).__name__}.")
        if not math.isfinite(val):
            raise ValueError(f"Weight '{name}' must be finite, got {val}.")
        if val < 0.0:
            raise ValueError(f"Weight '{name}' must be non-negative, got {val}.")

    flt_alpha = float(alpha)
    flt_beta = float(beta)

    # Determine GridEnv class
    cls = env_class if env_class is not None else GridEnv
    if cls is None:
        raise NotImplementedError(
            "Shared engine module 'src.placement.grid.GridEnv' is not available yet. "
            "Zoya's placement engine must be implemented to use Greedy placement."
        )

    # 1. Instantiate the grid environment
    env = cls(netlist)

    # 2. Extract macros and determine deterministic placement order
    macros = _extract_macros(netlist)
    sorted_macro_ids = _get_sorted_macro_ids(macros)

    # 3. Place macros sequentially
    for m_id in sorted_macro_ids:
        mask = env.legal_mask(m_id)

        if mask is None:
            partial_placement = env.placement() if hasattr(env, "placement") else None
            return GreedyPlacementResult(
                placement=partial_placement,
                success=False,
                hpwl=None,
                congestion=None,
                objective=None,
            )

        # Ensure mask is converted to 2D numpy array
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.ndim != 2 or not np.any(mask_arr):
            # Dead end: no legal position exists for this macro
            partial_placement = env.placement() if hasattr(env, "placement") else None
            return GreedyPlacementResult(
                placement=partial_placement,
                success=False,
                hpwl=None,
                congestion=None,
                objective=None,
            )

        grid_h, grid_w = mask_arr.shape
        best_anchor: tuple[int, int] | None = None
        min_delta = float("inf")

        # Enumerate legal anchors in deterministic row-major order (y then x)
        for y in range(grid_h):
            for x in range(grid_w):
                if mask_arr[y, x]:
                    delta = env.hpwl_delta_move(m_id, x, y)
                    if delta < min_delta:
                        min_delta = delta
                        best_anchor = (x, y)

        if best_anchor is None:
            partial_placement = env.placement() if hasattr(env, "placement") else None
            return GreedyPlacementResult(
                placement=partial_placement,
                success=False,
                hpwl=None,
                congestion=None,
                objective=None,
            )

        # Place macro at the best legal anchor
        place_x, place_y = best_anchor
        env.place(m_id, place_x, place_y)

    # 4. Retrieve complete placement
    final_placement = env.placement() if hasattr(env, "placement") else {}

    # Verify placement completeness
    macro_ids_set = set(macros.keys())
    placed_ids_set = _extract_placed_ids(final_placement)
    if macro_ids_set and not macro_ids_set.issubset(placed_ids_set):
        return GreedyPlacementResult(
            placement=final_placement,
            success=False,
            hpwl=None,
            congestion=None,
            objective=None,
        )

    # 5. Evaluate combined objective
    eval_res: EvaluationResult = evaluate_placement(
        netlist, final_placement, alpha=flt_alpha, beta=flt_beta
    )

    return GreedyPlacementResult(
        placement=final_placement,
        success=True,
        hpwl=eval_res.hpwl,
        congestion=eval_res.congestion,
        objective=eval_res.objective,
    )


greedy = greedy_placement

__all__ = [
    "GreedyPlacementResult",
    "PlacementResult",
    "greedy",
    "greedy_placement",
]
