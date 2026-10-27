"""Random Placement baseline for macro placement and floorplanning.

This module provides the Random Placement algorithm baseline. It delegates
feasible/legal coordinate generation to Zoya's shared engine helper
(`src.placement.legal.random_legal_placement`) to avoid duplicate boundary,
overlap, or collision checking. Upon generating a placement, it evaluates
Half-Perimeter Wirelength (HPWL), RUDY congestion proxy, and the combined
placement objective J(P) = alpha * HPWL + beta * congestion.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, NamedTuple, Sequence
import numpy as np

from src.placement.objective import EvaluationResult, evaluate_placement

# Dynamic import for Zoya's shared legal placement helper once implemented
try:
    from src.placement.legal import random_legal_placement  # type: ignore[import-untyped]
except ImportError:
    random_legal_placement = None


class RandomPlacementResult(NamedTuple):
    """Result of the random macro placement baseline.

    Attributes:
        placement: Placed macro positions mapping macro_id -> (x, y), or None if placement failed.
        success: True if a legal complete placement was found, False otherwise.
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


PlacementResult = RandomPlacementResult


def _extract_macro_ids(netlist: Any) -> set[Any]:
    """Extract the set of macro IDs defined in the netlist."""
    if isinstance(netlist, Mapping):
        raw_macros = netlist.get("macros", {})
    else:
        raw_macros = getattr(netlist, "macros", {})

    if isinstance(raw_macros, Mapping):
        return set(raw_macros.keys())
    elif isinstance(raw_macros, Sequence):
        ids: set[Any] = set()
        for item in raw_macros:
            if isinstance(item, Mapping):
                m_id = item.get("id")
            else:
                m_id = getattr(item, "id", None)
            if m_id is not None:
                ids.add(m_id)
        return ids
    return set()


def _extract_placed_ids(placement: Any) -> set[Any]:
    """Extract the set of placed macro IDs from placement object or mapping."""
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


def random_placement(
    netlist: Any,
    seed: int | None = 42,
    alpha: float = 1.0,
    beta: float = 1.0,
    *,
    legal_placement_fn: Callable[..., Any] | None = None,
) -> RandomPlacementResult:
    """Generate a random legal macro placement and evaluate the combined objective.

    This baseline delegates legal coordinate assignment to Zoya's shared
    engine (`src.placement.legal.random_legal_placement`) to avoid duplicating
    boundary or collision logic, and evaluates the resulting placement using
    the combined objective J(P) = alpha * HPWL + beta * C_score.

    Parameters
    ----------
    netlist : Any
        Circuit netlist providing grid dimensions, macros, and nets.
    seed : int | None, default=42
        Integer seed for reproducible placement generation.
    alpha : float, default=1.0
        Non-negative weight for HPWL in the combined objective.
    beta : float, default=1.0
        Non-negative weight for Congestion in the combined objective.
    legal_placement_fn : Callable[..., Any] | None, optional
        Custom legal placement generator function. If None, uses
        `src.placement.legal.random_legal_placement`.

    Returns
    -------
    RandomPlacementResult
        NamedTuple containing:
            - placement: macro_id -> (x, y) mapping or None if failed.
            - success: True if a legal complete placement was produced, False otherwise.
            - hpwl: Half-Perimeter Wirelength (float or None).
            - congestion: Mean-squared-demand congestion score (float or None).
            - objective: Combined objective value J(P) (float or None).

    Raises
    ------
    NotImplementedError
        If Zoya's shared helper `src.placement.legal.random_legal_placement`
        is not available and no `legal_placement_fn` is provided.
    ValueError
        If seed, alpha, or beta are invalid.
    """
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError(f"Seed must be an integer or None, got {type(seed).__name__}.")

    for name, val in [("alpha", alpha), ("beta", beta)]:
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"Weight '{name}' must be numeric, got {type(val).__name__}.")
        if not math.isfinite(val):
            raise ValueError(f"Weight '{name}' must be finite, got {val}.")
        if val < 0.0:
            raise ValueError(f"Weight '{name}' must be non-negative, got {val}.")

    flt_alpha = float(alpha)
    flt_beta = float(beta)

    # Determine legal placement generator
    placement_fn = legal_placement_fn if legal_placement_fn is not None else random_legal_placement
    if placement_fn is None:
        raise NotImplementedError(
            "Shared engine module 'src.placement.legal.random_legal_placement' is not available yet. "
            "Zoya's placement engine must be implemented to generate legal placements."
        )

    # Generate random legal placement with explicit seed
    placement = placement_fn(netlist, seed=seed)

    # Handle placement failure from helper
    if placement is None:
        return RandomPlacementResult(
            placement=None,
            success=False,
            hpwl=None,
            congestion=None,
            objective=None,
        )

    # Ensure incomplete placements are not silently marked as success
    macro_ids = _extract_macro_ids(netlist)
    placed_ids = _extract_placed_ids(placement)
    if macro_ids and not macro_ids.issubset(placed_ids):
        return RandomPlacementResult(
            placement=placement,
            success=False,
            hpwl=None,
            congestion=None,
            objective=None,
        )

    # Evaluate feasible complete placement
    eval_res: EvaluationResult = evaluate_placement(
        netlist, placement, alpha=flt_alpha, beta=flt_beta
    )

    return RandomPlacementResult(
        placement=placement,
        success=True,
        hpwl=eval_res.hpwl,
        congestion=eval_res.congestion,
        objective=eval_res.objective,
    )


__all__ = [
    "PlacementResult",
    "RandomPlacementResult",
    "random_placement",
]
