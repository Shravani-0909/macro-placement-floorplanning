"""RUDY Congestion Estimation and Combined Placement Objective.

===============================================================================
Module-Level Integration Note (Shared Engine Integration)
===============================================================================
This module integrates directly with the shared placement engine:
    - `src.placement.models` (`Netlist`, `Placement`, `Macro`, `Net`, `PlacedMacro`)
    - `src.placement.hpwl` (`calculate_hpwl`, `get_macro_center`)

Extraction helpers accept both canonical engine models (e.g. `list[PlacedMacro]`
placements and `Netlist` instances) and duck-typed/dictionary data structures,
preserving full backward compatibility while delegating wirelength and geometry
to the canonical engine implementations. Public APIs:
    - `rudy_map(netlist, placement)`
    - `congestion_score(netlist, placement)`
    - `evaluate_placement(netlist, placement, alpha, beta)`
    - `EvaluationResult`

===============================================================================
Algorithmic & Mathematical Specification
===============================================================================
1. Pin Coordinates:
   For a placed macro covering [x, x + w) x [y, y + h), the pin center is
   located at (x + w/2, y + h/2). These centers may lie on half-cell coordinates
   (e.g., odd dimensions w=1, h=1 yield center (0.5, 0.5)).

2. RUDY Routing Demand Density:
   For an eligible net with bounding box of pin centers [x_min, x_max] x [y_min, y_max],
   bounding box width is W = x_max - x_min and height is H = y_max - y_min.
   The net wirelength is HPWL = W + H.
   RUDY density is computed as:
       density = net.weight * (W + H) / (W_eff * H_eff)
   where W_eff = 1.0 if W == 0.0 else W, and H_eff = 1.0 if H == 0.0 else H.

3. Degenerate Bounding Boxes & Corridors:
   When pin centers share the same x-coordinate (W = 0) or y-coordinate (H = 0),
   the geometric area W * H is 0. To prevent division by zero and model wire
   routing along discrete tracks, any exactly degenerate axis is regularized using
   a unit-cell dimension: W_eff = 1.0 (if W == 0) and H_eff = 1.0 (if H == 0).
   - If H == 0 (same row, y_min == y_max == y_0), the routing is mapped to a
     1-cell-wide corridor at cell row y_cell = min(int(math.floor(y_0)), grid_h - 1),
     allocating unit height overlap (delta_y = 1.0) to row y_cell.
   - If W == 0 (same column, x_min == x_max == x_0), the routing is mapped to a
     1-cell-wide corridor at cell column x_cell = min(int(math.floor(x_0)), grid_w - 1),
     allocating unit width overlap (delta_x = 1.0) to column x_cell.
   - If both W == 0 and H == 0 (all pins share identical center), total HPWL is 0,
     and routing demand is 0.
   For non-degenerate axes (including positive sub-unit fractional dimensions like
   W = 0.5), W_eff = W, ensuring that accumulated demand over the continuous
   intersection cells exactly conserves total wirelength: sum(demand) == weight * HPWL.

4. Cell Intersection Rule:
   For non-degenerate dimensions, the continuous bounding box [min_coord, max_coord]
   intersects each grid cell [c, c + 1] with overlap length:
       delta_c = max(0.0, min(float(c + 1), max_coord) - max(float(c), min_coord))
   Demand added to cell (y, x) is:
       demand[y, x] += density * delta_x * delta_y
   This exact geometric intersection preserves wirelength without arbitrary rounding,
   correctly accounts for half-cell coordinates, keeps all demand within the grid,
   and ensures that the sum of demand over the grid equals the weighted HPWL.

5. Congestion Score Proxy:
   C_score = mean(demand ** 2) = (1 / (grid_w * grid_h)) * sum(demand[y, x] ** 2)
   Hotspots with concentrated routing demand are penalized quadratically.
   Note: This is a project-level proxy derived from RUDY wire density, NOT a
   measurement of actual routed congestion, detailed DRC violations, or proof
   of routability.

6. Combined Placement Objective:
   J(P) = alpha * HPWL(P) + beta * C_score(P)
   The research team must select and freeze fixed (alpha, beta) hyperparameters
   prior to conducting benchmarking experiments so that comparisons between
   Random, Greedy, Simulated Annealing, and Reinforcement Learning are valid
   and unbiased. Do not dynamically normalize or tune weights per placement.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, NamedTuple, Sequence
import numpy as np

from src.placement.hpwl import calculate_hpwl, get_macro_center
from src.placement.models import Netlist, PlacedMacro, Placement, placement_to_dict


class EvaluationResult(NamedTuple):
    """Result of placement evaluation containing HPWL, congestion, and combined objective."""

    hpwl: float
    congestion: float
    objective: float


# =============================================================================
# Temporary Internal Data Extraction Helpers
# =============================================================================
# These helpers abstract access to netlist and placement data structures
# (supporting dicts, dataclasses, or custom objects) until Zoya's shared models
# and engine functions are implemented.


def _extract_grid_dimensions(netlist: Any) -> tuple[int, int]:
    """Extract and validate (grid_w, grid_h) from netlist."""
    if isinstance(netlist, Mapping):
        grid_w = netlist.get("grid_w")
        grid_h = netlist.get("grid_h")
    else:
        grid_w = getattr(netlist, "grid_w", None)
        grid_h = getattr(netlist, "grid_h", None)

    if grid_w is None or grid_h is None:
        raise ValueError("Netlist must provide 'grid_w' and 'grid_h'.")

    if (
        isinstance(grid_w, bool)
        or isinstance(grid_h, bool)
        or not isinstance(grid_w, (int, float, np.integer, np.floating))
        or not isinstance(grid_h, (int, float, np.integer, np.floating))
    ):
        raise ValueError("Grid dimensions must be numeric.")

    if not math.isfinite(grid_w) or not math.isfinite(grid_h):
        raise ValueError("Grid dimensions must be finite.")

    int_w = int(grid_w)
    int_h = int(grid_h)

    if int_w != grid_w or int_h != grid_h:
        raise ValueError("Grid dimensions must be integers.")

    if int_w <= 0 or int_h <= 0:
        raise ValueError(
            f"Grid dimensions must be positive integers, got grid_w={int_w}, grid_h={int_h}."
        )

    return int_w, int_h


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

            macro_dict[m_id] = _validate_macro_dim(m_id, w, h)
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

            if m_id is None:
                raise ValueError("Macro object missing 'id'.")
            macro_dict[m_id] = _validate_macro_dim(m_id, w, h)

    return macro_dict


def _validate_macro_dim(macro_id: Any, w: Any, h: Any) -> tuple[int, int]:
    """Validate positive integer macro dimensions."""
    if w is None or h is None:
        raise ValueError(f"Macro '{macro_id}' missing width or height.")
    if (
        isinstance(w, bool)
        or isinstance(h, bool)
        or not isinstance(w, (int, float, np.integer, np.floating))
        or not isinstance(h, (int, float, np.integer, np.floating))
    ):
        raise ValueError(f"Macro '{macro_id}' dimensions must be numeric.")
    if not math.isfinite(w) or not math.isfinite(h):
        raise ValueError(f"Macro '{macro_id}' dimensions must be finite.")
    int_w = int(w)
    int_h = int(h)
    if int_w != w or int_h != h:
        raise ValueError(f"Macro '{macro_id}' dimensions must be integers.")
    if int_w <= 0 or int_h <= 0:
        raise ValueError(
            f"Macro '{macro_id}' dimensions must be positive integers, got w={int_w}, h={int_h}."
        )
    return int_w, int_h


def _extract_nets(netlist: Any) -> list[tuple[Any, list[Any], float]]:
    """Extract list of (net_id, pin_macro_ids, weight) from netlist."""
    if isinstance(netlist, Mapping):
        raw_nets = netlist.get("nets", [])
    else:
        raw_nets = getattr(netlist, "nets", [])

    nets: list[tuple[Any, list[Any], float]] = []

    if isinstance(raw_nets, Mapping):
        for net_id, net_info in raw_nets.items():
            pins, weight = _parse_net_info(net_id, net_info)
            nets.append((net_id, pins, weight))
    elif isinstance(raw_nets, Sequence):
        for idx, item in enumerate(raw_nets):
            if isinstance(item, (tuple, list)):
                net_id = idx
                pins = list(item[0]) if len(item) > 0 else []
                weight = float(item[1]) if len(item) > 1 else 1.0
                _validate_net_weight(net_id, weight)
                nets.append((net_id, pins, weight))
            else:
                net_id = getattr(item, "id", idx) if not isinstance(item, Mapping) else item.get("id", idx)
                pins, weight = _parse_net_info(net_id, item)
                nets.append((net_id, pins, weight))

    return nets


def _parse_net_info(net_id: Any, net_info: Any) -> tuple[list[Any], float]:
    """Parse pins and weight from a net representation."""
    if isinstance(net_info, Mapping):
        pins = net_info.get("pins", net_info.get("macro_ids", net_info.get("members", [])))
        weight = net_info.get("weight", 1.0)
    else:
        pins = getattr(
            net_info, "pins", getattr(net_info, "macro_ids", getattr(net_info, "members", []))
        )
        weight = getattr(net_info, "weight", 1.0)

    if weight is None:
        weight = 1.0
    _validate_net_weight(net_id, weight)
    return list(pins), float(weight)


def _validate_net_weight(net_id: Any, weight: Any) -> None:
    """Validate non-negative finite numeric net weight."""
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float, np.integer, np.floating))
    ):
        raise ValueError(f"Net '{net_id}' weight must be numeric, got {type(weight).__name__}.")
    if not math.isfinite(weight):
        raise ValueError(f"Net '{net_id}' weight must be finite, got {weight}.")
    if weight < 0.0:
        raise ValueError(f"Net '{net_id}' weight must be non-negative, got {weight}.")


def _extract_placements(
    placement: Any,
    macros: dict[Any, tuple[int, int]],
    grid_w: int,
    grid_h: int,
) -> dict[Any, tuple[int, int]]:
    """Extract and validate placed macro positions: macro_id -> (x, y).

    Supports:
        - Shared engine native Placement: `list[PlacedMacro]` or `Sequence[PlacedMacro]`
        - Mapping representations: `dict[macro_id, (x, y)]` or `dict[macro_id, {"x": x, "y": y}]`
        - Objects exposing a `.positions` attribute
        - Sequences of items with `macro_id`/`id` and `x`, `y` coordinates

    Distinguishes supported incomplete placements from malformed placements.
    Raises ValueError on out-of-bounds, negative, non-integer, duplicate,
    or unknown macro coordinates.
    """
    if placement is None:
        return {}

    # Unwrap .positions if present
    if hasattr(placement, "positions"):
        raw_source = placement.positions
    else:
        raw_source = placement

    raw_items: list[tuple[Any, Any]] = []

    if isinstance(raw_source, Mapping):
        raw_items = list(raw_source.items())
    elif isinstance(raw_source, (list, tuple, Sequence)) and not isinstance(
        raw_source, (str, bytes)
    ):
        for item in raw_source:
            if item is None:
                continue
            if hasattr(item, "macro_id") and hasattr(item, "x") and hasattr(item, "y"):
                raw_items.append((item.macro_id, (item.x, item.y)))
            elif hasattr(item, "id") and hasattr(item, "x") and hasattr(item, "y"):
                raw_items.append((item.id, (item.x, item.y)))
            elif isinstance(item, Mapping) and ("macro_id" in item or "id" in item):
                mid = item.get("macro_id", item.get("id"))
                raw_items.append((mid, (item.get("x"), item.get("y"))))
            elif isinstance(item, (tuple, list)):
                if len(item) == 2:
                    raw_items.append((item[0], item[1]))
                elif len(item) == 3:
                    raw_items.append((item[0], (item[1], item[2])))
                else:
                    raise ValueError(
                        f"Malformed placement item tuple of length {len(item)}: {item}"
                    )
            else:
                raise ValueError(
                    f"Unsupported placement item type '{type(item).__name__}': {item}"
                )
    else:
        # Check if placement implements mapping subscription protocol
        try:
            raw_items = [(m_id, raw_source[m_id]) for m_id in macros if m_id in raw_source]
        except (TypeError, AttributeError, KeyError):
            raw_items = []

    placed: dict[Any, tuple[int, int]] = {}

    for m_id, coord in raw_items:
        if coord is None:
            # Explicitly unplaced in incomplete placement
            continue

        if m_id in placed:
            raise ValueError(f"Duplicate placement for macro '{m_id}'.")

        if m_id not in macros:
            raise ValueError(f"Placement references macro '{m_id}' not found in netlist.")

        w, h = macros[m_id]

        if isinstance(coord, (tuple, list)) and len(coord) == 2:
            x, y = coord
        elif isinstance(coord, Mapping):
            x = coord.get("x")
            y = coord.get("y")
        else:
            x = getattr(coord, "x", None)
            y = getattr(coord, "y", None)

        if x is None or y is None:
            raise ValueError(f"Placement for macro '{m_id}' must provide x and y coordinates.")

        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float, np.integer, np.floating))
            or not isinstance(y, (int, float, np.integer, np.floating))
        ):
            raise ValueError(f"Macro '{m_id}' coordinates must be numeric.")

        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Macro '{m_id}' coordinates must be finite.")

        int_x = int(x)
        int_y = int(y)

        if int_x != x or int_y != y:
            raise ValueError(f"Macro '{m_id}' coordinates must be integer grid coordinates.")

        # Coordinate bounds checks (no silent clipping)
        if int_x < 0 or int_y < 0:
            raise ValueError(
                f"Macro '{m_id}' has negative coordinates ({int_x}, {int_y})."
            )
        if int_x + w > grid_w or int_y + h > grid_h:
            raise ValueError(
                f"Macro '{m_id}' of size ({w}, {h}) placed at ({int_x}, {int_y}) exceeds "
                f"grid boundaries ({grid_w}, {grid_h})."
            )

        placed[m_id] = (int_x, int_y)

    return placed


def _get_eligible_net_pin_centers(
    net: tuple[Any, list[Any], float],
    macros: dict[Any, tuple[int, int]],
    placements: dict[Any, tuple[int, int]],
) -> list[tuple[float, float]]:
    """Return placed pin centers (x + w/2, y + h/2) for a net.

    Reuses `get_macro_center` from `src.placement.hpwl`.
    """
    _, pins, _ = net
    centers: list[tuple[float, float]] = []
    for m_id in pins:
        if m_id in placements and m_id in macros:
            x, y = placements[m_id]
            w, h = macros[m_id]
            # Center of macro [x, x + w) x [y, y + h)
            centers.append(get_macro_center(x, y, w, h))
    return centers


def _calculate_hpwl(netlist: Any, placement: Any) -> float:
    """Calculate Half-Perimeter Wirelength (HPWL) for eligible nets.

    Reuses canonical `src.placement.hpwl.calculate_hpwl` when the netlist is an
    engine `Netlist` instance and placements are validated. Maintains a documented
    fallback for dictionary-based and mock netlists (e.g., `MockNetlist` with `pins`)
    to preserve backward compatibility across heterogeneous test configurations.
    """
    grid_w, grid_h = _extract_grid_dimensions(netlist)
    macros = _extract_macros(netlist)
    placements = _extract_placements(placement, macros, grid_w, grid_h)

    # If netlist is a canonical engine Netlist instance, delegate directly to calculate_hpwl
    if isinstance(netlist, Netlist):
        return float(calculate_hpwl(netlist, placements, allow_incomplete=True))

    # Compatibility fallback for dictionary/mock netlists (e.g. MockNetlist, dicts with 'pins')
    nets = _extract_nets(netlist)
    total_hpwl = 0.0

    for net in nets:
        _, _, weight = net
        centers = _get_eligible_net_pin_centers(net, macros, placements)
        if len(centers) < 2:
            continue

        x_coords = [c[0] for c in centers]
        y_coords = [c[1] for c in centers]
        w_bbox = max(x_coords) - min(x_coords)
        h_bbox = max(y_coords) - min(y_coords)
        net_hpwl = weight * (w_bbox + h_bbox)
        total_hpwl += net_hpwl

    return float(total_hpwl)


# =============================================================================
# Public API
# =============================================================================


def rudy_map(netlist: Any, placement: Any) -> np.ndarray:
    """Compute estimated routing demand map using RUDY wire density formulation.

    Parameters
    ----------
    netlist : Any
        Circuit netlist providing grid dimensions (grid_w, grid_h), macros, and nets.
    placement : Any
        Placement providing macro coordinates. Supports complete or incomplete placements.

    Returns
    -------
    np.ndarray
        2D NumPy array of shape (grid_h, grid_w) indexed as demand[y, x] containing
        estimated routing demand. All values are non-negative and finite.
    """
    grid_w, grid_h = _extract_grid_dimensions(netlist)
    macros = _extract_macros(netlist)
    nets = _extract_nets(netlist)
    placements = _extract_placements(placement, macros, grid_w, grid_h)

    demand = np.zeros((grid_h, grid_w), dtype=np.float64)

    for net in nets:
        _, _, weight = net
        centers = _get_eligible_net_pin_centers(net, macros, placements)

        # Ignore nets with fewer than 2 placed pins
        if len(centers) < 2:
            continue

        x_coords = [c[0] for c in centers]
        y_coords = [c[1] for c in centers]

        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        # Both coordinates identical: zero wirelength, zero demand
        if bbox_w == 0.0 and bbox_h == 0.0:
            continue

        # Regularization for degenerate bounding boxes:
        # If an axis has zero span (same-row or same-column net), regularize that axis
        # to a 1-cell corridor (effective dimension of 1.0), matching the cell assignment corridor.
        # For non-zero spans (including sub-unit fractional dimensions), use the actual span
        # so that accumulated demand integrates exactly to net.weight * HPWL.
        w_eff = 1.0 if bbox_w == 0.0 else bbox_w
        h_eff = 1.0 if bbox_h == 0.0 else bbox_h
        bbox_area = w_eff * h_eff

        density = weight * (bbox_w + bbox_h) / bbox_area

        # Determine cell intersection intervals in X
        if bbox_w == 0.0:
            # Same-column net: mapped to 1-cell-wide corridor
            x_cell = min(int(math.floor(x_min)), grid_w - 1)
            x_intersections = [(x_cell, 1.0)]
        else:
            x_start = max(0, int(math.floor(x_min)))
            x_end = min(grid_w - 1, int(math.floor(x_max)))
            x_intersections = []
            for cx in range(x_start, x_end + 1):
                overlap_x = max(0.0, min(float(cx + 1), x_max) - max(float(cx), x_min))
                if overlap_x > 0.0:
                    x_intersections.append((cx, overlap_x))

        # Determine cell intersection intervals in Y
        if bbox_h == 0.0:
            # Same-row net: mapped to 1-cell-wide corridor
            y_cell = min(int(math.floor(y_min)), grid_h - 1)
            y_intersections = [(y_cell, 1.0)]
        else:
            y_start = max(0, int(math.floor(y_min)))
            y_end = min(grid_h - 1, int(math.floor(y_max)))
            y_intersections = []
            for cy in range(y_start, y_end + 1):
                overlap_y = max(0.0, min(float(cy + 1), y_max) - max(float(cy), y_min))
                if overlap_y > 0.0:
                    y_intersections.append((cy, overlap_y))

        # Distribute demand to intersected cells
        for cy, overlap_y in y_intersections:
            for cx, overlap_x in x_intersections:
                demand[cy, cx] += density * overlap_x * overlap_y

    return demand


def congestion_score(netlist: Any, placement: Any) -> float:
    """Compute project-level scalar routing congestion proxy from RUDY demand map.

    Mathematically:
        C_score = (1 / (grid_w * grid_h)) * sum(demand[y, x] ** 2) = mean(demand ** 2)

    Hotspots with concentrated routing demand are penalized quadratically relative
    to evenly distributed demand.

    NOTE:
        This is a project-level proxy derived from RUDY wire density, NOT a
        measurement of actual routed congestion, detailed DRC violations, or
        guarantee of routability.

    Parameters
    ----------
    netlist : Any
        Circuit netlist providing grid dimensions, macros, and nets.
    placement : Any
        Placement providing macro coordinates.

    Returns
    -------
    float
        Congestion proxy score. Returns 0.0 if there are no eligible nets.
    """
    demand = rudy_map(netlist, placement)
    if demand.size == 0:
        raise ValueError("Grid has no cells.")

    return float(np.mean(demand**2))


def evaluate_placement(
    netlist: Any,
    placement: Any,
    alpha: float,
    beta: float,
) -> EvaluationResult:
    """Evaluate placement using combined objective: J(P) = alpha * HPWL(P) + beta * C_score(P).

    Parameters
    ----------
    netlist : Any
        Circuit netlist providing grid dimensions, macros, and nets.
    placement : Any
        Placement providing macro coordinates.
    alpha : float
        Non-negative weight for HPWL component.
    beta : float
        Non-negative weight for Congestion score component.

    Returns
    -------
    EvaluationResult
        NamedTuple containing:
            - hpwl: Half-Perimeter Wirelength (float)
            - congestion: Mean-squared-demand congestion proxy score (float)
            - objective: Combined objective J(P) (float)

    Notes
    -----
    The research team must select and freeze appropriate fixed weights (alpha, beta)
    prior to running comparative studies across classical heuristics (Random,
    Greedy, Simulated Annealing) and Reinforcement Learning. Do not dynamically
    normalize or tune weights based on the current placement.
    """
    # Validate alpha and beta
    for name, val in [("alpha", alpha), ("beta", beta)]:
        if (
            isinstance(val, bool)
            or not isinstance(val, (int, float, np.integer, np.floating))
        ):
            raise ValueError(f"Weight '{name}' must be numeric, got {type(val).__name__}.")
        if not math.isfinite(val):
            raise ValueError(f"Weight '{name}' must be finite, got {val}.")
        if val < 0.0:
            raise ValueError(f"Weight '{name}' must be non-negative, got {val}.")

    flt_alpha = float(alpha)
    flt_beta = float(beta)

    hpwl_val = _calculate_hpwl(netlist, placement)
    congestion_val = congestion_score(netlist, placement)
    objective_val = flt_alpha * hpwl_val + flt_beta * congestion_val

    return EvaluationResult(
        hpwl=hpwl_val,
        congestion=congestion_val,
        objective=objective_val,
    )
