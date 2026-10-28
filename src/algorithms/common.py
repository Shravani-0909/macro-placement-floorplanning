"""Common helper utilities and data adapters for classical placement algorithms."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.placement.engine import PlacementEngine
from src.placement.models import Macro, Net, Netlist, PlacedMacro, Placement


def to_canonical_netlist(netlist: Any) -> Netlist:
    """Convert any netlist representation (dict, MockNetlist, or Netlist) into a Netlist."""
    if isinstance(netlist, Netlist):
        return netlist

    if isinstance(netlist, Mapping):
        grid_w = int(netlist.get("grid_w", 0))
        grid_h = int(netlist.get("grid_h", 0))
        raw_macros = netlist.get("macros", {})
        raw_nets = netlist.get("nets", [])
    else:
        grid_w = int(getattr(netlist, "grid_w", 0))
        grid_h = int(getattr(netlist, "grid_h", 0))
        raw_macros = getattr(netlist, "macros", {})
        raw_nets = getattr(netlist, "nets", [])

    macro_objs: list[Macro] = []
    if isinstance(raw_macros, Mapping):
        for mid, m_dim in raw_macros.items():
            if isinstance(m_dim, (tuple, list)) and len(m_dim) == 2:
                w, h = m_dim
            elif isinstance(m_dim, Mapping):
                w = m_dim.get("w", m_dim.get("width"))
                h = m_dim.get("h", m_dim.get("height"))
            else:
                w = getattr(m_dim, "w", getattr(m_dim, "width", 1))
                h = getattr(m_dim, "h", getattr(m_dim, "height", 1))
            macro_objs.append(Macro(id=mid, w=int(w), h=int(h)))
    elif isinstance(raw_macros, Sequence) and not isinstance(raw_macros, (str, bytes)):
        for item in raw_macros:
            if isinstance(item, Mapping):
                mid = item.get("id")
                w = item.get("w", item.get("width", 1))
                h = item.get("h", item.get("height", 1))
            else:
                mid = getattr(item, "id", None)
                w = getattr(item, "w", getattr(item, "width", 1))
                h = getattr(item, "h", getattr(item, "height", 1))
            macro_objs.append(Macro(id=mid, w=int(w), h=int(h)))

    net_objs: list[Net] = []
    if isinstance(raw_nets, Mapping):
        for nid, net_info in raw_nets.items():
            if isinstance(net_info, Mapping):
                pins = net_info.get("pins", net_info.get("macro_ids", []))
                weight = float(net_info.get("weight", 1.0))
            else:
                pins = getattr(net_info, "pins", getattr(net_info, "macro_ids", []))
                weight = float(getattr(net_info, "weight", 1.0))
            net_objs.append(Net(id=nid, macro_ids=tuple(pins), weight=weight))
    elif isinstance(raw_nets, Sequence) and not isinstance(raw_nets, (str, bytes)):
        for idx, item in enumerate(raw_nets):
            if isinstance(item, (tuple, list)):
                pins = list(item[0]) if len(item) > 0 else []
                weight = float(item[1]) if len(item) > 1 else 1.0
                net_objs.append(Net(id=idx, macro_ids=tuple(pins), weight=weight))
            elif isinstance(item, Mapping):
                nid = item.get("id", idx)
                pins = item.get("pins", item.get("macro_ids", []))
                weight = float(item.get("weight", 1.0))
                net_objs.append(Net(id=nid, macro_ids=tuple(pins), weight=weight))
            else:
                nid = getattr(item, "id", idx)
                pins = getattr(item, "pins", getattr(item, "macro_ids", []))
                weight = float(getattr(item, "weight", 1.0))
                net_objs.append(Net(id=nid, macro_ids=tuple(pins), weight=weight))

    return Netlist(
        version="1.0",
        grid_w=grid_w,
        grid_h=grid_h,
        macros=tuple(macro_objs),
        nets=tuple(net_objs),
    )


def extract_placed_ids(placement: Any) -> set[Any]:
    """Extract the set of placed macro IDs from placement sequence, object, or mapping."""
    if placement is None:
        return set()
    if hasattr(placement, "positions"):
        raw_pos = placement.positions
    else:
        raw_pos = placement

    if isinstance(raw_pos, Mapping):
        return {k for k, v in raw_pos.items() if v is not None}
    elif isinstance(raw_pos, (list, tuple, Sequence)) and not isinstance(raw_pos, (str, bytes)):
        ids: set[Any] = set()
        for item in raw_pos:
            if item is None:
                continue
            if hasattr(item, "macro_id"):
                ids.add(item.macro_id)
            elif hasattr(item, "id"):
                ids.add(item.id)
            elif isinstance(item, Mapping) and ("macro_id" in item or "id" in item):
                ids.add(item.get("macro_id", item.get("id")))
            elif isinstance(item, (tuple, list)) and len(item) in (2, 3):
                ids.add(item[0])
        return ids
    else:
        try:
            return {k for k in raw_pos if raw_pos[k] is not None}
        except (TypeError, AttributeError, KeyError):
            return set()


def normalize_placement(placement: Any) -> dict[Any, tuple[int, int]]:
    """Convert any placement representation into a dictionary of macro_id -> (int(x), int(y))."""
    if placement is None:
        return {}
    if hasattr(placement, "positions"):
        raw_source = placement.positions
    else:
        raw_source = placement

    raw_items: list[tuple[Any, Any]] = []
    if isinstance(raw_source, Mapping):
        raw_items = list(raw_source.items())
    elif isinstance(raw_source, (list, tuple, Sequence)) and not isinstance(raw_source, (str, bytes)):
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
                    raise ValueError(f"Placement tuple/list must have length 2 or 3, got {len(item)}.")
            else:
                raise ValueError(f"Unsupported coordinate format in sequence: {item!r}.")
    else:
        try:
            raw_items = [(k, raw_source[k]) for k in raw_source]
        except (TypeError, AttributeError, KeyError):
            raw_items = []

    norm: dict[Any, tuple[int, int]] = {}
    for m_id, coord in raw_items:
        if coord is None:
            continue

        if isinstance(coord, (tuple, list)):
            if len(coord) != 2:
                raise ValueError(
                    f"Placement for macro '{m_id}' tuple/list must have length 2, got {len(coord)}."
                )
            x, y = coord[0], coord[1]
        elif isinstance(coord, Mapping):
            x = coord.get("x")
            y = coord.get("y")
            if x is None or y is None:
                raise ValueError(
                    f"Placement mapping for macro '{m_id}' must provide 'x' and 'y', got {coord}."
                )
        elif hasattr(coord, "x") and hasattr(coord, "y"):
            x = getattr(coord, "x")
            y = getattr(coord, "y")
        else:
            raise ValueError(f"Unsupported coordinate format for macro '{m_id}': {coord!r}.")

        if x is None or y is None:
            raise ValueError(f"Placement for macro '{m_id}' must provide x and y coordinates.")

        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float, np.integer, np.floating))
            or not isinstance(y, (int, float, np.integer, np.floating))
        ):
            raise ValueError(
                f"Macro '{m_id}' coordinates must be numeric, got x={type(x).__name__}, y={type(y).__name__}."
            )

        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Macro '{m_id}' coordinates must be finite, got x={x}, y={y}.")

        int_x = int(x)
        int_y = int(y)

        if int_x != x or int_y != y:
            raise ValueError(f"Macro '{m_id}' coordinates must be integers, got x={x}, y={y}.")

        norm[m_id] = (int_x, int_y)

    return norm


def random_legal_placement(netlist: Any, seed: int | None = 42) -> Placement | None:
    """Generate a complete legal placement using PlacementEngine.

    Parameters:
        netlist: Target circuit netlist (Netlist, dict, or duck-typed object).
        seed: Random seed for deterministic reproducibility, or None.

    Returns:
        A list of PlacedMacro objects (Placement) if successful, or None if dead end reached.
    """
    canonical_nl = to_canonical_netlist(netlist)
    engine = PlacementEngine(canonical_nl)
    rng = random.Random(seed) if seed is not None else random.Random()

    for macro in canonical_nl.macros:
        legal_positions = engine.get_legal_positions(macro.id)
        if not legal_positions:
            return None
        chosen_x, chosen_y = rng.choice(legal_positions)
        engine.place(macro.id, chosen_x, chosen_y)

    return engine.get_placement() if engine.is_complete() else None


def dict_to_placement(d: Mapping[Any, tuple[int, int]]) -> Placement:
    """Convert mapping from macro_id to (x, y) into a Placement list."""
    return [PlacedMacro(macro_id=mid, x=coord[0], y=coord[1]) for mid, coord in d.items()]


def compare_placements(p1: Any, p2: Any) -> bool:
    """Check if two placement representations describe identical macro positions."""
    if p1 is None and p2 is None:
        return True
    if p1 is None or p2 is None:
        return False
    d1 = normalize_placement(p1)
    d2 = normalize_placement(p2)
    return d1 == d2
