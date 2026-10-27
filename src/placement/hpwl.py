"""Half-Perimeter Wirelength (HPWL) calculation for circuit placements."""

from collections.abc import Mapping, Sequence

from .constraints import PlacementError
from .models import Macro, Net, Netlist, PlacedMacro, Placement


class IncompletePlacementError(PlacementError):
    """Raised when HPWL is requested on a placement with unplaced net macros."""


def get_macro_center(x: int, y: int, w: int, h: int) -> tuple[float, float]:
    """Compute the continuous center coordinates of a macro: (x + w/2, y + h/2)."""
    return (x + (w / 2.0), y + (h / 2.0))


def compute_net_hpwl(
    net: Net,
    placed_positions: Mapping[int, tuple[int, int]],
    macro_lookup: Mapping[int, Macro],
    allow_incomplete: bool = False,
) -> float:
    """Calculate HPWL for a single net.

    If allow_incomplete is False and any macro in the net is unplaced,
    raises IncompletePlacementError. If allow_incomplete is True and fewer
    than 2 macros are placed, returns 0.0.
    """
    placed_centers: list[tuple[float, float]] = []

    for macro_id in net.macro_ids:
        if macro_id not in placed_positions:
            if not allow_incomplete:
                raise IncompletePlacementError(
                    f"Net {net.id} includes macro {macro_id} which is not placed."
                )
            continue

        macro = macro_lookup.get(macro_id)
        if macro is None:
            raise IncompletePlacementError(
                f"Net {net.id} references undefined macro {macro_id}."
            )

        pos = placed_positions[macro_id]
        center = get_macro_center(pos[0], pos[1], macro.w, macro.h)
        placed_centers.append(center)

    if len(placed_centers) <= 1:
        return 0.0

    xs = [c[0] for c in placed_centers]
    ys = [c[1] for c in placed_centers]

    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    return float(dx + dy)


def calculate_hpwl(
    netlist: Netlist,
    placement: Placement | Sequence[PlacedMacro] | Mapping[int, tuple[int, int]],
    allow_incomplete: bool = False,
) -> float:
    """Calculate total weighted Half-Perimeter Wirelength across all nets in a netlist.

    Parameters:
        netlist: The circuit netlist containing macros and nets.
        placement: Placed macros as a Placement sequence or dict mapping macro_id to (x, y).
        allow_incomplete: If False, raises IncompletePlacementError if any net has unplaced
            macros. If True, computes HPWL over placed pins.

    Returns:
        Total weighted HPWL: sum(net.weight * net_hpwl).
    """
    if isinstance(placement, Mapping):
        pos_map = dict(placement)
    else:
        pos_map = {pm.macro_id: (pm.x, pm.y) for pm in placement}

    macro_lookup = {m.id: m for m in netlist.macros}
    total_hpwl = 0.0

    for net in netlist.nets:
        net_hpwl = compute_net_hpwl(
            net=net,
            placed_positions=pos_map,
            macro_lookup=macro_lookup,
            allow_incomplete=allow_incomplete,
        )
        total_hpwl += net.weight * net_hpwl

    return float(total_hpwl)


def calculate_net_hpwls(
    netlist: Netlist,
    placement: Placement | Sequence[PlacedMacro] | Mapping[int, tuple[int, int]],
    allow_incomplete: bool = False,
) -> dict[int, float]:
    """Calculate individual HPWL for each net, returned as a dict keyed by net.id."""
    if isinstance(placement, Mapping):
        pos_map = dict(placement)
    else:
        pos_map = {pm.macro_id: (pm.x, pm.y) for pm in placement}

    macro_lookup = {m.id: m for m in netlist.macros}
    results: dict[int, float] = {}

    for net in netlist.nets:
        results[net.id] = compute_net_hpwl(
            net=net,
            placed_positions=pos_map,
            macro_lookup=macro_lookup,
            allow_incomplete=allow_incomplete,
        )

    return results
