"""Tests for Half-Perimeter Wirelength (HPWL) calculation."""

import pytest

from src.placement.hpwl import (
    IncompletePlacementError,
    calculate_hpwl,
    calculate_net_hpwls,
    get_macro_center,
)
from src.placement.models import Macro, Net, Netlist, PlacedMacro


def test_get_macro_center() -> None:
    """Center of macro is (x + w/2, y + h/2) with float precision."""
    # Even dimensions
    assert get_macro_center(x=10, y=20, w=4, h=6) == (12.0, 23.0)
    # Odd dimensions
    assert get_macro_center(x=0, y=0, w=3, h=5) == (1.5, 2.5)


def test_hpwl_two_pin_and_multi_pin() -> None:
    """HPWL computed accurately across two-pin and multi-pin nets."""
    # Netlist with 3 macros of various sizes
    # Macro 0: w=4, h=4 at (0, 0) -> center (2.0, 2.0)
    # Macro 1: w=2, h=2 at (10, 0) -> center (11.0, 1.0)
    # Macro 2: w=6, h=2 at (4, 10) -> center (7.0, 11.0)
    macros = (
        Macro(id=0, w=4, h=4),
        Macro(id=1, w=2, h=2),
        Macro(id=2, w=6, h=2),
    )
    # Net 0 connects 0 and 1:
    # dx = |11.0 - 2.0| = 9.0, dy = |1.0 - 2.0| = 1.0 -> HPWL = 10.0
    # Net 1 connects 0, 1, and 2:
    # xs = [2.0, 11.0, 7.0] -> dx = 11.0 - 2.0 = 9.0
    # ys = [2.0, 1.0, 11.0] -> dy = 11.0 - 1.0 = 10.0 -> HPWL = 19.0
    nets = (
        Net(id=0, macro_ids=(0, 1), weight=1.0),
        Net(id=1, macro_ids=(0, 1, 2), weight=2.0),
    )
    netlist = Netlist("1.0", 30, 30, macros, nets)
    placement = [
        PlacedMacro(macro_id=0, x=0, y=0),
        PlacedMacro(macro_id=1, x=10, y=0),
        PlacedMacro(macro_id=2, x=4, y=10),
    ]

    net_hpwls = calculate_net_hpwls(netlist, placement)
    assert net_hpwls[0] == pytest.approx(10.0)
    assert net_hpwls[1] == pytest.approx(19.0)

    # Total weighted HPWL = 1.0 * 10.0 + 2.0 * 19.0 = 48.0
    total = calculate_hpwl(netlist, placement)
    assert total == pytest.approx(48.0)


def test_hpwl_single_pin_net() -> None:
    """A net with a single macro has zero HPWL."""
    netlist = Netlist(
        version="1.0",
        grid_w=10,
        grid_h=10,
        macros=(Macro(id=0, w=2, h=2),),
        nets=(Net(id=0, macro_ids=(0,), weight=1.5),),
    )
    placement = [PlacedMacro(macro_id=0, x=2, y=2)]
    assert calculate_hpwl(netlist, placement) == 0.0


def test_hpwl_incomplete_placement() -> None:
    """Incomplete placements raise IncompletePlacementError unless allow_incomplete=True."""
    netlist = Netlist(
        version="1.0",
        grid_w=20,
        grid_h=20,
        macros=(Macro(id=0, w=2, h=2), Macro(id=1, w=2, h=2)),
        nets=(Net(id=0, macro_ids=(0, 1)),),
    )
    partial_placement = [PlacedMacro(macro_id=0, x=0, y=0)]

    # Default raises
    with pytest.raises(IncompletePlacementError):
        calculate_hpwl(netlist, partial_placement, allow_incomplete=False)

    # allow_incomplete=True returns 0.0 because fewer than 2 pins are placed
    hpwl_partial = calculate_hpwl(netlist, partial_placement, allow_incomplete=True)
    assert hpwl_partial == 0.0
