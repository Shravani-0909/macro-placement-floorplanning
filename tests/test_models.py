"""Tests for placement data models."""

import json
from dataclasses import FrozenInstanceError

import pytest

from src.placement.models import (
    Macro,
    Net,
    Netlist,
    PlacedMacro,
    Placement,
    placement_to_dict,
)


@pytest.fixture
def sample_netlist() -> Netlist:
    """Return a representative Netlist instance for testing."""
    macros = (
        Macro(id=0, w=10, h=20),
        Macro(id=1, w=15, h=25),
        Macro(id=2, w=5, h=10),
    )
    nets = (
        Net(id=0, macro_ids=(0, 1), weight=1.0),
        Net(id=1, macro_ids=(1, 2, 0), weight=2.5),
    )
    return Netlist(
        version="1.0",
        grid_w=100,
        grid_h=80,
        macros=macros,
        nets=nets,
    )


def test_dict_round_trip(sample_netlist: Netlist) -> None:
    """Netlist.from_dict(netlist.to_dict()) must match the original instance."""
    d = sample_netlist.to_dict()
    restored = Netlist.from_dict(d)
    assert restored == sample_netlist


def test_json_round_trip(sample_netlist: Netlist) -> None:
    """Netlist survives serialization through json.dumps and json.loads."""
    d = sample_netlist.to_dict()
    raw_json = json.dumps(d)
    parsed_json = json.loads(raw_json)
    restored = Netlist.from_dict(parsed_json)
    assert restored == sample_netlist


def test_from_dict_default_weight() -> None:
    """Netlist.from_dict defaults net weight to 1.0 when omitted."""
    d = {
        "version": "1.0",
        "grid_w": 50,
        "grid_h": 50,
        "macros": [{"id": 0, "w": 5, "h": 5}],
        "nets": [{"id": 0, "macro_ids": [0]}],
    }
    netlist = Netlist.from_dict(d)
    assert netlist.nets[0].weight == 1.0


def test_frozen_instances(sample_netlist: Netlist) -> None:
    """Dataclasses must be immutable (raise FrozenInstanceError on mutation)."""
    macro = sample_netlist.macros[0]
    with pytest.raises(FrozenInstanceError):
        macro.w = 99  # type: ignore[misc]

    net = sample_netlist.nets[0]
    with pytest.raises(FrozenInstanceError):
        net.weight = 3.0  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        sample_netlist.grid_w = 200  # type: ignore[misc]

    placed = PlacedMacro(macro_id=0, x=10, y=20)
    with pytest.raises(FrozenInstanceError):
        placed.x = 30  # type: ignore[misc]


def test_placement_to_dict() -> None:
    """placement_to_dict converts list of PlacedMacro to dict[int, tuple[int, int]]."""
    placement: Placement = [
        PlacedMacro(macro_id=0, x=5, y=10),
        PlacedMacro(macro_id=1, x=20, y=30),
    ]
    expected = {
        0: (5, 10),
        1: (20, 30),
    }
    assert placement_to_dict(placement) == expected
