"""Tests for the stateful PlacementEngine, action masking, and position conversions."""

import numpy as np
import pytest

from src.placement.constraints import (
    DuplicateMacroError,
    OutOfBoundsError,
    OverlapError,
    UnknownMacroError,
)
from src.placement.engine import (
    PlacementEngine,
    action_to_coord,
    coord_to_action,
)
from src.placement.models import Macro, Net, Netlist, PlacedMacro


@pytest.fixture
def sample_engine() -> PlacementEngine:
    """Fixture returning an engine on a 10x10 grid with 3 macros."""
    netlist = Netlist(
        version="1.0",
        grid_w=10,
        grid_h=10,
        macros=(
            Macro(id=0, w=3, h=3),
            Macro(id=1, w=4, h=2),
            Macro(id=2, w=2, h=5),
        ),
        nets=(
            Net(id=0, macro_ids=(0, 1), weight=1.0),
            Net(id=1, macro_ids=(1, 2), weight=1.0),
        ),
    )
    return PlacementEngine(netlist)


def test_action_coord_roundtrip() -> None:
    """Converting between (x, y) coordinates and discrete action index is bijective."""
    grid_w = 15
    for y in range(8):
        for x in range(grid_w):
            action = coord_to_action(x, y, grid_w)
            rx, ry = action_to_coord(action, grid_w)
            assert (rx, ry) == (x, y)


def test_placement_flow(sample_engine: PlacementEngine) -> None:
    """Placing and unplacing macros updates engine state correctly."""
    assert not sample_engine.is_placed(0)
    assert not sample_engine.is_complete()

    pm0 = sample_engine.place(macro_id=0, x=0, y=0)
    assert pm0 == PlacedMacro(macro_id=0, x=0, y=0)
    assert sample_engine.is_placed(0)
    assert not sample_engine.is_complete()

    # Occupancy is set for [0, 3) x [0, 3)
    occ = sample_engine.occupancy_grid
    assert np.all(occ[0:3, 0:3])
    assert not np.any(occ[3:, :])

    # Unplace
    unplaced = sample_engine.unplace(0)
    assert unplaced == pm0
    assert not sample_engine.is_placed(0)
    assert not np.any(sample_engine.occupancy_grid)


def test_place_error_conditions(sample_engine: PlacementEngine) -> None:
    """Engine raises dedicated errors for out-of-bounds, overlaps, duplicates, unknown IDs."""
    sample_engine.place(0, 0, 0)

    # Duplicate placement
    with pytest.raises(DuplicateMacroError):
        sample_engine.place(0, 5, 5)

    # Out of bounds (macro 1 is 4x2 on 10x10, so x=7 exceeds 10 - 4 = 6)
    with pytest.raises(OutOfBoundsError):
        sample_engine.place(1, 7, 0)

    # Overlap (macro 1 at (2, 0) overlaps macro 0 at [0, 3) x [0, 3))
    with pytest.raises(OverlapError):
        sample_engine.place(1, 2, 0)

    # Unknown macro
    with pytest.raises(UnknownMacroError):
        sample_engine.place(99, 5, 5)

    # Unplace unplaced macro
    with pytest.raises(UnknownMacroError):
        sample_engine.unplace(1)


def test_engine_reset(sample_engine: PlacementEngine) -> None:
    """Reset completely clears placed macros and occupancy grid."""
    sample_engine.place(0, 0, 0)
    sample_engine.place(1, 4, 0)
    assert len(sample_engine.get_placement()) == 2

    sample_engine.reset()
    assert len(sample_engine.get_placement()) == 0
    assert not np.any(sample_engine.occupancy_grid)


def test_action_mask_and_legal_positions(sample_engine: PlacementEngine) -> None:
    """compute_action_mask corresponds exactly with get_legal_positions."""
    # Place macro 0 at (0, 0) (dimensions 3x3)
    sample_engine.place(0, 0, 0)

    legal_positions = sample_engine.get_legal_positions(1)
    mask_1d = sample_engine.compute_action_mask(1, flatten=True)
    mask_2d = sample_engine.compute_action_mask(1, flatten=False)

    assert mask_1d.shape == (100,)
    assert mask_2d.shape == (10, 10)

    # Check consistency between mask and legal_positions
    for y in range(10):
        for x in range(10):
            action = coord_to_action(x, y, 10)
            is_valid_in_mask = mask_1d[action]
            assert is_valid_in_mask == mask_2d[y, x]
            assert is_valid_in_mask == ((x, y) in legal_positions)

    # Touching position (3, 0) for macro 1 (w=4, h=2) must be legal
    assert sample_engine.is_legal(1, 3, 0)
    assert (3, 0) in legal_positions

    # Overlapping position (2, 0) must be illegal
    assert not sample_engine.is_legal(1, 2, 0)
    assert (2, 0) not in legal_positions

    # Boundary edge check: macro 1 (w=4, h=2) at (6, 8) must be legal
    assert sample_engine.is_legal(1, 6, 8)
    assert (6, 8) in legal_positions

    # Boundary violation: macro 1 at (7, 8) must be illegal
    assert not sample_engine.is_legal(1, 7, 8)
    assert (7, 8) not in legal_positions


def test_action_mask_dtype_and_placed_macro(sample_engine: PlacementEngine) -> None:
    """Action mask has bool dtype and becomes all False once macro is placed."""
    mask_before = sample_engine.compute_action_mask(0)
    assert mask_before.dtype == np.bool_
    assert np.any(mask_before)

    sample_engine.place(0, 0, 0)
    mask_after = sample_engine.compute_action_mask(0)
    assert mask_after.dtype == np.bool_
    assert not np.any(mask_after)


def test_occupancy_grid_consistency(sample_engine: PlacementEngine) -> None:
    """Occupancy grid cells exactly correspond to placed macros."""
    # Place macro 0 (3x3) at (1, 2)
    sample_engine.place(0, 1, 2)
    occ = sample_engine.occupancy_grid

    # Only cells in [1, 4) x [2, 5) are True
    for y in range(sample_engine.grid_h):
        for x in range(sample_engine.grid_w):
            expected = (1 <= x < 4) and (2 <= y < 5)
            assert occ[y, x] == expected

    # Unplace and verify completely cleared
    sample_engine.unplace(0)
    assert not np.any(sample_engine.occupancy_grid)


def test_engine_unplaced_helpers(sample_engine: PlacementEngine) -> None:
    """Engine correctly reports unplaced macro IDs and next unplaced macro."""
    assert sample_engine.get_unplaced_macro_ids() == [0, 1, 2]
    next_m = sample_engine.get_next_unplaced_macro()
    assert next_m is not None and next_m.id == 0

    sample_engine.place(0, 0, 0)
    assert sample_engine.get_unplaced_macro_ids() == [1, 2]
    next_m = sample_engine.get_next_unplaced_macro()
    assert next_m is not None and next_m.id == 1

    sample_engine.place(1, 4, 0)
    sample_engine.place(2, 0, 4)
    assert sample_engine.get_unplaced_macro_ids() == []
    assert sample_engine.get_next_unplaced_macro() is None


