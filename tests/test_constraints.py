"""Tests for placement constraints, boundary validation, and overlap checks."""

import pytest

from src.placement.constraints import (
    OutOfBoundsError,
    find_overlapping_macro,
    is_within_boundary,
    macros_overlap,
    validate_macro_boundary,
    validate_placement,
)
from src.placement.models import Macro, Net, Netlist, PlacedMacro


def test_is_within_boundary() -> None:
    """Verify boundary checks adhere to 0 <= x <= W - w and 0 <= y <= H - h."""
    # 10x10 grid, 2x3 macro
    # Valid corner placements
    assert is_within_boundary(x=0, y=0, w=2, h=3, grid_w=10, grid_h=10)
    assert is_within_boundary(x=8, y=7, w=2, h=3, grid_w=10, grid_h=10)

    # Invalid positions
    assert not is_within_boundary(x=-1, y=0, w=2, h=3, grid_w=10, grid_h=10)
    assert not is_within_boundary(x=0, y=-1, w=2, h=3, grid_w=10, grid_h=10)
    assert not is_within_boundary(x=9, y=0, w=2, h=3, grid_w=10, grid_h=10)
    assert not is_within_boundary(x=0, y=8, w=2, h=3, grid_w=10, grid_h=10)

    # Macro dimensions invalid or larger than grid
    assert not is_within_boundary(x=0, y=0, w=11, h=3, grid_w=10, grid_h=10)
    assert not is_within_boundary(x=0, y=0, w=0, h=3, grid_w=10, grid_h=10)


def test_validate_macro_boundary_raises() -> None:
    """validate_macro_boundary raises OutOfBoundsError for illegal coordinates."""
    with pytest.raises(OutOfBoundsError):
        validate_macro_boundary(x=9, y=5, w=3, h=2, grid_w=10, grid_h=10, macro_id=1)

    # Valid placement does not raise
    validate_macro_boundary(x=7, y=5, w=3, h=2, grid_w=10, grid_h=10, macro_id=1)


def test_macros_overlap_and_touching() -> None:
    """Half-open rectangles overlap only if interiors intersect; touching edges do not overlap."""
    # Macro 1: [2, 6) x [2, 5)  (w=4, h=3)
    # Touching on right edge: Macro 2: [6, 10) x [2, 5)
    assert not macros_overlap(2, 2, 4, 3, 6, 2, 4, 3)

    # Touching on left edge: Macro 2: [0, 2) x [2, 5)
    assert not macros_overlap(2, 2, 4, 3, 0, 2, 2, 3)

    # Touching on bottom edge: Macro 2: [2, 6) x [5, 8)
    assert not macros_overlap(2, 2, 4, 3, 2, 5, 4, 3)

    # Touching on corner: [6, 8) x [5, 7)
    assert not macros_overlap(2, 2, 4, 3, 6, 5, 2, 2)

    # Partial overlap by 1 cell
    assert macros_overlap(2, 2, 4, 3, 5, 2, 3, 3)

    # Complete enclosure
    assert macros_overlap(2, 2, 6, 6, 3, 3, 2, 2)


def test_find_overlapping_macro() -> None:
    """find_overlapping_macro correctly identifies conflicting placed macros."""
    macro_map = {
        0: Macro(id=0, w=4, h=4),
        1: Macro(id=1, w=3, h=3),
    }
    placed = {
        0: (0, 0),
    }

    # Macro 1 placed touching at (4, 0) does not overlap
    assert find_overlapping_macro(1, 4, 0, 3, 3, placed, macro_map) is None

    # Macro 1 placed at (3, 0) overlaps macro 0
    assert find_overlapping_macro(1, 3, 0, 3, 3, placed, macro_map) == 0


def test_validate_placement_report() -> None:
    """validate_placement produces comprehensive ValidationReport."""
    netlist = Netlist(
        version="1.0",
        grid_w=10,
        grid_h=10,
        macros=(
            Macro(id=0, w=4, h=4),
            Macro(id=1, w=4, h=4),
            Macro(id=2, w=2, h=2),
        ),
        nets=(Net(id=0, macro_ids=(0, 1, 2)),),
    )

    # Legal partial placement
    p_partial = [PlacedMacro(macro_id=0, x=0, y=0)]
    report = validate_placement(netlist, p_partial)
    assert report.is_valid
    assert not report.is_complete
    assert report.placed_count == 1
    assert len(report.errors) == 0

    # Legal complete placement
    p_complete = [
        PlacedMacro(macro_id=0, x=0, y=0),
        PlacedMacro(macro_id=1, x=4, y=0),
        PlacedMacro(macro_id=2, x=0, y=4),
    ]
    report_comp = validate_placement(netlist, p_complete)
    assert report_comp.is_valid
    assert report_comp.is_complete
    assert report_comp.placed_count == 3

    # Illegal: overlap, unknown ID, duplicate, out-of-bounds
    p_illegal = [
        PlacedMacro(macro_id=0, x=0, y=0),
        PlacedMacro(macro_id=0, x=1, y=1),  # duplicate
        PlacedMacro(macro_id=1, x=2, y=2),  # overlaps macro 0
        PlacedMacro(macro_id=99, x=0, y=0), # unknown
        PlacedMacro(macro_id=2, x=9, y=9),  # out of bounds (9+2 > 10)
    ]
    report_ill = validate_placement(netlist, p_illegal)
    assert not report_ill.is_valid
    assert not report_ill.is_complete
    assert len(report_ill.errors) >= 4


def test_validate_placement_empty() -> None:
    """Empty placement is valid but incomplete for non-empty netlist."""
    netlist = Netlist(
        version="1.0",
        grid_w=10,
        grid_h=10,
        macros=(Macro(id=0, w=2, h=2),),
        nets=(Net(id=0, macro_ids=(0,)),),
    )
    report = validate_placement(netlist, [])
    assert report.is_valid
    assert not report.is_complete
    assert report.placed_count == 0
    assert report.total_count == 1

