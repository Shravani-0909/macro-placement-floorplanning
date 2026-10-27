"""Validation rules and geometric constraint checks for macro placement."""

from collections.abc import Mapping
from dataclasses import dataclass

from .models import Macro, Netlist, Placement


class PlacementError(Exception):
    """Base exception for placement violations."""


class OutOfBoundsError(PlacementError):
    """Raised when a macro exceeds the grid boundaries."""


class OverlapError(PlacementError):
    """Raised when two placed macros overlap."""


class DuplicateMacroError(PlacementError):
    """Raised when a macro is placed more than once."""


class UnknownMacroError(PlacementError):
    """Raised when an unrecognised macro ID is encountered."""


def is_within_boundary(
    x: int,
    y: int,
    w: int,
    h: int,
    grid_w: int,
    grid_h: int,
) -> bool:
    """Return True if macro rectangle [x, x+w) x [y, y+h) lies strictly within the grid."""
    if w <= 0 or h <= 0:
        return False
    return 0 <= x <= grid_w - w and 0 <= y <= grid_h - h


def validate_macro_boundary(
    x: int,
    y: int,
    w: int,
    h: int,
    grid_w: int,
    grid_h: int,
    macro_id: int | None = None,
) -> None:
    """Validate that a macro fits within the grid boundary, raising OutOfBoundsError if not."""
    if not is_within_boundary(x, y, w, h, grid_w, grid_h):
        id_str = f"Macro {macro_id} " if macro_id is not None else "Macro "
        raise OutOfBoundsError(
            f"{id_str}with dimensions ({w}, {h}) placed at ({x}, {y}) "
            f"violates grid boundaries [0, {grid_w}] x [0, {grid_h}]."
        )


def macros_overlap(
    x1: int,
    y1: int,
    w1: int,
    h1: int,
    x2: int,
    y2: int,
    w2: int,
    h2: int,
) -> bool:
    """Return True if two half-open macro rectangles overlap. Touching edges do not overlap."""
    horizontal_overlap = max(x1, x2) < min(x1 + w1, x2 + w2)
    vertical_overlap = max(y1, y2) < min(y1 + h1, y2 + h2)
    return horizontal_overlap and vertical_overlap


def find_overlapping_macro(
    candidate_id: int,
    candidate_x: int,
    candidate_y: int,
    candidate_w: int,
    candidate_h: int,
    placed_positions: Mapping[int, tuple[int, int]],
    macro_lookup: Mapping[int, Macro],
) -> int | None:
    """Return macro_id of first placed macro overlapping candidate, or None if none."""
    for other_id, (ox, oy) in placed_positions.items():
        if other_id == candidate_id:
            continue
        other_macro = macro_lookup.get(other_id)
        if other_macro is None:
            continue
        if macros_overlap(
            candidate_x,
            candidate_y,
            candidate_w,
            candidate_h,
            ox,
            oy,
            other_macro.w,
            other_macro.h,
        ):
            return other_id
    return None


@dataclass(frozen=True)
class ValidationReport:
    """Result of a placement validation check."""

    is_valid: bool
    is_complete: bool
    placed_count: int
    total_count: int
    errors: tuple[str, ...] = ()


def validate_placement(
    netlist: Netlist,
    placement: Placement,
) -> ValidationReport:
    """Validate a placement against netlist dimensions and macro definitions."""
    macro_map: dict[int, Macro] = {m.id: m for m in netlist.macros}
    placed_positions: dict[int, tuple[int, int]] = {}
    errors: list[str] = []

    for pm in placement:
        if pm.macro_id not in macro_map:
            errors.append(f"Unknown macro ID: {pm.macro_id}")
            continue

        if pm.macro_id in placed_positions:
            errors.append(f"Duplicate placement for macro ID: {pm.macro_id}")
            continue

        macro = macro_map[pm.macro_id]
        if not is_within_boundary(pm.x, pm.y, macro.w, macro.h, netlist.grid_w, netlist.grid_h):
            errors.append(
                f"Macro {pm.macro_id} ({macro.w}x{macro.h}) at ({pm.x}, {pm.y}) is out of bounds."
            )

        overlapping = find_overlapping_macro(
            candidate_id=pm.macro_id,
            candidate_x=pm.x,
            candidate_y=pm.y,
            candidate_w=macro.w,
            candidate_h=macro.h,
            placed_positions=placed_positions,
            macro_lookup=macro_map,
        )
        if overlapping is not None:
            errors.append(
                f"Macro {pm.macro_id} overlaps with already placed macro {overlapping}."
            )

        placed_positions[pm.macro_id] = (pm.x, pm.y)

    total_count = len(netlist.macros)
    placed_count = len(placed_positions)
    is_complete = placed_count == total_count and len(errors) == 0

    return ValidationReport(
        is_valid=len(errors) == 0,
        is_complete=is_complete,
        placed_count=placed_count,
        total_count=total_count,
        errors=tuple(errors),
    )
