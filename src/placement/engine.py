"""Stateful placement engine managing grid layout, macro placement, and legal action masks."""

from collections.abc import Mapping

import numpy as np

from .constraints import (
    DuplicateMacroError,
    OverlapError,
    UnknownMacroError,
    find_overlapping_macro,
    is_within_boundary,
    validate_macro_boundary,
)
from .hpwl import calculate_hpwl
from .models import Macro, Netlist, PlacedMacro, Placement


def coord_to_action(x: int, y: int, grid_w: int) -> int:
    """Convert grid coordinate (x, y) to a 1D discrete action index: y * grid_w + x."""
    return int(y * grid_w + x)


def action_to_coord(action: int, grid_w: int) -> tuple[int, int]:
    """Convert a 1D discrete action index to grid coordinate (x, y)."""
    return (int(action % grid_w), int(action // grid_w))


class PlacementEngine:
    """Manages placement state, validation, and action masking on an integer grid.

    Conventions:
    - Origin (0, 0) is top-left.
    - Grid dimensions: width W = grid_w, height H = grid_h.
    - Placed macro occupies [x, x+w) x [y, y+h).
    - Discrete action index: action = y * grid_w + x.
    """

    def __init__(self, netlist: Netlist) -> None:
        """Initialize engine with a target netlist."""
        self.netlist: Netlist = netlist
        self._macro_map: dict[int, Macro] = {m.id: m for m in netlist.macros}
        self._placed: dict[int, PlacedMacro] = {}
        self._occupancy: np.ndarray = np.zeros(
            (netlist.grid_h, netlist.grid_w), dtype=bool
        )

    @property
    def grid_w(self) -> int:
        """Grid width in cells."""
        return self.netlist.grid_w

    @property
    def grid_h(self) -> int:
        """Grid height in cells."""
        return self.netlist.grid_h

    @property
    def macro_map(self) -> Mapping[int, Macro]:
        """Read-only view of macro ID to Macro definition."""
        return self._macro_map

    @property
    def placed_macros(self) -> Mapping[int, PlacedMacro]:
        """Read-only view of placed macros."""
        return self._placed

    @property
    def occupancy_grid(self) -> np.ndarray:
        """2D boolean grid where True indicates cell is occupied."""
        return self._occupancy.copy()

    def reset(self) -> None:
        """Clear all placed macros and reset grid occupancy."""
        self._placed.clear()
        self._occupancy.fill(False)

    def is_placed(self, macro_id: int) -> bool:
        """Return True if the macro has been placed."""
        return macro_id in self._placed

    def is_complete(self) -> bool:
        """Return True if all macros in the netlist have been placed."""
        return len(self._placed) == len(self._macro_map)

    def get_unplaced_macro_ids(self) -> list[int]:
        """Return list of macro IDs not yet placed, in netlist definition order."""
        return [m.id for m in self.netlist.macros if m.id not in self._placed]

    def get_next_unplaced_macro(self) -> Macro | None:
        """Return the next unplaced Macro in netlist definition order, or None if complete."""
        for m in self.netlist.macros:
            if m.id not in self._placed:
                return m
        return None

    def is_legal(self, macro_id: int, x: int, y: int) -> bool:
        """Check whether placing macro_id at (x, y) violates boundary or overlaps."""
        macro = self._macro_map.get(macro_id)
        if macro is None:
            return False

        if macro_id in self._placed:
            return False

        if not is_within_boundary(x, y, macro.w, macro.h, self.grid_w, self.grid_h):
            return False

        # Check occupancy grid slice for overlaps
        return not bool(np.any(self._occupancy[y : y + macro.h, x : x + macro.w]))

    def place(self, macro_id: int, x: int, y: int) -> PlacedMacro:
        """Place macro_id at (x, y). Raises exception if placement is illegal."""
        macro = self._macro_map.get(macro_id)
        if macro is None:
            raise UnknownMacroError(f"Macro ID {macro_id} not found in netlist.")

        if macro_id in self._placed:
            raise DuplicateMacroError(f"Macro {macro_id} is already placed.")

        validate_macro_boundary(x, y, macro.w, macro.h, self.grid_w, self.grid_h, macro_id=macro_id)

        # Check for overlap
        placed_positions = {pm.macro_id: (pm.x, pm.y) for pm in self._placed.values()}
        overlapping = find_overlapping_macro(
            candidate_id=macro_id,
            candidate_x=x,
            candidate_y=y,
            candidate_w=macro.w,
            candidate_h=macro.h,
            placed_positions=placed_positions,
            macro_lookup=self._macro_map,
        )
        if overlapping is not None:
            raise OverlapError(
                f"Placing macro {macro_id} at ({x}, {y}) overlaps with placed macro {overlapping}."
            )

        # Record placement and update occupancy
        placed = PlacedMacro(macro_id=macro_id, x=x, y=y)
        self._placed[macro_id] = placed
        self._occupancy[y : y + macro.h, x : x + macro.w] = True
        return placed

    def unplace(self, macro_id: int) -> PlacedMacro:
        """Remove a placed macro and update occupancy grid. Raises UnknownMacroError if not placed."""
        if macro_id not in self._placed:
            raise UnknownMacroError(f"Macro {macro_id} is not currently placed.")

        placed = self._placed.pop(macro_id)
        macro = self._macro_map[macro_id]
        self._occupancy[placed.y : placed.y + macro.h, placed.x : placed.x + macro.w] = False
        return placed

    def get_placement(self) -> Placement:
        """Return the current placement as a list of PlacedMacro."""
        return list(self._placed.values())

    def get_legal_positions(self, macro_id: int) -> list[tuple[int, int]]:
        """Return a list of all legal (x, y) top-left placement coordinates for a macro."""
        macro = self._macro_map.get(macro_id)
        if macro is None or macro_id in self._placed:
            return []

        legal_positions: list[tuple[int, int]] = []
        max_x = self.grid_w - macro.w
        max_y = self.grid_h - macro.h

        for y in range(max_y + 1):
            for x in range(max_x + 1):
                if not np.any(self._occupancy[y : y + macro.h, x : x + macro.w]):
                    legal_positions.append((x, y))

        return legal_positions

    def compute_action_mask(self, macro_id: int, flatten: bool = True) -> np.ndarray:
        """Generate a boolean mask of valid top-left placement coordinates for macro_id.

        Parameters:
            macro_id: The ID of the macro to place.
            flatten: If True, returns 1D array of shape (grid_w * grid_h,) matching
                     discrete action indices (y * grid_w + x). If False, returns 2D
                     array of shape (grid_h, grid_w).

        Returns:
            Boolean numpy ndarray where True represents a legal placement position.
        """
        mask = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        macro = self._macro_map.get(macro_id)

        if macro is None or macro_id in self._placed:
            return mask.ravel() if flatten else mask

        max_x = self.grid_w - macro.w
        max_y = self.grid_h - macro.h

        if max_x < 0 or max_y < 0:
            return mask.ravel() if flatten else mask

        for y in range(max_y + 1):
            for x in range(max_x + 1):
                if not np.any(self._occupancy[y : y + macro.h, x : x + macro.w]):
                    mask[y, x] = True

        return mask.ravel() if flatten else mask

    def current_hpwl(self, allow_incomplete: bool = True) -> float:
        """Calculate current HPWL of placed macros."""
        return calculate_hpwl(self.netlist, self.get_placement(), allow_incomplete=allow_incomplete)
