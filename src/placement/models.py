"""Data models for macro placement and netlists."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Macro:
    """A rectangular macro circuit block."""

    id: int
    w: int
    h: int


@dataclass(frozen=True)
class Net:
    """A net connecting multiple macros with an optional weight."""

    id: int
    macro_ids: tuple[int, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class Netlist:
    """A circuit netlist and placement grid specification."""

    version: str
    grid_w: int
    grid_h: int
    macros: tuple[Macro, ...]
    nets: tuple[Net, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize netlist to a JSON-safe dictionary."""
        return {
            "version": self.version,
            "grid_w": self.grid_w,
            "grid_h": self.grid_h,
            "macros": [
                {"id": m.id, "w": m.w, "h": m.h}
                for m in self.macros
            ],
            "nets": [
                {
                    "id": n.id,
                    "macro_ids": list(n.macro_ids),
                    "weight": n.weight,
                }
                for n in self.nets
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Netlist":
        """Deserialize netlist from a dictionary."""
        return cls(
            version=str(d["version"]),
            grid_w=int(d["grid_w"]),
            grid_h=int(d["grid_h"]),
            macros=tuple(
                Macro(id=int(m["id"]), w=int(m["w"]), h=int(m["h"]))
                for m in d["macros"]
            ),
            nets=tuple(
                Net(
                    id=int(n["id"]),
                    macro_ids=tuple(int(mid) for mid in n["macro_ids"]),
                    weight=float(n.get("weight", 1.0)),
                )
                for n in d["nets"]
            ),
        )


@dataclass(frozen=True)
class PlacedMacro:
    """Placement location for a macro."""

    macro_id: int
    x: int
    y: int


Placement = list[PlacedMacro]


def placement_to_dict(p: Placement) -> dict[int, tuple[int, int]]:
    """Convert placement to a mapping from macro_id to (x, y) coordinates."""
    return {pm.macro_id: (pm.x, pm.y) for pm in p}
