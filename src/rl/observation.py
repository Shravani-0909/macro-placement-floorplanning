"""Observation construction utilities for macro placement Gymnasium environment."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.placement.engine import PlacementEngine
from src.placement.hpwl import get_macro_center
from src.placement.models import Macro, Net, Netlist, Placement
from src.placement.objective import rudy_map


def build_connectivity_guidance_map(
    engine: PlacementEngine,
    current_macro: Macro | None,
    macro_to_nets: Mapping[int, Sequence[Net]],
) -> np.ndarray:
    """Build a 2D spatial channel showing pins of placed neighbors connected to current_macro.

    Parameters:
        engine: The active placement engine.
        current_macro: The macro currently being placed, or None if complete.
        macro_to_nets: Mapping from macro ID to list of incident Net objects.

    Returns:
        2D float32 ndarray of shape (grid_h, grid_w) where intensity reflects
        the weighted presence of already-placed connected neighbors.
    """
    guidance = np.zeros((engine.grid_h, engine.grid_w), dtype=np.float32)
    if current_macro is None:
        return guidance

    incident_nets = macro_to_nets.get(current_macro.id, ())
    for net in incident_nets:
        for neighbor_id in net.macro_ids:
            if neighbor_id == current_macro.id:
                continue
            if engine.is_placed(neighbor_id):
                placed = engine.placed_macros[neighbor_id]
                neighbor_def = engine.macro_map[neighbor_id]
                cx, cy = get_macro_center(
                    placed.x, placed.y, neighbor_def.w, neighbor_def.h
                )
                cell_x = min(math.floor(cx), engine.grid_w - 1)
                cell_y = min(math.floor(cy), engine.grid_h - 1)
                cell_x = max(0, cell_x)
                cell_y = max(0, cell_y)
                guidance[cell_y, cell_x] += float(net.weight)

    return guidance


def build_grid_observation(
    engine: PlacementEngine,
    current_macro: Macro | None,
    netlist: Netlist,
    macro_to_nets: Mapping[int, Sequence[Net]],
) -> np.ndarray:
    """Build the 4-channel spatial observation tensor.

    Channels:
        0: Binary occupancy grid (1.0 = occupied, 0.0 = free).
        1: Legal placement top-left mask for current_macro (1.0 = legal, 0.0 = illegal).
        2: Partial RUDY routing demand congestion map from already-placed nets.
        3: Net connectivity guidance map showing placed neighbors of current_macro.

    Returns:
        float32 ndarray of shape (4, grid_h, grid_w).
    """
    h, w = engine.grid_h, engine.grid_w

    # Channel 0: Occupancy
    ch0_occupancy = engine.occupancy_grid.astype(np.float32)

    # Channel 1: Legal placement mask
    if current_macro is not None and not engine.is_placed(current_macro.id):
        ch1_mask = engine.compute_action_mask(current_macro.id, flatten=False).astype(np.float32)
    else:
        ch1_mask = np.zeros((h, w), dtype=np.float32)

    # Channel 2: Partial RUDY map
    placement = engine.get_placement()
    if len(placement) >= 2:
        ch2_rudy = rudy_map(netlist, placement).astype(np.float32)
    else:
        ch2_rudy = np.zeros((h, w), dtype=np.float32)

    # Channel 3: Net Connectivity Guidance
    ch3_guidance = build_connectivity_guidance_map(engine, current_macro, macro_to_nets)

    return np.stack([ch0_occupancy, ch1_mask, ch2_rudy, ch3_guidance], axis=0)


def build_macro_features(
    current_macro: Macro | None,
    step: int,
    total_macros: int,
    grid_w: int,
    grid_h: int,
    incident_weight: float = 0.0,
    max_incident_weight: float = 1.0,
) -> np.ndarray:
    """Build a 1D normalized vector of active macro features and placement progress.

    Vector entries (6 dimensions, all in [0.0, 1.0]):
        [0]: normalized width (w / grid_w)
        [1]: normalized height (h / grid_h)
        [2]: normalized area ((w * h) / (grid_w * grid_h))
        [3]: normalized step progress (step / total_macros)
        [4]: normalized remaining macros ((total_macros - step) / total_macros)
        [5]: normalized incident net weight (incident_weight / max_incident_weight)

    Returns:
        float32 ndarray of shape (6,).
    """
    if current_macro is None or total_macros <= 0:
        return np.zeros((6,), dtype=np.float32)

    norm_w = min(1.0, max(0.0, float(current_macro.w) / float(grid_w)))
    norm_h = min(1.0, max(0.0, float(current_macro.h) / float(grid_h)))
    norm_area = min(1.0, max(0.0, float(current_macro.w * current_macro.h) / float(grid_w * grid_h)))
    norm_step = min(1.0, max(0.0, float(step) / float(total_macros)))
    norm_rem = min(1.0, max(0.0, float(total_macros - step) / float(total_macros)))

    safe_max_weight = max(1e-6, max_incident_weight)
    norm_weight = min(1.0, max(0.0, float(incident_weight) / safe_max_weight))

    return np.array(
        [norm_w, norm_h, norm_area, norm_step, norm_rem, norm_weight],
        dtype=np.float32,
    )


class FlattenPlacementObservation(gym.ObservationWrapper):
    """Flattens and normalizes dictionary observations into a single 1D Box vector for MlpPolicy.

    Converts Dict observation containing:
        - "grid": shape (4, H, W)
        - "macro_features": shape (6,)
    into a flat float32 Box of shape (4 * H * W + 6,) where every feature is
    strictly scaled into [0.0, 1.0].

    Feature Scaling Specification:
        - Channel 0 (Occupancy): already binary {0.0, 1.0}.
        - Channel 1 (Legal Action Mask): already binary {0.0, 1.0}.
        - Channel 2 (RUDY Routing Demand): nonlinear scaling x / (1.0 + x) mapping [0, inf) -> [0, 1).
          Even extreme values reaching 100,000.0 map strictly within [0.0, 1.0).
        - Channel 3 (Net Connectivity Guidance): nonlinear scaling x / (1.0 + x) mapping [0, inf) -> [0, 1).
        - Macro Features (6D): already normalized into [0.0, 1.0].
    """

    def __init__(self, env: gym.Env) -> None:
        """Initialize the observation flattening wrapper."""
        super().__init__(env)
        grid_w = getattr(env.unwrapped, "grid_w", None)
        grid_h = getattr(env.unwrapped, "grid_h", None)
        if grid_w is None or grid_h is None:
            dict_space = env.observation_space
            assert isinstance(dict_space, spaces.Dict)
            grid_space = dict_space["grid"]
            assert isinstance(grid_space, spaces.Box)
            grid_h = int(grid_space.shape[1])
            grid_w = int(grid_space.shape[2])

        self.flat_dim: int = int(4 * grid_w * grid_h + 6)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.flat_dim,),
            dtype=np.float32,
        )

    @staticmethod
    def scale_and_flatten_observation(obs: Mapping[str, np.ndarray]) -> np.ndarray:
        """Scale and flatten a dictionary observation into a normalized 1D float32 vector.

        Applies monotonic x / (1.0 + x) scaling to unbounded grid channels so that
        all features strictly lie within [0.0, 1.0] without saturation or gradient explosion.
        """
        grid = obs["grid"]
        macro_feats = obs["macro_features"]

        # Channel 0: Occupancy (0.0 or 1.0)
        ch0 = np.clip(grid[0].ravel(), 0.0, 1.0)
        # Channel 1: Legal placement mask (0.0 or 1.0)
        ch1 = np.clip(grid[1].ravel(), 0.0, 1.0)
        # Channel 2: RUDY demand scaled monotonically into [0.0, 1.0)
        ch2_raw = np.maximum(0.0, grid[2].ravel())
        ch2 = ch2_raw / (1.0 + ch2_raw)
        # Channel 3: Net guidance scaled monotonically into [0.0, 1.0)
        ch3_raw = np.maximum(0.0, grid[3].ravel())
        ch3 = ch3_raw / (1.0 + ch3_raw)

        # Macro features (6D normalized)
        norm_feats = np.clip(macro_feats.ravel(), 0.0, 1.0)

        return np.concatenate([ch0, ch1, ch2, ch3, norm_feats], axis=0).astype(np.float32)

    def observation(self, observation: dict[str, Any]) -> np.ndarray:
        """Convert dictionary observation to flattened and scaled 1D vector."""
        return self.scale_and_flatten_observation(observation)

    def action_masks(self) -> np.ndarray:
        """Forward action_masks call to inner environment for MaskablePPO."""
        if hasattr(self.env, "action_masks"):
            return self.env.action_masks()  # type: ignore[no-any-return]
        return self.env.unwrapped.action_masks()  # type: ignore[attr-defined, no-any-return]

    def get_placement(self) -> Placement:
        """Forward get_placement call to inner environment."""
        if hasattr(self.env, "get_placement"):
            return self.env.get_placement()  # type: ignore[no-any-return]
        return self.env.unwrapped.get_placement()  # type: ignore[attr-defined, no-any-return]

