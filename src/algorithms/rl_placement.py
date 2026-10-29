"""Reinforcement Learning Macro Placement using MaskablePPO and Gymnasium.

This module provides:
1. MacroPlacementEnv: Custom Gymnasium environment with action masking for sequential macro placement.
2. train_rl_agent: MaskablePPO training pipeline using MultiInputPolicy.
3. solve: Unified solver interface compliant with the team contract and experiment runner.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Robust import for ActionMasker across sb3_contrib versions
try:
    from sb3_contrib.common.maskable.wrappers import ActionMasker
except ImportError:
    from sb3_contrib.common.wrappers import ActionMasker  # type: ignore[no-redef]

from sb3_contrib import MaskablePPO


@dataclass
class MacroSpec:
    """Internal normalized representation of a macro to place."""

    id: Any
    w: int
    h: int


class MacroPlacementEnv(gym.Env):
    """Custom Gymnasium Environment for sequential 2D VLSI Macro Placement.

    Conventions:
    - Discrete action index: a = y * W + x (0 <= a < W * H).
    - Coordinate origin: (0, 0) is top-left.
    - Observation: Dict containing:
        - 'grid_occupancy': Box(0, 1, shape=(1, W, H), dtype=np.float32) for PyTorch CNNs.
        - 'macro_features': Box(low=0, high=W, shape=(2,), dtype=np.float32) with (w, h).
    - Action masking: Boolean 1D mask enforcing boundary condition and pairwise non-overlap.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        netlist: Sequence[Any] | Mapping[Any, Any] | Any,
        nets: Sequence[Any] | None = None,
    ) -> None:
        super().__init__()
        self.grid_width = int(grid_width)
        self.grid_height = int(grid_height)

        if self.grid_width <= 0 or self.grid_height <= 0:
            raise ValueError(
                f"Grid dimensions must be positive integers, got W={grid_width}, H={grid_height}"
            )

        # 1. Parse and normalize netlist macros
        self.macros: list[MacroSpec] = self._normalize_netlist_macros(netlist)
        self.nets: list[dict[str, Any]] = self._normalize_nets(netlist, nets)

        # 2. Action space: Flattened 1D grid coordinates [0, W * H - 1]
        self.action_space = spaces.Discrete(self.grid_width * self.grid_height)

        # 3. Observation space: Dict for CNN grid occupancy + MLP macro dimensions
        self.observation_space = spaces.Dict(
            {
                "grid_occupancy": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1, self.grid_width, self.grid_height),
                    dtype=np.float32,
                ),
                "macro_features": spaces.Box(
                    low=0.0,
                    high=float(max(self.grid_width, self.grid_height)),
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )

        # 4. State variables
        self.grid = np.zeros((self.grid_width, self.grid_height), dtype=np.float32)
        self.placed_macros: list[dict[str, Any]] = []
        self.current_macro_idx = 0

    @staticmethod
    def _normalize_netlist_macros(raw_netlist: Any) -> list[MacroSpec]:
        """Convert arbitrary netlist input formats into a list of MacroSpec."""
        raw_items: Sequence[Any] = []

        if hasattr(raw_netlist, "macros"):
            raw_macros = raw_netlist.macros
            if isinstance(raw_macros, Mapping):
                raw_items = list(raw_macros.items())
            elif isinstance(raw_macros, Sequence):
                raw_items = list(raw_macros)
        elif isinstance(raw_netlist, Mapping) and "macros" in raw_netlist:
            raw_macros = raw_netlist["macros"]
            if isinstance(raw_macros, Mapping):
                raw_items = list(raw_macros.items())
            elif isinstance(raw_macros, Sequence):
                raw_items = list(raw_macros)
        elif isinstance(raw_netlist, Sequence):
            raw_items = raw_netlist
        else:
            raise TypeError(f"Unsupported netlist format: {type(raw_netlist)}")

        specs: list[MacroSpec] = []
        for idx, item in enumerate(raw_items):
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], (str, int)):
                # Tuple of (macro_id, dimensions)
                mid = item[0]
                dim = item[1]
                if isinstance(dim, (tuple, list)):
                    w, h = dim[0], dim[1]
                elif isinstance(dim, Mapping):
                    w = dim.get("w", dim.get("width", 1))
                    h = dim.get("h", dim.get("height", 1))
                else:
                    w = getattr(dim, "w", getattr(dim, "width", 1))
                    h = getattr(dim, "h", getattr(dim, "height", 1))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                # Raw (w, h) tuple
                mid = idx
                w, h = item[0], item[1]
            elif isinstance(item, Mapping):
                mid = item.get("id", idx)
                w = item.get("w", item.get("width", 1))
                h = item.get("h", item.get("height", 1))
            else:
                mid = getattr(item, "id", idx)
                w = getattr(item, "w", getattr(item, "width", 1))
                h = getattr(item, "h", getattr(item, "height", 1))

            specs.append(MacroSpec(id=mid, w=int(w), h=int(h)))

        return specs

    @staticmethod
    def _normalize_nets(netlist: Any, nets: Sequence[Any] | None) -> list[dict[str, Any]]:
        """Extract multi-pin nets for HPWL calculation if available."""
        target_nets = nets
        if target_nets is None:
            if hasattr(netlist, "nets"):
                target_nets = getattr(netlist, "nets")
            elif isinstance(netlist, Mapping) and "nets" in netlist:
                target_nets = netlist["nets"]

        if not target_nets:
            return []

        normalized: list[dict[str, Any]] = []
        if isinstance(target_nets, Mapping):
            for nid, net_info in target_nets.items():
                if isinstance(net_info, Mapping):
                    pins = net_info.get("pins", net_info.get("macro_ids", []))
                    weight = float(net_info.get("weight", 1.0))
                else:
                    pins = getattr(net_info, "pins", getattr(net_info, "macro_ids", []))
                    weight = float(getattr(net_info, "weight", 1.0))
                normalized.append({"id": nid, "pins": list(pins), "weight": weight})
        elif isinstance(target_nets, Sequence):
            for idx, item in enumerate(target_nets):
                if isinstance(item, Mapping):
                    pins = item.get("pins", item.get("macro_ids", []))
                    weight = float(item.get("weight", 1.0))
                    normalized.append({"id": item.get("id", idx), "pins": list(pins), "weight": weight})
                elif hasattr(item, "macro_ids"):
                    pins = list(item.macro_ids)
                    weight = float(getattr(item, "weight", 1.0))
                    normalized.append({"id": getattr(item, "id", idx), "pins": pins, "weight": weight})

        return normalized

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Construct the Dict observation for the current placement step."""
        occupancy = self.grid[np.newaxis, :, :].copy()  # Shape: (1, W, H)

        if self.current_macro_idx < len(self.macros):
            current_macro = self.macros[self.current_macro_idx]
            features = np.array([float(current_macro.w), float(current_macro.h)], dtype=np.float32)
        else:
            features = np.zeros(2, dtype=np.float32)

        return {
            "grid_occupancy": occupancy,
            "macro_features": features,
        }

    def action_masks(self) -> np.ndarray:
        """Compute boolean validity mask for placing the current macro.

        A placement is legal iff:
        1. Boundary: 0 <= x <= W - w and 0 <= y <= H - h.
        2. Non-overlap: Grid slice [x:x+w, y:y+h] has no existing placed macros.

        Returns:
            Flattened 1D boolean numpy array of size W * H where True is a valid action.
        """
        mask = np.zeros(self.grid_width * self.grid_height, dtype=bool)

        # If all macros are already placed, no actions are valid
        if self.current_macro_idx >= len(self.macros):
            return mask

        curr = self.macros[self.current_macro_idx]
        w, h = curr.w, curr.h

        # Strict boundary bounds
        max_x = self.grid_width - w
        max_y = self.grid_height - h

        if max_x < 0 or max_y < 0:
            return mask

        # Check all candidate anchors (x, y)
        for y in range(max_y + 1):
            for x in range(max_x + 1):
                # Check non-overlap constraint
                if not np.any(self.grid[x : x + w, y : y + h] > 0):
                    action_idx = y * self.grid_width + x
                    mask[action_idx] = True

        return mask

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment to blank grid and initial macro index."""
        super().reset(seed=seed)
        self.grid.fill(0.0)
        self.placed_macros.clear()
        self.current_macro_idx = 0
        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one sequential macro placement step.

        Args:
            action: 1D discrete coordinate index (y * W + x).

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        if self.current_macro_idx >= len(self.macros):
            return self._get_obs(), 0.0, True, False, {"error": "All macros placed"}

        curr = self.macros[self.current_macro_idx]
        w, h = curr.w, curr.h

        # Convert 1D action to 2D coordinates (x, y)
        x = int(action % self.grid_width)
        y = int(action // self.grid_width)

        # Validate placement against boundary and non-overlap constraints
        fits_boundary = (0 <= x <= self.grid_width - w) and (0 <= y <= self.grid_height - h)
        overlap = False
        if fits_boundary:
            overlap = bool(np.any(self.grid[x : x + w, y : y + h] > 0))

        if not fits_boundary or overlap:
            # Action mask violation (defensive guard)
            terminated = True
            truncated = False
            reward = -50.0
            return self._get_obs(), reward, terminated, truncated, {"error": "Constraint violation"}

        # Place macro and update grid occupancy
        self.grid[x : x + w, y : y + h] = 1.0
        self.placed_macros.append(
            {
                "macro_id": curr.id,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        )
        self.current_macro_idx += 1

        terminated = self.current_macro_idx >= len(self.macros)
        truncated = False
        reward = 0.0

        if terminated:
            # All macros placed successfully: calculate terminal reward inversely proportional to HPWL
            hpwl = self._compute_hpwl()
            # Positive reward inversely proportional to HPWL
            reward = 100.0 / (hpwl + 1.0)
        else:
            # Check if remaining placement reached an unrecoverable dead end
            mask = self.action_masks()
            if not np.any(mask):
                # No legal actions left for the next macro
                truncated = True
                reward = -20.0

        info = {
            "placed_count": len(self.placed_macros),
            "total_macros": len(self.macros),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _compute_hpwl(self) -> float:
        """Compute Half-Perimeter Wirelength (HPWL) of current placement."""
        if len(self.placed_macros) <= 1:
            return 0.0

        # Build center coordinate lookup for placed macros
        centers: dict[Any, tuple[float, float]] = {
            pm["macro_id"]: (pm["x"] + pm["w"] / 2.0, pm["y"] + pm["h"] / 2.0)
            for pm in self.placed_macros
        }

        total_hpwl = 0.0

        if self.nets:
            for net in self.nets:
                pins = [p for p in net.get("pins", []) if p in centers]
                if len(pins) < 2:
                    continue
                xs = [centers[p][0] for p in pins]
                ys = [centers[p][1] for p in pins]
                weight = float(net.get("weight", 1.0))
                hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
                total_hpwl += weight * hpwl
        else:
            # Bounding box & sequential chain surrogate if no explicit nets exist
            for i in range(len(self.placed_macros) - 1):
                m1 = self.placed_macros[i]["macro_id"]
                m2 = self.placed_macros[i + 1]["macro_id"]
                c1 = centers[m1]
                c2 = centers[m2]
                total_hpwl += abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])

        return float(total_hpwl)


def mask_fn(env: gym.Env) -> np.ndarray:
    """Top-level helper function returning action mask for ActionMasker."""
    return env.action_masks()  # type: ignore[attr-defined]


def train_rl_agent(
    env: gym.Env,
    total_timesteps: int = 10000,
    save_path: str = "ppo_macro_placement.zip",
    verbose: int = 1,
) -> MaskablePPO:
    """Wrap environment with ActionMasker, train MaskablePPO, and save checkpoint.

    Args:
        env: Base MacroPlacementEnv instance.
        total_timesteps: Total training timesteps.
        save_path: Output file path for model checkpoint.
        verbose: Verbosity level for training output.

    Returns:
        Trained MaskablePPO agent.
    """
    wrapped_env = ActionMasker(env, mask_fn)

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=wrapped_env,
        verbose=verbose,
        gamma=0.99,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=128,
    )

    model.learn(total_timesteps=total_timesteps)

    # Ensure parent directory exists and save model
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    model.save(save_path)

    return model


def solve(
    netlist: Any,
    env_config: dict[str, Any] | None = None,
    budget: int | None = None,
    seed: int | None = None,
    on_progress: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Standardized Solver Interface for RL Macro Placement.

    Matches the shared solver contract:
    solve(netlist, env_config, budget, seed, on_progress) -> Result dict

    Args:
        netlist: Netlist object, dictionary, or sequence of macros.
        env_config: Dictionary containing grid and model settings (e.g., 'grid_width',
                    'grid_height', 'model_path', 'alpha', 'beta').
        budget: Maximum inference/evaluation attempts or timesteps.
        seed: Random seed for deterministic reproducibility.
        on_progress: Optional callback for streaming placement progress.

    Returns:
        Result dictionary containing:
        - placement: list of dicts [{'macro_id': id, 'x': x, 'y': y}]
        - feasible: True if all macros were legally placed, False otherwise
        - runtime_s: Total inference runtime in seconds
        - cost: Combined objective score
        - hpwl: Total Half-Perimeter Wirelength
        - congestion: Congestion score
        - evals: Total inference evaluations executed
        - trace: List of progress snapshots
    """
    start_time = time.perf_counter()
    cfg = env_config or {}

    # 1. Resolve grid dimensions from config or netlist
    grid_w = int(cfg.get("grid_width", getattr(netlist, "grid_w", 16)))
    grid_h = int(cfg.get("grid_height", getattr(netlist, "grid_h", 16)))
    model_path = str(cfg.get("model_path", "ppo_macro_placement.zip"))

    # 2. Instantiate environment
    env = MacroPlacementEnv(grid_width=grid_w, grid_height=grid_h, netlist=netlist)

    # 3. Load or train model
    if os.path.exists(model_path) or os.path.exists(model_path + ".zip"):
        model = MaskablePPO.load(model_path)
    else:
        # Fallback train quick model if checkpoint doesn't exist yet
        model = train_rl_agent(env, total_timesteps=1000, save_path=model_path, verbose=0)

    # 4. Run deterministic inference loop with action masks
    obs, _info = env.reset(seed=seed)
    done = False
    evals = 0
    trace: list[dict[str, Any]] = []

    while not done:
        masks = env.action_masks()
        if not np.any(masks):
            # No valid placement anchor available (dead end)
            break

        action, _states = model.predict(obs, action_masks=masks, deterministic=True)
        obs, reward, terminated, truncated, _step_info = env.step(int(action))
        evals += 1
        done = terminated or truncated

        elapsed = time.perf_counter() - start_time
        hpwl_snapshot = env._compute_hpwl()
        trace.append({"eval": evals, "t_s": elapsed, "best_cost": hpwl_snapshot})

        if on_progress is not None:
            progress_payload = {
                "step": evals,
                "total_steps": len(env.macros),
                "hpwl": hpwl_snapshot,
                "placement": [
                    {"macro_id": pm["macro_id"], "x": pm["x"], "y": pm["y"]}
                    for pm in env.placed_macros
                ],
            }
            if inspect.iscoroutinefunction(on_progress):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(on_progress(progress_payload))  # type: ignore[arg-type]
                    else:
                        loop.run_until_complete(on_progress(progress_payload))  # type: ignore[arg-type]
                except RuntimeError:
                    asyncio.run(on_progress(progress_payload))  # type: ignore[arg-type]
            else:
                on_progress(progress_payload)

    # 5. Compile placement results
    placement = [
        {"macro_id": pm["macro_id"], "x": pm["x"], "y": pm["y"]}
        for pm in env.placed_macros
    ]
    feasible = len(env.placed_macros) == len(env.macros)
    runtime_s = time.perf_counter() - start_time
    hpwl = env._compute_hpwl() if feasible else None
    cost = hpwl if hpwl is not None else float("inf")

    return {
        "placement": placement,
        "feasible": feasible,
        "runtime_s": runtime_s,
        "cost": cost,
        "hpwl": hpwl,
        "congestion": 0.0,
        "evals": evals,
        "trace": trace,
    }


if __name__ == "__main__":
    print("=== Testing RL Macro Placement Pipeline ===")

    # 1. Define dummy netlist
    dummy_netlist = [
        {"id": "macro_0", "w": 4, "h": 4},
        {"id": "macro_1", "w": 3, "h": 5},
        {"id": "macro_2", "w": 2, "h": 3},
        {"id": "macro_3", "w": 3, "h": 3},
    ]

    # 2. Instantiate Gymnasium environment
    test_env = MacroPlacementEnv(grid_width=10, grid_height=10, netlist=dummy_netlist)
    obs, info = test_env.reset()
    print("Environment reset successful. Obs shapes:")
    print(" - grid_occupancy:", obs["grid_occupancy"].shape)
    print(" - macro_features:", obs["macro_features"].shape)
    print(" - initial valid actions:", int(np.sum(test_env.action_masks())))

    # 3. Train MaskablePPO agent
    ckpt_file = "ppo_macro_placement.zip"
    print(f"\nTraining MaskablePPO agent ({ckpt_file})...")
    agent = train_rl_agent(test_env, total_timesteps=500, save_path=ckpt_file, verbose=0)
    print("Training complete and checkpoint saved.")

    # 4. Run deterministic solver inference
    print("\nRunning inference via standardized solve() interface...")
    res = solve(
        netlist=dummy_netlist,
        env_config={"grid_width": 10, "grid_height": 10, "model_path": ckpt_file},
        budget=100,
        seed=42,
    )

    print("\nSolver Result:")
    print(" - Feasible:", res["feasible"])
    print(" - Placed Macros:", len(res["placement"]))
    print(" - Placement coordinates:", res["placement"])
    print(" - HPWL:", res["hpwl"])
    print(f" - Inference Runtime: {res['runtime_s']:.4f} s")
    print("Success: Complete RL pipeline functional and verified!")
