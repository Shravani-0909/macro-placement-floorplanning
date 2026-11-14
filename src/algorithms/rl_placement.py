"""Reinforcement Learning Macro Placement using PlacementEngine, MaskablePPO, and Gymnasium.

This module provides:
1. MacroPlacementEnv: Custom Gymnasium environment wrapping the shared PlacementEngine
   with canonical action masking, objective evaluation, and sequential placement.
2. train_rl_agent: MaskablePPO training pipeline using MultiInputPolicy.
3. solve: Standardized solver interface compliant with the team contract.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Optional RL dependencies with informative error messages
try:
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYMNASIUM = True
except ImportError:
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    _HAS_GYMNASIUM = False

try:
    from sb3_contrib import MaskablePPO

    try:
        from sb3_contrib.common.maskable.wrappers import ActionMasker
    except ImportError:
        from sb3_contrib.common.wrappers import ActionMasker  # type: ignore[no-redef]
    _HAS_SB3 = True
except ImportError:
    MaskablePPO = None  # type: ignore[assignment, misc]
    ActionMasker = None  # type: ignore[assignment, misc]
    _HAS_SB3 = False

try:
    import torch
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    BaseFeaturesExtractor = object  # type: ignore[assignment, misc]
    _HAS_TORCH = False

from src.algorithms.common import to_canonical_netlist
from src.placement.constraints import validate_placement
from src.placement.engine import PlacementEngine, action_to_coord, coord_to_action
from src.placement.models import Macro, Netlist, PlacedMacro, Placement
from src.placement.objective import EvaluationResult, evaluate_placement

_BaseEnv: Any = gym.Env if gym is not None else object


def compute_connectivity_features(
    netlist: Any,
    placed_positions: Mapping[int, tuple[int, int]] | Sequence[PlacedMacro] | None = None,
    current_macro_idx: int = 0,
    cached_graph: tuple[dict[int, int], np.ndarray, np.ndarray, float, float] | None = None,
) -> np.ndarray:
    """Compute normalized, fixed-size connectivity and spatial context features for a macro placement step.

    Feature Vector Layout (size: 4 + 5 + 6 * N, where N = len(netlist.macros)):
    - [0:4] Current Macro Self-Features:
        0: norm_w = w / grid_w in (0, 1]
        1: norm_h = h / grid_h in (0, 1]
        2: norm_id = current_macro_idx / max(1, N - 1) in [0, 1] (deterministic macro identity)
        3: norm_step = current_macro_idx / max(1, N) in [0, 1) (placement episode progress)
    - [4:9] Aggregate Connectivity Context:
        4: norm_placed_connected = (number of placed connected macros) / max(1, N - 1)
        5: norm_unplaced_connected = (number of unplaced connected macros) / max(1, N - 1)
        6: norm_centroid_x = connection-weighted X center of placed connected pins / grid_w (0.5 if none)
        7: norm_centroid_y = connection-weighted Y center of placed connected pins / grid_h (0.5 if none)
        8: norm_total_weight = total net weight incident to current macro / max_macro_weight
    - [9 : 9 + 6*N] Per-Macro Slot Features (6 values per macro j in [0, N-1]):
        base + 0: is_placed = 1.0 if placed, 0.0 otherwise
        base + 1: norm_x = macro_j_x / grid_w if placed, 0.0 otherwise
        base + 2: norm_y = macro_j_y / grid_h if placed, 0.0 otherwise
        base + 3: norm_w = macro_j_w / grid_w
        base + 4: norm_h = macro_j_h / grid_h
        base + 5: connection_strength = net_weight(current_macro, macro_j) / max_conn_weight

    All values are continuous, normalized in [0.0, 1.0], with dtype np.float32.
    """
    canonical_nl = to_canonical_netlist(netlist)
    macros = list(canonical_nl.macros)
    num_macros = len(macros)
    if num_macros == 0:
        return np.zeros(0, dtype=np.float32)

    if cached_graph is not None:
        id_to_idx, adj_matrix, macro_net_weights, max_conn_weight, max_macro_weight = cached_graph
    else:
        id_to_idx = {m.id: i for i, m in enumerate(macros)}
        adj_matrix = np.zeros((num_macros, num_macros), dtype=np.float32)
        macro_net_weights = np.zeros(num_macros, dtype=np.float32)

        for net in canonical_nl.nets:
            wt = float(getattr(net, "weight", 1.0))
            mids = [id_to_idx[mid] for mid in net.macro_ids if mid in id_to_idx]
            for i_idx in mids:
                macro_net_weights[i_idx] += wt
                for j_idx in mids:
                    if i_idx != j_idx:
                        adj_matrix[i_idx, j_idx] += wt

        max_conn_weight = max(1.0, float(np.max(adj_matrix)))
        max_macro_weight = max(1.0, float(np.max(macro_net_weights)))

    # Parse placed positions into dict mapping macro_id -> (x, y)
    pos_map: dict[int, tuple[int, int]] = {}
    if placed_positions is not None:
        if isinstance(placed_positions, Mapping):
            pos_map = {int(k): (int(v[0]), int(v[1])) for k, v in placed_positions.items()}
        else:
            for pm in placed_positions:
                pos_map[int(pm.macro_id)] = (int(pm.x), int(pm.y))

    grid_w = float(canonical_nl.grid_w)
    grid_h = float(canonical_nl.grid_h)
    feat = np.zeros(4 + 5 + 6 * num_macros, dtype=np.float32)

    if 0 <= current_macro_idx < num_macros:
        curr_m = macros[current_macro_idx]

        # 1. Self features
        feat[0] = float(curr_m.w) / grid_w
        feat[1] = float(curr_m.h) / grid_h
        feat[2] = float(current_macro_idx) / max(1.0, float(num_macros - 1))
        feat[3] = float(current_macro_idx) / max(1.0, float(num_macros))

        # 2. Aggregate features of connected macros
        conn_mask = adj_matrix[current_macro_idx] > 0.0
        conn_indices = np.where(conn_mask)[0]

        placed_conn = [j for j in conn_indices if macros[j].id in pos_map]
        unplaced_conn = [j for j in conn_indices if macros[j].id not in pos_map]

        denom = max(1.0, float(num_macros - 1))
        feat[4] = float(len(placed_conn)) / denom
        feat[5] = float(len(unplaced_conn)) / denom

        if placed_conn:
            tot_w = 0.0
            sum_x = 0.0
            sum_y = 0.0
            for j in placed_conn:
                mj = macros[j]
                pos = pos_map[mj.id]
                cx = float(pos[0]) + float(mj.w) / 2.0
                cy = float(pos[1]) + float(mj.h) / 2.0
                w_ij = float(adj_matrix[current_macro_idx, j])
                sum_x += w_ij * cx
                sum_y += w_ij * cy
                tot_w += w_ij

            if tot_w > 0.0:
                feat[6] = (sum_x / tot_w) / grid_w
                feat[7] = (sum_y / tot_w) / grid_h
            else:
                feat[6] = 0.5
                feat[7] = 0.5
        else:
            feat[6] = 0.5
            feat[7] = 0.5

        feat[8] = float(macro_net_weights[current_macro_idx]) / float(max_macro_weight)

        # 3. Per-macro slot features
        offset = 9
        for j, mj in enumerate(macros):
            base = offset + j * 6
            if mj.id in pos_map:
                feat[base] = 1.0
                pos = pos_map[mj.id]
                feat[base + 1] = float(pos[0]) / grid_w
                feat[base + 2] = float(pos[1]) / grid_h
            else:
                feat[base] = 0.0
                feat[base + 1] = 0.0
                feat[base + 2] = 0.0

            feat[base + 3] = float(mj.w) / grid_w
            feat[base + 4] = float(mj.h) / grid_h
            feat[base + 5] = float(adj_matrix[current_macro_idx, j]) / float(max_conn_weight)

    return feat


if _HAS_TORCH and _HAS_SB3:

    class MacroPlacementFeaturesExtractor(BaseFeaturesExtractor):
        """Custom Multi-Input Features Extractor for MacroPlacementEnv.

        Processes:
        - 'grid_occupancy' (shape: [B, 1, H, W]) via a 2D spatial Convolutional Neural Network
          with adaptive pooling to preserve spatial geometry across arbitrary grid resolutions.
        - 'macro_features' (shape: [B, 2]) and 'connectivity_features' (shape: [B, K]) via an MLP.
        Combines spatial and graph-connectivity embeddings into a unified feature vector.
        """

        def __init__(
            self,
            observation_space: spaces.Dict,
            features_dim: int = 128,
        ) -> None:
            super().__init__(observation_space, features_dim=features_dim)

            # 1. Spatial CNN for grid occupancy
            occ_space = observation_space["grid_occupancy"]
            n_input_channels = occ_space.shape[0]  # 1

            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(32 * 4 * 4, 64),
                nn.ReLU(),
            )

            # 2. MLP for macro dimensions and connectivity features
            macro_dim = int(np.prod(observation_space["macro_features"].shape))
            conn_dim = int(np.prod(observation_space["connectivity_features"].shape))
            vector_in_dim = macro_dim + conn_dim

            self.mlp = nn.Sequential(
                nn.Linear(vector_in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )

            # 3. Fusion projection layer
            self.fusion = nn.Sequential(
                nn.Linear(64 + 64, features_dim),
                nn.ReLU(),
            )

        def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
            grid_occ = observations["grid_occupancy"]
            macro_feat = torch.flatten(observations["macro_features"], start_dim=1)
            conn_feat = torch.flatten(observations["connectivity_features"], start_dim=1)

            vector_inputs = torch.cat([macro_feat, conn_feat], dim=1)

            spatial_emb = self.cnn(grid_occ)
            vector_emb = self.mlp(vector_inputs)

            fused = torch.cat([spatial_emb, vector_emb], dim=1)
            return self.fusion(fused)

else:
    MacroPlacementFeaturesExtractor = None  # type: ignore[assignment, misc]


class MacroPlacementEnv(_BaseEnv):
    """Gymnasium Environment adapter wrapping the canonical PlacementEngine.

    Conventions:
    - Origin (0, 0) is top-left.
    - Grid dimensions: width W = netlist.grid_w, height H = netlist.grid_h.
    - Discrete action index: a = y * W + x (0 <= a < W * H).
    - Coordinates: x = a % W, y = a // W (action_to_coord).
    - Observation: Dict containing:
        - 'grid_occupancy': Box(low=0.0, high=1.0, shape=(1, H, W), dtype=np.float32)
          reflecting engine.occupancy_grid.
        - 'macro_features': Box(low=0.0, high=float(max(W, H)), shape=(2,), dtype=np.float32)
          reflecting current macro (w, h).
    - Action masking: Boolean 1D mask from engine.compute_action_mask(macro_id, flatten=True).
    - Objective: Evaluated via evaluate_placement(netlist, placement, alpha, beta),
      J(P) = alpha * HPWL(P) + beta * C_score(P).
    - Reward: Step reward based on objective improvement:
        reward = previous_objective - current_objective.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        netlist_or_w: Netlist | Any = None,
        grid_height: int | None = None,
        netlist: Any = None,
        *,
        grid_width: int | None = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        terminal_bonus: float = 0.0,
        dead_end_penalty: float = 20.0,
        illegal_action_penalty: float = 50.0,
        reward_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if not _HAS_GYMNASIUM:
            raise ImportError(
                "Gymnasium is required for MacroPlacementEnv. "
                "Install gymnasium>=1.4.0."
            )
        super().__init__()

        # Resolve netlist from diverse calling signatures (canonical, positional, or kwargs)
        target_netlist: Any
        if isinstance(netlist_or_w, (int, np.integer)):
            gw = int(netlist_or_w)
            gh = int(grid_height) if grid_height is not None else 10
            raw_nl = netlist if netlist is not None else kwargs.get("netlist", {})
            if isinstance(raw_nl, Mapping) and "grid_w" not in raw_nl:
                target_netlist = dict(raw_nl, grid_w=gw, grid_h=gh)
            elif isinstance(raw_nl, Sequence) and not isinstance(raw_nl, (str, bytes)):
                target_netlist = {"grid_w": gw, "grid_h": gh, "macros": raw_nl, "nets": kwargs.get("nets", [])}
            else:
                target_netlist = raw_nl
        else:
            raw_nl = netlist_or_w if netlist_or_w is not None else netlist
            if raw_nl is None:
                raw_nl = kwargs.get("netlist")
            gw_val = grid_width if grid_width is not None else kwargs.get("grid_w")
            gh_val = grid_height if grid_height is not None else kwargs.get("grid_h")
            if gw_val is not None and gh_val is not None:
                if isinstance(raw_nl, Mapping) and "grid_w" not in raw_nl:
                    target_netlist = dict(raw_nl, grid_w=int(gw_val), grid_h=int(gh_val))
                elif isinstance(raw_nl, Sequence) and not isinstance(raw_nl, (str, bytes)):
                    target_netlist = {
                        "grid_w": int(gw_val),
                        "grid_h": int(gh_val),
                        "macros": raw_nl,
                        "nets": kwargs.get("nets", []),
                    }
                else:
                    target_netlist = raw_nl
            else:
                target_netlist = raw_nl

        self.netlist: Netlist = to_canonical_netlist(target_netlist)
        self.alpha: float = float(alpha)
        self.beta: float = float(beta)
        self.terminal_bonus: float = float(terminal_bonus)
        self.dead_end_penalty: float = float(dead_end_penalty)
        self.illegal_action_penalty: float = float(illegal_action_penalty)
        self.reward_scale: float = float(reward_scale)

        if self.reward_scale <= 0.0:
            raise ValueError(f"reward_scale must be a positive number, got {self.reward_scale!r}.")

        self.grid_w: int = self.netlist.grid_w
        self.grid_h: int = self.netlist.grid_h

        if self.grid_w <= 0 or self.grid_h <= 0:
            raise ValueError(
                f"Grid dimensions must be positive integers, got W={self.grid_w}, H={self.grid_h}"
            )

        # Canonical PlacementEngine instance
        self.engine: PlacementEngine = PlacementEngine(self.netlist)
        self.macros: list[Macro] = list(self.netlist.macros)
        num_macros = len(self.macros)

        # Precompute static connectivity graph structures for fast observation generation
        id_to_idx = {m.id: i for i, m in enumerate(self.macros)}
        adj_matrix = np.zeros((num_macros, num_macros), dtype=np.float32)
        macro_net_weights = np.zeros(num_macros, dtype=np.float32)

        for net in self.netlist.nets:
            wt = float(getattr(net, "weight", 1.0))
            mids = [id_to_idx[mid] for mid in net.macro_ids if mid in id_to_idx]
            for i_idx in mids:
                macro_net_weights[i_idx] += wt
                for j_idx in mids:
                    if i_idx != j_idx:
                        adj_matrix[i_idx, j_idx] += wt

        max_conn_weight = max(1.0, float(np.max(adj_matrix))) if num_macros > 0 else 1.0
        max_macro_weight = max(1.0, float(np.max(macro_net_weights))) if num_macros > 0 else 1.0
        self._cached_graph = (id_to_idx, adj_matrix, macro_net_weights, max_conn_weight, max_macro_weight)

        # Action space: discrete coordinates [0, grid_w * grid_h - 1]
        self.action_space = spaces.Discrete(self.grid_w * self.grid_h)

        # Observation space: 2D occupancy grid + current macro dimensions + rich connectivity context
        conn_dim = 4 + 5 + 6 * num_macros
        self.observation_space = spaces.Dict(
            {
                "grid_occupancy": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1, self.grid_h, self.grid_w),
                    dtype=np.float32,
                ),
                "macro_features": spaces.Box(
                    low=0.0,
                    high=float(max(self.grid_w, self.grid_h)),
                    shape=(2,),
                    dtype=np.float32,
                ),
                "connectivity_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(conn_dim,),
                    dtype=np.float32,
                ),
            }
        )

        # State tracking
        self.current_macro_idx: int = 0
        self.current_objective: float = 0.0

    @property
    def placed_macros(self) -> Placement:
        """Return placed macros as canonical Placement list."""
        return self.engine.get_placement()

    @property
    def grid(self) -> np.ndarray:
        """Return occupancy grid from PlacementEngine."""
        return self.engine.occupancy_grid

    @property
    def grid_width(self) -> int:
        """Grid width in cells."""
        return self.grid_w

    @property
    def grid_height(self) -> int:
        """Grid height in cells."""
        return self.grid_h

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Construct observation dictionary for current placement state."""
        occupancy = self.engine.occupancy_grid.astype(np.float32)[np.newaxis, :, :]  # (1, H, W)

        if self.current_macro_idx < len(self.macros):
            curr = self.macros[self.current_macro_idx]
            features = np.array([float(curr.w), float(curr.h)], dtype=np.float32)
        else:
            features = np.zeros(2, dtype=np.float32)

        conn_features = compute_connectivity_features(
            self.netlist,
            self.engine.placed_macros,
            self.current_macro_idx,
            cached_graph=self._cached_graph,
        )

        return {
            "grid_occupancy": occupancy,
            "macro_features": features,
            "connectivity_features": conn_features,
        }

    def action_masks(self) -> np.ndarray:
        """Compute boolean validity mask for placing the current macro.

        Delegates directly to PlacementEngine.compute_action_mask.

        Returns:
            Flattened 1D boolean numpy array of size grid_w * grid_h.
        """
        if self.current_macro_idx >= len(self.macros):
            return np.zeros(self.grid_w * self.grid_h, dtype=bool)

        curr = self.macros[self.current_macro_idx]
        return self.engine.compute_action_mask(curr.id, flatten=True)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment and underlying PlacementEngine."""
        super().reset(seed=seed)
        self.engine.reset()
        self.current_macro_idx = 0

        init_eval = evaluate_placement(
            self.netlist, self.engine.get_placement(), alpha=self.alpha, beta=self.beta
        )
        self.current_objective = float(init_eval.objective)

        info = {
            "placed_count": 0,
            "total_macros": len(self.macros),
            "hpwl": init_eval.hpwl,
            "congestion": init_eval.congestion,
            "objective": init_eval.objective,
            "is_complete": False,
        }
        return self._get_obs(), info

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
            return (
                self._get_obs(),
                0.0,
                True,
                False,
                {
                    "error": "All macros placed",
                    "placed_count": len(self.engine.placed_macros),
                    "total_macros": len(self.macros),
                    "is_complete": self.engine.is_complete(),
                },
            )

        curr = self.macros[self.current_macro_idx]
        x, y = action_to_coord(int(action), self.grid_w)

        # Check legality against PlacementEngine
        if not self.engine.is_legal(curr.id, x, y):
            obs = self._get_obs()
            reward = - (self.illegal_action_penalty / self.reward_scale)
            terminated = True
            truncated = False
            info = {
                "error": "Illegal action",
                "macro_id": curr.id,
                "action": int(action),
                "coord": (x, y),
                "placed_count": len(self.engine.placed_macros),
                "total_macros": len(self.macros),
                "is_complete": False,
            }
            return obs, reward, terminated, truncated, info

        # Place macro in shared engine
        self.engine.place(curr.id, x, y)
        self.current_macro_idx += 1

        # Evaluate objective using canonical objective module
        placement = self.engine.get_placement()
        eval_res = evaluate_placement(
            self.netlist, placement, alpha=self.alpha, beta=self.beta
        )
        new_objective = float(eval_res.objective)

        # Objective-aligned reward: improvement in objective yields positive reward, scaled deterministically
        raw_reward = float(self.current_objective - new_objective)
        reward = raw_reward / self.reward_scale
        self.current_objective = new_objective

        is_complete = self.engine.is_complete()
        terminated = is_complete or (self.current_macro_idx >= len(self.macros))
        truncated = False

        if terminated:
            reward += (self.terminal_bonus / self.reward_scale)
        else:
            # Check if next macro has legal moves
            next_macro = self.macros[self.current_macro_idx]
            next_mask = self.engine.compute_action_mask(next_macro.id, flatten=True)
            if not np.any(next_mask):
                truncated = True
                reward -= (self.dead_end_penalty / self.reward_scale)

        info = {
            "placed_count": len(self.engine.placed_macros),
            "total_macros": len(self.macros),
            "hpwl": eval_res.hpwl,
            "congestion": eval_res.congestion,
            "objective": eval_res.objective,
            "is_complete": is_complete,
            "placement": placement,
        }
        return self._get_obs(), float(reward), terminated, truncated, info


def mask_fn(env: Any) -> np.ndarray:
    """Top-level helper function returning action mask for ActionMasker."""
    return env.action_masks()  # type: ignore[attr-defined]


def train_rl_agent(
    env: Any,
    total_timesteps: int = 10000,
    save_path: str = "ppo_macro_placement.zip",
    verbose: int = 1,
    features_dim: int = 128,
) -> Any:
    """Wrap environment with ActionMasker, train MaskablePPO, and save checkpoint.

    Args:
        env: Base MacroPlacementEnv instance.
        total_timesteps: Total training timesteps.
        save_path: Output file path for model checkpoint.
        verbose: Verbosity level for training output.
        features_dim: Output dimension of custom feature extractor.

    Returns:
        Trained MaskablePPO agent.
    """
    if not _HAS_SB3:
        raise ImportError(
            "sb3-contrib is required to train MaskablePPO. "
            "Install sb3-contrib>=2.9.2."
        )

    wrapped_env = ActionMasker(env, mask_fn)

    policy_kwargs = None
    if MacroPlacementFeaturesExtractor is not None:
        policy_kwargs = {
            "features_extractor_class": MacroPlacementFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": features_dim},
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        }

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=wrapped_env,
        verbose=verbose,
        gamma=0.99,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=128,
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=total_timesteps)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    model.save(save_path)

    return model


@dataclass
class PPOTrainingConfig:
    """Hyperparameters and configuration for PPO training and evaluation."""

    total_timesteps: int = 256
    learning_rate: float = 3e-4
    n_steps: int = 64
    batch_size: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    seed: int = 42
    alpha: float = 1.0
    beta: float = 1.0
    terminal_bonus: float = 0.0
    dead_end_penalty: float = 20.0
    illegal_action_penalty: float = 50.0
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "ppo_macro_placement.zip"
    verbose: int = 0
    reward_scale: float = 1.0
    features_dim: int = 128

    def validate(self) -> None:
        """Validate configuration settings and hyperparameter ranges."""
        if (
            isinstance(self.total_timesteps, bool)
            or not isinstance(self.total_timesteps, (int, np.integer))
            or self.total_timesteps <= 0
        ):
            raise ValueError(f"total_timesteps must be a positive integer, got {self.total_timesteps!r}.")

        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float, np.floating))
            or self.learning_rate <= 0
        ):
            raise ValueError(f"learning_rate must be a positive number, got {self.learning_rate!r}.")

        if (
            isinstance(self.n_steps, bool)
            or not isinstance(self.n_steps, (int, np.integer))
            or self.n_steps <= 0
        ):
            raise ValueError(f"n_steps must be a positive integer, got {self.n_steps!r}.")

        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, (int, np.integer))
            or self.batch_size <= 0
        ):
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size!r}.")

        if self.batch_size > self.n_steps:
            raise ValueError(
                f"batch_size ({self.batch_size}) cannot exceed n_steps ({self.n_steps})."
            )

        if (
            isinstance(self.gamma, bool)
            or not isinstance(self.gamma, (int, float, np.floating))
            or not (0.0 <= self.gamma <= 1.0)
        ):
            raise ValueError(f"gamma must be between 0.0 and 1.0, got {self.gamma!r}.")

        if (
            isinstance(self.gae_lambda, bool)
            or not isinstance(self.gae_lambda, (int, float, np.floating))
            or not (0.0 <= self.gae_lambda <= 1.0)
        ):
            raise ValueError(f"gae_lambda must be between 0.0 and 1.0, got {self.gae_lambda!r}.")

        if (
            isinstance(self.clip_range, bool)
            or not isinstance(self.clip_range, (int, float, np.floating))
            or self.clip_range <= 0.0
        ):
            raise ValueError(f"clip_range must be positive, got {self.clip_range!r}.")

        if (
            isinstance(self.ent_coef, bool)
            or not isinstance(self.ent_coef, (int, float, np.floating))
            or self.ent_coef < 0.0
        ):
            raise ValueError(f"ent_coef must be non-negative, got {self.ent_coef!r}.")

        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float, np.floating))
            or self.alpha < 0.0
            or isinstance(self.beta, bool)
            or not isinstance(self.beta, (int, float, np.floating))
            or self.beta < 0.0
        ):
            raise ValueError(
                f"alpha and beta must be non-negative numbers, got alpha={self.alpha!r}, beta={self.beta!r}."
            )

        if not self.checkpoint_dir or not isinstance(self.checkpoint_dir, str):
            raise ValueError(f"checkpoint_dir must be a non-empty string, got {self.checkpoint_dir!r}.")

        if not self.checkpoint_name or not isinstance(self.checkpoint_name, str):
            raise ValueError(f"checkpoint_name must be a non-empty string, got {self.checkpoint_name!r}.")

        if (
            isinstance(self.reward_scale, bool)
            or not isinstance(self.reward_scale, (int, float, np.floating))
            or self.reward_scale <= 0.0
        ):
            raise ValueError(f"reward_scale must be a positive number, got {self.reward_scale!r}.")

        if (
            isinstance(self.features_dim, bool)
            or not isinstance(self.features_dim, (int, np.integer))
            or self.features_dim <= 0
        ):
            raise ValueError(f"features_dim must be a positive integer, got {self.features_dim!r}.")


@dataclass(frozen=True)
class PPOEvalResult:
    """Evaluation metrics for a trained PPO model or RL inference run."""

    algorithm: str
    feasible: bool
    objective: float | None
    hpwl: float | None
    congestion: float | None
    runtime_sec: float
    placement: Placement
    steps: int
    total_macros: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Convert evaluation result to JSON-serializable dictionary."""
        return {
            "algorithm": self.algorithm,
            "feasible": self.feasible,
            "objective": self.objective,
            "hpwl": self.hpwl,
            "congestion": self.congestion,
            "runtime_sec": self.runtime_sec,
            "steps": self.steps,
            "total_macros": self.total_macros,
            "seed": self.seed,
            "placement": [
                {"macro_id": pm.macro_id, "x": pm.x, "y": pm.y}
                for pm in self.placement
            ],
        }


@dataclass(frozen=True)
class PPOTrainingResult:
    """Summary of a completed PPO training run."""

    algorithm: str
    netlist_id: str
    seed: int
    total_timesteps: int
    training_time_s: float
    checkpoint_path: str
    eval_result: PPOEvalResult

    def to_dict(self) -> dict[str, Any]:
        """Convert training result to JSON-serializable dictionary."""
        return {
            "algorithm": self.algorithm,
            "netlist_id": self.netlist_id,
            "seed": self.seed,
            "total_timesteps": self.total_timesteps,
            "training_time_s": self.training_time_s,
            "checkpoint_path": self.checkpoint_path,
            "eval_result": self.eval_result.to_dict(),
        }


def train_ppo(
    netlist: Any,
    config: PPOTrainingConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> PPOTrainingResult:
    """Train MaskablePPO agent on the placement environment and save checkpoint.

    Parameters:
        netlist: Circuit netlist (Netlist object, dict, or sequence).
        config: Training configuration with hyperparameters and settings.
        checkpoint_path: Optional explicit output path for model checkpoint.

    Returns:
        PPOTrainingResult containing training duration, checkpoint path, and evaluation result.
    """
    if not _HAS_GYMNASIUM:
        raise ImportError(
            "Gymnasium is required for PPO training. "
            "Please install gymnasium>=1.4.0."
        )
    if not _HAS_SB3:
        raise ImportError(
            "sb3-contrib is required for MaskablePPO training. "
            "Please install sb3-contrib>=2.9.2."
        )

    cfg = config if config is not None else PPOTrainingConfig()
    cfg.validate()

    # Deterministic RNG seeding across python, numpy, torch
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    canonical_nl = to_canonical_netlist(netlist)
    train_env = MacroPlacementEnv(
        netlist=canonical_nl,
        alpha=cfg.alpha,
        beta=cfg.beta,
        terminal_bonus=cfg.terminal_bonus,
        dead_end_penalty=cfg.dead_end_penalty,
        illegal_action_penalty=cfg.illegal_action_penalty,
        reward_scale=cfg.reward_scale,
    )
    wrapped_env = ActionMasker(train_env, mask_fn)

    policy_kwargs = None
    if MacroPlacementFeaturesExtractor is not None:
        policy_kwargs = {
            "features_extractor_class": MacroPlacementFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": cfg.features_dim},
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        }

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=wrapped_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        seed=seed,
        verbose=cfg.verbose,
        policy_kwargs=policy_kwargs,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=cfg.total_timesteps)
    train_time = time.perf_counter() - t0

    # Resolve checkpoint destination
    if checkpoint_path is not None:
        save_p = Path(checkpoint_path)
    else:
        save_p = Path(cfg.checkpoint_dir) / cfg.checkpoint_name

    save_p.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_p))

    # Evaluate trained model
    eval_res = evaluate_ppo(
        canonical_nl,
        model_or_path=model,
        seed=seed,
        alpha=cfg.alpha,
        beta=cfg.beta,
    )

    netlist_id = getattr(canonical_nl, "version", "custom")
    return PPOTrainingResult(
        algorithm="rl",
        netlist_id=netlist_id,
        seed=seed,
        total_timesteps=cfg.total_timesteps,
        training_time_s=train_time,
        checkpoint_path=str(save_p),
        eval_result=eval_res,
    )


def evaluate_ppo(
    netlist: Any,
    model_or_path: Any | str | Path | None = None,
    seed: int = 42,
    alpha: float = 1.0,
    beta: float = 1.0,
    deterministic: bool = True,
) -> PPOEvalResult:
    """Deterministically evaluate an RL policy or loaded MaskablePPO model on a netlist.

    Parameters:
        netlist: Circuit netlist.
        model_or_path: MaskablePPO model instance, checkpoint file path, or None for masked policy fallback.
        seed: Random seed for environment reset.
        alpha: HPWL weight.
        beta: Congestion weight.
        deterministic: Whether to use deterministic action selection.

    Returns:
        PPOEvalResult containing placement, feasibility, HPWL, congestion, and objective.
    """
    if not _HAS_GYMNASIUM:
        raise ImportError(
            "Gymnasium is required for PPO evaluation. "
            "Please install gymnasium>=1.4.0."
        )

    canonical_nl = to_canonical_netlist(netlist)

    model = None
    if model_or_path is not None:
        if isinstance(model_or_path, (str, Path)):
            if not _HAS_SB3:
                raise ImportError("sb3-contrib is required to load a MaskablePPO checkpoint.")
            m_str = str(model_or_path)
            if not (os.path.exists(m_str) or os.path.exists(f"{m_str}.zip")):
                raise FileNotFoundError(f"Checkpoint not found at '{m_str}'.")
            model = MaskablePPO.load(m_str)
        else:
            model = model_or_path

    env = MacroPlacementEnv(netlist=canonical_nl, alpha=alpha, beta=beta)
    obs, _info = env.reset(seed=seed)
    total_macros = len(env.macros)
    step = 0
    done = False
    start_t = time.perf_counter()

    while not done:
        masks = env.action_masks()
        if not np.any(masks):
            break

        if model is not None:
            action, _states = model.predict(obs, action_masks=masks, deterministic=deterministic)
            action_int = int(action)
        else:
            # Deterministic fallback: choose first legal position
            action_int = int(np.where(masks)[0][0])

        obs, _reward, terminated, truncated, _step_info = env.step(action_int)
        step += 1
        done = terminated or truncated

    runtime = time.perf_counter() - start_t
    placement = env.engine.get_placement()
    is_complete = env.engine.is_complete()
    val_report = validate_placement(canonical_nl, placement)
    feasible = is_complete and val_report.is_valid

    if feasible:
        eval_metrics = evaluate_placement(canonical_nl, placement, alpha=alpha, beta=beta)
        hpwl = eval_metrics.hpwl
        congestion = eval_metrics.congestion
        objective = eval_metrics.objective
    else:
        hpwl = None
        congestion = None
        objective = None

    return PPOEvalResult(
        algorithm="rl",
        feasible=feasible,
        objective=objective,
        hpwl=hpwl,
        congestion=congestion,
        runtime_sec=runtime,
        placement=placement,
        steps=step,
        total_macros=total_macros,
        seed=seed,
    )


def compare_baseline(
    netlist: Any,
    seed: int = 42,
    alpha: float = 1.0,
    beta: float = 1.0,
    rl_model_or_path: Any | None = None,
    sa_iterations: int = 100,
) -> dict[str, dict[str, Any]]:
    """Compare Random, Greedy, Simulated Annealing, and RL on a common netlist.

    Parameters:
        netlist: Canonical or dict/raw circuit netlist.
        seed: Random seed for reproducible comparison.
        alpha: Weight for HPWL in combined objective J(P) = alpha * HPWL + beta * C_score.
        beta: Weight for Congestion score in combined objective.
        rl_model_or_path: Optional trained MaskablePPO model or file path.
        sa_iterations: Maximum iterations for Simulated Annealing.

    Returns:
        Dictionary mapping algorithm name to its performance record:
        {
            'random': {'feasible': bool, 'hpwl': float, 'congestion': float, 'objective': float, 'runtime_sec': float},
            'greedy': {'feasible': bool, 'hpwl': float, 'congestion': float, 'objective': float, 'runtime_sec': float},
            'sa':     {'feasible': bool, 'hpwl': float, 'congestion': float, 'objective': float, 'runtime_sec': float},
            'rl':     {'feasible': bool, 'hpwl': float, 'congestion': float, 'objective': float, 'runtime_sec': float, 'error': str | None},
        }
    """
    canonical_nl = to_canonical_netlist(netlist)
    results: dict[str, dict[str, Any]] = {}

    # 1. Random Placement
    from src.algorithms.common import random_legal_placement

    t0 = time.perf_counter()
    rand_placement = random_legal_placement(canonical_nl, seed=seed)
    rand_time = time.perf_counter() - t0
    if rand_placement is not None:
        rand_eval = evaluate_placement(canonical_nl, rand_placement, alpha=alpha, beta=beta)
        results["random"] = {
            "feasible": True,
            "hpwl": rand_eval.hpwl,
            "congestion": rand_eval.congestion,
            "objective": rand_eval.objective,
            "runtime_sec": rand_time,
            "error": None,
        }
    else:
        results["random"] = {
            "feasible": False,
            "hpwl": None,
            "congestion": None,
            "objective": None,
            "runtime_sec": rand_time,
            "error": "Dead end reached during random placement",
        }

    # 2. Greedy Placement
    from src.algorithms.greedy import greedy_placement

    t0 = time.perf_counter()
    g_res = greedy_placement(canonical_nl, alpha=alpha, beta=beta)
    g_time = time.perf_counter() - t0
    results["greedy"] = {
        "feasible": g_res.success,
        "hpwl": g_res.hpwl if g_res.success else None,
        "congestion": g_res.congestion if g_res.success else None,
        "objective": g_res.objective if g_res.success else None,
        "runtime_sec": g_time,
        "error": None if g_res.success else "Greedy failed to place all macros",
    }

    # 3. Simulated Annealing
    from src.algorithms.simulated_annealing import simulated_annealing

    t0 = time.perf_counter()
    sa_res = simulated_annealing(
        canonical_nl,
        alpha=alpha,
        beta=beta,
        seed=seed,
        max_iterations=sa_iterations,
    )
    sa_time = time.perf_counter() - t0
    results["sa"] = {
        "feasible": sa_res.success,
        "hpwl": sa_res.hpwl if sa_res.success else None,
        "congestion": sa_res.congestion if sa_res.success else None,
        "objective": sa_res.objective if sa_res.success else None,
        "runtime_sec": sa_time,
        "error": None if sa_res.success else "SA failed to find legal placement",
    }

    # 4. Reinforcement Learning
    if not _HAS_GYMNASIUM:
        results["rl"] = {
            "feasible": False,
            "hpwl": None,
            "congestion": None,
            "objective": None,
            "runtime_sec": 0.0,
            "error": "Gymnasium is required for RL evaluation. Install gymnasium>=1.4.0.",
        }
    else:
        try:
            rl_eval = evaluate_ppo(
                canonical_nl,
                model_or_path=rl_model_or_path,
                seed=seed,
                alpha=alpha,
                beta=beta,
            )
            results["rl"] = {
                "feasible": rl_eval.feasible,
                "hpwl": rl_eval.hpwl,
                "congestion": rl_eval.congestion,
                "objective": rl_eval.objective,
                "runtime_sec": rl_eval.runtime_sec,
                "error": None if rl_eval.feasible else "RL policy reached dead end",
            }
        except Exception as exc:  # noqa: BLE001
            results["rl"] = {
                "feasible": False,
                "hpwl": None,
                "congestion": None,
                "objective": None,
                "runtime_sec": 0.0,
                "error": str(exc),
            }

    return results


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
        - objective: Combined objective value
        - evals: Total inference evaluations executed
        - trace: List of progress snapshots
    """
    start_time = time.perf_counter()
    cfg = env_config or {}

    alpha = float(cfg.get("alpha", 1.0))
    beta = float(cfg.get("beta", 1.0))
    model_path = str(cfg.get("model_path", "ppo_macro_placement.zip"))

    env = MacroPlacementEnv(netlist=netlist, alpha=alpha, beta=beta)

    model = None
    if _HAS_SB3:
        if os.path.exists(model_path) or os.path.exists(model_path + ".zip"):
            model = MaskablePPO.load(model_path)

    obs, _info = env.reset(seed=seed)
    done = False
    evals = 0
    trace: list[dict[str, Any]] = []

    while not done:
        masks = env.action_masks()
        if not np.any(masks):
            break

        if model is not None:
            action, _states = model.predict(obs, action_masks=masks, deterministic=True)
            action_int = int(action)
        else:
            # Deterministic fallback to first legal action
            action_int = int(np.where(masks)[0][0])

        obs, reward, terminated, truncated, step_info = env.step(action_int)
        evals += 1
        done = terminated or truncated

        elapsed = time.perf_counter() - start_time
        hpwl_snapshot = step_info.get("hpwl", 0.0)
        obj_snapshot = step_info.get("objective", 0.0)
        trace.append({
            "eval": evals,
            "t_s": elapsed,
            "best_cost": obj_snapshot,
            "hpwl": hpwl_snapshot,
        })

        if on_progress is not None:
            progress_payload = {
                "step": evals,
                "total_steps": len(env.macros),
                "hpwl": hpwl_snapshot,
                "objective": obj_snapshot,
                "placement": [
                    {"macro_id": pm.macro_id, "x": pm.x, "y": pm.y}
                    for pm in env.engine.get_placement()
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

    placement = [
        {"macro_id": pm.macro_id, "x": pm.x, "y": pm.y}
        for pm in env.engine.get_placement()
    ]
    feasible = env.engine.is_complete()
    runtime_s = time.perf_counter() - start_time
    eval_res = evaluate_placement(env.netlist, env.engine.get_placement(), alpha=alpha, beta=beta)

    return {
        "placement": placement,
        "feasible": feasible,
        "runtime_s": runtime_s,
        "cost": eval_res.objective if feasible else float("inf"),
        "hpwl": eval_res.hpwl if feasible else None,
        "congestion": eval_res.congestion if feasible else None,
        "objective": eval_res.objective if feasible else None,
        "evals": evals,
        "trace": trace,
    }
