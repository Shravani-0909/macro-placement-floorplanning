"""RL Placement runner for integration with placement jobs and benchmark suites."""

import asyncio
import random
from pathlib import Path
from typing import Any

import numpy as np

from src.placement.constraints import PlacementError, validate_placement
from src.placement.jobs import (
    JobProgress,
    ProgressCallback,
    register_algorithm,
)
from src.placement.models import Netlist, Placement

from .env import MacroPlacementEnv


async def rl_placement_runner(
    netlist: Netlist,
    seed: int,
    params: dict[str, Any],
    progress_cb: ProgressCallback,
) -> Placement:
    """Execute reinforcement learning policy inference runner.

    Parameters:
        netlist: Target netlist.
        seed: Random seed for deterministic inference.
        params: Configuration dictionary (can include 'policy', 'alpha', 'beta', 'order').
        progress_cb: Async callback for streaming placement progress snapshots.

    Returns:
        Final Placement sequence.
    """
    alpha = float(params.get("alpha", 1.0))
    beta = float(params.get("beta", 1.0))
    order = str(params.get("order", "netlist"))
    policy = params.get("policy", None)

    # Resolve saved model checkpoint if policy object is not directly provided
    checkpoint_path = params.get("model_path") or params.get("checkpoint_path")
    if policy is None and checkpoint_path is not None:
        cp_file = Path(checkpoint_path)
        if not cp_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            from sb3_contrib import MaskablePPO  # type: ignore[import-untyped, import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Loading saved MaskablePPO checkpoints requires 'sb3-contrib' and 'torch'. "
                "Install them with `pip install sb3-contrib torch` or provide an in-memory policy object in params['policy']."
            ) from exc

        policy = MaskablePPO.load(str(cp_file))

    env = MacroPlacementEnv(
        netlist=netlist,
        alpha=alpha,
        beta=beta,
        order=order,
    )

    obs, info = env.reset(seed=seed)
    rng = random.Random(seed)
    total_steps = len(netlist.macros)

    # Initial progress snapshot
    await progress_cb(
        JobProgress(
            run_id="",
            status="running",
            progress=0.0,
            step=0,
            total_steps=total_steps,
            current_hpwl=0.0,
            placement=[],
        )
    )

    terminated = False
    truncated = False

    while not (terminated or truncated):
        mask = env.action_masks()
        legal_actions = np.flatnonzero(mask)

        if len(legal_actions) == 0:
            raise PlacementError(
                f"Deadlock in RL placement: no legal actions available for macro {env.current_macro}."
            )

        if policy is not None and hasattr(policy, "predict"):
            # SB3 / MaskablePPO policy interface
            pred = policy.predict(obs, action_masks=mask, deterministic=True)
            action = int(pred[0] if isinstance(pred, tuple) else pred)
        elif callable(policy):
            action = int(policy(obs, mask))
        else:
            # Deterministic masked exploration fallback (for testing/unloaded policy)
            action = int(rng.choice(legal_actions))

        obs, _reward, terminated, truncated, info = env.step(action)
        step = int(info["step"])
        curr_hpwl = float(info["hpwl"])
        current_placement = env.get_placement()

        await progress_cb(
            JobProgress(
                run_id="",
                status="running",
                progress=float(step / total_steps),
                step=step,
                total_steps=total_steps,
                current_hpwl=curr_hpwl,
                placement=current_placement,
            )
        )
        await asyncio.sleep(0.001)

    final_placement = env.get_placement()
    validate_placement(netlist, final_placement)
    return final_placement


# Automatically register RL runner into ALGORITHM_REGISTRY
register_algorithm("rl", rl_placement_runner)
register_algorithm("maskable_ppo", rl_placement_runner)
