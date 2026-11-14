"""Placement algorithms package."""

from typing import Any

__all__ = [
    "MacroPlacementEnv",
    "MacroPlacementFeaturesExtractor",
    "PPOTrainingConfig",
    "PPOTrainingResult",
    "PPOEvalResult",
    "compare_baseline",
    "compute_connectivity_features",
    "evaluate_ppo",
    "rl_solve",
    "train_ppo",
    "train_rl_agent",
]


def __getattr__(name: str) -> Any:
    """Lazy import RL components to avoid requiring Gymnasium for classical algorithms."""
    if name in (
        "MacroPlacementEnv",
        "MacroPlacementFeaturesExtractor",
        "PPOTrainingConfig",
        "PPOTrainingResult",
        "PPOEvalResult",
        "compare_baseline",
        "compute_connectivity_features",
        "evaluate_ppo",
        "train_ppo",
        "train_rl_agent",
    ):
        from . import rl_placement

        return getattr(rl_placement, name)
    if name == "rl_solve":
        from .rl_placement import solve as rl_solve

        return rl_solve
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)
