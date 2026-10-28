"""Unit tests for Random Placement baseline algorithm.

Tests cover:
1. Successful placement returns the placement, success=True, and all three metrics.
2. Fixed seed produces reproducible output.
3. Helper result of None produces documented failure result without attempting objective evaluation.
4. Objective evaluator receives the correct netlist, placement, alpha, and beta.
5. Implementation propagates unexpected helper and evaluator errors.
6. Implementation does not mutate its inputs.
7. Integration with actual shared helper (when available) or marked pending/skipped.
8. Incomplete placement returned by helper produces success=False.
9. Invalid seed, alpha, and beta validations.
10. NotImplementedError when shared helper is absent and no generator is injected.
"""

from __future__ import annotations

import copy
from typing import Any
import numpy as np
import pytest

from src.placement.models import Macro, Net, Netlist, PlacedMacro, Placement
from src.algorithms.random_placement import (
    PlacementResult,
    RandomPlacementResult,
    random_placement,
)
from src.placement.objective import evaluate_placement


# =============================================================================
# Helper Fixtures and Test Doubles
# =============================================================================


@pytest.fixture
def sample_netlist() -> dict[str, Any]:
    """Return a standard test netlist."""
    return {
        "grid_w": 5,
        "grid_h": 5,
        "macros": {1: (1, 1), 2: (1, 1)},
        "nets": [{"pins": [1, 2], "weight": 1.0}],
    }


def deterministic_fake_legal_placement(netlist: Any, seed: int | None = 42) -> dict[int, tuple[int, int]]:
    """Deterministic fake legal placement generator for testing."""
    if seed == 999:
        # Placement with seed 999 places macros at distinct positions
        return {1: (1, 1), 2: (3, 3)}
    # Default placement
    return {1: (0, 0), 2: (2, 2)}


# =============================================================================
# 1. Successful Placement Returns Placement and All Three Metrics
# =============================================================================


def test_successful_placement_returns_placement_and_metrics(sample_netlist):
    """A successful placement returns the placement, success=True, and all three metrics."""
    result = random_placement(
        sample_netlist,
        seed=42,
        alpha=1.5,
        beta=2.5,
        legal_placement_fn=deterministic_fake_legal_placement,
    )

    assert isinstance(result, RandomPlacementResult)
    assert isinstance(result, PlacementResult)
    assert result.success is True
    assert result.placement == {1: (0, 0), 2: (2, 2)}

    # Verify metrics match evaluate_placement ground truth
    expected_eval = evaluate_placement(sample_netlist, result.placement, alpha=1.5, beta=2.5)
    assert np.isclose(result.hpwl, expected_eval.hpwl)
    assert np.isclose(result.congestion, expected_eval.congestion)
    assert np.isclose(result.objective, expected_eval.objective)
    assert np.isclose(result.objective, 1.5 * expected_eval.hpwl + 2.5 * expected_eval.congestion)


# =============================================================================
# 2. Fixed Seed Produces Reproducible Output
# =============================================================================


def test_fixed_seed_reproducible_output(sample_netlist):
    """A fixed seed produces reproducible output across repeated calls."""
    res1 = random_placement(
        sample_netlist,
        seed=42,
        alpha=1.0,
        beta=1.0,
        legal_placement_fn=deterministic_fake_legal_placement,
    )
    res2 = random_placement(
        sample_netlist,
        seed=42,
        alpha=1.0,
        beta=1.0,
        legal_placement_fn=deterministic_fake_legal_placement,
    )

    assert res1.placement == res2.placement
    assert res1.hpwl == res2.hpwl
    assert res1.congestion == res2.congestion
    assert res1.objective == res2.objective


def test_different_seeds_produce_different_placements(sample_netlist):
    """Different seeds pass the seed to the generator and allow different placements."""
    res_default = random_placement(
        sample_netlist,
        seed=42,
        legal_placement_fn=deterministic_fake_legal_placement,
    )
    res_other = random_placement(
        sample_netlist,
        seed=999,
        legal_placement_fn=deterministic_fake_legal_placement,
    )

    assert res_default.placement != res_other.placement
    assert res_default.placement == {1: (0, 0), 2: (2, 2)}
    assert res_other.placement == {1: (1, 1), 2: (3, 3)}


# =============================================================================
# 3. Helper Returning None Produces Documented Failure Result
# =============================================================================


def test_helper_returning_none_produces_failure_result(sample_netlist, monkeypatch):
    """When the legal placement helper returns None, return documented failure result."""
    eval_called = False

    def failing_helper(netlist: Any, seed: int | None = 42) -> None:
        return None

    def spy_evaluate(*args, **kwargs):
        nonlocal eval_called
        eval_called = True
        return evaluate_placement(*args, **kwargs)

    monkeypatch.setattr("src.algorithms.random_placement.evaluate_placement", spy_evaluate)

    result = random_placement(
        sample_netlist,
        seed=42,
        alpha=1.0,
        beta=1.0,
        legal_placement_fn=failing_helper,
    )

    assert isinstance(result, RandomPlacementResult)
    assert result.success is False
    assert result.placement is None
    assert result.hpwl is None
    assert result.congestion is None
    assert result.objective is None

    # Objective evaluation must not be attempted
    assert eval_called is False


# =============================================================================
# 4. Objective Evaluator Receives Correct Netlist, Placement, Alpha, Beta
# =============================================================================


def test_evaluator_receives_correct_parameters(sample_netlist, monkeypatch):
    """The objective evaluator receives the exact netlist, placement, alpha, and beta."""
    recorded_args: dict[str, Any] = {}

    def spy_evaluate(netlist, placement, alpha, beta):
        recorded_args["netlist"] = netlist
        recorded_args["placement"] = placement
        recorded_args["alpha"] = alpha
        recorded_args["beta"] = beta
        return evaluate_placement(netlist, placement, alpha, beta)

    monkeypatch.setattr("src.algorithms.random_placement.evaluate_placement", spy_evaluate)

    test_alpha = 3.25
    test_beta = 1.75
    random_placement(
        sample_netlist,
        seed=42,
        alpha=test_alpha,
        beta=test_beta,
        legal_placement_fn=deterministic_fake_legal_placement,
    )

    assert recorded_args["netlist"] is sample_netlist
    assert recorded_args["placement"] == {1: (0, 0), 2: (2, 2)}
    assert recorded_args["alpha"] == test_alpha
    assert recorded_args["beta"] == test_beta


# =============================================================================
# 5. Propagation of Unexpected Helper and Evaluator Errors
# =============================================================================


def test_helper_exception_propagates(sample_netlist):
    """Unexpected exceptions raised by the legal placement helper propagate cleanly."""
    def broken_helper(netlist: Any, seed: int | None = 42) -> Any:
        raise RuntimeError("Low-level grid error during placement generation.")

    with pytest.raises(RuntimeError, match="Low-level grid error"):
        random_placement(
            sample_netlist,
            legal_placement_fn=broken_helper,
        )


def test_evaluator_exception_propagates(sample_netlist, monkeypatch):
    """Unexpected evaluator exceptions propagate rather than being swallowed."""
    def broken_evaluator(netlist, placement, alpha, beta):
        raise ValueError("Malformed netlist detected during evaluation.")

    monkeypatch.setattr("src.algorithms.random_placement.evaluate_placement", broken_evaluator)

    with pytest.raises(ValueError, match="Malformed netlist detected"):
        random_placement(
            sample_netlist,
            legal_placement_fn=deterministic_fake_legal_placement,
        )


# =============================================================================
# 6. No Mutation of Inputs
# =============================================================================


def test_no_mutation_of_netlist(sample_netlist):
    """The random_placement function does not mutate caller-owned netlist data."""
    netlist_clone = copy.deepcopy(sample_netlist)

    random_placement(
        sample_netlist,
        seed=42,
        alpha=2.0,
        beta=3.0,
        legal_placement_fn=deterministic_fake_legal_placement,
    )

    assert sample_netlist == netlist_clone


# =============================================================================
# 7. True Integration Test (Skipped / Marked Pending if Helper Absent)
# =============================================================================


def test_true_integration_with_shared_helper(sample_netlist):
    """Integration test with shared engine random placement."""
    result = random_placement(sample_netlist, seed=42)
    assert isinstance(result, RandomPlacementResult)
    assert result.success is True
    assert result.placement is not None
    assert result.hpwl is not None
    assert result.congestion is not None
    assert result.objective is not None

    # Verify deterministic reproducibility with identical seed
    result2 = random_placement(sample_netlist, seed=42)
    assert result.placement == result2.placement
    assert np.isclose(result.objective, result2.objective)


# =============================================================================
# 8. Incomplete Placement Handling
# =============================================================================


def test_incomplete_placement_treated_as_failure(sample_netlist):
    """An incomplete placement missing netlist macros must return success=False."""
    # Netlist has macros {1: ..., 2: ...}, but helper only places macro 1
    def incomplete_helper(netlist: Any, seed: int | None = 42) -> dict[int, tuple[int, int]]:
        return {1: (0, 0)}

    result = random_placement(
        sample_netlist,
        legal_placement_fn=incomplete_helper,
    )

    assert result.success is False
    assert result.placement == {1: (0, 0)}
    assert result.hpwl is None
    assert result.congestion is None
    assert result.objective is None


# =============================================================================
# 9. Parameter Validations (Seed, Alpha, Beta)
# =============================================================================


@pytest.mark.parametrize("invalid_seed", ["seed_str", 3.14, True, [42]])
def test_invalid_seed_raises(sample_netlist, invalid_seed):
    """Non-integer, non-None seeds raise ValueError."""
    with pytest.raises(ValueError, match="Seed must be an integer"):
        random_placement(
            sample_netlist,
            seed=invalid_seed,
            legal_placement_fn=deterministic_fake_legal_placement,
        )


@pytest.mark.parametrize(
    "a, b",
    [
        (-1.0, 1.0),
        (1.0, -0.5),
        (float("nan"), 1.0),
        (1.0, float("nan")),
        (float("inf"), 1.0),
        (1.0, float("-inf")),
    ],
)
def test_invalid_alpha_beta_raises(sample_netlist, a, b):
    """Negative, NaN, or infinite alpha/beta raise ValueError."""
    with pytest.raises(ValueError):
        random_placement(
            sample_netlist,
            alpha=a,
            beta=b,
            legal_placement_fn=deterministic_fake_legal_placement,
        )


# =============================================================================
# 10. Missing Shared Helper Raises NotImplementedError
# =============================================================================


def test_missing_shared_helper_raises_not_implemented(sample_netlist, monkeypatch):
    """When no shared helper exists and no generator is injected, raise NotImplementedError."""
    monkeypatch.setattr("src.algorithms.random_placement.random_legal_placement", None)

    with pytest.raises(NotImplementedError, match="Shared engine module"):
        random_placement(sample_netlist, seed=42)


# =============================================================================
# 11. Shared Engine Canonical Integration & Dead End Detection
# =============================================================================


def test_random_placement_with_canonical_netlist():
    """Random placement works directly with canonical Netlist instances."""
    macros = (Macro(id=1, w=2, h=2), Macro(id=2, w=1, h=1))
    nets = (Net(id=10, macro_ids=(1, 2), weight=1.5),)
    netlist = Netlist(version="1.0", grid_w=6, grid_h=6, macros=macros, nets=nets)

    result = random_placement(netlist, seed=123)
    assert result.success is True
    assert result.placement is not None
    assert len(result.placement) == 2
    assert result.hpwl > 0.0
    assert result.congestion >= 0.0
    assert result.objective > 0.0


def test_random_placement_dead_end_cannot_fit():
    """When macros cannot legally fit within the grid, random placement detects dead end."""
    # 2x2 grid with two 2x2 macros: second macro cannot be placed
    macros = (Macro(id=1, w=2, h=2), Macro(id=2, w=2, h=2))
    netlist = Netlist(version="1.0", grid_w=2, grid_h=2, macros=macros, nets=())

    result = random_placement(netlist, seed=42)
    assert result.success is False
    assert result.placement is None
    assert result.hpwl is None
    assert result.congestion is None
    assert result.objective is None


def test_random_placement_placement_list_compatibility():
    """Custom helper returning Placement (list[PlacedMacro]) is fully supported."""
    sample = {
        "grid_w": 5,
        "grid_h": 5,
        "macros": {1: (1, 1), 2: (1, 1)},
        "nets": [{"pins": [1, 2], "weight": 1.0}],
    }

    def list_helper(nl, seed=42):
        return [PlacedMacro(macro_id=1, x=0, y=0), PlacedMacro(macro_id=2, x=2, y=2)]

    result = random_placement(sample, legal_placement_fn=list_helper)
    assert result.success is True
    assert isinstance(result.placement, list)
    assert len(result.placement) == 2
    assert result.hpwl == 4.0
