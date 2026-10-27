"""Unit tests for Greedy Placement baseline algorithm.

Tests cover:
1. Deterministic macro ordering and stable tie-breaking (largest area first, ID tie-break).
2. Selection of the legal position with minimum incremental HPWL (and deterministic anchor tie-break).
3. Correct conversion from mask coordinates mask[y, x] to env.place(macro_id, x, y).
4. Failure handling when a macro has no legal positions.
5. Successful placement and objective evaluation.
6. Correct forwarding of alpha and beta parameters.
7. Propagation of unexpected environment and evaluator exceptions.
8. No mutation of the input netlist.
9. Integration with the actual shared engine (skipped/pending when absent).
10. Incomplete placement verification.
11. Parameter validations (invalid alpha/beta).
12. NotImplementedError when shared GridEnv is missing and none is injected.
"""

from __future__ import annotations

import copy
from typing import Any
import numpy as np
import pytest

from src.algorithms.greedy import (
    GreedyPlacementResult,
    PlacementResult,
    _get_sorted_macro_ids,
    greedy,
    greedy_placement,
)
from src.placement.objective import evaluate_placement


# =============================================================================
# Test Doubles / Mock GridEnv
# =============================================================================


class MockGridEnv:
    """Mock GridEnv modeling Zoya's shared grid environment contract."""

    def __init__(self, netlist: Any):
        self.netlist = netlist
        if isinstance(netlist, dict):
            self.grid_w = netlist.get("grid_w", 5)
            self.grid_h = netlist.get("grid_h", 5)
        else:
            self.grid_w = getattr(netlist, "grid_w", 5)
            self.grid_h = getattr(netlist, "grid_h", 5)

        self.placed: dict[Any, tuple[int, int]] = {}
        self.history: list[tuple[str, Any]] = []
        self.custom_masks: dict[Any, np.ndarray] = {}
        self.custom_delta_fn: Any = None

    def legal_mask(self, macro_id: Any) -> np.ndarray:
        self.history.append(("legal_mask", macro_id))
        if macro_id in self.custom_masks:
            return self.custom_masks[macro_id]
        # Default: all coordinates legal except already occupied anchors
        mask = np.ones((self.grid_h, self.grid_w), dtype=bool)
        for _, (px, py) in self.placed.items():
            if 0 <= py < self.grid_h and 0 <= px < self.grid_w:
                mask[py, px] = False
        return mask

    def hpwl_delta_move(self, macro_id: Any, x: int, y: int) -> float:
        self.history.append(("hpwl_delta_move", (macro_id, x, y)))
        if self.custom_delta_fn is not None:
            return float(self.custom_delta_fn(macro_id, x, y))
        return float(x + 2 * y)

    def place(self, macro_id: Any, x: int, y: int) -> None:
        self.history.append(("place", (macro_id, x, y)))
        self.placed[macro_id] = (x, y)

    def placement(self) -> dict[Any, tuple[int, int]]:
        return dict(self.placed)


@pytest.fixture
def sample_netlist() -> dict[str, Any]:
    """Return a standard test netlist."""
    return {
        "grid_w": 5,
        "grid_h": 5,
        "macros": {1: (1, 1), 2: (1, 1)},
        "nets": [{"pins": [1, 2], "weight": 1.0}],
    }


# =============================================================================
# 1. Deterministic Macro Ordering and Stable Tie-Breaking
# =============================================================================


def test_deterministic_macro_ordering_largest_area_first():
    """Macros are placed in descending order of area, with stable tie-break by ID."""
    macros = {
        "small": (1, 1),       # area 1
        "large": (4, 3),       # area 12
        "medium_b": (3, 2),    # area 6, ID 'medium_b'
        "medium_a": (2, 3),    # area 6, ID 'medium_a'
    }

    sorted_ids = _get_sorted_macro_ids(macros)
    expected_order = ["large", "medium_a", "medium_b", "small"]
    assert sorted_ids == expected_order


def test_greedy_processes_macros_in_sorted_order():
    """Verify greedy_placement queries legal_mask in largest-area-first order."""
    netlist = {
        "grid_w": 6,
        "grid_h": 6,
        "macros": {
            "M1": (1, 2),  # area 2
            "M2": (3, 3),  # area 9
            "M3": (2, 2),  # area 4
        },
        "nets": [],
    }

    env_instance: MockGridEnv | None = None

    def env_factory(nl):
        nonlocal env_instance
        env_instance = MockGridEnv(nl)
        return env_instance

    res = greedy_placement(netlist, env_class=env_factory)
    assert res.success is True

    # Check order of legal_mask calls in history
    mask_calls = [m_id for op, m_id in env_instance.history if op == "legal_mask"]
    assert mask_calls == ["M2", "M3", "M1"]  # Area 9, then 4, then 2


# =============================================================================
# 2. Selection of Legal Position with Minimum Incremental HPWL
# =============================================================================


def test_selects_position_with_minimum_hpwl_delta():
    """Greedy chooses the anchor with the minimum incremental HPWL delta."""
    netlist = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {"A": (1, 1)},
        "nets": [],
    }

    env_instance: MockGridEnv | None = None

    def env_factory(nl):
        nonlocal env_instance
        env_instance = MockGridEnv(nl)
        # Position (2, 3) has the smallest delta (-10.5)
        env_instance.custom_delta_fn = lambda m_id, x, y: -10.5 if (x == 2 and y == 3) else float(x + y)
        return env_instance

    res = greedy_placement(netlist, env_class=env_factory)
    assert res.success is True
    assert res.placement == {"A": (2, 3)}

    place_calls = [args for op, args in env_instance.history if op == "place"]
    assert place_calls == [("A", 2, 3)]


def test_deterministic_tie_breaking_for_identical_deltas():
    """When multiple positions share the minimum delta, choose the first in row-major order."""
    netlist = {
        "grid_w": 3,
        "grid_h": 3,
        "macros": {"A": (1, 1)},
        "nets": [],
    }

    env_instance: MockGridEnv | None = None

    def env_factory(nl):
        nonlocal env_instance
        env_instance = MockGridEnv(nl)
        # All positions return the exact same delta (5.0)
        env_instance.custom_delta_fn = lambda m_id, x, y: 5.0
        return env_instance

    res = greedy_placement(netlist, env_class=env_factory)
    assert res.success is True
    # Row-major order (y=0, x=0) should be selected as the tie-break winner
    assert res.placement == {"A": (0, 0)}


# =============================================================================
# 3. Correct Coordinate Conversion: mask[y, x] -> env.place(macro_id, x, y)
# =============================================================================


def test_coordinate_conversion_from_mask_to_place():
    """Ensure mask row/col indexing mask[y, x] is mapped correctly to (x, y) coordinates."""
    netlist = {
        "grid_w": 5,
        "grid_h": 4,
        "macros": {"M": (1, 1)},
        "nets": [],
    }

    env_instance: MockGridEnv | None = None

    def env_factory(nl):
        nonlocal env_instance
        env_instance = MockGridEnv(nl)
        # Only position at row y=3, col x=1 is legal
        single_mask = np.zeros((4, 5), dtype=bool)
        single_mask[3, 1] = True
        env_instance.custom_masks["M"] = single_mask
        return env_instance

    res = greedy_placement(netlist, env_class=env_factory)
    assert res.success is True
    # Placed coordinate must be (x=1, y=3)
    assert res.placement == {"M": (1, 3)}

    place_calls = [args for op, args in env_instance.history if op == "place"]
    assert place_calls == [("M", 1, 3)]


# =============================================================================
# 4. Failure When Macro Has No Legal Positions
# =============================================================================


def test_failure_when_no_legal_positions_available():
    """When a macro has no legal positions in legal_mask, return failure result."""
    netlist = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {"A": (3, 3), "B": (2, 2)},
        "nets": [],
    }

    env_instance: MockGridEnv | None = None

    def env_factory(nl):
        nonlocal env_instance
        env_instance = MockGridEnv(nl)
        # Macro A places fine, but Macro B gets an all-False mask
        env_instance.custom_masks["B"] = np.zeros((4, 4), dtype=bool)
        return env_instance

    res = greedy_placement(netlist, env_class=env_factory)

    assert isinstance(res, GreedyPlacementResult)
    assert isinstance(res, PlacementResult)
    assert res.success is False
    assert res.hpwl is None
    assert res.congestion is None
    assert res.objective is None
    # Macro A was placed before failure, but B was not
    assert "A" in res.placement
    assert "B" not in res.placement


def test_failure_when_mask_is_none():
    """When legal_mask returns None, gracefully return documented failure result."""
    netlist = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {"A": (1, 1)},
        "nets": [],
    }

    def env_factory(nl):
        env = MockGridEnv(nl)
        env.legal_mask = lambda m_id: None
        return env

    res = greedy_placement(netlist, env_class=env_factory)
    assert res.success is False
    assert res.hpwl is None
    assert res.congestion is None
    assert res.objective is None


# =============================================================================
# 5. Successful Result and Objective Evaluation
# =============================================================================


def test_successful_result_and_objective_evaluation(sample_netlist):
    """Successful placement returns placement, success=True, and valid metrics."""
    res = greedy_placement(
        sample_netlist,
        alpha=1.5,
        beta=2.5,
        env_class=MockGridEnv,
    )

    assert isinstance(res, GreedyPlacementResult)
    assert res.success is True
    assert res.placement == {1: (0, 0), 2: (1, 0)}

    expected_eval = evaluate_placement(sample_netlist, res.placement, alpha=1.5, beta=2.5)
    assert np.isclose(res.hpwl, expected_eval.hpwl)
    assert np.isclose(res.congestion, expected_eval.congestion)
    assert np.isclose(res.objective, expected_eval.objective)


def test_alias_greedy_identical_to_greedy_placement(sample_netlist):
    """The alias `greedy` is identical to `greedy_placement`."""
    assert greedy is greedy_placement
    res = greedy(sample_netlist, env_class=MockGridEnv)
    assert res.success is True


# =============================================================================
# 6. Correct Forwarding of Alpha and Beta
# =============================================================================


def test_alpha_beta_forwarded_to_evaluator(sample_netlist, monkeypatch):
    """Verify alpha and beta are passed accurately to evaluate_placement."""
    recorded_args: dict[str, Any] = {}

    def spy_evaluate(netlist, placement, alpha, beta):
        recorded_args["alpha"] = alpha
        recorded_args["beta"] = beta
        return evaluate_placement(netlist, placement, alpha, beta)

    monkeypatch.setattr("src.algorithms.greedy.evaluate_placement", spy_evaluate)

    test_alpha = 4.25
    test_beta = 0.75
    greedy_placement(
        sample_netlist,
        alpha=test_alpha,
        beta=test_beta,
        env_class=MockGridEnv,
    )

    assert recorded_args["alpha"] == test_alpha
    assert recorded_args["beta"] == test_beta


# =============================================================================
# 7. Error Propagation
# =============================================================================


def test_environment_exception_propagates(sample_netlist):
    """Unexpected exceptions raised by the GridEnv propagate cleanly."""
    def broken_env_factory(nl):
        env = MockGridEnv(nl)
        def broken_place(*args):
            raise RuntimeError("Hardware/memory error during place operation.")
        env.place = broken_place
        return env

    with pytest.raises(RuntimeError, match="Hardware/memory error"):
        greedy_placement(sample_netlist, env_class=broken_env_factory)


def test_evaluator_exception_propagates(sample_netlist, monkeypatch):
    """Unexpected evaluator exceptions propagate rather than being swallowed."""
    def broken_evaluator(netlist, placement, alpha, beta):
        raise ValueError("Invalid placement detected during evaluation.")

    monkeypatch.setattr("src.algorithms.greedy.evaluate_placement", broken_evaluator)

    with pytest.raises(ValueError, match="Invalid placement detected"):
        greedy_placement(sample_netlist, env_class=MockGridEnv)


# =============================================================================
# 8. No Mutation of Input Netlist
# =============================================================================


def test_no_mutation_of_netlist(sample_netlist):
    """Greedy placement does not mutate the caller's netlist."""
    netlist_clone = copy.deepcopy(sample_netlist)

    greedy_placement(sample_netlist, alpha=2.0, beta=1.0, env_class=MockGridEnv)

    assert sample_netlist == netlist_clone


# =============================================================================
# 9. True Integration Test (Skipped / Marked Pending if GridEnv Absent)
# =============================================================================


def test_true_integration_with_shared_grid_env(sample_netlist):
    """Integration test with Zoya's shared GridEnv.

    Skipped if src.placement.grid.GridEnv is not available yet.
    """
    try:
        from src.placement.grid import GridEnv as SharedGridEnv
    except ImportError:
        pytest.skip(
            "Integration pending: Zoya's shared engine module 'src.placement.grid.GridEnv' is not yet implemented."
        )

    result = greedy_placement(sample_netlist)
    assert isinstance(result, GreedyPlacementResult)
    if result.success:
        assert result.placement is not None
        assert result.hpwl is not None
        assert result.congestion is not None
        assert result.objective is not None


# =============================================================================
# 10. Incomplete Placement Handled as Failure
# =============================================================================


def test_incomplete_placement_from_env_handled_as_failure(sample_netlist):
    """If the environment placement is missing macros, return success=False."""
    def incomplete_env_factory(nl):
        env = MockGridEnv(nl)
        # Override placement() to omit macro 2
        env.placement = lambda: {1: (0, 0)}
        return env

    res = greedy_placement(sample_netlist, env_class=incomplete_env_factory)
    assert res.success is False
    assert res.hpwl is None
    assert res.congestion is None
    assert res.objective is None


# =============================================================================
# 11. Parameter Validations
# =============================================================================


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
        greedy_placement(sample_netlist, alpha=a, beta=b, env_class=MockGridEnv)


# =============================================================================
# 12. Missing Shared GridEnv Raises NotImplementedError
# =============================================================================


def test_missing_shared_grid_env_raises_not_implemented(sample_netlist, monkeypatch):
    """When no shared GridEnv exists and none is injected, raise NotImplementedError."""
    monkeypatch.setattr("src.algorithms.greedy.GridEnv", None)

    with pytest.raises(NotImplementedError, match="Shared engine module"):
        greedy_placement(sample_netlist)
