"""Unit tests for RUDY congestion estimation and combined placement objective.

Tests cover all 20 required scenarios:
1. Empty placement.
2. Netlist with no nets.
3. A simple two-macro net.
4. A multi-pin net.
5. Same-row pins.
6. Same-column pins.
7. Pins or bounding boxes near grid boundaries.
8. Weighted nets.
9. Multiple overlapping net bounding boxes and additive demand.
10. Incomplete placement with fewer than two placed pins for a net.
11. A placement with multiple eligible nets.
12. Deterministic results across repeated calls.
13. Non-negative, finite demand values.
14. Correct output shape (grid_h, grid_w).
15. Correct mean-squared-demand congestion score.
16. Correct combined objective using known HPWL and congestion values.
17. Invalid zero or negative grid dimensions.
18. Invalid negative, NaN or infinite net weights.
19. Invalid negative, NaN or infinite alpha/beta.
20. No mutation of the input placement or netlist.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import numpy as np
import pytest

from src.placement.objective import (
    EvaluationResult,
    congestion_score,
    evaluate_placement,
    rudy_map,
)


# =============================================================================
# Helper Fixtures and Test Data Classes
# =============================================================================
# Lightweight temporary fixtures used only to supply data to objective functions.
# These avoid creating competing permanent engine models.


@dataclass(frozen=True)
class MockMacro:
    id: int | str
    w: int
    h: int


@dataclass(frozen=True)
class MockNet:
    id: int | str
    pins: list[int | str]
    weight: float = 1.0


@dataclass(frozen=True)
class MockNetlist:
    grid_w: int
    grid_h: int
    macros: dict[int | str, tuple[int, int]]
    nets: list[MockNet]


# =============================================================================
# 1. Empty Placement
# =============================================================================


def test_empty_placement():
    """An empty placement produces zero demand, zero HPWL, and zero congestion."""
    netlist = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1), 2: (1, 1)},
        "nets": [{"pins": [1, 2], "weight": 1.0}],
    }
    placement = {}

    demand = rudy_map(netlist, placement)
    assert demand.shape == (4, 4)
    assert np.all(demand == 0.0)

    score = congestion_score(netlist, placement)
    assert score == 0.0

    result = evaluate_placement(netlist, placement, alpha=1.0, beta=2.0)
    assert isinstance(result, EvaluationResult)
    assert result.hpwl == 0.0
    assert result.congestion == 0.0
    assert result.objective == 0.0


# =============================================================================
# 2. Netlist with No Nets
# =============================================================================


def test_netlist_with_no_nets():
    """A netlist with no nets produces zero demand, zero HPWL, and zero congestion."""
    netlist = {
        "grid_w": 5,
        "grid_h": 3,
        "macros": {1: (2, 2), 2: (1, 1)},
        "nets": [],
    }
    placement = {1: (0, 0), 2: (3, 1)}

    demand = rudy_map(netlist, placement)
    assert demand.shape == (3, 5)
    assert np.all(demand == 0.0)

    score = congestion_score(netlist, placement)
    assert score == 0.0

    result = evaluate_placement(netlist, placement, alpha=2.5, beta=1.5)
    assert result.hpwl == 0.0
    assert result.congestion == 0.0
    assert result.objective == 0.0


# =============================================================================
# 3. Simple Two-Macro Net (Hand-Calculated Verification)
# =============================================================================


def test_simple_two_macro_net():
    """Verify exact demand, HPWL, and congestion score against hand calculation.

    Grid: 4x4.
    Macro 1: (1, 1) placed at (0, 0) -> pin center (0.5, 0.5).
    Macro 2: (1, 1) placed at (2, 2) -> pin center (2.5, 2.5).
    Bbox: [0.5, 2.5] x [0.5, 2.5], W = 2.0, H = 2.0.
    Density: 1.0 * (2 + 2) / (2 * 2) = 1.0.
    Intersection x: cell 0 (0.5), cell 1 (1.0), cell 2 (0.5).
    Intersection y: cell 0 (0.5), cell 1 (1.0), cell 2 (0.5).
    """
    netlist = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1), 2: (1, 1)},
        "nets": [{"pins": [1, 2], "weight": 1.0}],
    }
    placement = {1: (0, 0), 2: (2, 2)}

    demand = rudy_map(netlist, placement)
    assert demand.shape == (4, 4)

    expected_demand = np.array(
        [
            [0.25, 0.50, 0.25, 0.00],
            [0.50, 1.00, 0.50, 0.00],
            [0.25, 0.50, 0.25, 0.00],
            [0.00, 0.00, 0.00, 0.00],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(demand, expected_demand, rtol=1e-12, atol=1e-12)

    # Wirelength conservation: sum of demand equals HPWL
    assert np.isclose(np.sum(demand), 4.0)

    # Congestion score: mean(demand ** 2) = 2.25 / 16 = 0.140625
    expected_congestion = 2.25 / 16.0
    score = congestion_score(netlist, placement)
    assert np.isclose(score, expected_congestion)

    result = evaluate_placement(netlist, placement, alpha=1.0, beta=2.0)
    assert np.isclose(result.hpwl, 4.0)
    assert np.isclose(result.congestion, expected_congestion)
    assert np.isclose(result.objective, 1.0 * 4.0 + 2.0 * expected_congestion)


# =============================================================================
# 4. Multi-Pin Net
# =============================================================================


def test_multi_pin_net():
    """A net with 3 pins spans the bounding box enclosing all 3 pin centers."""
    netlist = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1), 2: (1, 1), 3: (1, 1)},
        "nets": [{"pins": [1, 2, 3], "weight": 1.0}],
    }
    # Macro 1 at (0, 0) -> (0.5, 0.5)
    # Macro 2 at (2, 0) -> (2.5, 0.5)
    # Macro 3 at (1, 2) -> (1.5, 2.5)
    # Enclosing bbox: x in [0.5, 2.5], y in [0.5, 2.5]
    placement = {1: (0, 0), 2: (2, 0), 3: (1, 2)}

    demand = rudy_map(netlist, placement)
    expected_demand = np.array(
        [
            [0.25, 0.50, 0.25, 0.00],
            [0.50, 1.00, 0.50, 0.00],
            [0.25, 0.50, 0.25, 0.00],
            [0.00, 0.00, 0.00, 0.00],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(demand, expected_demand, rtol=1e-12)
    assert np.isclose(np.sum(demand), 4.0)


# =============================================================================
# 5. Same-Row Pins (Degenerate Bounding Box in Y)
# =============================================================================


def test_same_row_pins():
    """Pins sharing the same row are regularized to a 1-cell-wide corridor in row y.

    Grid: 4x4.
    Macro 1: at (0, 1) -> center (0.5, 1.5).
    Macro 2: at (2, 1) -> center (2.5, 1.5).
    Both pins have y = 1.5 -> mapped to corridor row y=1.
    W = 2.0, H = 0.0.
    Density = 1.0 * (2.0 + 0) / (2.0 * 1.0) = 1.0.
    """
    netlist = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1), 2: (1, 1)},
        "nets": [{"pins": [1, 2], "weight": 1.0}],
    }
    placement = {1: (0, 1), 2: (2, 1)}

    demand = rudy_map(netlist, placement)
    expected_demand = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 1.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(demand, expected_demand, rtol=1e-12)
    assert np.isclose(np.sum(demand), 2.0)  # HPWL = 2.0

    expected_congestion = (0.5**2 + 1.0**2 + 0.5**2) / 16.0  # 1.5 / 16 = 0.09375
    score = congestion_score(netlist, placement)
    assert np.isclose(score, expected_congestion)


# =============================================================================
# 6. Same-Column Pins (Degenerate Bounding Box in X)
# =============================================================================


def test_same_column_pins():
    """Pins sharing the same column are regularized to a 1-cell-wide corridor in column x.

    Grid: 4x4.
    Macro 1: at (1, 0) -> center (1.5, 0.5).
    Macro 2: at (1, 2) -> center (1.5, 2.5).
    Both pins have x = 1.5 -> mapped to corridor column x=1.
    W = 0.0, H = 2.0.
    Density = 1.0 * (0 + 2.0) / (1.0 * 2.0) = 1.0.
    """
    netlist = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1), 2: (1, 1)},
        "nets": [{"pins": [1, 2], "weight": 1.0}],
    }
    placement = {1: (1, 0), 2: (1, 2)}

    demand = rudy_map(netlist, placement)
    expected_demand = np.array(
        [
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(demand, expected_demand, rtol=1e-12)
    assert np.isclose(np.sum(demand), 2.0)  # HPWL = 2.0

    expected_congestion = 1.5 / 16.0
    score = congestion_score(netlist, placement)
    assert np.isclose(score, expected_congestion)


# =============================================================================
# 7. Boundary Placement
# =============================================================================


def test_boundary_placement():
    """Pins and bounding boxes touching grid boundaries do not exceed grid dimensions."""
    grid_w, grid_h = 5, 3
    netlist = {
        "grid_w": grid_w,
        "grid_h": grid_h,
        "macros": {1: (1, 1), 2: (1, 1)},
        "nets": [{"pins": [1, 2], "weight": 1.0}],
    }
    # Place at extreme opposite corners
    placement = {1: (0, 0), 2: (grid_w - 1, grid_h - 1)}

    demand = rudy_map(netlist, placement)
    assert demand.shape == (grid_h, grid_w)
    assert np.all(np.isfinite(demand))
    assert np.all(demand >= 0.0)

    # HPWL is (4.5 - 0.5) + (2.5 - 0.5) = 4.0 + 2.0 = 6.0
    assert np.isclose(np.sum(demand), 6.0)


# =============================================================================
# 8. Weighted Nets
# =============================================================================


def test_weighted_nets():
    """Net weights scale routing demand and HPWL proportionally."""
    weight = 3.5
    netlist_unweighted = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1), 2: (1, 1)},
        "nets": [{"pins": [1, 2], "weight": 1.0}],
    }
    netlist_weighted = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1), 2: (1, 1)},
        "nets": [{"pins": [1, 2], "weight": weight}],
    }
    placement = {1: (0, 0), 2: (2, 2)}

    demand_unweighted = rudy_map(netlist_unweighted, placement)
    demand_weighted = rudy_map(netlist_weighted, placement)

    np.testing.assert_allclose(demand_weighted, weight * demand_unweighted, rtol=1e-12)

    res_unweighted = evaluate_placement(netlist_unweighted, placement, alpha=1.0, beta=1.0)
    res_weighted = evaluate_placement(netlist_weighted, placement, alpha=1.0, beta=1.0)

    assert np.isclose(res_weighted.hpwl, weight * res_unweighted.hpwl)
    assert np.isclose(res_weighted.congestion, (weight**2) * res_unweighted.congestion)


# =============================================================================
# 9. Multiple Overlapping Net Bounding Boxes and Additive Demand
# =============================================================================


def test_multiple_overlapping_nets_additive_demand():
    """Routing demand from multiple nets is strictly additive across the grid."""
    netlist_net1 = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1), 2: (1, 1), 3: (1, 1)},
        "nets": [{"pins": [1, 2], "weight": 1.0}],
    }
    netlist_net2 = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1), 2: (1, 1), 3: (1, 1)},
        "nets": [{"pins": [2, 3], "weight": 2.0}],
    }
    netlist_combined = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1), 2: (1, 1), 3: (1, 1)},
        "nets": [
            {"pins": [1, 2], "weight": 1.0},
            {"pins": [2, 3], "weight": 2.0},
        ],
    }
    placement = {1: (0, 0), 2: (2, 2), 3: (3, 3)}

    demand1 = rudy_map(netlist_net1, placement)
    demand2 = rudy_map(netlist_net2, placement)
    demand_combined = rudy_map(netlist_combined, placement)

    np.testing.assert_allclose(demand_combined, demand1 + demand2, rtol=1e-12)


# =============================================================================
# 10. Incomplete Placement (Fewer Than Two Placed Pins)
# =============================================================================


def test_incomplete_placement():
    """Nets with fewer than two placed pins contribute zero to HPWL and demand."""
    netlist = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1), 2: (1, 1), 3: (1, 1), 4: (1, 1)},
        "nets": [
            {"pins": [1, 2], "weight": 1.0},  # Both placed (eligible)
            {"pins": [1, 3], "weight": 2.0},  # Only 1 placed (ineligible)
            {"pins": [3, 4], "weight": 3.0},  # 0 placed (ineligible)
        ],
    }
    # Only macro 1 and 2 are placed; 3 and 4 are unplaced
    placement = {1: (0, 0), 2: (2, 2)}

    # Netlist with only the eligible net
    netlist_only_net1 = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1), 2: (1, 1)},
        "nets": [{"pins": [1, 2], "weight": 1.0}],
    }

    demand_incomplete = rudy_map(netlist, placement)
    demand_only_net1 = rudy_map(netlist_only_net1, placement)

    np.testing.assert_allclose(demand_incomplete, demand_only_net1, rtol=1e-12)

    res_incomplete = evaluate_placement(netlist, placement, alpha=1.0, beta=1.0)
    res_only_net1 = evaluate_placement(netlist_only_net1, placement, alpha=1.0, beta=1.0)
    assert np.isclose(res_incomplete.hpwl, res_only_net1.hpwl)
    assert np.isclose(res_incomplete.congestion, res_only_net1.congestion)


# =============================================================================
# 11. Placement with Multiple Eligible Nets
# =============================================================================


def test_multiple_eligible_nets():
    """Placement with several active nets evaluates HPWL and congestion cumulatively."""
    netlist = {
        "grid_w": 5,
        "grid_h": 5,
        "macros": {1: (1, 1), 2: (1, 1), 3: (1, 1), 4: (1, 1)},
        "nets": [
            {"pins": [1, 2], "weight": 1.0},
            {"pins": [3, 4], "weight": 2.0},
        ],
    }
    placement = {
        1: (0, 0),
        2: (2, 0),  # HPWL = 2.0
        3: (1, 1),
        4: (1, 4),  # HPWL = 3.0 * weight 2.0 = 6.0
    }

    result = evaluate_placement(netlist, placement, alpha=1.0, beta=0.0)
    assert np.isclose(result.hpwl, 8.0)


# =============================================================================
# 12. Deterministic Results Across Repeated Calls
# =============================================================================


def test_determinism_across_repeated_calls():
    """Repeated calls with identical inputs must produce bitwise identical output."""
    netlist = {
        "grid_w": 6,
        "grid_h": 6,
        "macros": {1: (2, 1), 2: (1, 2), 3: (1, 1)},
        "nets": [
            {"pins": [1, 2, 3], "weight": 1.5},
            {"pins": [1, 3], "weight": 0.5},
        ],
    }
    placement = {1: (0, 0), 2: (4, 3), 3: (2, 2)}

    base_map = rudy_map(netlist, placement)
    base_res = evaluate_placement(netlist, placement, alpha=0.5, beta=1.5)

    for _ in range(10):
        m = rudy_map(netlist, placement)
        assert np.array_equal(m, base_map)
        res = evaluate_placement(netlist, placement, alpha=0.5, beta=1.5)
        assert res == base_res


# =============================================================================
# 13. Non-Negative and Finite Values
# =============================================================================


def test_non_negative_and_finite_demand():
    """Demand arrays and objective outputs must always be finite and non-negative."""
    netlist = {
        "grid_w": 8,
        "grid_h": 8,
        "macros": {i: (1, 1) for i in range(6)},
        "nets": [
            {"pins": [0, 1], "weight": 1.0},
            {"pins": [1, 2, 3], "weight": 2.0},
            {"pins": [4, 5], "weight": 0.0},  # Zero weight
        ],
    }
    placement = {i: (i, i) for i in range(6)}

    demand = rudy_map(netlist, placement)
    assert np.all(np.isfinite(demand))
    assert np.all(demand >= 0.0)

    score = congestion_score(netlist, placement)
    assert math.isfinite(score)
    assert score >= 0.0

    eval_res = evaluate_placement(netlist, placement, alpha=1.0, beta=1.0)
    assert math.isfinite(eval_res.hpwl) and eval_res.hpwl >= 0.0
    assert math.isfinite(eval_res.congestion) and eval_res.congestion >= 0.0
    assert math.isfinite(eval_res.objective) and eval_res.objective >= 0.0


# =============================================================================
# 14. Output Shape (grid_h, grid_w)
# =============================================================================


def test_correct_output_shape():
    """RUDY demand array has shape (grid_h, grid_w) with demand[y, x] indexing."""
    grid_w = 9
    grid_h = 5
    netlist = {
        "grid_w": grid_w,
        "grid_h": grid_h,
        "macros": {1: (1, 1), 2: (1, 1)},
        "nets": [{"pins": [1, 2], "weight": 1.0}],
    }
    placement = {1: (1, 1), 2: (7, 3)}

    demand = rudy_map(netlist, placement)
    assert demand.shape == (grid_h, grid_w)


# =============================================================================
# 15. Correct Mean-Squared-Demand Congestion Score
# =============================================================================


def test_mean_squared_demand_congestion_score():
    """Congestion score is precisely mean(demand ** 2)."""
    grid_w, grid_h = 4, 4
    netlist = {
        "grid_w": grid_w,
        "grid_h": grid_h,
        "macros": {1: (1, 1), 2: (1, 1)},
        "nets": [{"pins": [1, 2], "weight": 1.0}],
    }
    placement = {1: (0, 0), 2: (2, 2)}

    demand = rudy_map(netlist, placement)
    expected_score = float(np.sum(demand**2) / (grid_w * grid_h))
    actual_score = congestion_score(netlist, placement)

    assert np.isclose(actual_score, expected_score)
    assert np.isclose(actual_score, 2.25 / 16.0)


# =============================================================================
# 16. Combined Placement Objective Formulation
# =============================================================================


def test_evaluate_placement_objective():
    """Combined objective satisfies J(P) = alpha * HPWL + beta * congestion_score."""
    netlist = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1), 2: (1, 1)},
        "nets": [{"pins": [1, 2], "weight": 1.0}],
    }
    placement = {1: (0, 0), 2: (2, 2)}

    alpha = 3.0
    beta = 7.0
    result = evaluate_placement(netlist, placement, alpha=alpha, beta=beta)

    expected_hpwl = 4.0
    expected_congestion = 2.25 / 16.0
    expected_objective = alpha * expected_hpwl + beta * expected_congestion

    assert np.isclose(result.hpwl, expected_hpwl)
    assert np.isclose(result.congestion, expected_congestion)
    assert np.isclose(result.objective, expected_objective)


# =============================================================================
# 17. Invalid Zero or Negative Grid Dimensions
# =============================================================================


@pytest.mark.parametrize(
    "gw, gh",
    [
        (0, 5),
        (5, 0),
        (-2, 4),
        (4, -3),
        (float("nan"), 4),
        (4, float("inf")),
    ],
)
def test_invalid_grid_dimensions(gw, gh):
    """Zero, negative, NaN, or infinite grid dimensions raise ValueError."""
    netlist = {
        "grid_w": gw,
        "grid_h": gh,
        "macros": {1: (1, 1)},
        "nets": [],
    }
    placement = {1: (0, 0)}

    with pytest.raises(ValueError):
        rudy_map(netlist, placement)

    with pytest.raises(ValueError):
        congestion_score(netlist, placement)

    with pytest.raises(ValueError):
        evaluate_placement(netlist, placement, 1.0, 1.0)


# =============================================================================
# 18. Invalid Negative, NaN, or Infinite Net Weights
# =============================================================================


@pytest.mark.parametrize("invalid_weight", [-1.0, -0.001, float("nan"), float("inf"), float("-inf")])
def test_invalid_net_weights(invalid_weight):
    """Negative, NaN, or infinite net weights raise ValueError."""
    netlist = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1), 2: (1, 1)},
        "nets": [{"pins": [1, 2], "weight": invalid_weight}],
    }
    placement = {1: (0, 0), 2: (1, 1)}

    with pytest.raises(ValueError):
        rudy_map(netlist, placement)

    with pytest.raises(ValueError):
        congestion_score(netlist, placement)

    with pytest.raises(ValueError):
        evaluate_placement(netlist, placement, alpha=1.0, beta=1.0)


# =============================================================================
# 19. Invalid Negative, NaN, or Infinite Alpha/Beta
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
def test_invalid_alpha_beta(a, b):
    """Negative, NaN, or infinite alpha/beta raise ValueError."""
    netlist = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1), 2: (1, 1)},
        "nets": [{"pins": [1, 2], "weight": 1.0}],
    }
    placement = {1: (0, 0), 2: (1, 1)}

    with pytest.raises(ValueError):
        evaluate_placement(netlist, placement, alpha=a, beta=b)


# =============================================================================
# 20. No Mutation of Input Placement or Netlist
# =============================================================================


def test_no_mutation_of_inputs():
    """Functions must not alter the input placement or netlist."""
    netlist = {
        "grid_w": 5,
        "grid_h": 5,
        "macros": {1: (1, 1), 2: (2, 2)},
        "nets": [
            {"pins": [1, 2], "weight": 1.5},
            {"pins": [1], "weight": 2.0},
        ],
    }
    placement = {1: (0, 0), 2: (2, 2)}

    netlist_clone = copy.deepcopy(netlist)
    placement_clone = copy.deepcopy(placement)

    rudy_map(netlist, placement)
    assert netlist == netlist_clone
    assert placement == placement_clone

    congestion_score(netlist, placement)
    assert netlist == netlist_clone
    assert placement == placement_clone

    evaluate_placement(netlist, placement, alpha=1.0, beta=2.0)
    assert netlist == netlist_clone
    assert placement == placement_clone


# =============================================================================
# Additional Edge Case: Out-of-Bounds and Malformed Placement Validation
# =============================================================================


@pytest.mark.parametrize(
    "invalid_pos",
    [
        (-1, 0),       # Negative x
        (0, -1),       # Negative y
        (4, 0),        # x + w (4 + 1 = 5) exceeds grid_w=4
        (0, 4),        # y + h exceeds grid_h=4
        (1.5, 0),      # Non-integer coordinate
        (0, float("nan")),
    ],
)
def test_malformed_placement_coordinates(invalid_pos):
    """Malformed coordinates (out-of-bounds, negative, non-integer) raise ValueError."""
    netlist = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1)},
        "nets": [],
    }
    placement = {1: invalid_pos}

    with pytest.raises(ValueError):
        rudy_map(netlist, placement)


def test_placement_unknown_macro_raises():
    """Placement containing a macro not in the netlist raises ValueError."""
    netlist = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1)},
        "nets": [],
    }
    placement = {99: (0, 0)}

    with pytest.raises(ValueError):
        rudy_map(netlist, placement)


# =============================================================================
# Additional Edge Case: Dataclass Objects Support
# =============================================================================


def test_object_and_dataclass_support():
    """Ensure objective functions work seamlessly with dataclass/custom class models."""
    netlist = MockNetlist(
        grid_w=4,
        grid_h=4,
        macros={1: (1, 1), 2: (1, 1)},
        nets=[MockNet(id="N1", pins=[1, 2], weight=1.0)],
    )
    placement = {1: (0, 0), 2: (2, 2)}

    demand = rudy_map(netlist, placement)
    assert demand.shape == (4, 4)
    assert np.isclose(np.sum(demand), 4.0)

    res = evaluate_placement(netlist, placement, alpha=1.0, beta=1.0)
    assert np.isclose(res.hpwl, 4.0)
    assert np.isclose(res.congestion, 2.25 / 16.0)


# =============================================================================
# Additional Edge Case: Identical Pin Centers (W=0, H=0)
# =============================================================================


def test_identical_pin_centers_zero_demand():
    """Pins placed at identical positions have W=0, H=0 and contribute zero demand."""
    netlist = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1), 2: (1, 1)},
        "nets": [{"pins": [1, 2], "weight": 1.0}],
    }
    # Both macros at identical top-left (1, 1) -> centers both at (1.5, 1.5)
    placement = {1: (1, 1), 2: (1, 1)}

    demand = rudy_map(netlist, placement)
    assert np.all(demand == 0.0)

    res = evaluate_placement(netlist, placement, alpha=1.0, beta=1.0)
    assert res.hpwl == 0.0
    assert res.congestion == 0.0
    assert res.objective == 0.0


# =============================================================================
# Regression Tests: Fractional Bounding Boxes & Demand Conservation
# =============================================================================


def test_subunit_fractional_bounding_box_width_conservation():
    """Verify exact demand distribution and conservation for sub-unit width W=0.5, H=2.0.

    Grid: 4x4.
    Macro 1: (1, 1) placed at (0, 0) -> pin center (0.5, 0.5).
    Macro 2: (2, 1) placed at (0, 2) -> pin center (1.0, 2.5).
    Non-overlapping macros: macro 1 covers y in [0, 1), macro 2 covers y in [2, 3).
    Bbox: W = 1.0 - 0.5 = 0.5, H = 2.5 - 0.5 = 2.0.
    HPWL = 0.5 + 2.0 = 2.5.
    W_eff = 0.5, H_eff = 2.0 -> Area = 1.0.
    Density = 1.0 * (0.5 + 2.0) / 1.0 = 2.5.
    Intersections:
      X: cell 0 (0.5)
      Y: cell 0 (0.5), cell 1 (1.0), cell 2 (0.5)
    Expected demand:
      (0, 0): 2.5 * 0.5 * 0.5 = 0.625
      (1, 0): 2.5 * 0.5 * 1.0 = 1.250
      (2, 0): 2.5 * 0.5 * 0.5 = 0.625
    Total demand = 0.625 + 1.250 + 0.625 = 2.50 == net.weight * HPWL.
    """
    netlist = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1), 2: (2, 1)},
        "nets": [{"pins": [1, 2], "weight": 1.0}],
    }
    placement = {1: (0, 0), 2: (0, 2)}

    demand = rudy_map(netlist, placement)
    assert demand.shape == (4, 4)

    expected_demand = np.array(
        [
            [0.625, 0.0, 0.0, 0.0],
            [1.250, 0.0, 0.0, 0.0],
            [0.625, 0.0, 0.0, 0.0],
            [0.000, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(demand, expected_demand, rtol=1e-12, atol=1e-12)

    # Per-net demand conservation
    expected_hpwl = 2.5
    assert np.isclose(np.sum(demand), 1.0 * expected_hpwl)

    res = evaluate_placement(netlist, placement, alpha=1.0, beta=2.0)
    assert np.isclose(res.hpwl, expected_hpwl)
    expected_congestion = (0.625**2 + 1.25**2 + 0.625**2) / 16.0
    assert np.isclose(res.congestion, expected_congestion)
    assert np.isclose(res.objective, 1.0 * expected_hpwl + 2.0 * expected_congestion)


def test_subunit_fractional_bounding_box_both_axes_conservation():
    """Verify exact demand distribution and conservation for sub-unit dimensions on both axes.

    Grid: 4x4.
    Macro 1: (1, 1) placed at (0, 0) -> pin center (0.5, 0.5).
    Macro 2: (2, 2) placed at (0, 0) -> pin center (1.0, 1.0).
    Bbox: W = 0.5, H = 0.5.
    HPWL = 0.5 + 0.5 = 1.0.
    W_eff = 0.5, H_eff = 0.5 -> Area = 0.25.
    Density = 1.0 * (0.5 + 0.5) / 0.25 = 4.0.
    Intersections:
      X: cell 0 (0.5)
      Y: cell 0 (0.5)
    Expected demand:
      (0, 0): 4.0 * 0.5 * 0.5 = 1.0
    Total demand = 1.0 == net.weight * HPWL.
    """
    netlist = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1), 2: (2, 2)},
        "nets": [{"pins": [1, 2], "weight": 1.0}],
    }
    placement = {1: (0, 0), 2: (0, 0)}

    demand = rudy_map(netlist, placement)
    expected_demand = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(demand, expected_demand, rtol=1e-12, atol=1e-12)

    expected_hpwl = 1.0
    assert np.isclose(np.sum(demand), 1.0 * expected_hpwl)

    res = evaluate_placement(netlist, placement, alpha=2.0, beta=1.0)
    assert np.isclose(res.hpwl, expected_hpwl)
    expected_congestion = 1.0 / 16.0
    assert np.isclose(res.congestion, expected_congestion)


def test_subunit_degenerate_axis_conservation():
    """Verify demand conservation for a same-column net with sub-unit height H=0.5.

    Grid: 4x4.
    Macro 1: (2, 1) placed at (0, 0) -> pin center (1.0, 0.5).
    Macro 2: (2, 2) placed at (0, 0) -> pin center (1.0, 1.0).
    Bbox: W = 0.0, H = 0.5.
    HPWL = 0.0 + 0.5 = 0.5.
    W_eff = 1.0 (corridor at col 1), H_eff = 0.5 -> Area = 0.5.
    Density = 1.0 * (0.0 + 0.5) / 0.5 = 1.0.
    Expected demand:
      (0, 1): 1.0 * 1.0 * 0.5 = 0.5
    Total demand = 0.5 == net.weight * HPWL.
    """
    netlist = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (2, 1), 2: (2, 2)},
        "nets": [{"pins": [1, 2], "weight": 1.0}],
    }
    placement = {1: (0, 0), 2: (0, 0)}

    demand = rudy_map(netlist, placement)
    expected_demand = np.array(
        [
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(demand, expected_demand, rtol=1e-12, atol=1e-12)

    expected_hpwl = 0.5
    assert np.isclose(np.sum(demand), 1.0 * expected_hpwl)

    res = evaluate_placement(netlist, placement, alpha=1.0, beta=1.0)
    assert np.isclose(res.hpwl, expected_hpwl)
    assert np.isclose(res.congestion, 0.25 / 16.0)


def test_non_unit_weight_with_fractional_bounding_box_conservation():
    """Verify demand conservation with non-unit net weight and fractional dimensions W=1.5, H=2.0.

    Grid: 4x4.
    Macro 1: (1, 1) placed at (0, 0) -> pin center (0.5, 0.5).
    Macro 2: (2, 1) placed at (1, 2) -> pin center (2.0, 2.5).
    Net: weight = 2.4.
    Bbox: W = 2.0 - 0.5 = 1.5, H = 2.5 - 0.5 = 2.0.
    HPWL = 1.5 + 2.0 = 3.5.
    Weighted HPWL = 2.4 * 3.5 = 8.4.
    W_eff = 1.5, H_eff = 2.0 -> Area = 3.0.
    Density = 2.4 * (1.5 + 2.0) / 3.0 = 2.4 * 3.5 / 3.0 = 2.8.
    Intersections:
      X: cell 0 (0.5), cell 1 (1.0)
      Y: cell 0 (0.5), cell 1 (1.0), cell 2 (0.5)
    Expected demand:
      Row 0: [2.8 * 0.5 * 0.5, 2.8 * 1.0 * 0.5, 0, 0] = [0.70, 1.40, 0, 0]
      Row 1: [2.8 * 0.5 * 1.0, 2.8 * 1.0 * 1.0, 0, 0] = [1.40, 2.80, 0, 0]
      Row 2: [2.8 * 0.5 * 0.5, 2.8 * 1.0 * 0.5, 0, 0] = [0.70, 1.40, 0, 0]
      Row 3: [0, 0, 0, 0]
    Total demand = 0.70 + 1.40 + 1.40 + 2.80 + 0.70 + 1.40 = 8.40 == weight * HPWL.
    """
    weight = 2.4
    netlist = {
        "grid_w": 4,
        "grid_h": 4,
        "macros": {1: (1, 1), 2: (2, 1)},
        "nets": [{"pins": [1, 2], "weight": weight}],
    }
    placement = {1: (0, 0), 2: (1, 2)}

    demand = rudy_map(netlist, placement)
    assert demand.shape == (4, 4)

    expected_demand = np.array(
        [
            [0.70, 1.40, 0.0, 0.0],
            [1.40, 2.80, 0.0, 0.0],
            [0.70, 1.40, 0.0, 0.0],
            [0.00, 0.00, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(demand, expected_demand, rtol=1e-12, atol=1e-12)

    expected_unweighted_hpwl = 3.5
    expected_weighted_hpwl = weight * expected_unweighted_hpwl
    assert np.isclose(np.sum(demand), expected_weighted_hpwl)

    res = evaluate_placement(netlist, placement, alpha=1.0, beta=1.0)
    assert np.isclose(res.hpwl, expected_weighted_hpwl)
    expected_congestion = np.sum(expected_demand**2) / 16.0
    assert np.isclose(res.congestion, expected_congestion)
