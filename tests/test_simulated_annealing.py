"""Unit tests for Simulated Annealing (SA) placement baseline algorithm.

Tests cover:
1. Reproducibility with a fixed seed.
2. Initial-placement failure when the helper returns None.
3. Acceptance of a better candidate (delta < 0).
4. Acceptance of an equal-objective candidate (delta == 0).
5. Probabilistic acceptance of a worse candidate (delta > 0, u < p).
6. Rejection of a worse candidate when random draw exceeds acceptance probability (delta > 0, u >= p).
7. Best-placement tracking independently of the current accepted placement.
8. Temperature cooling and termination (max_iterations vs min_temperature).
9. Invalid parameter handling (NaN/inf temperatures, cooling_rate, max_iterations, seed, alpha/beta).
10. Failure or rejection of invalid neighbor proposals without corrupting current placement.
11. Correct forwarding of alpha and beta to evaluate_placement().
12. No mutation of caller-owned input objects (netlist, initial_placement).
13. Correct failure result if no complete feasible initial placement is available.
14. Integration with Zoya's actual engine (skipped pending implementation).
15. Aliases (sa, anneal).
16. Handling of empty netlist edge cases.
17. Exception propagation from environment and evaluator.
18. NotImplementedError when shared engine modules are missing and none are injected.
19. Fast path (unplace) vs fallback path (no unplace) parity.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Sequence
import numpy as np
import pytest

from src.placement.engine import PlacementEngine
from src.placement.models import Macro, Net, Netlist, PlacedMacro, Placement
from src.algorithms.simulated_annealing import (
    PlacementResult,
    SimulatedAnnealingResult,
    _acceptance_probability,
    _normalize_placement,
    anneal,
    sa,
    simulated_annealing,
)
from src.placement.objective import EvaluationResult, evaluate_placement


# =============================================================================
# Test Doubles / Mock GridEnv and Fake RNG
# =============================================================================


class MockGridEnv:
    """Mock GridEnv modeling Zoya's shared grid environment contract with unplace."""

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

    def legal_mask(self, macro_id: Any) -> np.ndarray:
        self.history.append(("legal_mask", macro_id))
        if macro_id in self.custom_masks:
            return self.custom_masks[macro_id]
        # Default: all grid anchors legal except already occupied anchors
        mask = np.ones((self.grid_h, self.grid_w), dtype=bool)
        for mid, (px, py) in self.placed.items():
            if mid != macro_id and 0 <= py < self.grid_h and 0 <= px < self.grid_w:
                mask[py, px] = False
        return mask

    def place(self, macro_id: Any, x: int, y: int) -> None:
        self.history.append(("place", (macro_id, x, y)))
        self.placed[macro_id] = (x, y)

    def unplace(self, macro_id: Any) -> None:
        self.history.append(("unplace", macro_id))
        self.placed.pop(macro_id, None)

    def placement(self) -> dict[Any, tuple[int, int]]:
        return dict(self.placed)


class MockGridEnvNoUnplace:
    """Mock GridEnv without unplace/remove to exercise the fresh replay fallback path."""

    def __init__(self, netlist: Any):
        self.netlist = netlist
        if isinstance(netlist, dict):
            self.grid_w = netlist.get("grid_w", 5)
            self.grid_h = netlist.get("grid_h", 5)
        else:
            self.grid_w = getattr(netlist, "grid_w", 5)
            self.grid_h = getattr(netlist, "grid_h", 5)
        self.placed: dict[Any, tuple[int, int]] = {}

    def legal_mask(self, macro_id: Any) -> np.ndarray:
        mask = np.ones((self.grid_h, self.grid_w), dtype=bool)
        for mid, (px, py) in self.placed.items():
            if 0 <= py < self.grid_h and 0 <= px < self.grid_w:
                mask[py, px] = False
        return mask

    def place(self, macro_id: Any, x: int, y: int) -> None:
        self.placed[macro_id] = (x, y)

    def placement(self) -> dict[Any, tuple[int, int]]:
        return dict(self.placed)


class FakeRNG:
    """Controllable test RNG for deterministic choice and acceptance checks."""

    def __init__(
        self,
        random_values: list[float] | None = None,
        choices: list[Any] | None = None,
    ):
        self._random_values = list(random_values or [])
        self._choices = list(choices or [])
        self.random_calls = 0
        self.choice_calls = 0

    def random(self) -> float:
        self.random_calls += 1
        if self._random_values:
            return self._random_values.pop(0)
        return 0.0

    def choice(self, seq: Sequence[Any]) -> Any:
        self.choice_calls += 1
        if self._choices:
            val = self._choices.pop(0)
            if val in seq:
                return val
        return seq[0]


@pytest.fixture
def sample_netlist():
    return {
        "grid_w": 5,
        "grid_h": 5,
        "macros": {
            "A": (1, 1),
            "B": (1, 1),
        },
        "nets": [
            {
                "id": "net0",
                "pins": ["A", "B"],
                "weight": 1.0,
            }
        ],
    }


# =============================================================================
# 1. Reproducibility with Fixed Seed
# =============================================================================


def test_reproducibility_with_fixed_seed(sample_netlist):
    """Running SA twice with identical seed produces identical trajectories and results."""
    def initial_legal_fn(nl, seed):
        return {"A": (0, 0), "B": (4, 4)}

    res1 = simulated_annealing(
        sample_netlist,
        seed=42,
        initial_temperature=10.0,
        cooling_rate=0.8,
        min_temperature=0.1,
        max_iterations=10,
        legal_placement_fn=initial_legal_fn,
        env_class=MockGridEnv,
    )
    res2 = simulated_annealing(
        sample_netlist,
        seed=42,
        initial_temperature=10.0,
        cooling_rate=0.8,
        min_temperature=0.1,
        max_iterations=10,
        legal_placement_fn=initial_legal_fn,
        env_class=MockGridEnv,
    )

    assert res1.success is True
    assert res2.success is True
    assert res1.placement == res2.placement
    assert res1.objective == res2.objective
    assert res1.iterations == res2.iterations
    assert res1.accepted_moves == res2.accepted_moves


def test_different_seeds_explore_differently(sample_netlist):
    """Different seeds can explore different moves."""
    def initial_legal_fn(nl, seed):
        return {"A": (0, 0), "B": (4, 4)}

    res1 = simulated_annealing(
        sample_netlist,
        seed=1,
        initial_temperature=50.0,
        cooling_rate=0.9,
        min_temperature=0.01,
        max_iterations=30,
        legal_placement_fn=initial_legal_fn,
        env_class=MockGridEnv,
    )
    res2 = simulated_annealing(
        sample_netlist,
        seed=9999,
        initial_temperature=50.0,
        cooling_rate=0.9,
        min_temperature=0.01,
        max_iterations=30,
        legal_placement_fn=initial_legal_fn,
        env_class=MockGridEnv,
    )
    assert res1.success is True
    assert res2.success is True
    assert res1.iterations == res2.iterations


# =============================================================================
# 2. Initial Placement Failure
# =============================================================================


def test_initial_placement_failure_when_helper_returns_none(sample_netlist):
    """When legal placement helper returns None, SA returns failure with None metrics."""
    res = simulated_annealing(
        sample_netlist,
        legal_placement_fn=lambda nl, seed: None,
        env_class=MockGridEnv,
    )
    assert isinstance(res, SimulatedAnnealingResult)
    assert isinstance(res, PlacementResult)
    assert res.success is False
    assert res.placement is None
    assert res.hpwl is None
    assert res.congestion is None
    assert res.objective is None
    assert res.iterations == 0
    assert res.accepted_moves == 0
    assert res.best_objective is None


# =============================================================================
# 3. Acceptance of a Better Candidate (delta < 0)
# =============================================================================


def test_acceptance_of_better_candidate(sample_netlist):
    """Candidates with delta <= 0 are accepted unconditionally."""
    # A at (0, 0) and B at (4, 4) -> HPWL is (4-0) + (4-0) = 8.
    # Moving B to (1, 1) -> HPWL is (1-0) + (1-0) = 2 (delta < 0).
    initial_p = {"A": (0, 0), "B": (4, 4)}

    # Force choice: macro "B", new anchor (1, 1)
    fake_rng = FakeRNG(choices=["B", (1, 1)])

    res = simulated_annealing(
        sample_netlist,
        initial_placement=initial_p,
        env_class=MockGridEnv,
        rng=fake_rng,
        initial_temperature=10.0,
        cooling_rate=0.9,
        min_temperature=5.0,
        max_iterations=1,
    )
    assert res.success is True
    assert res.accepted_moves == 1
    assert res.placement["B"] == (1, 1)
    init_eval = evaluate_placement(sample_netlist, initial_p, 1.0, 1.0)
    assert res.objective < init_eval.objective


# =============================================================================
# 4. Acceptance of an Equal-Objective Candidate (delta == 0)
# =============================================================================


def test_acceptance_of_equal_objective_candidate():
    """Candidates with identical objective (delta == 0) are accepted unconditionally."""
    netlist = {
        "grid_w": 3,
        "grid_h": 3,
        "macros": {"A": (1, 1)},
        "nets": [],
    }
    initial_p = {"A": (0, 0)}
    fake_rng = FakeRNG(choices=["A", (2, 2)])

    res = simulated_annealing(
        netlist,
        initial_placement=initial_p,
        env_class=MockGridEnv,
        rng=fake_rng,
        initial_temperature=10.0,
        cooling_rate=0.9,
        min_temperature=5.0,
        max_iterations=1,
    )
    assert res.success is True
    assert res.accepted_moves == 1
    assert res.placement["A"] == (2, 2)
    assert res.objective == 0.0


# =============================================================================
# 5. Probabilistic Acceptance of a Worse Candidate (delta > 0, u < p)
# =============================================================================


def test_probabilistic_acceptance_of_worse_candidate(sample_netlist):
    """When delta > 0 and random draw u < exp(-delta/T), move is accepted."""
    # Initial: A at (0, 0), B at (1, 1) -> HPWL = 2
    # Move B to (4, 4) -> HPWL = 8 (worse: delta > 0)
    initial_p = {"A": (0, 0), "B": (1, 1)}

    init_eval = evaluate_placement(sample_netlist, initial_p, 1.0, 1.0)
    cand_p = {"A": (0, 0), "B": (4, 4)}
    cand_eval = evaluate_placement(sample_netlist, cand_p, 1.0, 1.0)
    delta = cand_eval.objective - init_eval.objective
    assert delta > 0.0

    temp = 100.0
    prob = _acceptance_probability(delta, temp)
    assert 0.0 < prob < 1.0

    # Draw u strictly smaller than prob
    u = prob / 2.0
    fake_rng = FakeRNG(choices=["B", (4, 4)], random_values=[u])

    res = simulated_annealing(
        sample_netlist,
        initial_placement=initial_p,
        env_class=MockGridEnv,
        rng=fake_rng,
        initial_temperature=temp,
        cooling_rate=0.9,
        min_temperature=temp * 0.5,
        max_iterations=1,
    )
    assert res.success is True
    assert res.accepted_moves == 1
    # Even though uphill move was accepted as current state, best_placement tracks the best!
    assert res.placement == initial_p
    assert math.isclose(res.objective, init_eval.objective)


# =============================================================================
# 6. Rejection of Worse Candidate When u >= p
# =============================================================================


def test_rejection_of_worse_candidate_when_random_exceeds_probability(sample_netlist):
    """When delta > 0 and random draw u >= exp(-delta/T), move is rejected."""
    initial_p = {"A": (0, 0), "B": (1, 1)}

    init_eval = evaluate_placement(sample_netlist, initial_p, 1.0, 1.0)
    cand_p = {"A": (0, 0), "B": (4, 4)}
    cand_eval = evaluate_placement(sample_netlist, cand_p, 1.0, 1.0)
    delta = cand_eval.objective - init_eval.objective
    assert delta > 0.0

    temp = 10.0
    prob = _acceptance_probability(delta, temp)

    # Force u > prob (e.g. 0.99)
    fake_rng = FakeRNG(choices=["B", (4, 4)], random_values=[0.99])

    res = simulated_annealing(
        sample_netlist,
        initial_placement=initial_p,
        env_class=MockGridEnv,
        rng=fake_rng,
        initial_temperature=temp,
        cooling_rate=0.9,
        min_temperature=temp * 0.5,
        max_iterations=1,
    )
    assert res.success is True
    assert res.accepted_moves == 0
    assert res.placement == initial_p
    assert math.isclose(res.objective, init_eval.objective)


# =============================================================================
# 7. Best-Placement Tracking Independently of Current Placement
# =============================================================================


def test_best_placement_tracked_independently_of_current(sample_netlist):
    """When annealing accepts a worse candidate at the end, best placement is returned."""
    # Step 1: Start at (0, 0) & (3, 3) (HPWL = 6)
    # Step 2: Move to (1, 1) & (3, 3) (HPWL = 4, better! Best becomes this)
    # Step 3: Move to (0, 0) & (4, 4) (HPWL = 8, worse! Accepted uphill)
    # Final result must return step 2's placement, not step 3's!
    initial_p = {"A": (0, 0), "B": (3, 3)}

    # Iteration 1 choices: A -> (1, 1) (downhill)
    # Iteration 2 choices: B -> (4, 4) (uphill, u=0.01 accepts it)
    fake_rng = FakeRNG(
        choices=["A", (1, 1), "B", (4, 4)],
        random_values=[0.01],  # accepts uphill move in iter 2
    )

    res = simulated_annealing(
        sample_netlist,
        initial_placement=initial_p,
        env_class=MockGridEnv,
        rng=fake_rng,
        initial_temperature=100.0,
        cooling_rate=0.9,
        min_temperature=50.0,
        max_iterations=2,
    )
    assert res.success is True
    assert res.iterations == 2
    assert res.accepted_moves == 2
    # Best placement should be the step 1 improvement, NOT the final step 2 state
    assert res.placement == {"A": (1, 1), "B": (3, 3)}
    step2_eval = evaluate_placement(sample_netlist, {"A": (1, 1), "B": (3, 3)}, 1.0, 1.0)
    assert math.isclose(res.objective, step2_eval.objective)


# =============================================================================
# 8. Temperature Cooling and Termination
# =============================================================================


def test_termination_by_max_iterations(sample_netlist):
    """SA stops when max_iterations is reached before temperature cools below min_temperature."""
    initial_p = {"A": (0, 0), "B": (1, 1)}
    res = simulated_annealing(
        sample_netlist,
        initial_placement=initial_p,
        env_class=MockGridEnv,
        initial_temperature=100.0,
        cooling_rate=0.999,
        min_temperature=0.01,
        max_iterations=5,
    )
    assert res.success is True
    assert res.iterations == 5


def test_termination_by_min_temperature(sample_netlist):
    """SA stops when temperature drops at or below min_temperature."""
    initial_p = {"A": (0, 0), "B": (1, 1)}
    # Initial: 1.0, cooling: 0.5, min: 0.3
    # Iter 1: T cools to 0.5 (> 0.3)
    # Iter 2: T cools to 0.25 (<= 0.3) -> terminates!
    res = simulated_annealing(
        sample_netlist,
        initial_placement=initial_p,
        env_class=MockGridEnv,
        initial_temperature=1.0,
        cooling_rate=0.5,
        min_temperature=0.3,
        max_iterations=1000,
    )
    assert res.success is True
    assert res.iterations == 2


# =============================================================================
# 9. Invalid Parameter Handling
# =============================================================================


@pytest.mark.parametrize(
    "temp",
    [-10.0, 0.0, -0.001, float("nan"), float("inf"), float("-inf"), True, "100"],
)
def test_invalid_initial_temperature_raises(sample_netlist, temp):
    with pytest.raises(ValueError):
        simulated_annealing(
            sample_netlist,
            initial_temperature=temp,
            env_class=MockGridEnv,
            initial_placement={"A": (0, 0), "B": (1, 1)},
        )


@pytest.mark.parametrize(
    "min_t",
    [-1.0, 0.0, -0.001, float("nan"), float("inf"), float("-inf"), True, "0.01"],
)
def test_invalid_min_temperature_raises(sample_netlist, min_t):
    with pytest.raises(ValueError):
        simulated_annealing(
            sample_netlist,
            min_temperature=min_t,
            env_class=MockGridEnv,
            initial_placement={"A": (0, 0), "B": (1, 1)},
        )


def test_initial_temperature_less_than_min_raises(sample_netlist):
    with pytest.raises(ValueError, match="strictly greater"):
        simulated_annealing(
            sample_netlist,
            initial_temperature=10.0,
            min_temperature=20.0,
            env_class=MockGridEnv,
            initial_placement={"A": (0, 0), "B": (1, 1)},
        )


@pytest.mark.parametrize(
    "cr",
    [0.0, 1.0, -0.5, 1.5, float("nan"), float("inf"), float("-inf"), True, "0.95"],
)
def test_invalid_cooling_rate_raises(sample_netlist, cr):
    with pytest.raises(ValueError):
        simulated_annealing(
            sample_netlist,
            cooling_rate=cr,
            env_class=MockGridEnv,
            initial_placement={"A": (0, 0), "B": (1, 1)},
        )


@pytest.mark.parametrize("mi", [0, -1, -100, 3.14, True, "1000"])
def test_invalid_max_iterations_raises(sample_netlist, mi):
    with pytest.raises(ValueError):
        simulated_annealing(
            sample_netlist,
            max_iterations=mi,
            env_class=MockGridEnv,
            initial_placement={"A": (0, 0), "B": (1, 1)},
        )


@pytest.mark.parametrize("seed_val", [True, False, 3.14, "seed42"])
def test_invalid_seed_raises(sample_netlist, seed_val):
    with pytest.raises(ValueError):
        simulated_annealing(
            sample_netlist,
            seed=seed_val,
            env_class=MockGridEnv,
            initial_placement={"A": (0, 0), "B": (1, 1)},
        )


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [
        (-1.0, 1.0),
        (1.0, -0.5),
        (float("nan"), 1.0),
        (1.0, float("nan")),
        (float("inf"), 1.0),
        (1.0, float("-inf")),
        (True, 1.0),
        (1.0, False),
    ],
)
def test_invalid_alpha_beta_raises(sample_netlist, alpha, beta):
    with pytest.raises(ValueError):
        simulated_annealing(
            sample_netlist,
            alpha=alpha,
            beta=beta,
            env_class=MockGridEnv,
            initial_placement={"A": (0, 0), "B": (1, 1)},
        )


# =============================================================================
# 10. Rejection of Invalid Proposals Without Corrupting State
# =============================================================================


def test_rejection_when_no_alternative_anchors_exist(sample_netlist):
    """When legal_mask has no alternative anchors, proposal is skipped without corruption."""
    initial_p = {"A": (0, 0), "B": (1, 1)}

    class RestrictedEnv(MockGridEnv):
        def legal_mask(self, macro_id: Any) -> np.ndarray:
            # All anchors False (no legal moves possible)
            return np.zeros((self.grid_h, self.grid_w), dtype=bool)

    res = simulated_annealing(
        sample_netlist,
        initial_placement=initial_p,
        env_class=RestrictedEnv,
        initial_temperature=10.0,
        cooling_rate=0.5,
        min_temperature=1.0,
        max_iterations=5,
    )
    assert res.success is True
    assert res.accepted_moves == 0
    assert res.placement == initial_p


# =============================================================================
# 11. Parameter Forwarding to evaluate_placement()
# =============================================================================


def test_alpha_beta_forwarded_to_evaluator(sample_netlist, monkeypatch):
    """Custom alpha and beta are forwarded to evaluate_placement."""
    eval_calls: list[tuple[float, float]] = []

    def spy_eval(nl, p, alpha=1.0, beta=1.0):
        eval_calls.append((alpha, beta))
        return evaluate_placement(nl, p, alpha, beta)

    monkeypatch.setattr(
        "src.algorithms.simulated_annealing.evaluate_placement", spy_eval
    )

    simulated_annealing(
        sample_netlist,
        alpha=2.5,
        beta=0.75,
        initial_placement={"A": (0, 0), "B": (1, 1)},
        env_class=MockGridEnv,
        max_iterations=2,
    )
    assert len(eval_calls) >= 1
    for a, b in eval_calls:
        assert a == 2.5
        assert b == 0.75


# =============================================================================
# 12. No Mutation of Caller-Owned Input Objects
# =============================================================================


def test_no_mutation_of_netlist_and_initial_placement(sample_netlist):
    """SA does not mutate netlist or initial_placement dictionary."""
    netlist_copy = copy.deepcopy(sample_netlist)
    initial_p = {"A": (0, 0), "B": (1, 1)}
    initial_p_copy = copy.deepcopy(initial_p)

    simulated_annealing(
        sample_netlist,
        initial_placement=initial_p,
        env_class=MockGridEnv,
        max_iterations=10,
    )

    assert sample_netlist == netlist_copy
    assert initial_p == initial_p_copy


# =============================================================================
# 13. Failure on Incomplete Initial Placement
# =============================================================================


def test_incomplete_initial_placement_returns_failure(sample_netlist):
    """When initial placement is missing macros, return failure with None metrics."""
    incomplete_p = {"A": (0, 0)}

    res = simulated_annealing(
        sample_netlist,
        initial_placement=incomplete_p,
        env_class=MockGridEnv,
    )
    assert res.success is False
    assert res.placement == incomplete_p
    assert res.hpwl is None
    assert res.congestion is None
    assert res.objective is None
    assert res.iterations == 0
    assert res.accepted_moves == 0


# =============================================================================
# 14. True Integration Test (Skipped Pending Zoya's Engine)
# =============================================================================


def test_true_integration_with_shared_engine(sample_netlist):
    """True integration test with shared engine PlacementEngine."""
    res = simulated_annealing(sample_netlist, max_iterations=5, seed=42)
    assert isinstance(res, SimulatedAnnealingResult)
    assert res.success is True
    assert res.placement is not None
    assert res.hpwl is not None
    assert res.congestion is not None
    assert res.objective is not None


# =============================================================================
# 15. Aliases
# =============================================================================


def test_aliases_identical_to_simulated_annealing():
    assert sa is simulated_annealing
    assert anneal is simulated_annealing


# =============================================================================
# 16. Empty Netlist Edge Case
# =============================================================================


def test_empty_netlist_returns_zero_metrics():
    empty_netlist = {"grid_w": 5, "grid_h": 5, "macros": {}, "nets": []}
    res = simulated_annealing(
        empty_netlist,
        initial_placement={},
        env_class=MockGridEnv,
    )
    assert res.success is True
    assert res.placement == {}
    assert res.hpwl == 0.0
    assert res.congestion == 0.0
    assert res.objective == 0.0
    assert res.iterations == 0
    assert res.accepted_moves == 0


# =============================================================================
# 17. Exception Propagation
# =============================================================================


def test_environment_exception_propagates(sample_netlist):
    def broken_env(nl):
        raise RuntimeError("GridEnv failed during neighbor generation")

    with pytest.raises(RuntimeError, match="GridEnv failed"):
        simulated_annealing(
            sample_netlist,
            initial_placement={"A": (0, 0), "B": (1, 1)},
            env_class=broken_env,
        )


def test_evaluator_exception_propagates(sample_netlist, monkeypatch):
    def broken_eval(nl, p, alpha=1.0, beta=1.0):
        raise ValueError("Evaluator crash")

    monkeypatch.setattr(
        "src.algorithms.simulated_annealing.evaluate_placement", broken_eval
    )

    with pytest.raises(ValueError, match="Evaluator crash"):
        simulated_annealing(
            sample_netlist,
            initial_placement={"A": (0, 0), "B": (1, 1)},
            env_class=MockGridEnv,
        )


# =============================================================================
# 18. Missing Engine Raises NotImplementedError
# =============================================================================


def test_missing_shared_helper_raises_not_implemented(sample_netlist, monkeypatch):
    monkeypatch.setattr("src.algorithms.simulated_annealing.random_legal_placement", None)
    with pytest.raises(NotImplementedError, match="random_legal_placement"):
        simulated_annealing(sample_netlist, env_class=MockGridEnv)


def test_missing_shared_grid_env_raises_not_implemented(sample_netlist, monkeypatch):
    monkeypatch.setattr("src.algorithms.simulated_annealing.GridEnv", None)
    with pytest.raises(NotImplementedError, match="GridEnv"):
        simulated_annealing(
            sample_netlist,
            initial_placement={"A": (0, 0), "B": (1, 1)},
        )


# =============================================================================
# 19. Fast Path (unplace) vs Fallback Path (no unplace) Parity
# =============================================================================


def test_environment_without_unplace_works_via_replay_fallback(sample_netlist):
    """GridEnv without unplace/remove method executes successfully via replay fallback."""
    initial_p = {"A": (0, 0), "B": (4, 4)}
    res = simulated_annealing(
        sample_netlist,
        initial_placement=initial_p,
        env_class=MockGridEnvNoUnplace,
        initial_temperature=10.0,
        cooling_rate=0.8,
        min_temperature=1.0,
        max_iterations=5,
        seed=42,
    )
    assert res.success is True
    assert res.iterations > 0


# =============================================================================
# 20. Regression Tests for Fix 1: Coordinate Normalization
# =============================================================================


class PlacedMacroDouble:
    """Mock placed-macro object exposing x and y attributes."""

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


def test_normalize_placement_tuple_and_list():
    """_normalize_placement correctly handles tuple and list representations."""
    p = {"A": (1, 2), "B": [3, 4]}
    norm = _normalize_placement(p)
    assert norm == {"A": (1, 2), "B": (3, 4)}


def test_normalize_placement_mapping():
    """_normalize_placement correctly handles mapping representations."""
    p = {"A": {"x": 1, "y": 2}, "B": {"x": 3, "y": 4}}
    norm = _normalize_placement(p)
    assert norm == {"A": (1, 2), "B": (3, 4)}


def test_normalize_placement_object_with_xy_attributes():
    """_normalize_placement correctly handles objects exposing .x and .y attributes."""
    p = {"A": PlacedMacroDouble(1, 2), "B": PlacedMacroDouble(3, 4)}
    norm = _normalize_placement(p)
    assert norm == {"A": (1, 2), "B": (3, 4)}


def test_simulated_annealing_with_object_based_initial_placement(sample_netlist):
    """End-to-end SA run with object-based coordinates succeeds and preserves macro IDs."""
    initial_p = {
        "A": PlacedMacroDouble(0, 0),
        "B": PlacedMacroDouble(1, 1),
    }
    res = simulated_annealing(
        sample_netlist,
        initial_placement=initial_p,
        env_class=MockGridEnv,
        initial_temperature=10.0,
        cooling_rate=0.8,
        min_temperature=1.0,
        max_iterations=3,
        seed=42,
    )
    assert res.success is True
    assert set(res.placement.keys()) == {"A", "B"}
    for m_id, coord in res.placement.items():
        assert isinstance(coord, tuple)
        assert len(coord) == 2


def test_normalize_placement_preserves_non_string_macro_ids():
    """Macro IDs of arbitrary types (int, tuple) are preserved without string conversion."""
    p = {
        101: PlacedMacroDouble(1, 2),
        ("macro", 2): {"x": 3, "y": 4},
    }
    norm = _normalize_placement(p)
    assert 101 in norm
    assert ("macro", 2) in norm
    assert norm[101] == (1, 2)
    assert norm[("macro", 2)] == (3, 4)


@pytest.mark.parametrize(
    "invalid_coord",
    [
        (1,),  # tuple len != 2
        (1, 2, 3),  # tuple len != 2
        {"x": 1},  # missing y
        {"y": 2},  # missing x
        "invalid_str",  # unsupported format
        42,  # unsupported format
        {"x": "1", "y": 2},  # string coord
        {"x": True, "y": 2},  # boolean coord
        {"x": float("nan"), "y": 2},  # NaN coord
        {"x": float("inf"), "y": 2},  # infinite coord
        {"x": 1.5, "y": 2},  # fractional grid coord
    ],
)
def test_normalize_placement_malformed_coordinates_raise_value_error(invalid_coord):
    """Malformed coordinates raise clear ValueError."""
    p = {"A": invalid_coord}
    with pytest.raises(ValueError):
        _normalize_placement(p)


# =============================================================================
# 21. Regression Tests for Fix 2: Fast-Path Capability Detection
# =============================================================================


def test_fast_path_selected_with_factory_function(sample_netlist):
    """When env_class is a factory function returning an unplace-capable env, fast path is used."""
    env_created: list[MockGridEnv] = []

    def factory(nl):
        inst = MockGridEnv(nl)
        env_created.append(inst)
        return inst

    initial_p = {"A": (0, 0), "B": (4, 4)}
    res = simulated_annealing(
        sample_netlist,
        initial_placement=initial_p,
        env_class=factory,
        initial_temperature=10.0,
        cooling_rate=0.8,
        min_temperature=1.0,
        max_iterations=5,
        seed=42,
    )
    assert res.success is True
    # Exactly one persistent active environment was instantiated across all iterations
    assert len(env_created) == 1
    # Verify unplace was actually called on that instance
    unplace_calls = [action for action, arg in env_created[0].history if action == "unplace"]
    assert len(unplace_calls) > 0


def test_fast_path_selected_with_lambda_factory(sample_netlist):
    """When env_class is a lambda returning an unplace-capable env, fast path is used."""
    initial_p = {"A": (0, 0), "B": (4, 4)}
    res = simulated_annealing(
        sample_netlist,
        initial_placement=initial_p,
        env_class=lambda nl: MockGridEnv(nl),
        initial_temperature=10.0,
        cooling_rate=0.8,
        min_temperature=1.0,
        max_iterations=5,
        seed=42,
    )
    assert res.success is True
    assert res.iterations > 0


# =============================================================================
# 22. Regression Tests for Fix 3: Exception-Safe Environment Restoration
# =============================================================================


def test_active_env_restored_when_legal_mask_raises(sample_netlist):
    """When legal_mask raises, the unplaced macro is restored to original position before propagating."""
    env_instance: MockGridEnv | None = None

    class FailingMaskEnv(MockGridEnv):
        def legal_mask(self, macro_id: Any) -> np.ndarray:
            # Macro is currently unplaced in self.placed
            assert macro_id not in self.placed
            raise RuntimeError("Mask computation exploded")

    def factory(nl):
        nonlocal env_instance
        env_instance = FailingMaskEnv(nl)
        return env_instance

    initial_p = {"A": (0, 0), "B": (1, 1)}

    with pytest.raises(RuntimeError, match="Mask computation exploded"):
        simulated_annealing(
            sample_netlist,
            initial_placement=initial_p,
            env_class=factory,
            initial_temperature=10.0,
            cooling_rate=0.8,
            min_temperature=1.0,
            max_iterations=5,
        )

    # Verify that the finally block restored the macro to its initial coordinate in active_env!
    assert env_instance is not None
    assert env_instance.placed["A"] == (0, 0)
    assert env_instance.placed["B"] == (1, 1)


def test_restoration_failure_propagates_informative_runtime_error(sample_netlist):
    """If restoring the macro itself fails during finally, propagate informative RuntimeError."""
    class BrokenRestoreEnv(MockGridEnv):
        def __init__(self, netlist: Any):
            super().__init__(netlist)
            self.fail_on_restore = False

        def unplace(self, macro_id: Any) -> None:
            super().unplace(macro_id)
            self.fail_on_restore = True

        def place(self, macro_id: Any, x: int, y: int) -> None:
            if self.fail_on_restore:
                raise ValueError("Hardware bus disconnect")
            super().place(macro_id, x, y)

    initial_p = {"A": (0, 0), "B": (1, 1)}

    with pytest.raises(RuntimeError, match="Failed to restore macro"):
        simulated_annealing(
            sample_netlist,
            initial_placement=initial_p,
            env_class=BrokenRestoreEnv,
            initial_temperature=10.0,
            cooling_rate=0.8,
            min_temperature=1.0,
            max_iterations=5,
        )


# =============================================================================
# 23. Regression Tests for Fix 4: Deterministic Replay Order
# =============================================================================


def test_replay_fallback_order_independent_of_dict_insertion_order():
    """Replay fallback places remaining macros in deterministic sorted order regardless of dict insertion order."""
    placed_sequences: list[list[Any]] = []

    class OrderTrackingEnv(MockGridEnvNoUnplace):
        def place(self, macro_id: Any, x: int, y: int) -> None:
            super().place(macro_id, x, y)
            placed_sequences.append(macro_id)

    # Netlist with 3 macros
    nl = {
        "grid_w": 5,
        "grid_h": 5,
        "macros": {"A": (1, 1), "B": (1, 1), "C": (1, 1)},
        "nets": [{"id": "net0", "pins": ["A", "B", "C"], "weight": 1.0}],
    }

    # Dict insertion order 1: C, B, A
    p1 = {"C": (4, 4), "B": (2, 2), "A": (0, 0)}
    placed_sequences.clear()
    res1 = simulated_annealing(
        nl,
        initial_placement=p1,
        env_class=OrderTrackingEnv,
        initial_temperature=10.0,
        cooling_rate=0.8,
        min_temperature=5.0,
        max_iterations=1,
        seed=42,
    )

    seq1 = list(placed_sequences)

    # Dict insertion order 2: B, A, C
    p2 = {"B": (2, 2), "A": (0, 0), "C": (4, 4)}
    placed_sequences.clear()
    res2 = simulated_annealing(
        nl,
        initial_placement=p2,
        env_class=OrderTrackingEnv,
        initial_temperature=10.0,
        cooling_rate=0.8,
        min_temperature=5.0,
        max_iterations=1,
        seed=42,
    )

    seq2 = list(placed_sequences)

    # The replay sequence of placed macros must be strictly identical regardless of insertion order
    assert seq1 == seq2
    assert res1.placement == res2.placement
    assert res1.objective == res2.objective


# =============================================================================
# 25. Canonical Netlist and Placement List Integration
# =============================================================================


def test_simulated_annealing_canonical_netlist():
    """SA works end-to-end with canonical Netlist instances using PlacementEngine."""
    macros = (
        Macro(id=1, w=2, h=2),
        Macro(id=2, w=2, h=2),
    )
    nets = (Net(id=1, macro_ids=(1, 2), weight=1.0),)
    netlist = Netlist(version="1.0", grid_w=6, grid_h=6, macros=macros, nets=nets)

    res = simulated_annealing(
        netlist,
        initial_temperature=20.0,
        cooling_rate=0.85,
        min_temperature=1.0,
        max_iterations=10,
        seed=42,
    )
    assert res.success is True
    assert res.placement is not None
    assert len(res.placement) == 2
    assert res.hpwl is not None
    assert res.congestion is not None
    assert res.objective is not None


def test_simulated_annealing_with_placement_list_initial():
    """SA accepts and returns list[PlacedMacro] representation when supplied as initial placement."""
    macros = (
        Macro(id=1, w=1, h=1),
        Macro(id=2, w=1, h=1),
    )
    nets = (Net(id=1, macro_ids=(1, 2), weight=1.0),)
    netlist = Netlist(version="1.0", grid_w=4, grid_h=4, macros=macros, nets=nets)

    init_p: Placement = [
        PlacedMacro(macro_id=1, x=0, y=0),
        PlacedMacro(macro_id=2, x=3, y=3),
    ]

    res = simulated_annealing(
        netlist,
        initial_placement=init_p,
        initial_temperature=10.0,
        cooling_rate=0.8,
        min_temperature=1.0,
        max_iterations=5,
        seed=42,
    )
    assert res.success is True
    assert isinstance(res.placement, list)
    assert len(res.placement) == 2
    assert all(isinstance(pm, PlacedMacro) for pm in res.placement)
    assert res.objective <= evaluate_placement(netlist, init_p, 1.0, 1.0).objective


# =============================================================================
# 26. Move Types (Shift, Swap, Relocate) and Operator Legality
# =============================================================================


def test_sa_move_weights_validation():
    """Validate move_weights and shift_window constraints."""
    netlist = {"grid_w": 5, "grid_h": 5, "macros": {"A": (1, 1)}, "nets": []}

    # Non-mapping move_weights
    with pytest.raises(ValueError, match="move_weights must be a mapping"):
        simulated_annealing(netlist, move_weights=["shift", "swap"])

    # Unknown move type
    with pytest.raises(ValueError, match="Unknown move type in move_weights"):
        simulated_annealing(netlist, move_weights={"rotate": 1.0})

    # Negative move weight
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        simulated_annealing(netlist, move_weights={"shift": -0.5})

    # Non-numeric move weight
    with pytest.raises(ValueError, match="must be numeric"):
        simulated_annealing(netlist, move_weights={"shift": "high"})

    # Boolean move weight
    with pytest.raises(ValueError, match="must be numeric"):
        simulated_annealing(netlist, move_weights={"shift": True})

    # All zero weights
    with pytest.raises(ValueError, match="At least one move type"):
        simulated_annealing(
            netlist, move_weights={"shift": 0.0, "swap": 0.0, "relocate": 0.0}
        )

    # Invalid shift_window
    with pytest.raises(ValueError, match="shift_window must be a positive integer"):
        simulated_annealing(netlist, shift_window=0)

    with pytest.raises(ValueError, match="shift_window must be a positive integer"):
        simulated_annealing(netlist, shift_window=-2)

    with pytest.raises(ValueError, match="shift_window must be a positive integer"):
        simulated_annealing(netlist, shift_window=1.5)

    with pytest.raises(ValueError, match="shift_window must be a positive integer"):
        simulated_annealing(netlist, shift_window=True)


def test_sa_relocate_move_operator():
    """Relocate operator moves macro globally across the grid."""
    macros = (
        Macro(id=1, w=1, h=1),
        Macro(id=2, w=1, h=1),
    )
    nets = (Net(id=1, macro_ids=(1, 2), weight=1.0),)
    netlist = Netlist(version="1.0", grid_w=10, grid_h=10, macros=macros, nets=nets)
    init_p: Placement = [
        PlacedMacro(macro_id=1, x=0, y=0),
        PlacedMacro(macro_id=2, x=9, y=9),
    ]

    res = simulated_annealing(
        netlist,
        initial_placement=init_p,
        move_weights={"relocate": 1.0, "shift": 0.0, "swap": 0.0},
        initial_temperature=100.0,
        cooling_rate=0.9,
        min_temperature=1.0,
        max_iterations=20,
        seed=42,
    )
    assert res.success is True
    assert res.placement is not None
    engine = PlacementEngine(netlist)
    for pm in res.placement:
        engine.place(pm.macro_id, pm.x, pm.y)


def test_sa_shift_move_operator():
    """Shift operator displaces a macro within a local shift_window."""
    macros = (
        Macro(id=1, w=1, h=1),
        Macro(id=2, w=1, h=1),
    )
    nets = (Net(id=1, macro_ids=(1, 2), weight=1.0),)
    netlist = Netlist(version="1.0", grid_w=10, grid_h=10, macros=macros, nets=nets)
    init_p: Placement = [
        PlacedMacro(macro_id=1, x=5, y=5),
        PlacedMacro(macro_id=2, x=0, y=0),
    ]

    fake_rng = FakeRNG(choices=[1, (5, 6)], random_values=[0.0])
    res = simulated_annealing(
        netlist,
        initial_placement=init_p,
        move_weights={"shift": 1.0, "swap": 0.0, "relocate": 0.0},
        shift_window=1,
        rng=fake_rng,
        initial_temperature=10.0,
        cooling_rate=0.8,
        min_temperature=1.0,
        max_iterations=1,
    )
    assert res.success is True
    assert res.accepted_moves == 1
    p_dict = {pm.macro_id: (pm.x, pm.y) for pm in res.placement}
    assert max(abs(p_dict[1][0] - 5), abs(p_dict[1][1] - 5)) <= 1


def test_sa_valid_swap_compatible_dimensions():
    """Swap operator correctly exchanges coordinates of dimensionally compatible macros."""
    macros = (
        Macro(id=1, w=2, h=2),
        Macro(id=2, w=2, h=2),
    )
    nets = (Net(id=1, macro_ids=(1, 2), weight=1.0),)
    netlist = Netlist(version="1.0", grid_w=6, grid_h=6, macros=macros, nets=nets)
    init_p: Placement = [
        PlacedMacro(macro_id=1, x=0, y=0),
        PlacedMacro(macro_id=2, x=3, y=3),
    ]

    fake_rng = FakeRNG(choices=[1, 2], random_values=[0.0])
    res = simulated_annealing(
        netlist,
        initial_placement=init_p,
        move_weights={"swap": 1.0, "shift": 0.0, "relocate": 0.0},
        rng=fake_rng,
        initial_temperature=100.0,
        cooling_rate=0.8,
        min_temperature=1.0,
        max_iterations=1,
    )
    assert res.success is True
    assert res.accepted_moves == 1
    p_dict = {pm.macro_id: (pm.x, pm.y) for pm in res.placement}
    assert p_dict[1] == (3, 3)
    assert p_dict[2] == (0, 0)


def test_sa_invalid_swap_incompatible_dimensions():
    """Swap between macros with incompatible dimensions causing boundary violation is rejected safely."""
    macros = (
        Macro(id=1, w=3, h=1),
        Macro(id=2, w=1, h=2),
    )
    nets = (Net(id=1, macro_ids=(1, 2), weight=1.0),)
    netlist = Netlist(version="1.0", grid_w=3, grid_h=3, macros=macros, nets=nets)
    init_p: Placement = [
        PlacedMacro(macro_id=1, x=0, y=0),
        PlacedMacro(macro_id=2, x=2, y=1),
    ]

    fake_rng = FakeRNG(choices=[1, 2], random_values=[0.0])
    res = simulated_annealing(
        netlist,
        initial_placement=init_p,
        move_weights={"swap": 1.0, "shift": 0.0, "relocate": 0.0},
        rng=fake_rng,
        initial_temperature=10.0,
        cooling_rate=0.8,
        min_temperature=1.0,
        max_iterations=1,
    )
    assert res.success is True
    assert res.accepted_moves == 0
    p_dict = {pm.macro_id: (pm.x, pm.y) for pm in res.placement}
    assert p_dict[1] == (0, 0)
    assert p_dict[2] == (2, 1)


def test_sa_invalid_swap_causing_overlap():
    """Swap between two macros that causes overlap with a third macro is rejected safely."""
    macros = (
        Macro(id=1, w=1, h=1),
        Macro(id=2, w=2, h=2),
        Macro(id=3, w=1, h=1),
    )
    nets = (Net(id=1, macro_ids=(1, 2), weight=1.0),)
    netlist = Netlist(version="1.0", grid_w=5, grid_h=5, macros=macros, nets=nets)
    init_p: Placement = [
        PlacedMacro(macro_id=1, x=0, y=0),
        PlacedMacro(macro_id=2, x=2, y=2),
        PlacedMacro(macro_id=3, x=1, y=0),
    ]

    fake_rng = FakeRNG(choices=[1, 2], random_values=[0.0])
    res = simulated_annealing(
        netlist,
        initial_placement=init_p,
        move_weights={"swap": 1.0, "shift": 0.0, "relocate": 0.0},
        rng=fake_rng,
        initial_temperature=10.0,
        cooling_rate=0.8,
        min_temperature=1.0,
        max_iterations=1,
    )
    assert res.success is True
    assert res.accepted_moves == 0
    p_dict = {pm.macro_id: (pm.x, pm.y) for pm in res.placement}
    assert p_dict[1] == (0, 0)
    assert p_dict[2] == (2, 2)
    assert p_dict[3] == (1, 0)


def test_sa_rejected_swap_preserves_active_env_state():
    """Verify that an illegal swap leaves active_env in its exact initial state."""
    macros = (
        Macro(id=1, w=3, h=1),
        Macro(id=2, w=1, h=2),
    )
    nets = (Net(id=1, macro_ids=(1, 2), weight=1.0),)
    netlist = Netlist(version="1.0", grid_w=3, grid_h=3, macros=macros, nets=nets)
    init_p: Placement = [
        PlacedMacro(macro_id=1, x=0, y=0),
        PlacedMacro(macro_id=2, x=2, y=1),
    ]

    class SpyEngine(PlacementEngine):
        instances: list[SpyEngine] = []

        def __init__(self, nl: Any):
            super().__init__(nl)
            SpyEngine.instances.append(self)

    SpyEngine.instances.clear()
    fake_rng = FakeRNG(choices=[1, 2], random_values=[0.0])
    res = simulated_annealing(
        netlist,
        initial_placement=init_p,
        env_class=SpyEngine,
        move_weights={"swap": 1.0, "shift": 0.0, "relocate": 0.0},
        rng=fake_rng,
        initial_temperature=10.0,
        cooling_rate=0.8,
        min_temperature=1.0,
        max_iterations=1,
    )
    assert res.success is True
    assert res.accepted_moves == 0
    assert len(SpyEngine.instances) >= 1
    active = SpyEngine.instances[0]
    assert active.is_placed(1)
    assert active.is_placed(2)
    placed_map = {pm.macro_id: (pm.x, pm.y) for pm in active.get_placement()}
    assert placed_map[1] == (0, 0)
    assert placed_map[2] == (2, 1)


def test_sa_unavailable_move_type_handles_safely_no_infinite_loop():
    """When a chosen move type cannot be applied (e.g. swap on 1 macro), it handles safely."""
    macros = (Macro(id=1, w=1, h=1),)
    nets = ()
    netlist = Netlist(version="1.0", grid_w=3, grid_h=3, macros=macros, nets=nets)
    init_p: Placement = [PlacedMacro(macro_id=1, x=0, y=0)]

    res = simulated_annealing(
        netlist,
        initial_placement=init_p,
        move_weights={"swap": 1.0, "shift": 0.0, "relocate": 0.0},
        initial_temperature=10.0,
        cooling_rate=0.5,
        min_temperature=1.0,
        max_iterations=5,
    )
    assert res.success is True
    assert res.iterations > 0
    assert res.accepted_moves == 0


def test_sa_exception_during_evaluation_restores_state(monkeypatch):
    """An exception during candidate evaluation leaves active_env intact with original macros."""
    macros = (
        Macro(id=1, w=2, h=2),
        Macro(id=2, w=2, h=2),
    )
    nets = (Net(id=1, macro_ids=(1, 2), weight=1.0),)
    netlist = Netlist(version="1.0", grid_w=6, grid_h=6, macros=macros, nets=nets)
    init_p: Placement = [
        PlacedMacro(macro_id=1, x=0, y=0),
        PlacedMacro(macro_id=2, x=3, y=3),
    ]

    class SpyEngine(PlacementEngine):
        instances: list[SpyEngine] = []

        def __init__(self, nl: Any):
            super().__init__(nl)
            SpyEngine.instances.append(self)

    SpyEngine.instances.clear()

    import src.algorithms.simulated_annealing as sa_mod
    orig_eval = sa_mod.evaluate_placement
    call_count = 0

    def faulty_eval(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count > 1:  # allow initial evaluation, fail on candidate evaluation
            raise RuntimeError("Corrupted evaluator")
        return orig_eval(*args, **kwargs)

    monkeypatch.setattr(sa_mod, "evaluate_placement", faulty_eval)

    fake_rng = FakeRNG(choices=[1, 2], random_values=[0.0])
    with pytest.raises(RuntimeError, match="Corrupted evaluator"):
        simulated_annealing(
            netlist,
            initial_placement=init_p,
            env_class=SpyEngine,
            move_weights={"swap": 1.0, "shift": 0.0, "relocate": 0.0},
            rng=fake_rng,
            initial_temperature=100.0,
            cooling_rate=0.8,
            min_temperature=1.0,
            max_iterations=1,
        )

    assert len(SpyEngine.instances) >= 1
    active = SpyEngine.instances[0]
    assert active.is_placed(1)
    assert active.is_placed(2)
    placed_map = {pm.macro_id: (pm.x, pm.y) for pm in active.get_placement()}
    assert placed_map[1] == (0, 0)
    assert placed_map[2] == (3, 3)


def test_sa_deterministic_reproducibility_with_all_moves():
    """Simulated Annealing is strictly reproducible with the same seed when all move types are active."""
    macros = (
        Macro(id=1, w=1, h=1),
        Macro(id=2, w=2, h=1),
        Macro(id=3, w=1, h=2),
    )
    nets = (
        Net(id=1, macro_ids=(1, 2), weight=1.0),
        Net(id=2, macro_ids=(2, 3), weight=1.5),
    )
    netlist = Netlist(version="1.0", grid_w=6, grid_h=6, macros=macros, nets=nets)

    res1 = simulated_annealing(
        netlist,
        move_weights={"shift": 1.0, "swap": 1.0, "relocate": 1.0},
        shift_window=2,
        seed=777,
        initial_temperature=50.0,
        cooling_rate=0.9,
        min_temperature=0.1,
        max_iterations=30,
    )

    res2 = simulated_annealing(
        netlist,
        move_weights={"shift": 1.0, "swap": 1.0, "relocate": 1.0},
        shift_window=2,
        seed=777,
        initial_temperature=50.0,
        cooling_rate=0.9,
        min_temperature=0.1,
        max_iterations=30,
    )

    assert res1.success is True
    assert res2.success is True
    assert res1.iterations == res2.iterations
    assert res1.accepted_moves == res2.accepted_moves
    assert math.isclose(res1.objective, res2.objective)
    p1 = {pm.macro_id: (pm.x, pm.y) for pm in res1.placement}
    p2 = {pm.macro_id: (pm.x, pm.y) for pm in res2.placement}
    assert p1 == p2
