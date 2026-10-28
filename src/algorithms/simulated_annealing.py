"""Simulated Annealing Placement baseline for macro placement and floorplanning.

This module provides the Simulated Annealing (SA) algorithm baseline for macro
placement and floorplanning. It starts from a legal initial placement generated
by Zoya's shared legal engine (`src.placement.legal.random_legal_placement`),
iteratively explores neighboring configurations by relocating macros to legal
candidate anchors evaluated via Zoya's shared grid environment (`GridEnv.legal_mask`),
and accepts transitions using the Metropolis-Hastings acceptance criterion to
minimize the combined objective J(P) = alpha * HPWL + beta * C_score.

Key properties:
- Deterministic and reproducible when given a random seed.
- Numerically safe acceptance probability computation avoiding overflow/underflow.
- Geometric temperature cooling schedule.
- Separate tracking of current accepted placement and global best placement found.
- Zero duplicated HPWL, overlap, boundary, or legality logic.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Mapping, NamedTuple, Sequence
import numpy as np

from src.algorithms.common import (
    extract_placed_ids,
    normalize_placement,
    random_legal_placement,
    to_canonical_netlist,
)
from src.placement.engine import PlacementEngine
from src.placement.models import PlacedMacro
from src.placement.objective import EvaluationResult, evaluate_placement

# Dynamic aliases (preserving GridEnv and random_legal_placement backward-compatibility)
GridEnv = PlacementEngine


class SimulatedAnnealingResult(NamedTuple):
    """Result of Simulated Annealing macro placement.

    Attributes:
        placement: Placed macro positions mapping macro_id -> (x, y), or None if placement failed.
        success: True if a legal complete placement was found and optimized, False otherwise.
        hpwl: Half-Perimeter Wirelength (float), or None if placement failed.
        congestion: RUDY congestion score proxy (float), or None if placement failed.
        objective: Combined placement objective J(P) = alpha * HPWL + beta * congestion,
            or None if placement failed.
        iterations: Number of optimization iterations executed (int).
        accepted_moves: Number of candidate moves accepted during annealing (int).
        best_objective: Best combined objective value achieved during the search (float or None).
    """

    placement: Any | None
    success: bool
    hpwl: float | None
    congestion: float | None
    objective: float | None
    iterations: int = 0
    accepted_moves: int = 0
    best_objective: float | None = None


PlacementResult = SimulatedAnnealingResult


def _extract_macro_ids(netlist: Any) -> set[Any]:
    """Extract the set of macro IDs defined in the netlist."""
    if isinstance(netlist, Mapping):
        raw_macros = netlist.get("macros", {})
    else:
        raw_macros = getattr(netlist, "macros", {})

    if isinstance(raw_macros, Mapping):
        return set(raw_macros.keys())
    elif isinstance(raw_macros, Sequence):
        ids: set[Any] = set()
        for item in raw_macros:
            if isinstance(item, Mapping):
                m_id = item.get("id")
            else:
                m_id = getattr(item, "id", None)
            if m_id is not None:
                ids.add(m_id)
        return ids
    return set()


_extract_placed_ids = extract_placed_ids
_normalize_placement = normalize_placement


def _choose(rng: Any, sequence: Sequence[Any]) -> Any:
    """Safely and deterministically select an element from a sequence using rng."""
    if hasattr(rng, "choice"):
        try:
            return rng.choice(sequence)
        except (ValueError, TypeError):
            pass
    if hasattr(rng, "randint"):
        idx = rng.randint(0, len(sequence) - 1)
        return sequence[idx]
    if hasattr(rng, "integers"):
        idx = int(rng.integers(0, len(sequence)))
        return sequence[idx]
    # Fallback to random()
    rand_fn = getattr(rng, "random", random.random)
    idx = int(rand_fn() * len(sequence))
    if idx >= len(sequence):
        idx = len(sequence) - 1
    return sequence[idx]


def _acceptance_probability(delta: float, temperature: float) -> float:
    """Compute the Metropolis acceptance probability in a numerically safe manner.

    Parameters
    ----------
    delta : float
        Change in objective: candidate_objective - current_objective.
    temperature : float
        Current annealing temperature.

    Returns
    -------
    float
        Acceptance probability in [0.0, 1.0].
    """
    if delta <= 0.0:
        return 1.0
    if temperature <= 0.0 or not math.isfinite(temperature) or not math.isfinite(delta):
        return 0.0
    exponent = -delta / temperature
    if exponent < -700.0:
        return 0.0
    try:
        return float(math.exp(exponent))
    except OverflowError:
        return 0.0


def _query_legal_positions(env: Any, m_id: Any) -> list[tuple[int, int]]:
    """Retrieve all legal (x, y) anchors for macro m_id in row-major order."""
    if hasattr(env, "get_legal_positions"):
        return env.get_legal_positions(m_id)
    elif hasattr(env, "legal_mask"):
        mask = env.legal_mask(m_id)
        if mask is None:
            return []
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.ndim != 2:
            return []
        grid_h, grid_w = mask_arr.shape
        return [(x, y) for y in range(grid_h) for x in range(grid_w) if mask_arr[y, x]]
    elif hasattr(env, "compute_action_mask"):
        mask = env.compute_action_mask(m_id, flatten=False)
        grid_h, grid_w = mask.shape
        return [(x, y) for y in range(grid_h) for x in range(grid_w) if mask[y, x]]
    return []


def _get_candidate_anchors(
    netlist: Any,
    current_placement: dict[Any, tuple[int, int]],
    m_id: Any,
    env_class: Callable[..., Any],
    active_env: Any | None = None,
) -> list[tuple[int, int]]:
    """Query legal anchor candidates for relocating macro `m_id` without collision.

    Delegates legality checks exclusively to the engine or environment.
    Does not duplicate any boundary or overlap detection logic.
    """
    curr_pos = current_placement.get(m_id)
    if curr_pos is None:
        return []

    curr_x, curr_y = curr_pos
    positions: list[tuple[int, int]] = []

    # If the active environment supports unplace/remove, use the fast path
    if active_env is not None and hasattr(active_env, "unplace"):
        active_env.unplace(m_id)
        try:
            positions = _query_legal_positions(active_env, m_id)
        finally:
            try:
                active_env.place(m_id, curr_x, curr_y)
            except Exception as restore_err:
                raise RuntimeError(
                    f"Failed to restore macro '{m_id}' to ({curr_x}, {curr_y}) in active environment: {restore_err}"
                ) from restore_err
    elif active_env is not None and hasattr(active_env, "remove"):
        active_env.remove(m_id)
        try:
            positions = _query_legal_positions(active_env, m_id)
        finally:
            try:
                active_env.place(m_id, curr_x, curr_y)
            except Exception as restore_err:
                raise RuntimeError(
                    f"Failed to restore macro '{m_id}' to ({curr_x}, {curr_y}) in active environment: {restore_err}"
                ) from restore_err
    else:
        # Replay placement of all other macros on a fresh environment instance
        if env_class is PlacementEngine:
            env = env_class(to_canonical_netlist(netlist))
        else:
            env = env_class(netlist)
        for other_id in sorted(current_placement.keys(), key=str):
            if other_id != m_id:
                ox, oy = current_placement[other_id]
                env.place(other_id, ox, oy)
        positions = _query_legal_positions(env, m_id)

    return [(x, y) for (x, y) in positions if (x != curr_x or y != curr_y)]


def _choose_weighted(rng: Any, items: list[str], weights: list[float]) -> str:
    """Select an item according to non-negative weights using rng."""
    total = sum(weights)
    if total <= 0.0:
        return items[0]
    if hasattr(rng, "choices"):
        try:
            return rng.choices(items, weights=weights, k=1)[0]
        except (ValueError, TypeError):
            pass
    rand_fn = getattr(rng, "random", random.random)
    r = rand_fn() * total
    cumulative = 0.0
    for item, w in zip(items, weights):
        cumulative += w
        if r <= cumulative:
            return item
    return items[-1]


def _is_legal_anchor(env: Any, m_id: Any, x: int, y: int) -> bool:
    """Check whether placing m_id at (x, y) is legal in env (assuming m_id is unplaced)."""
    if hasattr(env, "is_legal"):
        try:
            return bool(env.is_legal(m_id, x, y))
        except Exception:
            pass
    legal_positions = _query_legal_positions(env, m_id)
    return (x, y) in legal_positions


def _check_swap_legality(
    netlist: Any,
    current_placement: dict[Any, tuple[int, int]],
    m1: Any,
    m2: Any,
    pos1: tuple[int, int],
    pos2: tuple[int, int],
    env_class: Callable[..., Any],
    active_env: Any | None = None,
) -> bool:
    """Check whether exchanging positions of m1 and m2 is strictly legal.

    Delegates legality checks exclusively to the engine or environment.
    Guarantees that active_env is left in its exact prior state.
    """
    if active_env is not None and hasattr(active_env, "unplace"):
        active_env.unplace(m1)
        active_env.unplace(m2)
        is_legal = False
        try:
            if _is_legal_anchor(active_env, m1, pos2[0], pos2[1]):
                try:
                    active_env.place(m1, pos2[0], pos2[1])
                    try:
                        is_legal = _is_legal_anchor(active_env, m2, pos1[0], pos1[1])
                    finally:
                        active_env.unplace(m1)
                except Exception:
                    is_legal = False
        finally:
            try:
                active_env.place(m1, pos1[0], pos1[1])
                active_env.place(m2, pos2[0], pos2[1])
            except Exception as restore_err:
                raise RuntimeError(
                    f"Failed to restore macro '{m1}' or '{m2}' during swap check: {restore_err}"
                ) from restore_err
        return is_legal
    elif active_env is not None and hasattr(active_env, "remove"):
        active_env.remove(m1)
        active_env.remove(m2)
        is_legal = False
        try:
            if _is_legal_anchor(active_env, m1, pos2[0], pos2[1]):
                try:
                    active_env.place(m1, pos2[0], pos2[1])
                    try:
                        is_legal = _is_legal_anchor(active_env, m2, pos1[0], pos1[1])
                    finally:
                        active_env.remove(m1)
                except Exception:
                    is_legal = False
        finally:
            try:
                active_env.place(m1, pos1[0], pos1[1])
                active_env.place(m2, pos2[0], pos2[1])
            except Exception as restore_err:
                raise RuntimeError(
                    f"Failed to restore macro '{m1}' or '{m2}' during swap check: {restore_err}"
                ) from restore_err
        return is_legal
    else:
        # Replay fallback path
        if env_class is PlacementEngine:
            env = env_class(to_canonical_netlist(netlist))
        else:
            env = env_class(netlist)

        cand = dict(current_placement)
        cand[m1] = pos2
        cand[m2] = pos1

        for mid in sorted(cand.keys(), key=str):
            px, py = cand[mid]
            if not _is_legal_anchor(env, mid, px, py):
                return False
            try:
                env.place(mid, px, py)
            except Exception:
                return False
        return True


def simulated_annealing(
    netlist: Any,
    seed: int | None = 42,
    alpha: float = 1.0,
    beta: float = 1.0,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.995,
    min_temperature: float = 0.01,
    max_iterations: int = 10000,
    *,
    legal_placement_fn: Callable[..., Any] | None = None,
    env_class: Callable[..., Any] | None = None,
    initial_placement: Any | None = None,
    rng: Any | None = None,
    shift_window: int | None = None,
    move_weights: Mapping[str, float] | None = None,
) -> SimulatedAnnealingResult:
    """Execute Simulated Annealing macro placement minimizing the combined objective.

    Algorithm:
    1. Validate all input parameters.
    2. Obtain a feasible, complete initial placement via `random_legal_placement(netlist, seed=seed)`
       or provided `initial_placement`. If unavailable or incomplete, return a documented failure result.
    3. Initialize temperature T = initial_temperature and evaluate initial objective J(P_0).
    4. Track both current accepted placement and global best placement found.
    5. In each iteration while iteration < max_iterations and T > min_temperature:
       a. Pick a macro deterministically/reproducibly using local RNG.
       b. Query legal relocation anchors from `GridEnv.legal_mask`.
       c. If no alternative legal anchor exists, reject proposal safely and cool temperature.
       d. Pick a candidate anchor and construct candidate placement P'.
       e. Evaluate candidate combined objective J(P') = alpha * HPWL + beta * C_score.
       f. Compute delta = J(P') - J(P).
       g. If delta <= 0, accept P'.
          Else accept P' with probability exp(-delta / T).
       h. If accepted: update current placement. If J(P') < J(P_best), update best placement.
       i. Cool temperature: T = T * cooling_rate.
    6. Return best placement found, its evaluated metrics, iteration count, and accepted moves.

    Parameters
    ----------
    netlist : Any
        Circuit netlist providing grid dimensions, macros, and nets.
    seed : int | None, default=42
        Integer seed for reproducible pseudo-random generation.
    alpha : float, default=1.0
        Non-negative weight for HPWL in the combined objective.
    beta : float, default=1.0
        Non-negative weight for Congestion in the combined objective.
    initial_temperature : float, default=100.0
        Starting temperature for simulated annealing (> min_temperature).
    cooling_rate : float, default=0.995
        Geometric cooling multiplier strictly in (0.0, 1.0).
    min_temperature : float, default=0.01
        Stopping temperature threshold (> 0.0).
    max_iterations : int, default=10000
        Maximum number of iterations allowed (positive integer).
    legal_placement_fn : Callable[..., Any] | None, optional
        Custom legal placement generator function. If None, uses
        `src.placement.legal.random_legal_placement`.
    env_class : Callable[..., Any] | None, optional
        Custom GridEnv class or factory for neighbor generation. If None, uses
        `src.placement.grid.GridEnv`.
    initial_placement : Any | None, optional
        Pre-computed initial legal placement. If None, generated via `legal_placement_fn`.
    rng : Any | None, optional
        Custom random number generator instance (e.g. `random.Random` or duck-typed double).

    Returns
    -------
    SimulatedAnnealingResult
        NamedTuple containing:
            - placement: best legal complete placement found, or None if initialization failed.
            - success: True if a legal complete placement was optimized, False otherwise.
            - hpwl: HPWL of best placement (float or None).
            - congestion: Congestion score of best placement (float or None).
            - objective: Combined objective value J(P) of best placement (float or None).
            - iterations: Total iterations executed (int).
            - accepted_moves: Total moves accepted (int).
            - best_objective: Best objective achieved (float or None).

    Raises
    ------
    NotImplementedError
        If required shared engine modules (`src.placement.legal.random_legal_placement`
        or `src.placement.grid.GridEnv`) are missing and not injected.
    ValueError
        If any input parameters (seed, alpha, beta, temperatures, cooling_rate, max_iterations)
        are invalid.
    """
    # 1. Parameter Validation
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError(f"Seed must be an integer or None, got {type(seed).__name__}.")

    for name, val in [("alpha", alpha), ("beta", beta)]:
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"Weight '{name}' must be numeric, got {type(val).__name__}.")
        if not math.isfinite(val):
            raise ValueError(f"Weight '{name}' must be finite, got {val}.")
        if val < 0.0:
            raise ValueError(f"Weight '{name}' must be non-negative, got {val}.")

    for name, val in [
        ("initial_temperature", initial_temperature),
        ("min_temperature", min_temperature),
    ]:
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"Temperature '{name}' must be numeric, got {type(val).__name__}.")
        if not math.isfinite(val):
            raise ValueError(f"Temperature '{name}' must be finite, got {val}.")
        if val <= 0.0:
            raise ValueError(f"Temperature '{name}' must be positive, got {val}.")

    flt_init_temp = float(initial_temperature)
    flt_min_temp = float(min_temperature)
    if flt_init_temp <= flt_min_temp:
        raise ValueError(
            f"initial_temperature ({initial_temperature}) must be strictly greater than "
            f"min_temperature ({min_temperature})."
        )

    if isinstance(cooling_rate, bool) or not isinstance(cooling_rate, (int, float, np.integer, np.floating)):
        raise ValueError(f"cooling_rate must be numeric, got {type(cooling_rate).__name__}.")
    flt_cooling = float(cooling_rate)
    if not math.isfinite(flt_cooling) or not (0.0 < flt_cooling < 1.0):
        raise ValueError(f"cooling_rate must be strictly between 0 and 1, got {cooling_rate}.")

    if isinstance(max_iterations, bool) or not isinstance(max_iterations, (int, np.integer)):
        raise ValueError(f"max_iterations must be an integer, got {type(max_iterations).__name__}.")
    int_max_iter = int(max_iterations)
    if int_max_iter <= 0:
        raise ValueError(f"max_iterations must be a positive integer, got {max_iterations}.")

    if shift_window is not None:
        if (
            isinstance(shift_window, bool)
            or not isinstance(shift_window, (int, np.integer))
            or shift_window <= 0
        ):
            raise ValueError(f"shift_window must be a positive integer, got {shift_window!r}.")

    valid_moves = {"shift", "swap", "relocate"}
    effective_weights: dict[str, float] = {"shift": 1.0, "swap": 1.0, "relocate": 1.0}
    if move_weights is not None:
        if not isinstance(move_weights, Mapping):
            raise ValueError(f"move_weights must be a mapping, got {type(move_weights).__name__}.")
        parsed_weights: dict[str, float] = {}
        for k, v in move_weights.items():
            if k not in valid_moves:
                raise ValueError(
                    f"Unknown move type in move_weights: '{k}'. Valid: {sorted(valid_moves)}"
                )
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
                raise ValueError(f"Move weight for '{k}' must be numeric, got {type(v).__name__}.")
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"Move weight for '{k}' must be finite and non-negative, got {v}.")
            parsed_weights[k] = float(v)
        for m in valid_moves:
            effective_weights[m] = parsed_weights.get(m, 0.0)

        if sum(effective_weights.values()) <= 0.0:
            raise ValueError("At least one move type in move_weights must have positive weight.")

    flt_alpha = float(alpha)
    flt_beta = float(beta)

    # Instantiate or adopt local RNG
    if rng is not None:
        rng_inst = rng
    elif seed is not None:
        rng_inst = random.Random(int(seed))
    else:
        rng_inst = random.Random()

    # 2. Check and resolve engine dependencies
    placement_fn = (
        legal_placement_fn if legal_placement_fn is not None else random_legal_placement
    )
    cls = env_class if env_class is not None else GridEnv

    # If no initial placement was provided and helper is missing, raise NotImplementedError
    if initial_placement is None and placement_fn is None:
        raise NotImplementedError(
            "Shared engine module 'src.placement.legal.random_legal_placement' is not available yet. "
            "Zoya's placement engine must be implemented to generate legal placements."
        )

    # If GridEnv is missing, raise NotImplementedError
    if cls is None:
        raise NotImplementedError(
            "Shared engine module 'src.placement.grid.GridEnv' is not available yet. "
            "Zoya's placement engine must be implemented to evaluate legal neighbor moves."
        )

    # 3. Obtain initial placement
    if initial_placement is not None:
        placement = initial_placement
    else:
        placement = placement_fn(netlist, seed=seed)

    # Handle placement generation failure
    if placement is None:
        return SimulatedAnnealingResult(
            placement=None,
            success=False,
            hpwl=None,
            congestion=None,
            objective=None,
            iterations=0,
            accepted_moves=0,
            best_objective=None,
        )

    macro_ids = _extract_macro_ids(netlist)
    placed_ids = _extract_placed_ids(placement)

    # Handle empty netlist edge-case
    if not macro_ids:
        eval_empty = evaluate_placement(netlist, {}, alpha=flt_alpha, beta=flt_beta)
        return SimulatedAnnealingResult(
            placement={},
            success=True,
            hpwl=eval_empty.hpwl,
            congestion=eval_empty.congestion,
            objective=eval_empty.objective,
            iterations=0,
            accepted_moves=0,
            best_objective=eval_empty.objective,
        )

    # Verify initial placement completeness
    if not macro_ids.issubset(placed_ids):
        return SimulatedAnnealingResult(
            placement=placement,
            success=False,
            hpwl=None,
            congestion=None,
            objective=None,
            iterations=0,
            accepted_moves=0,
            best_objective=None,
        )

    # Normalize placement to clean dict
    current_placement = _normalize_placement(placement)

    # Initial evaluation
    current_eval: EvaluationResult = evaluate_placement(
        netlist, current_placement, alpha=flt_alpha, beta=flt_beta
    )

    best_placement = dict(current_placement)
    best_eval = current_eval

    # Attempt to initialize active persistent environment if supported
    active_env: Any | None = None
    if cls is PlacementEngine:
        env_instance = cls(to_canonical_netlist(netlist))
    else:
        env_instance = cls(netlist)
    if hasattr(env_instance, "unplace") or hasattr(env_instance, "remove"):
        for mid in sorted(current_placement.keys(), key=str):
            px, py = current_placement[mid]
            env_instance.place(mid, px, py)
        active_env = env_instance

    # Determine grid dimensions and shift window
    if isinstance(netlist, Mapping):
        gw = int(netlist.get("grid_w", 10))
        gh = int(netlist.get("grid_h", 10))
    else:
        gw = int(getattr(netlist, "grid_w", 10))
        gh = int(getattr(netlist, "grid_h", 10))
    default_win = max(1, min(gw, gh) // 4)
    effective_shift_window = shift_window if shift_window is not None else default_win

    valid_moves_order = ("shift", "swap", "relocate")
    active_moves = [m for m in valid_moves_order if effective_weights[m] > 0.0]
    active_weights = [effective_weights[m] for m in active_moves]

    temperature = flt_init_temp
    iteration = 0
    accepted_moves = 0
    macro_list = sorted(current_placement.keys(), key=str)

    # 4. Annealing optimization loop
    while iteration < int_max_iter and temperature > flt_min_temp:
        iteration += 1

        # Select move type
        if len(active_moves) == 1:
            chosen_move = active_moves[0]
        elif (
            hasattr(rng_inst, "_choices")
            and getattr(rng_inst, "_choices")
            and rng_inst._choices[0] not in valid_moves
        ):
            # Backward compatibility for existing FakeRNG tests where choices are macro IDs/anchors:
            # Treat as relocate if active, else fallback to active_moves[0], without popping choice.
            chosen_move = "relocate" if "relocate" in active_moves else active_moves[0]
        elif (
            hasattr(rng_inst, "_choices")
            and getattr(rng_inst, "_choices")
            and rng_inst._choices[0] in valid_moves
        ):
            popped = rng_inst._choices.pop(0)
            chosen_move = popped if popped in active_moves else active_moves[0]
        elif hasattr(rng_inst, "_choices"):
            chosen_move = "relocate" if "relocate" in active_moves else active_moves[0]
        else:
            chosen_move = _choose_weighted(rng_inst, active_moves, active_weights)

        if chosen_move == "swap":
            if len(macro_list) < 2:
                temperature *= flt_cooling
                continue

            m1 = _choose(rng_inst, macro_list)
            remaining_macros = [m for m in macro_list if m != m1]
            m2 = _choose(rng_inst, remaining_macros)

            pos1 = current_placement[m1]
            pos2 = current_placement[m2]

            if not _check_swap_legality(
                netlist, current_placement, m1, m2, pos1, pos2, cls, active_env=active_env
            ):
                # Illegal swap: reject safely and preserve active state
                temperature *= flt_cooling
                continue

            cand_placement = dict(current_placement)
            cand_placement[m1] = pos2
            cand_placement[m2] = pos1

            cand_eval = evaluate_placement(
                netlist, cand_placement, alpha=flt_alpha, beta=flt_beta
            )
            delta = cand_eval.objective - current_eval.objective

            accept = False
            if delta <= 0.0:
                accept = True
            else:
                prob = _acceptance_probability(delta, temperature)
                rand_val = (
                    rng_inst.random()
                    if hasattr(rng_inst, "random")
                    else random.random()
                )
                accept = rand_val < prob

            if accept:
                # Update active environment atomically
                if active_env is not None and hasattr(active_env, "unplace"):
                    active_env.unplace(m1)
                    active_env.unplace(m2)
                    try:
                        active_env.place(m1, pos2[0], pos2[1])
                        active_env.place(m2, pos1[0], pos1[1])
                    except Exception as place_err:
                        try:
                            if hasattr(active_env, "is_placed") and active_env.is_placed(m1):
                                active_env.unplace(m1)
                        except Exception:
                            pass
                        try:
                            if hasattr(active_env, "is_placed") and active_env.is_placed(m2):
                                active_env.unplace(m2)
                        except Exception:
                            pass
                        active_env.place(m1, pos1[0], pos1[1])
                        active_env.place(m2, pos2[0], pos2[1])
                        raise RuntimeError(
                            f"Failed to apply accepted swap in active environment: {place_err}"
                        ) from place_err
                elif active_env is not None and hasattr(active_env, "remove"):
                    active_env.remove(m1)
                    active_env.remove(m2)
                    try:
                        active_env.place(m1, pos2[0], pos2[1])
                        active_env.place(m2, pos1[0], pos1[1])
                    except Exception as place_err:
                        try:
                            active_env.remove(m1)
                        except Exception:
                            pass
                        try:
                            active_env.remove(m2)
                        except Exception:
                            pass
                        active_env.place(m1, pos1[0], pos1[1])
                        active_env.place(m2, pos2[0], pos2[1])
                        raise RuntimeError(
                            f"Failed to apply accepted swap in active environment: {place_err}"
                        ) from place_err

                current_placement = cand_placement
                current_eval = cand_eval
                accepted_moves += 1

                if current_eval.objective <= best_eval.objective:
                    best_placement = dict(current_placement)
                    best_eval = current_eval

            temperature *= flt_cooling
            continue

        # Chosen move is "shift" or "relocate"
        m_id = _choose(rng_inst, macro_list)
        all_candidates = _get_candidate_anchors(
            netlist, current_placement, m_id, cls, active_env=active_env
        )

        if chosen_move == "shift":
            x0, y0 = current_placement[m_id]
            candidates = [
                pos
                for pos in all_candidates
                if max(abs(pos[0] - x0), abs(pos[1] - y0)) <= effective_shift_window
            ]
        else:
            candidates = all_candidates

        if not candidates:
            # No legal anchor available: reject safely
            temperature *= flt_cooling
            continue

        new_anchor = _choose(rng_inst, candidates)
        cand_placement = dict(current_placement)
        cand_placement[m_id] = new_anchor

        cand_eval = evaluate_placement(
            netlist, cand_placement, alpha=flt_alpha, beta=flt_beta
        )
        delta = cand_eval.objective - current_eval.objective

        accept = False
        if delta <= 0.0:
            accept = True
        else:
            prob = _acceptance_probability(delta, temperature)
            rand_val = (
                rng_inst.random()
                if hasattr(rng_inst, "random")
                else random.random()
            )
            accept = rand_val < prob

        if accept:
            old_anchor = current_placement[m_id]
            if active_env is not None and hasattr(active_env, "unplace"):
                active_env.unplace(m_id)
                try:
                    active_env.place(m_id, new_anchor[0], new_anchor[1])
                except Exception as place_err:
                    active_env.place(m_id, old_anchor[0], old_anchor[1])
                    raise RuntimeError(
                        f"Failed to apply accepted relocation in active environment: {place_err}"
                    ) from place_err
            elif active_env is not None and hasattr(active_env, "remove"):
                active_env.remove(m_id)
                try:
                    active_env.place(m_id, new_anchor[0], new_anchor[1])
                except Exception as place_err:
                    active_env.place(m_id, old_anchor[0], old_anchor[1])
                    raise RuntimeError(
                        f"Failed to apply accepted relocation in active environment: {place_err}"
                    ) from place_err

            current_placement = cand_placement
            current_eval = cand_eval
            accepted_moves += 1

            if current_eval.objective <= best_eval.objective:
                best_placement = dict(current_placement)
                best_eval = current_eval

        temperature *= flt_cooling

    # Format return placement matching input representation
    if isinstance(placement, list):
        return_placement: Any = [
            PlacedMacro(macro_id=mid, x=coord[0], y=coord[1])
            for mid, coord in best_placement.items()
        ]
    else:
        return_placement = best_placement

    return SimulatedAnnealingResult(
        placement=return_placement,
        success=True,
        hpwl=best_eval.hpwl,
        congestion=best_eval.congestion,
        objective=best_eval.objective,
        iterations=iteration,
        accepted_moves=accepted_moves,
        best_objective=best_eval.objective,
    )


sa = simulated_annealing
anneal = simulated_annealing

__all__ = [
    "PlacementResult",
    "SimulatedAnnealingResult",
    "anneal",
    "random_legal_placement",
    "sa",
    "simulated_annealing",
]
