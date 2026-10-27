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

from src.placement.objective import EvaluationResult, evaluate_placement

# Dynamic imports for Zoya's shared engine modules once implemented
try:
    from src.placement.legal import random_legal_placement  # type: ignore[import-untyped]
except ImportError:
    random_legal_placement = None

try:
    from src.placement.grid import GridEnv  # type: ignore[import-untyped]
except ImportError:
    GridEnv = None


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


def _extract_placed_ids(placement: Any) -> set[Any]:
    """Extract the set of placed macro IDs from placement object or mapping."""
    if placement is None:
        return set()
    if hasattr(placement, "positions"):
        raw_pos = placement.positions
    elif isinstance(placement, Mapping):
        raw_pos = placement
    else:
        try:
            raw_pos = {k: placement[k] for k in placement}
        except (TypeError, AttributeError):
            raw_pos = {}

    if isinstance(raw_pos, Mapping):
        return {k for k, v in raw_pos.items() if v is not None}
    return set()


def _normalize_placement(placement: Any) -> dict[Any, tuple[int, int]]:
    """Convert any placement representation into a dictionary of macro_id -> (int(x), int(y)).

    Supports:
        1. Tuple or list: (x, y)
        2. Mapping: {"x": x, "y": y}
        3. Object with .x and .y attributes (e.g. PlacedMacro)

    Raises
    ------
    ValueError
        If coordinates for any placed macro are missing, non-numeric, non-finite,
        or non-integer.
    """
    if placement is None:
        return {}
    if hasattr(placement, "positions"):
        raw_pos = placement.positions
    elif isinstance(placement, Mapping):
        raw_pos = placement
    else:
        try:
            raw_pos = {k: placement[k] for k in placement}
        except (TypeError, AttributeError):
            raw_pos = {}

    norm: dict[Any, tuple[int, int]] = {}
    if isinstance(raw_pos, Mapping):
        for m_id, coord in raw_pos.items():
            if coord is None:
                continue

            if isinstance(coord, (tuple, list)):
                if len(coord) != 2:
                    raise ValueError(
                        f"Placement for macro '{m_id}' tuple/list must have length 2, got {len(coord)}."
                    )
                x, y = coord[0], coord[1]
            elif isinstance(coord, Mapping):
                x = coord.get("x")
                y = coord.get("y")
                if x is None or y is None:
                    raise ValueError(
                        f"Placement mapping for macro '{m_id}' must provide 'x' and 'y', got {coord}."
                    )
            elif hasattr(coord, "x") and hasattr(coord, "y"):
                x = getattr(coord, "x")
                y = getattr(coord, "y")
            else:
                raise ValueError(
                    f"Unsupported coordinate format for macro '{m_id}': {coord!r}."
                )

            if x is None or y is None:
                raise ValueError(
                    f"Placement for macro '{m_id}' must provide x and y coordinates."
                )

            if (
                isinstance(x, bool)
                or isinstance(y, bool)
                or not isinstance(x, (int, float, np.integer, np.floating))
                or not isinstance(y, (int, float, np.integer, np.floating))
            ):
                raise ValueError(
                    f"Macro '{m_id}' coordinates must be numeric, got x={type(x).__name__}, y={type(y).__name__}."
                )

            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(
                    f"Macro '{m_id}' coordinates must be finite, got x={x}, y={y}."
                )

            int_x = int(x)
            int_y = int(y)

            if int_x != x or int_y != y:
                raise ValueError(
                    f"Macro '{m_id}' coordinates must be integer grid coordinates, got ({x}, {y})."
                )

            norm[m_id] = (int_x, int_y)

    return norm


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


def _get_candidate_anchors(
    netlist: Any,
    current_placement: dict[Any, tuple[int, int]],
    m_id: Any,
    env_class: Callable[..., Any],
    active_env: Any | None = None,
) -> list[tuple[int, int]]:
    """Query legal anchor candidates for relocating macro `m_id` without collision.

    Delegates legality checks exclusively to Zoya's shared `GridEnv.legal_mask`.
    Does not duplicate any boundary or overlap detection logic.
    """
    curr_pos = current_placement.get(m_id)
    if curr_pos is None:
        return []

    curr_x, curr_y = curr_pos

    mask: Any = None
    # If the active environment supports unplace/remove, use the fast path
    if active_env is not None and hasattr(active_env, "unplace"):
        active_env.unplace(m_id)
        try:
            mask = active_env.legal_mask(m_id)
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
            mask = active_env.legal_mask(m_id)
        finally:
            try:
                active_env.place(m_id, curr_x, curr_y)
            except Exception as restore_err:
                raise RuntimeError(
                    f"Failed to restore macro '{m_id}' to ({curr_x}, {curr_y}) in active environment: {restore_err}"
                ) from restore_err
    else:
        # Replay placement of all other macros on a fresh environment instance
        env = env_class(netlist)
        for other_id in sorted(current_placement.keys(), key=str):
            if other_id != m_id:
                ox, oy = current_placement[other_id]
                env.place(other_id, ox, oy)
        mask = env.legal_mask(m_id)

    if mask is None:
        return []

    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.ndim != 2:
        return []

    grid_h, grid_w = mask_arr.shape
    candidates: list[tuple[int, int]] = []
    # Enumerate legal anchors in deterministic row-major order (y then x)
    for y in range(grid_h):
        for x in range(grid_w):
            if mask_arr[y, x] and (x != curr_x or y != curr_y):
                candidates.append((x, y))

    return candidates


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
    env_instance = cls(netlist)
    if hasattr(env_instance, "unplace") or hasattr(env_instance, "remove"):
        for mid in sorted(current_placement.keys(), key=str):
            px, py = current_placement[mid]
            env_instance.place(mid, px, py)
        active_env = env_instance

    temperature = flt_init_temp
    iteration = 0
    accepted_moves = 0
    macro_list = sorted(current_placement.keys(), key=str)

    # 4. Annealing optimization loop
    while iteration < int_max_iter and temperature > flt_min_temp:
        iteration += 1

        # Pick candidate macro to relocate
        m_id = _choose(rng_inst, macro_list)

        # Query legal candidate anchors
        candidates = _get_candidate_anchors(
            netlist, current_placement, m_id, cls, active_env=active_env
        )

        if not candidates:
            # No legal relocation anchor available: reject unchanged safely
            temperature *= flt_cooling
            continue

        # Choose candidate anchor
        new_anchor = _choose(rng_inst, candidates)
        cand_placement = dict(current_placement)
        cand_placement[m_id] = new_anchor

        # Evaluate candidate placement
        cand_eval = evaluate_placement(
            netlist, cand_placement, alpha=flt_alpha, beta=flt_beta
        )
        delta = cand_eval.objective - current_eval.objective

        # Acceptance check
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
            # Apply transition
            current_placement = cand_placement
            current_eval = cand_eval
            accepted_moves += 1

            # Update active environment if present
            if active_env is not None and hasattr(active_env, "unplace"):
                active_env.unplace(m_id)
                active_env.place(m_id, new_anchor[0], new_anchor[1])
            elif active_env is not None and hasattr(active_env, "remove"):
                active_env.remove(m_id)
                active_env.place(m_id, new_anchor[0], new_anchor[1])

            # Update global best placement
            if current_eval.objective <= best_eval.objective:
                best_placement = dict(current_placement)
                best_eval = current_eval

        # Cool temperature
        temperature *= flt_cooling

    return SimulatedAnnealingResult(
        placement=best_placement,
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
    "sa",
    "simulated_annealing",
]
