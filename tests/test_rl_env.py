"""Unit and integration tests for MacroPlacementEnv and RL integration."""

import asyncio
import functools
from typing import Any
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.placement.benchmarks import (
    execute_single_benchmark_run,
    get_default_benchmark_suite,
)
from src.placement.constraints import validate_placement
from src.placement.engine import action_to_coord, coord_to_action
from src.placement.jobs import JobManager, get_registered_algorithms
from src.placement.models import Macro, Net, Netlist
from src.placement.objective import evaluate_placement
from src.rl.env import MacroPlacementEnv
from src.rl.observation import (
    build_connectivity_guidance_map,
    build_grid_observation,
    build_macro_features,
)
from src.rl.runner import rl_placement_runner


def async_test(func: Any) -> Any:
    """Helper decorator running coroutine tests via asyncio.run."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(func(*args, **kwargs))
    return wrapper


@pytest.fixture
def toy_netlist() -> Netlist:
    """Standard 4-macro toy benchmark netlist."""
    suite = get_default_benchmark_suite()
    return suite["toy_4"]


@pytest.fixture
def medium_netlist() -> Netlist:
    """Standard 8-macro medium benchmark netlist."""
    suite = get_default_benchmark_suite()
    return suite["medium_8"]


def test_gymnasium_env_compliance(toy_netlist: Netlist) -> None:
    """Verify MacroPlacementEnv passes Gymnasium's official env checker."""
    env = MacroPlacementEnv(netlist=toy_netlist, render_mode="ansi")
    # check_env verifies space properties, reset/step return types, and seeded reproducibility
    check_env(env.unwrapped, skip_render_check=False)


def test_observation_and_action_spaces(toy_netlist: Netlist) -> None:
    """Verify tensor shapes and space boundaries."""
    env = MacroPlacementEnv(netlist=toy_netlist)
    obs, info = env.reset(seed=42)

    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == toy_netlist.grid_w * toy_netlist.grid_h
    assert obs["grid"].shape == (4, toy_netlist.grid_h, toy_netlist.grid_w)
    assert obs["grid"].dtype == np.float32
    assert obs["macro_features"].shape == (6,)
    assert obs["macro_features"].dtype == np.float32
    assert obs["action_mask"].shape == (toy_netlist.grid_w * toy_netlist.grid_h,)
    assert obs["action_mask"].dtype == np.int8

    assert env.observation_space.contains(obs)
    assert info["step"] == 0
    assert info["total_steps"] == 4
    assert not info["is_complete"]
    assert info["is_legal"]


def test_action_mask_correctness(toy_netlist: Netlist) -> None:
    """Ensure every True/1 in action_mask is verified legal by PlacementEngine."""
    env = MacroPlacementEnv(netlist=toy_netlist)
    env.reset(seed=123)

    curr_macro = env.current_macro
    assert curr_macro is not None

    mask = env.action_masks()
    grid_w = env.grid_w

    for action_idx in range(len(mask)):
        x, y = action_to_coord(action_idx, grid_w)
        is_legal_engine = env.engine.is_legal(curr_macro.id, x, y)
        assert bool(mask[action_idx]) == is_legal_engine


def test_potential_reward_conservation(toy_netlist: Netlist) -> None:
    """Verify step rewards sum to -J(P) + completion_bonus over complete episode."""
    completion_bonus = 15.0
    alpha = 1.0
    beta = 1.0
    env = MacroPlacementEnv(
        netlist=toy_netlist,
        alpha=alpha,
        beta=beta,
        completion_bonus=completion_bonus,
    )
    _obs, info = env.reset(seed=100)

    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        mask = env.action_masks()
        legal_actions = np.flatnonzero(mask)
        assert len(legal_actions) > 0
        action = int(legal_actions[0])
        _obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    assert terminated
    assert not truncated
    assert info["is_complete"]

    final_placement = env.get_placement()
    eval_res = evaluate_placement(toy_netlist, final_placement, alpha, beta)
    expected_return = -eval_res.objective + completion_bonus

    assert pytest.approx(total_reward, rel=1e-5) == expected_return


def test_illegal_action_penalty_and_termination(toy_netlist: Netlist) -> None:
    """An illegal action must emit penalty and terminate immediately."""
    penalty = 75.0
    env = MacroPlacementEnv(netlist=toy_netlist, illegal_action_penalty=penalty)
    env.reset(seed=42)

    # Choose an out-of-bounds action (bottom right corner where macro won't fit)
    illegal_action = int(env.grid_w * env.grid_h) - 1
    mask = env.action_masks()
    assert mask[illegal_action] == 0

    _obs, reward, terminated, truncated, info = env.step(illegal_action)
    assert reward == -penalty
    assert terminated
    assert not truncated
    assert not info["is_legal"]
    assert info["failure_type"] == "illegal_action"


def test_deterministic_seeding_and_replay(toy_netlist: Netlist) -> None:
    """Identical seeds and actions must yield bit-exact trajectories."""
    env1 = MacroPlacementEnv(netlist=toy_netlist)
    env2 = MacroPlacementEnv(netlist=toy_netlist)

    obs1, _info1 = env1.reset(seed=999)
    obs2, _info2 = env2.reset(seed=999)

    np.testing.assert_array_equal(obs1["grid"], obs2["grid"])
    np.testing.assert_array_equal(obs1["macro_features"], obs2["macro_features"])
    np.testing.assert_array_equal(obs1["action_mask"], obs2["action_mask"])

    # Step identically
    for _ in range(len(toy_netlist.macros)):
        mask1 = env1.action_masks()
        legal_actions = np.flatnonzero(mask1)
        action = int(legal_actions[len(legal_actions) // 2])

        o1, r1, term1, trunc1, _inf1 = env1.step(action)
        o2, r2, term2, trunc2, _inf2 = env2.step(action)

        assert r1 == r2
        assert term1 == term2
        assert trunc1 == trunc2
        np.testing.assert_array_equal(o1["grid"], o2["grid"])
        np.testing.assert_array_equal(o1["macro_features"], o2["macro_features"])


def test_macro_ordering_modes(medium_netlist: Netlist) -> None:
    """Test macro ordering configurations ('netlist', 'descending_area', 'connectivity')."""
    env_netlist = MacroPlacementEnv(netlist=medium_netlist, order="netlist")
    env_area = MacroPlacementEnv(netlist=medium_netlist, order="descending_area")
    env_conn = MacroPlacementEnv(netlist=medium_netlist, order="connectivity")

    # Netlist order matches definition order
    assert [m.id for m in env_netlist._macro_sequence] == [m.id for m in medium_netlist.macros]

    # Area order sorts by w * h descending
    areas = [m.w * m.h for m in env_area._macro_sequence]
    assert areas == sorted(areas, reverse=True)

    # Connectivity order sorts by incident weight descending
    conn_weights = [env_conn._macro_incident_weight[m.id] for m in env_conn._macro_sequence]
    assert conn_weights == sorted(conn_weights, reverse=True)


def test_render_ansi(toy_netlist: Netlist) -> None:
    """Verify ANSI rendering returns readable grid text."""
    env = MacroPlacementEnv(netlist=toy_netlist, render_mode="ansi")
    env.reset(seed=42)

    # Place one macro
    env.step(coord_to_action(0, 0, env.grid_w))
    rendered = env.render()
    assert isinstance(rendered, str)
    assert len(rendered.splitlines()) == toy_netlist.grid_h
    assert "0" in rendered  # Macro 0 is rendered at top-left


@async_test
async def test_rl_runner_registered_and_solves(toy_netlist: Netlist) -> None:
    """Verify rl_placement_runner is registered and generates a valid placement."""
    registered = get_registered_algorithms()
    assert "rl" in registered
    assert "maskable_ppo" in registered

    progress_events = []

    async def on_progress(p: Any) -> None:
        progress_events.append(p)

    placement = await rl_placement_runner(
        netlist=toy_netlist,
        seed=42,
        params={"alpha": 1.0, "beta": 1.0},
        progress_cb=on_progress,
    )

    validate_placement(toy_netlist, placement)
    assert len(placement) == len(toy_netlist.macros)
    assert len(progress_events) >= len(toy_netlist.macros) + 1
    assert progress_events[-1].progress == 1.0


@async_test
async def test_rl_in_job_manager(toy_netlist: Netlist) -> None:
    """Verify RL jobs execute cleanly through JobManager."""
    manager = JobManager()
    run_id = manager.create_and_start_job(
        netlist=toy_netlist,
        algorithm="rl",
        seed=100,
    )

    for _ in range(50):
        job = manager.get_job(run_id)
        if job and job.status in ("completed", "failed"):
            break
        await asyncio.sleep(0.02)

    job = manager.get_job(run_id)
    assert job is not None
    assert job.status == "completed"
    assert job.error is None
    assert len(job.placement) == len(toy_netlist.macros)


@async_test
async def test_rl_in_benchmark_execution(toy_netlist: Netlist) -> None:
    """Verify RL runs through execute_single_benchmark_run."""
    result = await execute_single_benchmark_run(
        algorithm="rl",
        netlist_id="toy_4",
        netlist=toy_netlist,
        seed=42,
        config={"alpha": 1.0, "beta": 1.0},
    )

    assert result.success
    assert result.is_legal
    assert result.is_complete
    assert result.hpwl is not None
    assert result.congestion is not None
    assert result.objective is not None
    assert result.runtime_sec > 0.0


def test_observation_builders_direct(toy_netlist: Netlist) -> None:
    """Directly test observation construction utility functions."""
    env = MacroPlacementEnv(netlist=toy_netlist)
    env.reset(seed=42)

    # Place macro 0 at (0, 0)
    env.engine.place(0, 0, 0)

    # Next macro is macro 1, connected to macro 0 via net 0 (weight 1.0)
    m1 = toy_netlist.macros[1]
    guidance = build_connectivity_guidance_map(
        engine=env.engine,
        current_macro=m1,
        macro_to_nets=env._macro_to_nets,
    )
    assert guidance.shape == (toy_netlist.grid_h, toy_netlist.grid_w)
    # Pin center of macro 0 (size 3x3 at 0,0) is at (1.5, 1.5) -> cell (1, 1)
    assert guidance[1, 1] == 1.0

    # Grid observation
    grid_obs = build_grid_observation(
        engine=env.engine,
        current_macro=m1,
        netlist=toy_netlist,
        macro_to_nets=env._macro_to_nets,
    )
    assert grid_obs.shape == (4, toy_netlist.grid_h, toy_netlist.grid_w)

    # Macro features
    feats = build_macro_features(
        current_macro=m1,
        step=1,
        total_macros=4,
        grid_w=toy_netlist.grid_w,
        grid_h=toy_netlist.grid_h,
        incident_weight=2.5,
        max_incident_weight=5.0,
    )
    assert feats.shape == (6,)
    assert 0.0 <= feats[0] <= 1.0


def test_deadlock_truncation_and_penalty() -> None:
    """A canvas partitioned so remaining macros cannot fit must trigger truncation and penalty."""
    # 3x3 grid with two 2x2 macros. Macro 0 placed at (0, 0) leaves no 2x2 space for Macro 1.
    deadlock_netlist = Netlist(
        version="1.0",
        grid_w=3,
        grid_h=3,
        macros=(Macro(id=0, w=2, h=2), Macro(id=1, w=2, h=2)),
        nets=(Net(id=0, macro_ids=(0, 1), weight=1.0),),
    )

    deadlock_penalty = 60.0
    env = MacroPlacementEnv(
        netlist=deadlock_netlist,
        deadlock_penalty=deadlock_penalty,
    )
    _obs, _info = env.reset(seed=42)

    # Place macro 0 at (0, 0) -> action = 0
    _o, reward, terminated, truncated, info = env.step(0)

    assert not terminated
    assert truncated
    assert not info["is_complete"]
    assert info["placed_count"] == 1
    assert not np.any(env.action_masks())
    # Step 1 delta_obj is 0.0 (only 1 macro placed), so reward should be 0.0 - deadlock_penalty
    assert reward == -deadlock_penalty


def test_impossible_netlist_all_zero_mask() -> None:
    """An impossible macro larger than the canvas must produce an all-zero mask and terminate safely."""
    oversized_netlist = Netlist(
        version="1.0",
        grid_w=4,
        grid_h=4,
        macros=(Macro(id=0, w=5, h=3),),  # Width 5 exceeds grid width 4
        nets=(),
    )

    penalty = 40.0
    env = MacroPlacementEnv(netlist=oversized_netlist, illegal_action_penalty=penalty)
    obs, info = env.reset(seed=42)

    # Initial action mask must be strictly all zeros
    assert not np.any(env.action_masks())
    assert np.all(obs["action_mask"] == 0)
    assert np.all(obs["grid"][1] == 0.0)  # Legal mask channel is all zeros
    assert not info["is_complete"]

    # Any step taken on an impossible netlist must terminate as illegal action without exception
    _obs, reward, terminated, truncated, step_info = env.step(0)
    assert terminated
    assert not truncated
    assert reward == -penalty
    assert not step_info["is_legal"]
    assert step_info["failure_type"] == "illegal_action"
    assert len(env.get_placement()) == 0


def test_repeated_resets_no_state_accumulation(toy_netlist: Netlist) -> None:
    """Consecutive resets must cleanly wipe state without accumulating placements or metrics."""
    env = MacroPlacementEnv(netlist=toy_netlist)

    for iteration in range(5):
        obs, info = env.reset(seed=123)

        assert env.step_idx == 0
        assert len(env.get_placement()) == 0
        assert not np.any(env.engine.occupancy_grid)
        assert info["step"] == 0
        assert info["placed_count"] == 0
        assert not info["is_complete"]
        assert obs["macro_features"][3] == 0.0  # Normalized step is 0

        # Place one macro
        legal_actions = np.flatnonzero(env.action_masks())
        assert len(legal_actions) > 0
        env.step(int(legal_actions[0]))
        assert len(env.get_placement()) == 1


def test_illegal_action_preserves_engine_state() -> None:
    """An illegal action must not corrupt or partially mutate existing placements or occupancy."""
    netlist = Netlist(
        version="1.0",
        grid_w=5,
        grid_h=5,
        macros=(Macro(id=0, w=2, h=2), Macro(id=1, w=2, h=2)),
        nets=(Net(id=0, macro_ids=(0, 1), weight=1.0),),
    )

    penalty = 35.0
    env = MacroPlacementEnv(netlist=netlist, illegal_action_penalty=penalty)
    env.reset(seed=42)

    # Place macro 0 legally at (0, 0)
    env.step(coord_to_action(0, 0, env.grid_w))
    assert len(env.get_placement()) == 1

    placement_before = list(env.get_placement())
    occupancy_before = env.engine.occupancy_grid.copy()

    # Attempt illegal placement for macro 1 overlapping macro 0 at (0, 0)
    illegal_action = coord_to_action(0, 0, env.grid_w)
    _obs, reward, terminated, truncated, info = env.step(illegal_action)

    assert terminated
    assert not truncated
    assert reward == -penalty
    assert not info["is_legal"]
    assert info["failure_type"] == "illegal_action"

    # Placement and occupancy must remain identical to before the illegal move
    assert env.get_placement() == placement_before
    np.testing.assert_array_equal(env.engine.occupancy_grid, occupancy_before)
    assert env.engine.is_placed(0)
    assert not env.engine.is_placed(1)


class MockPredictPolicy:
    """Mock policy object mimicking Stable-Baselines3 MaskablePPO interface."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, np.ndarray, bool]] = []

    def predict(
        self,
        observation: dict[str, np.ndarray],
        action_masks: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[int, None]:
        self.calls.append((observation, action_masks, deterministic))
        legal = np.flatnonzero(action_masks)
        assert len(legal) > 0
        return int(legal[0]), None


@async_test
async def test_policy_dispatch_predict_object(toy_netlist: Netlist) -> None:
    """Verify rl_placement_runner dispatches to policy.predict with correct arguments."""
    mock_policy = MockPredictPolicy()
    progress_snapshots = []

    async def on_progress(p: Any) -> None:
        progress_snapshots.append(p)

    placement = await rl_placement_runner(
        netlist=toy_netlist,
        seed=42,
        params={"policy": mock_policy},
        progress_cb=on_progress,
    )

    validate_placement(toy_netlist, placement)
    assert len(placement) == len(toy_netlist.macros)
    assert len(mock_policy.calls) == len(toy_netlist.macros)
    # Check that deterministic=True and action_masks were passed
    for _obs, mask, det in mock_policy.calls:
        assert det is True
        assert isinstance(mask, np.ndarray)


@async_test
async def test_policy_dispatch_callable(toy_netlist: Netlist) -> None:
    """Verify rl_placement_runner dispatches to a callable policy(obs, mask)."""
    calls: list[tuple[Any, np.ndarray]] = []

    def callable_policy(obs: dict[str, np.ndarray], mask: np.ndarray) -> int:
        calls.append((obs, mask))
        legal = np.flatnonzero(mask)
        assert len(legal) > 0
        return int(legal[-1])  # Select last legal action

    progress_snapshots = []

    async def on_progress(p: Any) -> None:
        progress_snapshots.append(p)

    placement = await rl_placement_runner(
        netlist=toy_netlist,
        seed=42,
        params={"policy": callable_policy},
        progress_cb=on_progress,
    )

    validate_placement(toy_netlist, placement)
    assert len(placement) == len(toy_netlist.macros)
    assert len(calls) == len(toy_netlist.macros)


@async_test
async def test_policy_dispatch_fallback_deterministic(toy_netlist: Netlist) -> None:
    """Verify default fallback (policy=None) is deterministically reproducible with identical seeds."""
    async def noop_cb(p: Any) -> None:
        pass

    placement1 = await rl_placement_runner(
        netlist=toy_netlist,
        seed=101,
        params={},
        progress_cb=noop_cb,
    )
    placement2 = await rl_placement_runner(
        netlist=toy_netlist,
        seed=101,
        params={},
        progress_cb=noop_cb,
    )

    coords1 = [(pm.macro_id, pm.x, pm.y) for pm in placement1]
    coords2 = [(pm.macro_id, pm.x, pm.y) for pm in placement2]
    assert coords1 == coords2


@async_test
async def test_runner_checkpoint_not_found_raises(toy_netlist: Netlist) -> None:
    """Passing a non-existent model path raises FileNotFoundError."""
    async def noop_cb(p: Any) -> None:
        pass

    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        await rl_placement_runner(
            netlist=toy_netlist,
            seed=42,
            params={"model_path": "non_existent_checkpoint.zip"},
            progress_cb=noop_cb,
        )


@async_test
async def test_runner_checkpoint_missing_sb3_contrib_raises(toy_netlist: Netlist, tmp_path: Any) -> None:
    """Attempting to load a checkpoint when sb3-contrib is not installed raises an informative ImportError."""
    dummy_cp = tmp_path / "dummy.zip"
    dummy_cp.write_bytes(b"PK00dummy")

    async def noop_cb(p: Any) -> None:
        pass

    import sys
    with (
        mock.patch.dict(sys.modules, {"sb3_contrib": None}),
        pytest.raises(ImportError, match="Loading saved MaskablePPO checkpoints requires 'sb3-contrib'"),
    ):
        await rl_placement_runner(
            netlist=toy_netlist,
            seed=42,
            params={"model_path": str(dummy_cp)},
            progress_cb=noop_cb,
        )


@async_test
async def test_runner_loads_checkpoint_with_mocked_sb3(toy_netlist: Netlist, tmp_path: Any) -> None:
    """Verify that a saved checkpoint is successfully loaded via MaskablePPO.load."""
    dummy_cp = tmp_path / "model.zip"
    dummy_cp.write_bytes(b"dummy_weights")

    mock_model = MockPredictPolicy()
    mock_sb3 = mock.MagicMock()
    mock_sb3.MaskablePPO.load.return_value = mock_model

    async def noop_cb(p: Any) -> None:
        pass

    import sys
    with mock.patch.dict(sys.modules, {"sb3_contrib": mock_sb3}):
        placement = await rl_placement_runner(
            netlist=toy_netlist,
            seed=42,
            params={"checkpoint_path": str(dummy_cp)},
            progress_cb=noop_cb,
        )

    mock_sb3.MaskablePPO.load.assert_called_once_with(str(dummy_cp))
    assert len(mock_model.calls) == len(toy_netlist.macros)
    validate_placement(toy_netlist, placement)


@async_test
async def test_runner_policy_precedence_over_checkpoint(toy_netlist: Netlist) -> None:
    """In-memory policy takes precedence over model_path parameter."""
    mock_policy = MockPredictPolicy()

    async def noop_cb(p: Any) -> None:
        pass

    # Non-existent model path would fail if attempted, but policy is used instead
    placement = await rl_placement_runner(
        netlist=toy_netlist,
        seed=42,
        params={"policy": mock_policy, "model_path": "non_existent.zip"},
        progress_cb=noop_cb,
    )
    assert len(mock_policy.calls) == len(toy_netlist.macros)
    validate_placement(toy_netlist, placement)



