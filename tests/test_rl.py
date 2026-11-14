"""Unit tests for the Reinforcement Learning Macro Placement environment."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

try:
    import gymnasium as gym

    _HAS_GYMNASIUM = True
except ImportError:
    gym = None
    _HAS_GYMNASIUM = False

try:
    from sb3_contrib import MaskablePPO

    _HAS_SB3 = True
except ImportError:
    MaskablePPO = None
    _HAS_SB3 = False

needs_gym = pytest.mark.skipif(
    not _HAS_GYMNASIUM,
    reason="Gymnasium dependency unavailable in host environment",
)
needs_sb3 = pytest.mark.skipif(
    not _HAS_SB3,
    reason="sb3-contrib dependency unavailable in host environment",
)

from src.algorithms.rl_placement import (
    MacroPlacementEnv,
    compute_connectivity_features,
    _HAS_TORCH,
)
from src.placement.engine import PlacementEngine, action_to_coord, coord_to_action
from src.placement.models import Macro, Net, Netlist, PlacedMacro
from src.placement.objective import evaluate_placement


@pytest.fixture
def simple_netlist() -> Netlist:
    """Fixture providing a simple 2-macro, 1-net canonical Netlist."""
    macros = (
        Macro(id=0, w=2, h=2),
        Macro(id=1, w=2, h=2),
    )
    nets = (Net(id=0, macro_ids=(0, 1), weight=1.0),)
    return Netlist(version="1.0", grid_w=6, grid_h=6, macros=macros, nets=nets)


@pytest.fixture
def tight_netlist() -> Netlist:
    """Fixture where second macro cannot fit after placing first macro."""
    macros = (
        Macro(id=0, w=3, h=3),
        Macro(id=1, w=3, h=3),
    )
    nets = (Net(id=0, macro_ids=(0, 1), weight=1.0),)
    return Netlist(version="1.0", grid_w=4, grid_h=4, macros=macros, nets=nets)


@needs_gym
def test_env_construction(simple_netlist: Netlist) -> None:
    """Environment can be instantiated with canonical Netlist with valid spaces."""
    env = MacroPlacementEnv(netlist=simple_netlist)

    assert env.grid_w == 6
    assert env.grid_h == 6
    assert len(env.macros) == 2

    # Check action space
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 36

    # Check observation space
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert "grid_occupancy" in env.observation_space.spaces
    assert "macro_features" in env.observation_space.spaces
    assert "connectivity_features" in env.observation_space.spaces
    assert env.observation_space["grid_occupancy"].shape == (1, 6, 6)
    assert env.observation_space["macro_features"].shape == (2,)
    # conn_dim: 4 (self) + 5 (aggregate) + 6 * 2 (slots) = 21
    assert env.observation_space["connectivity_features"].shape == (21,)


@needs_gym
def test_env_reset(simple_netlist: Netlist) -> None:
    """Reset clears engine, returns valid observation and info, and does not accumulate."""
    env = MacroPlacementEnv(netlist=simple_netlist)

    obs, info = env.reset(seed=42)

    assert isinstance(obs, dict)
    assert "grid_occupancy" in obs
    assert "macro_features" in obs
    assert "connectivity_features" in obs
    assert np.all(obs["grid_occupancy"] == 0.0)
    assert np.array_equal(obs["macro_features"], np.array([2.0, 2.0], dtype=np.float32))
    assert obs["connectivity_features"].shape == (21,)
    assert obs["connectivity_features"].dtype == np.float32

    # Engine must be completely empty
    assert len(env.engine.placed_macros) == 0
    assert not np.any(env.engine.occupancy_grid)
    assert info["placed_count"] == 0
    assert info["total_macros"] == 2
    assert info["is_complete"] is False

    # Repeated reset must not accumulate
    env.reset(seed=42)
    assert len(env.engine.placed_macros) == 0
    assert not np.any(env.engine.occupancy_grid)


@needs_gym
def test_action_mapping(simple_netlist: Netlist) -> None:
    """Action index maps deterministically and matches PlacementEngine conventions."""
    env = MacroPlacementEnv(netlist=simple_netlist)

    # For 6x6 grid: action = y * 6 + x
    action = 14  # y = 2, x = 2
    x, y = action_to_coord(action, env.grid_w)
    assert x == 2
    assert y == 2
    assert coord_to_action(x, y, env.grid_w) == action

    action_corner = 35  # y = 5, x = 5
    xc, yc = action_to_coord(action_corner, env.grid_w)
    assert xc == 5
    assert yc == 5
    assert coord_to_action(xc, yc, env.grid_w) == action_corner


@needs_gym
def test_action_mask(simple_netlist: Netlist) -> None:
    """Action mask enables legal anchors and masks boundary & overlap conflicts."""
    env = MacroPlacementEnv(netlist=simple_netlist)
    env.reset()

    mask0 = env.action_masks()
    assert isinstance(mask0, np.ndarray)
    assert mask0.shape == (36,)
    assert mask0.dtype == bool

    # For 2x2 macro on 6x6 grid, valid anchors are x in [0, 4] and y in [0, 4]
    # (0, 0) should be legal: action = 0
    assert mask0[0] is True
    # (4, 4) should be legal: action = 4 * 6 + 4 = 28
    assert mask0[28] is True
    # (5, 0) out of bounds: action = 5
    assert mask0[5] is False
    # (0, 5) out of bounds: action = 30
    assert mask0[30] is False

    # Place first macro at (0, 0) -> action 0
    env.step(0)

    # Mask for macro 1 should now be updated
    mask1 = env.action_masks()
    # Placing macro 1 at (0, 0) or (1, 1) overlaps macro 0 ([0, 2) x [0, 2))
    assert mask1[0] is False  # (0, 0) overlap
    assert mask1[1] is False  # (1, 0) overlap
    assert mask1[6] is False  # (0, 1) overlap
    assert mask1[7] is False  # (1, 1) overlap

    # Placing macro 1 at (2, 0) touches macro 0, which is legal
    action_touching = 2  # x = 2, y = 0
    assert mask1[action_touching] is True


@needs_gym
def test_placement_engine_integration(simple_netlist: Netlist) -> None:
    """Stepping environment mutates the shared PlacementEngine correctly."""
    env = MacroPlacementEnv(netlist=simple_netlist)
    env.reset()

    # Step placing macro 0 at (1, 2) -> action = 2 * 6 + 1 = 13
    action = 13
    obs, reward, terminated, truncated, info = env.step(action)

    assert env.engine.is_placed(0)
    pm = env.engine.placed_macros[0]
    assert pm.x == 1
    assert pm.y == 2

    # Engine occupancy grid reflects [2:4, 1:3]
    occupancy = env.engine.occupancy_grid
    assert np.all(occupancy[2:4, 1:3])
    # Outside should be False
    assert not occupancy[0, 0]

    # Environment placed_macros property matches engine
    assert len(env.placed_macros) == 1
    assert env.placed_macros[0].macro_id == 0


@needs_gym
def test_objective_alignment(simple_netlist: Netlist) -> None:
    """Environment step uses canonical evaluate_placement including HPWL and RUDY congestion."""
    env = MacroPlacementEnv(netlist=simple_netlist, alpha=2.0, beta=3.0)
    env.reset()

    # Place macro 0 at (0, 0)
    env.step(coord_to_action(0, 0, env.grid_w))
    # Place macro 1 at (3, 3)
    obs, reward, terminated, truncated, info = env.step(coord_to_action(3, 3, env.grid_w))

    # Evaluate canonical placement independently
    expected_eval = evaluate_placement(simple_netlist, env.engine.get_placement(), alpha=2.0, beta=3.0)

    assert info["hpwl"] == expected_eval.hpwl
    assert info["congestion"] == expected_eval.congestion
    assert info["objective"] == expected_eval.objective
    assert info["hpwl"] > 0.0
    assert info["congestion"] > 0.0
    assert info["objective"] == 2.0 * expected_eval.hpwl + 3.0 * expected_eval.congestion


@needs_gym
def test_reward_direction() -> None:
    """Reward reflects improvement in objective: previous J > new J yields positive reward."""
    macros = (Macro(id=0, w=1, h=1),)
    nets = ()
    netlist = Netlist(version="1.0", grid_w=3, grid_h=3, macros=macros, nets=nets)

    env = MacroPlacementEnv(netlist=netlist)
    env.reset()

    # Simulate an objective improvement: set previous_objective higher than step evaluation
    env.current_objective = 50.0
    obs, reward, terminated, truncated, info = env.step(0)

    # For empty nets, objective is 0.0
    assert info["objective"] == 0.0
    # reward = previous_objective - new_objective = 50.0 - 0.0 = +50.0
    assert reward == 50.0

    # Test reverse: set previous_objective lower than new_objective
    env.reset()
    env.current_objective = -10.0
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == -10.0


@needs_gym
def test_completion_lifecycle(simple_netlist: Netlist) -> None:
    """Placing all macros completes the episode and reports metrics in info."""
    env = MacroPlacementEnv(netlist=simple_netlist, terminal_bonus=10.0)
    env.reset()

    # Place macro 0 at (0, 0)
    obs1, r1, term1, trunc1, info1 = env.step(coord_to_action(0, 0, env.grid_w))
    assert not term1
    assert not trunc1
    assert info1["placed_count"] == 1
    assert info1["is_complete"] is False

    # Place macro 1 at (2, 2)
    obs2, r2, term2, trunc2, info2 = env.step(coord_to_action(2, 2, env.grid_w))
    assert term2 is True
    assert trunc2 is False
    assert info2["placed_count"] == 2
    assert info2["is_complete"] is True
    assert "hpwl" in info2
    assert "congestion" in info2
    assert "objective" in info2


@needs_gym
def test_dead_end_truncation(tight_netlist: Netlist) -> None:
    """Oversubscribed / dead-end state triggers truncated flag and penalty."""
    # 4x4 grid with two 3x3 macros.
    # Placing macro 0 at (0, 0) occupies [0, 3) x [0, 3).
    # Remaining space cannot fit second 3x3 macro anywhere on 4x4 grid.
    env = MacroPlacementEnv(netlist=tight_netlist, dead_end_penalty=25.0)
    env.reset()

    # Place macro 0 at (0, 0)
    obs, reward, terminated, truncated, info = env.step(coord_to_action(0, 0, env.grid_w))

    assert truncated is True
    assert terminated is False
    assert info["is_complete"] is False


@needs_gym
def test_determinism_with_seed(simple_netlist: Netlist) -> None:
    """Environment reset with the same seed produces identical initial observations and info."""
    env1 = MacroPlacementEnv(netlist=simple_netlist)
    obs1, info1 = env1.reset(seed=123)

    env2 = MacroPlacementEnv(netlist=simple_netlist)
    obs2, info2 = env2.reset(seed=123)

    assert np.array_equal(obs1["grid_occupancy"], obs2["grid_occupancy"])
    assert np.array_equal(obs1["macro_features"], obs2["macro_features"])
    assert info1 == info2


@needs_gym
def test_illegal_action_handling(simple_netlist: Netlist) -> None:
    """Passing an out-of-bounds or overlapping action terminates cleanly without engine crash."""
    env = MacroPlacementEnv(netlist=simple_netlist, illegal_action_penalty=50.0)
    env.reset()

    # Out of bounds action: action = 35 is (5, 5), macro is 2x2 on 6x6 grid
    obs, reward, terminated, truncated, info = env.step(35)

    assert terminated is True
    assert truncated is False
    assert reward == -50.0
    assert info["error"] == "Illegal action"
    assert len(env.engine.placed_macros) == 0


@needs_sb3
def test_ppo_readiness_smoke_test(simple_netlist: Netlist) -> None:
    """Verify that MacroPlacementEnv wraps with ActionMasker and interfaces with MaskablePPO."""
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.wrappers import ActionMasker
    from src.algorithms.rl_placement import mask_fn

    env = MacroPlacementEnv(simple_netlist)
    wrapped_env = ActionMasker(env, mask_fn)

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=wrapped_env,
        learning_rate=3e-4,
        n_steps=16,
        batch_size=16,
        verbose=0,
    )

    obs, info = wrapped_env.reset(seed=42)
    masks = wrapped_env.action_masks()
    action, _states = model.predict(obs, action_masks=masks, deterministic=True)
    obs, reward, terminated, truncated, step_info = wrapped_env.step(action)
    assert isinstance(reward, (float, np.floating))
    assert "hpwl" in step_info


def test_ppo_training_config_defaults_and_validation() -> None:
    """PPOTrainingConfig validates hyperparameter types, ranges, and constraints."""
    from src.algorithms.rl_placement import PPOTrainingConfig

    cfg = PPOTrainingConfig()
    assert cfg.total_timesteps == 256
    assert cfg.learning_rate == 3e-4
    assert cfg.n_steps == 64
    assert cfg.batch_size == 32
    assert cfg.seed == 42
    cfg.validate()  # Valid default config passes

    # Invalid total_timesteps
    with pytest.raises(ValueError, match="total_timesteps"):
        PPOTrainingConfig(total_timesteps=0).validate()
    with pytest.raises(ValueError, match="total_timesteps"):
        PPOTrainingConfig(total_timesteps=True).validate()  # type: ignore[arg-type]

    # Invalid learning rate
    with pytest.raises(ValueError, match="learning_rate"):
        PPOTrainingConfig(learning_rate=-0.01).validate()

    # Invalid batch_size > n_steps
    with pytest.raises(ValueError, match="batch_size"):
        PPOTrainingConfig(n_steps=16, batch_size=32).validate()

    # Invalid gamma
    with pytest.raises(ValueError, match="gamma"):
        PPOTrainingConfig(gamma=1.5).validate()

    # Invalid clip_range
    with pytest.raises(ValueError, match="clip_range"):
        PPOTrainingConfig(clip_range=0.0).validate()


def test_ppo_missing_dependency_errors(simple_netlist: Netlist) -> None:
    """train_ppo and evaluate_ppo provide clean project-level errors if Gymnasium is missing."""
    import src.algorithms.rl_placement as rlp
    from src.algorithms.rl_placement import evaluate_ppo, train_ppo

    with patch.object(rlp, "_HAS_GYMNASIUM", False):
        with pytest.raises(ImportError, match="Gymnasium is required"):
            train_ppo(simple_netlist)
        with pytest.raises(ImportError, match="Gymnasium is required"):
            evaluate_ppo(simple_netlist)


def test_compare_baseline_classical_algorithms(simple_netlist: Netlist) -> None:
    """compare_baseline evaluates Random, Greedy, and SA on the same canonical netlist."""
    import src.algorithms.rl_placement as rlp
    from src.algorithms.rl_placement import compare_baseline

    results = compare_baseline(simple_netlist, seed=42, sa_iterations=20)
    assert "random" in results
    assert "greedy" in results
    assert "sa" in results
    assert "rl" in results

    # Verify classical algorithms succeeded and produced valid metrics
    for algo in ("random", "greedy", "sa"):
        res = results[algo]
        assert res["feasible"] is True
        assert res["hpwl"] is not None and res["hpwl"] > 0.0
        assert res["congestion"] is not None and res["congestion"] >= 0.0
        assert res["objective"] is not None and res["objective"] > 0.0
        assert res["runtime_sec"] >= 0.0
        assert res["error"] is None

    # RL in host environment (without Gymnasium) records informative error
    if not rlp._HAS_GYMNASIUM:
        assert results["rl"]["feasible"] is False
        assert "Gymnasium" in (results["rl"]["error"] or "")


@needs_sb3
def test_ppo_training_checkpoint_lifecycle(simple_netlist: Netlist, tmp_path: Path) -> None:
    """Tiny smoke training run saves a checkpoint, reloads it, and evaluates canonical placement."""
    from src.algorithms.rl_placement import PPOTrainingConfig, evaluate_ppo, train_ppo

    cfg = PPOTrainingConfig(
        total_timesteps=16,
        n_steps=16,
        batch_size=16,
        seed=101,
        verbose=0,
    )
    checkpoint_file = tmp_path / "smoke_ppo.zip"

    train_res = train_ppo(
        simple_netlist,
        config=cfg,
        checkpoint_path=checkpoint_file,
    )

    assert train_res.total_timesteps == 16
    assert train_res.seed == 101
    assert train_res.training_time_s >= 0.0
    assert Path(train_res.checkpoint_path).exists()
    assert train_res.eval_result.algorithm == "rl"

    # Reload checkpoint and evaluate
    eval_res = evaluate_ppo(
        simple_netlist,
        model_or_path=checkpoint_file,
        seed=101,
    )
    assert eval_res.feasible is True
    assert eval_res.hpwl is not None and eval_res.hpwl > 0.0
    assert eval_res.objective is not None and eval_res.objective > 0.0
    assert len(eval_res.placement) == len(simple_netlist.macros)


def test_compute_connectivity_features_shape_and_range() -> None:
    """compute_connectivity_features returns correctly shaped, bounded float32 array."""
    macros = (
        Macro(id=0, w=2, h=3),
        Macro(id=1, w=4, h=2),
        Macro(id=2, w=1, h=1),
    )
    nets = (
        Net(id=0, macro_ids=(0, 1), weight=2.0),
        Net(id=1, macro_ids=(1, 2), weight=1.0),
    )
    netlist = Netlist(version="1.0", grid_w=10, grid_h=10, macros=macros, nets=nets)

    # Initial state: no macros placed, current macro is 0
    feat0 = compute_connectivity_features(netlist, placed_positions={}, current_macro_idx=0)
    expected_dim = 4 + 5 + 6 * 3  # 27
    assert feat0.shape == (expected_dim,)
    assert feat0.dtype == np.float32
    assert np.all(feat0 >= 0.0)
    assert np.all(feat0 <= 1.0)

    # Self features: w/W = 2/10 = 0.2, h/H = 3/10 = 0.3, idx/N = 0/3 = 0.0, step/N = 0/3 = 0.0
    assert np.isclose(feat0[0], 0.2)
    assert np.isclose(feat0[1], 0.3)
    assert np.isclose(feat0[2], 0.0)
    assert np.isclose(feat0[3], 0.0)

    # Macro 1 placed at (4, 6)
    feat_placed = compute_connectivity_features(netlist, placed_positions={1: (4, 6)}, current_macro_idx=0)
    # Placed connected count is 1 / 2
    assert feat_placed[4] > 0.0
    # Centroid of placed connected macros: center of macro 1 (w=4, h=2) at (4, 6) is (6, 7) -> cx=0.6, cy=0.7
    assert np.isclose(feat_placed[6], 0.6)
    assert np.isclose(feat_placed[7], 0.7)
    # Slot features for macro 1 (index 1: base = 9 + 6 * 1 = 15):
    assert np.isclose(feat_placed[15], 1.0)  # is_placed
    assert np.isclose(feat_placed[16], 0.4)  # x/W
    assert np.isclose(feat_placed[17], 0.6)  # y/H


def test_compute_connectivity_features_with_placed_macro_objects() -> None:
    """compute_connectivity_features cleanly consumes PlacementEngine.placed_macros (PlacedMacro objects)."""
    macros = (
        Macro(id=0, w=2, h=3),
        Macro(id=1, w=4, h=2),
        Macro(id=2, w=1, h=1),
    )
    nets = (
        Net(id=0, macro_ids=(0, 1), weight=2.0),
        Net(id=1, macro_ids=(1, 2), weight=1.0),
    )
    netlist = Netlist(version="1.0", grid_w=10, grid_h=10, macros=macros, nets=nets)

    engine = PlacementEngine(netlist)
    engine.place(1, 4, 6)

    # 1. Test using PlacementEngine.placed_macros (dict[int, PlacedMacro])
    feat_engine = compute_connectivity_features(
        netlist,
        engine.placed_macros,
        current_macro_idx=0,
    )
    expected_dim = 4 + 5 + 6 * 3
    assert feat_engine.shape == (expected_dim,)
    assert feat_engine.dtype == np.float32
    assert np.all(feat_engine >= 0.0)
    assert np.all(feat_engine <= 1.0)
    # Centroid of placed connected macros (macro 1 center (6, 7)) -> cx=0.6, cy=0.7
    assert np.isclose(feat_engine[6], 0.6)
    assert np.isclose(feat_engine[7], 0.7)
    # Slot features for macro 1 (index 1: base = 15)
    assert np.isclose(feat_engine[15], 1.0)
    assert np.isclose(feat_engine[16], 0.4)
    assert np.isclose(feat_engine[17], 0.6)

    # 2. Test using Placement sequence (list[PlacedMacro])
    feat_seq = compute_connectivity_features(
        netlist,
        engine.get_placement(),
        current_macro_idx=0,
    )
    assert np.array_equal(feat_engine, feat_seq)

    # 3. Test equivalence with legacy tuple coordinates dict
    feat_tuples = compute_connectivity_features(
        netlist,
        {1: (4, 6)},
        current_macro_idx=0,
    )
    assert np.array_equal(feat_engine, feat_tuples)

    # 4. Moving macro in PlacementEngine updates centroid and slots dynamically
    engine.unplace(1)
    engine.place(1, 0, 0)
    feat_moved = compute_connectivity_features(
        netlist,
        engine.placed_macros,
        current_macro_idx=0,
    )
    # New center is (0 + 2, 0 + 1) = (2, 1) -> cx=0.2, cy=0.1
    assert np.isclose(feat_moved[6], 0.2)
    assert np.isclose(feat_moved[7], 0.1)
    assert np.isclose(feat_moved[15], 1.0)
    assert np.isclose(feat_moved[16], 0.0)
    assert np.isclose(feat_moved[17], 0.0)
    assert not np.allclose(feat_engine, feat_moved)


def test_connectivity_features_different_for_different_topologies() -> None:
    """Topologically distinct netlists produce distinct connectivity feature vectors."""
    macros = (
        Macro(id=0, w=2, h=2),
        Macro(id=1, w=2, h=2),
        Macro(id=2, w=2, h=2),
    )
    # Netlist A: macro 0 connected to macro 1
    netlist_a = Netlist(
        version="1.0",
        grid_w=10,
        grid_h=10,
        macros=macros,
        nets=(Net(id=0, macro_ids=(0, 1), weight=1.0),),
    )
    # Netlist B: macro 0 connected to macro 2
    netlist_b = Netlist(
        version="1.0",
        grid_w=10,
        grid_h=10,
        macros=macros,
        nets=(Net(id=0, macro_ids=(0, 2), weight=1.0),),
    )

    feat_a = compute_connectivity_features(netlist_a, current_macro_idx=0)
    feat_b = compute_connectivity_features(netlist_b, current_macro_idx=0)

    assert feat_a.shape == feat_b.shape
    assert not np.allclose(feat_a, feat_b)


def test_connectivity_features_change_when_macro_moves() -> None:
    """When a connected macro's placement changes, connectivity features update dynamically."""
    macros = (
        Macro(id=0, w=2, h=2),
        Macro(id=1, w=2, h=2),
    )
    netlist = Netlist(
        version="1.0",
        grid_w=20,
        grid_h=20,
        macros=macros,
        nets=(Net(id=0, macro_ids=(0, 1), weight=1.0),),
    )

    # Position 1: macro 1 at (2, 2)
    feat_pos1 = compute_connectivity_features(netlist, placed_positions={1: (2, 2)}, current_macro_idx=0)
    # Position 2: macro 1 at (16, 18)
    feat_pos2 = compute_connectivity_features(netlist, placed_positions={1: (16, 18)}, current_macro_idx=0)

    assert not np.allclose(feat_pos1, feat_pos2)
    # Centroid cx, cy should differ significantly (center of 2x2: (3, 3) vs (17, 19))
    assert feat_pos1[6] != feat_pos2[6]
    assert feat_pos1[7] != feat_pos2[7]
    assert np.isclose(feat_pos1[6], 3.0 / 20.0)
    assert np.isclose(feat_pos2[6], 17.0 / 20.0)


def test_reward_scale_and_features_dim_validation() -> None:
    """PPOTrainingConfig validates reward_scale and features_dim."""
    from src.algorithms.rl_placement import PPOTrainingConfig

    # Valid config with custom reward_scale and features_dim
    cfg = PPOTrainingConfig(reward_scale=0.01, features_dim=64)
    cfg.validate()
    assert cfg.reward_scale == 0.01
    assert cfg.features_dim == 64

    # Invalid reward_scale
    with pytest.raises(ValueError, match="reward_scale"):
        PPOTrainingConfig(reward_scale=0.0).validate()
    with pytest.raises(ValueError, match="reward_scale"):
        PPOTrainingConfig(reward_scale=-0.5).validate()

    # Invalid features_dim
    with pytest.raises(ValueError, match="features_dim"):
        PPOTrainingConfig(features_dim=0).validate()
    with pytest.raises(ValueError, match="features_dim"):
        PPOTrainingConfig(features_dim=-10).validate()


@needs_gym
def test_env_reward_scaling(simple_netlist: Netlist) -> None:
    """reward_scale scales step reward proportionally."""
    env_unscaled = MacroPlacementEnv(netlist=simple_netlist, reward_scale=1.0)
    env_unscaled.reset()
    env_unscaled.current_objective = 100.0
    _, reward_unscaled, _, _, _ = env_unscaled.step(0)

    env_scaled = MacroPlacementEnv(netlist=simple_netlist, reward_scale=0.1)
    env_scaled.reset()
    env_scaled.current_objective = 100.0
    _, reward_scaled, _, _, _ = env_scaled.step(0)

    assert np.isclose(reward_scaled, 0.1 * reward_unscaled)


def test_classical_neutrality_preserved(simple_netlist: Netlist) -> None:
    """Random, Greedy, and Simulated Annealing algorithms remain correct and unaffected."""
    from src.algorithms.common import random_legal_placement
    from src.algorithms.greedy import greedy_placement
    from src.algorithms.simulated_annealing import simulated_annealing

    # 1. Random
    rand_p = random_legal_placement(simple_netlist, seed=42)
    assert rand_p is not None
    assert len(rand_p) == 2
    rand_eval = evaluate_placement(simple_netlist, rand_p, alpha=1.0, beta=1.0)
    assert rand_eval.hpwl > 0.0
    assert rand_eval.objective > 0.0

    # 2. Greedy
    greedy_res = greedy_placement(simple_netlist, alpha=1.0, beta=1.0)
    assert greedy_res.success is True
    assert greedy_res.hpwl is not None and greedy_res.hpwl > 0.0
    assert greedy_res.objective is not None and greedy_res.objective > 0.0

    # 3. Simulated Annealing
    sa_res = simulated_annealing(simple_netlist, alpha=1.0, beta=1.0, seed=42, max_iterations=50)
    assert sa_res.success is True
    assert sa_res.hpwl is not None and sa_res.hpwl > 0.0
    assert sa_res.objective is not None and sa_res.objective > 0.0

