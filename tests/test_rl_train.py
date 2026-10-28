"""Unit and integration tests for MaskablePPO training pipeline and observation flattening."""

import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from gymnasium import spaces

from src.placement.benchmarks import get_default_benchmark_suite
from src.placement.models import Netlist
from src.rl.env import MacroPlacementEnv
from src.rl.observation import FlattenPlacementObservation
from src.rl.train import (
    export_training_metadata,
    make_training_env,
    train_maskable_ppo,
)


@pytest.fixture
def toy_netlist() -> Netlist:
    """Standard 4-macro toy benchmark netlist (15x15)."""
    suite = get_default_benchmark_suite()
    return suite["toy_4"]


@pytest.fixture
def medium_netlist() -> Netlist:
    """Standard 8-macro medium benchmark netlist (30x30)."""
    suite = get_default_benchmark_suite()
    return suite["medium_8"]


def test_flatten_observation_dimensions_and_scaling(toy_netlist: Netlist) -> None:
    """Verify FlattenPlacementObservation normalizes all features into a 1D [0.0, 1.0] Box."""
    raw_env = MacroPlacementEnv(netlist=toy_netlist)
    flat_env = FlattenPlacementObservation(raw_env)

    # 15x15 grid -> 4 channels * 225 cells + 6 macro features = 906
    expected_dim = 4 * 15 * 15 + 6
    assert isinstance(flat_env.observation_space, spaces.Box)
    assert flat_env.observation_space.shape == (expected_dim,)
    assert flat_env.observation_space.dtype == np.float32
    assert float(flat_env.observation_space.low.min()) == 0.0
    assert float(flat_env.observation_space.high.max()) == 1.0

    obs, info = flat_env.reset(seed=42)
    assert obs.shape == (expected_dim,)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)
    assert info["step"] == 0

    # Step legally
    legal_actions = np.flatnonzero(flat_env.action_masks())
    assert len(legal_actions) > 0
    next_obs, reward, terminated, truncated, _step_info = flat_env.step(int(legal_actions[0]))

    assert next_obs.shape == (expected_dim,)
    assert np.all(next_obs >= 0.0)
    assert np.all(next_obs <= 1.0)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated


def test_flatten_observation_scaling_extreme_values(toy_netlist: Netlist) -> None:
    """Extreme grid values (e.g. 100,000.0) must scale monotonically within [0.0, 1.0)."""
    raw_env = MacroPlacementEnv(netlist=toy_netlist)
    flat_env = FlattenPlacementObservation(raw_env)

    dict_obs, _ = raw_env.reset(seed=42)
    # Inject extreme congestion and guidance values
    dict_obs["grid"][2, 0, 0] = 100000.0
    dict_obs["grid"][3, 0, 0] = 50000.0

    flattened = flat_env.observation(dict_obs)
    assert np.all(np.isfinite(flattened))
    assert np.all(flattened >= 0.0)
    assert np.all(flattened <= 1.0)

    # Scaled 100,000.0 via x / (1.0 + x) should be approximately 0.99999
    ch2_scaled = flattened[2 * 225]
    assert pytest.approx(ch2_scaled, rel=1e-4) == (100000.0 / 100001.0)


def test_flatten_observation_delegates_methods(toy_netlist: Netlist) -> None:
    """Wrapper must properly delegate action_masks() and get_placement()."""
    raw_env = MacroPlacementEnv(netlist=toy_netlist)
    flat_env = FlattenPlacementObservation(raw_env)
    flat_env.reset(seed=42)

    masks = flat_env.action_masks()
    assert masks.shape == (toy_netlist.grid_w * toy_netlist.grid_h,)
    assert masks.dtype == np.int8
    assert len(flat_env.get_placement()) == 0


def test_make_training_env_missing_sb3_contrib(toy_netlist: Netlist) -> None:
    """make_training_env must raise a descriptive ImportError when sb3-contrib is absent."""
    with (
        mock.patch.dict(sys.modules, {"sb3_contrib": None, "sb3_contrib.common.wrappers": None}),
        pytest.raises(ImportError, match="Training requires 'sb3-contrib'"),
    ):
        make_training_env(netlist=toy_netlist, seed=42, rank=0)


def test_train_maskable_ppo_missing_dependencies() -> None:
    """train_maskable_ppo must raise a descriptive ImportError when dependencies are absent."""
    with (
        mock.patch.dict(sys.modules, {"sb3_contrib": None, "stable_baselines3": None}),
        pytest.raises(ImportError, match="Training requires 'sb3-contrib'"),
    ):
        train_maskable_ppo(netlist_id="toy_4", timesteps=100)


def test_export_training_metadata(toy_netlist: Netlist, tmp_path: Path) -> None:
    """Verify metadata export formats schema fields and persists valid JSON."""
    meta_file = tmp_path / "metadata.json"
    hyperparams = {"learning_rate": 3e-4, "n_steps": 128, "n_envs": 4}
    metrics = {"best_mean_reward": -8.5}

    result = export_training_metadata(
        metadata_path=meta_file,
        netlist_id="toy_4",
        netlist=toy_netlist,
        timesteps=5000,
        seed=100,
        hyperparameters=hyperparams,
        summary_metrics=metrics,
    )

    assert meta_file.exists()
    assert result["netlist_id"] == "toy_4"
    assert result["grid_w"] == 15
    assert result["grid_h"] == 15
    assert result["macro_count"] == 4
    assert result["policy_architecture"] == "MlpPolicy"
    assert result["hyperparameters"]["learning_rate"] == 3e-4

    with open(meta_file, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["netlist_id"] == "toy_4"


def test_train_maskable_ppo_mocked_orchestration(tmp_path: Path) -> None:
    """Verify train_maskable_ppo orchestrates vector envs, PPO learning, and metadata export."""
    mock_model = mock.MagicMock()
    mock_sb3 = mock.MagicMock()
    mock_sb3.MaskablePPO.return_value = mock_model

    mock_vec = mock.MagicMock()
    mock_sb3_vec = mock.MagicMock()
    mock_sb3_vec.DummyVecEnv.return_value = mock_vec

    mock_wrappers = mock.MagicMock()
    mock_callbacks = mock.MagicMock()

    modules_to_mock = {
        "sb3_contrib": mock_sb3,
        "sb3_contrib.common.wrappers": mock_wrappers,
        "sb3_contrib.common.maskable.callbacks": mock_callbacks,
        "stable_baselines3": mock.MagicMock(),
        "stable_baselines3.common.vec_env": mock_sb3_vec,
    }

    with mock.patch.dict(sys.modules, modules_to_mock):
        res = train_maskable_ppo(
            netlist_id="toy_4",
            timesteps=200,
            seed=42,
            n_envs=2,
            output_dir=tmp_path,
        )

    # Asserts
    mock_sb3.MaskablePPO.assert_called_once()
    mock_model.learn.assert_called_once()
    mock_model.save.assert_called_once()
    assert Path(res["final_model_path"]).name == "maskable_ppo_toy_4_final.zip"
    assert Path(res["metadata_path"]).exists()
