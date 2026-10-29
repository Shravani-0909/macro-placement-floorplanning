import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


class DummyMaskedEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self._step_count = 0

    def action_masks(self) -> np.ndarray:
        # Mask 1 or 2 actions dynamically while ensuring valid actions remain
        mask = np.ones(5, dtype=bool)
        mask[self._step_count % 5] = False
        mask[(self._step_count + 1) % 5] = False
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        obs = np.zeros(4, dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action: int):
        self._step_count += 1
        obs = np.full(4, 0.1 * self._step_count, dtype=np.float32)
        reward = 1.0
        terminated = self._step_count >= 10
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()


if __name__ == "__main__":
    env = DummyMaskedEnv()
    wrapped_env = ActionMasker(env, mask_fn)

    model = MaskablePPO("MlpPolicy", wrapped_env, verbose=0)
    model.learn(total_timesteps=100)

    print("Success: MaskablePPO and ActionMasker smoke test completed without crashing!")
