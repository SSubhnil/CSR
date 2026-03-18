"""
Procgen environment wrapper for CSR, following the same pattern as envs/atari.py.

Uses the procgen pip package (0.10.7) via shimmy's GymV21 bridge.
Returns CSR's 4-tuple step API: (obs_dict, reward, is_last, info).
"""

import gym
import gymnasium
import numpy as np
import cv2

# Shimmy 2.0 requires explicit registration of GymV21Environment-v0
from shimmy.registration import register_gymnasium_envs
register_gymnasium_envs()


class LegacyProcGenWrapper(gymnasium.Wrapper):
    """Strip seed/options kwargs that the legacy procgen env doesn't accept."""

    def reset(self, *, seed=None, options=None):
        return self.env.reset()


class ProcGen:
    metadata = {}

    def __init__(
        self,
        name,
        action_repeat=1,
        size=(64, 64),
        gray=False,
        seed=None,
        distribution_mode="hard",
        num_levels=0,
        start_level=0,
    ):
        self._size = tuple(size)
        self._gray = gray
        self._repeat = action_repeat
        self._step_count = 0
        self._done = True

        # Trigger procgen gym registration
        try:
            import procgen  # noqa: F401
        except ImportError:
            pass

        env_id = f"procgen-{name}-v0"
        base_env = gymnasium.make(
            "GymV21Environment-v0",
            env_id=env_id,
            make_kwargs=dict(
                distribution_mode=distribution_mode,
                num_levels=num_levels,
                start_level=start_level,
                center_agent=True,
                render_mode="rgb_array",
            ),
        )
        self._env = LegacyProcGenWrapper(base_env)
        self._num_actions = self._env.action_space.n

    @property
    def observation_space(self):
        img_shape = self._size + ((1,) if self._gray else (3,))
        return gym.spaces.Dict(
            {"image": gym.spaces.Box(0, 255, img_shape, np.uint8)}
        )

    @property
    def action_space(self):
        space = gym.spaces.Discrete(self._num_actions)
        space.discrete = True
        return space

    def _process_image(self, obs):
        if obs.shape[:2] != self._size:
            obs = cv2.resize(obs, self._size, interpolation=cv2.INTER_AREA)
        if self._gray:
            weights = [0.299, 0.587, 1 - (0.299 + 0.587)]
            obs = np.tensordot(obs, weights, (-1, 0)).astype(obs.dtype)
            obs = obs[:, :, None]
        return obs

    def step(self, action):
        if len(action.shape) >= 1:
            action = np.argmax(action)
        total_reward = 0.0
        for _ in range(self._repeat):
            obs, reward, terminated, truncated, info = self._env.step(action)
            self._step_count += 1
            total_reward += reward
            if terminated or truncated:
                break
        is_last = terminated or truncated
        self._done = is_last
        image = self._process_image(obs)
        return (
            {"image": image, "is_terminal": terminated, "is_first": False},
            total_reward,
            is_last,
            {},
        )

    def reset(self):
        obs, _info = self._env.reset()
        self._done = False
        self._step_count = 0
        image = self._process_image(obs)
        return {"image": image, "is_terminal": False, "is_first": True}

    def close(self):
        self._env.close()
