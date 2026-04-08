# envs/nethack.py
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from PIL import Image
import nle


class NetHack(gym.Env):
    metadata = {}

    def __init__(self, task, size=(64, 64), max_episode_steps=5000, seed=0):
        task = str(task)
        if task.startswith("nethack-"):
            task = task.split("-", 1)[1]
        if task.startswith("nethack_"):
            task = task.split("_", 1)[1]

        env_name = f"NetHack{task}-v0"
        base_env = gym.make(env_name)
        self._env = TimeLimit(base_env, max_episode_steps=max_episode_steps)
        self._size = tuple(size)
        self._seed = seed

        obs_space = self._env.observation_space
        self._blstats_shape = tuple(obs_space["blstats"].shape)

        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, (*self._size, 3), dtype=np.uint8),
            "blstats": gym.spaces.Box(-np.inf, np.inf, self._blstats_shape, dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, shape=(), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, shape=(), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, shape=(), dtype=bool),
        })
        self.action_space = gym.spaces.Discrete(self._env.action_space.n)

    def reset(self, *, seed=None, options=None):
        seed = self._seed if seed is None else seed
        obs, info = self._env.reset(seed=seed)
        return self._obs(obs, is_first=True, is_last=False, is_terminal=False)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        done = bool(terminated or truncated)
        out = self._obs(
            obs,
            is_first=False,
            is_last=done,
            is_terminal=bool(terminated),
        )
        return out, np.float32(reward), done, info

    def _obs(self, obs, is_first, is_last, is_terminal):
        image = self._tty_to_rgb(obs["tty_chars"], obs["tty_colors"])
        return {
            "image": image,
            "blstats": obs["blstats"].astype(np.float32),
            "is_first": np.array(is_first, dtype=bool),
            "is_last": np.array(is_last, dtype=bool),
            "is_terminal": np.array(is_terminal, dtype=bool),
        }

    def _tty_to_rgb(self, tty_chars, tty_colors):
        tty_chars = np.asarray(tty_chars, dtype=np.uint8)
        tty_colors = np.asarray(tty_colors, dtype=np.uint8)

        rgb = np.zeros((tty_chars.shape[0], tty_chars.shape[1], 3), dtype=np.uint8)
        rgb[..., 0] = tty_chars
        rgb[..., 1] = (tty_colors.astype(np.int32) * 16).clip(0, 255).astype(np.uint8)
        rgb[..., 2] = (
            (tty_chars.astype(np.int32) // 2) + (tty_colors.astype(np.int32) * 8)
        ).clip(0, 255).astype(np.uint8)

        image = Image.fromarray(rgb).resize(self._size, Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8)