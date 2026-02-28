"""GPU-resident vectorized IsaacLab environment wrapper for r2dreamer.

IsaacLab runs a fully GPU-resident simulation with N parallel environments
built in.  This wrapper adapts it to the same interface that ``ParallelEnv``
exposes to ``OnlineTrainer`` and ``Buffer``, but without any CPU round-trip:

  - ``step(action, done)`` accepts and returns GPU tensors directly.
  - The returned ``TensorDict`` is on the same CUDA device as the sim.
  - The ``done`` argument is accepted for API compatibility but ignored —
    IsaacLab performs per-environment auto-reset internally.
  - Scalar observation fields (``is_first``, ``is_terminal``, ``is_last``,
    ``reward``) are lifted from shape ``(B,)`` to ``(B, 1)`` to match the
    layout that ``Buffer.add_transition`` expects (it calls
    ``data.unsqueeze(1)`` which turns ``(B, 1, *)`` into ``(B, 1, 1, *)``
    for 1-D fields — the lift here keeps things consistent with
    ``ParallelEnv.lift_dim``).

Usage::

    isaac_env = gym.make("Isaac-Cartpole-Direct-v0", cfg=env_cfg)
    vec_env = IsaacLabVecEnv(isaac_env.unwrapped)
    # vec_env now satisfies the ParallelEnv interface expected by OnlineTrainer.
"""

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict


class IsaacLabVecEnv:
    """Wraps a vectorized IsaacLab DirectRLEnv for use with r2dreamer.

    All data remains on the GPU throughout.  The wrapper tracks ``is_first``,
    ``is_last``, and ``is_terminal`` from IsaacLab's ``terminated``/
    ``truncated`` signals and injects them into the returned ``TensorDict``.

    Parameters
    ----------
    env:
        An unwrapped IsaacLab ``DirectRLEnv`` instance (or a gymnasium
        wrapper around one — the unwrapped env is accessed via
        ``env.unwrapped`` if needed).
    """

    def __init__(self, env, simulation_app=None):
        self._env = env
        self._app = simulation_app
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        self._num_envs = unwrapped.num_envs
        self._device = unwrapped.device

        # On the very first step every environment is "first".
        self._is_first = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)

    # ------------------------------------------------------------------
    # ParallelEnv interface
    # ------------------------------------------------------------------

    @property
    def env_num(self) -> int:
        return self._num_envs

    @property
    def observation_space(self) -> gym.spaces.Dict:
        """Single-env observation space including ``is_first``/``is_terminal``/``is_last``."""
        unwrapped = self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env
        spaces = {}
        for key, box in unwrapped.single_observation_space.spaces.items():
            spaces[key] = gym.spaces.Box(
                low=float(np.array(box.low).flat[0]),
                high=float(np.array(box.high).flat[0]),
                shape=box.shape,
                dtype=box.dtype,
            )
        spaces["is_first"] = gym.spaces.Box(0, 1, (1,), dtype=bool)
        spaces["is_terminal"] = gym.spaces.Box(0, 1, (1,), dtype=bool)
        spaces["is_last"] = gym.spaces.Box(0, 1, (1,), dtype=bool)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self) -> gym.spaces.Box:
        """Single-env action space, clipped to [-1, 1]."""
        unwrapped = self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env
        space = unwrapped.single_action_space
        low = np.clip(np.array(space.low), -1.0, 1.0).astype(np.float32)
        high = np.clip(np.array(space.high), -1.0, 1.0).astype(np.float32)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def reset(self):
        """Reset all environments and mark the next step as first."""
        self._env.reset()
        self._is_first = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)

    def step(self, action: torch.Tensor, done: torch.Tensor):
        """Step all environments and return a GPU-resident TensorDict.

        Parameters
        ----------
        action:
            Float tensor of shape ``(B, A)`` **on any device** — passed
            directly to the IsaacLab env which expects a GPU tensor.
        done:
            Bool tensor of shape ``(B,)`` — accepted for API compatibility
            with ``ParallelEnv`` but intentionally ignored.  IsaacLab
            manages per-environment auto-resets internally; the resulting
            ``is_first`` flag is the authoritative reset signal.

        Returns
        -------
        td : TensorDict
            Shape ``(B,)``, on the simulation device (CUDA).  Contains all
            observation keys from the env plus ``reward``, ``is_first``,
            ``is_terminal``, and ``is_last``.  Scalar fields are lifted to
            ``(B, 1)`` to match ``ParallelEnv.lift_dim`` behaviour.
        done : torch.BoolTensor
            Shape ``(B,)`` on the simulation device.  True for environments
            whose episode just ended (terminated **or** truncated).
        """
        # IsaacLab expects the action on its own device.
        action = action.to(self._device)

        obs_dict, reward, terminated, truncated, _ = self._env.step(action)

        # Pump the Omniverse Kit event loop so livestream (WebRTC/WebSocket)
        # can accept connections and push frames.
        if self._app is not None:
            self._app.update()

        episode_done = terminated | truncated

        # Build the TensorDict directly on GPU — no CPU involved.
        data = {}
        for key, val in obs_dict.items():
            # Ensure float obs are float32; uint8 images stay uint8.
            if val.dtype == torch.float64:
                val = val.float()
            # Lift (B,) → (B, 1) for 1-D observations so the buffer sees a
            # consistent ndim across all fields after add_transition's unsqueeze.
            if val.ndim == 1:
                val = val.unsqueeze(-1)
            data[key] = val

        # is_first: True for envs that were reset (done on the *previous* step,
        # handled by IsaacLab auto-reset) — stored from the last call.
        data["is_first"] = self._is_first.unsqueeze(-1)
        # is_terminal: True when the episode ended due to a failure condition
        # (not time-based truncation).
        data["is_terminal"] = terminated.unsqueeze(-1)
        # is_last: True whenever the episode ended for any reason.
        data["is_last"] = episode_done.unsqueeze(-1)
        # reward: (B,) → (B, 1)
        data["reward"] = reward.float().unsqueeze(-1)

        td = TensorDict(data, batch_size=(self._num_envs,), device=self._device)

        # Update is_first for the *next* step: any env that just ended will
        # have been auto-reset by IsaacLab, so its next observation is a
        # first observation.
        self._is_first = episode_done.clone()

        return td, episode_done
