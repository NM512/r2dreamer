"""GPU-resident vectorized IsaacLab environment wrapper for r2dreamer.

IsaacLab runs a fully GPU-resident simulation with N parallel environments
built in.  This wrapper adapts it to the same interface that ``ParallelEnv``
exposes to ``OnlineTrainer`` and ``Buffer``, but without any CPU round-trip:
"""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from tensordict import TensorDict

# =============================================================================
# Terminal-observation capture subclasses
# =============================================================================
#
# Terminal observation capture

# IsaacLab runs N environments in parallel on the GPU. When an environment
# terminates or is truncated, IsaacLab **auto-resets it inside `step()`** and
# returns the post-reset observation as the env's entry in `obs_dict`. The true
# terminal observation is silently overwritten before it is ever returned.

# This is a problem for Dreamer because the terminal reward must be paired with
# the true terminal observation. Pairing it with the reset obs misattributes
# the reward to a state that never produced it.

# The `R2DreamerRLEnv` and `R2DreamerDirectRLEnv` base classes override
# `_reset_idx` to capture observations before the reset and store them in
# `extras["terminal_obs"]`. The `IsaacLabVecEnv` wrapper swaps these in on the
# step an episode ends, matching the data flow expected by Dreamer's world model.


class R2DreamerRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with pre-reset terminal observation capture."""

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # Clear stale terminal obs from the previous step so the wrapper
        # doesn't see data from a step where no envs terminated.
        self.extras.pop("terminal_obs", None)
        self.extras.pop("terminal_env_ids", None)
        return super().step(action)

    def _reset_idx(self, env_ids: Sequence[int]):
        # Capture observations BEFORE the reset so the wrapper can return
        # the true terminal obs to the agent.
        if len(env_ids) > 0:
            terminal_obs = self.observation_manager.compute()
            self.extras["terminal_obs"] = {key: val[env_ids].clone() for key, val in terminal_obs.items()}
            self.extras["terminal_env_ids"] = env_ids
        super()._reset_idx(env_ids)


class R2DreamerDirectRLEnv(DirectRLEnv):
    """DirectRLEnv with pre-reset terminal observation capture.

    For your own Direct envs, inherit from this class directly in the task
    definition.  For third-party Direct envs created via ``gym.make``, use
    ``_patch_direct_env`` in ``train_isaaclab.py`` to inject this class
    into the instance's MRO.
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        self.extras.pop("terminal_obs", None)
        self.extras.pop("terminal_env_ids", None)
        return super().step(action)

    def _reset_idx(self, env_ids: Sequence[int]):
        if len(env_ids) > 0:
            terminal_obs = self._get_observations()
            self.extras["terminal_obs"] = {key: val[env_ids].clone() for key, val in terminal_obs.items()}
            self.extras["terminal_env_ids"] = env_ids
        super()._reset_idx(env_ids)


# =============================================================================
# Vectorized wrapper
# =============================================================================


class IsaacLabVecEnv:
    """Wraps a vectorized IsaacLab env for use with r2dreamer.

    Parameters
    ----------
    env:
        An unwrapped IsaacLab env that captures terminal observations
        before auto-reset (e.g. ``R2DreamerRLEnv``, ``R2DreamerDirectRLEnv``,
        or any subclass that populates ``extras["terminal_obs"]``).
    """

    def __init__(self, env, simulation_app=None):
        self._env = env
        self._app = simulation_app
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        self._num_envs = unwrapped.num_envs
        self._device = unwrapped.device

        # On the very first step every environment is "first".
        self._is_first = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        # Envs whose previous step was a terminal step.  On the next call
        # these envs will emit is_first=True with reward=0.
        self._pending_first = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

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
        self._pending_first = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

    def step(self, action: torch.Tensor, done: torch.Tensor):
        """Step all environments and return a GPU-resident TensorDict.

        Parameters
        ----------
        action:
            Float tensor of shape ``(B, A)`` **on any device**, passed
            directly to the IsaacLab env which expects a GPU tensor.
        done:
            Bool tensor of shape ``(B,)``, accepted for API compatibility
            with ``ParallelEnv`` but intentionally ignored.  IsaacLab
            manages per-environment auto-resets internally.

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

        obs_dict, reward, terminated, truncated, extras = self._env.step(action)

        episode_done = terminated | truncated

        # ------------------------------------------------------------------
        # Build obs data dict
        # ------------------------------------------------------------------
        data = {}
        for key, val in obs_dict.items():
            if val.dtype == torch.float64:
                val = val.float()
            if val.ndim == 1:
                val = val.unsqueeze(-1)
            data[key] = val

        # ------------------------------------------------------------------
        # Swap in terminal observations for envs that just ended.
        # ------------------------------------------------------------------
        terminal_obs = extras.get("terminal_obs")
        terminal_env_ids = extras.get("terminal_env_ids")

        if terminal_obs is not None and terminal_env_ids is not None and len(terminal_env_ids) > 0:
            for key, val in terminal_obs.items():
                if val.dtype == torch.float64:
                    val = val.float()
                if val.ndim == 1:
                    val = val.unsqueeze(-1)
                data[key][terminal_env_ids] = val

        # ------------------------------------------------------------------
        # Flags & reward
        # ------------------------------------------------------------------
        # is_first: True for envs whose PREVIOUS step was an episode end
        # (they are now showing the first obs of the new episode).
        is_first_now = self._is_first | self._pending_first

        # We must NOT fire is_first on the SAME step as episode_done
        # for envs that are finishing normally (not pending).  is_first fires
        # on the NEXT step for those.
        data["is_first"] = is_first_now.unsqueeze(-1)

        # is_terminal: True only for actual termination (not truncation).
        data["is_terminal"] = terminated.unsqueeze(-1)

        # is_last: True whenever the episode ended for any reason.
        data["is_last"] = episode_done.unsqueeze(-1)

        # For pending_first envs (showing the first obs of a new episode),
        # zero the reward, the reset obs carries no meaningful reward.
        # Exception: if a pending_first env ALSO terminates this step (a
        # 1-step episode), keep the terminal reward.
        reward_out = reward.float()
        zero_reward = self._pending_first & ~episode_done
        reward_out = torch.where(zero_reward, torch.zeros_like(reward_out), reward_out)
        data["reward"] = reward_out.unsqueeze(-1)

        td = TensorDict(data, batch_size=(self._num_envs,), device=self._device)

        # ------------------------------------------------------------------
        # Update state for next call.
        # ------------------------------------------------------------------
        # _is_first consumed; will only be re-set by explicit reset().
        self._is_first = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        # Envs that ended this step will emit is_first=True on the NEXT call.
        self._pending_first = episode_done.clone()

        return td, episode_done
