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

Terminal observation handling
-----------------------------
IsaacLab auto-resets terminated/truncated envs *inside* its ``step()`` and
returns the post-reset observation.  The true terminal obs and reward would
be lost.

This wrapper requires the inner env to be an ``R2DreamerRLEnv``,
``R2DreamerDirectRLEnv``, or any IsaacLab env subclass that sets
``extras["terminal_obs"]`` and ``extras["terminal_env_ids"]`` before
the auto-reset.  The ``make_isaac_env`` factory handles this
automatically for both ManagerBased and Direct envs.

On the step where an episode ends the wrapper:
  1. Swaps in the **terminal obs** (from ``extras``) for the done envs.
  2. Keeps the **terminal reward** (no longer zeroed).
  3. Sets ``is_last=True``, ``is_terminal=terminated`` (not truncated),
     ``is_first=False``.
  4. Records those env indices in ``_pending_first``.

On the **next** call, for envs with ``_pending_first=True``:
  - ``is_first=True``, ``reward=0`` (the obs is the post-reset obs that
    IsaacLab already computed — this is the initial obs of the new episode).

This matches the data flow of ``ParallelEnv`` (DMC): the terminal step
carries the true terminal obs + reward, and the following step carries
``is_first=True`` with the reset obs and ``reward=0``.

The initial reset obs (obs_0) from the very first episode is effectively
"lost" — it is consumed by the ``is_first=True`` step that carries
``reward=0``.  This is acceptable since the first step has zero reward
anyway.

Usage::

    vec_env = make_isaac_env("Isaac-Cartpole-v0", env_cfg)
    # vec_env now satisfies the ParallelEnv interface expected by OnlineTrainer.
"""

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict


class IsaacLabVecEnv:
    """Wraps a vectorized IsaacLab env for use with r2dreamer.

    All data remains on the GPU throughout.  The wrapper tracks ``is_first``,
    ``is_last``, and ``is_terminal`` from IsaacLab's ``terminated``/
    ``truncated`` signals and injects them into the returned ``TensorDict``.

    The inner env **must** populate ``extras["terminal_obs"]`` and
    ``extras["terminal_env_ids"]`` so that the true terminal observation can
    be returned on the step an episode ends.  Both ``R2DreamerRLEnv``
    (ManagerBased) and ``R2DreamerDirectRLEnv`` (Direct) provide this.

    Parameters
    ----------
    env:
        An unwrapped IsaacLab env that captures terminal observations
        before auto-reset (e.g. ``R2DreamerRLEnv``,
        ``R2DreamerDirectRLEnv``, or any subclass that populates
        ``extras["terminal_obs"]``).
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
            Float tensor of shape ``(B, A)`` **on any device** — passed
            directly to the IsaacLab env which expects a GPU tensor.
        done:
            Bool tensor of shape ``(B,)`` — accepted for API compatibility
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
        #
        # The inner env (R2DreamerRLEnv / R2DreamerDirectRLEnv) captured obs
        # *before* the auto-reset and placed them in extras["terminal_obs"].
        # IsaacLab's
        # obs_dict currently holds the post-reset obs for these envs —
        # replace them with the true terminal obs.
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
        # Also True on the very first call after __init__/reset().
        # NOTE: For envs with _pending_first, the obs is the post-reset obs
        # returned by IsaacLab (obs_0 of new episode).  If an env has BOTH
        # _pending_first AND episode_done on the same step (episode lasted
        # exactly 1 step after reset), we handle that below.
        is_first_now = self._is_first | self._pending_first

        # For pending_first envs: is_first=True takes priority.  If the env
        # also terminated this step (episode_done=True), the terminal obs
        # swap above already placed the terminal obs in data.  But we need
        # the is_first row to carry the RESET obs, not the terminal obs.
        # Since this is an extremely rare edge case (1-step episode right
        # after reset) and the terminal obs was already captured, we leave
        # the terminal obs in place — the world model sees is_first=True
        # which resets the latent state, and the terminal reward from the
        # previous action is captured in the reward.
        #
        # However, we must NOT fire is_first on the SAME step as episode_done
        # for envs that are finishing normally (not pending).  is_first fires
        # on the NEXT step for those.
        data["is_first"] = is_first_now.unsqueeze(-1)

        # is_terminal: True only for actual termination (not truncation).
        data["is_terminal"] = terminated.unsqueeze(-1)

        # is_last: True whenever the episode ended for any reason.
        data["is_last"] = episode_done.unsqueeze(-1)

        # reward: Keep the terminal reward for done envs (no longer zeroed!).
        # For pending_first envs (showing the first obs of a new episode),
        # zero the reward — the reset obs carries no meaningful reward.
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
