"""ManagerBasedRLEnv and DirectRLEnv subclasses that capture terminal observations.

IsaacLab's ``ManagerBasedRLEnv.step()`` and ``DirectRLEnv.step()`` both
auto-reset terminated environments *before* computing the final observations,
so the returned obs for done envs is the post-reset obs of the **new**
episode — the true terminal obs is lost.

These subclasses override ``_reset_idx`` to insert an observation computation
**before** the reset, storing the result in ``self.extras["terminal_obs"]``.
A downstream wrapper (e.g. ``IsaacLabVecEnv``) can then use it to return the
correct terminal observation to the agent/buffer.

The override is intentionally minimal (~10 lines per class) to stay resilient
to base-class changes across IsaacLab versions.  All logic delegates to
the base class rather than duplicating the full ``step()`` body.

For Direct envs, ``make_isaac_env`` dynamically inserts
``R2DreamerDirectRLEnv`` into the instance's class hierarchy so that
``super()`` resolves correctly through the MRO.
"""

from __future__ import annotations

from typing import Sequence

import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn


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
            self.extras["terminal_obs"] = {
                key: val[env_ids].clone() for key, val in terminal_obs.items()
            }
            self.extras["terminal_env_ids"] = env_ids
        super()._reset_idx(env_ids)


class R2DreamerDirectRLEnv(DirectRLEnv):
    """DirectRLEnv with pre-reset terminal observation capture.

    This class is abstract (inherits abstract methods from ``DirectRLEnv``).
    It is not meant to be instantiated directly.  Instead,
    ``make_isaac_env`` monkey-patches its ``step`` and ``_reset_idx``
    methods onto concrete Direct env instances created via ``gym.make``.
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        self.extras.pop("terminal_obs", None)
        self.extras.pop("terminal_env_ids", None)
        return super().step(action)

    def _reset_idx(self, env_ids: Sequence[int]):
        if len(env_ids) > 0:
            terminal_obs = self._get_observations()
            self.extras["terminal_obs"] = {
                key: val[env_ids].clone() for key, val in terminal_obs.items()
            }
            self.extras["terminal_env_ids"] = env_ids
        super()._reset_idx(env_ids)
