"""Device-aware environment stepping adapters.

The ``OnlineTrainer`` training loop delegates all environment interaction and
device-transfer logic to an ``EnvStepper``.  Two concrete implementations are
provided:

* **CpuEnvStepper** — wraps ``ParallelEnv`` (subprocess workers on CPU).
  Actions are moved to CPU before stepping, observations are pinned and
  transferred to GPU asynchronously (H2D ``non_blocking=True``).

* **GpuEnvStepper** — wraps ``IsaacLabVecEnv`` (GPU-resident simulation).
  All tensors stay on the same CUDA device; no copies are needed.
"""

from __future__ import annotations

from typing import Protocol, Tuple

import torch
from tensordict import TensorDict


class EnvStepper(Protocol):
    """Minimal interface that ``OnlineTrainer`` requires from an env stepper."""

    @property
    def env_num(self) -> int:
        """Number of parallel environments."""
        ...

    def count_active_steps(self, done: torch.Tensor) -> int:
        """Return the number of environments that are actively stepping.

        Parameters
        ----------
        done:
            Bool tensor ``(B,)`` — True for environments whose episode just ended.

        Returns
        -------
        int
            Number of active (non-done) environments.
        """
        ...

    def reset(self) -> None:
        """Reset all environments to the start of a new episode."""
        ...

    def step(self, action: torch.Tensor, done: torch.Tensor) -> tuple[TensorDict, torch.Tensor]:
        """Step all environments.

        Parameters
        ----------
        action:
            Float tensor ``(B, A)`` on ``agent_device``.
        done:
            Bool tensor ``(B,)`` on ``agent_device``.

        Returns
        -------
        trans : TensorDict
            Transition data ``(B, ...)`` on ``agent_device``.
        done : torch.Tensor
            Bool tensor ``(B,)`` on ``agent_device``.
        """
        ...


# ---------------------------------------------------------------------------
# CPU subprocess environments (e.g. DMC / Atari via ParallelEnv)
# ---------------------------------------------------------------------------


class CpuEnvStepper:
    """Steps ``ParallelEnv`` on CPU and transfers results to GPU.

    GPU action -> CPU -> env.step() -> pinned CPU tensor -> GPU (non_blocking).
    """

    def __init__(self, env, agent_device: torch.device):
        self._env = env
        self._agent_device = torch.device(agent_device)

    @property
    def env_num(self) -> int:
        return self._env.env_num

    def count_active_steps(self, done: torch.Tensor) -> int:
        """Exact count — triggers a GPU→CPU sync via ``.item()``."""
        return int((~done).sum())

    def reset(self) -> None:
        """No-op — ParallelEnv resets via the ``done`` flag passed to ``step()``."""
        pass

    def step(self, action: torch.Tensor, done: torch.Tensor) -> tuple[TensorDict, torch.Tensor]:
        # Step environments on CPU
        # (B, A)
        act_cpu = action.detach().to("cpu")
        # (B,)
        done_cpu = done.detach().to("cpu")
        trans_cpu, done_cpu = self._env.step(act_cpu, done_cpu)
        # Move observations back to GPU asynchronously for agent.
        # dict of (B, 1, *)
        trans = trans_cpu.to(self._agent_device, non_blocking=True)
        # (B,)
        done = done_cpu.to(self._agent_device)

        return trans, done


# ---------------------------------------------------------------------------
# GPU-native environments (e.g. IsaacLab)
# ---------------------------------------------------------------------------


class GpuEnvStepper:
    """Steps a GPU-resident env (``IsaacLabVecEnv``) without any device transfers.

    Both input and output tensors stay on the simulation device (CUDA).
    The ``done`` tensor is forwarded for API compatibility but IsaacLab
    manages auto-resets internally, the ``is_first`` flag in the returned
    ``TensorDict`` is the authoritative reset signal.
    """

    def __init__(self, env, agent_device: torch.device):
        self._env = env
        self._agent_device = torch.device(agent_device)

    @property
    def env_num(self) -> int:
        return self._env.env_num

    def count_active_steps(self, done: torch.Tensor) -> int:
        """All envs are always active (IsaacLab auto-resets instantly)."""
        return self._env.env_num

    def reset(self) -> None:
        """Force-reset all IsaacLab environments to episode start."""
        self._env.reset()

    def step(self, action: torch.Tensor, done: torch.Tensor) -> tuple[TensorDict, torch.Tensor]:
        assert action.device.type == "cuda", f"GpuEnvStepper expects a CUDA action tensor, got {action.device}"
        # Pass tensors directly — no CPU round-trip.
        trans, done = self._env.step(action, done)
        # If the env device differs from the agent device (unusual), move.
        if trans.device != self._agent_device:
            trans = trans.to(self._agent_device, non_blocking=True)
            done = done.to(self._agent_device)
        return trans, done
