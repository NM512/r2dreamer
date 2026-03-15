"""ManagerBased term functions that replicate DMC cartpole behavior.

These functions follow IsaacLab's manager term API signatures and can be
used as ``func`` arguments in ``RewardTermCfg``, ``ObservationTermCfg``,
and ``EventTermCfg`` configs.  They allow a ManagerBased cartpole env
to produce the exact same reward, observations, and reset distribution
as the DeepMind Control Suite cartpole balance task.

The core reward math is shared with the Direct-env monkey patches in
``isaac_cartpole_overrides.py`` — the helpers ``_compute_dmc_balance_reward``,
``_torch_gaussian_tolerance``, and ``_torch_quadratic_tolerance`` are
imported from there to avoid duplication.
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg

from envs.isaac_cartpole_overrides import _compute_dmc_balance_reward


def dmc_balance_reward(
    env,
    asset_cfg: SceneEntityCfg,
    action_repeat: int,
) -> torch.Tensor:
    """DMC cartpole balance (smooth) reward for ManagerBased envs.

    Returns a per-env reward tensor of shape ``(num_envs,)``.

    The reward manager multiplies every term by ``dt`` (= ``step_dt``).
    The Direct env returns ``reward * action_repeat`` directly from
    ``_get_rewards()``.  To produce the same per-step total we return
    ``reward * action_repeat / step_dt`` so that after the ``* dt``
    multiplication the result is ``reward * action_repeat``.

    Config usage::

        RewTerm(func=dmc_balance_reward, weight=1.0, params={
            "asset_cfg": SceneEntityCfg("robot",
                joint_names=["slider_to_cart", "cart_to_pole"]),
            "action_repeat": 2,
        })
    """
    asset = env.scene[asset_cfg.name]
    # joint_ids may be a slice when joints are contiguous; index the data
    # first, then pick columns by position (0=cart, 1=pole).
    jpos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    jvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    cart_pos = jpos[:, 0]
    pole_angle = jpos[:, 1]
    pole_ang_vel = jvel[:, 1]
    # action_manager.action is the raw policy output (before the
    # JointEffortAction term scales it by action_scale).  This is
    # already in [-1, 1] — no need to divide by action_scale.
    raw_action = env.action_manager.action[:, 0]
    reward = _compute_dmc_balance_reward(pole_angle, pole_ang_vel, cart_pos, raw_action)
    # Compensate for the reward manager's ``* dt`` so the per-step
    # total matches the Direct env: reward * action_repeat.
    return reward * action_repeat / env.step_dt


def dmc_cartpole_obs(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """DMC-format cartpole observation: ``[cart_x, cos(θ), sin(θ), cart_vel, pole_vel]``.

    Returns a tensor of shape ``(num_envs, 5)``.

    Config usage::

        ObsTerm(func=dmc_cartpole_obs, params={
            "asset_cfg": SceneEntityCfg("robot",
                joint_names=["slider_to_cart", "cart_to_pole"]),
        })
    """
    asset = env.scene[asset_cfg.name]
    jpos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    jvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    cart_pos = jpos[:, 0:1]
    pole_angle = jpos[:, 1:2]
    cart_vel = jvel[:, 0:1]
    pole_vel = jvel[:, 1:2]
    return torch.cat(
        [cart_pos, torch.cos(pole_angle), torch.sin(pole_angle), cart_vel, pole_vel],
        dim=-1,
    )


def reset_dmc_cartpole_state(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Reset cartpole joints to the DMC balance initial-state distribution.

    DMC ``Balance(swing_up=False)`` initialises each episode as:
      - cart position:  ``Uniform(-0.1, 0.1)``
      - pole angle:     ``Uniform(-0.034, 0.034)`` rad (near-vertical)
      - cart velocity:  ``Normal(0, 0.01)``
      - pole velocity:  ``Normal(0, 0.01)``

    Config usage::

        EventTerm(func=reset_dmc_cartpole_state, mode="reset", params={
            "asset_cfg": SceneEntityCfg("robot",
                joint_names=["slider_to_cart", "cart_to_pole"]),
        })
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    n = len(env_ids)
    device = env.device

    # Cart position: Uniform(-0.1, 0.1)
    joint_pos[:, 0] = torch.rand(n, device=device) * 0.2 - 0.1
    # Pole angle: Uniform(-0.034, 0.034) rad
    joint_pos[:, 1] = torch.rand(n, device=device) * 0.068 - 0.034
    # Cart velocity: Normal(0, 0.01)
    joint_vel[:, 0] = torch.randn(n, device=device) * 0.01
    # Pole velocity: Normal(0, 0.01)
    joint_vel[:, 1] = torch.randn(n, device=device) * 0.01

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
