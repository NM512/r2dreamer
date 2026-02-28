"""DMC-compatible overrides for the IsaacLab cartpole environment.

These functions adapt the stock IsaacLab cartpole task to match the
DeepMind Control Suite cartpole as closely as possible, for benchmarking
purposes.  They monkey-patch the env **before** it is wrapped by the
generic ``IsaacLabVecEnv``.

Overrides applied:
  - Reward:       exact DMC balance (smooth) reward instead of the native one
  - Observations: remap to DMC format (position=[cart_x, cos θ, sin θ],
                  velocity=[cart_vel, pole_vel])
  - Image source: read raw uint8 from tiled camera buffer (bypass env's
                  normalisation that corrupts the uint8 preprocess)
  - Termination:  suppress early terminations (time-only truncation, like DMC)
  - Reset:        match DMC balance initial state distribution (cart pos,
                  pole angle, velocities)
  - Physics:      match DMC actuator damping at runtime (cart slider 5e-4,
                  pole hinge ~0)
  - Visuals:      match DMC scene colours for the RGB camera variant
"""

import math
import types

import numpy as np
import torch


# ---------------------------------------------------------------------------
# DMC reward helpers
# ---------------------------------------------------------------------------


def _torch_gaussian_tolerance(x, bounds=(0.0, 0.0), margin=0.0, value_at_margin=0.1):
    """Torch equivalent of dm_control's rewards.tolerance with gaussian sigmoid.

    Returns 1 when x falls inside bounds, decays smoothly outside.
    """
    lower, upper = bounds
    in_bounds = (x >= lower) & (x <= upper)
    if margin == 0:
        return torch.where(in_bounds, 1.0, 0.0)
    d = torch.where(x < lower, lower - x, x - upper) / margin
    scale = math.sqrt(-2 * math.log(value_at_margin))
    value = torch.exp(-0.5 * (d * scale) ** 2)
    return torch.where(in_bounds, 1.0, value)


def _torch_quadratic_tolerance(x, margin=1.0, value_at_margin=0.0):
    """Torch equivalent of dm_control's tolerance with quadratic sigmoid.

    Returns 1 at x==0, decays quadratically, reaches 0 at margin (when
    value_at_margin=0).
    """
    scale = math.sqrt(1.0 - value_at_margin)
    d = torch.abs(x) / margin
    scaled = d * scale
    return torch.where(scaled.abs() < 1.0, 1.0 - scaled**2, torch.zeros_like(x))


def _compute_dmc_balance_reward(pole_angle, pole_ang_vel, cart_pos, action):
    """Compute the exact DMC cartpole balance (smooth) reward in torch.

    Replicates dm_control/suite/cartpole.py Balance._get_reward(sparse=False):
        upright = (cos(pole_angle) + 1) / 2
        centered = (1 + tolerance(cart_pos, margin=2)) / 2
        small_control = (4 + tolerance(action, margin=1, v@m=0, quadratic)) / 5
        small_velocity = (1 + tolerance(ang_vel, margin=5)) / 2
        reward = upright * centered * small_control * small_velocity

    All inputs are (num_envs,) tensors.
    Returns (num_envs,) tensor with reward in [0, 1].
    """
    upright = (torch.cos(pole_angle) + 1.0) / 2.0

    centered = _torch_gaussian_tolerance(cart_pos, bounds=(0.0, 0.0), margin=2.0)
    centered = (1.0 + centered) / 2.0

    small_control = _torch_quadratic_tolerance(action, margin=1.0, value_at_margin=0.0)
    small_control = (4.0 + small_control) / 5.0

    small_velocity = _torch_gaussian_tolerance(pole_ang_vel, bounds=(0.0, 0.0), margin=5.0)
    small_velocity = (1.0 + small_velocity) / 2.0

    return upright * centered * small_control * small_velocity


# ---------------------------------------------------------------------------
# Pre-construction env patches
# ---------------------------------------------------------------------------


def patch_dmc_cartpole_reward(env, action_repeat):
    """Monkey-patch ``_get_rewards()`` to produce the exact DMC balance reward.

    The patched method ignores the native reward and computes the DMC
    cartpole balance (smooth) reward from joint state.  The reward is
    multiplied by *action_repeat* to match DMC's per-sub-step accumulation.

    Args:
        env: unwrapped DirectRLEnv (IsaacLab cartpole).
        action_repeat: decimation factor; DMC accumulates reward over sub-steps.
    """

    def _dmc_get_rewards(self):
        pole_angle = self.joint_pos[:, self._pole_dof_idx[0]]
        pole_ang_vel = self.joint_vel[:, self._pole_dof_idx[0]]
        cart_pos = self.joint_pos[:, self._cart_dof_idx[0]]
        # Recover raw action in [-1, 1] from the scaled action stored by
        # _pre_physics_step (self.actions = action_scale * raw_action).
        raw_action = self.actions[:, 0] / self.cfg.action_scale
        reward = _compute_dmc_balance_reward(pole_angle, pole_ang_vel, cart_pos, raw_action)
        return reward * action_repeat

    env._get_rewards = types.MethodType(_dmc_get_rewards, env)


def patch_dmc_cartpole_obs(env):
    """Monkey-patch the env to produce DMC-format observations.

    Replaces ``_get_observations()`` so that ``obs_dict["policy"]`` is a
    5-D vector ``[cart_x, cos(θ), sin(θ), cart_vel, pole_vel]`` matching
    DMC's cartpole observation format.  Also patches
    ``single_observation_space`` to report the correct shape ``(5,)``.

    For the RGB-camera variant (env has ``_tiled_camera``), the raw uint8
    camera image is additionally returned under ``obs_dict["image"]`` so
    that ``IsaacLabVecEnv`` auto-detects it as a 3-D observation and
    exposes it to the agent under the standard ``"image"`` key.
    """
    import gymnasium

    original_get_obs = env._get_observations
    has_camera = hasattr(env, "_tiled_camera")

    def _dmc_get_observations(self):
        # Call original to ensure camera buffers etc. are updated
        original_get_obs()
        pole_angle = self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(-1)
        pole_vel = self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(-1)
        cart_pos = self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(-1)
        cart_vel = self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(-1)
        # DMC order: [cart_x, cos(θ), sin(θ), cart_vel, pole_vel]
        obs = torch.cat(
            [cart_pos, torch.cos(pole_angle), torch.sin(pole_angle), cart_vel, pole_vel],
            dim=-1,
        )
        result = {"policy": obs}
        if has_camera:
            # Raw uint8 RGB from the tiled camera, bypassing the env's
            # normalisation (which divides by 255 and subtracts spatial mean).
            result["image"] = self._tiled_camera.data.output["rgb"]
        return result

    env._get_observations = types.MethodType(_dmc_get_observations, env)

    # Patch single_observation_space to reflect the new obs layout
    spaces = {"policy": gymnasium.spaces.Box(-np.inf, np.inf, (5,), dtype=np.float32)}
    if has_camera:
        cam = env._tiled_camera
        spaces["image"] = gymnasium.spaces.Box(0, 255, (cam.image_shape[0], cam.image_shape[1], 3), dtype=np.uint8)
    env.single_observation_space = gymnasium.spaces.Dict(spaces)
    env.observation_space = gymnasium.vector.utils.batch_space(env.single_observation_space, env.num_envs)


def patch_no_termination(env):
    """Monkey-patch _get_dones so terminated is always False.

    This makes the env behave like DMC: episodes only end via time-based
    truncation (max_episode_length), never via early failure. The original
    _get_dones is kept to still compute time_out correctly.
    """
    original_get_dones = env._get_dones

    def _no_termination_get_dones(self):
        terminated, time_out = original_get_dones()
        terminated = torch.zeros_like(terminated)
        return terminated, time_out

    env._get_dones = types.MethodType(_no_termination_get_dones, env)


def patch_dmc_cartpole_reset(env):
    """Monkey-patch ``_reset_idx`` to match DMC balance initial state.

    DMC ``Balance(swing_up=False)`` initialises each episode as:
      - cart position:  Uniform(-0.1, 0.1)
      - pole angle:     Uniform(-0.034, 0.034) rad  (near-vertical)
      - cart velocity:  Normal(0, 0.01)
      - pole velocity:  Normal(0, 0.01)

    The stock IsaacLab cartpole uses:
      - cart position:  0
      - pole angle:     Uniform(-0.25π, 0.25π) ≈ ±45°
      - velocities:     0

    This patch replaces the reset to match DMC exactly.
    """
    original_reset_idx = env._reset_idx

    # CartpoleEnv uses ``self.cartpole``; CartpoleCameraEnv uses
    # ``self._cartpole``.  Resolve once at patch time so the same
    # override works for both variants.
    _cartpole_attr = "cartpole" if hasattr(env, "cartpole") else "_cartpole"

    def _dmc_reset_idx(self, env_ids):
        # Call the original to handle scene.reset, event_manager, episode buf, etc.
        original_reset_idx(env_ids)

        cartpole = getattr(self, _cartpole_attr)
        n = len(env_ids)
        device = self.device

        # --- joint positions ---
        joint_pos = cartpole.data.default_joint_pos[env_ids].clone()
        # Cart position: Uniform(-0.1, 0.1)  — DMC balance init
        joint_pos[:, self._cart_dof_idx[0]] = torch.rand(n, device=device) * 0.2 - 0.1
        # Pole angle: Uniform(-0.034, 0.034) rad  — DMC near-vertical
        joint_pos[:, self._pole_dof_idx[0]] = torch.rand(n, device=device) * 0.068 - 0.034

        # --- joint velocities ---
        joint_vel = cartpole.data.default_joint_vel[env_ids].clone()
        # Both cart and pole: Normal(0, 0.01)  — DMC symmetry-breaking perturbation
        joint_vel[:, self._cart_dof_idx[0]] = torch.randn(n, device=device) * 0.01
        joint_vel[:, self._pole_dof_idx[0]] = torch.randn(n, device=device) * 0.01

        # --- root state (unchanged from default, just offset by env origins) ---
        default_root_state = cartpole.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Write back to simulation
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    env._reset_idx = types.MethodType(_dmc_reset_idx, env)


def apply_dmc_cartpole_colors(env):
    """Override cart/pole/slider/light colors to match DMC cartpole visuals.

    Must be called after the scene has been created and the simulation
    has been started (i.e. after ``gym.make``). Only touches env_0's
    prims — replicate_physics mirrors them to all other envs.

    DMC cartpole colours (linear RGB):
      - cart & pole ("self" material):     (0.89, 0.65, 0.41)  warm brown
      - slider / rail ("decoration"):      (0.24, 0.47, 0.61)  steel blue
      - dome light → approximate sky:      (0.18, 0.28, 0.37)  blue sky
      - ground plane:                      (0.04, 0.20, 0.31) dark blue-grey
    """
    import isaaclab.sim as sim_utils

    # ---- create materials under /World/Looks ----
    self_mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.89, 0.65, 0.41), roughness=0.6, metallic=0.0)
    deco_mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.24, 0.47, 0.61), roughness=0.6, metallic=0.0)
    self_mat_path = "/World/Looks/DmcSelf"
    deco_mat_path = "/World/Looks/DmcDecoration"

    sim_utils.spawn_preview_surface(self_mat_path, self_mat_cfg)
    sim_utils.spawn_preview_surface(deco_mat_path, deco_mat_cfg)

    # ---- bind materials to cart, pole, slider ----
    env0 = "/World/envs/env_0/Robot"
    for part in ("cart", "pole"):
        sim_utils.bind_visual_material(f"{env0}/{part}", self_mat_path, stronger_than_descendants=True)
    sim_utils.bind_visual_material(f"{env0}/slider", deco_mat_path, stronger_than_descendants=True)

    # ---- dome light → DMC-like sky colour ----
    stage = sim_utils.get_current_stage()
    light_prim = stage.GetPrimAtPath("/World/Light")
    if light_prim.IsValid():
        from pxr import Gf

        light_prim.GetAttribute("inputs:color").Set(Gf.Vec3f(0.18, 0.28, 0.37))
        light_prim.GetAttribute("inputs:intensity").Set(500.0)
        vis_attr = light_prim.GetAttribute("visibleInPrimaryRay")
        if vis_attr:
            vis_attr.Set(True)

    # ---- ground plane ----
    ground_cfg = sim_utils.CuboidCfg(
        size=(40.0, 40.0, 0.01),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.04, 0.20, 0.31),
            roughness=0.8,
            metallic=0.0,
        ),
    )
    ground_cfg.func(
        "/World/DmcGround",
        ground_cfg,
        translation=(0.0, 0.0, -0.005),
    )
