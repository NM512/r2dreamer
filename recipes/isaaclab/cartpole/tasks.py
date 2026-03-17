"""Cartpole task definitions for the IsaacLab r2dreamer recipe.

Defines four cartpole variants covering all combinations of env type
(ManagerBased vs Direct) and behaviour (stock IsaacLab vs DMC-exact):

  - ``cartpole_balance``            ManagerBased, stock IsaacLab
  - ``cartpole_balance_dmc``        ManagerBased, DMC-exact overrides
  - ``cartpole_balance_direct``     Direct, stock IsaacLab
  - ``cartpole_balance_direct_dmc`` Direct, DMC-exact monkey-patches

The stock IsaacLab cartpole envs don't inherit from ``R2DreamerRLEnv`` /
``R2DreamerDirectRLEnv``, so the helper ``_make_env`` below handles
instantiation and MRO patching.  For your own tasks, inherit from those
classes directly and you won't need any of this machinery.
"""

import functools

import train_isaaclab


# Known shorthand tasks.
# Maps shorthand -> {vision: gym_id, proprio: gym_id}.
KNOWN_TASKS = {
    # ManagerBased cartpole -- stock IsaacLab rewards/obs/terminations.
    "cartpole_balance": {
        "proprio": "Isaac-Cartpole-v0",
        "vision": "Isaac-Cartpole-RGB-v0",
    },
    # ManagerBased cartpole -- DMC-exact overrides via manager config.
    "cartpole_balance_dmc": {
        "proprio": "Isaac-Cartpole-v0",
        "vision": "Isaac-Cartpole-RGB-v0",
    },
    # Direct cartpole -- stock IsaacLab behaviour.
    "cartpole_balance_direct": {
        "proprio": "Isaac-Cartpole-Direct-v0",
        "vision": "Isaac-Cartpole-RGB-Camera-Direct-v0",
    },
    # Direct cartpole -- DMC-exact overrides via monkey-patching.
    "cartpole_balance_direct_dmc": {
        "proprio": "Isaac-Cartpole-Direct-v0",
        "vision": "Isaac-Cartpole-RGB-Camera-Direct-v0",
    },
}


# =============================================================================
# Env construction helpers for stock IsaacLab envs
# =============================================================================
#
# The stock IsaacLab cartpole envs don't inherit from R2DreamerRLEnv /
# R2DreamerDirectRLEnv, so we need to patch them.  If you write your own
# task and inherit from those classes directly, you can skip all of this
# and just instantiate + wrap with make_isaac_env().


def _patch_direct_env(unwrapped):
    """Inject ``R2DreamerDirectRLEnv`` into a third-party Direct env's MRO.

    Use this for Direct envs created via ``gym.make`` whose class you don't
    control (e.g. IsaacLab's built-in cartpole).  For your own Direct envs,
    inherit from ``R2DreamerDirectRLEnv`` directly instead.
    """
    from envs.isaaclab import R2DreamerDirectRLEnv

    ConcreteClass = type(unwrapped)
    PatchedClass = type(
        f"R2Dreamer{ConcreteClass.__name__}",
        (R2DreamerDirectRLEnv, ConcreteClass),
        {},
    )
    unwrapped.__class__ = PatchedClass


def _cartpole_camera_cfg(env_cfg, height, width):
    """Configure camera resolution, position, and rendering for cartpole.

    Sets camera resolution, positions the camera for the cartpole scene,
    and disables antialiasing.
    """
    _cam_pos = (-3.4, 0.0, 2.0)
    _cam_rot = (1.0, 0.0, 0.0, 0.0)  # identity -- look straight forward
    if hasattr(env_cfg, "tiled_camera"):
        # DirectRLEnv -- camera is a top-level env_cfg attribute.
        env_cfg.tiled_camera.width = width
        env_cfg.tiled_camera.height = height
        env_cfg.tiled_camera.offset.pos = _cam_pos
        env_cfg.tiled_camera.offset.rot = _cam_rot
    elif hasattr(env_cfg.scene, "tiled_camera"):
        # ManagerBased -- camera lives in the scene config.
        env_cfg.scene.tiled_camera.width = width
        env_cfg.scene.tiled_camera.height = height
        env_cfg.scene.tiled_camera.offset.pos = _cam_pos
        env_cfg.scene.tiled_camera.offset.rot = _cam_rot

    from isaaclab.sim import RenderCfg

    env_cfg.sim.render = RenderCfg(antialiasing_mode="Off")


def _make_env(config, gym_id, render_mode, simulation_app, pre_wrap_fns=(), post_create_fn=None, env_cfg_fns=()):
    """Construct a GPU-resident IsaacLab env from a stock gym ID.

    Args:
        config: Hydra env config with env_num, action_repeat, seed, etc.
        gym_id: Gymnasium env ID string.
        render_mode: ``"rgb_array"`` for vision, ``None`` for proprio.
        simulation_app: The IsaacLab simulation app instance.
        pre_wrap_fns: Callables applied to the unwrapped env before wrapping.
        post_create_fn: Callable applied after env creation.
        env_cfg_fns: Callables applied to env_cfg before instantiation.
            Use for task-specific overrides (camera, observations, ...).
    """
    import gymnasium as gym

    from envs import make_isaac_env
    from envs.isaaclab import R2DreamerRLEnv

    env_cfg, spec = train_isaaclab.make_env_cfg(config, gym_id)

    for fn in env_cfg_fns:
        fn(env_cfg)

    # Construct the unwrapped env with terminal-obs capture.
    is_manager_based = "ManagerBasedRLEnv" in str(spec.entry_point)
    if is_manager_based:
        unwrapped = R2DreamerRLEnv(cfg=env_cfg, render_mode=render_mode)
    else:
        # Third-party Direct env -- create via gym.make, then patch.
        isaac_env = gym.make(gym_id, cfg=env_cfg, render_mode=render_mode)
        unwrapped = isaac_env.unwrapped
        _patch_direct_env(unwrapped)

    for fn in pre_wrap_fns:
        fn(unwrapped)

    if post_create_fn is not None:
        post_create_fn(unwrapped)

    return make_isaac_env(
        unwrapped,
        simulation_app=simulation_app,
    )


# =============================================================================
# Task builder functions
# =============================================================================
#
# Each builder has the signature: builder(config, vision, simulation_app) -> vec_env


def _build_cartpole_env(config, vision, simulation_app):
    """Build a ManagerBased cartpole env (stock IsaacLab behaviour)."""
    ids = KNOWN_TASKS["cartpole_balance"]
    gym_id = ids["vision"] if vision else ids["proprio"]
    render_mode = "rgb_array" if vision else None

    env_cfg_fns = []
    if vision:
        env_cfg_fns.append(functools.partial(
            _cartpole_camera_cfg, height=config.size[0], width=config.size[1],
        ))
        env_cfg_fns.append(_cartpole_dreamer_vision_cfg_overrides)

    return _make_env(config, gym_id, render_mode, simulation_app, env_cfg_fns=env_cfg_fns)

def _build_cartpole_direct_env(config, vision, simulation_app):
    """Build a Direct cartpole env (stock IsaacLab behaviour)."""
    ids = KNOWN_TASKS["cartpole_balance_direct"]
    gym_id = ids["vision"] if vision else ids["proprio"]
    render_mode = "rgb_array" if vision else None

    env_cfg_fns = []
    if vision:
        env_cfg_fns.append(functools.partial(
            _cartpole_camera_cfg, height=config.size[0], width=config.size[1],
        ))

    return _make_env(config, gym_id, render_mode, simulation_app, env_cfg_fns=env_cfg_fns)


def _build_cartpole_direct_dmc_env(config, vision, simulation_app):
    """Build a Direct cartpole env with DMC-style monkey-patch overrides."""
    from cartpole.dmc_overrides import (
        apply_dmc_cartpole_colors,
        patch_dmc_cartpole_obs,
        patch_dmc_cartpole_reset,
        patch_dmc_cartpole_reward,
        patch_no_termination,
    )

    ids = KNOWN_TASKS["cartpole_balance_direct_dmc"]
    gym_id = ids["vision"] if vision else ids["proprio"]
    render_mode = "rgb_array" if vision else None
    post_create_fn = apply_dmc_cartpole_colors if vision else None

    pre_wrap_fns = [
        patch_no_termination,
        functools.partial(patch_dmc_cartpole_reward, action_repeat=int(config.action_repeat)),
        patch_dmc_cartpole_obs,
        patch_dmc_cartpole_reset,
    ]

    env_cfg_fns = []
    if vision:
        env_cfg_fns.append(functools.partial(
            _cartpole_camera_cfg, height=config.size[0], width=config.size[1],
        ))

    return _make_env(
        config,
        gym_id,
        render_mode,
        simulation_app,
        pre_wrap_fns,
        post_create_fn,
        env_cfg_fns=env_cfg_fns,
    )


def _build_cartpole_dmc_env(config, vision, simulation_app):
    """Build a ManagerBased cartpole env with DMC-style overrides via config."""
    from cartpole.dmc_overrides import apply_dmc_cartpole_colors

    ids = KNOWN_TASKS["cartpole_balance_dmc"]
    gym_id = ids["vision"] if vision else ids["proprio"]
    render_mode = "rgb_array" if vision else None
    post_create_fn = apply_dmc_cartpole_colors if vision else None

    env_cfg_fns = []
    if vision:
        env_cfg_fns.append(functools.partial(
            _cartpole_camera_cfg, height=config.size[0], width=config.size[1],
        ))
    env_cfg_fns.append(functools.partial(
        _cartpole_dmc_cfg_overrides,
        vision=vision,
        action_repeat=int(config.action_repeat),
    ))

    return _make_env(
        config,
        gym_id,
        render_mode,
        simulation_app,
        post_create_fn=post_create_fn,
        env_cfg_fns=env_cfg_fns,
    )


# =============================================================================
# Config override functions
# =============================================================================


def _cartpole_dmc_cfg_overrides(env_cfg, vision, action_repeat):
    """Override ManagerBased cartpole config to replicate DMC behaviour.

    Replaces observations, rewards, terminations, and events on the env_cfg
    so that a stock ManagerBased cartpole produces the exact same behaviour
    as the DMC cartpole balance task.
    """
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.utils import configclass

    from cartpole.dmc_overrides import dmc_balance_reward, dmc_cartpole_obs, reset_dmc_cartpole_state
    from isaaclab_tasks.manager_based.classic.cartpole.mdp import image as _image_fn
    from isaaclab_tasks.manager_based.classic.cartpole.mdp import time_out as _time_out_fn

    _robot_cfg = SceneEntityCfg("robot", joint_names=["slider_to_cart", "cart_to_pole"])

    # -- Observations: DMC 5D state --
    @configclass
    class _DmcPolicyObs(ObsGroup):
        dmc_obs = ObsTerm(func=dmc_cartpole_obs, params={"asset_cfg": _robot_cfg})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    if vision:

        @configclass
        class _DmcImageObs(ObsGroup):
            image = ObsTerm(
                func=_image_fn,
                params={
                    "sensor_cfg": SceneEntityCfg("tiled_camera"),
                    "data_type": "rgb",
                    "normalize": False,
                },
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        @configclass
        class _DmcObsCfg:
            policy: ObsGroup = _DmcPolicyObs()
            image: ObsGroup = _DmcImageObs()

    else:

        @configclass
        class _DmcObsCfg:
            policy: ObsGroup = _DmcPolicyObs()

    env_cfg.observations = _DmcObsCfg()

    # -- Rewards: single DMC balance reward --
    @configclass
    class _DmcRewardsCfg:
        dmc_balance = RewTerm(
            func=dmc_balance_reward,
            weight=1.0,
            params={
                "asset_cfg": _robot_cfg,
                "action_repeat": action_repeat,
            },
        )

    env_cfg.rewards = _DmcRewardsCfg()

    # -- Terminations: time-only (no cart_out_of_bounds) --
    @configclass
    class _DmcTerminationsCfg:
        time_out = DoneTerm(func=_time_out_fn, time_out=True)

    env_cfg.terminations = _DmcTerminationsCfg()

    # -- Events: DMC initial state distribution --
    @configclass
    class _DmcEventsCfg:
        reset_cart_pole = EventTerm(
            func=reset_dmc_cartpole_state,
            mode="reset",
            params={"asset_cfg": _robot_cfg},
        )

    env_cfg.events = _DmcEventsCfg()


def _cartpole_dreamer_vision_cfg_overrides(env_cfg):
    """Override the ManagerBased RGB cartpole obs layout for Dreamer.

    The stock ``CartpoleRGBCameraEnvCfg`` puts the camera image under a
    single ``"policy"`` observation group with ``concatenate_terms=True``,
    which flattens the image into 1-D.  Dreamer expects a separate
    ``"image"`` key with shape ``(H, W, C)``.

    This function replaces the observation config with two groups:
      - ``policy``: state observations (joint_pos_rel + joint_vel_rel)
      - ``image``:  raw RGB camera image (single term with
        ``concatenate_terms=True`` -- a no-op that returns the raw
        ``(B, H, W, C)`` tensor without flattening)
    """
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils import configclass

    from isaaclab_tasks.manager_based.classic.cartpole.mdp import image as _image_fn
    from isaaclab_tasks.manager_based.classic.cartpole.mdp import joint_pos_rel, joint_vel_rel

    @configclass
    class _PolicyObs(ObsGroup):
        joint_pos_rel = ObsTerm(func=joint_pos_rel)
        joint_vel_rel = ObsTerm(func=joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class _ImageObs(ObsGroup):
        image = ObsTerm(
            func=_image_fn,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class _DreamerObsCfg:
        policy: ObsGroup = _PolicyObs()
        image: ObsGroup = _ImageObs()

    env_cfg.observations = _DreamerObsCfg()


# Map shorthand task name -> builder(env_config, vision, simulation_app).
CARTPOLE_TASK_BUILDERS = {
    "cartpole_balance": _build_cartpole_env,
    "cartpole_balance_dmc": _build_cartpole_dmc_env,
    "cartpole_balance_direct": _build_cartpole_direct_env,
    "cartpole_balance_direct_dmc": _build_cartpole_direct_dmc_env,
}
