"""IsaacLab entry-point for r2dreamer.

This script serves two purposes:

  1. **Reproducibility** — runnable benchmark for the IsaacLab cartpole
     tasks that ship with IsaacLab, so reviewers and other users can
     reproduce results without any additional setup.

  2. **Template** — shows the exact pattern to follow when writing your
     own IsaacLab entry-point for a different task or downstream project.
     Copy this file, add your task to ``TASK_BUILDERS``, and you're done.
     The r2dreamer library (``envs/``, ``dreamer.py``, ``trainer.py``, …)
     does not need to be modified.

IsaacLab requires ``AppLauncher`` to be called before any Isaac/USD imports,
so this script uses a mandatory two-phase import pattern:

  Phase 1 — parse CLI args, launch AppLauncher, get simulation_app.
  Phase 2 — all other imports (torch, hydra, dreamer modules, …).

The CLI syntax mirrors the other r2dreamer benchmarks::

    python train_isaaclab.py env=isaaclab_<obs_mode> env.task=isaaclab_<task_name>

Usage
-----
Four cartpole variants are provided, covering all combinations of env type
(ManagerBased vs Direct) and behaviour (stock IsaacLab vs DMC-exact):

ManagerBased cartpole — stock IsaacLab rewards/obs/terminations::

    python train_isaaclab.py env=isaaclab_proprio env.task=isaaclab_cartpole_balance
    python train_isaaclab.py env=isaaclab_vision  env.task=isaaclab_cartpole_balance

ManagerBased cartpole — DMC-exact overrides (reward, obs, reset, time-only termination)::

    python train_isaaclab.py env=isaaclab_proprio env.task=isaaclab_cartpole_balance_dmc
    python train_isaaclab.py env=isaaclab_vision  env.task=isaaclab_cartpole_balance_dmc

Direct cartpole — stock IsaacLab behaviour::

    python train_isaaclab.py env=isaaclab_proprio env.task=isaaclab_cartpole_balance_direct
    python train_isaaclab.py env=isaaclab_vision  env.task=isaaclab_cartpole_balance_direct

Direct cartpole — DMC-exact overrides (monkey-patched reward, obs, reset, no termination)::

    python train_isaaclab.py env=isaaclab_proprio env.task=isaaclab_cartpole_balance_direct_dmc
    python train_isaaclab.py env=isaaclab_vision  env.task=isaaclab_cartpole_balance_direct_dmc

Any additional Hydra overrides can follow as usual::

    python train_isaaclab.py env=isaaclab_proprio \\
        env.task=isaaclab_cartpole_balance device=cuda:0 seed=42

Monitor with TensorBoard::

    tensorboard --logdir logdir/
"""

# =============================================================================
# Phase 1: AppLauncher must run before any IsaacLab / USD / warp imports.
# =============================================================================

import argparse
import pathlib
import sys

# Auto-detect vision env from CLI and enable cameras before AppLauncher
# parses argv, so the user doesn't have to pass --enable_cameras manually.
_vision = False
for _arg in sys.argv[1:]:
    if _arg.startswith("env=") and "vision" in _arg.split("=", 1)[1]:
        _vision = True
        if "--enable_cameras" not in sys.argv:
            sys.argv.insert(1, "--enable_cameras")
        break

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train r2dreamer with IsaacLab.")
AppLauncher.add_app_launcher_args(parser)
# Capture only the args AppLauncher understands; pass the rest to Hydra.
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =============================================================================
# Phase 2: Everything else — safe to import now that the sim is running.
# =============================================================================

import atexit
import functools
import signal
import warnings

import hydra
import torch
from omegaconf import OmegaConf

# Isaac Sim may override SIGINT; restore Python's default so Ctrl-C works.
signal.signal(signal.SIGINT, signal.default_int_handler)

sys.path.append(str(pathlib.Path(__file__).parent.parent))
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

# Register IsaacLab task environments (triggers gymnasium gym.register calls).
import isaaclab_tasks  # noqa: F401
import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs, make_isaac_env
from envs.isaaclab import R2DreamerDirectRLEnv, R2DreamerRLEnv
from trainer import OnlineTrainer

# =============================================================================
# Task registry — all task-specific knowledge lives here, not in envs/__init__.py
# =============================================================================

# Known shorthand tasks.
# Maps shorthand → {vision: gym_id, proprio: gym_id}.
KNOWN_TASKS = {
    # ManagerBased cartpole — stock IsaacLab rewards/obs/terminations.
    "cartpole_balance": {
        "proprio": "Isaac-Cartpole-v0",
        "vision": "Isaac-Cartpole-RGB-v0",
    },
    # ManagerBased cartpole — DMC-exact overrides via manager config.
    "cartpole_balance_dmc": {
        "proprio": "Isaac-Cartpole-v0",
        "vision": "Isaac-Cartpole-RGB-v0",
    },
    # Direct cartpole — stock IsaacLab behaviour.
    "cartpole_balance_direct": {
        "proprio": "Isaac-Cartpole-Direct-v0",
        "vision": "Isaac-Cartpole-RGB-Camera-Direct-v0",
    },
    # Direct cartpole — DMC-exact overrides via monkey-patching.
    "cartpole_balance_direct_dmc": {
        "proprio": "Isaac-Cartpole-Direct-v0",
        "vision": "Isaac-Cartpole-RGB-Camera-Direct-v0",
    },
}


def _build_cartpole_env(config, vision):
    """Build a ManagerBased cartpole env (stock IsaacLab behaviour)."""
    ids = KNOWN_TASKS["cartpole_balance"]
    gym_id = ids["vision"] if vision else ids["proprio"]
    render_mode = "rgb_array" if vision else None

    env_cfg_overrides = None
    if vision:
        env_cfg_overrides = _cartpole_dreamer_vision_cfg_overrides

    return _make_env(config, gym_id, render_mode, env_cfg_fn=env_cfg_overrides)


def _build_cartpole_direct_dmc_env(config, vision):
    """Build a Direct cartpole env with DMC-style monkey-patch overrides."""
    from envs.isaac_cartpole_overrides import (
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

    return _make_env(
        config,
        gym_id,
        render_mode,
        pre_wrap_fns,
        post_create_fn,
    )


def _build_cartpole_direct_env(config, vision):
    """Build a Direct cartpole env (stock IsaacLab behaviour, no patches)."""
    ids = KNOWN_TASKS["cartpole_balance_direct"]
    gym_id = ids["vision"] if vision else ids["proprio"]
    render_mode = "rgb_array" if vision else None
    return _make_env(config, gym_id, render_mode)


def _build_cartpole_dmc_env(config, vision):
    """Build a ManagerBased cartpole env with DMC-exact overrides via config."""
    from envs.isaac_cartpole_overrides import apply_dmc_cartpole_colors

    ids = KNOWN_TASKS["cartpole_balance_dmc"]
    gym_id = ids["vision"] if vision else ids["proprio"]
    render_mode = "rgb_array" if vision else None
    post_create_fn = apply_dmc_cartpole_colors if vision else None

    env_cfg_overrides = functools.partial(
        _cartpole_dmc_cfg_overrides,
        vision=vision,
        action_repeat=int(config.action_repeat),
    )

    return _make_env(
        config,
        gym_id,
        render_mode,
        post_create_fn=post_create_fn,
        env_cfg_fn=env_cfg_overrides,
    )


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

    from envs.isaac_cartpole_dmc_mdp import dmc_balance_reward, dmc_cartpole_obs, reset_dmc_cartpole_state
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
        ``concatenate_terms=True`` — a no-op that returns the raw
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


# Map shorthand task name → builder(env_config, vision).
TASK_BUILDERS = {
    "cartpole_balance": _build_cartpole_env,
    "cartpole_balance_dmc": _build_cartpole_dmc_env,
    "cartpole_balance_direct": _build_cartpole_direct_env,
    "cartpole_balance_direct_dmc": _build_cartpole_direct_dmc_env,
}


def _patch_direct_env(unwrapped):
    """Inject ``R2DreamerDirectRLEnv`` into a third-party Direct env's MRO.

    Use this for Direct envs created via ``gym.make`` whose class you don't
    control (e.g. IsaacLab's built-in cartpole).  For your own Direct envs,
    inherit from ``R2DreamerDirectRLEnv`` directly instead.
    """
    ConcreteClass = type(unwrapped)
    PatchedClass = type(
        f"R2Dreamer{ConcreteClass.__name__}",
        (R2DreamerDirectRLEnv, ConcreteClass),
        {},
    )
    unwrapped.__class__ = PatchedClass


def _make_env(config, gym_id, render_mode=None, pre_wrap_fns=(), post_create_fn=None, env_cfg_fn=None):
    """Generic helper to construct a GPU-resident IsaacLab env.

    For **ManagerBased** envs, instantiates ``R2DreamerRLEnv`` directly.
    For **Direct** (third-party) envs, uses ``gym.make`` then patches the
    class hierarchy to inject ``R2DreamerDirectRLEnv``.

    Args:
        config: Hydra env config with env_num, action_repeat, seed, etc.
        gym_id: Gymnasium env ID string.
        render_mode: ``"rgb_array"`` for vision, ``None`` for proprio.
        pre_wrap_fns: Callables applied to the unwrapped env before wrapping.
        post_create_fn: Callable applied after env creation.
        env_cfg_fn: Optional callable(env_cfg) applied after the standard
            fields are set but before the env is created.  Use this for
            task-specific config overrides (observations, camera, …).
    """
    import importlib

    import gymnasium as gym

    spec = gym.spec(gym_id)
    env_cfg_entry = spec.kwargs["env_cfg_entry_point"]
    if isinstance(env_cfg_entry, str):
        module_name, class_name = env_cfg_entry.rsplit(":", 1)
        env_cfg_class = getattr(importlib.import_module(module_name), class_name)
    else:
        env_cfg_class = env_cfg_entry

    env_cfg = env_cfg_class()

    sim_dt = getattr(config, "sim_dt", None)
    if sim_dt is not None:
        env_cfg.sim.dt = float(sim_dt)

    env_cfg.scene.num_envs = int(config.env_num)
    env_cfg.decimation = int(config.action_repeat)
    env_cfg.seed = int(config.seed)
    env_cfg.episode_length_s = config.time_limit * env_cfg.sim.dt

    if render_mode == "rgb_array":
        # Camera position/rotation matching DMC cartpole viewpoint.
        # Both Direct and ManagerBased envs get the same transform so
        # that the rendered images are identical regardless of env type.
        _cam_pos = (-3.4, 0.0, 2.0)
        _cam_rot = (1.0, 0.0, 0.0, 0.0)  # identity — look straight forward
        if hasattr(env_cfg, "tiled_camera"):
            # DirectRLEnv — camera is a top-level env_cfg attribute.
            env_cfg.tiled_camera.width = config.size[1]
            env_cfg.tiled_camera.height = config.size[0]
            env_cfg.tiled_camera.offset.pos = _cam_pos
            env_cfg.tiled_camera.offset.rot = _cam_rot
        elif hasattr(env_cfg.scene, "tiled_camera"):
            # ManagerBased — camera lives in the scene config.
            env_cfg.scene.tiled_camera.width = config.size[1]
            env_cfg.scene.tiled_camera.height = config.size[0]
            env_cfg.scene.tiled_camera.offset.pos = _cam_pos
            env_cfg.scene.tiled_camera.offset.rot = _cam_rot

        from isaaclab.sim import RenderCfg

        env_cfg.sim.render = RenderCfg(antialiasing_mode="Off")

    if env_cfg_fn is not None:
        env_cfg_fn(env_cfg)

    # Construct the unwrapped env with terminal-obs capture.
    is_manager_based = "ManagerBasedRLEnv" in str(spec.entry_point)
    if is_manager_based:
        unwrapped = R2DreamerRLEnv(cfg=env_cfg, render_mode=render_mode)
    else:
        # Third-party Direct env — create via gym.make, then patch.
        isaac_env = gym.make(gym_id, cfg=env_cfg, render_mode=render_mode)
        unwrapped = isaac_env.unwrapped
        _patch_direct_env(unwrapped)

    return make_isaac_env(
        unwrapped,
        pre_wrap_fns=pre_wrap_fns,
        post_create_fn=post_create_fn,
        simulation_app=simulation_app,
    )


# =============================================================================
# Main
# =============================================================================


@hydra.main(version_base=None, config_path="../configs", config_name="configs")
def main(config):
    # env.task follows the codebase convention: "isaaclab_<task_name>"
    # e.g. "isaaclab_cartpole_balance"
    full_task = config.env.task  # e.g. "isaaclab_cartpole_balance"
    _, task_name = full_task.split("_", 1)  # e.g. "cartpole_balance"

    builder = TASK_BUILDERS.get(task_name)
    if builder is not None:
        # Known task with custom overrides (e.g. cartpole_balance).
        vec_env = builder(config.env, _vision)
    else:
        # Unknown task — treat task_name as the gym ID and pass through.
        render_mode = "rgb_array" if _vision else None
        vec_env = _make_env(config.env, task_name, render_mode)

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    wandb_cfg = {
        "project": "r2dreamer-isaaclab",
        "name": f"{full_task}_{config.seed}",
        "dir": str(logdir),
    }
    logger = tools.Logger(
        logdir,
        backends=[
            tools.JSONLBackend(logdir),
            # tools.TensorBoardBackend(logdir),
            tools.WandbBackend(wandb_cfg),
        ],
    )
    logger.log_hydra_config(config)

    replay_buffer = Buffer(config.buffer)

    print("Create env.")
    # OmegaConf configs are read-only; use a plain object to pass the env through.
    env_config = OmegaConf.to_container(config.env, resolve=True)
    env_config = type("EnvConfig", (), env_config)()
    env_config.isaac_vec_env = vec_env
    train_envs, eval_envs, obs_space, act_space = make_envs(env_config)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)

    policy_trainer = OnlineTrainer(
        config.trainer,
        replay_buffer,
        logger,
        logdir,
        train_stepper=train_envs,
        eval_stepper=eval_envs,
    )

    exit_code = 0
    try:
        policy_trainer.begin(agent)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C).")
        exit_code = 1
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"TRAINING CRASHED: {type(e).__name__}: {e}")
        print(f"{'='*60}")
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
        print(f"Checkpoint saved to {logdir / 'latest.pt'}")

        logger.close(exit_code=exit_code)
        vec_env._env.close()
        simulation_app.close()


if __name__ == "__main__":
    # Forward only the Hydra-style args (everything after AppLauncher args).
    sys.argv = [sys.argv[0]] + hydra_args
    main()
