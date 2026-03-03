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
Proprio (state-based) cartpole balance::

    python train_isaaclab.py env=isaaclab_proprio env.task=isaaclab_cartpole_balance

Vision (RGB camera) cartpole balance (cameras are auto-enabled)::

    python train_isaaclab.py env=isaaclab_vision env.task=isaaclab_cartpole_balance

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
import sys
import pathlib

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

import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs, make_isaac_env
from trainer import OnlineTrainer

# Register IsaacLab task environments (triggers gymnasium gym.register calls).
import isaaclab_tasks  # noqa: F401

# =============================================================================
# Task registry — all task-specific knowledge lives here, not in envs/__init__.py
# =============================================================================

# Known shorthand tasks that need custom overrides.
# Maps shorthand → {vision: gym_id, proprio: gym_id}.
KNOWN_TASKS = {
    "cartpole_balance": {
        "proprio": "Isaac-Cartpole-Direct-v0",
        "vision": "Isaac-Cartpole-RGB-Camera-Direct-v0",
    },
}


def _build_cartpole_env(config, vision):
    """Build a cartpole env with DMC-style overrides."""
    from envs.isaac_cartpole_overrides import (
        patch_dmc_cartpole_obs,
        patch_dmc_cartpole_reward,
        patch_no_termination,
        apply_dmc_cartpole_colors,
    )

    ids = KNOWN_TASKS["cartpole_balance"]
    gym_id = ids["vision"] if vision else ids["proprio"]
    render_mode = "rgb_array" if vision else None
    post_create_fn = apply_dmc_cartpole_colors if vision else None

    pre_wrap_fns = [
        patch_no_termination,
        functools.partial(patch_dmc_cartpole_reward, action_repeat=int(config.action_repeat)),
        patch_dmc_cartpole_obs,
        # patch_dmc_cartpole_reset,
    ]

    return _make_env(
        config,
        gym_id,
        render_mode,
        pre_wrap_fns,
        post_create_fn,
    )


# Map shorthand task name → builder(env_config, vision).
# Only tasks that need custom overrides go here.
TASK_BUILDERS = {
    "cartpole_balance": _build_cartpole_env,
}


def _make_env(config, gym_id, render_mode=None, pre_wrap_fns=(), post_create_fn=None):
    """Generic helper to construct a GPU-resident IsaacLab env.

    Args:
        env_cfg_fn: Optional callable(env_cfg) applied after the standard
            fields are set but before the env is created.  Use this for
            task-specific config overrides (dt, action_scale, …).
    """
    import importlib
    import gymnasium as gym

    env_cfg_entry = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
    if isinstance(env_cfg_entry, str):
        module_name, class_name = env_cfg_entry.rsplit(":", 1)
        env_cfg_class = getattr(importlib.import_module(module_name), class_name)
    else:
        env_cfg_class = env_cfg_entry

    env_cfg = env_cfg_class()
    env_cfg.scene.num_envs = int(config.env_num)
    env_cfg.decimation = int(config.action_repeat)
    env_cfg.seed = int(config.seed)
    env_cfg.episode_length_s = config.time_limit * env_cfg.sim.dt

    # OmegaConf.update(config, "action_repeat", 1)

    if render_mode == "rgb_array":
        env_cfg.tiled_camera.width = config.size[1]  # TODO: move this to the overrides for vision tasks
        env_cfg.tiled_camera.height = config.size[0]
        env_cfg.tiled_camera.offset.pos = (-3.4, 0.0, 2.0)
        from isaaclab.sim import RenderCfg

        env_cfg.sim.render = RenderCfg(antialiasing_mode="Off")

    return make_isaac_env(
        gym_id=gym_id,
        env_cfg=env_cfg,
        render_mode=render_mode,
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

    logger = tools.Logger(logdir)
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

    try:
        policy_trainer.begin(agent)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
        vec_env._env.close()
        simulation_app.close()


if __name__ == "__main__":
    # Forward only the Hydra-style args (everything after AppLauncher args).
    sys.argv = [sys.argv[0]] + hydra_args
    main()
