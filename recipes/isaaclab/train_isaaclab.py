"""Generic IsaacLab entry-point for r2dreamer.

This script is a **template** showing the exact pattern to follow when
training r2dreamer on any IsaacLab task. See the /docs/isaaclab.md guide
for detailed instructions on how to use and customize this script for 
your own IsaacLab training runs.
"""

# =============================================================================
# Phase 1: AppLauncher must run before any IsaacLab / USD / warp imports.
#           Guarded by __name__ == "__main__" so that importers can run
#           their own Phase 1 first and set module globals afterwards.
# =============================================================================

import argparse
import pathlib
import sys

if __name__ == "__main__":
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

    sys.path.append(str(pathlib.Path(__file__).parent.parent))
else:
    # When imported, the caller must set these before calling main().
    _vision = False
    simulation_app = None

# =============================================================================
# Phase 2: Everything else -- safe to import now that the sim is running.
# =============================================================================

import atexit
import signal
import warnings

import hydra
import torch
from omegaconf import OmegaConf

# Isaac Sim may override SIGINT; restore Python's default so Ctrl-C works.
signal.signal(signal.SIGINT, signal.default_int_handler)

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

# Register IsaacLab task environments (triggers gymnasium gym.register calls).
import isaaclab_tasks  # noqa: F401
import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer

# Builder function: set by the entry point before calling main().
# Signature: build_env(env_config, vision, simulation_app) -> IsaacLabVecEnv
_build_env = None


# =============================================================================
# Generic env config construction
# =============================================================================


def make_env_cfg(config, gym_id):
    """Resolve a gymnasium ID and create an env config with common fields set.

    Looks up the gymnasium spec, resolves the env config class, instantiates
    it, and sets common fields (num_envs, decimation, seed, episode_length_s).

    Args:
        config: Hydra env config with env_num, action_repeat, seed, etc.
        gym_id: Gymnasium env ID string.

    Returns:
        Tuple of (env_cfg, spec) -- the configured env config and the
        gymnasium spec (useful for detecting ManagerBased vs Direct via
        ``spec.entry_point``).
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

    return env_cfg, spec


# =============================================================================
# Training
# =============================================================================


def run(config, build_env, simulation_app, vision=False):
    """Train r2dreamer on an IsaacLab environment.

    This is the core training function.  It takes the builder as an
    explicit parameter so it can be called directly without module state.

    Args:
        config: Hydra config object.
        build_env: Callable(env_config, vision, simulation_app) -> IsaacLabVecEnv.
        simulation_app: The IsaacLab simulation app instance.
        vision: Whether to use RGB camera observations.
    """
    vec_env = build_env(config.env, vision, simulation_app)
    # If you do not invoke this script as shown in the train_cartpole.py example,
    # but instead copy it into your repo and make extensive changes, you can simply
    # construct the environment config here with:
    # from envs import make_isaac_env
    # env_cfg, spec = train_isaaclab.make_env_cfg(config, "Your-IsaacLab-Task-Name")

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    # wandb_cfg = {
    #     "project": "r2dreamer-isaaclab",
    #     "name": f"{config.env.task}_{config.seed}",
    #     "dir": str(logdir),
    # }
    logger = tools.Logger(
        logdir,
        backends=[
            tools.JSONLBackend(logdir),
            tools.TensorBoardBackend(logdir),
            # tools.WandbBackend(wandb_cfg),
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

    # Resume from checkpoint if one exists in the logdir.
    _resume_step = 0
    checkpoint_path = logdir / "latest.pt"
    if checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        _resume_step = int(checkpoint.get("step", 0))
        if "scheduler_state_dict" in checkpoint:
            agent._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            agent._scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "slow_value_updates" in checkpoint:
            agent._slow_value_updates = checkpoint["slow_value_updates"]
        if "ema_updates" in checkpoint and hasattr(agent, "_ema_updates"):
            agent._ema_updates = checkpoint["ema_updates"]
        print(f"  Restored agent weights, optimizer states, step={_resume_step}")

    def _save_checkpoint(step):
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            "step": step,
            "scheduler_state_dict": agent._scheduler.state_dict(),
            "scaler_state_dict": agent._scaler.state_dict(),
            "slow_value_updates": agent._slow_value_updates,
            **({"ema_updates": agent._ema_updates} if hasattr(agent, "_ema_updates") else {}),
        }
        torch.save(items_to_save, logdir / f"checkpoint_{step}.pt")
        torch.save(items_to_save, logdir / "latest.pt")
        print(f"Checkpoint saved: checkpoint_{step}.pt + latest.pt")

    policy_trainer = OnlineTrainer(
        config.trainer,
        replay_buffer,
        logger,
        logdir,
        train_stepper=train_envs,
        eval_stepper=eval_envs,
        initial_step=_resume_step,
        save_fn=_save_checkpoint,
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
        _save_checkpoint(policy_trainer._step)

        logger.close(exit_code=exit_code)
        vec_env._env.close()
        simulation_app.close()


_CONFIGS_DIR = str((pathlib.Path(__file__).parent / "../../configs").resolve())


@hydra.main(version_base=None, config_path=_CONFIGS_DIR, config_name="configs")
def main(config):
    """Hydra entry point -- delegates to run() using module-level state.

    External entry points set ``_build_env``, ``simulation_app``, and
    ``_vision`` on this module before calling ``main()``.
    """
    if _build_env is None:
        raise ValueError(
            "No build_env function set. "
            "Set train_isaaclab._build_env before calling main(), "
            "or see cartpole/train_cartpole.py for an example."
        )
    run(config, _build_env, simulation_app, _vision)


if __name__ == "__main__":
    # Forward only the Hydra-style args (everything after AppLauncher args).
    sys.argv = [sys.argv[0]] + hydra_args
    main()
