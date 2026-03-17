"""Cartpole benchmark entry-point for r2dreamer + IsaacLab.

This script trains r2dreamer on one of four IsaacLab cartpole variants,
serving as a reproducible benchmark and sanity check. 
See the /docs/isaaclab.md guide for detailed instructions on how to 
run these examples.
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

parser = argparse.ArgumentParser(description="Train r2dreamer cartpole with IsaacLab.")
AppLauncher.add_app_launcher_args(parser)
# Capture only the args AppLauncher understands; pass the rest to Hydra.
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =============================================================================
# Phase 2: Everything else -- safe to import now that the sim is running.
# =============================================================================

_this_dir = pathlib.Path(__file__).parent                   # recipes/isaaclab/cartpole/
_recipe_dir = _this_dir.parent                              # recipes/isaaclab/
_r2dreamer_root = _recipe_dir.parent.parent                 # r2dreamer/
sys.path.insert(0, str(_r2dreamer_root))
sys.path.insert(0, str(_recipe_dir))

# Configure the generic template module before its @hydra.main is called.
import train_isaaclab  # noqa: E402
from cartpole.tasks import CARTPOLE_TASK_BUILDERS  # noqa: E402

train_isaaclab.simulation_app = simulation_app
train_isaaclab._vision = _vision


def _cartpole_build_env(env_config, vision, simulation_app):
    """Resolve the cartpole variant from env.task and build the env."""
    task_name = env_config.task.split("_", 1)[1]
    builder = CARTPOLE_TASK_BUILDERS.get(task_name)
    if builder is None:
        raise ValueError(
            f"Unknown cartpole task: {task_name!r}. "
            f"Available: {list(CARTPOLE_TASK_BUILDERS)}"
        )
    return builder(env_config, vision, simulation_app)


train_isaaclab._build_env = _cartpole_build_env

if __name__ == "__main__":
    # Forward only the Hydra-style args (everything after AppLauncher args).
    sys.argv = [sys.argv[0]] + hydra_args
    train_isaaclab.main()
