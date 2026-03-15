# r2dreamer + IsaacLab

IsaacLab differs from the other environments supported by r2dreamer (DMC, Atari, Crafter, MetaWorld, MemoryMaze). Those are benchmark-style task suites: you choose a predefined task and start training. IsaacLab is instead a GPU-accelerated robotics simulation platform. It provides the building blocks for creating physics-based tasks, but it does not come with a single, fixed benchmark collection for RL. That means integrating r2dreamer with IsaacLab requires a bit more setup on your side, including defining the task, configuring it, and providing the appropriate entry point.

We provide working recipes in the form of runnable examples that show the full integration pattern and serve as a foundation for your own robotic tasks.

## Environment Steup

For the easiest setup, we provide a ready-to-run Docker container with all dependencies preinstalled; see [docker.md](docker.md). If you already have a local IsaacLab setup or prefer not to use Docker, you can instead follow the official installation instructions in the[IsaacLab documentation](https://isaac-sim.github.io/IsaacLab/).

## File layout

```
recipes/isaaclab/
  train_isaaclab.py              # Generic template -- import or copy for your own task
  cartpole/
    train_cartpole.py            # Cartpole benchmark entry point
    tasks.py                     # Task registry, builder functions, config overrides
    dmc_overrides.py             # DMC-compatible reward/obs/reset/termination/visuals

envs/
  isaaclab.py                    # R2DreamerRLEnv, R2DreamerDirectRLEnv, IsaacLabVecEnv

configs/env/
  isaaclab_proprio.yaml          # Hydra config for proprioceptive observations
  isaaclab_vision.yaml           # Hydra config for RGB camera observations
```

## How `train_isaaclab.py` works

`train_isaaclab.py` is a generic recipe that handles:

- **Phase 1**: CLI parsing, `AppLauncher` startup (required before any IsaacLab imports)
- **Phase 2**: Hydra config, Dreamer agent, replay buffer, training loop, logging, checkpointing

All task-specific logic lives in a single **builder function** that you provide. The builder has the signature:

```python
def build_env(env_config, vision: bool, simulation_app) -> IsaacLabVecEnv
```

The core training logic is in `run()`, which takes the builder as an explicit parameter:

```python
train_isaaclab.run(config, build_env, simulation_app, vision)
```

`main()` is a thin `@hydra.main` wrapper that reads module-level state (`_build_env`, `simulation_app`, `_vision`) and delegates to `run()`. This is needed because Hydra only passes `config` to decorated functions.

## Terminal observation capture

IsaacLab runs N environments in parallel on the GPU. When an environment terminates or is truncated, IsaacLab **auto-resets it inside `step()`** and returns the post-reset observation as the env's entry in `obs_dict`. The true terminal observation is silently overwritten before it is ever returned.

This is a problem for Dreamer because the terminal reward must be paired with the true terminal observation. Pairing it with the reset obs misattributes the reward to a state that never produced it.

The `R2DreamerRLEnv` and `R2DreamerDirectRLEnv` base classes override `_reset_idx` to capture observations before the reset and store them in `extras["terminal_obs"]`. The `IsaacLabVecEnv` wrapper swaps these in on the step an episode ends, matching the data flow expected by Dreamer's world model.

For your own tasks, inherit from these classes and terminal observation handling works automatically.

## Adding your own task

1. Copy `train_isaaclab.py` or create a new entry point that imports it.
2. Define your IsaacLab env inheriting from `R2DreamerRLEnv` (ManagerBased) or `R2DreamerDirectRLEnv` (Direct) for terminal observation capture.
3. Write a builder function and set it on the module before calling `main()`.

Minimal example:

```python
from envs import make_isaac_env
from envs.isaaclab import R2DreamerRLEnv

class MyEnv(R2DreamerRLEnv):
    ...

def build_my_task(config, vision, simulation_app):
    cfg = MyEnvCfg()
    cfg.scene.num_envs = int(config.env_num)
    cfg.decimation = int(config.action_repeat)
    cfg.seed = int(config.seed)
    cfg.episode_length_s = config.time_limit * cfg.sim.dt
    render_mode = "rgb_array" if vision else None
    env = MyEnv(cfg=cfg, render_mode=render_mode)
    return make_isaac_env(env, simulation_app=simulation_app)

# In your entry point, before calling main():
import train_isaaclab
train_isaaclab._build_env = build_my_task
train_isaaclab.main()
```

## Cartpole benchmark

The cartpole example serves as a reproducible sanity check and demonstrates all the integration patterns. It covers four variants along two axes:

|                  | Stock IsaacLab            | DMC-exact                     |
| ---------------- | ------------------------- | ----------------------------- |
| **ManagerBased** | `cartpole_balance`        | `cartpole_balance_dmc`        |
| **Direct**       | `cartpole_balance_direct` | `cartpole_balance_direct_dmc` |

**Stock IsaacLab** variants use the default IsaacLab rewards, observations, terminations, and reset distribution.

**DMC-exact** variants override all four MDP components to exactly replicate the DeepMind Control Suite `cartpole:balance` task to make the learing comparable to the DMC setup:

- Reward: smooth balance reward (upright * centered * small_control * small_velocity)
- Observations: `[cart_x, cos(theta), sin(theta), cart_vel, pole_vel]`
- Termination: time-only (no early termination on cart out-of-bounds)
- Reset: DMC initial state distribution (small cart pos, near-vertical pole)
- Visuals (vision only): DMC colour scheme (warm brown cart/pole, steel blue rail)

**ManagerBased** envs use IsaacLab's config-driven manager system. DMC overrides are applied by replacing `ObservationTermCfg`, `RewardTermCfg`, etc. on the env config before instantiation.

**Direct** envs use IsaacLab's code-driven `DirectRLEnv`. DMC overrides are applied via monkey-patching (`_get_rewards`, `_get_observations`, `_reset_idx`, `_get_dones`) since the stock cartpole class can't be modified.

### Running the cartpole benchmark

Each variant can be run with either proprioceptive (`proprio`) or vision (`vision`) observations:

**ManagerBased -- stock IsaacLab:**

```bash
python recipes/isaaclab/cartpole/train_cartpole.py env=isaaclab_proprio env.task=isaaclab_cartpole_balance
python recipes/isaaclab/cartpole/train_cartpole.py env=isaaclab_vision  env.task=isaaclab_cartpole_balance
```

**ManagerBased -- DMC-exact:**

```bash
python recipes/isaaclab/cartpole/train_cartpole.py env=isaaclab_proprio env.task=isaaclab_cartpole_balance_dmc
python recipes/isaaclab/cartpole/train_cartpole.py env=isaaclab_vision  env.task=isaaclab_cartpole_balance_dmc
```

**Direct -- stock IsaacLab:**

```bash
python recipes/isaaclab/cartpole/train_cartpole.py env=isaaclab_proprio env.task=isaaclab_cartpole_balance_direct
python recipes/isaaclab/cartpole/train_cartpole.py env=isaaclab_vision  env.task=isaaclab_cartpole_balance_direct
```

**Direct -- DMC-exact:**

```bash
python recipes/isaaclab/cartpole/train_cartpole.py env=isaaclab_proprio env.task=isaaclab_cartpole_balance_direct_dmc
python recipes/isaaclab/cartpole/train_cartpole.py env=isaaclab_vision  env.task=isaaclab_cartpole_balance_direct_dmc
```
