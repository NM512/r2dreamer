from . import parallel, wrappers
from env_stepper import CpuEnvStepper, GpuEnvStepper


def make_envs(config):
    suite, _ = config.task.split("_", 1)
    if suite == "isaaclab":
        # IsaacLab is already a GPU-resident vectorized env — it cannot be
        # wrapped in ParallelEnv (single sim instance, no subprocess isolation
        # needed).  The caller (train_isaaclab.py) is responsible for building
        # env_cfg and patches, then passing the ready IsaacLabVecEnv via
        # config.isaac_vec_env before calling make_envs.
        vec_env = config.isaac_vec_env
        device = getattr(config, "device", "cuda:0")
        stepper = GpuEnvStepper(vec_env, device)
        return stepper, stepper, vec_env.observation_space, vec_env.action_space

    def env_constructor(idx):
        return lambda: make_env(config, idx)

    train_envs = parallel.ParallelEnv(env_constructor, config.env_num, config.device)
    eval_envs = parallel.ParallelEnv(env_constructor, config.eval_episode_num, config.device)
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    train_stepper = CpuEnvStepper(train_envs, config.device)
    eval_stepper = CpuEnvStepper(eval_envs, config.device)
    return train_stepper, eval_stepper, obs_space, act_space


def make_env(config, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(task, config.action_repeat, config.size, seed=config.seed + id)
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.gray,
            noops=config.noops,
            lives=config.lives,
            sticky=config.sticky,
            actions=config.actions,
            length=config.time_limit,
            pooling=config.pooling,
            aggregate=config.aggregate,
            resize=config.resize,
            autostart=config.autostart,
            clip_reward=config.clip_reward,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "metaworld":
        import envs.metaworld as metaworld

        env = metaworld.MetaWorld(
            task,
            config.action_repeat,
            config.size,
            config.camera,
            config.seed + id,
        )
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit // config.action_repeat)
    return wrappers.Dtype(env)


def make_isaac_env(gym_id, env_cfg, render_mode=None, pre_wrap_fns=(), post_create_fn=None, simulation_app=None):
    """Construct a GPU-resident IsaacLab env wrapped as an ``IsaacLabVecEnv``.

    This function is intentionally generic — it knows nothing about specific
    tasks.  All task-specific knowledge (gym_id, env_cfg construction, patch
    functions) belongs in the caller (e.g. ``train_isaaclab.py``).

    Parameters
    ----------
    gym_id:
        Gymnasium env ID string (e.g. ``"Isaac-Cartpole-Direct-v0"``).
    env_cfg:
        A fully configured IsaacLab ``EnvCfg`` instance.
    render_mode:
        ``"rgb_array"`` for vision tasks, ``None`` for proprio-only.
    pre_wrap_fns:
        Callables ``fn(unwrapped_env, config)`` applied to the unwrapped env
        before wrapping (e.g. reward/obs/termination patches).  Note: callers
        that don't need ``config`` can use ``functools.partial`` to pre-bind it.
    post_create_fn:
        Optional callable ``fn(unwrapped_env)`` applied after ``gym.make``
        (e.g. scene colour overrides that require the sim to be running).
    """
    import gymnasium as gym

    from envs.isaaclab import IsaacLabVecEnv

    isaac_env = gym.make(gym_id, cfg=env_cfg, render_mode=render_mode)
    unwrapped = isaac_env.unwrapped

    for fn in pre_wrap_fns:
        fn(unwrapped)

    if post_create_fn is not None:
        post_create_fn(unwrapped)

    return IsaacLabVecEnv(unwrapped, simulation_app=simulation_app)
