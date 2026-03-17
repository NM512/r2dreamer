from env_stepper import CpuEnvStepper, GpuEnvStepper

from . import parallel, wrappers


def make_envs(config):
    suite, _ = config.task.split("_", 1)
    if suite == "isaaclab":
        # IsaacLab is already a GPU-resident vectorized env, it cannot be
        # wrapped in ParallelEnv.
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


def make_isaac_env(unwrapped, pre_wrap_fns=(), post_create_fn=None, simulation_app=None):
    """Wrap an already-constructed IsaacLab env as an ``IsaacLabVecEnv``.

    Parameters
    ----------
    unwrapped:
        A fully constructed IsaacLab env instance (``R2DreamerRLEnv``,
        ``R2DreamerDirectRLEnv``, or a subclass of either).
    """
    from envs.isaaclab import IsaacLabVecEnv, R2DreamerDirectRLEnv, R2DreamerRLEnv

    assert isinstance(
        unwrapped, (R2DreamerRLEnv, R2DreamerDirectRLEnv)
    ), f"Expected R2DreamerRLEnv or R2DreamerDirectRLEnv, got {type(unwrapped).__name__}"

    return IsaacLabVecEnv(unwrapped, simulation_app=simulation_app)
