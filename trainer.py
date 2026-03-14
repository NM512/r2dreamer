import tools
import torch


class OnlineTrainer:
    def __init__(self, config, replay_buffer, logger, logdir, train_stepper, eval_stepper):
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.logdir = logdir
        self.train_stepper = train_stepper
        self.eval_stepper = eval_stepper
        self.steps = int(config.steps)
        self.pretrain = int(config.pretrain)
        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
        self.video_pred_log = bool(config.video_pred_log)
        self.params_hist_log = bool(config.params_hist_log)
        self.batch_length = int(config.batch_length)
        batch_steps = int(config.batch_size * config.batch_length)
        # train_ratio is based on data steps rather than environment steps.
        self._updates_needed = tools.Every(batch_steps / config.train_ratio * config.action_repeat)
        self._should_pretrain = tools.Once()
        self._should_log = tools.Every(config.update_log_every)
        self._should_eval = tools.Every(self.eval_every)
        self._action_repeat = config.action_repeat
        # Periodic checkpointing
        self._save_checkpoint_every = int(config.save_checkpoint_every)
        self._should_save = tools.Every(self._save_checkpoint_every) if self._save_checkpoint_every > 0 else None
        # Debug: memory profiling
        self._memory_history_snapshot = bool(config.memory_history_snapshot)
        self._memory_history_steps = int(config.memory_history_steps)

    def eval(self, agent, train_step):
        """Run evaluation episodes.

        Device handling is delegated to ``self.eval_stepper``.
        """
        print("Evaluating the policy...")
        stepper = self.eval_stepper
        # Reset all environments so eval always starts from fresh episodes.
        stepper.reset()
        agent.eval()
        # (B,)
        done = torch.ones(stepper.env_num, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(stepper.env_num, dtype=torch.bool, device=agent.device)
        steps = torch.zeros(stepper.env_num, dtype=torch.int32, device=agent.device)
        returns = torch.zeros(stepper.env_num, dtype=torch.float32, device=agent.device)
        log_metrics = {}
        # cache is only used for video logging / open-loop prediction.
        cache = []
        agent_state = agent.get_initial_state(stepper.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while not once_done.all():
            steps += ~done * ~once_done
            # Step environments via the stepper (handles device transfers).
            trans, done = stepper.step(act.detach(), done.detach())

            # Store transition.
            # We keep the observation and the action that produced it together.
            trans["action"] = act
            if len(cache) < self.batch_length:
                cache.append(trans.clone())
            # (B, A)
            act, agent_state = agent.act(trans, agent_state, eval=True)
            returns += trans["reward"][:, 0] * ~once_done
            for key, value in trans.items():
                if key.startswith("log_"):
                    if key not in log_metrics:
                        log_metrics[key] = torch.zeros_like(returns)
                    log_metrics[key] += value[:, 0] * ~once_done
            once_done |= done
        # dict of (B, T, *)
        cache = torch.stack(cache, dim=1) if len(cache) else None
        self.logger.scalar("episode/eval_score", returns.mean())
        self.logger.scalar("episode/eval_length", steps.to(torch.float32).mean())
        for key, value in log_metrics.items():
            if key == "log_success":
                value = torch.clip(value, max=1.0)  # make sure 1.0 for success episode
            self.logger.scalar(f"episode/eval_{key[4:]}", value.mean())
        if cache is not None and "image" in cache:
            self.logger.video("eval_video", tools.to_np(cache["image"][:1]))
        if self.video_pred_log and cache is not None:
            initial = agent.get_initial_state(1)
            self.logger.video(
                "eval_open_loop",
                tools.to_np(
                    agent.video_pred(
                        cache[:1],  # give only first batch
                        (initial["stoch"], initial["deter"]),
                    )
                ),
            )
        self.logger.write(train_step)
        agent.train()

    def begin(self, agent, initial_step=0, save_fn=None):
        """Main online training loop.

        Device handling is delegated to ``self.train_stepper``.

        Args:
            initial_step: Starting step count (used when resuming from checkpoint).
            save_fn: Optional callback ``save_fn(step)`` invoked periodically to
                save a checkpoint.
        """
        stepper = self.train_stepper
        video_cache = []
        if initial_step > 0:
            self._step = initial_step
            # Advance Every counters so they don't all fire on the first step.
            self._should_eval._last = self._step
            self._should_log._last = self._step
            self._updates_needed._last = self._step
            if self._should_save is not None:
                self._should_save._last = self._step
        else:
            self._step = self.replay_buffer.count() * self._action_repeat
        update_count = 0
        # Debug: memory history snapshot
        _mem_recording = False
        _mem_update_count = 0
        if self._memory_history_snapshot:
            print(
                f"[Debug] Memory history recording enabled — will record {self._memory_history_steps} training steps."
            )
            torch.cuda.memory._record_memory_history(max_entries=1048576)
            _mem_recording = True
        # (B,)
        done = torch.ones(stepper.env_num, dtype=torch.bool, device=agent.device)
        returns = torch.zeros(stepper.env_num, dtype=torch.float32, device=agent.device)
        lengths = torch.zeros(stepper.env_num, dtype=torch.int32, device=agent.device)
        episode_ids = torch.arange(stepper.env_num, dtype=torch.int32, device=agent.device)
        # Global counter for unique episode IDs — incremented whenever an env
        # resets so SliceSampler never samples across episode boundaries.
        _next_episode_id = stepper.env_num
        train_metrics = {}
        agent_state = agent.get_initial_state(stepper.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while self._step < self.steps:
            # Evaluation
            if self._should_eval(self._step) and self.eval_episode_num > 0:
                self.eval(agent, self._step)
                stepper.reset()
                done = torch.ones(stepper.env_num, dtype=torch.bool, device=agent.device)
                returns.zero_()
                lengths.zero_()
                agent_state = agent.get_initial_state(stepper.env_num)
                act = agent_state["prev_action"].clone()
                video_cache = []
            # Save metrics
            if done.any():
                for i, d in enumerate(done):
                    if d and lengths[i] > 0:
                        if i == 0 and len(video_cache) > 0:
                            video = torch.stack(video_cache, axis=0)
                            self.logger.video("train_video", tools.to_np(video[None]))
                            video_cache = []
                        self.logger.scalar("episode/score", returns[i])
                        self.logger.scalar("episode/length", lengths[i])
                        self.logger.write(self._step + i)  # to show all values on tensorboard
                        returns[i] = lengths[i] = 0
            self._step += stepper.count_active_steps(done) * self._action_repeat
            lengths += ~done

            # Step environments via the stepper (handles device transfers).
            trans, done = stepper.step(act.detach(), done.detach())

            # Policy inference on GPU.
            # "agent_state" is reset by the agent based on the "is_first" flag in trans.
            # (B, A)
            act, agent_state = agent.act(trans.clone(), agent_state, eval=False)

            # Store transition.
            # We keep the observation and the action that produced it together.
            # Mask actions after an episode has ended.
            trans["action"] = act * ~done.unsqueeze(-1)
            trans["stoch"] = agent_state["stoch"]
            trans["deter"] = agent_state["deter"]
            trans["episode"] = episode_ids  # Don't lift dim
            if "image" in trans:
                video_cache.append(trans["image"][0])
            self.replay_buffer.add_transition(trans.detach())
            returns += trans["reward"][:, 0]

            # Bump episode IDs AFTER storing the transition so the terminal
            # row (is_last=True) keeps the OLD episode ID.  The new ID takes
            # effect on the next iteration's row (the reset obs with
            # is_first=True), which is where SliceSampler should see the
            # trajectory boundary.
            for _i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                episode_ids[_i] = _next_episode_id
                _next_episode_id += 1

            # Update models after enough data has accumulated
            if self._step // (stepper.env_num * self._action_repeat) > self.batch_length + 1:
                if self._should_pretrain():
                    update_num = self.pretrain
                else:
                    update_num = self._updates_needed(self._step)
                for _ in range(update_num):
                    _metrics = agent.update(self.replay_buffer)
                    train_metrics = _metrics
                    if _mem_recording:
                        _mem_update_count += 1
                        if _mem_update_count >= self._memory_history_steps:
                            snapshot_path = self.logdir / "memory_snapshot.pickle"
                            torch.cuda.memory._dump_snapshot(str(snapshot_path))
                            torch.cuda.memory._record_memory_history(enabled=None)
                            _mem_recording = False
                            print(f"[Debug] Memory snapshot saved to {snapshot_path}")
                            print("[Debug] View it at: https://pytorch.org/memory_viz")
                update_count += update_num
                # Log training metrics
                if self._should_log(self._step):
                    for name, value in train_metrics.items():
                        value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                        self.logger.scalar(f"train/{name}", value)
                    self.logger.scalar("train/opt/updates", update_count)
                    if self.video_pred_log:
                        _sample = self.replay_buffer.sample()
                        if _sample is not None:
                            data, _, initial = _sample
                            self.logger.video("open_loop", tools.to_np(agent.video_pred(data, initial)))
                    if self.params_hist_log:
                        for name, param in agent._named_params.items():
                            self.logger.histogram(name, tools.to_np(param))
                    self.logger.write(self._step, fps=True)
            # Periodic checkpoint saving
            if save_fn is not None and self._should_save is not None and self._should_save(self._step):
                save_fn(self._step)
