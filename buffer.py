import warnings

import torch
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer:
    def __init__(self, config):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.num_eps = 0
        self._buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=config.max_size, device=self.storage_device, ndim=2),
            sampler=SliceSampler(
                num_slices=self.batch_size, end_key=None, traj_key="episode", truncated_key=None, strict_length=True
            ),
            prefetch=0,
            batch_size=self.batch_size * (self.batch_length + 1),  # +1 for context
        )

    def add_transition(self, data):
        # This is batched data and lifted for storage.
        # (B, ...) -> (B, 1, ...)
        self._buffer.extend(data.unsqueeze(1))

    def sample(self):
        try:
            sample_td, info = self._buffer.sample(return_info=True)
        except RuntimeError as e:
            if "sufficient length" in str(e):
                warnings.warn(f"Replay buffer sample skipped: {e}")
                return None
            raise
        # The sampler returns a flattened batch of length B*(T+1).
        # (B*(T+1), ...) -> (B, T+1, ...)
        sample_td = sample_td.view(-1, self.batch_length + 1)
        src_dev = sample_td.device
        if src_dev.type == "cpu" and self.device.type == "cuda":
            sample_td = sample_td.pin_memory().to(self.device, non_blocking=True)
        elif src_dev != self.device:
            sample_td = sample_td.to(self.device, non_blocking=True)
        # The initial ones are used only to extract the latent vector
        initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0])
        data = sample_td[:, 1:]
        data.set_("action", sample_td["action"][:, :-1])  # action is 1 step back
        index = [ind.view(-1, self.batch_length + 1)[:, 1:] for ind in info["index"]]
        return data, index, initial

    def update(self, index, stoch, deter):
        # Flatten the data
        index = [ind.reshape(-1) for ind in index]
        # (B, T, S, K) -> (B*T, S, K)
        stoch = stoch.reshape(-1, *stoch.shape[2:])
        # (B, T, D) -> (B*T, D)
        deter = deter.reshape(-1, *deter.shape[2:])
        # In storage, the length is the first dimension, and the batch (number of environments) is the second dimension.
        self._buffer[index[1], index[0]].set_("stoch", stoch)
        self._buffer[index[1], index[0]].set_("deter", deter)

    def count(self):
        if self._buffer.storage.shape is None:
            return 0
        return self._buffer.storage.shape.numel()
