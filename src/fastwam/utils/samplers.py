from math import ceil
from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        indices = self._epoch_indices(self.epoch + self.epoch_offset)
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def _epoch_indices(self, epoch: int) -> list[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + epoch)
        if hasattr(self.dataset, 'sample_epoch_indices'):
            indices = self.dataset.sample_epoch_indices(self.seed + epoch)
            if len(indices) != len(self.dataset):
                raise ValueError('Dataset epoch sampler must preserve dataset length')
        else:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        return indices

    def __len__(self) -> int:
        return len(self.dataset)


class ContinuousEpochSampler(ResumableEpochSampler):
    """Prefetch across epochs without changing Accelerate's per-rank batches.

    Each virtual epoch repeats its first indices to a global-batch boundary,
    exactly like BatchSamplerShard(even_batches=True, split_batches=False).
    Worker prefetch may run ahead: only the trainer's *consumed* epoch/batch
    cursor is checkpointed. Iterator-local progress must never update it.
    """

    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        super().__init__(dataset, seed, batch_size, num_processes)
        if len(dataset) <= 0 or self.batch_size <= 0 or self.num_processes <= 0:
            raise ValueError('Continuous sampling requires nonempty data and positive batch/world sizes')
        if self.num_processes == 1 and len(dataset) % self.batch_size:
            raise ValueError('Single-process continuous sampling requires full batches; '
                             'disable continuous_dataloader to preserve a partial final batch')
        self.global_batch_size = self.batch_size * self.num_processes
        self.batches_per_epoch = ceil(len(dataset) / self.global_batch_size)

    def __iter__(self) -> Iterator[int]:
        epoch = self.epoch + self.epoch_offset
        offset = self.resume_batch_offset if self.epoch == 0 else 0
        if not 0 <= offset <= self.batches_per_epoch:
            raise ValueError(f'Invalid resume batch offset: {offset}')
        # Accept legacy checkpoints saved on the last batch, before StopIteration.
        if offset == self.batches_per_epoch:
            epoch += 1
            offset = 0
        while True:
            indices = self._epoch_indices(epoch)
            padding = len(self) - len(indices)
            if padding:
                indices += (indices * ceil(padding / len(indices)))[:padding]
            yield from indices[offset * self.global_batch_size:]
            epoch += 1
            offset = 0

    def __len__(self) -> int:
        # A virtual epoch length for schedulers; iteration continues until the
        # trainer reaches max_steps rather than draining workers every epoch.
        return self.batches_per_epoch * self.global_batch_size
