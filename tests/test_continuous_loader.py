"""Batch parity, prefetch/consumer separation and worker lifetime contracts."""
import itertools
import copy
import os
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader
from torch.utils.data import DataLoader, Dataset

from fastwam.trainer import Wan22Trainer
from fastwam.utils.samplers import ContinuousEpochSampler, ResumableEpochSampler


class Indices(Dataset):
    deterministic_samples = True

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return {'index': index, 'pid': os.getpid()}


class Balanced(Indices):
    def sample_epoch_indices(self, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randint(self.n, (self.n,), generator=g).tolist()


def prepared(dataset, sampler, batch, world, rank, workers=0):
    kwargs = dict(num_workers=workers, generator=torch.Generator().manual_seed(42))
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(dataset, batch_size=batch, sampler=sampler, **kwargs)
    return prepare_data_loader(loader, num_processes=world, process_index=rank,
                               device=torch.device('cpu'), put_on_device=False,
                               split_batches=False, even_batches=True, rng_types=[])


def stop(loader, iterator):
    iterator.close()
    loader.end()
    workers = getattr(loader.base_dataloader, '_iterator', None)
    if workers is not None:
        workers._shutdown_workers()


@pytest.mark.parametrize('dataset_type', [Indices, Balanced])
@pytest.mark.parametrize('n,batch,world', [(3,4,8), (17,3,2), (32,4,2), (65,4,8), (120,4,1), (30470,64,8)])
def test_every_rank_matches_original_epochs(dataset_type, n, batch, world):
    dataset = dataset_type(n)
    for rank in range(world):
        old = ResumableEpochSampler(dataset, 42, batch, world)
        old_loader = prepared(dataset, old, batch, world, rank)
        expected = [b['index'].tolist() for _ in range(3) for b in old_loader]
        new = ContinuousEpochSampler(dataset, 42, batch, world)
        loader = prepared(dataset, new, batch, world, rank)
        iterator = iter(loader)
        actual = [b['index'].tolist() for b in itertools.islice(iterator, len(expected))]
        stop(loader, iterator)
        assert actual == expected


@pytest.mark.parametrize('offset', [0, 1, 3, 4])
def test_resume_uses_consumed_cursor_even_with_prefetch(offset):
    dataset, batch, world, rank, epoch = Balanced(25), 4, 2, 1, 7
    original = ResumableEpochSampler(dataset, 42, batch, world)
    original.set_epoch_offset(epoch)
    old_loader = prepared(dataset, original, batch, world, rank)
    expected = [b['index'].tolist() for _ in range(3) for b in old_loader][offset:]
    sampler = ContinuousEpochSampler(dataset, 42, batch, world)
    sampler.set_epoch_offset(epoch)
    sampler.set_resume_batch_offset(offset)
    loader = prepared(dataset, sampler, batch, world, rank, workers=2)
    iterator = iter(loader)
    try:
        batches = [next(iterator) for _ in range(len(expected))]
        assert [b['index'].tolist() for b in batches] == expected
        # Prefetch has crossed several epochs; the saved consumer cursor has not.
        assert sampler.epoch_offset == epoch
        assert sampler.resume_batch_offset == offset
        assert len({int(pid) for b in batches for pid in b['pid']}) == 2
        assert all(w.is_alive() for w in loader.base_dataloader._iterator._workers)
    finally:
        stop(loader, iterator)


def test_loader_restart_does_not_advance_model_rng():
    dataset = Indices(32)
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    sampler = ContinuousEpochSampler(dataset, 42, 4, 2)
    loader = prepared(dataset, sampler, 4, 2, 0, workers=2)
    iterator = iter(loader)
    try:
        next(iterator)
        assert torch.equal(before, torch.get_rng_state())
    finally:
        stop(loader, iterator)


@pytest.mark.parametrize('workers', [0, 2])
def test_trainer_constructs_continuous_loader(workers):
    trainer = SimpleNamespace(
        continuous_dataloader=True, seed=42, batch_size=4, num_workers=workers,
        gradient_accumulation_steps=1,
        accelerator=SimpleNamespace(num_processes=2, dataloader_config=SimpleNamespace(
            split_batches=False, even_batches=True, dispatch_batches=False)),
        cfg={'persistent_workers': True, 'prefetch_factor': 2},
    )
    loader = Wan22Trainer._build_loader(trainer, Indices(25))
    assert isinstance(loader.sampler, ContinuousEpochSampler)
    assert loader.persistent_workers == (workers > 0)
    trainer.gradient_accumulation_steps = 3
    with pytest.raises(ValueError, match='divisible'):
        Wan22Trainer._build_loader(trainer, Indices(25))


def test_invalid_resume_rejected():
    sampler = ContinuousEpochSampler(Indices(25), 42, 4, 2)
    sampler.set_resume_batch_offset(5)
    with pytest.raises(ValueError, match='offset'):
        next(iter(sampler))


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2))
        self.seen = []

    def training_loss(self, sample):
        self.seen.append(sample['index'].tolist())
        x = sample['index'].float()
        loss = (self.weight*x + torch.randn_like(x) - .5).square().mean()
        return loss, {'loss_action': float(loss.detach())}


def tiny_trainer(accumulation, resume=None):
    trainer = object.__new__(Wan22Trainer)
    trainer.accelerator = Accelerator(cpu=True, gradient_accumulation_steps=accumulation,
                                      step_scheduler_with_optimizer=False)
    trainer.model = TinyModel()
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .99**step)
    trainer.train_sampler = ContinuousEpochSampler(Indices(8), 42, 2, 1)
    loader = DataLoader(Indices(8), batch_size=2, sampler=trainer.train_sampler,
                        generator=torch.Generator().manual_seed(42))
    trainer.model, trainer.optimizer, trainer.train_loader, trainer.scheduler = trainer.accelerator.prepare(
        trainer.model, optimizer, loader, scheduler)
    trainer.optimizer.zero_grad()
    trainer.global_step = trainer.epoch = trainer.batch_in_epoch = 0
    trainer.continuous_dataloader = True
    trainer.max_steps, trainer.max_grad_norm = 7, 1.0
    trainer.log_every = trainer.eval_every = 0
    trainer.save_every = 1
    trainer.save_steps = frozenset()
    trainer.save_final_checkpoint = False
    trainer._set_dit_only_train_mode = lambda: None
    trainer.snapshots = {}

    def save():
        trainer.snapshots[trainer.global_step] = dict(
            global_step=trainer.global_step, epoch=trainer.epoch, batch_in_epoch=trainer.batch_in_epoch,
            model=copy.deepcopy(trainer.accelerator.unwrap_model(trainer.model).state_dict()),
            optimizer=copy.deepcopy(trainer.optimizer.state_dict()),
            scheduler=copy.deepcopy(trainer.scheduler.state_dict()), rng=torch.get_rng_state().clone())
        return {'weights_path': 'toy', 'state_path': 'toy'}

    trainer.save_checkpoint = save
    if resume is not None:
        trainer.accelerator.unwrap_model(trainer.model).load_state_dict(resume['model'])
        trainer.optimizer.load_state_dict(resume['optimizer'])
        trainer.scheduler.load_state_dict(resume['scheduler'])
        trainer.global_step = resume['global_step']
        trainer.epoch = resume['epoch']
        trainer.batch_in_epoch = resume['batch_in_epoch']
        trainer.train_sampler.set_epoch_offset(trainer.epoch)
        trainer.train_sampler.set_resume_batch_offset(trainer.batch_in_epoch)
        torch.set_rng_state(resume['rng'])
    return trainer


@pytest.mark.parametrize('accumulation,resume_step', [(1,2), (1,4), (2,1), (2,2)])
def test_actual_training_loop_resume_matches_uninterrupted(accumulation, resume_step):
    torch.manual_seed(123)
    original = tiny_trainer(accumulation)
    original.train()
    original.train_loader.end()
    snapshot = original.snapshots[resume_step]
    resumed = tiny_trainer(accumulation, snapshot)
    resumed.train()
    resumed.train_loader.end()
    original_model = original.accelerator.unwrap_model(original.model)
    resumed_model = resumed.accelerator.unwrap_model(resumed.model)
    assert torch.equal(original_model.weight, resumed_model.weight)
    assert original_model.seen[resume_step*accumulation:] == resumed_model.seen
    assert (resumed.epoch, resumed.batch_in_epoch) == (original.epoch, original.batch_in_epoch)
    assert resumed.scheduler.get_last_lr() == original.scheduler.get_last_lr()
