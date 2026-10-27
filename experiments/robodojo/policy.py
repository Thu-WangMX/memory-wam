"""Stateful RoboDojo boundary for the existing Anchor-Helios model.

The environment sends one observation per executed action, including inside a
chunk. We retain every fourth RGB observation and replan every 16 actions.
Only observed frames enter memory; predicted future video is never inserted.
"""
from pathlib import Path
import os

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.datasets.robodojo import (
    HORIZON, RGB_STRIDE, load_context, normalizers, observation_image,
)


class AnchorHeliosPolicy:
    def __init__(self, model, stats_path, text_cache_dir, *, num_inference_steps=50, seed=42):
        if num_inference_steps <= 0:
            raise ValueError('num_inference_steps must be positive')
        self.model = model
        self.norm = normalizers(stats_path)
        self.text_cache_dir = Path(text_cache_dir)
        self.num_inference_steps = int(num_inference_steps)
        self.seed = int(seed)
        self._contexts = {}
        self.reset()

    @classmethod
    def from_checkpoint(cls, config_path, checkpoint, *, device='cuda:0',
                        prepared_root=None, num_inference_steps=50, seed=42):
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        if cfg.model.memory_backend != 'helios':
            raise ValueError('RoboDojo AnchorHeliosPolicy requires a Helios training config')
        if cfg.data.train._target_ != 'fastwam.datasets.robodojo.RoboDojoHeliosDataset':
            raise ValueError('Expected the RoboDojo training config, not an RMBench checkpoint recipe')
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.skip_dit_load_from_pretrain = True
        model_cfg.action_dit_pretrained_path = None
        model_cfg.load_text_encoder = False
        model_cfg.helios_config.backbone_init_checkpoint = None
        model = instantiate(model_cfg, device=device, model_dtype=torch.bfloat16)
        step = model.load_checkpoint(str(checkpoint))  # strict weights + Helios/inference contract
        expected_step = os.getenv('POLICY_EXPECTED_STEP')
        if expected_step is not None and step != int(expected_step):
            raise ValueError(f'Checkpoint step {step} differs from requested {expected_step}')
        model.eval().requires_grad_(False)
        prepared = Path(prepared_root or cfg.data.train.prepared_root)
        stats = Path(config_path).parent / 'dataset_stats.json'
        if not stats.is_file():
            stats = prepared / 'dataset_stats.json'
        from fastwam.datasets.robodojo import sha256
        if sha256(stats) != cfg.robodojo_provenance.stats_sha256:
            raise ValueError('Evaluation normalization differs from training provenance')
        return cls(model, stats, prepared / 'text_cache',
                   num_inference_steps=num_inference_steps, seed=seed)

    def reset(self):
        self.observation_step = -1
        self.last_inferred_step = -HORIZON
        self.frames = []
        self.last_state = None
        self.instruction = None

    def observe(self, vision, state, instruction):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('Expected a finite 14D joint observation')
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('A nonempty task instruction is required')
        if self.instruction is not None and instruction != self.instruction:
            raise ValueError('Instruction changed inside an episode; reset the policy first')
        next_step = self.observation_step + 1
        if next_step > self.last_inferred_step + HORIZON:
            raise ValueError('Observations passed the next decision without an action request')
        if next_step % RGB_STRIDE == 0:
            self.frames.append(observation_image(vision))
        self.observation_step = next_step
        self.last_state = state.copy()
        self.instruction = instruction

    @torch.no_grad()
    def predict(self):
        step = self.observation_step
        if step < 0 or step != self.last_inferred_step + HORIZON:
            raise ValueError('Expected exactly 16 observed action steps between action requests')
        if len(self.frames) != 1 + step // RGB_STRIDE:
            raise ValueError('Continuous RGB history does not match the observation counter')
        if self.instruction not in self._contexts:
            try:
                self._contexts[self.instruction] = load_context(self.text_cache_dir, self.instruction)
            except FileNotFoundError:
                from experiments.robodojo.eval_text import request_context
                socket_path = os.environ.get('POLICY_TEXT_SOCKET')
                if not socket_path:
                    raise
                self._contexts[self.instruction] = request_context(self.instruction, socket_path)
        context, mask = self._contexts[self.instruction]
        state = self.norm['state'].forward(torch.from_numpy(self.last_state))
        prediction = self.model.infer_action(
            prompt=None, input_image=torch.stack(self.frames, dim=2).to(dtype=self.model.torch_dtype),
            action_horizon=HORIZON,
            proprio=state, context=context, context_mask=mask,
            full_kv_frame_index=step // HORIZON, num_inference_steps=self.num_inference_steps,
            seed=self.seed, rand_device='cpu', tiled=False,
        )
        action = prediction['action'].detach().float().cpu()
        if action.shape != (HORIZON, 14) or not torch.isfinite(action).all():
            raise ValueError(f'Invalid Anchor-Helios output: {action.shape}')
        self.last_inferred_step = step
        return self.norm['action'].backward(action).numpy().astype(np.float32)
