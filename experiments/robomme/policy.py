"""Official RoboMME observation-batch protocol, preserving demo memory."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from fastwam.datasets.robomme import mosaic, normalizers
from fastwam.datasets.robodojo import load_context, sha256
from fastwam.memory.robomme_clock import (CONTRACT, decision_frames, phase_sample_indices,
    LEGACY_ANCHOR_MODE, PHASE_ANCHOR_MODE, validate_anchor_mode)
from experiments.robomme.streaming_vae import CausalVAEStream

OBS_KEYS = ('front_rgb_list', 'wrist_rgb_list', 'joint_state_list', 'gripper_state_list')


def state8(inputs, index):
    joints = np.asarray(inputs['joint_state_list'][index], dtype=np.float32).reshape(-1)
    fingers = np.asarray(inputs['gripper_state_list'][index], dtype=np.float32).reshape(-1)
    if joints.shape != (7,) or fingers.shape != (2,):
        raise ValueError('Expected seven arm joints and two finger positions')
    state = np.concatenate([joints, [fingers.mean()]]).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError('Nonfinite proprioception')
    return state


class AnchorHeliosPolicy:
    def __init__(self, model, stats_path, text_cache_dir, num_inference_steps=50, seed=42):
        if num_inference_steps < 1 or model.helios_config.get('observation_contract') != CONTRACT:
            raise ValueError('Invalid RoboMME inference configuration')
        self.model, self.norm = model, normalizers(stats_path)
        self.text_cache_dir = Path(text_cache_dir)
        self.steps, self.seed = num_inference_steps, seed
        self._contexts = {}
        self._text_fallback = None
        self.execution_encoder = CausalVAEStream(model.vae)
        self.reset()

    @classmethod
    def from_checkpoint(cls, config_path, checkpoint, device='cuda:0', num_inference_steps=50, seed=42):
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        anchor_mode = validate_anchor_mode(cfg.model.helios_config.get('anchor_mode', LEGACY_ANCHOR_MODE))
        if cfg.data.train.get('anchor_mode', LEGACY_ANCHOR_MODE) != anchor_mode:
            raise ValueError('Dataset/model anchor modes disagree at deployment')
        if cfg.data.train.get('view_layout', 'legacy') != cfg.model.helios_config.get('view_layout', 'legacy'):
            raise ValueError('Dataset/model camera layouts disagree at deployment')
        past_enabled = cfg.model.helios_config.get('past_action', {}).get('enabled', False)
        expected_dataset = 'fastwam.datasets.robomme.RoboMMEHeliosDataset'
        if past_enabled:
            mode = cfg.model.helios_config.past_action.get('target_mode', 'endpoint')
            names = dict(endpoint='RoboMMEPastActionDataset',
                         ordered_interval16='RoboMMEPastActionBlock16Dataset',
                         uniform16_half8='RoboMMEPastActionSparse16x8FlowDataset',
                          past16_future2_half8='RoboMMEPastFutureActionDataset')
            if mode not in names:
                raise ValueError(f'Unknown past-action target mode: {mode}')
            expected_dataset = 'fastwam.datasets.robomme_past_action.' + names[mode]
            if mode == 'past16_future2_half8':
                expected_dataset = 'fastwam.datasets.robomme_past_future.' + names[mode]
            learning = cfg.model.helios_config.past_action.get('learning_mode', 'deterministic')
            if learning == 'flow_matching' and mode == 'ordered_interval16':
                expected_dataset = 'fastwam.datasets.robomme_past_action.RoboMMEPastActionBlock16FlowDataset'
            elif mode in ('uniform16_half8', 'past16_future2_half8') and learning == 'flow_matching':
                pass
            elif mode in ('uniform16_half8', 'past16_future2_half8'):
                raise ValueError('Sparse16x8 requires flow_matching')
            elif learning != 'deterministic':
                raise ValueError(f'Unsupported past-action learning mode: {learning}')
        if (cfg.data.train._target_ != expected_dataset
                or cfg.model.helios_config.observation_contract != CONTRACT):
            raise ValueError('Expected a RoboMME training configuration')
        prepared = Path(cfg.data.train.prepared_root)
        stats = Path(config_path).parent/'dataset_stats.json'
        if (sha256(stats) != cfg.robomme_provenance.stats_sha256
                or sha256(prepared/'provenance.json') != cfg.robomme_provenance.assets_sha256):
            raise ValueError('Deployment assets differ from training')
        provenance = json.loads((prepared/'provenance.json').read_text())
        for name, digest in provenance['text_sha256'].items():
            if sha256(prepared/'text_cache'/name) != digest:
                raise ValueError('Deployment text differs from training')
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.skip_dit_load_from_pretrain = True
        model_cfg.action_dit_pretrained_path = None
        model_cfg.load_text_encoder = False
        model_cfg.helios_config.backbone_init_checkpoint = None
        if 'asv_init_checkpoint' in model_cfg.helios_config:
            model_cfg.helios_config.asv_init_checkpoint = None
        model = instantiate(model_cfg, device=device, model_dtype=torch.bfloat16)
        if sha256(model.model_paths['vae']) != cfg.robomme_provenance.vae_sha256:
            raise ValueError('Deployment VAE differs from training cache')
        model.load_checkpoint(str(checkpoint))
        model.eval().requires_grad_(False)
        return cls(model, stats, prepared/'text_cache', num_inference_steps, seed)

    def reset(self):
        self.initialized = False
        self.instruction = None
        self.demo_latents = None
        self.execution_encoder.reset()
        self.last_prefix = None
        self.decisions = 0
        self.observation_frame_indices = []
        self.execution_start = None

    @torch.no_grad()
    def _encode(self, video):
        # Demonstrations are encoded once, with their own empty causal state.
        return self.model._encode_video_latents(
            video.unsqueeze(0).to(device=self.model.device, dtype=self.model.torch_dtype), tiled=False)

    @torch.no_grad()
    def infer(self, inputs):
        started = time.perf_counter()
        count = len(inputs['front_rgb_list'])
        if count == 0 or any(len(inputs[k]) != count for k in OBS_KEYS):
            raise ValueError('Observation lists must be nonempty and aligned')
        first = bool(inputs.get('is_first_step', False))
        if first == self.initialized:
            raise ValueError('Reset/first-step protocol mismatch')
        goal = inputs['task_goal']
        instruction = str(goal[0] if isinstance(goal, (list, tuple)) else goal)
        if self.instruction is not None and instruction != self.instruction:
            raise ValueError('Instruction changed without reset')
        if instruction not in self._contexts:
            try:
                self._contexts[instruction] = load_context(self.text_cache_dir, instruction)
            except FileNotFoundError:
                # Held-out episodes can introduce new language variants. Encode
                # their instructions with the same frozen Wan T5/prompt/length.
                # Keep generated evaluation contexts outside training assets.
                from experiments.robodojo.eval_text import TextCache
                from fastwam.datasets.robodojo import text_filename
                if self._text_fallback is None:
                    self._text_fallback = TextCache(self.text_cache_dir,
                                                   self.text_cache_dir.parent/'eval_text_cache')
                self._text_fallback.ensure(instruction)
                directory = self._text_fallback.output
                self._contexts[instruction] = load_context(directory, instruction)
                print('ROBOMME_EVAL_TEXT '+json.dumps({'instruction': instruction,
                      'cache': str(directory/text_filename(instruction)),
                      'sha256': sha256(directory/text_filename(instruction))}), flush=True)
        self.instruction = instruction
        def frame(i):
            return mosaic(inputs['front_rgb_list'][i], inputs['wrist_rgb_list'][i],
                          view_layout=self.model.helios_config.get('view_layout', 'legacy'))
        if first:
            # Official reset: complete demonstration followed by execution init.
            execution = count - 1
            points = decision_frames(count, execution)[:-1]
            self.execution_start = execution
            self.observation_frame_indices = points + [execution]
            if points:
                self.demo_latents = self._encode(torch.stack(
                    [frame(i) for i in phase_sample_indices(points)], dim=1)).cpu()
            execution_frames = [frame(execution)]
            self.initialized = True
        else:
            if count != 16:
                raise ValueError(f'Expected all 16 post-action observations, got {count}')
            execution_frames = [frame(i) for i in (3, 7, 11, 15)]
            self.observation_frame_indices.append(self.observation_frame_indices[-1] + 16)
        # Append only new RGB frames. The encoder's causal state is independent
        # of the demo; the returned execution latent prefix is unchanged.
        execution_latents = self.execution_encoder.append(
            torch.stack(execution_frames, dim=1).unsqueeze(0).to(
                device=self.model.device, dtype=self.model.torch_dtype))
        prefix = execution_latents if self.demo_latents is None else torch.cat(
            [self.demo_latents.to(execution_latents), execution_latents], dim=2)
        self.last_prefix = prefix.detach().cpu()
        encoded_at = time.perf_counter()
        context, mask = self._contexts[instruction]
        proprio = self.norm['state'].forward(torch.from_numpy(state8(inputs, -1)))
        query_clock = {}
        if (self.model.helios_config.get('past_action', {}).get('enabled', False)
                or self.model.helios_config.get('anchor_mode') == PHASE_ANCHOR_MODE):
            if len(self.observation_frame_indices) != prefix.shape[2]:
                raise ValueError('Online past-action clock differs from VAE decisions')
            query_clock = dict(observation_frame_indices=self.observation_frame_indices,
                               execution_start=self.execution_start)
        result = self.model.infer_action(
            prompt=None, input_image=None, observation_latents=prefix,
            action_horizon=int(self.model.helios_config.get('prediction_horizon', 16)),
            proprio=proprio, context=context, context_mask=mask,
            full_kv_frame_index=prefix.shape[2]-1, num_inference_steps=self.steps,
            seed=self.seed, rand_device='cpu', **query_clock)
        action = self.norm['action'].backward(result['action'].float().cpu()).numpy().astype(np.float32)
        if action.shape != (int(self.model.helios_config.get('prediction_horizon', 16)), 8) or not np.isfinite(action).all():
            raise ValueError('Invalid policy action chunk')
        self.decisions += 1
        if self.decisions == 1 or self.decisions % 10 == 0:
            print('ROBOMME_POLICY_PROGRESS '+json.dumps(dict(
                decision=self.decisions, prefix_latents=prefix.shape[2],
                demo_latents=0 if self.demo_latents is None else self.demo_latents.shape[2],
                encode_seconds=round(encoded_at-started, 3),
                action_seconds=round(time.perf_counter()-encoded_at, 3))), flush=True)
        # Execute only the observed-clock stride, even when predicting 32.
        return {'actions': action[:16]}
