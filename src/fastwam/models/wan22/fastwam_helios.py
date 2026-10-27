"""Helios latent memory on the shared RMBench FastWAM training/deployment contract.

No cross-decision Transformer KV is retained. The policy supplies its existing
continuous RGB prefix; action denoising reuses one compressed-history prefill.
"""
from copy import deepcopy
import logging
import torch
import torch.nn.functional as F
from .fastwam import FastWAM
from .helios_video import MTMVideoDiT
from fastwam.memory.helios_history import episode_anchors, select_history, validate_memory_variant, history_position_offset
from fastwam.memory.robomme_clock import (
    LEGACY_ANCHOR_MODE, PHASE_ANCHOR_MODE, select_robomme_anchors, validate_anchor_mode,
)
from fastwam.memory.helios_padding import (
    validate_history_padding, zero_invalid_history, apply_history_padding,
    history_padding_contract,
)


def load_backbone_initialization(model, path):
    """Strictly initialize the base FastWAM before adding Helios layers.

    This is weight initialization, not resume: optimizer and step are ignored.
    mmap avoids eagerly buffering the entire shared-filesystem checkpoint.
    """
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    model.mot.load_state_dict(payload['mot'], strict=True)
    if model.proprio_encoder is not None:
        model.proprio_encoder.load_state_dict(payload['proprio_encoder'], strict=True)
    elif payload.get('proprio_encoder'):
        raise ValueError('Initialization checkpoint has proprio weights but model does not')
    logging.getLogger(__name__).info(
        'BACKBONE_INIT_OK path=%s source_step=%s strict=True optimizer_and_step_reset=True',
        path, payload.get('step'))


class FastWAMHelios(FastWAM):
    @property
    def prediction_horizon(self):
        return int(self.helios_config.get('prediction_horizon', 16))

    @classmethod
    def from_wan22_pretrained(cls, *, helios_config=None, **kwargs):
        config = dict(helios_config or {})
        padding_mode = validate_history_padding(
            config.get('history_padding', 'legacy_zero'), config.get('memory_variant', 'multiscale'))
        anchor_mode = validate_anchor_mode(config.get('anchor_mode', LEGACY_ANCHOR_MODE))
        view_layout = config.get('view_layout', 'legacy')
        if view_layout != 'legacy':
            from fastwam.memory.robomme_views import view_spec
            view_spec(view_layout)
            if not config.get('observation_contract'):
                raise ValueError('Camera layout requires the RoboMME observation contract')
        if anchor_mode == PHASE_ANCHOR_MODE and not config.get('observation_contract'):
            raise ValueError('Phase anchors require the RoboMME observation contract')
        horizon = config.get('prediction_horizon', 16)
        if horizon not in (16, 32) or (horizon == 32 and not config.get('observation_contract')):
            raise ValueError('32-step prediction requires the RoboMME observation contract')
        if (kwargs.get('native_cache') or {}).get('enabled', False):
            raise ValueError('Helios must not be stacked with native_cache memory')
        if config.get('amplify_history', False):
            raise ValueError('Shared MoT port currently uses the original default amplify_history=false')
        model = super().from_wan22_pretrained(**kwargs)
        backbone_init = config.get('backbone_init_checkpoint')
        if backbone_init:
            load_backbone_initialization(model, backbone_init)
        sizes = tuple(config.get('history_sizes', (16, 2, 1)))
        anchor_frames = int(config.get('anchor_frames', 2))
        variant = config.get('memory_variant', 'multiscale')
        validate_memory_variant(variant, sizes, config.get('patch_long', (4,8,8)),
                                config.get('patch_mid', (2,4,4)), anchor_frames)
        if len(sizes) != 3 or sizes[-1] != 1:
            raise ValueError('Expected long/mid/current history sizes, current=1')
        if anchor_frames not in (0, 2):
            raise ValueError('Helios supports either zero (ablation) or two episode anchors')
        dit_cfg = deepcopy(dict(kwargs['video_dit_config']))
        dit_cfg['fuse_vae_embedding_in_latents'] = False
        dit_cfg.update(multi_term_memory=True, mtm_history_sizes=sizes,
                       mtm_anchor_size=anchor_frames,
                       mtm_full_history=(variant == 'full_history_dense'),
                       mtm_history_position_offset=history_position_offset(variant),
                       mtm_pred_size=horizon // 16, mtm_zero_history_timestep=True,
                       mtm_patch_kernel_long=tuple(config.get('patch_long', (4, 8, 8))),
                       mtm_patch_kernel_mid=tuple(config.get('patch_mid', (2, 4, 4))))
        # Share the already-loaded backbone parameters rather than allocating a
        # second 5B model. Only the two new patch convolutions need storage.
        with torch.device('meta'):
            video = MTMVideoDiT(**dit_cfg)
        result = video.load_state_dict(model.video_expert.state_dict(), strict=False, assign=True)
        expected = {'patch_mid.weight', 'patch_mid.bias', 'patch_long.weight', 'patch_long.bias'}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise ValueError(f'Unexpected backbone transfer: {result}')
        for patch in [video.patch_mid, video.patch_long]:
            patch.to_empty(device=model.device)
            patch.to(dtype=model.torch_dtype)
        video._init_mtm_patches_from_patch_embedding()
        # RoPE tables are plain tensors, not registered state-dict buffers.
        video.freqs = model.video_expert.freqs
        model.mot.mixtures['video'] = video
        model.helios_config = {'history_sizes': list(sizes),
                               'anchor_frames': anchor_frames,
                               'patch_long': list(video.mtm_patch_kernel_long),
                               'patch_mid': list(video.mtm_patch_kernel_mid),
                               'condition_augmentation': bool(config.get('condition_augmentation', True))}
        if horizon != 16:
            model.helios_config['prediction_horizon'] = horizon
        if anchor_mode != LEGACY_ANCHOR_MODE:
            model.helios_config['anchor_mode'] = anchor_mode
        if padding_mode != 'legacy_zero':
            model.helios_config['history_padding'] = padding_mode
        if view_layout != 'legacy':
            model.helios_config['view_layout'] = view_layout
        # Omit the default field to keep old Helios checkpoint contracts identical.
        if variant != 'multiscale':
            model.helios_config['memory_variant'] = variant
        observation_contract = config.get('observation_contract')
        if observation_contract is not None:
            from fastwam.memory.robomme_clock import CONTRACT
            if observation_contract != CONTRACT:
                raise ValueError('Unknown Helios observation contract')
            if model.proprio_dim != 8 or model.action_expert.action_dim != 8:
                raise ValueError('RoboMME requires native 8D action and proprioception')
            model.helios_config['observation_contract'] = observation_contract
            if not config.get('data_fingerprint'):
                raise ValueError('RoboMME requires a verified training data fingerprint')
            model.helios_config['data_fingerprint'] = config['data_fingerprint']
        model.freeze_unused_history_branches()
        model._initialize_future_mask_aux(config.get('future_mask_aux') or {})
        return model

    def freeze_unused_history_branches(self):
        # Trainer enables the full DiT first; keep removed branches out of ZeRO.
        long_size, mid_size, _ = self.helios_config['history_sizes']
        if long_size == 0:
            self.video_expert.patch_long.requires_grad_(False)
        if mid_size == 0:
            self.video_expert.patch_mid.requires_grad_(False)

    def _initialize_future_mask_aux(self, config):
        """Only the future branch accepts mask latents; inference stays RGB-only."""
        if not config.get('enabled', False):
            return
        weight = float(config.get('loss_weight', 1.0))
        row_start = int(config.get('latent_row_start', 8))
        if not 0 <= weight < float('inf') or row_start < 0:
            raise ValueError('Invalid future mask loss weight/latent row start')
        video = self.video_expert
        # W_rgb*x_rgb + W_mask*x_mask is a 96-channel concatenated projection,
        # with the RGB half shared with history/current, and exactly one bias.
        video.future_mask_patch = torch.nn.Conv3d(
            video.in_dim, video.hidden_dim, kernel_size=video.patch_size,
            stride=video.patch_size, bias=False,
            device=video.patch_embedding.weight.device,
            dtype=video.patch_embedding.weight.dtype)
        torch.nn.init.zeros_(video.future_mask_patch.weight)
        # Copy the RGB head so mask supervision reaches shared blocks on step 1.
        video.future_mask_head = deepcopy(video.head)
        self.helios_config['future_mask_aux'] = {
            'enabled': True, 'loss_weight': weight, 'latent_row_start': row_start,
            'schema': 'future_head_mask_local_clip_v1',
            'current_mask_input': False, 'action_sees_future': False}

    def _segments(self, history):
        # Input [B,H,C,1,h,w], output [B,C,H,h,w].
        sizes = self.helios_config['history_sizes']
        if self.helios_config.get('memory_variant') == 'full_history_dense':
            if history.ndim != 6 or history.shape[1] < sum(sizes):
                raise ValueError('Full history requires at least 19 padded latents')
            sizes = [history.shape[1] - 3, 2, 1]
        if history.ndim != 6 or history.shape[1] != sum(sizes):
            raise ValueError(f'Expected fixed {sum(sizes)}-frame Helios history')
        return history.squeeze(3).permute(0, 2, 1, 3, 4).split(sizes, dim=2)

    def _video_pre(self, anchors, history, future, timestep, context, context_mask,
                   future_mask=None, history_valid=None):
        padding_mode = validate_history_padding(
            self.helios_config.get('history_padding', 'legacy_zero'),
            self.helios_config.get('memory_variant', 'multiscale'))
        if padding_mode != 'legacy_zero':
            history = zero_invalid_history(history, history_valid)
        long, mid, current = self._segments(history)
        pre = self.video_expert.pre_dit(
            x=future, timestep=timestep, context=context, context_mask=context_mask,
            fuse_vae_embedding_in_latents=False,
            history_anchors=anchors.squeeze(3).permute(0, 2, 1, 3, 4),
            history_long=long, history_mid=mid, history_short=current)
        if future_mask is not None:
            if future_mask.shape != future.shape:
                raise ValueError('Future RGB and mask latent shapes must match')
            h = pre['meta']['mtm']['L_hist']
            mask_tokens = self.video_expert.future_mask_patch(future_mask).flatten(2).transpose(1, 2)
            pre['tokens'] = torch.cat((pre['tokens'][:, :h],
                                      pre['tokens'][:, h:] + mask_tokens), dim=1)
        if padding_mode != 'legacy_zero':
            pre = apply_history_padding(
                pre, history_valid, sizes=self.helios_config['history_sizes'],
                kernels=(self.video_expert.mtm_patch_kernel_long,
                         self.video_expert.mtm_patch_kernel_mid,
                         self.video_expert.patch_size),
                height=history.shape[-2], width=history.shape[-1],
                action_length=self.prediction_horizon, mode=padding_mode)
        return pre

    def _mask_post(self, tokens, pre):
        meta = pre['meta']['mtm']
        patches = self.video_expert.future_mask_head(tokens[:, meta['L_hist']:], pre['t'])
        return self.video_expert.unpatchify(patches, meta['grid_size_pred'])

    def _joint_mask(self, pre, action_length):
        m = pre['meta']['mtm']
        h, p = m['L_hist'], m['L_pred']
        mask = torch.zeros((h+p+action_length, h+p+action_length),
                           device=pre['tokens'].device, dtype=torch.bool)
        mask[:h, :h] = True
        mask[h:h+p, :h+p] = True
        mask[h+p:, :h] = True
        mask[h+p:, h+p:] = True
        if 'key_valid' in pre['meta']:
            key_valid = pre['meta']['key_valid']
            return mask.unsqueeze(0) & key_valid[:, None, :]
        return mask

    def _apply_history_batch_padding(self, pre, sample, action_length):
        if 'history_batch_pad' not in sample:
            return pre
        lengths = sample['history_unpadded_length'].tolist()
        batch_pad = sample['history_batch_pad'].to(pre['tokens'].device)
        b, max_frames = batch_pad.shape
        a = self.helios_config['anchor_frames']
        _, h, w = pre['meta']['grid_size']
        per_frame = h * w
        key_valid = torch.cat((
            torch.ones(b, a*per_frame, dtype=torch.bool, device=batch_pad.device),
            (~batch_pad).repeat_interleave(per_frame, dim=1),
            torch.ones(b, per_frame+action_length, dtype=torch.bool, device=batch_pad.device)), dim=1)
        freqs = []
        for length in lengths:
            # Keep every real token's RoPE identical to evaluation at this prefix,
            # independent of the longest episode in this minibatch.
            f = self.video_expert._build_segment_freqs(
                frame_indices=torch.arange(a+length+1, device=batch_pad.device),
                h=h, w=w, spatial_ratio=1, temporal_ratio=1)
            padding = f.new_ones(((max_frames-length)*per_frame, *f.shape[1:]))
            freqs.append(torch.cat((f[:a*per_frame], padding, f[a*per_frame:])))
        pre['freqs'] = torch.stack(freqs)
        pre['meta']['key_valid'] = key_valid
        pre['meta']['current_frame_indices'] = [a+length-1 for length in lengths]
        return pre

    def _action_freqs(self, action_length, pre):
        _, h, w = pre['meta']['grid_size']
        if self.prediction_horizon == 32:
            if action_length != 32:
                raise ValueError('Expected 32 action positions')
            def two_intervals(index):
                return torch.cat([self._build_video_aligned_action_freqs(
                    action_seq_len=16, temporal_base=index+block,
                    grid_h=h, grid_w=w, device=pre['tokens'].device)
                    for block in range(2)])
            if 'current_frame_indices' in pre['meta']:
                return torch.stack([two_intervals(i) for i in pre['meta']['current_frame_indices']])
            return two_intervals(pre['meta']['mtm']['current_frame_index'])
        if 'current_frame_indices' in pre['meta']:
            return torch.stack([self._build_video_aligned_action_freqs(
                action_seq_len=action_length, temporal_base=index,
                grid_h=h, grid_w=w, device=pre['tokens'].device)
                for index in pre['meta']['current_frame_indices']])
        return self._build_video_aligned_action_freqs(
            action_seq_len=action_length,
            temporal_base=pre['meta']['mtm']['current_frame_index'],
            grid_h=h, grid_w=w, device=pre['tokens'].device)

    def _apply_anchor_validity(self, pre, anchor_valid, action_length):
        # Legacy checkpoints retain their original padded-anchor semantics.
        if self.helios_config.get('anchor_mode', LEGACY_ANCHOR_MODE) != PHASE_ANCHOR_MODE:
            return pre
        batch = pre['tokens'].shape[0]
        anchors = self.helios_config['anchor_frames']
        if tuple(anchor_valid.shape) != (batch, anchors):
            raise ValueError('Phase anchor validity must be [B,A]')
        h, p = pre['meta']['mtm']['L_hist'], pre['meta']['mtm']['L_pred']
        _, height, width = pre['meta']['grid_size']
        anchor_tokens = anchors * height * width
        key_valid = pre['meta'].get('key_valid')
        if key_valid is None:
            key_valid = torch.ones((batch, h+p+action_length), dtype=torch.bool,
                                   device=pre['tokens'].device)
        else:
            key_valid = key_valid.clone()
        key_valid[:, :anchor_tokens] &= anchor_valid.to(
            device=key_valid.device, dtype=torch.bool).repeat_interleave(height*width, dim=1)
        pre['meta']['key_valid'] = key_valid
        pre['meta']['history_key_valid'] = key_valid[:, :h]
        return pre

    def _training_expert_forward(self, vp, ap, inputs, sample):
        return self.mot(
            embeds_all={'video': vp['tokens'], 'action': ap['tokens']},
            attention_mask=self._joint_mask(vp, ap['tokens'].shape[1]),
            freqs_all={'video': vp['freqs'], 'action': ap['freqs']},
            context_all={'video': {'context': vp['context'], 'mask': vp['context_mask']},
                         'action': {'context': ap['context'], 'mask': ap['context_mask']}},
            t_mod_all={'video': vp['t_mod'], 'action': ap['t_mod']})

    def _extra_training_loss(self, out, sample):
        return out['action'].new_zeros(()), {}

    def _prefill_inference_condition(self, vp, context, context_mask,
                                     observation_frame_indices, execution_start):
        h = vp['meta']['mtm']['L_hist']
        history_valid = vp['meta'].get('history_key_valid')
        if history_valid is None:
            history_mask = torch.ones((h, h), dtype=torch.bool, device=self.device)
            history_valid = torch.ones((vp['tokens'].shape[0], h), dtype=torch.bool, device=self.device)
        else:
            # Deployment serves one episode per policy; MoT prefill accepts 2D masks.
            if history_valid.shape != (1, h):
                raise ValueError('Helios cached inference requires one observed prefix')
            history_mask = history_valid.expand(h, -1)
        kv = self.mot.prefill_video_cache(
            video_tokens=vp['tokens'][:, :h], video_freqs=vp['freqs'][:h],
            video_t_mod=vp['t_mod'][:, :h],
            video_context_payload={'context': vp['context'], 'mask': vp['context_mask'][:, :h]},
            video_attention_mask=history_mask)
        return kv, history_valid, {}

    def training_loss(self, sample, tiled=False):
        inputs = self.build_inputs(sample, tiled=tiled)
        history = inputs['history_latents']
        anchors = sample.get('anchor_latents')
        anchor_valid = sample.get('anchor_valid')
        history_valid = sample.get('history_valid')
        expected_anchors = self.helios_config['anchor_frames']
        if (not isinstance(anchors, torch.Tensor) or anchors.ndim != 6
                or anchors.shape[0] != history.shape[0]
                or anchors.shape[1] != expected_anchors
                or tuple(anchors.shape[2:]) != tuple(history.shape[2:])):
            raise ValueError(
                'anchor_latents must be [B,A,C,1,h,w] and match history, '
                f'expected A={expected_anchors}')
        if (not isinstance(anchor_valid, torch.Tensor)
                or tuple(anchor_valid.shape) != tuple(anchors.shape[:2])):
            raise ValueError('anchor_valid must be [B,A] and match anchor_latents')
        if (not isinstance(history_valid, torch.Tensor)
                or tuple(history_valid.shape) != tuple(history.shape[:2])):
            raise ValueError('history_valid must be [B,H] and match history_latents')
        anchors = anchors.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True)
        future = inputs['input_latents'][:, :, 1:]
        action = inputs['action']
        if action.shape[1] != self.prediction_horizon or future.shape[2] != self.prediction_horizon // 16:
            raise ValueError('Action/video target lengths differ from the checkpoint prediction horizon')
        if self.helios_config['condition_augmentation']:
            ratio = torch.rand((history.shape[0], history.shape[1], 1, 1, 1, 1),
                               device=history.device).to(history.dtype)
            augmented = history*(1-ratio) + torch.randn_like(history)*ratio
            valid = history_valid.to(history.device).view(*history.shape[:2], 1, 1, 1, 1)
            history = torch.where(valid, augmented, history)
            anchor_ratio = torch.rand(
                (anchors.shape[0], anchors.shape[1], 1, 1, 1, 1),
                device=anchors.device, dtype=anchors.dtype)
            augmented_anchors = anchors * (1-anchor_ratio) + torch.randn_like(anchors) * anchor_ratio
            anchor_valid_mask = anchor_valid.to(anchors.device).view(
                *anchors.shape[:2], 1, 1, 1, 1)
            anchors = torch.where(anchor_valid_mask, augmented_anchors, anchors)
        tv = self.train_video_scheduler.sample_training_t(action.shape[0], self.device, future.dtype)
        ta = self.train_action_scheduler.sample_training_t(action.shape[0], self.device, action.dtype)
        nv, na = torch.randn_like(future), torch.randn_like(action)
        mask_config = self.helios_config.get('future_mask_aux')
        future_mask = noisy_mask = mask_noise = None
        if mask_config:
            future_mask = sample['future_mask_latent'].to(device=future.device, dtype=future.dtype)
            if future_mask.shape != future.shape or not torch.isfinite(future_mask).all():
                raise ValueError('Invalid cached future mask latent')
            if not torch.equal(sample['future_mask_image_is_pad'].to(inputs['image_is_pad'].device),
                               inputs['image_is_pad']):
                raise ValueError('RGB/mask temporal padding mismatch')
            mask_noise = torch.randn_like(future_mask)
            noisy_mask = self.train_video_scheduler.add_noise(future_mask, mask_noise, tv)
        vp = self._video_pre(anchors, history,
                             self.train_video_scheduler.add_noise(future, nv, tv), tv,
                             inputs['video_context'], inputs['video_context_mask'],
                             future_mask=noisy_mask, history_valid=history_valid)
        vp = self._apply_history_batch_padding(vp, sample, action.shape[1])
        vp = self._apply_anchor_validity(vp, anchor_valid, action.shape[1])
        ap = self.action_expert.pre_dit(
            self.train_action_scheduler.add_noise(action, na, ta), ta,
            inputs['context'], inputs['context_mask'])
        ap['freqs'] = self._action_freqs(action.shape[1], vp)
        out = self._training_expert_forward(
            vp, ap, dict(inputs, action_flow_timestep=ta), sample)
        pv = self.video_expert.post_dit(out['video'], vp)
        pa = self.action_expert.post_dit(out['action'], ap)
        lv = self._compute_video_loss_per_sample(
            pv, self.train_video_scheduler.training_target(future, nv, tv),
            inputs['image_is_pad'], include_initial_video_step=False)
        la = F.mse_loss(pa.float(), self.train_action_scheduler.training_target(action, na, ta).float(),
                        reduction='none').mean(dim=-1)
        if inputs['action_is_pad'] is not None:
            valid = (~inputs['action_is_pad']).to(la.dtype)
            la = (la*valid).sum(1)/valid.sum(1).clamp(min=1)
        else:
            la = la.mean(1)
        lv = (lv*self.train_video_scheduler.training_weight(tv)).mean()*self.loss_lambda_video
        la = (la*self.train_action_scheduler.training_weight(ta)).mean()*self.loss_lambda_action
        metrics = {'loss_video': lv.detach(), 'loss_action': la.detach(),
                   'helios_history_tokens': float(vp['meta']['mtm']['L_hist'])}
        extra_loss, extra_metrics = self._extra_training_loss(out, sample)
        metrics.update(extra_metrics)
        if mask_config:
            pm = self._mask_post(out['video'], vp)
            target = self.train_video_scheduler.training_target(future_mask, mask_noise, tv)
            row = mask_config['latent_row_start']
            if row >= pm.shape[-2]:
                raise ValueError('Mask supervision region is empty')
            # The top third is unannotated wrist-view padding, not negative labels.
            # Use the same temporal padding convention as the RGB auxiliary loss.
            lm = self._compute_video_loss_per_sample(
                pm[..., row:, :], target[..., row:, :], inputs['image_is_pad'],
                include_initial_video_step=False)
            lm = (lm * self.train_video_scheduler.training_weight(tv)).mean()
            weighted = lm * mask_config['loss_weight']
            metrics.update(loss_mask=weighted.detach(), loss_mask_unweighted=lm.detach())
            return lv + la + weighted + extra_loss, metrics
        return lv + la + extra_loss, metrics

    @torch.no_grad()
    def infer_action(self, prompt, input_image, action_horizon, proprio=None,
                     context=None, context_mask=None, negative_prompt=None, text_cfg_scale=1.,
                     num_inference_steps=50, sigma_shift=None, seed=None, rand_device='cpu',
                     tiled=False, full_kv_cache=None, full_kv_frame_index=0,
                     native_cache_state=None, observation_latents=None,
                     observation_frame_indices=None, execution_start=None, **kwargs):
        if action_horizon != self.prediction_horizon or full_kv_cache is not None or native_cache_state is not None:
            raise ValueError('Helios requires the trained prediction horizon and fresh per-decision Transformer KV')
        if self.proprio_encoder is not None and proprio is None:
            raise ValueError('This checkpoint requires proprioception, as in training')
        if text_cfg_scale != 1. or negative_prompt not in (None, '') or any(kwargs.values()):
            raise ValueError('Helios comparison does not enable guidance or external event-memory modes')
        self.eval()
        if self.helios_config.get('observation_contract') is not None:
            if input_image is not None or observation_latents is None:
                raise ValueError('RoboMME requires the phase-separated observed latent prefix')
            observations = observation_latents.to(device=self.device, dtype=self.torch_dtype)
            from fastwam.memory.robomme_views import view_spec
            expected_spatial = tuple(view_spec(self.helios_config.get('view_layout', 'legacy'))['latent_shape'][-2:])
            if (observations.ndim != 5 or observations.shape[:2] != (1, 48)
                    or observations.shape[-2:] != expected_spatial or not torch.isfinite(observations).all()):
                raise ValueError('Invalid RoboMME observed latent prefix')
        else:
            if observation_latents is not None:
                raise ValueError('External latents require an explicit observation contract')
            observations = self._encode_input_image_latents_tensor(input_image, tiled=tiled)
        if observations.shape[2] != full_kv_frame_index+1:
            raise ValueError('Online VAE prefix and decision index are inconsistent')
        if observation_frame_indices is not None and len(observation_frame_indices) != observations.shape[2]:
            raise ValueError('Original-frame query clock and VAE prefix length disagree')
        history, history_valid = select_history(observations[0].permute(1, 0, 2, 3).unsqueeze(2),
                                    self.helios_config['history_sizes'],
                                    self.helios_config.get('memory_variant', 'multiscale'))
        observed_prefix = observations[0].permute(1, 0, 2, 3).unsqueeze(2)
        if self.helios_config.get('observation_contract') is not None:
            anchors, anchor_valid = select_robomme_anchors(
                observed_prefix, observation_frame_indices, execution_start,
                size=self.helios_config['anchor_frames'],
                mode=self.helios_config.get('anchor_mode', LEGACY_ANCHOR_MODE))
        else:
            anchors, anchor_valid = episode_anchors(observed_prefix, self.helios_config['anchor_frames'])
        if prompt is not None:
            if context is not None or context_mask is not None:
                raise ValueError('Supply either prompt or cached text context')
            context, context_mask = self.encode_prompt(prompt)
        elif context is None or context_mask is None:
            raise ValueError('Text conditioning is required')
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        context = context.to(device=self.device, dtype=self.torch_dtype)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        action_context, action_mask = context, context_mask
        if proprio is not None:
            proprio = proprio.reshape(1, -1).to(device=self.device, dtype=self.torch_dtype)
            action_context, action_mask = self._append_proprio_to_context(context, context_mask, proprio)
        if action_horizon != self.prediction_horizon:
            raise ValueError('Requested action horizon differs from the trained model')
        future = observations.new_zeros((1, observations.shape[1], self.prediction_horizon // 16,
                                         *observations.shape[-2:]))
        vp = self._video_pre(anchors.unsqueeze(0), history.unsqueeze(0), future,
                             torch.zeros(1, device=self.device, dtype=self.torch_dtype), context, context_mask,
                             history_valid=history_valid.unsqueeze(0))
        vp = self._apply_anchor_validity(vp, anchor_valid.unsqueeze(0), action_horizon)
        # Future placeholders never influence history and are omitted from prefill.
        h = vp['meta']['mtm']['L_hist']
        kv, prefix_valid, extra = self._prefill_inference_condition(
            vp, context, context_mask, observation_frame_indices, execution_start)
        prefix_length = kv[0]['k'].shape[1]
        gen = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action = torch.randn((1, action_horizon, self.action_expert.action_dim),
                              generator=gen, device=rand_device).to(device=self.device, dtype=self.torch_dtype)
        mask = torch.ones((action_horizon, prefix_length+action_horizon),
                          dtype=torch.bool, device=self.device)
        mask[:, :prefix_length] = prefix_valid[0]
        freqs = self._action_freqs(action_horizon, vp)
        steps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps, self.device, action.dtype, shift_override=sigma_shift)
        action, extra = self._denoise_action_with_condition(
            action, steps, deltas, action_context, action_mask, freqs, kv, mask,
            prefix_length, extra, vp=vp, context=context, context_mask=context_mask,
            observation_frame_indices=observation_frame_indices, execution_start=execution_start,
            generator=gen, rand_device=rand_device)
        return {'action': action[0].float().cpu(), 'full_kv_cache': None, 'native_cache_state': None,
                'current_observation_latent': observations[:, :, -1:],
                'full_kv_frame_index': full_kv_frame_index, 'full_kv_video_tokens': h,
                'action_context': action_context, 'action_context_mask': action_mask,
                'video_context': context, 'video_context_mask': context_mask, **extra}

    def _denoise_action_with_condition(self, action, steps, deltas, action_context,
                                      action_mask, freqs, kv, mask, prefix_length,
                                      extra, **kwargs):
        for t, delta in zip(steps, deltas):
            prediction = self._predict_action_noise_with_cache(
                action, t.unsqueeze(0), action_context, action_mask, freqs, kv, mask, prefix_length)
            action = self.infer_action_scheduler.step(prediction, delta, action)
        return action, extra

    def _helios_inference_contract(self):
        contract = {'action_horizon': self.prediction_horizon, 'action_rope_spatial_mode': self.action_rope_spatial_mode,
                'action_scheduler_shift': float(self.infer_action_scheduler.shift),
                'video_scheduler_shift': float(self.infer_video_scheduler.shift),
                'action_num_train_timesteps': self.infer_action_scheduler.num_train_timesteps,
                'video_num_train_timesteps': self.infer_video_scheduler.num_train_timesteps,
                'history_vae': 'continuous_episode_stride4_causal_vae',
                'replan_stride': 16, 'anchor_frames': self.helios_config['anchor_frames'],
                'anchor_source_indices': ([0, 16] if self.helios_config['anchor_frames'] else []),
                'cross_decision_kv': False}
        if self.helios_config.get('observation_contract') is not None:
            from fastwam.memory.robomme_clock import inference_contract
            contract.update(inference_contract(
                self.helios_config['anchor_frames'],
                self.helios_config.get('anchor_mode', LEGACY_ANCHOR_MODE),
                self.helios_config.get('view_layout', 'legacy')))
        if self.prediction_horizon != 16:
            contract.update(future_video_latents=self.prediction_horizon // 16,
                            executed_action_horizon=16, action_selection='first16_of_prediction',
                            action_rope_time='two16step_blocks_base_plus_block_plus_i_div17',
                            prediction_contract='robomme_predict32_execute16_video2_v1')
        contract.update(history_padding_contract(self.helios_config.get('history_padding', 'legacy_zero')))
        return contract

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {'mot': self.mot.state_dict(), 'step': step,
                   'helios_contract': self.helios_config,
                   'inference_contract': self._helios_inference_contract(),
                   'memory_backend': 'helios', 'proprio_encoder': self.proprio_encoder.state_dict()
                   if self.proprio_encoder is not None else None}
        if optimizer is not None:
            payload['optimizer'] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        saved_helios = dict(payload.get('helios_contract') or {})
        saved_helios.setdefault('anchor_frames', 0)
        if payload.get('memory_backend') != 'helios' or saved_helios != self.helios_config:
            raise ValueError('Checkpoint memory backend/history contract differs from Helios config')
        saved_inference = dict(payload.get('inference_contract') or {})
        saved_inference.setdefault('anchor_frames', 0)
        saved_inference.setdefault('anchor_source_indices', [])
        if saved_inference != self._helios_inference_contract():
            raise ValueError('Checkpoint inference contract differs from runtime config')
        self.mot.load_state_dict(payload['mot'], strict=True)
        if self.proprio_encoder is not None:
            self.proprio_encoder.load_state_dict(payload['proprio_encoder'], strict=True)
        if optimizer is not None and 'optimizer' in payload:
            optimizer.load_state_dict(payload['optimizer'])
        return payload.get('step')
