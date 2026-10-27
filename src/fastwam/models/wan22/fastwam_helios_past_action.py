"""Visual history plus deterministic reconstruction of observed transitions."""
from copy import deepcopy
from functools import partial
import math

import torch
import torch.nn.functional as F

from fastwam.memory.past_action import LABEL_DIM, TARGET_SCHEMA, past_action_layout, validate_query_counts
from .fastwam_helios import FastWAMHelios
from .past_action_expert import PastActionExpert, PastActionMoT


class FastWAMHeliosPastAction(FastWAMHelios):
    target_schema = TARGET_SCHEMA
    prediction_width = LABEL_DIM
    expert_type = PastActionExpert
    query_layout = staticmethod(past_action_layout)
    teacher_input_contract = False
    query_slots = 99
    source_history_slots = 99
    past_query_slots = 16
    sparse_queries = False
    future_query_slots = 0

    @classmethod
    def from_wan22_pretrained(cls, *, helios_config=None, **kwargs):
        model = super().from_wan22_pretrained(helios_config=helios_config, **kwargs)
        model._initialize_past_action(
            dict(helios_config or {}).get('past_action') or {}, kwargs['action_dit_config'])
        return model

    def _configure_query_layout(self, config):
        source = sum(self.helios_config['history_sizes'])
        if config.get('source_history_slots', source) != source:
            raise ValueError('Past-action source_history_slots differs from visual history')
        self.source_history_slots, self.past_query_slots = validate_query_counts(
            source, config.get('past_query_slots', 16))
        kwargs = dict(slots=self.source_history_slots)
        self.query_slots = self.source_history_slots
        if self.sparse_queries:
            key = 'past_query_slots' if self.future_query_slots else 'query_slots'
            kwargs[key] = self.past_query_slots
            self.query_slots = self.past_query_slots + self.future_query_slots
        self.query_layout = partial(type(self).query_layout, **kwargs)

    def _initialize_past_action(self, config, action_config):
        self._configure_query_layout(config)
        if (not config.get('enabled') or self.helios_config['history_sizes'] not in ([96, 2, 1], [72, 2, 1])
                or self.helios_config['anchor_frames'] != 2):
            raise ValueError('Past-action expert requires long99/long75 + two anchors')
        contract = deepcopy(config.get('label_contract'))
        if (not isinstance(contract, dict) or contract.get('schema') != self.target_schema
                or contract.get('latent_dim') != LABEL_DIM
                or contract.get('teacher_input') is not self.teacher_input_contract
                or contract.get('transition_frame_stride') != 1
                or contract.get('boundary_supervision') is not False):
            raise ValueError('Past-action expert requires a verified DreamDojo label contract')
        for key in ('normalization_mean', 'normalization_std'):
            values = contract.get(key, [])
            if len(values) != LABEL_DIM or not all(math.isfinite(x) for x in values):
                raise ValueError('Invalid pinned DreamDojo normalization')
        if min(contract['normalization_std']) <= 1e-6:
            raise ValueError('Invalid pinned DreamDojo standard deviation')
        weight = float(config.get('loss_weight', 0.1))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('Past-action reconstruction loss weight must be positive')
        cfg = dict(action_config)
        cfg['action_dim'] = self.prediction_width
        # Independent parameters, initialized from the same action backbone.
        expert = self.expert_type(**cfg).to(device=self.device, dtype=self.torch_dtype)
        expert.initialize_backbone(self.action_expert)
        self.mot = PastActionMoT(
            {'video': self.video_expert, 'action': self.action_expert, 'past_action': expert},
            mot_checkpoint_mixed_attn=self.mot.mot_checkpoint_mixed_attn)
        self.helios_config['past_action'] = dict(
            enabled=True, mode='deterministic_past_v1', query_slots=self.query_slots,
            loss_weight=weight, action_reads_past=bool(config.get('action_reads_past', True)),
            label_contract=contract, query_clock='raw_frame_age_div16_and_phase_v1',
            rope='visual_slot_time_spatial_sentinel_minus1',
            attention='past_reads_observed_video_and_valid_past_only',
            initialization='independent_copy_of_action_backbone')
        if self.source_history_slots != 99:
            self.helios_config['past_action']['source_history_slots'] = self.source_history_slots

    @property
    def past_action_expert(self):
        return self.mot.mixtures['past_action']

    def _past_pre(self, sample, vp, context, context_mask):
        phase = sample['past_action_phase'].to(device=self.device, dtype=torch.long)
        relative = sample['past_action_relative_time'].to(device=self.device, dtype=torch.float32)
        valid = sample['past_action_valid'].to(device=self.device, dtype=torch.bool)
        if phase.shape != (vp['tokens'].shape[0], self.query_slots) or valid.shape != phase.shape:
            raise ValueError(f'Expected {self.query_slots} phase/time queries per example')
        freqs = self._past_freqs(sample)
        pre = self.past_action_expert.pre_queries(phase, relative, context, context_mask, freqs)
        if 'history_key_valid' in vp['meta']:
            pre['history_key_valid'] = vp['meta']['history_key_valid']
        return pre, valid

    def _past_freqs(self, sample):
        # A transition ending at a visual slot uses that slot's time and the
        # action modality's (-1,-1) spatial sentinel.
        base = self.video_expert.freqs
        positions = torch.arange(self.source_history_slots, device=self.device, dtype=torch.float32) + 2
        angles = positions[:, None] * torch.angle(base[0][1]).to(self.device).float()[None]
        temporal = torch.polar(torch.ones_like(angles), angles)
        spatial = torch.cat([base[1][1].conj(), base[2][1].conj()]).to(self.device)
        freqs = torch.cat((temporal, spatial[None].expand(self.source_history_slots, -1)), -1).unsqueeze(1)
        return freqs

    def _three_expert_mask(self, vp, action_length, valid):
        h, p = vp['meta']['mtm']['L_hist'], vp['meta']['mtm']['L_pred']
        n = h + p + action_length
        b, q = valid.shape
        mask = torch.zeros((b, n+q, n+q), dtype=torch.bool, device=valid.device)
        mask[:, :n, :n] = self._joint_mask(vp, action_length)
        mask[:, n:, :h] = True
        mask[:, n:, n:] = valid[:, None, :]
        if self.helios_config['past_action']['action_reads_past']:
            mask[:, h+p:n, n:] = valid[:, None, :]
        if 'history_key_valid' in vp['meta']:
            mask[:, :, :h] &= vp['meta']['history_key_valid'][:, None, :]
        return mask

    def _training_expert_forward(self, vp, ap, inputs, sample):
        # Only query metadata + observed visual/language features enter this path.
        # The teacher tensor is read exclusively in _extra_training_loss.
        pp, valid = self._past_pre(sample, vp, inputs['video_context'], inputs['video_context_mask'])
        return self._joint_past_forward(vp, ap, pp, valid)

    def _joint_past_forward(self, vp, ap, pp, valid):
        pre = dict(video=vp, action=ap, past_action=pp)
        return self.mot(
            embeds_all={k: v['tokens'] for k, v in pre.items()},
            attention_mask=self._three_expert_mask(vp, ap['tokens'].shape[1], valid),
            freqs_all={k: v['freqs'] for k, v in pre.items()},
            context_all={k: dict(context=v['context'], mask=v['context_mask']) for k, v in pre.items()},
            t_mod_all={k: v['t_mod'] for k, v in pre.items()})

    def _extra_training_loss(self, out, sample):
        pred = self._past_predictions(out['past_action'])
        target = sample['past_action_target'].to(device=pred.device, dtype=torch.float32)
        valid = sample['past_action_valid'].to(device=pred.device, dtype=torch.bool)
        if target.shape != pred.shape or not torch.isfinite(target).all():
            raise ValueError('Invalid DreamDojo supervision tensor')
        error = F.mse_loss(pred, target, reduction='none').mean(-1)
        counts = valid.sum(1)
        # Examples with no observed transition contribute zero, without NaNs.
        per_sample = (error * valid).sum(1) / counts.clamp(min=1)
        unweighted = per_sample.sum() / (counts > 0).sum().clamp(min=1)
        loss = unweighted * self.helios_config['past_action']['loss_weight']
        metrics = dict(loss_past_action=loss.detach(),
                       loss_past_action_unweighted=unweighted.detach(),
                       past_action_valid_queries=counts.float().mean().detach())
        for name, phase in [('demo', 1), ('execution', 3)]:
            mask = valid & (sample['past_action_phase'].to(valid.device) == phase)
            metrics['past_action_mse_' + name] = (
                (error.detach() * mask).sum() / mask.sum().clamp(min=1)).detach()
        return loss, metrics

    def _past_predictions(self, tokens):
        return self.past_action_expert.head(tokens).float()

    def _past_prediction_valid(self, sample):
        return sample['past_action_valid']

    def _past_inference_metadata(self, sample):
        return {}

    def _prefill_inference_condition(self, vp, context, context_mask,
                                     observation_frame_indices, execution_start):
        if observation_frame_indices is None or execution_start is None:
            raise ValueError('Past-action inference requires original-frame indices and execution_start')
        layout = self.query_layout(observation_frame_indices, execution_start)
        sample = {k: v.unsqueeze(0) for k, v in layout.items()}
        pp, valid = self._past_pre(sample, vp, context, context_mask)
        video_kv, video_valid, _ = super()._prefill_inference_condition(
            vp, context, context_mask, observation_frame_indices, execution_start)
        combined, key_valid, tokens = self.mot.prefill_past_action(pp, video_kv, valid)
        prediction_valid = self._past_prediction_valid(sample).to(device=tokens.device)
        prediction = self._past_predictions(tokens).masked_fill(~prediction_valid[..., None], 0)
        extra = dict(past_action_prediction=prediction[0].cpu(),
                     past_action_valid=valid[0].cpu(), past_action_query_tokens=self.query_slots,
                     **self._past_inference_metadata(sample))
        if not self.helios_config['past_action']['action_reads_past']:
            return video_kv, video_valid, extra
        return combined, key_valid, extra

    def _helios_inference_contract(self):
        contract = super()._helios_inference_contract()
        contract['past_action'] = deepcopy(self.helios_config['past_action'])
        return contract
