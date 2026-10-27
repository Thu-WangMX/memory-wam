"""One third expert with configurable past and two future queries: P -> Q -> A."""
import math
import torch
from torch import nn
from fastwam.memory.past_future_action import MODE, SCHEMA, sampling_contract, past_future_layout, target_schema
from .fastwam_helios_past_action_sparse import FastWAMHeliosPastActionSparse16x8Flow
from .fastwam_helios_past_action_flow import FastWAMHeliosPastActionFlow
from .past_action_expert import PastActionIntervalFlowExpert


class PastFutureExpert(PastActionIntervalFlowExpert):
    allow_future = True
    FRESH_PREFIXES = PastActionIntervalFlowExpert.FRESH_PREFIXES + ('direction_embedding.',)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.direction_embedding = nn.Embedding(2, self.hidden_dim)

    def pre_queries(self, phase, relative_time, context, context_mask, freqs):
        pre = super().pre_queries(phase, relative_time, context, context_mask, freqs)
        pre['tokens'] = pre['tokens'] + self.direction_embedding((relative_time > 0).long())
        return pre


class FastWAMHeliosPastFuture(FastWAMHeliosPastActionSparse16x8Flow):
    target_schema = SCHEMA
    query_slots = 18
    future_query_slots = 2
    expert_type = PastFutureExpert
    query_layout = staticmethod(past_future_layout)

    def _initialize_past_action(self, config, action_config):
        self._configure_query_layout(config)
        self.target_schema = target_schema(self.source_history_slots, self.past_query_slots)
        expected = sampling_contract(self.source_history_slots, self.past_query_slots)
        if (self.prediction_horizon != 32 or config.get('target_mode') != MODE
                or any((config.get('label_contract') or {}).get(k) != v for k, v in expected.items())):
            raise ValueError('Missing/incompatible past/future prediction contract')
        FastWAMHeliosPastActionFlow._initialize_past_action(self, config, action_config)
        weight = float(config.get('future_loss_weight', 0.1))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('Future transition loss weight must be positive')
        self.helios_config['past_action'].update(
            mode=f'joint_action_past{self.past_query_slots}_future2_half8_flow_v1', target_mode=MODE,
            future_loss_weight=weight, **expected,
            attention='H_to_P;H_P_Q_to_Q;H_P_Q_A_to_A;no_F_to_P_Q_A',
            loss_reduction='separate_past_future_valid_mean_per_example',
            rope=f'original_visual_slots_past_future_endpoints_{self.source_history_slots+2}_{self.source_history_slots+3}',
            inference='joint_old_P_Q_A_states_shared_euler;third_expert_directed_mask')

    def _past_freqs(self, sample):
        slots = sample['past_action_visual_slots'].to(self.device, torch.long)
        valid = sample['past_action_valid'].to(self.device, torch.bool)
        split, source = self.past_query_slots, self.source_history_slots
        if (slots.shape != valid.shape or slots.shape[-1] != self.query_slots
                or (slots[:, :split][valid[:, :split]] < 0).any()
                or (slots[:, :split][valid[:, :split]] >= source).any()
                or (slots[:, :split][~valid[:, :split]] != -1).any()
                or not (slots[:, split:] == torch.tensor([source, source+1], device=self.device)).all()
                or not valid[:, split:].all()):
            raise ValueError('Invalid past/future visual positions')
        base = self.video_expert.freqs
        positions = (slots+2).float().masked_fill(~valid, 0)
        angles = positions[..., None] * torch.angle(base[0][1]).to(self.device).float()
        spatial = torch.cat([base[1][1].conj(), base[2][1].conj()]).to(self.device)
        return torch.cat((torch.polar(torch.ones_like(angles), angles),
                          spatial.expand(*slots.shape, -1)), -1).unsqueeze(2)

    def _past_pre(self, sample, vp, context, context_mask):
        pre, valid = super()._past_pre(sample, vp, context, context_mask)
        relation = torch.ones(self.query_slots, self.query_slots, device=self.device, dtype=torch.bool)
        relation[:self.past_query_slots, self.past_query_slots:] = False
        pre['query_attention_mask'] = relation
        return pre, valid

    def _three_expert_mask(self, vp, action_length, valid):
        mask = super()._three_expert_mask(vp, action_length, valid)
        n = vp['meta']['mtm']['L_hist'] + vp['meta']['mtm']['L_pred'] + action_length
        mask[:, n:n+self.past_query_slots, n+self.past_query_slots:] = False
        return mask

    def _training_expert_forward(self, vp, ap, inputs, sample):
        valid = sample['past_action_transition_valid'].to(self.device, torch.bool)
        loss_valid = sample['past_action_loss_valid'].to(self.device, torch.bool)
        if loss_valid.shape != valid.shape or (loss_valid & ~valid).any():
            raise ValueError('Invalid independent transition supervision mask')
        target = sample['past_action_target'].to(self.device, torch.float32).masked_fill(~loss_valid[..., None], 0)
        t = inputs['action_flow_timestep']
        noise = torch.randn_like(target).masked_fill(~valid[..., None], 0)
        noisy = self.train_action_scheduler.add_noise(target, noise, t)
        # Unobserved future tail has no teacher signal at any noise time.
        # Keep its query and input-noise distribution; never reveal episode length.
        noisy = torch.where(loss_valid[..., None], noisy, noise)
        pp, qvalid = self._flow_pre(sample, vp, inputs['video_context'], inputs['video_context_mask'], noisy, t)
        out = self._joint_past_forward(vp, ap, pp, qvalid)
        out['past_action_flow_target'] = self.train_action_scheduler.training_target(target, noise, t)
        out['past_action_flow_timestep'] = t
        return out

    def _extra_training_loss(self, out, sample):
        pred = self._past_predictions(out['past_action'])
        target = out['past_action_flow_target']
        valid = sample['past_action_loss_valid'].to(self.device, torch.bool)
        if pred.shape != target.shape or valid.shape != pred.shape[:-1] or not torch.isfinite(target).all():
            raise ValueError('Invalid past/future flow supervision')
        error = (pred-target).square().mean(-1)
        weight = self.train_action_scheduler.training_weight(out['past_action_flow_timestep'])
        total = pred.sum()*0
        metrics = {}
        for name, sl, scale in [('past', slice(0, self.past_query_slots), self.helios_config['past_action']['loss_weight']),
                                ('future', slice(self.past_query_slots, self.query_slots), self.helios_config['past_action']['future_loss_weight'])]:
            v = valid[:, sl]
            counts = v.sum((1, 2))
            loss = (((error[:, sl]*v).sum((1, 2))/counts.clamp(min=1))*weight).sum() / (counts > 0).sum().clamp(min=1)
            total = total + loss*scale
            metrics['loss_'+name+'_action'] = (loss*scale).detach()
            metrics['loss_'+name+'_action_unweighted'] = loss.detach()
            metrics[name+'_action_valid_transitions'] = counts.float().mean().detach()
        metrics['past_action_flow_sigma'] = (out['past_action_flow_timestep'].float().mean()/self.train_action_scheduler.num_train_timesteps).detach()
        return total, metrics
