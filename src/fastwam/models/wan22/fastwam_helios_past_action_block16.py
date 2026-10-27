"""99 deterministic queries reconstruct ordered within-interval transitions."""
import torch
import torch.nn.functional as F

from fastwam.memory.past_action import (
    INTERVAL_SCHEMA, INTERVAL_STEPS, LABEL_DIM, past_action_interval_layout)
from .fastwam_helios_past_action import FastWAMHeliosPastAction
from .past_action_expert import PastActionIntervalExpert


class FastWAMHeliosPastActionBlock16(FastWAMHeliosPastAction):
    target_schema = INTERVAL_SCHEMA
    prediction_width = INTERVAL_STEPS * LABEL_DIM
    expert_type = PastActionIntervalExpert
    query_layout = staticmethod(past_action_interval_layout)
    transition_slots = INTERVAL_STEPS

    def _initialize_past_action(self, config, action_config):
        label = config.get('label_contract') or {}
        if (label.get('query_target_shape') != [self.transition_slots, LABEL_DIM]
                or label.get('interval_alignment')
                != 'previous_decision_exclusive_to_current_decision_inclusive'
                or label.get('interval_padding') != 'chronological_left_aligned_invalid_tail'
                or label.get('window_predecessor') != 'taken_before_cropping_history'
                or label.get('loss_reduction')
                != 'mean_valid_transitions_per_example_then_mean_nonempty_examples'):
            raise ValueError('Missing/incompatible ordered interval16 target contract')
        super()._initialize_past_action(config, action_config)
        self.helios_config['past_action'].update(
            mode='deterministic_past_interval16_v1', target_mode='ordered_interval16',
            query_target_shape=[self.transition_slots, LABEL_DIM],
            query_clock='endpoint_raw_frame_age_div16_phase_and_observed_interval_length_v1',
            head_order='reshape_flat_projection_as_transition_then_latent_dimension')

    def _past_pre(self, sample, vp, context, context_mask):
        pre, valid = super()._past_pre(sample, vp, context, context_mask)
        lengths = sample['past_action_interval_length'].to(device=self.device, dtype=torch.long)
        if (lengths.shape != valid.shape or lengths.min() < 0 or lengths.max() > INTERVAL_STEPS
                or (valid & (lengths == 0)).any()):
            raise ValueError('Invalid observed interval lengths')
        pre['tokens'] = pre['tokens'] + self.past_action_expert.interval_length_embedding(lengths)
        return pre, valid

    def _past_predictions(self, tokens):
        return super()._past_predictions(tokens).unflatten(-1, (self.transition_slots, LABEL_DIM))

    def _past_prediction_valid(self, sample):
        return sample['past_action_transition_valid']

    def _extra_training_loss(self, out, sample):
        pred = self._past_predictions(out['past_action'])
        target = out.get('past_action_flow_target', sample['past_action_target']).to(
            device=pred.device, dtype=torch.float32)
        valid = self._past_prediction_valid(sample).to(device=pred.device, dtype=torch.bool)
        query_valid = sample['past_action_valid'].to(device=pred.device, dtype=torch.bool)
        if (target.shape != pred.shape or valid.shape != pred.shape[:-1]
                or not torch.equal(query_valid, valid.any(-1)) or not torch.isfinite(target).all()):
            raise ValueError('Invalid ordered DreamDojo interval supervision')
        error = F.mse_loss(pred, target, reduction='none').mean(-1)
        counts = valid.sum((1, 2))
        per_sample = (error * valid).sum((1, 2)) / counts.clamp(min=1)
        if 'past_action_flow_timestep' in out:
            per_sample = per_sample * self.train_action_scheduler.training_weight(
                out['past_action_flow_timestep'])
        unweighted = per_sample.sum() / (counts > 0).sum().clamp(min=1)
        loss = unweighted * self.helios_config['past_action']['loss_weight']
        metrics = dict(
            loss_past_action=loss.detach(),
            loss_past_action_unweighted=unweighted.detach(),
            past_action_valid_queries=query_valid.sum(1).float().mean().detach(),
            past_action_valid_transitions=counts.float().mean().detach())
        phases = sample['past_action_phase'].to(valid.device)
        for name, phase in [('demo', 1), ('execution', 3)]:
            mask = valid & (phases == phase)[..., None]
            metrics['past_action_mse_' + name] = (
                (error.detach() * mask).sum() / mask.sum().clamp(min=1)).detach()
        return loss, metrics

    def _past_inference_metadata(self, sample):
        return {name: sample[name][0].cpu() for name in (
            'past_action_transition_valid', 'past_action_transition_frames',
            'past_action_interval_start', 'past_action_interval_length')}
