"""Joint action / ordered past-interval flow, with directed H -> P -> A attention."""
import torch

from .fastwam_helios import FastWAMHelios
from .fastwam_helios_past_action_block16 import FastWAMHeliosPastActionBlock16
from .past_action_expert import PastActionIntervalFlowExpert


class FastWAMHeliosPastActionFlow(FastWAMHeliosPastActionBlock16):
    expert_type = PastActionIntervalFlowExpert
    teacher_input_contract = True

    def _initialize_past_action(self, config, action_config):
        label = config.get('label_contract') or {}
        if (config.get('learning_mode') != 'flow_matching'
                or label.get('teacher_input_policy') != 'masked_flow_corrupted_training_only'
                or label.get('inference_teacher_input') is not False):
            raise ValueError('Missing/incompatible past-action flow training contract')
        super()._initialize_past_action(config, action_config)
        self.helios_config['past_action'].update(
            mode='joint_action_past_interval16_flow_v1', learning_mode='flow_matching',
            input='masked_noisy_ordered_block_plus_time_phase_length_queries',
            flow_path='x_sigma=(1-sigma)*z+sigma*eps;velocity=eps-z',
            flow_time='shared_with_action_branch',
            flow_weighting='same_scheduler_training_weight_as_action',
            inference='joint_euler_same_old_state_P_cache_then_A_simultaneous_update',
            cache_policy='H_once_P_recomputed_each_noise_step',
            inference_noise='independent_gaussian_draw_after_action_from_same_generator')

    def _flow_pre(self, sample, vp, context, context_mask, noisy, timestep):
        pre, valid = self._past_pre(sample, vp, context, context_mask)
        transitions = sample['past_action_transition_valid'].to(self.device, torch.bool)
        if (tuple(noisy.shape) != (*valid.shape, self.transition_slots, 64)
                or transitions.shape != noisy.shape[:-1]
                or not torch.equal(valid, transitions.any(-1))
                or timestep.shape != (valid.shape[0],)
                or not torch.isfinite(timestep).all()
                or (timestep < 0).any()
                or (timestep > self.train_action_scheduler.num_train_timesteps).any()):
            raise ValueError('Invalid past-action flow inputs/time/mask')
        # Tail padding and boundary values must never enter even the projection.
        noisy = noisy.masked_fill(~transitions[..., None], 0)
        if not torch.isfinite(noisy).all():
            raise ValueError('Nonfinite valid past-action flow input')
        return self.past_action_expert.add_flow_inputs(pre, noisy, timestep), valid

    def _training_expert_forward(self, vp, ap, inputs, sample):
        # Standard FM corruption; no clean-teacher-only conditioning shortcut.
        # P and A have the same sigma, independent Gaussian noises, and one
        # joint forward. P never reads future video, action, or proprioception.
        target = sample['past_action_target'].to(self.device, torch.float32)
        valid = sample['past_action_transition_valid'].to(self.device, torch.bool)
        target = target.masked_fill(~valid[..., None], 0)
        timestep = inputs['action_flow_timestep']
        noise = torch.randn_like(target).masked_fill(~valid[..., None], 0)
        noisy = self.train_action_scheduler.add_noise(target, noise, timestep)
        pp, query_valid = self._flow_pre(
            sample, vp, inputs['video_context'], inputs['video_context_mask'], noisy, timestep)
        out = self._joint_past_forward(vp, ap, pp, query_valid)
        out['past_action_flow_target'] = self.train_action_scheduler.training_target(
            target, noise, timestep)
        out['past_action_flow_timestep'] = timestep
        return out

    def _extra_training_loss(self, out, sample):
        if 'past_action_flow_target' not in out:
            raise ValueError('Past-action FM requires a velocity target')
        loss, metrics = super()._extra_training_loss(out, sample)
        # These are velocity errors, not z reconstruction errors.
        for phase in ('demo', 'execution'):
            metrics['past_action_velocity_mse_' + phase] = metrics.pop('past_action_mse_' + phase)
        metrics['past_action_flow_sigma'] = (
            out['past_action_flow_timestep'].detach().float().mean()
            / self.train_action_scheduler.num_train_timesteps).detach()
        return loss, metrics

    def _prefill_inference_condition(self, vp, context, context_mask,
                                     observation_frame_indices, execution_start):
        if observation_frame_indices is None or execution_start is None:
            raise ValueError('Past-action inference requires original-frame indices and execution_start')
        # Only H is invariant to diffusion time. No P KV may persist across steps.
        return FastWAMHelios._prefill_inference_condition(
            self, vp, context, context_mask, observation_frame_indices, execution_start)

    def _denoise_action_with_condition(self, action, steps, deltas, action_context,
                                      action_mask, freqs, kv, mask, prefix_length,
                                      extra, *, vp, context, context_mask,
                                      observation_frame_indices, execution_start,
                                      generator, rand_device):
        layout = self.query_layout(observation_frame_indices, execution_start)
        sample = {k: v.unsqueeze(0) for k, v in layout.items()}
        valid = sample['past_action_transition_valid'].to(self.device)
        # Integrate P in FP32, projecting to the expert's dtype only at its input.
        past = torch.randn((1, self.query_slots, self.transition_slots, 64),
                           generator=generator, device=rand_device,
                           dtype=torch.float32).to(self.device)
        past = past.masked_fill(~valid[..., None], 0)
        for t, delta in zip(steps, deltas):
            pp, query_valid = self._flow_pre(
                sample, vp, context, context_mask, past, t.unsqueeze(0))
            combined, key_valid, tokens = self.mot.prefill_past_action(pp, kv, query_valid)
            velocity_p = self._past_predictions(tokens).masked_fill(~valid[..., None], 0)
            if self.helios_config['past_action']['action_reads_past']:
                step_kv = combined
                length = combined[0]['k'].shape[1]
                step_mask = torch.cat((
                    key_valid[0], torch.ones(action.shape[1], device=self.device, dtype=torch.bool)
                ))[None].expand(action.shape[1], -1)
            else:
                step_kv, length, step_mask = kv, prefix_length, mask
            # Both velocities are evaluated at the OLD (A_sigma, P_sigma).
            velocity_a = self._predict_action_noise_with_cache(
                action, t.unsqueeze(0), action_context, action_mask,
                freqs, step_kv, step_mask, length)
            action = self.infer_action_scheduler.step(velocity_a, delta, action).to(self.torch_dtype)
            past = self.infer_action_scheduler.step(velocity_p, delta.float(), past)
            past = past.masked_fill(~valid[..., None], 0)
        extra = dict(
            extra, past_action_prediction=past[0].cpu(),
            past_action_valid=sample['past_action_valid'][0].cpu(),
            past_action_query_tokens=self.query_slots, past_action_flow_steps=len(steps),
            **self._past_inference_metadata(sample))
        return action, extra
