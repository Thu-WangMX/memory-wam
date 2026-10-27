"""ASV for Push75: action reads noisy future video; H and P/Q stay isolated.

Training uses the existing independent video/action noise times. Inference
integrates video and action/P/Q schedules together, evaluating every velocity
at the old state. Only the clean history's per-layer K/V survives noise steps.
"""
from copy import deepcopy
import logging
import torch

from .fastwam_helios_past_future import FastWAMHeliosPastFuture


class FastWAMHeliosPastFutureASV(FastWAMHeliosPastFuture):
    @classmethod
    def from_wan22_pretrained(cls, *, helios_config=None, **kwargs):
        config = dict(helios_config or {})
        if config.get('action_video_attention') != 'asv':
            raise ValueError('ASV needs an explicit saved action_video_attention=asv')
        if config.get('future_mask_aux', {}).get('enabled', False):
            raise ValueError('ASV currently supports RGB future prediction only')
        model = super().from_wan22_pretrained(helios_config=config, **kwargs)
        model._configure_asv()
        if config.get('asv_init_checkpoint'):
            model.initialize_asv_weights(config['asv_init_checkpoint'],
                                         config.get('asv_source_data_fingerprint'))
        return model

    def _configure_asv(self):
        # Capture the exact original contract before adding the new behavior.
        self._legacy_helios_contract = deepcopy(self.helios_config)
        self._legacy_inference_contract = deepcopy(
            FastWAMHeliosPastFuture._helios_inference_contract(self))
        self.helios_config['action_video_attention'] = 'asv'
        self.helios_config['past_action'].update(
            attention='H_to_P;H_P_Q_to_Q;H_F_P_Q_A_to_A;no_F_A_to_H_P_Q;no_A_P_Q_to_F',
            cache_policy='H_once;F_P_Q_A_recomputed_each_noise_step',
            inference='joint_old_F_P_Q_A_states;video_and_action_schedules;directed_ASV')

    def initialize_asv_weights(self, path, source_data_fingerprint):
        """Explicit migration, never a permissive checkpoint/resume loader."""
        if not source_data_fingerprint:
            raise ValueError('ASV initialization requires verified source data fingerprint')
        payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        expected = deepcopy(self._legacy_helios_contract)
        expected['data_fingerprint'] = source_data_fingerprint
        if (payload.get('memory_backend') != 'helios'
                or payload.get('helios_contract') != expected
                or payload.get('inference_contract') != self._legacy_inference_contract):
            raise ValueError('ASV source must match the old history-only architecture/data contract')
        self.mot.load_state_dict(payload['mot'], strict=True)
        if self.proprio_encoder is not None:
            self.proprio_encoder.load_state_dict(payload['proprio_encoder'], strict=True)
        elif payload.get('proprio_encoder'):
            raise ValueError('Unexpected source proprio encoder')
        logging.getLogger(__name__).info(
            'ASV_WEIGHT_INIT_OK path=%s source_step=%s strict=True optimizer_scheduler_step_reset=True',
            path, payload.get('step'))

    def _helios_inference_contract(self):
        result = super()._helios_inference_contract()
        result.update(action_video_attention='asv',
                      future_video_inference='joint_euler_from_independent_gaussian',
                      video_action_training_times='independent_existing_schedulers',
                      inference_state_update='all_velocities_from_same_old_state',
                      transformer_cache='observed_history_only',
                      noise_draw_order='action_then_latent_action_then_future_video')
        return result

    def _joint_mask(self, pre, action_length):
        mask = super()._joint_mask(pre, action_length)
        h, f = pre['meta']['mtm']['L_hist'], pre['meta']['mtm']['L_pred']
        mask[..., h+f:, h:h+f] = True
        return mask

    def _video_pre(self, anchors, history, future, timestep, context, context_mask,
                   future_mask=None, history_valid=None):
        pre = super()._video_pre(anchors, history, future, timestep, context,
                                 context_mask, future_mask, history_valid)
        pre['_asv_observed_inputs'] = (anchors, history, history_valid)
        pre['_asv_future_shape'] = tuple(future.shape)
        return pre

    def _joint_with_history_cache(self, vp, ap, pp, valid, history_cache):
        """Exact layer-wise joint MoT with only H's queries removed.

        Keys stay ordered [H,F,A,P,Q], exactly as in training. Cached history
        K/V are from the input to each layer, not the final history features.
        """
        h = vp['meta']['mtm']['L_hist']
        if len(history_cache) != self.mot.num_layers:
            raise ValueError('ASV history cache layer mismatch')
        future_pre = dict(vp)
        future_pre.update(tokens=vp['tokens'][:, h:], t_mod=vp['t_mod'][:, h:],
                          freqs=vp['freqs'][:, h:] if vp['freqs'].ndim == 4 else vp['freqs'][h:],
                          context_mask=vp['context_mask'][:, h:])
        pre = dict(video=future_pre, action=ap, past_action=pp)
        names = self.mot.expert_order
        states = {name: pre[name]['tokens'] for name in names}
        mask = self._three_expert_mask(vp, ap['tokens'].shape[1], valid)[:, h:, :]
        for index in range(self.mot.num_layers):
            cache = history_cache[index]
            if cache['k'].shape[1] != h or cache['v'].shape[1] != h:
                raise ValueError('ASV history cache layout mismatch')
            parts, post = [], []
            for name in names:
                expert = self.mot.mixtures[name]
                block = expert.blocks[index]
                q, k, v, residual, gate, shift, scale, mlp_gate, checkpoint = self.mot._build_expert_attention_io(
                    expert, block, states[name], pre[name]['freqs'], pre[name]['t_mod'])
                parts.append((q, k, v))
                post.append((block, residual, gate, shift, scale, mlp_gate, checkpoint))
            mixed = self.mot._mixed_attention(
                torch.cat([x[0] for x in parts], 1),
                torch.cat([cache['k']] + [x[1] for x in parts], 1),
                torch.cat([cache['v']] + [x[2] for x in parts], 1), mask)
            offset = 0
            for name, args in zip(names, post):
                length = states[name].shape[1]
                states[name] = self.mot._apply_post_with_optional_checkpoint(
                    *args, mixed[:, offset:offset+length],
                    dict(context=pre[name]['context'], mask=pre[name]['context_mask']))
                offset += length
        return states

    def _denoise_action_with_condition(self, action, steps, deltas, action_context,
                                      action_mask, freqs, kv, mask, prefix_length,
                                      extra, *, vp, context, context_mask,
                                      observation_frame_indices, execution_start,
                                      generator, rand_device):
        layout = self.query_layout(observation_frame_indices, execution_start)
        sample = {k: v.unsqueeze(0) for k, v in layout.items()}
        valid = sample['past_action_transition_valid'].to(self.device)
        past = torch.randn((1, self.query_slots, self.transition_slots, 64),
                           generator=generator, device=rand_device, dtype=torch.float32).to(self.device)
        past.masked_fill_(~valid[..., None], 0)
        future = torch.randn(vp['_asv_future_shape'], generator=generator,
                             device=rand_device, dtype=torch.float32).to(self.device)
        video_steps, video_deltas = self.infer_video_scheduler.build_inference_schedule(
            len(steps), self.device, self.torch_dtype)
        anchors, history, history_valid = vp['_asv_observed_inputs']
        for ta, da, tv, dv in zip(steps, deltas, video_steps, video_deltas):
            current = self._video_pre(anchors, history, future.to(self.torch_dtype), tv[None],
                                      context, context_mask, history_valid=history_valid)
            # Carry anchor-key validity from the one-time inference preparation.
            for key in ('key_valid', 'history_key_valid'):
                if key in vp['meta']:
                    current['meta'][key] = vp['meta'][key]
            ap = self.action_expert.pre_dit(action, ta[None], action_context, action_mask)
            ap['freqs'] = freqs
            pp, qvalid = self._flow_pre(sample, current, context, context_mask, past, ta[None])
            out = self._joint_with_history_cache(current, ap, pp, qvalid, kv)
            va = self.action_expert.post_dit(out['action'], ap)
            vl = self._past_predictions(out['past_action']).masked_fill(~valid[..., None], 0)
            vf = self.video_expert.unpatchify(
                self.video_expert.head(out['video'], current['t']),
                current['meta']['mtm']['grid_size_pred'])
            # No branch receives another branch's already-updated state.
            action = self.infer_action_scheduler.step(va, da, action).to(self.torch_dtype)
            past = self.infer_action_scheduler.step(vl.float(), da.float(), past).masked_fill(~valid[..., None], 0)
            future = self.infer_video_scheduler.step(vf.float(), dv.float(), future)
        return action, dict(extra, past_action_prediction=past[0].cpu(),
                            past_action_valid=sample['past_action_valid'][0].cpu(),
                            past_action_query_tokens=self.query_slots, past_action_flow_steps=len(steps),
                            future_video_prediction=future.cpu(), future_video_flow_steps=len(steps),
                            **self._past_inference_metadata(sample))
