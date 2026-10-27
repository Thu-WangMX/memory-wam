"""Visual memory plus configurable uniform, half-rate past-transition queries."""
import torch

from fastwam.memory.past_action import (
    SPARSE_MODE, SPARSE_SCHEMA, SPARSE_QUERIES, SPARSE_TRANSITIONS, LABEL_DIM,
    past_action_sparse_layout, sparse_sampling_contract, sparse_target_schema)
from .fastwam_helios_past_action_flow import FastWAMHeliosPastActionFlow


class FastWAMHeliosPastActionSparse16x8Flow(FastWAMHeliosPastActionFlow):
    target_schema = SPARSE_SCHEMA
    query_slots = SPARSE_QUERIES
    transition_slots = SPARSE_TRANSITIONS
    prediction_width = SPARSE_TRANSITIONS * LABEL_DIM
    query_layout = staticmethod(past_action_sparse_layout)
    sparse_queries = True

    def _initialize_past_action(self, config, action_config):
        self._configure_query_layout(config)
        self.target_schema = sparse_target_schema(self.source_history_slots, self.past_query_slots)
        label = config.get('label_contract') or {}
        sampling = sparse_sampling_contract(self.source_history_slots, self.past_query_slots)
        if (config.get('target_mode') != SPARSE_MODE
                or any(label.get(k) != v for k, v in sampling.items())):
            raise ValueError('Missing/incompatible sparse past-query sampling contract')
        super()._initialize_past_action(config, action_config)
        self.helios_config['past_action'].update(
            mode=f'joint_action_past_uniform{self.past_query_slots}_half8_flow_v1', target_mode=SPARSE_MODE,
            rope=f'original_long{self.source_history_slots}_visual_slot_time_spatial_sentinel_minus1',
            **sampling)

    def _past_freqs(self, sample):
        slots = sample['past_action_visual_slots'].to(self.device, torch.long)
        valid = sample['past_action_valid'].to(self.device, torch.bool)
        if (slots.shape != valid.shape or slots.shape[-1] != self.query_slots
                or (slots[valid] < 0).any()
                or (slots[valid] >= self.source_history_slots).any() or (slots[~valid] != -1).any()):
            raise ValueError('Invalid sparse query original visual slots')
        # Per-example original positions, never renumber the sampled subset.
        base = self.video_expert.freqs
        positions = (slots + 2).float().masked_fill(~valid, 0)
        angles = positions[..., None] * torch.angle(base[0][1]).to(self.device).float()
        temporal = torch.polar(torch.ones_like(angles), angles)
        spatial = torch.cat([base[1][1].conj(), base[2][1].conj()]).to(self.device)
        return torch.cat((temporal, spatial.expand(*slots.shape, -1)), -1).unsqueeze(2)

    def _past_inference_metadata(self, sample):
        return dict(super()._past_inference_metadata(sample),
                    past_action_visual_slots=sample['past_action_visual_slots'][0].cpu(),
                    past_action_transition_offsets=sample['past_action_transition_offsets'][0].cpu())
