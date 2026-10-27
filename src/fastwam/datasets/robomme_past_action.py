"""Opt-in DreamDojo supervision on the unchanged RoboMME Helios samples."""
from functools import partial
from .robomme import RoboMMEHeliosDataset
from fastwam.memory.past_action import (
    DreamDojoLabels, past_action_layout, past_action_interval_layout,
    SPARSE_MODE, past_action_sparse_layout, validate_query_counts)


class RoboMMEPastActionDataset(RoboMMEHeliosDataset):
    target_mode = 'endpoint'
    learning_mode = 'deterministic'
    query_layout = staticmethod(past_action_layout)
    sparse_queries = False
    future_query_slots = 0

    def __init__(self, *, dreamdojo_labels_root, past_query_slots=16, **kwargs):
        super().__init__(**kwargs)
        if self.history_sizes not in ((96, 2, 1), (72, 2, 1)) or self.anchor_frames != 2:
            raise ValueError('Past-action requires long99/long75 + two-anchor visual memory')
        self.source_history_slots, self.past_query_slots = validate_query_counts(
            sum(self.history_sizes), past_query_slots)
        layout_kwargs = dict(slots=self.source_history_slots)
        if self.sparse_queries:
            key = 'past_query_slots' if self.future_query_slots else 'query_slots'
            layout_kwargs[key] = self.past_query_slots
        self.query_layout = partial(type(self).query_layout, **layout_kwargs)
        self.dreamdojo = DreamDojoLabels(dreamdojo_labels_root, self.root, self.rows)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        row, current = self.samples[index]
        frames = self.observed_prefix(row, current)['frame_indices'].tolist()
        if frames[-1] != current:
            raise ValueError('Visual and latent-action query clocks disagree')
        layout = self.query_layout(frames, row['exec_start_idx'])
        sample.update(layout)
        sample['past_action_target'] = self.dreamdojo.targets(row['episode_index'], layout)
        return sample

    def dreamdojo_contract(self):
        return self.dreamdojo.contract(self.target_mode,
            source_history_slots=self.source_history_slots, past_query_slots=self.past_query_slots)


class RoboMMEPastActionBlock16Dataset(RoboMMEPastActionDataset):
    """Same 99 visual positions, with every within-interval adjacent transition."""
    target_mode = 'ordered_interval16'
    query_layout = staticmethod(past_action_interval_layout)


class RoboMMEPastActionBlock16FlowDataset(RoboMMEPastActionBlock16Dataset):
    learning_mode = 'flow_matching'

    def dreamdojo_contract(self):
        contract = super().dreamdojo_contract()
        contract.update(teacher_input=True,
                        teacher_input_policy='masked_flow_corrupted_training_only',
                        inference_teacher_input=False)
        return contract


class RoboMMEPastActionSparse16x8FlowDataset(RoboMMEPastActionBlock16FlowDataset):
    """Uniform half-rate blocks; the legacy import name keeps defaults 99/16."""
    target_mode = SPARSE_MODE
    sparse_queries = True
    query_layout = staticmethod(past_action_sparse_layout)
