"""Predict32/execute16 with independently normalized past/future flow losses."""
from .robomme import RoboMMEHeliosDataset
from .robomme_past_action import RoboMMEPastActionSparse16x8FlowDataset
from fastwam.memory.past_action import SPARSE_MODE
from fastwam.memory.past_future_action import (
    MODE, target_schema, sampling_contract, past_future_layout, training_targets)


class RoboMMEPastFutureActionDataset(RoboMMEPastActionSparse16x8FlowDataset):
    target_mode = MODE
    future_query_slots = 2
    query_layout = staticmethod(past_future_layout)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.prediction_horizon != 32:
            raise ValueError('Past/future expert requires prediction_horizon=32')

    def __getitem__(self, index):
        sample = RoboMMEHeliosDataset.__getitem__(self, index)
        row, current = self.samples[index]
        frames = self.observed_prefix(row, current)['frame_indices'].tolist()
        if frames[-1] != current:
            raise ValueError('Visual and transition clocks disagree')
        layout = self.query_layout(frames, row['exec_start_idx'])
        target, loss_valid = training_targets(self.dreamdojo, row['episode_index'], layout)
        return dict(sample, **layout, past_action_target=target, past_action_loss_valid=loss_valid)

    def dreamdojo_contract(self):
        source, past = self.source_history_slots, self.past_query_slots
        contract = self.dreamdojo.contract(SPARSE_MODE,
            source_history_slots=source, past_query_slots=past)
        contract.update(schema=target_schema(source, past), teacher_input=True,
                        teacher_input_policy='masked_flow_corrupted_training_only',
                        inference_teacher_input=False, **sampling_contract(source, past))
        return contract
