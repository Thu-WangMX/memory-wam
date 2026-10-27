"""XPolicyLab bridge: joint packing only; history lives in AnchorHeliosPolicy."""
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state

from experiments.robodojo.policy import AnchorHeliosPolicy


class Model(ModelTemplate):
    def __init__(self, config):
        super().__init__()
        if config['env_cfg_type'] != 'arx_x5' or config['action_type'] != 'joint':
            raise ValueError('Anchor-Helios supports RoboDojo arx_x5 / joint')
        self.dims = get_robot_action_dim_info('arx_x5')
        if self.dims != {'arm_dim': [6, 6], 'ee_dim': [1, 1]}:
            raise ValueError(f'Unexpected robot dimensions: {self.dims}')
        self.policy = AnchorHeliosPolicy.from_checkpoint(
            config['config_path'], config['checkpoint_path'],
            prepared_root=config.get('prepared_root'), device=config.get('device', 'cuda:0'),
            num_inference_steps=int(config.get('num_inference_steps', 50)), seed=int(config.get('seed', 42)))
        self.action_horizon = self.replan_steps = 16

    def update_obs(self, obs):
        instruction = obs.get('task_instruction', obs.get('instruction', obs.get('instructions')))
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if instruction else None
        state = pack_robot_state(obs, 'joint', self.dims, source_type='obs', state_type='state')
        self.policy.observe(obs['vision'], state, instruction)

    def get_action(self):
        import logging
        import time
        import torch
        started=time.monotonic()
        result=unpack_robot_state(self.policy.predict(), 'joint', self.dims, source_type='obs')
        logging.info('HELIOS_ACTION step=%d rgb_frames=%d seconds=%.3f peak_gpu_gib=%.3f',
                     self.policy.observation_step,len(self.policy.frames),time.monotonic()-started,
                     torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_initialized() else 0)
        return result

    def update_obs_batch(self, obs_list):
        raise NotImplementedError('Use eval_batch=false: one stateful policy per environment')

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError('Use eval_batch=false: one stateful policy per environment')

    def reset(self):
        import logging
        logging.info('HELIOS_RESET previous_step=%d previous_rgb_frames=%d',self.policy.observation_step,len(self.policy.frames))
        self.policy.reset()

    def status(self):
        import os
        return {'policy':'FastWAMAnchorHelios','checkpoint_step':int(os.getenv('POLICY_EXPECTED_STEP','0')),
                'action_horizon':16,'replan_steps':16,'rgb_stride':4,
                'inference_steps':self.policy.num_inference_steps,
                'seed':self.policy.seed,'observation_step':self.policy.observation_step}
