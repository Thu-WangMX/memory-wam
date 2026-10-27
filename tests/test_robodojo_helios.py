"""Causality, time alignment, reset, and real transport tests; no GPU needed."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from fastwam.datasets.robodojo import compose_rgb, text_filename
from experiments.robodojo.policy import AnchorHeliosPolicy


class RecordingModel:
    def __init__(self):
        self.calls = []
        self.torch_dtype = torch.float32

    def infer_action(self, **kwargs):
        self.calls.append(kwargs)
        return {'action': kwargs['proprio'].expand(16, 14).clone()}


def observation(step=0):
    return {
        'instruction': 'test task',
        'vision': {key: {'color': np.full((24, 32, 3), step, dtype=np.uint8)}
                   for key in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')},
        'state': {'left_arm_joint_state': np.arange(6, dtype=np.float32) / 10,
                  'left_ee_joint_state': np.array([0.6], np.float32),
                  'right_arm_joint_state': np.arange(7, 13, dtype=np.float32) / 10,
                  'right_ee_joint_state': np.array([1.3], np.float32)},
    }


class ContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        stats = {key: {'default': {'global_mean': [0.] * 14, 'global_std': [1.] * 14}}
                 for key in ('action', 'state')}
        (root / 'stats.json').write_text(json.dumps(stats))
        torch.save({'context': torch.ones(128, 4096), 'mask': torch.ones(128, dtype=torch.bool)},
                   root / text_filename('test task'))
        self.model = RecordingModel()
        self.policy = AnchorHeliosPolicy(self.model, root / 'stats.json', root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_camera_layout_and_rgb(self):
        cameras = [torch.full((1, 3, 12, 16), x, dtype=torch.uint8) for x in (25, 100, 200)]
        result = compose_rgb(cameras)
        torch.testing.assert_close(result[:, :, 0, 0], torch.full((1, 3), 100/255*2-1))
        torch.testing.assert_close(result[:, :, 0, -1], torch.full((1, 3), 200/255*2-1))
        torch.testing.assert_close(result[:, :, -1, 0], torch.full((1, 3), 25/255*2-1))

    def test_exact_history_and_reset(self):
        for step in range(33):
            self.policy.observe(observation(step)['vision'], np.arange(14, dtype=np.float32) / 10, 'test task')
            if step % 16 == 0:
                actions = self.policy.predict()
                np.testing.assert_allclose(actions[0], np.arange(14) / 10, atol=1e-6)
                call = self.model.calls[-1]
                self.assertEqual(call['full_kv_frame_index'], step // 16)
                pixels = call['input_image'][0, 0, :, 0, 0]
                torch.testing.assert_close(pixels, torch.arange(0, step+1, 4).float()/255*2-1)
        self.policy.reset()
        self.assertEqual(self.policy.frames, [])
        self.assertEqual(self.policy.observation_step, -1)
        with self.assertRaises(ValueError):
            self.policy.predict()
        self.policy.observe(observation()['vision'], np.zeros(14), 'test task')
        self.policy.predict()
        self.assertEqual(self.model.calls[-1]['full_kv_frame_index'], 0)
        self.assertEqual(self.model.calls[-1]['input_image'].shape[2], 1)

    def test_missing_observations_and_duplicate_inference_fail(self):
        self.policy.observe(observation()['vision'], np.zeros(14), 'test task')
        self.policy.predict()
        with self.assertRaises(ValueError):
            self.policy.predict()
        self.policy.observe(observation(1)['vision'], np.zeros(14), 'test task')
        with self.assertRaises(ValueError):
            self.policy.predict()
        with self.assertRaises(ValueError):
            self.policy.observe(observation()['vision'], np.zeros(14), 'different episode')

    def test_real_websocket_chunk_history_and_joint_order(self):
        from client_server.ws.model_client import WsModelClient
        from client_server.ws.model_server import PolicyServer, PolicyServerConfig
        from experiments.robodojo.xpolicy_model import Model
        bridge = Model.__new__(Model)
        bridge.dims = {'arm_dim': [6, 6], 'ee_dim': [1, 1]}
        bridge.policy = self.policy

        async def run():
            server = PolicyServer(bridge, PolicyServerConfig(host='127.0.0.1', port=0))
            await server.start()

            def client_test():
                client = WsModelClient(url=server.url, evaluation_id='helios-contract', trial_id='cpu',
                                       max_connect_attempts=1, request_timeout_s=30)
                try:
                    client.call(func_name='reset')
                    for step in range(17):
                        client.call(func_name='update_obs', obs=observation(step))
                        if step % 16 == 0:
                            actions = client.call(func_name='get_action')
                            self.assertEqual(len(actions), 16)
                            np.testing.assert_allclose(actions[0]['right_arm_joint_state'],
                                                       np.arange(7, 13)/10, atol=1e-6)
                    client.call(func_name='reset')
                    self.assertEqual(self.policy.frames, [])
                finally:
                    client.close()

            try:
                await asyncio.to_thread(client_test)
            finally:
                await server.stop()

        asyncio.run(run())


if __name__ == '__main__':
    unittest.main()
