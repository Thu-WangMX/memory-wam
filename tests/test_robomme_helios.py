"""Phase reset, RGB order, causal sampling and official batch protocol."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from fastwam.memory.robomme_clock import CONTRACT, decision_frames, phase_sample_indices
from fastwam.datasets.robomme import mosaic
from fastwam.datasets.robodojo import text_filename
from experiments.robomme.policy import AnchorHeliosPolicy
from fastwam.utils.samplers import ResumableEpochSampler


class RecordingModel:
    helios_config = {'observation_contract': CONTRACT}
    device, torch_dtype = 'cpu', torch.float32
    vae = None

    def _encode_video_latents(self, video, **kwargs):
        # Model the causal 4x temporal reduction, preserving endpoint identity.
        return video[:, :1, ::4, :1, :1].expand(-1, 48, -1, 24, 20).clone()

    def infer_action(self, **kwargs):
        self.call = kwargs
        return {'action': kwargs['proprio'].expand(16, 8).clone()}


class RecordingStream:
    def __init__(self, vae):
        self.reset()

    def reset(self):
        self.chunks = []
        self.inputs = []

    def append(self, video):
        self.inputs.append(video.clone())
        self.chunks.append(video[:, :1, -1:, :1, :1].expand(-1,48,-1,24,20).clone())
        return torch.cat(self.chunks,dim=2)


def obs(start, end, first=False):
    frames = [np.full((12, 16, 3), i, np.uint8) for i in range(start, end)]
    return dict(front_rgb_list=frames, wrist_rgb_list=frames,
                joint_state_list=[np.arange(7, dtype=np.float32)/10]*len(frames),
                gripper_state_list=[np.array([.02, .04], np.float32)]*len(frames),
                task_goal='test', is_first_step=first)


class RoboMMETests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root/'stats.json').write_text(json.dumps({k: {'default': {'global_mean': [0.]*8,
                                           'global_std': [1.]*8}} for k in ('state','action')}))
        torch.save(dict(context=torch.ones(128,4096), mask=torch.ones(128,dtype=torch.bool)),
                   root/text_filename('test'))
        self.model = RecordingModel()
        with patch('experiments.robomme.policy.CausalVAEStream',RecordingStream):
            self.policy = AnchorHeliosPolicy(self.model, root/'stats.json', root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_irregular_final_demo_and_no_demo(self):
        self.assertEqual(decision_frames(81,33), [0,16,32,33,49,65])
        self.assertEqual(decision_frames(96,48), [0,16,32,47,48,64,80])
        self.assertEqual(phase_sample_indices([32,47]), [32,36,40,43,47])
        self.assertEqual(decision_frames(33,0), [0,16,32])

    def test_online_phase_reset_keeps_memory_and_uses_current_state(self):
        out = self.policy.infer(obs(0,49,True))
        np.testing.assert_allclose(out['actions'][0], [0,.1,.2,.3,.4,.5,.6,.03], atol=1e-6)
        values = self.policy.last_prefix[0,0,:,0,0]
        torch.testing.assert_close(values, torch.tensor([0,16,32,47,48])/255*2-1)
        self.policy.infer(obs(49,65))
        torch.testing.assert_close(self.policy.last_prefix[0,0,:,0,0],
                                   torch.tensor([0,16,32,47,48,64])/255*2-1)
        self.assertEqual(self.model.call['full_kv_frame_index'],5)
        chunks=self.policy.execution_encoder.inputs
        self.assertEqual([x.shape[2] for x in chunks],[1,4])
        torch.testing.assert_close(chunks[1][0,0,:,0,0],
                                   torch.tensor([52,56,60,64])/255*2-1)
        self.policy.reset()
        self.assertEqual(self.policy.execution_encoder.inputs,[])
        self.policy.infer(obs(0,1,True))
        self.assertEqual(self.policy.last_prefix.shape[2],1)

    def test_missing_frames_and_duplicate_first_rejected(self):
        with self.assertRaises(ValueError):
            self.policy.infer(obs(0,1))
        self.policy.infer(obs(0,1,True))
        with self.assertRaises(ValueError):
            self.policy.infer(obs(1,16))
        with self.assertRaises(ValueError):
            self.policy.infer(obs(0,1,True))

    def test_rgb_mosaic_order(self):
        front = np.zeros((12,16,3),np.uint8); front[...,0]=255
        wrist = np.zeros_like(front); wrist[...,1]=255
        m = mosaic(front,wrist)
        torch.testing.assert_close(m[:,200,0],torch.tensor([1.,-1.,-1.]))
        torch.testing.assert_close(m[:,0,0],torch.tensor([-1.,1.,-1.]))
        torch.testing.assert_close(m[:,:128,:160],m[:,:128,160:])

    def test_balanced_sampler_resume(self):
        class Dataset:
            def __len__(self): return 32
            def sample_epoch_indices(self,seed): return np.random.default_rng(seed).permutation(32).tolist()
        sampler=ResumableEpochSampler(Dataset(),42,2,2)
        first=list(sampler)
        sampler.set_resume_batch_offset(3)
        self.assertEqual(list(sampler),first[12:])
        sampler.set_epoch(1)
        self.assertEqual(len(list(sampler)),32)
        self.assertNotEqual(first,list(sampler))


if __name__=='__main__':
    unittest.main()
