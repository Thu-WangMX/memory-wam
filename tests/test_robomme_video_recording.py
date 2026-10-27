import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import imageio_ffmpeg
from experiments.robomme.video_recording import EpisodeVideo
from experiments.robomme.campaign import collect_results, write_json, validate_resume


class VideoRecordingTests(unittest.TestCase):
    def test_demo_execution_terminal_and_none_are_recorded_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            camera=np.full((32,32,3),64,dtype=np.uint8)
            obs={'front_rgb_list':[camera.copy() for _ in range(3)],
                 'wrist_rgb_list':[camera.copy() for _ in range(3)]}
            video=EpisodeVideo(tmp,4)
            video.append(obs,0,initial=True)
            next_obs={'front_rgb_list':[camera], 'wrist_rgb_list':[camera]}
            video.append(next_obs,1)
            video.append(next_obs,2)  # terminal observation must not be lost
            video.append(None,3)     # simulator error may return no image
            result=video.close(complete=True)
            video.close()           # finally cleanup is idempotent
            self.assertEqual((result['frames'],result['demo_frames']),(5,2))
            self.assertTrue(all(np.all(x==64) for x in obs['front_rgb_list']))
            decoded=imageio_ffmpeg.read_frames(str(Path(tmp)/result['path']))
            metadata=next(decoded)
            self.assertEqual(metadata['size'],(64,32))
            self.assertEqual(len(list(decoded)),5)
            timeline=[json.loads(x) for x in (Path(tmp)/result['timeline']).read_text().splitlines()]
            self.assertEqual([r['step'] for r in timeline],[None,None,0,1,2])
            self.assertEqual(result['kind'],'original_rollout')

    def test_empty_video_cannot_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'missing or empty'):
                EpisodeVideo(tmp,0).close(complete=True)

    def test_campaign_rejects_missing_video_and_counts_verified_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            contract=dict(checkpoint='/weights/test.pt',data_fingerprint='test',denoising_steps=10,seed=42)
            manifest=dict(contract,tasks=['A'],episode_ids=[0],split='test',max_steps=1300,save_video=True)
            part=root/'tasks/A/chunk_000_000/attempt_1/summary.json'
            row=dict(episode=0,status='fail',success=False)
            payload=dict(task='A',split='test',max_steps=1300,contract=contract,results=[row])
            write_json(part,payload)
            with self.assertRaisesRegex(ValueError,'Missing scored-rollout video'):
                collect_results(root,manifest)
            video=EpisodeVideo(part.parent,0)
            camera=np.zeros((32,32,3),dtype=np.uint8)
            video.append(dict(front_rgb_list=[camera],wrist_rgb_list=[camera]),0,initial=True)
            row['video']=video.close(complete=True)
            write_json(part,payload)
            result=collect_results(root,manifest)
            self.assertEqual((result['completed'],result['videos_saved']),(1,1))
            (part.parent/row['video']['path']).unlink()
            with self.assertRaisesRegex(ValueError,'Missing scored-rollout video'):
                collect_results(root,manifest)

    def test_resume_cannot_mix_unrecorded_results(self):
        with self.assertRaisesRegex(ValueError,'save_video'):
            validate_resume({}, {'save_video':True})


if __name__=='__main__': unittest.main()
