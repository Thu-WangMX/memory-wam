"""Record the actual policy observations without changing the rollout or RNG."""
import json
from pathlib import Path
import numpy as np


class EpisodeVideo:
    def __init__(self, output, episode, fps=20):
        self.output = Path(output)
        self.path = self.output / f'episode_{episode:03d}_front_wrist.mp4'
        self.partial = self.path.with_suffix('.partial.mp4')
        self.timeline_path = self.path.with_suffix('.frames.jsonl')
        self.timeline = self.timeline_path.open('w')
        self.writer = None
        self.frames = self.demo_frames = 0
        self.fps = fps
        self.shape = None
        self.closed = False

    def append(self, obs, step, initial=False):
        if obs is None:
            return
        fronts, wrists = obs['front_rgb_list'], obs['wrist_rgb_list']
        if len(fronts) != len(wrists):
            raise ValueError('Video camera observation counts differ')
        for i, (front, wrist) in enumerate(zip(fronts, wrists)):
            front, wrist = np.asarray(front), np.asarray(wrist)
            if any(x.ndim != 3 or x.shape[-1] != 3 or x.dtype != np.uint8 for x in (front,wrist)):
                raise ValueError('Expected uint8 RGB camera observations')
            shapes = (front.shape, wrist.shape)
            if self.shape is None:
                import imageio_ffmpeg
                self.shape = shapes
                self.height = (max(front.shape[0], wrist.shape[0])+15)//16*16
                self.width = (front.shape[1]+wrist.shape[1]+15)//16*16
                self.writer = imageio_ffmpeg.write_frames(str(self.partial), (self.width,self.height),
                    fps=self.fps,codec='libx264',quality=7,pix_fmt_in='rgb24',pix_fmt_out='yuv420p',
                    output_params=['-movflags','+faststart','-threads','2'],ffmpeg_log_level='error')
                self.writer.send(None)
            elif shapes != self.shape:
                raise ValueError('Camera resolution changed during episode')
            canvas = np.zeros((self.height,self.width,3),dtype=np.uint8)
            canvas[:front.shape[0],:front.shape[1]] = front
            canvas[:wrist.shape[0],front.shape[1]:front.shape[1]+wrist.shape[1]] = wrist
            demo = initial and i < len(fronts)-1
            self.writer.send(canvas)
            self.timeline.write(json.dumps(dict(frame=self.frames,step=None if demo else step,
                stage='demo' if demo else 'execution',observation_index=i))+'\n')
            self.frames += 1
            self.demo_frames += int(demo)

    def close(self, complete=False):
        if self.closed:
            return
        self.closed = True
        try:
            if self.writer is not None:
                self.writer.close()
        finally:
            self.timeline.close()
        if not complete:
            return
        if not self.frames or not self.partial.exists() or self.partial.stat().st_size < 100:
            raise RuntimeError('Episode video missing or empty; result not committed')
        self.partial.replace(self.path)
        return dict(kind='original_rollout',path=self.path.name,camera='front_wrist',
            bytes=self.path.stat().st_size,frames=self.frames,demo_frames=self.demo_frames,
            playback_fps=self.fps,execution_start_seconds=self.demo_frames/self.fps,
            timeline=self.timeline_path.name,front_shape=list(self.shape[0]),wrist_shape=list(self.shape[1]),
            note='Recorded during this scored rollout; front left, wrist right; playback fps is not a control-frequency assertion.')
