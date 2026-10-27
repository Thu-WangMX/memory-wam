"""Fixed-duration joint-drive execution for RMBench policy waypoints.

Reference semantics: OpenBMB/SimpleMemVLA rmbench_sim/rmbench_env.py
take_action_dense. Implemented independently; preserves FastWAM video hooks.
"""
import os
import numpy as np


class RMBenchActionExecutor:
    def __init__(self, mode="legacy", substeps=15, state_source="drive_target"):
        if mode not in ("legacy", "dense"):
            raise ValueError("executor must be legacy or dense")
        if isinstance(substeps, bool) or int(substeps) != substeps or substeps < 1:
            raise ValueError("dense substeps must be a positive integer")
        if state_source not in ("drive_target", "measured"):
            raise ValueError("state source must be drive_target or measured")
        self.mode, self.substeps, self.state_source = mode, int(substeps), state_source
        self._logged = False

    @classmethod
    def from_env(cls):
        return cls(os.getenv("RMBENCH_ACTION_EXECUTOR", "legacy"),
                   int(os.getenv("RMBENCH_DENSE_SUBSTEPS", "15")),
                   os.getenv("RMBENCH_DENSE_STATE_SOURCE", "drive_target"))

    def execute(self, env, action):
        if self.mode == "legacy":
            return env.take_action(action, action_type="qpos")
        if env.take_action_cnt >= env.step_lim or env.eval_success:
            return bool(env.eval_success)
        if not env.is_dual_arm:
            raise ValueError("dense executor currently requires dual-arm RMBench")
        robot = env.robot
        suffix = "arm_jointState" if self.state_source == "drive_target" else "arm_real_jointState"
        left = np.asarray(getattr(robot, "get_left_" + suffix)(), dtype=np.float64)
        right = np.asarray(getattr(robot, "get_right_" + suffix)(), dtype=np.float64)
        q = np.asarray(action, dtype=np.float64)
        if left.shape != (7,) or right.shape != (7,) or q.shape != (14,):
            raise ValueError("Expected [left arm 6, gripper, right arm 6, gripper]")
        if not all(np.isfinite(v).all() for v in (left, right, q)):
            raise ValueError("Nonfinite joint action/state")
        dt = float(env.scene.get_timestep())
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid physics timestep")
        duration = self.substeps * dt
        lv, rv = (q[:6] - left[:6]) / duration, (q[7:13] - right[:6]) / duration
        lg = np.linspace(left[6], q[6], self.substeps + 1)[1:]
        rg = np.linspace(right[6], q[13], self.substeps + 1)[1:]
        if not self._logged:
            print(f"[RMBench executor] dense substeps={self.substeps} dt={dt} "
                  f"waypoint_seconds={duration} control_hz={1 / duration} "
                  f"state_source={self.state_source}", flush=True)
            self._logged = True
        # Match the existing harness: one video frame and one counter tick per waypoint.
        if env.eval_video_path is not None:
            env.eval_video_ffmpeg.stdin.write(env._get_eval_video_frame().tobytes())
        env.take_action_cnt += 1
        print(f"step: {env.take_action_cnt} / {env.step_lim}", flush=True)
        for i in range(self.substeps):
            robot.set_arm_joints(q[:6], lv, "left")
            robot.set_gripper(float(lg[i]), "left")
            robot.set_arm_joints(q[7:13], rv, "right")
            robot.set_gripper(float(rg[i]), "right")
            env.scene.step()
            if env.check_success():
                env.eval_success = True
                break
        env._update_render()
        if env.render_freq:
            env.viewer.render()
        if env.eval_success:
            env.get_obs()
            if env.eval_video_path is not None:
                env.eval_video_ffmpeg.stdin.write(env._get_eval_video_frame().tobytes())
        return bool(env.eval_success)
