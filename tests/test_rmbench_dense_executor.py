import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from fastwam.evaluation.rmbench_dense_executor import RMBenchActionExecutor


def environment(success_at=None):
    scene = SimpleNamespace(steps=0, get_timestep=lambda: 1 / 250)
    def step():
        scene.steps += 1
    scene.step = step
    robot = Mock()
    robot.get_left_arm_jointState.return_value = [0.] * 6 + [0.]
    robot.get_right_arm_jointState.return_value = [0.] * 6 + [1.]
    robot.get_left_arm_real_jointState.return_value = [.03] * 6 + [0.]
    robot.get_right_arm_real_jointState.return_value = [.03] * 6 + [1.]
    env = SimpleNamespace(scene=scene, robot=robot, take_action_cnt=0, step_lim=20,
        eval_success=False, is_dual_arm=True, eval_video_path='video',
        eval_video_ffmpeg=SimpleNamespace(stdin=Mock()),
        _get_eval_video_frame=lambda: np.zeros((2, 2, 3), dtype=np.uint8),
        _update_render=Mock(), get_obs=Mock(), render_freq=0, take_action=Mock())
    env.check_success = lambda: success_at is not None and scene.steps >= success_at
    return env


class TestDenseExecutor(unittest.TestCase):
    def test_fifteen_steps_drive_targets_velocities_and_gripper_ramp(self):
        env = environment()
        action = [.06] * 6 + [1.] + [-.06] * 6 + [0.]
        RMBenchActionExecutor('dense').execute(env, action)
        self.assertEqual(env.scene.steps, 15)
        self.assertEqual(env.take_action_cnt, 1)
        for call in env.robot.set_arm_joints.call_args_list:
            p, v, side = call.args
            np.testing.assert_allclose(p, .06 if side == 'left' else -.06)
            np.testing.assert_allclose(v, 1 if side == 'left' else -1)
        grips = env.robot.set_gripper.call_args_list
        np.testing.assert_allclose([c.args[0] for c in grips[::2]], np.arange(1,16)/15)
        np.testing.assert_allclose([c.args[0] for c in grips[1::2]], 1-np.arange(1,16)/15)
        self.assertEqual(env.eval_video_ffmpeg.stdin.write.call_count, 1)
        env.get_obs.assert_not_called()

    def test_success_latches_midway_and_no_further_action(self):
        env = environment(success_at=4)
        executor = RMBenchActionExecutor('dense')
        executor.execute(env, [0.] * 14)
        executor.execute(env, [0.] * 14)
        self.assertTrue(env.eval_success)
        self.assertEqual(env.scene.steps, 4)
        self.assertEqual(env.take_action_cnt, 1)
        self.assertEqual(env.eval_video_ffmpeg.stdin.write.call_count, 2)

    def test_limit(self):
        env = environment(); env.take_action_cnt = 20
        RMBenchActionExecutor('dense').execute(env, [0.] * 14)
        self.assertEqual(env.scene.steps, 0)

    def test_legacy_dispatch(self):
        env = environment(); action = [0.] * 14
        RMBenchActionExecutor().execute(env, action)
        env.take_action.assert_called_once_with(action, action_type='qpos')

    def test_measured_state_is_explicit(self):
        env = environment()
        RMBenchActionExecutor('dense', state_source='measured').execute(env, [.06]*14)
        np.testing.assert_allclose(env.robot.set_arm_joints.call_args_list[0].args[1], .5)

    def test_reject_bad_input_before_stepping(self):
        for action in ([0.]*16, [float('nan')]*14):
            env = environment()
            with self.assertRaises(ValueError):
                RMBenchActionExecutor('dense').execute(env, action)
            self.assertEqual(env.scene.steps, 0)
        for n in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                RMBenchActionExecutor('dense', n)


if __name__ == '__main__':
    unittest.main()
