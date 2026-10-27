import ast
import copy
from pathlib import Path
import unittest

import torch
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from fastwam.lr_schedules import WarmupStableDecay


class WSDScheduleTests(unittest.TestCase):
    def test_requested_100k_curve(self):
        schedule = WarmupStableDecay(100000, 2000, 30000)
        for step, expected in [(0, 0.0005), (999, 0.5), (1999, 1),
                               (2000, 1), (20000, 1), (70000, 1),
                               (85000, 0.505), (100000, 0.01), (110000, 0.01)]:
            self.assertAlmostEqual(schedule(step), expected, places=12)
        values = [schedule(x) for x in range(70000, 100001)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))

    def test_invalid_phases_fail_early(self):
        for total, warmup, decay in [(100, 90, 20), (100, 0, 0), (100, -1, 10)]:
            with self.assertRaises(ValueError):
                WarmupStableDecay(total, warmup, decay)

    def test_resume_through_decay_is_identical(self):
        def make():
            optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
            scheduler = LambdaLR(optimizer, WarmupStableDecay(100, 2, 30))
            return optimizer, scheduler
        optimizer, scheduler = make()
        for _ in range(72):
            optimizer.step()
            scheduler.step()
        opt_state, sched_state = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict())
        resumed_opt, resumed_sched = make()
        resumed_opt.load_state_dict(opt_state)
        resumed_sched.load_state_dict(sched_state)
        for _ in range(28):
            optimizer.step(); scheduler.step()
            resumed_opt.step(); resumed_sched.step()
            self.assertEqual(scheduler.get_last_lr(), resumed_sched.get_last_lr())
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1e-6, places=12)

    def test_trainer_dispatch_and_existing_schedules(self):
        # Compile the real method without importing video/model dependencies.
        source = Path(__file__).resolve().parents[1] / 'src/fastwam/trainer.py'
        tree = ast.parse(source.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == '_build_scheduler')
        scope = dict(LambdaLR=LambdaLR, WarmupStableDecay=WarmupStableDecay,
                     CosineAnnealingLR=CosineAnnealingLR, ConstantLR=ConstantLR,
                     LinearLR=LinearLR, SequentialLR=SequentialLR)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
        obj = type('Holder', (), {})()
        obj.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
        obj.learning_rate = 1e-4
        build = scope['_build_scheduler']
        self.assertIsInstance(build(obj, 'wsd', 100000, 2000, decay_steps=30000), LambdaLR)
        with self.assertRaises(ValueError):
            build(obj, 'wsd', 100000, 2000)
        self.assertIsInstance(build(obj, 'cosine', 100, 0), CosineAnnealingLR)
        self.assertIsInstance(build(obj, 'constant', 100, 0), ConstantLR)


if __name__ == '__main__':
    unittest.main()
