"""Exercise shell launch coordination without allocating GPUs or using credentials."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


REPO = Path(__file__).resolve().parents[1]


class RoboMMELaunchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='robomme-launch-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('experiments/robomme/env.sh', 'experiments/robomme/train.sh',
                     'experiments/robomme/train_volc_all16_16gpu.sh', 'scripts/distributed_env.sh',
                     'scripts/memory_variant_args.py', 'src/fastwam/memory/helios_history.py'):
            dest = self.root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, dest)
        for package in ('src/fastwam', 'src/fastwam/memory'):
            (self.root / package / '__init__.py').touch()
        (self.root / 'runs').mkdir()
        bindir = self.root / 'venv/bin'
        bindir.mkdir(parents=True)
        stub = f'#!{sys.executable}\n' + '''import json, os, pathlib, sys
program = pathlib.Path(sys.argv[0]).name
if program == 'python' and 'scripts/memory_variant_args.py' in sys.argv:
    # Run the actual profile helper, without pretending it is a GPU operation.
    import runpy
    sys.argv = sys.argv[sys.argv.index('scripts/memory_variant_args.py'):]
    runpy.run_path(sys.argv[0], run_name='__main__')
    sys.exit(0)
event = dict(program=program, argv=sys.argv[1:], node=os.environ['NODE_RANK'],
             world=os.environ['NUM_GPUS'], master=os.environ['MASTER_ADDR'],
             visible=os.environ.get('CUDA_VISIBLE_DEVICES'))
with open(os.environ['FAKE_EVENTS'], 'a') as f:
    f.write(json.dumps(event) + '\\n')
if program == 'accelerate':
    print('fake training on node ' + os.environ['NODE_RANK'])
    if os.environ.get('FAKE_FAIL_RANK') == os.environ['NODE_RANK']:
        sys.exit(17)
elif sys.argv[-1] == '-':
    sys.stdin.read()
'''
        for name in ('python', 'accelerate'):
            dest = bindir / name
            dest.write_text(stub)
            dest.chmod(0o755)
        # Do not inherit real credentials, rank variables, or user Python paths.
        self.env = dict(PATH='/usr/bin:/bin', HOME=str(self.root),
                        ROBOMME_TRAIN_VENV=str(self.root / 'venv'),
                        FAKE_EVENTS=str(self.root / 'events.jsonl'),
                        RUN_ID='test_fresh_run', NUM_GPUS='16', GPUS_PER_NODE='8', NNODES='2',
                        MASTER_PORT='29639', MLP_WORKER_0_HOST='worker0.example',
                        ROBOMME_LAUNCH_TIMEOUT='3',
                        TRITON_CACHE_DIR=str(self.root / 'triton'),
                        TORCH_EXTENSIONS_DIR=str(self.root / 'torch_extensions'),
                        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
                        WANDB_API_KEY='test-placeholder', TOS_ACCESS_KEY='test-placeholder',
                        TOS_SECRET_KEY='test-placeholder')

    def worker(self, rank, *, generic=False, args=(), **extra):
        env = dict(self.env, MLP_ROLE_INDEX=str(rank), **extra)
        name = 'train.sh' if generic else 'train_volc_all16_16gpu.sh'
        return subprocess.Popen(['bash', str(self.root / 'experiments/robomme' / name), *args],
                                env=env, cwd=self.root, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def finish(self, process):
        try:
            output = process.communicate(timeout=45)[0]
        except subprocess.TimeoutExpired:
            process.kill()
            output = process.communicate()[0]
            self.fail('Launcher hung: ' + output)
        return process.returncode, output

    def events(self):
        path = self.root / 'events.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_worker_one_can_start_first_and_only_zero_uploads(self):
        one = self.worker(1)
        time.sleep(0.1)
        zero = self.worker(0)
        self.assertEqual(self.finish(zero)[0], 0)
        self.assertEqual(self.finish(one)[0], 0)
        events = self.events()
        launches = [e for e in events if e['program'] == 'accelerate']
        self.assertEqual(len(launches), 2)
        for e in launches:
            args = e['argv']
            for key, expected in [('--num_processes', '16'), ('--num_machines', '2'),
                                  ('--machine_rank', e['node']), ('--main_process_ip', 'worker0.example'),
                                  ('--deepspeed_multinode_launcher', 'standard')]:
                self.assertEqual(args[args.index(key) + 1], expected)
            for arg in ('batch_size=32', 'gradient_accumulation_steps=1', 'max_steps=20000',
                        'model.helios_config.history_sizes=[16,2,1]',
                        'wandb.name=test_fresh_run', 'model.helios_config.backbone_init_checkpoint=null'):
                self.assertIn(arg, args)
            self.assertEqual(e['visible'], self.env['CUDA_VISIBLE_DEVICES'])
            self.assertTrue((self.root / f'runs/test_fresh_run/train.worker-{e["node"]}.log').is_file())
        uploads = [e for e in events if '--run' in e['argv']]
        doctors = [e for e in events if '--doctor' in e['argv']]
        self.assertEqual([e['node'] for e in uploads], ['0'])
        self.assertEqual([e['node'] for e in doctors], ['0'])

    def test_dry_run_never_creates_run_or_uses_gpu(self):
        for rank in (0, 1):
            code, output = self.finish(self.worker(rank, ROBOMME_LAUNCH_DRY_RUN='1'))
            self.assertEqual(code, 0, output)
            args = shlex.split(output)
            self.assertEqual(args[args.index('--machine_rank') + 1], str(rank))
        self.assertEqual(self.events(), [])
        self.assertFalse((self.root / 'runs/test_fresh_run').exists())

    def test_mismatched_worker_recipe_is_rejected(self):
        self.assertEqual(self.finish(self.worker(0))[0], 0)
        code, output = self.finish(self.worker(1, BATCH_SIZE='16'))
        self.assertNotEqual(code, 0)
        self.assertIn('Workers disagree', output)
        self.assertEqual(len([e for e in self.events() if e['program'] == 'accelerate']), 1)

    def test_long99_switch_and_hydra_selector(self):
        for generic in (False, True):
            for variant, sizes in (('multiscale_long99', '[96,2,1]'), ('multiscale', '[16,2,1]')):
                code, output = self.finish(self.worker(0, generic=generic,
                    ROBOMME_LAUNCH_DRY_RUN='1', ROBOMME_MEMORY_VARIANT=variant))
                self.assertEqual(code, 0, output)
                args = shlex.split(output)
                self.assertIn('model.helios_config.history_sizes='+sizes, args)
                self.assertIn('model.helios_config.patch_long=[4,8,8]', args)
                self.assertIn('model.helios_config.anchor_frames=2', args)
        code, output = self.finish(self.worker(0, ROBOMME_LAUNCH_DRY_RUN='1',
            args=('model.helios_config.memory_variant=multiscale_long99',)))
        self.assertEqual(code, 0, output)
        self.assertIn('model.helios_config.history_sizes=[96,2,1]', shlex.split(output))
        self.assertEqual(self.events(), [])
        self.assertFalse((self.root / 'runs/test_fresh_run').exists())

    def test_mismatched_worker_memory_is_rejected(self):
        self.assertEqual(self.finish(self.worker(0, ROBOMME_MEMORY_VARIANT='multiscale_long99'))[0], 0)
        code, output = self.finish(self.worker(1, ROBOMME_MEMORY_VARIANT='multiscale'))
        self.assertNotEqual(code, 0)
        self.assertIn('Workers disagree', output)

    def test_unknown_or_conflicting_selectors_fail_before_launch(self):
        for variant, args in (('unknown', ()), ('multiscale',
                ('model.helios_config.memory_variant=multiscale_long99',))):
            code, output = self.finish(self.worker(0, generic=True,
                ROBOMME_MEMORY_VARIANT=variant, args=args))
            self.assertNotEqual(code, 0, output)
        self.assertEqual(self.events(), [])
        self.assertFalse((self.root / 'runs/test_fresh_run').exists())

    def test_failed_training_skips_upload(self):
        code, output = self.finish(self.worker(0, FAKE_FAIL_RANK='0'))
        self.assertEqual(code, 17, output)
        self.assertFalse(any('--run' in e['argv'] for e in self.events()))

    def test_existing_run_and_duplicate_worker_are_rejected(self):
        self.assertEqual(self.finish(self.worker(0))[0], 0)
        self.assertEqual(self.finish(self.worker(1))[0], 0)
        self.assertNotEqual(self.finish(self.worker(0))[0], 0)
        self.assertNotEqual(self.finish(self.worker(1))[0], 0)
        self.assertEqual(len([e for e in self.events() if e['program'] == 'accelerate']), 2)

    def test_multinode_loopback_is_rejected(self):
        code, output = self.finish(self.worker(0, MASTER_ADDR='127.0.0.1', ROBOMME_LAUNCH_DRY_RUN='1'))
        self.assertNotEqual(code, 0)
        self.assertIn('cannot be loopback', output)
        self.assertEqual(self.events(), [])

    def test_missing_node_zero_times_out(self):
        code, output = self.finish(self.worker(1, ROBOMME_LAUNCH_TIMEOUT='1'))
        self.assertNotEqual(code, 0)
        self.assertIn('Timed out waiting', output)
        self.assertFalse(any(e['program'] == 'accelerate' for e in self.events()))

    def test_single_node_keeps_original_log(self):
        code, output = self.finish(self.worker(0, generic=True, NNODES='1', NUM_GPUS='8',
                                               MASTER_ADDR='127.0.0.1', args=('batch_size=1',)))
        self.assertEqual(code, 0, output)
        self.assertTrue((self.root / 'runs/test_fresh_run/train.log').is_file())


if __name__ == '__main__':
    unittest.main()
