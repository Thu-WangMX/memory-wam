"""Evaluation accounting must preserve failures and never silently drop episodes."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from collections import Counter

from experiments.robomme.campaign import (collect_results, write_json, pending_chunks,
                                         validate_resume, build_lanes, validate_memory_args)


class CampaignAccountingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = dict(tasks=['A', 'B'], episode_ids=[0, 1], split='val', max_steps=10000,
            checkpoint='/weights/step_020000.pt', data_fingerprint='audited', denoising_steps=50, seed=42)

    def part(self, task, attempt, rows, **contract_overrides):
        contract = {k: self.manifest[k] for k in ('checkpoint', 'data_fingerprint', 'denoising_steps', 'seed')}
        contract.update(contract_overrides)
        path = self.root / 'tasks' / task / 'chunk_000_001' / f'attempt_{attempt}' / 'summary.json'
        write_json(path, dict(task=task, split='val', max_steps=10000, contract=contract,
            results=[dict(episode=i, status=status, success=status=='success', steps=100, decisions=7)
                     for i, status in rows]))
        return path

    def test_partial_results_have_no_final_success_rate(self):
        self.part('A', 1, [(0, 'success'), (1, 'timeout')])
        result = collect_results(self.root, self.manifest)
        self.assertEqual((result['completed'], result['expected'], result['successes']), (2, 4, 1))
        self.assertIsNone(result['success_rate'])
        self.assertEqual(result['tasks']['A']['success_rate'], .5)
        self.assertEqual(result['tasks']['B']['missing_episode_ids'], [0, 1])

    def test_retry_keeps_completed_failures_and_counts_each_episode_once(self):
        self.part('A', 1, [(0, 'timeout')])
        self.part('A', 2, [(1, 'success')])
        self.part('B', 1, [(0, 'error'), (1, 'success')])
        result = collect_results(self.root, self.manifest)
        self.assertEqual(result['completed'], 4)
        self.assertEqual(result['success_rate'], .5)
        self.assertEqual(result['tasks']['B']['environment_errors'], 1)

    def test_duplicate_episode_is_rejected(self):
        self.part('A', 1, [(0, 'success')])
        self.part('A', 2, [(0, 'success')])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            collect_results(self.root, self.manifest)

    def test_wrong_checkpoint_and_invalid_success_labels_are_rejected(self):
        path = self.part('A', 1, [(0, 'success')], checkpoint='/other.pt')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            collect_results(self.root, self.manifest)
        path.unlink()
        path = self.part('A', 1, [(0, 'timeout')])
        payload = json.loads(path.read_text())
        payload['results'][0]['success'] = True
        write_json(path, payload)
        with self.assertRaisesRegex(ValueError, 'success label'):
            collect_results(self.root, self.manifest)

    def test_resume_schedules_only_missing_episodes_including_holes(self):
        manifest = dict(self.manifest, episode_ids=list(range(8)))
        self.part('A', 1, [(0, 'fail'), (2, 'success'), (3, 'timeout'), (6, 'success')])
        self.part('B', 1, [(0, 'error'), (1, 'success'), (2, 'success'), (3, 'fail'),
                           (4, 'fail'), (5, 'fail'), (6, 'fail'), (7, 'fail')])
        summary = collect_results(self.root, manifest)
        for size in (1, 3, 5, 10):
            jobs = list(pending_chunks(manifest, summary, size))
            episodes = [(task, episode) for task, start, count in jobs
                        for episode in range(start, start+count)]
            self.assertEqual(sorted(episodes), [('A', 1), ('A', 4), ('A', 5), ('A', 7)])

    def test_resume_allows_hardware_changes_but_rejects_protocol_changes(self):
        previous = dict(self.manifest, run='r', config='c', config_sha256='config',
            checkpoint_sha256='weights', history_sizes=[16,2,1], anchor_frames=2,
            official_episode_metadata={'A': 'fixed'}, gpu_indices=[6,7])
        validate_resume(previous, dict(previous, gpu_indices=list(range(8))))
        for key, value in [('seed',43), ('denoising_steps',25), ('split','test'),
                           ('checkpoint_sha256','different'), ('episode_ids',[0])]:
            with self.assertRaisesRegex(ValueError, key):
                validate_resume(previous, dict(previous, **{key: value}))

    def test_three_lanes_have_distinct_ports_and_storage_ids(self):
        lanes=build_lanes(list(range(8)),3,19300)
        self.assertEqual(len(lanes),24)
        self.assertEqual(Counter(gpu for gpu,_,_ in lanes),{gpu:3 for gpu in range(8)})
        self.assertEqual(len({lane for _,lane,_ in lanes}),24)
        self.assertEqual(len({port for _,_,port in lanes}),24)
        self.assertEqual(build_lanes([2,5],1,18260),[(2,'2',18260),(5,'5',18261)])
        for gpus,count,port in [([0,0],3,19300),([0],0,19300),([0],3,65535)]:
            with self.assertRaises(ValueError): build_lanes(gpus,count,port)

    def test_dedicated_budget_does_not_change_shared_gpu_rules(self):
        args=SimpleNamespace(policy_memory_gib=22,sim_memory_gib=1,gpu_budget_gib=80,
            stop_at_gib=78,robodojo_reserve_gib=0,memory_margin_gib=2,
            dedicated_gpus=True,fixed_memory=True,lanes_per_gpu=3)
        validate_memory_args(args)
        args.policy_memory_gib=26
        with self.assertRaisesRegex(ValueError,'exceed'): validate_memory_args(args)
        args.policy_memory_gib=18;args.dedicated_gpus=False
        with self.assertRaisesRegex(ValueError,'Shared'): validate_memory_args(args)
        args.lanes_per_gpu=1;args.gpu_budget_gib=35
        args.stop_at_gib=33;args.robodojo_reserve_gib=45
        validate_memory_args(args)

    def test_full_test50_queue_covers_all_800_distinct_episodes(self):
        manifest=dict(self.manifest,tasks=[f'task{i}' for i in range(16)],
                      episode_ids=list(range(50)),split='test',max_steps=1300)
        summary=collect_results(self.root,manifest)
        jobs=list(pending_chunks(manifest,summary,1))
        episodes=[(task,ep) for task,start,count in jobs for ep in range(start,start+count)]
        self.assertEqual(len(episodes),800)
        self.assertEqual(len(set(episodes)),800)
        self.assertEqual(Counter(task for task,_ in episodes),{task:50 for task in manifest['tasks']})


if __name__ == '__main__':
    unittest.main()
