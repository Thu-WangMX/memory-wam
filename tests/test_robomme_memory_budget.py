"""Capacity reservation and combined CUDA/graphics accounting regressions."""
from types import SimpleNamespace
import threading
from unittest.mock import patch
import unittest
from experiments.robomme.memory_budget import memory_allowance, process_memory, is_robodojo, MemoryGuard, GIB


class MemoryBudgetTest(unittest.TestCase):
    def test_current_robodojo_use_is_not_deducted_twice(self):
        for current in (0, 29, 40, 45):
            self.assertEqual(memory_allowance(80, current, 0, 0, 35, 45), 35)

    def test_other_processes_and_driver_overhead_reduce_our_allowance(self):
        self.assertAlmostEqual(memory_allowance(79.3, 29, 4, .3, 35, 45), 30)

    def test_reserve_actual_use_if_robodojo_exceeds_reservation(self):
        self.assertEqual(memory_allowance(80, 50, 5, 0, 35, 45), 25)
        self.assertEqual(memory_allowance(80, 50, 35, 0, 35, 45), 0)

    def test_compute_and_graphics_pid_deduplication(self):
        def record(pid, size):
            return SimpleNamespace(pid=pid, usedGpuMemory=size)
        result = process_memory([record(1, 100), record(2, 50)],
                                [record(1, 110), record(3, 25)])
        self.assertEqual(result, {1: 110, 2: 50, 3: 25})

    def test_unreadable_memory_fails_closed(self):
        for value in (None, -1, 2**64-1):
            with self.assertRaises(RuntimeError):
                process_memory([SimpleNamespace(pid=1, usedGpuMemory=value)], [])

    def test_unreadable_process_is_not_assumed_to_be_robodojo(self):
        with patch('pathlib.Path.read_bytes', side_effect=PermissionError):
            self.assertFalse(is_robodojo(1234, {}))

    def fake_guard(self, owned_bytes, external_bytes=0):
        guard=object.__new__(MemoryGuard)
        guard.lock=threading.Lock()
        guard.processes={123: SimpleNamespace(poll=lambda: None)}
        guard.handles=[0]
        guard.allowed_gpus={0}
        guard.stop_at=33*GIB;guard.budget=35*GIB;guard.reserve=45*GIB;guard.margin=2*GIB
        guard.peaks={};guard.violation=None
        guard.nvml=SimpleNamespace(
            nvmlDeviceGetComputeRunningProcesses=lambda h: [
                SimpleNamespace(pid=123, usedGpuMemory=owned_bytes),
                SimpleNamespace(pid=456, usedGpuMemory=external_bytes)],
            nvmlDeviceGetGraphicsRunningProcesses=lambda h: [],
            nvmlDeviceGetMemoryInfo=lambda h: SimpleNamespace(
                free=80*GIB-owned_bytes-external_bytes, used=owned_bytes+external_bytes))
        return guard

    @patch('experiments.robomme.memory_budget.is_robodojo', return_value=False)
    @patch('os.getpgid', side_effect=lambda pid: pid)
    def test_new_external_usage_triggers_stop_below_fixed_cap(self, *mocks):
        guard=self.fake_guard(20*GIB, 15*GIB)
        with self.assertRaisesRegex(RuntimeError, 'early-stop'):
            guard.snapshot()
        self.assertEqual(guard.last_sample['cards'][0]['budget_gib'],20)

    @patch('experiments.robomme.memory_budget.is_robodojo', return_value=False)
    @patch('os.getpgid', side_effect=lambda pid: pid)
    def test_only_small_accounted_enumeration_context_is_allowed_off_card(self, *mocks):
        guard=self.fake_guard(12*2**20)
        guard.allowed_gpus={1}
        self.assertGreater(guard.snapshot()['cards'][0]['owned_gib'],0)
        guard=self.fake_guard(65*2**20)
        guard.allowed_gpus={1}
        with self.assertRaisesRegex(RuntimeError, 'enumeration allowance'):
            guard.snapshot()


if __name__ == '__main__':
    unittest.main()
