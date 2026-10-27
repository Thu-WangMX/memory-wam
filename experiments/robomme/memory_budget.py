"""Reserve RoboDojo capacity while watching our CUDA and Vulkan usage.

Allocator caps live in serve.py/eval.py. This NVML monitor is an early-stop
guard for compute and graphics processes, not hardware memory isolation.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

GIB = 2**30
# SAPIEN's Vulkan device enumeration creates a ~6 MiB graphics context on GPU 0
# even when rendering explicitly on another GPU. Include it in accounting and
# permit only this small aggregate overhead outside the assigned cards.
ENUMERATION_ALLOWANCE = 64 * 2**20
ROBODOJO_PYTHON = '/mnt/vepfs01/output/spidy.wang/runtime-robodojo/venv/bin/python'


def process_memory(compute, graphics):
    # Both APIs report total memory; do not double-count a PID present in both.
    result = {}
    for process in [*compute, *graphics]:
        used = process.usedGpuMemory
        if used is None or used < 0 or used >= 2**63:
            raise RuntimeError(f'Cannot account for GPU process {process.pid}')
        result[process.pid] = max(result.get(process.pid, 0), int(used))
    return result


def memory_allowance(allocatable, robodojo, other, unattributed, cap, reserve):
    # Existing RoboDojo usage is part of its reservation, not an extra deduction.
    return max(0, min(cap, allocatable - max(reserve, robodojo) - other - unattributed))


def is_robodojo(pid, cache):
    if pid in cache:
        return cache[pid]
    ancestors = []
    result = False
    try:
        while pid > 1 and pid not in ancestors:
            if pid in cache:
                result = cache[pid]
                break
            ancestors.append(pid)
            argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            executable = os.fsdecode(argv[0])
            if executable in (ROBODOJO_PYTHON, ROBODOJO_PYTHON + '3', ROBODOJO_PYTHON + '3.11'):
                result = True
                break
            # The comm field may contain spaces and parentheses.
            stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            pid = int(stat[1])
    except (OSError, ValueError, IndexError):
        pass  # Unknown processes conservatively count as other users.
    for ancestor in ancestors:
        cache[ancestor] = result
    return result


class MemoryGuard:
    def __init__(self, output, nvml_path, allowed_gpus, stop_at_gib, budget_gib,
                 reserve_gib, margin_gib, on_violation):
        spec = importlib.util.spec_from_file_location('_robomme_nvml', nvml_path)
        self.nvml = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.nvml
        spec.loader.exec_module(self.nvml)
        self.nvml.nvmlInit()
        self.handles = [self.nvml.nvmlDeviceGetHandleByIndex(i)
                        for i in range(self.nvml.nvmlDeviceGetCount())]
        self.output = Path(output)
        self.allowed_gpus = set(allowed_gpus)
        if not self.allowed_gpus or not self.allowed_gpus.issubset(range(len(self.handles))):
            raise ValueError('Selected GPU indices do not exist')
        self.stop_at = stop_at_gib * GIB
        self.budget = budget_gib * GIB
        self.reserve = reserve_gib * GIB
        self.margin = margin_gib * GIB
        self.on_violation = on_violation
        self.processes = {}
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self.peaks = {}
        self.thread = None
        self.violation = None
        self.last_sample = None
        self.snapshot()  # Fail before launch if accounting is unavailable.

    def register(self, process):
        with self.lock:
            self.processes[process.pid] = process
            if self.violation is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def unregister(self, process):
        if process is not None:
            with self.lock:
                self.processes.pop(process.pid, None)

    def snapshot(self):
        with self.lock:
            groups = {pid for pid, process in self.processes.items() if process.poll() is None}

        def owns(pid):
            try:
                return os.getpgid(pid) in groups
            except (ProcessLookupError, PermissionError):
                return False

        cards, violations, classification = {}, [], {}
        for index, handle in enumerate(self.handles):
            usage = process_memory(self.nvml.nvmlDeviceGetComputeRunningProcesses(handle),
                                   self.nvml.nvmlDeviceGetGraphicsRunningProcesses(handle))
            owned = {pid: size for pid, size in usage.items() if owns(pid)}
            if not owned and index not in self.allowed_gpus:
                continue
            info = self.nvml.nvmlDeviceGetMemoryInfo(handle)
            robodojo = {pid: size for pid, size in usage.items()
                        if pid not in owned and is_robodojo(pid, classification)}
            others = {pid: size for pid, size in usage.items()
                      if pid not in owned and pid not in robodojo}
            # Exclude driver reservations and deduct usage absent from PID tables.
            unattributed = max(0, info.used - sum(usage.values()))
            budget = memory_allowance(info.free + info.used, sum(robodojo.values()),
                sum(others.values()), unattributed, self.budget, self.reserve)
            threshold = max(0, min(self.stop_at, budget - self.margin))
            total = sum(owned.values())
            self.peaks[index] = max(self.peaks.get(index, 0), total)
            cards[index] = dict(owned_gib=total / GIB, peak_gib=self.peaks[index] / GIB,
                robodojo_gib=sum(robodojo.values()) / GIB, other_gib=sum(others.values()) / GIB,
                unattributed_gib=unattributed / GIB, allocatable_gib=(info.free+info.used) / GIB,
                free_gib=info.free / GIB, budget_gib=budget / GIB, early_stop_gib=threshold / GIB,
                owned_processes_gib={str(pid): size / GIB for pid, size in owned.items()},
                robodojo_pids=list(robodojo), other_pids=list(others))
            if total and total >= threshold:
                violations.append(f'GPU {index}: owned {total/GIB:.3f} GiB >= early-stop '
                                  f'{threshold/GIB:.3f} GiB (budget {budget/GIB:.3f} GiB)')
            if total > ENUMERATION_ALLOWANCE and index not in self.allowed_gpus:
                violations.append(f'Owned usage exceeded 64 MiB enumeration allowance on unallocated GPU {index}')
        self.last_sample = dict(updated_unix=time.time(), maximum_budget_gib=self.budget/GIB,
            robodojo_reserve_gib=self.reserve/GIB, safety_margin_gib=self.margin/GIB,
            cards=cards, violation=self.violation, scope='this campaign process groups only')
        if violations:
            raise RuntimeError('; '.join(violations))
        return self.last_sample

    def save(self, sample):
        path = self.output / 'memory_watch.json'
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(json.dumps(sample, indent=2) + '\n')
        temporary.replace(path)

    def start(self):
        def monitor():
            last_saved = 0
            while not self.finished.wait(.25):
                try:
                    sample = self.snapshot()
                    if time.monotonic() - last_saved >= 2:
                        self.save(sample)
                        last_saved = time.monotonic()
                except Exception as exc:
                    self.violation = repr(exc)
                    self.on_violation(self.violation)
                    with self.lock:
                        for pid, process in self.processes.items():
                            if process.poll() is None:
                                try:
                                    os.killpg(pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                    self.save(dict(self.last_sample or {}, violation=self.violation))
                    return
        self.thread = threading.Thread(target=monitor, name='robomme-memory-watch', daemon=True)
        self.thread.start()

    def close(self):
        self.finished.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
        if self.violation is None:
            self.save(self.snapshot())
        self.nvml.nvmlShutdown()
