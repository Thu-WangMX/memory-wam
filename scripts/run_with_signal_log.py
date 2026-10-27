"""Run a process and record external termination signal metadata before forwarding.
Usage: python run_with_signal_log.py SIGNAL_LOG COMMAND [ARGS...]
Linux only. Does not ignore termination or automatically restart stopped jobs.
"""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    log = Path(sys.argv[1])
    signals = {signal.SIGTERM, signal.SIGINT, signal.SIGHUP}
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
    child = subprocess.Popen(sys.argv[2:], start_new_session=True,
                             preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, old_mask))
    def record(event, **values):
        with log.open("a") as f:
            f.write(json.dumps(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                   event=event, wrapper_pid=os.getpid(), child_pid=child.pid,
                                   **values)) + "\n")
            f.flush()
            os.fsync(f.fileno())
    record("started")
    while child.poll() is None:
        info = signal.sigtimedwait(signals, 1)
        if info is not None:
            sender = Path("/proc") / str(info.si_pid) / "comm"
            try:
                sender_name = sender.read_text().strip()
            except OSError:
                sender_name = None
            record("signal", signal=info.si_signo, sender_pid=info.si_pid,
                   sender_uid=info.si_uid, sender_name=sender_name, signal_code=info.si_code)
            try:
                os.killpg(child.pid, info.si_signo)
            except ProcessLookupError:
                pass
    record("exit", returncode=child.returncode)
    return child.returncode if child.returncode >= 0 else 128 - child.returncode

if __name__ == "__main__":
    sys.exit(main())
