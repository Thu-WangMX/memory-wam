"""User-authorized idle GPU memory reservations; release before evaluation."""
import argparse, os, signal, subprocess, time
from pathlib import Path

PYTHON='/mnt/vepfs01/output/spidy.wang/runtime/starwam-libero/venv/bin/python'


class AdoptedReservation:
    def __init__(self,pid): self.pid=pid
    def poll(self):
        try:
            state=Path(f'/proc/{self.pid}/stat').read_text().split(') ')[1][0]
            return 0 if state=='Z' else None
        except OSError:return 0
    def terminate(self):
        if self.poll() is None:os.kill(self.pid,signal.SIGTERM)
    def kill(self):
        if self.poll() is None:os.kill(self.pid,signal.SIGKILL)
    def wait(self,timeout):
        deadline=time.time()+timeout
        while self.poll() is None and time.time()<deadline:time.sleep(.2)
        if self.poll() is None:raise subprocess.TimeoutExpired(str(self.pid),timeout)
        return 0

class Reservations:
    def __init__(self, base, root):
        self.base,self.root,self.children=base,root,{}
        import json
        adoption=root/'adopt_reservations.json'
        self.predecessor=None
        if adoption.exists():
            info=json.loads(adoption.read_text())
            self.predecessor=info['predecessor']
            for gpu,pid in info['pids'].items():
                try:
                    cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
                    if b'robomme_gpu_reservation.py' in cmd and str(root).encode() in cmd:
                        self.children[int(gpu)]=AdoptedReservation(pid)
                except OSError:pass
            adoption.unlink()

    def refresh(self):
        for gpu,proc in list(self.children.items()):
            if proc.poll() is not None:self.children.pop(gpu)
        free,slots=self.base.free_slots()
        # Idle reservation needs 18 GiB plus driver margin, not the full
        # policy+simulator 20 GiB threshold used for actual evaluations.
        active_policy = False
        for proc in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                if b'experiments/robomme/serve.py' in proc.read_bytes():
                    active_policy = True
                    break
            except (OSError, PermissionError):
                pass
        if not active_policy:
            slots.update({g: 1 for g, mib in free.items() if mib >= 19*1024})
        for gpu in slots:
            if gpu in self.children:continue
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
            with (self.root/f'reservation_gpu{gpu}.log').open('a') as log:
                self.children[gpu]=subprocess.Popen([PYTHON,'-u',str(Path(__file__).resolve()),'--hold','--gpu',str(gpu),'--root',str(self.root)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        self.save('holding')
    def save(self,phase):
        self.base.write(self.root/'reservations.json',dict(phase=phase,buffer_gib=18,updated_at=self.base.now(),pids={str(g):p.pid for g,p in self.children.items() if p.poll() is None}))
    def release(self):
        for p in self.children.values():
            if p.poll() is None:p.terminate()
        for p in self.children.values():
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill();p.wait(timeout=10)
        self.children.clear();self.save('released_for_evaluation')
        if self.predecessor:
            try:os.kill(self.predecessor,signal.SIGKILL)
            except ProcessLookupError:pass
            self.predecessor=None

def hold(args):
    # If the supervisor dies unexpectedly, release only our own idle allocation.
    parent=os.getppid()
    import torch
    buffer=torch.empty(18*1024**3,dtype=torch.uint8,device='cuda')
    buffer.zero_();torch.cuda.synchronize()
    print(f'RESERVATION_READY gpu={args.gpu} pid={os.getpid()} buffer_gib=18',flush=True)
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while not stopped and os.getppid()==parent:time.sleep(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--hold',action='store_true');p.add_argument('--gpu',type=int);p.add_argument('--root');hold(p.parse_args())
