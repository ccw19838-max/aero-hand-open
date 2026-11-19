"""Bounded process supervisor. Restart restores the original job deadline."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,json,os,signal,subprocess,sys,time


def supervise(root):
    root=Path(root).resolve();job=json.loads((root/'job.json').read_text())
    deadline=datetime.fromisoformat(job['deadline_at']).timestamp();stopping=False;child=None
    def shutdown(*_):
        nonlocal stopping
        stopping=True
        if child and child.poll() is None:child.terminate()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,shutdown)
    state=dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),deadline_at=job['deadline_at'],attempts=[])
    def persist():
        temp=root/'.supervisor.json.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');os.replace(temp,root/'supervisor.json')
        active_path=Path(__file__).resolve().parent.parent/'ACTIVE_EXPERIMENT.json'
        if active_path.exists():
            active=json.loads(active_path.read_text())
            if active.get('root')==str(root):
                active.update(controller_pid=os.getpid(),trainer_pid=state.get('trainer_pid'),
                    runtime_status=state.get('status'),runtime_last_verified_at=datetime.now(timezone.utc).isoformat())
                status_path=root/'status.json'
                if status_path.exists():
                    progress=json.loads(status_path.read_text())
                    if progress.get('pid')==state.get('trainer_pid'):
                        active.update(worker_pids=progress['worker_pids'],latest_checkpoint=progress.get('latest_checkpoint'),
                            training_steps=progress['training_steps'],current_curriculum={'x':progress['stage']},
                            live_parallel_environments=4 if progress['status'] in ('training','evaluating','trajectory_probe','initializing') and state['status']=='running' else 0)
                temp=active_path.with_name('.ACTIVE_EXPERIMENT.json.tmp');temp.write_text(json.dumps(active,indent=2)+'\n');os.replace(temp,active_path)
    for attempt in range(4):
        if stopping or time.time()>=deadline:break
        with (root/'training.log').open('a',buffering=1) as log:
            child=subprocess.Popen([sys.executable,'-u','-m','experiments.cube_two_driver_v3.train','--root',str(root)],stdout=log,stderr=subprocess.STDOUT)
            row=dict(attempt=attempt,pid=child.pid,started_at=datetime.now(timezone.utc).isoformat());state['attempts'].append(row)
            state.update(status='running',trainer_pid=child.pid);persist()
            sent_at=None
            while child.poll() is None:
                if (stopping or time.time()>=deadline) and sent_at is None:child.terminate();sent_at=time.time()
                if sent_at is not None and time.time()-sent_at>120:child.kill()
                if int(time.time())%10==0:persist()
                time.sleep(1.)
            row.update(exit_code=child.returncode,finished_at=datetime.now(timezone.utc).isoformat());persist()
        if child.returncode==0 or stopping:break
    state.update(status='stopped',finished_at=datetime.now(timezone.utc).isoformat());persist()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    with (Path(args.root)/'supervisor.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Supervisor already active')
        supervise(args.root)
