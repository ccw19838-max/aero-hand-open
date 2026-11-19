"""New bounded 24-hour XY PPO experiment, two simulation workers per axis."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from datetime import datetime,timezone,timedelta
from pathlib import Path
import argparse
import json
import signal
import subprocess
import sys
import time
from .storage import atomic_json,implementation_digest


def supervisor(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    status_path=output/'status.json'
    if status_path.exists():
        if not args.resume:raise ValueError('Run exists; explicit stopped-run resume required')
        old=json.loads(status_path.read_text());pid=old.get('pid')
        process=Path('/proc')/str(pid)/'cmdline'
        if process.exists() and b'cube_contact_ppo.pipeline' in process.read_bytes():raise ValueError('Run is alive')
        deadline_at=old['deadline_at']
    else:
        deadline_at=(datetime.now(timezone.utc)+timedelta(hours=args.hours)).isoformat()
    deadline=datetime.fromisoformat(deadline_at).timestamp()
    if deadline<=time.time():raise ValueError('Preserved budget expired; a new authorized run is required')
    state=dict(task='xy_five_finger_closed_loop_ppo_v1',status='running',pid=os.getpid(),axes={},
        allocation={'x':2,'y':2,'z':0},parallel_environments=4,
        started_at=datetime.now(timezone.utc).isoformat(),deadline_at=deadline_at,budget_hours=args.hours,
        source_previous_run=str(Path(args.previous_run).resolve()),
        implementation_sha256=implementation_digest(),reference=str(Path(args.reference).resolve()),
        five_finger_hold_seconds=1.,final_goal_degrees=90.,temporary_regrasp_allowed=True,
        method='direct closed-loop PPO; 0/30/60/90 degree curriculum; mixed 1ms/2ms workers; small bounded DR after competence',
        xy_verified=False,full_cube_solver=False)
    stop={'value':False};children={};logs={}
    def handler(signum,frame):stop['value']=True
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    atomic_json(status_path,state)
    try:
        for index,axis in enumerate(('x','y')):
            logs[axis]=(output/f'{axis}.log').open('a')
            command=[sys.executable,'-m',__package__+'.train','--axis',axis,'--reference',args.reference,
                '--source-checkpoint',args.source_checkpoint,'--output',str(output/axis),'--deadline-at',deadline_at,
                '--seed',str(args.seed+1000*index)]
            if args.resume:
                latest=output/axis/'latest'
                if not latest.exists():raise ValueError('Missing resume checkpoint: '+axis)
                command+=['--resume',str(latest)]
            children[axis]=subprocess.Popen(command,stdout=logs[axis],stderr=subprocess.STDOUT)
        while any(c.poll() is None for c in children.values()):
            if stop['value'] or time.time()>=deadline:
                for c in children.values():
                    if c.poll() is None:c.terminate()
            for axis,child in children.items():
                path=output/axis/'status.json'
                record=json.loads(path.read_text()) if path.exists() else {'status':'initializing'}
                state['axes'][axis]={**record,'controller_pid':child.pid,'exit_code':child.poll()}
            state['updated_at']=datetime.now(timezone.utc).isoformat();atomic_json(status_path,state)
            time.sleep(3.)
        state['xy_verified']=all((output/a/'verified').exists() for a in ('x','y'))
        state['status']='completed' if state['xy_verified'] else 'stopped'
        state['stop_reason']='signal' if stop['value'] else 'wall_clock_budget' if time.time()>=deadline else 'axis_workers_finished'
    except BaseException as error:
        state.update(status='error',error=repr(error));raise
    finally:
        for c in children.values():
            if c.poll() is None:c.terminate()
        for axis,child in children.items():
            child.wait();path=output/axis/'status.json'
            state['axes'][axis]={**(json.loads(path.read_text()) if path.exists() else {}),'exit_code':child.returncode}
        for log in logs.values():log.close()
        state['finished_at']=datetime.now(timezone.utc).isoformat();atomic_json(status_path,state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True);p.add_argument('--source-checkpoint',required=True)
    p.add_argument('--previous-run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--hours',type=float,default=24.);p.add_argument('--seed',type=int,default=90000);p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if not 0<args.hours<=24.:p.error('Use a bounded budget up to 24h')
    supervisor(args)


if __name__=='__main__':main()
