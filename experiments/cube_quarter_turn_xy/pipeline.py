"""XY only: two physical simulator workers per axis until original deadline."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from experiments.cube_quarter_turn_v2.storage import atomic_json,implementation_digest

AXES=('x','y')


def is_live(pid):
    path=Path('/proc')/str(pid)/'cmdline'
    return path.exists() and b'experiments.cube_quarter_turn_xy.pipeline' in path.read_bytes()


def worker(args):
    from .planner import search
    from experiments.cube_quarter_turn_v2.train import train_axis
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    deadline=datetime.fromisoformat(args.deadline_at).timestamp()
    source=Path(args.source_run)/args.axis
    prepared=json.loads((source/'prepare/report.json').read_text())
    references=[entry['reference'] for entry in prepared['selected']]
    stopped={'value':False}
    def stop_signal(signum,frame):stopped['value']=True
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    should_stop=lambda:stopped['value']
    state=dict(axis=args.axis,pid=os.getpid(),phase='trajectory_search',parallel_environments=2,
        deadline_at=args.deadline_at,started_at=datetime.now(timezone.utc).isoformat(),
        final_task_verified=False,validated_initial_grips=references,source_run=str(source.resolve()))
    atomic_json(output/'status.json',state)
    try:
        plan_report=output/'planner/report.json'
        if args.resume and plan_report.exists() and json.loads(plan_report.read_text())['demonstrations']:
            plan=json.loads(plan_report.read_text())
        else:
            plan=search(references,args.axis,source/'planner',output/'planner',deadline,
                population=args.population,seed=args.seed,should_stop=should_stop,
                resume=args.resume and (output/'planner/search_state.npz').exists())
        state.update(search_attempts=plan['attempts'],demonstrations=plan['demonstrations'],best=plan['best'])
        if not plan['demonstrations'] or should_stop() or time.time()>=deadline:
            state.update(phase='stopped',stop_reason=plan['stop_reason']);return
        state['phase']='teacher_training';atomic_json(output/'status.json',state)
        checkpoint=output/'teacher/latest'
        teacher=train_axis(plan['demonstrations'],args.axis,output/'teacher',args.source_checkpoint,deadline,
            envs=2,seed=args.seed,should_stop=should_stop,
            resume=checkpoint if args.resume and checkpoint.exists() else None)
        state.update(phase='stopped',teacher=teacher)
    except BaseException as error:
        state.update(phase='error',error=repr(error));raise
    finally:
        state['finished_at']=datetime.now(timezone.utc).isoformat();atomic_json(output/'status.json',state)


def supervisor(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    deadline=datetime.fromisoformat(args.deadline_at).timestamp()
    if not 0<deadline-time.time()<=24*3600:raise ValueError('Deadline must be the original remaining budget')
    source=json.loads((Path(args.source_run)/'status.json').read_text())
    if args.deadline_at!=source['deadline_at']:raise ValueError('Original deadline must not be extended')
    if source['status']=='running':raise ValueError('Stop source XYZ supervisor before allocating XY workers')
    for a in 'xyz':
        pid=source['axes'][a].get('pid')
        path=Path('/proc')/str(pid)/'cmdline'
        if path.exists() and b'experiments.cube_quarter_turn_v2.pipeline' in path.read_bytes():
            raise ValueError('Source axis still running: '+a)
    prior=output/'status.json'
    if prior.exists():
        old=json.loads(prior.read_text())
        if not args.resume or is_live(old.get('pid')):raise ValueError('Run exists/alive; explicit stopped-run resume required')
        if old['deadline_at']!=args.deadline_at:raise ValueError('Resume deadline changed')
    stop={'value':False}
    def stop_signal(signum,frame):stop['value']=True
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    children={};logs={}
    state=dict(task='xy_quarter_turn_two_workers',pid=os.getpid(),status='running',axes={},
        allocation=dict(x=2,y=2,z=0),active_axes=list(AXES),parallel_environments=4,
        phase='continued_search_then_cloning_and_ppo',deadline_at=args.deadline_at,
        started_at=datetime.now(timezone.utc).isoformat(),source_run=str(Path(args.source_run).resolve()),
        implementation_sha256=implementation_digest(),xy_verified=False,final_task_verified=False,
        goal_degrees=dict(x=90,y=90),success='full 3D and signed 90±5deg; thumb+two fingers; low speeds; hold 1s',
        geometry_unchanged=True,source_manipulation_prior=args.source_checkpoint)
    atomic_json(prior,state)
    try:
        for index,axis in enumerate(AXES):
            logs[axis]=(output/f'{axis}.log').open('a')
            command=[sys.executable,'-m',__package__+'.pipeline','--axis',axis,'--source-run',args.source_run,
                '--source-checkpoint',args.source_checkpoint,'--output',str(output/axis),'--deadline-at',args.deadline_at,
                '--population',str(args.population),'--seed',str(args.seed+index*1000)]
            if args.resume:command+=['--resume']
            children[axis]=subprocess.Popen(command,stdout=logs[axis],stderr=subprocess.STDOUT)
        while any(child.poll() is None for child in children.values()):
            if stop['value'] or time.time()>=deadline:
                for child in children.values():
                    if child.poll() is None:child.terminate()
            for axis,child in children.items():
                path=output/axis/'status.json'
                record=json.loads(path.read_text()) if path.exists() else {'phase':'initializing'}
                sub=output/axis/('teacher/status.json' if record['phase']=='teacher_training' else 'planner/status.json')
                if sub.exists():record['detail']=json.loads(sub.read_text())
                state['axes'][axis]={**record,'worker_pid':child.pid,'exit_code':child.poll()}
            state['updated_at']=datetime.now(timezone.utc).isoformat();atomic_json(prior,state)
            time.sleep(3.)
        state['xy_verified']=all((output/a/'teacher/verified').exists() for a in AXES)
        # A claim about the full XYZ task would be false after deliberately
        # suspending Z. XY verification is reported as its own result.
        state['status']='completed' if state['xy_verified'] else 'stopped'
        state['stop_reason']='signal' if stop['value'] else 'deadline' if time.time()>=deadline else 'axis_workers_finished'
    except BaseException as error:
        state.update(status='error',error=repr(error));raise
    finally:
        for child in children.values():
            if child.poll() is None:child.terminate()
        for axis,child in children.items():
            child.wait();path=output/axis/'status.json'
            state['axes'][axis]={**(json.loads(path.read_text()) if path.exists() else {}),'exit_code':child.returncode}
        for log in logs.values():log.close()
        state['finished_at']=datetime.now(timezone.utc).isoformat();atomic_json(prior,state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--axis',choices=AXES);parser.add_argument('--source-run',required=True)
    parser.add_argument('--source-checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--deadline-at',required=True);parser.add_argument('--population',type=int,default=48)
    parser.add_argument('--seed',type=int,default=61000);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.population<8:parser.error('Population >=8 required')
    if args.axis:worker(args)
    else:supervisor(args)


if __name__=='__main__':main()
