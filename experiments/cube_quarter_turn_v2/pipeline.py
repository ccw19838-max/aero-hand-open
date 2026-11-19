"""CPU bounded per-axis prepare -> legal plan -> BC -> PPO, fixed global deadline."""
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
from .storage import atomic_json,digest


def worker(args):
    from .prepare import prepare
    from .planner import search
    from .train import train_axis
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    deadline=datetime.fromisoformat(args.deadline_at).timestamp();stop={'value':False}
    def stop_signal(signum,frame):stop['value']=True
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    should_stop=lambda:stop['value']
    state=dict(axis=args.axis,phase='grip_screening',pid=os.getpid(),deadline_at=args.deadline_at,
        final_task_verified=False,started_at=datetime.now(timezone.utc).isoformat())
    atomic_json(output/'status.json',state)
    try:
        prep_deadline=min(deadline,time.time()+args.prepare_minutes*60.)
        preparation=prepare(args.reference,args.axis,output/'prepare',prep_deadline,args.candidates,seed=args.seed,should_stop=should_stop)
        references=[item['reference'] for item in preparation['selected']]
        if not references:
            state.update(phase='blocked',reason='no_validated_initial_grip');return
        if should_stop() or time.time()>=deadline:
            state.update(phase='stopped',reason='signal_or_deadline');return
        state.update(phase='trajectory_search',validated_initial_grips=references);atomic_json(output/'status.json',state)
        plan=search(references,args.axis,output/'planner',min(deadline,time.time()+args.plan_minutes*60.),
            population=args.population,seed=args.seed+100,should_stop=should_stop)
        state.update(search_attempts=plan['attempts'],best=plan['best'],demonstrations=plan['demonstrations'])
        if not plan['demonstrations']:
            state.update(phase='blocked',reason='no_legal_full_90_degree_demonstration');return
        if should_stop() or time.time()>=deadline:
            state.update(phase='stopped',reason='signal_or_deadline');return
        state['phase']='teacher_training';atomic_json(output/'status.json',state)
        result=train_axis(plan['demonstrations'],args.axis,output/'teacher',args.source_checkpoint,deadline,
            envs=2 if args.axis=='y' else 1,seed=args.seed,should_stop=should_stop)
        state.update(phase='stopped',teacher=result)
    except BaseException as error:
        state.update(phase='error',error=repr(error));raise
    finally:
        state['finished_at']=datetime.now(timezone.utc).isoformat();atomic_json(output/'status.json',state)


def supervisor(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    deadline=datetime.fromisoformat(args.deadline_at).timestamp()
    if not 0<deadline-time.time()<=24*3600:raise ValueError('Use original deadline in next 24h')
    if (output/'status.json').exists():raise ValueError('Run exists; use a new output directory')
    children={};logs={};stop={'value':False}
    def stop_signal(signum,frame):stop['value']=True
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    state=dict(task='xyz_quarter_turn_v2',pid=os.getpid(),status='running',phase='gated_axis_teachers',
        deadline_at=args.deadline_at,started_at=datetime.now(timezone.utc).isoformat(),
        max_cpu_physics_workers=4,allocation={'x':1,'y':2,'z':1},
        physics=dict(cube_mm=57,mass_kg=.077966253,friction=.6,palm_tilt_degrees=20,sim_dt=.002),
        goal_degrees=dict(x=90,y=90,z=90),final_task_verified=False,axes={})
    atomic_json(output/'status.json',state)
    try:
        for index,axis in enumerate('xyz'):
            log=(output/f'{axis}.log').open('a');logs[axis]=log
            command=[sys.executable,'-m',__package__+'.pipeline','--axis',axis,'--reference',args.reference,
                '--source-checkpoint',args.source_checkpoint,'--output',str(output/axis),'--deadline-at',args.deadline_at,
                '--prepare-minutes',str(args.prepare_minutes),'--plan-minutes',str(args.plan_minutes),
                '--candidates',str(args.candidates),'--population',str(args.population),'--seed',str(args.seed+index*1000)]
            children[axis]=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        while any(child.poll() is None for child in children.values()):
            if time.time()>=deadline or stop['value']:
                for child in children.values():
                    if child.poll() is None:child.terminate()
            for axis,child in children.items():
                path=output/axis/'status.json'
                record=json.loads(path.read_text()) if path.exists() else {'phase':'initializing'}
                # Present current subphase while retaining the explicit PPO gate.
                sub=output/axis/({'trajectory_search':'planner/status.json','grip_screening':'prepare/search_status.json'}
                                 .get(record['phase'],'teacher/status.json'))
                if sub.exists():record['detail']=json.loads(sub.read_text())
                state['axes'][axis]={**record,'worker_pid':child.pid,'exit_code':child.poll()}
            state['updated_at']=datetime.now(timezone.utc).isoformat();atomic_json(output/'status.json',state)
            time.sleep(5.)
        # Independent teachers do not imply composability. Verify the no-reset
        # chain only if all three independent 100-start gates actually pass.
        verified={a:output/a/'teacher/verified' for a in 'xyz'}
        if all(p.exists() for p in verified.values()) and time.time()<deadline and not stop['value']:
            from .evaluate import chain
            report=chain(verified,100,deadline=deadline,should_stop=lambda:stop['value'])
            atomic_json(output/'chain_evaluation.json',report)
            state['final_task_verified']=bool(report['complete'] and report['success_rate']>=.9)
        state['status']='completed' if state['final_task_verified'] else 'stopped'
        state['stop_reason']='signal' if stop['value'] else 'deadline' if time.time()>=deadline else 'axis_workers_finished'
    finally:
        for child in children.values():
            if child.poll() is None:child.terminate()
        for axis,child in children.items():
            child.wait();path=output/axis/'status.json'
            state['axes'][axis]={**(json.loads(path.read_text()) if path.exists() else {}),'exit_code':child.returncode}
        for log in logs.values():log.close()
        state['finished_at']=datetime.now(timezone.utc).isoformat();atomic_json(output/'status.json',state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--axis',choices=list('xyz'));parser.add_argument('--reference',required=True)
    parser.add_argument('--source-checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--deadline-at',required=True);parser.add_argument('--prepare-minutes',type=float,default=12.)
    parser.add_argument('--plan-minutes',type=float,default=90.);parser.add_argument('--candidates',type=int,default=192)
    parser.add_argument('--population',type=int,default=48);parser.add_argument('--seed',type=int,default=25000)
    args=parser.parse_args()
    if min(args.prepare_minutes,args.plan_minutes,args.candidates)<=0 or args.population<8:parser.error('Invalid search budget')
    if args.axis:worker(args)
    else:supervisor(args)


if __name__=='__main__':main()
