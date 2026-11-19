"""Sixteen-hour supervisor: Z first, then independently advancing X and Y.

This is a bounded background job. It never weakens the five-contact recapture
criterion. Each configuration gets its own immutable checkpoint directory.
"""
import argparse
from dataclasses import replace
from datetime import datetime,timezone,timedelta
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .config import CubeReorientConfig,AXES,ANGLES,tolerance_for
from .evaluate import atomic_json


def now():return datetime.now(timezone.utc)


def read_json(path):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def advance_targets(targets,axes):
    result=list(targets)
    for axis in axes:
        index=AXES.index(axis)
        result[index]=ANGLES[min(ANGLES.index(float(result[index]))+1,len(ANGLES)-1)]
    return tuple(result)


class Runner:
    def __init__(self,reference,initial_checkpoint,output,hours=16.,z_hours=2.,steps=5000000,envs=4,resume=False):
        if not math.isfinite(hours) or not 0<hours<=24 or not 0<z_hours<hours:
            raise ValueError('Require a finite <=24h total budget and smaller Z budget')
        if not 1<=envs<=4 or not 0<steps<=10000000:raise ValueError('Invalid CPU/step budget')
        self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=True)
        self.reference=Path(reference).resolve();self.child=None;self.stop=None
        existing=read_json(self.output/'status.json')
        if existing and not resume:raise ValueError('Use a new output directory or explicit --resume')
        if resume:
            if not existing:raise ValueError('No saved curriculum to resume')
            pid=existing.get('pid')
            try:
                command=(Path('/proc')/str(pid)/'cmdline').read_bytes()
                if b'experiments.cube_envelop_gradual.curriculum' in command:
                    raise ValueError('The curriculum is already running')
            except OSError:pass
            child_pid=existing.get('child_pid')
            try:
                command=(Path('/proc')/str(child_pid)/'cmdline').read_bytes()
                if b'experiments.cube_envelop_gradual.train' in command:
                    raise ValueError('The previous trainer is still running; do not duplicate it')
            except OSError:pass
            if existing['reference']!=str(self.reference):raise ValueError('Resume reference changed')
            self.state=existing
            # Resume only the exact last configuration/checkpoint when available.
            active=existing.get('active_stage')
            self.resume_stage=active if active and read_json(self.output/active/'status.json') else None
        else:
            started=now()
            self.resume_stage=None
            self.state=dict(task='cube_envelope_gradual_v1',status='initializing',pid=os.getpid(),
                started_at=started.isoformat(),deadline_at=(started+timedelta(hours=hours)).isoformat(),
                z_deadline_at=(started+timedelta(hours=z_hours)).isoformat(),budget_hours=hours,
                z_budget_hours=z_hours,xy_budget_hours=hours-z_hours,step_budget=steps,envs=envs,
                reference=str(self.reference),source_checkpoint=str(Path(initial_checkpoint).resolve()),
                phase='z',training_axes=['z'],targets=list(ANGLES[:1])*3,
                passed_angles={a:0. for a in AXES},completed_steps=0,training_steps=0,
                stage_index=0,active_stage=None,stages={},axis_checkpoints={},
                final_test_failed=False,checkpoint_every_steps=8192,validation_every_steps=32768,
                independent_seed_start=50000000,validation_episodes_per_axis=20)
        self.state.update(pid=os.getpid(),status='running')
        self.deadline=datetime.fromisoformat(self.state['deadline_at']).timestamp()
        self.z_deadline=datetime.fromisoformat(self.state['z_deadline_at']).timestamp()
        self.write()

    def write(self):
        self.state['remaining_seconds']=max(0.,self.deadline-time.time())
        self.state['updated_at']=now().isoformat()
        atomic_json(self.output/'status.json',self.state)

    def request_stop(self,signum=signal.SIGTERM):
        self.stop=signal.Signals(signum).name
        if self.child and self.child.poll() is None:self.child.send_signal(signum)

    def finish(self,reason,status='stopped'):
        self.state.update(status=status,stop_reason=reason,finished_at=now().isoformat(),active_stage=None,child_pid=None)
        self.write();return self.state

    def run_child(self,name,command,deadline,training=True):
        stage=self.state['stages'].setdefault(name,{})
        stage.update(command=command,started_at=now().isoformat(),status='running')
        out=self.output/name;out.mkdir(exist_ok=True)
        self.state.update(active_stage=name)
        with (self.output/(name+'.log')).open('a') as log:
            self.child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                cwd=Path(__file__).resolve().parents[2],start_new_session=True)
            self.state['child_pid']=self.child.pid;self.write()
            stopping=None;last=0.
            while self.child.poll() is None:
                stamp=time.time()
                if (self.stop or stamp>=deadline or stamp>=self.deadline) and stopping is None:
                    self.child.send_signal(signal.SIGTERM);stopping=time.monotonic()
                if stopping is not None and time.monotonic()-stopping>60.:
                    try:os.killpg(self.child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    self.child.wait();stage['forced_cleanup']=True;break
                if stamp-last>=5:
                    data=read_json(out/'status.json')
                    if data:
                        stage['live']={k:data.get(k) for k in ('status','training_steps','latest_checkpoint','latest_validation')}
                        if training:
                            self.state['training_steps']=self.state['completed_steps']+int(data.get('training_steps',0))
                            self.state['current_stage_training_steps']=data.get('training_steps',0)
                    self.write();last=stamp
                time.sleep(.5)
            code=self.child.wait();self.child=None
        stage.update(returncode=code,status='finished' if code==0 else 'error',finished_at=now().isoformat())
        self.state['child_pid']=None;self.write()
        return code

    def switch_to_xy(self,checkpoint):
        self.state.update(phase='xy',training_axes=['x','y'],source_checkpoint=str(checkpoint),
                          z_finished_at=now().isoformat(),active_stage=None)
        self.write()

    def verify(self,name,checkpoint,axis='all'):
        command=[sys.executable,'-m','experiments.cube_envelop_gradual.evaluate','--checkpoint',str(checkpoint),
                 '--output',str(self.output/name),'--episodes','100','--seed-start','50000000','--axis',axis]
        self.run_child(name,command,self.deadline,training=False)
        return read_json(self.output/name/'evaluation.json')

    def run(self):
        try:
            while not self.stop and time.time()<self.deadline:
                if self.state['completed_steps']>=self.state['step_budget']:return self.finish('step_budget')
                phase=self.state['phase']
                if phase=='z' and time.time()>=self.z_deadline:
                    self.switch_to_xy(self.state['source_checkpoint']);phase='xy'
                axes=tuple(self.state['training_axes']);targets=tuple(self.state['targets'])
                scalar=min(targets[AXES.index(a)] for a in axes)
                config=CubeReorientConfig(training_axes=axes,target_angles=targets,target_degrees=scalar,
                                          angle_tolerance_degrees=tolerance_for(scalar))
                stage_deadline=min(self.deadline,self.z_deadline) if phase=='z' else self.deadline
                if self.resume_stage:
                    name=self.resume_stage;self.resume_stage=None
                    old=read_json(self.output/name/'status.json')
                    checkpoint=old.get('latest_checkpoint')
                    config=CubeReorientConfig.from_dict(old['config'])
                    option='--resume' if checkpoint else '--init-checkpoint'
                    checkpoint=checkpoint or self.state['source_checkpoint']
                    already_sampled=int(old.get('training_steps',0)) if option=='--resume' else 0
                else:
                    tag='_'.join(f'{a}{targets[AXES.index(a)]:g}' for a in axes)
                    name=f'stage_{self.state["stage_index"]:03d}_{tag}'
                    checkpoint=self.state['source_checkpoint'];option='--init-checkpoint'
                    already_sampled=0
                (self.output/name).mkdir(exist_ok=True)
                config_path=self.output/name/'stage_config.json';atomic_json(config_path,config.to_dict())
                hours=max(.0001,(stage_deadline-time.time())/3600.)
                remaining_steps=self.state['step_budget']-self.state['completed_steps']-already_sampled
                if remaining_steps<=0:return self.finish('step_budget')
                command=[sys.executable,'-m','experiments.cube_envelop_gradual.train',
                    '--reference',str(self.reference),'--config',str(config_path),'--output',str(self.output/name),
                    '--hours',str(hours),'--steps',str(remaining_steps),'--envs',str(self.state['envs']),
                    '--eval-every','32768','--eval-episodes','20','--baseline-episodes','3',
                    option,str(checkpoint)]
                if self.state['final_test_failed']:command+=['--keep-training-after-validation']
                code=self.run_child(name,command,stage_deadline)
                data=read_json(self.output/name/'status.json')
                if not data:return self.finish('child_did_not_write_status','error')
                self.state['completed_steps']+=int(data.get('training_steps',0))
                self.state['training_steps']=self.state['completed_steps']
                latest=data.get('latest_checkpoint')
                if latest:self.state['source_checkpoint']=latest
                self.state['stage_index']+=1;self.state['active_stage']=None;self.write()
                if self.stop:return self.finish(self.stop)
                if time.time()>=self.deadline:return self.finish('wall_clock_budget')
                if code!=0:return self.finish('training_child_failed','error')
                if phase=='z' and time.time()>=self.z_deadline:
                    best=self.output/name/'best'
                    self.switch_to_xy(best.resolve() if best.exists() else latest);continue
                result=data.get('latest_validation') or {}
                promoted=data.get('promote_axes',[])
                if data.get('stop_reason')=='angle_stage_ready':
                    for a in promoted:
                        if a not in axes or (result.get('consecutive_axis_passes') or {}).get(a,0)<2:
                            raise ValueError('Invalid angle promotion event')
                        angle=targets[AXES.index(a)];self.state['passed_angles'][a]=angle
                        self.state['axis_checkpoints'][f'{a}_{angle:g}']=latest
                    self.state['targets']=list(advance_targets(targets,promoted));self.write();continue
                if data.get('stop_reason')=='final_targets_validated':
                    for a in axes:
                        self.state['passed_angles'][a]=90.;self.state['axis_checkpoints'][f'{a}_90']=latest
                    if phase=='z':self.switch_to_xy(latest);continue
                    if phase=='z_finish':
                        test=self.verify(f'verify_z_{self.state["stage_index"]:03d}',latest,'z')
                        self.state['independent_z_test']=test and {k:test.get(k) for k in ('complete','axis_success_rates','goal_rotation_degrees')}
                        if test and test.get('complete') and test.get('axis_success_rates',{}).get('z',0)>=.9:
                            return self.finish('all_three_axes_independently_verified','completed')
                        self.state['final_test_failed']=True;self.write();continue
                    verify_name=f'verify_xy_{self.state["stage_index"]:03d}'
                    test=self.verify(verify_name,latest)
                    self.state['independent_xy_test']=test and {k:test.get(k) for k in ('complete','axis_success_rates','goal_rotation_degrees')}
                    if test and test.get('complete') and all((test.get('axis_success_rates') or {}).get(a,0)>=.9 for a in ('x','y')):
                        self.state['xy_verified_checkpoint']=latest
                        z_checkpoint=self.state['axis_checkpoints'].get('z_90')
                        if z_checkpoint:
                            z_test=self.verify(f'verify_z_{self.state["stage_index"]:03d}',z_checkpoint,'z')
                            self.state['independent_z_test']=z_test and {k:z_test.get(k) for k in ('complete','axis_success_rates','goal_rotation_degrees')}
                            if z_test and z_test.get('complete') and z_test.get('axis_success_rates',{}).get('z',0)>=.9:
                                return self.finish('all_three_axes_independently_verified','completed')
                        # X/Y are already certified and kept immutable. Only
                        # now may unfinished Z use the leftover wall-clock time.
                        self.state.update(phase='z_finish',training_axes=['z'],final_test_failed=False,
                                          source_checkpoint=z_checkpoint or latest)
                        self.write();continue
                    self.state['final_test_failed']=True;self.write();continue
                # A time-limited Z run must never consume X/Y's fourteen hours.
                if phase=='z':self.switch_to_xy(latest);continue
                return self.finish(data.get('stop_reason','stage_stopped'))
            return self.finish(self.stop or 'wall_clock_budget')
        except BaseException as error:
            self.state['error']=f'{type(error).__name__}: {error}'
            self.finish('supervisor_error','error');raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True);p.add_argument('--initial-checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--hours',type=float,default=16.)
    p.add_argument('--z-hours',type=float,default=2.);p.add_argument('--steps',type=int,default=5000000)
    p.add_argument('--envs',type=int,default=4);p.add_argument('--resume',action='store_true')
    a=p.parse_args();runner=Runner(a.reference,a.initial_checkpoint,a.output,a.hours,a.z_hours,a.steps,a.envs,a.resume)
    signal.signal(signal.SIGTERM,lambda sig,frame:runner.request_stop(sig))
    signal.signal(signal.SIGINT,lambda sig,frame:runner.request_stop(sig))
    print(json.dumps(runner.run(),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
