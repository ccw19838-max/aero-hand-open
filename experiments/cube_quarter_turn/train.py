"""Train all three 90-degree endpoint goals from the first rollout; bounded CPU."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv,VecNormalize

from .env import QuarterTurnEnv
from .policy import QuarterPolicy,initialize_prior
from .task import AXES
from .evaluation import evaluate,evaluate_chain
from .checkpoints import atomic_json,save,load


def new_model(vector,seed=0):
    return PPO(QuarterPolicy,vector,learning_rate=2e-4,n_steps=128,batch_size=128,n_epochs=4,
        gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=.005,max_grad_norm=.5,
        policy_kwargs={'log_std_init':float(np.log(.3))},seed=seed,device='cpu',verbose=0)


def selection_key(report):
    if not report or not report['complete']:return (-1.,-1.,-1.,-1.)
    groups=list(report['groups'].values());rates=[r['success_rate'] for r in groups]
    progress=[90.-r['mean_best_angle_error_degrees'] for r in groups]
    return (min(rates),float(np.mean(rates)),min(progress),float(np.mean(progress)))


def compact(report):
    return {**{k:v for k,v in report.items() if k!='groups'},
            'groups':{a:{k:v for k,v in r.items() if k!='records'} for a,r in report['groups'].items()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True);parser.add_argument('--source-checkpoint',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--deadline-at',required=True)
    parser.add_argument('--envs',type=int,default=4);parser.add_argument('--steps',type=int,default=100000000)
    parser.add_argument('--eval-every',type=int,default=32768);parser.add_argument('--checkpoint-every',type=int,default=16384)
    parser.add_argument('--initial-eval-episodes',type=int,default=3);parser.add_argument('--seed',type=int,default=16000)
    parser.add_argument('--resume')
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if not 1<=args.envs<=4 or min(args.steps,args.eval_every,args.checkpoint_every)<=0 or args.initial_eval_episodes<0:
        parser.error('Invalid CPU or evaluation budget')
    deadline=datetime.fromisoformat(args.deadline_at).timestamp()
    if not 0<deadline-time.time()<=24*3600:parser.error('Use a deadline within the next 24 hours')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True);reference=Path(args.reference).resolve()
    old=json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else None
    if old and not args.resume:raise ValueError('Output exists; use a new run or explicit resume')
    if old and old['deadline_at']!=args.deadline_at:raise ValueError('Resume must preserve its deadline')
    if old:
        cmd=Path('/proc')/str(old.get('pid'))/'cmdline'
        if cmd.exists() and b'experiments.cube_quarter_turn.train' in cmd.read_bytes():raise ValueError('Trainer is already alive')
    probe=QuarterTurnEnv(reference);reference_obs,_=probe.reset(seed=0,options={'reference':True,'axis':'z'})
    probe.close()
    def factory():return QuarterTurnEnv(reference)
    vector=DummyVecEnv([factory]) if args.envs==1 else SubprocVecEnv([factory]*args.envs,start_method='spawn')
    if args.resume:
        checkpoint=Path(args.resume).resolve()
        raw,metadata=load(checkpoint)
        normalizer=VecNormalize.load(checkpoint/'reward_normalizer.pkl',vector)
        normalizer.training=True;normalizer.norm_obs=False;normalizer.norm_reward=True
        model,_=load(checkpoint,normalizer);provenance=metadata['source_initialization']
        if reference.read_bytes()!=(checkpoint/'reference.npz').read_bytes():raise ValueError('Resume reference changed')
    else:
        from experiments.cube_official_baseline.train import load as load_prior
        source,metadata=load_prior(args.source_checkpoint)
        normalizer=VecNormalize(vector,norm_obs=False,norm_reward=True,gamma=.99,clip_reward=10000.)
        model=new_model(normalizer,args.seed)
        provenance=initialize_prior(model,source,reference_obs)
        provenance.update(source_checkpoint=str(Path(args.source_checkpoint).resolve()),
                          source_training_steps=metadata['training_steps'],new_goal_task_training_steps=0)
    stop={'reason':None}
    def request_stop(signum,frame):stop['reason']=signal.Signals(signum).name
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    initial_steps=int(model.num_timesteps);started=time.monotonic();normalizer.seed(args.seed)
    run=dict(task='xyz_quarter_turn_v1',status='initializing',pid=os.getpid(),envs=args.envs,
        phase='xyz_90_degrees_with_stop',goal_degrees={'x':90,'y':90,'z':90},
        axis_frame='fixed_cube_face_axes_at_initial_grasp',cube_size_mm=57,hand_tilt_degrees=20,
        training_steps=initial_steps,started_at=datetime.now(timezone.utc).isoformat(),deadline_at=args.deadline_at,
        source_initialization=provenance,reference=str(reference),final_task_verified=False,
        latest_evaluation=None,independent_attempts=old.get('independent_attempts',0) if old else 0)
    atomic_json(output/'status.json',run)
    checkpoint=save(model,normalizer,reference,output,'initialized',provenance);run['latest_checkpoint']=str(checkpoint)
    if not args.resume and args.initial_eval_episodes:
        run['status']='validating_initialization';atomic_json(output/'status.json',run)
        report=evaluate(model,reference,args.initial_eval_episodes,deadline=deadline,should_stop=lambda:bool(stop['reason']))
        atomic_json(output/'initialization_evaluation.json',report)
    best_key=selection_key(None);axis_best={a:(-1.,-1.) for a in AXES}
    if (output/'best/metadata.json').exists():best_key=selection_key(json.loads((output/'best/metadata.json').read_text()).get('evaluation'))
    class Monitor(BaseCallback):
        def __init__(self):
            super().__init__();self.next_eval=initial_steps+args.eval_every;self.next_save=initial_steps+args.checkpoint_every
            self.next_status=0.;self.reason='step_budget';self.last_report=None;self.streaks={a:0 for a in AXES}
        def _on_step(self):
            nonlocal best_key
            if stop['reason']:self.reason=stop['reason'];return False
            if time.time()>=deadline:self.reason='wall_clock_budget';return False
            if self.model.num_timesteps-initial_steps>=args.steps:self.reason='step_budget';return False
            stamp=time.monotonic()
            if stamp>=self.next_status:
                run.update(status='training',training_steps=int(self.model.num_timesteps),
                    elapsed_seconds=stamp-started,updated_at=datetime.now(timezone.utc).isoformat(),
                    environment_steps_per_second=(self.model.num_timesteps-initial_steps)/max(1.,stamp-started))
                atomic_json(output/'status.json',run);self.next_status=stamp+10.
            if self.model.num_timesteps>=self.next_eval:
                run.update(status='validating',training_steps=int(self.model.num_timesteps));atomic_json(output/'status.json',run)
                report=evaluate(self.model,reference,20,deadline=deadline,should_stop=lambda:bool(stop['reason']))
                self.last_report=report;aliases=[]
                if selection_key(report)>best_key:best_key=selection_key(report);aliases.append('best')
                if report['complete']:
                    for axis in AXES:
                        g=report['groups'][axis];key=(g['success_rate'],90.-g['mean_best_angle_error_degrees'])
                        if key>axis_best[axis]:axis_best[axis]=key;aliases.append('best_'+axis)
                        self.streaks[axis]=self.streaks[axis]+1 if g['success_rate']>=.9 else 0
                checkpoint=save(self.model,normalizer,reference,output,'validation',provenance,report,aliases)
                atomic_json(output/'validation'/f'step_{self.model.num_timesteps:09d}.json',report)
                summary=compact(report);summary.update(training_steps=int(self.model.num_timesteps),consecutive_passes=self.streaks.copy())
                with (output/'progress.jsonl').open('a') as stream:stream.write(json.dumps(summary)+'\n')
                run.update(latest_checkpoint=str(checkpoint),latest_evaluation=summary)
                atomic_json(output/'status.json',run)
                if all(n>=2 for n in self.streaks.values()):
                    attempt=run['independent_attempts'];run['independent_attempts']+=1
                    independent=evaluate(self.model,reference,100,50000000+attempt*1000,deadline,lambda:bool(stop['reason']))
                    atomic_json(output/'independent'/f'attempt_{attempt:03d}.json',independent)
                    chain=evaluate_chain(self.model,reference,20,80000000+attempt*1000,deadline,lambda:bool(stop['reason']))
                    atomic_json(output/'independent'/f'chain_{attempt:03d}.json',chain)
                    run.update(independent_evaluation=compact(independent),chain_evaluation={k:v for k,v in chain.items() if k!='records'})
                    if independent['complete'] and independent['min_axis_success_rate']>=.9 and chain['complete'] and chain['success_rate']>=.9:
                        self.reason='xyz90_and_chain_verified';run['final_task_verified']=True;return False
                    self.streaks={a:0 for a in AXES}
                self.next_eval+=args.eval_every;self.next_save=self.model.num_timesteps+args.checkpoint_every
                if not report['complete']:self.reason=stop['reason'] or 'wall_clock_budget';return False
            elif self.model.num_timesteps>=self.next_save:
                run['latest_checkpoint']=str(save(self.model,normalizer,reference,output,'periodic',provenance))
                atomic_json(output/'status.json',run);self.next_save=self.model.num_timesteps+args.checkpoint_every
            return True
    monitor=Monitor()
    try:
        if stop['reason']:monitor.reason=stop['reason']
        elif time.time()>=deadline:monitor.reason='wall_clock_budget'
        else:model.learn(total_timesteps=args.steps,callback=monitor,reset_num_timesteps=False)
        run.update(status='completed' if run['final_task_verified'] else 'stopped',stop_reason=monitor.reason)
    except BaseException as error:
        run.update(status='error',stop_reason=type(error).__name__,error=str(error));raise
    finally:
        checkpoint=save(model,normalizer,reference,output,run.get('stop_reason','interrupted'),provenance,monitor.last_report)
        run.update(training_steps=int(model.num_timesteps),latest_checkpoint=str(checkpoint),finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(output/'status.json',run);normalizer.close()
        print(json.dumps({k:run.get(k) for k in ('status','training_steps','stop_reason','latest_checkpoint')}),flush=True)


if __name__=='__main__':main()
