"""Bounded native-CPU fine tuning on the official continuous-rotation objective."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import signal
import time
import uuid

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv

from .env import OfficialRotateEnv,implementation_digest,source_manifest
from .policy import OfficialPolicy,initialize_source


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def new_model(vector,seed=0):
    return PPO(OfficialPolicy,vector,learning_rate=1e-4,n_steps=128,batch_size=128,n_epochs=4,
        gamma=.97,gae_lambda=.95,clip_range=.2,ent_coef=.001,max_grad_norm=.5,
        policy_kwargs={'log_std_init':float(np.log(.15))},seed=seed,device='cpu',verbose=0)


def evaluate(model,episodes=20,seed_start=10000,baseline=False,home=False,noise_level=1.,
             deadline=None,should_stop=lambda:False):
    env=OfficialRotateEnv(noise_level=noise_level);records=[];interrupted=False
    try:
        for episode in range(episodes):
            if should_stop() or (deadline is not None and time.time()>=deadline):interrupted=True;break
            obs,info=env.reset(seed=seed_start+episode,options={'home':home})
            for step in range(500):
                if should_stop() or (deadline is not None and time.time()>=deadline):interrupted=True;break
                action=np.zeros(7) if baseline else model.predict(obs,deterministic=True)[0]
                obs,reward,terminated,truncated,info=env.step(action)
                if terminated or truncated:break
            if interrupted:break
            records.append(dict(seed=seed_start+episode,**info))
    finally:env.close()
    def mean(key):return float(np.mean([r[key] for r in records])) if records else None
    return dict(task='official_z_velocity_cpu_v1',episodes_requested=episodes,episodes_completed=len(records),
        complete=len(records)==episodes,interrupted=interrupted,seed_start=seed_start,baseline=baseline,
        reset_mode='home' if home else 'official_distribution_numpy_rng',noise_level=noise_level,
        height_survival_rate=mean('horizon_reached'),mean_return=mean('episode_return'),
        mean_world_z_angular_velocity=mean('mean_world_z_angular_velocity'),
        mean_transverse_angular_speed=mean('mean_transverse_angular_speed'),
        mean_duration_seconds=mean('elapsed_seconds'),mean_max_palm_distance=mean('max_palm_distance'),
        mean_max_initial_tilt_deviation_degrees=mean('max_initial_tilt_deviation_degrees'),
        task_success_evaluated=False,
        note='Positive world-Z angular velocity and height survival do not certify a stable grasp, pure-axis rotation, or a 90-degree turn-and-stop.',records=records)


def score(report):
    if not report or not report['complete']:return (-1.,-float('inf'))
    return (report['height_survival_rate'],report['mean_return'])


def save(model,output,reason,provenance,best=False,report=None):
    token=uuid.uuid4().hex[:8];dest=output/'checkpoints'/f'step_{model.num_timesteps:09d}_{token}'
    temp=dest.with_name('.writing_'+token);temp.mkdir(parents=True)
    model.save(temp/'policy.zip')
    import hashlib
    metadata=dict(task='official_z_velocity_cpu_v1',format_version=1,training_steps=int(model.num_timesteps),
        created_at=datetime.now(timezone.utc).isoformat(),reason=reason,
        implementation_sha256=implementation_digest(),source_initialization=provenance,
        policy_sha256=hashlib.sha256((temp/'policy.zip').read_bytes()).hexdigest(),
        configuration=dict(sim_dt=.01,ctrl_dt=.05,cube_size=.05,cube_mass=.0692,
            initial_palm_tilt_degrees=20.,reward='dt*(omega_world_z-sum((a-last_a)^2)-100*height_failure)',
            action='home_ctrl + [0.02,0.02,0.02,0.02,0.7,0.003,0.012]*a',
            action_history_periods=2,actor_inputs=14,critic_inputs=81,hidden_layers=[512,256,128],
            engine='native_mujoco',algorithm='SB3 PPO',domain_randomization=False,
            source_observation_statistics='frozen',reward_normalization=False,
            stochastic_distribution='clipped Gaussian around tanh(source actor mean), global log_std',
            full_brax_continuation=False),evaluation=report)
    atomic_json(temp/'metadata.json',metadata);os.replace(temp,dest)
    for name in ('latest','best') if best else ('latest',):
        link=output/('.'+name+'_'+token);link.symlink_to(dest.relative_to(output),target_is_directory=True);os.replace(link,output/name)
    return dest


def load(checkpoint,vector=None):
    import hashlib
    checkpoint=Path(checkpoint).resolve();metadata=json.loads((checkpoint/'metadata.json').read_text())
    if metadata['task']!='official_z_velocity_cpu_v1' or metadata['implementation_sha256']!=implementation_digest():
        raise ValueError('Checkpoint semantics differ from this baseline')
    if hashlib.sha256((checkpoint/'policy.zip').read_bytes()).hexdigest()!=metadata['policy_sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    return PPO.load(checkpoint/'policy.zip',env=vector,device='cpu'),metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--deadline-at',required=True);parser.add_argument('--envs',type=int,default=4)
    parser.add_argument('--steps',type=int,default=100000000)
    parser.add_argument('--eval-every',type=int,default=131072)
    parser.add_argument('--checkpoint-every',type=int,default=65536)
    parser.add_argument('--resume');parser.add_argument('--seed',type=int,default=14000)
    parser.add_argument('--baseline-episodes',type=int,default=20)
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if not 1<=args.envs<=4 or min(args.steps,args.eval_every,args.checkpoint_every)<=0 or args.baseline_episodes<0:
        parser.error('Invalid CPU, step or evaluation budget')
    deadline=datetime.fromisoformat(args.deadline_at).timestamp()
    if not 0<deadline-time.time()<=24*3600:parser.error('Deadline must be within the next 24 hours')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'status.json').exists() and not args.resume:raise ValueError('Choose a fresh run or resume explicitly')
    existing=json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else None
    if existing and existing['deadline_at']!=args.deadline_at:raise ValueError('Resume must retain the original deadline')
    if existing:
        pid=existing.get('pid');cmd=Path('/proc')/str(pid)/'cmdline'
        if cmd.exists() and b'experiments.cube_official_baseline.train' in cmd.read_bytes():raise ValueError('Trainer is already alive')
    stop={'reason':None}
    def request_stop(signum,frame):stop['reason']=signal.Signals(signum).name
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    vector=DummyVecEnv([OfficialRotateEnv]) if args.envs==1 else SubprocVecEnv([OfficialRotateEnv]*args.envs,start_method='spawn')
    if args.resume:
        model,metadata=load(args.resume,vector);provenance=metadata['source_initialization']
    else:
        model=new_model(vector,args.seed);initialize_source(model,args.source)
        provenance=json.loads(Path(args.source).with_suffix('.json').read_text())
        provenance.update(mode='author actor and critic warm start; new PPO optimizer and exploration distribution',
                          stochastic_distribution_restored=False)
    model.set_random_seed(args.seed);vector.seed(args.seed)
    start=time.monotonic();initial_steps=int(model.num_timesteps)
    run=dict(status='initializing',pid=os.getpid(),task='official_z_velocity_cpu_v1',envs=args.envs,
        training_steps=initial_steps,started_at=datetime.now(timezone.utc).isoformat(),deadline_at=args.deadline_at,
        phase='official_continuous_rotation',target='positive world-Z velocity without height failure',
        source_initialization=provenance,final_90_degree_task_completed=False)
    atomic_json(output/'status.json',run)
    atomic_json(output/'source_manifest.json',source_manifest())
    best_key=score(None)
    if (output/'best/metadata.json').exists():best_key=score(json.loads((output/'best/metadata.json').read_text()).get('evaluation'))
    first=save(model,output,'initialized',provenance);run['latest_checkpoint']=str(first)
    if not args.resume and args.baseline_episodes:
        run['status']='validating_author_initialization';atomic_json(output/'status.json',run)
        report=evaluate(model,args.baseline_episodes,seed_start=10000,deadline=deadline,should_stop=lambda:bool(stop['reason']))
        atomic_json(output/'author_initialization_evaluation.json',report)
        if report['complete']:
            best_key=score(report);checkpoint=save(model,output,'author_initialization_evaluated',provenance,best=True,report=report)
            run.update(latest_checkpoint=str(checkpoint),latest_evaluation={k:v for k,v in report.items() if k!='records'})
    class Monitor(BaseCallback):
        def __init__(self):
            super().__init__();self.next_eval=initial_steps+args.eval_every;self.next_save=initial_steps+args.checkpoint_every
            self.next_status=0.;self.reason='step_budget';self.last_report=None
        def _on_step(self):
            nonlocal best_key
            if stop['reason']:self.reason=stop['reason'];return False
            if time.time()>=deadline:self.reason='wall_clock_budget';return False
            if self.model.num_timesteps-initial_steps>=args.steps:self.reason='step_budget';return False
            stamp=time.monotonic()
            if stamp>=self.next_status:
                run.update(status='training',training_steps=int(self.model.num_timesteps),updated_at=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=stamp-start,environment_steps_per_second=(self.model.num_timesteps-initial_steps)/max(1.,stamp-start))
                atomic_json(output/'status.json',run);self.next_status=stamp+10.
            if self.model.num_timesteps>=self.next_eval:
                run.update(status='validating',training_steps=int(self.model.num_timesteps));atomic_json(output/'status.json',run)
                result=evaluate(self.model,20,seed_start=10000,deadline=deadline,should_stop=lambda:bool(stop['reason']))
                self.last_report=result
                atomic_json(output/'validation'/f'step_{self.model.num_timesteps:09d}.json',result)
                improved=score(result)>best_key
                if improved:best_key=score(result)
                checkpoint=save(self.model,output,'validation',provenance,best=improved,report=result)
                summary={k:v for k,v in result.items() if k!='records'};summary['training_steps']=int(self.model.num_timesteps)
                with (output/'progress.jsonl').open('a') as stream:stream.write(json.dumps(summary)+'\n')
                run.update(latest_checkpoint=str(checkpoint),latest_evaluation=summary)
                self.next_eval+=args.eval_every;self.next_save=self.model.num_timesteps+args.checkpoint_every
                if result['interrupted']:self.reason=stop['reason'] or 'wall_clock_budget';return False
            elif self.model.num_timesteps>=self.next_save:
                run['latest_checkpoint']=str(save(self.model,output,'periodic',provenance))
                atomic_json(output/'status.json',run);self.next_save=self.model.num_timesteps+args.checkpoint_every
            return True
    monitor=Monitor()
    try:
        if stop['reason']:monitor.reason=stop['reason']
        elif time.time()>=deadline:monitor.reason='wall_clock_budget'
        else:model.learn(total_timesteps=args.steps,callback=monitor,reset_num_timesteps=False)
        run.update(status='stopped',stop_reason=monitor.reason)
    except BaseException as error:
        run.update(status='error',stop_reason=type(error).__name__,error=str(error));raise
    finally:
        checkpoint=save(model,output,run.get('stop_reason','interrupted'),provenance,report=monitor.last_report)
        run.update(training_steps=int(model.num_timesteps),latest_checkpoint=str(checkpoint),
                   finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(output/'status.json',run);vector.close()
        print(json.dumps({k:run.get(k) for k in ('status','training_steps','stop_reason','latest_checkpoint')}),flush=True)


if __name__=='__main__':main()
