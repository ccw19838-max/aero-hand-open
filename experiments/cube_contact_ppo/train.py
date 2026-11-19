"""Actual PPO begins immediately; five-finger curriculum is not a CEM demo gate."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import signal
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv,VecNormalize
from .env import ContactEnv
from .policy import Policy,initialize
from .evaluate import evaluate,compact
from .storage import atomic_json,save,load

TARGETS=(0.,30.,60.,90.)


def new_model(vector,seed=0):
    return PPO(Policy,vector,learning_rate=3e-4,n_steps=128,batch_size=128,n_epochs=4,
        gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=.005,max_grad_norm=.5,
        policy_kwargs={'log_std_init':float(np.log(.25))},seed=seed,device='cpu',verbose=0)


def advance(curriculum,report):
    """Curriculum success never substitutes for the final 90-degree task."""
    if report['target_degrees']!=curriculum['target_degrees']:raise ValueError('Evaluation is for a different curriculum goal')
    passed=bool(report['complete'] and report['min_integrator_success_rate']>=.9)
    curriculum['consecutive_passes']=curriculum.get('consecutive_passes',0)+1 if passed else 0
    if curriculum['consecutive_passes']<2:return False
    target=curriculum['target_degrees']
    if target<90.:
        curriculum.update(target_degrees=TARGETS[TARGETS.index(target)+1],consecutive_passes=0)
        return True
    return False


def train(args):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    deadline=datetime.fromisoformat(args.deadline_at).timestamp()
    if deadline<=time.time():raise ValueError('New training budget has expired')
    reference=Path(args.reference).resolve();stop={'value':False}
    def request_stop(signum,frame):stop['value']=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    should_stop=lambda:stop['value']
    curriculum=dict(target_degrees=0.,randomization=0.,consecutive_passes=0,independent_attempts=0)
    checkpoint=Path(args.resume).resolve() if args.resume else None
    if checkpoint:
        _,metadata=load(checkpoint);curriculum=metadata['curriculum'].copy()
        if metadata['axis']!=args.axis:raise ValueError('Resume axis changed')
    def factory(dt):
        return lambda:ContactEnv(reference,args.axis,physical_dt=dt,
            target_degrees=curriculum['target_degrees'],randomization=curriculum['randomization'])
    vector=SubprocVecEnv([factory(.001),factory(.002)],start_method='spawn')
    if checkpoint:
        normalizer=VecNormalize.load(checkpoint/'reward_normalizer.pkl',vector)
        normalizer.training=True;normalizer.norm_obs=False;normalizer.norm_reward=True
        model,metadata=load(checkpoint,normalizer);provenance=metadata['source_initialization']
        if provenance['deadline_at']!=args.deadline_at:raise ValueError('Resume deadline changed')
        if reference.read_bytes()!=(checkpoint/'reference.npz').read_bytes():raise ValueError('Resume reference changed')
    else:
        normalizer=VecNormalize(vector,norm_obs=False,norm_reward=True,gamma=.99,clip_reward=10000.)
        model=new_model(normalizer,args.seed)
        from experiments.cube_official_baseline.train import load as load_source
        source,metadata=load_source(args.source_checkpoint)
        probe=factory(.001)();observation,_=probe.reset(seed=0,options={'reference':True});probe.close()
        provenance=initialize(model,source,observation)
        provenance.update(source_checkpoint=str(Path(args.source_checkpoint).resolve()),
            source_training_steps=metadata['training_steps'],deadline_at=args.deadline_at,
            research='OpenAI 2019 relative actions, dense goal-distance differences, PPO; bounded randomization curriculum')
    initial_steps=int(model.num_timesteps);normalizer.seed(args.seed)
    state=dict(task='xy_five_finger_closed_loop_ppo_v1',axis=args.axis,status='initializing',phase='ppo_initialization',
        pid=os.getpid(),envs=2,worker_pids=[p.pid for p in vector.processes],
        physics_dt_per_worker=[.001,.002],training_steps=initial_steps,policy_updates=int(model._n_updates),
        curriculum=curriculum.copy(),started_at=datetime.now(timezone.utc).isoformat(),deadline_at=args.deadline_at,
        five_finger_static_requirement=True,temporary_regrasp_allowed=True,final_goal_degrees=90.,
        final_task_verified=False,demonstration_gate=False)
    atomic_json(output/'status.json',state)
    latest_report=None;best_keys={};stop_reason='step_budget';last_save=time.time()
    def record(report,reason):
        nonlocal latest_report,last_save
        latest_report=report;target=report['target_degrees'];key=(report['min_integrator_success_rate'],-report['mean_best_angle_error_degrees'],report['mean_five_finger_duty'])
        alias=[]
        if report['complete'] and key>best_keys.get(target,(-1.,-999.,-1.)):
            best_keys[target]=key;alias=['best_'+str(int(target)),'best']
        dest=save(model,normalizer,reference,output,args.axis,reason,provenance,curriculum.copy(),report,alias)
        atomic_json(output/'validation'/f'{reason}_{model.num_timesteps:09d}_{int(target):03d}.json',report)
        state.update(latest_checkpoint=str(dest),latest_evaluation=compact(report),curriculum=curriculum.copy())
        with (output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(training_steps=int(model.num_timesteps),**compact(report)))+'\n')
        atomic_json(output/'status.json',state);last_save=time.time()
    try:
        state['latest_checkpoint']=str(save(model,normalizer,reference,output,args.axis,'initialized',provenance,curriculum.copy()))
        if not checkpoint:
            # Two disjoint deterministic validation sets establish whether the
            # existing neutral five-finger grip already satisfies stage zero.
            # No synthetic demonstration or CEM success is needed for PPO.
            for attempt in range(2):
                if time.time()>=deadline or should_stop():break
                report=evaluate(model,reference,args.axis,0.,args.initial_eval_episodes,
                    seed=1000000+attempt*1000,deadline=deadline,should_stop=should_stop)
                record(report,'initial_grip_'+str(attempt))
                if advance(curriculum,report):
                    normalizer.env_method('set_task',curriculum['target_degrees'],curriculum['randomization'])
                    stage_checkpoint=save(model,normalizer,reference,output,args.axis,'initial_grip_verified_curriculum_advanced',
                        provenance,curriculum.copy(),report)
                    state.update(curriculum=curriculum.copy(),latest_checkpoint=str(stage_checkpoint));atomic_json(output/'status.json',state)
        class Monitor(BaseCallback):
            def __init__(self):
                super().__init__();self.next_eval=model.num_timesteps+args.eval_every
                self.next_status=0.;self.last_eval=time.time();self.last_save=time.time()
                self.start=time.monotonic();self.count=0
            def _on_step(self):
                nonlocal stop_reason,last_save
                if should_stop():stop_reason='signal';return False
                if time.time()>=deadline:stop_reason='wall_clock_budget';return False
                if time.time()>=self.next_status:
                    state.update(status='training',phase='ppo',training_steps=int(self.model.num_timesteps),
                        policy_updates=int(self.model._n_updates),curriculum=curriculum.copy(),
                        updated_at=datetime.now(timezone.utc).isoformat(),
                        environment_steps_per_second=(self.model.num_timesteps-initial_steps)/max(1.,time.monotonic()-self.start))
                    atomic_json(output/'status.json',state);self.next_status=time.time()+10.
                if self.model.num_timesteps>=self.next_eval or time.time()-self.last_eval>=args.eval_seconds:
                    state.update(status='validating',phase='validation');atomic_json(output/'status.json',state)
                    report=evaluate(self.model,reference,args.axis,curriculum['target_degrees'],20,
                        deadline=deadline,should_stop=should_stop,randomization=curriculum['randomization'])
                    record(report,'validation');self.count+=1
                    if advance(curriculum,report):
                        normalizer.env_method('set_task',curriculum['target_degrees'],curriculum['randomization'])
                        stage_checkpoint=save(self.model,normalizer,reference,output,args.axis,'curriculum_advanced',provenance,curriculum.copy(),report)
                        state.update(curriculum=curriculum.copy(),latest_checkpoint=str(stage_checkpoint));atomic_json(output/'status.json',state)
                    elif curriculum['target_degrees']==90. and curriculum['consecutive_passes']>=2:
                        attempt=curriculum['independent_attempts'];curriculum['independent_attempts']+=1
                        independent=evaluate(self.model,reference,args.axis,90.,100,seed=50000000+attempt*1000,
                            deadline=deadline,should_stop=should_stop,randomization=curriculum['randomization'])
                        atomic_json(output/'independent'/f'attempt_{attempt:03d}.json',independent)
                        if independent['complete'] and independent['min_integrator_success_rate']>=.9:
                            if curriculum['randomization']<.1:
                                save(self.model,normalizer,reference,output,args.axis,'nominal_or_partial_DR_verified',provenance,curriculum.copy(),independent,
                                    ('verified_dr_'+str(curriculum['randomization']),))
                                curriculum.update(randomization=round(curriculum['randomization']+.05,2),consecutive_passes=0)
                                normalizer.env_method('set_task',90.,curriculum['randomization'])
                            else:
                                nominal=evaluate(self.model,reference,args.axis,90.,100,seed=80000000,
                                    deadline=deadline,should_stop=should_stop)
                                atomic_json(output/'independent/final_nominal.json',nominal)
                                if nominal['complete'] and nominal['min_integrator_success_rate']>=.9:
                                    save(self.model,normalizer,reference,output,args.axis,'axis90_five_finger_verified',provenance,curriculum.copy(),independent,('verified',))
                                    state['final_task_verified']=True;stop_reason='axis90_five_finger_verified';return False
                        curriculum['consecutive_passes']=0
                    # Report final-goal performance periodically even while a
                    # smaller-angle curriculum is being learned.
                    if self.count%3==0 and curriculum['target_degrees']<90. and time.time()<deadline:
                        final=evaluate(self.model,reference,args.axis,90.,5,seed=30000000,
                            deadline=deadline,should_stop=should_stop)
                        atomic_json(output/'final_goal_probes'/f'step_{self.model.num_timesteps:09d}.json',final)
                        state['latest_90_degree_probe']=compact(final)
                    self.next_eval=self.model.num_timesteps+args.eval_every;self.last_eval=time.time()
                    if not report['complete']:stop_reason='signal' if should_stop() else 'wall_clock_budget';return False
                if time.time()-last_save>=args.checkpoint_seconds:
                    state['latest_checkpoint']=str(save(self.model,normalizer,reference,output,args.axis,'periodic',provenance,curriculum.copy()))
                    atomic_json(output/'status.json',state);last_save=time.time()
                return True
        monitor=Monitor()
        if time.time()<deadline and not should_stop():model.learn(total_timesteps=args.steps,callback=monitor,reset_num_timesteps=False)
        state.update(status='completed' if state['final_task_verified'] else 'stopped',stop_reason=stop_reason)
    except BaseException as error:
        state.update(status='error',stop_reason=type(error).__name__,error=str(error));raise
    finally:
        dest=save(model,normalizer,reference,output,args.axis,state.get('stop_reason','interrupted'),provenance,curriculum.copy(),latest_report)
        state.update(training_steps=int(model.num_timesteps),policy_updates=int(model._n_updates),latest_checkpoint=str(dest),
            curriculum=curriculum.copy(),finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(output/'status.json',state);normalizer.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--axis',choices=('x','y'),required=True);p.add_argument('--reference',required=True)
    p.add_argument('--source-checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--deadline-at',required=True);p.add_argument('--seed',type=int,default=90000)
    p.add_argument('--steps',type=int,default=200000000);p.add_argument('--eval-every',type=int,default=16384)
    p.add_argument('--eval-seconds',type=float,default=1200.);p.add_argument('--checkpoint-seconds',type=float,default=300.)
    p.add_argument('--initial-eval-episodes',type=int,default=10);p.add_argument('--resume')
    args=p.parse_args()
    if min(args.steps,args.eval_every,args.eval_seconds,args.checkpoint_seconds,args.initial_eval_episodes)<=0:p.error('Positive budgets required')
    train(args)


if __name__=='__main__':main()
