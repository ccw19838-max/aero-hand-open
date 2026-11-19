"""Validated trajectory cloning followed by one independent axis teacher's PPO."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from pathlib import Path
from datetime import datetime,timezone
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv,VecNormalize
from experiments.cube_quarter_turn.task import SYMMETRIES,matrix_quaternion,quaternion_multiply,canonical_quaternion
from .env import TeacherEnv
from .policy import TeacherPolicy,initialize_prior,cloning_update
from .planner import load_demonstration
from .evaluate import evaluate,compact
from .storage import atomic_json,save,load,digest


def augment(observations,actions):
    groups=[]
    for matrix in SYMMETRIES:
        symmetry=matrix_quaternion(matrix);copy=observations.copy()
        for start in (71,84,94):
            copy[:,start:start+4]=np.array([canonical_quaternion(quaternion_multiply(q,symmetry))
                                          for q in observations[:,start:start+4]])
        groups.append(copy)
    return np.concatenate(groups),np.tile(actions,(24,1))


class DemoPPO(PPO):
    def train(self):
        super().train()
        if getattr(self,'demo_observations',None) is not None:
            for _ in range(2):
                indices=self.demo_rng.integers(len(self.demo_actions),size=128)
                loss=cloning_update(self,self.demo_observations,self.demo_actions,indices,weight=.5)
            self.logger.record('train/demonstration_loss',loss)
    def _excluded_save_params(self):
        return super()._excluded_save_params()+['demo_observations','demo_actions','demo_rng']


def new_model(vector,seed=0):
    return DemoPPO(TeacherPolicy,vector,learning_rate=2e-4,n_steps=128,batch_size=128,n_epochs=4,
        gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=.001,max_grad_norm=.5,
        policy_kwargs={'log_std_init':float(np.log(.05))},seed=seed,device='cpu',verbose=0)


def train_axis(demos,axis,output,source_checkpoint,deadline,envs=1,seed=0,
               bc_steps=3000,eval_every=16384,should_stop=lambda:False,resume=None):
    if not demos:raise ValueError('PPO is gated on at least one complete replay-validated demonstration')
    if envs not in (1,2):raise ValueError('Bounded to one or two CPU workers per teacher')
    loaded=[load_demonstration(p) for p in demos]
    if any(meta['axis']!=axis for _,_,meta in loaded):raise ValueError('Axis mismatch')
    # Each teacher uses one physical starting grip. Do not mix incompatible resets.
    reference=Path(loaded[0][2]['reference']);reference_hash=digest(reference)
    loaded=[item for item in loaded if item[2]['reference_sha256']==reference_hash]
    observations,actions=augment(np.concatenate([x[0] for x in loaded]),np.concatenate([x[1] for x in loaded]))
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed)
    def factory():return TeacherEnv(reference,axis)
    vector=DummyVecEnv([factory]) if envs==1 else SubprocVecEnv([factory]*envs,start_method='spawn')
    if resume:
        checkpoint=Path(resume).resolve()
        normalizer=VecNormalize.load(checkpoint/'reward_normalizer.pkl',vector)
        normalizer.training=True;normalizer.norm_obs=False;normalizer.norm_reward=True
        _,metadata=load(checkpoint)
        model=DemoPPO.load(checkpoint/'policy.zip',env=normalizer,device='cpu')
        provenance=metadata['source_initialization']
        if metadata['axis']!=axis or digest(checkpoint/'reference.npz')!=reference_hash:raise ValueError('Resume reset/axis changed')
        if provenance.get('original_deadline_epoch')!=deadline:raise ValueError('Resume must preserve original deadline')
    else:
        normalizer=VecNormalize(vector,norm_obs=False,norm_reward=True,clip_reward=10000.,gamma=.99)
        model=new_model(normalizer,seed)
        from experiments.cube_official_baseline.train import load as load_prior
        source,metadata=load_prior(source_checkpoint)
        probe=factory();obs,_=probe.reset(seed=0,options={'reference':True});probe.close()
        provenance=initialize_prior(model,source,obs)
        provenance.update(source_checkpoint=str(source_checkpoint),demonstrations=list(map(str,demos)),
            source_training_steps=metadata['training_steps'],symmetry_augmentation=24,original_deadline_epoch=deadline)
    model.demo_observations=observations;model.demo_actions=actions;model.demo_rng=rng
    state=dict(phase='behavior_cloning',axis=axis,pid=os.getpid(),envs=envs,training_steps=model.num_timesteps,
        deadline_epoch=deadline,demonstrations=len(loaded),final_task_verified=False)
    atomic_json(output/'status.json',state)
    try:
        if not resume:
            losses=[]
            for step in range(bc_steps):
                if time.time()>=deadline or should_stop():break
                losses.append(cloning_update(model,observations,actions,rng.integers(len(actions),size=128)))
                if step%100==0:
                    state.update(bc_updates=step+1,bc_loss=losses[-1]);atomic_json(output/'status.json',state)
            with torch.no_grad():model.policy.log_std.fill_(float(np.log(.05)))
            state['cloning_updates']=len(losses)
        checkpoint=save(model,normalizer,reference,output,axis,'after_cloning',provenance)
        report=evaluate(model,reference,axis,20,deadline=deadline,should_stop=should_stop)
        atomic_json(output/'after_cloning_evaluation.json',report)
        best=(report['success_rate'],-report['mean_best_angle_error_degrees']) if report['complete'] else (-1.,-180.)
        save(model,normalizer,reference,output,axis,'cloning_validation',provenance,report,('best',))
        state.update(latest_checkpoint=str(checkpoint),evaluation=compact(report))
        class Monitor(BaseCallback):
            def __init__(self):
                super().__init__();self.next_eval=model.num_timesteps+eval_every;self.next_status=0.;self.streak=0
                self.reason='wall_clock_budget';self.last_report=report
            def _on_step(self):
                nonlocal best
                if should_stop():self.reason='signal';return False
                if time.time()>=deadline:return False
                if time.time()>=self.next_status:
                    state.update(phase='ppo',training_steps=int(self.model.num_timesteps),updated_at=datetime.now(timezone.utc).isoformat())
                    atomic_json(output/'status.json',state);self.next_status=time.time()+15.
                if self.model.num_timesteps>=self.next_eval:
                    state['phase']='validation';atomic_json(output/'status.json',state)
                    result=evaluate(self.model,reference,axis,20,deadline=deadline,should_stop=should_stop)
                    self.last_report=result
                    key=(result['success_rate'],-result['mean_best_angle_error_degrees'])
                    aliases=[]
                    if result['complete'] and key>best:best=key;aliases=['best']
                    saved=save(self.model,normalizer,reference,output,axis,'validation',provenance,result,aliases)
                    atomic_json(output/'validation'/f'step_{self.model.num_timesteps:09d}.json',result)
                    state.update(latest_checkpoint=str(saved),evaluation=compact(result))
                    self.streak=self.streak+1 if result['complete'] and result['success_rate']>=.9 else 0
                    if self.streak>=2:
                        independent=evaluate(self.model,reference,axis,100,seed=50000000,deadline=deadline,should_stop=should_stop)
                        atomic_json(output/'independent_evaluation.json',independent)
                        if independent['complete'] and independent['success_rate']>=.9:
                            self.reason='axis90_verified';state['axis_verified']=True
                            save(self.model,normalizer,reference,output,axis,self.reason,provenance,independent,('verified',))
                            return False
                        self.streak=0
                    self.next_eval=self.model.num_timesteps+eval_every
                    if not result['complete']:return False
                return True
        monitor=Monitor()
        if time.time()<deadline and not should_stop():
            model.learn(total_timesteps=100000000,callback=monitor,reset_num_timesteps=False)
        state.update(phase='stopped',stop_reason=monitor.reason,training_steps=int(model.num_timesteps),
            evaluation=compact(monitor.last_report))
    finally:
        checkpoint=save(model,normalizer,reference,output,axis,state.get('stop_reason','interrupted'),provenance)
        state['latest_checkpoint']=str(checkpoint);atomic_json(output/'status.json',state);normalizer.close()
    return state
