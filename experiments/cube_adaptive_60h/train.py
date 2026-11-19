import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,signal,time
from collections import deque
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv,VecNormalize
from .config import Profile,LADDER,validate_profile
from .env import AdaptiveEnv
from .policy import Policy,RetainPPO,transfer
from .evaluate import evaluate,compact,score
from .storage import atomic_json,save,load,load_source,digest
from .decisions import final_pass
from .training_parameters import temporal_parameters,worker_timesteps
from .structured_exploration import settings as exploration_settings,initialize as initialize_exploration,report_std


def train(job_path,resume=False):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    job=json.loads(Path(job_path).read_text());axis=job['axis'];output=Path(job['output']);output.mkdir(parents=True,exist_ok=True)
    temporal=temporal_parameters(job)
    timesteps=worker_timesteps(job)
    exploration=exploration_settings(job)
    profile=validate_profile(Profile(**job['profile']));deadline=datetime.fromisoformat(job['deadline_at']).timestamp()
    if deadline<=time.time():raise ValueError('Goal budget expired')
    reference=Path(job['reference']);stage=float(job['initial_target']);stop={'value':False}
    observation_transform=job.get('observation_transform')
    model_class=RetainPPO
    if observation_transform is not None:
        from .canonical_policy import CanonicalObservation,CanonicalRetainPPO,validate
        validate(observation_transform);model_class=CanonicalRetainPPO
    def handler(signum,frame):stop['value']=True
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    should_stop=lambda:stop['value']
    source_meta=None;streak=0;best_scores={};best_paths={};last_report=None;verification_attempts=0
    if resume:
        checkpoint=(output/'latest').resolve();_,source_meta=load(checkpoint)
        if source_meta['profile']!=profile.dictionary() or source_meta['state']['deadline_at']!=job['deadline_at']:raise ValueError('Resume changed profile/deadline')
        stage=source_meta['state']['stage'];streak=source_meta['state'].get('streak',0)
        verification_attempts=source_meta['state'].get('verification_attempts',0)
        best_scores={k:tuple(v) for k,v in source_meta['state'].get('best_scores',{}).items()}
        best_paths=source_meta['state'].get('best_paths',{}).copy()
        for alias in output.glob('best_*'):
            if not (alias/'metadata.json').exists():continue
            metadata=json.loads((alias/'metadata.json').read_text());report=metadata.get('evaluation')
            if report and report['complete']:
                key=str(int(report['target_degrees']))
                if score(report)>=best_scores.get(key,(-1.,-999.,-1.)):
                    best_scores[key]=score(report);best_paths[key]=str(alias.resolve())
    transform=job.get('reward_transform')
    from .reward_registry import TRANSFORMS,THUMB_DRIVE,wrap as wrap_reward
    if transform not in TRANSFORMS:raise ValueError('Unknown training reward transform')
    env_class=AdaptiveEnv;backend=job.get('native_contact_backend','python')
    if backend not in ('python','c_contacts_equivalent_v1'):raise ValueError('Unknown native contact backend')
    backend_info=dict(name='python')
    if backend=='c_contacts_equivalent_v1':
        from .fast_contacts import FastContactEnv,descriptor
        backend_info=descriptor()
        if backend_info!=job.get('native_contact_descriptor'):raise ValueError('Native contact implementation changed')
        env_class=FastContactEnv
    prefix_actions=None
    if job.get('prefix_source'):
        from .prefix_curriculum import read_actions
        if digest(job['prefix_source'])!=job['prefix_sha256']:raise ValueError('Actual prefix changed')
        prefix_actions=read_actions(job['prefix_source'],int(job['prefix_steps']))
    def factory(dt):
        env=env_class(reference,axis,physical_dt=dt,target_degrees=stage,profile=profile,training=True)
        env=wrap_reward(env,transform)
        if prefix_actions is not None:
            from .prefix_curriculum import ReachablePrefixCurriculum
            env=ReachablePrefixCurriculum(env,prefix_actions,job['prefix_probability'])
        if observation_transform is not None:env=CanonicalObservation(env,observation_transform)
        return env
    factories=[lambda dt=dt:factory(dt) for dt in timesteps]
    vector=SubprocVecEnv(factories,start_method='spawn')
    if resume:
        normalizer=VecNormalize.load(output/'latest/reward_normalizer.pkl',vector)
        normalizer.training=True;model,_=load(output/'latest',normalizer)
        if model.n_steps!=temporal['rollout_steps'] or model.gae_lambda!=temporal['gae_lambda']:
            raise ValueError('Resume changed PPO temporal parameters')
        if getattr(model,'canonical_spec',None)!=observation_transform:
            raise ValueError('Resume changed observation-label coordinates')
        if model.use_sde!=exploration['use_sde'] or model.sde_sample_freq!=exploration['sde_sample_freq']:
            raise ValueError('Resume changed exploration distribution')
        provenance=source_meta['state']['provenance']
        if tuple(provenance.get('training_physics_timesteps',(.001,.002)))!=timesteps:
            raise ValueError('Resume changed training worker allocation')
    else:
        normalizer=VecNormalize(vector,norm_obs=False,norm_reward=True,gamma=profile.gamma,clip_reward=10000.)
        extra=dict(canonical_spec=observation_transform) if observation_transform is not None else {}
        model=model_class(Policy,normalizer,learning_rate=profile.learning_rate,n_steps=temporal['rollout_steps'],batch_size=128,n_epochs=4,
            gamma=profile.gamma,gae_lambda=temporal['gae_lambda'],clip_range=.15,ent_coef=profile.entropy,max_grad_norm=.5,
            seed=job['seed'],device='cpu',verbose=0,**extra,**exploration)
        source,source_meta=load_source(job['source_checkpoint'])
        if source.use_sde and not model.use_sde:raise ValueError('SDE source requires a compatible exploration distribution')
        provenance=transfer(model,source)
        source_transform=source_meta.get('state',{}).get('provenance',{}).get('observation_transform')
        if source_transform is not None and source_transform!=observation_transform:
            raise ValueError('Canonical source requires the same fixed observation-label coordinates')
        provenance.update(source_checkpoint=job['source_checkpoint'],source_training_steps=source_meta['training_steps'])
        if observation_transform is not None:provenance['observation_transform']=observation_transform
        if job.get('initial_std') is not None:
            initial_std=float(job['initial_std'])
            if not .01<=initial_std<=.25:raise ValueError('Exploration initialization outside bounds')
            if model.use_sde:
                provenance['structured_exploration']=initialize_exploration(model,normalizer,initial_std,job['seed'])
            else:
                with torch.no_grad():model.policy.log_std.fill_(float(np.log(initial_std)))
            provenance['initial_action_std']=initial_std
        provenance['training_reward_transform']=transform
        if prefix_actions is not None:
            provenance['auxiliary_reachable_prefix']=dict(source=job['prefix_source'],sha256=job['prefix_sha256'],
                steps=job['prefix_steps'],probability=job['prefix_probability'],formal_evaluation_uses_prefix=False)
    max_std=job.get('max_std')
    noise_scale=1.
    if model.use_sde:
        from .structured_exploration import fingerprint
        spec=provenance.get('structured_exploration')
        if not spec or spec['source_sha256']!=fingerprint():raise ValueError('Structured exploration implementation changed')
        noise_scale=spec['parameter_scale']
    if transform in ('center_missing_tip_mount_v1',THUMB_DRIVE):
        from .capture_geometry import descriptor as geometry_descriptor
        geometry_spec=geometry_descriptor()
        if resume and provenance.get('capture_geometry_descriptor')!=geometry_spec:
            raise ValueError('Resume changed capture-geometry reward implementation')
        provenance['capture_geometry_descriptor']=geometry_spec
    if transform==THUMB_DRIVE:
        from .thumb_drive import descriptor as thumb_descriptor
        thumb_spec=thumb_descriptor()
        if resume and provenance.get('thumb_drive_descriptor')!=thumb_spec:
            raise ValueError('Resume changed thumb-drive reward implementation')
        provenance['thumb_drive_descriptor']=thumb_spec
    provenance['native_contact_backend']=backend_info
    provenance['ppo_temporal_parameters']=temporal
    provenance['training_physics_timesteps']=list(timesteps)
    if max_std is not None and not .01<=float(max_std)<=.25:raise ValueError('Exploration ceiling outside bounds')
    model.retention=None
    if job.get('retention') and profile.bc_weight>0.:
        from .retention import read
        observations,actions=read(job['retention'])
        if observation_transform is not None:
            from .cube_labels import transform as canonicalize_labels
            observations=canonicalize_labels(observations,observation_transform['anchor_quaternion'])
        if abs(profile.action_fraction-.02)>1e-12:raise ValueError('Retention action semantics changed')
        scale=job.get('retention_scale');coefficient=profile.bc_weight
        if scale is not None:
            if not .01<=float(scale)<=.25:raise ValueError('Invalid retention action scale')
            coefficient/=(2.*float(scale)**2)
            provenance['retention_objective']=dict(formula='0.5*bc_weight*mean((action_error/scale)^2)',
                bc_weight=profile.bc_weight,scale=float(scale),raw_mse_coefficient=coefficient,source=job['retention'])
        model.retention=(observations,actions,coefficient,np.random.default_rng(job['seed']))
    normalizer.seed(job['seed'])
    status=dict(task='adaptive_xy_60h_v1',axis=axis,status='initializing',pid=os.getpid(),worker_pids=[p.pid for p in vector.processes],
        parallel_environments=len(timesteps),training_physics_timesteps=list(timesteps),
        stage=stage,streak=streak,best_scores=best_scores,best_paths=best_paths,profile=profile.dictionary(),
        deadline_at=job['deadline_at'],provenance=provenance,training_steps=int(model.num_timesteps),
        ppo_training_epochs=int(model._n_updates),started_at=datetime.now(timezone.utc).isoformat(),verified=False)
    def state():
        return dict(stage=stage,streak=streak,best_scores=best_scores,best_paths=best_paths,
            deadline_at=job['deadline_at'],provenance=provenance,verification_attempts=verification_attempts)
    def checkpoint(reason,report=None,aliases=()):
        dest=save(model,normalizer,reference,output,axis,profile,state(),reason,report,aliases)
        status['latest_checkpoint']=str(dest);return dest
    def validation(episodes=20,seed=1000000):
        nonlocal last_report,streak,stage,verification_attempts
        status.update(status='validating',stage=stage);atomic_json(output/'status.json',status)
        report=evaluate(model,reference,axis,profile,stage,episodes,seed,deadline,should_stop)
        last_report=report;key=str(int(stage));aliases=[]
        if report['complete'] and score(report)>best_scores.get(key,(-1.,-999.,-1.)):
            best_scores[key]=score(report);aliases=['best_'+key,'best']
        dest=checkpoint('validation',report,aliases)
        if aliases:best_paths[key]=str(dest)
        atomic_json(output/'validation'/f'step_{model.num_timesteps:09d}_{int(stage):03d}.json',report)
        status.update(latest_evaluation=compact(report),best_paths=best_paths.copy())
        streak=streak+1 if report['complete'] and report['success_rate']>=.9 else 0
        if streak>=2 and stage<90.:
            stage=LADDER[LADDER.index(stage)+1];streak=0;normalizer.env_method('set_task',stage)
            checkpoint('curriculum_advanced',report)
        elif streak>=2 and stage==90.:
            attempt=verification_attempts;verification_attempts+=1
            checkpoint('verification_candidate',report)
            final_seed=job.get('verification_seed_base',60000000)+('xy'.index(axis))*10000000+job.get('segment_index',0)*100000+attempt*1000
            independent=evaluate(model,reference,axis,profile,90.,100,final_seed,deadline,should_stop)
            atomic_json(output/'independent'/f'step_{model.num_timesteps:09d}.json',independent)
            if final_pass(independent):
                checkpoint('verified_from_initial_90deg',independent,('verified',));status['verified']=True
            streak=0
        status.update(stage=stage,streak=streak);atomic_json(output/'status.json',status)
        return report
    atomic_json(output/'status.json',status)
    last_save=time.time();last_eval=time.time();reason='wall_clock_budget'
    try:
        checkpoint('initialized')
        if not resume and time.time()<deadline:validation(job.get('initial_eval_episodes',10))
        class Monitor(BaseCallback):
            def __init__(self):
                super().__init__();self.next_eval=model.num_timesteps+job.get('eval_steps',16384);self.next_status=0.
                self.thumb_samples=deque(maxlen=512)
            def _on_step(self):
                nonlocal last_save,last_eval,reason
                if transform==THUMB_DRIVE:
                    for info in self.locals.get('infos',[]):
                        if 'thumb_drive_quality' in info:
                            self.thumb_samples.append((info['thumb_drive_quality'],info['thumb_axis_moment_Nm'],
                                info['thumb_surface_speed_m_s']))
                if max_std is not None:
                    with torch.no_grad():self.model.policy.log_std.clamp_(max=float(np.log(max_std*noise_scale)))
                if should_stop():reason='signal';return False
                if status['verified']:reason='verified';return False
                if time.time()>=deadline:return False
                if time.time()>=self.next_status:
                    if self.thumb_samples:
                        values=np.asarray(self.thumb_samples)
                        status['thumb_drive_recent']=dict(control_samples=len(values),
                            mean_quality=float(values[:,0].mean()),positive_moment_fraction=float(np.mean(values[:,1]>0.)),
                            mean_axis_moment_Nm=float(values[:,1].mean()),mean_surface_speed_m_s=float(values[:,2].mean()),
                            scope='last512 stochastic training control samples; not independent success evaluation')
                    status.update(status='training',stage=stage,streak=streak,training_steps=int(self.model.num_timesteps),
                        ppo_training_epochs=int(self.model._n_updates),updated_at=datetime.now(timezone.utc).isoformat(),
                        best_paths=best_paths.copy(),action_std=report_std(self.model,provenance),
                        exploration_distribution='gSDE' if self.model.use_sde else 'independent_gaussian',
                        action_std_is_reference_calibration=bool(self.model.use_sde))
                    atomic_json(output/'status.json',status);self.next_status=time.time()+10.
                if self.model.num_timesteps>=self.next_eval or time.time()-last_eval>=job.get('eval_seconds',1200.):
                    report=validation();last_eval=time.time();last_save=time.time();self.next_eval=self.model.num_timesteps+job.get('eval_steps',16384)
                    if not report['complete']:return False
                if time.time()-last_save>=job.get('save_seconds',300.):checkpoint('periodic');last_save=time.time()
                return True
        if not status['verified'] and time.time()<deadline and not should_stop():
            model.learn(total_timesteps=job.get('steps',1000000000),callback=Monitor(),reset_num_timesteps=False)
        if should_stop():reason='signal'
        if status['verified']:reason='verified'
        status.update(status='completed' if status['verified'] else 'stopped',stop_reason=reason)
    except BaseException as error:
        status.update(status='error',error=repr(error));raise
    finally:
        checkpoint(status.get('stop_reason','interrupted'),last_report)
        status.update(training_steps=int(model.num_timesteps),ppo_training_epochs=int(model._n_updates),stage=stage,
            finished_at=datetime.now(timezone.utc).isoformat(),best_paths=best_paths.copy())
        atomic_json(output/'status.json',status);normalizer.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args();train(args.job,args.resume)
