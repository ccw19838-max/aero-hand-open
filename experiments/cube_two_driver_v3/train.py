import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from collections import deque
import argparse,fcntl,json,signal,time
import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv,VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from experiments.cube_adaptive_60h.config import Profile
from experiments.cube_adaptive_60h.policy import Policy,RetainPPO
from experiments.cube_adaptive_60h.storage import load as load_legacy
from .env import FastEnv,VERSION,LADDER,training_reward
from .io import atomic_json,digest,save,load
from .evaluation import evaluate,compact,score,final_pass


def train(root):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    root=Path(root).resolve();job=json.loads((root/'job.json').read_text());profile=Profile(**job['profile'])
    deadline=datetime.fromisoformat(job['deadline_at']).timestamp();job_sha=digest(root/'job.json')
    if time.time()>=deadline:raise ValueError('Original deadline expired; resume never resets the budget')
    stop={'requested':False}
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.update(requested=True))
    should_stop=lambda:stop['requested'];resume=(root/'latest').exists()
    stage=float(job['initial_stage']);streak=0;best={};reports={};verified=False;reviews=[]
    if resume:
        _,meta=load(root/'latest');state=meta['state']
        if state['job_sha256']!=job_sha:raise ValueError('Resume changed experiment configuration')
        stage=state['stage'];streak=state['streak'];best=state['best'];reports=state['reports'];reviews=state['reviews'];verified=state['verified'];provenance=state['provenance']
    def factory(dt):return training_reward(FastEnv(root/'inputs/reference.npz','x',physical_dt=dt,target_degrees=stage,profile=profile,training=True))
    vector=SubprocVecEnv([lambda dt=dt:factory(dt) for dt in job['physics_timesteps']],start_method='spawn')
    if resume:
        normalizer=VecNormalize.load(root/'latest/normalizer.pkl',vector);normalizer.training=True;model,_=load(root/'latest',normalizer)
    else:
        normalizer=VecNormalize(vector,norm_obs=False,norm_reward=True,gamma=profile.gamma,clip_reward=10000.)
        model=RetainPPO(Policy,normalizer,learning_rate=profile.learning_rate,n_steps=job['rollout_steps'],batch_size=128,n_epochs=4,
            gamma=profile.gamma,gae_lambda=.98,clip_range=.15,ent_coef=profile.entropy,max_grad_norm=.5,seed=job['seed'],device='cpu',verbose=0,use_sde=False)
        source,source_meta=load_legacy(job['source_checkpoint'])
        model.policy.features_extractor.load_state_dict(source.policy.features_extractor.state_dict())
        model.policy.mlp_extractor.load_state_dict(source.policy.mlp_extractor.state_dict())
        provenance=dict(hidden_features_transferred=True,observations=143,
            source_exploration='gSDE' if source.use_sde else 'Gaussian',new_exploration='bounded per-motor Gaussian')
        # Preserve hidden features, remove the learned action that opens the
        # index finger. All seven motors remain learnable; no force injection.
        with torch.no_grad():
            model.policy.value_net.weight.zero_();model.policy.value_net.bias.zero_()
            model.policy.action_net[0].weight.zero_();model.policy.action_net[0].bias.copy_(torch.as_tensor(np.arctanh(job['initial_action_mean']),dtype=torch.float32))
            model.policy.log_std.copy_(torch.as_tensor(np.log(job['initial_std']),dtype=torch.float32))
        provenance.update(source_checkpoint=job['source_checkpoint'],source_policy_sha256=digest(Path(job['source_checkpoint'])/'policy.zip'),
            source_training_steps=source_meta['training_steps'],action_head='reset to documented initial mean; all motors learned',
            initial_action_mean=job['initial_action_mean'],optimizer='fresh on V3 migration; restored on subsequent V3 resume')
    model.retention=None;normalizer.seed(job['seed'])
    previous=json.loads((root/'status.json').read_text()) if resume and (root/'status.json').exists() else {}
    last_eval=previous.get('last_eval_wall',time.time());last_review=previous.get('last_review_wall',time.time());last_save=time.time()
    status=dict(task=VERSION,status='initializing',pid=os.getpid(),worker_pids=[p.pid for p in vector.processes],parallel_environments=4,
        physics_timesteps=job['physics_timesteps'],stage=stage,started_at=job['started_at'],deadline_at=job['deadline_at'],last_eval_wall=last_eval,last_review_wall=last_review)
    def state():return dict(stage=stage,streak=streak,best=best,reports=reports,reviews=reviews,verified=verified,provenance=provenance,job_sha256=job_sha,deadline_at=job['deadline_at'])
    def persist():
        status.update(updated_at=datetime.now(timezone.utc).isoformat(),stage=stage,streak=streak,training_steps=int(model.num_timesteps),
            ppo_training_epochs=int(model._n_updates),best=best,verified=verified,latest_evaluations={k:compact(v) for k,v in reports.items()},
            next_evaluation_at=datetime.fromtimestamp(min(deadline,status['last_eval_wall']+job['eval_seconds']),timezone.utc).isoformat())
        atomic_json(root/'status.json',status)
        local=datetime.fromisoformat(job['deadline_at']).astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        lines=['# X轴：拇指/小指推进，食指/中指/无名指稳定',f'状态：{status["status"]}；当前课程：{stage:g}°；最终90°验收通过：{verified}',
            f'采样步：{model.num_timesteps:,}；并行环境：4；截止：{local}（北京时间）',
            '各课程均从原始初握出发；小角度成功不能算90°。最终要求五指受力、低速度保持1秒。','']
        for key,r in reports.items():lines.append(f'- {key}：成功率 {r["success_rate"]:.1%}；最大转角 {r["maximum_signed_angle"]:.2f}°；中间三指受力占比 {r["three_support_duty"]:.1%}')
        (root/'status.md').write_text('\n'.join(lines)+'\n')
    def checkpoint(reason,report=None,aliases=()):
        path=save(model,normalizer,root,state(),reason,report,aliases);status['latest_checkpoint']=str(path);persist();return path
    def assess(episodes=20,review=False):
        nonlocal stage,streak,verified,last_eval,last_review
        status['status']='evaluating';persist();seed=job['review_seed']+len(reviews)*100 if review else job['validation_seed']
        report=evaluate(model,root/'inputs',profile,stage,episodes,seed,deadline,should_stop)
        key=f'x{stage:g}';reports[key]=report
        atomic_json(root/'evaluations'/f'{model.num_timesteps:09d}_{key}_{seed}.json',report)
        if report['complete']:
            if score(report)>tuple(best.get(key,{}).get('score',(-1.,-999.,-1.))):
                path=checkpoint('best_'+key,report,('best_'+key,));best[key]=dict(path=str(path),score=score(report))
            if stage!=90.:
                probe=evaluate(model,root/'inputs',profile,90.,min(10,episodes),seed+1000000,deadline,should_stop)
                reports['x90_probe']=probe;atomic_json(root/'evaluations'/f'{model.num_timesteps:09d}_x90_probe_{seed}.json',probe)
            enough=all(g['trials']>=20 for g in report['groups'].values())
            streak=streak+1 if enough and report['success_rate']>=.8 else 0
            if streak>=2:
                if stage==90.:
                    independent=evaluate(model,root/'inputs',profile,90.,100,job['final_seed']+len(reviews)*1000,deadline,should_stop,trace=True)
                    atomic_json(root/'independent'/f'{model.num_timesteps:09d}.json',independent)
                    if final_pass(independent):verified=True;checkpoint('verified',independent,('verified',))
                else:
                    old=stage;checkpoint('qualified_'+key,report,('qualified_'+key,))
                    stage=LADDER[LADDER.index(stage)+1];normalizer.env_method('set_task',stage)
                    atomic_json(root/'transitions'/f'{model.num_timesteps:09d}.json',dict(from_degrees=old,to_degrees=stage,report=compact(report)))
                streak=0
        if review:
            reviews.append(dict(at=datetime.now(timezone.utc).isoformat(),stage=stage,report=compact(report),action='retain_named_support_and_fixed_physics'))
            atomic_json(root/'reviews'/f'{len(reviews):03d}.json',reviews[-1]);last_review=time.time();status['last_review_wall']=last_review
        last_eval=time.time();status['last_eval_wall']=last_eval;checkpoint('evaluation',report);return report
    reason='step_budget'
    try:
        checkpoint('resumed' if resume else 'initialized')
        if not resume:assess(job.get('initial_eval_episodes',20))
        class Monitor(BaseCallback):
            def __init__(self):super().__init__();self.next_status=0.;self.samples=deque(maxlen=512);self.first_saved=resume
            def _on_rollout_start(self):
                if not self.first_saved and self.model._n_updates>0:checkpoint('first_update');self.first_saved=True
            def _on_step(self):
                nonlocal reason,last_save
                with torch.no_grad():self.model.policy.log_std.copy_(torch.minimum(self.model.policy.log_std,torch.as_tensor(np.log(job['max_std']),dtype=torch.float32)))
                self.samples.extend(self.locals.get('infos',[]))
                if should_stop():reason='signal';return False
                if verified:reason='verified';return False
                if time.time()>=deadline:reason='wall_clock_budget';return False
                if time.time()>=self.next_status:
                    status['status']='training'
                    if self.samples:status['recent_training']=dict(three_support_fraction=float(np.mean([i['three_support'] for i in self.samples])),
                        five_finger_fraction=float(np.mean([i['five_finger_contact'] for i in self.samples])),samples=len(self.samples))
                    persist();self.next_status=time.time()+10.
                review=time.time()-last_review>=job['review_seconds']
                if review or time.time()-last_eval>=job['eval_seconds']:
                    r=assess(review=review);last_save=time.time()
                    if not r['complete']:reason='evaluation_interrupted';return False
                if time.time()-last_save>=job['save_seconds']:checkpoint('periodic');last_save=time.time()
                return True
        if not verified and not should_stop() and time.time()<deadline:model.learn(total_timesteps=job.get('steps',1000000000),callback=Monitor(),reset_num_timesteps=False)
        if should_stop():reason='signal'
        elif time.time()>=deadline:reason='wall_clock_budget'
        status.update(status='complete' if verified else 'stopped',stop_reason=reason)
    except BaseException as error:status.update(status='error',error=repr(error));raise
    finally:
        checkpoint(status.get('stop_reason','interrupted'));normalizer.close();status['finished_at']=datetime.now(timezone.utc).isoformat();persist()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    with (Path(a.root)/'train.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Training already active')
        train(a.root)
