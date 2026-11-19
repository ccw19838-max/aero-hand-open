import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from collections import deque
import argparse,fcntl,json,signal,time,shutil
import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv,VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from experiments.cube_adaptive_60h.config import Profile
from experiments.cube_adaptive_60h.policy import Policy,RetainPPO,transfer
from experiments.cube_adaptive_60h.storage import load as load_legacy
from .env import FastEnv,VERSION
from .reward import training_reward
from .io import atomic_json,digest,fingerprint,save,load,link
from .evaluation import evaluate,compact,score,final_pass


def train(root):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    root=Path(root).resolve();job=json.loads((root/'job.json').read_text());profile=Profile(**job['profile'])
    deadline=datetime.fromisoformat(job['deadline_at']).timestamp();job_sha=digest(root/'job.json')
    if time.time()>=deadline:raise ValueError('Experiment deadline expired; resume does not reset budget')
    stop={'requested':False}
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.update(requested=True))
    should_stop=lambda:stop['requested']
    stage=float(job['initial_stage']);streak=0;best={};last_reports={};reviews=[];verified=False
    resume=(root/'latest').exists()
    if resume:
        _,meta=load(root/'latest');state=meta['state']
        if state['job_sha256']!=job_sha:raise ValueError('Resume changed task configuration or budget')
        stage=state['stage'];streak=state['streak'];best=state['best'];last_reports=state['last_reports'];reviews=state['reviews']
        verified=state['verified'];provenance=state['provenance']
    def factory(dt):
        env=FastEnv(root/'inputs/reference.npz','x',physical_dt=dt,target_degrees=stage,profile=profile,training=True,
            prefix_bank=root/'inputs/prefix_bank.npz')
        return training_reward(env)
    vector=SubprocVecEnv([lambda dt=dt:factory(dt) for dt in job['physics_timesteps']],start_method='spawn')
    if resume:
        normalizer=VecNormalize.load(root/'latest/normalizer.pkl',vector);normalizer.training=True
        model,_=load(root/'latest',normalizer)
    else:
        normalizer=VecNormalize(vector,norm_obs=False,norm_reward=True,gamma=profile.gamma,clip_reward=10000.)
        model=RetainPPO(Policy,normalizer,learning_rate=profile.learning_rate,n_steps=job['rollout_steps'],batch_size=128,n_epochs=4,
            gamma=profile.gamma,gae_lambda=job['gae_lambda'],clip_range=.15,ent_coef=profile.entropy,max_grad_norm=.5,
            seed=job['seed'],device='cpu',verbose=0,use_sde=False)
        source,source_meta=load_legacy(job['source_checkpoint']);provenance=transfer(model,source)
        with torch.no_grad():
            model.policy.action_net[0].weight.mul_(job['action_head_scale']);model.policy.action_net[0].bias.mul_(job['action_head_scale'])
            model.policy.log_std.fill_(float(np.log(job['initial_std'])))
        provenance.update(source_checkpoint=job['source_checkpoint'],source_training_steps=source_meta['training_steps'],
            action_head_scale=job['action_head_scale'],force_feedback='kinematic family, including right_t_link',
            tip_distance_order=['thumb','index','middle','ring','pinky'],actor_tip_features_order_unchanged=True,
            reward_optimizer_normalization='fresh on version change; subsequent resume restores optimizer and reward statistics')
    model.retention=None;normalizer.seed(job['seed'])
    def use_verified_path():
        proof=root/'verified_path.json'
        if not proof.exists():return False
        evidence=json.loads(proof.read_text());path=root/'verified_path.npz'
        if not evidence['qualified'] or evidence['implementation_sha256']!=fingerprint() or digest(path)!=evidence['file_sha256']:
            raise ValueError('Invalid demonstration qualification')
        with np.load(path,allow_pickle=False) as data:observations=data['observations'].copy();actions=data['actions'].copy()
        model.retention=(observations,actions,.02/(2.*.15**2),np.random.default_rng(job['seed']+900))
        provenance['verified_path']=dict(sha256=evidence['file_sha256'],seed=evidence['seed'],bc_weight=.02,action_error_scale=.15)
        return True
    if stage==90.:use_verified_path()
    status=dict(task=VERSION,status='initializing',pid=os.getpid(),worker_pids=[p.pid for p in vector.processes],
        parallel_environments=len(vector.processes),physics_timesteps=job['physics_timesteps'],stage=stage,
        deadline_at=job['deadline_at'],started_at=job['started_at'],profile=job['profile'],verified=verified)
    now=time.time();last_save=now;last_eval=now
    previous_status=json.loads((root/'status.json').read_text()) if resume and (root/'status.json').exists() else {}
    last_eval=previous_status.get('last_eval_wall',now);last_review=previous_status.get('last_review_wall',datetime.fromisoformat(job['started_at']).timestamp())
    status.update(last_eval_wall=last_eval,last_review_wall=last_review)
    def state():return dict(stage=stage,streak=streak,best=best,last_reports=last_reports,reviews=reviews,
        verified=verified,job_sha256=job_sha,deadline_at=job['deadline_at'],provenance=provenance)
    def persist():
        status.update(updated_at=datetime.now(timezone.utc).isoformat(),stage=stage,streak=streak,
            training_steps=int(model.num_timesteps),ppo_training_epochs=int(model._n_updates),best=best,
            latest_evaluations={k:compact(v) for k,v in last_reports.items()},verified=verified,
            next_evaluation_at=datetime.fromtimestamp(min(deadline,status['last_eval_wall']+job['eval_seconds']),timezone.utc).isoformat())
        atomic_json(root/'status.json',status)
        local=lambda value:datetime.fromisoformat(value).astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        lines=['# 修正接触反馈后的 X 轴训练',f'状态：{status["status"]}；阶段：'+('真实失触状态补位' if stage==0. else '原始初握 X90°'),
            f'采样步：{model.num_timesteps:,}；训练环境：{len(vector.processes)}；最终通过：{verified}',
            f'预算截止：{local(job["deadline_at"])}（北京时间）',
            '辅助补位的成功率不能解释为完成90°。最终仍要求两步长各100个新初态全部通过。','']
        for name,report in last_reports.items():lines.append(f'- {name}：成功率 {report["success_rate"]:.1%}，最大目标保持 {report["maximum_stable_seconds"]:.2f}s')
        (root/'status.md').write_text('\n'.join(lines)+'\n')
    def checkpoint(reason,report=None,aliases=()):
        path=save(model,normalizer,root,state(),reason,report,aliases);status['latest_checkpoint']=str(path);persist();return path
    def run_evaluation(episodes=20,review=False):
        nonlocal stage,streak,verified,last_eval,last_review
        status['status']='evaluating';persist()
        seed=job['review_seed']+len(reviews)*100 if review else job['validation_seed']
        report=evaluate(model,root/'inputs',profile,stage,episodes,seed,deadline,should_stop)
        key='recovery' if stage==0. else 'x90';last_reports[key]=report
        if report['complete']:
            if score(report)>tuple(best.get(key,{}).get('score',(-1.,-999.,-1.))):
                path=checkpoint('best_'+key,report,('best_'+key,));best[key]=dict(path=str(path),score=score(report))
            atomic_json(root/'evaluations'/f'{model.num_timesteps:09d}_{key}_{seed}.json',report)
        if stage==0. and report['complete']:
            formal=evaluate(model,root/'inputs',profile,90.,min(10,episodes),seed+1000000,deadline,should_stop)
            last_reports['x90']=formal
            atomic_json(root/'evaluations'/f'{model.num_timesteps:09d}_x90_probe_{seed}.json',formal)
            enough=all(g['trials']>=20 for g in report['groups'].values())
            streak=streak+1 if enough and report['success_rate']>=job['auxiliary_promotion_threshold'] else 0
            if streak>=job['auxiliary_promotion_streak']:
                checkpoint('recovery_qualified',report,('qualified_recovery',))
                from .path_probe import search
                probe=root/'path_probes'/f'step_{model.num_timesteps:09d}_{int(time.time())}'
                status['status']='trajectory_probe';persist()
                evidence=search(model,root/'inputs',profile,probe,5700000+len(reviews)*100,
                    deadline,should_stop,seconds=job.get('path_probe_seconds',120.))
                if evidence['qualified']:
                    shutil.copy2(probe/'verified_path.npz',root/'verified_path.npz')
                    atomic_json(root/'verified_path.json',evidence);use_verified_path()
                    stage=90.;normalizer.env_method('set_task',90.)
                streak=0
                atomic_json(root/'curriculum_transition.json',dict(at=datetime.now(timezone.utc).isoformat(),
                    from_stage='contact_recovery',to_stage='original_initial_x90' if stage==90. else 'contact_recovery',
                    recovery_rate=report['success_rate'],complete_90_path_verified=evidence['qualified'],probe=str(probe)))
        elif stage==90. and report['complete']:
            enough=all(g['trials']>=20 for g in report['groups'].values())
            streak=streak+1 if enough and report['success_rate']>=.9 else 0
            if streak>=2:
                independent=evaluate(model,root/'inputs',profile,90.,100,job['final_seed']+len(reviews)*1000,deadline,should_stop,trace=True)
                atomic_json(root/'independent'/f'{model.num_timesteps:09d}.json',independent)
                if final_pass(independent):verified=True;checkpoint('verified_original_x90',independent,('verified',))
                streak=0
        if review:
            reviews.append(dict(at=datetime.now(timezone.utc).isoformat(),stage=stage,report=compact(report),
                action='preserve_recovery_gate_and_original_goal',complete_90_path_verified=verified))
            atomic_json(root/'reviews'/f'{len(reviews):03d}.json',reviews[-1]);last_review=time.time();status['last_review_wall']=last_review
        last_eval=time.time();status['last_eval_wall']=last_eval
        checkpoint('evaluation',report)
        return report
    reason='step_budget'
    try:
        checkpoint('resumed' if resume else 'initialized')
        if not resume and not verified:run_evaluation(job.get('initial_eval_episodes',20))
        class Monitor(BaseCallback):
            def __init__(self):super().__init__();self.next_status=0.;self.samples=deque(maxlen=512)
            def _on_step(self):
                nonlocal reason,last_save
                with torch.no_grad():self.model.policy.log_std.clamp_(max=float(np.log(job['max_std'])))
                for info in self.locals.get('infos',[]):self.samples.append(info)
                if should_stop():reason='signal';return False
                if verified:reason='verified';return False
                if time.time()>=deadline:reason='wall_clock_budget';return False
                if time.time()>=self.next_status:
                    status['status']='training';status['action_std']=model.policy.log_std.detach().exp().cpu().tolist()
                    if self.samples:status['recent_training']=dict(samples=len(self.samples),
                        recovering_fraction=float(np.mean([i['control_mode']=='recover' for i in self.samples])),
                        five_finger_fraction=float(np.mean([i['five_finger_contact'] for i in self.samples])),
                        prefix_setup_failures=sum(bool(i['auxiliary_setup_failure']) for i in self.samples))
                    persist();self.next_status=time.time()+10.
                review=time.time()-last_review>=job['review_seconds']
                if review or time.time()-last_eval>=job['eval_seconds']:
                    report=run_evaluation(review=review);last_save=time.time()
                    if not report['complete']:reason='evaluation_interrupted';return False
                if time.time()-last_save>=job['save_seconds']:checkpoint('periodic');last_save=time.time()
                return True
        if not verified and not should_stop() and time.time()<deadline:
            model.learn(total_timesteps=job.get('steps',1000000000),callback=Monitor(),reset_num_timesteps=False)
        if should_stop():reason='signal'
        elif time.time()>=deadline:reason='wall_clock_budget'
        status['status']='complete' if verified else 'stopped';status['stop_reason']=reason
    except BaseException as error:
        status.update(status='error',error=repr(error));raise
    finally:
        checkpoint(status.get('stop_reason','interrupted'));normalizer.close()
        status.update(finished_at=datetime.now(timezone.utc).isoformat());persist()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    with (Path(a.root)/'train.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Training already active')
        train(a.root)
