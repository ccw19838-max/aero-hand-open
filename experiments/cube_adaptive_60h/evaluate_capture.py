"""Measure auxiliary capture separately; export only real dual-step successes."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,time,json
import numpy as np
import torch
from .env import AdaptiveEnv
from .config import Profile
from .prefix_curriculum import read_actions
from .storage import load,digest,implementation_digest,atomic_json


def evaluate_capture(goal,checkpoint,episodes=6,seed=3000000,seconds=150.):
    torch.set_num_threads(1);root=Path(goal);checkpoint=Path(checkpoint).resolve();model,meta=load(checkpoint)
    provenance=meta['state']['provenance']['auxiliary_reachable_prefix']
    if digest(provenance['source'])!=provenance['sha256']:raise ValueError('Actual prefix changed')
    prefix=read_actions(provenance['source'],provenance['steps']);profile=Profile(**meta['profile'])
    target=meta['state']['stage'];axis=meta['axis']
    goal_meta=json.loads((root/'goal.json').read_text())
    deadline=min(time.time()+seconds,datetime.fromisoformat(goal_meta['deadline_at']).timestamp())
    output=root/'aux_capture_evaluation'/checkpoint.name;output.mkdir(parents=True,exist_ok=True)
    records=[];data={}
    for dt in (.001,.002):
        env=AdaptiveEnv(root/'sources/x/reference.npz',axis,physical_dt=dt,target_degrees=target,profile=profile)
        try:
            for index in range(episodes):
                if time.time()>=deadline:break
                case_seed=seed+index;obs,info=env.reset(seed=case_seed);observations=[];actions=[];interrupted=False
                for step in range(250):
                    if time.time()>=deadline:interrupted=True;break
                    action=prefix[step] if step<len(prefix) else model.predict(obs,deterministic=True)[0]
                    observations.append(obs.copy());actions.append(action.copy())
                    obs,_,done,_,info=env.step(action)
                    if done:break
                record=dict(seed=case_seed,dt=dt,success=bool(info['success'] and not interrupted),
                    interrupted=interrupted,scripted_prefix_controls=min(len(prefix),len(actions)),
                    neural_controls=max(0,len(actions)-len(prefix)),final=info)
                records.append(record);data[(case_seed,dt)]=(observations,actions)
                if interrupted:break
        finally:env.close()
    groups={}
    for dt in (.001,.002):
        cases=[r for r in records if r['dt']==dt]
        groups[str(dt)]=dict(trials=len(cases),requested_trials=episodes,
            successes=sum(r['success'] for r in cases),complete=len(cases)==episodes and not any(r['interrupted'] for r in cases))
    report=dict(observed_at=datetime.now(timezone.utc).isoformat(),checkpoint=str(checkpoint),axis=axis,target_degrees=target,
        controller='real shooting prefix followed by trained neural capture; not neural-only initial-grasp evaluation',
        neural_policy_from_initial_grasp=False,actual_trajectory_from_original_initial_grasp=True,
        groups=groups,records=records)
    atomic_json(output/'report.json',report)
    paired=sorted(set(r['seed'] for r in records if r['success'] and r['dt']==.001)&set(r['seed'] for r in records if r['success'] and r['dt']==.002))
    if paired:
        observations=[];actions=[];checks=[]
        for record in records:
            if record['seed'] not in paired or not record['success']:continue
            episode_obs,episode_actions=data[(record['seed'],record['dt'])];observations.extend(episode_obs);actions.extend(episode_actions)
            checks.append(dict(seed=record['seed'],dt=record['dt'],success=True,stable_seconds=record['final']['stable_seconds']))
        path=output/'qualified_hybrid_demonstrations.npz'
        np.savez_compressed(path,observations=np.asarray(observations,dtype=np.float32),actions=np.asarray(actions,dtype=np.float32))
        atomic_json(path.with_suffix('.json'),dict(goal_degrees=target,axis=axis,controller=report['controller'],
            neural_policy_from_initial_grasp=False,actual_trajectory_from_original_initial_grasp=True,
            implementation_sha256=implementation_digest(),sha256=digest(path),checks=checks,
            source_checkpoint=str(checkpoint),note='Only paired1ms/2ms successful physical trajectories. Not90deg or neural-only task completion.'))
    return {k:v for k,v in report.items() if k!='records'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--goal',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--episodes',type=int,default=6);args=p.parse_args()
    print(json.dumps(evaluate_capture(args.goal,args.checkpoint,args.episodes),ensure_ascii=False))
