"""Bounded true-dynamics shooting; a candidate is a diagnostic controller.

Only a full original-initial90deg trajectory replayed successfully at BOTH
integrator steps may be exported as a demonstration. No object-state reset.
"""
from pathlib import Path
import time
import numpy as np
from .env import NativeEnv
from .io import atomic_json,digest,fingerprint


def search(model,inputs,profile,output,seed,deadline,stop=lambda:False,seconds=120.):
    inputs=Path(inputs);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    until=min(deadline,time.time()+seconds);rng=np.random.default_rng(seed)
    mean=np.zeros((24,7));std=np.full((24,7),.25);best=None;trials=[]
    for iteration in range(3):
        population=[mean.copy()]+[np.clip(mean+rng.normal(size=mean.shape)*std,-.6,.6) for _ in range(15)]
        ranked=[]
        for offset in population:
            if time.time()>=until or stop():break
            env=NativeEnv(inputs/'reference.npz','x',physical_dt=.002,target_degrees=90.,profile=profile)
            observations=[];actions=[];trace=[];interrupted=False
            try:
                obs,info=env.reset(seed=seed)
                for step in range(250):
                    if time.time()>=until or stop():interrupted=True;break
                    command=np.clip(model.predict(obs,deterministic=True)[0]+offset[min(step//8,23)],-1.,1.).astype(np.float32)
                    observations.append(obs.copy());actions.append(command.copy())
                    obs,_,done,_,info=env.step(command);trace.append(info)
                    if done:break
                value=(1000.*info['success']+100.*info['max_stable_seconds']-info['best_angle_error_degrees']+3.*info['five_finger_duty'])
                row=dict(candidate=len(trials),iteration=iteration,seed=seed,interrupted=interrupted,final=info,score=value)
                trials.append(row)
                if not interrupted:
                    ranked.append((value,offset))
                    if best is None or value>best['score']:
                        best=row;np.savez_compressed(output/'best_candidate.npz',observations=np.asarray(observations),actions=np.asarray(actions))
                        atomic_json(output/'best_candidate_trace.json',trace)
                if info['success'] and not interrupted:
                    checks=[]
                    for dt in (.001,.002):
                        replay=NativeEnv(inputs/'reference.npz','x',physical_dt=dt,target_degrees=90.,profile=profile)
                        try:
                            _,final=replay.reset(seed=seed)
                            incomplete=False
                            for command in actions:
                                if time.time()>=until or stop():incomplete=True;break
                                _,_,done,_,final=replay.step(command)
                                if done:break
                            checks.append(dict(dt=dt,success=bool(final['success'] and not incomplete),interrupted=incomplete,final=final))
                        finally:replay.close()
                    if all(c['success'] for c in checks):
                        path=output/'verified_path.npz'
                        np.savez_compressed(path,observations=np.asarray(observations),actions=np.asarray(actions))
                        result=dict(qualified=True,seed=seed,implementation_sha256=fingerprint(),
                            file_sha256=digest(path),checks=checks,from_original_initial_grasp=True,
                            source='bounded shooting controller; demonstration, not a trained-policy success rate',trials=trials)
                        atomic_json(output/'result.json',result);return result
            finally:env.close()
        if not ranked:break
        elite=np.asarray([x[1] for x in sorted(ranked,key=lambda x:x[0],reverse=True)[:4]])
        mean=elite.mean(axis=0);std=np.clip(elite.std(axis=0),.08,.3)
    result=dict(qualified=False,seed=seed,trials=trials,best=best,
        maximum_angle=max((r['final']['max_progress_degrees'] for r in trials),default=0.),
        source='bounded diagnostic shooting only; no claim of hardware impossibility or task success')
    atomic_json(output/'result.json',result);return result
