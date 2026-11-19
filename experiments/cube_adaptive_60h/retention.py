from pathlib import Path
import json
import numpy as np
from .env import AdaptiveEnv
from .storage import atomic_json,digest,implementation_digest,load_source


def build(source,reference,axis,profile,output,per_dt=6,deadline=float('inf')):
    import time
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    model,meta=load_source(source);evaluation=meta.get('evaluation')
    if not evaluation or evaluation.get('target_degrees')!=30.:
        raise ValueError('Retention teacher must have an actual validated 30deg policy')
    input_size=model.observation_space.shape[0]
    if input_size not in (138,143):raise ValueError('Unknown retention teacher observations')
    observations=[];actions=[];checked=[]
    for dt in (.001,.002):
        chosen=[r for r in evaluation['records'] if r['success'] and r['dt']==dt][:per_dt]
        env=AdaptiveEnv(reference,axis,physical_dt=dt,target_degrees=30.,profile=profile)
        try:
            for record in chosen:
                if time.time()>=deadline:break
                obs,info=env.reset(seed=record['seed']);episode_obs=[];episode_actions=[]
                for _ in range(250):
                    action=model.predict(obs[:input_size],deterministic=True)[0]
                    episode_obs.append(obs.copy());episode_actions.append(action.copy())
                    obs,_,done,_,info=env.step(action)
                    if done:break
                if not info['success']:raise ValueError('Transferred retention case no longer meets five-finger criteria')
                observations.extend(episode_obs);actions.extend(episode_actions)
                checked.append(dict(seed=record['seed'],dt=dt,success=True,stable_seconds=info['stable_seconds']))
        finally:env.close()
    if not checked:raise ValueError('No genuine successful retention episodes')
    np.savez_compressed(output,observations=np.asarray(observations,dtype=np.float32),actions=np.asarray(actions,dtype=np.float32))
    atomic_json(output.with_suffix('.json'),dict(source_checkpoint=str(Path(source).resolve()),axis=axis,
        implementation_sha256=implementation_digest(),goal_degrees=30.,
        note='Successful 30-degree feedback rollouts only. Not a 90-degree demonstration or final evaluation.',
        checks=checked,sha256=digest(output)))
    return output


def read(path):
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
    if meta['implementation_sha256']!=implementation_digest() or meta['sha256']!=digest(path):raise ValueError('Retention data mismatch')
    if not all(r['success'] and r['stable_seconds']>=1.-1e-9 for r in meta['checks']):raise ValueError('Failed episode in retention set')
    if meta.get('goal_degrees')==90. and meta.get('neural_policy_from_initial_grasp') is False and not meta.get('saved_actions_native_verified'):
        raise ValueError('Hybrid90deg data require native checks of the exactly saved float32 commands')
    with np.load(path,allow_pickle=False) as data:return data['observations'].copy(),data['actions'].copy()
