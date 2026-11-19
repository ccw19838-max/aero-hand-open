"""Qualify the exact float32 saved commands with the original native monitor."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,time
import numpy as np
from .env import AdaptiveEnv
from .config import Profile
from .storage import atomic_json,digest,implementation_digest


def qualify(goal,teacher):
    root=Path(goal);teacher=Path(teacher);source=json.loads((teacher/'result.json').read_text())
    metadata=json.loads((Path(source['source_checkpoint'])/'metadata.json').read_text());profile=Profile(**metadata['profile'])
    deadline=datetime.fromisoformat(json.loads((root/'goal.json').read_text())['deadline_at']).timestamp()
    with np.load(teacher/'actual_trajectory.npz',allow_pickle=False) as data:commands=data['actions'].astype(np.float32)
    if commands.ndim!=2 or commands.shape[1]!=7 or not np.isfinite(commands).all() or np.max(np.abs(commands))>1.:
        raise ValueError('Invalid saved teacher commands')
    cases=[];all_observations=[];all_actions=[]
    for dt in (.001,.002):
        env=AdaptiveEnv(root/'sources/x/reference.npz',source['axis'],physical_dt=dt,target_degrees=90.,profile=profile)
        try:
            obs,info=env.reset(seed=source['seed']);observations=[];actions=[];interrupted=False
            for action in commands:
                if time.time()>=deadline:interrupted=True;break
                observations.append(obs.copy());actions.append(action.copy())
                obs,_,done,_,info=env.step(action)
                if done:break
            cases.append(dict(dt=dt,seed=source['seed'],success=bool(info['success'] and not interrupted),interrupted=interrupted,
                stable_seconds=info['stable_seconds'],final=info))
            all_observations.extend(observations);all_actions.extend(actions)
        finally:env.close()
    report=dict(observed_at=datetime.now(timezone.utc).isoformat(),saved_action_dtype='float32',
        saved_actions_native_verified=all(c['success'] for c in cases),checks=cases,
        source_commands_sha256=digest(teacher/'actual_trajectory.npz'),
        neural_policy_from_initial_grasp=False,actual_trajectory_from_original_initial_grasp=True)
    atomic_json(teacher/'saved_commands_validation.json',report)
    # Earlier raw-double diagnostic output must not be silently treated as an
    # already qualified float32 demonstration.
    for suffix in ('.npz','.json'):
        prior=teacher/('qualified_hybrid90'+suffix)
        if prior.exists():prior.rename(teacher/('raw_precision_candidate'+suffix))
    if report['saved_actions_native_verified']:
        path=teacher/'verified_teacher90.npz'
        np.savez_compressed(path,observations=np.asarray(all_observations,dtype=np.float32),actions=np.asarray(all_actions,dtype=np.float32))
        atomic_json(path.with_suffix('.json'),dict(axis=source['axis'],goal_degrees=90.,
            controller=source['controller'],neural_policy_from_initial_grasp=False,
            actual_trajectory_from_original_initial_grasp=True,saved_actions_native_verified=True,
            saved_action_dtype='float32',source_checkpoint=source['source_checkpoint'],
            implementation_sha256=implementation_digest(),sha256=digest(path),
            checks=[{k:c[k] for k in ('dt','seed','success','stable_seconds')} for c in cases],
            note='Two native replays of exactly saved commands on one seed; not standalone neural policy or100-case completion.'))
    return {k:v for k,v in report.items() if k!='checks'}|{'checks':[{k:c[k] for k in ('dt','seed','success','stable_seconds')} for c in cases]}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--goal',required=True);p.add_argument('--teacher',required=True)
    args=p.parse_args();print(json.dumps(qualify(args.goal,args.teacher),ensure_ascii=False))
