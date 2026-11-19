"""Retest archived target plans through the current bounded7-action interface."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,time
import numpy as np
from experiments.cube_quarter_turn.action_mapping_candidate import full_range_targets
from .env import AdaptiveEnv
from .config import Profile
from .storage import atomic_json,digest,implementation_digest


def probe(goal):
    root=Path(goal);repository=Path(__file__).resolve().parents[2]
    state=json.loads((root/'controller_state.json').read_text());output=root/'legacy_plan_probe';output.mkdir(exist_ok=True)
    deadline=min(time.time()+300.,datetime.fromisoformat(state['deadline_at']).timestamp());records=[];trajectories={}
    for axis in ('x','y'):
        old=repository/'experiments/cube_quarter_turn_xy/artifacts/xy_two_workers_v1'/axis/'planner'
        source_meta=json.loads((old/'best_candidate.json').read_text())
        with np.load(source_meta['reference']) as data:old_default=data['ctrl'].copy()
        with np.load(old/'best_candidate.npz') as data:old_actions=data['actions'].copy()
        profile=Profile(**state['axes'][axis]['profile'])
        for slowdown,intro in [(1,0),(1,10),(2,0),(2,10)]:
            for dt in (.001,.002):
                if time.time()>=deadline:break
                env=AdaptiveEnv(root/'sources/x/reference.npz',axis,physical_dt=dt,target_degrees=90.,profile=profile)
                try:
                    obs,info=env.reset(seed=0);actual=[];observations=[];interrupted=False
                    old_targets=[full_range_targets(a,old_default,env.ids['ctrl_lower'],env.ids['ctrl_upper']) for a in old_actions[5:]]
                    targets=[old_default.copy() for _ in range(intro)]
                    targets.extend(np.repeat(np.asarray(old_targets),slowdown,axis=0))
                    plan_index=0
                    for step in range(250):
                        if time.time()>=deadline:interrupted=True;break
                        if env.warming:action=np.zeros(7)
                        else:
                            desired=targets[plan_index] if plan_index<len(targets) else env.default
                            plan_index+=1
                            span=profile.action_fraction*(env.ids['ctrl_upper']-env.ids['ctrl_lower'])
                            action=np.clip((desired-env.data.ctrl)/span,-1.,1.)
                        observations.append(obs.copy());actual.append(action.copy())
                        obs,_,done,_,info=env.step(action)
                        if done:break
                    record=dict(axis=axis,dt=dt,slowdown=slowdown,intro_controls=intro,seed=0,
                        success=bool(info['success'] and not interrupted),interrupted=interrupted,
                        archived_plan=str(old/'best_candidate.npz'),archived_grip_not_used_as_reset=True,
                        controller='archived targets with bounded increments, then original grip targets; not trained policy',final=info)
                    records.append(record);trajectories[(axis,slowdown,intro,dt)]=(observations,actual)
                    print(json.dumps({k:record[k] for k in ['axis','dt','slowdown','intro_controls','success']}|
                        {k:info[k] for k in ['failure_reason','max_progress_degrees','max_stable_seconds']},ensure_ascii=False),flush=True)
                finally:env.close()
    atomic_json(output/'report.json',dict(observed_at=datetime.now(timezone.utc).isoformat(),
        scope='Current original initial grasp and strict task criteria; archived hand/cube initial states never substituted',records=records))
    for axis in ('x','y'):
        for slowdown,intro in [(1,0),(1,10),(2,0),(2,10)]:
            cases=[r for r in records if r['axis']==axis and r['slowdown']==slowdown and r['intro_controls']==intro]
            if len(cases)!=2 or not all(r['success'] for r in cases):continue
            observations=[];actions=[]
            for case in cases:
                x,a=trajectories[(axis,slowdown,intro,case['dt'])];observations.extend(x);actions.extend(a)
            path=output/f'qualified_{axis}_slow{slowdown}_intro{intro}.npz'
            np.savez_compressed(path,observations=np.asarray(observations,dtype=np.float32),actions=np.asarray(actions,dtype=np.float32))
            atomic_json(path.with_suffix('.json'),dict(axis=axis,goal_degrees=90.,implementation_sha256=implementation_digest(),
                sha256=digest(path),controller=cases[0]['controller'],neural_policy_from_initial_grasp=False,
                actual_trajectory_from_original_initial_grasp=True,
                checks=[dict(seed=0,dt=c['dt'],success=True,stable_seconds=c['final']['stable_seconds']) for c in cases],
                note='One archived-target controller replay on one seed, not a trained-policy or100-case result.'))
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--goal',required=True);args=p.parse_args();probe(args.goal)
