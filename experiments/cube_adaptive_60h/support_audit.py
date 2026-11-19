"""Read-only replay of real forbidden contacts; never alters collision rules."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,json
import numpy as np
import mujoco
import torch
from .env import AdaptiveEnv
from .config import Profile
from .storage import load,atomic_json


def audit(goal):
    root=Path(goal);state=json.loads((root/'controller_state.json').read_text())
    output=root/'support_audit';output.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    records=[]
    for axis,target,key,seeds in [('x',10.,'10',(1000000,1000013)),('y',30.,'30',(1000000,1000001))]:
        entry=state['axes'][axis];checkpoint=entry['best_by_target'][key]['path']
        policy,_=load(checkpoint);profile=Profile(**entry['profile'])
        env=AdaptiveEnv(root/'sources/x/reference.npz',axis,physical_dt=.002,target_degrees=target,profile=profile)
        try:
            for seed in seeds:
                obs,info=env.reset(seed=seed);origin=env.data.xpos[env.ids['cube_body']].copy()
                palm=env.model.body('palm').id;frame=env.data.xmat[palm].reshape(3,3).copy()
                poses=[];qs=[];forces=[];rewards=[];bad=[]
                for _ in range(250):
                    action=policy.predict(obs,deterministic=True)[0]
                    obs,reward,done,_,info=env.step(action)
                    poses.append(env.data.xpos[env.ids['cube_body']].copy());qs.append(env.data.qpos.copy())
                    forces.append(env.contacts['finger_forces'].copy());rewards.append(reward)
                    for index,c in enumerate(env.data.contact):
                        a,b=int(c.geom1),int(c.geom2);cube=env.ids['cube_geom']
                        other=b if a==cube else a if b==cube else None
                        if other is None or not env.ids['geom_forbidden'][other]:continue
                        force=np.zeros(6);mujoco.mj_contactForce(env.model,env.data,index,force)
                        bad.append(dict(time=float(env.data.time),geom_id=other,geom=env.model.geom(other).name,
                            body=env.model.body(int(env.model.geom_bodyid[other])).name,
                            normal_force=float(force[0]),distance=float(c.dist),position=c.pos.tolist()))
                    if done:break
                label=f'{axis}_seed{seed}'
                np.savez_compressed(output/(label+'.npz'),qpos=np.asarray(qs),cube_position=np.asarray(poses),
                    finger_forces=np.asarray(forces),reward=np.asarray(rewards))
                report=dict(axis=axis,target=target,seed=seed,dt=.002,checkpoint=checkpoint,
                    success=info['success'],failure=info['failure_reason'],signed_angle=info['signed_progress_degrees'],
                    max_angle=info['max_progress_degrees'],hold_seconds=info['max_stable_seconds'],
                    final_cube_translation_in_palm_axes=(frame.T@(poses[-1]-origin)).tolist(),
                    bad_contacts=bad,elapsed=info['elapsed_seconds'])
                records.append(report);print(json.dumps(report,ensure_ascii=False),flush=True)
        finally:env.close()
    result=dict(observed_at=datetime.now(timezone.utc).isoformat(),scope='Four selected deterministic replays, not a success-rate estimate',
        collision_classification='Only tetheria_mount and floor are forbidden. Palm contacts remain allowed.',records=records)
    atomic_json(output/'report.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--goal',required=True);args=p.parse_args();audit(args.goal)
