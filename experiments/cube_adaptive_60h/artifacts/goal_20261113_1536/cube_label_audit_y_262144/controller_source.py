"""Independent input-label canonicalization diagnostic for the symmetric cube.

Only the three observed quaternion blocks are right-multiplied by the same
cube symmetry. The original world, goal monitor, signed turn, contacts, forces,
and physics are never changed. This is not enabled in the active PPO jobs.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,time
import numpy as np
import torch
from experiments.cube_quarter_turn.task import SYMMETRIES,matrix_quaternion,orientation_error
from .config import Profile
from .env import AdaptiveEnv
from .storage import load,atomic_json,digest

SYMMETRY_QUATERNIONS=np.stack([matrix_quaternion(s) for s in SYMMETRIES])
QUATERNION_BLOCKS=(71,84,94)


def multiply(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    scalar=a[...,0]*b[...,0]-np.sum(a[...,1:]*b[...,1:],axis=-1)
    vector=a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+np.cross(a[...,1:],b[...,1:])
    return np.concatenate([scalar[...,None],vector],axis=-1)


def canonical(q):
    q=np.asarray(q,dtype=np.float64);length=np.linalg.norm(q,axis=-1,keepdims=True)
    if not np.isfinite(q).all() or np.any(length<1e-10):raise ValueError('Invalid observed quaternion')
    q=q/length;return q*np.where(q[...,:1]<0.,-1.,1.)


def initial_quaternion(observation):
    obs=np.asarray(observation)
    if obs.shape[-1]!=143:raise ValueError('Expected frozen143-input semantics')
    angle=obs[...,133]*np.pi/2.
    direction=obs[...,101:104].astype(np.float64)
    direction=direction/np.linalg.norm(direction,axis=-1,keepdims=True)
    inverse_turn=np.concatenate([np.cos(angle[...,None]/2.),-direction*np.sin(angle[...,None]/2.)],axis=-1)
    return canonical(multiply(inverse_turn,obs[...,94:98]))


def transform(observation,anchor):
    obs=np.asarray(observation);initial=initial_quaternion(obs);anchor=canonical(anchor)
    candidates=canonical(multiply(initial[...,None,:],SYMMETRY_QUATERNIONS))
    indices=np.argmax(np.abs(np.sum(candidates*anchor,axis=-1)),axis=-1)
    symmetry=SYMMETRY_QUATERNIONS[indices]
    result=obs.copy()
    for start in QUATERNION_BLOCKS:
        result[...,start:start+4]=canonical(multiply(obs[...,start:start+4],symmetry)).astype(obs.dtype)
    return result


def verify(env,obs,anchor,model):
    b=env.ids['cube_body'];inertia=env.model.body_inertia[b]
    np.testing.assert_allclose(inertia,np.full(3,inertia.mean()),rtol=0.,atol=1e-12)
    np.testing.assert_array_equal(env.model.body_ipos[b],np.zeros(3))
    geoms=[i for i in range(env.model.ngeom) if env.model.geom_bodyid[i]==b and
        (env.model.geom_contype[i] or env.model.geom_conaffinity[i])]
    assert geoms==[env.ids['cube_geom']]
    np.testing.assert_allclose(env.model.geom_size[geoms[0]],np.full(3,.0285),rtol=0.,atol=1e-12)
    np.testing.assert_allclose(obs[94:98],env.monitor.goal,atol=1e-7)
    assert obs[133]==1.
    target=transform(obs,anchor);variants=[];max_pose_error_difference=0.
    before=orientation_error(obs[84:88],obs[94:98])[1]
    for symmetry in SYMMETRY_QUATERNIONS:
        label=obs.copy()
        for start in QUATERNION_BLOCKS:
            label[start:start+4]=canonical(multiply(label[start:start+4],symmetry))
        variants.append(transform(label,anchor))
        max_pose_error_difference=max(max_pose_error_difference,abs(orientation_error(label[84:88],label[94:98])[1]-before))
    variants=np.stack(variants)
    np.testing.assert_allclose(variants,np.broadcast_to(target,variants.shape),rtol=0.,atol=2e-6)
    reference=model.predict(target,deterministic=True)[0]
    predicted=model.predict(variants,deterministic=True)[0]
    max_action_difference=float(np.max(np.abs(predicted-reference)))
    if max_action_difference>1e-5:raise AssertionError('Canonical action differs across cube labels')
    assert max_pose_error_difference<1e-6
    return dict(cube_mass=float(env.model.body_mass[b]),cube_inertia=inertia.tolist(),cube_com=env.model.body_ipos[b].tolist(),
        collision_geoms=geoms,all24_canonical_observations_agree=True,max_canonical_observation_difference=float(np.max(np.abs(variants-target))),
        max_canonical_action_difference=max_action_difference,max_goal_pose_error_difference=max_pose_error_difference)


def audit(goal,axis,checkpoint,anchor_seed=1000013,episodes=6,seed=5000000,seconds=600.):
    torch.set_num_threads(1);root=Path(goal);checkpoint=Path(checkpoint).resolve()
    model,metadata=load(checkpoint);profile=Profile(**metadata['profile'])
    output=root/f'cube_label_audit_{axis}_{metadata["training_steps"]}';output.mkdir(exist_ok=False)
    (output/'controller_source.py').write_bytes(Path(__file__).read_bytes())
    deadline=min(time.time()+seconds,datetime.fromisoformat(json.loads((root/'goal.json').read_text())['deadline_at']).timestamp())
    env=AdaptiveEnv(root/'sources/x/reference.npz',axis,physical_dt=.001,target_degrees=90.,profile=profile)
    try:
        obs,info=env.reset(seed=anchor_seed)
        for _ in range(5):obs,_,done,_,info=env.step(np.zeros(7,dtype=np.float32))
        if done:raise RuntimeError('Illegal anchor initial grasp')
        anchor=initial_quaternion(obs);checks=verify(env,obs,anchor,model)
    finally:env.close()
    atomic_json(output/'source.json',dict(checkpoint=str(checkpoint),checkpoint_metadata_sha256=digest(checkpoint/'metadata.json'),
        anchor_seed=anchor_seed,anchor_quaternion=anchor.tolist(),checks=checks,controller_sha256=digest(Path(__file__)),
        physical_and_success_monitor_changes=False,active_training_changed=False))
    cases=[]
    for index in range(episodes):
        for dt in (.001,.002):
            for mode in ('raw','canonical'):
                if time.time()>=deadline:break
                env=AdaptiveEnv(root/'sources/x/reference.npz',axis,physical_dt=dt,target_degrees=90.,profile=profile)
                try:
                    obs,info=env.reset(seed=seed+index);interrupted=False
                    for _ in range(250):
                        if time.time()>=deadline:interrupted=True;break
                        before=time.process_time();inputs=transform(obs,anchor) if mode=='canonical' else obs
                        action=model.predict(inputs,deterministic=True)[0]
                        obs,_,done,_,info=env.step(action)
                        if done:break
                        time.sleep(min(max(0.,deadline-time.time()),4.*(time.process_time()-before)))
                    cases.append(dict(mode=mode,seed=seed+index,dt=dt,interrupted=interrupted,final=info))
                    atomic_json(output/'status.json',dict(status='evaluating',completed_cases=len(cases),planned_cases=episodes*4,latest=cases[-1]))
                finally:env.close()
    groups={}
    for mode in ('raw','canonical'):
        chosen=[r for r in cases if r['mode']==mode]
        groups[mode]=dict(trials=len(chosen),complete=len(chosen)==episodes*2 and not any(c['interrupted'] for c in chosen),
            successes=sum(c['final']['success'] and not c['interrupted'] for c in chosen),
            maximum_angle=max((c['final']['max_progress_degrees'] for c in chosen),default=0.),
            mean_best_error=float(np.mean([c['final']['best_angle_error_degrees'] for c in chosen])) if chosen else None,
            max_goal_hold=max((c['final']['max_stable_seconds'] for c in chosen),default=0.))
    result=dict(status='diagnosis_complete',observed_at=datetime.now(timezone.utc).isoformat(),axis=axis,
        scope='Frozen actor, original initial grasp, paired raw versus observation-label preprocessing; not active policy scores',
        checkpoint=str(checkpoint),anchor_seed=anchor_seed,anchor_quaternion=anchor.tolist(),checks=checks,groups=groups,cases=cases)
    atomic_json(output/'result.json',result);atomic_json(output/'status.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--goal',required=True);p.add_argument('--axis',choices=('x','y'),required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--anchor-seed',type=int,default=1000013)
    p.add_argument('--episodes',type=int,default=6);p.add_argument('--seed',type=int,default=5000000)
    p.add_argument('--seconds',type=float,default=600.);args=p.parse_args()
    result=audit(args.goal,args.axis,args.checkpoint,args.anchor_seed,args.episodes,args.seed,args.seconds)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False))
