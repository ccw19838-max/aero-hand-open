"""Bounded online shooting diagnostic. Actual trajectory uses the unchanged env."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,time,json
import mujoco
import numpy as np
import torch
from experiments.cube_envelop.model import native_contacts,native_cube_velocity
from experiments.cube_quarter_turn.task import orientation_error
from .env import AdaptiveEnv
from .config import Profile
from .storage import load,atomic_json
from .shaping import colliding_forbidden_geoms


def run(goal,axis,target,seconds=120.,seed=0,population=18,generations=2):
    torch.set_num_threads(1);root=Path(goal);state=json.loads((root/'controller_state.json').read_text())
    entry=state['axes'][axis];key=str(int(10 if axis=='x' else 30))
    actor,_=load(entry['best_by_target'][key]['path']);profile=Profile(**entry['profile'])
    env=AdaptiveEnv(root/'sources/x/reference.npz',axis,physical_dt=.001,target_degrees=target,profile=profile)
    output=root/'mpc_diagnosis'/f'{axis}{int(target)}_seed{seed}';output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed);deadline=min(time.time()+seconds,datetime.fromisoformat(state['deadline_at']).timestamp())
    spec=mujoco.mjtState.mjSTATE_INTEGRATION;physics=np.empty(mujoco.mj_stateSize(env.model,spec));probe=mujoco.MjData(env.model)
    span=env.ids['ctrl_upper']-env.ids['ctrl_lower'];bad_geoms=colliding_forbidden_geoms(env)
    lower=np.array([.2,.1,.1,.1,.1]);upper=np.array([20.,12.,12.,12.,12.]);mean=np.zeros(7)
    actions=[];observations=[];trace=[];qs=[];controls=[];predictions=0;interrupted=False
    try:
        obs,info=env.reset(seed=seed);origin=env.initial_position.copy()
        def candidate_cost(action):
            nonlocal predictions
            predictions+=1;mujoco.mj_setState(env.model,probe,physics,spec);mujoco.mj_forward(env.model,probe)
            values=[]
            for _ in range(3):
                probe.ctrl[:]=np.clip(probe.ctrl+profile.action_fraction*span*action,env.ids['ctrl_lower'],env.ids['ctrl_upper'])
                mujoco.mj_step(env.model,probe,nstep=round(.05/env.physical_dt));mujoco.mj_forward(env.model,probe)
                contact=native_contacts(env.model,probe,env.ids);forces=contact['finger_forces']
                # Counterfactual screening is approximate; every committed
                # action is checked by the original substep task monitor.
                if contact['forbidden_support']:return 1e4
                _,error=orientation_error(probe.qpos[19:23],env.monitor.goal)
                linear,angular=native_cube_velocity(env.model,probe,env.ids)
                displacement=probe.xpos[env.ids['cube_body']]-origin
                cost=30.*(error/max(np.deg2rad(target),np.deg2rad(10.)))**2
                cost+=5.*float(displacement@displacement)/.03**2
                deficit=np.clip((lower-forces)/lower,0.,1.)
                cost+=(3. if error<np.deg2rad(20.) else .2)*float(np.sum(deficit**2))
                support=forces[0]>=lower[0] and np.count_nonzero(forces[1:]>=lower[1:])>=2
                if not support:cost+=10.
                cost+=float(np.mean((np.maximum(forces-upper,0.)/upper)**2))
                if error<np.deg2rad(10.):cost+=.25*(float(linear@linear)/.03**2+float(angular@angular)/.5**2)
                cost+=10.*max(0.,contact['max_penetration']/.001-1.)**2
                gap=min((float(mujoco.mj_geomDistance(env.model,probe,env.ids['cube_geom'],int(g),.1,None)) for g in bad_geoms),default=.1)
                cost+=3.*float(np.clip((.005-gap)/.005,0.,2.)**2)
                values.append(cost)
            return values[-1]+.2*float(np.mean(values))+.02*float(np.mean(action**2))
        for step in range(250):
            if time.time()>=deadline:interrupted=True;break
            if env.warming:action=np.zeros(7)
            else:
                mujoco.mj_getState(env.model,env.data,physics,spec)
                center=mean.copy();sigma=np.full(7,.35);best=None
                for generation in range(generations):
                    candidates=np.clip(rng.normal(center,sigma,(population,7)),-1.,1.)
                    candidates[0]=center;candidates[1]=np.zeros(7);candidates[2]=actor.predict(obs,deterministic=True)[0]
                    scored=[]
                    for candidate in candidates:
                        if time.time()>=deadline:interrupted=True;break
                        value=candidate_cost(candidate);scored.append((value,candidate.copy()))
                        if best is None or value<best[0]:best=(value,candidate.copy())
                    if interrupted:break
                    elites=np.array([a for _,a in sorted(scored,key=lambda p:p[0])[:max(3,population//4)]])
                    center=elites.mean(0);sigma=np.maximum(.1,elites.std(0))
                if interrupted:break
                action=best[1];mean=.7*action
            observations.append(obs.copy());actions.append(action.copy())
            obs,_,done,_,info=env.step(action);trace.append(info);qs.append(env.data.qpos.copy());controls.append(env.data.ctrl.copy())
            if step%10==0:
                atomic_json(output/'status.json',dict(status='diagnosing',step=step,predictions=predictions,
                    elapsed=info['elapsed_seconds'],signed_angle=info['signed_progress_degrees'],error=info['angle_error_degrees'],five_finger=info['five_finger_contact']))
            if done:break
        result=dict(observed_at=datetime.now(timezone.utc).isoformat(),controller='privileged native shooting; not trained policy',
            axis=axis,target_degrees=target,seed=seed,dt=.001,from_original_initial_grasp=True,
            success=bool(info['success'] and not interrupted),interrupted=interrupted,predictions=predictions,actual_controls=len(actions),final=info,
            general_success_rate_not_measured=True,counterfactual_screening_does_not_replace_original_monitor=True)
        np.savez_compressed(output/'actual_trajectory.npz',observations=np.asarray(observations),actions=np.asarray(actions),qpos=np.asarray(qs),ctrl=np.asarray(controls))
        atomic_json(output/'trace.json',trace);atomic_json(output/'result.json',result)
        atomic_json(output/'status.json',{'status':'completed_diagnosis',**result});return result
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--goal',required=True);p.add_argument('--axis',choices=('x','y'),required=True)
    p.add_argument('--target',type=float,required=True);p.add_argument('--seconds',type=float,default=120.);p.add_argument('--seed',type=int,default=0)
    args=p.parse_args();print(json.dumps(run(args.goal,args.axis,args.target,args.seconds,args.seed),ensure_ascii=False))
