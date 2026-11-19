"""Low-duty neural-prefix/model-predictive teacher search; separate from policy scores."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,json,time,hashlib
import numpy as np
import mujoco
import torch
from experiments.cube_envelop.model import native_contacts,native_cube_velocity
from experiments.cube_quarter_turn.task import orientation_error
from .env import AdaptiveEnv
from .fast_contacts import FastContactEnv
from .config import Profile
from .storage import load,atomic_json,digest,implementation_digest
from .shaping import colliding_forbidden_geoms


def search(goal,axis='y',seed=0,seconds=600.,checkpoint_override=None,variant='latched_capture_v3',actual_dt=.001):
    if variant not in ('latched_capture_v3','distance_capture_v4','long_horizon_v5'):raise ValueError('Unknown diagnostic variant')
    if actual_dt not in (.001,.002):raise ValueError('Only original validated integrators')
    distance_capture=variant=='distance_capture_v4'
    long_horizon=variant=='long_horizon_v5';horizon=12 if long_horizon else 6
    torch.set_num_threads(1);root=Path(goal);state=json.loads((root/'controller_state.json').read_text())
    checkpoint=Path(checkpoint_override or state['axes'][axis]['best_by_target']['90']['path']).resolve();actor,metadata=load(checkpoint)
    profile=Profile(**metadata['profile']);output=root/f'hybrid_teacher_{axis}90_seed{seed}_{variant}'
    output.mkdir(exist_ok=False)
    (output/'controller_source.py').write_bytes(Path(__file__).read_bytes())
    deadline=min(time.time()+seconds,datetime.fromisoformat(state['deadline_at']).timestamp())
    env=FastContactEnv(root/'sources/x/reference.npz',axis,physical_dt=actual_dt,target_degrees=90.,profile=profile)
    preview_dt=actual_dt if long_horizon else .002
    preview_model=copy.copy(env.model);preview_model.opt.timestep=preview_dt
    assert preview_model._address!=env.model._address
    preview=mujoco.MjData(preview_model);spec=mujoco.mjtState.mjSTATE_INTEGRATION
    physics=np.empty(mujoco.mj_stateSize(env.model,spec));rng=np.random.default_rng(seed+9401)
    lower=np.array([.2,.1,.1,.1,.1]);upper=np.array([20.,12.,12.,12.,12.])
    span=env.ids['ctrl_upper']-env.ids['ctrl_lower'];bad_geoms=colliding_forbidden_geoms(env)
    threshold=np.deg2rad(25. if axis=='y' or long_horizon else 10.);active=False;capturing=False;center=np.zeros(14)
    approach=0.
    finger_geoms=[[i for i in range(env.model.ngeom) if env.ids['geom_finger'][i]==finger and
        (env.model.geom_contype[i] or env.model.geom_conaffinity[i])] for finger in range(5)]
    preview_contact=np.zeros(8,dtype=np.float64);contact_pointer=preview_contact.ctypes.data
    source_digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    observations=[];actions=[];trace=[];predictions=0;neural_controls=0;mpc_controls=0;interrupted=False
    try:
        obs,info=env.reset(seed=seed);origin=env.initial_position.copy()
        def score(parameters):
            nonlocal predictions
            predictions+=1;mujoco.mj_setState(preview_model,preview,physics,spec);mujoco.mj_forward(preview_model,preview)
            values=[];contact_missing=env.monitor.insufficient_contact_seconds
            penetration_time=env.monitor.penetration_time;region_time=env.monitor.region_time
            unsupported_time=env.monitor.unsupported_time;predicted_elapsed=0.;predicted_hold=env.monitor.stable
            for index in range(horizon):
                action=parameters[:7] if index<horizon//2 else parameters[7:]
                preview.ctrl[:]=np.clip(preview.ctrl+profile.action_fraction*span*action,env.ids['ctrl_lower'],env.ids['ctrl_upper'])
                # Forecast the same10ms support/penetration/workspace clocks
                # as the actual task, including time already lost before MPC.
                for _ in range(5):
                    mujoco.mj_step(preview_model,preview,nstep=round(.01/preview_dt));mujoco.mj_forward(preview_model,preview);predicted_elapsed+=.01
                    if not np.isfinite(preview.qpos).all() or not np.isfinite(preview.qvel).all():return 1e6
                    env._contact_library.cube_contacts(preview_model._address,preview._address,env.ids['cube_geom'],
                        env._finger_pointer,env._palm_pointer,env._forbidden_pointer,env.config.contact_force,contact_pointer)
                    force=preview_contact[:5]
                    supported=force[0]>=lower[0] and np.count_nonzero(force[1:]>=lower[1:])>=2
                    contact_missing=0. if supported else contact_missing+.01
                    penetration_time=penetration_time+.01 if preview_contact[6]>env.config.max_penetration else 0.
                    distance=float(np.linalg.norm(preview.xpos[env.ids['cube_body']]-origin))
                    region_time=region_time+.01 if distance>env.config.max_displacement else 0.
                    hand_force=float(sum(force)+preview_contact[5])
                    unsupported_time=unsupported_time+.01 if hand_force<env.config.contact_force else 0.
                    failure=(bool(preview_contact[7]) or contact_missing>env.config.contact_grace_seconds+1e-10
                        or penetration_time+1e-10>=env.config.violation_seconds
                        or region_time+1e-10>=env.config.violation_seconds
                        or unsupported_time+1e-10>=env.config.max_unsupported_seconds)
                    if failure:return 1e4+100.*max(0.,horizon*.05-predicted_elapsed)
                    if distance_capture and approach>0.:
                        _,hold_error=orientation_error(preview.qpos[19:23],env.monitor.goal)
                        hold_linear,hold_angular=native_cube_velocity(preview_model,preview,env.ids)
                        five=bool(np.all(force>=lower) and np.all(force<=upper))
                        valid=bool(preview_contact[6]<=env.config.max_penetration and distance<=env.config.max_displacement)
                        slow=bool(np.linalg.norm(hold_linear)<.03 and np.linalg.norm(hold_angular)<.5)
                        # Planning preference only. Actual success remains the
                        # inherited monitor, including signed progress and1s.
                        predicted_hold=predicted_hold+.01 if five and valid and slow and hold_error<=np.deg2rad(5.) else 0.
                _,error=orientation_error(preview.qpos[19:23],env.monitor.goal)
                linear,angular=native_cube_velocity(preview_model,preview,env.ids)
                delta=preview.xpos[env.ids['cube_body']]-origin
                center_cost=2.*float(delta@delta)/.02**2 if long_horizon else .25*float(delta@delta)/.05**2
                value=20.*(error/(np.pi/2))**2+center_cost
                deficit=np.clip((lower-force)/lower,0.,1.)
                # The phase is latched from the actual state, not from each
                # candidate's predicted error. A cost jump at predicted20deg
                # otherwise makes staying just outside that boundary cheaper.
                force_weight=.2+2.8*approach if distance_capture else (3. if capturing else .2)
                value+=force_weight*float(np.sum(deficit**2))
                if distance_capture and approach>0.:
                    # Real collision-geometric gaps give a gradient while a
                    # missing finger still reads exactly0N. Visual geoms are
                    # excluded; distances never change contacts or physics.
                    gaps=np.array([min((float(mujoco.mj_geomDistance(preview_model,preview,
                        env.ids['cube_geom'],int(g),.1,None)) for g in geoms),default=.1) for geoms in finger_geoms])
                    value+=3.*approach*float(np.sum(np.clip((gaps+.00005)/.01,0.,2.)**2))
                if not (force[0]>=lower[0] and np.count_nonzero(force[1:]>=lower[1:])>=2):value+=5.
                value+=float(np.mean((np.maximum(force-upper,0.)/upper)**2))
                value+=10.*max(0.,preview_contact[6]/.001-1.)**2
                gap=min((float(mujoco.mj_geomDistance(preview_model,preview,env.ids['cube_geom'],int(g),.1,None)) for g in bad_geoms),default=.1)
                value+=2.*float(np.clip((.005-gap)/.005,0.,2.)**2)
                if capturing:
                    value+=.1*(float(linear@linear)/.03**2+float(angular@angular)/.5**2)
                    value+=.25*float(np.mean(((preview.ctrl-env.default)/span)**2))
                    value+=5.*max(0.,error/np.deg2rad(5.)-1.)**2
                values.append(value)
            hold_bonus=10.*min(predicted_hold,1.) if distance_capture else 0.
            return values[-1]+.1*float(np.mean(values))+.01*float(np.mean(parameters**2))-hold_bonus
        while len(actions)<250:
            if time.time()>=deadline:interrupted=True;break
            cpu_before=time.process_time()
            if not active and not env.warming and (env.monitor.progress>=threshold or
                    (env.monitor.progress>=threshold*.6 and env.monitor.insufficient_contact_seconds>.02)):active=True
            if not env.warming and env.monitor.error<=np.deg2rad(5.) and abs(env.monitor.progress-np.pi/2)<=np.deg2rad(5.):capturing=True
            # Held fixed while candidates are ranked, so no candidate can
            # avoid a phase-dependent cost by stopping just outside its edge.
            approach=1. if capturing else float(np.clip((np.deg2rad(25.)-env.monitor.error)/np.deg2rad(20.),0.,1.))
            if env.warming:action=np.zeros(7);commit=1
            elif not active:action=actor.predict(obs,deterministic=True)[0];commit=1;neural_controls+=1
            elif env.monitor.stable>=.2:action=np.zeros(7);commit=1;mpc_controls+=1
            else:
                mujoco.mj_getState(env.model,env.data,physics,spec)
                sigma=np.full(14,.18 if distance_capture and capturing else (.4 if long_horizon else .3));best=None
                prior=actor.predict(obs,deterministic=True)[0]
                for generation in range(2):
                    candidates=np.clip(rng.normal(center,sigma,(12,14)),-1.,1.)
                    candidates[0]=center;candidates[1]=np.zeros(14);candidates[2]=np.r_[prior,prior]
                    candidates[3::4,7:]=-.5*candidates[3::4,:7]
                    candidates[4::4,7:]=0.;ranked=[]
                    if distance_capture and approach>0.:
                        home=np.clip((env.default-env.data.ctrl)/(profile.action_fraction*span*5.),-1.,1.)
                        fingers_only=home.copy();fingers_only[4:]=0.
                        close_missing=np.zeros(7)
                        for motor,finger in enumerate(range(1,5)):
                            if info['finger_forces'][finger]<lower[finger]:close_missing[motor]=-.7
                        candidates[5]=np.r_[home,home]
                        candidates[6]=np.r_[fingers_only,fingers_only]
                        candidates[7]=np.r_[close_missing,close_missing]
                    for candidate in candidates:
                        if time.time()>=deadline:interrupted=True;break
                        cost=score(candidate);ranked.append((cost,candidate.copy()))
                        if best is None or cost<best[0]:best=(cost,candidate.copy())
                    if interrupted:break
                    elite=np.array([v for _,v in sorted(ranked,key=lambda item:item[0])[:3]])
                    center=elite.mean(0);sigma=np.maximum(.08,elite.std(0))
                if interrupted:break
                action=best[1][:7]
                center=np.r_[.5*(best[1][:7]+best[1][7:]),best[1][7:]] if long_horizon else np.r_[best[1][7:],best[1][7:]]
                commit=1 if env.monitor.insufficient_contact_seconds>.05 else (3 if long_horizon else 2);mpc_controls+=commit
            done=False
            for _ in range(commit):
                action=np.asarray(action,dtype=np.float32)
                observations.append(obs.copy());actions.append(action.copy())
                obs,_,done,_,info=env.step(action);trace.append(info)
                if done or len(actions)>=250:break
            atomic_json(output/'status.json',dict(status='searching',actual_controls=len(actions),neural_controls=neural_controls,
                mpc_controls=mpc_controls,predictions=predictions,signed_angle=info['signed_progress_degrees'],
                pose_error=info['angle_error_degrees'],hold=info['max_stable_seconds'],capturing=capturing))
            if done:break
            spent=time.process_time()-cpu_before;time.sleep(min(max(0.,deadline-time.time()),4.*spent))
        success=bool(info['success'] and not interrupted)
        record=dict(observed_at=datetime.now(timezone.utc).isoformat(),controller='neural prefix plus model-predictive correction; not standalone trained policy',
            axis=axis,target_degrees=90.,seed=seed,actual_dt=actual_dt,preview_dt=preview_dt,source_checkpoint=str(checkpoint),
            prediction_variant='constraint_clocks_'+variant,controller_source_sha256=source_digest,
            neural_policy_from_initial_grasp=False,actual_trajectory_from_original_initial_grasp=True,
            success=success,interrupted=interrupted,predictions=predictions,final=info)
        np.savez_compressed(output/'actual_trajectory.npz',observations=np.asarray(observations,dtype=np.float32),actions=np.asarray(actions,dtype=np.float32))
        atomic_json(output/'trace.json',trace)
        atomic_json(output/'result.json',record)
        if success:
            from .qualify_teacher import qualify
            record['saved_commands_validation']=qualify(root,output);atomic_json(output/'result.json',record)
        atomic_json(output/'status.json',{'status':'diagnosis_complete',**record});return record
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--goal',required=True);p.add_argument('--axis',choices=('x','y'),default='y')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--seconds',type=float,default=600.)
    p.add_argument('--checkpoint')
    p.add_argument('--variant',choices=('latched_capture_v3','distance_capture_v4','long_horizon_v5'),default='latched_capture_v3')
    p.add_argument('--actual-dt',type=float,choices=(.001,.002),default=.001)
    args=p.parse_args();print(json.dumps(search(args.goal,args.axis,args.seed,args.seconds,args.checkpoint,args.variant,args.actual_dt),ensure_ascii=False))
