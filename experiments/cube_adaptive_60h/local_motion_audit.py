"""Short real-physics pulses; local coupling evidence, not a reachability proof."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse
import numpy as np
import mujoco
from scipy.optimize import lsq_linear
from experiments.cube_envelop.model import native_contacts
from experiments.cube_quarter_turn.task import orientation_error
from .env import AdaptiveEnv
from .storage import atomic_json


def audit(goal):
    root=Path(goal);env=AdaptiveEnv(root/'sources/x/reference.npz','x',target_degrees=10.)
    try:
        env.reset(seed=0,options={'reference':True})
        for _ in range(5):env.step(np.zeros(7))
        spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(env.model,spec))
        mujoco.mj_getState(env.model,env.data,state,spec)
        position=env.data.xpos[env.ids['cube_body']].copy();quaternion=env.data.qpos[19:23].copy()
        span=env.ids['ctrl_upper']-env.ids['ctrl_lower'];horizon=.15
        def pulse(action):
            data=mujoco.MjData(env.model);mujoco.mj_setState(env.model,data,state,spec);mujoco.mj_forward(env.model,data)
            for _ in range(3):
                data.ctrl[:]=np.clip(data.ctrl+.02*span*action,env.ids['ctrl_lower'],env.ids['ctrl_upper'])
                mujoco.mj_step(env.model,data,nstep=round(.05/env.physical_dt))
            mujoco.mj_forward(env.model,data)
            rotation=orientation_error(quaternion,data.qpos[19:23])[0]
            translation=data.xpos[env.ids['cube_body']]-position
            contact=native_contacts(env.model,data,env.ids)
            return dict(twist=np.r_[translation,rotation]/horizon,translation=translation,rotation=rotation,
                finger_forces=contact['finger_forces'],penetration=contact['max_penetration'],forbidden=contact['forbidden_support'],qpos=data.qpos.copy())
        baseline=pulse(np.zeros(7));repeated=pulse(np.zeros(7))
        np.testing.assert_array_equal(baseline['qpos'],repeated['qpos'])
        epsilon=.2;columns=[];records=[]
        for motor in range(7):
            axis=np.eye(7)[motor];plus=pulse(epsilon*axis);minus=pulse(-epsilon*axis)
            columns.append((plus['twist']-minus['twist'])/(2*epsilon))
            for sign,result in [(1,plus),(-1,minus)]:
                records.append(dict(motor=motor,name=env.model.actuator(motor).name,action=sign*epsilon,
                    translation_mm=(1000*result['translation']).tolist(),rotation_degrees=np.rad2deg(result['rotation']).tolist(),
                    finger_forces=result['finger_forces'].tolist(),max_endpoint_penetration_m=result['penetration'],
                    endpoint_forbidden_contact=bool(result['forbidden'])))
        jacobian=np.array(columns).T;scale=np.r_[np.full(3,.03),np.full(3,.5)]
        matrix=jacobian/scale[:,None];singular=np.linalg.svd(matrix,compute_uv=False);candidates=[]
        for axis_index,label in enumerate('xy'):
            direction=env.basis[:,axis_index];desired=np.r_[np.zeros(3),.2*direction]
            target=(desired-baseline['twist'])/scale
            solution=lsq_linear(np.r_[matrix,.03*np.eye(7)],np.r_[target,np.zeros(7)],bounds=(-.3,.3))
            observed=pulse(solution.x)
            candidates.append(dict(axis=label,action=solution.x.tolist(),
                requested_angular_velocity_rad_s=(.2*direction).tolist(),
                predicted_twist=(baseline['twist']+jacobian@solution.x).tolist(),observed_twist=observed['twist'].tolist(),
                observed_translation_mm=(1000*observed['translation']).tolist(),
                observed_rotation_degrees=np.rad2deg(observed['rotation']).tolist(),
                endpoint_finger_forces=observed['finger_forces'].tolist(),
                endpoint_forbidden_contact=bool(observed['forbidden'])))
        report=dict(observed_at=datetime.now(timezone.utc).isoformat(),seed=0,dt=env.physical_dt,horizon_seconds=horizon,
            scope='Initial-grasp0.15s local finite-difference pulses, not full-turn attempts, success-rate tests or a global reachability proof',
            physics_restore_repeat_identical=True,jacobian_world_twist_per_normalized_action=jacobian.tolist(),
            singular_values_with_speed_scaling=singular.tolist(),effective_rank_at2percent=int(np.sum(singular>.02*singular[0])),
            pulses=records,least_squares_local_candidates=candidates)
        atomic_json(root/'local_motion_audit.json',report)
        return report
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--goal',required=True);args=p.parse_args();result=audit(args.goal)
    print({k:result[k] for k in ['effective_rank_at2percent','singular_values_with_speed_scaling','least_squares_local_candidates']})
