"""Independent endpoint construction only: never credited as a turn trajectory."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,json
import numpy as np
import mujoco
from experiments.cube_envelop.model import native_contacts,native_cube_velocity
from experiments.cube_quarter_turn.task import quaternion_multiply,canonical_quaternion,orientation_error
from .env import AdaptiveEnv
from .storage import atomic_json


def audit(goal):
    root=Path(goal);rows=[]
    for axis in ('x','y'):
        for dt in (.001,.002):
            env=AdaptiveEnv(root/'sources/x/reference.npz',axis,physical_dt=dt,target_degrees=90.)
            try:
                for degrees in (0.,10.,30.,45.,60.,90.):
                    env.reset(seed=0,options={'reference':True})
                    original=env.data.qpos[19:23].copy();direction=env.monitor.direction.copy();angle=np.deg2rad(degrees)
                    terminal=canonical_quaternion(quaternion_multiply(np.r_[np.cos(angle/2),direction*np.sin(angle/2)],original))
                    # This is an independently constructed terminal-grasp test.
                    # No task episode, policy grade or trajectory is evaluated.
                    env.data.qpos[19:23]=terminal;mujoco.mj_forward(env.model,env.data)
                    initial=native_contacts(env.model,env.data,env.ids);valid=not initial['forbidden_support'] and initial['max_penetration']<=.001
                    stable=0.;maximum=0.;max_penetration=initial['max_penetration'];mins=np.full(5,np.inf)
                    if valid:
                        for step in range(round(1.5/dt)):
                            mujoco.mj_step(env.model,env.data)
                            if (step+1)%round(.01/dt):continue
                            mujoco.mj_forward(env.model,env.data);contact=native_contacts(env.model,env.data,env.ids)
                            linear,angular=native_cube_velocity(env.model,env.data,env.ids);forces=contact['finger_forces']
                            _,error=orientation_error(env.data.qpos[19:23],terminal)
                            max_penetration=max(max_penetration,contact['max_penetration']);mins=np.minimum(mins,forces)
                            good=bool(np.all(forces>=np.array([.2,.1,.1,.1,.1])) and np.all(forces<=np.array([20.,12.,12.,12.,12.]))
                                and not contact['forbidden_support'] and contact['max_penetration']<=.001
                                and np.linalg.norm(linear)<.03 and np.linalg.norm(angular)<.5 and error<=np.deg2rad(5.))
                            stable=stable+.01 if good else 0.;maximum=max(maximum,stable)
                    row=dict(axis=axis,dt=dt,constructed_endpoint_degrees=degrees,initial_geometry_valid=bool(valid),
                        initial_penetration_m=initial['max_penetration'],peak_penetration_m=max_penetration,
                        static_five_finger_hold_seconds=maximum,min_finger_forces=mins.tolist() if valid else None,
                        rotation_trajectory_tested=False)
                    rows.append(row)
            finally:env.close()
    report=dict(observed_at=datetime.now(timezone.utc).isoformat(),
        scope='Independently constructed endpoints, original hand targets and gravity. No rotation path or learned policy was evaluated.',
        used_in_training_or_policy_scores=False,records=rows)
    atomic_json(root/'static_endpoint_audit.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--goal',required=True);args=p.parse_args();result=audit(args.goal)
    print(json.dumps(result,ensure_ascii=False))
