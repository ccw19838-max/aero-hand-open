"""Recheck a naturally settled five-finger candidate with fresh, unforced data."""
import json
from pathlib import Path
from dataclasses import replace
import mujoco,numpy as np
from experiments.cube_reorient.model import build_model,model_ids,native_contacts,native_cube_velocity
from experiments.cube_reorient.config import CubeReorientConfig

OUT=Path(__file__).parent
z=np.load(OUT/'success.npz');m=build_model();ids=model_ids(m);d=mujoco.MjData(m)
d.qpos[:]=z['qpos'];d.ctrl[:]=z['ctrl'];mujoco.mj_forward(m,d)
for _ in range(1500):mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
qpos=d.qpos.copy();ctrl=d.ctrl.copy();tests={};ends=[]
np.savez(OUT/'settled_candidate.npz',qpos=qpos,qvel=np.zeros(m.nv),ctrl=ctrl,grasp_ctrl=ctrl)
for dt in [.002,.001]:
    m=build_model(replace(CubeReorientConfig(),sim_dt=dt));ids=model_ids(m);d=mujoco.MjData(m)
    d.qpos[:]=qpos;d.ctrl[:]=ctrl;mujoco.mj_forward(m,d)
    c=native_contacts(m,d,ids);peak=c['max_penetration'];initial=c['finger_forces'].copy();initdepth=peak
    trace=[];stable=best=contact_time=over=longest_over=0.;forbidden=False;min_last=np.full(5,np.inf)
    start=d.xpos[ids['cube_body']].copy();maxdist=0.
    for k in range(round(3/dt)):
        mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        c=native_contacts(m,d,ids);v,w=native_cube_velocity(m,d,ids);speed=float(np.linalg.norm(v));omega=float(np.linalg.norm(w))
        five=bool((c['finger_forces']>=.02).all());valid=c['max_penetration']<=.001 and not c['forbidden_support']
        stable=stable+dt if five and valid and speed<.03 and omega<.5 else 0.;best=max(best,stable)
        contact_time=contact_time+dt if five else 0.
        peak=max(peak,c['max_penetration']);forbidden|=c['forbidden_support'];over=over+dt if c['max_penetration']>.001 else 0.;longest_over=max(longest_over,over)
        maxdist=max(maxdist,float(np.linalg.norm(d.xpos[ids['cube_body']]-start)))
        if d.time>=2: min_last=np.minimum(min_last,c['finger_forces'])
        trace.append([d.time,*c['finger_forces'],speed,omega,c['max_penetration']])
    name=str(dt)
    tests[name]=dict(stable_end=stable,longest_stable=best,five_contact_end=contact_time,
                     initial_forces=initial.tolist(),initial_depth=initdepth,peak_penetration=peak,
                     longest_penetration_overlimit=longest_over,forbidden_support=bool(forbidden),
                     last_second_min_forces=min_last.tolist(),final_forces=c['finger_forces'].tolist(),
                     final_speed=speed,final_omega=omega,max_displacement=maxdist,
                     final_position=d.xpos[ids['cube_body']].tolist(),
                     passed=bool(stable>=.5 and contact_time>=1 and longest_over<.1 and not forbidden))
    ends.append(d.xpos[ids['cube_body']].copy())
    np.savez(OUT/f'settled_trace_{round(dt*1e6)}us.npz',trace=np.array(trace),columns=np.array(['time','thumb','index','middle','ring','pinky','speed','omega','penetration']))
report=dict(passed=all(v['passed'] for v in tests.values()),tests=tests,final_position_difference=float(np.linalg.norm(ends[0]-ends[1])),
            scope='Five-finger initial holding feasibility, not a learned policy',
            state_source='success.npz followed by 3 seconds of native free evolution; qvel reset to zero before independent tests')
(OUT/'settled_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
