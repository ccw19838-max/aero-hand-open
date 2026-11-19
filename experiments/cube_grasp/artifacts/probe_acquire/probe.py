"""Short, unassisted release/reclosure diagnostic; never a training score."""
import json, time
from pathlib import Path
import mujoco
import numpy as np
from experiments.cube_grasp.model import build_model, model_ids, native_contacts, native_cube_velocity
from experiments.cube_grasp.config import CubeGraspConfig
from experiments.cube_grasp.env import GraspMonitor

OUT=Path(__file__).parent
cfg=CubeGraspConfig();started=time.monotonic();deadline=started+59.
m=build_model(cfg);ids=model_ids(m);qa=ids['cube_qpos_adr'];span=ids['ctrl_upper']-ids['ctrl_lower']
with np.load(OUT.parent/'prepared_v2/reference.npz') as z:
    base={k:z[k].copy() for k in z.files}
rows=[];found=0

def fresh(qpos,qvel,ctrl):
    d=mujoco.MjData(m);d.qpos[:]=qpos;d.qvel[:]=qvel;d.ctrl[:]=ctrl;mujoco.mj_forward(m,d)
    return d

def step(d):
    if time.monotonic()>deadline:raise TimeoutError
    mujoco.mj_step(m,d);mujoco.mj_forward(m,d)

def evaluate(qpos,qvel,ctrl,close):
    d=fresh(qpos,qvel,ctrl);monitor=GraspMonitor(cfg);origin=d.xpos[ids['cube_body']].copy()
    while not monitor.done:
        if close:
            d.ctrl[:]=np.clip(d.ctrl+np.clip(base['grasp_ctrl']-d.ctrl,-.01*span,.01*span),ids['ctrl_lower'],ids['ctrl_upper'])
        for _ in range(cfg.substeps):
            step(d);c=native_contacts(m,d,ids);v,w=native_cube_velocity(m,d,ids)
            monitor.update(c,v,w,np.linalg.norm(d.xpos[ids['cube_body']]-origin),d.xpos[ids['cube_body'],2],not np.isfinite(d.qpos).all())
            if monitor.done:break
    return dict(success=monitor.success,failure=monitor.failure_reason,time=monitor.elapsed,
                stable=monitor.max_stable_seconds,peak_penetration=monitor.peak_penetration,
                final_forces=c['finger_forces'].tolist())

spec=[]
for duration in [.05,.1]:
    for channel in [2,6]:
        for offset in [.002,.004,.006,.010,.015]:
            spec.append((duration,{channel:offset}))
for duration in [.05,.1]:
    for ring,thumb in [(.002,.002),(.004,.002),(.004,.004),(.006,.004),(.01,.006)]:
        spec.append((duration,{2:ring,6:thumb}))
try:
    for index,(duration,offsets) in enumerate(spec[:60]):
        ctrl=base['ctrl'].copy()
        for ch,delta in offsets.items():ctrl[ch]+=delta
        ctrl=np.clip(ctrl,ids['ctrl_lower'],ids['ctrl_upper'])
        d=fresh(base['qpos'],base['qvel'],ctrl);peak=0.
        for _ in range(round(duration/cfg.sim_dt)):
            step(d);c=native_contacts(m,d,ids);peak=max(peak,c['max_penetration'])
        c=native_contacts(m,d,ids);v,w=native_cube_velocity(m,d,ids)
        initial_stable=bool(c['finger_forces'][0]>=cfg.contact_force and (c['finger_forces']>=cfg.contact_force).sum()>=cfg.min_fingers and np.linalg.norm(v)<cfg.grip_velocity and np.linalg.norm(w)<cfg.grip_omega)
        row=dict(index=index,duration=duration,offsets=offsets,initial_penetration=c['max_penetration'],prelude_peak_penetration=peak,
                 initial_stable=initial_stable,initial_forces=c['finger_forces'].tolist(),initial_speed=float(np.linalg.norm(v)),initial_omega=float(np.linalg.norm(w)))
        if c['forbidden_support'] or c['max_penetration']>cfg.max_penetration or initial_stable:
            row['rejected']=True;rows.append(row);continue
        qpos=d.qpos.copy();qvel=d.qvel.copy()
        row['zero']=evaluate(qpos,qvel,ctrl,False)
        if not row['zero']['success']:
            row['close']=evaluate(qpos,qvel,ctrl,True)
            if row['close']['success'] and row['close']['peak_penetration']<=cfg.max_penetration:
                found+=1
                np.savez(OUT/f'candidate_{found}.npz',qpos=qpos,qvel=qvel,ctrl=ctrl,grasp_ctrl=base['grasp_ctrl'])
                row['candidate_file']=f'candidate_{found}.npz'
        rows.append(row);print(json.dumps(row),flush=True)
        (OUT/'results.json').write_text(json.dumps(dict(found=found,elapsed=time.monotonic()-started,rows=rows),indent=2))
        if found>=2:break
except TimeoutError:
    pass
(OUT/'results.json').write_text(json.dumps(dict(found=found,elapsed=time.monotonic()-started,rows=rows),indent=2))
print('DONE',len(rows),found,time.monotonic()-started,flush=True)
