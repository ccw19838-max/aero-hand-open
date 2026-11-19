"""Read-only source audit: transform existing grasp to two palm-up root poses."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from dataclasses import replace
import numpy as np
import mujoco
from experiments.cube_grasp.config import CubeGraspConfig
from experiments.cube_grasp.model import scene_and_assets, model_ids, native_contacts, native_cube_velocity

OUT=Path(__file__).parent
SOURCE=Path('experiments/cube_grasp/artifacts/prepared_v2/reference.npz')
old=np.load(SOURCE);root_pos=np.array([0.,0.,.15]);result={}

def yquat(angle):return np.array([np.cos(angle/2),0.,np.sin(angle/2),0.])

def build(angle,dt):
    xml,assets=scene_and_assets(replace(CubeGraspConfig(),sim_dt=dt))
    root=ET.fromstring(xml);root.find('.//body[@name="tetheria_mount"]').set('quat',' '.join(map(str,yquat(angle))))
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'),assets)

for name,angle in [('mount_identity',0.),('palm_horizontal',-np.deg2rad(19.994717308376508))]:
    rotate=yquat(angle+np.pi/2);matrix=np.empty(9);mujoco.mju_quat2Mat(matrix,rotate);matrix=matrix.reshape(3,3)
    qpos=old['qpos'].copy();qpos[-7:-4]=root_pos+matrix@(qpos[-7:-4]-root_pos)
    quat=np.empty(4);mujoco.mju_mulQuat(quat,rotate,qpos[-4:].copy());qpos[-4:]=quat
    end_positions=[];tests={}
    for dt in [.002,.001]:
        m=build(angle,dt);ids=model_ids(m);d=mujoco.MjData(m);d.qpos[:]=qpos;d.qvel[:]=0.;d.ctrl[:]=old['ctrl'];mujoco.mj_forward(m,d)
        palm_matrix=d.xmat[m.body('palm').id].reshape(3,3).copy()
        initial=native_contacts(m,d,ids);maxdepth=initial['max_penetration'];stable=best=0.;support=False;maxv=maxw=0.;invalid_duration=0.
        start=d.xpos[ids['cube_body']].copy();maxdist=0.
        for _ in range(round(3/dt)):
            mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
            c=native_contacts(m,d,ids);v,w=native_cube_velocity(m,d,ids)
            maxdepth=max(maxdepth,c['max_penetration']);support|=c['forbidden_support'];maxv=max(maxv,float(np.linalg.norm(v)));maxw=max(maxw,float(np.linalg.norm(w)))
            maxdist=max(maxdist,float(np.linalg.norm(d.xpos[ids['cube_body']]-start)))
            fingers=c['finger_forces']>=.02
            valid=not c['forbidden_support'] and c['max_penetration']<=.001
            good=valid and fingers[0] and fingers.sum()>=3 and np.linalg.norm(v)<.03 and np.linalg.norm(w)<.5
            stable=stable+dt if good else 0.;best=max(best,stable)
            if not valid:invalid_duration+=dt
        tests[str(dt)]=dict(stable_at_end=stable,longest_stable=best,initial_depth=initial['max_penetration'],max_depth=maxdepth,
                            forbidden_support=bool(support),max_displacement=maxdist,max_speed=maxv,max_omega=maxw,
                            final_speed=float(np.linalg.norm(v)),final_omega=float(np.linalg.norm(w)),
                            final_forces=c['finger_forces'].tolist(),final_center=d.xpos[ids['cube_body']].tolist(),
                            contact_violation_total_seconds=invalid_duration,
                            strict_hold_success=bool(stable>=2-1e-8 and maxdepth<=.001 and not support))
        end_positions.append(d.xpos[ids['cube_body']].copy())
    result[name]=dict(root_quaternion=yquat(angle).tolist(),relative_quaternion=rotate.tolist(),
                     initial_cube_position=qpos[-7:-4].tolist(),initial_cube_quaternion=qpos[-4:].tolist(),
                     palm_outward_normal=(-palm_matrix[:,1]).tolist(),wrist_to_fingers=palm_matrix[:,2].tolist(),
                     index_to_pinky=(-palm_matrix[:,0]).tolist(),gravity=[0,0,-9.81],
                     timestep_end_position_difference=float(np.linalg.norm(end_positions[0]-end_positions[1])),tests=tests)
    np.savez(OUT/f'{name}_candidate.npz',qpos=qpos,qvel=np.zeros(m.nv),ctrl=old['ctrl'],grasp_ctrl=old['grasp_ctrl'])
(OUT/'orientation_report.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
