import time,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco
from experiments.cube_grasp.model import *
from experiments.cube_grasp.config import CubeGraspConfig
from experiments.cube_grasp.prepare import trial,approve
cfg=CubeGraspConfig();xml,assets=scene_and_assets(cfg);root=ET.fromstring(xml);cube=root.find('.//body[@name="cube"]');cube.remove(cube.find('freejoint'))
fixed=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'),assets);fd=mujoco.MjData(fixed);bid=fixed.body('cube').id
free=build_model(cfg);ids=model_ids(free);rng=np.random.default_rng(21);best=-1;log=[];start=time.monotonic()
base=np.load('experiments/cube_grasp/artifacts/fixture_search/reference.npz');baseq=base['qpos'];basec=base['ctrl'];pos=baseq[-7:-4]
out=Path('experiments/cube_grasp/artifacts/refine');out.mkdir(exist_ok=True)
for i in range(180):
 ctrl=basec.copy()
 if i<30:
  ctrl[[0,1,2,3,6]]+=.0005*(i%6);ctrl[4]-=.025*(i//6)
 else:
  ctrl+=rng.normal(0,[.002,.002,.002,.002,.06,.0008,.002])
 fixed.body_pos[bid]=pos;fixed.body_quat[bid]=baseq[-4:];mujoco.mj_setConst(fixed,fd);mujoco.mj_resetData(fixed,fd)
 fd.qpos[:]=baseq[:16];fd.ctrl[:]=ctrl;mujoco.mj_step(fixed,fd,nstep=500)
 qpos=np.r_[fd.qpos,baseq[-7:]];test=mujoco.MjData(free);test.qpos[:]=qpos;test.ctrl[:]=ctrl;mujoco.mj_forward(free,test)
 c=native_contacts(free,test,ids)
 if c['max_penetration']>.0004:continue
 row,end=trial(free,qpos,ctrl,seconds=3.);row.update(index=i,initial_penetration=c['max_penetration'],ctrl=ctrl.tolist());log.append(row)
 score=row['best_hold_seconds']-.1*c['finger_forces'].max()
 if score>best:
  best=score;print(json.dumps(row),flush=True);np.savez(out/'candidate.npz',qpos=qpos,qvel=np.zeros(free.nv),ctrl=ctrl,grasp_ctrl=ctrl)
 if row['success']:
  r=approve(out,cfg,qpos,ctrl,log,i+1);print('APPROVED',r['approved'],flush=True)
  if r['approved']:break
(out/'history.json').write_text(json.dumps(log,indent=2));print('DONE',time.monotonic()-start,'trials',len(log),flush=True)
