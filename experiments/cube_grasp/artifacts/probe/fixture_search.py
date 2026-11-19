import time,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco
from experiments.cube_grasp.model import *
from experiments.cube_grasp.config import CubeGraspConfig
from experiments.cube_grasp.prepare import trial,approve
cfg=CubeGraspConfig();xml,assets=scene_and_assets(cfg);root=ET.fromstring(xml);cube=root.find('.//body[@name="cube"]');cube.remove(cube.find('freejoint'))
fixed=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'),assets);fd=mujoco.MjData(fixed);bid=fixed.body('cube').id
free=build_model(cfg);ids=model_ids(free);rng=np.random.default_rng(17);best=-1;log=[];start=time.monotonic()
out=Path('experiments/cube_grasp/artifacts/fixture_search');out.mkdir(exist_ok=True)
for i in range(400):
    if time.monotonic()-start>300:break
    if i<30:
        pos=np.array([[-.03,-.02,-.01][i%3],-.005,[.265,.275,.285][i//3%3]])
        ctrl=np.r_[np.full(4,[.075,.08,.085][i//9%3]),.8,.033,.097]
    else:
        pos=rng.uniform([-.04,-.025,.255],[-.005,.015,.31])
        ctrl=np.r_[np.clip(np.full(4,rng.uniform(.069,.09))+rng.normal(0,.0015,4),.065,.095),rng.uniform(.2,1.45),rng.uniform(.0275,.0375),rng.uniform(.087,.108)]
    quat=np.array([1,0,0,0])
    fixed.body_pos[bid]=pos;fixed.body_quat[bid]=quat;mujoco.mj_setConst(fixed,fd);mujoco.mj_resetData(fixed,fd)
    fd.qpos[:]=HOME_QPOS;fd.ctrl[:]=ctrl;mujoco.mj_step(fixed,fd,nstep=750)
    qpos=np.r_[fd.qpos,pos,quat];test=mujoco.MjData(free);test.qpos[:]=qpos;test.ctrl[:]=ctrl;mujoco.mj_forward(free,test)
    c=native_contacts(free,test,ids)
    if c['max_penetration']>.001:continue
    row,end=trial(free,qpos,ctrl,seconds=3.)
    score=row['best_hold_seconds']+min(row['seconds'],3)*.05+(np.array(row['finger_forces'])>.02).sum()*.01
    row.update(index=i,position=pos.tolist(),ctrl=ctrl.tolist());log.append(row)
    if score>best:
        best=score;np.savez(out/'candidate.npz',qpos=qpos,qvel=np.zeros(free.nv),ctrl=ctrl,grasp_ctrl=ctrl)
        print(json.dumps(row),flush=True)
    if row['success']:
        result=approve(out,cfg,qpos,ctrl,log,i+1);print(json.dumps(result),flush=True)
        if result['approved']:break
(out/'history.json').write_text(json.dumps(log,indent=2));print('DONE',time.monotonic()-start,'trials',len(log),flush=True)
