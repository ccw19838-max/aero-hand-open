from pathlib import Path
import json
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
from experiments.pen_spin.policy import load_checkpoint
from experiments.pen_spin.env import make_env
root=Path('experiments/pen_spin/artifacts')
_,cfg,ref,meta=load_checkpoint(root/'grasp_train_v4/review_20261111_current/checkpoint_stopped_65536')
env=make_env(cfg,ref,backend='native');state=env.reset(jax.random.PRNGKey(30000));m=env.mj_model;d=state.data
# The zero-increment action still applies the environment's effective limits.
d.ctrl[:]=np.clip(d.ctrl,np.asarray(env._ctrl_low),np.asarray(env._ctrl_high));mujoco.mj_forward(m,d)
xm=mjx.put_model(m)
xd=mjx.make_data(m).replace(qpos=jp.array(d.qpos),qvel=jp.array(d.qvel),ctrl=jp.array(d.ctrl))
print('compiling forward/step',flush=True)
xd=jax.jit(mjx.forward)(xm,xd)
step=jax.jit(mjx.step)
keys=['qpos','qvel','qacc','qfrc_bias','qfrc_passive','qfrc_actuator','qfrc_constraint','ten_length']
report={'initial':{},'steps':[],'options':{'timestep':m.opt.timestep,'solver':int(m.opt.solver),'iterations':m.opt.iterations,'ls_iterations':m.opt.ls_iterations,'integrator':int(m.opt.integrator)}}
for k in keys:
 a=np.asarray(getattr(d,k));b=np.asarray(getattr(xd,k));diff=np.abs(a-b);idx=int(diff.argmax())
 report['initial'][k]={'max_abs':float(diff.max()),'index':idx,'native':float(a[idx]),'mjx':float(b[idx])}
print(report['initial'],flush=True)
pen=env.ids['pen_geom']
def contacts(geoms,dist):
 return [{'pair':[m.geom(int(x)).name or str(x) for x in g],'dist':float(t)} for g,t in zip(geoms,dist) if pen in g and t<.002]
for i in range(50):
 mujoco.mj_step(m,d);xd=step(xm,xd)
 if i in [0,1,2,4,9,14,17,24,49]:
  qdiff=np.abs(d.qpos-np.asarray(xd.qpos));row={'time':(i+1)*cfg.sim_dt,'qpos_max_abs':float(qdiff.max()),'qpos_worst_index':int(qdiff.argmax()),'pen_position_distance':float(np.linalg.norm(d.qpos[-7:-4]-np.asarray(xd.qpos[-7:-4]))),'native_pen_contacts':contacts(d.contact.geom,d.contact.dist),'mjx_pen_contacts':contacts(np.asarray(xd._impl.contact.geom),np.asarray(xd._impl.contact.dist))}
  report['steps'].append(row);print(json.dumps(row),flush=True)
(root/'grasp_open_source_v3/parity_probe.json').write_text(json.dumps(report,indent=2)+'\n')
