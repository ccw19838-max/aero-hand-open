import json
from pathlib import Path
import jax,jax.numpy as jp,numpy as np,mujoco
from mujoco import mjx
from experiments.pen_spin.policy import load_checkpoint
from experiments.pen_spin.env import make_env
_,cfg,ref,_=load_checkpoint('experiments/pen_spin/artifacts/grasp_train_v4/review_20261111_current/checkpoint_stopped_65536')
env=make_env(cfg,ref,backend='native');d=env.reset(jax.random.PRNGKey(30000)).data;m=env.mj_model
for _ in range(10):mujoco.mj_step(m,d)
mujoco.mj_forward(m,d)
xm=mjx.put_model(m);xd=mjx.make_data(m).replace(qpos=jp.array(d.qpos),qvel=jp.array(d.qvel),ctrl=jp.array(d.ctrl))
xd=jax.jit(mjx.forward)(xm,xd)
report={}
for name in ['th_tip','if_tip','mf_tip','rf_tip','pf_tip']:
 g=m.geom(name).id;p=env.ids['pen_geom'];pairs=np.asarray(xd._impl.contact.geom);dist=np.asarray(xd._impl.contact.dist)
 mask=((pairs[:,0]==g)&(pairs[:,1]==p))|((pairs[:,1]==g)&(pairs[:,0]==p))
 closest=np.zeros(6);native=mujoco.mj_geomDistance(m,d,g,p,.5,closest)
 report[name]={'native_geom_distance':native,'mjx_distances':dist[mask].tolist()}
print(json.dumps(report,indent=2))
Path('experiments/pen_spin/artifacts/grasp_open_source_v3/contact_probe.json').write_text(json.dumps(report,indent=2)+'\n')
