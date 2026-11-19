from experiments.pen_spin.tripod_prepare import tripod_config,hold_trial
from experiments.pen_spin.model import build_model,model_ids
from pathlib import Path
import numpy as np,mujoco,json,itertools,time
cfg=tripod_config();m=build_model(cfg);ids=model_ids(m);span=m.actuator_ctrlrange[:,1]-m.actuator_ctrlrange[:,0]
with np.load('experiments/pen_spin/artifacts/tripod_prepare_v3d/best_geometry_candidate.npz') as r:ref={k:r[k].copy() for k in r.files}
d=mujoco.MjData(m);d.qpos[:]=ref['qpos'];mujoco.mj_forward(m,d);axis=(d.site('pen_a').xpos-d.site('pen_b').xpos)/.13
results=[];best=0.;q=ids['pen_qpos_adr'];start=time.monotonic()
for i,(offset,preload) in enumerate(itertools.product([-.006,-.008,-.004,-.002,0.,.002,.004,.006,.008,-.012,.012],[0.,.002,.005,.01,.02,.04])):
 candidate={k:v.copy() for k,v in ref.items()};candidate['qpos'][q:q+3]+=offset*axis;candidate['c0']+=offset*axis
 candidate['ctrl'][[0,1,6]]+=span[[0,1,6]]*np.array([-preload,-preload,preload])
 trial,state=hold_trial(m,candidate,cfg)
 trial.update(offset=offset,preload=preload);results.append(trial)
 if trial['peak_stable_seconds']>best:best=trial['peak_stable_seconds'];print('best',i,trial,flush=True)
 if i%20==0:print('balance tested',i,'seconds',time.monotonic()-start,flush=True)
 if trial['passed']:
  np.savez('experiments/pen_spin/artifacts/grasp_open_source_v3/balance_success.npz',**state);print('SUCCESS',trial,flush=True);break
Path('experiments/pen_spin/artifacts/grasp_open_source_v3/balance_probe.json').write_text(json.dumps(results,indent=2)+'\n')
