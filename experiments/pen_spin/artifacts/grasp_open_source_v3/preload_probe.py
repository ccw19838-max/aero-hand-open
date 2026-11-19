from experiments.pen_spin.tripod_prepare import tripod_config,hold_trial
from experiments.pen_spin.model import build_model,model_ids
from pathlib import Path
import numpy as np,json,itertools,time
cfg=tripod_config();m=build_model(cfg);span=m.actuator_ctrlrange[:,1]-m.actuator_ctrlrange[:,0]
with np.load('experiments/pen_spin/artifacts/tripod_prepare_v3d/best_geometry_candidate.npz') as r:ref={k:r[k].copy() for k in r.files}
results=[];start=time.monotonic();best=0
for i,(a,b,c) in enumerate(itertools.product([.001,.002,.004,.008,.0005],[.001,.002,.004,.008,.0005],[.001,.002,.004,.008,.0005])):
 candidate={k:v.copy() for k,v in ref.items()};candidate['ctrl'][[0,1,6]]+=span[[0,1,6]]*np.array([-a,-b,c])
 trial,state=hold_trial(m,candidate,cfg)
 trial['preload']=[a,b,c];results.append(trial)
 if trial['peak_stable_seconds']>best:best=trial['peak_stable_seconds'];print('best',i,trial,flush=True)
 if i%20==0:print('preload tested',i,'seconds',time.monotonic()-start,flush=True)
 if trial['passed']:
  np.savez('experiments/pen_spin/artifacts/grasp_open_source_v3/preload_success.npz',**state);print('SUCCESS',trial,flush=True);break
Path('experiments/pen_spin/artifacts/grasp_open_source_v3/preload_probe.json').write_text(json.dumps(results,indent=2)+'\n')
