import json, time
from pathlib import Path
import numpy as np
import mujoco
from experiments.cube_grasp.model import build_model, model_ids, native_contacts, native_cube_velocity, HOME_CTRL, HOME_QPOS

OUT=Path(__file__).parent
m=build_model(); ids=model_ids(m); qa=ids['cube_qpos_adr']; va=ids['cube_dof_adr']
d=mujoco.MjData(m); rng=np.random.default_rng(871)
lo=np.array([.083,.083,.083,.083,1.0,.032,.088,-.075,-.015,.285,-.5])
hi=np.array([.099,.099,.099,.099,1.7,.038,.103,-.025,.025,.335,.5])
center=np.array([.09,.09,.09,.09,1.4,.037,.096,-.05,0,.316,0.])
spread=(hi-lo)*.3
deadline=time.monotonic()+145
best=-1e9; history=[]; good=0

def trial(x):
    mujoco.mj_resetData(m,d)
    d.qpos[:qa]=HOME_QPOS;d.qpos[qa:qa+7]=[1,0,.6,1,0,0,0];d.ctrl[:]=x[:7]
    mujoco.mj_forward(m,d);mujoco.mj_step(m,d,nstep=300)
    d.qpos[qa:qa+3]=x[7:10];d.qpos[qa+3:qa+7]=[np.cos(x[10]/2),0,np.sin(x[10]/2),0]
    d.qvel[:]=0;mujoco.mj_forward(m,d)
    c=native_contacts(m,d,ids)
    if c['max_penetration']>.0005 or c['forbidden_support']:
        return -30-1000*c['max_penetration'], None, {'initial_penetration':float(c['max_penetration'])}
    initial=dict(qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy())
    stable=0.;longest=0.;maxcontacts=0;maxpen=0.;totcontacts=0.;maxthumb=0.;elapsed=0
    for k in range(100):
        mujoco.mj_step(m,d,nstep=10);elapsed=(k+1)*.02
        c=native_contacts(m,d,ids);v,w=native_cube_velocity(m,d,ids)
        n=int((c['finger_forces']>.02).sum());th=c['finger_forces'][0]>.02
        maxcontacts=max(maxcontacts,n);totcontacts+=n*.02;maxthumb=max(maxthumb,int(th));maxpen=max(maxpen,c['max_penetration'])
        valid=not c['forbidden_support'] and c['max_penetration']<.001 and np.linalg.norm(d.qpos[qa:qa+3]-x[7:10])<.08
        held=valid and th and n>=3 and np.linalg.norm(v)<.03 and np.linalg.norm(w)<.5
        stable=stable+.02 if held else 0.;longest=max(longest,stable)
        if not valid or d.qpos[qa+2]<.205:break
    score=elapsed+totcontacts*.5+maxthumb*.5+longest*20-100*max(0,maxpen-.001)
    success=elapsed>=2 and stable>=.5
    if success: score+=100
    return score,initial,dict(elapsed=elapsed,stable=stable,longest=longest,contacts=maxcontacts,thumb=maxthumb,max_penetration=maxpen,success=success,final_center=d.qpos[qa:qa+3].tolist(),forces=c['finger_forces'].tolist())

for generation in range(30):
    batch=[]
    for j in range(32):
        if time.monotonic()>deadline:break
        x=np.clip(center+rng.normal(size=len(lo))*spread,lo,hi)
        if generation==0 and j==0:x=center.copy()
        score,state,info=trial(x)
        batch.append((score,x));history.append(dict(generation=generation,x=x.tolist(),score=score,**info))
        if score>best:
            best=score
            print(json.dumps(dict(best=best,x=x.tolist(),**info)),flush=True)
            if state is not None:np.savez(OUT/'candidate_v2.npz',**state,params=x,score=score)
            (OUT/'best_v2.json').write_text(json.dumps(history[-1],indent=2))
        if info.get('success'):
            good+=1;np.savez(OUT/f'success_{good}.npz',**state,params=x,score=score)
            print('SUCCESS',good,flush=True)
            break
    if good or time.monotonic()>deadline:break
    if batch:
        elite=np.stack([x for _,x in sorted(batch,key=lambda t:t[0],reverse=True)[:8]])
        center=.4*center+.6*elite.mean(0)
        spread=np.maximum(.45*spread+.55*elite.std(0),(hi-lo)*.03)
    (OUT/'history_v2.json').write_text(json.dumps(history,indent=2))
(OUT/'history_v2.json').write_text(json.dumps(history,indent=2))
print('FINISHED',len(history),best,good,flush=True)
