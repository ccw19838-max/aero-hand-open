"""Bounded five-finger fixture construction, followed by free-body validation."""
import json,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco
from experiments.cube_reorient.model import scene_and_assets,build_model,fixed_axis_basis,model_ids,native_contacts,native_cube_velocity,HOME_QPOS

OUT=Path(__file__).parent;started=time.monotonic();deadline=started+285.;rng=np.random.default_rng(73)
xml,assets=scene_and_assets();root=ET.fromstring(xml);b=root.find('.//body[@name="cube"]');b.remove(b.find('freejoint'))
fixed=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'),assets);fd=mujoco.MjData(fixed);cb=fixed.body('cube').id
free=build_model();ids=model_ids(free);qa=ids['cube_qpos_adr'];basis=fixed_axis_basis(free)
tmp=mujoco.MjData(free);mujoco.mj_forward(free,tmp);palm_origin=tmp.xpos[free.body('palm').id].copy()
finger_geoms=[[g for g in range(free.ngeom) if ids['geom_finger'][g]==f and (free.geom_contype[g] or free.geom_conaffinity[g])] for f in range(5)]
lo=np.array([.061,.061,.061,.061,.25,.0265,.083,.035,-.015,.04,-1.05])
hi=np.array([.092,.092,.092,.092,1.74,.0378,.108,.10,.025,.09,1.05])
center=np.array([.073,.075,.075,.073,1.40,.031,.094,.065,.005,.065,.7854]);spread=(hi-lo)*.22
history=[];best=-1e9;best_free=-1e9;candidate_count=0;success=None

def construct(x):
    yaw=x[10];r=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    q=np.empty(4);mujoco.mju_mat2Quat(q,(basis@r).ravel());position=palm_origin+basis@x[7:10]
    fixed.body_pos[cb]=position;fixed.body_quat[cb]=q;mujoco.mj_setConst(fixed,fd);mujoco.mj_resetData(fixed,fd)
    fd.qpos[:]=HOME_QPOS;fd.ctrl[:]=x[:7];mujoco.mj_step(fixed,fd,nstep=400)
    d=mujoco.MjData(free);d.qpos[:qa]=fd.qpos;d.qpos[qa:qa+7]=np.r_[position,q];d.ctrl[:]=x[:7];mujoco.mj_forward(free,d)
    c=native_contacts(free,d,ids)
    gaps=np.array([min(mujoco.mj_geomDistance(free,d,ids['cube_geom'],g,.12,None) for g in gs) for gs in finger_geoms])
    score=10*np.minimum(c['finger_forces']/.15,1).sum()-100*np.maximum(gaps,0).sum()-2000*max(0,c['max_penetration']-.0009)
    if c['forbidden_support']:score-=100
    return score,d,c,gaps

def rollout(state):
    d=mujoco.MjData(free);d.qpos[:]=state.qpos;d.ctrl[:]=state.ctrl;mujoco.mj_forward(free,d)
    origin=d.xpos[ids['cube_body']].copy();stable=best_stable=allcontact=peak=violation=0.;forbidden=False;trace=[];reason=''
    for k in range(1500):
        if time.monotonic()>deadline:raise TimeoutError
        mujoco.mj_step(free,d);mujoco.mj_forward(free,d);c=native_contacts(free,d,ids);v,w=native_cube_velocity(free,d,ids)
        speed=float(np.linalg.norm(v));omega=float(np.linalg.norm(w));peak=max(peak,c['max_penetration']);forbidden|=c['forbidden_support']
        valid=not c['forbidden_support'] and c['max_penetration']<=.001
        five=bool((c['finger_forces']>=.02).all())
        allcontact=allcontact+.002 if five else 0.
        stable=stable+.002 if five and valid and speed<.03 and omega<.5 else 0.;best_stable=max(best_stable,stable)
        violation=violation+.002 if not valid else 0.
        trace.append([d.time,*c['finger_forces'],speed,omega,c['max_penetration']])
        if forbidden:reason='forbidden_support'
        elif violation>=.1:reason='penetration'
        elif np.linalg.norm(d.xpos[ids['cube_body']]-origin)>.08:reason='workspace'
        if reason:break
    passed=bool(d.time>=2.999 and stable>=.5 and allcontact>=1 and not reason)
    row=dict(success=passed,seconds=d.time,stable_at_end=stable,best_stable=best_stable,
             five_contact_at_end=allcontact,peak_penetration=peak,failure=reason,forces=c['finger_forces'].tolist(),
             final_position=d.xpos[ids['cube_body']].tolist())
    return row,np.array(trace)

try:
    for generation in range(30):
        batch=[]
        for j in range(48):
            if time.monotonic()>deadline:raise TimeoutError
            x=np.clip(center+rng.normal(size=len(lo))*spread,lo,hi)
            if generation==0 and j==0:x=center.copy()
            score,d,c,gaps=construct(x);candidate_count+=1
            batch.append((score,x));row=dict(index=candidate_count,generation=generation,score=score,parameters=x.tolist(),forces=c['finger_forces'].tolist(),gaps=gaps.tolist(),penetration=c['max_penetration'])
            if score>best:
                best=score;print(json.dumps(dict(fixture_best=row)),flush=True);(OUT/'best_fixture.json').write_text(json.dumps(row,indent=2))
            if (c['finger_forces']>=.02).all() and c['max_penetration']<=.001 and not c['forbidden_support']:
                result,trace=rollout(d);row['free']=result
                quality=result['best_stable']+result['five_contact_at_end']+.01*np.count_nonzero(np.array(result['forces'])>=.02)
                print(json.dumps(dict(free_candidate=row)),flush=True)
                if quality>best_free:
                    best_free=quality
                    np.savez(OUT/'candidate.npz',qpos=d.qpos,qvel=np.zeros(free.nv),ctrl=d.ctrl,grasp_ctrl=d.ctrl)
                    np.savez(OUT/'candidate_contacts.npz',trace=trace,columns=np.array(['time','thumb','index','middle','ring','pinky','speed','omega','penetration']))
                    (OUT/'best_free.json').write_text(json.dumps(row,indent=2))
                if result['success']:
                    success=row
                    np.savez(OUT/'success.npz',qpos=d.qpos,qvel=np.zeros(free.nv),ctrl=d.ctrl,grasp_ctrl=d.ctrl)
                    np.savez(OUT/'success_contacts.npz',trace=trace,columns=np.array(['time','thumb','index','middle','ring','pinky','speed','omega','penetration']))
            history.append(row)
            if success:break
        if success:break
        elites=np.stack([x for _,x in sorted(batch,key=lambda z:z[0],reverse=True)[:10]])
        center=.3*center+.7*elites.mean(0);spread=np.maximum(.4*spread+.6*elites.std(0),(hi-lo)*.025)
        (OUT/'history.json').write_text(json.dumps(history,indent=2))
except TimeoutError:
    pass
(OUT/'history.json').write_text(json.dumps(history,indent=2))
report=dict(passed=success is not None,scope='Untrained five-finger initial-state feasibility only',seed=73,
            candidates=candidate_count,elapsed_seconds=time.monotonic()-started,success=success)
(OUT/'search_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
