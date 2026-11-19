"""Twenty matched randomized physical resets of the settled five-finger pose."""
import json,time
from pathlib import Path
import numpy as np,mujoco
from experiments.cube_reorient.model import build_model,model_ids,native_contacts,native_cube_velocity

OUT=Path(__file__).parent;z=np.load(OUT/'success_settled.npz');m=build_model();ids=model_ids(m)
mass=m.body_mass[ids['cube_body']].copy();inertia=m.body_inertia[ids['cube_body']].copy();friction=m.geom_friction[ids['cube_geom']].copy()
qa=ids['cube_qpos_adr'];records=[];started=time.monotonic()
for seed in range(20):
    rng=np.random.default_rng(seed);mf=float(rng.uniform(.9,1.1));ff=float(rng.uniform(.9,1.1))
    m.body_mass[ids['cube_body']]=mass*mf;m.body_inertia[ids['cube_body']]=inertia*mf
    m.geom_friction[ids['cube_geom']]=friction;m.geom_friction[ids['cube_geom'],0]*=ff
    d=mujoco.MjData(m);mujoco.mj_setConst(m,d)
    reset_valid=False
    for attempt in range(1,33):
        mujoco.mj_resetData(m,d);d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel'];d.ctrl[:]=z['ctrl']
        d.qpos[qa:qa+3]+=rng.uniform(-.001,.001,3)
        axis=rng.normal(size=3);axis/=np.linalg.norm(axis);angle=np.deg2rad(rng.uniform(-2,2));noise=np.r_[np.cos(angle/2),axis*np.sin(angle/2)]
        q=np.empty(4);mujoco.mju_mulQuat(q,noise,d.qpos[qa+3:qa+7].copy());d.qpos[qa+3:qa+7]=q/np.linalg.norm(q)
        mujoco.mj_forward(m,d);c=native_contacts(m,d,ids)
        if c['max_penetration']<=.001 and not c['forbidden_support']:reset_valid=True;break
    row=dict(seed=seed,mass_factor=mf,friction_factor=ff,reset_attempts=attempt,reset_valid=reset_valid,initial_depth=c['max_penetration'])
    if not reset_valid:
        row.update(passed=False,reason='reset_rejected');records.append(row);continue
    stable=best=contacts_time=over=longest_over=0.;peak=c['max_penetration'];reason='';trace=[];min_last=np.full(5,np.inf)
    initialq=d.qpos.copy();initialv=d.qvel.copy()
    for _ in range(1500):
        mujoco.mj_step(m,d);mujoco.mj_forward(m,d);c=native_contacts(m,d,ids);v,w=native_cube_velocity(m,d,ids)
        five=bool((c['finger_forces']>=.02).all());valid=c['max_penetration']<=.001 and not c['forbidden_support']
        stable=stable+.002 if five and valid and np.linalg.norm(v)<.03 and np.linalg.norm(w)<.5 else 0.;best=max(best,stable)
        contacts_time=contacts_time+.002 if five else 0.;peak=max(peak,c['max_penetration']);over=over+.002 if c['max_penetration']>.001 else 0.;longest_over=max(over,longest_over)
        if c['forbidden_support']:reason='forbidden_support'
        if over>=.1:reason='sustained_penetration'
        if d.time>=2:min_last=np.minimum(min_last,c['finger_forces'])
        trace.append([d.time,*c['finger_forces'],np.linalg.norm(v),np.linalg.norm(w),c['max_penetration']])
        if reason:break
    passed=bool(d.time>=2.999 and stable>=.5 and contacts_time>=1 and not reason)
    row.update(passed=passed,reason=reason,stable_at_end=stable,max_stable=best,five_contact_end=contacts_time,
               peak_penetration=peak,longest_overlimit=longest_over,final_forces=c['finger_forces'].tolist(),
               last_second_min_forces=min_last.tolist() if d.time>=2 else None,seconds=d.time)
    records.append(row);print(json.dumps(row),flush=True)
    np.savez_compressed(OUT/f'jitter_seed_{seed:02d}.npz',trace=np.array(trace),initial_qpos=initialq,initial_qvel=initialv,
                        columns=np.array(['time','thumb','index','middle','ring','pinky','speed','omega','penetration']))
result=dict(episodes=20,successes=sum(r['passed'] for r in records),success_rate=sum(r['passed'] for r in records)/20,
            elapsed_seconds=time.monotonic()-started,scope='Untrained fixed-target five-finger holding feasibility, not learned policy performance or real fingertip forces',
            randomization=dict(position_m=.001,angle_degrees=2,mass_fraction=.1,friction_fraction=.1,reset_attempt_limit=32),records=records)
result['half_second_stability_successes']=sum(r.get('max_stable',0)>=.5-1e-9 for r in records)
result['half_second_stability_rate']=result['half_second_stability_successes']/20
result['strict_success_definition']='Complete 3-second free rollout, final stable >=0.5 s, five-finger contact continuous >=1 s, no physical failure'
(OUT/'jitter_report.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
