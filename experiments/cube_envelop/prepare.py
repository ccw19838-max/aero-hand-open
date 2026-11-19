"""Search and verify true five-finger envelope resets without changing physics."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .config import CubeReorientConfig
from .model import (scene_and_assets,build_model,model_ids,model_fingerprint,
                    native_contacts,native_cube_velocity,fixed_axis_basis,HOME_QPOS)


class SearchTimeout(Exception):pass


def validate(model,qpos,ctrl,qvel=None,seconds=3.,deadline=float('inf')):
    ids=model_ids(model);d=mujoco.MjData(model)
    d.qpos[:]=qpos;d.ctrl[:]=ctrl
    if qvel is not None:d.qvel[:]=qvel
    mujoco.mj_forward(model,d)
    initial=native_contacts(model,d,ids)
    initial_position=d.xpos[ids['cube_body']].copy()
    stable=best_stable=five=peak=penetrating=longest_penetration=0.
    minimum=np.full(5,np.inf);trace=[];reason=None;dt=float(model.opt.timestep)
    if initial['max_penetration']>.001 or initial['forbidden_support']:
        reason='invalid_initial_contact'
    for _ in range(round(seconds/dt)):
        if reason:break
        if time.monotonic()>deadline:raise SearchTimeout
        mujoco.mj_step(model,d);mujoco.mj_forward(model,d)
        c=native_contacts(model,d,ids);v,w=native_cube_velocity(model,d,ids)
        speed=float(np.linalg.norm(v));omega=float(np.linalg.norm(w))
        all_five=bool(np.all(c['finger_forces']>=.02))
        five=five+dt if all_five else 0.
        valid=c['max_penetration']<=.001 and not c['forbidden_support']
        stable=stable+dt if valid and all_five and speed<.03 and omega<.5 else 0.
        best_stable=max(best_stable,stable);peak=max(peak,c['max_penetration'])
        penetrating=penetrating+dt if c['max_penetration']>.001 else 0.
        longest_penetration=max(longest_penetration,penetrating)
        if d.time>seconds-1:minimum=np.minimum(minimum,c['finger_forces'])
        trace.append([d.time,*c['finger_forces'],speed,omega,c['max_penetration']])
        if c['forbidden_support']:reason='forbidden_support'
        elif penetrating>=.1-1e-10:reason='penetration'
        elif np.linalg.norm(d.xpos[ids['cube_body']]-initial_position)>.08:reason='workspace'
        elif not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():reason='numerical'
    result=dict(passed=bool(not reason and d.time>=seconds-1e-8 and stable>=.5 and five>=1.),
                seconds=float(d.time),stable_at_end=stable,best_stable_seconds=best_stable,
                five_contact_at_end=five,initial_penetration=float(initial['max_penetration']),
                initial_forces=initial['finger_forces'].tolist(),final_forces=c['finger_forces'].tolist() if trace else [],
                last_second_min_forces=minimum.tolist() if trace and d.time>seconds-1 else [],
                peak_penetration=peak,longest_penetration_seconds=longest_penetration,
                failure_reason=reason,final_position=d.xpos[ids['cube_body']].tolist())
    return result,d,np.asarray(trace)


def approve(output,config,qpos,ctrl,construction,deadline=float('inf')):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    checks={};ends=[]
    for dt in (config.sim_dt,config.sim_dt/2):
        model=build_model(replace(config,sim_dt=dt))
        row,d,trace=validate(model,qpos,ctrl,deadline=deadline)
        checks[str(dt)]=row;ends.append(d.xpos[model.body('cube').id].copy())
        np.savez_compressed(output/f'contact_trace_{round(dt*1e6)}us.npz',trace=trace,
                            columns=np.array(['time','thumb','index','middle','ring','pinky','speed','omega','penetration']))
    difference=float(np.linalg.norm(ends[0]-ends[1]))
    passed=all(r['passed'] for r in checks.values()) and difference<.005
    report=dict(approved=passed,task='cube_envelope_xyz90_v1',config=config.to_dict(),
                model_fingerprint=model_fingerprint(config),construction=construction,
                scope='Five-finger envelope feasibility, not learned rotation or grasp performance',
                numerical_checks=checks,final_position_difference=difference,
                force_order=['thumb','index','middle','ring','pinky'],
                contact_rule='Every finger >=0.02 N; final low-speed stability >=0.5s; all five contact for final >=1s',
                penetration_rule='Depth<=1mm for stability; >1mm continuously for 0.1s fails; peaks retained')
    if passed:
        model=build_model(config)
        np.savez(output/'reference.npz',qpos=qpos,qvel=np.zeros(model.nv),ctrl=ctrl,grasp_ctrl=ctrl)
        report['reference_sha256']=hashlib.sha256((output/'reference.npz').read_bytes()).hexdigest()
    (output/'prepare_report.json').write_text(json.dumps(report,indent=2))
    (output/'config.json').write_text(json.dumps(config.to_dict(),indent=2))
    return report


def search(output,config,seconds=300.,seed=73):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+seconds;rng=np.random.default_rng(seed)
    xml,assets=scene_and_assets(config);root=ET.fromstring(xml)
    body=root.find('.//body[@name="cube"]');body.remove(body.find('freejoint'))
    fixed=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'),assets)
    fd=mujoco.MjData(fixed);cb=fixed.body('cube').id
    free=build_model(config);ids=model_ids(free);qa=ids['cube_qpos_adr'];basis=fixed_axis_basis(free)
    state=mujoco.MjData(free);mujoco.mj_forward(free,state);palm_origin=state.xpos[free.body('palm').id].copy()
    finger_geoms=[[g for g in range(free.ngeom) if ids['geom_finger'][g]==finger and
                  (free.geom_contype[g] or free.geom_conaffinity[g])] for finger in range(5)]
    low=np.array([.061,.061,.061,.061,.25,.0265,.083,.035,-.015,.04,-1.05])
    high=np.array([.092,.092,.092,.092,1.74,.0378,.108,.10,.025,.09,1.05])
    center=np.array([.073,.075,.075,.073,1.40,.031,.094,.065,.005,.065,.7854]);spread=(high-low)*.22
    count=0;history=[]
    try:
        for generation in range(30):
            batch=[]
            for j in range(48):
                if time.monotonic()>deadline:raise SearchTimeout
                x=np.clip(center+rng.normal(size=len(low))*spread,low,high)
                if generation==0 and j==0:x=center.copy()
                yaw=x[10];r=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
                quat=np.empty(4);mujoco.mju_mat2Quat(quat,(basis@r).ravel())
                position=palm_origin+basis@x[7:10]
                fixed.body_pos[cb]=position;fixed.body_quat[cb]=quat
                mujoco.mj_setConst(fixed,fd);mujoco.mj_resetData(fixed,fd)
                fd.qpos[:]=HOME_QPOS;fd.ctrl[:]=x[:7];mujoco.mj_step(fixed,fd,nstep=400)
                d=mujoco.MjData(free);d.qpos[:qa]=fd.qpos;d.qpos[qa:qa+7]=np.r_[position,quat]
                d.ctrl[:]=x[:7];mujoco.mj_forward(free,d)
                c=native_contacts(free,d,ids)
                gaps=np.array([min(mujoco.mj_geomDistance(free,d,ids['cube_geom'],g,.12,None) for g in gs) for gs in finger_geoms])
                score=float(10*np.minimum(c['finger_forces']/.15,1).sum()-100*np.maximum(gaps,0).sum()
                            -2000*max(0,c['max_penetration']-.0009)-100*c['forbidden_support'])
                count+=1;batch.append((score,x))
                row=dict(candidate=count,score=score,forces=c['finger_forces'].tolist(),parameters=x.tolist())
                if np.all(c['finger_forces']>=.02) and c['max_penetration']<=.001 and not c['forbidden_support']:
                    check,end,_=validate(free,d.qpos,d.ctrl,deadline=deadline);row['free']=check
                    if check['passed']:
                        # Two settling periods remove initialization transients.
                        for _ in range(2):
                            check,end,_=validate(free,end.qpos,d.ctrl,deadline=deadline)
                        if check['passed']:
                            report=approve(output,config,end.qpos.copy(),d.ctrl.copy(),
                                           dict(method='temporary_static_cube_then_free_release_and_settle',seed=seed,
                                                candidates=count,parameters=x.tolist(),fixture_seconds=.8,
                                                free_settling_seconds=9.,physics_unchanged=True),deadline)
                            if report['approved']:return report
                history.append(row)
            elites=np.stack([x for _,x in sorted(batch,key=lambda item:item[0],reverse=True)[:10]])
            center=.3*center+.7*elites.mean(0)
            spread=np.maximum(.4*spread+.6*elites.std(0),(high-low)*.025)
    except SearchTimeout:pass
    report=dict(approved=False,scope='No validated five-finger envelope in budget',candidates=count,
                config=config.to_dict(),model_fingerprint=model_fingerprint(config),seed=seed,budget_seconds=seconds)
    (output/'search_history.json').write_text(json.dumps(history,indent=2))
    (output/'prepare_report.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--candidate');p.add_argument('--seconds',type=float,default=300.);p.add_argument('--seed',type=int,default=73)
    args=p.parse_args();output=Path(args.output)
    if (output/'reference.npz').exists():raise FileExistsError('Use a new output directory')
    config=CubeReorientConfig()
    if args.candidate:
        path=Path(args.candidate)
        with np.load(path,allow_pickle=False) as data:qpos=data['qpos'].copy();ctrl=data['ctrl'].copy()
        report=approve(output,config,qpos,ctrl,dict(method='independent_free_validation_of_candidate',source=str(path.resolve()),
                                                  source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    else:report=search(output,config,args.seconds,args.seed)
    print(json.dumps(report,indent=2));return 0 if report['approved'] else 2


if __name__=='__main__':raise SystemExit(main())
