"""Bounded, unassisted control search to diagnose the current quarter-turn task.

No training code, checkpoint, reward, contact constraint or physical parameter is
changed. The full-range ablation only changes the action-to-target mapping in
this separate process. Every candidate uses the real environment step/checks.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import multiprocessing as mp
import os
from pathlib import Path
import time
import json

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')

import numpy as np
from .env import QuarterTurnEnv
from . import env as environment_module
from .task import AXES
from .checkpoints import atomic_json,load

DURATIONS=(4,6,6,24) # 0.2s release, 0.3s drive, 0.3s transfer, 1.2s capture.


def mapped_step(env,action,mapping):
    old=environment_module.SCALE
    try:
        if mapping=='full_range':
            action=np.clip(np.asarray(action),-1.,1.)
            environment_module.SCALE=np.where(action>=0,env.ids['ctrl_upper']-env.default,
                                               env.default-env.ids['ctrl_lower'])
        return env.step(action)
    finally:environment_module.SCALE=old


def trial(env,axis,mapping,knots,seed=0):
    obs,info=env.reset(seed=seed,options={'axis':axis,'reference':True,'allow_bank':False})
    for _ in range(5):
        obs,reward,done,_,info=mapped_step(env,np.zeros(7),mapping)
        if done:return dict(score=-100.,peak_valid_progress=0.,final_info=info,knots=knots.tolist())
    peak=0.;best_error=90.;peak_time=0.;end=0.;max_force=0.;max_depth=0.
    for knot,duration in zip(knots,DURATIONS):
        for _ in range(duration):
            obs,reward,done,_,info=mapped_step(env,knot,mapping)
            valid=(not info['failure_reason'] and env.contacts['hand_force']>=env.config.contact_force
                   and env.contacts['max_penetration']<=env.config.max_penetration
                   and env.displacement<=env.config.max_displacement)
            max_force=max(max_force,float(sum(env.contacts['finger_forces'])))
            max_depth=max(max_depth,env.contacts['max_penetration'])
            end=min(max(0.,90.-info['angle_error_degrees']),max(0.,min(90.,info['signed_progress_degrees']))) if valid else 0.
            if end>peak:peak=end;peak_time=info['goal_elapsed_seconds'];best_error=info['angle_error_degrees']
            if done:break
        if done:break
    score=peak+.25*end+3.*info['max_stable_seconds']-(8. if info['failure_reason'] else 0.)
    return dict(score=float(score),peak_valid_progress=float(peak),peak_time=float(peak_time),
                error_at_peak=float(best_error),final_valid_progress=float(end),
                max_finger_normal_force_sum=float(max_force),max_penetration=float(max_depth),
                final_info=info,knots=knots.tolist())


def structured_candidates():
    patterns=[np.zeros((4,7))]
    for fingers in [(0,1),(2,3),(0,1,2,3)]:
        for sweep in (-1.,1.):
            p=np.zeros((4,7));p[0,list(fingers)]=.7;p[0,4]=-.3
            p[1,list(fingers)]=.8;p[1,4]=sweep*.8;p[1,5]=sweep*.5;p[1,6]=-sweep*.5
            p[2,list(fingers)]=.3;p[2,4]=sweep*.4
            patterns.append(p)
    return patterns


def search(job):
    reference,axis,mapping,output,deadline,population,iterations=job
    env=QuarterTurnEnv(reference,axis=axis);rng=np.random.default_rng(34000+AXES.index(axis))
    mean=np.zeros((4,7));std=np.full((4,7),.65);rows=[];best=None;best_legal=None
    try:
        for generation in range(iterations):
            candidates=np.clip(rng.normal(mean,std,size=(population,4,7)),-1.,1.)
            for i,pattern in enumerate(structured_candidates()[:population//2]):candidates[i]=pattern
            # Preserve broad coherent exploration instead of collapsing around
            # the stationary clamp when early samples all fail.
            count=max(1,population//4);candidates[-count:]=rng.uniform(-1,1,(count,4,7))
            generation_rows=[]
            for candidate in candidates:
                if time.monotonic()>=deadline:break
                result=trial(env,axis,mapping,candidate);result.update(generation=generation,candidate=len(rows))
                rows.append(result);generation_rows.append(result)
                if best is None or result['score']>best['score']:best=result
                if not result['final_info']['failure_reason'] and (best_legal is None or result['score']>best_legal['score']):best_legal=result
            if not generation_rows:break
            elite=sorted(generation_rows,key=lambda r:r['score'],reverse=True)[:max(4,len(generation_rows)//5)]
            elite_knots=np.array([r['knots'] for r in elite]);mean=.3*mean+.7*elite_knots.mean(axis=0)
            std=np.maximum(.25,.3*std+.7*elite_knots.std(axis=0))
            atomic_json(Path(output)/(axis+'_'+mapping+'.json'),dict(axis=axis,mapping=mapping,
                candidates=len(rows),completed_generations=generation+1,best=best,best_legal=best_legal,
                successes=sum(r['final_info']['success'] for r in rows),durations_control_steps=DURATIONS,
                geometry_and_failure_rules='Unmodified original environment; only action mapping differs',
                rows=rows))
            print(json.dumps(dict(axis=axis,mapping=mapping,generation=generation,candidates=len(rows),
                peak=best['peak_valid_progress'],legal_peak=None if best_legal is None else best_legal['peak_valid_progress'])),flush=True)
            if time.monotonic()>=deadline:break
    finally:env.close()
    return dict(axis=axis,mapping=mapping,candidates=len(rows),best=best,best_legal=best_legal,
                successes=sum(r['final_info']['success'] for r in rows))


def audit(reference,run,output):
    import torch
    torch.set_num_threads(1)
    env=QuarterTurnEnv(reference);obs,info=env.reset(seed=0,options={'reference':True,'axis':'z'})
    lo=env.ids['ctrl_lower'];hi=env.ids['ctrl_upper'];base=env.default;scale=environment_module.SCALE
    audit=dict(created_at=datetime.now(timezone.utc).isoformat(),reference=str(Path(reference).resolve()),
        cube_weight_newtons=env.config.cube_mass*9.81,
        initial_finger_normal_force_sum=float(sum(env.contacts['finger_forces'])),
        note='Normal contact force sum is not the resultant gripping force or a direct friction margin.',
        actuator_ranges=[dict(channel=i+1,lower=float(lo[i]),upper=float(hi[i]),reference=float(base[i]),
            current_lower=float(max(lo[i],base[i]-scale[i])),current_upper=float(min(hi[i],base[i]+scale[i])),
            reachable_fraction=float((min(hi[i],base[i]+scale[i])-max(lo[i],base[i]-scale[i]))/(hi[i]-lo[i]))) for i in range(7)],
        deterministic_rollouts=[])
    try:
        for axis in AXES:
            checkpoint=(Path(run)/('best_'+axis)).resolve();model,metadata=load(checkpoint)
            for mapping in ('current','full_range'):
                obs,info=env.reset(seed=0,options={'reference':True,'axis':axis,'allow_bank':False})
                samples=[]
                for step in range(260):
                    action=model.predict(obs,deterministic=True)[0]
                    obs,reward,done,_,info=mapped_step(env,action,mapping)
                    samples.append(dict(action=info['applied_action'],forces=info['finger_forces'],
                        turn=info['signed_progress_degrees'],error=info['angle_error_degrees'],
                        stable=info['stable_seconds']))
                    if done:break
                audit['deterministic_rollouts'].append(dict(axis=axis,mapping=mapping,
                    checkpoint=str(checkpoint.resolve()),checkpoint_steps=metadata['training_steps'],
                    final_info=info,samples=samples))
    finally:env.close()
    atomic_json(Path(output)/'audit.json',audit)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True);parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--minutes',type=float,default=7.);parser.add_argument('--population',type=int,default=32)
    parser.add_argument('--iterations',type=int,default=3);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    if not 0<args.minutes<=10 or not 1<=args.workers<=2:parser.error('Bound diagnosis to <=10 minutes and <=2 workers')
    audit(args.reference,args.run,output)
    deadline=time.monotonic()+args.minutes*60
    jobs=[(args.reference,a,m,str(output),deadline,args.population,args.iterations) for a in AXES for m in ('current','full_range')]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as executor:
        results=list(executor.map(search,jobs))
    atomic_json(output/'summary.json',dict(results=results,
        scope='Bounded search from one physical reference reset. Not a success-rate estimate or a proof of global mechanical impossibility.',
        constraints='Original gravity, mass, friction, motor limits, contact/failure checks; no object forces or pose edits during a trajectory.'))
    print(json.dumps({'finished':True,'output':str(output)}),flush=True)


if __name__=='__main__':main()
