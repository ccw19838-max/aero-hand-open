from pathlib import Path
from collections import Counter
import time
import numpy as np
from .env import NativeEnv,VERSION


def evaluate(model,inputs,profile,target,episodes=20,seed=6200000,deadline=float('inf'),stop=lambda:False,trace=False):
    inputs=Path(inputs);groups={};records=[]
    for dt in (.001,.002):
        env=NativeEnv(inputs/'reference.npz','x',physical_dt=dt,target_degrees=target,profile=profile);group=[]
        try:
            for index in range(episodes):
                if time.time()>=deadline or stop():break
                obs,info=env.reset(seed=seed+index);frames=[];interrupted=False
                for _ in range(250):
                    if time.time()>=deadline or stop():interrupted=True;break
                    obs,_,done,_,info=env.step(model.predict(obs,deterministic=True)[0])
                    if trace:frames.append(info)
                    if done:break
                assert info['from_original_initial_grasp'] and not info['training_prefix_used']
                row=dict(seed=seed+index,dt=dt,success=bool(info['success'] and not interrupted),interrupted=interrupted,final=info)
                if trace:row['trace']=frames
                group.append(row);records.append(row)
                if interrupted:break
        finally:env.close()
        groups[str(dt)]=dict(trials=len(group),requested_trials=episodes,successes=sum(r['success'] for r in group),
            success_rate=sum(r['success'] for r in group)/max(1,len(group)),complete=len(group)==episodes and not any(r['interrupted'] for r in group),
            failures=dict(Counter(r['final']['failure_reason'] or 'unfinished' for r in group if not r['success'])))
    return dict(task=VERSION,target_degrees=target,from_original_initial_grasp=True,
        complete=all(g['complete'] for g in groups.values()),groups=groups,
        success_rate=min(g['success_rate'] for g in groups.values()),
        mean_best_error=float(np.mean([r['final']['best_angle_error_degrees'] for r in records])) if records else 90.,
        three_support_duty=float(np.mean([r['final']['three_support_duty'] for r in records])) if records else 0.,
        five_finger_duty=float(np.mean([r['final']['five_finger_duty'] for r in records])) if records else 0.,
        maximum_signed_angle=max((r['final']['max_progress_degrees'] for r in records),default=0.),
        maximum_supported_angle=max((r['final']['supported_progress_degrees'] for r in records),default=0.),
        maximum_stable_seconds=max((r['final']['max_stable_seconds'] for r in records),default=0.),records=records)


def compact(report):return {k:v for k,v in report.items() if k!='records'}


def score(report):
    if not report or not report['complete']:return (-1.,-999.,-1.)
    return report['success_rate'],-report['mean_best_error'],report['three_support_duty']


def final_pass(report):
    return bool(report and report['task']==VERSION and report['target_degrees']==90. and report['from_original_initial_grasp']
        and report['complete'] and set(report['groups'])=={'0.001','0.002'}
        and all(g['requested_trials']>=100 and g['successes']==g['requested_trials'] for g in report['groups'].values()))
