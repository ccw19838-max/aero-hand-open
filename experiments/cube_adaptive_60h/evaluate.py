from collections import Counter
import time
import numpy as np
from .env import AdaptiveEnv


def evaluate(model,reference,axis,profile,target=90.,episodes=20,seed=1000000,deadline=float('inf'),should_stop=lambda:False,trace=False):
    groups={};records=[];zeros=np.zeros(5);samples=0
    for dt in (.001,.002):
        env=AdaptiveEnv(reference,axis,physical_dt=dt,target_degrees=target,profile=profile,training=False);group=[]
        try:
            for index in range(episodes):
                if time.time()>=deadline or should_stop():break
                obs,info=env.reset(seed=seed+index);frames=[];interrupted=False
                for _ in range(250):
                    if time.time()>=deadline or should_stop():interrupted=True;break
                    obs,_,done,_,info=env.step(model.predict(obs,deterministic=True)[0])
                    zeros+=np.asarray(info['finger_forces'])<.02;samples+=1
                    if trace:frames.append(info)
                    if done:break
                record=dict(seed=seed+index,dt=dt,success=bool(info['success'] and not interrupted),
                    interrupted=interrupted,final=info)
                if trace:record['trace']=frames
                records.append(record);group.append(record)
                if interrupted:break
        finally:env.close()
        groups[str(dt)]=dict(trials=len(group),requested_trials=episodes,successes=sum(r['success'] for r in group),
            success_rate=sum(r['success'] for r in group)/max(1,len(group)),
            complete=len(group)==episodes and not any(r['interrupted'] for r in group),
            failures=dict(Counter(r['final']['failure_reason'] for r in group if not r['success'])))
    finals=[r['final'] for r in records]
    return dict(axis=axis,target_degrees=target,from_original_initial_grasp=True,
        complete=all(g['complete'] for g in groups.values()),groups=groups,
        success_rate=min(g['success_rate'] for g in groups.values()),
        mean_best_error=float(np.mean([r['best_angle_error_degrees'] for r in finals])) if finals else 90.,
        five_finger_duty=float(np.mean([r['five_finger_duty'] for r in finals])) if finals else 0.,
        zero_fraction=(zeros/max(1,samples)).tolist(),
        maximum_signed_angle=max((f['max_progress_degrees'] for f in finals),default=0.),
        maximum_stable_seconds=max((f['max_stable_seconds'] for f in finals),default=0.),records=records,
        verified=bool(target==90. and episodes>=100 and all(g['complete'] and g['successes']==g['requested_trials'] for g in groups.values())))


def compact(report):return {k:v for k,v in report.items() if k!='records'}


def score(report):
    if not report or not report['complete']:return (-1.,-999.,-1.)
    return (report['success_rate'],-report['mean_best_error'],report['five_finger_duty'])
