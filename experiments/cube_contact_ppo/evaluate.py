"""Always distinguish curriculum holds/angles from final strict 90deg validation."""
from collections import Counter
import argparse
from pathlib import Path
import json
import time
import numpy as np
from .env import ContactEnv
from .storage import atomic_json,load


def evaluate(model,reference,axis,target_degrees=90.,episodes=20,seed=1000000,
             deadline=float('inf'),should_stop=lambda:False,randomization=0.):
    groups={};records=[]
    for dt in (.001,.002):
        group=[];env=ContactEnv(reference,axis,physical_dt=dt,target_degrees=target_degrees,randomization=randomization)
        try:
            for index in range(episodes):
                if time.time()>=deadline or should_stop():break
                obs,info=env.reset(seed=seed+index);trace=[];interrupted=False
                for _ in range(250):
                    if time.time()>=deadline or should_stop():interrupted=True;break
                    obs,_,done,_,info=env.step(model.predict(obs,deterministic=True)[0]);trace.append(info)
                    if done:break
                record=dict(seed=seed+index,dt=dt,success=bool(info['success'] and not interrupted),
                    interrupted=interrupted,final=info,trace=trace)
                records.append(record);group.append(record)
                if interrupted:break
        finally:env.close()
        groups[str(dt)]=dict(episodes=len(group),requested_episodes=episodes,
            complete=len(group)==episodes and not any(r['interrupted'] for r in group),
            success_rate=sum(r['success'] for r in group)/max(1,len(group)),
            failures=dict(Counter(r['final']['failure_reason'] for r in group if not r['success'])))
    finals=[r['final'] for r in records]
    return dict(axis=axis,target_degrees=target_degrees,final_goal_evaluation=target_degrees==90.,
        groups=groups,complete=all(g['complete'] for g in groups.values()),
        min_integrator_success_rate=min(g['success_rate'] for g in groups.values()),
        mean_best_angle_error_degrees=float(np.mean([f['best_angle_error_degrees'] for f in finals])) if finals else 90.,
        mean_five_finger_duty=float(np.mean([f['five_finger_duty'] for f in finals])) if finals else 0.,
        mean_max_stable_seconds=float(np.mean([f['max_stable_seconds'] for f in finals])) if finals else 0.,
        per_finger_zero_contact_fraction=(np.mean([np.array(t['finger_forces'])<.02 for r in records for t in r['trace']],axis=0).tolist()
                                       if any(r['trace'] for r in records) else [0.]*5),
        records=records,final_task_verified=False)


def compact(report):return {k:v for k,v in report.items() if k!='records'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--episodes',type=int,default=100);p.add_argument('--target',type=float,default=90.)
    args=p.parse_args();model,meta=load(args.checkpoint)
    report=evaluate(model,Path(args.checkpoint).resolve()/'reference.npz',meta['axis'],args.target,args.episodes,seed=50000000)
    atomic_json(Path(args.output)/'evaluation.json',report);print(json.dumps(compact(report)))


if __name__=='__main__':main()
