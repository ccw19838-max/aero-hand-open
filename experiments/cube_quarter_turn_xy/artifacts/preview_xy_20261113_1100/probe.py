"""Independent diagnostic: exact saved trajectory then unchanged final control.

No learning, no physical parameter changes, and no forced object motion.
The extra holding duration is an evaluation intervention and is labeled as such.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json
import numpy as np
from experiments.cube_quarter_turn_v2.env import TeacherEnv
from experiments.cube_quarter_turn_v2.storage import atomic_json,implementation_digest

OUT=Path(__file__).resolve().parent


def probe(axis,dt):
    folder=OUT/axis
    meta=json.loads((folder/'candidate.json').read_text())
    with np.load(folder/'candidate.npz',allow_pickle=False) as data:actions=data['actions'].copy()
    env=TeacherEnv(folder/'reference.npz',axis,physical_dt=dt)
    trace=[]
    try:
        _,info=env.reset(seed=0,options={'reference':True})
        done=False
        for action in actions:
            _,_,done,_,info=env.step(action);trace.append({'extension':False,**info})
            if done:break
        endpoint=info.copy()
        if dt==.002:
            difference=max(abs(info[k]-meta['final'][k]) for k in
                ('signed_progress_degrees','max_progress_degrees','angle_error_degrees','max_stable_seconds','elapsed_seconds'))
            if difference>1e-7:raise RuntimeError('Exact replay differs from saved source: '+str(difference))
        # Hold the same targets; let normal success/failure/stagnation gates run.
        for _ in range(60):
            if done:break
            _,_,done,_,info=env.step(actions[-1]);trace.append({'extension':True,**info})
        return dict(axis=axis,dt=dt,original_endpoint=endpoint,extension_seconds=info['elapsed_seconds']-endpoint['elapsed_seconds'],
            final=info,success=info['success'],trace=trace,implementation_sha256=implementation_digest())
    finally:env.close()


if __name__=='__main__':
    result={}
    for axis in 'xy':
        result[axis]=[]
        for dt in (.002,.001):
            record=probe(axis,dt);result[axis].append(record)
            print(json.dumps(dict(axis=axis,dt=dt,original_hold=record['original_endpoint']['max_stable_seconds'],
                extension_seconds=record['extension_seconds'],success=record['success'],
                max_hold=record['final']['max_stable_seconds'],error=record['final']['angle_error_degrees'],
                failure=record['final']['failure_reason'])),flush=True)
    atomic_json(OUT/'holding_probe.json',result)
