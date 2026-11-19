import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from pathlib import Path
import argparse,json
import numpy as np
from experiments.cube_contact_recovery_v2.env import NativeEnv
from experiments.cube_adaptive_60h.config import Profile
from experiments.cube_adaptive_60h.storage import atomic_json
from .roles import wrenches,support_state,driving_quality


def probe(reference,output):
    controls={'hold':np.zeros(7)}
    for motor in (3,4,5,6):
        for sign in (-1.,1.):
            action=np.zeros(7);action[motor]=sign*.3;controls[f'motor_{motor}_{sign:+g}']=action
    for thumb in (4,5,6):
        for t in (-.3,.3):
            for p in (-.3,.3):
                action=np.zeros(7);action[thumb]=t;action[3]=p;controls[f'thumb_{thumb}_{t:+g}_pinky_{p:+g}']=action
    records=[]
    for dt in (.001,.002):
        env=NativeEnv(reference,'x',physical_dt=dt,target_degrees=90.,profile=Profile())
        try:
            for name,action in controls.items():
                obs,info=env.reset(seed=6100000)
                for _ in range(5):obs,_,done,_,info=env.step(np.zeros(7))
                initial=info;trace=[]
                for _ in range(10):
                    obs,_,done,_,info=env.step(action);w=wrenches(env)
                    trace.append(dict(info=info,moments=w['axis_moments_Nm'].tolist(),speeds=w['tangential_speeds_m_s'].tolist(),
                        three_support=support_state(info['finger_forces']),quality=[driving_quality(w['axis_moments_Nm'][i],w['tangential_speeds_m_s'][i]) for i in (0,4)]))
                    if done:break
                records.append(dict(name=name,dt=dt,action=action.tolist(),initial=initial,final=info,trace=trace,
                    peak=info['max_progress_degrees'],support_fraction=float(np.mean([x['three_support'] for x in trace])),
                    average_moments=np.mean([x['moments'] for x in trace],axis=0).tolist(),
                    maximum_driver_quality=np.max([x['quality'] for x in trace],axis=0).tolist()))
        finally:env.close()
    atomic_json(output,dict(description='Real actuator pulses from original prepared grasp; index/middle/ring targets held, no external force, no dynamics changes.',records=records))
    for row in sorted(records,key=lambda r:(r['support_fraction'],r['peak']),reverse=True)[:10]:
        print({k:row[k] for k in ('name','dt','peak','support_fraction','average_moments','maximum_driver_quality')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args();probe(a.reference,a.output)
