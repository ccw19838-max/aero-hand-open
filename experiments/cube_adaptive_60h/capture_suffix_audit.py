"""Replay an actual diagnostic prefix, then test physical capture commands."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np
from .env import AdaptiveEnv
from .config import Profile
from .storage import atomic_json


def audit(goal):
    root=Path(goal);state=json.loads((root/'controller_state.json').read_text());profile=Profile(**state['axes']['x']['profile'])
    source=root/'mpc_diagnosis/x10_seed0';output=root/'capture_suffix_audit';output.mkdir(exist_ok=True)
    with np.load(source/'actual_trajectory.npz') as data:
        original_actions=data['actions'].copy();original_qpos=data['qpos'].copy()
    records=[]
    for cut,mode in [(28,'zero'),(34,'zero'),(28,'finger_guard')]:
        for dt in (.001,.002):
            env=AdaptiveEnv(root/'sources/x/reference.npz','x',physical_dt=dt,target_degrees=10.,profile=profile)
            actual_actions=[];observations=[];trace=[];qs=[]
            try:
                obs,info=env.reset(seed=0);prefix_identical=None
                for step in range(250):
                    if step<cut:action=original_actions[step]
                    else:
                        action=np.zeros(7)
                        if mode=='finger_guard':
                            # A negative-tendon correction candidate. Coupled
                            # contact forces need not vary monotonically; the
                            # unchanged monitor checks the actual outcome.
                            action[:4]=np.where(env.contacts['finger_forces'][1:]<.3,-.1,0.)
                    observations.append(obs.copy());actual_actions.append(action.copy())
                    obs,_,done,_,info=env.step(action);trace.append(info);qs.append(env.data.qpos.copy())
                    if step==cut-1 and dt==.001:prefix_identical=bool(np.array_equal(env.data.qpos,original_qpos[cut-1]))
                    if done:break
                record=dict(observed_at=datetime.now(timezone.utc).isoformat(),cut=cut,mode=mode,dt=dt,
                    source=str(source),prefix_identical_at1ms=prefix_identical,success=bool(info['success']),
                    controller='actual shooting prefix + physical capture commands; not a trained policy',final=info)
                label=f'cut{cut}_{mode}_dt{int(dt*1000)}ms'
                np.savez_compressed(output/(label+'.npz'),observations=np.asarray(observations),actions=np.asarray(actual_actions),qpos=np.asarray(qs))
                atomic_json(output/(label+'.json'),record);atomic_json(output/(label+'_trace.json'),trace);records.append(record)
                print(json.dumps({k:record[k] for k in ['cut','mode','dt','prefix_identical_at1ms','success']}|{k:info[k] for k in ['failure_reason','signed_progress_degrees','max_stable_seconds','finger_forces']},ensure_ascii=False),flush=True)
            finally:env.close()
    atomic_json(output/'report.json',dict(scope='Six selected physical replay tests; not a trained-policy success rate',records=records))
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--goal',required=True);args=p.parse_args();audit(args.goal)
