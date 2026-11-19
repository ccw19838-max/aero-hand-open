import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,shutil
import numpy as np
from experiments.cube_adaptive_60h.config import Profile
from experiments.cube_adaptive_60h.storage import atomic_json,digest
from .evaluation import evaluate,compact
from .env import VERSION


class Hold:
    def predict(self,obs,deterministic=True):return np.zeros(7,dtype=np.float32),None


def prepare(root,source,previous,probe):
    root=Path(root).resolve();source=Path(source).resolve();previous=Path(previous).resolve()
    if root.exists():raise ValueError('Do not overwrite an existing experiment')
    previous_job=json.loads((previous/'job.json').read_text());deadline=previous_job['deadline_at']
    if datetime.fromisoformat(deadline)<=datetime.now(timezone.utc):raise ValueError('Existing budget expired')
    (root/'inputs').mkdir(parents=True);shutil.copytree(source,root/'legacy_source')
    for name in ('reference.npz','prepare_report.json'):shutil.copy2(source/name,root/'inputs'/name)
    shutil.copy2(probe,root/'actuator_probe.json')
    profile=Profile(name='thumb_pinky_three_support_v3',learning_rate=1e-4,gamma=.999,entropy=.001,
        contact_weight=1.,loss_weight=1.,loss_grace=.15,pressure_potential_gain=.5,stall_weight=.5,progress_gain=20.,grip_mix=0.,bc_weight=0.)
    holds=[]
    for seed in (6200000,6210000):
        report=evaluate(Hold(),root/'inputs',profile,0.,20,seed)
        holds.append(report);atomic_json(root/f'initial_hold_{seed}.json',report)
        print('hold',seed,report['success_rate'],flush=True)
    qualified=all(r['complete'] and r['success_rate']>=.8 for r in holds)
    now=datetime.now(timezone.utc)
    job=dict(task=VERSION,root=str(root),source_checkpoint=str(root/'legacy_source'),profile=profile.dictionary(),
        started_at=now.isoformat(),deadline_at=deadline,original_budget_started_at=previous_job['started_at'],previous_experiment=str(previous),
        budget_policy='Continue remaining original 24h window including pause and preparation; no deadline reset',
        physics_timesteps=[.001,.002,.001,.002],initial_stage=5. if qualified else 0.,rollout_steps=256,
        initial_action_mean=[0.,0.,0.,.15,-.15,0.,0.] if qualified else [0.]*7,
        initial_std=[.04,.04,.04,.12,.12,.08,.08],max_std=[.1,.1,.1,.2,.2,.15,.15],
        seed=6300000,validation_seed=6400000,review_seed=6500000,final_seed=6600000,
        initial_eval_episodes=20,save_seconds=300.,eval_seconds=1800.,review_seconds=10800.,
        final_goal='Original initial grasp X+90±5deg and full orientation error<=5deg, all five fingers loaded and low speeds for1s; 100/100 at dt1ms and2ms. Named index/middle/ring support condition enforced during motion.',
        static_hold_validation=[compact(r) for r in holds],preserved_y_manifest=previous_job['preserved_y_manifest'])
    atomic_json(root/'job.json',job);print(json.dumps(dict(prepared=str(root),initial_stage=job['initial_stage'],deadline=deadline)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--previous',required=True);p.add_argument('--probe',required=True)
    a=p.parse_args();prepare(a.root,a.source,a.previous,a.probe)
