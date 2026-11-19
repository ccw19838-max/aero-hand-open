"""Actual action prefixes and a separate budget; no physical state injection."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from pathlib import Path
from datetime import datetime,timezone,timedelta
from zoneinfo import ZoneInfo
import argparse,json,shutil
import numpy as np
import torch
from experiments.cube_adaptive_60h.storage import load as load_legacy,digest,atomic_json
from experiments.cube_adaptive_60h.config import Profile
from .env import NativeEnv,VERSION


def prepare(root,source,hours=24.):
    root=Path(root).resolve();source=Path(source).resolve()
    if not 1.<=hours<=72.:raise ValueError('Use a bounded1-72h budget')
    if (root/'job.json').exists():raise ValueError('Prepared experiment already exists; do not reset its budget')
    now=datetime.fromtimestamp(root.stat().st_mtime,timezone.utc) if root.exists() else datetime.now(timezone.utc)
    (root/'inputs').mkdir(parents=True,exist_ok=True)
    if not (root/'legacy_source').exists():shutil.copytree(source,root/'legacy_source')
    elif digest(root/'legacy_source/policy.zip')!=digest(source/'policy.zip'):raise ValueError('Preparation source changed')
    for name in ('reference.npz','prepare_report.json'):shutil.copy2(source/name,root/'inputs'/name)
    policy,meta=load_legacy(source);torch.set_num_threads(1)
    prefixes={};details=[];reference=root/'inputs/reference.npz'
    for index in range(8):
        env=NativeEnv(reference,'x',physical_dt=.001,target_degrees=90.,profile=Profile(**meta['profile']))
        obs,info=env.reset(seed=5100000+index);actions=[]
        try:
            for step in range(16):
                action=policy.predict(obs,deterministic=True)[0].astype(np.float32);actions.append(action.copy())
                obs,_,done,_,info=env.step(action)
                if done:raise ValueError('Source prefix failed before preparation endpoint')
            prefixes[f'prefix_{index:02d}']=np.asarray(actions)
            details.append(dict(index=index,source_seed=5100000+index,controls=len(actions),final=info))
        finally:env.close()
    np.savez_compressed(root/'inputs/prefix_bank.npz',**prefixes)
    checks=[]
    for dt in (.001,.002):
        env=NativeEnv(reference,'x',physical_dt=dt,target_degrees=0.,prefix_bank=root/'inputs/prefix_bank.npz')
        try:
            for seed in range(5200000,5200032):
                _,info=env.reset(seed=seed)
                checks.append(dict(seed=seed,dt=dt,valid=info['training_prefix_used'] and not info['auxiliary_setup_failure'],
                    forces=info['finger_forces'],angle=info['prefix_start_angle_degrees'],failure=info['auxiliary_setup_failure']))
        finally:env.close()
    valid=sum(r['valid'] for r in checks)
    atomic_json(root/'inputs/prefix_manifest.json',dict(source_checkpoint=str(source),source_policy_sha256=digest(source/'policy.zip'),
        prefixes=details,checks=checks,valid=valid,total=len(checks),all_states_reached_by_actual_mj_steps=True,
        formal_x90_uses_prefix=False,check_seeds_not_final_acceptance=True))
    if valid<len(checks):raise ValueError(f'Prefix preparation rejected: {valid}/{len(checks)} valid')
    profile=Profile(name='corrected_contact_recovery_v2',learning_rate=1e-4,gamma=.999,entropy=.001,
        contact_weight=2.,loss_weight=2.,loss_grace=.15,pressure_potential_gain=.5,
        stall_weight=.5,progress_gain=20.,grip_mix=.15,bc_weight=0.).dictionary()
    job=dict(task=VERSION,root=str(root),source_checkpoint=str(root/'legacy_source'),profile=profile,
        started_at=now.isoformat(),deadline_at=(now+timedelta(hours=hours)).isoformat(),hours=hours,
        physics_timesteps=[.001,.002,.001,.002],initial_stage=0.,rollout_steps=256,gae_lambda=.98,
        action_head_scale=.5,initial_std=.08,max_std=.12,seed=5300000,
        save_seconds=300.,eval_seconds=1800.,review_seconds=10800.,initial_eval_episodes=20,
        validation_seed=5400000,review_seed=5500000,final_seed=5600000,
        auxiliary_promotion_threshold=.8,auxiliary_promotion_streak=2,
        final_goal='original initial X+90±5deg, full pose error<=5deg, five loaded fingers and low speeds continuously1s; 100/100 at each dt',
        preserved_y_manifest=str((root.parent.parent.parent/'preserved_policies/y90_best_20261116_155308/manifest.json').resolve()))
    atomic_json(root/'job.json',job)
    print(json.dumps(dict(prepared=str(root),prefix_checks=f'{valid}/{len(checks)}',deadline=job['deadline_at'])),flush=True)
    return job


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--hours',type=float,default=24.)
    a=p.parse_args();prepare(a.root,a.source,a.hours)
