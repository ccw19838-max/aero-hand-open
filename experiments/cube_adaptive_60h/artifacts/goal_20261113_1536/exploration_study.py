"""Isolated sampling-noise comparison on known successful teacher initial states.

This is not an unbiased overall success-rate evaluation and does not change live
training. Identical initial states and random draws are paired across sigmas.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
from pathlib import Path
import json
import numpy as np
import torch
from experiments.cube_adaptive_60h.storage import load,atomic_json
from experiments.cube_adaptive_60h.env import AdaptiveEnv
from experiments.cube_adaptive_60h.config import Profile

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    protected=json.loads((ROOT/'protected_best_y30.json').read_text())
    model,meta=load(protected['path']);profile=Profile(**meta['profile'])
    baseline=json.loads(Path(protected['report']).read_text())
    seeds=[r['seed'] for r in baseline['records'] if r['success'] and r['dt']==.001][:6]
    original=model.policy.log_std.detach().clone();results=[]
    for sigma in (0.,.03,.08,.25):
        if sigma:
            with torch.no_grad():model.policy.log_std.fill_(float(np.log(sigma)))
        episodes=[]
        env=AdaptiveEnv(ROOT/'sources/x/reference.npz','y',physical_dt=.001,target_degrees=30.,profile=profile)
        try:
            for i,seed in enumerate(seeds):
                torch.manual_seed(800000+i);obs,info=env.reset(seed=seed)
                for _ in range(250):
                    obs,_,done,_,info=env.step(model.predict(obs,deterministic=sigma==0.)[0])
                    if done:break
                episodes.append(dict(seed=seed,success=info['success'],failure=info['failure_reason'],
                    stable_seconds=info['max_stable_seconds'],error=info['angle_error_degrees']))
        finally:env.close()
        result=dict(sigma=sigma,trials=len(episodes),successes=sum(e['success'] for e in episodes),episodes=episodes)
        results.append(result);print(json.dumps(result),flush=True)
    with torch.no_grad():model.policy.log_std.copy_(original)
    atomic_json(ROOT/'exploration_study.json',dict(
        limitation='Six selected teacher-success initial states at 1ms only; comparison isolates noise sensitivity, not general success rate.',
        live_training_changed=False,results=results))


if __name__=='__main__':main()
