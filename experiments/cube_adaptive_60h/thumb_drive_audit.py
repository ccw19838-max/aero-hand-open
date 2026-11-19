"""Short physical thumb pulses with the other four motors held at their targets.

Diagnostic only: every trial replays the same original grasp/policy prefix;
no exported demonstrations, curriculum resets or proof of a complete turn.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
from datetime import datetime, timezone
import argparse
import itertools
import numpy as np
import torch
from .env import AdaptiveEnv
from .config import Profile
from .storage import load, atomic_json
from .thumb_drive import ThumbDriveReward


def audit(checkpoint, output):
    torch.set_num_threads(1)
    model, metadata = load(checkpoint)
    records = []; actuators = []
    actions = [('zero', np.zeros(3))]
    actions += [(f'motor_{i}_{sign:+d}', sign * .5 * np.eye(3)[i]) for i in range(3) for sign in (-1, 1)]
    actions += [(f'combo_{i}', np.array(a) * .5) for i, a in enumerate(itertools.product((-1, 1), repeat=3))]
    for dt in (.001, .002):
        env = ThumbDriveReward(AdaptiveEnv(Path(checkpoint) / 'reference.npz', 'x', physical_dt=dt,
                                          target_degrees=90., profile=Profile(**metadata['profile'])))
        base = env.unwrapped
        actuators = [base.model.actuator(i).name for i in range(7)]
        try:
            for prefix_angle in (0., 25.):
                for label, thumb in actions:
                    obs, info = env.reset(seed=3100007)
                    for _ in range(200):
                        if not base.warming and info['signed_progress_degrees'] >= prefix_angle:
                            break
                        obs, _, done, _, info = env.step(model.predict(obs, deterministic=True)[0])
                        if done:
                            break
                    if base.monitor.done:
                        records.append(dict(dt=dt, prefix_angle=prefix_angle, action=label, prefix_failed=True))
                        continue
                    start = info['signed_progress_degrees']; start_error = info['angle_error_degrees']
                    command = np.r_[np.zeros(4), thumb].astype(np.float32)
                    trace = []
                    for _ in range(6):
                        obs, _, done, _, info = env.step(command)
                        trace.append(info)
                        if done:
                            break
                    records.append(dict(dt=dt, prefix_angle=prefix_angle, action=label,
                        thumb_command=thumb.tolist(), start_angle=start, final_angle=info['signed_progress_degrees'],
                        signed_increment=info['signed_progress_degrees']-start,
                        full_pose_improvement=start_error-info['angle_error_degrees'],
                        failure=info['failure_reason'], success=info['success'],
                        final_forces=info['finger_forces'],
                        max_thumb_moment=max((r.get('thumb_axis_moment_Nm', 0.) for r in trace), default=0.),
                        max_drive_quality=max((r.get('thumb_drive_quality', 0.) for r in trace), default=0.), trace=trace))
        finally:
            env.close()
    result=dict(created_at=datetime.now(timezone.utc).isoformat(), checkpoint=str(checkpoint), seed=3100007,
        scope='0.3s local thumb pulses, four other motor targets held; from original grasp or replayed 25deg prefix; not trained-policy success',
        actuators=actuators, records=records)
    atomic_json(output,result)
    for dt in (.001,.002):
        for angle in (0.,25.):
            selected=[r for r in records if r['dt']==dt and r['prefix_angle']==angle and not r.get('prefix_failed')]
            best=max(selected,key=lambda r:r['full_pose_improvement'],default={})
            print({k:v for k,v in best.items() if k!='trace'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();audit(Path(args.checkpoint).resolve(),args.output)
