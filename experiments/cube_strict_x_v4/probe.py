"""Paired real-actuator pulses from the prepared initial grasp."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from experiments.cube_adaptive_60h.config import Profile
from experiments.cube_adaptive_60h.storage import atomic_json, digest
from experiments.cube_two_driver_v3.env import NativeEnv
from experiments.cube_two_driver_v3.roles import support_state


PULSES = {
    'hold': (0., 0.),
    'thumb_only': (-.3, 0.),
    'pinky_only': (0., .3),
    'both': (-.3, .3),
}


def _pulse(env, seed, name):
    env.reset(seed=seed)
    for _ in range(5):
        _, _, done, _, _ = env.step(np.zeros(7))
        if done:
            raise RuntimeError(f'Prepared grasp ended during warmup at seed {seed}')
    assert not env.warming
    state_hash = hashlib.sha256()
    for state in (env.data.qpos, env.data.qvel, env.data.ctrl):
        state_hash.update(np.ascontiguousarray(state, dtype='<f8').tobytes())
    cube = env.ids['cube_body']
    starting_center = env.data.xipos[cube].copy()
    start = float(env.data.time)
    trace = []
    monitor = env.monitor
    original = monitor.checkpoint

    def sampled(q, contacts, linear, angular, displacement, dt=.01):
        value = original(q, contacts, linear, angular, displacement, dt)
        force = contacts['finger_forces']
        trace.append(dict(
            t=float(env.data.time-start), force_N=force.tolist(),
            moment_Nm=monitor.last_moments.tolist(),
            tangent_speed_m_s=monitor.last_speeds.tolist(),
            com_offset_m=(env.data.xipos[cube]-starting_center).tolist(),
            angle_deg=float(np.rad2deg(monitor.progress)), three_support=support_state(force),
        ))
        return value

    monitor.checkpoint = sampled
    action = np.zeros(7, dtype=np.float32)
    action[4], action[3] = PULSES[name]
    try:
        for _ in range(10):
            _, _, done, _, info = env.step(action)
            if done:
                break
    finally:
        del monitor.checkpoint

    if not trace:
        raise RuntimeError('Missing 10 ms force samples in pulse')
    forces = np.asarray([sample['force_N'] for sample in trace])
    return dict(name=name, seed=seed, dt=env.physical_dt, action=action.tolist(),
                post_warmup_sha256=state_hash.hexdigest(), samples=len(trace),
                active_seconds=float(env.data.time-start), failure_reason=info['failure_reason'],
                terminal_angle_deg=info['signed_progress_degrees'], peak_angle_deg=info['max_progress_degrees'],
                mean_moment_Nm=np.mean([s['moment_Nm'] for s in trace],axis=0).tolist(),
                minimum_force_N=forces.min(axis=0).tolist(), maximum_force_N=forces.max(axis=0).tolist(),
                all_three_supported=all(s['three_support'] for s in trace),
                maximum_com_displacement_m=float(max(np.linalg.norm(s['com_offset_m']) for s in trace)),trace=trace)


def probe(reference, output, seed=6100000, episodes=3, deadline=float('inf')):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    rows = []
    reference = Path(reference)
    for dt in (.001, .002):
        env = NativeEnv(reference, 'x', physical_dt=dt, target_degrees=90., profile=Profile(grip_mix=0.))
        try:
            for initial_seed in range(seed, seed + episodes):
                matched = []
                for name in PULSES:
                    if time.time() >= deadline:
                        raise TimeoutError('Pulse experiment reached its deadline')
                    matched.append(_pulse(env, initial_seed, name))
                if len({r['post_warmup_sha256'] for r in matched}) != 1:
                    raise RuntimeError('Pulse initial states differ within a matched set')
                rows.extend(matched)
        finally:
            env.close()
    result = dict(reference_sha256=digest(reference), initial_seed=seed,
                  matched_initial_states=True, sample_seconds=.01, signed_joint_order=['index','middle','ring','pinky','thumb_cmc','thumb_1','thumb_2'],
                  force_order=['thumb','index','middle','ring','pinky'], rows=rows)
    atomic_json(output,result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=3)
    args = parser.parse_args()
    result = probe(args.reference,args.output,episodes=args.episodes)
    print(json.dumps([{k:r[k] for k in ('name','seed','dt','peak_angle_deg','all_three_supported')}
                      for r in result['rows'] if r['seed']==result['initial_seed']],indent=2))
