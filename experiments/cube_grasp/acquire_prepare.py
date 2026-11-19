"""Derive and independently gate an unassisted slip-recovery curriculum reset.

The approved parent holding pose is physically released for 50 ms by relaxing
one actuator. The exported moving state is not an open-hand pickup demonstration.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .config import CubeGraspConfig
from .env import CubeGraspEnv, GraspMonitor
from .evaluate import atomic_json, _json_value
from .model import build_model, model_ids, model_fingerprint, native_contacts, native_cube_velocity


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _initial(model, data, ids, config):
    contacts = native_contacts(model, data, ids)
    linear, angular = native_cube_velocity(model, data, ids)
    touching = contacts['finger_forces'] >= config.contact_force
    stable = bool(touching[0] and np.count_nonzero(touching) >= config.min_fingers
                  and np.linalg.norm(linear) < config.grip_velocity
                  and np.linalg.norm(angular) < config.grip_omega
                  and contacts['max_penetration'] <= config.max_penetration
                  and not contacts['forbidden_support'])
    return dict(stable=stable, penetration=float(contacts['max_penetration']),
                forbidden_support=bool(contacts['forbidden_support']),
                finger_forces=contacts['finger_forces'].tolist(),
                linear_speed=float(np.linalg.norm(linear)), angular_speed=float(np.linalg.norm(angular)))


def _nominal(candidate, config, mode, deadline):
    model = build_model(config)
    ids = model_ids(model)
    data = mujoco.MjData(model)
    data.qpos[:] = candidate['qpos']
    data.qvel[:] = candidate['qvel']
    data.ctrl[:] = candidate['ctrl']
    mujoco.mj_forward(model, data)
    origin = data.xpos[ids['cube_body']].copy()
    initial = _initial(model, data, ids, config)
    monitor = GraspMonitor(config)
    span = ids['ctrl_upper'] - ids['ctrl_lower']
    while not monitor.done:
        if time.monotonic() >= deadline:
            return dict(mode=mode, complete=False, reason='time_budget', initial=initial)
        if mode == 'closure':
            data.ctrl[:] = np.clip(data.ctrl + np.clip(candidate['grasp_ctrl'] - data.ctrl,
                                      -config.action_delta_fraction*span,
                                      config.action_delta_fraction*span), ids['ctrl_lower'], ids['ctrl_upper'])
        for _ in range(config.substeps):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            contact = native_contacts(model, data, ids)
            linear, angular = native_cube_velocity(model, data, ids)
            monitor.update(contact, linear, angular,
                           np.linalg.norm(data.xpos[ids['cube_body']] - origin),
                           data.xpos[ids['cube_body'], 2],
                           not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()))
            if monitor.done:
                break
    return dict(mode=mode, sim_dt=config.sim_dt, complete=True, success=monitor.success,
                failure_reason=monitor.failure_reason, stable_seconds=monitor.max_stable_seconds,
                elapsed_seconds=monitor.elapsed, peak_penetration=monitor.peak_penetration, initial=initial)


def _seeded(parent_reference, candidate, config, mode, episodes, deadline):
    # The parent gate is checked normally. Only this private environment instance
    # receives the candidate in memory; no candidate approval file is fabricated.
    env = CubeGraspEnv(parent_reference, config=config)
    env.reference = {key:value.copy() for key,value in candidate.items()}
    records = []
    try:
        for seed in range(episodes):
            if time.monotonic() >= deadline:
                break
            try:
                _, info = env.reset(seed=seed)
            except RuntimeError as error:
                records.append(dict(seed=seed, success=False, failure_reason='invalid_reset', error=str(error)))
                continue
            initial = _initial(env.model, env.data, env.ids, config)
            finished = False
            for _ in range(round(config.episode_seconds/config.ctrl_dt)+2):
                if time.monotonic() >= deadline:
                    break
                action = env.baseline_action('zero' if mode == 'zero' else 'close')
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    finished = True
                    break
            if not finished:
                break
            records.append(dict(seed=seed, success=bool(info['success']),
                                failure_reason=info['failure_reason'], initial=initial,
                                final_info=_json_value(info)))
            print(json.dumps(dict(baseline=mode, seed=seed, success=info['success'],
                                  failure=info['failure_reason'])), flush=True)
    finally:
        env.close()
    return dict(mode=mode, episodes_requested=episodes, episodes_completed=len(records),
                complete=len(records)==episodes,
                successes=sum(row['success'] for row in records),
                success_rate=sum(row['success'] for row in records)/len(records) if records else None,
                initial_stable_count=sum(row.get('initial', {}).get('stable', False) for row in records),
                records=records)


def prepare(reference, output, episodes=20, reset_position_noise=None,
            reset_angle_noise_degrees=None, seconds=120.):
    if episodes < 1 or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('Positive episode count and finite positive budget required')
    started = time.monotonic()
    deadline = started + seconds
    reference = Path(reference).resolve()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'reference.npz').exists():
        raise FileExistsError('Use a fresh output directory; approved references are immutable')
    parent_report = json.loads((reference.parent/'prepare_report.json').read_text())
    config = CubeGraspConfig.from_dict(parent_report['config'])
    changes = {}
    if reset_position_noise is not None:
        changes['reset_position_noise'] = reset_position_noise
    if reset_angle_noise_degrees is not None:
        changes['reset_angle_noise_degrees'] = reset_angle_noise_degrees
    config = replace(config, **changes)
    parent = CubeGraspEnv(reference, config=config)
    try:
        model, ids = parent.model, parent.ids
        source = parent.reference
        data = mujoco.MjData(model)
        data.qpos[:] = source['qpos']
        data.qvel[:] = source['qvel']
        data.ctrl[:] = source['ctrl']
        data.ctrl[2] = min(data.ctrl[2]+.015, ids['ctrl_upper'][2])
        applied_offset = float(data.ctrl[2]-source['ctrl'][2])
        mujoco.mj_forward(model, data)
        peak = 0.
        construction_support = False
        for _ in range(round(.05/config.sim_dt)):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            contact = native_contacts(model, data, ids)
            peak = max(peak, float(contact['max_penetration']))
            construction_support |= bool(contact['forbidden_support'])
        candidate = dict(qpos=data.qpos.copy(), qvel=data.qvel.copy(), ctrl=data.ctrl.copy(),
                         grasp_ctrl=source.get('grasp_ctrl', source['ctrl']).copy())
        initial = _initial(model, data, ids, config)
    finally:
        parent.close()
    construction = dict(method='free_body_partial_release', actuator_index=2,
                        actuator_name='right_ring_A_tendon', requested_control_offset=.015,
                        applied_control_offset=applied_offset, free_evolution_seconds=.05,
                        parent_reference=str(reference), parent_reference_sha256=_digest(reference),
                        parent_report_sha256=_digest(reference.parent/'prepare_report.json'),
                        peak_penetration=peak, forbidden_support=construction_support,
                        description='Only the ring target is relaxed; the free-body object and hand evolve under gravity without fixtures or applied forces. Velocities are preserved.')
    report = dict(approved=False, config=config.to_dict(), model_fingerprint=model_fingerprint(config),
                  scope='Slip recovery from a moving, partially released pregrasp; not open-hand pickup or learned-policy performance',
                  construction=construction, initial=initial, checks={},
                  validation_seed_start=0, validation_episodes=episodes,
                  thresholds=dict(closure_success_rate=.8, zero_success_rate_max=.2,
                                  max_penetration=config.max_penetration,
                                  penetration_violation_seconds=config.violation_seconds,
                                  penetration_semantics='Above the limit clears stability; continuous excess for violation_seconds fails, as defined by the shared GraspMonitor'),
                  budget_seconds=seconds)
    atomic_json(output/'prepare_report.json', report)
    atomic_json(output/'config.json', config.to_dict())
    np.savez(output/'candidate.npz', **candidate)
    initial_ok = (not initial['stable'] and not initial['forbidden_support']
                  and initial['penetration'] <= config.max_penetration
                  and peak <= config.max_penetration and not construction_support
                  and all(np.isfinite(value).all() for value in candidate.values()))
    if not initial_ok:
        report.update(reason='Released candidate does not meet initial-state requirements',
                      elapsed_seconds=time.monotonic()-started)
        atomic_json(output/'prepare_report.json', report)
        return report
    try:
        for dt in (config.sim_dt, config.sim_dt/2):
            for mode in ('zero', 'closure'):
                result = _nominal(candidate, replace(config, sim_dt=dt), mode, deadline)
                report['checks'][f'nominal_{dt}_{mode}'] = result
                atomic_json(output/'prepare_report.json', report)
        for mode in ('zero', 'closure'):
            report['checks'][f'randomized_{mode}'] = _seeded(reference, candidate, config,
                                                          mode, episodes, deadline)
            atomic_json(output/'prepare_report.json', report)
        zero = report['checks']['randomized_zero']
        closure = report['checks']['randomized_closure']
        nominal = [report['checks'][f'nominal_{dt}_{mode}']
                   for dt in (config.sim_dt, config.sim_dt/2) for mode in ('zero', 'closure')]
        nominal_ok = all(row['complete'] and row.get('success') == (row['mode']=='closure') for row in nominal)
        randomized_ok = (zero['complete'] and closure['complete'] and zero['success_rate'] <= .2
                         and closure['success_rate'] >= .8
                         and zero['initial_stable_count'] == 0 and closure['initial_stable_count'] == 0)
        report['approved'] = bool(nominal_ok and randomized_ok)
        report['reason'] = None if report['approved'] else 'Nominal or randomized baseline separation gate failed; candidate not approved'
    except Exception as error:
        report.update(approved=False, reason=f'Validation error: {type(error).__name__}: {error}')
    if report['approved']:
        temporary = output/'reference.tmp.npz'
        np.savez(temporary, **candidate)
        temporary.replace(output/'reference.npz')
        report['reference_sha256'] = _digest(output/'reference.npz')
    report['elapsed_seconds'] = time.monotonic()-started
    atomic_json(output/'prepare_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seconds', type=float, default=120.)
    parser.add_argument('--reset-position-noise', type=float, default=None,
                        help='Explicit position randomization half-width in metres; default inherits parent')
    parser.add_argument('--reset-angle-noise-degrees', type=float, default=None,
                        help='Explicit angle randomization half-width; default inherits parent')
    args = parser.parse_args()
    result = prepare(args.reference, args.output, args.episodes, args.reset_position_noise,
                     args.reset_angle_noise_degrees, args.seconds)
    print(json.dumps({key:value for key,value in result.items() if key != 'checks'}, indent=2))
    return 0 if result['approved'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
