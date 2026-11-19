"""Paired support-finger interventions before the first observed force violation.

All branches start at reset(seed) and execute actions through native physics. The
only commands changed after the fork belong to the three support tendons.
"""
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.cube_adaptive_60h.storage import atomic_json, digest

INTEGRATORS = (.001, .002)
CONTROL_PERIOD = .05
MAX_SECONDS = 600.
MIN_GAIN_DEG = .1
SCHEMA = 'preloss_support_adjustment_v1'


def implementation_hash():
    """Bind segment qualifications to the scoring and physical environment."""
    from experiments.cube_contact_recovery_v2.io import fingerprint
    sha = hashlib.sha256(fingerprint().encode())
    package = Path(__file__).resolve().parent
    for path in (package/'contact_history.py', package/'env.py', package/'stall.py',
                 package.parent/'cube_two_driver_v3/env.py', package.parent/'cube_two_driver_v3/roles.py'):
        sha.update(path.read_bytes())
    return sha.hexdigest()


class ClosedWindow(Exception):
    """The original stage budget or the caller's wall-clock deadline expired."""


def _reset(env, seed):
    obs, info = env.reset(seed=seed)
    if not info['from_original_initial_grasp'] or info['training_prefix_used']:
        raise ValueError('Replay must begin at original grasp')
    return obs, info


def _fork_candidates(first_loss_s, warmup_s, length):
    loss = first_loss_s + warmup_s
    return sorted((k for k in range(length)
                   if k*CONTROL_PERIOD + 1e-9 >= warmup_s and .1-1e-9 <= loss-k*CONTROL_PERIOD <= .2+1e-9),
                  key=lambda k:abs(loss-k*CONTROL_PERIOD-.15))


def _commands(base, triple_a, triple_b):
    out = base.copy()
    mid = (len(out) + 1) // 2
    out[:mid, :3] = triple_a
    out[mid:, :3] = triple_b
    assert np.array_equal(base[:, 3:], out[:, 3:])
    return out


def _profiles(generator):
    # Deterministic single-tendon and coordinated gentle shifts, followed by
    # two-piece commands to allow unloading and subsequent support recovery.
    for amplitude in (.18, -.18, .35, -.35):
        for direction in np.eye(3):
            action = amplitude * direction
            yield action, action
        yield np.full(3, amplitude), np.full(3, amplitude)
    for _ in range(24):
        first = generator.uniform(-.5, .5, 3)
        second = np.clip(first + generator.normal(0., .18, 3), -.6, .6)
        yield first, second


def _pose_gain(info, target):
    return min(float(info['signed_progress_degrees']), float(target - info['angle_error_degrees']))


def _physical_seed_trace(env, model, seed, guard):
    obs, info = _reset(env, seed)
    controls = []
    for _ in range(250):
        guard()
        action = np.asarray(model.predict(obs, deterministic=True)[0], dtype=np.float32).copy()
        controls.append(action)
        obs, _, done, truncated, info = env.step(action)
        lost = info['first_violation']
        if done or truncated or (lost and info['goal_elapsed_seconds'] >= lost['time_s'] + .2):
            break
    return np.asarray(controls), info['first_violation']


def _digest_fork(env, obs):
    sha = hashlib.sha256()
    for state in (env.data.qpos, env.data.qvel, env.data.ctrl, obs):
        sha.update(np.ascontiguousarray(state).tobytes())
    return sha.hexdigest()


def _physical_window_valid(env, info):
    return (not info['first_violation'] and not info['failure_reason']
            and not env.contacts['forbidden_support']
            and env.contacts['max_penetration'] <= env.config.max_penetration
            and env.displacement <= env.config.max_displacement)


@dataclass
class Branch:
    summary: dict
    observations: list


def _branch(env, seed, prefix, segment, target, guard, capture=False):
    """Replay the entire prefix, then run the candidate at the same clock time."""
    obs, info = _reset(env, seed)
    # Check clearance over every 10 ms scoring sample too. An intermediate
    # workspace/penetration violation need not immediately terminate the task.
    physical_clean = True
    original_refresh = env._refresh
    def checked_refresh():
        nonlocal physical_clean
        original_refresh()
        if not env.warming:
            physical_clean &= (not env.contacts['forbidden_support']
                               and env.contacts['max_penetration'] <= env.config.max_penetration
                               and env.displacement <= env.config.max_displacement)
    env._refresh = checked_refresh
    try:
        for action in prefix:
            guard()
            obs, _, done, truncated, info = env.step(action)
            if done or truncated or not physical_clean or not _physical_window_valid(env, info):
                return Branch({'valid_prefix': False}, [])
        stamp = _digest_fork(env, obs)
        start = _pose_gain(info, target)
        start_signed = info['signed_progress_degrees']
        start_error = info['angle_error_degrees']
        observations = []
        clean = True
        strictly_supported = 0
        for i, action in enumerate(segment):
            guard()
            if capture:
                observations.append(obs.copy())
            obs, _, done, truncated, info = env.step(action)
            clean &= physical_clean and _physical_window_valid(env, info)
            if clean:
                strictly_supported += 1
            if done or truncated:
                break
        complete = i+1 == len(segment)
        summary = dict(valid_prefix=True, fork_sha256=stamp, fork_objective_deg=start,
                       fork_signed_angle_deg=start_signed, fork_pose_error_deg=start_error,
                       complete=complete, strict_eligible=bool(clean and complete),
                       full_controls=i+1, strict_controls=strictly_supported,
                       final_signed_angle_deg=info['signed_progress_degrees'],
                       final_pose_error_deg=info['angle_error_degrees'],
                       final_objective_deg=_pose_gain(info, target),
                       eligible_peak_deg=info['eligible_peak_degrees'],
                       first_violation=info['first_violation'], failure_reason=info['failure_reason'],
                       final_displacement_m=info['displacement'], max_strict_hold_s=info['max_strict_hold_seconds'])
        return Branch(summary, observations)
    finally:
        del env._refresh  # Return class method resolution, keeping the next reset independent.


def _improved(fixed, candidate, target):
    if not (fixed.get('valid_prefix') and candidate.get('strict_eligible')):
        return False
    if candidate['fork_sha256'] != fixed['fork_sha256']:
        raise ValueError('Matched-seed physical fork differs')
    # A failed baseline may spin past its first violation: only credit its
    # pre-violation peak. For a clean fixed branch compare its terminal pose.
    if fixed['strict_eligible']:
        comparator = fixed['final_objective_deg']
        signed_comparator = fixed['final_signed_angle_deg']
    else:
        comparator = min(target, fixed['eligible_peak_deg'])
        signed_comparator = fixed['eligible_peak_deg']
    return (candidate['final_objective_deg'] >= max(candidate['fork_objective_deg'], comparator) + MIN_GAIN_DEG
            and candidate['final_signed_angle_deg'] >= max(candidate['fork_signed_angle_deg'], signed_comparator) + MIN_GAIN_DEG
            and candidate['final_pose_error_deg'] <= candidate['fork_pose_error_deg'] - MIN_GAIN_DEG)


def verify_practice(output, target):
    """Return the NPZ only for an intact, two-integrator qualified stage segment."""
    output = Path(output)
    evidence = output/'result.json'
    if not evidence.is_file():
        return None
    result = json.loads(evidence.read_text())
    if not result.get('qualified'):
        return None
    checks = result['verification']
    if (result['schema'] != SCHEMA or result['target_degrees'] != target
            or result['implementation_sha256'] != implementation_hash()
            or set(checks) != {str(dt) for dt in INTEGRATORS}):
        raise ValueError('Segment stage, provenance or integrator set changed')
    for check in checks.values():
        if not (.1 - 1e-9 <= check['seconds_before_loss'] <= .2 + 1e-9
                and _improved(check['fixed'], check['adjusted'], target)):
            raise ValueError('Segment replay evidence failed')
    path = output/'verified_segment.npz'
    if digest(path) != result['npz_sha256']:
        raise ValueError('Verified actions or observations changed')
    with np.load(path, allow_pickle=False) as data:
        obs, actions = data['observations'], data['actions']
        if (obs.shape != (len(actions), 149) or actions.ndim != 2 or actions.shape[1] != 7
                or len(actions) == 0 or not np.isfinite(obs).all()
                or not np.isfinite(actions).all() or np.any(np.abs(actions)>1.)
                or not np.array_equal(data['nominal_segment'][:, 3:], data['segment'][:, 3:])
                or not np.array_equal(actions, np.tile(data['segment'], (len(INTEGRATORS), 1)))):
            raise ValueError('Invalid matched driver commands or practice samples')
    return path


def search(model, inputs, profile, target, output, seeds, deadline, stop=lambda:False, seconds=600):
    """Run at most once per stage directory, preserving its original time window.

    seeds may be an iterable of diagnostic seeds or the first of eight seeds.
    deadline is a Unix timestamp. The caller must keep the same output directory
    throughout the stage, including after checkpoint recovery.
    """
    from .env import NativeEnv
    inputs, output = Path(inputs), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'search.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/'result.json').is_file():
            old = json.loads((output/'result.json').read_text())
            if old['target_degrees'] != target:
                raise ValueError('Stage directory reused for another target')
            if old['qualified']:
                verify_practice(output, target)
            return old
        seeds = list(range(int(seeds), int(seeds)+8)) if isinstance(seeds, (int, np.integer)) else list(seeds)[:8]
        seeds = [int(s) for s in seeds]
        if not seeds:
            raise ValueError('At least one diagnostic seed is required')
        clockfile = output/'clock.json'
        if clockfile.exists():
            clock = json.loads(clockfile.read_text())
            if clock['stage'] != target or clock['seeds'] != seeds:
                raise ValueError('Existing probe window belongs to another stage or seed set')
        else:
            clock = dict(start=time.time(), stage=target, seeds=seeds)
            clock['until'] = min(deadline, clock['start']+min(MAX_SECONDS,max(0.,seconds)))
            atomic_json(clockfile, clock)
        until = min(clock['until'], deadline)
        def guard():
            if stop() or time.time() >= until:
                raise ClosedWindow
        result = dict(schema=SCHEMA, target_degrees=float(target), qualified=False, comparisons=[], seeds=seeds,
                      reference_sha256=digest(inputs/'reference.npz'), profile=profile.dictionary(),
                      training_step=getattr(model,'num_timesteps',None), source_traces=[],
                      source='partial anticipatory support exercise from original initial grasp, identical open-loop drivers')
        envs = {}
        try:
            for dt in INTEGRATORS:
                guard()
                envs[dt] = NativeEnv(inputs/'reference.npz', 'x', physical_dt=dt, target_degrees=target, profile=profile)
            for seed in seeds:
                nominal, loss = _physical_seed_trace(envs[.002], model, seed, guard)
                trace = dict(seed=seed, nominal_actions_sha256=hashlib.sha256(nominal.tobytes()).hexdigest(),
                             coarse_first_loss=loss, fine_first_loss=None, matched_fork=False)
                result['source_traces'].append(trace)
                if not loss:
                    continue
                warmup = envs[.002].config.warmup_seconds
                choices = _fork_candidates(loss['time_s'], warmup, len(nominal))
                # Apply the SAME baseline policy commands across integrators.
                guard()
                _, info = _reset(envs[.001], seed)
                for action in nominal:
                    guard()
                    _, _, done, truncated, info = envs[.001].step(action)
                    if done or truncated or info['first_violation']:
                        break
                fine_loss = info['first_violation']
                trace['fine_first_loss'] = fine_loss
                if not fine_loss:
                    continue
                choices = [k for k in choices if k in _fork_candidates(fine_loss['time_s'], warmup, len(nominal))]
                if not choices:
                    continue
                fork = choices[0]
                trace['matched_fork'] = True
                trace['fork_controls'] = fork
                prefix, original = nominal[:fork], nominal[fork:fork+10]
                if len(original) < 3:  # allow at least 150 ms for the intervention
                    continue
                guard()
                fixed_actions = _commands(original, np.zeros(3), np.zeros(3))
                fixed = {dt:_branch(e,seed,prefix,fixed_actions,target,guard).summary for dt,e in envs.items()}
                if not all(x['valid_prefix'] for x in fixed.values()):
                    continue
                for first, second in _profiles(np.random.default_rng(seed)):
                    guard()
                    actions = _commands(original,first,second)
                    verification = {}
                    observations = []
                    row = dict(seed=seed, prefix_controls=fork, controls=len(actions),
                               support_first=first.tolist(), support_second=second.tolist(), pairs={})
                    for dt in (.002,.001):
                        adjusted = _branch(envs[dt],seed,prefix,actions,target,guard,capture=True)
                        loss_s = loss['time_s'] if dt==.002 else fine_loss['time_s']
                        check = dict(seconds_before_loss=float(warmup+loss_s-fork*CONTROL_PERIOD),
                                     fixed=fixed[dt], adjusted=adjusted.summary)
                        row['pairs'][str(dt)] = check
                        if not _improved(fixed[dt], adjusted.summary, target):
                            break
                        observations.extend(adjusted.observations)
                        verification[str(dt)] = check
                    result['comparisons'].append(row)
                    if len(verification) == 2:
                        path = output/'verified_segment.npz'
                        guard()
                        with (output/'.verified_segment.tmp').open('wb') as f:
                            np.savez_compressed(f, prefix=prefix, nominal_segment=original, segment=actions,
                                                observations=np.asarray(observations,dtype=np.float32),
                                                actions=np.tile(actions,(len(INTEGRATORS),1)))
                        os.replace(output/'.verified_segment.tmp',path)
                        result.update(qualified=True,seed=seed,verification=verification,npz_sha256=digest(path),
                                      implementation_sha256=implementation_hash(), observation_order_dt=[.002,.001],
                                      prefix_controls=fork, segment_controls=len(actions))
                        atomic_json(output/'result.json',result)
                        verify_practice(output,target)
                        return result
            result['reason']='no_two_integrator_qualified_segment'
        except ClosedWindow:
            result['reason']='search_time_expired_or_interrupted'
        finally:
            for env in envs.values():env.close()
        atomic_json(output/'result.json',result)
        return result
