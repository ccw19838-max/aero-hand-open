"""Deterministic, original-grasp evaluation with a common 10 ms strict scorer."""
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .env import BaselineEvalEnv, NativeEnv, ScoredMonitor, StrictMonitor
from .reward_trace import record_task_rewards

TIMESTEPS = (.001, .002)
FINGERS = ('index', 'middle', 'ring')
DRIVERS = ('thumb', 'pinky')


class DriverTelemetry:
    """Integrate each driver's signed contact moment on the physical scoring grid."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.push_impulse = np.zeros(2)
        self.counter_impulse = np.zeros(2)

    def checkpoint(self, q, contacts, linear, angular, displacement, dt=.01):
        active = not self.done and not self.env.warming
        value = super().checkpoint(q, contacts, linear, angular, displacement, dt)
        if active and self.strict.first_violation is None:
            moments = self.last_moments[[0, 4]]
            speeds = self.last_speeds[[0, 4]]
            if self.new_fraction > 0.:
                self.push_impulse += dt * np.maximum(0., moments) * (speeds > 0.)
            # Opposing torque at any qualified sample; static support can also contribute.
            self.counter_impulse += dt * np.maximum(0., -moments)
        return value


class BaselineTelemetry(DriverTelemetry, ScoredMonitor):
    pass


class ReformTelemetry(DriverTelemetry, StrictMonitor):
    pass


def make_env(inputs, profile, target, dt, arm):
    classes = {'v3': BaselineEvalEnv, 'v4': NativeEnv}
    monitors = {'v3': BaselineTelemetry, 'v4': ReformTelemetry}
    env = classes[arm](Path(inputs)/'reference.npz', 'x', physical_dt=dt,
                       target_degrees=target, profile=profile, training=False)
    env.monitor_type = monitors[arm]
    return record_task_rewards(env, arm)


def episode(env, model, seed, *, trace=False, capture=None,
            deadline=float('inf'), stop=lambda: False):
    observation, info = env.reset(seed=seed)
    if not info['from_original_initial_grasp'] or info['training_prefix_used']:
        raise RuntimeError('Formal evaluation must start from the original prepared grasp')
    if capture is not None:
        capture(env, info)
    frames = []
    done = False
    interrupted = False
    while not done:
        if stop() or time.time() >= deadline:
            interrupted = True
            break
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if trace:
            frames.append(info)
        if capture is not None:
            capture(env, info)
        if env.data.time > env.config.episode_seconds + env.config.warmup_seconds + env.config.ctrl_dt and not done:
            raise RuntimeError('Environment failed to terminate at its episode limit')
    strict = bool(done and info['strict_success'] and not interrupted)
    if strict and (info['first_violation'] is not None or info['max_strict_hold_seconds'] + 1e-8 < env.config.hold_seconds):
        raise RuntimeError('Inconsistent strict success flags')
    # The native monitor gets replaced after the physical warmup and on each reset.
    monitor = env.monitor
    if isinstance(monitor, DriverTelemetry):
        info['driver_push_impulse_Nm_s'] = dict(zip(DRIVERS, monitor.push_impulse.tolist()))
        info['driver_counter_impulse_Nm_s'] = dict(zip(DRIVERS, monitor.counter_impulse.tolist()))
    row = dict(seed=int(seed), dt=float(env.physical_dt), success=strict,
               interrupted=interrupted, final=info)
    if trace:
        row['trace'] = frames
    return row


def failure_reason(row):
    if row['interrupted']:
        return 'interrupted'
    final = row['final']
    if final['first_violation'] is not None:
        return 'support_out_of_range'
    return final['failure_reason'] or 'hold_incomplete'


def summarize(rows):
    finals = [r['final'] for r in rows if not r['interrupted']]
    violations = [f['first_violation'] for f in finals if f['first_violation'] is not None]
    lead_ms = [ms for f in finals for ms in f['warning_lead_ms']]
    median = lambda values: float(np.median(values)) if values else None
    angles_by_finger = {finger: [v['angle_deg'] for v in violations if finger in v['fingers']]
                        for finger in FINGERS}
    valid_median = median([f['eligible_peak_degrees'] for f in finals])
    def driver_medians(field):
        return {finger:median([f[field][finger] for f in finals if field in f]) for finger in DRIVERS}
    return dict(
        median_valid_peak_degrees=valid_median, median_eligible_peak_degrees=valid_median,
        maximum_eligible_peak_degrees=max((f['eligible_peak_degrees'] for f in finals),default=0.),
        maximum_signed_angle=max((f['max_progress_degrees'] for f in finals),default=0.),
        maximum_stable_seconds=max((f['max_strict_hold_seconds'] for f in finals),default=0.),
        mean_best_error=float(np.mean([f['best_angle_error_degrees'] for f in finals])) if finals else None,
        first_violation_episodes=len(violations),
        first_violation_finger_counts={finger:len(values) for finger,values in angles_by_finger.items()},
        median_first_violation_angle_by_finger={finger:median(values) for finger,values in angles_by_finger.items()},
        median_first_violation_seconds=median([v['time_s'] for v in violations]),
        warning_count=sum(f['warning_count'] for f in finals),
        warning_false_count=sum(f['warning_false_count'] for f in finals),
        warning_censored_count=sum(f['warning_censored_count'] for f in finals),
        warning_matched_count=len(lead_ms), warning_median_lead_ms=median(lead_ms),
        median_driver_push_impulse_Nm_s=driver_medians('driver_push_impulse_Nm_s'),
        median_driver_counter_impulse_Nm_s=driver_medians('driver_counter_impulse_Nm_s'),
    )


def evaluate(model, inputs, profile, target, episodes=20, seed=6400000,
             deadline=float('inf'), stop=lambda:False, trace=False, arm='v4', purpose='validation'):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    records = []
    for dt in TIMESTEPS:
        if time.time() < deadline and not stop():
            env = make_env(inputs,profile,target,dt,arm)
            try:
                for initial_seed in range(seed,seed+episodes):
                    if time.time() >= deadline or stop():
                        break
                    row = episode(env,model,initial_seed,trace=trace,deadline=deadline,stop=stop)
                    records.append(row)
                    if row['interrupted']:
                        break
            finally:
                env.close()
    return summarize_report(records,target,episodes,seed,arm,purpose)


def summarize_report(records, target, episodes, seed, arm='v4', purpose='validation'):
    from .env import VERSION
    groups = {}
    expected_seeds = list(range(seed, seed+episodes))
    for dt in TIMESTEPS:
        group = [r for r in records if r['dt']==dt]
        successes = sum(r['success'] for r in group)
        groups[str(dt)] = dict(trials=len(group), requested_trials=episodes, successes=successes,
                               success_rate=successes/max(len(group),1),
                               complete=sorted(r['seed'] for r in group)==expected_seeds and not any(r['interrupted'] for r in group),
                               failures=dict(Counter(failure_reason(r) for r in group if not r['success'])),
                               **summarize(group))
    return dict(task=VERSION,arm=arm,purpose=purpose,target_degrees=float(target),
                seed_start=seed,from_original_initial_grasp=True,groups=groups,
                complete=all(g['complete'] for g in groups.values()),
                success_rate=min(g['success_rate'] for g in groups.values()),
                **summarize(records),records=records)


def compact(report):
    return {key:value for key,value in report.items() if key!='records'}


def score(report):
    if not report or not report['complete']:
        return (-1., -1., -1000.)
    # The slower integrator / harder initials cannot be hidden by pooling.
    median_peak = min(g['median_eligible_peak_degrees'] for g in report['groups'].values())
    return (report['success_rate'],min(report['target_degrees'],median_peak),-report['mean_best_error'])


def final_pass(report):
    from .env import VERSION
    if not (report and report['task']==VERSION and report['arm']=='v4' and report['purpose']=='independent_final'
            and report['from_original_initial_grasp'] and report['complete'] and report['target_degrees']==90.):
        return False
    required = {str(dt) for dt in TIMESTEPS}
    if set(report['groups']) != required:
        return False
    if not all(g['complete'] and g['successes']==g['requested_trials']==g['trials']==100 for g in report['groups'].values()):
        return False
    pairs = {(r['dt'],r['seed']) for r in report['records'] if r['success'] and not r['interrupted']
             and r['final']['first_violation'] is None and r['final']['strict_success']}
    return pairs == {(dt, seed) for dt in TIMESTEPS for seed in range(report['seed_start'],report['seed_start']+100)}
