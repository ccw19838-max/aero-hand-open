"""Strict X flip, using the original V3 physics and physical contact probes."""
from dataclasses import replace

import gymnasium as gym
import numpy as np

from experiments.cube_adaptive_60h.shaping import MountMarginReward
from experiments.cube_contact_ppo.task import force_limits, force_state
from experiments.cube_two_driver_v3.env import (LADDER, FastEnv as V3Fast,
                                                NativeEnv as V3Native, RoleMonitor)
from experiments.cube_two_driver_v3.roles import DRIVERS, driving_quality
from .contact_history import ContactHistory, SAMPLE_SECONDS

VERSION = 'x_strict_three_support_v4'
OBSERVATION_VERSION = 'v3_143_plus_index_middle_ring_margin_then_warning'
OBS_SIZE = 149


class ScoredMonitor(RoleMonitor):
    """Passive scoring for V3. Preserve all legacy observations, rewards and terminations."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strict = ContactHistory()
        self.previous_sample_angle = 0.
        self.new_fraction = 0.
        self.base_progress_return = self.driver_return = 0.

    def checkpoint(self, q, contacts, linear, angular, displacement, dt=.01):
        if self.done:
            return 0.
        if abs(dt - SAMPLE_SECONDS) > 1e-10:
            raise ValueError('Use the 10 ms force sampling grid')
        previous_error = self.error
        reward = super().checkpoint(q, contacts, linear, angular, displacement, dt)
        if self.env.warming:
            return reward
        angle = float(np.rad2deg(self.progress))
        physical_ok = (not self.failure and not contacts['forbidden_support']
                       and contacts['max_penetration'] <= self.config.max_penetration
                       and displacement <= self.config.max_displacement)
        self.new_fraction = self.strict.sample(
            contacts['finger_forces'], angle, np.rad2deg(self.error), self.target_degrees,
            self.previous_sample_angle, np.rad2deg(previous_error), physical_ok, self.stable)
        self.previous_sample_angle = angle
        return reward


class StrictMonitor(ScoredMonitor):
    def checkpoint(self, q, contacts, linear, angular, displacement, dt=.01):
        if self.done:
            return 0.
        previous_error = self.error
        previous_bonus = self.role_bonus
        previous_penalty = self.role_penalty
        previous_potential = self.support_potential
        reward = super().checkpoint(q, contacts, linear, angular, displacement, dt)
        if self.env.warming:
            return reward
        first = self.strict.first_violation
        if first is not None:
            # Discard inherited hold/goal/progress rewards after the first raw failure.
            # Support-potential differences and the V3 named-force deficit train recovery.
            change = self.support_potential - previous_potential if previous_potential is not None else 0.
            recovery = change + self.role_penalty - previous_penalty
            recovered = self.three_missing == 0.
            expired = self.strict.sample_count * SAMPLE_SECONDS - first.time_s >= .15 - 1e-10
            if self.failure or expired or recovered or self.success:
                self.success = False
                self.failure = self.failure or ('recovered_after_violation' if recovered else 'three_support_lost')
                if not recovered:
                    recovery -= self.config.failure_penalty
            return float(recovery)

        # V3 pays positive angular-error reduction and a driver bonus. Remove both
        # before awarding strictly supported new progress. Keep reversal costs.
        valid = (not contacts['forbidden_support'] and contacts['max_penetration'] <= self.config.max_penetration
                 and displacement <= self.config.max_displacement)
        forces = contacts['finger_forces']
        if valid and force_state(forces, self.config)[0]:
            reward -= self.profile.progress_gain * max(0., (previous_error - self.error) / (np.pi / 2))
        reward -= self.role_bonus - previous_bonus
        progress_credit = 20. * self.new_fraction
        driver_credit = 0.
        lower, upper = force_limits(self.config)
        if self.new_fraction and np.all(forces <= upper):
            qualities = [driving_quality(self.last_moments[i], self.last_speeds[i]) if forces[i] >= lower[i] else 0.
                         for i in DRIVERS]
            # Either digit may advance the turn. Static squeezing adds no credit.
            quality = .5 * np.mean(qualities) + .5 * min(qualities)
            driver_credit = 8. * self.new_fraction * quality * (.5 if self.strict.warnings.any() else 1.)
        self.base_progress_return += progress_credit
        self.driver_return += driver_credit
        return float(reward + progress_credit + driver_credit)


class ScoredEnvironment:
    monitor_type = ScoredMonitor

    def replace_monitor(self):
        old = self.monitor
        if isinstance(old, self.monitor_type):
            return
        self.monitor = self.monitor_type(old.initial, old.axis, old.basis, self.config, old.target_degrees,
                                         ideal_initial=old.ideal_initial, profile=self.profile, env=self)

    def _info(self):
        info = super()._info()
        monitor = self.monitor
        history = getattr(monitor, 'strict', None)
        info.update((history or ContactHistory()).info())
        info['legacy_sampled_supported_peak_degrees'] = info['supported_progress_degrees']
        info['supported_progress_degrees'] = info['eligible_peak_degrees']
        info['strict_success'] = bool(history and history.first_violation is None and monitor.success
                                      and not monitor.failure and not self.warming)
        return info


class BaselineEvalEnv(ScoredEnvironment, V3Native):
    """143 inputs and unmodified V3 control; scores available in info only."""


class StrictEnvironment(ScoredEnvironment):
    monitor_type = StrictMonitor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = replace(self.config, version=VERSION)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_SIZE,), dtype=np.float32)

    def _observation(self, history):
        original = super()._observation(history)
        score = getattr(self.monitor, 'strict', None)
        extra = np.r_[score.margins, score.warnings] if score else np.zeros(6)
        return np.concatenate((original, extra)).astype(np.float32)

    def _info(self):
        info = super()._info()
        m = self.monitor
        info.update(task=VERSION, environment_version=VERSION, actor_inputs=OBS_SIZE,
                    observation_version=OBSERVATION_VERSION,
                    normalized_progress_return=getattr(m, 'base_progress_return', 0.),
                    dual_driver_bonus=getattr(m, 'driver_return', 0.))
        return info

    def step(self, action):
        previous = self.last_action.copy()
        obs, reward, done, truncated, info = super().step(action)
        if info['first_violation'] is not None:
            # The parent adds an action-slew penalty outside the monitor. In a
            # recovery interval only the contact-recovery signal is used.
            reward += .01 * float(np.mean((self.last_action - previous) ** 2))
        return obs, reward, done, truncated, info


class NativeEnv(StrictEnvironment, V3Native):
    pass


class FastEnv(StrictEnvironment, V3Fast):
    pass


class TrainingReward(MountMarginReward):
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if info['first_violation'] is not None:
            penalty = info['mount_proximity_penalty']
            reward += penalty
            self.total_penalty -= penalty
            info.update(mount_proximity_penalty=0., mount_proximity_penalty_total=self.total_penalty)
        return obs, reward, done, truncated, info


def training_reward(env):
    return TrainingReward(env)
