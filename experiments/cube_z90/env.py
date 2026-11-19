"""Real contacts, fixed initial acquisition, and strong goal-directed shaping.

Only the initial acquire phase holds the existing actuator targets unchanged.
Rotation and recapture are controlled entirely by the learned seven actions.
No external forces, object pose changes, or assistance are applied in a rollout.
"""
import mujoco
import numpy as np

from experiments.cube_envelop.env import (
    CubeReorientEnv as EnvelopeEnv, ReorientMonitor, target_quaternion,
    orientation_error, quaternion_multiply, quaternion_conjugate,
)
from .config import CubeReorientConfig


def swing_radians(quaternion, initial, axis):
    """Swing-twist decomposition around an immutable world-frame hand normal."""
    relative = quaternion_multiply(quaternion, quaternion_conjugate(initial))
    cosine_half = np.hypot(relative[0], np.dot(relative[1:], axis))
    return float(2 * np.arccos(np.clip(cosine_half, 0., 1.)))


def physical_substep_reward(config, monitor, contacts, linear, angular,
                            stable, valid, potential_delta, swing=0.):
    c = config
    touching = np.asarray(contacts['finger_forces']) >= c.contact_force
    supported = bool(touching.any())
    # Positive progress only counts with valid finger support. Reverse motion
    # always costs reward, including during contact loss or illegal penetration.
    delta = potential_delta if valid and supported else min(potential_delta, 0.)
    reward = c.progress_reward * delta if monitor.target_active else 0.
    if not valid:
        return float(reward - c.sim_dt * 5.)
    if not monitor.target_active:
        return 0.  # No reward for spending longer in initial acquisition.

    rate = -c.time_penalty
    rate -= c.swing_penalty * min((swing / np.deg2rad(15.)) ** 2, 25.)
    rate -= .2 * min((np.linalg.norm(linear) / .1) ** 2, 25.)
    # Goal error, not an endless angular-speed target: slow down at the face.
    desired = np.clip(3. * (np.deg2rad(c.target_degrees)
                            - monitor.signed_progress_radians), -2.5, 2.5)
    axis = monitor.axis_basis[:, 2]
    axial_speed = float(np.dot(angular, axis))
    perpendicular = angular - axial_speed * axis
    rate -= c.velocity_tracking_penalty * min((axial_speed - desired) ** 2, 25.)
    rate -= .05 * min(float(np.dot(perpendicular, perpendicular)), 100.)
    if supported:
        radius = np.deg2rad(max(1., min(12., c.target_degrees / 3.)))
        # Signed progress also gates the attractive pose term. -270/450 degrees
        # must never collect the +90 goal bonus despite identical quaternions.
        progress_error = abs(monitor.signed_progress_radians - np.deg2rad(c.target_degrees))
        error = max(monitor.angle_error_radians, progress_error)
        rate += c.near_goal_reward * np.exp(-(error / radius) ** 2)
        rate += c.stable_reward * bool(stable)
        rate += .05 * bool(touching[0])
        # Once near the target, each missing contact has a reason to return.
        near = error <= np.deg2rad(c.angle_tolerance_degrees)
        if near:
            rate += 2. * float(np.mean(touching))
        excess = max(0., float(contacts['finger_forces'][0]) / c.thumb_force_soft_limit - 1.)
        rate -= c.thumb_force_penalty * min(excess ** 2, 25.)
    else:
        rate -= 2.
    return float(reward + c.sim_dt * rate)


class CubeReorientEnv(EnvelopeEnv):
    def __init__(self, reference, config=None, render_mode=None, axis=None):
        if axis not in (None, 'z'):
            raise ValueError('Z-only environment')
        config = config or CubeReorientConfig()
        if not isinstance(config, CubeReorientConfig):
            raise ValueError('Dedicated Z90 configuration required')
        super().__init__(reference, config, render_mode, axis='z')

    def reset(self, *, seed=None, options=None):
        if (options or {}).get('axis', 'z') != 'z':
            raise ValueError('Z-only reset')
        return super().reset(seed=seed, options={**(options or {}), 'axis': 'z'})

    def _info(self):
        info = super()._info()
        q = self.ids['cube_qpos_adr']
        info.update(target_degrees=self.config.target_degrees,
                    angle_tolerance_degrees=self.config.angle_tolerance_degrees,
                    required_hold_seconds=self.config.target_hold_seconds,
                    reward_version=self.config.reward_version,
                    acquisition_controller=self.config.initial_capture_controller,
                    rotation_controller='learned_ppo',
                    swing_degrees=np.rad2deg(swing_radians(
                        self.data.qpos[q+3:q+7], self.monitor.initial_quaternion,
                        self.axis_basis[:, 2])))
        return info

    def step(self, action):
        if not self._has_reset or self.monitor.done:
            raise RuntimeError('Reset before stepping a new episode')
        requested = np.asarray(action, dtype=float)
        if requested.shape != (7,) or not np.isfinite(requested).all():
            raise ValueError('Expected seven finite actions')
        # This is a documented phase controller, used identically in training,
        # evaluation and replay; only actuator targets are held, never the cube.
        applied = np.zeros(7) if not self.monitor.capture_succeeded else np.clip(requested, -1., 1.)
        c = self.config
        self.data.ctrl[:] = np.clip(self.data.ctrl + c.action_delta_fraction * self._range * applied,
                                    self.ids['ctrl_lower'], self.ids['ctrl_upper'])
        reward = 0.
        before = self.monitor.elapsed
        captured_before = self.monitor.capture_succeeded
        for _ in range(c.substeps):
            old_time = self.data.time
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            bad = (not np.isfinite(self.data.qpos).all()
                   or not np.isfinite(self.data.qvel).all() or self.data.time < old_time)
            self._refresh()
            q = self.ids['cube_qpos_adr']
            quaternion = self.data.qpos[q+3:q+7]
            stable, valid, delta = self.monitor.update(
                quaternion, self._contacts, self._linear, self._angular,
                self._displacement, float(self.data.xpos[self.ids['cube_body'], 2]), bad)
            swing = swing_radians(quaternion, self.monitor.initial_quaternion, self.axis_basis[:, 2])
            reward += physical_substep_reward(c, self.monitor, self._contacts,
                                              self._linear, self._angular, stable, valid, delta, swing)
            if self.monitor.done:
                reward += c.success_reward if self.monitor.success else -c.failure_penalty
                break
        if self.monitor.capture_succeeded and not captured_before:
            reward += c.capture_reward
        elapsed = self.monitor.elapsed - before
        reward -= elapsed * (.02 * np.mean((applied - self.previous_action) ** 2)
                             + .002 * np.mean(applied ** 2))
        self.previous_action = applied.copy()
        info = self._info()
        info.update(requested_action=np.clip(requested, -1., 1.), applied_action=applied.copy())
        # The finite 16-second task deadline is a failed attempt, rather than an
        # arbitrary rollout cut. Do not bootstrap a critic value past failure.
        return (self._observation(), float(reward), bool(self.monitor.done), False, info)
