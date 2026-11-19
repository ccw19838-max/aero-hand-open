"""Versioned, read-only thumb-driven rotation shaping; no action controller.

Contact frame rows are world axes. MuJoCo's contact wrench acts on geom2;
negate it when the cube is geom1. All moments are about the cube mass center.
"""
from pathlib import Path
from types import MethodType
import hashlib
import gymnasium as gym
import mujoco
import numpy as np
from experiments.cube_contact_ppo.task import force_limits, force_state

TRANSFORM = 'thumb_drive_four_support_v1'


def descriptor():
    return dict(name=TRANSFORM, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                reward_budget=12., torque_scale_Nm=.005, surface_speed_scale_m_s=.005,
                drive_auxiliary_deficit_rate=.2, capture_auxiliary_deficit_rate=1.5,
                wrong_thumb_moment_rate=.25, sample_seconds=.01,
                physics_actions_observations_termination_unchanged=True)


def contact_wrenches(base):
    """Measured finger forces, moments and moving-thumb tangential speed."""
    model, data, ids = base.model, base.data, base.ids
    center = data.xipos[ids['cube_body']]
    direction = base.monitor.direction
    forces = np.zeros((5, 3)); moments = np.zeros((5, 3))
    velocity_numerator = 0.; normal_total = 0.
    wrench = np.zeros(6); jac = np.zeros((3, model.nv))
    cube_jac = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, cube_jac, None, center, ids['cube_body'])
    center_velocity = cube_jac @ data.qvel
    for i, contact in enumerate(data.contact):
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if ids['cube_geom'] not in (g1, g2):
            continue
        other = g1 if g2 == ids['cube_geom'] else g2
        finger = int(ids['geom_finger'][other])
        if finger < 0:
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        sign = 1. if g2 == ids['cube_geom'] else -1.
        frame = contact.frame.reshape(3, 3)
        force = sign * (frame.T @ wrench[:3])
        arm = contact.pos - center
        forces[finger] += force
        moments[finger] += np.cross(arm, force) + sign * (frame.T @ wrench[3:])
        if finger == 0 and wrench[0] > 0.:
            tangent = np.cross(direction, arm)
            norm = np.linalg.norm(tangent)
            if norm > 1e-8:
                mujoco.mj_jac(model, data, jac, None, contact.pos, int(model.geom_bodyid[other]))
                velocity_numerator += float(wrench[0]) * float((jac @ data.qvel - center_velocity) @ (tangent / norm))
                normal_total += float(wrench[0])
    return dict(forces_world_N=forces, moments_world_Nm=moments,
                axis_moments_Nm=moments @ direction,
                thumb_surface_speed_m_s=velocity_numerator / max(normal_total, 1e-12))


def drive_quality(moment, speed, normal_force, support, valid):
    # A stationary squeeze, reverse push, over-force or unsupported throw earns
    # no drive reward. Stronger squeezing beyond the scale adds no reward.
    if not valid or not support or not .2 <= normal_force <= 20.:
        return 0.
    return float(np.clip(moment / .005, 0., 1.) * np.clip(speed / .005, 0., 1.))


def credit_progress(progress, error, target, high_water):
    reached = float(np.clip(min(progress, target - error), 0., target))
    return max(0., reached - high_water), max(high_water, reached)


class ThumbDriveReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        if env.unwrapped.fixed_axis != 'x':
            raise ValueError('Thumb-drive v1 is an X-axis experiment')
        self.patched = None
        self._clear()

    def _clear(self):
        self.high_water = 0.; self.bonus_total = 0.; self.cost_total = 0.
        self.positive_moment_seconds = 0.; self.active_drive_seconds = 0.
        self.sample_seconds = 0.; self.last_metrics = {}

    def install(self):
        base = self.unwrapped; monitor = base.monitor
        if self.patched is monitor:
            return
        original = monitor.checkpoint

        def checkpoint(this, q, contacts, linear, angular, displacement, dt=.01):
            reward = original(q, contacts, linear, angular, displacement, dt)
            if base.warming or this.target_degrees <= 0.:
                return reward
            metrics = contact_wrenches(base)
            forces = np.asarray(contacts['finger_forces'])
            lower, _ = force_limits(base.config)
            support = force_state(forces, base.config)[0]
            valid = (not contacts['forbidden_support'] and not this.failure
                     and contacts['max_penetration'] <= base.config.max_penetration
                     and displacement <= base.config.max_displacement)
            moment = float(metrics['axis_moments_Nm'][0])
            speed = float(metrics['thumb_surface_speed_m_s'])
            quality = drive_quality(moment, speed, forces[0], support, valid)
            target = np.deg2rad(this.target_degrees)
            # Advance even outside the valid region: cycling through invalid
            # states and back cannot re-earn previously visited progress.
            increment, self.high_water = credit_progress(this.progress, this.error, target, self.high_water)
            drive_phase = float(np.clip((np.rad2deg(this.error) - 5.) / 20., 0., 1.))
            bonus = 12. * increment / (np.pi / 2.) * quality * drive_phase
            deficit = np.clip((lower[1:] - forces[1:]) / lower[1:], 0., 1.)
            assist_cost = (.2 * drive_phase + 1.5 * (1. - drive_phase)) * float(np.mean(deficit**2))
            reverse_cost = .25 * drive_phase * float(np.clip(-moment / .005, 0., 2.))
            cost = dt * (assist_cost + reverse_cost)
            self.bonus_total += bonus; self.cost_total += cost
            self.sample_seconds += dt
            self.positive_moment_seconds += dt * (moment > 0.)
            self.active_drive_seconds += dt * (quality > .1)
            self.last_metrics = dict(thumb_axis_moment_Nm=moment, thumb_surface_speed_m_s=speed,
                                     finger_axis_moments_Nm=metrics['axis_moments_Nm'].tolist(),
                                     thumb_drive_quality=quality)
            return float(reward + bonus - cost)

        monitor.checkpoint = MethodType(checkpoint, monitor)
        self.patched = monitor

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs); self._clear(); self.install()
        return result

    def step(self, action):
        self.install()
        obs, reward, done, truncated, info = self.env.step(action)
        info = {**info, **self.last_metrics, 'training_reward_transform': TRANSFORM,
                'thumb_drive_bonus_total': self.bonus_total, 'thumb_support_cost_total': self.cost_total,
                'thumb_positive_moment_fraction': self.positive_moment_seconds / max(self.sample_seconds, .01),
                'thumb_active_drive_fraction': self.active_drive_seconds / max(self.sample_seconds, .01)}
        return obs, reward, done, truncated, info
