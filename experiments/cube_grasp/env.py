"""Native MuJoCo grasp acquisition/holding environment shared by train and replay.

The actor observes simulated joint/object state and contacts. These inputs are
not a claim that a physical Aero hand provides joint encoders or tactile sensing.
Objects are initialized once, then evolve without constraints or support forces.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from .config import CubeGraspConfig
from .model import (build_model, model_fingerprint, model_ids, native_contacts,
                    native_cube_velocity)


@dataclass
class GraspMonitor:
    """Physical-substep success/failure state, independent of policy frequency."""
    config: CubeGraspConfig
    elapsed: float = 0.
    stable_seconds: float = 0.
    max_stable_seconds: float = 0.
    penetration_seconds: float = 0.
    region_seconds: float = 0.
    peak_penetration: float = 0.
    success: bool = False
    failure_reason: str = ''

    @property
    def done(self):
        return self.success or bool(self.failure_reason)

    def update(self, contacts, linear, angular, displacement, cube_height,
               numerical_failure=False):
        """Returns stable/valid flags; terminal events are latched exactly once."""
        if self.done:
            return False, False
        c = self.config
        self.elapsed += c.sim_dt
        forces = np.asarray(contacts['finger_forces'])
        touching = forces >= c.contact_force
        enough = bool(touching[0] and np.count_nonzero(touching) >= c.min_fingers)
        depth = float(contacts['max_penetration'])
        self.peak_penetration = max(self.peak_penetration, depth)
        penetrated = depth > c.max_penetration
        outside = displacement > c.max_displacement
        forbidden = bool(contacts['forbidden_support'])
        self.penetration_seconds = self.penetration_seconds + c.sim_dt if penetrated else 0.
        self.region_seconds = self.region_seconds + c.sim_dt if outside else 0.
        valid = not (penetrated or outside or forbidden or numerical_failure)
        stable = bool(valid and enough and np.linalg.norm(linear) < c.grip_velocity
                      and np.linalg.norm(angular) < c.grip_omega)
        self.stable_seconds = self.stable_seconds + c.sim_dt if stable else 0.
        self.max_stable_seconds = max(self.max_stable_seconds, self.stable_seconds)
        if numerical_failure:
            self.failure_reason = 'numerical'
        elif forbidden:
            self.failure_reason = 'forbidden_support'
        elif cube_height < c.cube_size / 2:
            self.failure_reason = 'dropped'
        elif self.penetration_seconds + 1e-10 >= c.violation_seconds:
            self.failure_reason = 'penetration'
        elif self.region_seconds + 1e-10 >= c.violation_seconds:
            self.failure_reason = 'workspace'
        elif self.stable_seconds + 1e-10 >= c.hold_seconds:
            self.success = True
        elif self.elapsed + 1e-10 >= c.episode_seconds:
            self.failure_reason = 'timeout'
        return stable, valid


class CubeGraspEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    def __init__(self, reference, config=None, render_mode=None):
        super().__init__()
        self.reference_path = Path(reference).resolve()
        report_path = self.reference_path.parent / 'prepare_report.json'
        if not report_path.is_file():
            raise ValueError('Prepared reference requires prepare_report.json')
        report = json.loads(report_path.read_text())
        if report.get('approved') is not True:
            raise ValueError('Preparation has not approved this reference')
        self.config = config or CubeGraspConfig.from_dict(report['config'])
        if report.get('model_fingerprint') != model_fingerprint(self.config):
            raise ValueError('Reference and requested physics fingerprint differ')
        if report.get('reference_sha256') and report['reference_sha256'] != hashlib.sha256(
                self.reference_path.read_bytes()).hexdigest():
            raise ValueError('Prepared reference checksum differs')
        with np.load(self.reference_path, allow_pickle=False) as archive:
            self.reference = {key: archive[key].copy() for key in archive.files}
        self.model = build_model(self.config)
        self.data = mujoco.MjData(self.model)
        self.ids = model_ids(self.model)
        for key, shape in [('qpos', (self.model.nq,)), ('qvel', (self.model.nv,)),
                           ('ctrl', (self.model.nu,))]:
            if key not in self.reference or self.reference[key].shape != shape:
                raise ValueError(f'Invalid prepared {key} shape')
            if not np.all(np.isfinite(self.reference[key])):
                raise ValueError(f'Prepared {key} contains non-finite values')
        if self.model.nu != 7:
            raise ValueError('Expected seven Aero actuator channels')
        self.action_space = gym.spaces.Box(-1., 1., (7,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (65,), dtype=np.float32)
        self.render_mode = render_mode
        if render_mode not in (None, 'rgb_array'):
            raise ValueError('Only rgb_array rendering is supported')
        self._renderer = None
        self._camera = mujoco.MjvCamera()
        self._camera.lookat[:] = [-.025, 0., .26]
        self._camera.distance = .46
        self._camera.azimuth = 155.
        self._camera.elevation = -10.
        self._body_mass = self.model.body_mass[self.ids['cube_body']].copy()
        self._body_inertia = self.model.body_inertia[self.ids['cube_body']].copy()
        self._friction = self.model.geom_friction[self.ids['cube_geom']].copy()
        self._range = self.ids['ctrl_upper'] - self.ids['ctrl_lower']
        self.previous_action = np.zeros(7)
        self.monitor = GraspMonitor(self.config)
        self._contacts = {'finger_forces': np.zeros(5), 'forbidden_support': False,
                          'max_penetration': 0.}
        self._linear = np.zeros(3)
        self._angular = np.zeros(3)
        self._initial_position = np.zeros(3)
        self._displacement = 0.
        self._has_reset = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.config
        mujoco.mj_resetData(self.model, self.data)
        mass_factor = self.np_random.uniform(1 - c.mass_randomization, 1 + c.mass_randomization)
        friction_factor = self.np_random.uniform(1 - c.friction_randomization,
                                                 1 + c.friction_randomization)
        self.model.body_mass[self.ids['cube_body']] = self._body_mass * mass_factor
        self.model.body_inertia[self.ids['cube_body']] = self._body_inertia * mass_factor
        self.model.geom_friction[self.ids['cube_geom']] = self._friction
        self.model.geom_friction[self.ids['cube_geom'], 0] *= friction_factor
        mujoco.mj_setConst(self.model, self.data)
        q = self.ids['cube_qpos_adr']
        reset_contacts = None
        for attempt in range(1, c.reset_max_attempts + 1):
            self.data.qpos[:] = self.reference['qpos']
            self.data.qvel[:] = self.reference['qvel']
            self.data.qacc_warmstart[:] = 0.
            self.data.ctrl[:] = np.clip(self.reference['ctrl'], self.ids['ctrl_lower'],
                                        self.ids['ctrl_upper'])
            self.data.qpos[q:q + 3] += self.np_random.uniform(-c.reset_position_noise,
                                                             c.reset_position_noise, 3)
            axis = self.np_random.normal(size=3)
            axis /= max(float(np.linalg.norm(axis)), 1e-12)
            angle = np.deg2rad(self.np_random.uniform(-c.reset_angle_noise_degrees,
                                                      c.reset_angle_noise_degrees))
            noise_q = np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]
            result_q = np.empty(4)
            mujoco.mju_mulQuat(result_q, noise_q, self.data.qpos[q + 3:q + 7].copy())
            self.data.qpos[q + 3:q + 7] = result_q / np.linalg.norm(result_q)
            mujoco.mj_forward(self.model, self.data)
            reset_contacts = native_contacts(self.model, self.data, self.ids)
            if (reset_contacts['max_penetration'] <= c.reset_max_penetration
                    and not reset_contacts['forbidden_support']):
                break
        else:
            self._has_reset = False
            raise RuntimeError(
                f'No valid randomized reset in {c.reset_max_attempts} attempts; '
                f'last penetration={reset_contacts["max_penetration"]:.6g} m, '
                f'forbidden_support={reset_contacts["forbidden_support"]}')
        self._reset_attempts = attempt
        self._reset_penetration = float(reset_contacts['max_penetration'])
        self._initial_position = self.data.xpos[self.ids['cube_body']].copy()
        self.previous_action[:] = 0.
        self.monitor = GraspMonitor(c)
        self._has_reset = True
        self._refresh()
        info = self._info()
        info.update(mass_factor=float(mass_factor), friction_factor=float(friction_factor))
        return self._observation(), info

    def _refresh(self):
        self._contacts = native_contacts(self.model, self.data, self.ids)
        self._linear, self._angular = native_cube_velocity(self.model, self.data, self.ids)
        self._displacement = float(np.linalg.norm(
            self.data.xpos[self.ids['cube_body']] - self._initial_position))

    def _observation(self):
        q = self.ids['cube_qpos_adr']
        quat = self.data.qpos[q + 3:q + 7].copy()
        if quat[0] < 0:
            quat = -quat
        ctrl = 2 * (self.data.ctrl - self.ids['ctrl_lower']) / self._range - 1
        values = np.concatenate([
            self.data.qpos[self.ids['hand_qpos_ids']],
            self.data.qvel[self.ids['hand_dof_ids']], ctrl, self.previous_action,
            self.data.xpos[self.ids['cube_body']] - self._initial_position, quat,
            self._linear, self._angular,
            (self._contacts['finger_forces'] >= self.config.contact_force).astype(float),
            [self.monitor.stable_seconds / self.config.hold_seconds],
        ])
        return np.nan_to_num(values, nan=0., posinf=1e6, neginf=-1e6).astype(np.float32)

    def _info(self):
        return {
            'success': bool(self.monitor.success), 'is_success': bool(self.monitor.success),
            'failure_reason': self.monitor.failure_reason,
            'stable_seconds': float(self.monitor.stable_seconds),
            'max_stable_seconds': float(self.monitor.max_stable_seconds),
            'grasp_acquired': bool(self.monitor.max_stable_seconds >= .1 - 1e-10),
            'cube_displacement': self._displacement,
            'finger_contacts': int(np.count_nonzero(
                self._contacts['finger_forces'] >= self.config.contact_force)),
            'finger_forces': np.asarray(self._contacts['finger_forces']).copy(),
            'peak_penetration': float(self.monitor.peak_penetration),
            'elapsed_seconds': float(self.monitor.elapsed),
            'linear_speed': float(np.linalg.norm(self._linear)),
            'angular_speed': float(np.linalg.norm(self._angular)),
            'forbidden_support': bool(self._contacts['forbidden_support']),
            'reset_attempts': int(self._reset_attempts),
            'reset_rejected': int(self._reset_attempts - 1),
            'reset_penetration': self._reset_penetration,
        }

    def step(self, action):
        if not self._has_reset:
            raise RuntimeError('Call reset before step')
        if self.monitor.done:
            raise RuntimeError('Episode ended; call reset before the next step')
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError('Action must be seven finite values')
        action = np.clip(action, -1., 1.)
        c = self.config
        self.data.ctrl[:] = np.clip(self.data.ctrl + c.action_delta_fraction * self._range * action,
                                    self.ids['ctrl_lower'], self.ids['ctrl_upper'])
        reward = 0.
        elapsed_before = self.monitor.elapsed
        for _ in range(c.substeps):
            time_before = self.data.time
            mujoco.mj_step(self.model, self.data)
            # mj_step leaves position-dependent fields from the beginning of
            # its final integration step; forward makes replay/observations agree.
            mujoco.mj_forward(self.model, self.data)
            bad = not (np.all(np.isfinite(self.data.qpos)) and
                       np.all(np.isfinite(self.data.qvel))) or self.data.time < time_before
            self._refresh()
            stable, valid = self.monitor.update(
                self._contacts, self._linear, self._angular, self._displacement,
                float(self.data.xpos[self.ids['cube_body'], 2]), bad)
            contacts = self._contacts['finger_forces'] >= c.contact_force
            if valid:
                contact_reward = (min(np.count_nonzero(contacts), c.min_fingers)
                                  / c.min_fingers) * (.5 + .5 * contacts[0])
                reward += c.sim_dt * (
                    contact_reward + 2. * stable
                    + .2 * np.exp(-(self._displacement / .02) ** 2)
                    - .1 * min(float(np.linalg.norm(self._linear) / .1) ** 2, 25.)
                    - .02 * min(float(np.linalg.norm(self._angular)) ** 2, 25.)
                    - .02)
            else:
                reward -= c.sim_dt
            if self.monitor.done:
                reward += 10. if self.monitor.success else -5.
                break
        elapsed = self.monitor.elapsed - elapsed_before
        reward -= elapsed * (.05 * float(np.mean((action - self.previous_action) ** 2))
                             + .005 * float(np.mean(action ** 2)))
        self.previous_action = action.copy()
        timeout = self.monitor.failure_reason == 'timeout'
        return self._observation(), float(reward), bool(self.monitor.done and not timeout), \
            bool(timeout), self._info()

    def baseline_action(self, mode='close'):
        if mode == 'zero':
            return np.zeros(7, dtype=np.float32)
        if mode == 'random':
            return self.np_random.uniform(-1, 1, 7).astype(np.float32)
        if mode != 'close':
            raise ValueError('Unknown baseline mode')
        if 'grasp_ctrl' in self.reference:
            target = self.reference['grasp_ctrl']
        else:
            target = self.reference['ctrl'].copy()
            target[[0, 1, 2, 3, 6]] -= .1 * self._range[[0, 1, 2, 3, 6]]
        return np.clip((target - self.data.ctrl) /
                       (self.config.action_delta_fraction * self._range), -1, 1).astype(np.float32)

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.config.render_height,
                                            width=self.config.render_width)
        self._renderer.update_scene(self.data, camera=self._camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
