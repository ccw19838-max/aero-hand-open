"""Five-finger envelope acquisition followed by whole-cube manipulation.

X points from wrist to fingertips, Y across the hand, and Z out of its upward
palm. The original approximately 20-degree palm tilt is retained. These axes
are fixed by the installation and need not coincide with world XYZ. Goals are
composed on the left of the actual randomized reset orientation; colored faces
are never interchangeable. Actor inputs include simulation ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from .config import CubeReorientConfig
from .model import (build_model, fixed_axis_basis, model_fingerprint, model_ids,
                    native_contacts, native_cube_velocity)

AXES = ('x', 'y', 'z')


def unit_quaternion(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (4,) or not np.all(np.isfinite(value)) or np.linalg.norm(value) < 1e-12:
        raise ValueError('Quaternion must contain four finite values and have nonzero norm')
    return value / np.linalg.norm(value)


def quaternion_multiply(left, right):
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,
                     w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2])


def quaternion_conjugate(quaternion):
    return np.asarray(quaternion) * np.array([1., -1., -1., -1.])


def canonical_quaternion(quaternion):
    quaternion = unit_quaternion(quaternion)
    # At exactly pi choose a deterministic vector sign as well as w's sign.
    first = next((v for v in quaternion if abs(v) > 1e-12), 1.)
    return quaternion if first >= 0 else -quaternion


def validated_axis_basis(axis_basis=None):
    basis = np.eye(3) if axis_basis is None else np.asarray(axis_basis, dtype=float)
    if (basis.shape != (3, 3) or not np.all(np.isfinite(basis))
            or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-6, rtol=0.)
            or not np.isclose(np.linalg.det(basis), 1., atol=1e-6, rtol=0.)):
        raise ValueError('Axis basis must be a right-handed orthonormal 3x3 matrix')
    return basis.copy()


def target_quaternion(initial, axis, degrees=90., axis_basis=None):
    if axis not in AXES:
        raise ValueError('Axis must be x, y, or z')
    rotation = np.zeros(4)
    half = np.deg2rad(degrees) / 2
    rotation[0] = np.cos(half)
    rotation[1:] = validated_axis_basis(axis_basis)[:, AXES.index(axis)] * np.sin(half)
    return canonical_quaternion(quaternion_multiply(rotation, unit_quaternion(initial)))


def orientation_error(current, target):
    """World-frame shortest rotation vector and SO(3) angle, invariant to q/-q."""
    error = canonical_quaternion(quaternion_multiply(
        unit_quaternion(target), quaternion_conjugate(unit_quaternion(current))))
    length = float(np.linalg.norm(error[1:]))
    angle = 2 * np.arctan2(length, max(0., float(error[0])))
    rotvec = error[1:] * (angle / length) if length > 1e-12 else np.zeros(3)
    return rotvec, float(angle)


@dataclass
class ReorientMonitor:
    """Acquire all five contacts, rotate freely, then recapture with all five."""
    config: CubeReorientConfig
    initial_quaternion: np.ndarray
    axis: str
    axis_basis: np.ndarray = field(default_factory=lambda: np.eye(3))
    elapsed: float = 0.
    stable_seconds: float = 0.
    max_stable_seconds: float = 0.
    grasp_stable_seconds: float = 0.
    max_grasp_stable_seconds: float = 0.
    penetration_seconds: float = 0.
    region_seconds: float = 0.
    peak_penetration: float = 0.
    success: bool = False
    failure_reason: str = ''
    signed_progress_radians: float = 0.
    goal_quaternion: np.ndarray = field(init=False)
    error_rotvec: np.ndarray = field(init=False)
    angle_error_radians: float = field(init=False)
    best_angle_error_radians: float = field(init=False)
    _last_twist: float = 0.
    phase: str = 'acquire'
    capture_succeeded: bool = False
    recapture_succeeded: bool = False
    target_active: bool = False
    capture_time: float | None = None
    hold_stable_seconds: float = 0.
    no_finger_contact_seconds: float = 0.
    contact_seconds: np.ndarray = field(default_factory=lambda: np.zeros(5))
    failure_phase: str = ''

    def __post_init__(self):
        self.initial_quaternion = unit_quaternion(self.initial_quaternion)
        self.axis_basis = validated_axis_basis(self.axis_basis)
        self._rebase_goal(self.initial_quaternion)

    def _rebase_goal(self, quaternion):
        """Before capture the goal is provisional; capture freezes its origin."""
        self.initial_quaternion = unit_quaternion(quaternion)
        self.goal_quaternion = (self.initial_quaternion.copy() if self.config.task_mode == 'hold'
                                else target_quaternion(self.initial_quaternion, self.axis,
                                                       self.config.target_degrees, self.axis_basis))
        self.error_rotvec, self.angle_error_radians = orientation_error(
            self.initial_quaternion, self.goal_quaternion)
        self.best_angle_error_radians = self.angle_error_radians
        self.signed_progress_radians = 0.
        self._last_twist = 0.

    @property
    def done(self):
        return self.success or bool(self.failure_reason)

    @property
    def progress_in_window(self):
        return abs(np.rad2deg(self.signed_progress_radians) - self.config.target_degrees) \
            <= self.config.angle_tolerance_degrees + 1e-10

    def _update_orientation(self, quaternion):
        previous = self.angle_error_radians
        self.error_rotvec, self.angle_error_radians = orientation_error(
            quaternion, self.goal_quaternion)
        self.best_angle_error_radians = min(self.best_angle_error_radians,
                                            self.angle_error_radians)
        relative = quaternion_multiply(unit_quaternion(quaternion),
                                       quaternion_conjugate(self.initial_quaternion))
        component = float(np.dot(relative[1:], self.axis_basis[:, AXES.index(self.axis)]))
        # A 180-degree swing perpendicular to the requested axis has undefined
        # twist. Leave the diagnostic unchanged; success still uses full SO(3).
        if relative[0] ** 2 + component ** 2 > 1e-12:
            twist = 2 * np.arctan2(component, relative[0])
            increment = (twist - self._last_twist + np.pi) % (2 * np.pi) - np.pi
            self.signed_progress_radians += increment
            self._last_twist = twist
        return (previous - self.angle_error_radians) / np.deg2rad(self.config.target_degrees)

    def update(self, quaternion, contacts, linear, angular, displacement,
               cube_height, numerical_failure=False):
        """Return (stable, physically_valid, potential_delta), with latched events."""
        if self.done:
            return False, False, 0.
        c = self.config
        self.elapsed += c.sim_dt
        if self.target_active and not numerical_failure:
            delta = self._update_orientation(quaternion)
        else:
            delta = 0.
            if not numerical_failure:
                self._rebase_goal(quaternion)
        touching = np.asarray(contacts['finger_forces']) >= c.contact_force
        self.contact_seconds += touching.astype(float) * c.sim_dt
        enough = bool(np.all(touching))
        depth = float(contacts['max_penetration'])
        self.peak_penetration = max(self.peak_penetration, depth)
        penetrated = depth > c.max_penetration
        outside = displacement > c.max_displacement
        forbidden = bool(contacts['forbidden_support'])
        self.penetration_seconds = self.penetration_seconds + c.sim_dt if penetrated else 0.
        self.region_seconds = self.region_seconds + c.sim_dt if outside else 0.
        valid = not (penetrated or outside or forbidden or numerical_failure)
        near = self.angle_error_radians <= np.deg2rad(c.angle_tolerance_degrees) + 1e-12
        grasp_stable = bool(valid and enough and np.linalg.norm(linear) < c.grip_velocity
                            and np.linalg.norm(angular) < c.grip_omega)
        self.grasp_stable_seconds = self.grasp_stable_seconds + c.sim_dt if grasp_stable else 0.
        self.max_grasp_stable_seconds = max(self.max_grasp_stable_seconds,
                                            self.grasp_stable_seconds)
        just_captured = False
        if not self.capture_succeeded and self.grasp_stable_seconds + 1e-10 >= c.capture_seconds:
            self.capture_succeeded = True
            self.capture_time = self.elapsed
            just_captured = True
            self._rebase_goal(quaternion)
            if c.task_mode == 'reorient':
                self.target_active = True
                self.phase = 'rotate'
            delta = 0.
        if self.capture_succeeded and not touching.any():
            self.no_finger_contact_seconds += c.sim_dt
        else:
            self.no_finger_contact_seconds = 0.
        if c.task_mode == 'hold':
            if self.capture_succeeded and not just_captured:
                self.hold_stable_seconds = self.hold_stable_seconds + c.sim_dt if grasp_stable else 0.
            stable = bool(self.capture_succeeded and grasp_stable and not just_captured)
            self.stable_seconds = self.hold_stable_seconds
        else:
            # A full turn or -270 has the same pose as +90, but fails the net
            # single-quarter-turn window. Individual fingers may leave in rotate.
            at_goal = bool(self.target_active and near and self.progress_in_window)
            if self.target_active:
                self.phase = 'recapture' if at_goal else 'rotate'
            stable = bool(at_goal and grasp_stable)
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
        elif self.no_finger_contact_seconds + 1e-10 >= c.max_no_finger_contact_seconds:
            self.failure_reason = 'lost_finger_contact'
        elif not self.capture_succeeded and self.elapsed + 1e-10 >= c.capture_timeout:
            self.failure_reason = 'capture_timeout'
        elif c.task_mode == 'hold' and self.hold_stable_seconds + 1e-10 >= c.grip_hold_seconds:
            self.success = True
        elif c.task_mode == 'reorient' and self.target_active and self.stable_seconds + 1e-10 >= c.target_hold_seconds:
            self.success = True
            self.recapture_succeeded = True
        elif self.elapsed + 1e-10 >= c.episode_seconds:
            self.failure_reason = 'timeout'
        if self.failure_reason:
            self.failure_phase = self.phase
        return stable, valid, float(delta)


def physical_substep_reward(config, monitor, contacts, linear, angular,
                            stable, valid, potential_delta):
    """Task shaping only; terminal bonuses and action smoothness are separate."""
    c = config
    touching = np.asarray(contacts['finger_forces']) >= c.contact_force
    finger_supported = bool(touching.any())
    reward = 0.
    if c.task_mode == 'reorient' and monitor.target_active:
        credited = potential_delta if valid and finger_supported else min(potential_delta, 0.)
        reward += c.progress_reward * credited
    if not valid:
        return float(reward - c.sim_dt)
    # Every finger has marginal value. There is no three-finger cap.
    contact_reward = float(np.mean(touching))
    envelope_stage = c.task_mode == 'hold' or monitor.phase in ('acquire', 'recapture')
    envelope_stable = bool(np.all(touching) and np.linalg.norm(linear) < c.grip_velocity
                           and np.linalg.norm(angular) < c.grip_omega)
    stage_reward = (c.envelope_contact_reward * contact_reward
                    + c.envelope_stable_reward * envelope_stable) if envelope_stage else .02 * contact_reward
    # A brief flight/palm-only interval may precede contact recovery, but it
    # earns neither positive rotation progress nor a near-goal pose bonus.
    if c.task_mode == 'reorient' and monitor.target_active and finger_supported:
        nearby = np.exp(-(monitor.angle_error_radians / np.deg2rad(30.)) ** 2)
        stage_reward += c.near_goal_reward * nearby + c.stable_reward * stable
    thumb_excess = max(0., float(contacts['finger_forces'][0]) / c.thumb_force_soft_limit - 1.)
    reward += c.sim_dt * (stage_reward + c.thumb_contact_reward * touching[0]
        - c.thumb_force_penalty * min(thumb_excess ** 2, 25.) - .05
        - .05 * min(float(np.linalg.norm(linear) / .1) ** 2, 25.)
        - .005 * min(float(np.linalg.norm(angular)) ** 2, 100.))
    return float(reward)


class CubeReorientEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    def __init__(self, reference, config=None, render_mode=None, axis=None):
        super().__init__()
        if axis is not None and axis not in AXES:
            raise ValueError('Axis must be x, y, z, or None for uniform sampling')
        self.fixed_axis = axis
        self.reference_path = Path(reference).resolve()
        report_path = self.reference_path.parent / 'prepare_report.json'
        if not report_path.is_file():
            raise ValueError('Prepared reference requires prepare_report.json')
        report = json.loads(report_path.read_text())
        if report.get('approved') is not True:
            raise ValueError('Preparation has not approved this reference')
        if report.get('config', {}).get('variant') != 'four_finger_envelope_v1':
            raise ValueError('Reference must be prepared for five-finger envelope capture')
        self.config = config or CubeReorientConfig.from_dict(report['config'])
        if not isinstance(self.config, CubeReorientConfig):
            raise ValueError('Reorientation requires CubeReorientConfig')
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
        self.axis_basis = validated_axis_basis(fixed_axis_basis(self.model))
        for key, shape in [('qpos', (self.model.nq,)), ('qvel', (self.model.nv,)),
                           ('ctrl', (self.model.nu,))]:
            if key not in self.reference or self.reference[key].shape != shape:
                raise ValueError(f'Invalid prepared {key} shape')
            if not np.all(np.isfinite(self.reference[key])):
                raise ValueError(f'Prepared {key} contains non-finite values')
        if self.model.nu != 7:
            raise ValueError('Expected seven Aero actuator channels')
        self.action_space = gym.spaces.Box(-1., 1., (7,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (78,), dtype=np.float32)
        self.render_mode = render_mode
        if render_mode not in (None, 'rgb_array'):
            raise ValueError('Only rgb_array rendering is supported')
        self._renderer = None
        self._camera = mujoco.MjvCamera()
        self._camera.lookat[:] = [.1, 0., .19]
        self._camera.distance = .42
        self._camera.azimuth = 135.
        self._camera.elevation = -35.
        self._body_mass = self.model.body_mass[self.ids['cube_body']].copy()
        self._body_inertia = self.model.body_inertia[self.ids['cube_body']].copy()
        self._friction = self.model.geom_friction[self.ids['cube_geom']].copy()
        self._range = self.ids['ctrl_upper'] - self.ids['ctrl_lower']
        self.previous_action = np.zeros(7)
        self.monitor = None
        self._contacts = {'finger_forces': np.zeros(5), 'forbidden_support': False,
                          'max_penetration': 0.}
        self._linear = np.zeros(3)
        self._angular = np.zeros(3)
        self._initial_position = np.zeros(3)
        self._displacement = 0.
        self._has_reset = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        chosen_axis = (options or {}).get('axis', self.fixed_axis)
        if chosen_axis is not None and chosen_axis not in AXES:
            raise ValueError('Reset axis must be x, y, or z')
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
            noise_axis = self.np_random.normal(size=3)
            noise_axis /= max(float(np.linalg.norm(noise_axis)), 1e-12)
            angle = np.deg2rad(self.np_random.uniform(-c.reset_angle_noise_degrees,
                                                      c.reset_angle_noise_degrees))
            noise_q = np.r_[np.cos(angle / 2), noise_axis * np.sin(angle / 2)]
            self.data.qpos[q + 3:q + 7] = unit_quaternion(quaternion_multiply(
                noise_q, self.data.qpos[q + 3:q + 7]))
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
        if chosen_axis is None:
            chosen_axis = AXES[int(self.np_random.integers(3))]
        self.monitor = ReorientMonitor(c, self.data.qpos[q + 3:q + 7].copy(), chosen_axis,
                                        axis_basis=self.axis_basis)
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
        raw_quat = self.data.qpos[q + 3:q + 7]
        if not np.all(np.isfinite(raw_quat)) or np.linalg.norm(raw_quat) < 1e-12:
            # A numerical terminal transition still needs a finite Gym
            # observation; its failure flag prevents further rollout steps.
            raw_quat = self.monitor.initial_quaternion
        quat = canonical_quaternion(raw_quat)
        ctrl = 2 * (self.data.ctrl - self.ids['ctrl_lower']) / self._range - 1
        if self.monitor.phase == 'acquire' and not self.monitor.capture_succeeded:
            stable_progress = self.monitor.grasp_stable_seconds / self.config.capture_seconds
        elif self.config.task_mode == 'hold':
            stable_progress = self.monitor.hold_stable_seconds / self.config.grip_hold_seconds
        else:
            stable_progress = self.monitor.stable_seconds / self.config.target_hold_seconds
        values = np.concatenate([
            self.data.qpos[self.ids['hand_qpos_ids']],
            self.data.qvel[self.ids['hand_dof_ids']], ctrl, self.previous_action,
            self.data.xpos[self.ids['cube_body']] - self._initial_position, quat,
            self._linear, self._angular,
            (self._contacts['finger_forces'] >= self.config.contact_force).astype(float),
            [stable_progress],
            self.monitor.goal_quaternion, self.monitor.error_rotvec,
            np.eye(3)[AXES.index(self.monitor.axis)],
            np.eye(3)[('acquire', 'rotate', 'recapture').index(self.monitor.phase)],
        ])
        return np.nan_to_num(values, nan=0., posinf=1e6, neginf=-1e6).astype(np.float32)

    def _info(self):
        height = float(self.data.xpos[self.ids['cube_body'], 2])
        floor = self.model.geom('floor').id
        floor_contact = any(self.ids['cube_geom'] in (int(contact.geom1), int(contact.geom2))
                            and floor in (int(contact.geom1), int(contact.geom2))
                            and contact.dist <= 0 for contact in self.data.contact)
        return {
            'axis': self.monitor.axis,
            'task_mode': self.config.task_mode,
            'variant': self.config.variant,
            'phase': self.monitor.phase,
            'failure_phase': self.monitor.failure_phase,
            'capture_succeeded': bool(self.monitor.capture_succeeded),
            'envelope_acquired': bool(self.monitor.capture_succeeded),
            'recapture_succeeded': bool(self.monitor.recapture_succeeded),
            'target_active': bool(self.monitor.target_active),
            'capture_time': self.monitor.capture_time,
            'envelope_stable_seconds': float(self.monitor.grasp_stable_seconds),
            'hold_stable_seconds': float(self.monitor.hold_stable_seconds),
            'contact_seconds': self.monitor.contact_seconds.copy(),
            'each_finger_contact': (self._contacts['finger_forces'] >= self.config.contact_force).copy(),
            'thumb_force': float(self._contacts['finger_forces'][0]),
            'no_finger_contact_seconds': float(self.monitor.no_finger_contact_seconds),
            'goal_frame': 'hand_fixed',
            'axis_world_vector': self.axis_basis[:, AXES.index(self.monitor.axis)].copy(),
            'angle_error_degrees': float(np.rad2deg(self.monitor.angle_error_radians)),
            'best_angle_error_degrees': float(np.rad2deg(self.monitor.best_angle_error_radians)),
            'signed_progress_degrees': float(np.rad2deg(self.monitor.signed_progress_radians)),
            'progress_in_window': bool(self.monitor.progress_in_window),
            'success': bool(self.monitor.success), 'is_success': bool(self.monitor.success),
            'failure_reason': self.monitor.failure_reason,
            'stable_seconds': float(self.monitor.stable_seconds),
            'max_stable_seconds': float(self.monitor.max_stable_seconds),
            'grasp_acquired': bool(self.monitor.capture_succeeded),
            'goal_hold_acquired': bool(self.monitor.max_stable_seconds >= .1 - 1e-10),
            'grasp_stable_seconds': float(self.monitor.grasp_stable_seconds),
            'cube_displacement': self._displacement,
            'cube_height': height,
            'cube_position': self.data.xpos[self.ids['cube_body']].copy(),
            'dropped': bool(floor_contact or height < self.config.cube_size / 2),
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
            mujoco.mj_forward(self.model, self.data)
            bad = not (np.all(np.isfinite(self.data.qpos)) and
                       np.all(np.isfinite(self.data.qvel))) or self.data.time < time_before
            self._refresh()
            q = self.ids['cube_qpos_adr']
            stable, valid, delta = self.monitor.update(
                self.data.qpos[q + 3:q + 7], self._contacts, self._linear, self._angular,
                self._displacement, float(self.data.xpos[self.ids['cube_body'], 2]), bad)
            reward += physical_substep_reward(c, self.monitor, self._contacts,
                                               self._linear, self._angular, stable, valid, delta)
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
