"""Five-finger phase transitions and non-equivalent rotation trajectories."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from experiments.cube_envelop.config import CubeReorientConfig
from experiments.cube_envelop.env import (
    AXES, CubeReorientEnv, ReorientMonitor, orientation_error,
    physical_substep_reward, quaternion_multiply, target_quaternion,
)
from experiments.cube_envelop.model import build_model, model_fingerprint

IDENTITY = np.array([1., 0., 0., 0.])


class EnvelopePhaseTests(unittest.TestCase):
    def setUp(self):
        self.config = CubeReorientConfig()
        self.contacts = dict(finger_forces=np.full(5, .2), forbidden_support=False,
                             max_penetration=0.)

    def tick(self, monitor, quaternion=IDENTITY, contacts=None, speed=0.):
        return monitor.update(quaternion, self.contacts if contacts is None else contacts,
                              np.array([speed, 0., 0.]), np.zeros(3), 0., .3)

    def capture(self, monitor, quaternion=IDENTITY):
        for _ in range(round(monitor.config.capture_seconds / monitor.config.sim_dt)):
            self.tick(monitor, quaternion)
        self.assertTrue(monitor.capture_succeeded)

    def test_three_fingers_and_four_without_thumb_cannot_capture(self):
        for forces in ([.2, .2, .2, 0., 0.], [0., .2, .2, .2, .2]):
            monitor = ReorientMonitor(self.config, IDENTITY, 'x')
            contacts = dict(self.contacts, finger_forces=np.array(forces))
            for _ in range(1000):
                self.tick(monitor, contacts=contacts)
            self.assertFalse(monitor.capture_succeeded or monitor.success)
            self.assertEqual(monitor.failure_reason, 'capture_timeout')
            self.assertEqual(monitor.failure_phase, 'acquire')

    def test_capture_requires_uninterrupted_five_finger_contact(self):
        monitor = ReorientMonitor(self.config, IDENTITY, 'x')
        for _ in range(150):
            self.tick(monitor)
        missing = dict(self.contacts, finger_forces=np.array([.2, .2, .2, .2, 0.]))
        self.tick(monitor, contacts=missing)
        for _ in range(249):
            self.tick(monitor)
        self.assertFalse(monitor.capture_succeeded)
        self.tick(monitor)
        self.assertTrue(monitor.capture_succeeded)
        self.assertEqual(monitor.phase, 'rotate')
        self.assertFalse(monitor.success)

    def test_goal_and_progress_start_at_capture_pose(self):
        monitor = ReorientMonitor(self.config, IDENTITY, 'x')
        settled = target_quaternion(IDENTITY, 'z', 25.)
        self.capture(monitor, settled)
        expected = target_quaternion(settled, 'x')
        _, error = orientation_error(monitor.goal_quaternion, expected)
        self.assertLess(error, 1e-12)
        self.assertAlmostEqual(monitor.signed_progress_radians, 0.)
        self.assertAlmostEqual(np.rad2deg(monitor.angle_error_radians), 90.)
        _, _, progress = self.tick(monitor, settled)
        self.assertAlmostEqual(progress, 0.)

    def test_rotate_permits_some_fingers_to_leave(self):
        monitor = ReorientMonitor(self.config, IDENTITY, 'x')
        self.capture(monitor)
        partial = dict(self.contacts, finger_forces=np.array([.2, 0., .2, .2, 0.]))
        for angle in np.linspace(0., 45., 100):
            self.tick(monitor, target_quaternion(IDENTITY, 'x', angle), partial, speed=.1)
        self.assertFalse(monitor.done)
        self.assertEqual(monitor.phase, 'rotate')
        self.assertAlmostEqual(np.rad2deg(monitor.signed_progress_radians), 45.)

    def test_recapture_missing_one_finger_cannot_succeed_then_all_five_can(self):
        monitor = ReorientMonitor(self.config, IDENTITY, 'z')
        self.capture(monitor)
        missing = dict(self.contacts, finger_forces=np.array([.2, .2, .2, .2, 0.]))
        goal = monitor.goal_quaternion
        for _ in range(300):
            self.tick(monitor, goal, missing)
        self.assertEqual(monitor.phase, 'recapture')
        self.assertFalse(monitor.success)
        self.assertEqual(monitor.stable_seconds, 0.)
        for _ in range(249):
            self.tick(monitor, goal)
        self.assertFalse(monitor.done)
        self.tick(monitor, goal)
        self.assertTrue(monitor.success and monitor.recapture_succeeded)

    def test_hold_is_two_additional_seconds_after_capture_and_has_no_rotation(self):
        config = replace(self.config, task_mode='hold')
        monitor = ReorientMonitor(config, IDENTITY, 'y')
        self.capture(monitor)
        self.assertEqual(monitor.hold_stable_seconds, 0.)
        self.assertFalse(monitor.target_active)
        for _ in range(999):
            _, _, delta = self.tick(monitor)
            self.assertEqual(delta, 0.)
        self.assertFalse(monitor.done)
        self.tick(monitor)
        self.assertTrue(monitor.success)
        self.assertFalse(monitor.recapture_succeeded)
        self.assertAlmostEqual(monitor.elapsed, 2.5)

    def test_full_contact_loss_is_bounded_and_failure_latches(self):
        monitor = ReorientMonitor(self.config, IDENTITY, 'x')
        self.capture(monitor)
        empty = dict(self.contacts, finger_forces=np.zeros(5))
        for _ in range(74):
            self.tick(monitor, contacts=empty)
        self.assertFalse(monitor.done)
        self.tick(monitor, contacts=empty)
        self.assertEqual(monitor.failure_reason, 'lost_finger_contact')
        elapsed = monitor.elapsed
        self.tick(monitor, monitor.goal_quaternion)
        self.assertEqual(monitor.elapsed, elapsed)
        self.assertFalse(monitor.success)

    def test_each_finger_contact_duration_is_recorded(self):
        monitor = ReorientMonitor(self.config, IDENTITY, 'x')
        self.capture(monitor)
        missing = dict(self.contacts, finger_forces=np.array([.2, 0., .2, .2, .2]))
        for _ in range(10):
            self.tick(monitor, contacts=missing)
        np.testing.assert_allclose(monitor.contact_seconds, [.52, .5, .52, .52, .52])

    def test_wrong_axis_partial_and_extra_turns_cannot_complete(self):
        for axis, finish in [('y', 90), ('x', 45), ('x', -270), ('x', 450)]:
            monitor = ReorientMonitor(self.config, IDENTITY, 'x')
            self.capture(monitor)
            direction = 10 if finish > 0 else -10
            for angle in np.arange(0., finish, direction):
                self.tick(monitor, target_quaternion(IDENTITY, axis, angle), speed=.1)
            final = target_quaternion(IDENTITY, axis, finish)
            for _ in range(300):
                self.tick(monitor, final)
            self.assertFalse(monitor.success)
            self.assertEqual(monitor.stable_seconds, 0.)

    def test_static_envelope_is_not_a_rotation_success(self):
        monitor = ReorientMonitor(replace(self.config, episode_seconds=3.), IDENTITY, 'x')
        self.capture(monitor)
        for _ in range(1250):
            self.tick(monitor)
        self.assertFalse(monitor.success)
        self.assertEqual(monitor.failure_reason, 'timeout')

    def test_zero_finger_contact_cannot_earn_rotation_or_near_goal_reward(self):
        monitor = ReorientMonitor(self.config, IDENTITY, 'x')
        self.capture(monitor)
        monitor.angle_error_radians = 0.
        empty = dict(self.contacts, finger_forces=np.zeros(5))
        kwargs = (self.config, monitor, empty, np.zeros(3), np.zeros(3), False, True)
        forward = physical_substep_reward(*kwargs, .5)
        stationary = physical_substep_reward(*kwargs, 0.)
        reverse = physical_substep_reward(*kwargs, -.5)
        self.assertEqual(forward, stationary)
        self.assertLess(forward, 0.)
        self.assertLess(reverse, stationary)

    def test_fourth_and_fifth_fingers_have_marginal_acquisition_reward(self):
        monitor = ReorientMonitor(self.config, IDENTITY, 'x')
        rewards = []
        for count in [3, 4, 5]:
            forces = np.zeros(5); forces[:count] = .2
            rewards.append(physical_substep_reward(
                self.config, monitor, dict(self.contacts, finger_forces=forces),
                np.zeros(3), np.zeros(3), False, True, 0.))
        self.assertLess(rewards[0], rewards[1])
        self.assertLess(rewards[1], rewards[2])


class QuaternionTests(unittest.TestCase):
    def test_sign_fixed_tilt_axis_and_colored_faces(self):
        angle = np.deg2rad(20.)
        basis = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                          [-np.sin(angle), 0., np.cos(angle)]])
        initial = target_quaternion(IDENTITY, 'z', 30.)
        goal = target_quaternion(initial, 'x', axis_basis=basis)
        for qsign in (-1, 1):
            for tsign in (-1, 1):
                _, error = orientation_error(qsign * initial, tsign * goal)
                self.assertAlmostEqual(np.rad2deg(error), 90.)
        hand_rotation = np.r_[np.sqrt(.5), basis[:, 0] * np.sqrt(.5)]
        _, error = orientation_error(goal, quaternion_multiply(hand_rotation, initial))
        self.assertLess(error, 1e-12)
        _, error = orientation_error(goal, target_quaternion(initial, 'y'))
        self.assertGreater(np.rad2deg(error), 10.)


class EnvironmentContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.reference = Path(self.directory.name) / 'reference.npz'
        self.config = CubeReorientConfig()
        model = build_model(self.config); data = mujoco.MjData(model)
        data.ctrl[:] = model.actuator_ctrlrange.mean(axis=1)
        np.savez(self.reference, qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl)
        self.report = dict(approved=True, config=self.config.to_dict(),
                           model_fingerprint=model_fingerprint(self.config))
        self.report_path = self.reference.parent / 'prepare_report.json'
        self.report_path.write_text(json.dumps(self.report))

    def tearDown(self):
        self.directory.cleanup()

    def test_observation_78_phase_and_seeded_action_contract(self):
        env = CubeReorientEnv(self.reference)
        try:
            first, info = env.reset(seed=7, options={'axis': 'y'})
            second, _ = env.reset(seed=7, options={'axis': 'y'})
            np.testing.assert_array_equal(first, second)
            self.assertEqual(first.shape, (78,))
            np.testing.assert_array_equal(first[75:78], [1., 0., 0.])
            self.assertEqual(info['phase'], 'acquire')
            self.assertFalse(info['target_active'] or info['capture_succeeded'])
            before = env.data.ctrl.copy()
            action = np.array([2., -2., 0., .1, -.1, 0., .5])
            obs, reward, _, _, info = env.step(action)
            expected = np.clip(before + .01 * env._range * np.clip(action, -1, 1),
                               env.ids['ctrl_lower'], env.ids['ctrl_upper'])
            np.testing.assert_allclose(env.data.ctrl, expected)
            self.assertTrue(np.all(np.isfinite(obs)) and np.isfinite(reward))
            self.assertEqual(info['contact_seconds'].shape, (5,))
            self.assertEqual(info['each_finger_contact'].shape, (5,))
        finally:
            env.close()

    def test_old_three_finger_reference_cannot_be_relabelled(self):
        self.report['config'].pop('variant')
        self.report_path.write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, 'five-finger envelope'):
            CubeReorientEnv(self.reference, config=self.config)

    def test_config_roundtrip_and_five_finger_requirement(self):
        self.assertEqual(CubeReorientConfig.from_dict(self.config.to_dict()), self.config)
        with self.assertRaises(ValueError):
            replace(self.config, min_fingers=3)


if __name__ == '__main__':
    unittest.main()
