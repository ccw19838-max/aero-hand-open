"""Goal orientation, temporal success, and native Gym I/O regressions."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from experiments.cube_reorient.config import CubeReorientConfig
from experiments.cube_reorient.env import (
    AXES, CubeReorientEnv, ReorientMonitor, orientation_error,
    quaternion_multiply, target_quaternion,
)
from experiments.cube_reorient.model import build_model, model_fingerprint


IDENTITY = np.array([1., 0., 0., 0.])


class OrientationTests(unittest.TestCase):
    def test_quaternion_sign_does_not_change_error(self):
        current = target_quaternion(IDENTITY, 'z', 32.)
        target = target_quaternion(IDENTITY, 'x', 90.)
        expected_vector, expected_angle = orientation_error(current, target)
        for qsign in (-1, 1):
            for tsign in (-1, 1):
                vector, angle = orientation_error(qsign * current, tsign * target)
                np.testing.assert_allclose(vector, expected_vector, atol=1e-12)
                self.assertAlmostEqual(angle, expected_angle)

    def test_target_composes_about_fixed_world_axis(self):
        initial = target_quaternion(IDENTITY, 'z', 60.)
        x90 = target_quaternion(IDENTITY, 'x')
        goal = target_quaternion(initial, 'x')
        expected = quaternion_multiply(x90, initial)
        _, error = orientation_error(goal, expected)
        self.assertLess(error, 1e-12)
        _, wrong = orientation_error(goal, quaternion_multiply(initial, x90))
        self.assertGreater(np.rad2deg(wrong), 30.)

    def test_colored_cube_symmetries_are_not_equivalent(self):
        target = target_quaternion(IDENTITY, 'x')
        _, error = orientation_error(IDENTITY, target)
        self.assertAlmostEqual(np.rad2deg(error), 90.)
        wrong = target_quaternion(IDENTITY, 'y')
        _, error = orientation_error(wrong, target)
        self.assertAlmostEqual(np.rad2deg(error), 120.)

    def test_installed_axis_basis_rotates_goal_away_from_world_axis(self):
        a = np.deg2rad(20.)
        basis = np.array([[np.cos(a), 0., np.sin(a)], [0., 1., 0.],
                          [-np.sin(a), 0., np.cos(a)]])
        for index, axis in enumerate(AXES):
            goal = target_quaternion(IDENTITY, axis, axis_basis=basis)
            np.testing.assert_allclose(goal[1:], basis[:, index] / np.sqrt(2), atol=1e-12)
            monitor = ReorientMonitor(CubeReorientConfig(), IDENTITY, axis, axis_basis=basis)
            contacts = dict(finger_forces=np.zeros(5), forbidden_support=False, max_penetration=0.)
            monitor.update(goal, contacts, np.zeros(3), np.zeros(3), 0., .3)
            self.assertLess(monitor.angle_error_radians, 1e-12)
            self.assertAlmostEqual(np.rad2deg(monitor.signed_progress_radians), 90.)
        world_z_goal = target_quaternion(IDENTITY, 'z')
        hand_z_goal = target_quaternion(IDENTITY, 'z', axis_basis=basis)
        _, error = orientation_error(world_z_goal, hand_z_goal)
        self.assertGreater(np.rad2deg(error), 10.)


class ReorientationMonitorTests(unittest.TestCase):
    def setUp(self):
        self.config = CubeReorientConfig()
        self.contacts = dict(finger_forces=np.array([.2, .2, .2, 0., 0.]),
                             forbidden_support=False, max_penetration=0.)

    def tick(self, monitor, quaternion=None, contacts=None, speed=0., displacement=0.):
        return monitor.update(IDENTITY if quaternion is None else quaternion,
                              self.contacts if contacts is None else contacts,
                              np.array([speed, 0., 0.]), np.zeros(3), displacement, .3)

    def test_correct_goal_requires_continuous_half_second(self):
        monitor = ReorientMonitor(self.config, IDENTITY, 'x')
        goal = monitor.goal_quaternion
        for _ in range(249):
            self.tick(monitor, goal)
        self.assertFalse(monitor.done)
        self.tick(monitor, goal, speed=.04)
        self.assertEqual(monitor.stable_seconds, 0.)
        for _ in range(249):
            self.tick(monitor, goal)
        self.assertFalse(monitor.done)
        self.tick(monitor, goal)
        self.assertTrue(monitor.success)

    def test_static_wrong_axis_partial_and_negative_turns_fail(self):
        config = replace(self.config, episode_seconds=1.)
        for quaternion in (IDENTITY, target_quaternion(IDENTITY, 'y'),
                           target_quaternion(IDENTITY, 'x', 45.),
                           target_quaternion(IDENTITY, 'x', -90.),
                           target_quaternion(IDENTITY, 'x', 180.)):
            monitor = ReorientMonitor(config, IDENTITY, 'x')
            for _ in range(500):
                self.tick(monitor, quaternion)
            self.assertFalse(monitor.success)
            self.assertEqual(monitor.failure_reason, 'timeout')
            self.assertEqual(monitor.stable_seconds, 0.)

    def test_finger_contact_and_angle_departure_reset_hold(self):
        monitor = ReorientMonitor(self.config, IDENTITY, 'z')
        for _ in range(100):
            self.tick(monitor, monitor.goal_quaternion)
        no_thumb = dict(self.contacts, finger_forces=np.array([0., .3, .3, .3, 0.]))
        self.tick(monitor, monitor.goal_quaternion, no_thumb)
        self.assertEqual(monitor.stable_seconds, 0.)
        for _ in range(100):
            self.tick(monitor, monitor.goal_quaternion)
        self.tick(monitor, target_quaternion(IDENTITY, 'z', 79.))
        self.assertEqual(monitor.stable_seconds, 0.)

    def test_failure_latches_and_no_future_goal_can_revive_it(self):
        monitor = ReorientMonitor(self.config, IDENTITY, 'y')
        self.tick(monitor, monitor.goal_quaternion,
                  dict(self.contacts, forbidden_support=True))
        self.assertEqual(monitor.failure_reason, 'forbidden_support')
        elapsed = monitor.elapsed
        for _ in range(300):
            self.assertEqual(self.tick(monitor, monitor.goal_quaternion), (False, False, 0.))
        self.assertEqual(monitor.elapsed, elapsed)
        self.assertFalse(monitor.success)

    def test_penetration_blocks_progress_and_times_out_after_point_one_second(self):
        monitor = ReorientMonitor(self.config, IDENTITY, 'z')
        contacts = dict(self.contacts, max_penetration=.0011)
        for _ in range(49):
            stable, valid, _ = self.tick(monitor, monitor.goal_quaternion, contacts)
        self.assertFalse(stable or valid or monitor.done)
        self.tick(monitor, monitor.goal_quaternion, contacts)
        self.assertEqual(monitor.failure_reason, 'penetration')

    def test_reverse_progress_is_negative_and_round_trip_has_zero_potential(self):
        monitor = ReorientMonitor(self.config, IDENTITY, 'x')
        _, _, forward = self.tick(monitor, target_quaternion(IDENTITY, 'x', 45.))
        _, _, reverse = self.tick(monitor, IDENTITY)
        self.assertGreater(forward, 0.)
        self.assertLess(reverse, 0.)
        self.assertAlmostEqual(forward + reverse, 0.)

    def test_signed_progress_survives_pi_and_quaternion_sign_changes(self):
        monitor = ReorientMonitor(self.config, IDENTITY, 'z')
        for index, angle in enumerate((0., 90., 179., 181., 270., 359.)):
            quaternion = target_quaternion(IDENTITY, 'z', angle)
            self.tick(monitor, (-1 if index % 2 else 1) * quaternion, speed=1.)
        self.assertAlmostEqual(np.rad2deg(monitor.signed_progress_radians), 359.)
        self.assertFalse(monitor.success)

    def test_negative_270_and_positive_450_do_not_alias_single_positive_90(self):
        for finish in (-270, 450):
            monitor = ReorientMonitor(self.config, IDENTITY, 'z')
            direction = 10 if finish > 0 else -10
            for angle in range(0, finish + direction, direction):
                self.tick(monitor, target_quaternion(IDENTITY, 'z', angle), speed=1.)
            final = target_quaternion(IDENTITY, 'z', finish)
            for _ in range(300):
                self.tick(monitor, final)
            self.assertLess(monitor.angle_error_radians, 1e-12)
            self.assertAlmostEqual(np.rad2deg(monitor.signed_progress_radians), finish)
            self.assertFalse(monitor.progress_in_window)
            self.assertFalse(monitor.success)
            self.assertEqual(monitor.stable_seconds, 0.)


class EnvironmentContractTests(unittest.TestCase):
    """Synthetic free-space fixture verifies task plumbing, not prepared grasp quality."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.reference = Path(self.directory.name) / 'reference.npz'
        self.config = CubeReorientConfig()
        model = build_model(self.config)
        data = mujoco.MjData(model)
        data.ctrl[:] = np.mean(model.actuator_ctrlrange, axis=1)
        np.savez(self.reference, qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl,
                 grasp_ctrl=data.ctrl - .01 * np.ptp(model.actuator_ctrlrange, axis=1))
        self.report = dict(approved=True, config=self.config.to_dict(),
                           model_fingerprint=model_fingerprint(self.config))
        self.report_path = self.reference.parent / 'prepare_report.json'
        self.report_path.write_text(json.dumps(self.report))

    def tearDown(self):
        self.directory.cleanup()

    def test_config_roundtrip_and_goal_validation(self):
        self.assertEqual(CubeReorientConfig.from_dict(self.config.to_dict()), self.config)
        with self.assertRaises(ValueError):
            replace(self.config, target_degrees=180.)
        with self.assertRaises(ValueError):
            replace(self.config, angle_tolerance_degrees=90.)

    def test_axis_goals_use_actual_randomized_reset_and_observation_is_75(self):
        env = CubeReorientEnv(self.reference)
        initial_positions = []
        try:
            for index, axis in enumerate(AXES):
                obs, info = env.reset(seed=13, options={'axis': axis})
                self.assertEqual(info['axis'], axis)
                self.assertAlmostEqual(info['angle_error_degrees'], 90.)
                self.assertEqual(obs.shape, (75,))
                np.testing.assert_array_equal(obs[72:75], np.eye(3)[index])
                qa = env.ids['cube_qpos_adr']
                initial_positions.append(env.data.qpos[qa:qa+7].copy())
                expected = target_quaternion(env.data.qpos[qa+3:qa+7], axis,
                                             axis_basis=env.axis_basis)
                np.testing.assert_allclose(obs[65:69], expected, atol=1e-6)
                self.assertEqual(info['goal_frame'], 'hand_fixed')
                np.testing.assert_allclose(info['axis_world_vector'], env.axis_basis[:, index])
            for position in initial_positions[1:]:
                np.testing.assert_array_equal(position, initial_positions[0])
        finally:
            env.close()

    def test_seed_reproducibility_and_action_mapping(self):
        env = CubeReorientEnv(self.reference, axis='y')
        try:
            first, info = env.reset(seed=11)
            second, second_info = env.reset(seed=11)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(second_info['axis'], 'y')
            before = env.data.ctrl.copy()
            action = np.array([2., -2., .5, -.5, 0., .25, -.25])
            clipped = np.clip(action, -1., 1.)
            obs, reward, _, _, info = env.step(action)
            expected = np.clip(before + .01 * env._range * clipped,
                               env.ids['ctrl_lower'], env.ids['ctrl_upper'])
            np.testing.assert_allclose(env.data.ctrl, expected)
            np.testing.assert_array_equal(obs[39:46], clipped.astype(np.float32))
            self.assertAlmostEqual(info['elapsed_seconds'], .05)
            self.assertEqual(obs.shape, (75,))
            self.assertTrue(np.all(np.isfinite(obs)) and np.isfinite(reward))
        finally:
            env.close()

    def test_invalid_reset_is_rejected_after_a_bounded_number_of_attempts(self):
        env = CubeReorientEnv(self.reference)
        invalid = dict(finger_forces=np.zeros(5), max_penetration=.0011,
                       forbidden_support=False)
        try:
            with patch('experiments.cube_reorient.env.native_contacts', return_value=invalid) as fn:
                with self.assertRaisesRegex(RuntimeError, 'No valid randomized reset'):
                    env.reset(seed=7)
            self.assertEqual(fn.call_count, 32)
            with self.assertRaisesRegex(RuntimeError, 'reset before step'):
                env.step(np.zeros(7))
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
