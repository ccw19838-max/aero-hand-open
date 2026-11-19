"""Task checks cover temporal success and rejection of physically invalid holds."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from experiments.cube_grasp.config import CubeGraspConfig
from experiments.cube_grasp.env import CubeGraspEnv, GraspMonitor
from experiments.cube_grasp.model import build_model, model_fingerprint


class TemporalGraspTests(unittest.TestCase):
    def setUp(self):
        self.config = CubeGraspConfig()
        self.contacts = dict(finger_forces=np.array([.2, .2, .2, 0., 0.]),
                             forbidden_support=False, max_penetration=0.)

    def tick(self, monitor, contacts=None, speed=0., displacement=0.):
        return monitor.update(self.contacts if contacts is None else contacts,
                              np.array([speed, 0., 0.]), np.zeros(3), displacement, .3)

    def test_requires_uninterrupted_hold_and_latches_once(self):
        monitor = GraspMonitor(self.config)
        for _ in range(750):
            self.tick(monitor)
        self.assertFalse(monitor.done)
        self.tick(monitor, speed=.04)
        self.assertEqual(monitor.stable_seconds, 0.)
        for _ in range(999):
            self.tick(monitor)
        self.assertFalse(monitor.done)
        self.tick(monitor)
        self.assertTrue(monitor.success)
        elapsed = monitor.elapsed
        self.tick(monitor)
        self.assertEqual(monitor.elapsed, elapsed)

    def test_three_fingers_without_thumb_do_not_pass(self):
        monitor = GraspMonitor(self.config)
        contacts = dict(self.contacts, finger_forces=np.array([0., .3, .3, .3, 0.]))
        for _ in range(1001):
            self.tick(monitor, contacts)
        self.assertFalse(monitor.success)
        self.assertEqual(monitor.stable_seconds, 0.)

    def test_support_contact_immediately_fails(self):
        monitor = GraspMonitor(self.config)
        stable, valid = self.tick(monitor, dict(self.contacts, forbidden_support=True))
        self.assertFalse(stable or valid)
        self.assertEqual(monitor.failure_reason, 'forbidden_support')

    def test_penetration_blocks_progress_then_fails(self):
        monitor = GraspMonitor(self.config)
        contacts = dict(self.contacts, max_penetration=.0011)
        for _ in range(49):
            stable, valid = self.tick(monitor, contacts)
        self.assertFalse(stable or valid or monitor.done)
        self.tick(monitor, contacts)
        self.assertEqual(monitor.failure_reason, 'penetration')
        self.assertEqual(monitor.stable_seconds, 0.)

    def test_workspace_failure_and_timeout_are_distinct(self):
        monitor = GraspMonitor(self.config)
        for _ in range(50):
            self.tick(monitor, displacement=.081)
        self.assertEqual(monitor.failure_reason, 'workspace')
        monitor = GraspMonitor(replace(self.config, episode_seconds=2.))
        for _ in range(1000):
            self.tick(monitor, speed=.04)
        self.assertEqual(monitor.failure_reason, 'timeout')

    def test_config_validates_discretization_and_actuation(self):
        self.assertEqual(CubeGraspConfig.from_dict(self.config.to_dict()), self.config)
        with self.assertRaises(ValueError):
            replace(self.config, ctrl_dt=.051)
        with self.assertRaises(ValueError):
            replace(self.config, action_delta_fraction=.06)


class EnvironmentContractTests(unittest.TestCase):
    """Synthetic test state exercises I/O and gating, not grasp preparation."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.reference = Path(self.directory.name) / 'reference.npz'
        self.config = CubeGraspConfig()
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

    def test_preparation_failure_and_changed_physics_are_rejected(self):
        self.report['approved'] = False
        self.report_path.write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, 'not approved'):
            CubeGraspEnv(self.reference)
        self.report['approved'] = True
        self.report_path.write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            CubeGraspEnv(self.reference, replace(self.config, cube_size=.05))

    def test_seeded_reset_action_limit_and_previous_action_observation(self):
        env = CubeGraspEnv(self.reference)
        try:
            initial, info = env.reset(seed=7)
            qpos = env.data.qpos.copy()
            obs, repeat_info = env.reset(seed=7)
            np.testing.assert_array_equal(obs, initial)
            np.testing.assert_array_equal(env.data.qpos, qpos)
            self.assertEqual(info['mass_factor'], repeat_info['mass_factor'])
            before = env.data.ctrl.copy()
            action = np.array([2., -2., .5, -.5, 0., .25, -.25])
            clipped = np.clip(action, -1, 1)
            obs, reward, terminated, truncated, info = env.step(action)
            expected = np.clip(before + self.config.action_delta_fraction * env._range * clipped,
                               env.ids['ctrl_lower'], env.ids['ctrl_upper'])
            np.testing.assert_allclose(env.data.ctrl, expected)
            np.testing.assert_array_equal(obs[39:46], clipped.astype(np.float32))
            self.assertEqual(obs.shape, env.observation_space.shape)
            self.assertTrue(np.all(np.isfinite(obs)))
            self.assertTrue(np.isfinite(reward))
            self.assertAlmostEqual(info['elapsed_seconds'], self.config.ctrl_dt)
        finally:
            env.close()

    def test_close_baseline_uses_prepared_target(self):
        env = CubeGraspEnv(self.reference)
        try:
            env.reset(seed=0)
            np.testing.assert_allclose(env.baseline_action('close'), -1., atol=1e-6)
            np.testing.assert_array_equal(env.baseline_action('zero'), np.zeros(7))
            self.assertTrue(np.all(np.abs(env.baseline_action('random')) <= 1))
        finally:
            env.close()

    def test_invalid_randomized_resets_are_rejected_and_bounded(self):
        env = CubeGraspEnv(self.reference)
        invalid = dict(finger_forces=np.zeros(5), max_penetration=.0011,
                       forbidden_support=False)
        valid = dict(finger_forces=np.zeros(5), max_penetration=.0001,
                     forbidden_support=False)
        try:
            with patch('experiments.cube_grasp.env.native_contacts',
                       side_effect=[invalid, valid, valid]):
                _, info = env.reset(seed=7)
            self.assertEqual(info['reset_attempts'], 2)
            self.assertEqual(info['reset_rejected'], 1)
            with patch('experiments.cube_grasp.env.native_contacts', return_value=invalid) as fn:
                with self.assertRaisesRegex(RuntimeError, 'No valid randomized reset'):
                    env.reset(seed=7)
            self.assertEqual(fn.call_count, self.config.reset_max_attempts)
            with self.assertRaisesRegex(RuntimeError, 'reset before step'):
                env.step(np.zeros(7))
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
