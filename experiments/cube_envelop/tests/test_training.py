"""Checkpoint/normalizer round trips and honest independent evaluation counts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.cube_envelop.evaluate import summarize_episodes


class EvaluationAccountingTests(unittest.TestCase):
    def test_incomplete_evaluation_is_not_reported_as_complete(self):
        records = [{"seed": 20000, "success": True, "failure_reason": "",
                    "final_angle_error_degrees": 5., "best_angle_error_degrees": 4.,
                    "signed_progress_degrees": 86., "dropped": False,
                    "capture_succeeded": True, "recapture_succeeded": True,
                    "contact_seconds": [2., 2., 2., 2., 2.], "failure_phase": "",
                    "duration_seconds": 2., "max_stable_seconds": 2.,
                    "max_displacement_m": .001, "mean_action_change_squared": .02,
                    "peak_penetration_m": .0002}]
        report = summarize_episodes(records, 100, 20000)
        self.assertEqual(report["episodes_completed"], 1)
        self.assertEqual(report["success_rate"], 1.)
        self.assertFalse(report["complete"])
        self.assertEqual(report["seeds"], [20000])
        empty = summarize_episodes([], 100, 20000)
        self.assertIsNone(empty["success_rate"])
        self.assertFalse(empty["complete"])

    def test_only_unsuccessful_episodes_count_as_failures(self):
        base = {"duration_seconds": 2., "max_stable_seconds": 1.,
                "final_angle_error_degrees": 5., "best_angle_error_degrees": 4.,
                "signed_progress_degrees": 86., "dropped": False,
                "capture_succeeded": True, "recapture_succeeded": False,
                "contact_seconds": [1., 1., 1., 1., 1.], "failure_phase": "rotate",
                "max_displacement_m": .001, "mean_action_change_squared": .02,
                "peak_penetration_m": .0002}
        records = [{**base, "seed": 20000, "success": True, "failure_reason": ""},
                   {**base, "seed": 20001, "success": False, "failure_reason": "floor_contact"},
                   {**base, "seed": 20002, "success": False, "failure_reason": ""}]
        report = summarize_episodes(records, 3, 20000)
        self.assertAlmostEqual(report["success_rate"], 1/3)
        self.assertEqual(report["failure_counts"], {"floor_contact": 1, "unsuccessful_timeout": 1})

    def test_video_accounts_for_early_terminal_physics_steps(self):
        from types import SimpleNamespace
        from experiments.cube_envelop.evaluate import record_video

        class EarlyFailureEnv:
            def __init__(self):
                self.requested_axes = []

            def reset(self, **kwargs):
                self.requested_axes.append(kwargs["options"]["axis"])
                return np.zeros(3), {}

            def step(self, action):
                return np.zeros(3), 0., True, False, {
                    "elapsed_seconds": .01, "success": False,
                    "failure_reason": "dropped", "stable_seconds": 0.}

            def render(self):
                return np.zeros((100, 640, 3), dtype=np.uint8)

            def close(self):
                pass

        class Writer:
            def __init__(self):
                self.frames = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def append_data(self, frame):
                self.frames.append(frame)

        writer = Writer()
        environment = EarlyFailureEnv()
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("experiments.cube_envelop.evaluate._environment", return_value=environment), \
                mock.patch("experiments.cube_envelop.evaluate.policy_action", return_value=np.zeros(7)), \
                mock.patch("imageio.v2.get_writer", return_value=writer):
            result = record_video(None, None, "unused", SimpleNamespace(ctrl_dt=.05),
                                  Path(directory) / "video.mp4", seconds=.2, axis="x")
            self.assertEqual(result["frames"], 4)
            self.assertEqual(len(writer.frames), 4)
            self.assertAlmostEqual(result["simulation_seconds"], .2)
            self.assertEqual(result["episodes_shown"], 20)
            self.assertEqual(len(result["resets"]), 20)
            result_all = record_video(None, None, "unused", SimpleNamespace(ctrl_dt=.05),
                                      Path(directory) / "all.mp4", seconds=.3, axis="all")
            self.assertEqual(result_all["seconds_by_axis"], {"x": .1, "y": .1, "z": .1})
            self.assertEqual(set(environment.requested_axes), {"x", "y", "z"})
            self.assertEqual({row["axis"] for row in result_all["resets"]}, {"x", "y", "z"})

    def test_each_axis_uses_disjoint_seeds_and_reports_its_own_success(self):
        import gymnasium as gym
        from types import SimpleNamespace
        from experiments.cube_envelop.evaluate import evaluate_policy

        class AxisEnv:
            action_space = gym.spaces.Box(-1., 1., shape=(7,), dtype=np.float32)

            def __init__(self, axis, mode="reorient"):
                self.axis = axis
                self.mode = mode

            def reset(self, *, seed, options):
                self.assert_axis = options["axis"] == self.axis
                self.seed = seed
                return np.zeros(78), {"axis": self.axis, "angle_error_degrees": 90.}

            def step(self, action):
                assert self.assert_axis
                success = self.axis != "z" or self.seed % 5 != 0
                return np.zeros(78), 0., True, False, {
                    "axis": self.axis, "angle_error_degrees": 5. if success else 90.,
                    "signed_progress_degrees": 85. if success else 0.,
                    "success": success, "elapsed_seconds": .05,
                    "failure_reason": "" if success else "timeout", "dropped": False,
                    "capture_succeeded": True, "recapture_succeeded": success and self.mode == "reorient",
                    "phase": "recapture" if self.mode == "reorient" else "acquire",
                    "failure_phase": "" if success else "rotate",
                    "contact_seconds": [.01, .02, .03, .04, .05]}

            def close(self):
                pass

        with mock.patch("experiments.cube_envelop.evaluate._environment",
                        side_effect=lambda *a, axis=None, **k: AxisEnv(axis, getattr(a[1], "task_mode", "reorient"))):
            report = evaluate_policy(None, None, "unused", SimpleNamespace(ctrl_dt=.05, episode_seconds=.05),
                                     episodes=5, baseline="zero", axis="all", seed_start=20000)
            hold = evaluate_policy(None, None, "unused", SimpleNamespace(ctrl_dt=.05, episode_seconds=.05, task_mode="hold"),
                                   episodes=5, baseline="zero", axis="all", seed_start=20000)
        self.assertEqual(report["episodes_completed"], 15)
        self.assertEqual(report["axis_success_rates"], {"x": 1., "y": 1., "z": .8})
        self.assertGreater(report["success_rate"], .9)
        self.assertFalse(report["all_axes_at_least_90_percent"])
        self.assertEqual(report["min_axis_success_rate"], .8)
        seeds = [set(report["axes"][axis]["seeds"]) for axis in ("x", "y", "z")]
        self.assertFalse(seeds[0] & seeds[1] | seeds[0] & seeds[2] | seeds[1] & seeds[2])
        self.assertEqual(report["capture_rate"], 1.)
        self.assertAlmostEqual(report["recapture_rate"], 14/15)
        self.assertAlmostEqual(report["contact_duty_cycle"]["thumb"], .2)
        self.assertAlmostEqual(report["contact_duty_cycle"]["pinky"], 1.)
        self.assertEqual(report["phase_failure_counts"], {"rotate:timeout": 1})
        self.assertEqual(hold["episodes_completed"], 5)
        self.assertEqual(hold["axes"], {})
        self.assertEqual(hold["group_success_rates"], {"hold": 1.})
        self.assertEqual(hold["capture_rate"], 1.)
        self.assertIsNone(hold["recapture_rate"])

    def test_promotion_requires_two_checks_for_every_axis(self):
        from experiments.cube_envelop.train import AxisPromotionMonitor
        monitor = AxisPromotionMonitor()
        def report(rates, complete=True):
            return {"axes": {axis: {"complete": complete, "success_rate": rate}
                             for axis, rate in zip(("x", "y", "z"), rates)}}
        self.assertFalse(monitor.update(report([1., 1., .8])))
        self.assertFalse(monitor.update(report([1., 1., 1.])))
        self.assertTrue(monitor.update(report([1., 1., 1.])))
        self.assertFalse(monitor.update(report([1., 1., 1.], complete=False)))
        self.assertEqual(monitor.streaks, {"x": 0, "y": 0, "z": 0})
        hold_monitor = AxisPromotionMonitor(("hold",))
        hold_report = {"groups": {"hold": {"complete": True, "success_rate": .9}}}
        self.assertFalse(hold_monitor.update(hold_report))
        self.assertTrue(hold_monitor.update(hold_report))


class CheckpointTests(unittest.TestCase):
    def test_hold_initialization_transfers_weights_and_stats_but_resets_optimizer(self):
        from dataclasses import replace
        import gymnasium as gym
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        import torch
        from experiments.cube_envelop.config import CubeReorientConfig
        from experiments.cube_envelop.evaluate import policy_action
        from experiments.cube_envelop.train import (initialize_from_checkpoint,
            new_policy, save_checkpoint, validate_initialization_checkpoint)

        torch.set_num_threads(1)

        class ArrayEnv(gym.Env):
            observation_space = gym.spaces.Box(-100., 100., shape=(78,), dtype=np.float32)
            action_space = gym.spaces.Box(-1., 1., shape=(7,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.t = 0
                return np.linspace(1., 3., 78, dtype=np.float32), {}

            def step(self, action):
                self.t += 1
                return np.linspace(1., 3., 78, dtype=np.float32) + self.t / 10., float(action[0]), False, self.t >= 4, {}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.npz"
            np.savez(reference, qpos=np.zeros(23), ctrl=np.zeros(7))
            (root / "prepare_report.json").write_text(json.dumps({"approved": True}))
            source_norm = VecNormalize(DummyVecEnv([ArrayEnv]), norm_obs=True, norm_reward=False)
            source = new_policy(source_norm, 123)
            source.learn(128)
            self.assertTrue(source.policy.optimizer.state)
            hold_config = CubeReorientConfig(task_mode="hold")
            target_config = replace(hold_config, task_mode="reorient")
            checkpoint = save_checkpoint(source, source_norm, reference, hold_config, root / "run",
                                         elapsed_seconds=1., reason="unit_test")
            metadata = validate_initialization_checkpoint(checkpoint, reference, target_config)
            self.assertEqual(metadata["config"]["task_mode"], "hold")
            target, target_norm, provenance = initialize_from_checkpoint(
                checkpoint, reference, target_config, DummyVecEnv([ArrayEnv]), 456)
            observation = np.linspace(2., 4., 78, dtype=np.float32)
            np.testing.assert_array_equal(policy_action(target, target_norm, observation),
                                          policy_action(source, source_norm, observation))
            np.testing.assert_array_equal(target_norm.obs_rms.mean, source_norm.obs_rms.mean)
            np.testing.assert_array_equal(target_norm.obs_rms.var, source_norm.obs_rms.var)
            self.assertEqual(target_norm.obs_rms.count, source_norm.obs_rms.count)
            self.assertEqual(target.num_timesteps, 0)
            self.assertEqual(target._n_updates, 0)
            self.assertFalse(target.policy.optimizer.state)
            self.assertEqual(provenance["source_training_steps"], 128)
            self.assertEqual(provenance["source_task_mode"], "hold")
            self.assertEqual(provenance["target_task_mode"], "reorient")
            changed = root / "changed_reference.npz"
            np.savez(changed, qpos=np.ones(23), ctrl=np.zeros(7))
            with self.assertRaisesRegex(ValueError, "reference SHA256 differs"):
                validate_initialization_checkpoint(checkpoint, changed, target_config)
            with self.assertRaisesRegex(ValueError, "configs must match"):
                validate_initialization_checkpoint(checkpoint, reference, replace(target_config, cube_mass=.1))
            with self.assertRaisesRegex(ValueError, "only supported from hold to reorient"):
                validate_initialization_checkpoint(checkpoint, reference, hold_config)
            legacy = root / "legacy75"
            legacy.mkdir()
            (legacy / "metadata.json").write_text(json.dumps({**metadata, "observation_size": 75}))
            with self.assertRaisesRegex(ValueError, "75-observation"):
                validate_initialization_checkpoint(legacy, reference, target_config)
            target_norm.close()
            source_norm.close()

    def test_policy_and_running_observation_normalizer_round_trip(self):
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        import torch
        from experiments.cube_envelop.config import CubeReorientConfig
        from experiments.cube_envelop.evaluate import load_checkpoint, policy_action
        from experiments.cube_envelop.train import save_checkpoint

        torch.set_num_threads(1)

        class TinyEnv(gym.Env):
            observation_space = gym.spaces.Box(-100., 100., shape=(78,), dtype=np.float32)
            action_space = gym.spaces.Box(-1., 1., shape=(7,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.t = 0
                return np.r_[np.array([4., -2., .7]), np.zeros(75)].astype(np.float32), {}

            def step(self, action):
                self.t += 1
                return np.r_[np.array([4. + self.t, -2., .7]), np.zeros(75)].astype(np.float32), float(action[0]), False, self.t >= 4, {}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.npz"
            np.savez(reference, qpos=np.zeros(23), ctrl=np.zeros(7))
            (root / "prepare_report.json").write_text(json.dumps({"approved": True}))
            normalizer = VecNormalize(DummyVecEnv([TinyEnv]), norm_obs=True, norm_reward=False)
            model = PPO("MlpPolicy", normalizer, n_steps=8, batch_size=8, n_epochs=1,
                        policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}}, seed=123, device="cpu")
            model.learn(16)
            observation = np.r_[np.array([8., -2., .7]), np.zeros(75)].astype(np.float32)
            expected = policy_action(model, normalizer, observation)
            config = CubeReorientConfig()
            bundle = save_checkpoint(model, normalizer, reference, config, root / "run",
                                     elapsed_seconds=1., reason="unit_test")
            with mock.patch("experiments.cube_envelop.evaluate._environment", side_effect=lambda *a, **k: TinyEnv()):
                restored, restored_norm, _, _, metadata = load_checkpoint(bundle)
                actual = policy_action(restored, restored_norm, observation)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(restored_norm.obs_rms.mean, normalizer.obs_rms.mean)
                np.testing.assert_array_equal(restored_norm.obs_rms.var, normalizer.obs_rms.var)
                self.assertEqual(metadata["training_steps"], 16)
                self.assertFalse(restored_norm.training)
                restored_norm.close()
                legacy = root / "legacy"
                legacy.mkdir()
                (legacy / "metadata.json").write_text(json.dumps({
                    "format_version": 1, "task": "horizontal_cube_reorient_xyz90",
                    "observation_size": 75}))
                with self.assertRaisesRegex(ValueError, "78-observation"):
                    load_checkpoint(legacy)
                altered = root / "different_reference.npz"
                np.savez(altered, qpos=np.ones(23), ctrl=np.zeros(7))
                with self.assertRaisesRegex(ValueError, "reference differs"):
                    load_checkpoint(bundle, altered)
                (bundle / "prepare_report.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    load_checkpoint(bundle)
            normalizer.close()


if __name__ == "__main__":
    unittest.main()
