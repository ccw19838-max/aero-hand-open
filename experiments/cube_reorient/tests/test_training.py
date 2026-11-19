"""Checkpoint/normalizer round trips and honest independent evaluation counts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.cube_reorient.evaluate import summarize_episodes


class EvaluationAccountingTests(unittest.TestCase):
    def test_incomplete_evaluation_is_not_reported_as_complete(self):
        records = [{"seed": 20000, "success": True, "failure_reason": "",
                    "final_angle_error_degrees": 5., "best_angle_error_degrees": 4.,
                    "signed_progress_degrees": 86., "dropped": False,
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
        from experiments.cube_reorient.evaluate import record_video

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
                mock.patch("experiments.cube_reorient.evaluate._environment", return_value=environment), \
                mock.patch("experiments.cube_reorient.evaluate.policy_action", return_value=np.zeros(7)), \
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
        from experiments.cube_reorient.evaluate import evaluate_policy

        class AxisEnv:
            action_space = gym.spaces.Box(-1., 1., shape=(7,), dtype=np.float32)

            def __init__(self, axis):
                self.axis = axis

            def reset(self, *, seed, options):
                self.assert_axis = options["axis"] == self.axis
                self.seed = seed
                return np.zeros(75), {"axis": self.axis, "angle_error_degrees": 90.}

            def step(self, action):
                assert self.assert_axis
                success = self.axis != "z" or self.seed % 5 != 0
                return np.zeros(75), 0., True, False, {
                    "axis": self.axis, "angle_error_degrees": 5. if success else 90.,
                    "signed_progress_degrees": 85. if success else 0.,
                    "success": success, "elapsed_seconds": .05,
                    "failure_reason": "" if success else "timeout", "dropped": False}

            def close(self):
                pass

        with mock.patch("experiments.cube_reorient.evaluate._environment",
                        side_effect=lambda *a, axis=None, **k: AxisEnv(axis)):
            report = evaluate_policy(None, None, "unused", SimpleNamespace(ctrl_dt=.05, episode_seconds=.05),
                                     episodes=5, baseline="zero", axis="all", seed_start=20000)
        self.assertEqual(report["episodes_completed"], 15)
        self.assertEqual(report["axis_success_rates"], {"x": 1., "y": 1., "z": .8})
        self.assertGreater(report["success_rate"], .9)
        self.assertFalse(report["all_axes_at_least_90_percent"])
        self.assertEqual(report["min_axis_success_rate"], .8)
        seeds = [set(report["axes"][axis]["seeds"]) for axis in ("x", "y", "z")]
        self.assertFalse(seeds[0] & seeds[1] | seeds[0] & seeds[2] | seeds[1] & seeds[2])

    def test_promotion_requires_two_checks_for_every_axis(self):
        from experiments.cube_reorient.train import AxisPromotionMonitor
        monitor = AxisPromotionMonitor()
        def report(rates, complete=True):
            return {"axes": {axis: {"complete": complete, "success_rate": rate}
                             for axis, rate in zip(("x", "y", "z"), rates)}}
        self.assertFalse(monitor.update(report([1., 1., .8])))
        self.assertFalse(monitor.update(report([1., 1., 1.])))
        self.assertTrue(monitor.update(report([1., 1., 1.])))
        self.assertFalse(monitor.update(report([1., 1., 1.], complete=False)))
        self.assertEqual(monitor.streaks, {"x": 0, "y": 0, "z": 0})


class CheckpointTests(unittest.TestCase):
    def test_policy_and_running_observation_normalizer_round_trip(self):
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        import torch
        from experiments.cube_reorient.config import CubeReorientConfig
        from experiments.cube_reorient.evaluate import load_checkpoint, policy_action
        from experiments.cube_reorient.train import save_checkpoint

        torch.set_num_threads(1)

        class TinyEnv(gym.Env):
            observation_space = gym.spaces.Box(-100., 100., shape=(3,), dtype=np.float32)
            action_space = gym.spaces.Box(-1., 1., shape=(7,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.t = 0
                return np.array([4., -2., .7], np.float32), {}

            def step(self, action):
                self.t += 1
                return np.array([4. + self.t, -2., .7], np.float32), float(action[0]), False, self.t >= 4, {}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.npz"
            np.savez(reference, qpos=np.zeros(23), ctrl=np.zeros(7))
            (root / "prepare_report.json").write_text(json.dumps({"approved": True}))
            normalizer = VecNormalize(DummyVecEnv([TinyEnv]), norm_obs=True, norm_reward=False)
            model = PPO("MlpPolicy", normalizer, n_steps=8, batch_size=8, n_epochs=1,
                        policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}}, seed=123, device="cpu")
            model.learn(16)
            observation = np.array([8., -2., .7], np.float32)
            expected = policy_action(model, normalizer, observation)
            config = CubeReorientConfig()
            bundle = save_checkpoint(model, normalizer, reference, config, root / "run",
                                     elapsed_seconds=1., reason="unit_test")
            with mock.patch("experiments.cube_reorient.evaluate._environment", side_effect=lambda *a, **k: TinyEnv()):
                restored, restored_norm, _, _, metadata = load_checkpoint(bundle)
                actual = policy_action(restored, restored_norm, observation)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(restored_norm.obs_rms.mean, normalizer.obs_rms.mean)
                np.testing.assert_array_equal(restored_norm.obs_rms.var, normalizer.obs_rms.var)
                self.assertEqual(metadata["training_steps"], 16)
                self.assertFalse(restored_norm.training)
                restored_norm.close()
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
