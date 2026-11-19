"""Checkpoint/normalizer round trips and honest independent evaluation counts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.cube_grasp.evaluate import summarize_episodes


class EvaluationAccountingTests(unittest.TestCase):
    def test_incomplete_evaluation_is_not_reported_as_complete(self):
        records = [{"seed": 20000, "success": True, "failure_reason": "",
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
        from experiments.cube_grasp.evaluate import record_video

        class EarlyFailureEnv:
            def reset(self, **kwargs):
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
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("experiments.cube_grasp.evaluate._environment", return_value=EarlyFailureEnv()), \
                mock.patch("experiments.cube_grasp.evaluate.policy_action", return_value=np.zeros(7)), \
                mock.patch("imageio.v2.get_writer", return_value=writer):
            result = record_video(None, None, "unused", SimpleNamespace(ctrl_dt=.05),
                                  Path(directory) / "video.mp4", seconds=.2)
            self.assertEqual(result["frames"], 4)
            self.assertEqual(len(writer.frames), 4)
            self.assertAlmostEqual(result["simulation_seconds"], .2)
            self.assertEqual(result["episodes_shown"], 20)
            self.assertEqual(len(result["resets"]), 20)


class CheckpointTests(unittest.TestCase):
    def test_policy_and_running_observation_normalizer_round_trip(self):
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        import torch
        from experiments.cube_grasp.config import CubeGraspConfig
        from experiments.cube_grasp.evaluate import load_checkpoint, policy_action
        from experiments.cube_grasp.train import save_checkpoint

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
            config = CubeGraspConfig()
            bundle = save_checkpoint(model, normalizer, reference, config, root / "run",
                                     elapsed_seconds=1., reason="unit_test")
            with mock.patch("experiments.cube_grasp.evaluate._environment", side_effect=lambda *a, **k: TinyEnv()):
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
