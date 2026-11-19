"""Regression for the exact native evaluation path and the actual CPU training worker."""
import json
from pathlib import Path
import unittest

import numpy as np

from experiments.cube_adaptive_60h.config import Profile
from .engine import _worker
from .evaluation import episode, make_env


SOURCE = Path(__file__).resolve().parents[1] / 'cube_two_driver_v3/artifacts/thumb_pinky_20261117'
PROFILE = Profile(**json.loads((SOURCE/'job.json').read_text())['profile'])


class Push:
    def predict(self, observation, deterministic=True):
        return np.array([.35, 0., 0., .3, -.3, 0., 0.], dtype=np.float32), None


class ReplayRewardTests(unittest.TestCase):
    def compare(self, dt, arm, forced_clearance):
        reference = SOURCE/'inputs/reference.npz'
        replay = make_env(reference.parent, PROFILE, 5., dt, arm)
        worker = _worker(arm, reference, dt, 5., PROFILE)
        self.addCleanup(replay.close)
        self.addCleanup(worker.close)
        # Inject only the distance measurement. Forces, actions and trajectory stay physical.
        if forced_clearance:
            replay.env.clearance = worker.env.clearance = lambda: .001
        controller = Push()
        row = episode(replay, controller, 692, trace=True)
        obs, _ = worker.reset(seed=692)
        total = 0.
        for index, sample in enumerate(row['trace']):
            obs, reward, done, truncated, state = worker.step(controller.predict(obs)[0])
            total += reward
            self.assertEqual(sample['raw_task_reward'], reward)
            self.assertEqual(sample['raw_task_return'], total)
            self.assertEqual(sample['mount_proximity_penalty'], state['mount_proximity_penalty'])
            np.testing.assert_array_equal(sample['finger_forces'], state['finger_forces'])
            self.assertEqual(done or truncated, index == len(row['trace'])-1)
        self.assertEqual(row['final']['raw_task_return'], total)
        if forced_clearance:
            self.assertTrue(any(s['mount_proximity_penalty'] > 0. for s in row['trace']))
            recovery = [s for s in row['trace'] if s['first_violation']]
            self.assertTrue(recovery)
            if arm == 'v4':
                self.assertTrue(all(s['mount_proximity_penalty'] == 0. for s in recovery))
            else:
                self.assertTrue(all(s['mount_proximity_penalty'] > 0. for s in recovery))

    def test_native_rewards_match_worker_before_reward_normalization(self):
        for dt in (.001, .002):
            for arm in ('v3', 'v4'):
                for forced in (False, True):
                    with self.subTest(dt=dt,arm=arm,forced_clearance=forced):
                        self.compare(dt,arm,forced)


if __name__ == '__main__':
    unittest.main()
