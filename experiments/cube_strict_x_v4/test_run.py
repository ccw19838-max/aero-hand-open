"""Promotion, plateau detection and matched-step comparisons."""
import json
import tempfile
import unittest
from pathlib import Path

from .run import Experiment, plateau, qualifies


def result(a, b):
    return dict(complete=True, groups={
        dt: dict(complete=True, requested_trials=20, trials=20, successes=count)
        for dt, count in zip(("0.001", "0.002"), (a, b))})


class ScheduleTests(unittest.TestCase):
    def test_both_timesteps_are_required(self):
        self.assertTrue(qualifies(result(16, 16)))
        self.assertFalse(qualifies(result(20, 15)))
        partial = result(20, 20)
        partial["groups"]["0.002"]["complete"] = False
        self.assertFalse(qualifies(partial))
        del partial["groups"]["0.002"]
        self.assertFalse(qualifies(partial))

    def test_four_scores_without_improvement(self):
        rows = [dict(rate=0., median=x) for x in (2., 2.1, 2.8, 2.4)]
        self.assertFalse(plateau(rows[:3]))
        self.assertTrue(plateau(rows))
        rows[-1]["median"] = 3.
        self.assertFalse(plateau(rows))
        rows[-1].update(median=2., rate=.05)
        self.assertFalse(plateau(rows))

    def test_paired_comparison_uses_only_shared_checkpoint_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Experiment.__new__(Experiment)
            experiment.root = Path(directory)
            experiment.job = {"validation_seed": 7200000}
            row = dict(report="eval.json", groups={"0.001": dict(
                successes=0, trials=20, median_valid_peak_degrees=2.1, maximum_eligible_peak_degrees=3.4,
                maximum_signed_angle=8., maximum_stable_seconds=.1)})
            experiment.state = {"milestones": {
                "v3": dict.fromkeys(("0", "8192", "4096"), row),
                "v4": dict.fromkeys(("0", "4096", "16384"), row)}}
            experiment.write_comparison()
            pairs = json.loads((experiment.root/"comparison.json").read_text())["pairs"]
            self.assertEqual([row["steps"] for row in pairs], [0, 4096])
            table = (experiment.root/"comparison.md").read_text()
            self.assertIn("4,096", table)
            self.assertNotIn("16,384", table)
