import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import mujoco
import numpy as np

from experiments.pen_spin.config import PenSpinConfig
from experiments.pen_spin.model import build_model, model_ids, thumb_frame
from experiments.pen_spin.prepare import (
    check_freeflight, check_model, check_reference_timesteps, reference_geometry,
    run_prepare, search_grasp, stable_sample, candidate_initial_state, _plane, _quat_z_to,
)


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = PenSpinConfig()
        cls.model = build_model(cls.config)

    def test_numerical_model_and_freeflight_gates(self):
        self.assertTrue(check_model(self.model)["passed"])
        self.assertTrue(check_freeflight(self.config)["passed"])

    def test_ungrasped_pen_never_counts_as_stable_grasp(self):
        d = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, d, 0)
        mujoco.mj_forward(self.model, d)
        valid, _ = stable_sample(self.model, d, model_ids(self.model), self.config)
        self.assertFalse(valid)
        c0, n0 = thumb_frame(self.model, d)
        ref = dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=d.ctrl.copy(), c0=c0, n0=n0)
        self.assertFalse(check_reference_timesteps(ref, self.config)["passed"])

    def test_reference_must_fit_frozen_axis_task_envelope(self):
        m = self.model; ids = model_ids(m); d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
        c0, n0 = thumb_frame(m, d, ids)
        reference = dict(c0=c0, n0=n0)
        e1, e2 = _plane(n0); q = ids["pen_qpos_adr"]
        def geometry(center, direction):
            d.qpos[q:q+3] = center
            d.qpos[q+3:q+7] = _quat_z_to(direction)
            mujoco.mj_forward(m, d)
            return reference_geometry(m, d, ids, reference, self.config)
        self.assertTrue(geometry(c0 + .025 * e1, e2)["passed"])
        axial = geometry(c0 + .025 * e1 + .03 * n0, e2)
        self.assertFalse(axial["checks"]["axial_distance"])
        singular = geometry(c0 + .025 * e1, n0)
        self.assertFalse(singular["checks"]["plane"])
        self.assertFalse(singular["checks"]["direction_projection"])
        endpoint_at_pivot = geometry(c0 - .065 * e2, e2)
        self.assertFalse(endpoint_at_pivot["checks"]["endpoint_radius"])
        far = geometry(c0 + .09 * e1, e2)
        self.assertFalse(far["checks"]["center_distance"])
        geometry(c0 + .025 * e1, e2)
        rotated_reference = reference_geometry(m, d, ids, dict(c0=c0, n0=e1), self.config)
        self.assertFalse(rotated_reference["checks"]["thumb_axis"])
        self.assertAlmostEqual(rotated_reference["thumb_axis_degrees"], 90.)

    def test_empty_search_cannot_approve_and_budget_bounds_hold(self):
        ref, report = search_grasp(self.model, self.config, max_candidates=0)
        self.assertIsNone(ref)
        self.assertFalse(report["passed"])
        self.assertEqual(report["examined"], 0)
        with self.assertRaises(ValueError):
            search_grasp(self.model, self.config, max_candidates=2049)
        with self.assertRaises(ValueError):
            search_grasp(self.model, self.config, budget_seconds=901)

    def test_valid_reference_outranks_longer_hold_outside_task_envelope(self):
        invalid = {"reason": "initial_task_envelope", "final_hold_seconds": 2.,
                   "best_hold_seconds": 2., "mean_linear_speed": 0.}
        valid = {"reason": "passed", "final_hold_seconds": .5,
                 "best_hold_seconds": .5, "mean_linear_speed": .01}
        with patch("experiments.pen_spin.prepare._candidate", side_effect=[
                ({"diagnostic": True}, invalid), ({"valid": True}, valid)]):
            ref, report = search_grasp(self.model, self.config, max_candidates=2)
        self.assertTrue(report["passed"])
        self.assertEqual(ref, {"valid": True})

    def test_diagnostic_reset_is_deterministic_and_has_no_external_assistance(self):
        point = np.linspace(.05, .95, 11)
        first = candidate_initial_state(self.model, point, self.config)
        second = candidate_initial_state(self.model, point, self.config)
        np.testing.assert_array_equal(first.qpos, second.qpos)
        np.testing.assert_array_equal(first.qvel, second.qvel)
        np.testing.assert_array_equal(first.ctrl, second.ctrl)
        self.assertFalse(np.any(first.xfrc_applied))
        self.assertFalse(np.any(first.qfrc_applied))

    def test_skipping_mjx_never_produces_an_approved_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            report = run_prepare(temp, max_candidates=0, check_mjx=False)
            self.assertFalse(report["passed"])
            self.assertIsNone(report["search"])
            self.assertFalse((Path(temp) / "reference.npz").exists())
            self.assertTrue((Path(temp) / "prepare_report.json").is_file())

    def test_existing_reference_is_not_silently_reused_or_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            reference = Path(temp) / "reference.npz"
            reference.write_bytes(b"preserve existing user artifact")
            with self.assertRaises(FileExistsError):
                run_prepare(temp, max_candidates=0, check_mjx=False)
            self.assertEqual(reference.read_bytes(), b"preserve existing user artifact")


if __name__ == "__main__":
    unittest.main()
