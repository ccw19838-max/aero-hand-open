import unittest
from dataclasses import replace
import mujoco
import numpy as np

from experiments.pen_spin.config import PenSpinConfig
from experiments.pen_spin.model import (
    PEN_COM, PEN_INERTIA, PEN_MASS, build_model, model_ids, native_contacts,
    model_fingerprint, task_fingerprint,
)


class PenModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model(PenSpinConfig())
        cls.ids = model_ids(cls.model)

    def test_physical_specification_and_collision_length(self):
        m, ids = self.model, self.ids
        self.assertEqual(m.nu, 7)
        np.testing.assert_allclose(m.body_mass[ids["pen_body"]], PEN_MASS, atol=1e-12)
        np.testing.assert_allclose(m.body_ipos[ids["pen_body"]], PEN_COM, atol=1e-12)
        np.testing.assert_allclose(m.body_inertia[ids["pen_body"]], PEN_INERTIA, atol=1e-12)
        radius, segment_half = m.geom_size[ids["pen_geom"], :2]
        self.assertAlmostEqual(2 * (radius + segment_half), .130)
        self.assertEqual(m.geom_type[ids["pen_geom"]], mujoco.mjtGeom.mjGEOM_CAPSULE)
        for name in ("pen_middle_visual", "pen_heavy_visual", "pen_light_visual"):
            self.assertEqual(m.geom(name).contype, 0)
            self.assertEqual(m.geom(name).conaffinity, 0)

    def test_vertical_grasp_frame_and_endpoint_markers(self):
        m, ids = self.model, self.ids
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        np.testing.assert_allclose(m.opt.gravity, [0, 0, -9.81])
        np.testing.assert_allclose(d.site("grasp_site").xpos, [-.03, 0, .26], atol=1e-7)
        self.assertAlmostEqual(np.linalg.norm(d.site_xpos[ids["site_a"]] -
                                             d.site_xpos[ids["site_b"]]), .13)

    def test_thumb_base_is_not_effective_thumb_contact(self):
        m, ids = self.model, self.ids
        effective = invalid = 0
        for g in range(m.ngeom):
            name = m.body(int(m.geom_bodyid[g])).name
            if name in ("right_t_link", "right_thumb_mcp_link"):
                self.assertEqual(ids["geom_finger"][g], -1)
                self.assertTrue(ids["geom_invalid"][g])
                invalid += 1
            if name in ("right_thumb_proximal_link", "right_thumb_distal_link"):
                self.assertEqual(ids["geom_finger"][g], 0)
                effective += 1
        self.assertGreater(invalid, 0)
        self.assertGreater(effective, 0)

    def test_floor_support_is_detected(self):
        m, ids = self.model, self.ids
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        q = ids["pen_qpos_adr"]
        d.qpos[q:q+3] = [.6, .6, .064]  # 1mm capsule penetration of floor.
        mujoco.mj_forward(m, d)
        contacts = native_contacts(m, d, ids)
        self.assertTrue(contacts["invalid_support"])
        self.assertGreater(contacts["max_penetration"], 0)
        self.assertFalse(contacts["finger_forces"].any())

    def test_actual_thumb_collision_mask_and_positive_force(self):
        m, ids = self.model, self.ids
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
        thumb = m.geom("th_tip").id
        rotation = d.geom_xmat[thumb].reshape(3, 3)
        pen = ids["pen_geom"]; q = ids["pen_qpos_adr"]
        self.assertTrue((m.geom_contype[pen] & m.geom_conaffinity[thumb])
                        or (m.geom_contype[thumb] & m.geom_conaffinity[pen]))
        d.qpos[q:q+3] = d.geom_xpos[thumb] + rotation[:, 0] * .012
        from experiments.pen_spin.prepare import _quat_z_to
        d.qpos[q+3:q+7] = _quat_z_to(rotation[:, 2])
        mujoco.mj_forward(m, d)
        self.assertGreater(native_contacts(m, d, ids)["finger_forces"][0], 0)
        pen_contacts = [c for c in d.contact if pen in (c.geom1, c.geom2)]
        self.assertTrue(pen_contacts)
        for contact in pen_contacts:
            self.assertEqual(contact.dim, 3)
            self.assertAlmostEqual(contact.friction[0], .6)

    def test_preparation_provenance_ignores_training_budget_only(self):
        cfg = PenSpinConfig()
        runtime = replace(cfg, budget_hours=1, learning_rate=1e-4)
        physics = replace(cfg, max_axial_distance=.03)
        self.assertEqual(task_fingerprint(cfg), task_fingerprint(runtime))
        self.assertNotEqual(task_fingerprint(cfg), task_fingerprint(physics))
        self.assertEqual(model_fingerprint(cfg), model_fingerprint(runtime))
        self.assertNotEqual(model_fingerprint(cfg), model_fingerprint(replace(cfg, sim_dt=.001)))


if __name__ == "__main__":
    unittest.main()
