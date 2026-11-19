import unittest

import mujoco
import numpy as np

from experiments.cube_reorient.model import build_model, fixed_axis_basis, model_fingerprint
from experiments.cube_grasp.model import model_fingerprint as upright_fingerprint


class MountingTests(unittest.TestCase):
    def test_original_twenty_degree_mount_and_fixed_palm_axes(self):
        m=build_model();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        np.testing.assert_allclose(m.body('tetheria_mount').quat,[1,0,0,0])
        matrix=d.xmat[m.body('palm').id].reshape(3,3)
        basis=fixed_axis_basis(m)
        np.testing.assert_allclose(basis[:,0],matrix[:,2],atol=1e-12)
        np.testing.assert_allclose(basis[:,1],-matrix[:,0],atol=1e-12)
        np.testing.assert_allclose(basis[:,2],-matrix[:,1],atol=1e-12)
        np.testing.assert_allclose(basis.T@basis,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(basis),1.)
        self.assertAlmostEqual(np.rad2deg(np.arccos(basis[2,2])),19.9947173,places=5)
        before=basis.copy();d.qpos[:16]=.5;mujoco.mj_forward(m,d)
        np.testing.assert_allclose(fixed_axis_basis(m),before)

    def test_physics_not_gravity_was_rotated(self):
        m=build_model()
        np.testing.assert_allclose(m.opt.gravity,[0,0,-9.81])
        self.assertAlmostEqual(float(m.body('cube').mass[0]),.077966253)
        self.assertEqual(m.nu,7)
        self.assertNotEqual(model_fingerprint(),upright_fingerprint())


if __name__=='__main__':unittest.main()
