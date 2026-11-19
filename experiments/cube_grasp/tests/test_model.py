import hashlib
import json
import unittest

import mujoco
import numpy as np

from experiments.cube_grasp.model import CUBE_ASSETS, build_model, model_ids, native_contacts


class ModelTests(unittest.TestCase):
    def test_pinned_cube_sources(self):
        manifest=json.loads((CUBE_ASSETS/'source_manifest.json').read_text())
        for record in manifest['files']:
            local=CUBE_ASSETS/record['path'].split('/')[-1]
            if local.exists():
                self.assertEqual(hashlib.sha256(local.read_bytes()).hexdigest(),record['sha256'])

    def test_rigid_cube_mass_inertia_and_size(self):
        model=build_model();ids=model_ids(model);body=ids['cube_body'];geom=ids['cube_geom']
        self.assertEqual(model.nq,23);self.assertEqual(model.nv,22);self.assertEqual(model.nu,7)
        np.testing.assert_allclose(model.geom_size[geom],[.0285]*3)
        self.assertAlmostEqual(float(model.body_mass[body]),.077966253)
        np.testing.assert_allclose(model.body_inertia[body],[4.22187259995e-5]*3)
        np.testing.assert_allclose(model.opt.gravity,[0,0,-9.81])
        self.assertEqual(model.geom_condim[geom],3)
        self.assertEqual(sum(model.geom_bodyid==body),1)

    def test_floor_contact_is_detected(self):
        model=build_model();data=mujoco.MjData(model);ids=model_ids(model)
        data.qpos[-7:]=[.5,0,.028,1,0,0,0]
        mujoco.mj_forward(model,data)
        contact=native_contacts(model,data,ids)
        self.assertTrue(contact['forbidden_support'])
        self.assertGreater(contact['max_penetration'],.0004)


if __name__=='__main__':unittest.main()
