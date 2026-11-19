from pathlib import Path
import unittest
import numpy as np
from experiments.cube_adaptive_60h.env import AdaptiveEnv
from experiments.cube_adaptive_60h.reward_registry import wrap
from experiments.cube_adaptive_60h.thumb_drive import (TRANSFORM, ThumbDriveReward, contact_wrenches,
                                                      drive_quality, credit_progress)

REFERENCE=Path(__file__).resolve().parents[3]/'experiments/cube_envelop/artifacts/prepared_envelope_v1/reference.npz'


class ThumbDrive(unittest.TestCase):
    def test_squeezing_wrong_direction_overforce_and_throw_have_no_bonus(self):
        self.assertEqual(drive_quality(.01,.01,2.,True,True),1.)
        for args in ((.01,0.,2.,True,True),(-.01,.01,2.,True,True),(.01,-.01,2.,True,True),
                     (.01,.01,21.,True,True),(.01,.01,2.,False,True),(.01,.01,2.,True,False)):
            self.assertEqual(drive_quality(*args),0.)

    def test_progress_reward_cannot_be_reearned_by_backtracking_or_wrong_pose(self):
        high=0.;total=0.;target=np.pi/2
        for degrees in (0,20,40,20,40,30,50,10,90,10,90):
            p=np.deg2rad(degrees);increment,high=credit_progress(p,target-p,target,high)
            total+=12.*increment/target
        self.assertAlmostEqual(total,12.)
        self.assertEqual(credit_progress(target,target,target,0.),(0.,0.))

    def test_world_wrench_sign_matches_mujoco_generalized_cube_contact_force(self):
        env=AdaptiveEnv(REFERENCE,'x',physical_dt=.001,target_degrees=90.)
        try:
            env.reset(seed=0,options={'reference':True})
            for _ in range(5):env.step(np.zeros(7))
            # Include any palm contacts solely for this independent sum check.
            env.ids['geom_finger'][env.ids['geom_finger']<0]=4
            measured=contact_wrenches(env);adr=env.ids['cube_dof_adr']
            np.testing.assert_allclose(measured['forces_world_N'].sum(axis=0),env.data.qfrc_constraint[adr:adr+3],atol=1e-9)
            rotation=env.data.xmat[env.ids['cube_body']].reshape(3,3)
            np.testing.assert_allclose(measured['moments_world_Nm'].sum(axis=0),rotation@env.data.qfrc_constraint[adr+3:adr+6],atol=1e-9)
            self.assertGreater(np.linalg.norm(measured['forces_world_N']),.1)
        finally:env.close()

    def test_both_integrators_keep_original_physics_observations_and_termination(self):
        for dt in (.001,.002):
            for target in (0.,90.):
                plain=AdaptiveEnv(REFERENCE,'x',physical_dt=dt,target_degrees=target)
                wrapped=ThumbDriveReward(AdaptiveEnv(REFERENCE,'x',physical_dt=dt,target_degrees=target))
                try:
                    a=plain.reset(seed=0);b=wrapped.reset(seed=0)
                    np.testing.assert_array_equal(a[0],b[0])
                    for i in range(250):
                        action=np.zeros(7) if target==0 else np.array([.1,-.2,.15,-.1,.3,.1,-.2])
                        a=plain.step(action);b=wrapped.step(action)
                        for key in ('qpos','qvel','ctrl'):
                            np.testing.assert_array_equal(getattr(plain.data,key),getattr(wrapped.unwrapped.data,key))
                        np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[2:4],b[2:4])
                        self.assertEqual(a[4],{k:b[4][k] for k in a[4]})
                        self.assertTrue(np.isfinite(b[1]))
                        if a[2]:break
                    self.assertTrue(a[2]);self.assertLessEqual(wrapped.bonus_total,12.)
                    if target==0:self.assertEqual(wrapped.bonus_total,0.)
                    else:self.assertGreater(wrapped.sample_seconds,0.)
                finally:plain.close();wrapped.close()

    def test_composed_reward_can_start_before_first_reset(self):
        env=wrap(AdaptiveEnv(REFERENCE,'x',target_degrees=90.),TRANSFORM)
        try:
            env.reset(seed=0)
            for _ in range(6):result=env.step(np.zeros(7))
            self.assertEqual(result[4]['training_reward_transform'],TRANSFORM)
            self.assertGreater(env.sample_seconds,0.)
        finally:env.close()


if __name__=='__main__':unittest.main()
