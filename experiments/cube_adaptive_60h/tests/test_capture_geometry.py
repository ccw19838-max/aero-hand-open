from pathlib import Path
import unittest
import numpy as np
from experiments.cube_adaptive_60h.env import AdaptiveEnv
from experiments.cube_adaptive_60h.capture_geometry import CaptureGeometryReward

REFERENCE=Path(__file__).resolve().parents[3]/'experiments/cube_envelop/artifacts/prepared_envelope_v1/reference.npz'


class GeometryReward(unittest.TestCase):
    def test_original_numerical_failure_keeps_finite_reward(self):
        from unittest.mock import patch
        wrapped=CaptureGeometryReward(AdaptiveEnv(REFERENCE,'y',target_degrees=90.))
        try:
            obs,_=wrapped.reset(seed=0);wrapped.unwrapped.warming=False
            def original_failure(action):
                # Synthetic numerical terminal state, never a scored rollout.
                wrapped.unwrapped.data.time=float('inf')
                return obs,-20.,True,False,dict(failure_reason='numerical')
            with patch.object(wrapped.env,'step',side_effect=original_failure):result=wrapped.step(np.zeros(7))
            self.assertEqual(result[1],-20.);self.assertTrue(np.isfinite(result[1]))
            self.assertTrue(result[2]);self.assertEqual(result[4]['failure_reason'],'numerical')
            self.assertEqual(result[4]['geometry_penalty'],0.)
        finally:wrapped.close()
    def test_both_integrators_keep_physics_observations_history_and_original_success(self):
        for dt in (.001,.002):
            plain=AdaptiveEnv(REFERENCE,'y',physical_dt=dt,target_degrees=0.)
            wrapped=CaptureGeometryReward(AdaptiveEnv(REFERENCE,'y',physical_dt=dt,target_degrees=0.))
            try:
                plain.reset(seed=0,options={'reference':True});wrapped.reset(seed=0,options={'reference':True})
                adjustment=0.
                for _ in range(35):
                    warming=plain.warming;a=plain.step(np.zeros(7));b=wrapped.step(np.zeros(7))
                    np.testing.assert_array_equal(plain.data.qpos,wrapped.unwrapped.data.qpos)
                    np.testing.assert_array_equal(plain.data.qvel,wrapped.unwrapped.data.qvel)
                    np.testing.assert_array_equal(plain.data.ctrl,wrapped.unwrapped.data.ctrl)
                    np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[2:4],b[2:4])
                    self.assertEqual(a[4],{k:b[4][k] for k in a[4]})
                    if warming:self.assertEqual(a[1],b[1])
                    self.assertLessEqual(b[1],a[1]+1e-12);adjustment+=a[1]-b[1]
                    if a[2]:break
                self.assertTrue(a[4]['success']);self.assertTrue(b[4]['success'])
                self.assertAlmostEqual(adjustment,wrapped.total_penalty,places=10)
                self.assertGreater(wrapped.total_penalty,0.)
                self.assertAlmostEqual(wrapped.components()[1],0.)
            finally:plain.close();wrapped.close()


if __name__=='__main__':unittest.main()
