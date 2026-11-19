from pathlib import Path
import unittest
import numpy as np
import mujoco
from experiments.cube_adaptive_60h.env import AdaptiveEnv
from experiments.cube_adaptive_60h.fast_contacts import FastContactEnv

REFERENCE=Path(__file__).resolve().parents[3]/'experiments/cube_envelop/artifacts/prepared_envelope_v1/reference.npz'


class ContactEquivalence(unittest.TestCase):
    def test_known_real_30degree_policy_success_matches_both_steps(self):
        import torch
        from experiments.cube_adaptive_60h.storage import load_source
        torch.set_num_threads(1)
        checkpoint=Path(__file__).resolve().parents[3]/'experiments/cube_contact_ppo/artifacts/contact_ppo_v1/y/best_30'
        policy,_=load_source(checkpoint)
        for dt in (.001,.002):
            plain=AdaptiveEnv(REFERENCE,'y',physical_dt=dt,target_degrees=30.)
            fast=FastContactEnv(REFERENCE,'y',physical_dt=dt,target_degrees=30.)
            try:
                obs,_=plain.reset(seed=1000001);fast.reset(seed=1000001)
                for _ in range(250):
                    action=policy.predict(obs[:138],deterministic=True)[0]
                    a=plain.step(action);b=fast.step(action);obs=a[0]
                    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(plain.data.qpos,fast.data.qpos)
                    self.assertEqual(a[1:4],b[1:4]);self.assertEqual(a[4],b[4])
                    if a[2]:break
                self.assertTrue(a[4]['success']);self.assertTrue(b[4]['success'])
            finally:plain.close();fast.close()
    def test_full_state_observation_reward_and_failure_are_bitwise_identical(self):
        for dt in (.001,.002):
            for seed in (0,1000000):
                plain=AdaptiveEnv(REFERENCE,'x',physical_dt=dt,target_degrees=10.)
                fast=FastContactEnv(REFERENCE,'x',physical_dt=dt,target_degrees=10.)
                try:
                    a=plain.reset(seed=seed);b=fast.reset(seed=seed);np.testing.assert_array_equal(a[0],b[0])
                    rng=np.random.default_rng(seed+5)
                    for _ in range(120):
                        action=rng.normal(0.,.15,7);a=plain.step(action);b=fast.step(action)
                        np.testing.assert_array_equal(plain.data.qpos,fast.data.qpos)
                        np.testing.assert_array_equal(plain.data.qvel,fast.data.qvel)
                        np.testing.assert_array_equal(plain.data.ctrl,fast.data.ctrl)
                        np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[1:4],b[1:4])
                        self.assertEqual(a[4],b[4])
                        if a[2]:break
                finally:plain.close();fast.close()
    def test_real_five_finger_success_and_floor_force_detection(self):
        for dt in (.001,.002):
            plain=AdaptiveEnv(REFERENCE,'x',physical_dt=dt,target_degrees=0.)
            fast=FastContactEnv(REFERENCE,'x',physical_dt=dt,target_degrees=0.)
            try:
                plain.reset(seed=0,options={'reference':True});fast.reset(seed=0,options={'reference':True})
                for _ in range(35):
                    a=plain.step(np.zeros(7));b=fast.step(np.zeros(7));self.assertEqual(a[4],b[4])
                    if a[2]:break
                self.assertTrue(a[4]['success']);self.assertTrue(b[4]['success'])
                for env in (plain,fast):
                    env.data.qpos[18]=.02;mujoco.mj_forward(env.model,env.data)
                self.assertTrue(plain._forbidden());self.assertTrue(fast._forbidden())
                a=plain._contacts();b=fast._contacts()
                np.testing.assert_array_equal(a['finger_forces'],b['finger_forces'])
                for key in ('hand_force','max_penetration','forbidden_support'):self.assertEqual(a[key],b[key])
            finally:plain.close();fast.close()


if __name__=='__main__':unittest.main()
