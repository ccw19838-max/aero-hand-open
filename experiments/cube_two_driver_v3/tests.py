from pathlib import Path
from dataclasses import replace
import unittest,time
import numpy as np
from experiments.cube_adaptive_60h.config import Profile
from experiments.cube_adaptive_60h.thumb_drive import contact_wrenches
from .env import NativeEnv,FastEnv,training_reward,RoleMonitor
from .roles import support_state,wrenches,driving_quality
from .evaluation import evaluate,final_pass

INPUTS=Path(__file__).parent/'artifacts/thumb_pinky_20261117/inputs'
PROFILE=Profile(grip_mix=0.)


class Zero:
    def predict(self,obs,deterministic=True):return np.zeros(7,dtype=np.float32),None


class RoleTests(unittest.TestCase):
    def make(self,kind=NativeEnv,dt=.002,target=5.):return kind(INPUTS/'reference.npz','x',physical_dt=dt,target_degrees=target,profile=PROFILE)
    def test_named_three_support_not_any_three_fingers(self):
        self.assertTrue(support_state([0.,.1,.1,.1,0.]))
        for i in (1,2,3):
            f=np.ones(5);f[i]=0.;self.assertFalse(support_state(f))
        self.assertFalse(support_state([1.,13.,1.,1.,1.]))
        self.assertEqual(driving_quality(1.,0.),0.)
        self.assertEqual(driving_quality(-1.,1.),0.)
    def test_measured_wrenches_match_independent_existing_thumb_implementation(self):
        env=self.make()
        try:
            env.reset(seed=6100000)
            for _ in range(5):env.step(np.zeros(7))
            for _ in range(6):
                env.step(np.array([0.,0.,0.,.3,-.3,0.,0.]))
                new=wrenches(env);old=contact_wrenches(env)
                np.testing.assert_allclose(new['axis_moments_Nm'],old['axis_moments_Nm'],atol=1e-12)
                self.assertAlmostEqual(new['tangential_speeds_m_s'][0],old['thumb_surface_speed_m_s'],places=12)
        finally:env.close()
    def test_native_fast_actions_observations_rewards_and_state_agree(self):
        for dt in (.001,.002):
            a=training_reward(self.make(dt=dt));b=training_reward(self.make(FastEnv,dt=dt));rng=np.random.default_rng(0)
            try:
                x,i=a.reset(seed=6100001);y,j=b.reset(seed=6100001);np.testing.assert_array_equal(x,y);self.assertEqual(i,j)
                for _ in range(100):
                    command=np.clip(rng.normal([0.,0.,0.,.1,-.1,0.,0.],.03),-1,1).astype(np.float32)
                    left=a.step(command);right=b.step(command)
                    np.testing.assert_array_equal(left[0],right[0]);self.assertEqual(left[1:],right[1:])
                    for key in ('qpos','qvel','ctrl'):np.testing.assert_array_equal(getattr(a.unwrapped.data,key),getattr(b.unwrapped.data,key))
                    if left[2]:break
            finally:a.close();b.close()
    def test_small_target_cannot_pass_by_holding_initial_orientation(self):
        env=self.make(target=5.)
        try:
            env.reset(seed=6100000)
            for _ in range(5):env.step(np.zeros(7))
            self.assertIsInstance(env.monitor,RoleMonitor);self.assertEqual(env.config.tolerance_degrees,1.25)
            for _ in range(30):
                _,_,done,_,info=env.step(np.zeros(7))
                if done:break
            self.assertFalse(info['success']);self.assertEqual(info['max_stable_seconds'],0.)
            self.assertTrue(info['from_original_initial_grasp']);self.assertFalse(info['training_prefix_used'])
        finally:env.close()
    def test_named_support_loss_terminates_even_with_old_minimum_three_support(self):
        env=self.make(target=90.)
        try:
            env.reset(seed=6100000)
            for _ in range(5):env.step(np.zeros(7))
            m=env.monitor;c=dict(env.contacts);c.update(finger_forces=np.array([1.,0.,1.,1.,1.]),forbidden_support=False,max_penetration=0.)
            for _ in range(16):m.checkpoint(env.data.qpos[19:23],c,np.zeros(3),np.zeros(3),0.,.01)
            self.assertEqual(m.failure,'three_support_lost');self.assertFalse(m.success)
        finally:env.close()
    def test_expired_evaluation_cannot_qualify(self):
        report=evaluate(Zero(),INPUTS,PROFILE,90.,1,deadline=time.time()-1)
        self.assertFalse(report['complete']);self.assertFalse(final_pass(report))


if __name__=='__main__':unittest.main()
