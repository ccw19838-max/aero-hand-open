"""Physics parity, strict score counterexamples, and paired actor initialization."""
import json
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.cube_adaptive_60h.config import Profile
from experiments.cube_adaptive_60h.policy import Policy as OriginalPolicy
from experiments.cube_two_driver_v3.env import NativeEnv as OriginalEnv
from experiments.cube_two_driver_v3.roles import driving_quality
from .contact_history import ContactHistory
from .env import BaselineEvalEnv, FastEnv, NativeEnv, training_reward
from .policy import Policy, initialize_from_v3

RUN = Path(__file__).resolve().parents[1] / 'cube_two_driver_v3/artifacts/thumb_pinky_20261117'
PROFILE = Profile(**json.loads((RUN / 'job.json').read_text())['profile'])


def make(cls, dt=.002):
    return cls(RUN / 'inputs/reference.npz', 'x', physical_dt=dt, target_degrees=5., profile=PROFILE)


class StrictEnvironmentTests(unittest.TestCase):
    def test_credit_only_once_per_supported_new_pose_and_latched_violation(self):
        def simulate(target, inputs):
            history = ContactHistory(); previous, previous_error, total = 0., target, 0.
            for angle, error, index, valid in inputs:
                forces = [1., index, 1., 1., 1.]
                total += history.sample(forces, angle, error, target, previous, previous_error, valid, 0.)
                previous, previous_error = angle, error
            return total, history
        for target in (5., 10., 90.):
            samples = [(fraction * target, (1.-fraction)*target, 1., True) for fraction in (0., .3, .1, .5, .3, 1., 1.)]
            total, history = simulate(target, samples)
            self.assertAlmostEqual(20*total, 20.); self.assertIsNone(history.first_violation)
            self.assertAlmostEqual(8*total, 8.)
        total, history = simulate(5., [(2.,3.,.1,True),(3.,2.,0.,True),(5.,0.,12.,True)])
        self.assertAlmostEqual(total*20, 8.); self.assertEqual(history.valid_peak_deg, 2.)
        self.assertEqual(history.first_violation.fingers,('index',)); self.assertEqual(history.first_violation.time_s,.02)
        for samples in [[(0.,5.,1.,True)],[(3.,6.,1.,True)],[(5.,0.,1.,False)]]:
            self.assertEqual(simulate(5.,samples)[0],0.)
        self.assertEqual(driving_quality(.2,0.),0.)

    def test_causal_warning_horizon_false_censored_and_simultaneous_support_loss(self):
        h=ContactHistory()
        def sample(f):return h.sample([1.,*f,1.],0.,5.,5.,0.,5.,True,0.)
        for f in (.38,.35,.32,.29,.26,.23,.20):sample([f,f,f])
        np.testing.assert_array_equal(h.warnings,[True,True,True]); self.assertEqual(h.warning_count,3)
        self.assertEqual(h.info()['warning_censored_count'],3);self.assertIsNone(h.first_violation)
        sample([0.,0.,.6]);self.assertEqual(h.first_violation.fingers,('index','middle'))
        self.assertEqual(h.warning_lead_ms,[10,10]); self.assertEqual(h.info()['warning_censored_count'],1)
        for _ in range(4):sample([1.,1.,.6])
        self.assertEqual(h.warning_false_count,1); self.assertEqual(h.info()['warning_censored_count'],0)
        self.assertFalse(h.warnings[2]); self.assertTrue(h.warnings[0]); sample([1.,1.,1.]);self.assertFalse(h.warnings.any())

    def test_v3_sidecar_never_changes_policy_observation_rewards_or_termination(self):
        for dt in (.001,.002):
            original=make(OriginalEnv,dt);scored=make(BaselineEvalEnv,dt)
            try:
                a,_=original.reset(seed=1198); b,_=scored.reset(seed=1198);np.testing.assert_array_equal(a,b)
                for _ in range(250):
                    action=np.array([.4,0.,0.,.3,-.3,0.,0.],dtype=np.float32)
                    left=original.step(action); right=scored.step(action)
                    np.testing.assert_array_equal(left[0],right[0]);self.assertEqual(left[1:4],right[1:4]);
                    if left[2]:break
                self.assertTrue(left[2]);self.assertFalse(right[4]['strict_success']);self.assertIsNotNone(right[4]['first_violation'])
            finally:original.close();scored.close()

    def test_native_fast_physics_observation_reward_and_first_loss(self):
        for dt in (.001,.002):
            a=training_reward(make(NativeEnv,dt)); b=training_reward(make(FastEnv,dt))
            try:
                x,i=a.reset(seed=692);y,j=b.reset(seed=692);np.testing.assert_array_equal(x,y)
                self.assertEqual(x.shape,(149,));self.assertEqual(i,j)
                for step in range(250):
                    action=np.array([0. if step<10 else .35,0.,0.,.3,-.3,0.,0.],dtype=np.float32)
                    l=a.step(action);r=b.step(action)
                    np.testing.assert_array_equal(l[0],r[0]);self.assertEqual(l[1:],r[1:])
                    for key in ('qpos','qvel','ctrl'):np.testing.assert_array_equal(getattr(a.unwrapped.data,key),getattr(b.unwrapped.data,key))
                    if l[2]:break
                self.assertTrue(l[2]);self.assertIsNotNone(l[4]['first_violation']);self.assertFalse(l[4]['strict_success'])
            finally:a.close();b.close()

    def test_lost_then_regrasped_cannot_earn_final_bonus(self):
        env=make(NativeEnv)
        try:
            env.reset(seed=212)
            for _ in range(5):env.step(np.zeros(7))
            m=env.monitor;c=dict(env.contacts);c.update(finger_forces=np.array([1.,0.,1.,1.,1.]),forbidden_support=False,max_penetration=0.)
            m.checkpoint(env.data.qpos[19:23],c,np.zeros(3),np.zeros(3),0.)
            m.progress=np.deg2rad(5.);m.stable=.99;c['finger_forces']=np.ones(5)
            value=m.checkpoint(m.goal,c,np.zeros(3),np.zeros(3),0.)
            info=env._info();self.assertTrue(m.done);self.assertFalse(m.success);self.assertFalse(info['strict_success'])
            self.assertEqual(info['eligible_peak_degrees'],0.);self.assertEqual(info['strict_hold_seconds'],0.)
            self.assertEqual(info['normalized_progress_return'],0.);self.assertLess(value,env.config.success_bonus)
        finally:env.close()

    def test_forbidden_support_partial_turn_and_retreat_do_not_succeed(self):
        for kind in ('forbidden','partial','retreat'):
            env=make(NativeEnv)
            try:
                env.reset(seed=118)
                for _ in range(5):env.step(np.zeros(7))
                m=env.monitor;c=dict(env.contacts);c.update(finger_forces=np.ones(5),forbidden_support=kind=='forbidden',max_penetration=0.)
                m.progress=np.deg2rad(5. if kind!='partial' else 1.)
                m.stable=.99 if kind!='retreat' else .98
                m.checkpoint(m.goal,c,np.zeros(3),np.zeros(3),0.)
                if kind=='retreat':m.progress=0.;m.checkpoint(m.initial,c,np.zeros(3),np.zeros(3),0.)
                self.assertFalse(env._info()['strict_success'])
            finally:env.close()

    def test_identical_initialized_gaussian_and_critic_with_nonzero_new_features(self):
        torch.set_num_threads(1);torch.manual_seed(5)
        action=gym.spaces.Box(-1.,1.,shape=(7,),dtype=np.float32)
        obs=lambda n:gym.spaces.Box(-np.inf,np.inf,shape=(n,),dtype=np.float32)
        baseline=OriginalPolicy(obs(143),action,lambda _:1.e-4)
        candidate=Policy(obs(149),action,lambda _:1.e-4)
        initialize_from_v3(candidate,baseline)
        x=torch.rand(8,143);y=torch.cat((x,torch.randn(8,6)),dim=1)
        with torch.no_grad():
            b=baseline.get_distribution(x).distribution;a=candidate.get_distribution(y).distribution
            torch.testing.assert_close(a.mean,b.mean,atol=2e-7,rtol=3e-6)
            torch.testing.assert_close(a.stddev,b.stddev,atol=0.,rtol=0.)
            torch.testing.assert_close(baseline.predict_values(x),candidate.predict_values(y),atol=2e-7,rtol=3e-6)
            self.assertFalse(candidate.mlp_extractor.policy_net[0].weight[:,143:].any())


if __name__=='__main__':unittest.main()
