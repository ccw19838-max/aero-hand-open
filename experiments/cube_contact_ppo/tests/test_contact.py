"""Reject the previous three-contact loophole; test actual closed-loop PPO."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv,VecNormalize
from experiments.cube_contact_ppo.task import Config,ContactMonitor,force_state
from experiments.cube_contact_ppo.env import ContactEnv
from experiments.cube_contact_ppo.train import new_model,advance
from experiments.cube_contact_ppo.policy import initialize
from experiments.cube_contact_ppo.storage import save,load
from experiments.cube_contact_ppo.evaluate import evaluate
from experiments.cube_official_baseline.train import load as load_source

ROOT=Path(__file__).resolve().parents[3]
REFERENCE=ROOT/'experiments/cube_envelop/artifacts/prepared_envelope_v1/reference.npz'
SOURCE=ROOT/'experiments/cube_official_baseline/artifacts/official_warmstart_v1/checkpoints/step_001441792_47e53726'


def contacts(forces=None):
    return dict(finger_forces=np.array(forces if forces is not None else [1.,1.,1.,1.,1.]),
        hand_force=float(sum(forces)) if forces is not None else 5.,max_penetration=0.,forbidden_support=False)


class ContactTests(unittest.TestCase):
    def monitor(self,target=90.):return ContactMonitor(np.array([1.,0.,0.,0.]),'x',np.eye(3),target_degrees=target)
    def test_zero_middle_ring_contacts_cannot_count_as_stable_capture(self):
        m=self.monitor();m.track(m.goal)
        rewards=[]
        for _ in range(110):rewards.append(m.checkpoint(m.goal,contacts([2.,1.,0.,0.,1.]),np.zeros(3),np.zeros(3),0.))
        self.assertFalse(m.success);self.assertEqual(m.max_stable,0.);self.assertFalse(m.failure)
    def test_all_five_real_forces_need_a_full_second_at_the_actual_target(self):
        m=self.monitor();m.track(m.goal)
        for _ in range(99):m.checkpoint(m.goal,contacts(),np.zeros(3),np.zeros(3),0.)
        self.assertFalse(m.success)
        m.checkpoint(m.goal,contacts(),np.zeros(3),np.zeros(3),0.);self.assertTrue(m.success)
    def test_losing_one_capture_finger_resets_timer_but_allows_regrasp(self):
        m=self.monitor();m.track(m.goal)
        for _ in range(60):m.checkpoint(m.goal,contacts(),np.zeros(3),np.zeros(3),0.)
        m.checkpoint(m.goal,contacts([1,1,1,1,0]),np.zeros(3),np.zeros(3),0.)
        self.assertEqual(m.stable,0.);self.assertFalse(m.failure)
        for _ in range(100):m.checkpoint(m.goal,contacts(),np.zeros(3),np.zeros(3),0.)
        self.assertTrue(m.success)
    def test_rotation_support_loss_has_a_bounded_grace_period(self):
        m=self.monitor();c=contacts([1,1,0,0,0])
        for _ in range(15):m.checkpoint(m.initial,c,np.zeros(3),np.zeros(3),0.)
        self.assertFalse(m.failure)
        m.checkpoint(m.initial,c,np.zeros(3),np.zeros(3),0.)
        self.assertEqual(m.failure,'insufficient_rotation_contacts')
    def test_huge_or_invalid_forces_cannot_manufacture_success(self):
        self.assertFalse(force_state([100]*5,Config())[1])
        with self.assertRaises(ValueError):force_state([np.nan]*5,Config())
        m=self.monitor();m.track(m.goal)
        for _ in range(110):m.checkpoint(m.goal,contacts([100]*5),np.zeros(3),np.zeros(3),0.)
        self.assertFalse(m.success);self.assertEqual(m.max_stable,0.)
    def test_physics_failures_stay_unchanged(self):
        for key,value,reason in [('forbidden_support',True,'forbidden_support'),('max_penetration',.002,'penetration')]:
            m=self.monitor();c=contacts();c[key]=value
            for _ in range(11):m.checkpoint(m.initial,c,np.zeros(3),np.zeros(3),0.)
            self.assertEqual(m.failure,reason)
    def test_full_goal_does_not_accept_a_small_angle_or_extra_turn_alias(self):
        for degrees in (30.,-270.,450.):
            m=self.monitor()
            for angle in np.linspace(0.,np.deg2rad(degrees),180):
                q=np.array([np.cos(angle/2),np.sin(angle/2),0,0]);m.track(q)
                m.checkpoint(q,contacts(),np.zeros(3),np.zeros(3),0.)
            for _ in range(110):m.checkpoint(q,contacts(),np.zeros(3),np.zeros(3),0.)
            self.assertFalse(m.success)
    def test_curriculum_advancement_requires_two_real_passes_of_the_same_target(self):
        c=dict(target_degrees=0.,consecutive_passes=0)
        r=dict(target_degrees=0.,complete=True,min_integrator_success_rate=.9)
        self.assertFalse(advance(c,r));self.assertTrue(advance(c,r));self.assertEqual(c['target_degrees'],30.)
        with self.assertRaises(ValueError):advance(c,r)


class NativeTests(unittest.TestCase):
    def test_neutral_real_five_finger_grip_passes_both_integrators(self):
        for dt in (.001,.002):
            e=ContactEnv(REFERENCE,'x',physical_dt=dt)
            try:
                obs,_=e.reset(seed=0,options={'reference':True})
                self.assertEqual(obs.shape,(138,))
                for _ in range(30):
                    obs,_,done,_,info=e.step(np.zeros(7))
                    if done:break
                self.assertTrue(info['success']);self.assertTrue(info['five_finger_contact'])
                self.assertGreaterEqual(info['stable_seconds'],1.)
                self.assertTrue(np.all(np.array(info['finger_forces'])>0.))
            finally:e.close()
    def test_incremental_targets_are_bounded_and_history_is_the_previous_action(self):
        e=ContactEnv(REFERENCE,'x',target_degrees=30.)
        try:
            e.reset(seed=0,options={'reference':True})
            for _ in range(5):e.step(np.zeros(7))
            before=e.data.ctrl.copy();obs,_,_,_,info=e.step(np.ones(7))
            expected=np.clip(before+.02*(e.ids['ctrl_upper']-e.ids['ctrl_lower']),e.ids['ctrl_lower'],e.ids['ctrl_upper'])
            np.testing.assert_allclose(info['motor_targets'],expected,atol=1e-12)
            np.testing.assert_array_equal(obs[7:14],np.ones(7))
        finally:e.close()
    def test_curriculum_change_waits_for_reset(self):
        e=ContactEnv(REFERENCE,'x')
        try:
            obs,_=e.reset(seed=0);goal=e.monitor.goal.copy()
            e.set_task(30.,.05)
            self.assertEqual(e.target_degrees,0.);np.testing.assert_array_equal(e.monitor.goal,goal)
            obs,info=e.reset(seed=0)
            self.assertEqual(info['goal_degrees'],30.);self.assertAlmostEqual(obs[133],30/90,places=6)
            self.assertEqual(info['applied_randomization'],.05)
        finally:e.close()
    def test_numerical_failure_returns_finite_terminal_observation(self):
        e=ContactEnv(REFERENCE,'y')
        try:
            e.reset(seed=0)
            def corrupt(*args):e.data.qvel[0]=np.nan
            with patch('experiments.cube_contact_ppo.env.mujoco.mj_step',side_effect=corrupt):
                obs,_,done,_,info=e.step(np.zeros(7))
            self.assertTrue(done);self.assertEqual(info['failure_reason'],'numerical');self.assertTrue(np.isfinite(obs).all())
        finally:e.close()
    def test_next_goal_preserves_physics_and_keeps_five_finger_monitor(self):
        e=ContactEnv(REFERENCE,'x')
        try:
            e.reset(seed=0);e.monitor.success=True
            before={k:getattr(e.data,k).copy() for k in ('qpos','qvel','ctrl')};clock=e.data.time
            _,info=e.start_goal('y')
            for key,value in before.items():np.testing.assert_array_equal(getattr(e.data,key),value)
            self.assertEqual(clock,e.data.time);self.assertIsInstance(e.monitor,ContactMonitor)
            self.assertEqual(info['goal_degrees'],90.)
        finally:e.close()


class PolicyTests(unittest.TestCase):
    def test_actual_two_worker_ppo_has_no_demo_gate_and_checkpoint_roundtrips(self):
        torch.set_num_threads(1)
        factories=[lambda:ContactEnv(REFERENCE,'x',physical_dt=.001),lambda:ContactEnv(REFERENCE,'x',physical_dt=.002)]
        vector=SubprocVecEnv(factories,start_method='spawn');normalizer=VecNormalize(vector,norm_obs=False)
        probe=ContactEnv(REFERENCE,'x')
        try:
            observation,_=probe.reset(seed=0,options={'reference':True})
            model=new_model(normalizer);source,_=load_source(SOURCE);provenance=initialize(model,source,observation)
            np.testing.assert_array_equal(model.predict(observation,deterministic=True)[0],np.zeros(7))
            self.assertFalse(provenance['demonstrations_used'])
            model.learn(total_timesteps=256)
            self.assertEqual(model.num_timesteps,256);self.assertGreater(model._n_updates,0)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.policy.parameters()))
            action=model.predict(observation,deterministic=True)[0]
            with tempfile.TemporaryDirectory() as directory:
                path=save(model,normalizer,REFERENCE,directory,'x','test',provenance,dict(target_degrees=0.))
                loaded,metadata=load(path)
                np.testing.assert_array_equal(action,loaded.predict(observation,deterministic=True)[0])
                self.assertEqual(metadata['observation_size'],138)
                report=evaluate(loaded,REFERENCE,'x',0.,episodes=1)
                self.assertFalse(report['final_goal_evaluation'])
                self.assertEqual(set(report['groups']),{'0.001','0.002'})
        finally:normalizer.close();probe.close()


if __name__=='__main__':unittest.main()
