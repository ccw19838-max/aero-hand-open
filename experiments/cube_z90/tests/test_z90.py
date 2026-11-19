from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.cube_z90.config import ANGLES, CubeReorientConfig, tolerance_for
from experiments.cube_z90.env import (CubeReorientEnv, ReorientMonitor,
    physical_substep_reward, swing_radians, target_quaternion)
from experiments.cube_z90.curriculum import Runner
from experiments.cube_z90.train import AxisPromotionMonitor

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT/'experiments/cube_envelop/artifacts/prepared_envelope_v1/reference.npz'


class GoalTests(unittest.TestCase):
    def prepared(self, angle=90.):
        c=CubeReorientConfig(target_degrees=angle,angle_tolerance_degrees=tolerance_for(angle))
        m=ReorientMonitor(c,np.array([1.,0.,0.,0.]),'z')
        contacts={'finger_forces':np.ones(5),'max_penetration':0.,'forbidden_support':False}
        self.feed(m,np.array([1.,0.,0.,0.]),contacts,250)
        self.assertTrue(m.capture_succeeded)
        return c,m,contacts

    @staticmethod
    def feed(m, q, contacts, count, angular=None):
        for _ in range(count):
            m.update(q,contacts,np.zeros(3),np.zeros(3) if angular is None else angular,0.,.2)

    def test_all_curriculum_levels_exclude_no_motion(self):
        for angle in ANGLES:
            c,m,contacts=self.prepared(angle)
            self.feed(m,np.array([1.,0.,0.,0.]),contacts,1000)
            self.assertFalse(m.success)
            self.assertLess(c.angle_tolerance_degrees,angle)

    def test_real_quarter_turn_requires_a_full_second(self):
        c,m,contacts=self.prepared()
        q=target_quaternion(np.array([1.,0.,0.,0.]),'z',90.)
        self.feed(m,q,contacts,499)
        self.assertFalse(m.success)
        self.feed(m,q,contacts,1)
        self.assertTrue(m.success)
        self.assertAlmostEqual(np.rad2deg(m.signed_progress_radians),90.)

    def test_final_fast_motion_or_missing_one_finger_does_not_pass(self):
        for mode in ['speed','contact']:
            c,m,contacts=self.prepared()
            q=target_quaternion(np.array([1.,0.,0.,0.]),'z',90.)
            if mode=='contact':contacts['finger_forces'][2]=0.
            self.feed(m,q,contacts,600,angular=np.array([0.,0.,.6]) if mode=='speed' else None)
            self.assertFalse(m.success)

    def test_wrong_axis_and_wrong_direction_fail(self):
        for axis,angle in [('x',90.),('y',90.),('z',-90.)]:
            c,m,contacts=self.prepared()
            q=target_quaternion(np.array([1.,0.,0.,0.]),axis,angle)
            self.feed(m,q,contacts,600)
            self.assertFalse(m.success)

    def test_extra_turns_do_not_alias_to_90(self):
        for angle in [-270.,450.]:
            c,m,contacts=self.prepared()
            for value in np.linspace(0.,angle,251):
                self.feed(m,target_quaternion(np.array([1.,0.,0.,0.]),'z',value),contacts,1)
            self.feed(m,target_quaternion(np.array([1.,0.,0.,0.]),'z',angle),contacts,600)
            self.assertFalse(m.success)
            self.assertAlmostEqual(np.rad2deg(m.signed_progress_radians),angle)

    def test_no_positive_progress_credit_without_valid_contact(self):
        c,m,contacts=self.prepared()
        empty=dict(contacts,finger_forces=np.zeros(5))
        for valid,contact in [(False,contacts),(True,empty)]:
            reward=physical_substep_reward(c,m,contact,np.zeros(3),np.zeros(3),False,valid,.1)
            self.assertLess(reward,0.)
            reverse=physical_substep_reward(c,m,contact,np.zeros(3),np.zeros(3),False,valid,-.1)
            self.assertLess(reverse,-7.9)

    def test_oscillation_does_not_multiply_progress_reward(self):
        c,m,contacts=self.prepared()
        before=physical_substep_reward(c,m,contacts,np.zeros(3),np.zeros(3),False,True,0.)
        forward=physical_substep_reward(c,m,contacts,np.zeros(3),np.zeros(3),False,True,.1)
        reverse=physical_substep_reward(c,m,contacts,np.zeros(3),np.zeros(3),False,True,-.1)
        self.assertAlmostEqual(forward+reverse,2*before)

    def test_swing_uses_fixed_hand_axis(self):
        axis=np.array([.3419335,0.,.9397242]);axis/=np.linalg.norm(axis)
        q=np.r_[np.cos(np.pi/4),axis*np.sin(np.pi/4)]
        self.assertAlmostEqual(swing_radians(q,np.array([1.,0.,0.,0.]),axis),0.,places=6)
        self.assertAlmostEqual(swing_radians(np.array([0.,0.,1.,0.]),np.array([1.,0.,0.,0.]),axis),np.pi)

    def test_z_only_and_dominant_terminal_reward(self):
        c=CubeReorientConfig()
        self.assertEqual(c.training_axes,('z',))
        with self.assertRaises(ValueError):replace(c,training_axes=('x',))
        with self.assertRaises(ValueError):replace(c,target_hold_seconds=.5)
        with self.assertRaises(ValueError):replace(c,success_reward=100.)


class NativeAndBudgetTests(unittest.TestCase):
    @unittest.skipUnless(REFERENCE.exists(),'Prepared reference is an artifact')
    def test_task_timeout_is_a_terminal_failure(self):
        env=CubeReorientEnv(REFERENCE,replace(CubeReorientConfig(),episode_seconds=3.))
        try:
            env.reset(seed=10000)
            for _ in range(61):
                obs,reward,terminated,truncated,info=env.step(np.zeros(7))
                if terminated or truncated:break
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info['failure_reason'],'timeout')
            self.assertLess(reward,-99.)
        finally:env.close()

    @unittest.skipUnless(REFERENCE.exists(),'Prepared reference is an artifact')
    def test_acquire_holds_motor_targets_then_actor_controls_rotation(self):
        env=CubeReorientEnv(REFERENCE)
        try:
            obs,info=env.reset(seed=10000)
            initial=env.data.ctrl.copy()
            for _ in range(40):
                obs,r,term,trunc,info=env.step(np.ones(7))
                np.testing.assert_array_equal(env.data.ctrl,initial)
                np.testing.assert_array_equal(env.previous_action,np.zeros(7))
                self.assertTrue(np.isfinite(r))
                if info['capture_succeeded']:break
            self.assertTrue(info['capture_succeeded'])
            self.assertFalse(term or trunc)
            env.step(np.ones(7))
            expected=np.clip(initial+.02*env._range,env.ids['ctrl_lower'],env.ids['ctrl_upper'])
            np.testing.assert_allclose(env.data.ctrl,expected)
            self.assertFalse(np.array_equal(initial,env.data.ctrl))
        finally:env.close()

    def test_promotion_needs_two_complete_checks(self):
        promotion=AxisPromotionMonitor(CubeReorientConfig())
        passed={'groups':{'z':{'complete':True,'success_rate':.9}}}
        partial={'groups':{'z':{'complete':False,'success_rate':1.}}}
        self.assertFalse(promotion.update(passed))
        self.assertFalse(promotion.update(partial))
        self.assertFalse(promotion.update(passed))
        self.assertTrue(promotion.update(passed))
        self.assertEqual(promotion.ready_axes,['z'])

    def test_resume_preserves_original_deadline_and_blocks_duplicate(self):
        stamp=datetime(2026,11,12,3,tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp,patch('experiments.cube_z90.curriculum.now',return_value=stamp):
            folder=Path(tmp)/'run'
            runner=Runner('ref','ckpt',folder)
            deadline=runner.state['deadline_at']
            self.assertEqual(runner.state['budget_hours'],24.)
            self.assertEqual(runner.state['training_axes'],['z'])
            with self.assertRaises(ValueError):Runner('ref','ckpt',folder,resume=True)
            runner.lock.close()
            resumed=Runner('ref','ckpt',folder,hours=1.,resume=True)
            self.assertEqual(resumed.state['deadline_at'],deadline)
            resumed.lock.close()

    def test_only_z_advances_and_failed_independent_test_is_not_completion(self):
        stamp=datetime(2026,11,12,3,tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp,patch('experiments.cube_z90.curriculum.now',return_value=stamp),patch('experiments.cube_z90.curriculum.time.time',return_value=stamp.timestamp()):
            runner=Runner('ref','source',Path(tmp)/'run')
            angles=[];verification_seeds=[]
            def child(name,command,training=True):
                folder=runner.output/name;folder.mkdir(exist_ok=True)
                if training:
                    config=json.loads((folder/'stage_config.json').read_text())
                    self.assertEqual(config['training_axes'],['z'])
                    angle=config['target_degrees'];angles.append(angle)
                    report={'training_steps':1024,'latest_checkpoint':str(folder/'saved'),
                            'stop_reason':'final_targets_validated' if angle==90 else 'angle_stage_ready',
                            'latest_validation':{'validation_complete':True,'consecutive_axis_passes':{'z':2}}}
                    (folder/'status.json').write_text(json.dumps(report))
                elif name.startswith('verify90'):
                    verification_seeds.append(int(command[command.index('--seed-start')+1]))
                    # Course validation alone cannot certify the task. Reject
                    # the first independent set, pass a new one after retraining.
                    (folder/'evaluation.json').write_text(json.dumps({
                        'complete':True,'success_rate':.8 if len(verification_seeds)==1 else .95}))
                return 0
            runner.run_child=child
            result=runner.run()
            self.assertEqual(angles,list(ANGLES)+[90.])
            self.assertEqual(verification_seeds,[50000000,50001000])
            self.assertEqual(result['stop_reason'],'z90_independently_verified')
            self.assertEqual(result['training_steps'],1024*(len(ANGLES)+1))


if __name__=='__main__':unittest.main()
