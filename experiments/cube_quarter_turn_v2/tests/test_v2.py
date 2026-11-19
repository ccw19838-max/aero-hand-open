"""Tests target failure modes, physical gates and train/replay semantics."""
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import time
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from experiments.cube_quarter_turn.task import quarter_target,rotation_matrix,orientation_error
from experiments.cube_quarter_turn.action_mapping_candidate import full_range_targets
from experiments.cube_quarter_turn_v2.env import TeacherEnv,targets_to_action
from experiments.cube_quarter_turn_v2.task import Config,GoalMonitor
from experiments.cube_quarter_turn_v2.prepare import prepare,hold_check
from experiments.cube_quarter_turn_v2.planner import (sequence,rollout,validate_demonstration,
    load_demonstration,PARAMETERS)
from experiments.cube_quarter_turn_v2.train import new_model,augment,train_axis
from experiments.cube_quarter_turn_v2.policy import initialize_prior,cloning_update
from experiments.cube_quarter_turn_v2.storage import save,load,atomic_json
from experiments.cube_official_baseline.train import load as load_source

ROOT=Path(__file__).resolve().parents[3]
REFERENCE=ROOT/'experiments/cube_envelop/artifacts/prepared_envelope_v1/reference.npz'
SOURCE=ROOT/'experiments/cube_official_baseline/artifacts/official_warmstart_v1/checkpoints/step_001441792_47e53726'


def contacts():
    return dict(finger_forces=np.array([1.,1.,1.,0.,0.]),hand_force=3.,max_penetration=0.,forbidden_support=False)


class MonitorTests(unittest.TestCase):
    def monitor(self):return GoalMonitor(np.array([1.,0.,0.,0.]),'z',np.eye(3))
    def test_idle_stops_early_without_changing_goal(self):
        m=self.monitor()
        for _ in range(251):m.checkpoint(m.initial,contacts(),np.zeros(3),np.zeros(3),0.)
        self.assertEqual(m.failure,'stagnation');self.assertFalse(m.success)
        self.assertAlmostEqual(np.rad2deg(m.error),90.)
    def test_stability_still_requires_one_second(self):
        m=self.monitor();m.track(m.goal)
        for _ in range(99):m.checkpoint(m.goal,contacts(),np.zeros(3),np.zeros(3),0.)
        self.assertFalse(m.success)
        m.checkpoint(m.goal,contacts(),np.zeros(3),np.zeros(3),0.)
        self.assertTrue(m.success)
    def test_failures_have_bounded_cost_and_unchanged_physical_gates(self):
        m=self.monitor();c=contacts();c['forbidden_support']=True
        value=m.checkpoint(m.initial,c,np.zeros(3),np.zeros(3),0.)
        self.assertEqual(m.failure,'forbidden_support');self.assertGreater(value,-11.)
        for key,reason in (('max_penetration','penetration'),('hand_force','lost_hand_contact')):
            m=self.monitor();c=contacts();c[key]=.002 if key=='max_penetration' else 0.
            for _ in range(21):m.checkpoint(m.initial,c,np.zeros(3),np.zeros(3),0.)
            self.assertEqual(m.failure,reason)
    def test_wrong_axis_flight_and_extra_turn_never_pass(self):
        for axis,angle in (('x',90),('z',5),('z',-270),('z',450)):
            m=self.monitor()
            for value in np.linspace(0,np.deg2rad(angle),150):
                direction=np.eye(3)[:, 'xyz'.index(axis)]
                q=np.r_[np.cos(value/2),direction*np.sin(value/2)];m.track(q)
                m.checkpoint(q,contacts(),np.zeros(3),np.zeros(3),0.)
            for _ in range(110):m.checkpoint(q,contacts(),np.zeros(3),np.zeros(3),0.)
            self.assertFalse(m.success)
        m=self.monitor();m.track(m.goal);c=contacts();c['finger_forces'][:]=0.;c['hand_force']=0.
        value=m.checkpoint(m.goal,c,np.zeros(3),np.zeros(3),0.)
        self.assertLess(value,0.);self.assertFalse(m.success)


class NativeTests(unittest.TestCase):
    def setUp(self):self.env=TeacherEnv(REFERENCE,'z')
    def tearDown(self):self.env.close()
    def test_teacher_contains_full_state_forces_targets_and_previous_action(self):
        e=self.env;o,_=e.reset(seed=0,options={'reference':True})
        self.assertEqual(o.shape,(133,));self.assertTrue(np.isfinite(o).all())
        np.testing.assert_allclose(o[14:30],e.data.qpos[:16],atol=1e-6)
        np.testing.assert_allclose(o[30:46],e.data.qvel[:16],atol=1e-6)
        np.testing.assert_allclose(o[112:117],np.clip(e.contacts['finger_forces']/5.,0.,20.),atol=1e-6)
        for _ in range(5):e.step(np.zeros(7))
        o,_,_,_,i=e.step(np.full(7,.02))
        np.testing.assert_allclose(o[7:14],np.full(7,.02),atol=1e-7)
        np.testing.assert_allclose(i['motor_targets'],full_range_targets(np.full(7,.02),e.default,e.ids['ctrl_lower'],e.ids['ctrl_upper']))
    def test_mapping_roundtrip_and_full_endpoints(self):
        e=self.env;r=e.default;lo=e.ids['ctrl_lower'];hi=e.ids['ctrl_upper']
        np.testing.assert_allclose(full_range_targets(-np.ones(7),r,lo,hi),lo)
        np.testing.assert_allclose(full_range_targets(np.ones(7),r,lo,hi),hi)
        for a in np.random.default_rng(0).uniform(-1,1,(20,7)):
            np.testing.assert_allclose(targets_to_action(full_range_targets(a,r,lo,hi),r,lo,hi),a,atol=1e-12)
    def test_numerical_failure_returns_finite_cached_observation(self):
        e=self.env;e.reset(seed=0)
        def corrupt(*args):e.data.qvel[0]=np.nan
        with patch('experiments.cube_quarter_turn_v2.env.mujoco.mj_step',side_effect=corrupt):
            obs,_,done,_,info=e.step(np.zeros(7))
        self.assertTrue(done);self.assertEqual(info['failure_reason'],'numerical');self.assertTrue(np.isfinite(obs).all())
    def test_chain_goal_change_never_resets_physics(self):
        e=self.env;e.reset(seed=0);e.monitor.success=True
        before=[getattr(e.data,k).copy() for k in ('qpos','qvel','ctrl')];stamp=e.data.time
        e.start_goal('x')
        for k,a in zip(('qpos','qvel','ctrl'),before):np.testing.assert_array_equal(getattr(e.data,k),a)
        self.assertEqual(e.data.time,stamp)
    def test_symmetry_augmentation_preserves_orientation_error(self):
        obs,_=self.env.reset(seed=0,options={'reference':True})
        augmented,actions=augment(obs[None],np.zeros((1,7),dtype=np.float32))
        self.assertEqual(augmented.shape,(24,133))
        for row in augmented:
            _,error=orientation_error(row[84:88],row[94:98]);self.assertAlmostEqual(error,np.pi/2,places=5)
            np.testing.assert_allclose(row[71:75],row[84:88],atol=1e-6)
    def test_static_failed_trajectory_cannot_become_demo(self):
        parameters=np.r_[np.zeros(35),[7,8,8,10,27]]
        approved,results=validate_demonstration(REFERENCE,'z',sequence(parameters),time.time()+20.)
        self.assertFalse(approved);self.assertTrue(all(not r['success'] for r in results))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'failed.npz';np.savez(path,observations=np.zeros((1,133)),actions=np.zeros((1,7)))
            atomic_json(path.with_suffix('.json'),dict(approved=False,goal_degrees=90.))
            with self.assertRaises(ValueError):load_demonstration(path)
    def test_deadline_interrupt_is_explicit_and_never_success(self):
        result=rollout(self.env,np.zeros((10,7)),deadline=time.time()-1.)
        self.assertTrue(result['interrupted']);self.assertFalse(result['success'])
    def test_preparation_checks_both_timesteps_before_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=prepare(REFERENCE,'z',tmp,time.time()+30.,candidates=1,keep=1)
            self.assertEqual(report['checked'],1)
            self.assertEqual(len(report['selected']),1)
            path=Path(report['selected'][0]['reference'])
            meta=json.loads(path.with_name('prepare_report.json').read_text())
            self.assertTrue(meta['approved']);self.assertEqual([r['dt'] for r in meta['validation']],[.002,.001])
            self.assertTrue(all(r['approved'] for r in meta['validation']))


class PolicyTests(unittest.TestCase):
    def test_prior_bc_update_checkpoint_and_inference_roundtrip(self):
        torch.set_num_threads(1)
        probe=TeacherEnv(REFERENCE,'z');obs,_=probe.reset(seed=0,options={'reference':True});probe.close()
        vector=VecNormalize(DummyVecEnv([lambda:TeacherEnv(REFERENCE,'z')]),norm_obs=False)
        try:
            model=new_model(vector);source,_=load_source(SOURCE);provenance=initialize_prior(model,source,obs)
            np.testing.assert_allclose(model.predict(obs,deterministic=True)[0],np.zeros(7),atol=2e-5)
            self.assertEqual(model.policy.mlp_extractor.policy_net[0].in_features,133)
            # Synthetic labels below test optimizer behavior only; never saved as a demo.
            observations=np.tile(obs,(32,1));actions=np.full((32,7),.05,dtype=np.float32)
            loss=cloning_update(model,observations,actions,np.arange(32));self.assertTrue(np.isfinite(loss))
            action=model.predict(obs,deterministic=True)[0]
            with tempfile.TemporaryDirectory() as tmp:
                checkpoint=save(model,vector,REFERENCE,tmp,'z','test',provenance)
                loaded,meta=load(checkpoint)
                np.testing.assert_array_equal(action,loaded.predict(obs,deterministic=True)[0])
                self.assertEqual(meta['goal_degrees'],90.)
            # PPO smoke exercises the shared actor/critic/normalizer path.
            model.learn(total_timesteps=128)
            self.assertEqual(model.num_timesteps,128)
        finally:vector.close()
    def test_training_is_blocked_without_complete_demos(self):
        with self.assertRaises(ValueError):train_axis([], 'x','/tmp/unused-v2',SOURCE,time.time()+1.)


if __name__=='__main__':unittest.main()
