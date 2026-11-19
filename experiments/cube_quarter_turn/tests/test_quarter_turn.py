from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize

from experiments.cube_quarter_turn.task import (SYMMETRIES,AXES,GoalMonitor,quarter_target,
    rotation_matrix,matrix_quaternion,aligned_cube_basis,quaternion_multiply)
from experiments.cube_quarter_turn.env import QuarterTurnEnv
from experiments.cube_quarter_turn.train import new_model
from experiments.cube_quarter_turn.policy import initialize_prior
from experiments.cube_quarter_turn.checkpoints import save,load
from experiments.cube_official_baseline.train import load as load_source

ROOT=Path(__file__).resolve().parents[3]
REFERENCE=ROOT/'experiments/cube_envelop/artifacts/prepared_envelope_v1/reference.npz'
SOURCE=ROOT/'experiments/cube_official_baseline/artifacts/official_warmstart_v1/checkpoints/step_001441792_47e53726'


def contacts(forces=None):
    forces=np.array([1.,1.,1.,0.,0.]) if forces is None else np.array(forces,dtype=float)
    return {'finger_forces':forces,'hand_force':float(sum(forces)),'max_penetration':0.,'forbidden_support':False}


class GeometryAndTaskTests(unittest.TestCase):
    def test_three_positive_quarter_turns_generate_all_cube_orientations(self):
        self.assertEqual(len(SYMMETRIES),24)
        turns=[np.rint(rotation_matrix(quarter_target(np.array([1.,0.,0.,0.]),a,np.eye(3)))).astype(int) for a in AXES]
        found={tuple(np.eye(3,dtype=int).ravel())};pending=[np.eye(3,dtype=int)]
        while pending:
            current=pending.pop()
            for turn in turns:
                next_=turn@current;key=tuple(next_.ravel())
                if key not in found:found.add(key);pending.append(next_)
        self.assertEqual(len(found),24)
        normals={tuple(np.array(k).reshape(3,3)@np.array([0,0,1])) for k in found}
        self.assertEqual(len(normals),6)

    def test_90_degree_endpoint_preserves_box_collision_geometry(self):
        initial=np.array([.89201518,-.06371006,.11692644,.43194693]);initial/=np.linalg.norm(initial)
        basis=aligned_cube_basis(initial,np.eye(3))
        corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*.0285
        for symmetry in SYMMETRIES:
            current=quaternion_multiply(initial,matrix_quaternion(symmetry));before=corners@rotation_matrix(current).T
            for axis in AXES:
                target=quarter_target(current,axis,basis);after=corners@rotation_matrix(target).T
                distance=np.linalg.norm(before[:,None,:]-after[None,:,:],axis=-1)
                self.assertLess(np.max(np.min(distance,axis=1)),1e-10)

    def monitor(self):return GoalMonitor(np.array([1.,0.,0.,0.]),'z',np.eye(3))

    def test_waiting_or_stopping_at_5_degrees_cannot_pass(self):
        for angle in (0.,5.):
            m=self.monitor();q=np.r_[np.cos(np.deg2rad(angle)/2),0,0,np.sin(np.deg2rad(angle)/2)]
            m.track(q)
            for _ in range(110):m.checkpoint(q,contacts(),np.zeros(3),np.zeros(3),0.)
            self.assertFalse(m.success)
            self.assertGreater(np.rad2deg(m.error),80.)

    def test_actual_90_with_three_fingers_needs_full_second(self):
        m=self.monitor();q=m.goal;m.track(q)
        for _ in range(99):m.checkpoint(q,contacts(),np.zeros(3),np.zeros(3),0.)
        self.assertFalse(m.success)
        m.checkpoint(q,contacts(),np.zeros(3),np.zeros(3),0.)
        self.assertTrue(m.success)

    def test_missing_thumb_or_fast_object_does_not_pass(self):
        for force,omega in [(contacts([0,1,1,1,1]),np.zeros(3)),(contacts(),np.array([0,0,.6]))]:
            m=self.monitor();m.track(m.goal)
            for _ in range(110):m.checkpoint(m.goal,force,np.zeros(3),omega,0.)
            self.assertFalse(m.success)

    def test_wrong_axis_and_extra_turns_fail(self):
        for angle in (-270.,450.):
            m=self.monitor()
            for theta in np.linspace(0,np.deg2rad(angle),200):
                q=np.array([np.cos(theta/2),0,0,np.sin(theta/2)]);m.track(q)
                m.checkpoint(q,contacts(),np.zeros(3),np.zeros(3),0.)
            for _ in range(110):m.checkpoint(q,contacts(),np.zeros(3),np.zeros(3),0.)
            self.assertFalse(m.success)
        m=self.monitor();wrong=quarter_target(m.initial,'x',np.eye(3));m.track(wrong)
        for _ in range(110):m.checkpoint(wrong,contacts(),np.zeros(3),np.zeros(3),0.)
        self.assertFalse(m.success)

    def test_flight_cannot_collect_positive_pose_progress(self):
        m=self.monitor();q=quarter_target(m.initial,'z',np.eye(3));m.track(q)
        reward=m.checkpoint(q,contacts([0]*5),np.zeros(3),np.zeros(3),0.)
        self.assertLess(reward,0.)


@unittest.skipUnless(REFERENCE.exists(),'Prepared reference required')
class NativeTests(unittest.TestCase):
    def setUp(self):self.env=QuarterTurnEnv(REFERENCE)
    def tearDown(self):self.env.close()

    def test_all_axes_start_with_90_degree_goal(self):
        for axis in AXES:
            obs,info=self.env.reset(seed=0,options={'axis':axis,'reference':True})
            self.assertEqual(info['goal_degrees'],90.)
            self.assertAlmostEqual(info['angle_error_degrees'],90.)
            self.assertEqual(obs['goal_state'].shape,(31,))

    def test_no_state_reset_between_successful_goal_commands(self):
        self.env.reset(seed=0,options={'axis':'x','reference':True})
        self.env.monitor.success=True
        before=[getattr(self.env.data,k).copy() for k in ('qpos','qvel','ctrl')];before_time=self.env.data.time
        self.env.start_goal('y')
        for key,value in zip(('qpos','qvel','ctrl'),before):np.testing.assert_array_equal(getattr(self.env.data,key),value)
        self.assertEqual(self.env.data.time,before_time)

    def test_numerical_failure_returns_finite_terminal_observation(self):
        self.env.reset(seed=0)
        def corrupt(*args,**kwargs):self.env.data.qvel[0]=np.nan
        with patch('experiments.cube_quarter_turn.env.mujoco.mj_step',side_effect=corrupt):
            obs,reward,terminated,truncated,info=self.env.step(np.ones(7))
        self.assertTrue(terminated);self.assertFalse(truncated);self.assertEqual(info['failure_reason'],'numerical')
        self.assertTrue(all(np.isfinite(v).all() for v in obs.values()))


@unittest.skipUnless(SOURCE.exists() and REFERENCE.exists(),'Local source and reference required')
class PolicyTests(unittest.TestCase):
    def test_calibration_goal_input_and_checkpoint_roundtrip(self):
        torch.set_num_threads(1)
        env=QuarterTurnEnv(REFERENCE);obs,_=env.reset(seed=0,options={'reference':True,'axis':'z'})
        vector=DummyVecEnv([lambda:QuarterTurnEnv(REFERENCE)])
        normalizer=VecNormalize(vector,norm_obs=False,norm_reward=True,clip_reward=10000.)
        try:
            source,_=load_source(SOURCE);model=new_model(normalizer);provenance=initialize_prior(model,source,obs)
            action=model.predict(obs,deterministic=True)[0]
            np.testing.assert_allclose(action,np.zeros(7),atol=2e-5)
            self.assertEqual(model.policy.mlp_extractor.policy_net[0].in_features,45)
            self.assertFalse(model.policy.optimizer.state)
            with tempfile.TemporaryDirectory() as tmp:
                checkpoint=save(model,normalizer,REFERENCE,tmp,'test',provenance)
                loaded,metadata=load(checkpoint)
                np.testing.assert_array_equal(action,loaded.predict(obs,deterministic=True)[0])
                self.assertEqual(metadata['goal_degrees'],90.)
        finally:normalizer.close();env.close()


if __name__=='__main__':unittest.main()
