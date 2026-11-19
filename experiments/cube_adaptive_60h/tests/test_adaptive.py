from pathlib import Path
from datetime import datetime,timezone,timedelta
import tempfile,unittest
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from experiments.cube_contact_ppo.task import ContactMonitor
from experiments.cube_contact_ppo.storage import load as old_load
from experiments.cube_adaptive_60h.config import Profile,Config,validate_profile
from experiments.cube_adaptive_60h.env import AdaptiveEnv,AdaptiveMonitor
from experiments.cube_adaptive_60h.policy import Policy,RetainPPO,transfer
from experiments.cube_adaptive_60h.storage import save,load
from experiments.cube_adaptive_60h.decisions import choose,next_review,final_pass

ROOT=Path(__file__).resolve().parents[3]
REFERENCE=ROOT/'experiments/cube_envelop/artifacts/prepared_envelope_v1/reference.npz'
SOURCE=ROOT/'experiments/cube_contact_ppo/artifacts/contact_ppo_v1/y/best_30'


def contacts(f=None):
    f=np.array([1.]*5 if f is None else f)
    return dict(finger_forces=f,hand_force=float(sum(f)),max_penetration=0.,forbidden_support=False,tip_distances=np.zeros(5))


class Rules(unittest.TestCase):
    def test_hybrid90_data_require_exact_saved_command_validation(self):
        import json
        from experiments.cube_adaptive_60h.retention import read
        from experiments.cube_adaptive_60h.storage import digest,implementation_digest
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'unverified.npz'
            np.savez_compressed(path,observations=np.zeros((1,143),dtype=np.float32),actions=np.zeros((1,7),dtype=np.float32))
            path.with_suffix('.json').write_text(json.dumps(dict(implementation_sha256=implementation_digest(),sha256=digest(path),
                goal_degrees=90.,neural_policy_from_initial_grasp=False,checks=[dict(success=True,stable_seconds=1.)])))
            with self.assertRaisesRegex(ValueError,'exactly saved'):read(path)
    def test_temporal_horizons_are_explicit_and_bounded(self):
        from experiments.cube_adaptive_60h.training_parameters import temporal_parameters
        self.assertEqual(temporal_parameters({}),dict(rollout_steps=128,gae_lambda=.95))
        self.assertEqual(temporal_parameters(dict(rollout_steps=256,gae_lambda=.98)),dict(rollout_steps=256,gae_lambda=.98))
        with self.assertRaises(ValueError):temporal_parameters(dict(rollout_steps=0))
        with self.assertRaises(ValueError):temporal_parameters(dict(gae_lambda=1.1))
    def test_rotation_summary_separates_failed_peak_and_held_endpoint(self):
        from experiments.cube_adaptive_60h.rotation_report import summarize
        failed=dict(seed=1,dt=.002,success=False,final=dict(max_progress_degrees=90.,signed_progress_degrees=89.,success=False,max_stable_seconds=.9,failure_reason='lost_hand_contact'))
        good=dict(seed=2,dt=.001,success=True,final=dict(max_progress_degrees=33.,signed_progress_degrees=32.,success=True,max_stable_seconds=1.,failure_reason=''))
        report=dict(target_degrees=30.,success_rate=.5,complete=True,groups={},records=[failed,good])
        summary=summarize(report,'saved_validation.json')
        self.assertEqual(summary['peak']['degrees'],90.)
        self.assertFalse(summary['peak']['success'])
        self.assertEqual(summary['qualified_endpoint']['degrees'],32.)
        self.assertEqual(summary['qualified_endpoint']['target_degrees'],30.)
    def test_rotation_summary_includes_final_probe_without_calling_it_curriculum(self):
        import json
        from experiments.cube_adaptive_60h.rotation_report import RotationEvidence
        final=dict(max_progress_degrees=88.,signed_progress_degrees=86.,success=False,max_stable_seconds=0.,failure_reason='lost_hand_contact')
        report=dict(axis='x',target_degrees=90.,success_rate=0.,complete=True,groups={},records=[dict(seed=1,dt=.002,success=False,final=final)])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'reviews';path.mkdir()
            (path/'001_review.json').write_text(json.dumps(dict(axes=dict(x=dict(final90=report)))))
            result=RotationEvidence(tmp).write()['axes']['x']
            self.assertEqual(result['observed_peak']['degrees'],88.)
            self.assertIsNone(result['qualified_endpoint'])
            self.assertIsNone(result['latest_completed_validation'])
    def test_schedule_stays_anchored_after_restart(self):
        start=datetime(2026,11,13,7,36,27,tzinfo=timezone.utc)
        self.assertEqual(next_review(start.isoformat(),start+timedelta(hours=3,minutes=25)),start+timedelta(hours=6))
        self.assertEqual(next_review(start.isoformat(),start+timedelta(hours=10)),start+timedelta(hours=12))
    def test_regression_rolls_back_without_lowering_goal(self):
        p=Profile();c=dict(complete=True,target_degrees=30.,success_rate=0.,mean_best_error=7.,five_finger_duty=.1)
        b={**c,'success_rate':.65}
        d=choose(p,c,b,[],30.);self.assertEqual(d['action'],'rollback_best')
        self.assertEqual(d['target'],30.);self.assertLess(d['profile']['learning_rate'],p.learning_rate)
    def test_plateau_is_bounded_and_does_not_confuse_different_stages(self):
        p=Profile();r=dict(complete=True,target_degrees=10.,success_rate=0.,mean_best_error=8.,five_finger_duty=.1)
        self.assertEqual(choose(p,r,None,[{**r,'target_degrees':0.}]*3,10.)['action'],'continue')
        d=choose(p,r,None,[r]*3,10.)
        self.assertEqual(d['action'],'new_profile');self.assertEqual(d['target'],90.)
        validate_profile(Profile(**d['profile']))
    def test_final_requires_both_integrators_100_of_100_from_initial_state(self):
        group=dict(complete=True,requested_trials=100,successes=100)
        report=dict(complete=True,target_degrees=90.,from_original_initial_grasp=True,groups={'0.001':group.copy(),'0.002':group.copy()})
        self.assertTrue(final_pass(report))
        report['groups']['0.001']['successes']=99;self.assertFalse(final_pass(report))
        report['groups']['0.001']['successes']=100;report['target_degrees']=30.;self.assertFalse(final_pass(report))
    def test_unsafe_profile_rejected(self):
        from dataclasses import replace
        with self.assertRaises(ValueError):validate_profile(replace(Profile(),action_fraction=1.))


class ContactChecks(unittest.TestCase):
    def test_auxiliary_prefix_is_real_replay_with_identical_state(self):
        from experiments.cube_adaptive_60h.prefix_curriculum import ReachablePrefixCurriculum
        actions=np.zeros((6,7))
        plain=AdaptiveEnv(REFERENCE,'x',target_degrees=10.)
        wrapped=ReachablePrefixCurriculum(AdaptiveEnv(REFERENCE,'x',target_degrees=10.),actions,probability=1.)
        try:
            plain.reset(seed=0)
            for action in actions:a=plain.step(action)
            obs,info=wrapped.reset(seed=0)
            np.testing.assert_array_equal(plain.data.qpos,wrapped.unwrapped.data.qpos)
            np.testing.assert_array_equal(a[0],obs)
            self.assertTrue(info['training_prefix_used'])
            self.assertFalse(info['neural_controls_from_initial_grasp'])
            self.assertEqual(info['training_prefix_controls'],6)
            self.assertGreater(info['elapsed_seconds'],.25)
        finally:plain.close();wrapped.close()
    def test_mount_warning_excludes_visual_and_preserves_physics(self):
        from unittest.mock import patch
        from experiments.cube_adaptive_60h.shaping import MountMarginReward,colliding_forbidden_geoms
        plain=AdaptiveEnv(REFERENCE,'x',target_degrees=10.)
        wrapped=MountMarginReward(AdaptiveEnv(REFERENCE,'x',target_degrees=10.))
        try:
            plain.reset(seed=0);wrapped.reset(seed=0)
            names=[plain.model.geom(int(g)).name for g in colliding_forbidden_geoms(plain)]
            self.assertIn('tetheria_mount_collision_0',names)
            self.assertNotIn('tetheria_mount_visual',names)
            self.assertGreater(wrapped.clearance(),.005)
            # Both run the same real simulation; only the read-only distance
            # warning is mocked to exercise a near-support reward safely.
            reduction=0.
            for _ in range(10):
                a=plain.step(np.zeros(7))
                with patch('experiments.cube_adaptive_60h.shaping.mujoco.mj_geomDistance',return_value=.001):
                    b=wrapped.step(np.zeros(7))
                np.testing.assert_array_equal(plain.data.qpos,wrapped.unwrapped.data.qpos)
                np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[2:4],b[2:4])
                reduction+=b[1]-a[1]
            self.assertLess(reduction,0.)
            self.assertGreater(wrapped.total_penalty,0.)
        finally:plain.close();wrapped.close()
    def test_small_angle_shaping_does_not_change_physics_or_termination(self):
        from experiments.cube_adaptive_60h.shaping import NarrowGoalReward
        plain=AdaptiveEnv(REFERENCE,'x',target_degrees=10.)
        wrapped=NarrowGoalReward(AdaptiveEnv(REFERENCE,'x',target_degrees=10.))
        try:
            plain.reset(seed=0);wrapped.reset(seed=0);reduction=0.
            for _ in range(20):
                a=plain.step(np.zeros(7));b=wrapped.step(np.zeros(7))
                np.testing.assert_array_equal(plain.data.qpos,wrapped.env.data.qpos)
                np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[2],b[2])
                reduction+=b[1]-a[1]
                if a[2]:break
            self.assertLess(reduction,0.)
        finally:plain.close();wrapped.close()
    def test_missing_penalty_operates_before_capture_region(self):
        m=AdaptiveMonitor(np.array([1.,0.,0.,0.]),'x',np.eye(3),Config(),30.,profile=Profile())
        for _ in range(60):m.checkpoint(m.initial,contacts([1,1,0,1,1]),np.zeros(3),np.zeros(3),0.)
        self.assertEqual(m.phase,'drive');self.assertLess(m.recovery_penalty,0.)
        self.assertGreater(m.missing_seconds[2],.5);self.assertEqual(m.stable,0.)
    def test_three_finger_loophole_stays_closed_at_goal(self):
        m=AdaptiveMonitor(np.array([1.,0.,0.,0.]),'x',np.eye(3),Config(),90.,profile=Profile());m.track(m.goal)
        for _ in range(110):m.checkpoint(m.goal,contacts([1,1,0,0,1]),np.zeros(3),np.zeros(3),0.)
        self.assertFalse(m.success);self.assertEqual(m.max_stable,0.)
    def test_native_real_five_finger_hold_both_steps(self):
        for dt in (.001,.002):
            e=AdaptiveEnv(REFERENCE,'x',physical_dt=dt,target_degrees=0.)
            try:
                o,_=e.reset(seed=0,options={'reference':True});self.assertEqual(o.shape,(143,))
                for _ in range(30):
                    o,_,done,_,i=e.step(np.zeros(7))
                    if done:break
                self.assertTrue(i['success']);self.assertTrue(i['five_finger_contact'])
                self.assertTrue(np.isfinite(o).all())
            finally:e.close()


class PolicyChecks(unittest.TestCase):
    def test_zero_padding_preserves_actor_and_checkpoint_roundtrip(self):
        torch.set_num_threads(1)
        source,meta=old_load(SOURCE)
        e=AdaptiveEnv(REFERENCE,'y',target_degrees=30.)
        vector=VecNormalize(DummyVecEnv([lambda:AdaptiveEnv(REFERENCE,'y',target_degrees=30.)]),norm_obs=False)
        try:
            o,_=e.reset(seed=1000001)
            m=RetainPPO(Policy,vector,n_steps=16,batch_size=16,n_epochs=1,device='cpu')
            transfer(m,source)
            np.testing.assert_allclose(m.predict(o,deterministic=True)[0],source.predict(o[:138],deterministic=True)[0],atol=2e-6)
            with tempfile.TemporaryDirectory() as tmp:
                path=save(m,vector,REFERENCE,tmp,'y',Profile(),{'stage':30.,'deadline_at':'fixed'},'test')
                restored,_=load(path)
                np.testing.assert_array_equal(m.predict(o,deterministic=True)[0],restored.predict(o,deterministic=True)[0])
            m.learn(total_timesteps=32);self.assertGreater(m._n_updates,0)
            self.assertTrue(all(torch.isfinite(p).all() for p in m.policy.parameters()))
        finally:e.close();vector.close()


if __name__=='__main__':unittest.main()
