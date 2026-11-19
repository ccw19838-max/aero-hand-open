"""Real two-process simulation, bounded stopping and resumable search state."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
from experiments.cube_quarter_turn_xy.planner import (worker_init,bounded_map,
    initialize_distributions,save_state,restore_state)
from experiments.cube_quarter_turn_v2.env import TeacherEnv
from experiments.cube_quarter_turn_v2.planner import sequence,rollout,summary

ROOT=Path(__file__).resolve().parents[3]
REFERENCE=ROOT/'experiments/cube_envelop/artifacts/prepared_envelope_v1/reference.npz'


class ParallelTests(unittest.TestCase):
    def test_two_environment_ppo_rollout_and_reward_normalizer(self):
        import torch
        from stable_baselines3.common.vec_env import SubprocVecEnv,VecNormalize
        from experiments.cube_quarter_turn_v2.train import new_model
        torch.set_num_threads(1)
        def factory():return TeacherEnv(REFERENCE,'y')
        vector=SubprocVecEnv([factory,factory],start_method='spawn')
        normalizer=VecNormalize(vector,norm_obs=False,norm_reward=True)
        try:
            self.assertEqual(len({p.pid for p in vector.processes}),2)
            self.assertEqual(normalizer.reset().shape,(2,133))
            model=new_model(normalizer,seed=12)
            model.learn(total_timesteps=256)
            self.assertEqual(model.num_timesteps,256)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.policy.parameters()))
            self.assertGreater(float(normalizer.ret_rms.count),256.)
        finally:normalizer.close()
    def test_two_worker_results_match_single_physics_and_stop_bounds_queue(self):
        context=mp.get_context('spawn');stop=context.Event()
        parameters=np.r_[np.zeros(35),[7,8,8,10,27]]
        deadline=time.time()+45.
        env=TeacherEnv(REFERENCE,'x')
        try:expected=summary(rollout(env,sequence(parameters),deadline=deadline))
        finally:env.close()
        with ProcessPoolExecutor(max_workers=2,mp_context=context,initializer=worker_init,
                initargs=([str(REFERENCE)],'x',stop)) as pool:
            results=list(bounded_map(pool,[(0,parameters,deadline)]*6,stop,deadline))
            self.assertEqual(len(results),6)
            self.assertEqual(len({r['simulator_pid'] for _,r in results}),2)
            for _,r in results:
                self.assertAlmostEqual(r['score'],expected['score'],places=10)
                self.assertEqual(r['final']['failure_reason'],expected['final']['failure_reason'])
            stopped=list(bounded_map(pool,[(0,parameters,deadline)]*20,stop,deadline,should_stop=lambda:True))
            self.assertLessEqual(len(stopped),2);self.assertTrue(stop.is_set())
    def test_state_restores_distribution_and_next_rng_draw_and_rejects_deadline_change(self):
        means=np.zeros((3,40));scales=np.ones((3,40));rng=np.random.default_rng(123)
        manifest=dict(axis='x',workers=2,deadline_epoch=1234567.,references={'a':'hash'})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.npz'
            save_state(path,means,scales,rng,17,800,None,None,None,[],manifest,stalls=12)
            restored_mean,restored_scale,restored_rng,metadata,p=restore_state(path,manifest)
            np.testing.assert_array_equal(restored_mean,means);np.testing.assert_array_equal(restored_scale,scales)
            np.testing.assert_array_equal(rng.normal(size=128),restored_rng.normal(size=128))
            self.assertEqual(metadata['generation'],17);self.assertIsNone(p)
            self.assertEqual(metadata['stalls'],12)
            with self.assertRaises(ValueError):restore_state(path,{**manifest,'deadline_epoch':1234568.})


if __name__=='__main__':unittest.main()
