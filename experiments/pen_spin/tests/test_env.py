"""Native adapter tests use an explicitly unapproved diagnostic reset only."""
from dataclasses import replace
from pathlib import Path
import os
import tempfile
import unittest

import jax
import numpy as np
import mujoco

from experiments.pen_spin.config import PenSpinConfig
from experiments.pen_spin.env import make_env
from experiments.pen_spin.model import build_model, thumb_frame


class EnvironmentContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.reference = Path(cls.directory.name)/'diagnostic-not-approved.npz'
        cls.config = replace(PenSpinConfig(),reset_position_noise=0.,reset_angle_noise_degrees=0.)
        model = build_model(cls.config)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model,data,0)
        mujoco.mj_forward(model,data)
        c0,n0 = thumb_frame(model,data)
        np.savez(cls.reference,qpos=data.qpos,qvel=data.qvel,ctrl=data.ctrl,c0=c0,n0=n0)
        cls.env = make_env(cls.config,cls.reference,backend='native')

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_history_and_bounded_increment(self):
        env = self.env
        state = env.reset(jax.random.PRNGKey(7))
        self.assertEqual(state.obs['state'].shape,(54,))
        self.assertEqual(state.obs['privileged_state'].shape,(93,))
        np.testing.assert_array_equal(state.obs['state'][14:21],0.)
        previous = np.asarray(state.info['motor_targets']).copy()
        action = np.array([2.,-2.,1.,-1.,1.,-1.,.5],dtype=np.float32)
        next_state = env.step(state,action)
        clipped = np.clip(action,-1,1)
        expected = np.clip(previous+.05*np.asarray(env._ctrl_high-env._ctrl_low)*clipped,
                           env._ctrl_low,env._ctrl_high)
        np.testing.assert_allclose(next_state.data.ctrl,expected,atol=1e-7)
        np.testing.assert_array_equal(next_state.obs['state'][14:21],clipped)
        self.assertTrue(np.isfinite(next_state.obs['state']).all())
        self.assertEqual(next_state.info['substep_trace'].shape,(25,24))
        np.testing.assert_allclose(next_state.info['substep_trace'][:,0],
                                   np.arange(1,26)*self.config.sim_dt,atol=1e-7)

    def test_seeded_reset_and_shared_observation(self):
        env = self.env
        first = env.reset(jax.random.PRNGKey(11))
        second = env.reset(jax.random.PRNGKey(11))
        np.testing.assert_array_equal(first.obs['state'],second.obs['state'])
        sample = env._sample(first.data)
        reconstructed = env.observe(first.data,sample,first.info['trace'],
                                    first.info['last_action'],np.zeros(7))
        for name in reconstructed:
            np.testing.assert_array_equal(first.obs[name],reconstructed[name])
        self.assertIn('stable_seconds',first.metrics)

    @unittest.skipUnless(os.environ.get('PEN_SPIN_MJX_TESTS')=='1',
                         'Set PEN_SPIN_MJX_TESTS=1 for full CPU MJX compilation')
    def test_mjx_four_environment_step_and_native_reset_observation(self):
        env = make_env(self.config,self.reference,backend='mjx')
        key = jax.random.PRNGKey(11)
        native = self.env.reset(key)
        state = jax.jit(env.reset)(key)
        np.testing.assert_allclose(state.obs['state'],native.obs['state'],rtol=1e-5,atol=1e-5)
        keys = jax.random.split(key,4)
        batch = jax.jit(jax.vmap(env.reset))(keys)
        batch = jax.jit(jax.vmap(env.step))(batch,np.zeros((4,7),np.float32))
        batch.reward.block_until_ready()
        self.assertEqual(batch.obs['state'].shape,(4,54))
        self.assertEqual(batch.info['substep_trace'].shape,(4,25,24))
        self.assertTrue(np.isfinite(batch.obs['state']).all())
        self.assertTrue(np.isfinite(batch.reward).all())


if __name__ == '__main__':
    unittest.main()
