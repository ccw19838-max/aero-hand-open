"""Exercise actor-only transfer and a real immutable checkpoint round trip."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from experiments.cube_z90.config import CubeReorientConfig
from experiments.cube_z90.env import CubeReorientEnv
from experiments.cube_z90.evaluate import load_checkpoint, policy_action
from experiments.cube_z90.train import initialize_from_checkpoint, save_checkpoint

ROOT=Path(__file__).resolve().parents[3]
REFERENCE=ROOT/'experiments/cube_envelop/artifacts/prepared_envelope_v1/reference.npz'
SOURCE=ROOT/'experiments/cube_envelop_gradual/artifacts/long16h_z_then_xy_v1/stage_000_z5/checkpoints/step_000229376_25a1a93b'


@unittest.skipUnless(SOURCE.exists() and REFERENCE.exists(),'Warm-start artifacts are local')
class CheckpointTests(unittest.TestCase):
    def test_only_actor_transfers_and_saved_inference_matches(self):
        torch.set_num_threads(1)
        config=CubeReorientConfig()
        vector=DummyVecEnv([lambda:CubeReorientEnv(REFERENCE,config)])
        model,normalizer,provenance=initialize_from_checkpoint(SOURCE,REFERENCE,config,vector,123)
        try:
            source=PPO.load(SOURCE/'policy.zip',device='cpu')
            source_state=source.policy.state_dict()
            state=model.policy.state_dict()
            for key in state:
                if key.startswith(('mlp_extractor.policy_net.','action_net.')):
                    torch.testing.assert_close(state[key],source_state[key],rtol=0,atol=0)
            self.assertFalse(torch.equal(state['value_net.weight'],source_state['value_net.weight']))
            self.assertEqual(model.num_timesteps,0)
            self.assertFalse(model.policy.optimizer.state)
            self.assertTrue(normalizer.norm_reward)
            self.assertEqual(normalizer.clip_reward,10000.)
            normalizer.ret_rms.var=np.array(100.)
            # Default clip=10 would turn this 500-point bonus into 10 rather
            # than 50 and could favour lingering over completing the task.
            np.testing.assert_allclose(normalizer.normalize_reward(np.array([500.])),[50.])
            np.testing.assert_allclose(state['log_std'].numpy(),-.5)
            env=CubeReorientEnv(REFERENCE,config)
            try:obs,_=env.reset(seed=20000)
            finally:env.close()
            expected=policy_action(model,normalizer,obs)
            with tempfile.TemporaryDirectory() as tmp:
                checkpoint=save_checkpoint(model,normalizer,REFERENCE,config,tmp,elapsed_seconds=0.,
                                           reason='test',initialization=provenance)
                loaded,frozen,loaded_config,ref,metadata=load_checkpoint(checkpoint)
                try:
                    np.testing.assert_array_equal(policy_action(loaded,frozen,obs),expected)
                    self.assertFalse(frozen.training)
                    self.assertFalse(frozen.norm_reward)
                    self.assertEqual(loaded_config.target_hold_seconds,1.)
                    self.assertEqual(metadata['task'],'cube_z90_aggressive_v1')
                finally:frozen.close()
        finally:normalizer.close()


if __name__=='__main__':unittest.main()
