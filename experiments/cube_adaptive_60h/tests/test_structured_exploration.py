from pathlib import Path
from datetime import datetime,timezone,timedelta
import tempfile,unittest
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from experiments.cube_adaptive_60h.env import AdaptiveEnv
from experiments.cube_adaptive_60h.config import Profile
from experiments.cube_adaptive_60h.policy import Policy,RetainPPO,transfer
from experiments.cube_adaptive_60h.storage import load,save
from experiments.cube_adaptive_60h.structured_exploration import settings,initialize,report_std,fingerprint

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'experiments/cube_adaptive_60h/artifacts/goal_20261113_1536/x/segment_008/checkpoints/step_000622592_1817c346'
REFERENCE=SOURCE/'reference.npz'


class StructuredExploration(unittest.TestCase):
    def test_configuration_is_optional_and_bounded(self):
        self.assertEqual(settings({}),dict(use_sde=False,sde_sample_freq=-1))
        self.assertEqual(settings(dict(use_sde=True,sde_sample_freq=8)),dict(use_sde=True,sde_sample_freq=8))
        for bad in (dict(use_sde='true'),dict(use_sde=True,sde_sample_freq=0),dict(use_sde=True,sde_sample_freq=True)):
            with self.assertRaises(ValueError):settings(bad)

    def test_mean_preservation_coherence_ppo_save_reload_and_continued_update(self):
        torch.set_num_threads(1);source,_=load(SOURCE)
        vector=VecNormalize(DummyVecEnv([lambda dt=dt:AdaptiveEnv(REFERENCE,'x',physical_dt=dt,
            target_degrees=90.) for dt in (.001,.002)]),norm_obs=False,norm_reward=True)
        try:
            model=RetainPPO(Policy,vector,use_sde=True,sde_sample_freq=8,n_steps=128,batch_size=128,n_epochs=1,
                seed=4401,device='cpu',verbose=0)
            transfer(model,source);spec=initialize(model,vector,.08,4401)
            self.assertEqual(spec['source_sha256'],fingerprint())
            obs=vector.venv._obs_from_buf().copy()
            np.testing.assert_allclose(model.predict(obs,deterministic=True)[0],source.predict(obs,deterministic=True)[0],rtol=0.,atol=1e-7)
            np.testing.assert_allclose(report_std(model,{'structured_exploration':spec}),np.full(7,.08),rtol=0.,atol=1e-6)
            model.policy.reset_noise(2);first=model.predict(obs,deterministic=False)[0]
            np.testing.assert_array_equal(first,model.predict(obs,deterministic=False)[0])
            model.policy.reset_noise(2);self.assertFalse(np.array_equal(first,model.predict(obs,deterministic=False)[0]))
            model.learn(total_timesteps=256);expected=model.predict(obs,deterministic=True)[0]
            self.assertTrue(np.isfinite(model.policy.log_std.detach().numpy()).all())
            with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
                state=dict(stage=90.,deadline_at=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
                    provenance=dict(structured_exploration=spec))
                checkpoint=save(model,vector,REFERENCE,tmp,'x',Profile(bc_weight=0.),state,'test_only')
                restored,_=load(checkpoint);self.assertTrue(restored.use_sde);self.assertEqual(restored.sde_sample_freq,8)
                np.testing.assert_array_equal(restored.predict(obs,deterministic=True)[0],expected)
                steps=restored.num_timesteps;epochs=restored._n_updates
                restored.set_env(vector);restored.learn(total_timesteps=256,reset_num_timesteps=False)
                self.assertEqual(restored.num_timesteps,steps+256);self.assertGreater(restored._n_updates,epochs)
                self.assertTrue(np.isfinite(restored.policy.log_std.detach().numpy()).all())
        finally:vector.close()


if __name__=='__main__':unittest.main()
