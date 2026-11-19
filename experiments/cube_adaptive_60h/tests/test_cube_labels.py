from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,tempfile,unittest
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from experiments.cube_quarter_turn.task import orientation_error
from experiments.cube_adaptive_60h.env import AdaptiveEnv
from experiments.cube_adaptive_60h.config import Profile
from experiments.cube_adaptive_60h.policy import Policy,transfer
from experiments.cube_adaptive_60h.cube_labels import (initial_quaternion,transform,
    multiply,canonical,SYMMETRY_QUATERNIONS,QUATERNION_BLOCKS)
from experiments.cube_adaptive_60h.canonical_policy import (descriptor,CanonicalObservation,
    CanonicalRetainPPO)
from experiments.cube_adaptive_60h.storage import load_source,save,load

ROOT=Path(__file__).resolve().parents[3]
REFERENCE=ROOT/'experiments/cube_envelop/artifacts/prepared_envelope_v1/reference.npz'
SOURCE=ROOT/'experiments/cube_contact_ppo/artifacts/contact_ppo_v1/y/best_30'


def anchor():
    env=AdaptiveEnv(REFERENCE,'y',target_degrees=90.)
    try:
        obs,_=env.reset(seed=1000013)
        for _ in range(5):obs,_,_,_,_=env.step(np.zeros(7,dtype=np.float32))
        return initial_quaternion(obs)
    finally:env.close()


class CubeLabels(unittest.TestCase):
    def test_all24_labels_preserve_full_goal_error_and_canonical_input(self):
        env=AdaptiveEnv(REFERENCE,'y',target_degrees=90.)
        try:
            obs,_=env.reset(seed=1000000);fixed=anchor();expected=transform(obs,fixed)
            physical_before=env.data.qpos.copy();goal_before=env.monitor.goal.copy()
            error_before=orientation_error(obs[84:88],obs[94:98])[0]
            variants=[]
            for symmetry in SYMMETRY_QUATERNIONS:
                label=obs.copy()
                for start in QUATERNION_BLOCKS:
                    label[start:start+4]=canonical(multiply(obs[start:start+4],symmetry))
                np.testing.assert_allclose(orientation_error(label[84:88],label[94:98])[0],error_before,atol=2e-7)
                variants.append(label)
            actual=transform(np.stack(variants),fixed)
            np.testing.assert_allclose(actual,np.broadcast_to(expected,actual.shape),rtol=0.,atol=2e-6)
            np.testing.assert_array_equal(env.data.qpos,physical_before)
            np.testing.assert_array_equal(env.monitor.goal,goal_before)
        finally:env.close()

    def test_physics_history_reward_and_monitor_match_both_integrators(self):
        spec=descriptor(anchor());rng=np.random.default_rng(91)
        for dt in (.001,.002):
            plain=AdaptiveEnv(REFERENCE,'x',physical_dt=dt,target_degrees=90.)
            wrapped=CanonicalObservation(AdaptiveEnv(REFERENCE,'x',physical_dt=dt,target_degrees=90.),spec)
            try:
                plain.reset(seed=1000000);wrapped.reset(seed=1000000)
                unchanged=np.ones(143,dtype=bool)
                for start in QUATERNION_BLOCKS:unchanged[start:start+4]=False
                for _ in range(35):
                    action=rng.normal(0.,.05,7).astype(np.float32);a=plain.step(action);b=wrapped.step(action)
                    np.testing.assert_array_equal(plain.data.qpos,wrapped.unwrapped.data.qpos)
                    np.testing.assert_array_equal(plain.data.qvel,wrapped.unwrapped.data.qvel)
                    np.testing.assert_array_equal(plain.data.ctrl,wrapped.unwrapped.data.ctrl)
                    np.testing.assert_array_equal(a[0][unchanged],b[0][unchanged])
                    np.testing.assert_array_equal(plain.monitor.goal,wrapped.unwrapped.monitor.goal)
                    self.assertEqual(a[1:],b[1:])
                    if a[2]:break
            finally:plain.close();wrapped.close()

    def test_short_ppo_rollout_reload_and_raw_state_inference_agree(self):
        torch.set_num_threads(1);spec=descriptor(anchor())
        factories=[lambda dt=dt:CanonicalObservation(AdaptiveEnv(REFERENCE,'y',physical_dt=dt,
            target_degrees=90.,training=True),spec) for dt in (.001,.002)]
        vector=VecNormalize(DummyVecEnv(factories),norm_obs=False,norm_reward=True)
        try:
            model=CanonicalRetainPPO(Policy,vector,canonical_spec=spec,n_steps=16,batch_size=16,n_epochs=1,
                seed=991,device='cpu',verbose=0)
            source,_=load_source(SOURCE);transfer(model,source)
            model.learn(total_timesteps=32)
            raw=vector.venv.envs[0].unwrapped.last_observation.copy()
            expected=model.predict(raw,deterministic=True)[0]
            direct=model.policy.predict(transform(raw,spec['anchor_quaternion']),deterministic=True)[0]
            np.testing.assert_array_equal(expected,direct)
            with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
                state=dict(stage=90.,deadline_at=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
                    provenance=dict(observation_transform=spec))
                checkpoint=save(model,vector,REFERENCE,tmp,'y',Profile(),state,'test_only')
                loaded,metadata=load(checkpoint)
                self.assertIsInstance(loaded,CanonicalRetainPPO)
                self.assertEqual(loaded.canonical_spec,spec)
                np.testing.assert_array_equal(loaded.predict(raw,deterministic=True)[0],expected)
                bad=json.loads((checkpoint/'metadata.json').read_text())
                bad['state']['provenance']['observation_transform']['files']['cube_labels.py']='0'*64
                (checkpoint/'metadata.json').write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError,'implementation'):load(checkpoint)
        finally:vector.close()


if __name__=='__main__':unittest.main()
