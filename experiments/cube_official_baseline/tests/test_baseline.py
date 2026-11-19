import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')

from pathlib import Path
import tempfile
import unittest
import numpy as np
import mujoco
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from experiments.cube_official_baseline.env import OfficialRotateEnv,SCALE
from experiments.cube_official_baseline.policy import initialize_source
from experiments.cube_official_baseline.train import new_model,save,load,evaluate

SOURCE=Path(__file__).resolve().parents[1]/'artifacts/source/author_actor_critic.npz'


class EnvironmentTests(unittest.TestCase):
    def setUp(self):self.env=OfficialRotateEnv(noise_level=0.);self.env.reset(seed=0,options={'home':True})
    def tearDown(self):self.env.close()

    def test_exact_original_object_mount_and_steps(self):
        m=self.env.model
        np.testing.assert_allclose(m.geom('cube').size,[.025]*3)
        self.assertAlmostEqual(m.body_mass[m.body('cube').id],.0692)
        np.testing.assert_array_equal(m.body('tetheria_mount').quat,[1,0,0,0])
        self.assertAlmostEqual(m.opt.timestep,.01)
        self.env.step(np.zeros(7));self.assertAlmostEqual(self.env.data.time,.05)

    def test_actions_are_absolute_offsets_not_accumulated_increments(self):
        action=np.full(7,.1)
        self.env.step(action);first=self.env.data.ctrl.copy()
        self.env.step(action)
        np.testing.assert_allclose(first,self.env.default+SCALE*action)
        np.testing.assert_array_equal(first,self.env.data.ctrl)

    def test_legacy_two_period_history_and_asymmetric_observations(self):
        first=np.full(7,.1);second=np.full(7,-.1)
        a,*_=self.env.step(first);b,*_=self.env.step(second)
        np.testing.assert_array_equal(a['state'][7:],np.zeros(7))
        np.testing.assert_allclose(b['state'][7:],first)
        self.assertEqual(b['state'].shape,(14,));self.assertEqual(b['privileged_state'].shape,(81,))
        np.testing.assert_array_equal(b['privileged_state'][:14],b['state'])

    def test_official_reward_formula(self):
        action=np.full(7,.1);obs,reward,terminated,truncated,info=self.env.step(action)
        expected=.05*(info['world_z_angular_velocity']-7*.1**2-100.*info['height_termination'])
        self.assertAlmostEqual(reward,expected)

    def test_free_flight_is_not_terminated_by_a_five_finger_gate(self):
        self.env.data.qpos[18]=.3;mujoco.mj_forward(self.env.model,self.env.data)
        obs,reward,terminated,truncated,info=self.env.step(np.zeros(7))
        self.assertFalse(terminated or truncated)
        self.assertFalse(info['task_success_evaluated'])

    def test_height_failure(self):
        self.env.data.qpos[18]=-.08;mujoco.mj_forward(self.env.model,self.env.data)
        obs,reward,terminated,truncated,info=self.env.step(np.zeros(7))
        self.assertTrue(terminated);self.assertFalse(truncated);self.assertEqual(info['failure_reason'],'height')

    def test_random_reset_reproducibility(self):
        a,_=self.env.reset(seed=123);q=self.env.data.qpos.copy()
        b,_=self.env.reset(seed=123)
        np.testing.assert_array_equal(q,self.env.data.qpos)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        self.assertAlmostEqual(np.linalg.norm(q[19:23]),1.)


@unittest.skipUnless(SOURCE.exists(),'Exported source checkpoint required')
class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.vector=DummyVecEnv([OfficialRotateEnv])
        self.model=new_model(self.vector);initialize_source(self.model,SOURCE)
    def tearDown(self):self.vector.close()

    def test_actions_and_values_match_brax_on_64_observations(self):
        with np.load(SOURCE,allow_pickle=False) as data:
            obs={'state':data['test_state'],'privileged_state':data['test_privileged_state']}
            action=self.model.predict(obs,deterministic=True)[0]
            tensors=self.model.policy.obs_to_tensor(obs)[0]
            with torch.no_grad():value=self.model.policy.predict_values(tensors).numpy().reshape(-1)
            np.testing.assert_allclose(action,data['test_actions'],atol=1e-5,rtol=1e-5)
            np.testing.assert_allclose(value,data['test_values'].reshape(-1),atol=1e-5,rtol=1e-5)

    def test_actor_does_not_read_privileged_cube_truth(self):
        obs=self.vector.reset();before=self.model.predict(obs,deterministic=True)[0]
        obs['privileged_state']+=100.
        after=self.model.predict(obs,deterministic=True)[0]
        np.testing.assert_array_equal(before,after)

    def test_checkpoint_inference_roundtrip(self):
        obs=self.vector.reset();before=self.model.predict(obs,deterministic=True)[0]
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=save(self.model,Path(tmp),'test',{'source':'test'})
            loaded,meta=load(checkpoint)
            np.testing.assert_array_equal(before,loaded.predict(obs,deterministic=True)[0])
            self.assertEqual(meta['configuration']['actor_inputs'],14)

    def test_height_survival_is_never_called_quarter_turn_success(self):
        report=evaluate(self.model,1,baseline=True)
        self.assertTrue(report['complete'])
        self.assertFalse(report['task_success_evaluated'])
        self.assertNotIn('success_rate',report)


if __name__=='__main__':unittest.main()
