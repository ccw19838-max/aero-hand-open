from pathlib import Path
import unittest
import numpy as np
import mujoco,torch
from experiments.cube_adaptive_60h.config import Profile
from experiments.cube_adaptive_60h.storage import load as load_legacy
from experiments.cube_contact_recovery_v2.env import NativeEnv,FastEnv,TIP_ORDER
from experiments.cube_contact_recovery_v2.reward import training_reward
from experiments.cube_contact_recovery_v2.evaluation import evaluate,final_pass

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'experiments/cube_adaptive_60h/artifacts/x_focus_20261116_155308/sources/x'
INPUTS=ROOT/'experiments/cube_contact_recovery_v2/artifacts/recovery_20261117_v2/inputs'


class Semantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);cls.policy,cls.metadata=load_legacy(SOURCE)
    def make(self,kind=NativeEnv,dt=.002,target=90.,bank=False):
        return kind(INPUTS/'reference.npz','x',physical_dt=dt,target_degrees=target,
            profile=Profile(**self.metadata['profile']),prefix_bank=INPUTS/'prefix_bank.npz' if bank else None)
    def test_motor_driven_thumb_root_is_part_of_thumb_and_total_hand_force(self):
        env=self.make();env.reset(seed=1000006)
        try:
            g=env.model.geom('th_mp_collision').id;self.assertEqual(env.ids['geom_finger'][g],0)
            found=False;obs=env.last_observation
            for _ in range(50):
                obs,_,done,_,info=env.step(self.policy.predict(obs,deterministic=True)[0])
                total=np.zeros(5);root_force=0.;palm=0.;w=np.zeros(6)
                # Independent physical oracle: classify via actuator-bearing
                # ancestor names, including right_t_link, not a text prefix.
                for i,c in enumerate(env.data.contact):
                    a,b=int(c.geom1),int(c.geom2);other=b if a==env.ids['cube_geom'] else a if b==env.ids['cube_geom'] else None
                    if other is None:continue
                    mujoco.mj_contactForce(env.model,env.data,i,w);f=max(0.,float(w[0]));body=int(env.model.geom_bodyid[other])
                    name=env.model.body(body).name
                    if name=='right_t_link':root_force+=f
                    if name=='palm':palm+=f
                    while body>0:
                        name=env.model.body(body).name
                        roots=['right_t_link','right_index_f_link','right_middle_f_link','right_ring_f_link','right_pinky_f_link']
                        if name in roots:total[roots.index(name)]+=f;break
                        body=int(env.model.body_parentid[body])
                np.testing.assert_allclose(info['finger_forces'],total,atol=1e-12)
                self.assertAlmostEqual(info['hand_force'],sum(total)+palm,places=10)
                if root_force>.2:found=True
                if done:break
            self.assertTrue(found)
        finally:env.close()
    def test_each_distance_matches_its_named_finger_not_legacy_feature_order(self):
        env=self.make()
        try:
            env.reset(seed=0)
            cube=env.data.xpos[env.ids['cube_body']];rotation=env.data.xmat[env.ids['cube_body']].reshape(3,3)
            for i,name in enumerate(TIP_ORDER):
                point=rotation.T@(env.data.site_xpos[env.model.site(name).id]-cube)
                d=np.abs(point)-env.config.cube_size/2
                expected=np.linalg.norm(np.maximum(d,0.))+min(max(d),0.)
                self.assertAlmostEqual(env.tip_distances[i],expected,places=12)
            self.assertEqual([env.model.site(int(s)).name for s in env.tip_sites],['if_tip','mf_tip','rf_tip','pf_tip','th_tip'])
        finally:env.close()
    def test_native_fast_full_state_observation_reward_and_termination(self):
        for dt in (.001,.002):
            a=training_reward(self.make(dt=dt,target=0.,bank=True));b=training_reward(self.make(FastEnv,dt=dt,target=0.,bank=True))
            try:
                x,info=a.reset(seed=5400001);y,other=b.reset(seed=5400001);np.testing.assert_array_equal(x,y);self.assertEqual(info,other)
                for _ in range(250):
                    command=self.policy.predict(x,deterministic=True)[0];left=a.step(command);right=b.step(command);x=left[0]
                    for key in ('qpos','qvel','ctrl'):np.testing.assert_array_equal(getattr(a.unwrapped.data,key),getattr(b.unwrapped.data,key))
                    np.testing.assert_array_equal(left[0],right[0]);self.assertEqual(left[1:],right[1:])
                    if left[2]:break
                self.assertTrue(left[2])
            finally:a.close();b.close()
    def test_auxiliary_prefix_is_real_motion_and_formal_goal_starts_original(self):
        aux=self.make(target=0.,bank=True);formal=self.make(target=90.,bank=True)
        try:
            _,info=aux.reset(seed=5400000);_,finfo=formal.reset(seed=5400000)
            self.assertTrue(info['training_prefix_used']);self.assertFalse(info['from_original_initial_grasp'])
            self.assertAlmostEqual(aux.data.time,.8,places=10);self.assertEqual(formal.data.time,0.)
            self.assertFalse(finfo['training_prefix_used']);self.assertTrue(finfo['from_original_initial_grasp'])
            self.assertGreater(info['prefix_start_angle_degrees'],0.)
            self.assertLess(min(info['finger_forces']),.1)
            self.assertAlmostEqual(aux.monitor.error,0.,places=12)
            np.testing.assert_array_equal(aux.last_action,aux.bank[aux.prefix_index][-1])
        finally:aux.close();formal.close()
    def test_auxiliary_success_cannot_become_final_x90_verification(self):
        report=evaluate(self.policy,INPUTS,Profile(),0.,episodes=1,seed=5400000)
        self.assertFalse(report['from_original_initial_grasp']);self.assertFalse(final_pass(report))
        self.assertEqual(report['evaluation_kind'],'reachable_contact_recovery')


if __name__=='__main__':unittest.main()
