from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch

import numpy as np

from experiments.cube_envelop_gradual.config import CubeReorientConfig,ANGLES,tolerance_for
from experiments.cube_envelop_gradual.env import ReorientMonitor,target_quaternion,physical_substep_reward
from experiments.cube_envelop_gradual.train import AxisPromotionMonitor
from experiments.cube_envelop_gradual.curriculum import Runner,advance_targets


class SmallAngleTests(unittest.TestCase):
    def test_tolerances_exclude_initial_pose_at_every_level(self):
        for angle in ANGLES:
            c=CubeReorientConfig(target_degrees=angle,angle_tolerance_degrees=tolerance_for(angle))
            self.assertLess(c.angle_tolerance_degrees,angle)
        with self.assertRaises(ValueError):CubeReorientConfig(target_degrees=5,angle_tolerance_degrees=5)

    def monitor(self):
        c=CubeReorientConfig();m=ReorientMonitor(c,np.array([1.,0,0,0]),'z')
        contact={'finger_forces':np.ones(5),'max_penetration':0.,'forbidden_support':False}
        for _ in range(250):m.update(np.array([1.,0,0,0]),contact,np.zeros(3),np.zeros(3),0.,.2)
        self.assertTrue(m.capture_succeeded)
        return c,m,contact

    def test_static_does_not_pass_five_degrees(self):
        c,m,contact=self.monitor()
        for _ in range(500):m.update(np.array([1.,0,0,0]),contact,np.zeros(3),np.zeros(3),0.,.2)
        self.assertFalse(m.success);self.assertAlmostEqual(np.rad2deg(m.angle_error_radians),5.)

    def test_actual_five_degree_turn_and_five_contact_recapture(self):
        c,m,contact=self.monitor();goal=target_quaternion(np.array([1.,0,0,0]),'z',5.)
        for _ in range(250):m.update(goal,contact,np.zeros(3),np.zeros(3),0.,.2)
        self.assertTrue(m.success);self.assertAlmostEqual(np.rad2deg(m.signed_progress_radians),5.)

    def test_new_near_goal_bonus_is_not_free_at_small_initial_error(self):
        c,m,contact=self.monitor()
        reward=physical_substep_reward(c,m,contact,np.zeros(3),np.zeros(3),False,True,0.)
        self.assertLess(reward,.001) # Old 30-degree radius would yield >.001 here.
        none=dict(contact,finger_forces=np.zeros(5))
        self.assertLess(physical_substep_reward(c,m,none,np.zeros(3),np.zeros(3),False,True,.5),0.)


class PromotionTests(unittest.TestCase):
    def report(self,x,y):
        return {'groups':{'x':{'complete':True,'success_rate':x},'y':{'complete':True,'success_rate':y}}}

    def test_x_can_advance_without_y(self):
        m=AxisPromotionMonitor(CubeReorientConfig(training_axes=('x','y')))
        self.assertFalse(m.update(self.report(.95,0.)))
        self.assertTrue(m.update(self.report(.95,0.)));self.assertEqual(m.ready_axes,['x'])
        self.assertEqual(advance_targets((5.,5.,5.),m.ready_axes),(10.,5.,5.))

    def test_final_requires_both_axes(self):
        m=AxisPromotionMonitor(CubeReorientConfig(training_axes=('x','y'),target_angles=(90,90,90)))
        m.update(self.report(1.,.8));self.assertFalse(m.update(self.report(1.,.8)))
        m.update(self.report(1.,1.));self.assertTrue(m.update(self.report(1.,1.)))
        self.assertTrue(m.final_pass);self.assertEqual(m.ready_axes,[])

    def test_sixteen_hour_budget_and_two_hour_z_cap(self):
        fixed=datetime(2026,11,11,10,tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp,patch('experiments.cube_envelop_gradual.curriculum.now',return_value=fixed),patch('experiments.cube_envelop_gradual.curriculum.time.time',return_value=fixed.timestamp()):
            runner=Runner(Path(tmp)/'reference',Path(tmp)/'checkpoint',Path(tmp)/'run')
            self.assertEqual(runner.deadline-fixed.timestamp(),16*3600)
            self.assertEqual(runner.z_deadline-fixed.timestamp(),2*3600)
            runner.switch_to_xy(Path(tmp)/'checkpoint')
            self.assertEqual(runner.state['training_axes'],['x','y'])
            with self.assertRaises(ValueError):Runner('ref','ckpt',Path(tmp)/'bad',hours=25)

    def test_z_timeout_hands_off_to_xy_and_x_advances_independently(self):
        fixed=datetime(2026,11,11,10,tzinfo=timezone.utc);clock=[fixed.timestamp()]
        with tempfile.TemporaryDirectory() as tmp,patch('experiments.cube_envelop_gradual.curriculum.now',return_value=fixed),patch('experiments.cube_envelop_gradual.curriculum.time.time',side_effect=lambda:clock[0]):
            runner=Runner(Path(tmp)/'reference',Path(tmp)/'checkpoint',Path(tmp)/'run')
            seen=[]
            def child(name,command,deadline,training=True):
                folder=runner.output/name
                config=json.loads((folder/'stage_config.json').read_text());seen.append(config)
                data={'training_steps':100,'latest_checkpoint':str(folder/'checkpoint'),
                      'stop_reason':'wall_clock_budget','latest_validation':{}}
                if len(seen)==1:
                    clock[0]=runner.z_deadline+1 # Z has not passed even 5 degrees.
                elif len(seen)==2:
                    data.update(stop_reason='angle_stage_ready',promote_axes=['x'],
                                latest_validation={'consecutive_axis_passes':{'x':2,'y':0}})
                (folder/'status.json').write_text(json.dumps(data))
                return 1 if len(seen)==3 else 0
            runner.run_child=child
            runner.run()
            self.assertEqual(seen[0]['training_axes'],['z'])
            self.assertEqual(seen[1]['training_axes'],['x','y'])
            self.assertEqual(seen[2]['target_angles'],[10.,5.,5.])
            self.assertEqual(runner.state['passed_angles']['z'],0.)
            self.assertEqual(runner.state['passed_angles']['x'],5.)


if __name__=='__main__':unittest.main()
