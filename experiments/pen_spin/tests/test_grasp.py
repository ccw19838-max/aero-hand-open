"""Active acquisition has its own gate and never counts initial proximity as success."""
from dataclasses import replace
import unittest
import jax
import jax.numpy as jp
from experiments.pen_spin.config import PenSpinConfig
from experiments.pen_spin.trajectory import Sample,initialize_trace,SUPPORT,PENETRATION
from experiments.pen_spin.grasp import advance_grasp_trace


class GraspTests(unittest.TestCase):
    def setUp(self):
        self.cfg=replace(PenSpinConfig(),schema_version=2,task_mode='grasp')
        self.sample=Sample(center=jp.zeros(3),endpoint_a=jp.array([.065,0.,0.]),
                           direction=jp.array([1.,0.,0.]),linvel=jp.zeros(3),angvel=jp.zeros(3),
                           thumb_center=jp.zeros(3),finger_forces=jp.array([.1,.1,.1,0.,0.]),
                           invalid_contact=jp.array(False),penetration=jp.array(False),
                           finger_distances=jp.zeros(5))
        self.trace=initialize_trace(self.sample,jp.array([0.,0.,1.]),jp.zeros(3),0.,self.cfg)

    def run_steps(self,sample,count):
        def body(trace,_):return advance_grasp_trace(trace,sample,.002,self.cfg)[0],None
        return jax.jit(lambda t:jax.lax.scan(body,t,None,length=count)[0])(self.trace)

    def test_three_distinct_fingers_hold_for_one_second(self):
        self.assertFalse(bool(self.run_steps(self.sample,499).success))
        state=self.run_steps(self.sample,500)
        self.assertTrue(bool(state.success))
        after,reward,_=advance_grasp_trace(state,self.sample,.002,self.cfg)
        self.assertEqual(float(reward),0.)
        self.assertEqual(float(after.elapsed),float(state.elapsed))

    def test_two_fingers_or_three_without_thumb_cannot_pass(self):
        for forces in ([.1,.1,0.,0.,0.],[0.,.1,.1,.1,0.]):
            sample=self.sample.replace(finger_forces=jp.array(forces))
            self.assertFalse(bool(self.run_steps(sample,600).success))

    def test_proximity_and_high_speed_do_not_count_as_hold(self):
        for sample in (self.sample.replace(finger_forces=jp.zeros(5)),
                       self.sample.replace(linvel=jp.array([.04,0.,0.]))):
            self.assertFalse(bool(self.run_steps(sample,600).success))

    def test_acquisition_time_does_not_excuse_support_or_penetration(self):
        for sample,reason in ((self.sample.replace(invalid_contact=jp.array(True)),SUPPORT),
                              (self.sample.replace(penetration=jp.array(True)),PENETRATION)):
            state,_,_=advance_grasp_trace(self.trace,sample,.002,self.cfg)
            self.assertTrue(bool(state.done))
            self.assertFalse(bool(state.success))
            self.assertEqual(int(state.failure_reason),reason)


if __name__=='__main__':unittest.main()
