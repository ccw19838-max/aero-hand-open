"""Quarter-turn geometry and endpoint/holding checks, separate from spin speed."""
from dataclasses import dataclass
from itertools import permutations,product
import math
import numpy as np
import mujoco

from experiments.cube_envelop.env import (canonical_quaternion,quaternion_multiply,
    quaternion_conjugate,orientation_error,unit_quaternion)

AXES=('x','y','z')


@dataclass(frozen=True)
class Config:
    version:str='xyz_quarter_turn_v1'
    cube_size:float=.057
    cube_mass:float=.077966253
    cube_friction:float=.6
    sim_dt:float=.002
    ctrl_dt:float=.05
    solver_iterations:int=20
    render_width:int=960
    render_height:int=720
    warmup_seconds:float=.25
    episode_seconds:float=12.
    hold_seconds:float=1.
    tolerance_degrees:float=5.
    contact_force:float=.02
    max_penetration:float=.001
    violation_seconds:float=.1
    max_displacement:float=.08
    max_unsupported_seconds:float=.2
    position_noise:float=.0005
    orientation_noise_degrees:float=.5


def cube_symmetries():
    result=[]
    for perm in permutations(range(3)):
        for signs in product((-1.,1.),repeat=3):
            matrix=np.eye(3)[:,perm]*np.array(signs)
            if np.linalg.det(matrix)>.5:result.append(matrix)
    return tuple(result)


SYMMETRIES=cube_symmetries()


def rotation_matrix(q):
    value=np.empty(9);mujoco.mju_quat2Mat(value,unit_quaternion(q));return value.reshape(3,3)


def matrix_quaternion(matrix):
    value=np.empty(4);mujoco.mju_mat2Quat(value,np.asarray(matrix).reshape(-1))
    return canonical_quaternion(value)


def aligned_cube_basis(quaternion,preferred_basis):
    """Fixed cube-face axes closest to the mounting axes; endpoint box fits again.

    Aligning arbitrary hand axes with a non-aligned cube does not produce the
    cube's 24 standard orientations. Choose a proper signed face permutation.
    """
    rotation=rotation_matrix(quaternion)
    permutation=max(SYMMETRIES,key=lambda p:float(np.trace(preferred_basis.T@rotation@p)))
    return rotation@permutation


def quarter_target(initial,axis,basis):
    direction=basis[:,AXES.index(axis)]
    turn=np.r_[np.sqrt(.5),direction*np.sqrt(.5)]
    return canonical_quaternion(quaternion_multiply(turn,unit_quaternion(initial)))


class GoalMonitor:
    def __init__(self,initial,axis,basis,config=None,ideal_initial=None):
        self.config=config or Config();self.axis=axis;self.basis=np.asarray(basis).copy()
        self.initial=unit_quaternion(initial);self.ideal_initial=unit_quaternion(ideal_initial if ideal_initial is not None else initial)
        self.goal=quarter_target(self.ideal_initial,axis,self.basis)
        self.direction=self.basis[:,AXES.index(axis)]
        self.progress=0.;self.last_twist=0.;self.elapsed=0.;self.stable=0.;self.max_stable=0.
        self.error_vector,self.error=orientation_error(initial,self.goal);self.best_error=self.error
        self.success=False;self.failure='';self.penetration_time=0.;self.region_time=0.;self.unsupported_time=0.
        self.milestones=set();self.max_progress=0.

    @property
    def done(self):return self.success or bool(self.failure)

    def track(self,quaternion):
        relative=quaternion_multiply(quaternion,quaternion_conjugate(self.initial))
        component=float(np.dot(relative[1:],self.direction))
        if relative[0]**2+component**2>1e-12:
            twist=2*math.atan2(component,float(relative[0]))
            self.progress+=(twist-self.last_twist+math.pi)%(2*math.pi)-math.pi
            self.last_twist=twist;self.max_progress=max(self.max_progress,self.progress)

    def checkpoint(self,quaternion,contacts,linear,angular,displacement,dt=.01):
        if self.done:return 0.
        c=self.config;self.elapsed+=dt;old_error=self.error;old_stable=self.stable
        self.error_vector,self.error=orientation_error(quaternion,self.goal);self.best_error=min(self.best_error,self.error)
        valid=not(contacts['forbidden_support'] or contacts['max_penetration']>c.max_penetration or displacement>c.max_displacement)
        supported=contacts['hand_force']>=c.contact_force
        self.penetration_time=self.penetration_time+dt if contacts['max_penetration']>c.max_penetration else 0.
        self.region_time=self.region_time+dt if displacement>c.max_displacement else 0.
        self.unsupported_time=self.unsupported_time+dt if not supported else 0.
        touching=np.asarray(contacts['finger_forces'])>=c.contact_force
        secure=bool(touching[0] and np.count_nonzero(touching[1:])>=2)
        at_goal=self.error<=np.deg2rad(c.tolerance_degrees)+1e-12 and abs(self.progress-np.pi/2)<=np.deg2rad(c.tolerance_degrees)+1e-12
        stable=bool(valid and at_goal and secure and np.linalg.norm(linear)<.03 and np.linalg.norm(angular)<.5)
        self.stable=self.stable+dt if stable else 0.;self.max_stable=max(self.max_stable,self.stable)
        delta=(old_error-self.error)/(np.pi/2)
        reward=40.*(delta if valid and supported else min(delta,0.))
        # Holding progress is a potential: losing the hold undoes its shaping
        # reward, so repeatedly breaking a 0.99-second hold cannot farm it.
        reward+=20.*(self.stable-old_stable)/c.hold_seconds
        speed=float(np.dot(angular,self.direction));perpendicular=angular-speed*self.direction
        desired=float(np.clip(3.*np.dot(self.error_vector,self.direction),-3.,3.))
        rate=-.2-.05*min((speed-desired)**2,25.)-.05*min(float(perpendicular@perpendicular),100.)
        rate-=.1*min(float(np.linalg.norm(linear)/.1)**2,25.)
        if valid and supported:
            rate+=4.*np.exp(-(self.error/np.deg2rad(15.))**2)
            if self.error<np.deg2rad(20.):rate+=float(np.mean(touching))
            # Reward intermediate progress once, without changing the 90-degree
            # task or imposing a stop/holding gate at 30 or 60 degrees.
            for angle in (30,60):
                if angle not in self.milestones and self.progress>=np.deg2rad(angle) and self.error<=np.deg2rad(95-angle):
                    self.milestones.add(angle);reward+=5.
        reward+=dt*rate
        if contacts['forbidden_support']:self.failure='forbidden_support'
        elif self.penetration_time+1e-10>=c.violation_seconds:self.failure='penetration'
        elif self.region_time+1e-10>=c.violation_seconds:self.failure='workspace'
        elif self.unsupported_time+1e-10>=c.max_unsupported_seconds:self.failure='lost_hand_contact'
        elif self.stable+1e-10>=c.hold_seconds:self.success=True
        elif self.elapsed+1e-10>=c.episode_seconds:self.failure='timeout'
        if self.done:reward+=200. if self.success else -100.
        return float(reward)
