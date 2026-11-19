"""Phase-aware force constraints and angular-goal rewards, without a demo gate."""
from dataclasses import dataclass
import numpy as np
from experiments.cube_quarter_turn_v2.task import Config as V2Config
from experiments.cube_quarter_turn.task import (GoalMonitor as BaseMonitor,orientation_error,
    quaternion_multiply,canonical_quaternion)


@dataclass(frozen=True)
class Config(V2Config):
    version: str = 'xy_five_finger_closed_loop_ppo_v1'
    max_action_delta_fraction: float = .02
    finger_min_force: float = .1
    thumb_min_force: float = .2
    finger_force_soft_limit: float = 12.
    thumb_force_soft_limit: float = 20.
    contact_grace_seconds: float = .15
    stagnation_seconds: float = 6.
    failure_penalty: float = 20.
    success_bonus: float = 50.


def force_limits(config):
    return (np.r_[config.thumb_min_force,np.full(4,config.finger_min_force)],
            np.r_[config.thumb_force_soft_limit,np.full(4,config.finger_force_soft_limit)])


def force_state(forces,config):
    forces=np.asarray(forces,dtype=float)
    if forces.shape!=(5,) or not np.isfinite(forces).all():raise ValueError('Five real finite contact forces required')
    lower,upper=force_limits(config)
    firm=forces>=lower
    rotation_support=bool(firm[0] and np.count_nonzero(firm[1:])>=2)
    five_finger=bool(np.all(firm) and np.all(forces<=upper))
    coverage=float(np.mean(np.clip(forces/lower,0.,1.)))
    return rotation_support,five_finger,coverage


class ContactMonitor(BaseMonitor):
    def __init__(self,initial,axis,basis,config=None,target_degrees=90.,ideal_initial=None):
        config=config or Config()
        if axis not in ('x','y') or not 0.<=target_degrees<=90.:raise ValueError('Invalid XY curriculum goal')
        super().__init__(initial,axis,basis,config,ideal_initial=ideal_initial)
        self.target_degrees=float(target_degrees)
        angle=np.deg2rad(target_degrees)
        self.goal=canonical_quaternion(quaternion_multiply(np.r_[np.cos(angle/2),self.direction*np.sin(angle/2)],self.ideal_initial))
        self.error_vector,self.error=orientation_error(initial,self.goal);self.best_error=self.error
        self.insufficient_contact_seconds=0.;self.stagnant_seconds=0.;self.progress_anchor=self.error
        self.all_five_seconds=0.;self.max_all_five_seconds=0.;self.five_finger_duty_seconds=0.
        self.rotation_support_seconds=0.;self.last_coverage=1.

    @property
    def phase(self):
        if self.target_degrees==0. or self.stable>0.:return 'hold'
        return 'capture' if self.error<=np.deg2rad(20.) else 'drive'

    def checkpoint(self,quaternion,contacts,linear,angular,displacement,dt=.01):
        if self.done:return 0.
        c=self.config;old_error=self.error;old_stable=self.stable;self.elapsed+=dt
        self.error_vector,self.error=orientation_error(quaternion,self.goal);self.best_error=min(self.best_error,self.error)
        forces=np.asarray(contacts['finger_forces'])
        rotation_support,five_finger,coverage=force_state(forces,c)
        valid=bool(not contacts['forbidden_support'] and contacts['max_penetration']<=c.max_penetration and displacement<=c.max_displacement)
        self.penetration_time=self.penetration_time+dt if contacts['max_penetration']>c.max_penetration else 0.
        self.region_time=self.region_time+dt if displacement>c.max_displacement else 0.
        self.unsupported_time=self.unsupported_time+dt if contacts['hand_force']<c.contact_force else 0.
        self.insufficient_contact_seconds=0. if rotation_support else self.insufficient_contact_seconds+dt
        self.rotation_support_seconds+=dt*(rotation_support and valid)
        self.five_finger_duty_seconds+=dt*(five_finger and valid)
        low_speed=bool(np.linalg.norm(linear)<.03 and np.linalg.norm(angular)<.5)
        self.all_five_seconds=self.all_five_seconds+dt if valid and five_finger and low_speed else 0.
        self.max_all_five_seconds=max(self.max_all_five_seconds,self.all_five_seconds)
        signed_error=abs(self.progress-np.deg2rad(self.target_degrees))
        at_goal=self.error<=np.deg2rad(c.tolerance_degrees)+1e-12 and signed_error<=np.deg2rad(c.tolerance_degrees)+1e-12
        stable=bool(valid and five_finger and low_speed and at_goal)
        self.stable=self.stable+dt if stable else 0.;self.max_stable=max(self.max_stable,self.stable)
        delta=(old_error-self.error)/(np.pi/2)
        reward=20.*(delta if valid and rotation_support else min(delta,0.))
        reward+=20.*(self.stable-old_stable)/c.hold_seconds
        reward+=2.*(coverage-self.last_coverage);self.last_coverage=coverage
        lower,upper=force_limits(c)
        force_excess=np.maximum(forces-upper,0.)/upper
        deficit=np.clip((lower-forces)/lower,0.,1.)
        speed=float(np.dot(angular,self.direction));perpendicular=angular-speed*self.direction
        rate=-1.-2.*(self.error/(np.pi/2))**2+.5*coverage
        rate-=.05*min(float(perpendicular@perpendicular),100.)
        rate-=.1*min(float(np.linalg.norm(linear)/.1)**2,25.)
        rate-=.2*float(np.mean(force_excess**2))
        # Missing fingers cost more during capture/holding. Saturating contact
        # coverage gives no incentive to manufacture huge normal forces.
        if self.phase in ('capture','hold'):rate-=float(np.mean(deficit**2))
        if valid and rotation_support:
            rate+=2.*np.exp(-(self.error/np.deg2rad(15.))**2)*coverage**2
            distances=np.asarray(contacts.get('tip_distances',np.zeros(5)))
            rate+=.25*float(np.mean(np.exp(-(np.maximum(distances,0.)/.02)**2)))
        reward+=dt*rate
        if valid and rotation_support and self.progress_anchor-self.error>=np.deg2rad(2.):
            self.progress_anchor=self.error;self.stagnant_seconds=0.
        elif self.stable>0. or self.target_degrees==0.:self.stagnant_seconds=0.
        else:self.stagnant_seconds+=dt
        if contacts['forbidden_support']:self.failure='forbidden_support'
        elif self.penetration_time+1e-10>=c.violation_seconds:self.failure='penetration'
        elif self.region_time+1e-10>=c.violation_seconds:self.failure='workspace'
        elif self.unsupported_time+1e-10>=c.max_unsupported_seconds:self.failure='lost_hand_contact'
        elif self.insufficient_contact_seconds>c.contact_grace_seconds+1e-10:self.failure='insufficient_rotation_contacts'
        elif self.stable+1e-10>=c.hold_seconds:self.success=True
        elif self.stagnant_seconds>=c.stagnation_seconds:self.failure='stagnation'
        elif self.elapsed+1e-10>=c.episode_seconds:self.failure='timeout'
        if self.done:reward+=c.success_bonus if self.success else -c.failure_penalty
        return float(reward)
