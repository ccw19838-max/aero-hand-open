from dataclasses import replace
import numpy as np
from experiments.cube_contact_recovery_v2.env import NativeEnv as NativeBase,FastEnv as FastBase
from experiments.cube_adaptive_60h.env import AdaptiveMonitor
from experiments.cube_adaptive_60h.thumb_drive import credit_progress
from experiments.cube_contact_ppo.task import force_state,force_limits
from experiments.cube_adaptive_60h.shaping import MountMarginReward
from .roles import SUPPORT,DRIVERS,support_state,wrenches,driving_quality

VERSION='x_thumb_pinky_three_support_v3'
LADDER=(0.,5.,10.,20.,30.,45.,60.,75.,90.)


class RoleMonitor(AdaptiveMonitor):
    def __init__(self,*args,env,**kwargs):
        super().__init__(*args,**kwargs);self.env=env
        self.three_missing=0.;self.max_three_missing=0.;self.support_seconds=0.
        self.support_potential=None;self.high_water=0.;self.supported_progress=0.
        self.role_bonus=0.;self.role_penalty=0.;self.last_moments=np.zeros(5);self.last_speeds=np.zeros(5)
    def checkpoint(self,q,contacts,linear,angular,displacement,dt=.01):
        if self.done:return 0.
        before=self.error;value=super().checkpoint(q,contacts,linear,angular,displacement,dt)
        if self.env.warming:return value
        f=np.asarray(contacts['finger_forces']);support=support_state(f);lower,upper=force_limits(self.config)
        self.three_missing=0. if support else self.three_missing+dt
        self.max_three_missing=max(self.max_three_missing,self.three_missing)
        valid=bool(not contacts['forbidden_support'] and contacts['max_penetration']<=self.config.max_penetration and displacement<=self.config.max_displacement)
        self.support_seconds+=dt*(support and valid)
        if support and valid:self.supported_progress=max(self.supported_progress,self.progress)
        fraction=np.clip(f[list(SUPPORT)]/.1,0.,1.);deficit=1.-fraction
        proximity=np.exp(-(np.maximum(self.env.tip_distances[list(SUPPORT)],0.)/.02)**2)
        potential=10.*float(np.min(fraction))+5.*float(np.mean(fraction))+3.*float(np.mean(proximity))
        if self.support_potential is not None:value+=potential-self.support_potential
        self.support_potential=potential
        penalty=dt*(12.*float(np.mean(deficit**2))+8.*float(np.max(deficit**2)))
        value-=penalty;self.role_penalty-=penalty
        delta=(before-self.error)/(np.pi/2)
        if not support and valid and force_state(f,self.config)[0]:value-=self.profile.progress_gain*max(delta,0.)
        if self.target_degrees>0. and valid and force_state(f,self.config)[0]:
            # The inherited 15-degree near-goal bonus would already be large
            # at the start of a 5-degree lesson. Tie its width to the actual
            # tolerance, and require the three named support fingers.
            coverage=force_state(f,self.config)[2]
            old_near=2.*np.exp(-(self.error/np.deg2rad(15.))**2)*coverage**2
            new_near=2.*np.exp(-(self.error/np.deg2rad(self.config.tolerance_degrees))**2)*coverage**2 if support else 0.
            value+=dt*(new_near-old_near)
        metrics=wrenches(self.env);self.last_moments=metrics['axis_moments_Nm'];self.last_speeds=metrics['tangential_speeds_m_s']
        if self.target_degrees>0.:
            increment,self.high_water=credit_progress(self.progress,self.error,np.deg2rad(self.target_degrees),self.high_water)
            if support and valid and not self.failure and np.all(f<=upper):
                qualities=[driving_quality(self.last_moments[i],self.last_speeds[i]) if f[i]>=lower[i] else 0. for i in DRIVERS]
                quality=.5*float(np.mean(qualities))+.5*min(qualities)
                bonus=8.*increment/(np.pi/2)*quality
                value+=bonus;self.role_bonus+=bonus
        # Preserve all old safety failures and add the named support condition.
        if self.three_missing>.15+1e-10 and not self.done:
            self.failure='three_support_lost';value-=self.config.failure_penalty
        return float(value)


class Roles:
    def __init__(self,*args,**kwargs):
        if kwargs.get('prefix_bank') is not None:raise ValueError('V3 always starts from original prepared grasp')
        super().__init__(*args,**kwargs)
        if self.profile.grip_mix!=0.:raise ValueError('Explicit curriculum only; no hidden goal mixing')
        self.config=replace(self.config,version=VERSION)
    def replace_monitor(self):
        old=self.monitor
        if isinstance(old,RoleMonitor):return
        self.monitor=RoleMonitor(old.initial,old.axis,old.basis,self.config,old.target_degrees,
            ideal_initial=old.ideal_initial,profile=self.profile,env=self)
    def set_task(self,target_degrees,randomization=0.):
        if target_degrees not in LADDER or randomization!=0.:raise ValueError('Invalid fixed-physics role curriculum')
        self.stage_target=float(target_degrees)
    def reset(self,**kwargs):
        tolerance=5. if self.stage_target==0. else min(5.,self.stage_target/4.)
        self.config=replace(self.config,tolerance_degrees=tolerance)
        return super().reset(**kwargs)
    def control_mode(self):
        if self.warming:return 'setup'
        if not support_state(self.contacts['finger_forces']):return 'recover'
        if getattr(self.monitor,'target_degrees',self.target_degrees)==0. or self.monitor.stable>0.:return 'hold'
        if self.monitor.error<=np.deg2rad(max(2.,min(20.,self.target_degrees/3.))):return 'capture'
        return 'drive'
    def _info(self):
        info=super()._info();m=self.monitor
        info.update(task=VERSION,environment_version=VERSION,
            from_original_initial_grasp=True,evaluation_kind='original_grasp_named_finger_roles',
            three_support=support_state(self.contacts['finger_forces']),
            three_support_duty=getattr(m,'support_seconds',0.)/max(.01,m.elapsed),
            three_support_missing_seconds=getattr(m,'three_missing',0.),
            max_three_support_missing_seconds=getattr(m,'max_three_missing',0.),
            supported_progress_degrees=float(np.rad2deg(getattr(m,'supported_progress',0.))),
            driver_moments_Nm=getattr(m,'last_moments',np.zeros(5))[[0,4]].tolist(),
            driver_tangential_speeds_m_s=getattr(m,'last_speeds',np.zeros(5))[[0,4]].tolist(),
            dual_driver_bonus=getattr(m,'role_bonus',0.),three_support_penalty=getattr(m,'role_penalty',0.),
            tolerance_degrees=self.config.tolerance_degrees)
        return info


class NativeEnv(Roles,NativeBase):pass
class FastEnv(Roles,FastBase):pass


def training_reward(env):return MountMarginReward(env)
