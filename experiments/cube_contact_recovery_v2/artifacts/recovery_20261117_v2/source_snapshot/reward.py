"""Read-only contact-state shaping; original physical goal checks are retained."""
from types import MethodType
import gymnasium as gym
import numpy as np
from experiments.cube_contact_ppo.task import force_limits,force_state
from experiments.cube_adaptive_60h.thumb_drive import contact_wrenches,drive_quality,credit_progress
from experiments.cube_adaptive_60h.shaping import MountMarginReward


class RecoveryReward(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env);self.patched=None;self.previous=None;self.high_water=0.;self.total=0.
    def install(self):
        base=self.unwrapped;m=base.monitor
        if self.patched is m:return
        original=m.checkpoint;self.previous=None;self.high_water=0.
        def checkpoint(this,q,contacts,linear,angular,displacement,dt=.01):
            before=this.error
            value=original(q,contacts,linear,angular,displacement,dt)
            if base.warming:return value
            forces=np.asarray(contacts['finger_forces']);lower,upper=force_limits(base.config)
            deficit=np.clip((lower-forces)/lower,0.,1.)
            # Distances are explicitly thumb-first in the V2 environment.
            proximity=np.exp(-(np.maximum(base.tip_distances,0.)/.02)**2)
            potential=8.*float(np.mean(np.clip(forces/lower,0.,1.)))+4.*float(np.mean(proximity))
            adjustment=0. if self.previous is None else potential-self.previous
            self.previous=potential
            recovering=base.control_mode()=='recover'
            adjustment-=dt*(12. if recovering else 3.)*float(np.mean(deficit**2))
            anchor=base.recovery_anchor if this.target_degrees==0. else base.initial_position
            if anchor is not None:
                distance=np.linalg.norm(base.data.xpos[base.ids['cube_body']]-anchor)
                adjustment-=dt*.8*min((distance/.02)**2,16.)
            support=force_state(forces,base.config)[0]
            valid=not contacts['forbidden_support'] and not this.failure and contacts['max_penetration']<=base.config.max_penetration and displacement<=base.config.max_displacement
            if this.target_degrees>0.:
                delta=(before-this.error)/(np.pi/2)
                if recovering and support and valid:
                    # Don't pay for advancing the turn while repair is due.
                    adjustment-=base.profile.progress_gain*max(delta,0.)
                increment,self.high_water=credit_progress(this.progress,this.error,np.deg2rad(this.target_degrees),self.high_water)
                if not recovering and valid and support and np.all(forces<=upper):
                    metrics=contact_wrenches(base)
                    quality=drive_quality(metrics['axis_moments_Nm'][0],metrics['thumb_surface_speed_m_s'],forces[0],support,valid)
                    adjustment+=6.*increment/(np.pi/2)*quality*float(np.clip((np.rad2deg(this.error)-5.)/20.,0.,1.))
            self.total+=adjustment
            return float(value+adjustment)
        m.checkpoint=MethodType(checkpoint,m);self.patched=m
    def reset(self,**kwargs):
        result=self.env.reset(**kwargs);self.total=0.;self.install();return result
    def step(self,action):
        self.install();obs,reward,done,truncated,info=self.env.step(action)
        return obs,reward,done,truncated,{**info,'recovery_reward_adjustment':self.total}


def training_reward(env):return RecoveryReward(MountMarginReward(env))
