"""Always-on contact recovery; strict final criterion is inherited unchanged."""
from dataclasses import replace
import gymnasium as gym
import numpy as np
from experiments.cube_contact_ppo.env import ContactEnv
from experiments.cube_contact_ppo.task import ContactMonitor,force_limits,force_state
from .config import Config,Profile,validate_profile,LADDER

OBS_SIZE=143


class AdaptiveMonitor(ContactMonitor):
    def __init__(self,*args,profile=None,**kwargs):
        self.profile=profile or Profile()
        super().__init__(*args,**kwargs)
        self.missing_seconds=np.zeros(5);self.max_missing_seconds=np.zeros(5)
        self.pressure_potential=None;self.recovery_penalty=0.;self.stall_penalty=0.
    def checkpoint(self,q,contacts,linear,angular,displacement,dt=.01):
        if self.done:return 0.
        previous=self.error
        reward=super().checkpoint(q,contacts,linear,angular,displacement,dt)
        p=self.profile;f=np.asarray(contacts['finger_forces']);lower,_=force_limits(self.config)
        self.missing_seconds=np.where(f<lower,self.missing_seconds+dt,0.)
        self.max_missing_seconds=np.maximum(self.max_missing_seconds,self.missing_seconds)
        deficit=np.clip((lower-f)/lower,0.,1.)
        overdue=np.clip((self.missing_seconds-p.loss_grace),0.,1.)
        penalty=dt*(p.contact_weight*float(np.mean(deficit**2))+p.loss_weight*float(np.mean(overdue)))
        reward-=penalty;self.recovery_penalty-=penalty
        pressure=p.pressure_potential_gain*float(np.mean(np.clip(f/np.array([1.,.5,.5,.5,.5]),0.,1.)))
        if self.pressure_potential is not None:reward+=pressure-self.pressure_potential
        self.pressure_potential=pressure
        valid=not contacts['forbidden_support'] and contacts['max_penetration']<=self.config.max_penetration and displacement<=self.config.max_displacement
        support=force_state(f,self.config)[0]
        delta=(previous-self.error)/(np.pi/2)
        reward+=(p.progress_gain-20.)*(delta if valid and support else min(delta,0.))
        if self.target_degrees>0.:
            penalty=dt*p.stall_weight*float(np.clip((self.stagnant_seconds-.5)/4.,0.,1.))
            reward-=penalty;self.stall_penalty-=penalty
        return float(reward)


class AdaptiveEnv(ContactEnv):
    def __init__(self,reference,axis='x',physical_dt=.001,target_degrees=10.,profile=None,training=False):
        self.profile=validate_profile(profile or Profile());self.stage_target=float(target_degrees)
        self.training_mode=training
        config=replace(Config(),max_action_delta_fraction=self.profile.action_fraction)
        super().__init__(reference,axis,config=config,physical_dt=physical_dt,target_degrees=target_degrees)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(OBS_SIZE,),dtype=np.float32)
    def set_task(self,target_degrees,randomization=0.):
        if target_degrees not in LADDER or randomization!=0.:raise ValueError('Fixed-physics curriculum only')
        self.stage_target=float(target_degrees)
    def replace_monitor(self):
        old=self.monitor
        if isinstance(old,AdaptiveMonitor):return
        self.monitor=AdaptiveMonitor(old.initial,old.axis,old.basis,self.config,old.target_degrees,
            ideal_initial=old.ideal_initial,profile=self.profile)
    def reset(self,*,seed=None,options=None):
        if seed is not None:self.np_random=np.random.default_rng(seed)
        target=(0. if self.training_mode and self.stage_target>0. and self.np_random.random()<self.profile.grip_mix else self.stage_target)
        self.next_target_degrees=target;self.next_randomization=0.
        super().reset(seed=seed,options=options)
        self.replace_monitor()
        self.last_observation=self._observation(self.last_action)
        return self.last_observation.copy(),self._info()
    def _observation(self,history):
        base=super()._observation(history)
        # Preserve the frozen 138-input feature units on transfer, even though
        # the new stagnation termination gives coordination two extra seconds.
        base[129]=min(1.,getattr(self.monitor,'stagnant_seconds',0.)/6.)
        missing=getattr(self.monitor,'missing_seconds',np.zeros(5))
        return np.r_[base,np.clip(missing/2.,0.,1.)].astype(np.float32)
    def _info(self):
        info=super()._info();m=self.monitor
        info.update(stage_target_degrees=self.stage_target,training_grip_mix=self.training_mode,
            contact_loss_seconds=getattr(m,'missing_seconds',np.zeros(5)).tolist(),
            maximum_contact_loss_seconds=getattr(m,'max_missing_seconds',np.zeros(5)).tolist(),
            recovery_penalty=getattr(m,'recovery_penalty',0.),stall_penalty=getattr(m,'stall_penalty',0.),
            action_mapping=f'incremental motor target, at most {self.profile.action_fraction*100:g}% range/cycle',
            actor_inputs=OBS_SIZE)
        return info
    def step(self,action):
        result=super().step(action)
        if not isinstance(self.monitor,AdaptiveMonitor):
            # Parent replaces the goal at the end of the physical warmup.
            self.replace_monitor();obs=self._observation(self.last_action);self.last_observation=obs.copy()
            return obs,result[1],result[2],result[3],self._info_with_action()
        return result
    def _info_with_action(self):
        info=self._info();info.update(applied_action=self.last_action.tolist(),motor_targets=self.data.ctrl.tolist());return info
