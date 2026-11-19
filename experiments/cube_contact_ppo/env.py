"""Incremental tendon targets, actual five-finger forces, mixed integrator workers."""
import gymnasium as gym
import mujoco
import numpy as np
from experiments.cube_quarter_turn_v2.env import TeacherEnv
from experiments.cube_quarter_turn.task import aligned_cube_basis
from .task import Config,ContactMonitor,force_state

OBS_SIZE=138


class ContactEnv(TeacherEnv):
    def __init__(self,reference,axis='x',config=None,physical_dt=.001,target_degrees=0.,randomization=0.):
        self.target_degrees=float(target_degrees);self.randomization=float(randomization)
        self.next_target_degrees=self.target_degrees;self.next_randomization=self.randomization
        self.applied_randomization=0.;self.tip_distances=np.zeros(5)
        super().__init__(reference,axis,config or Config(),physical_dt)
        self.nominal_friction=self.model.geom_friction.copy();self.nominal_damping=self.model.dof_damping.copy()
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(OBS_SIZE,),dtype=np.float32)
        self.force_min=np.full(5,np.inf);self.force_max=np.zeros(5)

    def set_task(self,target_degrees,randomization=0.):
        if target_degrees not in (0.,30.,60.,90.) or not 0.<=randomization<=.1:raise ValueError('Invalid bounded curriculum')
        # Effective at the next reset: never move a goal inside a live rollout.
        self.next_target_degrees=float(target_degrees);self.next_randomization=float(randomization)

    def reset(self,*,seed=None,options=None):
        if seed is not None:self.np_random=np.random.default_rng(seed)
        self.target_degrees=self.next_target_degrees;self.randomization=self.next_randomization
        level=self.randomization
        self.applied_randomization=level
        self.model.geom_friction[:]=self.nominal_friction
        self.model.dof_damping[:]=self.nominal_damping
        if level:
            # Small bounded randomization only after the nominal final task is
            # competent; never change nominal size, gravity or contact limits.
            self.model.geom_friction[self.ids['cube_geom'],0]*=self.np_random.uniform(1-level,1+level)
            self.model.dof_damping[:16]*=self.np_random.uniform(1-level,1+level)
        self.force_min=np.full(5,np.inf);self.force_max=np.zeros(5)
        super().reset(seed=seed,options=options)
        self.monitor=ContactMonitor(self.data.qpos[19:23].copy(),self.monitor.axis,self.basis,self.config,self.target_degrees)
        self.last_observation=self._observation(self.last_action)
        return self.last_observation.copy(),self._info()

    def _refresh(self):
        super()._refresh()
        rotation=self.data.xmat[self.ids['cube_body']].reshape(3,3)
        local=(self.data.site_xpos[self.tip_sites]-self.data.xpos[self.ids['cube_body']])@rotation
        excess=np.abs(local)-self.config.cube_size/2
        self.tip_distances=np.linalg.norm(np.maximum(excess,0.),axis=1)+np.minimum(np.max(excess,axis=1),0.)
        self.contacts['tip_distances']=self.tip_distances.copy()

    def start_goal(self,axis):
        if not self.monitor.success:raise RuntimeError('A successful five-finger endpoint is required')
        if axis not in ('x','y'):raise ValueError('Only X/Y enabled')
        ideal=self.monitor.goal.copy()
        self.target_degrees=self.next_target_degrees=90.
        self.monitor=ContactMonitor(self.data.qpos[19:23].copy(),axis,self.basis,self.config,90.,ideal_initial=ideal)
        self.warming=False
        self.last_observation=self._observation(self.last_action)
        return self.last_observation.copy(),self._info()

    def _observation(self,history):
        base=super()._observation(history)
        m=self.monitor
        _,_,coverage=force_state(self.contacts['finger_forces'],self.config)
        return np.r_[base,self.target_degrees/90.,coverage,
            min(1.,getattr(m,'all_five_seconds',0.)/self.config.hold_seconds),self.physical_dt/.002,
            min(1.,getattr(m,'insufficient_contact_seconds',0.)/self.config.contact_grace_seconds)].astype(np.float32)

    def _info(self):
        info=super()._info();m=self.monitor
        rotating,five,coverage=force_state(self.contacts['finger_forces'],self.config)
        info.update(task=self.config.version,goal_degrees=getattr(m,'target_degrees',self.target_degrees),
            final_goal_degrees=90.,curriculum_complete=getattr(m,'target_degrees',self.target_degrees)==90.,
            actor_inputs=OBS_SIZE,action_mapping='incremental motor target, at most 2% valid range per control cycle',
            five_finger_contact=five,rotation_support=rotating,contact_coverage=coverage,
            all_five_stable_seconds=getattr(m,'all_five_seconds',0.),
            max_all_five_stable_seconds=getattr(m,'max_all_five_seconds',0.),
            five_finger_duty=getattr(m,'five_finger_duty_seconds',0.)/max(.01,m.elapsed),
            insufficient_contact_seconds=getattr(m,'insufficient_contact_seconds',0.),
            applied_randomization=self.applied_randomization)
        if hasattr(self,'force_min') and np.isfinite(self.force_min).all():
            info.update(control_cycle_min_finger_forces=self.force_min.tolist(),control_cycle_max_finger_forces=self.force_max.tolist())
        return info

    def step(self,action):
        if not self._has_reset or self.monitor.done:raise RuntimeError('Reset first')
        action=np.asarray(action,dtype=float)
        if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('Seven finite actions required')
        applied=np.zeros(7) if self.warming else np.clip(action,-1.,1.)
        if self.warming:self.data.ctrl[:]=self.default
        else:
            delta=self.config.max_action_delta_fraction*(self.ids['ctrl_upper']-self.ids['ctrl_lower'])*applied
            self.data.ctrl[:]=np.clip(self.data.ctrl+delta,self.ids['ctrl_lower'],self.ids['ctrl_upper'])
        reward=0.;self.force_min[:]=np.inf;self.force_max[:]=0.
        for i in range(round(self.config.ctrl_dt/self.physical_dt)):
            before=self.data.time;mujoco.mj_step(self.model,self.data);self.elapsed+=self.physical_dt
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or self.data.time<before:
                self.monitor.failure='numerical';reward-=self.config.failure_penalty;break
            if self._forbidden():self.monitor.failure='forbidden_support';reward-=self.config.failure_penalty;break
            if not self.warming:self.monitor.track(self.data.qpos[19:23])
            if (i+1)%round(.01/self.physical_dt)==0:
                mujoco.mj_forward(self.model,self.data);self._refresh()
                self.force_min=np.minimum(self.force_min,self.contacts['finger_forces'])
                self.force_max=np.maximum(self.force_max,self.contacts['finger_forces'])
                value=self.monitor.checkpoint(self.data.qpos[19:23],self.contacts,self.linear,self.angular,self.displacement,.01)
                self.grip_stable_seconds=self.monitor.all_five_seconds
                if not self.warming:reward+=value
                if self.monitor.done:break
        if self.warming and self.elapsed+1e-10>=self.config.warmup_seconds and not self.monitor.done:
            self.basis=aligned_cube_basis(self.data.qpos[19:23],self.mount_basis)
            self.monitor=ContactMonitor(self.data.qpos[19:23].copy(),self.monitor.axis,self.basis,self.config,self.target_degrees)
            self.warming=False
        reward-=.01*float(np.mean((applied-self.last_action)**2))
        self.last_action=applied.copy();self.returned_history=applied.copy()
        obs=self.last_observation.copy() if self.monitor.failure=='numerical' else self._observation(applied)
        self.last_observation=obs.copy();info=self._info()
        info.update(applied_action=applied.tolist(),motor_targets=self.data.ctrl.tolist())
        return obs,float(reward),bool(self.monitor.done),False,info
