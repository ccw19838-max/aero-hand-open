"""Reuse tested five-finger phases, with per-axis small-angle goals."""
from dataclasses import replace
import gymnasium as gym
import mujoco
import numpy as np

from experiments.cube_envelop.env import (
    CubeReorientEnv as EnvelopeEnv, physical_substep_reward as envelope_reward,
    ReorientMonitor,target_quaternion,orientation_error,AXES,
)
from .config import CubeReorientConfig,tolerance_for


def physical_substep_reward(config,monitor,contacts,linear,angular,stable,valid,delta):
    reward=envelope_reward(config,monitor,contacts,linear,angular,stable,valid,delta)
    # The former 30-degree proximity radius would reward standing still when
    # the target itself is only 5 degrees away. Scale the radius with the goal.
    if (config.task_mode=='reorient' and monitor.target_active and valid
            and np.any(np.asarray(contacts['finger_forces'])>=config.contact_force)):
        error=monitor.angle_error_radians
        radius=np.deg2rad(max(1.,min(30.,config.target_degrees/3.)))
        old=np.exp(-(error/np.deg2rad(30.))**2)
        new=np.exp(-(error/radius)**2)
        reward+=config.sim_dt*config.near_goal_reward*(new-old)
    return float(reward)


class CubeReorientEnv(EnvelopeEnv):
    def __init__(self,reference,config=None,render_mode=None,axis=None):
        self.curriculum_config=config or CubeReorientConfig()
        if not isinstance(self.curriculum_config,CubeReorientConfig):raise ValueError('Gradual curriculum config required')
        super().__init__(reference,config=self.curriculum_config,render_mode=render_mode,axis=axis)

    def reset(self,*,seed=None,options=None):
        gym.Env.reset(self,seed=seed)
        axis=(options or {}).get('axis',self.fixed_axis)
        if axis is None:axis=str(self.np_random.choice(self.curriculum_config.training_axes))
        if axis not in AXES:raise ValueError('Invalid axis')
        angle=self.curriculum_config.angle_for(axis)
        self.config=replace(self.curriculum_config,target_degrees=angle,
                            angle_tolerance_degrees=tolerance_for(angle))
        # Keep the RNG stream after axis selection. Parent reset performs the
        # exact same physical randomization and legal-contact rejection gate.
        return super().reset(seed=None,options={**(options or {}),'axis':axis})

    def _info(self):
        info=super()._info()
        info.update(target_degrees=self.config.target_degrees,
                    angle_tolerance_degrees=self.config.angle_tolerance_degrees,
                    curriculum_version=self.curriculum_config.curriculum_version)
        return info

    def step(self,action):
        if not self._has_reset:raise RuntimeError('Call reset before step')
        if self.monitor.done:raise RuntimeError('Episode ended; reset required')
        action=np.asarray(action,dtype=float)
        if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('Expected seven finite actions')
        action=np.clip(action,-1.,1.);c=self.config
        self.data.ctrl[:]=np.clip(self.data.ctrl+c.action_delta_fraction*self._range*action,
                                  self.ids['ctrl_lower'],self.ids['ctrl_upper'])
        reward=0.;before=self.monitor.elapsed
        for _ in range(c.substeps):
            previous_time=self.data.time
            mujoco.mj_step(self.model,self.data);mujoco.mj_forward(self.model,self.data)
            bad=not(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()) or self.data.time<previous_time
            self._refresh();q=self.ids['cube_qpos_adr']
            stable,valid,delta=self.monitor.update(self.data.qpos[q+3:q+7],self._contacts,self._linear,self._angular,
                                                   self._displacement,float(self.data.xpos[self.ids['cube_body'],2]),bad)
            reward+=physical_substep_reward(c,self.monitor,self._contacts,self._linear,self._angular,stable,valid,delta)
            if self.monitor.done:
                reward+=10. if self.monitor.success else -5.
                break
        elapsed=self.monitor.elapsed-before
        reward-=elapsed*(.05*float(np.mean((action-self.previous_action)**2))+.005*float(np.mean(action**2)))
        self.previous_action=action.copy();timeout=self.monitor.failure_reason=='timeout'
        return self._observation(),float(reward),bool(self.monitor.done and not timeout),bool(timeout),self._info()
