"""Training-only precision shaping; observations/physics/termination unchanged."""
from types import MethodType
import gymnasium as gym
import numpy as np
import mujoco
from experiments.cube_contact_ppo.task import force_state

TRANSFORMS=(None,'narrow_small_angle_v1','narrow_small_angle_mount_margin_v1','center_missing_tip_mount_v1')


def colliding_forbidden_geoms(env):
    """Visual meshes on the mount have no collision and must be excluded."""
    base=env.unwrapped;model=base.model;cube=base.ids['cube_geom']
    pairs=((model.geom_contype&model.geom_conaffinity[cube])!=0)|((model.geom_conaffinity&model.geom_contype[cube])!=0)
    return np.flatnonzero(base.ids['geom_forbidden']&pairs)


class NarrowGoalReward(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env);self.patched=None;self.total_adjustment=0.
    def install(self):
        m=self.env.monitor
        if self.patched is m:return
        original=m.checkpoint
        def checkpoint(this,q,contacts,linear,angular,displacement,dt=.01):
            value=original(q,contacts,linear,angular,displacement,dt)
            target=this.target_degrees
            if target<=0. or target>=45.:return value
            support,_,coverage=force_state(contacts['finger_forces'],this.config)
            valid=not contacts['forbidden_support'] and contacts['max_penetration']<=this.config.max_penetration and displacement<=this.config.max_displacement
            if valid and support:
                old=2.*np.exp(-(this.error/np.deg2rad(15.))**2)*coverage**2
                width=np.deg2rad(max(5.,target/3.))
                new=2.*np.exp(-(this.error/width)**2)*coverage**2
                adjustment=dt*(new-old);value+=adjustment
                if not self.env.warming:self.total_adjustment+=adjustment
            return float(value)
        m.checkpoint=MethodType(checkpoint,m);self.patched=m
    def reset(self,**kwargs):
        result=self.env.reset(**kwargs);self.total_adjustment=0.;self.install();return result
    def step(self,action):
        self.install();result=self.env.step(action)
        info={**result[4],'training_reward_transform':'narrow_small_angle_v1','reward_transform_total':self.total_adjustment}
        return result[0],result[1],result[2],result[3],info


class MountMarginReward(gym.Wrapper):
    """Soft warning within5mm of collidable fixed support; no safety override."""
    def __init__(self,env,margin=.005,weight=2.):
        super().__init__(env);self.margin=float(margin);self.weight=float(weight)
        self.geoms=colliding_forbidden_geoms(env);self.total_penalty=0.
    def clearance(self):
        base=self.unwrapped
        return min((float(mujoco.mj_geomDistance(base.model,base.data,base.ids['cube_geom'],int(g),.1,None))
            for g in self.geoms),default=.1)
    def reset(self,**kwargs):
        result=self.env.reset(**kwargs);self.total_penalty=0.;return result
    def step(self,action):
        base=self.unwrapped;before=base.data.time;warming=base.warming
        obs,reward,done,truncated,info=self.env.step(action)
        gap=self.clearance();elapsed=max(0.,float(base.data.time-before))
        penalty=0. if warming else elapsed*self.weight*float(np.clip((self.margin-gap)/self.margin,0.,2.)**2)
        self.total_penalty+=penalty
        info={**info,'training_reward_transform':'narrow_small_angle_mount_margin_v1',
            'mount_clearance_m':gap,'mount_proximity_penalty':penalty,'mount_proximity_penalty_total':self.total_penalty}
        return obs,float(reward-penalty),done,truncated,info
