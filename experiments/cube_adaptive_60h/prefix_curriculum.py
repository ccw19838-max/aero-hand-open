"""Auxiliary capture practice after replaying a physically reachable prefix."""
from pathlib import Path
import gymnasium as gym
import numpy as np


def read_actions(path,steps):
    with np.load(Path(path),allow_pickle=False) as data:actions=data['actions'].copy()
    if (actions.ndim!=2 or actions.shape[1]!=7 or not 1<=steps<=len(actions)
            or not np.isfinite(actions).all() or np.max(np.abs(actions))>1.+1e-8):
        raise ValueError('Invalid actual trajectory prefix')
    return actions[:steps].astype(np.float64)


class ReachablePrefixCurriculum(gym.Wrapper):
    def __init__(self,env,actions,probability=.5):
        super().__init__(env)
        if not 0.<=probability<=1.:raise ValueError('Invalid prefix probability')
        self.actions=np.asarray(actions);self.probability=float(probability);self.prefix_rng=np.random.default_rng(8129)
        self.used=False;self.rejected=None
    def annotate(self,info):
        return {**info,'training_prefix_used':self.used,'training_prefix_controls':len(self.actions) if self.used else 0,
            'neural_controls_from_initial_grasp':not self.used,'training_prefix_rejected':self.rejected}
    def reset(self,**kwargs):
        if kwargs.get('seed') is not None:self.prefix_rng=np.random.default_rng(int(kwargs['seed'])+8129)
        obs,info=self.env.reset(**kwargs);self.used=False;self.rejected=None
        if self.unwrapped.monitor.target_degrees>0. and self.prefix_rng.random()<self.probability:
            self.used=True
            for action in self.actions:
                obs,_,done,truncated,info=self.env.step(action)
                if done or truncated:
                    # Reject invalid auxiliary starts visibly; fixed formal
                    # evaluation never uses this wrapper or this selection.
                    self.rejected=info.get('failure_reason') or 'prefix_ended_episode'
                    self.used=False;obs,info=self.env.reset();break
        return obs,self.annotate(info)
    def step(self,action):
        obs,reward,done,truncated,info=self.env.step(action)
        return obs,reward,done,truncated,self.annotate(info)
