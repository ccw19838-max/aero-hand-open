"""Training-only center and missing-fingertip distance costs, versioned v1.

Read-only geometry feedback fills the zero-force interval before contact. It
does not apply forces, move a body, change targets or alter goal/failure checks.
"""
from pathlib import Path
import hashlib
import gymnasium as gym
import numpy as np
from experiments.cube_contact_ppo.task import force_limits

TRANSFORM='center_missing_tip_mount_v1'
CENTER_SCALE=.02
CENTER_WEIGHT=1.
MISSING_TIP_SCALE=.02
MISSING_TIP_WEIGHT=1.


def descriptor():
    return dict(name=TRANSFORM,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mount_shaping_sha256=hashlib.sha256(Path(__file__).with_name('shaping.py').read_bytes()).hexdigest(),
        mount_margin_m=.005,mount_rate_weight=2.,
        center_scale_m=CENTER_SCALE,center_rate_weight=CENTER_WEIGHT,
        missing_tip_scale_m=MISSING_TIP_SCALE,missing_tip_rate_weight=MISSING_TIP_WEIGHT,
        center_squared_clip=16.,missing_tip_normalized_clip=2.,physical_changes=False,
        applies='non-warmup training reward only; original success and failure monitor')


class CaptureGeometryReward(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env);self.total_penalty=0.
    def components(self):
        base=self.unwrapped;lower,_=force_limits(base.config)
        center=CENTER_WEIGHT*min((base.displacement/CENTER_SCALE)**2,16.)
        missing=np.asarray(base.contacts['finger_forces'])<lower
        distances=np.maximum(np.asarray(base.tip_distances),0.)
        tip=MISSING_TIP_WEIGHT*float(np.mean(missing*np.clip(distances/MISSING_TIP_SCALE,0.,2.)**2))
        return float(center),float(tip)
    def reset(self,**kwargs):
        result=self.env.reset(**kwargs);self.total_penalty=0.;return result
    def step(self,action):
        base=self.unwrapped;before=base.data.time;warming=base.warming
        obs,reward,done,truncated,info=self.env.step(action)
        numerical=info.get('failure_reason')=='numerical'
        center,tip=self.components() if not numerical else (0.,0.)
        penalty=0. if warming or numerical else max(0.,float(base.data.time-before))*(center+tip)
        self.total_penalty+=penalty
        return obs,float(reward-penalty),done,truncated,{**info,
            'training_reward_transform':TRANSFORM,'geometry_center_rate':center,'geometry_missing_tip_rate':tip,
            'geometry_penalty':penalty,'geometry_penalty_total':self.total_penalty}
