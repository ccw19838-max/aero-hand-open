"""Small-angle, independently promoted hand-axis targets; old tasks stay frozen."""
from dataclasses import dataclass
import math

from experiments.cube_envelop.config import CubeReorientConfig as EnvelopeConfig
from experiments.cube_grasp.config import CubeGraspConfig

ANGLES=(5.,10.,15.,30.,45.,60.,90.)
AXES=('x','y','z')


def tolerance_for(angle):
    return max(1.,min(10.,float(angle)/5.))


@dataclass(frozen=True)
class CubeReorientConfig(EnvelopeConfig):
    curriculum_version: str = 'gradual_axes_v1'
    training_axes: tuple = ('z',)
    target_angles: tuple = (5.,5.,5.)
    target_degrees: float = 5.
    angle_tolerance_degrees: float = 1.

    def __post_init__(self):
        # The legacy reorientation class intentionally accepts only 90 degrees.
        # Revalidate its physics and envelope invariants explicitly here rather
        # than relaxing old checkpoints' meaning or modifying their source.
        object.__setattr__(self,'training_axes',tuple(self.training_axes))
        object.__setattr__(self,'target_angles',tuple(float(v) for v in self.target_angles))
        object.__setattr__(self,'hold_seconds',self.target_hold_seconds)
        CubeGraspConfig.__post_init__(self)
        if self.curriculum_version!='gradual_axes_v1':raise ValueError('Unknown curriculum')
        if self.task!='fixed_hand_xyz_reorient_v1' or self.goal_frame!='hand_fixed':raise ValueError('Wrong goal frame')
        if self.variant!='four_finger_envelope_v1' or self.min_fingers!=5:raise ValueError('Five-finger envelope required')
        if not self.training_axes or len(set(self.training_axes))!=len(self.training_axes) or any(a not in AXES for a in self.training_axes):
            raise ValueError('Training axes must be a nonempty unique selection of x/y/z')
        if len(self.target_angles)!=3 or any(v not in ANGLES for v in self.target_angles):raise ValueError('Invalid axis targets')
        if self.target_degrees not in ANGLES:raise ValueError('Unsupported target angle')
        if not 0<self.angle_tolerance_degrees<=tolerance_for(self.target_degrees):raise ValueError('Small-angle tolerance must exclude zero motion')
        if self.task_mode not in ('hold','reorient'):raise ValueError('Invalid mode')
        if not 0<self.capture_seconds<=self.capture_timeout<=self.episode_seconds:raise ValueError('Invalid capture time')
        if not 0<self.target_hold_seconds<=self.episode_seconds:raise ValueError('Invalid recapture time')
        if self.capture_seconds+self.grip_hold_seconds>self.episode_seconds:raise ValueError('Invalid hold timing')
        if min(self.max_no_finger_contact_seconds,self.thumb_force_soft_limit,self.grip_hold_seconds)<=0:raise ValueError('Invalid contact thresholds')
        if min(self.progress_reward,self.near_goal_reward,self.stable_reward,self.envelope_contact_reward,
               self.envelope_stable_reward,self.thumb_contact_reward,self.thumb_force_penalty)<0:raise ValueError('Negative reward weight')

    def angle_for(self,axis):return self.target_angles[AXES.index(axis)]


# Only these settings may change when transferring between angle levels. The
# underlying hand, object, observation shape, controls and contacts must match.
TRANSFER_FIELDS={'task_mode','curriculum_version','training_axes','target_angles',
                 'target_degrees','angle_tolerance_degrees'}
