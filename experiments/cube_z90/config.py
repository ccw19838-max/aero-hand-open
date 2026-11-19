"""Frozen, more forceful Z-only reward experiment, not a relaxed success test."""
from dataclasses import dataclass

from experiments.cube_envelop.config import CubeReorientConfig as EnvelopeConfig
from experiments.cube_grasp.config import CubeGraspConfig

ANGLES = (5., 10., 15., 20., 30., 45., 60., 75., 90.)
AXES = ('z',)


def tolerance_for(angle):
    return min(5., float(angle) / 5.)


@dataclass(frozen=True)
class CubeReorientConfig(EnvelopeConfig):
    reward_version: str = 'z90_aggressive_v1'
    training_axes: tuple = ('z',)
    target_degrees: float = 5.
    angle_tolerance_degrees: float = 1.
    target_hold_seconds: float = 1.
    episode_seconds: float = 16.
    action_delta_fraction: float = .02
    progress_reward: float = 80.
    near_goal_reward: float = 4.
    stable_reward: float = 20.
    success_reward: float = 500.
    failure_penalty: float = 100.
    capture_reward: float = 2.
    swing_penalty: float = 2.
    velocity_tracking_penalty: float = .3
    time_penalty: float = .5
    initial_capture_controller: str = 'fixed_reference_control'

    def __post_init__(self):
        object.__setattr__(self, 'training_axes', tuple(self.training_axes))
        object.__setattr__(self, 'hold_seconds', self.target_hold_seconds)
        CubeGraspConfig.__post_init__(self)
        if self.reward_version != 'z90_aggressive_v1':
            raise ValueError('Unknown reward version')
        if self.training_axes != ('z',) or self.task_mode != 'reorient':
            raise ValueError('This experiment only trains positive Z reorientation')
        if self.variant != 'four_finger_envelope_v1' or self.min_fingers != 5:
            raise ValueError('Five-finger acquisition and recapture remain mandatory')
        if self.goal_frame != 'hand_fixed' or self.task != 'fixed_hand_xyz_reorient_v1':
            raise ValueError('Expected the original fixed hand reference axes')
        if self.target_degrees not in ANGLES:
            raise ValueError('Unsupported curriculum angle')
        if not 0 < self.angle_tolerance_degrees <= tolerance_for(self.target_degrees):
            raise ValueError('The target tolerance cannot admit standing still')
        if self.target_hold_seconds < 1.:
            raise ValueError('Require at least one continuous second of stable recapture')
        if not 0 < self.capture_seconds <= self.capture_timeout < self.episode_seconds:
            raise ValueError('Invalid capture timing')
        if self.capture_seconds + self.target_hold_seconds > self.episode_seconds:
            raise ValueError('Episode is too short')
        if self.initial_capture_controller != 'fixed_reference_control':
            raise ValueError('Unknown acquisition controller')
        if min(self.max_no_finger_contact_seconds, self.thumb_force_soft_limit) <= 0:
            raise ValueError('Invalid contact limits')
        for name in ('progress_reward', 'near_goal_reward', 'stable_reward',
                     'success_reward', 'failure_penalty', 'capture_reward',
                     'swing_penalty', 'velocity_tracking_penalty', 'time_penalty',
                     'thumb_force_penalty', 'thumb_contact_reward'):
            if getattr(self, name) < 0:
                raise ValueError(f'Negative reward weight: {name}')
        # Even a full episode of near-goal/stable shaping cannot pay more than
        # completing the task. This discourages repeatedly breaking the timer.
        if self.success_reward <= self.episode_seconds * (self.near_goal_reward+self.stable_reward+2.05):
            raise ValueError('Terminal success must dominate lingering near the goal')

    def angle_for(self, axis):
        if axis != 'z':
            raise ValueError('This experiment has no X/Y goals')
        return self.target_degrees
