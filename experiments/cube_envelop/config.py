"""Five-contact envelope acquisition, reorientation, and recapture settings."""
from dataclasses import dataclass

from experiments.cube_reorient.config import CubeReorientConfig as ReorientConfig


@dataclass(frozen=True)
class CubeReorientConfig(ReorientConfig):
    variant: str = 'four_finger_envelope_v1'
    task_mode: str = 'reorient'
    episode_seconds: float = 12.
    min_fingers: int = 5
    capture_seconds: float = .5
    capture_timeout: float = 2.
    grip_hold_seconds: float = 2.
    max_no_finger_contact_seconds: float = .15
    envelope_contact_reward: float = .5
    envelope_stable_reward: float = 2.
    thumb_contact_reward: float = .1
    thumb_force_soft_limit: float = 20.
    thumb_force_penalty: float = .005

    def __post_init__(self):
        super().__post_init__()
        if self.variant != 'four_finger_envelope_v1':
            raise ValueError('Unsupported envelope task variant')
        if self.task_mode not in ('hold', 'reorient'):
            raise ValueError('task_mode must be hold or reorient')
        if self.min_fingers != 5:
            raise ValueError('Envelope acquisition and recapture require all five fingers')
        if not 0 < self.capture_seconds <= self.capture_timeout <= self.episode_seconds:
            raise ValueError('Invalid initial envelope acquisition timing')
        if not 0 < self.grip_hold_seconds < self.episode_seconds:
            raise ValueError('Invalid holding-only interval')
        if self.capture_seconds + self.grip_hold_seconds > self.episode_seconds:
            raise ValueError('Episode cannot fit acquisition followed by holding')
        if min(self.max_no_finger_contact_seconds, self.thumb_force_soft_limit) <= 0:
            raise ValueError('Contact-loss time and thumb force scale must be positive')
        if min(self.envelope_contact_reward, self.envelope_stable_reward,
               self.thumb_contact_reward, self.thumb_force_penalty) < 0:
            raise ValueError('Envelope reward weights cannot be negative')
