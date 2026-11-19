"""A separate task configuration; existing vertical-grasp checkpoints stay valid."""
from dataclasses import dataclass

from experiments.cube_grasp.config import CubeGraspConfig


@dataclass(frozen=True)
class CubeReorientConfig(CubeGraspConfig):
    task: str = 'fixed_hand_xyz_reorient_v1'
    goal_frame: str = 'hand_fixed'
    episode_seconds: float = 10.
    hold_seconds: float = .5
    target_degrees: float = 90.
    target_hold_seconds: float = .5
    angle_tolerance_degrees: float = 10.
    progress_reward: float = 5.
    near_goal_reward: float = .5
    stable_reward: float = 2.

    def __post_init__(self):
        # The inherited field remains an explicit alias for utilities that
        # display a task's holding interval. Reorientation uses target_hold_seconds.
        object.__setattr__(self, 'hold_seconds', self.target_hold_seconds)
        super().__post_init__()
        if self.task != 'fixed_hand_xyz_reorient_v1' or self.goal_frame != 'hand_fixed':
            raise ValueError('Unsupported reorientation task')
        if self.target_degrees != 90.:
            raise ValueError('This task is a single positive 90-degree turn')
        if not 0 < self.target_hold_seconds <= self.episode_seconds:
            raise ValueError('Invalid goal holding interval')
        if not 0 < self.angle_tolerance_degrees < self.target_degrees:
            raise ValueError('Angular tolerance must distinguish the initial pose')
        if min(self.progress_reward, self.near_goal_reward, self.stable_reward) < 0:
            raise ValueError('Reward weights cannot be negative')
