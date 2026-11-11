"""Frozen v1 physical/task/CPU defaults; saved with each run."""
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class PenSpinConfig:
    schema_version: int = 1
    task_mode: str = 'spin'
    grasp_min_fingers: int = 3
    grasp_acquisition_seconds: float = 0.3
    sim_dt: float = 0.002
    ctrl_dt: float = 0.05
    episode_seconds: float = 10.0
    contact_force: float = 0.02
    grip_velocity: float = 0.03
    grip_omega: float = 0.5
    capture_seconds: float = 0.5
    hold_seconds: float = 1.0
    initial_hold_seconds: float = 0.3
    free_contact_seconds: float = 0.15
    violation_seconds: float = 0.1
    max_plane_degrees: float = 45.0
    soft_plane_degrees: float = 15.0
    max_pivot_drift: float = 0.01
    max_thumb_axis_degrees: float = 30.0
    max_center_distance: float = 0.08
    max_axial_distance: float = 0.02
    min_endpoint_radius: float = 0.015
    contact_sector_seconds: float = 0.020
    angle_tolerance_degrees: float = 10.0
    capture_direction_degrees: float = 15.0
    max_penetration: float = 0.001
    action_delta_fraction: float = 0.05
    reset_position_noise: float = 0.001
    reset_angle_noise_degrees: float = 2.0
    num_envs: int = 8
    num_eval_envs: int = 4
    hidden_layers: tuple = (128, 128)
    unroll_length: int = 16
    batch_size: int = 32
    num_minibatches: int = 4
    num_updates_per_batch: int = 4
    learning_rate: float = 3e-4
    discounting: float = 0.99
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.2
    entropy_cost: float = 0.005
    max_grad_norm: float = 1.0
    stage_hours: tuple = (2.0, 4.0, 6.0, 12.0)
    budget_hours: float = 24.0
    validation_interval_seconds: float = 1800.0
    validation_episodes: int = 20
    promotion_success_rate: float = 0.8
    promotion_checks: int = 2
    stagnation_checks: int = 4
    seed: int = 0

    def __post_init__(self):
        if self.task_mode not in ('spin','grasp'):
            raise ValueError('task_mode must be spin or grasp')
        if not 2 <= self.grasp_min_fingers <= 5 or self.grasp_acquisition_seconds < 0:
            raise ValueError('Invalid grasp curriculum parameters')
        if self.sim_dt <= 0 or self.ctrl_dt <= 0:
            raise ValueError('Time steps must be positive')
        if abs(self.ctrl_dt/self.sim_dt-round(self.ctrl_dt/self.sim_dt)) > 1e-8:
            raise ValueError('ctrl_dt must be an integral multiple of sim_dt')
        if self.budget_hours <= 0 or self.budget_hours > 24:
            raise ValueError('First-batch budget must be in (0, 24] hours')
        if self.num_envs < 1 or self.batch_size*self.num_minibatches % self.num_envs:
            raise ValueError('PPO batch_size * num_minibatches must divide by num_envs')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        unknown = set(values)-{f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
        values = dict(values)
        for key in ('hidden_layers', 'stage_hours'):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2)+'\n')

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()
