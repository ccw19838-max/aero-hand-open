"""Versioned CPU grasp task settings; object properties are explicit assumptions."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class CubeGraspConfig:
    schema_version: int = 1
    cube_size: float = .057
    cube_mass: float = .077966253
    cube_friction: float = .6
    sim_dt: float = .002
    ctrl_dt: float = .05
    solver_iterations: int = 20
    episode_seconds: float = 5.
    hold_seconds: float = 2.
    contact_force: float = .02
    min_fingers: int = 3
    grip_velocity: float = .03
    grip_omega: float = .5
    max_displacement: float = .08
    max_penetration: float = .001
    violation_seconds: float = .1
    action_delta_fraction: float = .01
    reset_position_noise: float = .001
    reset_angle_noise_degrees: float = 2.
    reset_max_penetration: float = .001
    reset_max_attempts: int = 32
    mass_randomization: float = .1
    friction_randomization: float = .1
    render_width: int = 960
    render_height: int = 720

    def __post_init__(self):
        numeric = [v for v in asdict(self).values() if isinstance(v, (float, int))]
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError('Configuration contains non-finite values')
        if self.schema_version != 1:
            raise ValueError('Unsupported cube grasp schema')
        positive = ('cube_size', 'cube_mass', 'cube_friction', 'sim_dt', 'ctrl_dt',
                    'episode_seconds', 'hold_seconds', 'contact_force',
                    'grip_velocity', 'grip_omega', 'max_displacement',
                    'max_penetration', 'violation_seconds', 'action_delta_fraction')
        if any(getattr(self, name) <= 0 for name in positive):
            raise ValueError('Physical parameters and thresholds must be positive')
        ratio = self.ctrl_dt / self.sim_dt
        if not math.isclose(ratio, round(ratio), abs_tol=1e-9) or ratio < 1:
            raise ValueError('ctrl_dt must contain an integer number of physics steps')
        if not 2 <= self.min_fingers <= 5:
            raise ValueError('A grasp requires the thumb and at least one other finger')
        if self.hold_seconds > self.episode_seconds:
            raise ValueError('hold_seconds cannot exceed episode duration')
        if not 0 < self.action_delta_fraction <= .05:
            raise ValueError('Action increment exceeds the 5% range limit')
        if min(self.reset_position_noise, self.reset_angle_noise_degrees) < 0:
            raise ValueError('Reset noise cannot be negative')
        if self.reset_max_penetration < 0 or self.reset_max_attempts < 1:
            raise ValueError('Invalid reset validity limits')
        if not all(0 <= value < 1 for value in
                   (self.mass_randomization, self.friction_randomization)):
            raise ValueError('Physical randomization must be in [0, 1)')
        if self.solver_iterations < 1 or min(self.render_width, self.render_height) < 1:
            raise ValueError('Solver and image dimensions must be positive')

    @property
    def substeps(self):
        return round(self.ctrl_dt / self.sim_dt)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)
