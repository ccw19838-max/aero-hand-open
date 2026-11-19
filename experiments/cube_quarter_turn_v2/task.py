"""Full 90-degree endpoint task with bounded failure cost and stagnation detection."""
from dataclasses import dataclass
import numpy as np
from experiments.cube_quarter_turn.task import Config as V1Config, GoalMonitor as V1Monitor


@dataclass(frozen=True)
class Config(V1Config):
    version: str = 'xyz_quarter_turn_v2'
    failure_penalty: float = 10.
    stagnation_seconds: float = 2.5
    progress_threshold_degrees: float = 2.


class GoalMonitor(V1Monitor):
    def __init__(self, initial, axis, basis, config=None, ideal_initial=None):
        super().__init__(initial, axis, basis, config or Config(), ideal_initial)
        self.progress_anchor = self.error
        self.stagnant_seconds = 0.

    @property
    def phase(self):
        if self.stable > 0.: return 'hold'
        if self.error <= np.deg2rad(20.): return 'capture'
        return 'drive'

    def checkpoint(self, quaternion, contacts, linear, angular, displacement, dt=.01):
        if self.done: return 0.
        reward = super().checkpoint(quaternion, contacts, linear, angular, displacement, dt)
        # Keep all v1 physical gates and the full 90±5 / one-second endpoint.
        # The old -100 encourages postponing a failure by waiting to time out.
        if self.failure: reward += 100. - self.config.failure_penalty
        reward -= .8 * dt  # total time cost is now 1 unit per second
        force = float(sum(contacts['finger_forces']))
        reward -= .02 * dt * max(0., force / (self.config.cube_mass * 9.81) - 8.)
        valid=not(contacts['forbidden_support'] or contacts['max_penetration']>self.config.max_penetration
                  or displacement>self.config.max_displacement) and contacts['hand_force']>=self.config.contact_force
        if valid and self.progress_anchor-self.error >= np.deg2rad(self.config.progress_threshold_degrees):
            self.progress_anchor = self.error
            self.stagnant_seconds = 0.
        elif self.stable > 0.:
            self.stagnant_seconds = 0.
        else:
            self.stagnant_seconds += dt
        if not self.done and self.stagnant_seconds+1e-10 >= self.config.stagnation_seconds:
            self.failure = 'stagnation'
            reward -= self.config.failure_penalty
        return float(reward)
