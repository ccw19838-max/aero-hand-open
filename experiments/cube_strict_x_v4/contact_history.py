"""Strict assessment on the 10 ms active time grid, in the fixed task frame."""
from collections import deque
from dataclasses import dataclass

import numpy as np

NAMES = ('index', 'middle', 'ring')
SAMPLE_SECONDS = 0.01


@dataclass(frozen=True)
class Violation:
    time_s: float
    angle_deg: float
    fingers: tuple[str, ...]


class ContactHistory:
    """One instance per episode. All predictors use present and past forces only."""
    def __init__(self):
        self.samples = deque(maxlen=6)
        self.warnings = np.zeros(3, dtype=bool)
        self.margins = np.zeros(3)
        self.low_ticks = np.zeros(3, dtype=int)
        self.safe_ticks = np.zeros(3, dtype=int)
        self.pending_onset = [None] * 3
        self.warning_count = self.warning_false_count = 0
        self.warning_lead_ms = []
        self.first_violation = None
        self.valid_peak_deg = self.fraction_record = 0.
        self.max_hold = self.hold = 0.
        self.sample_count = 0

    def sample(self, forces, angle, goal_error, target, previous_angle, previous_error,
               physical_ok, hold_seconds):
        """Angles in degrees; returns newly earned, eligible fraction of this goal."""
        self.sample_count += 1
        values = np.asarray(forces, dtype=float)[1:4].copy()
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError('Expected finite index, middle and ring forces')
        self.samples.append(values)
        predicted = 2 * values - self.samples[0] if len(self.samples) == 6 else values
        self.margins = np.clip((predicted - .1) / .1, -1., 1.)
        # The policy sees the three margins immediately, even before the two-tick warning.
        low = (len(self.samples) == 6) & (values < .3) & (predicted < .1)
        safe = (values >= .2) & (predicted >= .1)
        self.low_ticks = np.where(low, self.low_ticks + 1, 0)
        self.safe_ticks = np.where(safe, self.safe_ticks + 1, 0)
        previous_warnings = self.warnings.copy()
        self.warnings |= self.low_ticks >= 2
        self.warnings &= self.safe_ticks < 5
        for i in range(3):
            if self.warnings[i] and not previous_warnings[i]:
                self.warning_count += 1
                self.pending_onset[i] = self.sample_count
            onset = self.pending_onset[i]
            if onset is None:
                continue
            age = self.sample_count - onset
            if values[i] < .1 and age <= 5:
                self.warning_lead_ms.append(10 * age)
                self.pending_onset[i] = None
            elif age >= 5:
                self.warning_false_count += 1
                self.pending_onset[i] = None
        invalid = tuple(name for name, force in zip(NAMES, values) if not .1 <= force <= 12.)
        if invalid and self.first_violation is None:
            self.first_violation = Violation(self.sample_count * SAMPLE_SECONDS, float(angle), invalid)
        eligible = self.first_violation is None and physical_ok
        self.hold = float(hold_seconds) if eligible else 0.
        self.max_hold = max(self.hold, self.max_hold)
        if eligible:
            self.valid_peak_deg = max(self.valid_peak_deg, float(angle))
        fraction = max(0., min(1., angle / target, 1. - goal_error / target)) if target else 0.
        gain = max(0., fraction - self.fraction_record)
        self.fraction_record = max(self.fraction_record, fraction)
        return gain if eligible and angle > previous_angle and goal_error < previous_error else 0.

    def info(self):
        first = self.first_violation
        return dict(eligible_peak_degrees=self.valid_peak_deg,
                    first_violation=None if first is None else dict(time_s=first.time_s,
                            angle_deg=first.angle_deg, fingers=list(first.fingers)),
                    warning_count=self.warning_count, warning_false_count=self.warning_false_count,
                    warning_censored_count=sum(onset is not None for onset in self.pending_onset),
                    warning_lead_ms=list(self.warning_lead_ms), force_margins=self.margins.tolist(),
                    support_warnings=self.warnings.tolist(), strict_hold_seconds=self.hold,
                    max_strict_hold_seconds=self.max_hold)
