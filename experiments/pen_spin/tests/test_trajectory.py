"""Task semantics and anti-cheating tests, independent of physics engines."""
import dataclasses
import unittest

import jax
import jax.numpy as jp
import numpy as np

from experiments.pen_spin.config import PenSpinConfig
from experiments.pen_spin.trajectory import (
    Sample, initialize_trace, advance_trace, HOLD, SPIN, CATCH, SUCCESS, FAILED,
    SUPPORT, PENETRATION, LOST_CONTACT, ENVELOPE, NONFINITE,
)

DT = .002


def sequence(goal=2*np.pi, kind='circle', count=2000):
    """A tangent rod circles a fixed fulcrum, then comes to rest."""
    t = np.arange(count)*DT
    angle = np.clip((t-.35)/3., 0., 1.)*goal
    active = (t > .35) & (t < 3.35)
    if kind == 'static' or kind == 'axial':
        angle[:] = 0.
    if kind == 'reverse':
        angle = np.where(t < 1.85,
                         np.clip((t-.35)/1.5, 0., 1.)*np.pi,
                         np.clip(1.-(t-1.85)/1.5, 0., 1.)*np.pi)
    direction = np.c_[np.cos(angle), np.sin(angle), np.zeros(count)]
    center = np.c_[-.02*np.sin(angle), .02*np.cos(angle), np.zeros(count)]
    angular = np.c_[np.zeros(count), np.zeros(count), active*goal/3.]
    linear = np.c_[-.02*np.cos(angle), -.02*np.sin(angle), np.zeros(count)]*(active*goal/3.)[:, None]
    if kind == 'static':
        angular[:] = 0.; linear[:] = 0.
    if kind == 'axial':
        angular[:] = 0.; angular[active, 0] = 5.; linear[:] = 0.
    return Sample(
        center=jp.asarray(center, dtype=jp.float32),
        endpoint_a=jp.asarray(center+.065*direction, dtype=jp.float32),
        direction=jp.asarray(direction, dtype=jp.float32),
        linvel=jp.asarray(linear, dtype=jp.float32),
        angvel=jp.asarray(angular, dtype=jp.float32),
        thumb_center=jp.zeros((count, 3), dtype=jp.float32),
        finger_forces=jp.tile(jp.array([.1, .1, 0., 0., 0.]), (count, 1)),
        invalid_contact=jp.zeros(count, dtype=bool),
        penetration=jp.zeros(count, dtype=bool),
    )


def _run(samples, goal, config):
    first = jax.tree.map(lambda v: v[0], samples)
    trace = initialize_trace(first, jp.array([0., 0., 1.]), jp.zeros(3), goal, config)

    def step(trace, sample):
        state, reward, metrics = advance_trace(trace, sample, DT, config)
        return state, (reward, metrics)

    return jax.lax.scan(step, trace, samples)


run = jax.jit(_run, static_argnames=('config',))


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.config = PenSpinConfig()

    def execute(self, samples, goal=2*np.pi, config=None):
        state, (rewards, metrics) = run(samples, jp.asarray(goal, dtype=jp.float32), config or self.config)
        state.elapsed.block_until_ready()
        return state, np.asarray(rewards), jax.tree.map(np.asarray, metrics)

    def test_complete_turn_and_continuous_capture(self):
        state, rewards, _ = self.execute(sequence())
        self.assertTrue(bool(state.success))
        self.assertEqual(int(state.phase), SUCCESS)
        self.assertGreaterEqual(float(state.stable_time), .5-1e-6)
        self.assertTrue(np.all(np.asarray(state.sector_contact) >= .02-1e-6))
        # A single terminal bonus, with absorbing zero reward thereafter.
        self.assertEqual(int(np.sum(rewards > 40)), 1)
        self.assertTrue(np.all(rewards[-50:] == 0))

    def test_quarter_turn_captures_at_goal_orientation(self):
        state, _, _ = self.execute(sequence(goal=np.pi/2), goal=np.pi/2)
        self.assertTrue(bool(state.success))
        self.assertLess(float(state.theta), 1.7)

    def test_static_hold_stage_is_separate_from_spin(self):
        samples = sequence(kind='static')
        held, _, _ = self.execute(samples, goal=0.)
        still, _, _ = self.execute(samples)
        self.assertTrue(bool(held.success))
        self.assertAlmostEqual(float(held.elapsed), 1., places=4)
        self.assertFalse(bool(still.success))

    def test_axial_spinning_is_not_a_thumb_around(self):
        state, _, _ = self.execute(sequence(kind='axial'))
        self.assertFalse(bool(state.success))
        self.assertAlmostEqual(float(state.theta), 0., places=5)
        self.assertAlmostEqual(float(state.phi), 0., places=5)

    def test_throw_without_contacts_is_failure(self):
        samples = sequence()
        forces = samples.finger_forces.at[220:].set(0.)
        state, _, _ = self.execute(samples.replace(finger_forces=forces))
        self.assertFalse(bool(state.success))
        self.assertEqual(int(state.failure_reason), LOST_CONTACT)

    def test_reversing_before_a_full_turn_does_not_complete(self):
        state, _, _ = self.execute(sequence(kind='reverse'))
        self.assertFalse(bool(state.success))
        self.assertAlmostEqual(float(state.theta), 0., places=4)
        self.assertAlmostEqual(float(state.phi), 0., places=4)

    def test_moving_thumb_cannot_move_the_reference_axis(self):
        samples = sequence()
        thumb = samples.thumb_center.at[220:, 0].set(.025)
        state, _, _ = self.execute(samples.replace(thumb_center=thumb))
        self.assertEqual(int(state.failure_reason), ENVELOPE)
        np.testing.assert_array_equal(np.asarray(state.c0), [0., 0., 0.])

    def test_rotating_thumb_about_fixed_midpoint_fails(self):
        samples = sequence()
        thumb_direction = jp.tile(jp.array([0., 0., 1.]), (2000, 1))
        thumb_direction = thumb_direction.at[220:].set(
            jp.array([jp.sin(jp.pi/4), 0., jp.cos(jp.pi/4)]))
        state, _, metrics = self.execute(samples.replace(thumb_direction=thumb_direction))
        self.assertEqual(int(state.failure_reason), ENVELOPE)
        # The midpoint never moved; this must be the angular envelope gate.
        np.testing.assert_array_equal(np.asarray(samples.thumb_center), 0.)
        self.assertAlmostEqual(float(metrics['thumb_axis_degrees'][-1]), 45., places=4)
        self.assertGreaterEqual(float(state.violation_time), .1-1e-6)
        self.assertLess(float(state.elapsed), .55)

    def test_support_and_penetration_fail_immediately(self):
        samples = sequence(kind='static')
        for field, reason in [('invalid_contact', SUPPORT), ('penetration', PENETRATION)]:
            with self.subTest(field=field):
                bad = samples.replace(**{field: jp.arange(2000) >= 50})
                state, rewards, _ = self.execute(bad)
                self.assertEqual(int(state.failure_reason), reason)
                self.assertLess(float(state.elapsed), .11)
                self.assertEqual(int(np.sum(rewards < -20)), 1)

    def test_angle_wrap_boundary_is_not_a_reset(self):
        state, _, metrics = self.execute(sequence())
        self.assertGreater(float(state.theta), 6.1)
        self.assertGreater(float(state.phi), 6.1)
        self.assertLess(float(np.max(np.abs(np.diff(metrics['theta'])))), .01)

    def test_singular_radius_does_not_reanchor_a_free_turn(self):
        samples = sequence()
        endpoint = samples.endpoint_a.at[300:303].set(jp.array([0., 0., .01]))
        state, _, _ = self.execute(samples.replace(endpoint_a=endpoint))
        self.assertFalse(bool(state.success))
        self.assertFalse(bool(state.trajectory_proven))

    def test_projection_singularity_and_plane_violation_fail(self):
        samples = sequence()
        direction = samples.direction.at[220:].set(jp.array([0., 0., 1.]))
        state, _, _ = self.execute(samples.replace(direction=direction))
        self.assertEqual(int(state.failure_reason), ENVELOPE)

    def test_negative_progress_is_charged_outside_envelope(self):
        samples = sequence()
        first = jax.tree.map(lambda x: x[0], samples)
        trace = initialize_trace(first, jp.array([0., 0., 1.]), jp.zeros(3), 2*jp.pi, self.config).replace(phase=jp.array(SPIN))

        def sample_at(angle, drift):
            direction = jp.array([jp.cos(angle), jp.sin(angle), 0.])
            center = .02*jp.array([-jp.sin(angle), jp.cos(angle), 0.])
            return first.replace(direction=direction, center=center,
                                 endpoint_a=center+.065*direction,
                                 thumb_center=jp.array([drift, 0., 0.]))

        forward, _, _ = advance_trace(trace, sample_at(.5, 0.), DT, self.config)
        back, reward, _ = advance_trace(forward, sample_at(.25, .02), DT, self.config)
        self.assertLess(float(back.potential), float(forward.potential))
        self.assertLess(float(reward), -.5)

    def test_valid_progress_metric_masks_out_of_region_angles(self):
        samples = sequence(kind='static')
        first = jax.tree.map(lambda value: value[0], samples)
        trace = initialize_trace(first, jp.array([0., 0., 1.]), jp.zeros(3),
                                 2*jp.pi, self.config).replace(phase=jp.array(SPIN))

        def sample_at(angle, drift):
            direction = jp.array([jp.cos(angle), jp.sin(angle), 0.])
            center = .02*jp.array([-jp.sin(angle), jp.cos(angle), 0.])
            return first.replace(direction=direction, center=center,
                                 endpoint_a=center+.065*direction,
                                 thumb_center=jp.array([drift, 0., 0.]))

        valid, _, valid_metrics = advance_trace(trace, sample_at(.5, 0.), DT, self.config)
        invalid, _, invalid_metrics = advance_trace(valid, sample_at(1., .02), DT, self.config)
        self.assertAlmostEqual(float(valid_metrics['valid_progress_radians']), .5, places=6)
        self.assertAlmostEqual(float(invalid.potential), 1., places=6)
        self.assertEqual(float(invalid_metrics['valid_progress_radians']), 0.)

    def test_custom_contact_threshold_is_used(self):
        cfg = dataclasses.replace(self.config, contact_force=.2)
        state, _, _ = self.execute(sequence(kind='static'), goal=0., config=cfg)
        self.assertEqual(int(state.failure_reason), LOST_CONTACT)

    def test_isolated_thumb_contact_earns_contact_shaping(self):
        samples = sequence(kind='static')
        first = jax.tree.map(lambda value: value[0], samples)
        trace = initialize_trace(first, jp.array([0., 0., 1.]), jp.zeros(3),
                                 2*jp.pi, self.config).replace(phase=jp.array(SPIN))
        thumb_only = first.replace(finger_forces=jp.array([.1, 0., 0., 0., 0.]))
        no_contact = first.replace(finger_forces=jp.zeros(5))
        _, thumb_reward, thumb_metrics = advance_trace(trace, thumb_only, DT, self.config)
        _, free_reward, _ = advance_trace(trace, no_contact, DT, self.config)
        # A thumb-only pivot is useful contact during rotation, but is not a
        # stable two-finger grasp and must not earn the separate holding term.
        self.assertEqual(float(thumb_metrics['valid_grip']), 0.)
        self.assertAlmostEqual(float(thumb_reward), DT*(.2-.02), places=7)
        self.assertAlmostEqual(float(thumb_reward-free_reward), DT*.2, places=7)

    def test_capture_hold_reward_requires_target_orientation(self):
        samples = sequence(kind='static')
        first = jax.tree.map(lambda value: value[0], samples)
        trace = initialize_trace(first, jp.array([0., 0., 1.]), jp.zeros(3),
                                 2*jp.pi, self.config).replace(
            phase=jp.array(CATCH), theta=jp.array(2*jp.pi), phi=jp.array(2*jp.pi),
            potential=jp.array(2*jp.pi), sector_contact=jp.full(4, .03))
        wrong_direction = jp.array([0., 1., 0.])
        wrong_center = jp.array([-.02, 0., 0.])
        wrong = first.replace(direction=wrong_direction, center=wrong_center,
                              endpoint_a=wrong_center+.065*wrong_direction)
        wrong_state, wrong_reward, wrong_metrics = advance_trace(trace, wrong, DT, self.config)
        right_state, right_reward, _ = advance_trace(trace, first, DT, self.config)
        # Both grasps are stable and retain full winding progress. Only the
        # requested final orientation establishes the capture/holding reward.
        self.assertEqual(float(wrong_metrics['valid_grip']), 1.)
        self.assertEqual(float(wrong_state.stable_time), 0.)
        self.assertAlmostEqual(float(right_state.stable_time), DT, places=7)
        self.assertAlmostEqual(float(wrong_reward), DT*(.2-.02), places=7)
        self.assertAlmostEqual(float(right_reward-wrong_reward), DT, places=7)

    def test_nonfinite_sample_is_a_finite_terminal_failure(self):
        samples = sequence()
        bad = samples.replace(center=samples.center.at[220, 0].set(jp.nan))
        state, rewards, metrics = self.execute(bad)
        self.assertEqual(int(state.failure_reason), NONFINITE)
        self.assertTrue(np.isfinite(rewards).all())
        self.assertTrue(all(np.isfinite(v).all() for v in metrics.values()))


if __name__ == '__main__':
    unittest.main()
