"""Shared, JIT-compatible trajectory judgment and reward for PenSpin v1.

This module is independent of MuJoCo/Playground: native and MJX adapters supply
the same physical Sample on every integration substep. Distances are metres,
forces newtons, angles radians, and velocities SI. Action-change cost is applied
once per control step by the environment, not here.
"""
from flax import struct
import jax
import jax.numpy as jp

HOLD, SPIN, CATCH, SUCCESS, FAILED = range(5)
NO_FAILURE, SUPPORT, PENETRATION, LOST_CONTACT, ENVELOPE, TIMEOUT, NONFINITE = range(7)
FAILURE_NAMES = {
    NO_FAILURE: "none", SUPPORT: "invalid_support", PENETRATION: "penetration",
    LOST_CONTACT: "lost_contact", ENVELOPE: "trajectory_envelope",
    TIMEOUT: "timeout", NONFINITE: "nonfinite_state",
}


@struct.dataclass
class Sample:
    center: jax.Array
    endpoint_a: jax.Array
    direction: jax.Array
    linvel: jax.Array
    angvel: jax.Array
    thumb_center: jax.Array
    finger_forces: jax.Array
    invalid_contact: jax.Array
    penetration: jax.Array
    # Native/MJX adapters provide IP→tip. None keeps legacy synthetic samples
    # aligned with their fixed reference axis rather than inventing a vector.
    thumb_direction: jax.Array | None = None
    finger_distances: jax.Array | None = None


@struct.dataclass
class TraceState:
    n0: jax.Array
    c0: jax.Array
    e1: jax.Array
    e2: jax.Array
    initial_direction: jax.Array
    initial_radius: jax.Array
    raw_theta: jax.Array
    raw_phi: jax.Array
    theta: jax.Array
    phi: jax.Array
    potential: jax.Array
    sector_contact: jax.Array
    substeps: jax.Array
    elapsed: jax.Array
    initial_hold_time: jax.Array
    stable_time: jax.Array
    no_contact_time: jax.Array
    violation_time: jax.Array
    phase: jax.Array
    done: jax.Array
    success: jax.Array
    failure_reason: jax.Array
    goal_angle: jax.Array
    previous_angles_valid: jax.Array
    trajectory_proven: jax.Array


def _cfg(config, name, default):
    # Explicit configurations must contain the actual field: misspellings must
    # not silently replace an experiment's settings with defaults.
    return default if config is None else getattr(config, name)


def _unit(vector):
    vector = jp.asarray(vector, dtype=jp.float32)
    return vector / jp.maximum(jp.linalg.norm(vector), 1e-8)


def _wrapped_delta(new, old):
    return jp.arctan2(jp.sin(new-old), jp.cos(new-old))


def _angles(sample, n0, c0, e1, e2):
    direction = _unit(sample.direction)
    dp = direction-jp.dot(direction, n0)*n0
    endpoint = jp.asarray(sample.endpoint_a)-c0
    ep = endpoint-jp.dot(endpoint, n0)*n0
    return (jp.arctan2(jp.dot(dp, e2), jp.dot(dp, e1)),
            jp.arctan2(jp.dot(ep, e2), jp.dot(ep, e1)),
            jp.linalg.norm(dp), jp.linalg.norm(ep))


def initialize_trace(sample, n0, c0, goal_angle, config=None):
    """Start a fresh episode; goal 0 is the static-hold curriculum stage."""
    n0 = _unit(n0)
    c0 = jp.asarray(c0, dtype=jp.float32)
    direction = _unit(sample.direction)
    projected = direction-jp.dot(direction, n0)*n0
    fallback_axis = jp.where(jp.abs(n0[0]) < .9, jp.array([1., 0., 0.]),
                            jp.array([0., 1., 0.]))
    fallback = fallback_axis-jp.dot(fallback_axis, n0)*n0
    e1 = _unit(jp.where(jp.linalg.norm(projected) > 1e-6, projected, fallback))
    e2 = _unit(jp.cross(n0, e1))
    raw_theta, raw_phi, projection, radius = _angles(sample, n0, c0, e1, e2)
    valid = (projection > 1e-6) & (radius >= _cfg(config, 'min_endpoint_radius', .015))
    zero = jp.array(0., dtype=jp.float32)
    return TraceState(
        n0=n0, c0=c0, e1=e1, e2=e2, initial_direction=direction,
        initial_radius=radius, raw_theta=raw_theta, raw_phi=raw_phi,
        theta=zero, phi=zero, potential=zero,
        sector_contact=jp.zeros(4, dtype=jp.float32),
        substeps=jp.array(0, dtype=jp.int32), elapsed=zero,
        initial_hold_time=zero, stable_time=zero, no_contact_time=zero,
        violation_time=zero, phase=jp.array(HOLD, dtype=jp.int32),
        done=jp.array(False), success=jp.array(False),
        failure_reason=jp.array(NO_FAILURE, dtype=jp.int32),
        goal_angle=jp.asarray(goal_angle, dtype=jp.float32),
        previous_angles_valid=valid, trajectory_proven=valid,
    )


def advance_trace(trace, sample, dt, config=None):
    """Advance one physical substep, including winding/contact and events.

    A terminal trace is absorbing and earns zero further reward. A lost angular
    observation cannot be repaired by inventing a winding angle on re-entry.
    """
    dt = jp.asarray(dt, dtype=jp.float32)
    time_epsilon = dt*.02  # float32 timer accumulation, much smaller than a substep.
    direction = _unit(sample.direction)
    theta_raw, phi_raw, projection, radius = _angles(
        sample, trace.n0, trace.c0, trace.e1, trace.e2)
    angles_valid = (projection > 1e-6) & (radius >= _cfg(config, 'min_endpoint_radius', .015))
    active_spin = (trace.phase == SPIN) | (trace.phase == CATCH)
    observable_step = angles_valid & trace.previous_angles_valid
    dtheta = jp.where(active_spin & observable_step,
                     _wrapped_delta(theta_raw, trace.raw_theta), 0.)
    dphi = jp.where(active_spin & observable_step,
                   _wrapped_delta(phi_raw, trace.raw_phi), 0.)
    theta, phi = trace.theta+dtheta, trace.phi+dphi
    potential = jp.clip(jp.minimum(theta, phi), 0., jp.maximum(trace.goal_angle, 0.))

    relative = jp.asarray(sample.center)-trace.c0
    axial = jp.abs(jp.dot(relative, trace.n0))
    center_distance = jp.linalg.norm(relative)
    thumb_relative = jp.asarray(sample.thumb_center)-trace.c0
    pivot_drift = jp.linalg.norm(thumb_relative-jp.dot(thumb_relative, trace.n0)*trace.n0)
    thumb_direction = (trace.n0 if sample.thumb_direction is None
                       else _unit(sample.thumb_direction))
    thumb_axis_degrees = jp.rad2deg(jp.arccos(
        jp.clip(jp.dot(thumb_direction, trace.n0), -1., 1.)))
    plane_angle = jp.arcsin(jp.clip(jp.abs(jp.dot(direction, trace.n0)), 0., 1.))
    plane_degrees = jp.rad2deg(plane_angle)
    envelope_valid = (
        angles_valid
        & (plane_degrees <= _cfg(config, 'max_plane_degrees', 45.))
        & (pivot_drift <= _cfg(config, 'max_pivot_drift', .01))
        & (thumb_axis_degrees <= _cfg(config, 'max_thumb_axis_degrees', 30.))
        & (center_distance <= _cfg(config, 'max_center_distance', .08))
        & (axial <= _cfg(config, 'max_axial_distance', .02))
    )
    forces = jp.asarray(sample.finger_forces)
    contacts = forces >= _cfg(config, 'contact_force', .02)
    thumb_contact = contacts[0]
    has_contact = jp.any(contacts)
    two_finger = thumb_contact & jp.any(contacts[1:])
    speed_ok = ((jp.linalg.norm(sample.linvel) < _cfg(config, 'grip_velocity', .03))
                & (jp.linalg.norm(sample.angvel) < _cfg(config, 'grip_omega', .5)))
    unsafe_support = jp.asarray(sample.invalid_contact, dtype=bool)
    penetrated = jp.asarray(sample.penetration, dtype=bool)
    geometry_finite = jp.all(jp.isfinite(jp.concatenate([
        jp.asarray(sample.center), jp.asarray(sample.endpoint_a), direction,
        jp.asarray(sample.linvel), jp.asarray(sample.angvel),
        jp.asarray(sample.thumb_center), thumb_direction, forces,
    ])))
    valid_grip = two_finger & speed_ok & envelope_valid & ~unsafe_support & ~penetrated
    no_contact_time = jp.where(has_contact, 0., trace.no_contact_time+dt)
    violation_time = jp.where(envelope_valid, 0., trace.violation_time+dt)
    substeps = trace.substeps+1
    elapsed = substeps*dt

    # A near-axis angular singularity is unobservable even if it lasts less than
    # the hard-envelope timeout. Retain evidence that a complete turn has not
    # been established; do not re-anchor it as a successful turn.
    proven = jp.where(trace.phase == HOLD, angles_valid,
                      trace.trajectory_proven & angles_valid)
    sector_index = jp.clip(jp.floor(theta/(jp.pi/2)).astype(jp.int32), 0, 3)
    sector_active = (active_spin & thumb_contact & envelope_valid
                     & (theta >= 0.) & (theta <= trace.goal_angle+1e-5))
    sector_contact = trace.sector_contact.at[sector_index].add(
        jp.where(sector_active, dt, 0.))
    required = jp.clip(jp.ceil(trace.goal_angle/(jp.pi/2)).astype(jp.int32), 1, 4)
    sectors_ok = jp.all(jp.where(jp.arange(4) < required,
                               sector_contact+1e-7 >= _cfg(config, 'contact_sector_seconds', .02), True))
    tolerance = jp.deg2rad(_cfg(config, 'angle_tolerance_degrees', 10.))
    completed = ((theta >= trace.goal_angle-tolerance)
                 & (phi >= trace.goal_angle-tolerance) & sectors_ok & proven)
    # Partial-turn curricula capture at their requested orientation; only the
    # full 2π task returns to its original directed (heavy-end) orientation.
    target_direction = (
        jp.cos(trace.goal_angle)*trace.initial_direction
        +jp.sin(trace.goal_angle)*jp.cross(trace.n0, trace.initial_direction)
        +(1.-jp.cos(trace.goal_angle))*jp.dot(trace.n0, trace.initial_direction)*trace.n0)
    direction_error = jp.arccos(jp.clip(jp.dot(direction, target_direction), -1., 1.))
    returned = direction_error <= jp.deg2rad(_cfg(config, 'capture_direction_degrees', 15.))
    hold_time = jp.where((trace.phase == HOLD) & valid_grip,
                        trace.initial_hold_time+dt, 0.)
    holding_goal = trace.goal_angle <= 0.
    ready = hold_time+time_epsilon >= jp.where(
        holding_goal, _cfg(config, 'hold_seconds', 1.),
        _cfg(config, 'initial_hold_seconds', .3))
    capture = active_spin & completed & returned & valid_grip
    stable_time = jp.where(capture, trace.stable_time+dt, 0.)
    successful = (((trace.phase == HOLD) & holding_goal & ready)
                  | (capture & (stable_time+time_epsilon >= _cfg(config, 'capture_seconds', .5))))
    candidate_phase = jp.where(
        trace.phase == HOLD, jp.where(ready & ~holding_goal, SPIN, HOLD),
        jp.where(completed, CATCH, SPIN))

    failure_reason = jp.array(NO_FAILURE, dtype=jp.int32)
    failure_reason = jp.where(elapsed+time_epsilon >= _cfg(config, 'episode_seconds', 10.), TIMEOUT, failure_reason)
    failure_reason = jp.where(violation_time+time_epsilon >= _cfg(config, 'violation_seconds', .1), ENVELOPE, failure_reason)
    failure_reason = jp.where(no_contact_time > _cfg(config, 'free_contact_seconds', .15)+time_epsilon, LOST_CONTACT, failure_reason)
    failure_reason = jp.where(penetrated, PENETRATION, failure_reason)
    failure_reason = jp.where(unsafe_support, SUPPORT, failure_reason)
    failure_reason = jp.where(~geometry_finite, NONFINITE, failure_reason)
    failed = failure_reason != NO_FAILURE
    successful = successful & ~failed
    done = successful | failed
    phase = jp.where(successful, SUCCESS, jp.where(failed, FAILED, candidate_phase)).astype(jp.int32)

    dp = potential-trace.potential
    rewarded_dp = jp.minimum(dp, 0.)+jp.where(envelope_valid & proven, jp.maximum(dp, 0.), 0.)
    progress_reward = jp.where(trace.goal_angle > 0.,
                               20.*rewarded_dp/jp.maximum(trace.goal_angle, 1e-8), 0.)
    soft_angle = jp.deg2rad(_cfg(config, 'soft_plane_degrees', 15.))
    plane_error = jp.maximum(0., plane_angle-soft_angle)/jp.maximum(soft_angle, 1e-8)
    omega = jp.asarray(sample.angvel)
    perpendicular = omega-jp.dot(omega, trace.n0)*trace.n0
    radius_error = (radius-trace.initial_radius)/.03
    holding_reward = (((trace.phase == HOLD) & valid_grip)
                      | ((trace.phase == CATCH) & capture))
    continuous_reward = dt*(
        holding_reward.astype(jp.float32)+.2*has_contact.astype(jp.float32)
        -2.*plane_error**2-.05*jp.sum(perpendicular**2)-.1*radius_error**2-.02)
    reward = progress_reward+continuous_reward+50.*successful.astype(jp.float32)-30.*failed.astype(jp.float32)
    reward = jp.where(geometry_finite, reward, -30.)
    new_trace = trace.replace(
        raw_theta=theta_raw, raw_phi=phi_raw, theta=theta, phi=phi,
        potential=potential, sector_contact=sector_contact, substeps=substeps, elapsed=elapsed,
        initial_hold_time=hold_time, stable_time=stable_time,
        no_contact_time=no_contact_time, violation_time=violation_time,
        phase=phase, done=done, success=successful, failure_reason=failure_reason,
        previous_angles_valid=angles_valid, trajectory_proven=proven,
    )
    # Environment wrappers may step an already terminal state before resetting.
    # It is absorbing and cannot repeatedly earn its terminal bonus/penalty.
    new_trace = jax.tree.map(lambda new, old: jp.where(trace.done, old, new), new_trace, trace)
    new_trace = jax.tree.map(lambda new, old: jp.where(jp.isfinite(new), new, old), new_trace, trace)
    reward = jp.where(trace.done, 0., reward)
    metrics = {
        'success': new_trace.success.astype(jp.float32),
        'failure': ((new_trace.done) & ~new_trace.success).astype(jp.float32),
        'phase': new_trace.phase.astype(jp.float32),
        'theta': new_trace.theta, 'phi': new_trace.phi,
        'progress': jp.where(trace.goal_angle > 0., new_trace.potential/jp.maximum(trace.goal_angle, 1e-8), 0.),
        'valid_progress_radians': jp.where(envelope_valid & proven, new_trace.potential, 0.),
        'plane_degrees': plane_degrees, 'pivot_drift': pivot_drift,
        'thumb_axis_degrees': thumb_axis_degrees,
        'endpoint_radius': radius, 'free_contact_time': new_trace.no_contact_time,
        'stable_time': new_trace.stable_time, 'valid_grip': valid_grip.astype(jp.float32),
        'contact': has_contact.astype(jp.float32),
        'failure_reason': new_trace.failure_reason.astype(jp.float32),
        'trajectory_proven': new_trace.trajectory_proven.astype(jp.float32),
    }
    metrics = jax.tree.map(jp.nan_to_num, metrics)
    return new_trace, reward, metrics
