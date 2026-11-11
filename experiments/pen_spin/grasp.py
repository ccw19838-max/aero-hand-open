"""V2 active, unsupported multi-finger acquisition; independent of spin reward."""
import jax
import jax.numpy as jp
from .trajectory import HOLD,SUCCESS,FAILED,SUPPORT,PENETRATION,LOST_CONTACT,ENVELOPE,TIMEOUT,NONFINITE


def advance_grasp_trace(trace,sample,dt,config):
    contacts = sample.finger_forces>=config.contact_force
    count = jp.sum(contacts)
    multi = contacts[0] & (count>=config.grasp_min_fingers)
    speed = jp.linalg.norm(sample.linvel)
    omega = jp.linalg.norm(sample.angvel)
    region = jp.linalg.norm(sample.center-trace.c0)<=config.max_center_distance
    finite = jp.all(jp.isfinite(jp.concatenate([sample.center,sample.direction,sample.linvel,
                                               sample.angvel,sample.finger_forces])))
    stable = multi & (speed<config.grip_velocity) & (omega<config.grip_omega) & region
    stable &= ~sample.invalid_contact & ~sample.penetration & finite
    substeps = trace.substeps+1
    elapsed = substeps*dt
    hold = jp.where(stable,trace.initial_hold_time+dt,0.)
    # Time to close fingers is a new task rule, never assistance or a success grace period.
    acquiring = elapsed<=config.grasp_acquisition_seconds
    lost = jp.where(jp.any(contacts) | acquiring,0.,trace.no_contact_time+dt)
    violation = jp.where(region,0.,trace.violation_time+dt)
    reason = jp.array(0,jp.int32)
    reason = jp.where(elapsed>=config.episode_seconds-dt*.02,TIMEOUT,reason)
    reason = jp.where(violation>=config.violation_seconds-dt*.02,ENVELOPE,reason)
    reason = jp.where(lost>config.free_contact_seconds+dt*.02,LOST_CONTACT,reason)
    reason = jp.where(sample.penetration,PENETRATION,reason)
    reason = jp.where(sample.invalid_contact,SUPPORT,reason)
    reason = jp.where(~finite,NONFINITE,reason)
    failed = reason!=0
    success = (hold>=config.hold_seconds-dt*.02) & ~failed
    near = jp.zeros(5) if sample.finger_distances is None else jp.exp(-sample.finger_distances/.015)
    reward = dt*(2.*jp.sum(near)+3.*count+5.*multi+20.*stable
                 -2.*speed**2-.1*omega**2-.02)+50.*success-30.*failed
    reward = jp.where(finite,reward,-30.)
    new = trace.replace(substeps=substeps,elapsed=elapsed,initial_hold_time=hold,
                        no_contact_time=lost,violation_time=violation,
                        phase=jp.where(success,SUCCESS,jp.where(failed,FAILED,HOLD)),
                        done=success|failed,success=success,failure_reason=reason)
    new = jax.tree.map(lambda a,b:jp.where(trace.done,b,a),new,trace)
    reward = jp.where(trace.done,0.,reward)
    metrics = {'valid_progress_radians':jp.array(0.),'contact_fingers':count,
               'stable_seconds':new.initial_hold_time}
    return new,jp.nan_to_num(reward),metrics
