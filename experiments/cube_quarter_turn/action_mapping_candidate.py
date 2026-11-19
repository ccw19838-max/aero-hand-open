"""Candidate for the next experiment; not wired into the running v1 policy.

Keep the reference as action zero while exposing both full actuator endpoints.
The current frozen v1 environment/checkpoints retain their original semantics.
"""
import numpy as np


def full_range_targets(action,reference,lower,upper):
    arrays=[np.asarray(v,dtype=float) for v in (action,reference,lower,upper)]
    if any(a.shape!=(7,) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Expected seven finite values for each argument')
    action,reference,lower,upper=arrays
    if np.any(lower>=upper) or np.any(reference<lower) or np.any(reference>upper):
        raise ValueError('Reference must be within ordered actuator bounds')
    action=np.clip(action,-1.,1.)
    return reference+action*np.where(action>=0.,upper-reference,reference-lower)
