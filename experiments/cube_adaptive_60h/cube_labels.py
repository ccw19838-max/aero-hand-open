"""Fixed initial-label coordinates for the dynamically symmetric rigid cube.

Current and target orientations receive the same right cube symmetry. Relative
rotation and all physical and goal-monitor states remain unchanged. The anchor
is fixed for an experiment, not selected from the outcome of a live episode.
"""
import numpy as np
from experiments.cube_quarter_turn.task import SYMMETRIES,matrix_quaternion

SYMMETRY_QUATERNIONS=np.stack([matrix_quaternion(s) for s in SYMMETRIES])
QUATERNION_BLOCKS=(71,84,94)


def multiply(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    scalar=a[...,0]*b[...,0]-np.sum(a[...,1:]*b[...,1:],axis=-1)
    vector=a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+np.cross(a[...,1:],b[...,1:])
    return np.concatenate([scalar[...,None],vector],axis=-1)


def canonical(q):
    q=np.asarray(q,dtype=np.float64);length=np.linalg.norm(q,axis=-1,keepdims=True)
    if not np.isfinite(q).all() or np.any(length<1e-10):raise ValueError('Invalid observed quaternion')
    q=q/length;return q*np.where(q[...,:1]<0.,-1.,1.)


def initial_quaternion(observation):
    obs=np.asarray(observation)
    if obs.shape[-1]!=143:raise ValueError('Expected frozen143-input semantics')
    angle=obs[...,133]*np.pi/2.
    direction=obs[...,101:104].astype(np.float64)
    norm=np.linalg.norm(direction,axis=-1,keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm<1e-10):raise ValueError('Invalid fixed task axis')
    direction=direction/norm
    inverse_turn=np.concatenate([np.cos(angle[...,None]/2.),-direction*np.sin(angle[...,None]/2.)],axis=-1)
    return canonical(multiply(inverse_turn,obs[...,94:98]))


def transform(observation,anchor):
    obs=np.asarray(observation);initial=initial_quaternion(obs);anchor=canonical(anchor)
    candidates=canonical(multiply(initial[...,None,:],SYMMETRY_QUATERNIONS))
    indices=np.argmax(np.abs(np.sum(candidates*anchor,axis=-1)),axis=-1)
    symmetry=SYMMETRY_QUATERNIONS[indices]
    result=obs.copy()
    for start in QUATERNION_BLOCKS:
        result[...,start:start+4]=canonical(multiply(obs[...,start:start+4],symmetry)).astype(obs.dtype)
    return result
