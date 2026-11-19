"""Optional explicitly fingerprinted inference and rollout observation adapter."""
from pathlib import Path
import hashlib
import gymnasium as gym
import numpy as np
from .cube_labels import canonical,transform
from .policy import RetainPPO


def descriptor(anchor):
    anchor=canonical(anchor)
    if anchor.shape!=(4,):raise ValueError('A fixed quaternion anchor is required')
    return dict(name='fixed_initial_cube_label_v1',anchor_quaternion=anchor.tolist(),
        quaternion_blocks=[71,84,94],physical_changes=False,
        files={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('cube_labels.py','canonical_policy.py')})


def validate(spec):
    if not isinstance(spec,dict):raise ValueError('Missing observation-label specification')
    expected=descriptor(spec.get('anchor_quaternion'))
    if any(spec.get(k)!=v for k,v in expected.items() if k!='anchor_quaternion') or set(spec)!=set(expected):
        raise ValueError('Observation-label implementation or anchor mismatch')
    anchor=np.asarray(spec['anchor_quaternion'],dtype=np.float64)
    if not np.allclose(anchor,expected['anchor_quaternion'],rtol=0.,atol=1e-12):
        raise ValueError('Observation-label anchor must be canonical and normalized')
    return canonical(anchor)


class CanonicalObservation(gym.ObservationWrapper):
    def __init__(self,env,spec):
        super().__init__(env);self.anchor=validate(spec)
    def observation(self,observation):return transform(observation,self.anchor)


class CanonicalRetainPPO(RetainPPO):
    """Rollouts see wrapped observations; predict accepts original raw states.

    SB3 collects rollouts through policy.forward directly. The training env
    therefore supplies canonical observations, while external inference and
    original-environment evaluation pass raw observations through predict.
    """
    def __init__(self,*args,canonical_spec=None,**kwargs):
        self.canonical_spec=canonical_spec
        self._canonical_anchor=validate(canonical_spec) if canonical_spec is not None else None
        super().__init__(*args,**kwargs)
    @classmethod
    def load(cls,path,*args,**kwargs):
        model=super().load(path,*args,**kwargs)
        model._canonical_anchor=validate(model.canonical_spec)
        return model
    def predict(self,observation,state=None,episode_start=None,deterministic=False):
        if self._canonical_anchor is None:raise ValueError('Missing observation-label anchor')
        return super().predict(transform(observation,self._canonical_anchor),state=state,
            episode_start=episode_start,deterministic=deterministic)
