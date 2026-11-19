"""Bounded optional PPO horizons; do not alter inference or task criteria."""
import math


def worker_timesteps(job):
    values=job.get('physics_timesteps',(.001,.002))
    if not isinstance(values,(list,tuple)) or len(values) not in (2,4):
        raise ValueError('Training requires2 or4 explicit CPU workers')
    if any(isinstance(v,bool) or v not in (.001,.002) for v in values):
        raise ValueError('Only validated1ms/2ms workers are allowed')
    if values.count(.001)!=values.count(.002):
        raise ValueError('Both integrators require equal training allocation')
    return tuple(float(v) for v in values)


def temporal_parameters(job):
    steps=job.get('rollout_steps',128);gae=float(job.get('gae_lambda',.95))
    if steps not in (128,256,512) or isinstance(steps,bool):raise ValueError('Invalid PPO rollout horizon')
    if not math.isfinite(gae) or not .9<=gae<=1.:raise ValueError('Invalid PPO GAE lambda')
    return dict(rollout_steps=int(steps),gae_lambda=gae)
