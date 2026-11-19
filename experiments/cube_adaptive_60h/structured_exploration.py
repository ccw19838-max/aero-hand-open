"""Optional SB3 gSDE exploration; deterministic inference and physics unchanged."""
from pathlib import Path
import hashlib
import numpy as np
import torch


def settings(job):
    enabled=job.get('use_sde',False)
    if not isinstance(enabled,bool):raise ValueError('use_sde must be boolean')
    frequency=job.get('sde_sample_freq',8) if enabled else -1
    if enabled and (isinstance(frequency,bool) or frequency not in (4,8,16,32)):
        raise ValueError('Unsupported structured exploration interval')
    return dict(use_sde=enabled,sde_sample_freq=frequency)


def fingerprint():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def initialize(model,normalizer,target_std,seed):
    normalizer.seed(seed+7000000);obs=normalizer.reset()
    for _ in range(5):obs,_,_,_=normalizer.step(np.zeros((normalizer.num_envs,7),dtype=np.float32))
    with torch.no_grad():
        tensor=model.policy.obs_to_tensor(obs)[0]
        latent=model.policy.mlp_extractor.forward_actor(model.policy.features_extractor(tensor))
        square=latent.square().mean(0).cpu().numpy()
        norm=float(np.sqrt(square.sum()))
        if not np.isfinite(norm) or norm<1e-6:raise ValueError('Invalid exploration calibration features')
        scale=1./norm;model.policy.log_std.fill_(float(np.log(target_std*scale)))
    return dict(mode='SB3_gSDE_v1',source_sha256=fingerprint(),sample_frequency=model.sde_sample_freq,
        initial_reference_action_std=target_std,parameter_scale=scale,mean_feature_square=square.tolist(),
        calibration_seed=seed+7000000,calibration_zero_action_cycles=5,
        note='Noise weights persist over several control cycles; action variance depends on state. Physical clipping and goal/failure checks unchanged.')


def report_std(model,provenance):
    values=model.policy.log_std.detach().exp().cpu().numpy()
    if not model.use_sde:return values.tolist()
    reference=np.asarray(provenance['structured_exploration']['mean_feature_square'])
    return np.sqrt(reference@np.square(values)).tolist()
