"""Shared deterministic inference and atomic, provenance-bound policy bundles.

Bundles contain trusted Python/Brax serialized data, not MCU firmware.  Resume
restores parameters and normalization; Brax initializes a new optimizer.
"""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import jax
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks

from .config import PenSpinConfig


FORMAT_VERSION = 1


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_prepared(reference, config=None):
    """Training cannot bypass the static-grip and numerical preparation gate."""
    reference = Path(reference).resolve()
    report_path = reference.parent / 'prepare_report.json'
    if not reference.is_file() or not report_path.is_file():
        raise ValueError('Run prepare successfully first: reference.npz and prepare_report.json are required')
    report = json.loads(report_path.read_text())
    if config is None:
        config = PenSpinConfig.load(reference.parent/'config.json')
    if config.task_mode=='grasp':
        passed = (report.get('passed') is True and report.get('status')=='grasp_reset_ready'
                  and report.get('scope')=='grasp_acquisition_only'
                  and all(g.get('passed') is True for g in report.get('gates',{}).values())
                  and bool(report.get('gates')))
    else:
        passed = report.get('passed') is True and report.get('status') == 'passed'
    if not passed:
        raise ValueError('Preparation failed; refusing to train or label a policy successful')
    expected = report.get('reference_sha256')
    if not expected or expected != digest_file(reference):
        raise ValueError('Reference hash differs from the prepared reference')
    from .model import model_fingerprint, task_fingerprint
    if config is None:
        config_path = reference.parent / 'config.json'
        if not config_path.is_file():
            raise ValueError('Prepared configuration is missing')
        config = PenSpinConfig.load(config_path)
    if report.get('model_digest') != model_fingerprint(config):
        raise ValueError('Prepared model differs from the current model/assets')
    if report.get('task_digest') != task_fingerprint(config):
        raise ValueError('Prepared physics/task differs from the requested configuration')
    with np.load(reference, allow_pickle=False) as data:
        for key in ('qpos', 'qvel', 'ctrl', 'n0', 'c0'):
            if key not in data or not np.isfinite(data[key]).all():
                raise ValueError(f'Missing or nonfinite prepared reference field: {key}')
        for key in ('config_digest', 'model_digest', 'task_digest'):
            if key not in data or str(data[key].item()) != report.get(key):
                raise ValueError(f'Reference/report provenance mismatch: {key}')
    return report


def network_factory(config):
    from functools import partial
    return partial(networks.make_ppo_networks,
                   policy_hidden_layer_sizes=tuple(config.hidden_layers),
                   value_hidden_layer_sizes=tuple(config.hidden_layers),
                   policy_obs_key='state', value_obs_key='privileged_state')


@lru_cache(maxsize=16)
def _inference_function(obs_sizes, hidden_layers, action_size):
    net = networks.make_ppo_networks(
        observation_size=dict(obs_sizes), action_size=action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=hidden_layers,
        value_hidden_layer_sizes=hidden_layers,
        policy_obs_key='state', value_obs_key='privileged_state')
    make_policy = networks.make_inference_fn(net)
    # Parameters are dynamic arguments so validation does not recompile for
    # each new checkpoint's values.
    return jax.jit(lambda params, obs, key: make_policy(params, deterministic=True)(obs, key)[0])


def inference_function(params, config):
    sizes = tuple(sorted((key, int(value.shape[-1]))
                         for key, value in params[0].mean.items()))
    return _inference_function(sizes, tuple(config.hidden_layers), 7)


def save_checkpoint(directory, params, config, reference, goal_angle, steps, **metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reference = Path(reference).resolve()
    prepared = require_prepared(reference, config)
    ref_digest = digest_file(reference)
    ref_destination = directory / 'reference.npz'
    if not ref_destination.exists():
        temporary_reference=directory/f'.reference.{os.getpid()}.tmp'
        shutil.copyfile(reference, temporary_reference)
        os.replace(temporary_reference,ref_destination)
    elif digest_file(ref_destination) != ref_digest:
        raise ValueError('A checkpoint directory cannot mix prepared references')
    temporary_report=directory/f'.prepare_report.{os.getpid()}.tmp'
    temporary_report.write_text(json.dumps(prepared,indent=2)+'\n')
    os.replace(temporary_report,directory/'prepare_report.json')
    info = dict(format_version=FORMAT_VERSION, saved_at_unix=time.time(),
                goal_angle=float(goal_angle), training_steps=int(steps),
                reference_sha256=ref_digest, config_digest=config.digest,
                backend='cpu', truth_observations=True,
                task_mode=config.task_mode,
                resume_semantics='weights_and_normalizer_only_optimizer_reinitialized',
                task_success_assessed=False, **metadata)
    payload = dict(format_version=FORMAT_VERSION, params=params,
                   config=config.to_dict(), metadata=info)
    temporary = directory / f'.checkpoint.{os.getpid()}.tmp'
    model.save_params(str(temporary), payload)
    os.replace(temporary, directory / 'checkpoint.pkl')
    # The pickle above is the atomic source of truth. Human-readable files may
    # lag if a process is interrupted, and are never used to load parameters.
    (directory / 'metadata.json').write_text(json.dumps(info, indent=2)+'\n')
    config.save(directory / 'config.json')
    return directory / 'checkpoint.pkl'


def load_checkpoint(path):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / 'checkpoint.pkl'
    if not path.is_file():
        raise ValueError(f'Policy checkpoint not found: {path}')
    payload = model.load_params(str(path))
    if not isinstance(payload, dict) or payload.get('format_version') != FORMAT_VERSION:
        raise ValueError('Not an AeroPenSpin checkpoint bundle')
    config = PenSpinConfig.from_dict(payload['config'])
    info = payload['metadata']
    if info['config_digest'] != config.digest:
        raise ValueError('Checkpoint configuration digest mismatch')
    reference = path.parent / 'reference.npz'
    if not reference.exists() or digest_file(reference) != info['reference_sha256']:
        raise ValueError('Checkpoint prepared reference is missing or altered')
    require_prepared(reference, config)
    return payload['params'], config, reference, info
