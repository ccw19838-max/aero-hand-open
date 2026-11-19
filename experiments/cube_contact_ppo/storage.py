"""Independent contact-task checkpoints; do not change frozen previous tasks."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import os
import shutil
import uuid
from experiments.cube_quarter_turn_v2.storage import (atomic_json,digest,
    implementation_digest as prior_digest)
from .task import Config


def implementation_digest():
    sha=hashlib.sha256(prior_digest().encode())
    for path in sorted(Path(__file__).parent.glob('*.py')):
        sha.update(path.name.encode());sha.update(path.read_bytes())
    return sha.hexdigest()


def save(model,normalizer,reference,output,axis,reason,provenance,curriculum,evaluation=None,aliases=()):
    output=Path(output);token=uuid.uuid4().hex[:8]
    dest=output/'checkpoints'/f'step_{model.num_timesteps:09d}_{token}'
    temp=dest.with_name('.writing_'+token);temp.mkdir(parents=True)
    model.save(temp/'policy.zip');normalizer.save(temp/'reward_normalizer.pkl')
    for name in ('reference.npz','prepare_report.json'):shutil.copy2(Path(reference).with_name(name),temp/name)
    files=('policy.zip','reward_normalizer.pkl','reference.npz','prepare_report.json')
    atomic_json(temp/'metadata.json',dict(task=Config().version,axis=axis,training_steps=int(model.num_timesteps),
        implementation_sha256=implementation_digest(),created_at=datetime.now(timezone.utc).isoformat(),
        reason=reason,configuration=Config().__dict__,curriculum=curriculum,
        source_initialization=provenance,observation_size=138,goal_degrees=90.,
        success='five real per-finger forces within limits + full orientation/signed angle ±5deg + low speeds + continuous 1s',
        evaluation=evaluation,files={n:digest(temp/n) for n in files}))
    os.replace(temp,dest)
    for name in ('latest',*aliases):
        link=output/('.'+name+'_'+token);link.symlink_to(dest.relative_to(output),target_is_directory=True);os.replace(link,output/name)
    return dest


def load(path,vector=None):
    from stable_baselines3 import PPO
    path=Path(path).resolve();metadata=json.loads((path/'metadata.json').read_text())
    if metadata['task']!=Config().version or metadata['implementation_sha256']!=implementation_digest():
        raise ValueError('Contact-task checkpoint semantics changed')
    for name,sha in metadata['files'].items():
        if digest(path/name)!=sha:raise ValueError('Checkpoint checksum mismatch: '+name)
    return PPO.load(path/'policy.zip',env=vector,device='cpu'),metadata
