"""Checksummed, immutable teacher checkpoints and references."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime,timezone
from experiments.cube_quarter_turn.checkpoints import atomic_json,implementation_digest as v1_digest
from experiments.cube_envelop.model import model_fingerprint
from .task import Config


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_digest():
    sha=hashlib.sha256(v1_digest().encode())
    root=Path(__file__).parent
    sha.update((root.parent/'cube_quarter_turn/action_mapping_candidate.py').read_bytes())
    for path in sorted(root.glob('*.py')):
        sha.update(path.name.encode());sha.update(path.read_bytes())
    return sha.hexdigest()


def save_reference(directory,qpos,ctrl,validation,origin):
    import numpy as np
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'reference.npz'
    np.savez_compressed(path,qpos=qpos,qvel=np.zeros(22),ctrl=ctrl)
    atomic_json(directory/'prepare_report.json',dict(approved=True,model_fingerprint=model_fingerprint(Config()),
        reference_sha256=digest(path),variant='three-finger-or-better unassisted v2 grip',
        origin=str(origin),validation=validation,physics_unchanged=True))
    return path


def save(model,normalizer,reference,output,axis,reason,provenance,report=None,aliases=()):
    output=Path(output);token=uuid.uuid4().hex[:8]
    dest=output/'checkpoints'/f'step_{model.num_timesteps:09d}_{token}'
    temp=dest.with_name('.writing_'+token);temp.mkdir(parents=True)
    model.save(temp/'policy.zip');normalizer.save(temp/'reward_normalizer.pkl')
    for name in ('reference.npz','prepare_report.json'):
        shutil.copy2(Path(reference).with_name(name),temp/name)
    names=('policy.zip','reward_normalizer.pkl','reference.npz','prepare_report.json')
    atomic_json(temp/'metadata.json',dict(task='xyz_quarter_turn_v2',axis=axis,training_steps=int(model.num_timesteps),
        implementation_sha256=implementation_digest(),source_initialization=provenance,
        created_at=datetime.now(timezone.utc).isoformat(),reason=reason,configuration=Config().__dict__,
        actor_inputs=133,critic_inputs=133,goal_degrees=90.,evaluation=report,files={n:digest(temp/n) for n in names}))
    os.replace(temp,dest)
    for alias in ('latest',*aliases):
        link=output/('.'+alias+'_'+token);link.symlink_to(dest.relative_to(output),target_is_directory=True);os.replace(link,output/alias)
    return dest


def load(checkpoint,vector=None):
    from stable_baselines3 import PPO
    path=Path(checkpoint).resolve();metadata=json.loads((path/'metadata.json').read_text())
    if metadata['task']!='xyz_quarter_turn_v2' or metadata['implementation_sha256']!=implementation_digest():
        raise ValueError('Teacher implementation mismatch')
    for name,sha in metadata['files'].items():
        if digest(path/name)!=sha: raise ValueError('Checkpoint checksum mismatch: '+name)
    return PPO.load(path/'policy.zip',env=vector,device='cpu'),metadata
