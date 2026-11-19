"""Immutable quarter-turn policy, reward statistics and reference-state bundles."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import os
import shutil
import uuid

from experiments.cube_envelop.model import model_fingerprint
from .task import Config


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_digest():
    root=Path(__file__).resolve().parent;sha=hashlib.sha256()
    paths=[root/n for n in ('task.py','env.py','policy.py')]
    paths.extend(root.parent/package/name for package,names in (
        ('cube_grasp',('config.py','model.py')),('cube_reorient',('model.py',)),
        ('cube_envelop',('model.py','env.py')),('cube_official_baseline',('env.py','policy.py')))
        for name in names)
    for p in paths:sha.update(str(p.relative_to(root.parent)).encode());sha.update(p.read_bytes())
    sha.update(model_fingerprint(Config()).encode());return sha.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def save(model,normalizer,reference,output,reason,provenance,report=None,aliases=()):
    output=Path(output);token=uuid.uuid4().hex[:8]
    dest=output/'checkpoints'/f'step_{model.num_timesteps:09d}_{token}';temp=dest.with_name('.writing_'+token);temp.mkdir(parents=True)
    model.save(temp/'policy.zip');normalizer.save(temp/'reward_normalizer.pkl')
    shutil.copy2(reference,temp/'reference.npz');shutil.copy2(Path(reference).with_name('prepare_report.json'),temp/'prepare_report.json')
    atomic_json(temp/'metadata.json',dict(task='xyz_quarter_turn_v1',format_version=1,
        created_at=datetime.now(timezone.utc).isoformat(),training_steps=int(model.num_timesteps),reason=reason,
        implementation_sha256=implementation_digest(),source_initialization=provenance,
        observations={'actor':45,'critic':112,'goal_context':31},goal_degrees=90.,
        success='full pose and signed 90±5 degrees, thumb plus two fingers, low speeds, continuous 1 second',
        trajectory_checks={'angle_and_forbidden_contact_seconds':.002,'force_and_stability_seconds':.01},
        configuration=Config().__dict__,evaluation=report,
        files={n:digest(temp/n) for n in ('policy.zip','reward_normalizer.pkl','reference.npz','prepare_report.json')}))
    os.replace(temp,dest)
    for name in ('latest',*aliases):
        link=output/('.'+name+'_'+token);link.symlink_to(dest.relative_to(output),target_is_directory=True);os.replace(link,output/name)
    return dest


def load(checkpoint,vector=None):
    from stable_baselines3 import PPO
    checkpoint=Path(checkpoint).resolve();m=json.loads((checkpoint/'metadata.json').read_text())
    if m['task']!='xyz_quarter_turn_v1' or m['implementation_sha256']!=implementation_digest():raise ValueError('Different quarter-turn semantics')
    for name,sha in m['files'].items():
        if digest(checkpoint/name)!=sha:raise ValueError('Checkpoint checksum mismatch: '+name)
    return PPO.load(checkpoint/'policy.zip',env=vector,device='cpu'),m
