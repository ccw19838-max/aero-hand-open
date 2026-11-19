from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,uuid
from experiments.cube_adaptive_60h.storage import atomic_json,digest,implementation_digest as legacy_digest
from experiments.cube_adaptive_60h.policy import RetainPPO
from .env import VERSION


def fingerprint():
    sha=hashlib.sha256(legacy_digest().encode())
    for name in ('env.py','reward.py','io.py','evaluation.py','train.py','path_probe.py'):
        path=Path(__file__).with_name(name);sha.update(name.encode());sha.update(path.read_bytes())
    for name in ('fast_contacts.py','native_contacts.c','shaping.py','thumb_drive.py'):
        path=Path(__file__).parent.parent/'cube_adaptive_60h'/name;sha.update(name.encode());sha.update(path.read_bytes())
    return sha.hexdigest()


def link(path,target):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name('.'+path.name+'.new')
    temp.unlink(missing_ok=True);temp.symlink_to(Path(target).resolve());os.replace(temp,path)


def save(model,normalizer,root,state,reason,evaluation=None,aliases=()):
    root=Path(root);token=uuid.uuid4().hex[:8];dest=root/'checkpoints'/f'step_{model.num_timesteps:09d}_{token}'
    temp=dest.with_name('.writing_'+token);temp.mkdir(parents=True)
    model.save(temp/'policy.zip');normalizer.save(temp/'normalizer.pkl')
    for name in ('reference.npz','prepare_report.json','prefix_bank.npz','prefix_manifest.json'):
        shutil.copy2(root/'inputs'/name,temp/name)
    names=['policy.zip','normalizer.pkl','reference.npz','prepare_report.json','prefix_bank.npz','prefix_manifest.json']
    atomic_json(temp/'metadata.json',dict(task=VERSION,implementation_sha256=fingerprint(),created_at=datetime.now(timezone.utc).isoformat(),
        training_steps=int(model.num_timesteps),reason=reason,state=state,evaluation=evaluation,
        profile=json.loads((root/'job.json').read_text())['profile'],
        files={n:digest(temp/n) for n in names},inference_environment='experiments.cube_contact_recovery_v2.env.NativeEnv'))
    os.replace(temp,dest)
    for name in ('latest',*aliases):link(root/name,dest)
    return dest


def load(path,env=None):
    path=Path(path).resolve();meta=json.loads((path/'metadata.json').read_text())
    if meta['task']!=VERSION or meta['implementation_sha256']!=fingerprint():raise ValueError('V2 checkpoint implementation mismatch')
    for name,sha in meta['files'].items():
        if digest(path/name)!=sha:raise ValueError('Checkpoint content changed: '+name)
    return RetainPPO.load(path/'policy.zip',env=env,device='cpu'),meta
