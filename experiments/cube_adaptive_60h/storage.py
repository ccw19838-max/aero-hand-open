from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,uuid
from experiments.cube_contact_ppo.storage import atomic_json,digest,implementation_digest as source_digest


def implementation_digest():
    sha=hashlib.sha256(source_digest().encode())
    # Operational monitor/review fixes must not invalidate an unchanged policy.
    for name in ('config.py','env.py','policy.py'):
        path=Path(__file__).parent/name;sha.update(path.name.encode());sha.update(path.read_bytes())
    return sha.hexdigest()


def save(model,normalizer,reference,output,axis,profile,state,reason,evaluation=None,aliases=()):
    output=Path(output);token=uuid.uuid4().hex[:8];dest=output/'checkpoints'/f'step_{model.num_timesteps:09d}_{token}'
    temp=dest.with_name('.writing_'+token);temp.mkdir(parents=True)
    model.save(temp/'policy.zip');normalizer.save(temp/'reward_normalizer.pkl')
    for name in ('reference.npz','prepare_report.json'):shutil.copy2(Path(reference).with_name(name),temp/name)
    names=('policy.zip','reward_normalizer.pkl','reference.npz','prepare_report.json')
    atomic_json(temp/'metadata.json',dict(task='adaptive_xy_60h_v1',axis=axis,profile=profile.dictionary(),
        state=state,training_steps=int(model.num_timesteps),reason=reason,created_at=datetime.now(timezone.utc).isoformat(),
        implementation_sha256=implementation_digest(),evaluation=evaluation,files={n:digest(temp/n) for n in names}))
    os.replace(temp,dest)
    for alias in ('latest',*aliases):
        link=output/('.'+alias+'_'+token);link.symlink_to(dest.relative_to(output),target_is_directory=True);os.replace(link,output/alias)
    return dest


def load(path,env=None):
    from .policy import RetainPPO
    path=Path(path).resolve();meta=json.loads((path/'metadata.json').read_text())
    if meta['task']!='adaptive_xy_60h_v1' or meta['implementation_sha256']!=implementation_digest():raise ValueError('Adaptive checkpoint implementation mismatch')
    for name,sha in meta['files'].items():
        if digest(path/name)!=sha:raise ValueError('Bad checkpoint file: '+name)
    spec=meta.get('state',{}).get('provenance',{}).get('observation_transform')
    model_class=RetainPPO
    if spec is not None:
        from .canonical_policy import CanonicalRetainPPO,validate
        validate(spec);model_class=CanonicalRetainPPO
    model=model_class.load(path/'policy.zip',env=env,device='cpu')
    if getattr(model,'canonical_spec',None)!=spec:
        raise ValueError('Saved policy observation adapter disagrees with metadata')
    return model,meta


def load_source(path):
    path=Path(path).resolve();m=json.loads((path/'metadata.json').read_text())
    if m['task']=='adaptive_xy_60h_v1':return load(path)
    from experiments.cube_contact_ppo.storage import load as old_load
    return old_load(path)
