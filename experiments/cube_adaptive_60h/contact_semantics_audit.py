"""Diagnostic adapters for two discovered contact semantic defects.

No legacy/core files or live jobs are changed. This is not a new trained policy.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from pathlib import Path
import argparse,json
import numpy as np
import torch,mujoco
from .env import AdaptiveEnv
from .config import Profile
from .storage import load,atomic_json
from experiments.cube_official_baseline.env import TIP_NAMES


class CorrectedDiagnosticEnv(AdaptiveEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        root=self.model.body('right_t_link').id
        for g in range(self.model.ngeom):
            body=int(self.model.geom_bodyid[g])
            while body>0:
                if body==root:
                    self.ids['geom_finger'][g]=0;break
                body=int(self.model.body_parentid[body])
        # Keep the original actor's tip-feature order intact; only reorder the
        # per-finger distance used together with thumb-first force/missing flags.
        self.distance_order=np.array([TIP_NAMES.index(name) for name in ('th_tip','if_tip','mf_tip','rf_tip','pf_tip')])
    def _refresh(self):
        super()._refresh()
        self.tip_distances=self.tip_distances[self.distance_order]
        self.contacts['tip_distances']=self.tip_distances.copy()


def main(output):
    torch.set_num_threads(1);out=Path(output);manifest=json.loads((out/'manifest.json').read_text());records=[];proofs=[]
    for label,path in manifest['pins'].items():
        path=Path(path);policy,metadata=load(path);profile=Profile(**metadata['profile'])
        for dt in (.001,.002):
            plain=AdaptiveEnv(path/'reference.npz','x',physical_dt=dt,target_degrees=90.,profile=profile)
            corrected=CorrectedDiagnosticEnv(path/'reference.npz','x',physical_dt=dt,target_degrees=90.,profile=profile)
            try:
                for seed in (1000006,9400000,9400001):
                    obs,a=plain.reset(seed=seed);fixed_obs,b=corrected.reset(seed=seed);trace=[];identical=True;shared_steps=0
                    for step in range(250):
                        command=policy.predict(obs,deterministic=True)[0]
                        obs,_,adone,_,a=plain.step(command);fixed_obs,_,bdone,_,b=corrected.step(command)
                        if abs(plain.data.time-corrected.data.time)<1e-12:
                            shared_steps+=1
                            identical&=bool(np.array_equal(plain.data.qpos,corrected.data.qpos) and np.array_equal(plain.data.qvel,corrected.data.qvel))
                        else:
                            # A terminal check can stop one env partway through
                            # its 50ms cycle. Compare physics only at equal times.
                            assert adone or bdone
                        if adone or bdone:break
                    missing=0.;wrench=np.zeros(6);root=plain.model.body('right_t_link').id;cube=plain.ids['cube_geom']
                    for i,c in enumerate(plain.data.contact):
                        g1,g2=int(c.geom1),int(c.geom2);other=g2 if g1==cube else g1 if g2==cube else None
                        if other is not None and int(plain.model.geom_bodyid[other])==root:
                            mujoco.mj_contactForce(plain.model,plain.data,i,wrench);missing+=max(0.,float(wrench[0]))
                    actual=np.asarray(a['finger_forces']).copy();actual[0]+=missing
                    proofs.append(dict(policy=label,dt=dt,seed=seed,shared_action_prefix_bitwise_physics=identical,
                        equal_time_control_steps=shared_steps,legacy_terminal_missing_thumb_force=missing,
                        corrected_forces_at_exact_legacy_terminal=actual.tolist(),
                        plain_failure=a['failure_reason'],corrected_failure=b['failure_reason'],
                        plain_forces=a['finger_forces'],corrected_forces=b['finger_forces'],time=a['elapsed_seconds']))
                    assert identical
                    # Continue feedback using the corrected force observation.
                    while not bdone and step<249:
                        step+=1;fixed_obs,_,bdone,_,b=corrected.step(policy.predict(fixed_obs,deterministic=True)[0])
                    records.append(dict(policy=label,dt=dt,seed=seed,mode='identical_action_prefix_then_corrected_feedback',
                        diagnostic_only=True,final=b))
                    # Separate fully closed-loop rerun with corrected contacts
                    # from reset; does not imply the old weights were retrained.
                    fixed_obs,b=corrected.reset(seed=seed)
                    for _ in range(250):
                        fixed_obs,_,bdone,_,b=corrected.step(policy.predict(fixed_obs,deterministic=True)[0]);trace.append(b)
                        if bdone:break
                    records.append(dict(policy=label,dt=dt,seed=seed,mode='corrected_feedback_from_initial',diagnostic_only=True,final=b,trace=trace))
            finally:plain.close();corrected.close()
    atomic_json(out/'contact_semantics_checks.json',dict(
        scope='Mapping correction diagnostics only; no learning or modification of legacy evaluation data',
        force_order=['thumb','index','middle','ring','pinky'],legacy_tip_order=list(TIP_NAMES),
        missing_thumb_body='right_t_link',missing_thumb_geom='th_mp_collision',proofs=proofs,records=records))
    for key in manifest['pins']:
        for mode in ('identical_action_prefix_then_corrected_feedback','corrected_feedback_from_initial'):
            rs=[r for r in records if r['policy']==key and r['mode']==mode]
            print(json.dumps(dict(policy=key,mode=mode,count=len(rs),max_angle=max(r['final']['max_progress_degrees'] for r in rs),
                max_stable=max(r['final']['max_stable_seconds'] for r in rs),successes=sum(r['final']['success'] for r in rs),
                failure_reasons=list(set(r['final']['failure_reason'] for r in rs)))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
