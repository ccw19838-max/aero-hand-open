"""Replay a real reachable prefix, then try bounded reference-grip feedback.

Every candidate starts from the original initial grasp. There is no state bank
or object placement at an intermediate angle. Only complete native replays of
the exact saved float32 controls can qualify a demonstration; none of these
controllers is a standalone neural policy.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,itertools,json,time
import numpy as np
from .config import Profile
from .fast_contacts import FastContactEnv
from .qualify_teacher import qualify
from .storage import atomic_json,digest


def search(goal,source,seconds=600.,contact_first=False):
    root=Path(goal);source=Path(source)
    source_info=json.loads((source/'result.json').read_text())
    if source_info['axis']!='y' or source_info['seed']!=0 or source_info['actual_dt']!=.001:
        raise ValueError('This diagnostic is pinned to the recorded Y seed0 1ms path')
    checkpoint=Path(source_info['source_checkpoint'])
    profile=Profile(**json.loads((checkpoint/'metadata.json').read_text())['profile'])
    if profile.action_fraction!=.02:raise ValueError('Action semantics changed')
    original_trace=json.loads((source/'trace.json').read_text())
    with np.load(source/'actual_trajectory.npz',allow_pickle=False) as data:
        prefix=data['actions'].astype(np.float32)
    if prefix.ndim!=2 or prefix.shape[1]!=7 or not np.isfinite(prefix).all() or np.max(np.abs(prefix))>1.:
        raise ValueError('Invalid recorded prefix')
    variant='contact_first_v2' if contact_first else 'v1'
    output=root/f'return_grip_teacher_y90_seed0_{variant}';output.mkdir(exist_ok=False)
    (output/'controller_source.py').write_bytes(Path(__file__).read_bytes())
    deadline=min(time.time()+seconds,datetime.fromisoformat(json.loads((root/'goal.json').read_text())['deadline_at']).timestamp())
    atomic_json(output/'source.json',dict(source=str(source.resolve()),checkpoint=str(checkpoint),
        prefix_sha256=digest(source/'actual_trajectory.npz'),controller_sha256=digest(Path(__file__)),
        seed=0,axis='y',actual_dt=.001,action_fraction=.02,from_original_initial_grasp=True))
    masks=dict(fingers=np.array([1.,1.,1.,1.,0.,0.,0.]),all=np.ones(7),
        fingers_thumb_tendons=np.array([1.,1.,1.,1.,0.,1.,1.]))
    results=[];qualified=False;best=None
    modes=('together','middle_first','pinky_first') if contact_first else tuple(masks)
    gains=(.8,.4) if contact_first else (1.,.5)
    candidates=list(itertools.product((184,176,196),modes,gains))
    for case_index,(cut,mode,gain) in enumerate(candidates):
        if time.time()>=deadline:break
        if cut>len(prefix):raise ValueError('Source prefix shorter than required')
        case=output/f'case_{case_index:03d}';case.mkdir()
        env=FastContactEnv(root/'sources/x/reference.npz','y',physical_dt=.001,target_degrees=90.,profile=profile)
        observations=[];actions=[];trace=[];interrupted=False;prefix_exact=False
        try:
            obs,info=env.reset(seed=0)
            span=env.ids['ctrl_upper']-env.ids['ctrl_lower']
            for control in range(250):
                if time.time()>=deadline:interrupted=True;break
                before=time.process_time()
                if control<cut:action=prefix[control]
                elif contact_first:
                    # Keep existing supporting fingers at their actual held
                    # targets. Establish replacement contacts before releasing
                    # any of the three fingers carrying the cube. Negative
                    # tendon target increments flex these four actuators.
                    action=np.zeros(7);forces=info['finger_forces']
                    order=(2,4) if mode!='pinky_first' else (4,2)
                    missing=[finger for finger in order if forces[finger]<.15]
                    if mode!='together':missing=missing[:1]
                    for finger in missing:action[finger-1]=-gain
                else:
                    error=env.default-env.data.ctrl
                    action=np.clip(gain*error/(profile.action_fraction*span),-1.,1.)*masks[mode]
                action=np.asarray(action,dtype=np.float32)
                observations.append(obs.copy());actions.append(action.copy())
                obs,_,done,_,info=env.step(action);trace.append(info)
                if control+1==cut:
                    expected=original_trace[cut-1]
                    error=max(abs(info[k]-expected[k]) for k in
                        ('signed_progress_degrees','angle_error_degrees','elapsed_seconds'))
                    prefix_exact=bool(error<1e-7 and not done)
                    if not prefix_exact:raise RuntimeError('Exact recorded prefix did not reproduce')
                if done:break
                spent=time.process_time()-before
                time.sleep(min(max(0.,deadline-time.time()),4.*spent))
            record=dict(observed_at=datetime.now(timezone.utc).isoformat(),axis='y',seed=0,actual_dt=.001,
                target_degrees=90.,source_checkpoint=str(checkpoint),source_prefix=str(source.resolve()),
                prefix_controls=cut,prefix_reproduced=prefix_exact,feedback_mask=mode,feedback_gain=gain,
                controller='saved reachable prefix plus bounded '+('contact-first' if contact_first else 'reference-grip')+' feedback; not standalone neural policy',
                neural_policy_from_initial_grasp=False,actual_trajectory_from_original_initial_grasp=True,
                success=bool(info['success'] and not interrupted and prefix_exact),interrupted=interrupted,final=info)
            np.savez_compressed(case/'actual_trajectory.npz',observations=np.asarray(observations,dtype=np.float32),
                actions=np.asarray(actions,dtype=np.float32))
            atomic_json(case/'trace.json',trace);atomic_json(case/'result.json',record)
            if record['success']:
                validation=qualify(root,case);record['saved_commands_validation']=validation
                atomic_json(case/'result.json',record);qualified=validation['saved_actions_native_verified']
            value=(record['success'],info['max_stable_seconds'],-info['best_angle_error_degrees'])
            if best is None or value>best[0]:best=(value,str(case.resolve()))
            results.append(dict(case=case.name,cut=cut,mode=mode,gain=gain,success=record['success'],
                native_both_dt_verified=qualified,interrupted=interrupted,prefix_reproduced=prefix_exact,
                maximum_angle=info['max_progress_degrees'],best_error=info['best_angle_error_degrees'],
                goal_hold=info['max_stable_seconds'],five_finger_hold=info['max_all_five_stable_seconds'],
                failure=info['failure_reason']))
            atomic_json(output/'status.json',dict(status='searching',cases_finished=len(results),
                cases_planned=len(candidates),qualified_demonstration=qualified,best_case=best[1],latest=results[-1]))
            print(json.dumps(results[-1]),flush=True)
            if qualified:break
        finally:env.close()
    report=dict(observed_at=datetime.now(timezone.utc).isoformat(),status='diagnosis_complete',
        qualified_demonstration=qualified,neural_policy_success=False,best_case=best[1] if best else None,
        scope='Independent diagnostic candidates from original grasp, not standalone neural-policy evaluation',
        cases=results,local_budget_seconds=seconds,budget_exhausted=time.time()>=deadline)
    atomic_json(output/'result.json',report);atomic_json(output/'status.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--goal',required=True);p.add_argument('--source',required=True)
    p.add_argument('--seconds',type=float,default=600.);p.add_argument('--contact-first',action='store_true');args=p.parse_args()
    print(json.dumps(search(args.goal,args.source,args.seconds,args.contact_first),ensure_ascii=False))
