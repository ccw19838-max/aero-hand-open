"""Read-only X failure audit and labelled physical counterfactuals.

Does not modify training jobs, sources, weights or acceptance criteria. The
grace diagnostic changes only its private monitor and is never a valid trial.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import replace
from types import MethodType
from collections import defaultdict
import argparse,json
import numpy as np
import torch,mujoco
from .config import Profile
from .env import AdaptiveEnv
from .storage import load,atomic_json
from .reward_registry import wrap
from .thumb_drive import TRANSFORM
from experiments.cube_contact_ppo.task import force_state


def run(policy,reference,profile,dt,seed,mode='policy',detail=False):
    env=wrap(AdaptiveEnv(reference,'x',physical_dt=dt,target_degrees=90.,profile=profile),TRANSFORM)
    base=env.unwrapped;rows=[];ticks=[];pieces=defaultdict(float);patched=None;total=0.
    def install():
        nonlocal patched
        env.install();m=base.monitor
        if mode=='grace_500ms':m.config=replace(m.config,contact_grace_seconds=.5)
        if patched is m:return
        original=m.checkpoint
        def checkpoint(this,q,contacts,linear,angular,displacement,delta_t=.01):
            old_error=this.error;old_stable=this.stable;old_coverage=this.last_coverage
            recovery=this.recovery_penalty;stall=this.stall_penalty;pressure=this.pressure_potential
            bonus=env.bonus_total;cost=env.cost_total
            value=original(q,contacts,linear,angular,displacement,delta_t)
            if base.warming:return value
            support=force_state(contacts['finger_forces'],base.config)[0]
            valid=not contacts['forbidden_support'] and contacts['max_penetration']<=base.config.max_penetration and displacement<=base.config.max_displacement
            progress=(old_error-this.error)/(np.pi/2)
            parts=dict(rotation_progress=profile.progress_gain*(progress if support and valid else min(progress,0.)),
                goal_hold_potential=20.*(this.stable-old_stable)/base.config.hold_seconds,
                coverage_potential=2.*(this.last_coverage-old_coverage),
                pressure_potential=0. if pressure is None else this.pressure_potential-pressure,
                missing_contact_cost=this.recovery_penalty-recovery,stall_cost=this.stall_penalty-stall,
                terminal=(50. if this.success else -20.) if this.done else 0.,
                thumb_bonus=env.bonus_total-bonus,thumb_assist_and_reverse_cost=-(env.cost_total-cost))
            parts['remaining_monitor_rates']=value-sum(parts.values())
            for key,val in parts.items():pieces[key]+=val
            if detail:
                ticks.append(dict(t=float(base.elapsed),angle=float(np.rad2deg(this.progress)),error=float(np.rad2deg(this.error)),
                    forces=contacts['finger_forces'].tolist(),phase=this.phase,tip_distance=base.tip_distances.tolist(),
                    position=base.data.xpos[base.ids['cube_body']].tolist(),displacement=float(displacement),
                    thumb_quality=env.last_metrics.get('thumb_drive_quality',0.),
                    thumb_moment=env.last_metrics.get('thumb_axis_moment_Nm',0.),parts=parts))
            return value
        m.checkpoint=MethodType(checkpoint,m);patched=m
    try:
        obs,info=env.reset(seed=seed)
        for _ in range(250):
            install()
            action=policy.predict(obs,deterministic=True)[0]
            if mode=='zero':action[:]=0.
            elif mode=='index_hold':action[0]=0.
            elif mode in ('guard_index_pinky','guard_four'):
                for finger in ([0,3] if mode=='guard_index_pinky' else range(4)):
                    if base.contacts['finger_forces'][finger+1]<.3:action[finger]=-.5
            elif mode=='thumb_half':action[4:]*=.5
            elif mode=='thumb_hold':action[4:]=0.
            before=base.data.time
            if detail:
                with torch.no_grad():
                    tensor=policy.policy.obs_to_tensor(obs)[0];z=policy.policy.features_extractor(tensor)
                    latent=policy.policy.mlp_extractor.forward_actor(z)
                    mean=policy.policy.action_net(latent).cpu().numpy()[0]
                    std=policy.policy.get_distribution(tensor).distribution.scale.cpu().numpy()[0]
                    extreme=np.argsort(np.abs(z.cpu().numpy()[0]))[-5:]
                extra=dict(mean_action=mean.tolist(),action_std=std.tolist(),max_normalized_feature=float(z.abs().max()),
                    extreme_feature_indices=extreme.tolist(),tanh_slope=(1-mean**2).tolist())
            obs,reward,done,_,info=env.step(action);total+=reward
            if detail:
                rows.append(dict(t=float(base.elapsed),physical_delta=base.data.time-before,info=info,
                    action=action.tolist(),qpos=base.data.qpos[:16].tolist(),ctrl=base.data.ctrl.tolist(),
                    normalized_ctrl=((base.data.ctrl-base.ids['ctrl_lower'])/(base.ids['ctrl_upper']-base.ids['ctrl_lower'])).tolist(),
                    tip_distance=base.tip_distances.tolist(),**extra))
            if done:break
        pieces['geometry_mount_action_costs']=total-sum(pieces.values())
        result=dict(dt=dt,seed=seed,mode=mode,valid_acceptance_trial=mode!='grace_500ms',
            final=info,return_common_new_reward=total,reward_parts=dict(pieces),steps=_+1)
        if detail:result.update(controls=rows,monitor_ticks=ticks)
        return result
    finally:env.close()


def pulse_sign(reference):
    records=[]
    for dt in (.001,.002):
        env=AdaptiveEnv(reference,'x',physical_dt=dt,target_degrees=90.)
        try:
            for motor in (0,3):
                for sign in (-1,0,1):
                    env.reset(seed=1000006)
                    for _ in range(5):env.step(np.zeros(7))
                    before=env.data.qpos.copy();initial_forces=env.contacts['finger_forces'].copy()
                    for _ in range(6):
                        a=np.zeros(7);a[motor]=sign*.5
                        _,_,done,_,info=env.step(a)
                        if done:break
                    records.append(dict(dt=dt,motor=motor,action=sign*.5,
                        flexion_change_degrees=np.rad2deg(env.data.qpos[3*motor:3*motor+3]-before[3*motor:3*motor+3]).tolist(),
                        before_force=initial_forces[motor+1],after_force=env.contacts['finger_forces'][motor+1],final=info))
        finally:env.close()
    return records


def main(goal,output):
    torch.set_num_threads(1);root=Path(goal).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    pins=dict(old_best=root/'sources/x',new_peak=root/'x/segment_006/checkpoints/step_000753664_682d0b14',
        latest=(root/'x/current/latest').resolve())
    models={key:load(path)[0] for key,path in pins.items()}
    common=Profile(**json.loads((pins['new_peak']/'metadata.json').read_text())['profile'])
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),pins={k:str(v) for k,v in pins.items()},
        common_profile=common.dictionary(),scope='Native read-only diagnosis. Same original initial states; no training changes. Counterfactual actions are diagnostic controllers, not learned success.')
    atomic_json(output/'manifest.json',manifest)
    traces=[]
    for key,policy in models.items():
        for dt in (.001,.002):
            for seed in (1000006,9400000):
                r=run(policy,pins[key]/'reference.npz',common,dt,seed,detail=True);r['policy']=key
                name=f'{key}_{dt}_{seed}.json';atomic_json(output/name,r);traces.append(dict(file=name,policy=key,
                    **{k:v for k,v in r.items() if k not in ('controls','monitor_ticks','policy')}))
                print(json.dumps(dict(phase='trace',policy=key,dt=dt,seed=seed,angle=r['final']['max_progress_degrees'],reward=r['return_common_new_reward'])),flush=True)
    atomic_json(output/'traces_summary.json',traces)
    pulses=pulse_sign(pins['new_peak']/'reference.npz');atomic_json(output/'pulse_signs.json',pulses)
    interventions=[]
    for mode in ('zero','index_hold','guard_index_pinky','guard_four','thumb_half','thumb_hold','grace_500ms'):
        for dt in (.001,.002):
            for seed in (1000006,9400000,9400001):
                r=run(models['new_peak'],pins['new_peak']/'reference.npz',common,dt,seed,mode=mode)
                interventions.append(r);atomic_json(output/'interventions.json',interventions)
        subset=[r for r in interventions if r['mode']==mode]
        print(json.dumps(dict(phase='intervention',mode=mode,max_angle=max(r['final']['max_progress_degrees'] for r in subset),
            successes=sum(r['final']['success'] for r in subset))),flush=True)
    (output/'audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(complete=True,output=str(output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--goal',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.goal,a.output)
