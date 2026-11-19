"""Bounded staged-action CEM. Only strictly replayed 90+hold successes are demos."""
from dataclasses import replace
from pathlib import Path
import json
import time
import numpy as np
from .env import TeacherEnv
from .task import Config
from .storage import atomic_json,digest,implementation_digest

KNOTS=5
PARAMETERS=KNOTS*7+KNOTS


def sequence(parameters):
    parameters=np.asarray(parameters,dtype=float)
    if parameters.shape!=(PARAMETERS,) or not np.isfinite(parameters).all():raise ValueError('Invalid staged plan')
    knots=np.clip(parameters[:KNOTS*7].reshape(KNOTS,7),-1.,1.)
    # Release / drive / transfer / capture / final capture. Last command remains
    # constant for >=1.25s, allowing the actual 1-second stability criterion.
    durations=np.rint(np.clip(parameters[KNOTS*7:],[3,3,3,4,25],[18,18,18,20,40])).astype(int)
    actions=[np.zeros(7) for _ in range(5)]  # normal .25s warmup, no object fixation
    previous=np.zeros(7)
    for stage,(knot,count) in enumerate(zip(knots,durations)):
        blend=1 if stage==4 else min(4,count)
        for i in range(count):actions.append(previous+(knot-previous)*min(1.,(i+1)/blend))
        previous=knot
    return np.asarray(actions,dtype=np.float32)


def rollout(env,actions,seed=0,reference=True,deadline=float('inf'),should_stop=lambda:False,record=False):
    obs,info=env.reset(seed=seed,options={'reference':reference})
    observations=[];used=[];trace=[];best=0.;total=0.;done=False;interrupted=False
    for action in actions:
        if time.time()>=deadline or should_stop():interrupted=True;break
        if record:observations.append(obs.copy());used.append(np.asarray(action).copy())
        obs,reward,done,_,info=env.step(action);total+=reward
        valid=not(env.contacts['forbidden_support'] or env.contacts['max_penetration']>env.config.max_penetration
            or info['displacement']>env.config.max_displacement) and info['hand_force']>=env.config.contact_force
        progress=min(info['signed_progress_degrees'],90.-info['angle_error_degrees'])
        if valid:best=max(best,progress)
        if record:trace.append({**info,'cube_qpos':env.data.qpos[16:23].tolist()})
        if done:break
    final=min(info['signed_progress_degrees'],90.-info['angle_error_degrees'])
    penalty=20. if info['failure_reason'] else 0.
    score=best+.6*final+30.*info['max_stable_seconds']-penalty
    score-=.1*min(100.,info['angular_speed'])+2.*min(10.,info['linear_speed'])
    fingers=np.asarray(info['finger_forces'])>=env.config.contact_force
    secure=bool(fingers[0] and np.count_nonzero(fingers[1:])>=2)
    capture=np.exp(-(info['angle_error_degrees']/20.)**2)
    capture*=np.exp(-(info['linear_speed']/.03)**2-(info['angular_speed']/.5)**2)
    if secure:score+=20.*float(capture)
    # A soft terminal grip preference helps different axes end in a reusable
    # grasp. It never replaces the independent no-reset chain test.
    if info['angle_error_degrees']<20.:
        score-=2.*float(np.mean(((env.data.ctrl-env.default)/(env.ids['ctrl_upper']-env.ids['ctrl_lower']))**2))
    if info['success']:score+=10000.
    return dict(score=float(score),success=bool(info['success'] and not interrupted),interrupted=interrupted,
        best_valid_progress_degrees=float(best),return_=float(total),final=info,
        observations=np.asarray(observations,dtype=np.float32),actions=np.asarray(used,dtype=np.float32),trace=trace)


def summary(result):return {k:v for k,v in result.items() if k not in ('observations','actions','trace')}


def validate_demonstration(reference,axis,actions,deadline,should_stop=lambda:False):
    results=[]
    for dt in (.002,.001):
        env=TeacherEnv(reference,axis,physical_dt=dt)
        try:results.append(rollout(env,actions,deadline=deadline,should_stop=should_stop,record=dt==.002))
        finally:env.close()
    approved=all(r['success'] and not r['interrupted'] for r in results)
    return approved,results


def load_demonstration(path):
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
    if meta.get('approved') is not True or meta.get('goal_degrees')!=90.:
        raise ValueError('Only full successful demonstrations are usable')
    if meta['implementation_sha256']!=implementation_digest() or meta['sha256']!=digest(path):
        raise ValueError('Demonstration semantics/checksum mismatch')
    if meta['reference_sha256']!=digest(meta['reference']):raise ValueError('Demonstration reference changed')
    if not all(c['success'] and not c['interrupted'] for c in meta['replay_checks']):
        raise ValueError('Demonstration did not pass both replay checks')
    with np.load(path,allow_pickle=False) as data:
        observations=data['observations'].copy();actions=data['actions'].copy()
    if observations.ndim!=2 or observations.shape[1]!=133 or actions.shape!=(len(observations),7):
        raise ValueError('Malformed demonstration')
    if not np.isfinite(observations).all() or not np.isfinite(actions).all() or np.max(np.abs(actions))>1.:
        raise ValueError('Nonfinite/out-of-range demonstration')
    return observations,actions,meta


def search(references,axis,output,deadline,population=48,generations=10000,seed=0,should_stop=lambda:False):
    if not references:raise ValueError('A validated initial grip is required')
    if population<8 or generations<1:raise ValueError('Invalid search budget')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    envs=[TeacherEnv(path,axis) for path in references]
    rng=np.random.default_rng(seed);best=None;best_actions=None;best_reference=None;demos=[];attempts=0
    means=[np.r_[np.zeros(KNOTS*7),[7,8,8,10,27]].astype(float) for _ in references]
    scales=[np.r_[np.full((KNOTS-1)*7,.4),np.full(7,.15),[3,3,3,3,3]].astype(float) for _ in references]
    try:
        for generation in range(generations):
            if time.time()>=deadline or should_stop():break
            which=generation%len(envs);env=envs[which];mean=means[which];scale=scales[which]
            candidates=rng.normal(mean,scale,(population,PARAMETERS));candidates[0]=mean
            # Persistent global samples avoid an early small-angle CEM collapse.
            global_count=max(1,population//5)
            candidates[-global_count:,:KNOTS*7]=rng.uniform(-1.,1.,(global_count,KNOTS*7))
            # Every fourth proposal explicitly returns to its legal reference
            # actuator targets for capture, instead of randomizing all 7 there.
            candidates[1::4,(KNOTS-1)*7:KNOTS*7]=0.
            scored=[]
            for parameters in candidates:
                if time.time()>=deadline or should_stop():break
                actions=sequence(parameters)
                result=rollout(env,actions,deadline=deadline,should_stop=should_stop);attempts+=1
                scored.append((result['score'],parameters.copy()))
                if best is None or result['score']>best['score']:
                    best=summary(result);best_actions=actions;best_reference=references[which]
                    np.savez_compressed(output/'best_candidate.npz',actions=actions,parameters=parameters)
                    atomic_json(output/'best_candidate.json',{**best,'reference':str(best_reference),
                        'generation':generation,'attempts':attempts,'qualified_demonstration':False})
                if result['success']:
                    approved,checks=validate_demonstration(references[which],axis,actions,deadline,should_stop)
                    if approved:
                        path=output/f'demo_{len(demos):03d}.npz';data=checks[0]
                        np.savez_compressed(path,observations=data['observations'],actions=data['actions'])
                        atomic_json(path.with_suffix('.json'),dict(approved=True,axis=axis,goal_degrees=90.,
                            reference=str(Path(references[which]).resolve()),reference_sha256=digest(references[which]),
                            sha256=digest(path),implementation_sha256=implementation_digest(),
                            replay_checks=[summary(c) for c in checks],trace_file=path.with_suffix('.trace.json').name))
                        atomic_json(path.with_suffix('.trace.json'),data['trace']);demos.append(str(path.resolve()))
                        if len(demos)>=3:break
            if scored:
                elites=np.array([p for _,p in sorted(scored,key=lambda item:item[0],reverse=True)[:max(2,population//6)]])
                means[which]=.7*elites.mean(0)+.3*mean
                scales[which]=np.maximum(.7*elites.std(0)+.3*scale,np.r_[np.full(KNOTS*7,.08),np.ones(KNOTS)])
            status=dict(phase='trajectory_search',axis=axis,generation=generation,attempts=attempts,
                best=best,demonstrations=demos,deadline_epoch=deadline)
            atomic_json(output/'status.json',status)
            if len(demos)>=3:break
        if best_actions is not None and time.time()<deadline and not should_stop():
            # Preserve the best FAILED trajectory explicitly, too, for diagnosis.
            index=references.index(best_reference)
            diagnostic=rollout(envs[index],best_actions,deadline=deadline,should_stop=should_stop,record=True)
            atomic_json(output/'best_trajectory.json',dict(summary=summary(diagnostic),trace=diagnostic['trace']))
    finally:
        for env in envs:env.close()
    report=dict(axis=axis,attempts=attempts,best=best,demonstrations=demos,
        status='demonstrations_ready' if demos else 'blocked_no_legal_90_degree_demonstration',
        goal_degrees=90.,deadline_epoch=deadline,final_task_verified=False)
    atomic_json(output/'report.json',report)
    return report
