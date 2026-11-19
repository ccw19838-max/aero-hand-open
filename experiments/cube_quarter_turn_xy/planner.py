"""Two-worker CEM with warm starts and resumable distribution/RNG state.

Action semantics, score and strict demonstration checks come from frozen v2.
Only scheduling, initialization, persistence and the time allocation change.
"""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np
from experiments.cube_quarter_turn_v2.env import TeacherEnv
from experiments.cube_quarter_turn_v2.planner import (KNOTS,PARAMETERS,sequence,rollout,summary,
    validate_demonstration,load_demonstration)
from experiments.cube_quarter_turn_v2.storage import atomic_json,digest,implementation_digest

_ENVS=None
_STOP=None


def worker_init(references,axis,stop):
    global _ENVS,_STOP
    _STOP=stop
    _ENVS=[TeacherEnv(path,axis) for path in references]


def worker_rollout(job):
    which,parameters,deadline=job
    result=rollout(_ENVS[which],sequence(parameters),deadline=deadline,should_stop=_STOP.is_set)
    return {**summary(result),'simulator_pid':os.getpid()}


def bounded_map(executor,jobs,stop,deadline,should_stop=lambda:False):
    """Keep only two live jobs; stopping never drains a queued population."""
    iterator=iter(enumerate(jobs));pending={}
    def submit():
        try:index,job=next(iterator)
        except StopIteration:return False
        pending[executor.submit(worker_rollout,job)]=index
        return True
    for _ in range(2):submit()
    while pending:
        if time.time()>=deadline or should_stop():stop.set()
        completed,_=wait(pending,timeout=.25,return_when=FIRST_COMPLETED)
        for future in completed:
            index=pending.pop(future)
            result=future.result()
            yield index,result
            if not stop.is_set():submit()


def initialize_distributions(references,source):
    means=np.array([np.r_[np.zeros(KNOTS*7),[7,8,8,10,27]] for _ in references],dtype=float)
    scales=np.array([np.r_[np.full((KNOTS-1)*7,.4),np.full(7,.15),[3,3,3,3,3]] for _ in references])
    source=Path(source);path=source/'best_candidate.json';archive=source/'best_candidate.npz'
    warm=None
    if path.exists() and archive.exists():
        meta=json.loads(path.read_text());ref=Path(meta['reference']).resolve()
        if ref not in [Path(r).resolve() for r in references]:raise ValueError('Warm-start grip not in approved pool')
        index=[Path(r).resolve() for r in references].index(ref)
        with np.load(archive,allow_pickle=False) as arrays:parameters=arrays['parameters'].copy()
        if parameters.shape!=(PARAMETERS,) or not np.isfinite(parameters).all():raise ValueError('Invalid warm-start parameters')
        means[index]=parameters
        scales[index]=np.r_[np.full(KNOTS*7,.18),np.full(KNOTS,2.)]
        warm=dict(reference_index=index,parameters=parameters,metadata=meta,
            source=str(source.resolve()),previous_attempts=meta.get('attempts',0))
    return means,scales,warm


def save_state(path,means,scales,rng,generation,attempts,best,best_parameters,best_reference,demos,manifest,stalls=0):
    # Arrays and metadata in ONE atomic archive avoid pairing two generations.
    payload=dict(means=means,scales=scales,
        best_parameters=np.asarray(best_parameters) if best_parameters is not None else np.empty(0),
        metadata=np.array(json.dumps(dict(generation=generation,attempts=attempts,best=best,stalls=stalls,
            best_reference=best_reference,demonstrations=demos,rng_state=rng.bit_generator.state,
            manifest=manifest),allow_nan=False)))
    path=Path(path);temp=path.with_name('.'+path.name+'.tmp')
    with temp.open('wb') as stream:np.savez_compressed(stream,**payload)
    os.replace(temp,path)


def restore_state(path,manifest):
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']))
        if meta['manifest']!=manifest:raise ValueError('Resume changes code, references, task or deadline')
        means=data['means'].copy();scales=data['scales'].copy();best_parameters=data['best_parameters'].copy()
    rng=np.random.default_rng();rng.bit_generator.state=meta['rng_state']
    return means,scales,rng,meta,best_parameters if best_parameters.size else None


def planner_digest():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def search(references,axis,source,output,deadline,population=48,seed=0,should_stop=lambda:False,resume=False):
    if axis not in ('x','y') or not references or population<8:raise ValueError('XY and a valid search population required')
    references=list(map(lambda p:str(Path(p).resolve()),references))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    checkpoint=output/'search_state.npz'
    previous_report=Path(source)/'report.json'
    previous_attempts=json.loads(previous_report.read_text())['attempts'] if previous_report.exists() else 0
    manifest=dict(axis=axis,references={p:digest(p) for p in references},deadline_epoch=deadline,
        implementation_sha256=implementation_digest(),planner_sha256=planner_digest(),population=population,workers=2)
    rng=np.random.default_rng(seed);means,scales,warm=initialize_distributions(references,source)
    best=None;best_parameters=None;best_reference=None;demos=[];attempts=0;generation=0;stalls=0
    if resume:
        means,scales,rng,meta,best_parameters=restore_state(checkpoint,manifest)
        generation=meta['generation'];attempts=meta['attempts'];best=meta['best'];best_reference=meta['best_reference'];demos=meta['demonstrations'];stalls=meta['stalls']
        for demo in demos:load_demonstration(demo)
    context=mp.get_context('spawn');stop=context.Event()
    pool=ProcessPoolExecutor(max_workers=2,mp_context=context,initializer=worker_init,initargs=(references,axis,stop))
    worker_pids=[];last_status=0.
    def status(phase='trajectory_search'):
        nonlocal last_status,worker_pids
        worker_pids=sorted(p.pid for p in pool._processes.values()) if pool._processes else worker_pids
        value=dict(phase=phase,axis=axis,pid=os.getpid(),parallel_environments=2,worker_pids=worker_pids,
            generation=generation,attempts=attempts,best=best,demonstrations=demos,
            previous_run_attempts=previous_attempts,total_attempts=previous_attempts+attempts,
            deadline_epoch=deadline,source_previous_search=str(Path(source).resolve()),
            warm_start='best parameters and validated initial grips; original v2 did not persist distribution/RNG',
            updated_at=datetime.now(timezone.utc).isoformat(),final_task_verified=False)
        atomic_json(output/'status.json',value);last_status=time.time()
    def incumbent(result,parameters,which):
        nonlocal best,best_parameters,best_reference
        if result['interrupted']:return False
        if best is None or result['score']>best['score']:
            best=result;best_parameters=parameters.copy();best_reference=references[which]
            np.savez_compressed(output/'best_candidate.npz',actions=sequence(parameters),parameters=parameters)
            atomic_json(output/'best_candidate.json',{**best,'reference':best_reference,
                'generation':generation,'attempts':attempts,'qualified_demonstration':False})
            return True
        return False
    try:
        if warm and not resume and time.time()<deadline and not should_stop():
            # Verify migrated JSON, reference and controls reproduce the source.
            which=warm['reference_index'];parameters=warm['parameters']
            initial=pool.submit(worker_rollout,(which,parameters,deadline)).result()
            if not initial['interrupted']:
                old=warm['metadata'];error=max(abs(initial['final'][k]-old['final'][k]) for k in
                    ('signed_progress_degrees','max_progress_degrees','angle_error_degrees','elapsed_seconds'))
                if error>1e-7 or initial['final']['failure_reason']!=old['final']['failure_reason']:
                    raise ValueError('Warm-start replay differs from source')
                incumbent(initial,parameters,which)
                atomic_json(output/'warm_start.json',dict(source=warm['source'],source_best_attempt=warm['previous_attempts'],
                    replay_matches=True,max_absolute_difference=error,distribution_resume=False))
        status()
        while time.time()<deadline and not should_stop():
            which=generation%len(references);mean=means[which];scale=scales[which]
            candidates=rng.normal(mean,scale,(population,PARAMETERS));candidates[0]=mean
            # Keep the carried best in the population, so adding a worker does
            # not discard the existing physically reproduced trajectory.
            if best_reference==references[which]:candidates[1]=best_parameters
            global_count=max(1,population//5)
            candidates[-global_count:,:KNOTS*7]=rng.uniform(-1.,1.,(global_count,KNOTS*7))
            candidates[2::4,(KNOTS-1)*7:KNOTS*7]=0.
            scored=[];improved=False
            jobs=[(which,p,deadline) for p in candidates]
            for index,result in bounded_map(pool,jobs,stop,deadline,should_stop):
                attempts+=1
                if result['interrupted']:continue
                scored.append((result['score'],candidates[index].copy()))
                improved=incumbent(result,candidates[index],which) or improved
                if time.time()-last_status>=10.:status()
                if result['success']:
                    # Other worker is idle/finishing its at most one pending
                    # candidate; perform both strict validation rollouts there.
                    actions=sequence(candidates[index])
                    status('validating_demonstration')
                    approved,checks=validate_demonstration(references[which],axis,actions,deadline,should_stop)
                    if approved:
                        path=output/f'demo_{len(demos):03d}.npz';data=checks[0]
                        np.savez_compressed(path,observations=data['observations'],actions=data['actions'])
                        atomic_json(path.with_suffix('.json'),dict(approved=True,axis=axis,goal_degrees=90.,
                            reference=references[which],reference_sha256=digest(references[which]),sha256=digest(path),
                            implementation_sha256=implementation_digest(),replay_checks=[summary(c) for c in checks]))
                        atomic_json(path.with_suffix('.trace.json'),data['trace']);demos.append(str(path.resolve()))
                        # One qualified full demonstration is enough to unlock
                        # the existing BC/PPO gate; failure prefixes never do.
                        stop.set()
            if scored:
                elites=np.array([p for _,p in sorted(scored,key=lambda v:v[0],reverse=True)[:max(2,population//6)]])
                means[which]=.7*elites.mean(0)+.3*mean
                scales[which]=np.maximum(.7*elites.std(0)+.3*scale,np.r_[np.full(KNOTS*7,.08),np.ones(KNOTS)])
            stalls=0 if improved else stalls+1
            generation+=1
            # Long-running search retains broad restarts rather than collapsing
            # to one local mode for the entire remaining wall-clock budget.
            if stalls>=30:
                scales=np.maximum(scales,np.r_[np.full(KNOTS*7,.25),np.full(KNOTS,2.)]);stalls=0
            save_state(checkpoint,means,scales,rng,generation,attempts,best,best_parameters,best_reference,demos,manifest,stalls)
            status()
            if demos:break
    finally:
        stop.set();pool.shutdown(wait=True,cancel_futures=True)
        save_state(checkpoint,means,scales,rng,generation,attempts,best,best_parameters,best_reference,demos,manifest,stalls)
    report=dict(axis=axis,attempts=attempts,generation=generation,best=best,demonstrations=demos,
        parallel_environments=2,worker_pids=worker_pids,deadline_epoch=deadline,
        status='demonstrations_ready' if demos else 'stopped_without_qualified_demonstration',
        stop_reason='demonstration' if demos else 'signal' if should_stop() else 'deadline',final_task_verified=False)
    atomic_json(output/'report.json',report);return report
