"""Separate XYZ quarter-turn and no-reset composition evaluation."""
import time
import numpy as np
from .env import QuarterTurnEnv
from .task import AXES


def summarize(records,requested):
    def mean(key):return float(np.mean([r[key] for r in records])) if records else None
    return dict(complete=len(records)==requested,episodes_completed=len(records),episodes_requested=requested,
        success_rate=mean('success'),mean_final_angle_error_degrees=mean('angle_error_degrees'),
        mean_best_angle_error_degrees=mean('best_angle_error_degrees'),mean_max_progress_degrees=mean('max_progress_degrees'),
        mean_max_stable_seconds=mean('max_stable_seconds'),reach30_rate=mean('reach30'),reach60_rate=mean('reach60'),
        failure_counts={reason:sum(r['failure_reason']==reason for r in records) for reason in sorted({r['failure_reason'] for r in records if r['failure_reason']})},
        records=records)


def rollout(model,env,observation,deadline=None,should_stop=lambda:False):
    for _ in range(260):
        if should_stop() or (deadline is not None and time.time()>=deadline):return None
        action=model.predict(observation,deterministic=True)[0]
        observation,reward,terminated,truncated,info=env.step(action)
        if terminated or truncated:return info
    raise RuntimeError('Quarter-turn environment did not finish within its task horizon')


def evaluate(model,reference,episodes=20,seed_start=10000,deadline=None,should_stop=lambda:False):
    groups={}
    for index,axis in enumerate(AXES):
        env=QuarterTurnEnv(reference,axis=axis);records=[]
        try:
            for episode in range(episodes):
                if should_stop() or (deadline is not None and time.time()>=deadline):break
                seed=seed_start+index*1000000+episode
                obs,_=env.reset(seed=seed,options={'allow_bank':False})
                info=rollout(model,env,obs,deadline,should_stop)
                if info is None:break
                records.append(dict(seed=seed,**info,reach30=30 in info['milestones'],reach60=60 in info['milestones']))
        finally:env.close()
        groups[axis]=summarize(records,episodes)
    complete=all(g['complete'] for g in groups.values())
    return dict(task='xyz_quarter_turn_v1',goal_degrees=90.,axis_frame='fixed_cube_face_axes_at_initial_grasp',
        complete=complete,groups=groups,axis_success_rates={a:g['success_rate'] for a,g in groups.items()},
        min_axis_success_rate=min(g['success_rate'] for g in groups.values()) if complete else None,
        episodes_per_axis=episodes,seed_start=seed_start)


def evaluate_chain(model,reference,episodes=20,seed_start=80000000,deadline=None,should_stop=lambda:False):
    env=QuarterTurnEnv(reference);records=[]
    try:
        for episode in range(episodes):
            if should_stop() or (deadline is not None and time.time()>=deadline):break
            obs,_=env.reset(seed=seed_start+episode,options={'axis':'x','allow_bank':False})
            results=[]
            for index,axis in enumerate(AXES):
                if index:obs,_=env.start_goal(axis)
                info=rollout(model,env,obs,deadline,should_stop)
                if info is None:break
                results.append(info)
                if not info['success']:break
            if not results or len(results)<3 and (deadline is not None and time.time()>=deadline or should_stop()):break
            records.append(dict(seed=seed_start+episode,success=len(results)==3 and all(r['success'] for r in results),
                completed_actions=len(results),results=results,physical_time=float(env.data.time)))
    finally:env.close()
    return dict(task='xyz_quarter_turn_chain_v1',sequence=['x90','y90','z90'],
        complete=len(records)==episodes,episodes_completed=len(records),episodes_requested=episodes,
        success_rate=float(np.mean([r['success'] for r in records])) if records else None,
        environment_resets_between_actions=False,records=records)
