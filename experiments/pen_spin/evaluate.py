"""Independent CPU policy evaluation and genuine-time, reset-labelled videos."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import jax
import jax.numpy as jp
import numpy as np

from .policy import load_checkpoint, inference_function


def _scalar_metrics(state):
    result = {}
    for key, value in state.metrics.items():
        arr = np.asarray(value)
        if arr.size == 1:
            result[key] = float(arr.reshape(()))
    return result


def _summary(seed, rows, dt):
    final = rows[-1]
    return dict(seed=seed, steps=len(rows), seconds=final['time'],control_timeline_seconds=len(rows)*dt,
                return_value=sum(r['reward'] for r in rows),
                success=bool(final['success']), failure=bool(final['failure']),
                reached_spin=any(int(r['phase']) == 1 for r in rows),
                reached_catch=any(int(r['phase']) == 2 or r['success'] for r in rows),
                theta=final['theta'], phi=final['phi'],
                max_theta=max(r['theta'] for r in rows),
                max_phi=max(r['phi'] for r in rows),
                max_joint_angle_peak=max(min(r['theta'], r['phi']) for r in rows),
                max_valid_progress=max(r.get('valid_progress_radians',0.) for r in rows),
                max_stable_seconds=max(r.get('stable_seconds',r.get('capture',0.)) for r in rows),
                peak_tilt_degrees=max(r.get('tilt_degrees',0.) for r in rows),
                mean_action_change_squared=float(np.mean([r.get('action_change_squared',0.) for r in rows])),
                observed_dropped=any(bool(r.get('dropped',False)) for r in rows),
                final_metrics=final)


def _row(state, action, step, include_substeps=False):
    metrics = _scalar_metrics(state)
    for name in ('success', 'failure', 'phase', 'theta', 'phi'):
        if name not in metrics:
            raise ValueError(f'Environment omitted required evaluation metric: {name}')
    metrics.update(step=step, time=float(state.data.time), reward=float(state.reward),
                   action=np.asarray(action).tolist())
    if include_substeps and 'substep_trace' in state.info:
        trace=np.asarray(state.info['substep_trace'])
        # Terminal padding repeats elapsed time; retain only integrated steps.
        elapsed=trace[:,0]
        unique=np.r_[True,np.diff(elapsed)>1e-9] & (elapsed>0)
        metrics['substeps']=trace[unique].tolist()
    return metrics


def _episode(env, params, apply, seed, collect=False, advance=None, collect_substeps=False):
    state = env.reset(jax.random.PRNGKey(seed))
    if advance is None:
        advance = jax.jit(env.step) if env.backend == 'mjx' else env.step
    rows, states = [], []
    for step in range(int(round(env.config.episode_seconds/env.config.ctrl_dt))):
        action = apply(params, state.obs, jax.random.PRNGKey(seed+step+1))
        if not np.isfinite(np.asarray(action)).all():
            raise ValueError('Policy produced nonfinite actions')
        state = advance(state, action)
        rows.append(_row(state, action, step+1,collect_substeps))
        if collect:
            # Native MjData is mutable; keep numerical snapshots, not aliases.
            states.append(dict(qpos=np.array(state.data.qpos,copy=True),
                               qvel=np.array(state.data.qvel,copy=True),time=float(state.data.time)))
        if bool(state.done):
            break
    return _summary(seed, rows, env.config.ctrl_dt), rows, states


def _parallel_episodes(env, params, apply, seeds,collect_substeps=False):
    """Four MJX evaluations per batch; completed states remain frozen."""
    width = min(env.config.num_eval_envs, len(seeds))
    reset = jax.jit(jax.vmap(env.reset))
    def advance(state, action, finished):
        proposed = jax.vmap(env.step)(state, action)
        def select(new, old):
            mask = finished.reshape((width,)+(1,)*(new.ndim-1))
            return jp.where(mask, old, new)
        return jax.tree.map(select, proposed, state)
    advance = jax.jit(advance)
    for offset in range(0,len(seeds),width):
        actual = seeds[offset:offset+width]
        batch = actual+[actual[-1]]*(width-len(actual))
        state = reset(jp.array([jax.random.PRNGKey(seed) for seed in batch]))
        finished = np.zeros(width,dtype=bool)
        trajectories = [[] for _ in batch]
        for step in range(round(env.config.episode_seconds/env.config.ctrl_dt)):
            action = apply(params,state.obs,jax.random.PRNGKey(batch[0]+step+1))
            if not np.isfinite(np.asarray(action)).all():
                raise ValueError('Policy produced nonfinite actions')
            state = advance(state,action,jp.asarray(finished))
            for i in range(len(actual)):
                if not finished[i]:
                    one = jax.tree.map(lambda x:x[i],state)
                    trajectories[i].append(_row(one,action[i],step+1,collect_substeps))
            finished |= np.asarray(state.done,dtype=bool)
            if finished[:len(actual)].all():break
        for seed,rows in zip(actual,trajectories):
            yield _summary(seed,rows,env.config.ctrl_dt), rows, []


def evaluate_params(config, reference, params, goal_angle, episodes=20, seed_start=1000,
                    backend='mjx', output=None):
    """Same evaluation implementation for curriculum gates and final reports."""
    if episodes < 1:
        raise ValueError('At least one evaluation episode is required')
    from .env import make_env
    env = make_env(config, reference, goal_angle=goal_angle, backend=backend)
    apply = inference_function(params, config)
    advance = jax.jit(env.step) if backend == 'mjx' else env.step
    results, failures = [], []
    start = time.monotonic()
    seeds = list(range(seed_start,seed_start+episodes))
    rollouts = (_parallel_episodes(env,params,apply,seeds,output is not None) if backend=='mjx' else
                (_episode(env,params,apply,seed,advance=advance,collect_substeps=output is not None) for seed in seeds))
    for summary, rows, _ in rollouts:
        results.append(summary)
        if not summary['success'] and len(failures) < 10:
            failures.append(dict(summary=summary, trajectory=rows))
        if output is not None:
            destination=Path(output);destination.mkdir(parents=True,exist_ok=True)
            if any('substeps' in row for row in rows):
                from .env import SUBSTEP_FIELDS
                trace=np.asarray([record for row in rows for record in row.get('substeps',[])],dtype=np.float32)
                trajectory_directory=destination/'trajectories'
                trajectory_directory.mkdir(exist_ok=True)
                np.savez_compressed(trajectory_directory/f"episode_{summary['seed']}.npz",
                                    seed=summary['seed'],fields=np.asarray(SUBSTEP_FIELDS),trace=trace,
                                    actions=np.asarray([row['action'] for row in rows],dtype=np.float32))
            partial=dict(status='running',requested_episodes=episodes,completed_episodes=len(results),
                         goal_angle=float(goal_angle),backend=backend,results=results,
                         success_rate=float(np.mean([r['success'] for r in results])))
            temporary=destination/'.partial_evaluation.tmp'
            temporary.write_text(json.dumps(partial,indent=2)+'\n')
            os.replace(temporary,destination/'partial_evaluation.json')
            (destination/'failure_samples.json').write_text(json.dumps(failures,indent=2)+'\n')
    success = np.array([r['success'] for r in results], dtype=float)
    report = dict(status='completed',episodes=len(results),completed_episodes=len(results),requested_episodes=episodes,
                  seed_start=seed_start, seed_end=seed_start+episodes-1,
                  goal_angle=float(goal_angle), backend=backend,
                  truth_observations=True, task_scope='simulation only; fixed upright right hand',
                  success_rate=float(success.mean()), success_rate_std=float(success.std()),
                  failure_rate=float(np.mean([r['failure'] for r in results])),
                  reached_spin_rate=float(np.mean([r['reached_spin'] for r in results])),
                  reached_catch_rate=float(np.mean([r['reached_catch'] for r in results])),
                  mean_return=float(np.mean([r['return_value'] for r in results])),
                  std_return=float(np.std([r['return_value'] for r in results])),
                  mean_episode_seconds=float(np.mean([r['seconds'] for r in results])),
                  median_valid_progress=float(np.median([r['max_valid_progress'] for r in results])),
                  median_joint_angle_peak=float(np.median([r['max_joint_angle_peak'] for r in results])),
                  progress_sampling='Valid-envelope progress sampled at control boundaries; full 2ms evidence in trajectory files',
                  median_stable_seconds=float(np.median([r['max_stable_seconds'] for r in results])),
                  peak_tilt_degrees_mean=float(np.mean([r['peak_tilt_degrees'] for r in results])),
                  mean_action_change_squared=float(np.mean([r['mean_action_change_squared'] for r in results])),
                  evaluation_parallelism=min(config.num_eval_envs,episodes) if backend=='mjx' else 1,
                  wall_seconds=time.monotonic()-start, results=results)
    for metric in ('theta','phi','tilt_degrees','action_change_squared'):
        if all(metric in r['final_metrics'] for r in results):
            values = [r['final_metrics'][metric] for r in results]
            report['final_'+metric+'_mean'] = float(np.mean(values))
            report['final_'+metric+'_std'] = float(np.std(values))
    if all('failure_reason' in r['final_metrics'] for r in results):
        from .trajectory import NO_FAILURE, TIMEOUT, ENVELOPE, LOST_CONTACT, PENETRATION, SUPPORT, NONFINITE
        names = {NO_FAILURE:'none',TIMEOUT:'timeout',ENVELOPE:'envelope',
                 LOST_CONTACT:'lost_contact',PENETRATION:'penetration',SUPPORT:'support',NONFINITE:'nonfinite'}
        report['failure_reason_counts'] = {name:sum(int(r['final_metrics']['failure_reason'])==code for r in results)
                                           for code,name in names.items()}
    # Only report failure causes when the environment explicitly supplies them.
    for metric, field in [('height_drop','height_drop_rate'),('invalid_contact','invalid_contact_rate')]:
        if all(metric in r['final_metrics'] for r in results):
            report[field] = float(np.mean([bool(r['final_metrics'][metric]) for r in results]))
    if all('dropped' in r['final_metrics'] for r in results):
        report['observed_drop_rate']=float(np.mean([r['observed_dropped'] for r in results]))
        report['drop_definition']='No finger force >= contact_force and pen center world Z below reference pivot by > max_center_distance; observed before task termination only'
    if output is not None:
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        (output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
        (output/'failure_samples.json').write_text(json.dumps(failures,indent=2)+'\n')
        report['substep_trajectory_files']=[f"trajectories/episode_{result['seed']}.npz" for result in results
                                          if (output/'trajectories'/f"episode_{result['seed']}.npz").is_file()]
        (output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def _reference_markers(scene, reference, mujoco):
    """Blue reference axis and orange right-hand arc, independent of hand motion."""
    axis=np.asarray(reference['n0']);center=np.asarray(reference['c0'])
    helper=np.array([1.,0.,0.]) if abs(axis[0])<.9 else np.array([0.,1.,0.])
    e1=helper-axis*np.dot(axis,helper);e1/=np.linalg.norm(e1)
    e2=np.cross(axis,e1)
    def connector(start,end,arrow,color):
        if scene.ngeom>=scene.maxgeom:return
        geom=scene.geoms[scene.ngeom]
        kind=mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_LINE
        mujoco.mjv_initGeom(geom,kind,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color))
        mujoco.mjv_connector(geom,kind,.002 if arrow else .001,start,end)
        scene.ngeom+=1
    connector(center,center+.045*axis,True,[.1,.4,1.,1.])
    angles=np.linspace(0,1.6*math.pi,18)
    points=[center+.028*(math.cos(a)*e1+math.sin(a)*e2) for a in angles]
    for i,(start,end) in enumerate(zip(points,points[1:])):
        connector(start,end,i==len(points)-2,[1.,.5,.05,1.])


def record_video(config, reference, params, goal_angle, output, seconds=30., backend='native', seed=20000):
    """Each output frame advances exactly one control period; episodes are labelled."""
    import os
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('EGL_PLATFORM','surfaceless')
    os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
    import mujoco
    import mediapy
    from PIL import Image, ImageDraw
    from .env import make_env
    if seconds <= 0:
        raise ValueError('Video duration must be positive')
    env = make_env(config,reference,goal_angle=goal_angle,backend=backend)
    apply = inference_function(params,config)
    frame_count = int(round(seconds/config.ctrl_dt))
    if abs(frame_count*config.ctrl_dt-seconds)>1e-8:
        raise ValueError('Video duration must be a whole number of control periods')
    model = env.mj_model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model,height=480,width=640)
    advance = jax.jit(env.step) if backend=='mjx' else env.step
    if backend=='mjx':
        from mujoco import mjx
        physics_step=jax.jit(lambda d:mjx.step(env.mjx_model,d))
    state = env.reset(jax.random.PRNGKey(seed))
    frames, labels = [], []
    episode, episode_steps,integrated_seconds = 1, 0,0.
    try:
        for index in range(frame_count):
            reset = False
            if bool(state.done):
                episode += 1; episode_steps = 0; reset = True
                state = env.reset(jax.random.PRNGKey(seed+episode-1))
            action = apply(params,state.obs,jax.random.PRNGKey(seed+index+1))
            previous_time=float(state.data.time)
            state = advance(state,action); episode_steps += 1
            event_advance=float(state.data.time)-previous_time
            remaining=round((config.ctrl_dt-event_advance)/config.sim_dt)
            if remaining<0 or not 0<event_advance<=config.ctrl_dt+1e-5:
                raise ValueError('Unexpected physics time advancement during video')
            render_state=state.data
            if backend=='mjx':
                for _ in range(remaining):render_state=physics_step(render_state)
            for name in ('qpos','qvel','ctrl','mocap_pos','mocap_quat'):
                if hasattr(render_state,name):
                    getattr(data,name)[:] = np.asarray(getattr(render_state,name))
            data.time=float(render_state.time)
            if backend=='native':
                # Only the video copy advances after a terminal task event.
                # Training and success metrics remain frozen at the event.
                for _ in range(remaining):mujoco.mj_step(model,data)
            integrated_seconds+=float(data.time)-previous_time
            mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera='overview')
            _reference_markers(renderer.scene,env.reference,mujoco)
            frame=Image.fromarray(renderer.render())
            drawing=ImageDraw.Draw(frame)
            drawing.rectangle((0,0,640,48),fill='white')
            label=f'Episode {episode} | episode {episode_steps*config.ctrl_dt:.2f}s | physics {integrated_seconds:.2f}s'
            if reset: label += ' | RESET'
            if remaining:label+=f' | TERMINAL +{remaining*config.sim_dt:.3f}s physics'
            drawing.text((8,7),label,fill='black')
            drawing.text((8,28),'Blue: reference axis | Orange: positive right-hand rotation',fill='black')
            frames.append(np.asarray(frame))
            labels.append(dict(frame=index,episode=episode,episode_step=episode_steps,reset=reset,
                               total_sim_seconds=integrated_seconds,
                               terminal_event_step_seconds=event_advance,
                               post_terminal_physics_seconds=remaining*config.sim_dt,
                               **_scalar_metrics(state)))
    finally:
        renderer.close()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    mediapy.write_video(str(output/'rollout.mp4'),frames,fps=1/config.ctrl_dt)
    (output/'video_trajectory.json').write_text(json.dumps(labels,indent=2)+'\n')
    info=dict(seconds=seconds,frames=frame_count,fps=1/config.ctrl_dt,episodes=episode,
              integrated_physics_seconds=integrated_seconds,
              backend=backend,resets_labelled=True,frames_are_actual_simulated_control_steps=True,
              terminal_remainder_policy='Continue physics on video copy to control boundary; task metrics freeze at terminal event',
              task_success_not_implied_by_video_completion=True)
    (output/'video_info.json').write_text(json.dumps(info,indent=2)+'\n')
    return info


def _evaluation_worker(request):
    params,config,reference,metadata=load_checkpoint(request['checkpoint'])
    goal=request['goal_angle']
    report=evaluate_params(config,reference,params,goal,request['episodes'],
                           request['seed_start'],request['backend'],request['output'])
    if request['video_seconds']:
        report['video']=record_video(config,reference,params,goal,request['output'],
                                      request['video_seconds'],request['backend'],request['seed_start'])
    report['checkpoint_metadata']=metadata
    report['trained_stage_goal_angle']=metadata['goal_angle']
    report['evaluated_full_turn_task']=abs(goal-2*math.pi)<1e-6
    report['evaluation_deadline_unix']=request.get('evaluation_deadline_unix')
    report['standalone_budget']=request.get('standalone_budget',False)
    (Path(request['output'])/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--checkpoint',type=Path)
    source.add_argument('--diagnostic-reference',type=Path,help='Record failed preparation with fixed motor targets; no learned policy is used')
    parser.add_argument('--output',type=Path,default=Path('experiments/pen_spin/artifacts/evaluation'))
    parser.add_argument('--episodes',type=int,default=100)
    parser.add_argument('--seed-start',type=int,default=10000)
    parser.add_argument('--backend',choices=['native','mjx'],default='native')
    parser.add_argument('--video-seconds',type=float,default=30.)
    parser.add_argument('--goal-degrees',type=float,help='Evaluate a different curriculum goal, including the full 360-degree task')
    parser.add_argument('--budget-seconds',type=float,help='Explicit later standalone evaluation budget; excluded from the original first-batch report')
    parser.add_argument('--_worker',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args._worker:return _evaluation_worker(json.loads(args._worker.read_text()))
    if args.diagnostic_reference:
        from .diagnostics import record_diagnostic
        config_path=args.diagnostic_reference.parent/'config.json'
        from .config import PenSpinConfig
        config=PenSpinConfig.load(config_path) if config_path.exists() else PenSpinConfig()
        report=record_diagnostic(args.diagnostic_reference,args.output,config=config,seconds=args.video_seconds)
        print(json.dumps(report,indent=2))
        return 0
    if not args.checkpoint:parser.error('--checkpoint is required')
    if args.episodes<1 or args.video_seconds<0:parser.error('episodes must be positive; video-seconds must be nonnegative')
    params,config,reference,metadata=load_checkpoint(args.checkpoint)
    goal=metadata['goal_angle'] if args.goal_degrees is None else math.radians(args.goal_degrees)
    if not 0<=goal<=2*math.pi+1e-8:parser.error('goal-degrees must be in [0,360]')
    if args.budget_seconds is not None and not 0<args.budget_seconds<=86400:
        parser.error('budget-seconds must be in (0,86400]')
    deadline=time.time()+args.budget_seconds if args.budget_seconds is not None else metadata.get('batch_deadline_unix')
    if not deadline:parser.error('Checkpoint lacks a batch deadline; create a prepared training bundle first')
    args.output.mkdir(parents=True,exist_ok=True)
    request=dict(checkpoint=str(args.checkpoint.resolve()),output=str(args.output.resolve()),
                 episodes=args.episodes,seed_start=args.seed_start,backend=args.backend,
                 video_seconds=args.video_seconds,goal_angle=goal,evaluation_deadline_unix=deadline,
                 standalone_budget=args.budget_seconds is not None)
    request_path=args.output/'evaluation_request.json'
    request_path.write_text(json.dumps(request,indent=2)+'\n')
    (args.output/'partial_evaluation.json').unlink(missing_ok=True)
    remaining=deadline-time.time()
    if remaining<=0:
        report=dict(status='budget_expired',completed_episodes=0,requested_episodes=args.episodes,
                    batch_deadline_unix=deadline,full_skill_success_not_assumed=True)
        (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2));return 2
    command=[sys.executable,'-m','experiments.pen_spin.evaluate','--_worker',str(request_path.resolve())]
    with (args.output/'worker.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=process.wait(timeout=max(0.,deadline-time.time()))
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
            partial=args.output/'partial_evaluation.json'
            report=json.loads(partial.read_text()) if partial.exists() else dict(completed_episodes=0,requested_episodes=args.episodes)
            report.update(status='budget_stopped',batch_deadline_unix=deadline,
                          video_completed=False,full_skill_success_not_assumed=True)
            (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
            return 0
    if code:
        partial=args.output/'partial_evaluation.json'
        report=json.loads(partial.read_text()) if partial.exists() else dict(completed_episodes=0,requested_episodes=args.episodes)
        report.update(status='error',worker_exit_code=code,log=str(args.output/'worker.log'),full_skill_success_not_assumed=True)
        (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    else:report=json.loads((args.output/'evaluation.json').read_text())
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','checkpoint_metadata')},indent=2))
    return code


if __name__=='__main__':
    raise SystemExit(main())
