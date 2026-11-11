"""Passive failure evidence; never creates weights or a preparation approval."""
from dataclasses import replace
import json
from pathlib import Path
import time


def record_diagnostic(reference, output, config=None, seconds=30.):
    """Repeat a diagnostic reset every 10s, with fixed actuator targets.

    Physics continues after the task's latched failure so the fall is visible.
    These are passive simulations of an unapproved reset, not policy tests.
    """
    import os
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('EGL_PLATFORM','surfaceless')
    os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
    import jax
    import mujoco
    import mediapy
    import numpy as np
    from PIL import Image,ImageDraw
    from .config import PenSpinConfig
    from .env import make_env,SUBSTEP_FIELDS,_substep_row
    from .evaluate import _reference_markers
    from .trajectory import advance_trace,FAILURE_NAMES

    reference,output = Path(reference),Path(output)
    if config is None:
        config = PenSpinConfig.load(reference.parent/'config.json')
    config = replace(config,reset_position_noise=0.,reset_angle_noise_degrees=0.)
    if seconds <= 0 or abs(seconds/config.ctrl_dt-round(seconds/config.ctrl_dt))>1e-8:
        raise ValueError('Duration must be a positive whole number of control periods')
    output.mkdir(parents=True,exist_ok=True)
    config.save(output/'diagnostic_config.json')
    env = make_env(config,reference,goal_angle=2*np.pi,backend='native')
    def update(trace,sample):
        trace,reward,metrics = advance_trace(trace,sample,config.sim_dt,config)
        return trace,_substep_row(trace,sample,reward)
    update = jax.jit(update)
    renderer = mujoco.Renderer(env.mj_model,height=480,width=640)
    frame_count = round(seconds/config.ctrl_dt)
    frames_per_episode = round(config.episode_seconds/config.ctrl_dt)
    records,qpos,qvel,global_times,episode_ids,labels,events = [],[],[],[],[],[],[]
    start = time.monotonic()
    try:
        with mediapy.VideoWriter(str(output/'diagnostic.mp4'),shape=(480,640),fps=1/config.ctrl_dt) as writer:
            for frame_id in range(frame_count):
                reset = frame_id%frames_per_episode==0
                episode = frame_id//frames_per_episode+1
                if reset:
                    state = env.reset(jax.random.PRNGKey(0))
                    data,trace = state.data,state.info['trace']
                    event_saved = False
                for _ in range(env.n_substeps):
                    mujoco.mj_step(env.mj_model,data)
                    trace,row = update(trace,env._sample(data))
                    records.append(np.asarray(row))
                    qpos.append(data.qpos.copy());qvel.append(data.qvel.copy())
                    global_times.append(len(records)*config.sim_dt);episode_ids.append(episode)
                    if bool(trace.done) and not event_saved:
                        events.append(dict(episode=episode,event_seconds=float(data.time),
                                           failure_reason=FAILURE_NAMES[int(trace.failure_reason)],
                                           task_success=bool(trace.success)))
                        event_saved = True
                mujoco.mj_forward(env.mj_model,data)
                renderer.update_scene(data,camera='overview')
                _reference_markers(renderer.scene,env.reference,mujoco)
                frame = Image.fromarray(renderer.render())
                drawing = ImageDraw.Draw(frame)
                drawing.rectangle((0,0,640,66),fill='white')
                drawing.text((8,5),'PASSIVE DIAGNOSTIC | NO TRAINED POLICY | UNAPPROVED RESET',fill='black')
                reason = FAILURE_NAMES[int(trace.failure_reason)]
                drawing.text((8,25),f'Episode {episode} | physics {(frame_id+1)*config.ctrl_dt:.2f}s | task: {reason}',fill='black')
                drawing.text((8,45),'Blue: fixed thumb axis | Orange: positive rotation | fixed motor targets',fill='black')
                array = np.asarray(frame)
                writer.add_image(array)
                if frame_id==0:
                    frame.save(output/'preview.png')
                labels.append(dict(frame=frame_id,episode=episode,reset=reset,
                                   physical_seconds=(frame_id+1)*config.ctrl_dt,
                                   episode_seconds=float(data.time),failure_reason=reason))
    finally:
        renderer.close()
    np.savez_compressed(output/'diagnostic_trajectory.npz',
                        fields=np.array(SUBSTEP_FIELDS),trace=np.asarray(records),
                        physical_seconds=np.asarray(global_times),episode=np.asarray(episode_ids),
                        qpos=np.asarray(qpos),qvel=np.asarray(qvel))
    (output/'video_trajectory.json').write_text(json.dumps(labels,indent=2)+'\n')
    info = dict(status='preparation_failed',artifact_type='passive_diagnostic',
                policy_checkpoint=None,trained_policy=False,approved_reference=False,
                reference=str(reference.resolve()),seconds=seconds,
                integrated_physics_seconds=len(records)*config.sim_dt,
                frames=frame_count,fps=1/config.ctrl_dt,physics_samples=len(records),
                episodes=episode,reset_rule='Repeat the same seed-0 diagnostic initial state every10s',
                task_failures=events,heldout_policy_evaluation_performed=False,
                wall_seconds=time.monotonic()-start,
                task_success_not_implied_by_video_completion=True)
    (output/'diagnostic_report.json').write_text(json.dumps(info,indent=2)+'\n')
    return info
