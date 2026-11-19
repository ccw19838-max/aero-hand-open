"""Four independent checkpoint replays and a complete-hand view, with provenance.

This module only evaluates frozen policies; it never modifies the trainer or its
observation statistics. Recorded states are rendered offline with the same model.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')

from .evaluate import load_checkpoint, policy_action, atomic_json, _json_value
from .env import CubeReorientEnv


def capture(checkpoint, output, seconds=30., seed_start=80000000):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    model, norm, config, reference, metadata=load_checkpoint(checkpoint)
    fps=20
    frames=int(round(seconds*fps))
    if frames<1:
        raise ValueError('Positive duration required')
    envs=[CubeReorientEnv(reference,config,axis='z') for _ in range(4)]
    lanes=[];events=[];records=[[] for _ in envs];states=[[] for _ in envs]
    for lane,env in enumerate(envs):
        seed=seed_start+lane*1000
        observation,info=env.reset(seed=seed)
        lanes.append(dict(observation=observation,info=info,total=0.,episode_time=0.,
                          episode=0,seed=seed,terminal=False,last_reset=0.))
        events.append(dict(lane=lane,video_seconds=0.,event='reset',seed=seed,episode=0))
    try:
        with (output/'trajectory.jsonl').open('w') as trajectory:
            for frame in range(frames):
                target=(frame+1)/fps
                for lane,(env,state) in enumerate(zip(envs,lanes)):
                    while state['total']<target-1e-9:
                        if state['terminal']:
                            state['episode']+=1
                            state['seed']=seed_start+lane*1000+state['episode']
                            state['observation'],state['info']=env.reset(seed=state['seed'])
                            state.update(episode_time=0.,terminal=False,last_reset=state['total'])
                            events.append(dict(lane=lane,video_seconds=state['total'],event='reset',
                                               seed=state['seed'],episode=state['episode']))
                        action=policy_action(model,norm,state['observation'])
                        obs,reward,terminated,truncated,info=env.step(action)
                        advanced=info['elapsed_seconds']-state['episode_time']
                        if advanced<=0:
                            raise RuntimeError('Physical time did not advance')
                        state.update(observation=obs,info=info,episode_time=info['elapsed_seconds'],
                                     total=state['total']+advanced,terminal=bool(terminated or truncated))
                        trajectory.write(json.dumps(dict(lane=lane,video_seconds=state['total'],
                            episode=state['episode'],seed=state['seed'],reward=reward,
                            info=_json_value(info),qpos=env.data.qpos.tolist(),
                            qvel=env.data.qvel.tolist(),ctrl=env.data.ctrl.tolist()),allow_nan=False)+'\n')
                        if state['terminal']:
                            events.append(dict(lane=lane,video_seconds=state['total'],event='terminal',
                                episode=state['episode'],seed=state['seed'],success=info['success'],
                                failure=info['failure_reason'],max_stable_seconds=info['max_stable_seconds'],
                                signed_progress_degrees=info['signed_progress_degrees']))
                    info=state['info']
                    records[lane].append(dict(video_seconds=frame/fps,simulation_seconds=state['total'],
                        episode=state['episode'],seed=state['seed'],phase=info['phase'],
                        progress=info['signed_progress_degrees'],error=info['angle_error_degrees'],
                        stable=info['stable_seconds'],contact=info['each_finger_contact'].tolist(),
                        force=info['finger_forces'].tolist(),action=info['applied_action'].tolist(),
                        ctrl=env.data.ctrl.tolist(),linear_speed=info['linear_speed'],
                        angular_speed=info['angular_speed'],success=info['success'],
                        failure=info['failure_reason'],capture=info['capture_succeeded'],
                        reset_recent=(frame/fps-state['last_reset']<.5)))
                    states[lane].append(dict(qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),
                                             ctrl=env.data.ctrl.copy()))
                if frame%100==0:
                    print(json.dumps(dict(phase='capturing',frame=frame,frames=frames)),flush=True)
        np.savez_compressed(output/'states.npz',qpos=np.array([[s['qpos'] for s in lane] for lane in states]),
                            qvel=np.array([[s['qvel'] for s in lane] for lane in states]),
                            ctrl=np.array([[s['ctrl'] for s in lane] for lane in states]))
        terminals=[event for event in events if event['event']=='terminal']
        manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),checkpoint=str(Path(checkpoint).resolve()),
            checkpoint_steps=metadata['training_steps'],target_degrees=config.target_degrees,
            required_stable_seconds=config.target_hold_seconds,angle_tolerance_degrees=config.angle_tolerance_degrees,
            hand_tilt_degrees=20.,fps=fps,frames=frames,seconds=frames/fps,
            replay_environments=4,seed_start=seed_start,simulation_seconds=[s['total'] for s in lanes],
            sampling_tolerance_seconds=config.ctrl_dt,events=events,
            completed_attempts=len(terminals),successful_attempts=sum(e['success'] for e in terminals),
            method='Frozen deterministic policy in four independent native MuJoCo environments. Initial acquire holds reference motor targets; rotation/recapture use learned actions. Offline rendering of exact recorded states.',
            note='Checkpoint evaluation replay. These seeds and trajectories are separate from the four live training workers. Resets are separate attempts; a partial final attempt is not a success or failure.')
        atomic_json(output/'replay.json',manifest)
        atomic_json(output/'frames.json',records)
        return manifest,records,states,envs[0]
    except BaseException:
        for env in envs:env.close()
        raise
    finally:
        norm.close()
        for env in envs[1:]:env.close()


def render(manifest,records,states,env,output):
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    def font(size):
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
    title_font=font(23);small_font=font(18)
    renderer=mujoco.Renderer(env.model,height=480,width=640)
    large=mujoco.Renderer(env.model,height=720,width=960)
    env._camera.distance=.36
    outputs=[]
    try:
        for filename,grid in [('four_hands_30s.mp4',True),('single_hand_30s.mp4',False)]:
            path=output/filename
            with imageio.get_writer(path,fps=manifest['fps'],codec='libx264',quality=7,
                    macro_block_size=1,ffmpeg_params=['-threads','1','-movflags','+faststart']) as writer:
                for frame in range(manifest['frames']):
                    canvas=Image.new('RGB',(1280,1184) if grid else (960,816),(15,23,37))
                    draw=ImageDraw.Draw(canvas)
                    draw.text((18,12),f"FROZEN CHECKPOINT REPLAY  |  step {manifest['checkpoint_steps']:,}  |  {frame/20:05.2f}s / {manifest['seconds']:g}s",font=title_font,fill='#E6F0FF')
                    draw.text((18,42),f"Current Z goal: +{manifest['target_degrees']:g}deg  |  Required stable recapture: {manifest['required_stable_seconds']:g}s",font=small_font,fill='#9DB2CC')
                    for lane in range(4) if grid else range(1):
                        s=states[lane][frame];r=records[lane][frame]
                        env.data.qpos[:]=s['qpos'];env.data.qvel[:]=s['qvel'];env.data.ctrl[:]=s['ctrl']
                        mujoco.mj_forward(env.model,env.data)
                        active=renderer if grid else large
                        active.update_scene(env.data,camera=env._camera)
                        rgb=Image.fromarray(active.render())
                        x=(lane%2)*640 if grid else 0
                        y=64+(lane//2)*560 if grid else 96
                        if grid:canvas.paste(rgb,(x,y+80))
                        else:canvas.paste(rgb,(0,96))
                        outcome='SUCCESS' if r['success'] else (r['failure'] or r['phase'].upper())
                        reset=' | RESET' if r['reset_recent'] else ''
                        if grid:
                            draw.text((x+14,y+7),f"HAND {lane+1} | attempt {r['episode']+1} | seed {r['seed']}{reset}",font=small_font,fill='#DBEAFE')
                            draw.text((x+14,y+30),f"{outcome} | turn {r['progress']:+.2f}deg | error {r['error']:.2f}deg",font=small_font,fill='#FBBF24' if not r['success'] else '#34D399')
                            draw.text((x+14,y+53),f"contacts {sum(r['contact'])}/5 | stable {r['stable']:.3f}/{manifest['required_stable_seconds']:g}s",font=small_font,fill='#ADC0D8')
                        else:
                            draw.text((18,68),f"Hand 1 | {outcome}{reset} | turn {r['progress']:+.2f}deg | error {r['error']:.2f}deg | stable {r['stable']:.3f}s",font=small_font,fill='#FBBF24')
                    writer.append_data(np.asarray(canvas))
                    if frame==100:
                        canvas.save(output/('four_hands_preview.png' if grid else 'single_hand_preview.png'))
                    if frame%100==0:
                        print(json.dumps(dict(phase='rendering',video=filename,frame=frame,frames=manifest['frames'])),flush=True)
            outputs.append(str(path))
    finally:
        renderer.close();large.close();env.close()
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seconds',type=float,default=30.)
    parser.add_argument('--seed-start',type=int,default=80000000)
    parser.add_argument('--run',help='Training output directory, to retain status provenance for the HTML builder')
    args=parser.parse_args()
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    if args.run:
        run=Path(args.run).resolve()
        atomic_json(output/'snapshot.json',dict(run_directory=str(run),
            run=json.loads((run/'status.json').read_text()),
            checkpoint=str(Path(args.checkpoint).resolve()),
            checkpoint_metadata=json.loads((Path(args.checkpoint)/'metadata.json').read_text())))
    started=time.monotonic()
    manifest,records,states,env=capture(args.checkpoint,output,args.seconds,args.seed_start)
    paths=render(manifest,records,states,env,output)
    print(json.dumps(dict(outputs=paths,elapsed_seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':main()
