"""Independent strict rollouts; optional real-time video with no simulated resets hidden."""
from collections import Counter
from pathlib import Path
import argparse
import json
import time
import numpy as np
from .env import TeacherEnv
from .storage import atomic_json,load


def evaluate(model,reference,axis,episodes=20,seed=1000000,deadline=float('inf'),should_stop=lambda:False):
    env=TeacherEnv(reference,axis);records=[]
    try:
        for index in range(episodes):
            if time.time()>=deadline or should_stop():break
            obs,info=env.reset(seed=seed+index);trace=[];interrupted=False
            for _ in range(250):
                if time.time()>=deadline or should_stop():interrupted=True;break
                action=model.predict(obs,deterministic=True)[0]
                obs,_,done,_,info=env.step(action)
                trace.append(info)
                if done:break
            records.append(dict(seed=seed+index,success=bool(info['success'] and not interrupted),
                interrupted=interrupted,final=info,trace=trace))
            if interrupted:break
    finally:env.close()
    return dict(axis=axis,episodes=len(records),requested_episodes=episodes,complete=len(records)==episodes
        and not any(r['interrupted'] for r in records),
        success_rate=sum(r['success'] for r in records)/max(1,len(records)),
        mean_best_angle_error_degrees=float(np.mean([r['final']['best_angle_error_degrees'] for r in records])) if records else 90.,
        mean_max_progress_degrees=float(np.mean([r['final']['max_progress_degrees'] for r in records])) if records else 0.,
        failures=dict(Counter(r['final']['failure_reason'] for r in records if not r['success'])),
        records=records,final_task_verified=False)


def compact(report):return {k:v for k,v in report.items() if k!='records'}


def chain(checkpoints,episodes=20,seed=70000000,deadline=float('inf'),should_stop=lambda:False):
    models={a:load(checkpoints[a])[0] for a in 'xyz'}
    refs={a:Path(checkpoints[a]).resolve()/'reference.npz' for a in 'xyz'}
    env=TeacherEnv(refs['x'],'x');records=[]
    try:
        for index in range(episodes):
            if time.time()>=deadline or should_stop():break
            env.default=np.load(refs['x'],allow_pickle=False)['ctrl'].copy()
            obs,_=env.reset(seed=seed+index);completed=[];interrupted=False
            for axis in 'xyz':
                if completed:
                    # Change the command map and goal, never the physical state.
                    with np.load(refs[axis],allow_pickle=False) as state:env.default=state['ctrl'].copy()
                    obs,_=env.start_goal(axis)
                for _ in range(250):
                    if time.time()>=deadline or should_stop():interrupted=True;break
                    obs,_,done,_,info=env.step(models[axis].predict(obs,deterministic=True)[0])
                    if done:break
                if interrupted or not info['success']:break
                completed.append(axis)
            records.append(dict(seed=seed+index,completed=completed,success=len(completed)==3,
                interrupted=interrupted,final=info))
            if interrupted:break
    finally:env.close()
    return dict(episodes=len(records),complete=len(records)==episodes and not any(r['interrupted'] for r in records),
        success_rate=sum(r['success'] for r in records)/max(1,len(records)),records=records,
        physical_reset_between_axes=False)


def video(model,reference,axis,path,seconds=30,seed=90000000):
    import imageio.v2 as imageio
    env=TeacherEnv(reference,axis);obs,_=env.reset(seed=seed);records=[];trial=0
    try:
        with imageio.get_writer(path,fps=20,codec='libx264',quality=7) as writer:
            for frame in range(round(seconds*20)):
                action=model.predict(obs,deterministic=True)[0]
                obs,_,done,_,info=env.step(action)
                from PIL import Image,ImageDraw
                image=Image.fromarray(env.render());draw=ImageDraw.Draw(image)
                draw.rectangle((0,0,960,80),fill=(15,15,15))
                draw.text((14,10),f'{axis.upper()} +90 | trial {trial} | 20fps, resets marked | video t={frame/20:.2f}s',fill='white')
                draw.text((14,35),f'angle={info["signed_progress_degrees"]:.1f} error={info["angle_error_degrees"]:.1f} hold={info["stable_seconds"]:.2f}s',fill='white')
                draw.text((14,56),f'{info["phase"]} | {info["failure_reason"]} | success={info["success"]}',fill='white')
                writer.append_data(np.asarray(image));records.append(info)
                if done:trial+=1;obs,_=env.reset(seed=seed+trial)
        atomic_json(Path(path).with_suffix('.json'),dict(seconds=seconds,fps=20,axis=axis,
            resets_visible_in_trial_counter=True,records=records))
    finally:env.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--episodes',type=int,default=100)
    parser.add_argument('--video',action='store_true');args=parser.parse_args()
    model,meta=load(args.checkpoint);path=Path(args.checkpoint).resolve();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    report=evaluate(model,path/'reference.npz',meta['axis'],args.episodes,seed=50000000)
    atomic_json(output/'evaluation.json',report)
    if args.video:video(model,path/'reference.npz',meta['axis'],output/'replay_30s.mp4')
    print(json.dumps(compact(report)))


if __name__=='__main__':main()
