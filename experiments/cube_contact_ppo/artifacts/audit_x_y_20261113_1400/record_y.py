"""Frozen Y PPO replay: a success and a failure, clearly marked as +30deg."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from pathlib import Path
import importlib.util
import json
import base64
import hashlib
import itertools
import time
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import ImageDraw
from experiments.cube_contact_ppo.env import ContactEnv
from experiments.cube_contact_ppo.storage import load,atomic_json

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
helper=ROOT/'experiments/cube_quarter_turn_v2/artifacts/preview_xyz_20261112_1640/record.py'
spec=importlib.util.spec_from_file_location('native_preview_renderer',helper)
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
renderer.FAILURES['insufficient_rotation_contacts']='换指后未恢复最低支撑'


def capture(model,source_record,checkpoint):
    env=ContactEnv(checkpoint/'reference.npz','y',physical_dt=source_record['dt'],target_degrees=30.)
    obs,info=env.reset(seed=source_record['seed']);states=[];records=[]
    def save():
        states.append({k:getattr(env.data,k).copy() for k in ('qpos','qvel','ctrl')})
        records.append(info.copy())
    save()
    for _ in range(250):
        obs,_,done,_,info=env.step(model.predict(obs,deterministic=True)[0]);save()
        if done:break
    expected=source_record['final']
    difference=max(abs(info[k]-expected[k]) for k in
        ('elapsed_seconds','signed_progress_degrees','angle_error_degrees','max_stable_seconds'))
    if difference>1e-7 or info['success']!=source_record['success'] or info['failure_reason']!=expected['failure_reason']:
        raise RuntimeError('Pinned policy replay disagrees with its saved validation record')
    return env,states,records


def annotate(frames,records,seed,outcome,steps):
    for frame,info in zip(frames,records):
        draw=ImageDraw.Draw(frame)
        draw.rectangle((0,0,848,54),fill=(13,20,32))
        draw.text((22,9),'Y 轴 PPO：+30°后五指抓稳',font=renderer.FONTS[33],fill=renderer.COLORS['y'])
        draw.rectangle((0,751,1280,800),fill=(13,20,32))
        text=f'权重 {steps} 步 | 30°验证65% | {outcome}，seed={seed} | 非90°演示'
        draw.text((22,758),text,font=renderer.FONTS[19],fill='#b9cbe1')
    return frames


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    started=time.monotonic();checkpoint=OUT/'pinned_y_best_30';model,meta=load(checkpoint)
    report=meta['evaluation'];source=report['records']
    chosen={'success':next(r for r in source if r['success'] and r['dt']==.001),
            'failure':next(r for r in source if not r['success'] and r['dt']==.001)}
    rendered={};episode_reports={}
    for name,record in chosen.items():
        env,states,records=capture(model,record,checkpoint)
        np.savez_compressed(OUT/f'y_{name}_states.npz',**{k:np.array([s[k] for s in states]) for k in states[0]})
        episode_reports[name]=dict(seed=record['seed'],dt=record['dt'],final=records[-1],trace=records,
            exact_saved_validation_replay=True,selected_outcome=name)
        atomic_json(OUT/f'y_{name}_trajectory.json',episode_reports[name])
        frames=renderer.render_frames('y',env,states,records)
        frames=annotate(frames,records,record['seed'],'成功样例' if name=='success' else '失败样例',meta['training_steps'])
        frames[-1].save(OUT/f'y_{name}_final.png');frames[len(frames)//2].save(OUT/f'y_{name}_poster.png')
        rendered[name]=frames
    a=rendered['success'];b=rendered['failure']
    segments=[('初始状态 · 静帧',[a[0]]*14,0.),('成功样例 · 原速 1×',a[1:],1.),
        ('成功终点 · 静帧',[a[-1]]*24,0.),('同一成功样例重新慢放',[a[0]]*14,0.),
        ('同一动作 · 慢放0.25×',list(itertools.chain.from_iterable([f]*4 for f in a[1:])),.25),
        ('成功终点 · 静帧',[a[-1]]*20,0.),('以下为失败样例 · 重置',[b[0]]*20,0.),
        ('失败样例 · 原速 1×',b[1:],1.),('失败终点 · 静帧',[b[-1]]*30,0.)]
    video=OUT/'y_30deg_best.mp4';count=0;timeline=[]
    with imageio.get_writer(video,fps=20,codec='libx264',quality=7,macro_block_size=1,
            ffmpeg_params=['-threads','1','-movflags','+faststart']) as writer:
        for label,frames,speed in segments:
            timeline.append(dict(label=label,start_seconds=count/20,speed=speed))
            for frame in frames:writer.append_data(renderer.tagged(frame,label));count+=1
    videos={'y':dict(path=video.name,frames=count,fps=20,seconds=count/20,width=1280,height=800,
        sha256=hashlib.sha256(video.read_bytes()).hexdigest(),segments=timeline)}
    atomic_json(OUT/'recording_report.json',dict(videos=videos,checkpoint=str(checkpoint),training_steps=meta['training_steps'],
        target_degrees=30.,validated_success_rate=report['min_integrator_success_rate'],
        selected_success_seed=chosen['success']['seed'],selected_failure_seed=chosen['failure']['seed'],
        note='Two independent attempts of the same saved best PPO. The slow-motion section repeats the same success, not a second trial. No extra control extension.',
        elapsed_seconds=time.monotonic()-started,live_training_unchanged=True))
    webpage(video,meta,episode_reports)
    print(json.dumps(dict(completed=True,seconds=count/20,video=str(video))),flush=True)


def webpage(video,metadata,episodes):
    encoded=base64.b64encode(video.read_bytes()).decode()
    chart=OUT/'x_diagnosis.png'
    chart_html='<h2>X轴诊断对照</h2><img style="width:100%" src="data:image/png;base64,'+base64.b64encode(chart.read_bytes()).decode()+'">' if chart.exists() else ''
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Y轴30°五指重抓录像与X轴诊断</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#0d1420;color:#e5edf9;font:16px/1.7 system-ui,sans-serif}}main{{max-width:1400px;margin:auto;padding:26px}}p{{color:#b9cbe1}}video{{display:block;width:100%;background:#050b12}}button{{padding:12px 20px;margin:10px 12px 10px 0;border:0;border-radius:8px;background:#2563eb;color:white;font-size:16px}}.notice{{padding:12px;border-left:4px solid #fbbf24;background:#202b3d}}</style>
    <main><h1>Y轴：30°转动后五指抓稳</h1><p>固定已保存最佳 PPO 权重（{metadata['training_steps']:,}步）。该版本在1 ms和2 ms两组验证中各成功13/20，总计26/40。后续训练曾退化，本片不代表最新权重，也不是90°完成演示。</p>
    <p class="notice">先展示一次成功：原速，再慢放同一次动作。之后明确重置，展示同一权重的一次失败。静帧不计入物理保持时间。没有追加控制、改变力阈值或移动物体。</p>
    <button onclick="const v=document.querySelector('video');v.currentTime=0;v.play()">从头播放</button><button onclick="document.querySelector('video').pause()">暂停</button>
    <video controls playsinline preload="metadata" src="data:video/mp4;base64,{encoded}"></video>
    <p>成功样例seed={episodes['success']['seed']}；失败样例seed={episodes['failure']['seed']}。视频含全手与近景、每指真实接触力、姿态误差和一秒连续保持计时。</p>{chart_html}
    <p>页面中的视频和图已嵌入，下载后可离线观看。诊断对照为独立评估，正在运行的训练未改动。</p></main></html>'''
    (OUT/'index.html').write_text(text)


if __name__=='__main__':main()
