"""Pinned planner trajectories, exact native replay, offline two-camera recording.

Run from the repository root with its Python on PYTHONPATH. This file is outside
the frozen teacher implementation and never modifies live workers/checkpoints.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from pathlib import Path
import base64
import hashlib
import itertools
import json
import math
import time
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from experiments.cube_quarter_turn_v2.env import TeacherEnv
from experiments.cube_quarter_turn_v2.storage import atomic_json,implementation_digest

OUTPUT=Path(__file__).resolve().parent
FPS=20
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONTS={n:ImageFont.truetype(FONT,n,index=2) for n in (19,22,25,30,33)}
COLORS={'x':(248,113,113),'y':(74,222,128),'z':(96,165,250)}
FAILURES={'forbidden_support':'碰到安装支架或地面','penetration':'持续穿透超限',
    'workspace':'离开操作区域','lost_hand_contact':'脱离手部接触','stagnation':'转动进展停滞',
    'timeout':'回合超时','numerical':'数值异常'}
PHASES={'setup':'初始保持','drive':'转动','capture':'重抓','hold':'目标姿态保持'}


def capture(axis):
    directory=OUTPUT/axis;meta=json.loads((directory/'candidate.json').read_text())
    with np.load(directory/'candidate.npz',allow_pickle=False) as archive:actions=archive['actions'].copy()
    env=TeacherEnv(directory/'reference.npz',axis)
    _,info=env.reset(seed=0,options={'reference':True})
    states=[];records=[]
    def save(info):
        states.append({k:getattr(env.data,k).copy() for k in ('qpos','qvel','ctrl')})
        records.append({**info,'max_penetration_m':env.contacts['max_penetration']})
    save(info)
    for action in actions:
        _,_,done,_,info=env.step(action);save(info)
        if done:break
    differences={key:abs(info[key]-meta['final'][key]) for key in
        ('signed_progress_degrees','max_progress_degrees','angle_error_degrees','max_stable_seconds','elapsed_seconds')}
    if max(differences.values())>1e-7 or info['failure_reason']!=meta['final']['failure_reason']:
        raise RuntimeError(f'{axis}: pinned JSON/NPZ replay mismatch {differences}')
    np.savez_compressed(directory/'replay_states.npz',**{k:np.array([s[k] for s in states]) for k in states[0]})
    atomic_json(directory/'trajectory.json',dict(records=records,replay_matches_snapshot=True,
        absolute_differences=differences,end_reason=info['failure_reason'] or 'candidate_sequence_completed',
        simulated_seconds=info['elapsed_seconds'],source_candidate=meta,
        implementation_sha256=implementation_digest()))
    print(json.dumps(dict(phase='captured',axis=axis,frames=len(states),final=info)),flush=True)
    return env,states,records


def render_frames(axis,env,states,records):
    renderer=mujoco.Renderer(env.model,height=480,width=640)
    main=mujoco.MjvCamera();main.lookat[:]=[.105,0.,.195];main.distance=.36;main.azimuth=135.;main.elevation=-35.
    close=mujoco.MjvCamera();close.lookat[:]=states[0]['qpos'][16:19];close.distance=.235;close.azimuth=100.;close.elevation=-75.
    cameras=(main,close);frames=[]
    direction=np.array(records[-1]['axis_world_vector'])
    arrow_start=states[0]['qpos'][16:19]+np.array([-.035,-.04,.035]);arrow_end=arrow_start+.055*direction
    try:
        for index,(state,info) in enumerate(zip(states,records)):
            for key,value in state.items():getattr(env.data,key)[:]=value
            mujoco.mj_forward(env.model,env.data)
            views=[]
            for camera in cameras:
                renderer.update_scene(env.data,camera=camera)
                geom=renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.ones(3)*.001,
                    np.zeros(3),np.eye(3).ravel(),np.r_[np.array(COLORS[axis])/255.,1.])
                mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.002,arrow_start,arrow_end)
                renderer.scene.ngeom+=1
                views.append(Image.fromarray(renderer.render()))
            base=Image.new('RGB',(1280,800),(13,20,32))
            base.paste(views[0],(0,150));base.paste(views[1],(640,150))
            draw=ImageDraw.Draw(base)
            draw.text((22,9),f'{axis.upper()} 轴整体转动 +90°  |  当前最佳候选轨迹',font=FONTS[33],fill=COLORS[axis])
            draw.text((22,55),f'仿真 t={info["elapsed_seconds"]:.3f}s   转角 {info["signed_progress_degrees"]:+.1f}°   三维姿态误差 {info["angle_error_degrees"]:.1f}°',font=FONTS[25],fill='white')
            draw.text((22,109),'全手视角',font=FONTS[22],fill='#b9cbe1')
            draw.text((665,109),'近景 / 俯视  ·  彩色箭头表示目标正轴',font=FONTS[22],fill='#b9cbe1')
            draw.text((22,638),f'目标处稳定保持 {info["stable_seconds"]:.2f} / 1.00 s   |   线速度 {info["linear_speed"]:.3f} m/s   |   角速度 {info["angular_speed"]:.2f} rad/s',font=FONTS[22],fill='#dbe7f6')
            for finger,(name,force) in enumerate(zip(('拇指','食指','中指','无名指','小指'),info['finger_forces'])):
                x=22+finger*246;color='#4ade80' if force>=.02 else '#94a3b8'
                draw.ellipse((x,687,x+12,699),fill=color)
                draw.text((x+20,675),f'{name} {force:.2f}N',font=FONTS[22],fill=color)
            reason=FAILURES.get(info['failure_reason'],info['failure_reason'])
            final=index==len(states)-1
            status=('成功' if info['success'] else f'结束：{reason}' if reason else '候选动作执行完毕，未通过验收' if final else PHASES.get(info['phase'],info['phase']))
            draw.text((22,714),f'{status}   |   本次最高转角 {info["max_progress_degrees"]:.1f}°',font=FONTS[25],fill='#fbbf24')
            draw.text((22,758),'57 mm 魔方 / 掌面保留 20°倾角 / 无辅助外力 / 原始失败终点完整保留',font=FONTS[19],fill='#94a3b8')
            frames.append(base)
            if index%30==0:print(json.dumps(dict(phase='render',axis=axis,frame=index,frames=len(states))),flush=True)
    finally:renderer.close();env.close()
    return frames


def tagged(frame,label):
    frame=frame.copy();draw=ImageDraw.Draw(frame)
    draw.rounded_rectangle((850,5,1262,54),radius=9,fill=(26,39,59))
    draw.text((866,9),label,font=FONTS[25],fill='white')
    return np.asarray(frame)


def encode(axis,frames):
    path=OUTPUT/f'{axis}_axis_demo.mp4'
    segments=[dict(label='初始状态 · 静帧',indices=[0]*14,speed=0.),
        dict(label='原速回放 1×',indices=list(range(1,len(frames))),speed=1.),
        dict(label='结束状态 · 静帧',indices=[len(frames)-1]*36,speed=0.),
        dict(label='同一轨迹重新慢放',indices=[0]*14,speed=0.),
        dict(label='慢动作 0.25×',indices=list(itertools.chain.from_iterable([i]*4 for i in range(1,len(frames)))) ,speed=.25),
        dict(label='结束状态 · 静帧',indices=[len(frames)-1]*40,speed=0.)]
    count=0;timeline=[]
    with imageio.get_writer(path,fps=FPS,codec='libx264',quality=7,macro_block_size=1,
            ffmpeg_params=['-threads','1','-movflags','+faststart','-pix_fmt','yuv420p']) as writer:
        for segment in segments:
            timeline.append(dict(label=segment['label'],start_seconds=count/FPS,speed=segment['speed']))
            for i in segment['indices']:writer.append_data(tagged(frames[i],segment['label']));count+=1
    frames[0].save(OUTPUT/f'{axis}_initial.png')
    frames[-1].save(OUTPUT/f'{axis}_final.png')
    frames[max(1,int(.6*(len(frames)-1)))].save(OUTPUT/f'{axis}_poster.png')
    metadata=dict(path=path.name,width=1280,height=800,frames=count,fps=FPS,seconds=count/FPS,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),segments=timeline,
        note='One pinned candidate only. Slow motion repeats the SAME attempt; static pauses are labeled. Terminal contact checks stop physics.')
    atomic_json(OUTPUT/axis/'video.json',metadata)
    print(json.dumps(dict(phase='encoded',axis=axis,**metadata)),flush=True)
    return metadata


def webpage(videos):
    cards=[]
    for axis in 'xyz':
        meta=json.loads((OUTPUT/axis/'trajectory.json').read_text());final=meta['records'][-1]
        reason=FAILURES.get(final['failure_reason'],'候选序列执行完毕，未通过目标验收')
        encoded=base64.b64encode((OUTPUT/videos[axis]['path']).read_bytes()).decode()
        cards.append(f'''<article><h2>{axis.upper()} 轴</h2><p>最高转角 {final['max_progress_degrees']:.1f}° · 最小姿态误差 {final['best_angle_error_degrees']:.1f}° · {reason}</p>
        <video controls playsinline preload="metadata" src="data:video/mp4;base64,{encoded}"></video>
        <a download="{axis}_axis_demo.mp4" href="data:video/mp4;base64,{encoded}">下载 {axis.upper()} 轴视频</a></article>''')
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>XYZ 魔方转动 · 当前最佳候选回放</title><style>*{box-sizing:border-box}body{margin:0;background:#0d1420;color:#e5edf9;font:16px/1.65 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:28px}h1{font-size:28px}p{color:#bacbdd}article{background:#152136;border:1px solid #293d58;border-radius:14px;padding:20px;margin:22px 0}h2{margin:0}video{display:block;width:100%;background:#050b12;border-radius:8px}a{color:#93c5fd;display:inline-block;padding:12px 0}button{padding:12px 20px;border:0;border-radius:8px;background:#3b82f6;color:white;font-size:16px;margin:0 12px 8px 0}</style>
    <main><h1>XYZ 整体转动 · 各轴一个候选演示</h1><p>从当前搜索结果固定每轴最高评分轨迹，真实物理回放。尚无 90°后稳定保持 1 秒的成功结果。每段先原速、再重播同一轨迹的 0.25× 慢动作；静帧、失败原因和仿真时间均已标出。</p>
    <button onclick="document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play()})">同时从头播放</button>
    <button onclick="document.querySelectorAll('video').forEach(v=>v.pause())">全部暂停</button>'''+''.join(cards)+'''<p>视频已嵌入，下载此 HTML 后可离线观看。各轴只展示同一次尝试；当前手模型为 7 电机、魔方为刚体，尚未拧动魔方层。</p></main></html>'''
    (OUTPUT/'index.html').write_text(document)


def main():
    started=time.monotonic();videos={};all_frames={}
    for axis in 'xyz':
        env,states,records=capture(axis)
        frames=render_frames(axis,env,states,records)
        videos[axis]=encode(axis,frames)
    webpage(videos)
    atomic_json(OUTPUT/'recording_report.json',dict(videos=videos,elapsed_seconds=time.monotonic()-started,
        origin='Pinned CEM candidate trajectories, not trained policy rollouts',training_workers_unmodified=True))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
