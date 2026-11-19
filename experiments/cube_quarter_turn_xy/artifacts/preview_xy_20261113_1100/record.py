"""Show the exact saved best actions and a labeled extra constant-control test."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from pathlib import Path
import importlib.util
import base64
import json
import time
import numpy as np
from PIL import ImageDraw
from experiments.cube_quarter_turn_v2.env import TeacherEnv
from experiments.cube_quarter_turn_v2.storage import atomic_json,implementation_digest

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
HELPER=ROOT/'experiments/cube_quarter_turn_v2/artifacts/preview_xyz_20261112_1640/record.py'
spec=importlib.util.spec_from_file_location('previous_preview_renderer',HELPER)
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
renderer.OUTPUT=OUT


def capture(axis):
    folder=OUT/axis;meta=json.loads((folder/'candidate.json').read_text())
    with np.load(folder/'candidate.npz',allow_pickle=False) as data:actions=data['actions'].copy()
    env=TeacherEnv(folder/'reference.npz',axis)
    _,info=env.reset(seed=0,options={'reference':True})
    states=[];records=[]
    def save(extension):
        states.append({k:getattr(env.data,k).copy() for k in ('qpos','qvel','ctrl')})
        records.append({**info,'extension':extension,'max_penetration_m':env.contacts['max_penetration']})
    save(False);done=False
    for action in actions:
        _,_,done,_,info=env.step(action);save(False)
        if done:break
    endpoint=info.copy();endpoint_index=len(records)-1
    differences={k:abs(info[k]-meta['final'][k]) for k in
        ('signed_progress_degrees','max_progress_degrees','angle_error_degrees','max_stable_seconds','elapsed_seconds')}
    if max(differences.values())>1e-7:raise RuntimeError('Saved trajectory mismatch')
    for _ in range(60):
        if done:break
        _,_,done,_,info=env.step(actions[-1]);save(True)
    np.savez_compressed(folder/'replay_states.npz',**{k:np.array([s[k] for s in states]) for k in states[0]})
    atomic_json(folder/'trajectory.json',dict(records=records,original_endpoint=endpoint,original_endpoint_index=endpoint_index,
        original_replay_matches_snapshot=True,absolute_differences=differences,final=info,
        extra_test='After the saved sequence, keep the last normalized action unchanged, with original success/failure gates.',
        extension_seconds=info['elapsed_seconds']-endpoint['elapsed_seconds'],source_candidate=meta,
        implementation_sha256=implementation_digest(),dt=.002,
        note='The saved search had no qualified demonstration. An extension passing 2ms alone is NOT the two-timestep acceptance gate.'))
    print(json.dumps(dict(phase='captured',axis=axis,frames=len(records),original_hold=endpoint['max_stable_seconds'],
        final_hold=info['max_stable_seconds'],success_2ms=info['success'])),flush=True)
    return env,states,records,endpoint_index


def annotate(axis,frames,records,endpoint_index):
    for frame,info in zip(frames,records):
        draw=ImageDraw.Draw(frame)
        draw.rectangle((0,0,848,54),fill=(13,20,32))
        draw.text((22,9),f'{axis.upper()} 轴最佳动作 · 含末动作保持检查',font=renderer.FONTS[33],fill=renderer.COLORS[axis])
        draw.rectangle((0,751,1280,800),fill=(13,20,32))
        phase='追加验证：继续保持原末动作' if info['extension'] else '原始保存动作回放'
        draw.text((22,758),f'{phase}  |  2 ms 录像；1 ms 复测未通过  |  尚未进入 PPO',
            font=renderer.FONTS[19],fill='#fbbf24' if info['extension'] else '#b9cbe1')
    draw=ImageDraw.Draw(frames[-1]);draw.rectangle((0,710,1280,750),fill=(13,20,32))
    label='2 ms 单次保持满 1 秒；1 ms 复测失败，未通过完整验收' if records[-1]['success'] else '追加保持后偏离目标，未通过 1 秒验收'
    draw.text((22,714),label,font=renderer.FONTS[25],fill='#fbbf24')
    frames[endpoint_index].save(OUT/f'{axis}_saved_endpoint.png')
    return frames


def webpage(videos):
    cards=[];probes=json.loads((OUT/'holding_probe.json').read_text())
    for axis in 'xy':
        data=json.loads((OUT/axis/'trajectory.json').read_text());original=data['original_endpoint']
        result='延长原末动作 0.36 秒后，2 ms 单次回放保持满 1 秒；1 ms 复测出现穿透。' if axis=='x' else '延长原末动作后最高保持 0.82 秒，随后偏离；1 ms 复测也出现穿透。'
        encoded=base64.b64encode((OUT/videos[axis]['path']).read_bytes()).decode()
        cards.append(f'''<article><h2>{axis.upper()} 轴</h2><p>保存动作末尾：转角 {original['signed_progress_degrees']:.1f}°，姿态误差 {original['angle_error_degrees']:.2f}°，保持 {original['max_stable_seconds']:.2f} 秒。{result}</p>
        <video controls playsinline preload="metadata" src="data:video/mp4;base64,{encoded}"></video>
        <button onclick="downloadVideo(this)">下载 {axis.upper()} 轴视频</button></article>''')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>XY 最佳动作与保持验证 · 11月13日</title><style>*{box-sizing:border-box}body{margin:0;background:#0d1420;color:#e5edf9;font:16px/1.65 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:28px}h1{font-size:28px}p{color:#bacbdd}article{background:#152136;border:1px solid #293d58;border-radius:14px;padding:20px;margin:22px 0}h2{margin:0}video{display:block;width:100%;background:#050b12;border-radius:8px}button{padding:12px 20px;border:0;border-radius:8px;background:#3b82f6;color:white;font-size:16px;margin:8px 12px 8px 0}.notice{border-left:4px solid #fbbf24;padding:10px 16px;background:#202b3d}</style>
    <main><h1>X/Y 最佳动作与末动作保持验证</h1><p>2026年11月13日回放。训练最后可核实记录为北京时间06:52，恢复环境后原进程已不存在。两轴累计约56万条候选尝试，尚未通过完整的双步长成功验收，也未进入行为克隆或PPO。</p>
    <p class="notice">每轴完整回放一次保存的最佳动作，然后额外保持同一末动作。视频明确标出追加验证段。这是本次独立检查，不能把追加保持当作已训练策略。先原速，再慢放同一次尝试；静帧不计入物理保持时间。</p>
    <button onclick="document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play()})">同时从头播放</button>
    <button onclick="document.querySelectorAll('video').forEach(v=>v.pause())">全部暂停</button>'''+''.join(cards)+'''
    <p>2 ms 单次回放通过不等于稳健成功：X/Y 在1 ms复测中均触发穿透失败。画面包含全手和近景、实际仿真时间、转角、完整姿态误差、接触力和连续保持计时。下载此 HTML 后可离线观看。</p></main>
    <script>function downloadVideo(b){const v=b.closest('article').querySelector('video');const a=document.createElement('a');a.href=v.src;a.download=b.closest('article').querySelector('h2').textContent.trim()+'_demo.mp4';a.click()}</script></html>'''
    (OUT/'index.html').write_text(html)


def main():
    started=time.monotonic();videos={}
    for axis in 'xy':
        env,states,records,endpoint=capture(axis)
        frames=renderer.render_frames(axis,env,states,records)
        frames=annotate(axis,frames,records,endpoint)
        videos[axis]=renderer.encode(axis,frames)
    webpage(videos)
    atomic_json(OUT/'recording_report.json',dict(videos=videos,elapsed_seconds=time.monotonic()-started,
        origin='Saved best CEM candidates plus explicitly labeled constant-final-control extension; not PPO policies',
        observations='2ms X extension passes once; 1ms X/Y checks fail penetration; no robust task claim',
        original_training_processes_absent=True,new_training_started=False))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
