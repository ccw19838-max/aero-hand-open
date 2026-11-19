"""Pinned best Y90 policy: one complete native attempt, then labelled slow replay."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,base64,importlib.util,itertools,json,time
import numpy as np
import torch,mujoco,imageio.v2 as imageio
from PIL import ImageDraw
from .env import AdaptiveEnv
from .config import Profile
from .storage import load,atomic_json,digest


def record(goal):
    torch.set_num_threads(1);root=Path(goal).resolve();started=time.monotonic()
    state=json.loads((root/'controller_state.json').read_text())
    checkpoint=Path(state['axes']['y']['best_by_target']['90']['path']).resolve()
    policy,metadata=load(checkpoint);report=metadata['evaluation']
    if not report['complete'] or report['target_degrees']!=90.:raise ValueError('No completed Y90 checkpoint evaluation')
    selected=max(report['records'],key=lambda r:(r['success'],r['final']['max_stable_seconds'],-r['final']['best_angle_error_degrees']))
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output=root/'previews'/f'y_best_{stamp}';output.mkdir(parents=True,exist_ok=False)
    (output/'recording_source.py').write_bytes(Path(__file__).read_bytes())
    env=AdaptiveEnv(checkpoint/'reference.npz','y',physical_dt=selected['dt'],target_degrees=90.,profile=Profile(**metadata['profile']))
    states=[];trace=[]
    try:
        obs,info=env.reset(seed=selected['seed'])
        def save():
            states.append({k:getattr(env.data,k).copy() for k in ('qpos','qvel','ctrl')});trace.append(info.copy())
        save()
        for _ in range(250):
            obs,_,done,_,info=env.step(policy.predict(obs,deterministic=True)[0]);save()
            if done:break
        differences={k:abs(info[k]-selected['final'][k]) for k in
            ('elapsed_seconds','signed_progress_degrees','angle_error_degrees','best_angle_error_degrees','max_stable_seconds')}
        if max(differences.values())>1e-7 or info['success']!=selected['success'] or info['failure_reason']!=selected['final']['failure_reason']:
            raise RuntimeError('Saved/native policy replay mismatch: '+str(differences))
        terminal_support=[];force=np.zeros(6);cube=env.ids['cube_geom']
        for i,c in enumerate(env.data.contact):
            a,b=int(c.geom1),int(c.geom2);other=b if a==cube else a if b==cube else None
            if other is not None and env.ids['geom_forbidden'][other]:
                mujoco.mj_contactForce(env.model,env.data,i,force)
                if force[0]>=env.config.contact_force:terminal_support.append(dict(geom=env.model.geom(other).name,normal_force_N=float(force[0])))
        np.savez_compressed(output/'states.npz',**{k:np.asarray([s[k] for s in states]) for k in states[0]})
        atomic_json(output/'trajectory.json',dict(checkpoint=str(checkpoint),seed=selected['seed'],dt=selected['dt'],
            exact_saved_validation_replay=True,differences=differences,terminal_support=terminal_support,trace=trace))
        print(json.dumps(dict(phase='captured',output=str(output),simulated_seconds=info['elapsed_seconds'],
            angle=info['signed_progress_degrees'],best_error=info['best_angle_error_degrees'],terminal_support=terminal_support)),flush=True)
        helper=Path(__file__).resolve().parents[2]/'cube_quarter_turn_v2/artifacts/preview_xyz_20261112_1640/record.py'
        # Resolve from experiments/, avoiding a copy of any frozen task code.
        helper=Path(__file__).resolve().parent.parent/'cube_quarter_turn_v2/artifacts/preview_xyz_20261112_1640/record.py'
        spec=importlib.util.spec_from_file_location('y_best_native_renderer',helper)
        renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
        env.model.vis.quality.shadowsize=256  # display only, after all physics is captured
        frames=renderer.render_frames('y',env,states,trace)
    finally:env.close()
    for frame,item in zip(frames,trace):
        draw=ImageDraw.Draw(frame)
        draw.rectangle((0,0,848,54),fill=(13,20,32))
        draw.text((22,9),'Y轴 +90°目标｜最佳检查点回放',font=renderer.FONTS[33],fill=renderer.COLORS['y'])
        draw.rectangle((0,672,1280,712),fill=(13,20,32))
        for finger,(name,f) in enumerate(zip(('拇指','食指','中指','无名指','小指'),item['finger_forces'])):
            lower=.2 if finger==0 else .1;upper=20. if finger==0 else 12.
            color='#fbbf24' if f>upper else '#4ade80' if f>=lower else '#94a3b8'
            x=22+finger*246;draw.ellipse((x,687,x+12,699),fill=color)
            draw.text((x+20,675),f'{name} {f:.2f}N',font=renderer.FONTS[22],fill=color)
        draw.rectangle((0,751,1280,800),fill=(13,20,32))
        draw.text((22,758),f'本段 {metadata["training_steps"]:,}步｜seed={selected["seed"]}｜{selected["dt"]*1000:g}ms｜最小姿态误差 {info["best_angle_error_degrees"]:.2f}°｜未通过五指抓稳验收',font=renderer.FONTS[19],fill='#b9cbe1')
    segments=[('初始状态 · 静帧',[frames[0]]*10,0.),('原速 1×',frames[1:],1.),
        ('失败终点 · 静帧',[frames[-1]]*16,0.),('同一动作重新慢放',[frames[0]]*10,0.),
        ('同一动作 · 0.25×',list(itertools.chain.from_iterable([f]*4 for f in frames[1:])),.25),
        ('终点静帧 · 非保持',[frames[-1]]*20,0.)]
    video=output/'y_best_policy.mp4';count=0;timeline=[]
    with imageio.get_writer(video,fps=20,codec='libx264',quality=7,macro_block_size=1,
            ffmpeg_params=['-threads','1','-movflags','+faststart']) as writer:
        for label,part,speed in segments:
            timeline.append(dict(label=label,start_seconds=count/20,speed=speed))
            for frame in part:writer.append_data(renderer.tagged(frame,label));count+=1
    frames[max(1,len(frames)*3//4)].save(output/'poster.png');frames[-1].save(output/'final.png')
    result=dict(created_at=datetime.now(timezone.utc).isoformat(),video=video.name,seconds=count/20,fps=20,frames=count,
        width=1280,height=800,sha256=digest(video),checkpoint=str(checkpoint),training_steps=metadata['training_steps'],
        selection='Best protected Y90 checkpoint by aggregate evaluation; its example with the smallest best full-pose error',
        seed=selected['seed'],physical_dt=selected['dt'],final=info,terminal_support=terminal_support,
        saved_validation_reproduced=True,segments=timeline,
        note='One complete failed native neural-policy attempt. Slow replay repeats the same attempt; static frames do not advance physics or hold time. No training was restarted.',
        elapsed_wall_seconds=time.monotonic()-started)
    atomic_json(output/'recording_report.json',result)
    encoded=base64.b64encode(video.read_bytes()).decode()
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Y轴最佳检查点回放</title>
<style>*{box-sizing:border-box}body{margin:0;background:#0d1420;color:#e5edf9;font:16px/1.7 system-ui,sans-serif}main{max-width:1360px;margin:auto;padding:22px}p{color:#b9cbe1}video{display:block;width:100%;background:#050b12}button{padding:10px 18px;margin:8px 8px 8px 0;border:0;border-radius:8px;background:#2563eb;color:white;font-size:16px}</style>
<main><h1>Y轴：最佳检查点的姿态接近样例</h1><p>先原速，再以0.25倍速重播同一次动作。完整保留失败终点；静帧及慢放不增加物理保持时间。</p>
<p>本次转到约90°，最小完整姿态误差约3.33°。食指、中指未重新受力，拇指力超出设定上限，并碰到支架，未完成五指抓稳1秒。绿色表示达到受力门槛，橙色表示超过上限。</p>
<button onclick="const v=document.querySelector('video');v.currentTime=0;v.play()">从头播放</button><button onclick="document.querySelector('video').pause()">暂停</button>
<video controls playsinline preload="metadata" src="data:video/mp4;base64,'''+encoded+'''"></video>
<p>本段589,824步检查点；seed1000007；1ms物理步长。选择标准是完整目标姿态的接近程度，不能视为90°抓稳成功。</p></main></html>'''
    (output/'index.html').write_text(html)
    print(json.dumps(dict(completed=True,output=str(output),video=str(video),seconds=count/20)),flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--goal',required=True);args=parser.parse_args();record(args.goal)
