"""Native X policy comparison at an identical initial state, with failure shown."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from pathlib import Path
from datetime import datetime,timezone
import argparse,base64,json,time
import numpy as np
import torch,imageio.v2 as imageio
from PIL import Image,ImageDraw
from .env import AdaptiveEnv
from .config import Profile
from .storage import load,atomic_json,digest
from .record_preview import Panel,FONTS


def capture(output,key,pin):
    checkpoint=Path(pin['checkpoint']);policy,metadata=load(checkpoint)
    selected=pin['selected'];frames=[]
    env=AdaptiveEnv(checkpoint/'reference.npz','x',physical_dt=selected['dt'],target_degrees=90.,profile=Profile(**metadata['profile']))
    try:
        obs,info=env.reset(seed=selected['seed'])
        def save():
            frames.append(dict(qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),ctrl=env.data.ctrl.copy(),
                time=float(env.data.time),info=info.copy(),seed=selected['seed'],episode=0,kind='原速 1×'))
        save()
        for _ in range(250):
            obs,_,done,_,info=env.step(policy.predict(obs,deterministic=True)[0]);save()
            if done:break
        expected=selected['final']
        differences={k:abs(info[k]-expected[k]) for k in ('elapsed_seconds','signed_progress_degrees',
            'max_progress_degrees','angle_error_degrees','max_stable_seconds')}
        assert max(differences.values())<1e-7 and info['failure_reason']==expected['failure_reason']
        assert info['success']==selected['success']
        np.savez_compressed(output/f'{key}_states.npz',**{k:np.asarray([f[k] for f in frames]) for k in ('qpos','qvel','ctrl','time')})
        atomic_json(output/f'{key}_trajectory.json',dict(checkpoint=str(checkpoint),seed=selected['seed'],dt=selected['dt'],
            exact_validation_reproduced=True,differences=differences,trace=[f['info'] for f in frames]))
        panel=Panel('x',env.model,frames,metadata)
        print(json.dumps(dict(phase='captured',policy=key,angle=info['max_progress_degrees'],failure=info['failure_reason'])),flush=True)
        return panel
    finally:env.close()


def main(output):
    torch.set_num_threads(1);output=Path(output);manifest=json.loads((output/'manifest.json').read_text())
    panels={};started=time.monotonic()
    try:
        for key in ('new','old'):panels[key]=capture(output,key,manifest['pins'][key])
        for name in ('qpos','qvel','ctrl'):
            np.testing.assert_array_equal(panels['new'].frames[0][name],panels['old'].frames[0][name])
        # Render physical states once. Repeated video frames are labelled pauses
        # or slow motion and never contribute physical hold time.
        rendered={}
        for key,panel in panels.items():
            rendered[key]=[]
            for i in range(len(panel.frames)):
                frame=panel.draw(i);draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,1280,59),fill=(13,20,32))
                draw.text((18,1),manifest['pins'][key]['label']+' · X轴目标 +90°',font=FONTS[25],fill='#f87171' if key=='new' else '#7dd3fc')
                draw.text((18,34),f'本段 {panel.metadata["training_steps"]:,} 步 | seed={manifest["seed"]} | {manifest["dt"]*1000:g}ms | 同一初态',font=FONTS[18],fill='#b9cbe1')
                rendered[key].append(frame)
            print(json.dumps(dict(phase='rendered',policy=key,frames=len(rendered[key]))),flush=True)
        length=max(len(v) for v in rendered.values());timeline=[];count=0
        sequences=[('初态静帧',[(0,0)]*10),('原速 1×',[(i,1) for i in range(1,length)]),
                   ('失败终点静帧',[(length-1,0)]*20),('同一次动作 · 0.25×慢放',[(i,.25) for i in range(1,length) for _ in range(4)]),
                   ('失败终点静帧',[(length-1,0)]*30)]
        video=output/'x_thumb_comparison.mp4'
        with imageio.get_writer(video,fps=20,codec='libx264',quality=7,macro_block_size=1,
                ffmpeg_params=['-threads','1','-movflags','+faststart','-pix_fmt','yuv420p']) as writer:
            for label,sequence in sequences:
                timeline.append(dict(label=label,start_seconds=count/20))
                for index,speed in sequence:
                    image=Image.new('RGB',(1280,1040),(8,13,23));draw=ImageDraw.Draw(image)
                    draw.text((18,5),'X轴新旧策略对照｜'+label,font=FONTS[28],fill='white')
                    for key,y in [('new',50),('old',540)]:
                        image.paste(rendered[key][min(index,len(rendered[key])-1)],(0,y))
                        if index>=len(rendered[key])-1 or speed==0:
                            draw.rectangle((950,y+65,1270,y+98),fill=(13,20,32))
                            draw.text((958,y+66),'静帧 · 不计入保持',font=FONTS[20],fill='#fbbf24')
                    draw.text((18,1018),'上：新方案最大转角样例；下：旧策略同初态。两者均未完成90°。绿色达门槛，橙色超力。',font=FONTS[18],fill='#b9cbe1')
                    writer.append_data(np.asarray(image));count+=1
                    if label=='原速 1×' and index==length-1:image.save(output/'final.png')
                    if label=='原速 1×' and index==length//2:image.save(output/'poster.png')
        result=dict(created_at=datetime.now(timezone.utc).isoformat(),video=video.name,sha256=digest(video),
            fps=20,frames=count,seconds=count/20,width=1280,height=1040,identical_initial_state=True,
            replay_matches_saved_validation=True,selection=manifest['selection'],
            finals={k:p.frames[-1]['info'] for k,p in panels.items()},segments=timeline,
            wall_seconds=time.monotonic()-started,training_untouched=True)
        atomic_json(output/'recording_report.json',result)
        new=result['finals']['new']['max_progress_degrees'];old=result['finals']['old']['max_progress_degrees']
        encoded=base64.b64encode(video.read_bytes()).decode()
        html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>X轴拇指推动方案对照</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#0d1420;color:#e5edf9;font:16px/1.7 system-ui,sans-serif}}main{{max-width:1360px;margin:auto;padding:20px}}p{{color:#b9cbe1}}video{{display:block;width:100%;background:#050b12}}button{{padding:10px 18px;margin:8px 8px 8px 0;border:0;border-radius:8px;background:#2563eb;color:white;font-size:16px}}</style>
<main><h1>X轴：拇指推动方案的实际效果</h1><p>上方是新方案训练后最大转角的验证样例，下方是保留的旧策略。两者使用相同初态、相同2ms物理步长，目标均为正向90°并五指稳定保持1秒。</p>
<p>本例新方案峰值 {new:.2f}°，旧策略 {old:.2f}°，均因支撑不足失败，目标保持0秒。先原速，再慢放同一次动作；所有静帧明确标注，不增加保持时间。这是选取的样例，不能代表成功率。</p>
<button onclick="const v=document.querySelector('video');v.currentTime=0;v.play()">从头播放</button><button onclick="document.querySelector('video').pause()">暂停</button>
<video controls playsinline preload="metadata" src="data:video/mp4;base64,{encoded}"></video>
<p>真实神经网络策略回放，无外力、无脚本接管。显示五指法向接触力，绿色为达到门槛，橙色为超过上限。当前训练继续运行。</p></main></html>'''
        (output/'index.html').write_text(html)
        (output/'recording_source.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps(dict(completed=True,output=str(output),seconds=count/20,angles=dict(new=new,old=old))),flush=True)
    finally:
        for panel in panels.values():panel.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();main(args.output)
