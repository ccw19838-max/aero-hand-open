"""Select and record existing X checkpoints under corrected contact feedback.

Read-only policy evaluation, no training or automatic restart. The candidate
screen uses selected diagnostic seeds and is not a success-rate benchmark.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from pathlib import Path
from datetime import datetime,timezone
import argparse,base64,json,math,time,shutil
import numpy as np
import torch,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw
from experiments.cube_adaptive_60h.config import Profile
from experiments.cube_adaptive_60h.storage import load as legacy_load
from experiments.cube_adaptive_60h.record_preview import Panel,FONTS
from .env import NativeEnv
from .io import load,atomic_json,digest,fingerprint


def policy_for(path):
    meta=json.loads((path/'metadata.json').read_text())
    model,meta=(load if meta['task']=='x_contact_recovery_v2' else legacy_load)(path)
    return model,meta


def rollout(path,seed,dt,record=False):
    model,meta=policy_for(path)
    env=NativeEnv(path/'reference.npz','x',physical_dt=dt,target_degrees=90.,profile=Profile(**meta['profile']))
    frames=[]
    try:
        obs,info=env.reset(seed=seed)
        def save():
            if record:frames.append(dict(qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),ctrl=env.data.ctrl.copy(),
                time=float(env.data.time),info=info.copy(),seed=seed,episode=0,kind='原速 1×'))
        save()
        for _ in range(250):
            obs,_,done,_,info=env.step(model.predict(obs,deterministic=True)[0]);save()
            if done:break
        result=dict(seed=seed,dt=dt,final=info,success=bool(info['success']))
        return result,(env.model,frames,meta) if record else None
    finally:env.close()


class CenteredPanel(Panel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cube=self.model.body('cube').id
    def draw(self,index):
        # Arrow origin follows the center; its direction is fixed at initial
        # grasp. This is a display change only, not a constraint on the cube.
        frame=self.frames[index]
        self.arrow_start=frame['qpos'][16:19].copy()
        self.arrow_end=self.arrow_start+.07*np.asarray(frame['info']['axis_world_vector'])
        row=super().draw(index);draw=ImageDraw.Draw(row)
        for panel_index,camera in enumerate(self.cameras):
            self.renderer.update_scene(self.data,camera=camera)
            cameras=self.renderer.scene.camera
            position=(np.asarray(cameras[0].pos)+np.asarray(cameras[1].pos))/2
            forward=np.asarray(cameras[0].forward);up=np.asarray(cameras[0].up)
            right=np.cross(forward,up);focal=300*cameras[0].frustum_near/(cameras[0].frustum_top-cameras[0].frustum_bottom)
            def project(point):
                relative=point-position;depth=float(relative@forward)
                if depth<=0:raise ValueError('Axis behind camera')
                return (panel_index*640+320+focal*float(relative@right)/depth,
                    60+150-focal*float(relative@up)/depth)
            a=np.asarray(project(self.arrow_start));b=np.asarray(project(self.arrow_end));v=b-a
            norm=np.linalg.norm(v)
            if norm>4:
                v/=norm;normal=np.array([-v[1],v[0]])
                draw.line([tuple(a),tuple(b)],fill='#101827',width=7)
                draw.line([tuple(a),tuple(b)],fill='#ff5864',width=3)
                draw.polygon([tuple(b),tuple(b-12*v+5*normal),tuple(b-12*v-5*normal)],fill='#ff5864')
                draw.ellipse((a[0]-4,a[1]-4,a[0]+4,a[1]+4),fill='white',outline='#ff5864',width=2)
                draw.text(tuple(b+6*normal),'X',font=FONTS[22],fill='#ff5864',stroke_width=2,stroke_fill='#101827')
        return row


def main(output,render_only=False):
    torch.set_num_threads(1);output=Path(output).resolve();manifest=json.loads((output/'manifest.json').read_text())
    started=time.monotonic();candidates=[]
    if render_only:
        selected=json.loads((output/'selected.json').read_text())
        assert selected['implementation_sha256']==fingerprint()
        assert selected['policy_sha256']==digest(Path(selected['checkpoint'])/'policy.zip')
        candidates=json.loads((output/'candidate_screen.json').read_text())
    for candidate in ([] if render_only else manifest['candidates']):
        path=Path(candidate['checkpoint']).resolve();records=[]
        for seed in manifest['seeds']:
            for dt in (.001,.002):
                result,_=rollout(path,seed,dt);records.append(result)
        result=dict(candidate,checkpoint=str(path),records=records,
            success_count=sum(r['success'] for r in records),trials=len(records),
            peak=max(r['final']['max_progress_degrees'] for r in records),
            mean_best_error=float(np.mean([r['final']['best_angle_error_degrees'] for r in records])))
        candidates.append(result);atomic_json(output/'candidate_screen.json',candidates)
        print(json.dumps(dict(candidate=candidate['name'],peak=result['peak'],successes=result['success_count'],trials=result['trials'])),flush=True)
    chosen=max(candidates,key=lambda c:(c['success_count'],c['peak'],-c['mean_best_error']))
    case=max(chosen['records'],key=lambda r:(r['success'],r['final']['max_progress_degrees']))
    selected=dict(candidate=chosen['name'],checkpoint=chosen['checkpoint'],case=case,
        selection='Existing high-performing checkpoint candidates; successful trials first, then maximum signed X angle. Selected diagnostic seeds, not an independent success-rate estimate.',
        inference='V2 corrected thumb forces, per-finger distances and contact-aware phase observation; old policy weights unchanged',
        implementation_sha256=fingerprint(),policy_sha256=digest(Path(chosen['checkpoint'])/'policy.zip'))
    atomic_json(output/'selected.json',selected)
    result,(model,frames,meta)=rollout(Path(chosen['checkpoint']),case['seed'],case['dt'],record=True)
    differences={k:abs(result['final'][k]-case['final'][k]) for k in
        ('max_progress_degrees','signed_progress_degrees','angle_error_degrees','elapsed_seconds','max_stable_seconds')}
    assert max(differences.values())<1e-9 and result['final']['failure_reason']==case['final']['failure_reason']
    np.savez_compressed(output/'replay_states.npz',**{key:np.asarray([f[key] for f in frames]) for key in ('qpos','qvel','ctrl','time')})
    atomic_json(output/'trajectory.json',dict(selection=selected,exact_reproduction=differences,trace=[f['info'] for f in frames]))
    panel=CenteredPanel('x',model,frames,meta);rendered=[]
    try:
        for index in range(len(frames)):
            frame=panel.draw(index);draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,1280,59),fill=(13,20,32))
            draw.text((18,1),'X轴最佳转角候选 · 修正接触反馈重评 · 目标 +90°',font=FONTS[25],fill='#f87171')
            draw.text((18,34),f'选取的最好样例 | seed={case["seed"]} | {case["dt"]*1000:g}ms | 左：整体  右：近景',font=FONTS[18],fill='#b9cbe1')
            rendered.append(frame)
        fps=20;duration=float(frames[-1]['time']);times=np.asarray([f['time'] for f in frames]);timeline=[];sequence=[]
        def add(label,indices):
            timeline.append(dict(label=label,start_seconds=len(sequence)/fps))
            sequence.extend((label,index) for index in indices)
        add('初态静帧 · 不计入保持',[0]*10)
        for speed,label in [(1.,'原速 1×'),(.25,'同一次动作 · 0.25×慢放')]:
            ticks=np.arange(0.,duration,.05*speed)
            indices=np.clip(np.searchsorted(times,ticks,side='right')-1,0,len(frames)-1).tolist()
            indices.append(len(frames)-1);add(label,indices)
            add('失败终点静帧 · 不计入保持' if not result['success'] else '成功终点静帧 · 不计入保持',[len(frames)-1]*20)
        add('结果静帧 · 训练已暂停 · 不计入保持',[len(frames)-1]*40)
        video=output/'x_best_replay.mp4';peak_index=max(range(len(frames)),key=lambda i:frames[i]['info']['signed_progress_degrees'])
        def compose(label,index):
            canvas=Image.new('RGB',(1280,640),(8,13,23));draw=ImageDraw.Draw(canvas)
            draw.text((18,10),'X轴策略回放｜'+label,font=FONTS[28],fill='white')
            canvas.paste(rendered[index],(0,58))
            draw.text((18,550),'红色 X：穿过魔方重心的目标轴示意（叠加显示）；轴方向固定，不锁住魔方。',font=FONTS[20],fill='#b9cbe1')
            draw.text((18,579),'五指力已修正拇指根部漏计。绿色达门槛，橙色超力；慢放和静帧不增加保持时间。',font=FONTS[20],fill='#b9cbe1')
            draw.text((18,608),'历史检查点重评样例；未完成90°。当前训练已保存暂停，待观看后再启动。' if not result['success'] else '本例成功不代表通过独立100初态验收。训练已保存暂停。',font=FONTS[18],fill='#fbbf24')
            return canvas
        with imageio.get_writer(video,fps=fps,codec='libx264',quality=8,macro_block_size=1,
                ffmpeg_params=['-threads','1','-movflags','+faststart','-pix_fmt','yuv420p']) as writer:
            for label,index in sequence:writer.append_data(np.asarray(compose(label,index)))
        compose('最大转角静帧 · 不计入保持',peak_index).save(output/'poster.png')
        compose('终点静帧 · 不计入保持',len(frames)-1).save(output/'final.png')
        report=dict(created_at=datetime.now(timezone.utc).isoformat(),video=video.name,sha256=digest(video),
            width=1280,height=640,fps=fps,frames=len(sequence),seconds=len(sequence)/fps,physical_seconds=duration,
            final=result['final'],timeline=timeline,selected=selected,wall_seconds=time.monotonic()-started,
            policy_actions_only=True,no_external_forces_or_state_resets_within_episode=True,training_paused=True)
        atomic_json(output/'recording_report.json',report)
        encoded=base64.b64encode(video.read_bytes()).decode();peak=result['final']['max_progress_degrees']
        html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>X轴最佳候选回放</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#0d1420;color:#e5edf9;font:16px/1.7 system-ui,sans-serif}}main{{max-width:1360px;margin:auto;padding:22px}}p{{color:#b9cbe1}}video{{display:block;width:100%;background:#050b12}}button{{padding:10px 18px;margin:8px;border:0;border-radius:8px;background:#2563eb;color:white;font-size:16px}}</style>
<main><h1>X轴：最佳转角候选回放</h1><p>本次选取样例最大转角 {peak:.2f}°，目标仍是正向90°并五指稳定保持1秒。左侧整体、右侧近景，显示修正后的五指接触力。原速之后是同一次动作的慢放，静帧不计保持时间。</p>
<p>从历史候选和当前检查点中重评选择，使用修正后的反馈；这段视频不是成功率证明。没有外力、脚本接管或回合内瞬移。训练已保存并暂停，待观看后再启动。</p>
<button onclick="const v=document.querySelector('video');v.currentTime=0;v.play()">从头播放</button><button onclick="document.querySelector('video').pause()">暂停</button>
<video controls playsinline preload="metadata" src="data:video/mp4;base64,{encoded}"></video></main></html>'''
        (output/'index.html').write_text(html)
        shutil.copy2(Path(__file__),output/'recording_source.py')
        print(json.dumps(dict(completed=True,output=str(output),peak=peak,seconds=report['seconds'],candidate=chosen['name'])),flush=True)
    finally:panel.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--render-only',action='store_true');args=parser.parse_args();main(args.output,args.render_only)
