"""Thirty-second, native-policy X/Y replay with real force and hold telemetry."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):
    os.environ[_key]='1'
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from pathlib import Path
from datetime import datetime,timezone
import argparse,base64,json,hashlib,time
import numpy as np
import torch,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from .env import AdaptiveEnv
from .config import Profile
from .storage import load,atomic_json,digest

FPS=20
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONTS={n:ImageFont.truetype(FONT,n,index=2) for n in (18,20,22,25,28)}
COLORS={'x':(248,113,113),'y':(74,222,128)}
FAILURES={'insufficient_rotation_contacts':'换指后支撑不足','forbidden_support':'碰到支架或地面',
    'penetration':'穿透超限','workspace':'离开操作区','lost_hand_contact':'脱离手部接触',
    'stagnation':'进展停滞','timeout':'回合超时','numerical':'数值异常'}


def capture(output,axis,pin,count):
    checkpoint=Path(pin['checkpoint']);model,metadata=load(checkpoint)
    selected=pin['selected_case'];dt=selected['dt'];profile=Profile(**metadata['profile'])
    env=AdaptiveEnv(checkpoint/'reference.npz',axis,physical_dt=dt,target_degrees=90.,profile=profile)
    frames=[];episodes=[];episode=0;verified=False
    def save(info,kind,seed):
        frames.append(dict(qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),ctrl=env.data.ctrl.copy(),
            time=float(env.data.time),info=info.copy(),kind=kind,seed=seed,episode=episode))
    try:
        while len(frames)<count:
            seed=selected['seed'] if episode==0 else 7300000+(100000 if axis=='y' else 0)+episode
            obs,info=env.reset(seed=seed)
            for _ in range(min(5,count-len(frames))):save(info,'新回合 · 初始静帧',seed)
            done=False
            while len(frames)<count:
                action=model.predict(obs,deterministic=True)[0]
                obs,_,done,_,info=env.step(action);save(info,'原速 1×',seed)
                if done:break
            if done:
                if episode==0:
                    expected=selected['final']
                    difference=max(abs(info[k]-expected[k]) for k in
                        ('elapsed_seconds','signed_progress_degrees','max_progress_degrees','angle_error_degrees','max_stable_seconds'))
                    if difference>1e-7 or info['success']!=selected['success'] or info['failure_reason']!=expected['failure_reason']:
                        raise RuntimeError('Pinned validation/native replay mismatch: '+str(difference))
                    verified=True
                episodes.append(dict(seed=seed,dt=dt,selected_validation_example=episode==0,final=info))
                for _ in range(min(10,count-len(frames))):save(info,'终点静帧 · 不计入保持',seed)
            else:
                episodes.append(dict(seed=seed,dt=dt,selected_validation_example=episode==0,clip_ended_mid_episode=True,final=info))
            episode+=1
        atomic_json(output/f'{axis}_trajectory.json',dict(axis=axis,checkpoint=str(checkpoint),
            training_steps=metadata['training_steps'],physical_dt=dt,selected_validation_reproduced=verified,
            first_episode_selection=pin['selection'],episodes=episodes,
            frames=[{k:v for k,v in f.items() if k not in ('qpos','qvel','ctrl')} for f in frames]))
        np.savez_compressed(output/f'{axis}_states.npz',**{k:np.asarray([f[k] for f in frames]) for k in ('qpos','qvel','ctrl','time')})
        print(json.dumps(dict(phase='captured',axis=axis,episodes=len(episodes),frames=len(frames),pinned_case_reproduced=verified)),flush=True)
        return env.model,frames,metadata
    finally:env.close()


class Panel:
    def __init__(self,axis,model,frames,metadata):
        self.axis=axis;self.model=model;self.frames=frames;self.metadata=metadata
        self.data=mujoco.MjData(model);self.renderer=mujoco.Renderer(model,height=300,width=640)
        main=mujoco.MjvCamera();main.lookat[:]=[.105,0.,.195];main.distance=.36;main.azimuth=135.;main.elevation=-35.
        close=mujoco.MjvCamera();close.lookat[:]=frames[0]['qpos'][16:19];close.distance=.235;close.azimuth=100.;close.elevation=-75.
        self.cameras=(main,close);self.arrow_start=frames[0]['qpos'][16:19]+np.array([-.035,-.04,.035])
        self.arrow_end=self.arrow_start+.055*np.array(frames[0]['info']['axis_world_vector'])
    def draw(self,index):
        frame=self.frames[index];info=frame['info'];axis=self.axis
        for key in ('qpos','qvel','ctrl'):getattr(self.data,key)[:]=frame[key]
        self.data.time=frame['time'];mujoco.mj_forward(self.model,self.data)
        row=Image.new('RGB',(1280,480),(13,20,32))
        for i,camera in enumerate(self.cameras):
            self.renderer.update_scene(self.data,camera=camera)
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
            geom=self.renderer.scene.geoms[self.renderer.scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.ones(3)*.001,
                np.zeros(3),np.eye(3).ravel(),np.r_[np.array(COLORS[axis])/255.,1.])
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.002,self.arrow_start,self.arrow_end)
            self.renderer.scene.ngeom+=1
            row.paste(Image.fromarray(self.renderer.render()),(i*640,60))
        draw=ImageDraw.Draw(row)
        draw.text((18,1),f'{axis.upper()}轴 · 目标 +90° · 第{frame["episode"]+1}回合 · {frame["kind"]}',font=FONTS[25],fill=COLORS[axis])
        source='选取的验证样例' if frame['episode']==0 else '新初态尝试'
        draw.text((18,34),f'保留最佳权重 {self.metadata["training_steps"]:,}步  |  {info["physical_dt"]*1000:g}ms  |  seed={frame["seed"]}  |  {source}',font=FONTS[18],fill='#b9cbe1')
        draw.text((18,365),f'转角 {info["signed_progress_degrees"]:+.1f}°   三维误差 {info["angle_error_degrees"]:.1f}°   五指目标保持 {info["stable_seconds"]:.2f} / 1.00 s',font=FONTS[22],fill='white')
        for finger,(name,force) in enumerate(zip(('拇指','食指','中指','无名指','小指'),info['finger_forces'])):
            lower=.2 if finger==0 else .1;upper=20. if finger==0 else 12.
            color='#fbbf24' if force>upper else '#4ade80' if force>=lower else '#94a3b8'
            x=18+finger*250
            draw.ellipse((x,405,x+10,415),fill=color)
            draw.text((x+16,395),f'{name} {force:.2f}N',font=FONTS[22],fill=color)
        reason=FAILURES.get(info['failure_reason'],info['failure_reason'])
        status='成功' if info['success'] else '失败：'+reason if reason else '仍在尝试'
        draw.text((18,435),f'{status}  |  本回合峰值 {info["max_progress_degrees"]:.1f}°  |  仿真 t={info["elapsed_seconds"]:.2f}s',font=FONTS[22],fill='#fbbf24')
        return row
    def close(self):self.renderer.close()


def captured(output,axis,pin,count):
    checkpoint=Path(pin['checkpoint']);metadata=json.loads((checkpoint/'metadata.json').read_text())
    trace=json.loads((output/f'{axis}_trajectory.json').read_text())
    if not trace['selected_validation_reproduced'] or trace['checkpoint']!=str(checkpoint):raise ValueError('Unverified cached trajectory')
    with np.load(output/f'{axis}_states.npz',allow_pickle=False) as archive:
        states={k:archive[k].copy() for k in ('qpos','qvel','ctrl','time')}
    frames=trace['frames']
    if len(frames)!=count or any(len(v)!=count for v in states.values()):raise ValueError('Incomplete saved video frames')
    for index,frame in enumerate(frames):
        for key,value in states.items():frame[key]=value[index]
    env=AdaptiveEnv(checkpoint/'reference.npz',axis,physical_dt=trace['physical_dt'],target_degrees=90.,profile=Profile(**metadata['profile']))
    try:return env.model,frames,metadata
    finally:env.close()


def main(output,seconds=30.,render_only=False):
    torch.set_num_threads(1);output=Path(output);manifest=json.loads((output/'manifest.json').read_text())
    count=round(seconds*FPS);panels={};started=time.monotonic()
    try:
        for axis in ('x','y'):
            model,frames,metadata=(captured if render_only else capture)(output,axis,manifest['pins'][axis],count)
            panels[axis]=Panel(axis,model,frames,metadata)
        video=output/'xy_policy_30s.mp4'
        with imageio.get_writer(video,fps=FPS,codec='libx264',quality=7,macro_block_size=1,
                ffmpeg_params=['-threads','1','-movflags','+faststart','-pix_fmt','yuv420p']) as writer:
            for index in range(count):
                image=Image.new('RGB',(1280,1040),(8,13,23));draw=ImageDraw.Draw(image)
                draw.text((18,5),'X / Y 轴转魔方策略回放 · 原速 · 手掌保留20°倾角',font=FONTS[28],fill='white')
                image.paste(panels['x'].draw(index),(0,50));image.paste(panels['y'].draw(index),(0,540))
                draw.text((18,1018),'绿色：达到受力门槛；橙色：超过力上限。静帧不计保持；失败后明确复位。',font=FONTS[18],fill='#b9cbe1')
                writer.append_data(np.asarray(image))
                if index in (0,round(count*.45),count-1):image.save(output/('poster.png' if index==round(count*.45) else f'frame_{index:04d}.png'))
                if index%60==0:print(json.dumps(dict(phase='encoding',frame=index,total=count)),flush=True)
        info=dict(created_at=datetime.now(timezone.utc).isoformat(),video=video.name,width=1280,height=1040,
            fps=FPS,frames=count,seconds=count/FPS,sha256=digest(video),checkpoint_pins=manifest['pins'],
            source='native deterministic policy replays, not live PPO weights or planning controllers',
            recording_source_sha256=digest(Path(__file__)),
            trajectory_sha256={axis:digest(output/f'{axis}_trajectory.json') for axis in ('x','y')},
            note='First episode selected from each protected checkpoint validation; subsequent episodes use fresh seeds. Static terminal frames explicitly labelled; no additional physics after termination.',
            elapsed_wall_seconds=time.monotonic()-started,original_goal_unchanged=True)
        atomic_json(output/'recording_report.json',info)
        encoded=base64.b64encode(video.read_bytes()).decode()
        html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>X/Y魔方策略30秒回放</title>
<style>*{box-sizing:border-box}body{margin:0;background:#0d1420;color:#e5edf9;font:16px/1.7 system-ui,sans-serif}main{max-width:1360px;margin:auto;padding:22px}p{color:#b9cbe1}video{display:block;width:100%;background:#050b12}button,a{display:inline-block;padding:10px 18px;margin:8px 8px 8px 0;border:0;border-radius:8px;background:#2563eb;color:white;font-size:16px;text-decoration:none}</style>
<main><h1>X / Y 轴转魔方：30秒真实策略回放</h1><p>上方X轴，下方Y轴。使用恢复时保留的较好检查点；首轮是选取的验证样例，之后为新初态尝试。显示原速物理运动、五指真实接触力、姿态误差和目标保持计时。静帧已标明，不计入保持；失败后重新开始。</p>
<p>目标保持不变：从初握转+90°、完整姿态误差≤5°，五指受力并稳定保持1秒。此次录屏不能视为通过验收。</p>
<button onclick="const v=document.querySelector('video');v.currentTime=0;v.play()">从头播放</button><button onclick="document.querySelector('video').pause()">暂停</button>
<video controls playsinline preload="metadata" src="data:video/mp4;base64,'''+encoded+'''"></video>
<p>绿色表示达到各指受力门槛，橙色表示超过上限。保留失败终点，最后一帧若回合仍在进行，不能推断该回合结果。此页面含嵌入视频，可离线观看。</p></main></html>'''
        (output/'index.html').write_text(html)
        print(json.dumps(dict(completed=True,output=str(output),video=str(video))),flush=True)
    finally:
        for panel in panels.values():panel.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--seconds',type=float,default=30.)
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args();main(args.output,args.seconds,args.render_only)
