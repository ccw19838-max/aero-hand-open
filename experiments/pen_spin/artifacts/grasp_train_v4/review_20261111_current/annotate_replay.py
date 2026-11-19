from pathlib import Path
import json
import argparse
import numpy as np
import mediapy as media
from PIL import Image, ImageDraw, ImageFont

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--evaluation',default='native_evaluation')
args=parser.parse_args()
source=root/args.evaluation
assessment=json.loads((source/'evaluation.json').read_text())
steps=int(assessment.get('checkpoint_metadata',{}).get('training_steps',51200))
counts=assessment['failure_reason_counts']
successes=sum(r['success'] for r in assessment['results'])
episodes=assessment['episodes']
stable=assessment['median_stable_seconds']
labels=json.loads((source/'video_trajectory.json').read_text())
info=json.loads((source/'video_info.json').read_text())
font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fonts={n:ImageFont.truetype(font_path,n,index=2) for n in [16,18,20,22,26,30]}
reasons={0:'进行中',1:'无效支撑',2:'穿透超限',3:'连续脱手',4:'离开工作区',5:'超时',6:'非有限数值'}
out=root/f'grasp_{steps}_steps_30s.mp4'
with media.VideoReader(str(source/'rollout.mp4')) as reader, media.VideoWriter(str(out),shape=(720,1120),fps=20,codec='h264') as writer:
 for i,frame in enumerate(reader):
  row=labels[i]
  canvas=Image.new('RGB',(1120,720),(18,24,34));d=ImageDraw.Draw(canvas)
  d.text((22,9),f'多指握持策略 · {steps:,} 步检查点',font=fonts[30],fill=(245,247,250))
  d.text((24,48),'原生 MuJoCo  |  1× 仿真速度  |  竖直右手 / 130 mm 配重笔',font=fonts[18],fill=(166,183,203))
  closeup=Image.fromarray(frame).crop((150,60,470,380)).resize((640,640),Image.Resampling.LANCZOS)
  canvas.paste(closeup,(8,76))
  d.rounded_rectangle((660,83,1106,291),radius=12,fill=(32,42,57))
  d.text((681,99),'尚未学会稳定握持',font=fonts[26],fill=(255,172,133))
  d.text((681,145),f'独立评估：成功 {successes} / {episodes}',font=fonts[22],fill='white')
  d.text((681,184),f"穿透超限 {counts['penetration']} · 无效支撑 {counts['support']}",font=fonts[20],fill=(203,214,229))
  d.text((681,215),f"越界 {counts['envelope']} · 稳定保持中位数 {stable:g} s",font=fonts[20],fill=(203,214,229))
  d.text((681,252),'目标：拇指 + 至少另外两指，保持 1 s',font=fonts[18],fill=(158,182,207))
  terminal=bool(row.get('failure',0) or row.get('success',0))
  event_time=(row['episode_step']-1)*.05+row['terminal_event_step_seconds']
  d.text((681,317),f"回合 {row['episode']} / {info['episodes']}",font=fonts[26],fill='white')
  d.text((681,359),f"本回合任务时间：{event_time:.3f} s",font=fonts[20],fill=(216,224,236))
  status='成功' if row.get('success') else ('失败：'+reasons.get(int(row.get('failure_reason',0)),'其他') if terminal else '正在尝试接触')
  d.text((681,396),status,font=fonts[22],fill=(255,172,133) if terminal else (160,204,245))
  d.text((681,436),f"有效接触手指：{int(row.get('contact_fingers',0))} / 5",font=fonts[20],fill=(216,224,236))
  d.text((681,473),f"稳定保持：{row.get('stable_seconds',0):.3f} / 1.000 s",font=fonts[20],fill=(216,224,236))
  d.text((681,520),'失败后自动重置；本片包含多次尝试。',font=fonts[18],fill=(177,192,213))
  d.text((681,550),'彩色箭头是转笔参考标记；当前只练握持。',font=fonts[16],fill=(148,169,194))
  d.text((681,593),f"累计仿真时间 {row['total_sim_seconds']:.2f} / 30.00 s",font=fonts[20],fill=(224,232,243))
  d.rounded_rectangle((681,636,1083,644),radius=4,fill=(57,73,91))
  d.rounded_rectangle((681,636,681+402*(i+1)/len(labels),644),radius=4,fill=(79,170,218))
  d.text((681,668),'录像使用固定权重；训练已暂停，待修复初态检验。',font=fonts[16],fill=(148,169,194))
  writer.add_image(np.asarray(canvas))
  if i==2:canvas.save(root/'grasp_preview.png')
  if i%200==0:print('annotated_frames',i+1,flush=True)
(root/f'presentation_{steps}.json').write_text(json.dumps({'source':str(source.relative_to(root)/'rollout.mp4'),'output':out.name,'frames':len(labels),'fps':20,'seconds':30,'crop':[150,60,470,380],'playback_rate':1,'physics_and_episode_order_unchanged':True,'annotations':'Chinese task criteria, evaluation results, per-frame terminal status and reset count'},ensure_ascii=False,indent=2)+'\n')
print(out)
