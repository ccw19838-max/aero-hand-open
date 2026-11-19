"""Replay a pinned checkpoint on original initial states, showing the strict score."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import tempfile
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from experiments.cube_adaptive_60h.config import Profile
from experiments.cube_adaptive_60h.record_preview import FAILURES, FONTS
from experiments.cube_adaptive_60h.storage import atomic_json, digest
from experiments.cube_contact_recovery_v2.record_best_x import CenteredPanel
from .evaluation import episode, evaluate, failure_reason, make_env
from .reward_trace import record_task_rewards


NAMES = {'index':'食指', 'middle':'中指', 'ring':'无名指'}
KINDS = {'best':'最佳样例','typical_failure':'典型失败样例'}
REASONS = {**FAILURES, 'three_support_lost': '三根稳定指中的接触越界',
           'recovered_after_violation': '三指越界后重新抓住', '': '目标保持不足'}
FPS = 20


def select_cases(report):
    rows = [r for r in report['records'] if not r['interrupted']]
    if not rows:
        return {}
    best = max(rows, key=lambda r: (r['success'], r['final']['max_strict_hold_seconds'],
                                    min(report['target_degrees'],r['final']['eligible_peak_degrees']),
                                    -r['final']['best_angle_error_degrees']))
    choices = {'best':best}
    failures = [r for r in rows if not r['success']]
    if failures:
        def signature(row):
            violation = row['final']['first_violation']
            return (failure_reason(row), tuple(violation['fingers']) if violation else ())
        counts = Counter(signature(r) for r in failures)
        typical_class = min(counts, key=lambda reason:(-counts[reason],reason))
        similar = [r for r in failures if signature(r)==typical_class]
        center = np.median([r['final']['eligible_peak_degrees'] for r in similar])
        choices['typical_failure'] = min(similar,key=lambda r:abs(r['final']['eligible_peak_degrees']-center))
    return choices


def _verify(saved, replayed):
    for field in ('success','dt','seed','interrupted'):
        if saved[field] != replayed[field]:
            raise ValueError(f'Replay differs from the selected episode in {field}')
    for field in ('strict_success','first_violation','failure_reason'):
        if saved['final'][field] != replayed['final'][field]:
            raise ValueError(f'Replay differs in {field}')
    fields = ('eligible_peak_degrees','max_progress_degrees','elapsed_seconds','max_strict_hold_seconds',
              'signed_progress_degrees','angle_error_degrees')
    for field in fields:
        if not np.isclose(saved['final'][field],replayed['final'][field],rtol=0,atol=1e-8):
            raise ValueError(f'Replay differs in {field}')
    if 'raw_task_return' in saved['final'] and not np.isclose(
            saved['final']['raw_task_return'], replayed['final']['raw_task_return'], rtol=0, atol=1e-8):
        raise ValueError('Replay differs in raw task return')


def _draw(panel, index, target, kind, held_frame):
    # The existing renderer projects the fixed X direction through the instantaneous cube center.
    body = panel.draw(index)
    info = panel.frames[index]['info']
    text = ImageDraw.Draw(body)
    text.rectangle((0,0,1279,59),fill='#0d1420')
    label = '终点静帧' if held_frame else '原速'
    text.text((16,1), f'{KINDS[kind]}  ·  X 轴目标 +{target:g}°  ·  {label}',font=FONTS[25],fill='white')
    text.text((16,34),f'左侧全景  右侧近景   红箭头沿重心处 X 轴   初态 {panel.frames[index]["seed"]}   物理步长 {1000*info["physical_dt"]:g} ms',font=FONTS[18],fill='#bfccde')
    text.rectangle((0,362,1279,395),fill='#0d1420')
    text.text((16,363),f'转角 {info["signed_progress_degrees"]:+.2f}°   首次违规前峰值 {info["eligible_peak_degrees"]:.2f}°   三维误差 {info["angle_error_degrees"]:.2f}°',font=FONTS[22],fill='white')
    text.rectangle((0,432,1279,479),fill='#0d1420')
    text.text((16,435),f'五指到位保持 {info["strict_hold_seconds"]:.2f} / 1.00 秒    仿真时间 {info["elapsed_seconds"]:.2f} 秒',font=FONTS[22],fill='#e6edf7')
    reward = info['raw_task_reward']
    current = '初态' if reward is None else f'{reward:+.2f}'
    text.text((783,435),f'本步奖励 {current}    累计 {info["raw_task_return"]:+.2f}',
              font=FONTS[22],fill='#e6edf7')
    canvas = Image.new('RGB',(1280,608),'#101b2b')
    canvas.paste(body,(0,0))
    draw = ImageDraw.Draw(canvas)
    thumb_moment,pinky_moment = info['driver_moments_Nm']
    draw.text((16,489),f'绕 X 轴力矩    拇指 {thumb_moment:+.4f} N·m     小指 {pinky_moment:+.4f} N·m',font=FONTS[22],fill='#7dd3fc')
    first = info['first_violation']
    if first:
        fingers = '、'.join(NAMES[name] for name in first['fingers'])
        detail = f'首次违规    {fingers}    动作开始后 {first["time_s"]:.2f} 秒    当时转角 {first["angle_deg"]:+.2f}°'
    else:
        detail = '首次违规    本段尚无三指越界记录'
    draw.text((16,530),detail,font=FONTS[22],fill='#fda4af' if first else '#86efac')
    reason = info['failure_reason']
    terminal = index == len(panel.frames)-1
    result = '进行中'
    if terminal:
        result = '严格通过' if info['strict_success'] else f'严格未通过    {REASONS.get(reason,reason)}'
    draw.text((16,571),f'{result}     接触力与力矩来自仿真采样',font=FONTS[18],fill='#c4ccdc')
    return np.asarray(canvas)


def record_case(model, inputs, profile, target, row, output, checkpoint, *, arm='v4',kind='best',
                deadline=float('inf'),stop=lambda:False):
    """Reproduce exactly one selected result. Publish only after complete encoding and decode."""
    if kind not in KINDS or row['interrupted'] or (kind=='typical_failure' and row['success']):
        raise ValueError('Expected a completed episode of the requested kind')
    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    def check_budget():
        if stop() or time.time() >= deadline:
            raise TimeoutError('Replay interrupted or the experiment budget expired')
    check_budget()
    # An already running experiment can still supply the earlier bare evaluation factory.
    env = record_task_rewards(make_env(inputs,profile,target,row['dt'],arm),arm)
    frames = []
    def capture(base,info):
        frames.append(dict(qpos=base.data.qpos.copy(),qvel=base.data.qvel.copy(),ctrl=base.data.ctrl.copy(),
                           time=float(base.data.time),info=info,seed=row['seed'],episode=0,kind=KINDS[kind]))
    panel = None
    try:
        replayed = episode(env,model,row['seed'],capture=capture,deadline=deadline,stop=stop)
        check_budget()
        _verify(row,replayed)
        checkpoint_file = checkpoint if checkpoint.is_file() else checkpoint/'policy.zip'
        policy_hash = digest(checkpoint_file)
        panel = CenteredPanel('x',env.model,frames,{'training_steps':int(model.num_timesteps)})
        with tempfile.TemporaryDirectory(dir=output,prefix='.recording_') as directory:
            staging = Path(directory)
            video = staging/f'{kind}.mp4'
            times = np.asarray([frame['time'] for frame in frames])
            times -= times[0]
            ticks = np.arange(0.,times[-1],1/FPS)
            sources = np.clip(np.searchsorted(times,ticks+1e-10,side='right')-1,0,len(frames)-1).tolist()
            # The terminal sample may fall between video frame boundaries.
            sources.append(len(frames)-1)
            with imageio.get_writer(video,fps=FPS,codec='libx264',quality=8,macro_block_size=1,
                                   ffmpeg_params=['-threads','1','-movflags','+faststart']) as writer:
                for index in sources:
                    check_budget()
                    writer.append_data(_draw(panel,index,target,kind,False))
                last = _draw(panel,len(frames)-1,target,kind,True)
                for _ in range(FPS):
                    check_budget()
                    writer.append_data(last)
            check_budget()
            timeout = min(120.,max(.1,deadline-time.time()))
            subprocess.run(['ffmpeg','-v','error','-xerror','-threads','1','-i',str(video),'-f','null','-'],
                           check=True,timeout=timeout,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            check_budget()
            np.savez_compressed(staging/f'{kind}_states.npz',**{
                key:np.asarray([f[key] for f in frames]) for key in ('qpos','qvel','ctrl','time')})
            atomic_json(staging/f'{kind}_trajectory.json',dict(seed=row['seed'],dt=row['dt'],
                       reward_units='task reward before VecNormalize, including warmup and terminal transition',
                       raw_task_return=replayed['final']['raw_task_return'],
                       trace=[f['info'] for f in frames],result=replayed))
            metadata = dict(kind=kind,video=str(output/video.name),video_sha256=digest(video),
                            checkpoint=str(checkpoint),policy_sha256=policy_hash,arm=arm,target_degrees=target,
                            selected={key:value for key,value in row.items() if key!='trace'},
                            raw_task_return=replayed['final']['raw_task_return'],
                            elapsed_physics_seconds=float(times[-1]),fps=FPS,frames=len(sources)+FPS,
                            terminal_still_seconds=1.,full_decode_passed=True)
            for file in staging.iterdir():
                os.replace(file,output/file.name)
            atomic_json(output/f'{kind}.json',metadata)
            return metadata
    finally:
        if panel is not None:
            panel.close()
        env.close()


def record_examples(model, inputs, profile, report, output, checkpoint, *, arm='v4',
                    deadline=float('inf'),kinds=('best','typical_failure'),stop=lambda:False):
    if report['arm'] != arm:
        raise ValueError('Report and policy arm disagree')
    cases = select_cases(report)
    artifacts = {}
    for kind in kinds:
        if kind not in KINDS:
            raise ValueError(f'Unknown replay kind {kind}')
        if kind in cases:
            artifacts[kind] = record_case(model,inputs,profile,report['target_degrees'],cases[kind],
                                          output,checkpoint,arm=arm,kind=kind,deadline=deadline,stop=stop)
    result = dict(selection='Best ranks strict completion, qualified hold time, eligible angle capped at target, then pose error. '
                            'Typical failure uses the modal failure class and its median eligible angle.',
                  checkpoint=str(Path(checkpoint).resolve()),target_degrees=report['target_degrees'],
                  requested=list(kinds),artifacts=artifacts,
                  unavailable=[kind for kind in kinds if kind not in artifacts])
    atomic_json(Path(output)/'replay_index.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--target',type=float,default=5.)
    parser.add_argument('--report',type=Path,help='Existing completed evaluation of this same checkpoint')
    parser.add_argument('--arm',choices=('v4','v3'),help='Defaults to the checkpoint arm')
    parser.add_argument('--episodes',type=int,default=5)
    parser.add_argument('--seed',type=int,default=6900000)
    args = parser.parse_args()
    from .engine import load_for_eval
    model,metadata = load_for_eval(args.checkpoint)
    arm = args.arm or metadata['arm']
    if arm != metadata['arm']:
        parser.error('Requested policy arm differs from the checkpoint')
    inputs = args.inputs or args.checkpoint.resolve()
    profile = Profile(**metadata['profile'])
    report = (json.loads(args.report.read_text()) if args.report
              else evaluate(model,inputs,profile,args.target,args.episodes,args.seed,arm=arm,purpose='preview'))
    output = record_examples(model,inputs,profile,report,args.output,args.checkpoint,arm=arm)
    print(json.dumps({kind:value['video'] for kind,value in output['artifacts'].items()},ensure_ascii=False))
