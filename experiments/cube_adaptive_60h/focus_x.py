"""Preserve Y independently and start a bounded four-worker X-only phase."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from pathlib import Path
from datetime import datetime,timezone,timedelta
from zoneinfo import ZoneInfo
import argparse,json,shutil,subprocess,sys,time
from .config import Profile,validate_profile
from .storage import load_source,atomic_json,digest
from .evaluate import score
from .controller import alive,link
from .fast_contacts import descriptor as contact_descriptor


def launch(root):
    root=Path(root).resolve();goal=json.loads((root/'goal.json').read_text())
    if datetime.now(timezone.utc)>=datetime.fromisoformat(goal['deadline_at']):
        raise ValueError('X phase budget expired; this resume will not reset its clock')
    state=json.loads((root/'controller_state.json').read_text())
    if alive(state.get('pid'),'experiments.cube_adaptive_60h.controller'):
        return state['pid']
    with (root/'controller.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,'-m','experiments.cube_adaptive_60h.controller','--goal',str(root)],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return child.pid


def create(previous,hours=24.):
    if not 1.<=hours<=72.:raise ValueError('Use a bounded1-72h phase')
    previous=Path(previous).resolve();old=json.loads((previous/'controller_state.json').read_text())
    active=Path(__file__).resolve().parents[1]/'ACTIVE_EXPERIMENT.json'
    if active.exists():
        existing=json.loads(active.read_text())
        if existing.get('task')=='x_focus_v1':
            candidate=Path(existing['goal_file']).parent;g=json.loads((candidate/'goal.json').read_text())
            if datetime.now(timezone.utc)<datetime.fromisoformat(g['deadline_at']):
                return candidate,launch(candidate)
    # Do not create a competing old controller/worker population.
    if alive(old.get('pid'),'experiments.cube_adaptive_60h.controller'):
        raise ValueError('Previous controller is active; stop and save it before changing allocation')
    for item in old['axes'].values():
        if alive(item.get('pid'),'experiments.cube_adaptive_60h.train'):
            raise ValueError('Previous trainer is still active; save it before changing allocation')
    x_source=Path(old['axes']['x']['best_by_target']['90']['path']).resolve()
    y_source=Path(old['axes']['y']['best_by_target']['90']['path']).resolve()
    _,x_meta=load_source(x_source);_,y_meta=load_source(y_source)
    now=datetime.now(timezone.utc);local=now.astimezone(ZoneInfo('Asia/Shanghai'));stamp=local.strftime('%Y%m%d_%H%M%S')
    project=Path(__file__).resolve().parents[2]
    preserved=project/'experiments/preserved_policies'/f'y90_best_{stamp}'
    checkpoint=preserved/'checkpoint';checkpoint.parent.mkdir(parents=True,exist_ok=False)
    shutil.copytree(y_source,checkpoint)
    names=[f.name for f in y_source.iterdir() if f.is_file()]
    source_hashes={n:digest(y_source/n) for n in names};saved_hashes={n:digest(checkpoint/n) for n in names}
    if source_hashes!=saved_hashes:raise RuntimeError('Y policy backup differs from its source')
    _,saved_y=load_source(checkpoint)
    if saved_y!=y_meta:raise RuntimeError('Y backup metadata changed')
    previews=sorted((previous/'previews').glob('y_best_*'))
    for preview in reversed(previews):
        r=preview/'recording_report.json'
        if not r.exists() or json.loads(r.read_text())['checkpoint']!=str(y_source):continue
        target=preserved/'preview';target.mkdir()
        for name in ('y_best_policy.mp4','index.html','recording_report.json','qa.json','final.png'):
            if (preview/name).exists():shutil.copy2(preview/name,target/name)
        break
    atomic_json(preserved/'manifest.json',dict(preserved_at=now.isoformat(),source=str(y_source),checkpoint=str(checkpoint),
        sha256=saved_hashes,training_steps=y_meta['training_steps'],evaluation=y_meta['evaluation'],
        read_only_checkpoint=True,training_allocation=0))
    preserved.joinpath('README.md').write_text(f'''# Y轴最佳策略保留\n\n保存于{local:%Y-%m-%d %H:%M:%S}（北京时间），后续X轴训练不会更新这些参数。\n\n策略为此前选出的Y90°最佳检查点，本段训练步数{y_meta['training_steps']:,}，平均最佳姿态误差4.68°。当前保存的是已录制约90°姿态接近效果的版本；五指稳握1秒尚未通过。\n\n- 完整检查点：checkpoint/（policy.zip、奖励归一化状态、参考状态、模型报告、元数据）。\n- 每个文件SHA256与原件逐一相同，见manifest.json。\n- 检查点文件及目录设为只读，训练分配为0。\n- 原速与慢放回放保留在preview/（如存在）。\n- 源检查点：{y_source}\n\n恢复推理可用experiments.cube_adaptive_60h.storage.load读取checkpoint目录；需保留其记录的匹配代码和依赖。\n''')
    for f in checkpoint.iterdir():
        if f.is_file():f.chmod(0o444)
    checkpoint.chmod(0o555)
    root=project/'experiments/cube_adaptive_60h/artifacts'/f'x_focus_{stamp}';(root/'sources').mkdir(parents=True,exist_ok=False)
    shutil.copytree(x_source,root/'sources/x');reference=str(root/'sources/x/reference.npz')
    profile=validate_profile(Profile(name='x_focus_capture_explore_v1',learning_rate=7.5e-5,gamma=.999,entropy=.002,
        action_fraction=.02,contact_weight=1.2,loss_weight=1.4,loss_grace=.35,pressure_potential_gain=2.,
        stall_weight=1.25,progress_gain=40.,grip_mix=.15,bc_weight=0.))
    goal=dict(task_id='x_focus_v1',objective='X from original initial grasp to signed+90deg and stable five-finger1s capture; preserve Y unchanged',
        active_axes=['x'],started_at=now.isoformat(),deadline_at=(now+timedelta(hours=hours)).isoformat(),hours=hours,
        review_interval_hours=3,next_review_at=(now+timedelta(hours=3)).isoformat(),allocation={'x':4,'y':0,'z':0},
        previous_run=str(previous),reference=reference,goal_complete=False,final_verification_trials_per_integrator=100,
        final_required_successes_per_integrator=100,user_authorized_adjustments=True,native_contact_backend='c_contacts_equivalent_v1',
        native_contact_descriptor=contact_descriptor(),preserved_y_checkpoint=str(checkpoint),preserved_y_manifest=str(preserved/'manifest.json'),
        training_seed_base=310000,review_seed_base=12000000,verification_seed_base=200000000,review_verification_seed_base=220000000)
    entry=dict(segment=0,source_checkpoint=str(root/'sources/x'),source_training_steps=x_meta['training_steps'],profile=profile.dictionary(),
        initial_target=90.,history=[],best_by_target={'90':dict(path=str(root/'sources/x'),score=score(x_meta['evaluation']))},
        crashes=0,verified=False,status='preparing',exploration={'initial_std':.12,'max_std':.16},
        reward_transform='center_missing_tip_mount_v1',rollout_steps=256,gae_lambda=.98,
        physics_timesteps=[.001,.002,.001,.002],observation_transform=None)
    state=dict(status='prepared',pid=None,last_review_index=0,axes={'x':entry},reviews=[],goal_complete=False,
        preserved_y_checkpoint=str(checkpoint))
    atomic_json(root/'goal.json',goal);atomic_json(root/'controller_state.json',state)
    link(root/'x/best_90',root/'sources/x')
    (root/'README.md').write_text(f'''# X轴专项训练\n\n本轮预算{hours:g}小时，X4个CPU训练环境（1ms/2ms各2），Y/Z均0。每3小时复查，从原始初握转+90°并五指稳握1秒，最后两步长各100个新初态全部通过。\n\n继承X最佳权重（源检查点本段{x_meta['training_steps']:,}步），使用已验证的位置/缺指接近/支架辅助奖励，探索标准差0.12、上限0.16。保留原143维观测、7维动作、每周期2%限幅、57mm/77.97g魔方及20°安装倾角。新奖励使用新的critic输出/优化器/奖励统计，actor及输入归一化继承，非从零训练。\n\nY最佳策略已另行只读保存：{preserved}\n\n运行状态status.md；权重x/latest和x/best_90。恢复命令：\n\n/workspace/.venvs/aero-cube/bin/python -m experiments.cube_adaptive_60h.controller --goal {root}\n\n恢复会沿用本轮截止时间，不自动重置24小时。环境不可用期间不得虚构训练或复查进展。\n''')
    manifest=dict(task='x_focus_v1',goal_file=str(root/'goal.json'),status_file=str(root/'status.json'),human_status=str(root/'status.md'),
        active_axes=['x'],suspended_axes=['y','z'],allocation=goal['allocation'],parallel_environments=4,live_parallel_environments=0,
        runtime_status='starting',started_at=goal['started_at'],deadline_at=goal['deadline_at'],review_interval_hours=3,
        next_review_at=goal['next_review_at'],preserved_y_checkpoint=str(checkpoint),preserved_y_manifest=str(preserved/'manifest.json'),
        initial_grasp_evaluation='Unassisted neural policy from original initial grasp; no reset bank or scripted prefix',
        goal_degrees={'x':90},final_criterion='X signed+90±5deg, full pose error≤5deg, five real loaded fingers and low speeds for1s; dt1ms/2ms100of100 each',
        updated_at=now.isoformat(),readme=str(root/'README.md'))
    atomic_json(active,manifest)
    pid=launch(root)
    atomic_json(root/'launch.json',dict(started_at=now.isoformat(),controller_pid=pid,preserved_y=str(checkpoint),
        y_backup_hashes_equal=True,source_x=str(x_source),original_run_unmodified=str(previous)))
    return root,pid


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous-goal');p.add_argument('--hours',type=float,default=24.)
    p.add_argument('--resume-goal');args=p.parse_args()
    if args.resume_goal:
        root=Path(args.resume_goal).resolve();pid=launch(root)
    else:
        if not args.previous_goal:p.error('--previous-goal or --resume-goal is required')
        root,pid=create(args.previous_goal,args.hours)
    print(json.dumps(dict(goal=str(root),controller_pid=pid),ensure_ascii=False),flush=True)
