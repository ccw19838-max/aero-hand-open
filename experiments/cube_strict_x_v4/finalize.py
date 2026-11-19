"""Freeze policies before opening sealed seeds. Every completed episode survives restart."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from experiments.cube_adaptive_60h.config import Profile
from experiments.cube_adaptive_60h.storage import atomic_json, digest
from . import evaluation
from experiments.cube_two_driver_v3.env import LADDER as LEGACY_LADDER


def read(path):
    return json.loads(Path(path).read_text())


def active(deadline, stop):
    return time.time() < deadline and not stop()


def freeze_plan(state, final_seed):
    """Choose checkpoints using training/validation only, before any sealed outcome."""
    best = state.get('best', {})
    qualified = state.get('qualified', {})
    stages = [int(angle) for angle in state.get('curriculum_degrees', LEGACY_LADDER) if angle > 0]
    highest = max((angle for angle in stages if str(angle) in qualified), default=0)
    stage = int(state['stage'])

    def checkpoint(target, require_qualified=False):
        key = str(int(target))
        # A qualified checkpoint actually met the two-round progression test.
        item = qualified.get(key) if require_qualified else best.get(key)
        source = (item or {}).get('checkpoint')
        if source is None and not require_qualified:
            source = state.get('last_v4_checkpoint')
        if not source:
            raise ValueError(f'Missing frozen checkpoint for {target:g} degrees')
        return str(Path(source).resolve(strict=True))

    if state.get('final_quorum'):
        if stage != 90:
            raise ValueError('90-degree quorum must refer to the final stage')
        return [dict(name='acceptance_90', target=90, source=checkpoint(90, require_qualified='90' in qualified),
                     seed=final_seed, episodes=100, purpose='independent_final')]

    if highest == 90:
        raise ValueError('Qualified 90-degree stage missing final quorum')
    tasks = []
    if highest:
        tasks.append(dict(name=f'qualified_{highest}', target=highest, source=checkpoint(highest, True),
                          seed=final_seed + 1000, episodes=20, purpose='qualified_stage_review'))
    challenge = stages[stages.index(highest) + 1] if highest else stages[0]
    tasks.append(dict(name=f'challenge_{challenge}', target=challenge, source=checkpoint(challenge),
                      seed=final_seed + 2000, episodes=20, purpose='next_stage_diagnostic'))
    return tasks


def preserve_checkpoint(source, destination):
    """Copy the complete engine directory and atomically install the frozen version."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest(source/'policy.zip') != digest(destination/'policy.zip'):
            raise ValueError('Frozen policy differs from selected policy')
        return
    tmp = destination.with_name(f'.{destination.name}-{uuid.uuid4().hex[:8]}')
    try:
        shutil.copytree(source, tmp)
        if digest(source/'policy.zip') != digest(tmp/'policy.zip'):
            raise ValueError('Frozen policy integrity check failed')
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def frozen_manifest(root, state, job):
    path = root/'final'/'frozen.json'
    job_hash = digest(root/'job.json')
    if path.exists():
        manifest = read(path)
        if manifest['job_sha256'] != job_hash:
            raise ValueError('The frozen final evaluation belongs to a different job')
        return manifest
    tasks = freeze_plan(state, job['final_seed'])
    for task in tasks:
        dest = root/'final'/'checkpoints'/task['name']
        preserve_checkpoint(task['source'], dest)
        task.update(checkpoint=str(dest), policy_sha256=digest(dest/'policy.zip'))
    manifest = dict(frozen_at=datetime.now(timezone.utc).isoformat(), job_sha256=job_hash,
                    highest_qualified_degrees=max((t['target'] for t in tasks if t['purpose'] != 'next_stage_diagnostic'), default=0),
                    final_quorum=bool(state.get('final_quorum')), tasks=tasks,
                    resumable_checkpoint=state.get('last_v4_checkpoint'))
    atomic_json(path, manifest)
    return manifest


def evaluate_frozen(task, folder, profile, deadline, stop, model):
    folder.mkdir(parents=True,exist_ok=True)
    rows = []
    for dt in evaluation.TIMESTEPS:
        env = None
        try:
            for seed in range(task['seed'], task['seed'] + task['episodes']):
                path = folder/'episodes'/f'{round(dt*1000)}ms_{seed}.json'
                if path.exists():
                    row = read(path)
                    if (row['seed'] != seed or row['dt'] != dt or row['interrupted'] or
                            row.get('policy_sha256') != task['policy_sha256']):
                        raise ValueError(f'Invalid completed final row in {path.name}')
                elif active(deadline,stop):
                    if env is None:
                        env = evaluation.make_env(task['checkpoint'],profile,task['target'],dt,'v4')
                    row = evaluation.episode(env,model,seed,deadline=deadline,stop=stop)
                    if row['interrupted']:
                        break
                    row['policy_sha256'] = task['policy_sha256']
                    atomic_json(path,row)
                else:
                    # Read any completed rows after this seed on a resumed run.
                    continue
                rows.append(row)
        finally:
            if env is not None:
                env.close()
    report = evaluation.summarize_report(rows,task['target'],task['episodes'],task['seed'],
                                         arm='v4',purpose=task['purpose'])
    report.update(checkpoint=task['checkpoint'], policy_sha256=task['policy_sha256'])
    atomic_json(folder/'evaluation.json',report)
    return report


def find_failure(root, tasks_and_reports, deadline, stop):
    from .replay import select_cases
    # Try the next-stage challenge before the qualified lower stage.
    for task,report in reversed(tasks_and_reports):
        row = select_cases(report).get('typical_failure')
        if row is not None:
            return task,row,str(root/'final'/task['name']/'evaluation.json')
    # If the sealed policy passed all trials, failures from an older checkpoint
    # are labelled as such and never change the final denominator.
    for path in sorted((root/'evaluations').glob('**/*.json'),reverse=True):
        if not active(deadline,stop):
            break
        report = read(path)
        if (report.get('arm')!='v4' or not report.get('complete') or
                not report.get('from_original_initial_grasp') or not report.get('checkpoint')):
            continue
        row = select_cases(report).get('typical_failure')
        if row is None:
            continue
        checkpoint = Path(report['checkpoint']).resolve()
        if not (checkpoint/'policy.zip').exists():
            continue
        task = dict(name='archived_validation',checkpoint=str(checkpoint),target=report['target_degrees'],
                    policy_sha256=digest(checkpoint/'policy.zip'),purpose='archived_validation')
        return task,row,str(path)
    return None


def intact(media):
    return (Path(media['video']).is_file() and digest(media['video']) == media['sha256']
            and Path(media['trajectory']).is_file() and Path(media['states']).is_file())


def render_example(root, kind, example, profile, deadline, stop, loader, recorder):
    task, row, source_report = example
    folder = root/'final'/'videos'/kind
    pin = dict(target=task['target'],checkpoint=task['checkpoint'],policy_sha256=task['policy_sha256'],
               purpose=task['purpose'],source_report=source_report,seed=row['seed'],dt=row['dt'],kind=kind)
    marker = folder/'delivery.json'
    if marker.exists():
        previous = read(marker)
        if previous['pin'] == pin and intact(previous):
            return previous
    if not active(deadline,stop):
        return None
    model,_ = loader(task['checkpoint'])
    # The recorder checks the full terminal replay against the selected eval row.
    folder.mkdir(parents=True,exist_ok=True)
    media = recorder(model,Path(task['checkpoint']),profile,task['target'],row,folder,task['checkpoint'],
                     arm='v4',kind=kind,deadline=deadline,stop=stop)
    if not media:
        return None
    if not media.get('full_decode_passed') or media['policy_sha256'] != task['policy_sha256']:
        raise ValueError('Replay has not verified the frozen policy and encoded video')
    video = Path(media['video']).resolve(strict=True)
    if digest(video) != media['video_sha256']:
        raise ValueError('The delivered video does not match the verified encoding')
    trajectory = (folder / f'{kind}_trajectory.json').resolve(strict=True)
    states = (folder / f'{kind}_states.npz').resolve(strict=True)
    delivery = dict(pin=pin, video=str(video), sha256=media['video_sha256'], metadata=media,
                    trajectory=str(trajectory), states=str(states))
    atomic_json(marker,delivery)
    return delivery


def write_report(root, manifest, reports, result):
    path = root / 'final' / 'REPORT.md'
    if result['verified']:
        verdict = '两种物理步长分别通过了全部 100 个封存初态。'
    elif not manifest['final_quorum']:
        verdict = '训练尚未取得 90° 独立验收资格。'
    elif result['evaluation_complete']:
        verdict = '90° 独立验收结束，未达到两种物理步长各 100/100 的要求。'
    else:
        verdict = '90° 独立验收尚有未测初态。'
    attained = manifest['highest_qualified_degrees']
    first_target = manifest['tasks'][0]['target']
    stage = (f'训练中最高连续两轮达到晋级要求的角度为 {attained}°。' if attained
             else f'截至冻结时，{first_target:g}° 课程尚未连续两轮达到晋级要求。')
    lines = [
        '# X 轴翻面冻结评估', '', verdict, '', stage, '',
        '|冻结评估目标|物理步长|严格成功 / 已测 / 计划|首次违规前转角中位数|瞬时角度峰值|目标保持峰值|',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for task, report in reports:
        for dt, group in report['groups'].items():
            median = group['median_valid_peak_degrees']
            peak = '—' if median is None else f'{median:.2f}°'
            lines.append(f'|{task["target"]}°|{1000 * float(dt):g} ms|'
                         f'{group["successes"]} / {group["trials"]} / {group["requested_trials"]}|'
                         f'{peak}|{group["maximum_signed_angle"]:.2f}°|{group["maximum_stable_seconds"]:.2f} s|')
    lines.append('')
    for task, _ in reports:
        lines.append(f'- [{task["target"]}° 冻结策略]({os.path.relpath(task["checkpoint"], path.parent)}) · '
                     f'[逐初态结果]({task["name"]}/evaluation.json)')
    if manifest.get('resumable_checkpoint'):
        lines.append(f'- [可续训检查点]({os.path.relpath(manifest["resumable_checkpoint"], path.parent)})')
    for kind, title in [('best', '最佳样例'), ('typical_failure', '典型失败')]:
        media = result['artifacts'].get(kind)
        if media:
            pin = media['pin']
            scope = '历史验证' if pin['purpose'] == 'archived_validation' else '冻结评估'
            lines.append(f'- [{title}录像]({os.path.relpath(media["video"], path.parent)}) · '
                         f'[受力与轨迹记录]({os.path.relpath(media["trajectory"], path.parent)}) · '
                         f'[仿真状态]({os.path.relpath(media["states"], path.parent)})。'
                         f'{scope} {pin["target"]:g}°，{pin["dt"] * 1000:g} ms，初态种子 {pin["seed"]}。')
    lines.extend(['', result['note'], ''])
    temp = path.with_suffix('.md.tmp')
    temp.write_text('\n'.join(lines))
    os.replace(temp, path)


def finalize(root, state, deadline, stop=lambda: False):
    from .engine import load_for_eval
    from .replay import record_case, select_cases

    root = Path(root).resolve()
    job = read(root / 'job.json')
    deadline = min(float(deadline), datetime.fromisoformat(job['deadline_at']).timestamp())
    destination = root / 'final'
    destination.mkdir(exist_ok=True)
    result_path = destination / 'result.json'
    previous = read(result_path) if result_path.exists() else {}
    if previous.get('complete') and all(intact(media) for media in previous['artifacts'].values()):
        return previous

    if not (destination / 'frozen.json').exists() and not any(
        state.get(k) for k in ('last_v4_checkpoint', 'best', 'qualified')
    ):
        report = destination / 'REPORT.md'
        report.write_text('# X 轴翻面冻结评估\n\n冻结时没有可用的 V4 检查点，无法进行独立评估或录制回放。\n')
        result = dict(complete=False, verified=False, evaluation_complete=False,
                      highest_qualified_degrees=0, failure_status='pending', artifacts={}, reports=[],
                      report=str(report), error='No V4 checkpoint available when training closed')
        atomic_json(result_path, result)
        return result

    manifest = frozen_manifest(root, state, job)
    profile = Profile(**job['profile'])
    reports = []
    media = {}
    note = '封存初态逐项记录了三指全程受力和完整保持条件。'
    # Reserve enough time for two original-grasp replays; with recordings saved,
    # a retry can spend the remaining interval finishing unopened trials.
    reserve_path = destination / 'evaluation_deadline.json'
    if not reserve_path.exists():
        remaining = max(0., deadline - time.time())
        reserve = min(600., remaining * .12)
        atomic_json(reserve_path, dict(until=deadline - reserve, deadline=deadline))
    eval_deadline = min(deadline, read(reserve_path)['until'])
    prior_media = previous.get('artifacts', {})
    if prior_media.get('best') and prior_media.get('typical_failure'):
        eval_deadline = max(eval_deadline, deadline - 60.)

    for task in manifest['tasks']:
        candidate = Path(task['checkpoint'])
        if digest(candidate / 'policy.zip') != task['policy_sha256']:
            raise ValueError('Frozen checkpoint changed before final evaluation')
        model, metadata = load_for_eval(candidate)
        if metadata['arm'] != 'v4' or metadata['profile'] != job['profile']:
            raise ValueError('Frozen policy does not belong to the V4 arm and profile')
        report = evaluate_frozen(task, destination / task['name'], profile, eval_deadline, stop, model)
        reports.append((task, report))

    eval_complete = all(report['complete'] for _, report in reports)
    verified = any(evaluation.final_pass(report) for _, report in reports)
    error = None
    failure_status = 'pending'
    if not stop():
        try:
            selectable = [entry for entry in reports if entry[1]['records']]
            if selectable:
                task, report = selectable[0]
                row = select_cases(report).get('best')
                if row is not None:
                    example = (task, row, str(destination / task['name'] / 'evaluation.json'))
                    media['best'] = render_example(root, 'best', example, profile, deadline, stop, load_for_eval, record_case)
                failure = find_failure(root, reports, deadline, stop)
                if failure is not None:
                    media['typical_failure'] = render_example(root, 'typical_failure', failure, profile,
                                                               deadline, stop, load_for_eval, record_case)
                    failure_status = 'recorded' if media['typical_failure'] else 'pending'
                    if failure[0]['purpose'] == 'archived_validation':
                        note = '本轮封存评估的初态全部通过。失败录像来自先前的训练验证，相关检查点和初态已单独记录。'
                elif eval_complete and active(deadline, stop):
                    failure_status = 'none_observed'
                    note = '封存评估和已保存的可回放 V4 验证回合中均未找到失败样例。'
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            error = str(exc)

    complete = bool(eval_complete and media.get('best') and failure_status in ('recorded', 'none_observed'))
    if not complete:
        note = '表格列出了已完成数量，仍有未完成的评估或录像。'
    result = dict(complete=complete, verified=verified, evaluation_complete=eval_complete, artifacts=media,
                  failure_status=failure_status, highest_qualified_degrees=manifest['highest_qualified_degrees'],
                  frozen=str(destination / 'frozen.json'), report=str(destination / 'REPORT.md'), note=note,
                  reports=[str(destination / task['name'] / 'evaluation.json') for task, _ in reports],
                  updated_at=datetime.now(timezone.utc).isoformat(), error=error)
    write_report(root, manifest, reports, result)
    atomic_json(result_path, result)
    return result
