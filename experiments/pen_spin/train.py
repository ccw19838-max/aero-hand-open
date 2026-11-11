"""Preparation-gated CPU PPO curriculum with a hard parent wall-clock limit."""
import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .config import PenSpinConfig
from .policy import require_prepared, save_checkpoint, load_checkpoint, network_factory, digest_file


STAGES = (('hold',0.),('turn90',math.pi/2),('turn180',math.pi),('turn360',2*math.pi))


class StopStage(Exception):
    def __init__(self, reason):
        self.reason=reason
        super().__init__(reason)


class CurriculumMonitor:
    """Deterministic promotion/stagnation decisions, separate from task success."""
    def __init__(self, config):
        self.config=config
        self.consecutive=0
        self.stagnant=0
        self.best_success=-1.
        self.best_progress=-1.

    def update(self, success_rate, progress):
        if not 0 <= success_rate <= 1 or not math.isfinite(progress):
            raise ValueError('Invalid validation metrics')
        self.consecutive=self.consecutive+1 if success_rate>=self.config.promotion_success_rate else 0
        improved=success_rate>self.best_success+1e-8 or progress>self.best_progress+1e-4
        self.stagnant=0 if improved else self.stagnant+1
        self.best_success=max(self.best_success,success_rate)
        self.best_progress=max(self.best_progress,progress)
        if self.consecutive>=self.config.promotion_checks:
            return 'promote'
        if self.stagnant>=self.config.stagnation_checks:
            return 'stagnant'
        return 'continue'


def stage_budget_seconds(config):
    if config.task_mode=='grasp':
        return [config.budget_hours*3600]
    weights=[float(x) for x in config.stage_hours]
    if len(weights)!=4 or min(weights)<=0:
        raise ValueError('Exactly four positive curriculum stage budgets are required')
    total=config.budget_hours*3600
    return [total*w/sum(weights) for w in weights]


def _append(path,record):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(record)+'\n')
        stream.flush()


def _worker(request_path):
    # Import heavy libraries only after the CPU package setup and the gate.
    request=json.loads(Path(request_path).read_text())
    config=PenSpinConfig.from_dict(request['config'])
    reference=Path(request['reference'])
    require_prepared(reference, config)
    output=Path(request['output']);output.mkdir(parents=True,exist_ok=True)
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper
    from .env import make_env
    from .evaluate import evaluate_params

    start=time.monotonic()
    deadline=start+config.budget_hours*3600
    stop_requested=False
    def stop_handler(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,stop_handler)
    signal.signal(signal.SIGINT,stop_handler)
    params=None; total_steps=0; first_stage=0
    if request.get('resume'):
        params,old_config,old_reference,meta=load_checkpoint(request['resume'])
        # Budget/seed are run controls; observation and physics definitions may
        # not silently change while reusing a saved normalizer.
        exempt={'budget_hours','stage_hours','seed','validation_interval_seconds'}
        old={k:v for k,v in old_config.to_dict().items() if k not in exempt}
        new={k:v for k,v in config.to_dict().items() if k not in exempt}
        if old!=new or digest_file(old_reference)!=digest_file(reference):
            raise ValueError('Resume requires matching physics, observation, and prepared reference')
        first_stage=int(meta.get('stage_index',0))
        if not 0 <= first_stage < len(STAGES):
            raise ValueError('Checkpoint curriculum stage is outside the supported range')
        total_steps=int(meta.get('total_training_steps',meta.get('training_steps',0)))
    batch_start_steps=total_steps
    max_steps=request.get('num_timesteps')
    stage_reports=[]
    selected=None
    completed=False
    try:
        stages = STAGES[:1] if config.task_mode=='grasp' else STAGES
        for stage_index in range(first_stage,len(stages)):
            name,goal=STAGES[stage_index]
            stage_dir=output/'stages'/name;stage_dir.mkdir(parents=True,exist_ok=True)
            stage_start=time.monotonic()
            stage_deadline=min(deadline,stage_start+stage_budget_seconds(config)[stage_index])
            stage_clock=output/'active_stage.json'
            temporary_clock=output/'.active_stage.tmp'
            temporary_clock.write_text(json.dumps(dict(stage=name,deadline_monotonic=stage_deadline)))
            os.replace(temporary_clock,stage_clock)
            monitor=CurriculumMonitor(config)
            next_validation=stage_start+config.validation_interval_seconds
            stage_steps_base=total_steps
            last_params=params
            best_rank=(-1.,-float('inf'),-float('inf'))
            last_validation=None
            stage_result='unknown'
            env=make_env(config,reference,goal_angle=goal,backend='mjx')
            stage_max_steps=(max_steps-(total_steps-batch_start_steps)) if max_steps else 2**31
            if stage_max_steps<=0:
                raise StopStage('timestep_limit')
            update_steps=config.batch_size*config.num_minibatches*config.unroll_length
            evaluations=math.ceil(stage_max_steps/update_steps)+1

            def checkpoint_callback(steps,make_policy,current_params):
                nonlocal last_params,total_steps,next_validation,last_validation,best_rank,selected
                last_params=current_params
                total_steps=stage_steps_base+int(steps)
                selected=str(stage_dir/'last')
                save_checkpoint(stage_dir/'last',current_params,config,reference,goal,steps,
                                stage=name,stage_index=stage_index,total_training_steps=total_steps,
                                validated=False,batch_started_unix=request.get('batch_started_unix'),
                                run_id=request.get('run_id'),
                                batch_deadline_unix=request.get('batch_deadline_unix'),
                                evaluation_reserved_seconds=request.get('evaluation_reserved_seconds',0.))
                now=time.monotonic()
                live=dict(status='training',task_mode=config.task_mode,stage=name,
                          total_training_steps=total_steps,elapsed_seconds=now-start,
                          selected_checkpoint=selected,last_validation=last_validation,
                          run_id=request.get('run_id'),
                          batch_deadline_unix=request.get('batch_deadline_unix'))
                live_tmp=output/'.live_status.tmp'
                live_tmp.write_text(json.dumps(live,indent=2)+'\n')
                os.replace(live_tmp,output/'live_status.json')
                if stop_requested or now>=deadline:
                    raise StopStage('total_budget_or_interrupt')
                if now>=stage_deadline:
                    raise StopStage('stage_budget')
                if max_steps and total_steps-batch_start_steps>=max_steps:
                    raise StopStage('timestep_limit')
                if now>=next_validation:
                    report=evaluate_params(config,reference,current_params,goal,
                                            episodes=config.validation_episodes,
                                            seed_start=1000,backend='mjx')
                    progress=report['median_valid_progress'] if goal else report['median_stable_seconds']
                    decision=monitor.update(report['success_rate'],progress)
                    last_validation=dict(report=report,decision=decision)
                    _append(output/'curriculum.jsonl',dict(stage=name,training_steps=total_steps,
                            elapsed_seconds=time.monotonic()-start,**last_validation))
                    rank=(report['success_rate'],progress,report['mean_return'])
                    if rank>best_rank:
                        best_rank=rank
                        save_checkpoint(stage_dir/'best',current_params,config,reference,goal,steps,
                                        stage=name,stage_index=stage_index,total_training_steps=total_steps,
                                        validated=True,validation_success_rate=report['success_rate'],
                                        run_id=request.get('run_id'),
                                        batch_started_unix=request.get('batch_started_unix'),
                                        batch_deadline_unix=request.get('batch_deadline_unix'),
                                        evaluation_reserved_seconds=request.get('evaluation_reserved_seconds',0.))
                    elapsed=time.monotonic()-stage_start
                    next_validation=stage_start+(math.floor(elapsed/config.validation_interval_seconds)+1)*config.validation_interval_seconds
                    if decision=='promote':raise StopStage('promote')
                    if decision=='stagnant':raise StopStage('stagnant')

            try:
                _,returned_params,_=ppo.train(
                    environment=env,eval_env=env,num_timesteps=stage_max_steps,
                    num_envs=config.num_envs,num_eval_envs=config.num_eval_envs,
                    episode_length=round(config.episode_seconds/config.ctrl_dt),
                    action_repeat=1,wrap_env_fn=wrapper.wrap_for_brax_training,
                    learning_rate=config.learning_rate,entropy_cost=config.entropy_cost,
                    discounting=config.discounting,unroll_length=config.unroll_length,
                    batch_size=config.batch_size,num_minibatches=config.num_minibatches,
                    num_updates_per_batch=config.num_updates_per_batch,
                    normalize_observations=True,reward_scaling=1.,
                    clipping_epsilon=config.clipping_epsilon,gae_lambda=config.gae_lambda,
                    max_grad_norm=config.max_grad_norm,network_factory=network_factory(config),
                    seed=config.seed+stage_index,num_evals=evaluations,
                    run_evals=False,policy_params_fn=checkpoint_callback,
                    restore_params=params,restore_value_fn=True,
                    num_resets_per_eval=0,log_training_metrics=False)
                last_params=returned_params
                stage_result='timestep_limit' if max_steps else 'sample_limit'
            except StopStage as event:
                stage_result=event.reason
            params=last_params
            if stage_result=='promote' and (stage_dir/'best/checkpoint.pkl').exists():
                params,_,_,_=load_checkpoint(stage_dir/'best')
                selected=str(stage_dir/'best')
            stage_reports.append(dict(stage=name,goal_angle=goal,result=stage_result,
                                      elapsed_seconds=time.monotonic()-stage_start,
                                      total_training_steps=total_steps,validation=last_validation))
            report=dict(status='running',stage_reports=stage_reports,selected_checkpoint=selected,
                        training_steps_this_batch=total_steps-batch_start_steps,
                        total_training_steps=total_steps,elapsed_seconds=time.monotonic()-start,
                        curriculum_complete=False,full_skill_success_not_assumed=True,
                        resume_semantics='weights_and_normalizer_only_optimizer_reinitialized')
            (output/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
            if stage_result!='promote':break
        else:
            completed=True
    except StopStage as event:
        stage_reports.append(dict(result=event.reason))
    finally:
        report=dict(status='completed' if completed else 'stopped',stage_reports=stage_reports,
                    run_id=request.get('run_id'),
                    selected_checkpoint=selected,training_steps_this_batch=total_steps-batch_start_steps,
                    total_training_steps=total_steps,elapsed_seconds=time.monotonic()-start,
                    curriculum_complete=completed and config.task_mode=='spin',
                    grasp_training_complete=completed and config.task_mode=='grasp',
                    task_mode=config.task_mode,formal_heldout_evaluation_required=True,
                    full_skill_success_not_assumed=True,
                    batch_deadline_unix=request.get('batch_deadline_unix'),
                    evaluation_reserved_seconds=request.get('evaluation_reserved_seconds',0.),
                    resume_semantics='weights_and_normalizer_only_optimizer_reinitialized')
        (output/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0


def run_guarded(request, output):
    """An outer watchdog bounds initial compilation as well as Python callbacks."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    request=dict(request,run_id=uuid.uuid4().hex)
    path=output/'training_request.json';path.write_text(json.dumps(request,indent=2)+'\n')
    configured_seconds=float(request['config']['budget_hours'])*3600
    seconds=min(configured_seconds,max(0.,request.get('training_deadline_unix',time.time()+configured_seconds)-time.time()))
    command=[sys.executable,'-m','experiments.pen_spin.train','--_worker',str(path.resolve())]
    start=time.monotonic()
    stage_clock=output/'active_stage.json'
    stage_clock.unlink(missing_ok=True)
    scope='total_training_budget'
    with (output/'worker.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            # Ask the worker to finish before the hard deadline; compilation
            # and validation are still forcibly bounded if they cannot return.
            hard_deadline=start+seconds
            while True:
                stage_deadline=float('inf')
                if stage_clock.exists():
                    active=json.loads(stage_clock.read_text())
                    stage_deadline=active['deadline_monotonic']
                effective_deadline=min(hard_deadline,stage_deadline)
                scope='stage_budget' if stage_deadline<hard_deadline else 'total_training_budget'
                grace=min(10.,max(0.,effective_deadline-start)*.01)
                remaining=effective_deadline-grace-time.monotonic()
                if remaining<=0:raise subprocess.TimeoutExpired(command,seconds)
                try:
                    code=process.wait(timeout=min(1.,remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except (subprocess.TimeoutExpired,KeyboardInterrupt) as event:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            remaining=max(0.,effective_deadline-time.monotonic())
            if isinstance(event,KeyboardInterrupt):remaining=min(10.,remaining)
            try:process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait()
            existing=output/'training_report.json'
            report=json.loads(existing.read_text()) if existing.exists() else {}
            if report.get('run_id')!=request['run_id']:report={}
            if not report.get('selected_checkpoint'):
                candidates=[]
                for bundle in output.glob('stages/*/last/checkpoint.pkl'):
                    try:
                        _,_,_,info=load_checkpoint(bundle)
                        if info.get('run_id')==request['run_id']:
                            candidates.append((info.get('total_training_steps',0),str(bundle.parent)))
                    except (ValueError,OSError):
                        continue
                if candidates:report['selected_checkpoint']=max(candidates)[1]
            report.update(status='wallclock_stopped',elapsed_seconds=time.monotonic()-start,
                          run_id=request['run_id'],
                          watchdog_limit_seconds=seconds,watchdog_scope=scope,
                          batch_deadline_unix=request.get('batch_deadline_unix'),
                          evaluation_reserved_seconds=request.get('evaluation_reserved_seconds',0.),
                          full_skill_success_not_assumed=True)
            existing.write_text(json.dumps(report,indent=2)+'\n')
            return 0
    if code:
        report=dict(status='error',worker_exit_code=code,elapsed_seconds=time.monotonic()-start,
                    log=str(output/'worker.log'),full_skill_success_not_assumed=True)
        (output/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return code


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--output',type=Path,default=Path('experiments/pen_spin/artifacts/train'))
    parser.add_argument('--config',type=Path)
    parser.add_argument('--budget-hours',type=float)
    parser.add_argument('--num-timesteps',type=int,help='Optional short-run total sample cap, not a claim of skill convergence')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--new-batch',action='store_true',help='With --resume, explicitly start a new bounded batch clock')
    parser.add_argument('--_worker',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args._worker:return _worker(args._worker)
    if args.reference is None:parser.error('--reference is required')
    if args.new_batch and not args.resume:parser.error('--new-batch requires --resume')
    config=(PenSpinConfig.load(args.config) if args.config else
            load_checkpoint(args.resume)[1] if args.resume else
            PenSpinConfig.load(args.reference.parent/'config.json') if (args.reference.parent/'config.json').exists()
            else PenSpinConfig())
    if args.budget_hours is not None:config=replace(config,budget_hours=args.budget_hours)
    if args.num_timesteps is not None and args.num_timesteps<1:parser.error('--num-timesteps must be positive')
    try:
        prepared=require_prepared(args.reference, config)
    except (ValueError,OSError) as error:
        parser.error(str(error))
    now=time.time()
    if args.new_batch:
        batch_start=now;batch_deadline=now+config.budget_hours*3600
    else:
        clock=load_checkpoint(args.resume)[3] if args.resume else prepared
        batch_start=clock.get('batch_started_unix')
        batch_deadline=clock.get('batch_deadline_unix')
        if not batch_start or not batch_deadline:
            parser.error('Preparation lacks the first-batch clock; rerun prepare in a new directory')
        batch_deadline=min(batch_deadline,batch_start+config.budget_hours*3600)
    remaining=max(0.,batch_deadline-now)
    reserve=min(900.,remaining*.1)
    request=dict(config=config.to_dict(),reference=str(args.reference.resolve()),
                 output=str(args.output.resolve()),num_timesteps=args.num_timesteps,
                 resume=str(args.resume.resolve()) if args.resume else None,
                 batch_started_unix=batch_start,batch_deadline_unix=batch_deadline,
                 training_deadline_unix=batch_deadline-reserve,evaluation_reserved_seconds=reserve)
    code=run_guarded(request,args.output)
    report_path=args.output/'training_report.json'
    report=json.loads(report_path.read_text())
    selected=report.get('selected_checkpoint')
    if code==0 and selected:
        from .evaluate import main as evaluate_main
        evaluation_output=args.output/'final_evaluation'
        final_goal = '0' if config.task_mode=='grasp' else '360'
        evaluation_code=evaluate_main(['--checkpoint',selected,'--output',str(evaluation_output),
                                      '--episodes','100','--seed-start','10000','--goal-degrees',final_goal,
                                      '--video-seconds','30'])
        report['final_evaluation']=dict(output=str(evaluation_output),exit_code=evaluation_code,
                                        requested_goal_degrees=int(final_goal),requested_episodes=100)
    else:
        report['final_evaluation']=dict(status='not_run',reason='No completed checkpoint is available' if not selected else 'Training worker failed')
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print((args.output/'training_report.json').read_text())
    return code


if __name__=='__main__':
    raise SystemExit(main())
