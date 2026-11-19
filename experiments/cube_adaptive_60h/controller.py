"""Persistent, budget-anchored three-hour review controller; no external messages."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
from pathlib import Path
from datetime import datetime,timezone,timedelta
from zoneinfo import ZoneInfo
from dataclasses import replace
import argparse,fcntl,json,signal,subprocess,sys,time,shutil
from .config import Profile,LADDER,validate_profile
from .storage import atomic_json,load,implementation_digest,digest
from .evaluate import evaluate,compact,score
from .decisions import choose,next_review,final_pass
from .rotation_report import RotationEvidence


def alive(pid,identity):
    p=Path('/proc')/str(pid)/'cmdline'
    return p.exists() and identity.encode() in p.read_bytes()


def link(path,target):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.new');temp.unlink(missing_ok=True)
    temp.symlink_to(Path(target).resolve(),target_is_directory=True);os.replace(temp,path)


class Controller:
    def __init__(self,goal_path):
        self.root=Path(goal_path).resolve();self.goal=json.loads((self.root/'goal.json').read_text())
        self.axes=tuple(self.goal.get('active_axes',('x','y')))
        if not self.axes or len(set(self.axes))!=len(self.axes) or any(a not in ('x','y') for a in self.axes):
            raise ValueError('Invalid active training axes')
        self.rotation_evidence=RotationEvidence(self.root)
        self.deadline=datetime.fromisoformat(self.goal['deadline_at']).timestamp()
        self.path=self.root/'controller_state.json';self.stop=False;self.children={};self.logs={}
        self.state=json.loads(self.path.read_text()) if self.path.exists() else dict(
            status='preparing',pid=os.getpid(),last_review_index=0,axes={},reviews=[],goal_complete=False)
        self.state['pid']=os.getpid()
        self.reference=str(Path(self.goal['previous_run'])/'x/latest/reference.npz')
        for key in ('SIGTERM','SIGINT'):signal.signal(getattr(signal,key),lambda a,b:setattr(self,'stop',True))
    def persist(self):
        self.state['updated_at']=datetime.now(timezone.utc).isoformat()
        self.state['deadline_at']=self.goal['deadline_at'];self.state['next_review_at']=self.goal.get('next_review_at')
        self.state['allocation']=self.goal['allocation'];self.state['task']=self.goal.get('task_id','adaptive_xy_60h_v1')
        self.state['active_axes']=list(self.axes)
        atomic_json(self.path,self.state);atomic_json(self.root/'status.json',self.state)
        atomic_json(self.root/'goal.json',self.goal)
        local=lambda value:datetime.fromisoformat(value).astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        lines=[f'# {self.goal.get("hours",60):g}小时训练监控',f'窗口：{local(self.goal["started_at"])} 至 {local(self.goal["deadline_at"])}（北京时间）',
            f'下次定时复查：{local(self.goal["next_review_at"])}',f'状态：{self.state["status"]}；最终完成：{self.state.get("goal_complete",False)}',
            '', '| 轴 | 分段 | 当前课程 | 本段采样步 | 状态 |','|---|---:|---:|---:|---|']
        for axis,e in self.state['axes'].items():
            live=e.get('live',{})
            lines.append(f'| {axis.upper()} | {e["segment"]} | {live.get("stage",e["initial_target"])}° | {live.get("training_steps",0)} | {e.get("status","preparing")} |')
        lines+=['','最终要求：从原始初握转+90°，五指实际受力保持1秒；每轴在1ms和2ms两组各100个新初态全部通过。',
            '详细定时复查见 reviews/latest.md；配置和检查点在各分段内保存。']
        (self.root/'status.md').write_text('\n'.join(lines)+'\n')
        self.rotation_evidence.write()
    def initialize(self):
        if self.state['axes']:return
        old=Path(self.goal['previous_run']);sources=self.root/'sources';sources.mkdir(exist_ok=True)
        baseline={}
        for axis,alias in (('x','best_0'),('y','best_30')):
            if axis not in self.axes:continue
            source=(old/axis/alias).resolve();dest=sources/axis
            if not dest.exists():shutil.copytree(source,dest)
            meta=json.loads((dest/'metadata.json').read_text())
            baseline[axis]=dict(checkpoint=str(source),training_steps=meta['training_steps'],evaluation=meta.get('evaluation'))
            profile=Profile(name='x_contact_recovery',bc_weight=0.) if axis=='x' else Profile(name='y_best_retention',learning_rate=7.5e-5)
            self.state['axes'][axis]=dict(segment=0,source_checkpoint=str(dest),profile=profile.dictionary(),
                initial_target=10. if axis=='x' else 30.,history=[],best_by_target={},crashes=0,verified=False)
        self.reference=str(sources/'x/reference.npz')
        self.goal['reference']=self.reference
        atomic_json(self.root/'reviews/000_initial_baseline.json',dict(
            observed_at=datetime.now(timezone.utc).isoformat(),baseline=baseline,
            decisions=dict(x='recover contact throughout rotation; finer angle curriculum; higher gamma',
                y='restore protected 65% teacher; lower LR; retain successful 30deg feedback actions'),
            strict_90_verified=False))
        self.persist()
    def stop_previous(self):
        old=Path(self.goal['previous_run']);s=json.loads((old/'status.json').read_text());pid=s.get('pid')
        if alive(pid,'experiments.cube_contact_ppo.pipeline'):
            os.kill(pid,signal.SIGTERM)
            until=time.time()+60.
            while time.time()<until and alive(pid,'experiments.cube_contact_ppo.pipeline'):time.sleep(.25)
            if alive(pid,'experiments.cube_contact_ppo.pipeline'):raise RuntimeError('Old training did not stop safely')
        for axis in self.axes:
            p=old/axis/'status.json'
            if p.exists():
                s=json.loads(p.read_text())
                if alive(s.get('pid'),'experiments.cube_contact_ppo.train'):raise RuntimeError('Old axis worker still active')
    def prepare_retention(self):
        if 'y' not in self.axes:return
        p=self.root/'retention_y.npz'
        if p.exists():return
        from .retention import build
        build(self.root/'sources/y',self.reference,'y',Profile(**self.state['axes']['y']['profile']),p,deadline=self.deadline)
    def segment_path(self,axis):return self.root/axis/f'segment_{self.state["axes"][axis]["segment"]:03d}'
    def job_path(self,axis):return self.segment_path(axis)/'job.json'
    def ensure_job(self,axis):
        entry=self.state['axes'][axis];output=self.segment_path(axis);output.mkdir(parents=True,exist_ok=True)
        path=output/'job.json'
        if not path.exists():
            job=dict(axis=axis,output=str(output),reference=self.reference,
                profile=entry['profile'],initial_target=entry['initial_target'],source_checkpoint=entry['source_checkpoint'],
                deadline_at=self.goal['deadline_at'],seed=self.goal.get('training_seed_base',120000)+1000*('xy'.index(axis))+entry['segment'],
                segment_index=entry['segment'],
                retention=entry.get('retention_override',str(self.root/'retention_y.npz') if axis=='y' else None),
                native_contact_backend=self.goal.get('native_contact_backend','python'),
                native_contact_descriptor=self.goal.get('native_contact_descriptor'),
                initial_eval_episodes=20,save_seconds=300.,eval_seconds=1200.,eval_steps=16384)
            job['verification_seed_base']=self.goal.get('verification_seed_base',60000000)
            for field in ('retention_scale','prefix_source','prefix_steps','prefix_probability','prefix_sha256','rollout_steps','gae_lambda','observation_transform','physics_timesteps','use_sde','sde_sample_freq'):
                if field in entry:job[field]=entry[field]
            job.update(entry.get('exploration',{}));job['reward_transform']=entry.get('reward_transform');atomic_json(path,job)
        link(self.root/axis/'current',output)
        link(self.root/axis/'status.json',output/'status.json')
        return path
    def spawn(self,axis):
        entry=self.state['axes'][axis]
        if entry['verified'] or self.stop or time.time()>=self.deadline:return
        job=self.ensure_job(axis)
        if alive(entry.get('pid'),str(job)):return
        output=self.segment_path(axis);command=[sys.executable,'-m','experiments.cube_adaptive_60h.train','--job',str(job)]
        if (output/'latest').exists():command+=['--resume']
        log=(output/'training.log').open('a');self.logs[axis]=log
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);self.children[axis]=child
        entry.update(pid=child.pid,status='starting');self.persist()
    def stop_axis(self,axis):
        entry=self.state['axes'][axis];job=self.job_path(axis)
        if alive(entry.get('pid'),str(job)):
            os.kill(entry['pid'],signal.SIGTERM);until=time.time()+60.
            while time.time()<until and alive(entry['pid'],str(job)):time.sleep(.25)
            if alive(entry['pid'],str(job)):raise RuntimeError('Training stop timeout: '+axis)
        child=self.children.get(axis)
        if child is not None:child.poll()
        if axis in self.logs:self.logs.pop(axis).close()
    def collect(self,axis):
        entry=self.state['axes'][axis];output=self.segment_path(axis);p=output/'status.json'
        if not p.exists():return None
        status=json.loads(p.read_text());entry['live']=status
        entry['status']=status['status'];entry['training_steps']=status.get('training_steps',0)
        for key,path in status.get('best_paths',{}).items():
            target=Path(path)
            if not (target/'metadata.json').exists():continue
            meta=json.loads((target/'metadata.json').read_text());evaluation=meta.get('evaluation')
            previous=entry['best_by_target'].get(key)
            if evaluation and (previous is None or score(evaluation)>tuple(previous['score'])):
                entry['best_by_target'][key]=dict(path=str(target),score=score(evaluation))
                link(self.root/axis/('best_'+key),target)
        protected=self.root/f'protected_best_{axis}30.json'
        if protected.exists():
            teacher=json.loads(protected.read_text());previous=entry['best_by_target'].get('30')
            if previous is None or tuple(teacher['score'])>tuple(previous['score']):
                entry['best_by_target']['30']=dict(path=teacher['path'],score=teacher['score'])
                link(self.root/axis/'best_30',teacher['path'])
        if (output/'latest').exists():link(self.root/axis/'latest',output/'latest')
        verified=output/'verified'
        if verified.exists():
            meta=json.loads((verified/'metadata.json').read_text())
            if final_pass(meta.get('evaluation')):
                entry['verified']=True;entry['verified_checkpoint']=str(verified.resolve())
                link(self.root/axis/'verified',verified)
        return status
    def apply_operator_request(self,spawn=True):
        path=self.root/'operator_request.json'
        if not path.exists():return
        request=json.loads(path.read_text());identity=request['id']
        if identity in self.state.get('applied_operator_requests',[]):return
        axis=request['axis']
        if axis not in self.axes or request['initial_target'] not in LADDER:raise ValueError('Invalid review action')
        profile=validate_profile(Profile(**request['profile']))
        exploration={key:float(request[key]) for key in ('initial_std','max_std') if key in request}
        if any(not .01<=value<=.25 for value in exploration.values()):raise ValueError('Invalid exploration setting')
        if exploration.get('initial_std',0.)>exploration.get('max_std',.25):raise ValueError('Initial noise exceeds ceiling')
        from .reward_registry import TRANSFORMS
        if request.get('reward_transform') not in TRANSFORMS:raise ValueError('Unknown reward transform')
        from .training_parameters import temporal_parameters,worker_timesteps
        temporal_parameters(request)
        worker_timesteps({'physics_timesteps':request.get('physics_timesteps',self.state['axes'][axis].get('physics_timesteps',(.001,.002)))})
        from .structured_exploration import settings as exploration_settings
        exploration_settings({k:request.get(k,self.state['axes'][axis].get(k,default))
            for k,default in (('use_sde',False),('sde_sample_freq',8))})
        if request.get('retention_scale') is not None and not .01<=float(request['retention_scale'])<=.25:
            raise ValueError('Invalid retention action scale')
        if request.get('retention_override'):
            from .retention import read
            read(request['retention_override'])
        observation_transform=request.get('observation_transform',self.state['axes'][axis].get('observation_transform'))
        if observation_transform is not None:
            from .canonical_policy import validate
            validate(observation_transform)
        if request.get('prefix_source'):
            from .prefix_curriculum import read_actions
            read_actions(request['prefix_source'],int(request['prefix_steps']))
            if not 0.<float(request['prefix_probability'])<=.75:raise ValueError('Invalid auxiliary prefix mix')
            if digest(request['prefix_source'])!=request['prefix_sha256']:raise ValueError('Actual prefix changed')
        from .storage import load_source
        _,source_meta=load_source(request['source_checkpoint'])  # validate before stopping a live worker
        source_transform=source_meta.get('state',{}).get('provenance',{}).get('observation_transform')
        if source_transform is not None and source_transform!=observation_transform:
            raise ValueError('Canonical source requires the same observation-label coordinates')
        entry=self.state['axes'][axis]
        pending=self.state.get('pending_operator_request')
        if not pending or pending['id']!=identity:
            pending={'id':identity,'new_segment':entry['segment']+1,'request':request}
            self.state['pending_operator_request']=pending;self.persist()
        if entry['segment']<pending['new_segment']:
            self.stop_axis(axis);self.collect(axis)
            entry.setdefault('segment_history',[]).append(dict(output=str(self.segment_path(axis)),
                training_steps=entry.get('training_steps',0),reason=request['reason']))
            entry.update(segment=pending['new_segment'],source_checkpoint=request['source_checkpoint'],
                profile=profile.dictionary(),initial_target=request['initial_target'],exploration=exploration,crashes=0,
                reward_transform=request.get('reward_transform'))
            for field in ('retention_override','retention_scale','prefix_source','prefix_steps','prefix_probability','prefix_sha256','rollout_steps','gae_lambda','observation_transform','physics_timesteps','use_sde','sde_sample_freq'):
                if field in request:entry[field]=request[field]
        self.state.setdefault('applied_operator_requests',[]).append(identity)
        self.state['pending_operator_request']=None
        atomic_json(self.root/'operator_reviews'/f'{identity}.json',dict(applied_at=datetime.now(timezone.utc).isoformat(),
            request=request,new_segment=entry['segment'],formal_review_schedule_unchanged=self.goal['next_review_at']))
        self.persist()
        if spawn:self.spawn(axis)
    def snapshot_review(self,index):
        for axis in self.axes:self.stop_axis(axis);self.collect(axis)
        report=dict(review_index=index,scheduled_at=self.goal['next_review_at'],started_at=datetime.now(timezone.utc).isoformat(),axes={})
        for axis in self.axes:
            entry=self.state['axes'][axis]
            if entry['verified']:
                report['axes'][axis]=dict(verified=True,action='keep_verified_checkpoint');continue
            status=entry['live'];stage=status['stage'];profile=Profile(**entry['profile'])
            checkpoint=(self.segment_path(axis)/'latest').resolve();model,_=load(checkpoint)
            seed=self.goal.get('review_seed_base',2000000)+index*100
            current=evaluate(model,self.reference,axis,profile,stage,20,seed,self.deadline,lambda:self.stop)
            final=current if stage==90. else evaluate(model,self.reference,axis,profile,90.,10,seed,self.deadline,lambda:self.stop)
            best_entry=entry['best_by_target'].get(str(int(stage)));best=None
            if best_entry:
                best_model,_=load(best_entry['path'])
                best=evaluate(best_model,self.reference,axis,profile,stage,20,seed,self.deadline,lambda:self.stop)
            if final['complete'] and final['success_rate']>=.9:
                final_seed=self.goal.get('review_verification_seed_base',90000000)+('xy'.index(axis))*10000000+index*1000
                independent=evaluate(model,self.reference,axis,profile,90.,100,final_seed,self.deadline,lambda:self.stop)
                if final_pass(independent):
                    entry['verified']=True;entry['verified_checkpoint']=str(checkpoint)
                    link(self.root/axis/'verified',checkpoint)
                    atomic_json(self.root/'reviews'/f'{index:03d}_{axis}_verified.json',independent)
            decision=choose(profile,current,best,entry['history'],stage)
            record=dict(stage=stage,current=current,best=best,final90=final,decision=decision,checkpoint=str(checkpoint),verified=entry['verified'])
            report['axes'][axis]=record;entry['history'].append(compact(current))
            if not current['complete']:continue
            if decision['action']!='continue' and not entry['verified']:
                source=best_entry['path'] if decision['action']=='rollback_best' and best_entry else str(checkpoint)
                entry.update(segment=entry['segment']+1,source_checkpoint=source,profile=decision['profile'],
                    initial_target=decision['target'],crashes=0)
        report['finished_at']=datetime.now(timezone.utc).isoformat()
        atomic_json(self.root/'reviews'/f'{index:03d}_review.json',report)
        self.write_review_text(index,report)
        self.state['last_review_index']=index;self.state['reviews'].append(str(self.root/'reviews'/f'{index:03d}_review.json'))
        self.goal['next_review_at']=next_review(self.goal['started_at'],datetime.now(timezone.utc)).isoformat()
        self.persist()
    def write_review_text(self,index,report):
        observed=datetime.fromisoformat(report['started_at']).astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        lines=[f'# {self.goal.get("hours",60):g}小时训练：第{index}次复查',f'北京时间：{observed}',
            '最终成功：从原始初握出发，有向+90°±5°、完整姿态误差≤5°、五指实际受力、连续稳定保持1秒。',
            '', '| 轴 | 当前课程 | 当前课程成功率 | 90°成功率 | 决定 |','|---|---:|---:|---:|---|']
        for axis,r in report['axes'].items():
            if 'current' in r:lines.append(f'| {axis.upper()} | {r["stage"]} | {r["current"]["success_rate"]:.1%} | {r["final90"]["success_rate"]:.1%} | {r["decision"]["action"]}: {r["decision"]["reason"]} |')
            else:lines.append(f'| {axis.upper()} | 90 | 100% verified | 100% verified | preserve |')
        lines+=['','最终要求：每轴在1ms、2ms两组各100个新初态中全部通过。较小角度或瞬时转过目标不算最终完成。']
        (self.root/'reviews'/f'{index:03d}_review.md').write_text('\n'.join(lines)+'\n')
        link(self.root/'reviews/latest.md',self.root/'reviews'/f'{index:03d}_review.md')
    def run(self):
        self.initialize();self.reference=self.goal['reference']
        if self.state.get('goal_complete'):
            self.state['status']='complete';self.persist();return
        if time.time()>=self.deadline:
            self.state['status']='budget_expired_unverified';self.persist();return
        self.prepare_retention();self.stop_previous()
        if self.goal['next_review_at']==self.goal['started_at']:
            self.goal['next_review_at']=(datetime.fromisoformat(self.goal['started_at'])+timedelta(hours=3)).isoformat()
        self.state['status']='running';self.persist()
        self.apply_operator_request(spawn=False)
        # After a long executor interruption, review saved checkpoints before
        # starting workers which would otherwise be stopped immediately.
        if time.time()<datetime.fromisoformat(self.goal['next_review_at']).timestamp():
            for axis in self.axes:self.spawn(axis)
        try:
            while not self.stop and time.time()<self.deadline:
                self.apply_operator_request()
                for axis in self.axes:
                    entry=self.state['axes'][axis];status=self.collect(axis)
                    child=self.children.get(axis)
                    if child is not None:child.poll()
                    if not entry['verified'] and not alive(entry.get('pid'),str(self.job_path(axis))):
                        entry['crashes']+=1
                        if entry['crashes']<=3:self.spawn(axis)
                        else:
                            self.state['status']='needs_attention';entry['restart_blocked']=True
                if all(self.state['axes'][a]['verified'] for a in self.axes):
                    self.state['goal_complete']=self.goal['goal_complete']=True;self.state['status']='complete';break
                now=datetime.now(timezone.utc)
                if now.timestamp()>=datetime.fromisoformat(self.goal['next_review_at']).timestamp():
                    index=int((now-datetime.fromisoformat(self.goal['started_at'])).total_seconds()//10800)
                    # Missing intervals are logged as missing, never fabricated.
                    for missed in range(self.state['last_review_index']+1,index):
                        atomic_json(self.root/'reviews'/f'{missed:03d}_missed.json',dict(status='not_observed',reason='controller unavailable or review delayed'))
                    self.goal['next_review_at']=(datetime.fromisoformat(self.goal['started_at'])+timedelta(hours=3*index)).isoformat()
                    self.snapshot_review(index)
                    for axis in self.axes:self.spawn(axis)
                self.persist();time.sleep(10.)
            if not self.state['goal_complete']:
                self.state['status']='interrupted' if self.stop else 'budget_expired_unverified'
        finally:
            for axis in self.axes:self.stop_axis(axis);self.collect(axis)
            self.persist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--goal',required=True);args=p.parse_args()
    root=Path(args.goal).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Controller already running; did not launch a duplicate')
        controller=Controller(root)
        try:controller.run()
        except BaseException as error:
            controller.state.update(status='error',error=repr(error));controller.persist();raise


if __name__=='__main__':main()
