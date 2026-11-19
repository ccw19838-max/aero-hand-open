"""Search actuator-only releases from the approved grip; validate 2ms and 1ms."""
from dataclasses import replace
from pathlib import Path
import time
import mujoco
import numpy as np
from experiments.cube_quarter_turn.task import orientation_error
from .env import TeacherEnv
from .task import Config
from .storage import atomic_json,save_reference


def secure(info):
    forces=np.array(info['finger_forces'])
    return bool(forces[0]>=.02 and np.count_nonzero(forces[1:]>=.02)>=2)


def stable(info):
    return secure(info) and info['linear_speed']<.03 and info['angular_speed']<.5


def clearance(env):
    distances=[]
    for finger in range(1,5):
        ids=np.flatnonzero((env.ids['geom_finger']==finger)&((env.model.geom_contype!=0)|(env.model.geom_conaffinity!=0)))
        distances.append(min((mujoco.mj_geomDistance(env.model,env.data,int(g),env.ids['cube_geom'],.05,None)
                              for g in ids),default=.05))
    return np.array(distances)


def hold_check(reference,axis,dt,deadline,state=None):
    env=TeacherEnv(reference,axis,config=replace(Config(),stagnation_seconds=100.),physical_dt=dt)
    if state is not None:
        qpos,ctrl=state
        env.reference['qpos']=qpos.copy();env.reference['ctrl']=ctrl.copy();env.default=ctrl.copy()
    consecutive=0.;maximum=0.;worst_penetration=0.;result={}
    try:
        _,info=env.reset(seed=0,options={'reference':True})
        for _ in range(65):  # 0.25 setup + 3 seconds free-body hold
            if time.time()>=deadline:return {'approved':False,'reason':'deadline','dt':dt}
            _,_,done,_,info=env.step(np.zeros(7))
            worst_penetration=max(worst_penetration,env.contacts['max_penetration'])
            consecutive=info['grip_stable_seconds'] if not done else 0.
            maximum=max(maximum,consecutive)
            if done:break
        result=dict(approved=not done and consecutive>=.5,dt=dt,final_stable_seconds=consecutive,
            max_stable_seconds=maximum,worst_penetration_m=worst_penetration,final=info,
            final_cube_qpos=env.data.qpos[16:23].tolist())
    except RuntimeError as error:
        result=dict(approved=False,dt=dt,reason='illegal_reference_reset',error=str(error))
    finally:env.close()
    return result


def prepare(reference,axis,output,deadline,candidates=192,keep=3,seed=0,should_stop=lambda:False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    env=TeacherEnv(reference,axis,config=replace(Config(),stagnation_seconds=100.))
    rng=np.random.default_rng(seed);records=[];pool=[]
    # Diverse support pairs: no fixed assumption about which sign closes a tendon.
    pairs=[(3,4),(1,2),(1,4),(2,3),(1,3),(2,4)]
    try:
        for index in range(candidates):
            if time.time()>=deadline or should_stop():break
            _,info=env.reset(seed=0,options={'reference':True})
            support=pairs[(index+('xyz'.index(axis)*2))%len(pairs)]
            action=np.zeros(7)
            if index:
                action=rng.normal(0.,.12,7)
                for finger in range(1,5):
                    if finger not in support:action[finger-1]=rng.choice((-1.,1.))*rng.uniform(.15,.9)
                action=np.clip(action,-1.,1.)
            consecutive=0.;done=False
            for step in range(45):
                if time.time()>=deadline or should_stop():break
                # First five controls are the normal prepared-reference setup.
                applied=action*min(1.,max(0.,(step-4)/8.))
                _,_,done,_,info=env.step(applied)
                consecutive=info['grip_stable_seconds'] if not done else 0.
                if done:break
            accepted=not done and consecutive>=.5 and time.time()<deadline and not should_stop() \
                and env.contacts['max_penetration']<=env.config.max_penetration \
                and not env.contacts['forbidden_support']
            record=dict(candidate=index,requested_support=list(support),action=action.tolist(),
                stable_seconds=consecutive,accepted_for_validation=accepted,final=info)
            if accepted:
                gaps=clearance(env)
                # Prefer clearance of the two least-loaded non-thumb fingers,
                # with a modest force cost; stable holds remain a hard gate.
                spare=np.argsort(info['finger_forces'][1:])[:2]
                score=float(100.*np.sum(np.maximum(gaps[spare],0.))-.01*sum(info['finger_forces'])
                    -10.*info['linear_speed']-info['angular_speed'])
                record.update(score=score,finger_clearance_m=gaps.tolist(),spare_fingers=(spare+1).tolist())
                pool.append((score,index,env.data.qpos.copy(),env.data.ctrl.copy(),record))
            records.append(record)
            if index%8==0:
                atomic_json(output/'search_status.json',dict(phase='grip_screening',checked=index+1,
                    candidates=candidates,eligible=len(pool),axis=axis,deadline_epoch=deadline))
    finally:env.close()
    selected=[]
    for score,index,qpos,ctrl,record in sorted(pool,key=lambda item:item[0],reverse=True):
        if len(selected)>=keep or time.time()>=deadline or should_stop():break
        directory=output/f'candidate_{index:04d}'
        # Validate raw candidate states using the source's approved geometry.
        # No approved candidate file exists before BOTH checks finish.
        path=directory/'reference.npz'
        checks=[hold_check(reference,axis,dt,deadline,(qpos,ctrl)) for dt in (.002,.001)]
        valid=all(c['approved'] for c in checks)
        # Large timestep disagreement invalidates a seemingly stable reference.
        if valid:
            a,b=[np.asarray(c['final_cube_qpos']) for c in checks]
            valid=np.linalg.norm(a[:3]-b[:3])<.003 and orientation_error(a[3:],b[3:])[1]<np.deg2rad(5.)
        if valid:
            save_reference(directory,qpos,ctrl,checks,reference)
            selected.append(dict(reference=str(path.resolve()),score=score,candidate=index,
                spare_fingers=record['spare_fingers'],finger_clearance_m=record['finger_clearance_m']))
        else:
            atomic_json(directory/'prepare_report.json',{'approved':False,'checks':checks})
    report=dict(axis=axis,checked=len(records),candidates=candidates,selected=selected,records=records,
        complete=len(records)==candidates,physical_assistance=False,
        criterion='thumb plus two fingers, low speeds, final 0.5s of 3s hold, 1ms and 2ms; no forbidden support')
    atomic_json(output/'report.json',report)
    return report
