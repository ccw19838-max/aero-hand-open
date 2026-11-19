"""Low-duty capture search. Every candidate starts with the real initial grasp."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,time,json
import numpy as np
from .env import AdaptiveEnv
from .config import Profile
from .prefix_curriculum import read_actions
from .storage import atomic_json,digest,implementation_digest


def search(goal,seconds=900.,population=16,generations=100):
    root=Path(goal);state=json.loads((root/'controller_state.json').read_text());profile=Profile(**state['axes']['x']['profile'])
    output=root/'capture_search_v1';output.mkdir(exist_ok=True)
    prefix_path=root/'mpc_diagnosis/x10_seed0/actual_trajectory.npz';prefix=read_actions(prefix_path,28)
    deadline=min(time.time()+seconds,datetime.fromisoformat(state['deadline_at']).timestamp())
    rng=np.random.default_rng(137);env=AdaptiveEnv(root/'sources/x/reference.npz','x',physical_dt=.002,target_degrees=10.,profile=profile)
    def sequence(parameters):
        return np.r_[prefix,np.repeat(parameters[:7][None,:],5,axis=0),
            np.repeat(parameters[7:][None,:],5,axis=0),np.zeros((50,7))]
    def rollout(episode_env,commands,record=False):
        obs,info=episode_env.reset(seed=0);observations=[];used=[];interrupted=False
        for action in commands:
            if time.time()>=deadline:interrupted=True;break
            if record:observations.append(obs.copy());used.append(action.copy())
            obs,_,done,_,info=episode_env.step(action)
            if done:break
        score=-2.*info['best_angle_error_degrees']-.5*info['angle_error_degrees']+40.*info['max_stable_seconds']
        score+=5.*info['five_finger_duty']+2.*info['contact_coverage']-10.*info['displacement']/.03
        if info['failure_reason']:score-=20.
        if info['success'] and not interrupted:score+=10000.
        return dict(score=float(score),success=bool(info['success'] and not interrupted),interrupted=interrupted,final=info,
            observations=np.asarray(observations,dtype=np.float32),actions=np.asarray(used,dtype=np.float32))
    def summary(result):return {k:v for k,v in result.items() if k not in ('observations','actions')}
    mean=np.zeros(14);sigma=np.r_[np.full(7,.2),np.full(7,.15)];best=None;best_parameters=None;attempts=0;qualified=[]
    try:
        for generation in range(generations):
            if time.time()>=deadline:break
            candidates=np.clip(rng.normal(mean,sigma,(population,14)),-1.,1.)
            candidates[0]=best_parameters if best_parameters is not None else mean
            candidates[-2:]=rng.uniform(-.7,.7,(2,14));scored=[]
            for parameters in candidates:
                if time.time()>=deadline:break
                cpu_before=time.process_time();commands=sequence(parameters);result=rollout(env,commands);attempts+=1
                scored.append((result['score'],parameters.copy()))
                if best is None or result['score']>best['score']:
                    best=summary(result);best_parameters=parameters.copy()
                    np.savez_compressed(output/'best_candidate.npz',actions=commands,parameters=parameters)
                    atomic_json(output/'best_candidate.json',dict(**best,attempts=attempts,generation=generation,
                        controller='actual-prefix plus two capture pulses and fixed-target hold; not trained policy',qualified=False))
                if result['success']:
                    checks=[]
                    for dt in (.001,.002):
                        replay=AdaptiveEnv(root/'sources/x/reference.npz','x',physical_dt=dt,target_degrees=10.,profile=profile)
                        try:checks.append((dt,rollout(replay,commands,record=True)))
                        finally:replay.close()
                    atomic_json(output/f'dual_check_{attempts:05d}.json',[dict(dt=dt,**summary(value)) for dt,value in checks])
                    if all(value['success'] for _,value in checks):
                        observations=np.concatenate([v['observations'] for _,v in checks]);actions=np.concatenate([v['actions'] for _,v in checks])
                        path=output/'qualified_hybrid_x10.npz';np.savez_compressed(path,observations=observations,actions=actions)
                        atomic_json(path.with_suffix('.json'),dict(axis='x',goal_degrees=10.,
                            controller='actual-prefix plus searched capture commands; not trained policy',neural_policy_from_initial_grasp=False,
                            actual_trajectory_from_original_initial_grasp=True,source_prefix_sha256=digest(prefix_path),
                            implementation_sha256=implementation_digest(),sha256=digest(path),
                            checks=[dict(dt=dt,seed=0,success=True,stable_seconds=v['final']['stable_seconds']) for dt,v in checks],
                            note='One seed and one open-loop hybrid sequence. No90deg or trained-policy success claim.'))
                        qualified.append(str(path));break
                # Deliberately average roughly0.2CPU for this extra diagnostic;
                # the four PPO simulation workers retain their allocation.
                spent=time.process_time()-cpu_before
                time.sleep(min(max(0.,deadline-time.time()),spent*4.))
            if scored:
                elites=np.array([p for _,p in sorted(scored,key=lambda v:v[0],reverse=True)[:max(3,population//4)]])
                mean=.7*elites.mean(0)+.3*mean;sigma=np.maximum(.035,.7*elites.std(0)+.3*sigma)
            status=dict(observed_at=datetime.now(timezone.utc).isoformat(),phase='searching',attempts=attempts,
                generation=generation,best=best,qualified_demonstrations=qualified,deadline_epoch=deadline)
            atomic_json(output/'status.json',status)
            if qualified:break
        status.update(phase='qualified_small_angle_demonstration' if qualified else 'no_qualified_demonstration',
            observed_at=datetime.now(timezone.utc).isoformat(),attempts=attempts)
        atomic_json(output/'status.json',status);return status
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--goal',required=True);p.add_argument('--seconds',type=float,default=900.)
    args=p.parse_args();print(json.dumps(search(args.goal,args.seconds),ensure_ascii=False))
