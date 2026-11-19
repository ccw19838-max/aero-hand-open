"""Read-only native policy diagnosis with reconciled reward components.

All variants are isolated evaluations of pinned checkpoints. Live trainers,
models, reward definitions and optimizer states are never mutated.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
from pathlib import Path
from types import MethodType
import json
import time
import numpy as np
import torch
from experiments.cube_contact_ppo.env import ContactEnv
from experiments.cube_contact_ppo.task import force_state,force_limits
from experiments.cube_contact_ppo.storage import load,atomic_json

OUT=Path(__file__).resolve().parent


class AuditEnv(ContactEnv):
    def reset(self,**kwargs):
        result=super().reset(**kwargs);self.components=[];self.patched_monitor=None
        return result

    def install_monitor(self):
        m=self.monitor
        if m is self.patched_monitor:return
        original=m.checkpoint
        def checkpoint(this,q,contacts,linear,angular,displacement,dt=.01):
            before=(this.error,this.stable,this.last_coverage)
            value=original(q,contacts,linear,angular,displacement,dt)
            c=this.config;f=np.asarray(contacts['finger_forces']);support,five,coverage=force_state(f,c)
            valid=not contacts['forbidden_support'] and contacts['max_penetration']<=c.max_penetration and displacement<=c.max_displacement
            delta=(before[0]-this.error)/(np.pi/2);lower,upper=force_limits(c)
            speed=float(np.dot(angular,this.direction));perp=angular-speed*this.direction
            parts=dict(progress=20.*(delta if valid and support else min(delta,0.)),
                holding_potential=20.*(this.stable-before[1])/c.hold_seconds,
                coverage_potential=2.*(coverage-before[2]),time=-dt,
                pose=-2.*(this.error/(np.pi/2))**2*dt,coverage=.5*coverage*dt,
                perpendicular_speed=-.05*min(float(perp@perp),100.)*dt,
                linear_speed=-.1*min(float(np.linalg.norm(linear)/.1)**2,25.)*dt,
                force_excess=-.2*float(np.mean((np.maximum(f-upper,0.)/upper)**2))*dt,
                missing_fingers=(-float(np.mean(np.clip((lower-f)/lower,0.,1.)**2))*dt if this.phase in ('capture','hold') else 0.),
                goal_proximity=(2.*np.exp(-(this.error/np.deg2rad(15.))**2)*coverage**2*dt if valid and support else 0.),
                fingertip_proximity=(.25*float(np.mean(np.exp(-(np.maximum(contacts['tip_distances'],0.)/.02)**2)))*dt if valid and support else 0.),
                terminal=(c.success_bonus if this.success else -c.failure_penalty) if this.done else 0.)
            discrepancy=float(value-sum(parts.values()))
            if abs(discrepancy)>1e-8:raise RuntimeError('Reward audit mismatch '+str(discrepancy))
            self.components.append(dict(simulation_seconds=self.elapsed,warming=self.warming,**parts,discrepancy=discrepancy))
            return value
        m.checkpoint=MethodType(checkpoint,m);self.patched_monitor=m

    def step(self,action):
        self.install_monitor()
        return super().step(action)


def run(model,reference,mode,seed=1000000,dt=.001):
    env=AuditEnv(reference,'x',physical_dt=dt,target_degrees=30.)
    trace=[];rewards=[];parts=[]
    try:
        obs,info=env.reset(seed=seed)
        for _ in range(250):
            if mode=='neutral':action=np.zeros(7)
            elif mode.startswith('pulse_'):
                action=np.zeros(7);action[1]=.25 if mode=='pulse_positive' else -.25
                if env.elapsed>=.75:break
            else:
                action=model.predict(obs,deterministic=mode!='stochastic')[0]
                if mode=='middle_guard' and env.contacts['finger_forces'][2]<.5:
                    action[1]=min(float(action[1]),-.2)
            start=len(env.components);previous_action=env.last_action.copy()
            obs,reward,done,_,info=env.step(action)
            new=[c for c in env.components[start:] if not c['warming']]
            step_parts={k:sum(p[k] for p in new) for k in new[0] if k not in ('simulation_seconds','discrepancy','warming')} if new else {}
            step_parts['action_smoothness']=-.01*float(np.mean((np.asarray(info['applied_action'])-previous_action)**2))
            residual=reward-sum(step_parts.values())
            if abs(residual)>1e-8:
                if info['failure_reason'] not in ('forbidden_support','numerical') or abs(residual+env.config.failure_penalty)>1e-8:
                    raise RuntimeError('Control-cycle reward mismatch '+str(residual))
                step_parts['external_failure']=residual
            trace.append({**info,'normalized_motor_targets':((env.data.ctrl-env.ids['ctrl_lower'])/(env.ids['ctrl_upper']-env.ids['ctrl_lower'])).tolist(),
                'tip_distances':env.tip_distances.tolist(),'reward':reward,'components':step_parts})
            rewards.append(reward);parts.append(step_parts)
            if done:break
        f=np.array([t['finger_forces'] for t in trace]);a=np.array([t['applied_action'] for t in trace]);norm=np.array([t['normalized_motor_targets'] for t in trace])
        positive=[t for t in trace if t['phase']!='setup']
        totals={k:sum(v.get(k,0.) for v in parts) for k in {key for p in parts for key in p}}
        discounted={k:sum(.99**i*p.get(k,0.) for i,p in enumerate(parts)) for k in totals}
        return dict(mode=mode,seed=seed,dt=dt,final=info,raw_return=sum(rewards),
            discounted_return=sum(.99**i*r for i,r in enumerate(rewards)),
            force_zero_fraction=(f<.02).mean(0).tolist(),mean_action=a.mean(0).tolist(),
            actuator_names=[env.model.actuator(i).name for i in range(7)],
            min_normalized_targets=norm.min(0).tolist(),max_normalized_targets=norm.max(0).tolist(),
            actuator_near_limit_fraction=((norm<.01)|(norm>.99)).mean(0).tolist(),
            missing_finger_penalty_active_fraction=sum(t['phase'] in ('capture','hold') for t in positive)/max(1,len(positive)),
            reward_totals=totals,discounted_reward_totals=discounted,reward_reconciliation_max_error=max((abs(c['discrepancy']) for c in env.components),default=0.),
            trace=trace)
    finally:env.close()


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model,meta=load(OUT/'pinned_x_latest');reference=OUT/'pinned_x_latest/reference.npz'
    cases=[]
    for mode in ('deterministic','neutral','pulse_positive','pulse_negative','middle_guard'):
        cases.append(run(model,reference,mode))
        r=cases[-1];print(json.dumps(dict(mode=mode,raw_return=r['raw_return'],discounted_return=r['discounted_return'],
            progress=r['final']['max_progress_degrees'],middle_force=r['final']['finger_forces'][2],
            failure=r['final']['failure_reason'],missing_penalty_active=r['missing_finger_penalty_active_fraction'])),flush=True)
    for seed in range(4):
        torch.manual_seed(500+seed)
        cases.append(run(model,reference,'stochastic',seed=1000000+seed))
    best,bmeta=load(OUT/'pinned_x_best_30')
    r=run(best,OUT/'pinned_x_best_30/reference.npz','deterministic');r['mode']='earlier_best';cases.append(r)
    log_std=model.policy.log_std.detach().numpy()
    result=dict(checkpoint_step=meta['training_steps'],source='pinned_x_latest',cases=cases,
        gaussian_std=np.exp(log_std).tolist(),source_learning_rate=float(model.learning_rate),
        nominal_max_delta_fraction=.02,discount=.99,live_training_unchanged=True,
        limitations='Small controlled evaluations establish action/contact effects and reward bookkeeping; they do not prove a new training recipe will solve the task.')
    atomic_json(OUT/'x_diagnosis.json',result)
    make_plot(result)


def make_plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='DejaVu Sans'
    fig,axes=plt.subplots(4,1,figsize=(10,11),sharex=True)
    for mode,label,color in [('deterministic','Current X policy','#d9534f'),('neutral','Neutral motor targets','#4c78a8'),('middle_guard','Isolated middle-finger guard','#59a14f')]:
        record=next(r for r in result['cases'] if r['mode']==mode);trace=record['trace'];t=[x['elapsed_seconds'] for x in trace]
        axes[0].plot(t,[x['signed_progress_degrees'] for x in trace],label=label,color=color)
        axes[1].plot(t,[x['finger_forces'][2] for x in trace],color=color)
        axes[2].plot(t,[x['normalized_motor_targets'][1] for x in trace],color=color)
        axes[3].plot(t,np.cumsum([.99**i*x['reward'] for i,x in enumerate(trace)]),color=color)
    axes[0].axhline(30,color='black',ls='--',lw=1,label='Goal: +30 degrees')
    axes[1].axhline(.1,color='black',ls='--',lw=1)
    axes[0].set_ylabel('Signed angle (deg)');axes[1].set_ylabel('Middle force (N)')
    axes[2].set_ylabel('Middle motor target\n(normalized range)');axes[3].set_ylabel('Discounted return');axes[3].set_xlabel('Simulation time (s)')
    for ax in axes:ax.grid(alpha=.25)
    axes[0].legend(loc='upper left',fontsize=9)
    fig.suptitle('X-axis diagnosis: pinned policy, unchanged physics and success criteria')
    fig.tight_layout();fig.savefig(OUT/'x_diagnosis.png',dpi=160);fig.savefig(OUT/'x_diagnosis.pdf');plt.close(fig)


if __name__=='__main__':main()
