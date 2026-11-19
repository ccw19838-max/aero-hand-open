from dataclasses import replace
from pathlib import Path
import numpy as np
from experiments.cube_adaptive_60h.env import AdaptiveEnv,AdaptiveMonitor
from experiments.cube_adaptive_60h.fast_contacts import FastContactEnv
from experiments.cube_contact_ppo.task import force_limits,force_state
from experiments.cube_official_baseline.env import TIP_NAMES

VERSION='x_contact_recovery_v2'
FINGERS=('thumb','index','middle','ring','pinky')
TIP_ORDER=('th_tip','if_tip','mf_tip','rf_tip','pf_tip')
ROOTS=('right_t_link','right_index_f_link','right_middle_f_link','right_ring_f_link','right_pinky_f_link')


class ContactSemantics:
    def __init__(self,*args,prefix_bank=None,**kwargs):
        super().__init__(*args,**kwargs)
        if self.fixed_axis!='x':raise ValueError('This experiment trains X only')
        self.config=replace(self.config,version=VERSION)
        roots={self.model.body(name).id:i for i,name in enumerate(ROOTS)}
        mapping=np.full(self.model.ngeom,-1,dtype=np.int32)
        for geom in range(self.model.ngeom):
            body=int(self.model.geom_bodyid[geom])
            while body>0:
                if body in roots:mapping[geom]=roots[body];break
                body=int(self.model.body_parentid[body])
        self.ids['geom_finger'][:]=mapping
        if hasattr(self,'_finger_map'):self._finger_map[:]=mapping
        self.distance_order=np.array([TIP_NAMES.index(name) for name in TIP_ORDER])
        self.bank=[]
        if prefix_bank is not None:
            with np.load(Path(prefix_bank),allow_pickle=False) as data:
                self.bank=[data[key].astype(np.float32) for key in sorted(data.files)]
            if not self.bank or any(a.ndim!=2 or a.shape[1]!=7 or not 6<=len(a)<=20 or
                    not np.isfinite(a).all() or np.max(np.abs(a))>1. for a in self.bank):
                raise ValueError('Invalid physically reached prefix bank')
        self.auxiliary_requested=False;self.prefix_used=False;self.prefix_index=None
        self.prefix_failure='';self.prefix_start_angle=0.;self.recovery_anchor=None
    def _refresh(self):
        super()._refresh()
        self.tip_distances=self.tip_distances[self.distance_order]
        self.contacts['tip_distances']=self.tip_distances.copy()
    def control_mode(self):
        if self.warming:return 'setup'
        m=self.monitor;forces=self.contacts['finger_forces'];lower,upper=force_limits(self.config)
        missing=getattr(m,'missing_seconds',np.zeros(5))
        support,five,_=force_state(forces,self.config)
        if m.target_degrees==0.:return 'hold' if five else 'recover'
        if np.any(missing>.10) or not support:return 'recover'
        if m.stable>0.:return 'hold'
        if np.any(forces>upper) or m.error<np.deg2rad(20.):return 'capture'
        return 'drive'
    def _observation(self,history):
        obs=super()._observation(history)
        mode=self.control_mode();obs[125:129]=np.array([mode=='setup',mode=='drive',mode in ('recover','capture'),mode=='hold'])
        return obs
    def _info(self):
        info=super()._info()
        target=getattr(self.monitor,'target_degrees',self.target_degrees)
        info.update(environment_version=VERSION,control_mode=self.control_mode(),
            auxiliary_requested=self.auxiliary_requested,training_prefix_used=self.prefix_used,
            prefix_index=self.prefix_index,auxiliary_setup_failure=self.prefix_failure,
            prefix_start_angle_degrees=self.prefix_start_angle,
            evaluation_kind='contact_recovery' if target==0. else 'original_initial_x90',
            from_original_initial_grasp=bool(target==90. and not self.prefix_used))
        return info
    def set_task(self,target_degrees,randomization=0.):
        if target_degrees not in (0.,90.):raise ValueError('Only contact recovery and original X90')
        super().set_task(target_degrees,randomization)
    def reset(self,*,seed=None,options=None):
        self.auxiliary_requested=False;self.prefix_used=False;self.prefix_index=None
        self.prefix_failure='';self.prefix_start_angle=0.;self.recovery_anchor=None
        obs,info=super().reset(seed=seed,options=options)
        # Formal X90 never uses a prefix. During recovery training retain 20%
        # original-grasp holds; validation always tests actual missing contact.
        requested=bool(self.bank and self.monitor.target_degrees==0. and
                       (not self.training_mode or self.np_random.random()>=.2))
        if requested:
            self.auxiliary_requested=True
            self.prefix_index=int(seed%len(self.bank)) if seed is not None else int(self.np_random.integers(len(self.bank)))
            for action in self.bank[self.prefix_index]:
                obs,_,done,_,info=super().step(action)
                if done:
                    self.prefix_failure=info['failure_reason'] or 'prefix_ended_episode';break
            lower,_=force_limits(self.config)
            if not self.prefix_failure and not np.any(self.contacts['finger_forces']<lower):
                self.prefix_failure='prefix_has_no_missing_finger'
            if self.prefix_failure:
                failure=self.prefix_failure;index=self.prefix_index
                obs,info=super().reset(seed=seed,options=options)
                self.auxiliary_requested=True;self.prefix_failure=failure;self.prefix_index=index
            else:
                self.prefix_used=True;self.prefix_start_angle=float(np.rad2deg(self.monitor.progress))
                # No physical state or actuator target is changed. Only the
                # auxiliary goal becomes "regrasp at this reached orientation".
                self.monitor=AdaptiveMonitor(self.data.qpos[19:23].copy(),'x',self.basis,self.config,0.,profile=self.profile)
                self.warming=False;self.grip_stable_seconds=0.
        self.recovery_anchor=self.data.xpos[self.ids['cube_body']].copy()
        obs=self._observation(self.last_action);self.last_observation=obs.copy()
        return obs,self._info()


class NativeEnv(ContactSemantics,AdaptiveEnv):pass
class FastEnv(ContactSemantics,FastContactEnv):pass
