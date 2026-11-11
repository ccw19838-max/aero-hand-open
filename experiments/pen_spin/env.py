"""One observation/action/trajectory contract for native MuJoCo and MJX.

The native backend is for diagnostics, not differentiable training. Both
backends invoke the identical task kernel at every physics substep.
"""
from pathlib import Path
import math
import numpy as np
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import support
from mujoco_playground._src import mjx_env

from .config import PenSpinConfig
from .model import build_model, model_ids, native_contacts, native_pen_velocity
from .trajectory import Sample, initialize_trace, advance_trace
from .grasp import advance_grasp_trace


SENSOR_NAMES = ('len_if','len_mf','len_rf','len_pf','len_th1','len_th2','len_th_abd')
SUBSTEP_FIELDS = ('time','theta','phi','phase','force_thumb','force_index','force_middle',
                  'force_ring','force_pinky','center_x','center_y','center_z',
                  'direction_x','direction_y','direction_z','contact_sector_0',
                  'contact_sector_1','contact_sector_2','contact_sector_3',
                  'stable_seconds','no_contact_seconds','done','success','reward')


def _substep_row(trace,sample,reward):
    stable = jp.where(trace.goal_angle > 0.,trace.stable_time,trace.initial_hold_time)
    return jp.concatenate([jp.array([trace.elapsed,trace.theta,trace.phi,trace.phase]),
                           sample.finger_forces,sample.center,sample.direction,
                           trace.sector_contact,jp.array([stable,trace.no_contact_time,
                                                         trace.done,trace.success,reward])]).astype(jp.float32)


def load_reference(path, model):
    """Only numerical state; never deserialize executable objects."""
    with np.load(Path(path), allow_pickle=False) as archive:
        ref = {k: archive[k].copy() for k in ('qpos','qvel','ctrl','c0','n0')}
    for key, shape in [('qpos',(model.nq,)),('qvel',(model.nv,)),
                       ('ctrl',(model.nu,)),('c0',(3,)),('n0',(3,))]:
        if ref[key].shape != shape or not np.isfinite(ref[key]).all():
            raise ValueError(f'Invalid reference {key}; expected {shape}')
    if not np.isclose(np.linalg.norm(ref['n0']),1,atol=1e-5):
        raise ValueError('Reference rotation axis must be normalized')
    if np.any(ref['ctrl'] < model.actuator_ctrlrange[:,0]-1e-8) or np.any(ref['ctrl'] > model.actuator_ctrlrange[:,1]+1e-8):
        raise ValueError('Reference control outside model limits')
    return ref


def _rot_quat(n, angle):
    return jp.concatenate([jp.cos(angle/2)[None], n*jp.sin(angle/2)])


def _quat_mul(a,b):
    return jp.concatenate([(a[0]*b[0]-jp.dot(a[1:],b[1:]))[None],
                           a[0]*b[1:]+b[0]*a[1:]+jp.cross(a[1:],b[1:])])


class PenSpinEnv(mjx_env.MjxEnv):
    def __init__(self, config, reference_path, goal_angle=0., backend='mjx'):
        self.config = config or PenSpinConfig()
        if backend not in ('native','mjx'):
            raise ValueError('backend must be native or mjx')
        if not (0 <= goal_angle <= 2*math.pi+1e-6):
            raise ValueError('Goal angle must be in [0, 2pi]')
        self.backend = backend
        self.goal_angle = float(goal_angle)
        if self.config.task_mode=='grasp' and goal_angle!=0:
            raise ValueError('Acquisition policy must pass holding before any rotation goal')
        self._task_advance = advance_grasp_trace if self.config.task_mode=='grasp' else advance_trace
        self._ctrl_dt, self._sim_dt = self.config.ctrl_dt, self.config.sim_dt
        self._mj_model = build_model(self.config)
        self.ids = model_ids(self._mj_model)
        self.reference_path = str(Path(reference_path).resolve())
        self.reference = load_reference(reference_path,self._mj_model)
        self._mjx_model = mjx.put_model(self._mj_model) if backend=='mjx' else None
        self._blank_data = mjx.make_data(self._mj_model) if backend=='mjx' else None
        self._ctrl_low = jp.array(self.ids['ctrl_lower'],dtype=jp.float32)
        self._ctrl_high = jp.array(self.ids['ctrl_upper'],dtype=jp.float32)
        # Joint-actuated thumb target should also respect its physical range.
        self._ctrl_low = self._ctrl_low.at[4].set(max(float(self._ctrl_low[4]),0.))
        self._ctrl_high = self._ctrl_high.at[4].set(min(float(self._ctrl_high[4]),1.7453))
        self._sensor_ids = np.array([self._mj_model.sensor(n).adr[0] for n in SENSOR_NAMES])
        self._pen_vel_adr = int(self._mj_model.sensor('pen_linvel').adr[0])
        self._pen_ang_adr = int(self._mj_model.sensor('pen_angvel').adr[0])
        self._quat_adr = int(self._mj_model.sensor('pen_orientation').adr[0])
        self._advance_native = jax.jit(lambda t,s: self._task_advance(t,s,self.sim_dt,self.config))
        self._initialize = jax.jit(lambda s: initialize_trace(s,jp.array(self.reference['n0']),
                                  jp.array(self.reference['c0']),self.goal_angle,self.config))

    @property
    def action_size(self): return 7
    @property
    def observation_size(self): return {'state':(54,), 'privileged_state':(93,)}
    @property
    def mj_model(self): return self._mj_model
    @property
    def mjx_model(self): return self._mjx_model
    @property
    def xml_path(self): return 'generated:aero-upright-weighted-pen-v1'
    @property
    def unwrapped(self): return self

    def _sample(self, data):
        ids = self.ids
        if self.backend=='native':
            contact = native_contacts(self.mj_model,data,ids)
            linear,angular = native_pen_velocity(self.mj_model,data,ids)
            forces = jp.array(contact['finger_forces'],dtype=jp.float32)
            invalid = jp.array(contact['invalid_support'])
            penetration = jp.array(contact['max_penetration'] > self.config.max_penetration)
        else:
            contact = data._impl.contact
            force,indices = support.contact_force_dim(self.mjx_model,data,3)
            geom = contact.geom[indices]
            is_pen = (geom[:,0]==ids['pen_geom']) | (geom[:,1]==ids['pen_geom'])
            other = jp.where(geom[:,0]==ids['pen_geom'],geom[:,1],geom[:,0])
            other = jp.clip(other,0,self.mj_model.ngeom-1)
            normals = jp.maximum(force[:,0],0.)*is_pen
            group = jp.array(ids['geom_finger'])[other]
            forces = jp.stack([jp.sum(jp.where(group==i,normals,0.)) for i in range(5)])
            invalid = jp.any(is_pen & jp.array(ids['geom_invalid'])[other] & (normals>1e-6))
            penetration = jp.any(is_pen & (contact.dist[indices] < -self.config.max_penetration))
            linear = data.sensordata[self._pen_vel_adr:self._pen_vel_adr+3]
            angular = data.sensordata[self._pen_ang_adr:self._pen_ang_adr+3]
        center = jp.array(data.site_xpos[ids['site_center']],dtype=jp.float32)
        a = jp.array(data.site_xpos[ids['site_a']],dtype=jp.float32)
        b = jp.array(data.site_xpos[ids['site_b']],dtype=jp.float32)
        thumb_anchor = jp.array(data.xanchor[ids['thumb_ip_joint']])
        thumb_tip = jp.array(data.site_xpos[ids['thumb_tip_site']])
        thumb = (thumb_anchor+thumb_tip)*.5
        thumb_direction = (thumb_tip-thumb_anchor)/jp.maximum(jp.linalg.norm(thumb_tip-thumb_anchor),1e-8)
        tips = jp.stack([jp.array(data.site_xpos[self.mj_model.site(name).id])
                         for name in ('th_tip','if_tip','mf_tip','rf_tip','pf_tip')])
        segment = a-b
        fractions = jp.clip(jp.sum((tips-b)*segment,axis=-1)/jp.maximum(jp.dot(segment,segment),1e-8),0.,1.)
        distances = jp.maximum(0.,jp.linalg.norm(tips-(b+fractions[:,None]*segment),axis=-1)-.005)
        return Sample(center=center,endpoint_a=a,direction=(a-b)/jp.maximum(jp.linalg.norm(a-b),1e-8),
                      linvel=jp.array(linear,dtype=jp.float32),angvel=jp.array(angular,dtype=jp.float32),
                      thumb_center=thumb,thumb_direction=thumb_direction,finger_forces=forces,
                      invalid_contact=invalid,penetration=penetration,finger_distances=distances)

    def observe(self,data,sample,trace,last_action,feedback_velocity):
        """54 versioned truth features; same order and units on both backends."""
        phase = jp.asarray(trace.phase,dtype=jp.int32)
        fields = [jp.asarray(data.sensordata)[self._sensor_ids],feedback_velocity,last_action,
                  sample.center-trace.c0,sample.direction,
                  jp.asarray(data.sensordata)[self._quat_adr:self._quat_adr+4],sample.linvel,sample.angvel,
                  sample.finger_forces,jp.array([self.goal_angle]),jax.nn.one_hot(phase,5),
                  jp.array([trace.theta/(2*jp.pi),trace.phi/(2*jp.pi),
                            trace.initial_hold_time if self.config.task_mode=='grasp' else trace.stable_time,
                            trace.no_contact_time,trace.violation_time,
                            jp.maximum(0.,self.config.episode_seconds-trace.elapsed)])]
        obs = jp.concatenate(fields).astype(jp.float32)
        privileged = jp.concatenate([obs,jp.asarray(data.qpos)[self.ids['hand_qpos_ids']],
                                    jp.asarray(data.qvel)[self.ids['hand_dof_ids']],
                                    jp.asarray(data.actuator_force)]).astype(jp.float32)
        return {'state':obs,'privileged_state':privileged}

    def _metrics(self,trace,task_metrics,sample,action_change=0.):
        stable = jp.where(self.goal_angle > 0., trace.stable_time, trace.initial_hold_time)
        finger_contact = jp.any(sample.finger_forces >= self.config.contact_force)
        dropped = ~finger_contact & (sample.center[2] < trace.c0[2]-self.config.max_center_distance)
        return {k:jp.asarray(v,dtype=jp.float32) for k,v in {
            'success':trace.success,'failure':trace.done & ~trace.success,'phase':trace.phase,
            'theta':trace.theta,'phi':trace.phi,'capture':trace.stable_time,
            'stable_seconds':stable,
            'contact_fingers':jp.sum(sample.finger_forces>=self.config.contact_force),
            'dropped':dropped,'pen_center_z':sample.center[2],'finger_contact':finger_contact,
            'tilt_degrees':task_metrics.get('plane_degrees',0.),
            'action_change_squared':action_change,'invalid_contact':sample.invalid_contact,
            'failure_reason':trace.failure_reason,
            'valid_progress_radians':task_metrics.get('valid_progress_radians',0.),
            'progress':jp.where(self.goal_angle>0,trace.potential/jp.maximum(self.goal_angle,1e-8),
                                stable/self.config.hold_seconds),
        }.items()}

    def reset(self,rng):
        rng,pk,ak = jax.random.split(rng,3)
        q = jp.array(self.reference['qpos'],dtype=jp.float32)
        p = self.ids['pen_qpos_adr']
        dp = jax.random.uniform(pk,(3,),minval=-1.,maxval=1.)*self.config.reset_position_noise
        angle = jax.random.uniform(ak,(),minval=-1.,maxval=1.)*jp.deg2rad(self.config.reset_angle_noise_degrees)
        q = q.at[p:p+3].add(dp)
        q = q.at[p+3:p+7].set(_quat_mul(_rot_quat(jp.array(self.reference['n0']),angle),q[p+3:p+7]))
        if self.backend=='mjx':
            data = self._blank_data.replace(qpos=q,qvel=jp.array(self.reference['qvel'],dtype=jp.float32),
                          ctrl=jp.array(self.reference['ctrl'],dtype=jp.float32))
            data = mjx.forward(self.mjx_model,data)
        else:
            data = mujoco.MjData(self.mj_model)
            data.qpos[:] = np.asarray(q);data.qvel[:] = self.reference['qvel'];data.ctrl[:] = self.reference['ctrl']
            mujoco.mj_forward(self.mj_model,data)
        sample = self._sample(data)
        trace = self._initialize(sample)
        last = jp.zeros(7,dtype=jp.float32)
        feedback = jp.asarray(data.sensordata)[self._sensor_ids]
        obs = self.observe(data,sample,trace,last,jp.zeros(7))
        return mjx_env.State(data,obs,jp.array(0.),jp.array(0.),self._metrics(trace,{},sample),
                             {'rng':rng,'trace':trace,'last_action':last,'last_feedback':feedback,
                              'motor_targets':jp.array(data.ctrl),
                              'substep_trace':jp.zeros((self.n_substeps,len(SUBSTEP_FIELDS)),dtype=jp.float32)})

    def step(self,state,action):
        action = jp.clip(jp.asarray(action,dtype=jp.float32),-1.,1.)
        targets = jp.clip(state.info['motor_targets']+self.config.action_delta_fraction*
                          (self._ctrl_high-self._ctrl_low)*action,self._ctrl_low,self._ctrl_high)
        change = jp.mean((action-state.info['last_action'])**2)
        if self.backend=='mjx':
            def substep(carry,_):
                data,trace,reward,valid_progress = carry
                data = jax.lax.cond(trace.done,lambda d:d,lambda d:mjx.step(self.mjx_model,d),data)
                sample = self._sample(data)
                new_trace,r,metrics = self._task_advance(trace,sample,self.sim_dt,self.config)
                valid_progress = jp.maximum(valid_progress,metrics['valid_progress_radians'])
                return (data,new_trace,reward+r,valid_progress),_substep_row(new_trace,sample,r)
            data = state.data.replace(ctrl=targets)
            (data,trace,reward,valid_progress),rows = jax.lax.scan(
                substep,(data,state.info['trace'],jp.array(0.),jp.array(0.)),None,length=self.n_substeps)
        else:
            data = state.data
            data.ctrl[:] = np.asarray(targets)
            trace = state.info['trace'];reward = jp.array(0.);valid_progress = jp.array(0.)
            substep_rows = []
            for _ in range(self.n_substeps):
                if bool(trace.done): break
                mujoco.mj_step(self.mj_model,data)
                physical_sample = self._sample(data)
                trace,r,submetrics = self._advance_native(trace,physical_sample)
                reward += r
                valid_progress = jp.maximum(valid_progress,submetrics['valid_progress_radians'])
                substep_rows.append(_substep_row(trace,physical_sample,r))
            padding = _substep_row(trace,self._sample(data),0.)
            substep_rows += [padding]*(self.n_substeps-len(substep_rows))
            rows = jp.stack(substep_rows)
        reward -= jp.where(state.info['trace'].done,0.,self.dt*.2*change)
        sample = self._sample(data)
        plane = jp.rad2deg(jp.arcsin(jp.clip(jp.abs(jp.dot(sample.direction,trace.n0)),0.,1.)))
        diagnostics = {'plane_degrees':plane,'valid_progress_radians':valid_progress}
        feedback = jp.asarray(data.sensordata)[self._sensor_ids]
        info = dict(state.info)
        info.update(trace=trace,last_action=action,last_feedback=feedback,motor_targets=targets,substep_trace=rows)
        return state.replace(data=data,obs=self.observe(data,sample,trace,action,(feedback-state.info['last_feedback'])/self.dt),
                             reward=reward,done=trace.done.astype(jp.float32),
                             metrics=self._metrics(trace,diagnostics,sample,change),info=info)


def make_env(config=None,reference_path=None,goal_angle=0.,backend='mjx'):
    if reference_path is None:
        raise ValueError('A validated reference.npz path is required')
    return PenSpinEnv(config or PenSpinConfig(),reference_path,goal_angle,backend)
