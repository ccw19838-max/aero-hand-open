"""57-mm locked cube, 20-degree mounting, absolute control and true XYZ90 goals."""
from pathlib import Path
import hashlib
import json
import gymnasium as gym
import mujoco
import numpy as np

from experiments.cube_envelop.model import (build_model,model_fingerprint,model_ids,
    native_contacts,native_cube_velocity,fixed_axis_basis)
from experiments.cube_official_baseline.env import SCALE,SENSOR_NAMES,TIP_NAMES
from .task import (Config,GoalMonitor,AXES,SYMMETRIES,aligned_cube_basis,canonical_quaternion,
                   matrix_quaternion,quaternion_multiply,rotation_matrix)


class QuarterTurnEnv(gym.Env):
    metadata={'render_modes':['rgb_array'],'render_fps':20}
    def __init__(self,reference,axis=None,render_mode=None,config=None):
        super().__init__();self.config=config or Config();self.reference_path=Path(reference).resolve()
        report=json.loads(self.reference_path.with_name('prepare_report.json').read_text())
        if report.get('approved') is not True or report['model_fingerprint']!=model_fingerprint(self.config):
            raise ValueError('Approved reference with identical 57-mm physics required')
        if report.get('reference_sha256')!=hashlib.sha256(self.reference_path.read_bytes()).hexdigest():
            raise ValueError('Reference checksum mismatch')
        with np.load(self.reference_path,allow_pickle=False) as a:self.reference={k:a[k].copy() for k in a.files}
        self.model=build_model(self.config);self.data=mujoco.MjData(self.model);self.ids=model_ids(self.model)
        self.mount_basis=fixed_axis_basis(self.model)
        self.sensor_indices=np.array([int(self.model.sensor(n).adr[0]) for n in SENSOR_NAMES])
        self.grasp_site=self.model.site('grasp_site').id
        self.tip_sites=[self.model.site(n).id for n in TIP_NAMES]
        self.palm_geoms=np.array([self.model.body(int(b)).name=='palm' for b in self.model.geom_bodyid])
        self.fixed_axis=axis
        if axis is not None and axis not in AXES:raise ValueError('Invalid axis')
        self.action_space=gym.spaces.Box(-1.,1.,(7,),dtype=np.float32)
        self.observation_space=gym.spaces.Dict({'state':gym.spaces.Box(-np.inf,np.inf,(14,),dtype=np.float32),
            'privileged_state':gym.spaces.Box(-np.inf,np.inf,(81,),dtype=np.float32),
            'goal_state':gym.spaces.Box(-np.inf,np.inf,(31,),dtype=np.float32)})
        self.render_mode=render_mode;self.renderer=None;self.camera=mujoco.MjvCamera()
        self.camera.lookat[:]=[.1,0.,.19];self.camera.distance=.36;self.camera.azimuth=135.;self.camera.elevation=-35.
        self.default=self.reference['ctrl'].copy();self.last_action=np.zeros(7);self.returned_history=np.zeros(7)
        self._has_reset=False;self.success_bank=[]

    def _contacts(self):
        contacts=native_contacts(self.model,self.data,self.ids);palm_force=0.;force=np.zeros(6)
        for i,c in enumerate(self.data.contact):
            g1,g2=int(c.geom1),int(c.geom2)
            other=g2 if g1==self.ids['cube_geom'] else g1 if g2==self.ids['cube_geom'] else None
            if other is not None and self.palm_geoms[other]:
                mujoco.mj_contactForce(self.model,self.data,i,force);palm_force+=max(0.,float(force[0]))
        contacts['hand_force']=float(sum(contacts['finger_forces'])+palm_force)
        contacts['forbidden_support']=self._forbidden()
        return contacts

    def _refresh(self):
        self.contacts=self._contacts();self.linear,self.angular=native_cube_velocity(self.model,self.data,self.ids)
        self.displacement=float(np.linalg.norm(self.data.xpos[self.ids['cube_body']]-self.initial_position))

    def _forbidden(self):
        if not self.data.ncon:return False
        pairs=self.data.contact.geom
        cube=self.ids['cube_geom'];mask=(pairs[:,0]==cube)|(pairs[:,1]==cube)
        if not mask.any():return False
        indices=np.flatnonzero(mask);force=np.zeros(6)
        for i in indices:
            a,b=map(int,pairs[i]);other=b if a==cube else a
            if self.ids['geom_forbidden'][other]:
                mujoco.mj_contactForce(self.model,self.data,int(i),force)
                if force[0]>=self.config.contact_force:return True
        return False

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);options=options or {};c=self.config
        axis=options.get('axis',self.fixed_axis) or str(self.np_random.choice(AXES))
        if axis not in AXES:raise ValueError('Invalid axis')
        reference_only=options.get('reference',False)
        bank=(self.success_bank[int(self.np_random.integers(len(self.success_bank)))]
              if self.success_bank and not reference_only and options.get('allow_bank',True)
              and self.np_random.random()<.25 else None)
        for _ in range(32):
            mujoco.mj_resetData(self.model,self.data)
            self.data.qpos[:]=self.reference['qpos'];self.data.qvel[:]=0.;self.data.ctrl[:]=self.default
            if bank:
                self.data.qpos[:]=bank['qpos'];self.data.qvel[:]=bank['qvel'];self.data.ctrl[:]=bank['ctrl']
            elif not reference_only:
                symmetry=SYMMETRIES[int(self.np_random.integers(24))]
                direction=self.np_random.normal(size=3);direction/=np.linalg.norm(direction)
                angle=np.deg2rad(self.np_random.uniform(-c.orientation_noise_degrees,c.orientation_noise_degrees))
                noise=np.r_[np.cos(angle/2),direction*np.sin(angle/2)]
                self.data.qpos[19:23]=quaternion_multiply(noise,quaternion_multiply(self.reference['qpos'][19:23],matrix_quaternion(symmetry)))
                self.data.qpos[16:19]+=self.np_random.uniform(-c.position_noise,c.position_noise,3)
            mujoco.mj_forward(self.model,self.data);contacts=self._contacts()
            if contacts['max_penetration']<=c.max_penetration and not contacts['forbidden_support']:break
        else:raise RuntimeError('No legal randomized reference reset')
        self.initial_position=self.data.xpos[self.ids['cube_body']].copy()
        self.elapsed=0.;self.warming=bank is None
        self.last_action[:]=bank['action'] if bank else 0.
        self.returned_history[:]=bank['history'] if bank else 0.
        self.reset_mode='successful_terminal_bank' if bank else 'prepared_reference_24_orientations'
        self.basis=bank['basis'].copy() if bank else aligned_cube_basis(self.data.qpos[19:23],self.mount_basis)
        self.monitor=GoalMonitor(self.data.qpos[19:23].copy(),axis,self.basis,c,
                                 ideal_initial=bank['ideal'] if bank else None)
        self._has_reset=True;self._refresh()
        self.last_observation=self._observation(self.returned_history)
        return self.last_observation,self._info()

    def _observation(self,history):
        base=np.r_[self.data.sensordata[self.sensor_indices],history].astype(np.float32)
        grasp=self.data.site_xpos[self.grasp_site];rotation=self.data.site_xmat[self.grasp_site].reshape(3,3)
        tips=np.concatenate([rotation.T@(self.data.site_xpos[s]-grasp) for s in self.tip_sites])
        quat=canonical_quaternion(self.data.qpos[19:23]);position=self.data.xpos[self.ids['cube_body']]
        privileged=np.r_[base,self.data.qpos[:16],self.data.qvel[:16],self.data.actuator_force,tips,
            grasp-position,quat,self.angular,self.linear].astype(np.float32)
        goal=np.r_[(position-grasp)/.1,quat,self.linear/.1,self.angular/5.,self.monitor.goal,
            self.monitor.error_vector/np.pi,self.monitor.direction,self.monitor.progress/np.pi,
            self.monitor.stable/self.config.hold_seconds,self.monitor.elapsed/self.config.episode_seconds,
            (self.contacts['finger_forces']>=self.config.contact_force).astype(float)].astype(np.float32)
        return {'state':base,'privileged_state':privileged,'goal_state':goal}

    def _info(self):
        m=self.monitor
        return dict(axis=m.axis,goal_degrees=90.,axis_frame='fixed_cube_face_axes_at_initial_grasp',
            axis_world_vector=m.direction.tolist(),goal_quaternion=m.goal.tolist(),
            phase='initial_reference_hold' if self.warming else 'recapture' if m.error<=np.deg2rad(5) else 'rotate',
            elapsed_seconds=self.elapsed,goal_elapsed_seconds=m.elapsed,success=bool(m.success),is_success=bool(m.success),
            failure_reason=m.failure,angle_error_degrees=float(np.rad2deg(m.error)),
            best_angle_error_degrees=float(np.rad2deg(m.best_error)),signed_progress_degrees=float(np.rad2deg(m.progress)),
            max_progress_degrees=float(np.rad2deg(m.max_progress)),stable_seconds=m.stable,max_stable_seconds=m.max_stable,
            finger_forces=self.contacts['finger_forces'].tolist(),finger_contacts=int(np.count_nonzero(self.contacts['finger_forces']>=self.config.contact_force)),
            hand_force=self.contacts['hand_force'],linear_speed=float(np.linalg.norm(self.linear)),
            angular_speed=float(np.linalg.norm(self.angular)),displacement=self.displacement,
            milestones=sorted(m.milestones),cube_layers_locked=True,reset_mode=self.reset_mode)

    def start_goal(self,axis):
        """Continue a successful chain without resetting the cube, hand or action history."""
        if not self.monitor.success:raise RuntimeError('Only a successful stable endpoint can start a chain')
        if axis not in AXES:raise ValueError('Invalid axis')
        previous_ideal=self.monitor.goal.copy()
        self.monitor=GoalMonitor(self.data.qpos[19:23].copy(),axis,self.basis,self.config,ideal_initial=previous_ideal)
        self.warming=False
        self.last_observation=self._observation(self.returned_history)
        return self.last_observation,self._info()

    def step(self,action):
        if not self._has_reset or self.monitor.done:raise RuntimeError('Reset or start a new goal before stepping')
        action=np.asarray(action,dtype=float)
        if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('Seven finite actions required')
        applied=np.zeros(7) if self.warming else np.clip(action,-1.,1.)
        self.data.ctrl[:]=np.clip(self.default+SCALE*applied,self.ids['ctrl_lower'],self.ids['ctrl_upper'])
        reward=0.;substeps=round(self.config.ctrl_dt/self.config.sim_dt)
        for substep in range(substeps):
            before=self.data.time;mujoco.mj_step(self.model,self.data);self.elapsed+=self.config.sim_dt
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or self.data.time<before:
                self.monitor.failure='numerical';reward-=100.;break
            if self._forbidden():self.monitor.failure='forbidden_support';reward-=100.;break
            if not self.warming:self.monitor.track(self.data.qpos[19:23])
            if (substep+1)%5==0:
                mujoco.mj_forward(self.model,self.data);self._refresh()
                if not self.warming:
                    reward+=self.monitor.checkpoint(self.data.qpos[19:23],self.contacts,self.linear,self.angular,self.displacement,.01)
                if self.monitor.done:break
        if self.warming and self.elapsed+1e-10>=self.config.warmup_seconds and not self.monitor.done:
            self.basis=aligned_cube_basis(self.data.qpos[19:23],self.mount_basis)
            self.monitor=GoalMonitor(self.data.qpos[19:23].copy(),self.monitor.axis,self.basis,self.config)
            self.warming=False
        reward-=.001*float(np.sum((applied-self.last_action)**2))
        self.returned_history=self.last_action.copy()
        obs=({k:v.copy() for k,v in self.last_observation.items()} if self.monitor.failure=='numerical'
             else self._observation(self.returned_history))
        self.last_observation=obs
        self.last_action=applied.copy();info=self._info()
        info.update(applied_action=applied.tolist(),motor_targets=self.data.ctrl.tolist())
        if self.monitor.success:
            self.success_bank.append(dict(qpos=self.data.qpos.copy(),qvel=self.data.qvel.copy(),
                ctrl=self.data.ctrl.copy(),basis=self.basis.copy(),ideal=self.monitor.goal.copy(),
                action=self.last_action.copy(),history=self.returned_history.copy()))
            self.success_bank=self.success_bank[-16:]
        return obs,float(reward),bool(self.monitor.done),False,info

    def render(self):
        if self.renderer is None:self.renderer=mujoco.Renderer(self.model,height=720,width=960)
        self.renderer.update_scene(self.data,camera=self.camera);return self.renderer.render()

    def close(self):
        if self.renderer:self.renderer.close();self.renderer=None
