"""Native CPU adaptation of the repository's official AeroCubeRotateZAxis task.

Source task: sim_rl/mujoco_playground/.../aero_hand/rotate_z.py. Keep its 50 mm
cube, 69.2 g mass, original mount/20-degree palm tilt, world-Z reward, absolute
motor targets, random reset, sensor noise and two-period action-history timing.
Domain randomization and MJX/Brax's accelerator execution are not reproduced.
"""
from pathlib import Path
import hashlib
import json

import gymnasium as gym
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
TASK=ROOT/'sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand'
WEIGHTS=ROOT/'ros2/src/aero_hand_open_rl/aero_hand_open_rl/ppo_TetheriaCubeRotateZAxisTendon_20250926_152920'
SENSOR_NAMES=('len_if','len_mf','len_rf','len_pf','len_th1','len_th2','len_th_abd')
TIP_NAMES=('if_tip','mf_tip','rf_tip','pf_tip','th_tip')
SCALE=np.array([.02,.02,.02,.02,.7,.003,.012])


def build_model():
    assets={p.name:p.read_bytes() for p in (ROOT/'sim_rl/simulation/mujoco/assets').rglob('*') if p.is_file()}
    assets.update({p.name:p.read_bytes() for p in (TASK/'xmls').rglob('*') if p.is_file()})
    model=mujoco.MjModel.from_xml_string((TASK/'xmls/scene_mjx_cube.xml').read_text(),assets)
    model.opt.timestep=.01
    model.vis.global_.offwidth=960;model.vis.global_.offheight=720
    return model


def source_manifest():
    files=[TASK/'rotate_z.py',TASK/'aero_hand_constants.py',TASK/'base.py',
           TASK/'xmls/right_hand.xml',TASK/'xmls/scene_mjx_cube.xml',
           TASK/'xmls/reorientation_cube.xml',WEIGHTS]
    files.extend(p for p in (ROOT/'sim_rl/simulation/mujoco/assets').rglob('*') if p.is_file())
    files.extend(p for p in (TASK/'xmls').rglob('*') if p.is_file() and p not in files)
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def implementation_digest():
    digest=hashlib.sha256()
    for name in ('env.py','policy.py'):
        p=Path(__file__).with_name(name);digest.update(p.name.encode());digest.update(p.read_bytes())
    digest.update(json.dumps(source_manifest(),sort_keys=True).encode())
    return digest.hexdigest()


class OfficialRotateEnv(gym.Env):
    metadata={'render_modes':['rgb_array'],'render_fps':20}
    def __init__(self,noise_level=1.,render_mode=None):
        super().__init__()
        self.model=build_model();self.data=mujoco.MjData(self.model)
        self.noise_level=float(noise_level)
        if not np.isfinite(self.noise_level) or self.noise_level<0:raise ValueError('Invalid noise level')
        self.render_mode=render_mode;self.renderer=None
        self.action_space=gym.spaces.Box(-1.,1.,(7,),dtype=np.float32)
        self.observation_space=gym.spaces.Dict({'state':gym.spaces.Box(-np.inf,np.inf,(14,),dtype=np.float32),
            'privileged_state':gym.spaces.Box(-np.inf,np.inf,(81,),dtype=np.float32)})
        self.home=self.model.keyframe('home').qpos.copy();self.default=self.model.keyframe('home').ctrl.copy()
        self.sensor_indices=np.array([int(self.model.sensor(n).adr[0]) for n in SENSOR_NAMES])
        self.last_action=np.zeros(7);self.step_count=0;self.done=True
        self.cube=self.model.body('cube').id

    def sensor(self,name):return np.asarray(self.data.sensor(name).data).copy()

    def _observation(self,history):
        lengths=self.data.sensordata[self.sensor_indices[:6]].copy()
        joint=self.data.sensordata[self.sensor_indices[6:]].copy()
        lengths+=self.np_random.uniform(-1.,1.,6)*self.noise_level*.005
        joint+=self.np_random.uniform(-1.,1.,1)*self.noise_level*.05
        state=np.concatenate((lengths,joint,history)).astype(np.float32)
        privileged=np.concatenate((state,self.data.qpos[:16],self.data.qvel[:16],self.data.actuator_force,
            *[self.sensor(n+'_position') for n in TIP_NAMES],
            self.sensor('palm_position')-self.sensor('cube_position'),self.sensor('cube_orientation'),
            self.sensor('cube_angvel'),self.sensor('cube_linvel'))).astype(np.float32)
        return {'state':state,'privileged_state':privileged}

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        if not (options or {}).get('home',False):
            self.data.qpos[:16]=np.clip(self.home[:16]+.1*self.np_random.normal(size=16),
                self.model.jnt_range[:16,0],self.model.jnt_range[:16,1])
            self.data.qpos[16:19]=np.array([.1,0.,.05])+self.np_random.uniform(-.01,.01,3)
            q=self.np_random.normal(size=4);self.data.qpos[19:23]=q/np.linalg.norm(q)
        self.data.qvel[:]=0.;self.data.ctrl[:]=self.default;self.data.mocap_pos[:]=-100.
        mujoco.mj_forward(self.model,self.data)
        self.last_action[:]=0.;self.step_count=0;self.done=False;self.integrated_world_z=0.
        self.angular_xy_sum=0.;self.angular_z_sum=0.;self.return_sum=0.
        self.initial_up=self._body_up();self.max_tilt=0.;self.max_distance=0.
        return self._observation(np.zeros(7)),self._info()

    def _body_up(self):
        # Pure rotation around world Z leaves R^T*Z invariant, even if the cube
        # starts with an arbitrary tilt. This diagnoses off-axis tumbling.
        rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,self.data.qpos[19:23])
        return rotation.reshape(3,3).T@np.array([0.,0.,1.])

    def _info(self):
        position=self.sensor('cube_position');omega=self.sensor('cube_angvel')
        distance=float(np.linalg.norm(position-self.sensor('palm_position')))
        tilt=float(np.rad2deg(np.arccos(np.clip(self._body_up()@self.initial_up,-1.,1.)))) if hasattr(self,'initial_up') else 0.
        self.max_distance=max(self.max_distance,distance);self.max_tilt=max(self.max_tilt,tilt)
        return dict(phase='continuous_rotation',axis='world_z',elapsed_seconds=self.step_count*.05,
            cube_position=position.tolist(),world_z_angular_velocity=float(omega[2]),
            transverse_angular_speed=float(np.linalg.norm(omega[:2])),
            initial_tilt_deviation_degrees=tilt,max_initial_tilt_deviation_degrees=self.max_tilt,
            palm_distance=distance,max_palm_distance=self.max_distance,
            integrated_world_z_angular_velocity_rad=self.integrated_world_z,
            mean_world_z_angular_velocity=self.angular_z_sum/max(1,self.step_count),
            mean_transverse_angular_speed=self.angular_xy_sum/max(1,self.step_count),
            height_termination=bool(position[2]<-.05),episode_return=self.return_sum,
            action_history_periods=2,task_success_evaluated=False)

    def step(self,action):
        if self.done:raise RuntimeError('Reset before stepping')
        action=np.asarray(action,dtype=float)
        if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('Seven finite actions required')
        action=np.clip(action,-1.,1.)
        self.data.ctrl[:]=self.default+SCALE*action
        before=self.data.time
        for _ in range(5):mujoco.mj_step(self.model,self.data)
        numerical=not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or self.data.time<before
        omega=self.sensor('cube_angvel');fall=bool(self.sensor('cube_position')[2]<-.05)
        change=float(np.sum((action-self.last_action)**2))
        reward=.05*(float(omega[2])-change-100.*fall) if not numerical else -5.
        self.step_count+=1;self.angular_z_sum+=float(omega[2]);self.angular_xy_sum+=float(np.linalg.norm(omega[:2]))
        self.integrated_world_z+=float(omega[2])*.05;self.return_sum+=reward
        # Match official source order: construct the next obs BEFORE updating
        # last_act. A control-time t observation therefore contains a(t-2).
        observation=self._observation(self.last_action)
        self.last_action=action.copy()
        terminated=fall or numerical;truncated=self.step_count>=500 and not terminated
        self.done=terminated or truncated
        info=self._info();info.update(failure_reason='numerical' if numerical else 'height' if fall else '',
            horizon_reached=bool(truncated),action_rate_cost=change,applied_action=action.tolist(),
            motor_targets=self.data.ctrl.tolist())
        return observation,float(reward),bool(terminated),bool(truncated),info

    def render(self):
        if self.renderer is None:self.renderer=mujoco.Renderer(self.model,height=720,width=960)
        self.renderer.update_scene(self.data,camera='side')
        return self.renderer.render()

    def close(self):
        if self.renderer:self.renderer.close();self.renderer=None
