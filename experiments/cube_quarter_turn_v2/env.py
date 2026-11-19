"""Full-range control and a shared 133-value teacher observation for plan/train/replay."""
import gymnasium as gym
import mujoco
import numpy as np
from experiments.cube_quarter_turn.env import QuarterTurnEnv
from experiments.cube_quarter_turn.task import AXES, aligned_cube_basis
from experiments.cube_quarter_turn.action_mapping_candidate import full_range_targets
from .task import Config, GoalMonitor

OBS_SIZE = 133


def targets_to_action(targets, reference, lower, upper):
    delta = np.asarray(targets)-reference
    span = np.where(delta >= 0., upper-reference, reference-lower)
    return np.clip(np.divide(delta, span, out=np.zeros(7), where=span>1e-12), -1., 1.)


class TeacherEnv(QuarterTurnEnv):
    def __init__(self, reference, axis=None, config=None, physical_dt=.002, render_mode=None):
        if physical_dt not in (.001, .002): raise ValueError('Only validated 1ms/2ms integration')
        super().__init__(reference, axis, render_mode, config or Config())
        self.physical_dt = physical_dt
        self.model.opt.timestep = physical_dt
        self.grip_stable_seconds = 0.
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_SIZE,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # v2 has no terminal-state reset bank. Chaining uses start_goal only.
        self.success_bank = []
        self.grip_stable_seconds = 0.
        options = {**(options or {}), 'allow_bank': False}
        super().reset(seed=seed, options=options)
        self.monitor = GoalMonitor(self.data.qpos[19:23].copy(), self.monitor.axis, self.basis, self.config)
        self.last_observation = self._observation(self.last_action)
        return self.last_observation.copy(), self._info()

    def _observation(self, history):
        source = super()._observation(history)
        m = self.monitor
        phase = 'setup' if self.warming else getattr(m, 'phase', 'drive')
        phases = np.array([phase == p for p in ('setup','drive','capture','hold')], dtype=float)
        forces = self.contacts['finger_forces']
        palm = max(0., self.contacts['hand_force']-sum(forces))
        targets = 2*(self.data.ctrl-self.ids['ctrl_lower'])/(self.ids['ctrl_upper']-self.ids['ctrl_lower'])-1
        return np.r_[source['privileged_state'], source['goal_state'],
            np.clip(np.r_[forces,palm]/5.,0.,20.), targets, phases,
            min(1.,getattr(m,'stagnant_seconds',0.)/self.config.stagnation_seconds),
            m.penetration_time/self.config.violation_seconds,
            m.region_time/self.config.violation_seconds,
            m.unsupported_time/self.config.max_unsupported_seconds].astype(np.float32)

    def _info(self):
        info = super()._info()
        info.update(task=self.config.version, phase='setup' if self.warming else getattr(self.monitor,'phase','drive'),
            stagnation_seconds=getattr(self.monitor,'stagnant_seconds',0.), actor_inputs=OBS_SIZE,
            action_mapping='piecewise full range, zero at reference', physical_dt=self.physical_dt,
            action_history='previous applied control action a(t-1)',
            grip_stable_seconds=self.grip_stable_seconds)
        return info

    def start_goal(self, axis):
        if not self.monitor.success: raise RuntimeError('Successful endpoint required before chaining')
        if axis not in AXES: raise ValueError('Invalid axis')
        ideal = self.monitor.goal.copy()
        self.monitor = GoalMonitor(self.data.qpos[19:23].copy(),axis,self.basis,self.config,ideal)
        self.warming = False
        self.last_observation = self._observation(self.last_action)
        return self.last_observation.copy(), self._info()

    def step(self, action):
        if not self._has_reset or self.monitor.done: raise RuntimeError('Reset or start a new goal first')
        action = np.asarray(action,dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all(): raise ValueError('Seven finite actions required')
        applied = np.zeros(7) if self.warming else np.clip(action,-1.,1.)
        self.data.ctrl[:] = full_range_targets(applied,self.default,self.ids['ctrl_lower'],self.ids['ctrl_upper'])
        reward = 0.
        stride = round(.01/self.physical_dt)
        for i in range(round(self.config.ctrl_dt/self.physical_dt)):
            before = self.data.time
            mujoco.mj_step(self.model,self.data)
            self.elapsed += self.physical_dt
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or self.data.time < before:
                self.monitor.failure = 'numerical'; reward -= self.config.failure_penalty; break
            if self._forbidden():
                self.monitor.failure = 'forbidden_support'; reward -= self.config.failure_penalty; break
            if not self.warming: self.monitor.track(self.data.qpos[19:23])
            if (i+1)%stride == 0:
                mujoco.mj_forward(self.model,self.data); self._refresh()
                fingers=self.contacts['finger_forces']>=self.config.contact_force
                grip_stable=bool(fingers[0] and np.count_nonzero(fingers[1:])>=2
                    and np.linalg.norm(self.linear)<.03 and np.linalg.norm(self.angular)<.5
                    and self.contacts['max_penetration']<=self.config.max_penetration
                    and self.displacement<=self.config.max_displacement and not self.contacts['forbidden_support'])
                self.grip_stable_seconds=self.grip_stable_seconds+.01 if grip_stable else 0.
                # Also enforce contact, penetration and workspace limits during setup.
                value = self.monitor.checkpoint(self.data.qpos[19:23], self.contacts,
                    self.linear,self.angular,self.displacement,.01)
                if not self.warming: reward += value
                if self.monitor.done: break
        if self.warming and self.elapsed+1e-10 >= self.config.warmup_seconds and not self.monitor.done:
            self.basis = aligned_cube_basis(self.data.qpos[19:23],self.mount_basis)
            self.monitor = GoalMonitor(self.data.qpos[19:23].copy(),self.monitor.axis,self.basis,self.config)
            self.warming = False
        reward -= .001*float(np.sum((applied-self.last_action)**2))
        self.returned_history = applied.copy()
        self.last_action = applied.copy()
        obs = self.last_observation.copy() if self.monitor.failure == 'numerical' else self._observation(applied)
        self.last_observation = obs.copy()
        info = self._info()
        info.update(applied_action=applied.tolist(), motor_targets=self.data.ctrl.tolist())
        return obs, float(reward), bool(self.monitor.done), False, info
