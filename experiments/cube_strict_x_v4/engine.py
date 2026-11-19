"""Four CPU environments per arm. Checkpoints commit complete PPO rollouts."""

import os
for thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_var] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/aero-cube-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/aero-cube-cache")

from collections import deque
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import copy
import fcntl
import hashlib
import json
import pickle
import random
import shutil
import time
import uuid

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from experiments.cube_adaptive_60h.config import Profile
from experiments.cube_adaptive_60h.policy import Policy as BaselinePolicy, RetainPPO
from experiments.cube_adaptive_60h.storage import atomic_json, digest, load as load_legacy
from experiments.cube_two_driver_v3 import env as baseline
from experiments.cube_two_driver_v3.io import fingerprint as baseline_fingerprint

SCHEMA = "strict_x_rollout_1"
TIMESTEPS = (.001, .002, .001, .002)
ROLLOUT = 256
BATCH_STEPS = ROLLOUT * len(TIMESTEPS)
AUX_WEIGHT = .02 / (2 * .15**2)


def _fingerprint(arm):
    checksum = hashlib.sha256(baseline_fingerprint().encode())
    files = ["engine.py"]
    if arm == "v4":
        files.extend(("env.py", "contact_history.py", "policy.py"))
    for name in files:
        checksum.update(name.encode())
        checksum.update(Path(__file__).with_name(name).read_bytes())
    return checksum.hexdigest()


def _rng_state():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state()


def _set_rng_state(state):
    py, numpy, pytorch = state
    random.setstate(py)
    np.random.set_state(numpy)
    torch.random.set_rng_state(pytorch)


def _published_checkpoint(path):
    path = Path(path).resolve(strict=True)
    meta = json.loads((path / "metadata.json").read_text())
    if meta.get("schema") != SCHEMA or meta.get("arm") not in ("v3", "v4"):
        raise ValueError("Unsupported strict-X checkpoint")
    if meta["implementation_sha256"] != _fingerprint(meta["arm"]):
        raise ValueError("Training implementation changed since checkpoint creation")
    for name, expected in meta["files"].items():
        if digest(path / name) != expected:
            raise ValueError(f"Damaged checkpoint file: {name}")
    return path, meta


def load_for_eval(path):
    """Return a detached model and its metadata without reseeding active training."""
    path, meta = _published_checkpoint(path)
    state = _rng_state()
    try:
        model = RetainPPO.load(path / "policy.zip", device="cpu")
    finally:
        _set_rng_state(state)
    if model.num_timesteps != meta["training_steps"] or model._n_updates != meta["ppo_training_epochs"]:
        raise ValueError("PPO counters disagree with the checkpoint")
    return model, meta


class ReplayableEpisode(gym.Wrapper):
    """Save the active episode as its actual reset seed and motor commands."""
    def __init__(self, env):
        super().__init__(env)
        self.generator = np.random.default_rng()
        self.episode_seed = None
        self.actions = []
        self.observation = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.generator = np.random.default_rng(seed)
        else:
            seed = int(self.generator.integers(0, 2**31 - 1))
        self.episode_seed = int(seed)
        self.actions = []
        self.observation, info = self.env.reset(seed=seed, options=options)
        return self.observation, info

    def step(self, action):
        result = self.env.step(action)
        self.actions.append(np.array(action, dtype=np.float32, copy=True))
        self.observation = result[0]
        return result

    def physical_digest(self):
        checksum = hashlib.sha256()
        data = self.unwrapped.data
        for value in (data.qpos, data.qvel, data.ctrl, data.time, self.observation):
            checksum.update(np.asarray(value).tobytes())
        return checksum.hexdigest()

    def snapshot_episode(self):
        return {
            "seed": self.episode_seed,
            "actions": np.asarray(self.actions, dtype=np.float32).reshape(-1, 7),
            "generator": self.generator.bit_generator.state,
            "physical_digest": self.physical_digest(),
        }

    def restore_episode(self, state):
        self.reset(seed=state["seed"])
        for action in state["actions"]:
            _, _, done, truncated, _ = self.step(action)
            if done or truncated:
                raise ValueError("Checkpoint replay unexpectedly ended the episode")
        if state["physical_digest"] != self.physical_digest():
            raise ValueError("Checkpoint replay changed physical state or observation")
        self.generator.bit_generator.state = state["generator"]
        return self.observation


def _worker(arm, reference, dt, stage, profile):
    if arm == "v3":
        module = baseline
    else:
        from . import env as module
    task = module.FastEnv(reference, "x", physical_dt=dt, target_degrees=stage,
                          profile=profile, training=True)
    return ReplayableEpisode(module.training_reward(task))


class _StopAt(BaseCallback):
    def __init__(self, deadline, stop):
        super().__init__()
        self.deadline = deadline
        self.stop = stop

    def _on_step(self):
        return time.time() < self.deadline and not self.stop()


class Engine:
    def __init__(self, root, arm, stage=5., checkpoint=None, stop=lambda: False):
        if arm not in ("v3", "v4"):
            raise ValueError("Arm must be v3 or v4")
        if stage not in baseline.LADDER:
            raise ValueError("Unknown curriculum stage")
        torch.set_num_threads(1)
        self.root = Path(root).resolve()
        self.arm = arm
        self.job = json.loads((self.root / "job.json").read_text())
        self.job_hash = digest(self.root / "job.json")
        self.profile = Profile(**self.job["profile"])
        self.deadline = datetime.fromisoformat(self.job["deadline_at"]).timestamp()
        if time.time() >= self.deadline:
            raise ValueError("Experiment has reached its original deadline")
        self.steps_per_update = BATCH_STEPS
        self.stop = stop
        self.arm_root = self.root / "arms" / arm
        self.arm_root.mkdir(parents=True, exist_ok=True)
        self._lock = (self.arm_root / "engine.lock").open("a")
        self.vector = None
        self._closed = False
        self._pending_cursor = None
        self.stage = float(stage)
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._open(checkpoint)
        except BaseException:
            self.close()
            raise

    def _open(self, checkpoint):
        latest = self.arm_root / "latest"
        if checkpoint is None and latest.exists():
            checkpoint = latest
        source = Path(self.job["source_checkpoint"])
        self.source_sha = digest(source / "policy.zip")
        self.input_sha = digest(self.root / "inputs/reference.npz")
        meta = None
        if checkpoint is not None:
            checkpoint, meta = _published_checkpoint(checkpoint)
            if (meta["arm"], meta["job_sha256"], meta["source_policy_sha256"], meta["input_sha256"]) != (
                    self.arm, self.job_hash, self.source_sha, self.input_sha):
                raise ValueError("Checkpoint arm, job, reference or transfer source does not match")
            self.stage = meta["stage"]
        functions = [partial(_worker, self.arm, self.root / "inputs/reference.npz",
                             dt, self.stage, self.profile) for dt in TIMESTEPS]
        self.vector = SubprocVecEnv(functions, start_method="spawn")
        self.max_std = torch.as_tensor(np.log(self.job.get("max_std", [.10]*3+[.20]*2+[.15]*2)), dtype=torch.float32)
        self.demo = None
        if checkpoint is None:
            self.normalizer = VecNormalize(self.vector, norm_obs=False, norm_reward=True,
                                           gamma=self.profile.gamma, clip_reward=10000.)
            self.model = self._fresh_model(source)
            # Consume identical exploration and minibatch RNG streams in both arms.
            random.seed(self.job["seed"])
            np.random.seed(self.job["seed"])
            torch.manual_seed(self.job["seed"])
            self.normalizer.seed(self.job["seed"])
            self._reset_batch()
        else:
            self.normalizer = VecNormalize.load(checkpoint / "normalizer.pkl", self.vector)
            self.normalizer.training = True
            self.model = RetainPPO.load(checkpoint / "policy.zip", env=self.normalizer,
                                        device="cpu", force_reset=False)
            self.model.retention = None
            if (checkpoint / "demonstration.npz").exists():
                self.set_demonstration(checkpoint / "demonstration.npz")
            with (checkpoint / "cursor.pkl").open("rb") as file:
                cursor = pickle.load(file)
            self._restore_cursor(cursor)
            if self.steps != meta["training_steps"] or self.epochs != meta["ppo_training_epochs"]:
                raise ValueError("Saved PPO counters disagree")
        self.model.set_logger(configure(format_strings=[]))

    def _fresh_model(self, source):
        if self.arm == "v4":
            from .policy import Policy, initialize_from_v3
        else:
            Policy = BaselinePolicy
        model = RetainPPO(Policy, self.normalizer, learning_rate=self.profile.learning_rate,
                          n_steps=ROLLOUT, batch_size=128, n_epochs=4, gamma=self.profile.gamma,
                          gae_lambda=.98, clip_range=.15, ent_coef=self.profile.entropy,
                          max_grad_norm=.5, seed=self.job["seed"], device="cpu", verbose=0,
                          use_sde=False)
        if self.arm == "v3":
            initial = model.policy
        else:
            initial = BaselinePolicy(gym.spaces.Box(-np.inf, np.inf, (143,), np.float32),
                                     self.normalizer.action_space, lambda _: self.profile.learning_rate)
        legacy, _ = load_legacy(source)
        initial.features_extractor.load_state_dict(legacy.policy.features_extractor.state_dict())
        initial.mlp_extractor.load_state_dict(legacy.policy.mlp_extractor.state_dict())
        mean = self.job.get("initial_action_mean", [0., 0., 0., .15, -.15, 0., 0.])
        std = self.job.get("initial_std", [.04]*3 + [.12]*2 + [.08]*2)
        with torch.no_grad():
            initial.value_net.weight.zero_()
            initial.value_net.bias.zero_()
            initial.action_net[0].weight.zero_()
            initial.action_net[0].bias.copy_(torch.as_tensor(np.arctanh(mean)))
            initial.log_std.copy_(torch.as_tensor(np.log(std)))
        if self.arm == "v4":
            initialize_from_v3(model.policy, initial)
        model.retention = None
        return model

    @property
    def worker_pids(self):
        return [worker.pid for worker in self.vector.processes]

    @property
    def steps(self):
        return int(self.model.num_timesteps)

    @property
    def epochs(self):
        return int(self.model._n_updates)

    def _reset_batch(self):
        self.model._last_obs = self.normalizer.reset()
        self.model._last_episode_starts = np.ones(len(TIMESTEPS), dtype=bool)
        self.model._last_original_obs = self.normalizer.get_original_obs()

    def _cursor(self):
        if self._pending_cursor is not None:
            return self._pending_cursor
        cursor = {
            "steps": self.steps,
            "workers": self.vector.env_method("snapshot_episode"),
            "rng": _rng_state(),
            "obs": self.model._last_obs.copy(),
            "episode_starts": self.model._last_episode_starts.copy(),
            "returns": self.normalizer.returns.copy(),
            "reward_rms": copy.deepcopy(self.normalizer.ret_rms),
            "demo_rng": None if self.demo is None else self.model.retention[3].bit_generator.state,
        }
        return cursor

    def _restore_cursor(self, cursor):
        for i, worker in enumerate(cursor["workers"]):
            result, = self.vector.env_method("restore_episode", worker, indices=i)
            np.testing.assert_array_equal(result, cursor["obs"][i])
        # Model.load seeds its VecEnv for the next explicit reset. The workers
        # already reconstructed their saved episode generators from the cursor.
        self.vector._reset_seeds()
        self.model._last_obs = cursor["obs"].copy()
        self.model._last_original_obs = cursor["obs"].copy()
        self.model._last_episode_starts = cursor["episode_starts"].copy()
        self.model.num_timesteps = cursor["steps"]
        self.normalizer.returns = cursor["returns"].copy()
        self.normalizer.ret_rms = copy.deepcopy(cursor["reward_rms"])
        self.normalizer.old_obs = cursor["obs"].copy()
        _set_rng_state(cursor["rng"])
        if self.demo is not None:
            self.model.retention[3].bit_generator.state = cursor["demo_rng"]
        self._pending_cursor = None

    def set_stage(self, degrees):
        if degrees not in baseline.LADDER:
            raise ValueError("Invalid curriculum angle")
        if self._pending_cursor is not None:
            self._restore_cursor(self._pending_cursor)
        if degrees != self.stage:
            self.vector.env_method("set_task", float(degrees))
            self.stage = float(degrees)
            self.set_demonstration(None)
            self._reset_batch()

    def set_demonstration(self, path):
        if path is None:
            self.demo = None
            self.model.retention = None
            return
        if self.arm != "v4":
            raise ValueError("Auxiliary practice is restricted to the V4 arm")
        with np.load(path, allow_pickle=False) as file:
            obs = np.asarray(file["observations"], dtype=np.float32)
            actions = np.asarray(file["actions"], dtype=np.float32)
        if (obs.ndim != 2 or obs.shape[1] != 149 or not len(obs) or actions.shape != (len(obs), 7)
            or not np.all(np.isfinite(obs)) or not np.all(np.isfinite(actions)) or np.max(abs(actions)) > 1):
            raise ValueError("Invalid auxiliary observations/actions")
        self.demo = (obs, actions)
        self.model.retention = (*self.demo, AUX_WEIGHT, np.random.default_rng(self.job["seed"] + int(self.stage)))

    def advance(self, deadline=None):
        cutoff = min(self.deadline, float(deadline) if deadline is not None else self.deadline)
        if time.time() >= cutoff or self.stop():
            return False
        if self._pending_cursor is not None:
            self._restore_cursor(self._pending_cursor)
        before = self._cursor()
        old_epochs = self.epochs
        self.model.learn(total_timesteps=BATCH_STEPS, reset_num_timesteps=False,
                         log_interval=None, callback=_StopAt(cutoff, self.stop))
        if self.epochs == old_epochs:
            # No parameters have changed. Persist the last complete rollout
            # boundary and reconstruct it lazily only if sampling continues.
            self._pending_cursor = before
            self.model.num_timesteps = before["steps"]
            self.normalizer.ret_rms = before["reward_rms"]
            self.normalizer.returns = before["returns"]
            self.model._last_obs = before["obs"]
            self.model._last_episode_starts = before["episode_starts"]
            return False
        if self.steps - before["steps"] != BATCH_STEPS or self.epochs != old_epochs + 4:
            raise RuntimeError("Incomplete PPO update")
        with torch.no_grad():
            self.model.policy.log_std.copy_(torch.minimum(self.model.policy.log_std, self.max_std))
        return True

    def save(self, reason, aliases=("latest",)):
        directory = self.arm_root / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex[:10]
        path = directory / f"step_{self.steps:09d}_{token}"
        temp = directory / f".writing_{token}"
        temp.mkdir()
        try:
            self.model.save(temp / "policy.zip")
            self.normalizer.save(temp / "normalizer.pkl")
            with (temp / "cursor.pkl").open("wb") as file:
                pickle.dump(self._cursor(), file, protocol=pickle.HIGHEST_PROTOCOL)
            for name in ("reference.npz", "prepare_report.json"):
                shutil.copy2(self.root / "inputs" / name, temp / name)
            if self.demo is not None:
                np.savez_compressed(temp / "demonstration.npz", observations=self.demo[0], actions=self.demo[1])
            files = {p.name: digest(p) for p in temp.iterdir()}
            meta = dict(schema=SCHEMA, task="x90_strict_support", arm=self.arm, stage=self.stage,
                        profile=self.profile.dictionary(), reason=reason, created_at=datetime.now(timezone.utc).isoformat(),
                        training_steps=self.steps, ppo_training_epochs=self.epochs,
                        rollout_count=self.steps // BATCH_STEPS, observation_size=self.normalizer.observation_space.shape[0],
                        job_sha256=self.job_hash, deadline_at=self.job["deadline_at"],
                        implementation_sha256=_fingerprint(self.arm), input_sha256=self.input_sha,
                        source_checkpoint=self.job["source_checkpoint"], source_policy_sha256=self.source_sha,
                        files=files)
            atomic_json(temp / "metadata.json", meta)
            os.replace(temp, path)
        except BaseException:
            if temp.exists():
                shutil.rmtree(temp)
            raise
        for name in aliases:
            if Path(name).name != name or name in ("", ".", ".."):
                raise ValueError("Checkpoint alias must be a direct child of the arm directory")
            temp_link = self.arm_root / f".{name}_{token}"
            temp_link.symlink_to(path.relative_to(self.arm_root), target_is_directory=True)
            os.replace(temp_link, self.arm_root / name)
        return path

    def close(self):
        if not self._closed:
            try:
                if self.vector is not None:
                    self.vector.close()
            finally:
                self._lock.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
