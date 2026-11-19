"""Continue a relocated strict-X V4 checkpoint with CUDA PPO and four native MuJoCo workers."""
import argparse
from datetime import datetime, timezone
from functools import partial
import fcntl
import json
import os
from pathlib import Path
import pickle
import shutil
import signal
import time

# The engine limits per-worker BLAS threads before importing torch.
from .engine import (
    BATCH_STEPS, TIMESTEPS, Engine, RetainPPO, SubprocVecEnv, VecNormalize,
    _published_checkpoint, _set_rng_state, _worker, configure, load_for_eval,
)
import gymnasium as gym
import numpy as np
import torch
from .curriculum import REVISED_DEGREES
from .evaluation import compact, evaluate, final_pass, score
from .run import plateau, qualifies
from experiments.cube_adaptive_60h.config import Profile
from experiments.cube_adaptive_60h.storage import atomic_json, digest


VALIDATION_SECONDS = 1800
SAVE_SECONDS = 300
VALIDATION_SEED = 7200000


def read(path):
    return json.loads(path.read_text())


class PortableEpisode(gym.Wrapper):
    def restore_episode(self, state):
        # SB3 exits the worker on an uncaught env_method exception. Return an
        # error to the parent's exact-observation check so all pipes stay live.
        try:
            return self.env.restore_episode(state)
        except (ValueError, RuntimeError, FloatingPointError, AssertionError) as exc:
            return f"Cannot reproduce the saved physical episode: {exc}"


def portable_worker(reference, dt, stage, profile):
    return PortableEpisode(_worker("v4", reference, dt, stage, profile))


class Continuation(Engine):
    """Reuse original rollouts, action and exploration limits, cursor and checkpoint format."""

    def __init__(self, output, checkpoint, meta, config, device, stop, reset_episodes):
        self.root = self.arm_root = output
        self.arm = "v4"
        self.stage = float(meta["stage"])
        self.profile = Profile(**meta["profile"])
        self.stop = stop
        self.deadline = float("inf")
        self.job = config
        self.job_hash = digest(output / "continuation.json")
        self.source_sha = config["source_policy_sha256"]
        self.input_sha = digest(checkpoint / "reference.npz")
        self._pending_cursor = None
        self.demo = None
        self.device = torch.device(device)
        self.max_std = torch.tensor(np.log([.1]*3 + [.2]*2 + [.15]*2), dtype=torch.float32, device=self.device)
        torch.set_num_threads(1)
        workers = [partial(portable_worker, output / "inputs/reference.npz", dt, self.stage, self.profile)
                   for dt in TIMESTEPS]
        self.vector = SubprocVecEnv(workers, start_method="spawn")
        try:
            self.normalizer = VecNormalize.load(checkpoint / "normalizer.pkl", self.vector)
            self.normalizer.training = True
            self.model = RetainPPO.load(checkpoint / "policy.zip", env=self.normalizer,
                                        device=device, force_reset=False)
            if (self.steps != meta["training_steps"] or self.epochs != meta["ppo_training_epochs"]
                    or self.model.n_steps * self.vector.num_envs != BATCH_STEPS
                    or self.model.device.type != self.device.type):
                raise ValueError("PPO counters, rollout size or device differ from the requested continuation")
            self.model.retention = None
            if (checkpoint / "demonstration.npz").is_file():
                self.set_demonstration(checkpoint / "demonstration.npz")
            with (checkpoint / "cursor.pkl").open("rb") as source:
                cursor = pickle.load(source)
            if cursor["steps"] != self.steps or len(cursor["workers"]) != len(TIMESTEPS):
                raise ValueError("Checkpoint cursor does not belong to the loaded PPO update")
            if reset_episodes:
                self.normalizer.seed((self.job["seed"] + self.steps) % (2**31))
                self._reset_batch()
                _set_rng_state(cursor["rng"])
                if self.demo is not None:
                    self.model.retention[3].bit_generator.state = cursor["demo_rng"]
                self.restore_cuda_rng(cursor)
            else:
                try:
                    self._restore_cursor(cursor)
                except (ValueError, AssertionError) as exc:
                    raise ValueError("Episode replay differs on this host. Use --reset-episodes to start "
                                     "fresh physical episodes with the saved PPO and reward statistics.") from exc
            self.model.set_logger(configure(format_strings=[]))
        except BaseException:
            self.vector.close()
            raise

    def restore_cuda_rng(self, cursor):
        if self.device.type == "cuda" and cursor.get("torch_cuda_rng") is not None:
            torch.cuda.set_rng_state(cursor["torch_cuda_rng"], self.device)

    def _restore_cursor(self, cursor):
        super()._restore_cursor(cursor)
        self.restore_cuda_rng(cursor)

    def _cursor(self):
        snapshot = super()._cursor()
        if self._pending_cursor is None and self.device.type == "cuda":
            snapshot["torch_cuda_rng"] = torch.cuda.get_rng_state(self.device)
        return snapshot

    def close(self):
        self.vector.close()


def initial_config(output, checkpoint):
    if checkpoint is None:
        raise ValueError("A new output directory needs --checkpoint")
    path, meta = _published_checkpoint(checkpoint)
    if meta["arm"] != "v4" or meta["stage"] not in REVISED_DEGREES:
        raise ValueError("Expected a strict-X V4 checkpoint at 30, 60 or 90 degrees")
    if any(p.name != ".continuation.lock" for p in output.iterdir()):
        raise ValueError("Choose an empty output directory for a new continuation")
    sha = meta["files"]["policy.zip"]
    config = dict(schema="portable_strict_x_1", source_checkpoint="sha256:" + sha, source_policy_sha256=sha,
                  source_metadata_sha256=digest(path / "metadata.json"), input_sha256=meta["files"]["reference.npz"],
                  seed=7100000, deadline_at=None, curriculum_degrees=list(REVISED_DEGREES))
    atomic_json(output / "continuation.json", config)
    return config


def checked_path(output, relative):
    candidate = output / relative
    if (candidate.parent != output / "checkpoints" or not candidate.name.startswith("step_")
            or candidate.is_symlink() or not candidate.is_dir()):
        raise ValueError("Invalid committed checkpoint path")
    return candidate


def repair_aliases(output, state):
    for name in ("latest", "best"):
        relative = state.get(name)
        if relative:
            checked_path(output, relative)
            link = output / ("." + name + ".new")
            link.unlink(missing_ok=True)
            link.symlink_to(relative, target_is_directory=True)
            link.replace(output / name)


def commit(output, engine, state, reason, best=None):
    latest = engine.save(reason, aliases=()).relative_to(output)
    state.update(latest=str(latest), steps=engine.steps, stage=engine.stage,
                 updated_at=datetime.now(timezone.utc).isoformat())
    if best is True:
        state["best"] = str(latest)
    elif isinstance(best, Path):
        state["best"] = str(best.relative_to(output))
    atomic_json(output / "state.json", state)
    repair_aliases(output, state)
    retained = {state["latest"], state.get("best")}
    for directory in (output / "checkpoints").glob("step_*"):
        if (not directory.is_symlink() and directory.is_dir()
                and str(directory.relative_to(output)) not in retained):
            shutil.rmtree(directory)


def prepare_session(output, source):
    config_file = output / "continuation.json"
    config = read(config_file) if config_file.exists() else initial_config(output, source)
    if config.get("schema") != "portable_strict_x_1":
        raise ValueError("Unexpected continuation configuration")
    state_path = output / "state.json"
    if state_path.exists():
        state = read(state_path)
        if state["config_sha256"] != digest(config_file):
            raise ValueError("Continuation configuration changed after the run started")
        repair_aliases(output, state)
        checkpoint = checked_path(output, state["latest"])
        if source is not None and source.resolve() != checkpoint:
            if digest(source / "metadata.json") != config["source_metadata_sha256"]:
                raise ValueError("This output already belongs to a different source checkpoint")
    else:
        if source is None or digest(source / "metadata.json") != config["source_metadata_sha256"]:
            raise ValueError("Supply the original checkpoint to finish initializing this output")
        checkpoint = source
        state = dict(config_sha256=digest(config_file), streak=0, best_score=None, best=None,
                     next_validation=time.time()+VALIDATION_SECONDS, last_validation_steps=None, complete=False)
    checkpoint, meta = _published_checkpoint(checkpoint)
    if (meta["arm"] != "v4" or meta["stage"] not in REVISED_DEGREES or
            meta["files"]["reference.npz"] != config["input_sha256"]):
        raise ValueError("Incompatible strict-X checkpoint")
    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)
    for name in ("reference.npz", "prepare_report.json"):
        target = inputs / name
        if not target.exists():
            staging = inputs / (name + ".tmp")
            shutil.copy2(checkpoint / name, staging)
            os.replace(staging, target)
        if digest(target) != meta["files"][name]:
            raise ValueError("Changed task input: " + name)
    return config, state, checkpoint, meta


def review(engine, state, output, stop):
    report = evaluate(engine.model, output / "inputs", engine.profile, engine.stage,
                      episodes=20, seed=VALIDATION_SEED, stop=stop)
    if not report["complete"]:
        return
    report.update(steps=engine.steps, ppo_epochs=engine.epochs)
    report_path = output / "evaluations" / f"{int(engine.stage):02d}_{engine.steps:09d}.json"
    atomic_json(report_path, report)
    rank = [engine.stage, *score(report)]
    improved = state["best_score"] is None or rank > state["best_score"]
    if improved:
        state["best_score"] = rank
    state["last_validation_steps"] = engine.steps
    state["last_validation"] = str(report_path.relative_to(output))
    state["next_validation"] = time.time() + VALIDATION_SECONDS
    state["streak"] = state["streak"] + 1 if qualifies(report) else 0
    history = state.setdefault("recent_validations", [])
    history.append(dict(rate=report["success_rate"], median=report["median_valid_peak_degrees"]))
    del history[:-4]
    print(json.dumps(compact(report)), flush=True)
    best_path = True if improved else None
    if state["streak"] == 2:
        if engine.stage == REVISED_DEGREES[-1]:
            state["curriculum_qualified"] = True
        else:
            if improved:
                best_path = engine.save("qualified_stage", aliases=())
            next_stage = REVISED_DEGREES[REVISED_DEGREES.index(engine.stage) + 1]
            engine.set_stage(next_stage)
        history.clear()
        state["streak"] = 0
    elif plateau(history):
        stage_key = str(int(engine.stage))
        practice = state.setdefault("practice", {})
        if stage_key not in practice:
            from .stall import search, verify_practice
            directory = output / "practice" / stage_key
            result = search(engine.model, output / "inputs", engine.profile, engine.stage, directory,
                            7400000 + int(engine.stage) * 100, time.time() + 600, stop)
            practice[stage_key] = result["qualified"]
            if result["qualified"]:
                engine.set_demonstration(verify_practice(directory, engine.stage))
    commit(output, engine, state, "validation", best=best_path)


def final_acceptance(output, state, stop):
    from .finalize import evaluate_frozen
    checkpoint = checked_path(output, state["latest"])
    model, meta = load_for_eval(checkpoint)
    if meta["stage"] != 90:
        raise ValueError("A frozen 90-degree checkpoint is required for final acceptance")
    task = dict(checkpoint=str(checkpoint), target=90, seed=8100000, episodes=100,
                purpose="independent_final", policy_sha256=meta["files"]["policy.zip"])
    report = evaluate_frozen(task, output / "final", Profile(**meta["profile"]), float("inf"), stop, model)
    if report["complete"]:
        state.update(complete=True, final_passed=final_pass(report), final_report="final/evaluation.json")
        atomic_json(output / "state.json", state)
        print("Final strict success counts:", {dt: g["successes"] for dt, g in report["groups"].items()}, flush=True)


def train(source, output, device, updates, reset_episodes=False):
    output = Path(output).resolve()
    source = Path(source).resolve(strict=True) if source else None
    if source is not None and output.is_relative_to(source):
        raise ValueError("Choose an output separate from the input checkpoint")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requires a visible NVIDIA GPU, driver and CUDA-enabled PyTorch")
    if updates < 1:
        raise ValueError("--updates must be positive")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".continuation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, state, checkpoint, meta = prepare_session(output, source)
        if state["complete"]:
            print("Independent 90-degree evaluation complete. Passed:", state["final_passed"])
            return
        stopping = False

        def request_stop(*_):
            nonlocal stopping
            stopping = True

        previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            if state.get("curriculum_qualified"):
                final_acceptance(output, state, lambda: stopping)
                return
            with Continuation(output, checkpoint, meta, config, device, lambda: stopping, reset_episodes) as engine:
                print(f"PPO on {engine.model.device}; four CPU MuJoCo workers. Starting step {engine.steps} at {engine.stage:g} degrees.", flush=True)
                next_save = time.time() + SAVE_SECONDS
                for _ in range(updates):
                    if stopping or not engine.advance():
                        break
                    if time.time() >= state["next_validation"] and state["last_validation_steps"] != engine.steps:
                        review(engine, state, output, lambda: stopping)
                        next_save = time.time() + SAVE_SECONDS
                    elif time.time() >= next_save:
                        commit(output, engine, state, "periodic")
                        next_save = time.time() + SAVE_SECONDS
                    if state.get("curriculum_qualified"):
                        break
                # Validation already persisted the terminal step and candidate together.
                if (state.get("steps") != engine.steps or not (output / "state.json").exists()
                        or (reset_episodes and not state.get("curriculum_qualified"))):
                    commit(output, engine, state, "paused" if stopping else "requested_updates_complete")
                print(f"Saved {engine.steps} steps at {output / 'latest'}.", flush=True)
            if state.get("curriculum_qualified") and not stopping:
                final_acceptance(output, state, lambda: stopping)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Original V4 directory; optional when resuming this output.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1024, help="Maximum additional complete 1,024-step rollouts.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--reset-episodes", action="store_true", help="Begin fresh physical episodes and keep learned state.")
    args = parser.parse_args()
    train(args.checkpoint, args.output, args.device, args.updates, args.reset_episodes)
