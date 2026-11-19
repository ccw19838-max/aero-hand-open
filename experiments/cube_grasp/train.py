"""Bounded CPU PPO training for native MuJoCo cube grasping."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import time
import uuid

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from .evaluate import (atomic_json, config_dict, evaluate_policy, file_digest,
                       implementation_digest)


def _append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


def _alias(output, name, destination):
    alias = output / name
    temporary = output / ("." + name + "." + uuid.uuid4().hex)
    temporary.symlink_to(destination.relative_to(output), target_is_directory=True)
    os.replace(temporary, alias)


def save_checkpoint(model, normalizer, reference, config, output, *, elapsed_seconds,
                    reason, score=None, best=False):
    """Write a complete immutable bundle, then atomically update aliases."""
    output = Path(output).resolve()
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    destination = checkpoints / f"step_{model.num_timesteps:09d}_{token}"
    temporary = checkpoints / (".writing_" + token)
    temporary.mkdir()
    try:
        model.save(temporary / "policy.zip")
        normalizer.save(temporary / "vecnormalize.pkl")
        shutil.copy2(reference, temporary / "reference.npz")
        report = Path(reference).parent / "prepare_report.json"
        if not report.is_file():
            raise ValueError("A checkpoint requires the preparation report alongside the reference")
        shutil.copy2(report, temporary / "prepare_report.json")
        import stable_baselines3
        import torch
        import mujoco
        metadata = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task": "upright_rigid_rubik_cube_grasp",
            "training_steps": int(model.num_timesteps),
            "elapsed_seconds": float(elapsed_seconds),
            "reason": reason,
            "validation_success_rate": score,
            "config": config_dict(config),
            "implementation_sha256": implementation_digest(),
            "files": {name: file_digest(temporary / name) for name in
                      ("policy.zip", "vecnormalize.pkl", "reference.npz", "prepare_report.json")},
            "versions": {"stable_baselines3": stable_baselines3.__version__,
                         "torch": torch.__version__, "mujoco": mujoco.__version__},
            "resume_semantics": "SB3 policy, optimizer, and VecNormalize statistics are restored; environment episode state and RNG streams are restarted, so resume is not bitwise identical.",
        }
        atomic_json(temporary / "metadata.json", metadata)
        os.replace(temporary, destination)
        _alias(output, "latest", destination)
        if best:
            _alias(output, "best", destination)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hours", type=float, default=2.)
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=8192)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--baseline-episodes", type=int, default=20)
    parser.add_argument("--config", help="Optional task-config JSON; must match resume config")
    args = parser.parse_args(argv)
    if min(args.hours, args.steps, args.envs, args.eval_every, args.eval_episodes) <= 0 or args.baseline_episodes < 0:
        parser.error("Budgets, steps, envs, and evaluation counts must be positive")
    if args.envs > 4:
        parser.error("This CPU baseline limits parallel environments to four")

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from .config import CubeGraspConfig
    from .env import CubeGraspEnv

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "status.json").exists() and not args.resume:
        raise ValueError("Output already contains a run; choose a new output or explicitly resume")
    reference = Path(args.reference).resolve()
    config = CubeGraspConfig.from_dict(json.loads(Path(args.config).read_text())) if args.config else CubeGraspConfig()
    if args.resume:
        metadata = json.loads((Path(args.resume) / "metadata.json").read_text())
        resumed_config = CubeGraspConfig.from_dict(metadata["config"])
        if args.config and config_dict(config) != config_dict(resumed_config):
            raise ValueError("Resume task config must exactly match its checkpoint")
        config = resumed_config
        if implementation_digest() != metadata["implementation_sha256"]:
            raise ValueError("Resume observation/action/model implementation changed")
        for name, expected in metadata["files"].items():
            if file_digest(Path(args.resume) / name) != expected:
                raise ValueError(f"Resume bundle failed SHA256 validation: {name}")
        if file_digest(reference) != metadata["files"]["reference.npz"]:
            raise ValueError("Resume reference changed")

    stop = {"requested": False, "reason": None}
    def request_stop(signum, frame):
        stop.update(requested=True, reason=signal.Signals(signum).name)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    deadline = started + args.hours * 3600.
    run = {"status": "initializing", "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
           "budget_hours": args.hours, "requested_additional_steps": args.steps,
           "parallel_environments": args.envs, "reference": str(reference), "seed": args.seed,
           "config": config_dict(config), "engine": "native_mujoco", "device": "cpu",
           "training_steps": 0, "elapsed_seconds": 0.}
    atomic_json(output / "status.json", run)
    atomic_json(output / "run_config.json", {**run, "arguments": vars(args)})

    def make_environment():
        return CubeGraspEnv(reference=reference, config=config)

    # Verify preparation in the parent before creating subprocesses.
    probe = make_environment()
    probe.reset(seed=args.seed)
    probe.close()
    vector = DummyVecEnv([make_environment]) if args.envs == 1 else SubprocVecEnv([make_environment] * args.envs, start_method="spawn")
    if args.resume:
        normalizer = VecNormalize.load(Path(args.resume) / "vecnormalize.pkl", vector)
        normalizer.training = True
        normalizer.norm_reward = False
        model = PPO.load(Path(args.resume) / "policy.zip", env=normalizer, device="cpu")
        model.set_random_seed(args.seed)
    else:
        normalizer = VecNormalize(vector, norm_obs=True, norm_reward=False, clip_obs=10.)
        model = PPO("MlpPolicy", normalizer, learning_rate=3e-4, n_steps=128,
                    batch_size=128, n_epochs=4, gamma=.99, gae_lambda=.95,
                    clip_range=.2, ent_coef=.005, vf_coef=.5, max_grad_norm=.5,
                    policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "log_std_init": -1.0},
                    seed=args.seed, device="cpu", verbose=0)
    normalizer.seed(args.seed)
    initial_steps = int(model.num_timesteps)
    checkpoint = save_checkpoint(model, normalizer, reference, config, output,
                                 elapsed_seconds=time.monotonic()-started,
                                 reason="initialized_before_first_training_step")
    run.update(training_steps=initial_steps, latest_checkpoint=str(checkpoint), status="baselines")
    atomic_json(output / "status.json", run)

    class TrainingCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_evaluation = initial_steps + args.eval_every
            self.next_status = time.monotonic()
            self.best_success = -1.
            self.consecutive_passes = 0
            self.reason = "step_budget"
            self.last_evaluation = None

        def _on_step(self):
            now = time.monotonic()
            if stop["requested"]:
                self.reason = stop["reason"]
                return False
            if now >= deadline:
                self.reason = "wall_clock_budget"
                return False
            if self.model.num_timesteps - initial_steps >= args.steps:
                self.reason = "step_budget"
                return False
            if now >= self.next_status:
                run.update(status="training", training_steps=int(self.model.num_timesteps),
                           elapsed_seconds=now-started,
                           environment_steps_per_second=(self.model.num_timesteps-initial_steps)/max(now-started, 1e-6))
                atomic_json(output / "status.json", run)
                self.next_status = now + 10.
            if self.model.num_timesteps >= self.next_evaluation:
                run.update(status="validating", training_steps=int(self.model.num_timesteps), elapsed_seconds=now-started)
                atomic_json(output / "status.json", run)
                result = evaluate_policy(self.model, normalizer, reference, config,
                                         episodes=args.eval_episodes, seed_start=10000,
                                         output=output / "validation" / f"step_{self.model.num_timesteps:09d}",
                                         deadline=deadline, should_stop=lambda: stop["requested"])
                self.last_evaluation = result
                score = result.get("success_rate")
                improved = result["complete"] and score > self.best_success
                if improved:
                    self.best_success = score
                self.consecutive_passes = self.consecutive_passes + 1 if result["complete"] and score >= .9 else 0
                checkpoint = save_checkpoint(self.model, normalizer, reference, config, output,
                                             elapsed_seconds=time.monotonic()-started,
                                             reason="validation", score=score, best=improved)
                summary = {"training_steps": int(self.model.num_timesteps),
                           "elapsed_seconds": time.monotonic()-started,
                           "validation_success_rate": score,
                           "validation_episodes": result["episodes_completed"],
                           "validation_complete": result["complete"],
                           "mean_max_stable_seconds": result.get("mean_max_stable_seconds"),
                           "failure_counts": result.get("failure_counts"),
                           "consecutive_validation_passes": self.consecutive_passes,
                           "checkpoint": str(checkpoint),
                           "ppo": {str(k): float(v) for k, v in self.model.logger.name_to_value.items()
                                   if isinstance(v, (int, float)) and __import__('math').isfinite(v)}}
                _append(output / "progress.jsonl", summary)
                print(json.dumps(summary), flush=True)
                run.update(latest_checkpoint=str(checkpoint), latest_validation=summary)
                atomic_json(output / "status.json", run)
                self.next_evaluation += args.eval_every
                if self.consecutive_passes >= 2:
                    self.reason = "two_validation_checks_at_least_90_percent"
                    return False
                if result["interrupted"]:
                    self.reason = stop["reason"] or "wall_clock_budget"
                    return False
            return True

    callback = TrainingCallback()
    try:
        baseline_reports = {}
        if not args.resume:
            for baseline in ("zero", "closure", "random"):
                if not args.baseline_episodes or stop["requested"] or time.monotonic() >= deadline:
                    break
                baseline_reports[baseline] = evaluate_policy(None, None, reference, config,
                                                            episodes=args.baseline_episodes, seed_start=10000,
                                                            baseline=baseline, output=output / "baselines" / baseline,
                                                            deadline=deadline, should_stop=lambda: stop["requested"])
            atomic_json(output / "baselines.json", baseline_reports)
        if stop["requested"]:
            callback.reason = stop["reason"]
        elif time.monotonic() >= deadline:
            callback.reason = "wall_clock_budget"
        else:
            run.update(status="training")
            atomic_json(output / "status.json", run)
            model.learn(total_timesteps=args.steps, callback=callback, reset_num_timesteps=False)
        run["status"] = "completed" if callback.reason == "two_validation_checks_at_least_90_percent" else "stopped"
        run["stop_reason"] = callback.reason
    except BaseException as exc:
        run.update(status="error", stop_reason=type(exc).__name__, error=str(exc))
        raise
    finally:
        checkpoint = save_checkpoint(model, normalizer, reference, config, output,
                                     elapsed_seconds=time.monotonic()-started,
                                     reason=run.get("stop_reason", "interrupted"),
                                     score=(callback.last_evaluation or {}).get("success_rate"))
        run.update(training_steps=int(model.num_timesteps), elapsed_seconds=time.monotonic()-started,
                   latest_checkpoint=str(checkpoint),
                   finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(output / "status.json", run)
        normalizer.close()
        print(json.dumps(run, indent=2), flush=True)


if __name__ == "__main__":
    main()
