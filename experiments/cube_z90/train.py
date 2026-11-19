"""Bounded CPU PPO training for the aggressive Z-only cube curriculum."""
from __future__ import annotations

import argparse
from dataclasses import replace
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
                       implementation_digest, AXES)


class AxisPromotionMonitor:
    """Advance each unfinished axis independently; require two complete checks."""
    def __init__(self, config):
        self.groups=tuple(config.training_axes)
        self.angles={a:config.angle_for(a) for a in self.groups}
        self.streaks={a:0 for a in self.groups}
        self.ready_axes=[]
        self.final_pass=False

    def update(self, report):
        for axis in self.groups:
            result=report.get("groups",{}).get(axis,{})
            passed=result.get("complete",False) and result.get("success_rate",0.)>=.9
            self.streaks[axis]=self.streaks[axis]+1 if passed else 0
        self.ready_axes=[a for a in self.groups if self.angles[a]<90 and self.streaks[a]>=2]
        self.final_pass=all(self.angles[a]==90 and self.streaks[a]>=2 for a in self.groups)
        return bool(self.ready_axes or self.final_pass)


def validation_score(report):
    if not report or not report.get("complete"):
        return None
    return {"group_success_rates": report["group_success_rates"],
            "min_group_success_rate": report["min_group_success_rate"],
            "capture_rate": report["capture_rate"],
            "task_mode": report["task_mode"],
            "mean_max_hold_stable_seconds": report["mean_max_hold_stable_seconds"],
            "min_contact_duty_cycle": min(report["contact_duty_cycle"].values()),
            "mean_final_angle_error_degrees": report["mean_final_angle_error_degrees"]}


def selection_key(score):
    if score is None:
        return (-1., -1., -1., -float("inf"), -1.)
    rates = list(score["group_success_rates"].values())
    progress = score["mean_max_hold_stable_seconds"] if score["task_mode"] == "hold" else -score["mean_final_angle_error_degrees"]
    return (min(rates), sum(rates)/len(rates), score["capture_rate"], progress, score["min_contact_duty_cycle"])


def _append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


def _alias(output, name, destination):
    alias = output / name
    temporary = output / ("." + name + "." + uuid.uuid4().hex)
    temporary.symlink_to(destination.relative_to(output), target_is_directory=True)
    os.replace(temporary, alias)


def validate_initialization_checkpoint(checkpoint, reference, config):
    """Check source provenance and identical physics before explicit warm start."""
    from experiments.cube_envelop.evaluate import implementation_digest as legacy_digest
    from experiments.cube_envelop.config import CubeReorientConfig as LegacyConfig
    from experiments.cube_envelop_gradual.evaluate import implementation_digest as gradual_digest
    from experiments.cube_envelop_gradual.config import CubeReorientConfig as GradualConfig
    from .config import CubeReorientConfig
    from .model import model_fingerprint
    checkpoint=Path(checkpoint).resolve()
    metadata=json.loads((checkpoint/"metadata.json").read_text())
    source_task=metadata.get("task")
    sources={"cube_envelope_xyz90_v1":(LegacyConfig,legacy_digest),
             "cube_envelope_gradual_v1":(GradualConfig,gradual_digest),
             "cube_z90_aggressive_v1":(CubeReorientConfig,implementation_digest)}
    if metadata.get("format_version")!=1 or metadata.get("observation_size")!=78 or source_task not in sources:
        raise ValueError("A trusted compatible 78-observation checkpoint is required")
    source_class,source_digest=sources[source_task]
    if metadata.get("implementation_sha256")!=source_digest():
        raise ValueError("Initialization source implementation changed")
    source=source_class.from_dict(metadata["config"])
    if model_fingerprint(source)!=model_fingerprint(config):
        raise ValueError("Initialization changed the physical hand/object model")
    # The observable layout and physical meaning are shared. Changes to action
    # delta, reward, holding duration and phase controller are deliberate and
    # recorded, not an exact continuation of the old policy's training run.
    for name,expected in metadata["files"].items():
        if file_digest(checkpoint/name)!=expected:raise ValueError(f"Corrupt initialization file: {name}")
    if file_digest(reference)!=metadata["files"]["reference.npz"]:raise ValueError("Initialization reference changed")
    return metadata


def new_policy(normalizer, seed):
    from stable_baselines3 import PPO
    return PPO("MlpPolicy", normalizer, learning_rate=2e-4, n_steps=256,
               batch_size=256, n_epochs=6, gamma=.995, gae_lambda=.95,
               clip_range=.2, ent_coef=.01, vf_coef=.5, max_grad_norm=.5,
               policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "log_std_init": -.5},
               seed=seed, device="cpu", verbose=0)


def configure_reward_normalization(normalizer):
    """Scale rewards without flattening the completion bonus at SB3's default 10."""
    normalizer.norm_reward = True
    normalizer.gamma = .995
    normalizer.clip_reward = 10000.


def initialize_from_checkpoint(checkpoint, reference, config, vector, seed):
    """Warm start the actor; rebuild critic/Adam for the different reward scale."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    checkpoint = Path(checkpoint).resolve()
    metadata = validate_initialization_checkpoint(checkpoint, reference, config)
    normalizer = VecNormalize.load(checkpoint / "vecnormalize.pkl", vector)
    normalizer.training = True
    configure_reward_normalization(normalizer)
    if tuple(normalizer.observation_space.shape) != (78,) or tuple(normalizer.action_space.shape) != (7,):
        raise ValueError("Initialization environment must have 78 observations and 7 actions")
    source = PPO.load(checkpoint / "policy.zip", device="cpu")
    if tuple(source.observation_space.shape) != (78,) or tuple(source.action_space.shape) != (7,):
        raise ValueError("Initialization policy must have 78 observations and 7 actions")
    model = new_policy(normalizer, seed)
    source_state=source.policy.state_dict()
    actor_keys=[key for key in source_state if key.startswith(('mlp_extractor.policy_net.', 'action_net.'))]
    weights=model.policy.state_dict()
    for key in actor_keys:
        weights[key]=source_state[key]
    # Larger, fresh exploration rather than inheriting a near-static old std.
    model.policy.load_state_dict(weights,strict=True)
    from stable_baselines3.common.running_mean_std import RunningMeanStd
    normalizer.ret_rms=RunningMeanStd(shape=())
    normalizer.returns[:]=0.
    if model.num_timesteps != 0 or model._n_updates != 0 or model.policy.optimizer.state:
        raise RuntimeError("Initialization accidentally retained training/optimizer state")
    provenance = {
        "checkpoint": str(checkpoint),
        "source_task_mode": metadata["config"]["task_mode"], "target_task_mode": config.task_mode,
        "source_task": metadata["task"], "source_targets": metadata["config"].get("target_angles"),
        "target_angles": {'z':config.target_degrees}, "training_axes": config.training_axes,
        "source_training_steps": int(metadata["training_steps"]),
        "source_policy_sha256": metadata["files"]["policy.zip"],
        "reference_sha256": metadata["files"]["reference.npz"],
        "source_action_delta_fraction":metadata['config']['action_delta_fraction'],
        "target_action_delta_fraction":config.action_delta_fraction,
        "semantics": "Copied deterministic actor weights and observation statistics only. Critic, log_std=-0.5, reward statistics, Adam, training counters and environment/RNG streams are new. Initial acquire uses fixed reference motor targets; rotation/recapture use the actor.",
    }
    return model, normalizer, provenance


def save_checkpoint(model, normalizer, reference, config, output, *, elapsed_seconds,
                    reason, score=None, best=False, initialization=None):
    """Write a complete immutable bundle, then atomically update aliases."""
    output = Path(output).resolve()
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    destination = checkpoints / f"step_{model.num_timesteps:09d}_{token}"
    temporary = checkpoints / (".writing_" + token)
    temporary.mkdir()
    try:
        if tuple(model.observation_space.shape) != (78,) or tuple(model.action_space.shape) != (7,):
            raise ValueError("Envelope checkpoints require 78 observations and 7 actions")
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
            "task": "cube_z90_aggressive_v1",
            "task_mode": config.task_mode,
            "observation_size": 78,
            "axis_frame": "fixed_hand_axes",
            "goal_rotation_degrees": {a:config.angle_for(a) for a in config.training_axes},
            "training_steps": int(model.num_timesteps),
            "elapsed_seconds": float(elapsed_seconds),
            "reason": reason,
            "initialization": initialization,
            "validation": score,
            "config": config_dict(config),
            "reward_normalization": {"enabled_for_training": True,
                "gamma": normalizer.gamma, "clip_reward": normalizer.clip_reward,
                "note": "High clip preserves the relative size of terminal success and dense holding rewards."},
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
    parser.add_argument("--hours", type=float, default=24.)
    parser.add_argument("--keep-training-after-validation", action="store_true", help="Continue the current targets after a failed independent test")
    parser.add_argument("--steps", type=int, default=5000000)
    parser.add_argument("--envs", type=int, default=4)
    starting = parser.add_mutually_exclusive_group()
    starting.add_argument("--resume")
    starting.add_argument("--init-checkpoint", help="Initialize reorientation from compatible hold weights/statistics with a new optimizer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=16384)
    parser.add_argument("--eval-episodes", type=int, default=20, help="Validation episodes per axis, or one hold group")
    parser.add_argument("--baseline-episodes", type=int, default=5, help="Baseline episodes per axis, or one hold group")
    parser.add_argument("--config", help="Optional task-config JSON; must match resume config")
    parser.add_argument("--task-mode", choices=("hold", "reorient"), help="Override config mode for a new run; resume mode must match")
    args = parser.parse_args(argv)
    if min(args.hours, args.steps, args.envs, args.eval_every, args.eval_episodes) <= 0 or args.baseline_episodes < 0:
        parser.error("Budgets, steps, envs, and evaluation counts must be positive")
    if args.envs > 4:
        parser.error("This CPU baseline limits parallel environments to four")
    if args.hours > 24 or args.steps > 10000000:
        parser.error("Curriculum stage is capped at 24 hours and 10000000 additional steps")

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from .config import CubeReorientConfig
    from .env import CubeReorientEnv

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "status.json").exists() and not args.resume:
        raise ValueError("Output already contains a run; choose a new output or explicitly resume")
    reference = Path(args.reference).resolve()
    config = CubeReorientConfig.from_dict(json.loads(Path(args.config).read_text())) if args.config else CubeReorientConfig()
    if args.task_mode:
        config = replace(config, task_mode=args.task_mode)
    initialization = None
    if args.resume:
        metadata = json.loads((Path(args.resume) / "metadata.json").read_text())
        if metadata.get("task") != "cube_z90_aggressive_v1":
            raise ValueError("Old 75-observation cube policies cannot resume this 78-observation task; start a fresh envelope run")
        if metadata.get("observation_size") != 78:
            raise ValueError("Resume requires an envelope checkpoint with 78 observations")
        resumed_config = CubeReorientConfig.from_dict(metadata["config"])
        if args.config and config_dict(config) != config_dict(resumed_config):
            raise ValueError("Resume task config must exactly match its checkpoint")
        config = resumed_config
        if args.task_mode and args.task_mode != resumed_config.task_mode:
            raise ValueError("Resume cannot switch hold/reorient task modes")
        if implementation_digest() != metadata["implementation_sha256"]:
            raise ValueError("Resume observation/action/model implementation changed")
        for name, expected in metadata["files"].items():
            if file_digest(Path(args.resume) / name) != expected:
                raise ValueError(f"Resume bundle failed SHA256 validation: {name}")
        if file_digest(reference) != metadata["files"]["reference.npz"]:
            raise ValueError("Resume reference changed")
        initialization = metadata.get("initialization")
    elif args.init_checkpoint:
        validate_initialization_checkpoint(args.init_checkpoint, reference, config)

    stop = {"requested": False, "reason": None}
    def request_stop(signum, frame):
        stop.update(requested=True, reason=signal.Signals(signum).name)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    deadline = started + args.hours * 3600.
    run = {"status": "initializing", "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
           "task": "cube_z90_aggressive_v1",
           "budget_hours": args.hours, "requested_additional_steps": args.steps,
           "parallel_environments": args.envs, "reference": str(reference), "seed": args.seed,
           "config": config_dict(config), "engine": "native_mujoco", "device": "cpu",
           "task_mode": config.task_mode, "observation_size": 78,
           "training_axes": list(config.training_axes) if config.task_mode == "reorient" else [],
           "target_rotation_degrees": {a:config.angle_for(a) for a in config.training_axes},
           "axis_frame": "fixed_hand_axes",
           "training_steps": 0, "elapsed_seconds": 0.}
    atomic_json(output / "status.json", run)
    atomic_json(output / "run_config.json", {**run, "arguments": vars(args)})

    def make_environment():
        return CubeReorientEnv(reference=reference, config=config)

    # Verify preparation in the parent before creating subprocesses.
    probe = make_environment()
    probe.reset(seed=args.seed)
    probe.close()
    vector = DummyVecEnv([make_environment]) if args.envs == 1 else SubprocVecEnv([make_environment] * args.envs, start_method="spawn")
    if args.resume:
        normalizer = VecNormalize.load(Path(args.resume) / "vecnormalize.pkl", vector)
        normalizer.training = True
        configure_reward_normalization(normalizer)
        model = PPO.load(Path(args.resume) / "policy.zip", env=normalizer, device="cpu")
        model.set_random_seed(args.seed)
    elif args.init_checkpoint:
        model, normalizer, initialization = initialize_from_checkpoint(
            args.init_checkpoint, reference, config, vector, args.seed)
    else:
        normalizer = VecNormalize(vector, norm_obs=True, norm_reward=True, clip_obs=10., gamma=.995, clip_reward=10000.)
        model = new_policy(normalizer, args.seed)
    run["initialization"] = initialization
    atomic_json(output / "run_config.json", {**run, "arguments": vars(args)})
    normalizer.seed(args.seed)
    initial_steps = int(model.num_timesteps)
    checkpoint = save_checkpoint(model, normalizer, reference, config, output,
                                 elapsed_seconds=time.monotonic()-started,
                                 reason="initialized_before_first_training_step", initialization=initialization)
    run.update(training_steps=initial_steps, latest_checkpoint=str(checkpoint), status="baselines")
    atomic_json(output / "status.json", run)

    class TrainingCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_evaluation = initial_steps + args.eval_every
            self.next_status = time.monotonic()
            self.best_key = selection_key(None)
            if (output / "best" / "metadata.json").is_file():
                self.best_key = selection_key(json.loads((output / "best" / "metadata.json").read_text()).get("validation"))
            self.promotion = AxisPromotionMonitor(config)
            self.next_periodic_save = initial_steps + 4096
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
            if self.model.num_timesteps >= self.next_periodic_save and self.model.num_timesteps < self.next_evaluation:
                checkpoint=save_checkpoint(self.model, normalizer, reference, config, output,
                                           elapsed_seconds=time.monotonic()-started,
                                           reason="periodic", initialization=initialization)
                run.update(latest_checkpoint=str(checkpoint))
                atomic_json(output/"status.json",run)
                self.next_periodic_save=self.model.num_timesteps+4096
            if self.model.num_timesteps >= self.next_evaluation:
                run.update(status="validating", training_steps=int(self.model.num_timesteps), elapsed_seconds=now-started)
                atomic_json(output / "status.json", run)
                result = evaluate_policy(self.model, normalizer, reference, config,
                                         episodes=args.eval_episodes, seed_start=10000,
                                         output=output / "validation" / f"step_{self.model.num_timesteps:09d}",
                                         deadline=deadline, should_stop=lambda: stop["requested"])
                self.last_evaluation = result
                score = validation_score(result)
                improved = score is not None and selection_key(score) > self.best_key
                if improved:
                    self.best_key = selection_key(score)
                promoted = self.promotion.update(result)
                checkpoint = save_checkpoint(self.model, normalizer, reference, config, output,
                                             elapsed_seconds=time.monotonic()-started,
                                             reason="validation", score=score, best=improved, initialization=initialization)
                summary = {"training_steps": int(self.model.num_timesteps),
                           "elapsed_seconds": time.monotonic()-started,
                           "axis_success_rates": result["axis_success_rates"],
                           "group_success_rates": result["group_success_rates"],
                           "task_mode": config.task_mode,
                           "min_axis_success_rate": result["min_axis_success_rate"],
                           "validation_episodes": result["episodes_completed"],
                           "validation_complete": result["complete"],
                           "group_angle_error_degrees": {axis: value["mean_final_angle_error_degrees"] for axis, value in result["groups"].items()},
                           "group_best_angle_error_degrees": {axis: value["mean_best_angle_error_degrees"] for axis, value in result["groups"].items()},
                           "group_drop_rates": {axis: value["drop_rate"] for axis, value in result["groups"].items()},
                           "group_capture_rates": {axis: value["capture_rate"] for axis, value in result["groups"].items()},
                           "group_recapture_rates": {axis: value["recapture_rate"] for axis, value in result["groups"].items()},
                           "group_contact_duty_cycle": {axis: value["contact_duty_cycle"] for axis, value in result["groups"].items()},
                           "group_max_hold_stable_seconds": {axis: value["mean_max_hold_stable_seconds"] for axis, value in result["groups"].items()},
                           "group_max_envelope_stable_seconds": {axis: value["mean_max_envelope_stable_seconds"] for axis, value in result["groups"].items()},
                           "phase_failure_counts": result["phase_failure_counts"],
                           "failure_counts": result.get("failure_counts"),
                           "consecutive_axis_passes": dict(self.promotion.streaks),
                           "target_angles": {a:config.angle_for(a) for a in config.training_axes},
                           "promote_axes": list(self.promotion.ready_axes),
                           "checkpoint": str(checkpoint),
                           "ppo": {str(k): float(v) for k, v in self.model.logger.name_to_value.items()
                                   if isinstance(v, (int, float)) and __import__('math').isfinite(v)}}
                _append(output / "progress.jsonl", summary)
                print(json.dumps(summary), flush=True)
                run.update(latest_checkpoint=str(checkpoint), latest_validation=summary)
                atomic_json(output / "status.json", run)
                self.next_evaluation += args.eval_every
                if promoted and not args.keep_training_after_validation:
                    self.reason = "final_targets_validated" if self.promotion.final_pass else "angle_stage_ready"
                    run.update(promote_axes=list(self.promotion.ready_axes), final_targets_validated=self.promotion.final_pass)
                    return False
                if result["interrupted"]:
                    self.reason = stop["reason"] or "wall_clock_budget"
                    return False
            return True

    callback = TrainingCallback()
    try:
        baseline_reports = {}
        if not args.resume:
            for baseline in ("zero", "random"):
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
        run["status"] = "completed" if callback.reason in ("angle_stage_ready", "final_targets_validated") else "stopped"
        run["stop_reason"] = callback.reason
    except BaseException as exc:
        run.update(status="error", stop_reason=type(exc).__name__, error=str(exc))
        raise
    finally:
        checkpoint = save_checkpoint(model, normalizer, reference, config, output,
                                     elapsed_seconds=time.monotonic()-started,
                                     reason=run.get("stop_reason", "interrupted"),
                                     score=validation_score(callback.last_evaluation), initialization=initialization)
        run.update(training_steps=int(model.num_timesteps), elapsed_seconds=time.monotonic()-started,
                   latest_checkpoint=str(checkpoint),
                   finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(output / "status.json", run)
        normalizer.close()
        print(json.dumps(run, indent=2), flush=True)


if __name__ == "__main__":
    main()
