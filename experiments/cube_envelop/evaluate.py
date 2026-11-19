"""Native MuJoCo evaluation, shared normalization, and honest multi-episode video."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

AXES = ("x", "y", "z")
AXIS_SEED_STRIDE = 1_000_000
FINGERS = ("thumb", "index", "middle", "ring", "pinky")

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_digest():
    """Bind checkpoint semantics to the observation, action, and physical model."""
    digest = hashlib.sha256()
    here = Path(__file__).resolve().parent
    paths = [here / name for name in ("config.py", "model.py", "env.py")]
    paths.extend(here.parent / "cube_reorient" / name for name in ("model.py", "config.py", "env.py"))
    paths.extend(here.parent / "cube_grasp" / name for name in ("model.py", "config.py"))
    paths.append(here.parents[1] / "sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/xmls/right_hand.xml")
    for path in paths:
        digest.update(str(path.relative_to(here.parents[1])).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def config_dict(config):
    return config.to_dict() if hasattr(config, "to_dict") else asdict(config)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def selected_axes(axis):
    if axis == "all":
        return AXES
    if axis not in AXES:
        raise ValueError("axis must be x, y, z, or all")
    return (axis,)


def _environment(reference, config, render_mode=None, axis=None):
    from .env import CubeReorientEnv
    return CubeReorientEnv(reference=reference, config=config, render_mode=render_mode, axis=axis)


def load_checkpoint(checkpoint, reference=None):
    """Return model, frozen normalizer, task config, and bundled reference."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from .config import CubeReorientConfig

    checkpoint = Path(checkpoint).resolve()
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    if metadata.get("format_version") != 1 or metadata.get("task") != "cube_envelope_xyz90_v1":
        raise ValueError("This task requires a new 78-observation envelope checkpoint; old 75-observation policies cannot be resumed")
    if metadata.get("observation_size") != 78:
        raise ValueError("Envelope checkpoint must have 78 observations")
    if metadata["implementation_sha256"] != implementation_digest():
        raise ValueError("Checkpoint observation/action/model code differs from the current task")
    for name, expected in metadata["files"].items():
        if file_digest(checkpoint / name) != expected:
            raise ValueError(f"Checkpoint file failed SHA256 verification: {name}")
    reference = Path(reference).resolve() if reference else checkpoint / "reference.npz"
    if file_digest(reference) != metadata["files"]["reference.npz"]:
        raise ValueError("Evaluation reference differs from the checkpoint reference")
    config = CubeReorientConfig.from_dict(metadata["config"])
    holder = DummyVecEnv([lambda: _environment(reference, config)])
    normalizer = VecNormalize.load(checkpoint / "vecnormalize.pkl", holder)
    normalizer.training = False
    normalizer.norm_reward = False
    model = PPO.load(checkpoint / "policy.zip", device="cpu")
    if tuple(model.observation_space.shape) != (78,) or tuple(model.action_space.shape) != (7,):
        normalizer.close()
        raise ValueError("Envelope policy must have 78 observations and 7 actions")
    return model, normalizer, config, reference, metadata


def policy_action(model, normalizer, observation):
    normalized = normalizer.normalize_obs(np.asarray(observation, dtype=np.float32))
    action, _ = model.predict(normalized, deterministic=True)
    return np.asarray(action, dtype=np.float32)


def baseline_action(env, name, rng):
    if name == "zero":
        return np.zeros(env.action_space.shape, dtype=np.float32)
    if name == "random":
        return rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
    if name == "closure" and hasattr(env, "baseline_action"):
        return np.asarray(env.baseline_action("close"), dtype=np.float32)
    raise ValueError(f"Environment does not implement the {name!r} baseline")


def summarize_episodes(records, requested_episodes, seed_start):
    completed = len(records)
    duration = sum(record["duration_seconds"] for record in records)
    return {
        "episodes_requested": int(requested_episodes),
        "episodes_completed": completed,
        "seed_start": int(seed_start),
        "seeds": [int(record["seed"]) for record in records],
        "successes": sum(record["success"] for record in records),
        "success_rate": float(np.mean([record["success"] for record in records])) if records else None,
        "mean_final_angle_error_degrees": float(np.mean([record["final_angle_error_degrees"] for record in records])) if records else None,
        "mean_best_angle_error_degrees": float(np.mean([record["best_angle_error_degrees"] for record in records])) if records else None,
        "mean_signed_progress_degrees": float(np.mean([record["signed_progress_degrees"] for record in records])) if records else None,
        "drop_rate": float(np.mean([record["dropped"] for record in records])) if records else None,
        "capture_rate": float(np.mean([record["capture_succeeded"] for record in records])) if records else None,
        "recapture_rate": float(np.mean([record["recapture_succeeded"] for record in records])) if records else None,
        "contact_duty_cycle": {finger: sum(record["contact_seconds"][index] for record in records) / duration
                               if duration > 0 else None for index, finger in enumerate(FINGERS)},
        "contact_duty_cycle_definition": "Total physical-substep contact time above the configured force threshold divided by total completed episode time.",
        "capture_definition": "All five fingers continuously satisfy the initial stable envelope criterion for capture_seconds.",
        "recapture_definition": "After entering the rotation phase and reaching the goal, all five fingers satisfy the stable goal recapture criterion; initial capture does not count.",
        "failure_phase_counts": dict(Counter(record["failure_phase"] or "unknown" for record in records if not record["success"])),
        "phase_failure_counts": dict(Counter((record["failure_phase"] or "unknown") + ":" +
                                             (record["failure_reason"] or "unsuccessful_timeout")
                                             for record in records if not record["success"])),
        "mean_duration_seconds": float(np.mean([record["duration_seconds"] for record in records])) if records else None,
        "mean_max_stable_seconds": float(np.mean([record["max_stable_seconds"] for record in records])) if records else None,
        "mean_max_envelope_stable_seconds": float(np.mean([record.get("max_envelope_stable_seconds", 0.) for record in records])) if records else None,
        "mean_max_hold_stable_seconds": float(np.mean([record.get("max_hold_stable_seconds", 0.) for record in records])) if records else None,
        "mean_max_displacement_m": float(np.mean([record["max_displacement_m"] for record in records])) if records else None,
        "mean_action_change_squared": float(np.mean([record["mean_action_change_squared"] for record in records])) if records else None,
        "peak_penetration_m": max((record["peak_penetration_m"] for record in records), default=0.),
        "failure_counts": dict(Counter(record["failure_reason"] or "unsuccessful_timeout" for record in records if not record["success"])),
        "complete": completed == requested_episodes,
        "records": records,
    }


def _evaluate_axis(model, normalizer, reference, config, *, axis, episodes=20,
                    seed_start=10000, baseline=None, output=None, deadline=None,
                    should_stop=None, trajectories=False):
    """Evaluate independent, seeded native episodes without touching training stats."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    physical_axis = "x" if axis == "hold" else axis
    env = _environment(reference, config, axis=physical_axis)
    if baseline == "closure" and not hasattr(env, "baseline_action"):
        env.close()
        return {"baseline": baseline, "skipped": True, "reason": "No closure baseline supplied by environment"}
    output = Path(output) if output else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    trajectory = (output / "trajectories.jsonl").open("w") if trajectories and output else None
    records = []
    interrupted = False
    start = time.monotonic()
    ctrl_dt = float(getattr(config, "ctrl_dt", .05))
    episode_limit = int(np.ceil(float(getattr(config, "episode_seconds", 10.)) / ctrl_dt)) + 2
    try:
        for episode in range(episodes):
            if (deadline is not None and time.monotonic() >= deadline) or (should_stop and should_stop()):
                interrupted = True
                break
            seed = seed_start + episode
            observation, info = env.reset(seed=seed, options={"axis": physical_axis})
            initial_contacts = int(info.get("finger_contacts", 0))
            if trajectory:
                row = {"axis": axis, "episode": episode, "seed": seed, "step": 0, "sim_time": 0.,
                       "event": "reset", "observation": observation.tolist(), "info": _json_value(info)}
                if hasattr(env, "data"):
                    row.update(qpos=env.data.qpos.tolist(), qvel=env.data.qvel.tolist())
                trajectory.write(json.dumps(row, allow_nan=False) + "\n")
            rng = np.random.default_rng(seed + 1_000_000)
            previous = np.zeros(env.action_space.shape, dtype=np.float32)
            action_changes = []
            stable = displacement = penetration = 0.
            envelope_stable = hold_stable = 0.
            best_error = float(info.get("angle_error_degrees", 90.))
            info = dict(info)
            finished = False
            for step in range(episode_limit):
                if (deadline is not None and time.monotonic() >= deadline) or (should_stop and should_stop()):
                    interrupted = True
                    break
                action = baseline_action(env, baseline, rng) if baseline else policy_action(model, normalizer, observation)
                action_changes.append(float(np.mean((action - previous) ** 2)))
                previous = action.copy()
                observation, reward, terminated, truncated, info = env.step(action)
                info = _json_value(info)
                if info.get("axis", physical_axis) != physical_axis:
                    raise RuntimeError("Evaluation environment changed the requested target axis")
                best_error = min(best_error, float(info["angle_error_degrees"]))
                stable = max(stable, float(info.get("max_stable_seconds", info.get("stable_seconds", 0.))))
                envelope_stable = max(envelope_stable, float(info.get("envelope_stable_seconds", 0.)))
                hold_stable = max(hold_stable, float(info.get("hold_stable_seconds", 0.)))
                displacement = max(displacement, float(info.get("cube_displacement", 0.)))
                penetration = max(penetration, float(info.get("peak_penetration", 0.)))
                if trajectory:
                    row = {"axis": axis, "episode": episode, "seed": seed, "step": step + 1,
                           "sim_time": float(info.get("elapsed_seconds", (step + 1) * ctrl_dt)), "observation": observation.tolist(),
                           "action": action.tolist(), "reward": float(reward),
                           "terminated": bool(terminated), "truncated": bool(truncated), "info": info}
                    if hasattr(env, "data"):
                        row.update(qpos=env.data.qpos.tolist(), qvel=env.data.qvel.tolist())
                    trajectory.write(json.dumps(row, allow_nan=False) + "\n")
                if terminated or truncated:
                    finished = True
                    break
            if interrupted:
                break  # An interrupted partial episode is never scored as success/failure.
            if not finished:
                raise RuntimeError("Environment exceeded its configured episode duration")
            contact_seconds = np.asarray(info["contact_seconds"], dtype=float)
            if contact_seconds.shape != (5,) or not np.all(np.isfinite(contact_seconds)):
                raise ValueError("Environment must supply five finite physical contact durations")
            capture = bool(info["capture_succeeded"])
            recapture = bool(info["recapture_succeeded"])
            if recapture and (not capture or getattr(config, "task_mode", "reorient") == "hold"):
                raise ValueError("Initial capture/hold cannot be reported as rotational recapture")
            if info.get("success") and (not capture or (getattr(config, "task_mode", "reorient") == "reorient" and not recapture)):
                raise ValueError("Envelope success is missing its required capture/recapture event")
            records.append({"axis": axis, "seed": seed, "success": bool(info.get("success", False)),
                            "capture_succeeded": capture, "recapture_succeeded": recapture,
                            "contact_seconds": contact_seconds.tolist(),
                            "failure_phase": str(info.get("failure_phase") or ""),
                            "final_phase": str(info["phase"]),
                            "final_angle_error_degrees": float(info["angle_error_degrees"]),
                            "best_angle_error_degrees": float(info.get("best_angle_error_degrees", best_error)),
                            "signed_progress_degrees": float(info.get("signed_progress_degrees", 0.)),
                            "dropped": bool(info.get("dropped", info.get("failure_reason") == "dropped")),
                            "initial_finger_contacts": initial_contacts,
                            "failure_reason": str(info.get("failure_reason") or ""),
                            "steps": step + 1, "duration_seconds": float(info.get("elapsed_seconds", (step + 1) * ctrl_dt)),
                            "max_stable_seconds": stable, "max_displacement_m": displacement,
                            "max_envelope_stable_seconds": envelope_stable,
                            "max_hold_stable_seconds": hold_stable,
                            "peak_penetration_m": penetration,
                            "mean_action_change_squared": float(np.mean(action_changes)),
                            "final_info": info})
    finally:
        env.close()
        if trajectory:
            trajectory.close()
    result = summarize_episodes(records, episodes, seed_start)
    result.update(axis=axis, baseline=baseline, interrupted=interrupted,
                  elapsed_seconds=time.monotonic() - start,
                  task="cube_envelope_xyz90_v1", engine="native_mujoco",
                  task_mode=getattr(config, "task_mode", "reorient"))
    if getattr(config, "task_mode", "reorient") == "hold":
        result["recapture_rate"] = None
        result["recapture_definition"] = "Not applicable in hold-only mode; initial envelope capture is not recapture."
    if output:
        atomic_json(output / "evaluation.json", result)
    return result


def evaluate_policy(model, normalizer, reference, config, *, episodes=20,
                    seed_start=10000, baseline=None, output=None, deadline=None,
                    should_stop=None, trajectories=False, axis="all"):
    """The episode count is PER AXIS; an average cannot satisfy the all-axis gate."""
    task_mode = getattr(config, "task_mode", "reorient")
    axes = ("hold",) if task_mode == "hold" else selected_axes(axis)
    reports = {}
    started = time.monotonic()
    for name in axes:
        reports[name] = _evaluate_axis(
            model, normalizer, reference, config, axis=name, episodes=episodes,
            seed_start=seed_start + (AXES.index(name) if name in AXES else 0) * AXIS_SEED_STRIDE,
            baseline=baseline, output=Path(output) / name if output else None,
            deadline=deadline, should_stop=should_stop, trajectories=trajectories)
    records = [record for report in reports.values() for record in report.get("records", [])]
    result = summarize_episodes(records, episodes * len(axes), seed_start)
    complete = all(report.get("complete", False) for report in reports.values())
    rates = {name: report.get("success_rate") for name, report in reports.items()}
    result.update(task="cube_envelope_xyz90_v1", engine="native_mujoco",
                  task_mode=task_mode, axis_frame="fixed_hand_axes",
                  goal_rotation_degrees=90 if task_mode == "reorient" else None,
                  axis_selection=axis if task_mode == "reorient" else None,
                  episodes_per_group=episodes, groups=reports,
                  axes=reports if task_mode == "reorient" else {},
                  axis_seed_stride=AXIS_SEED_STRIDE,
                  group_success_rates=rates,
                  axis_success_rates=rates if task_mode == "reorient" else {},
                  min_group_success_rate=min(rates.values()) if complete else None,
                  min_axis_success_rate=min(rates.values()) if complete and task_mode == "reorient" else None,
                  all_groups_at_least_90_percent=complete and all(rate >= .9 for rate in rates.values()),
                  all_axes_at_least_90_percent=complete and task_mode == "reorient" and all(rate >= .9 for rate in rates.values()),
                  complete=complete, interrupted=any(report.get("interrupted", False) for report in reports.values()),
                  baseline=baseline, elapsed_seconds=time.monotonic()-started)
    if task_mode == "hold":
        result["recapture_rate"] = None
        result["recapture_definition"] = "Not applicable in hold-only mode; initial envelope capture is not recapture."
        result["hold"] = reports["hold"]
        reports["hold"]["recapture_rate"] = None
        reports["hold"]["recapture_definition"] = result["recapture_definition"]
    if output:
        atomic_json(Path(output) / "evaluation.json", result)
    return result


def record_video(model, normalizer, reference, config, destination, *, seconds=30., seed_start=30000, axis="all"):
    """Real simulation time; episode resets are labelled and listed in a sidecar."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    if seconds <= 0:
        raise ValueError("Video duration must be positive")
    task_mode = getattr(config, "task_mode", "reorient")
    axes = ("x",) if task_mode == "hold" else selected_axes(axis)
    env = _environment(reference, config, render_mode="rgb_array", axis=axes[0])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fps = 20
    frame_count = int(round(seconds * fps))
    if frame_count < len(axes):
        raise ValueError("Video must contain at least one frame for each requested axis")
    ctrl_dt = float(getattr(config, "ctrl_dt", .05))
    simulation_time = 0.
    episode_time = 0.
    current_axis_index = 0
    current_axis = axes[0]
    segment_start = 0.
    observation, info = env.reset(seed=seed_start + AXES.index(current_axis) * AXIS_SEED_STRIDE,
                                  options={"axis": current_axis})
    episode = 0
    resets = [{"video_seconds": 0., "episode": 0, "axis": current_axis,
               "seed": seed_start + AXES.index(current_axis) * AXIS_SEED_STRIDE}]
    terminal = False
    terminal_info = None
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    try:
        with imageio.get_writer(destination, fps=fps, codec="libx264", quality=7, macro_block_size=1) as writer:
            for frame in range(frame_count):
                video_time = frame / fps
                segment = min(len(axes)-1, int(frame * len(axes) / frame_count))
                if segment != current_axis_index:
                    current_axis_index = segment
                    current_axis = axes[segment]
                    segment_start = video_time
                    episode += 1
                    seed = seed_start + AXES.index(current_axis) * AXIS_SEED_STRIDE + episode
                    observation, info = env.reset(seed=seed, options={"axis": current_axis})
                    episode_time = 0.
                    terminal = False
                    resets.append({"video_seconds": video_time, "episode": episode,
                                   "axis": current_axis, "seed": seed, "reason": "axis_segment_change"})
                resets_before = len(resets)
                while simulation_time < (frame + 1) / fps - 1e-9:
                    if terminal:
                        episode += 1
                        seed = seed_start + AXES.index(current_axis) * AXIS_SEED_STRIDE + episode
                        observation, info = env.reset(seed=seed, options={"axis": current_axis})
                        episode_time = 0.
                        resets.append({"video_seconds": simulation_time, "episode": episode,
                                       "axis": current_axis, "seed": seed, "previous_outcome": terminal_info})
                        terminal = False
                    action = policy_action(model, normalizer, observation)
                    observation, _, terminated, truncated, info = env.step(action)
                    terminal = bool(terminated or truncated)
                    if terminal:
                        terminal_info = _json_value(info)
                    new_episode_time = float(info.get("elapsed_seconds", episode_time + ctrl_dt))
                    advanced = new_episode_time - episode_time
                    if advanced <= 0:
                        raise RuntimeError("Replay environment did not advance physical simulation time")
                    simulation_time += advanced
                    episode_time = new_episode_time
                frame_rgb = env.render()
                if frame_rgb is None:
                    raise RuntimeError("CubeReorientEnv.render() did not return an RGB image")
                canvas = Image.fromarray(np.asarray(frame_rgb, dtype=np.uint8))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, canvas.width, 108), fill=(20, 24, 31))
                outcome = "SUCCESS" if info.get("success") else str(info.get("failure_reason") or ("HOLDING" if task_mode == "hold" else "TURNING"))
                reset_label = " | RESET" if len(resets) != resets_before or frame == 0 or video_time == segment_start else ""
                phase = str(info.get("phase", "acquire"))
                phase_label = "CAPTURE" if phase == "acquire" else phase.upper()
                target = f"Hand axis {current_axis.upper()} +90deg | Error {info.get('angle_error_degrees', 90.):.1f}deg" if task_mode == "reorient" else "Five-finger envelope HOLD"
                draw.text((12, 8), f"{target} | Episode {episode + 1}", font=font, fill="white")
                draw.text((12, 36), f"{phase_label} | {outcome}{reset_label} | Time {video_time:05.2f}s", font=font, fill="white")
                draw.text((12, 64), f"Envelope {info.get('envelope_stable_seconds', 0.):.2f}s | Capture {bool(info.get('capture_succeeded'))} | Recapture {bool(info.get('recapture_succeeded'))}", font=font, fill="white")
                writer.append_data(np.asarray(canvas))
    finally:
        env.close()
    metadata = {"seconds": frame_count / fps, "simulation_seconds": simulation_time,
                "sampling_tolerance_seconds": ctrl_dt, "fps": fps, "frames": frame_count,
                "task_mode": task_mode, "axis_selection": axis if task_mode == "reorient" else None,
                "axes": list(axes) if task_mode == "reorient" else [], "axis_frame": "fixed_hand_axes",
                "seconds_by_axis": {name: sum(min(len(axes)-1, int(frame * len(axes) / frame_count)) == index
                                             for frame in range(frame_count)) / fps
                                    for index, name in enumerate(axes)},
                "episodes_shown": episode + 1, "resets": resets,
                "note": "Resets are separate attempts. Video includes failed attempts and is not proof of successful rotation."}
    atomic_json(destination.with_suffix(".json"), metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--axis", choices=("all", *AXES), default="all")
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--video", type=float, default=0., metavar="SECONDS")
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    model, normalizer, config, reference, metadata = load_checkpoint(args.checkpoint, args.reference)
    try:
        report = evaluate_policy(model, normalizer, reference, config,
                                 episodes=args.episodes, seed_start=args.seed_start,
                                 output=args.output, trajectories=True, axis=args.axis)
        report.update(checkpoint=str(Path(args.checkpoint).resolve()),
                      training_steps=metadata["training_steps"],
                      checkpoint_reference_sha256=metadata["files"]["reference.npz"])
        if args.video:
            report["video"] = record_video(model, normalizer, reference, config,
                                           Path(args.output) / "replay.mp4", seconds=args.video, axis=args.axis)
        atomic_json(Path(args.output) / "evaluation.json", report)
        print(json.dumps({k: v for k, v in report.items() if k not in ("records", "axes", "groups", "hold")}, indent=2))
    finally:
        normalizer.close()


if __name__ == "__main__":
    main()
