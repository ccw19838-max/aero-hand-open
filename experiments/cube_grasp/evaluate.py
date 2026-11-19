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
    paths.append(here.parents[1] / "sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/xmls/right_hand.xml")
    for path in paths:
        digest.update(path.name.encode())
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


def _environment(reference, config, render_mode=None):
    from .env import CubeGraspEnv
    return CubeGraspEnv(reference=reference, config=config, render_mode=render_mode)


def load_checkpoint(checkpoint, reference=None):
    """Return model, frozen normalizer, task config, and bundled reference."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from .config import CubeGraspConfig

    checkpoint = Path(checkpoint).resolve()
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported cube-grasp checkpoint format")
    if metadata["implementation_sha256"] != implementation_digest():
        raise ValueError("Checkpoint observation/action/model code differs from the current task")
    for name, expected in metadata["files"].items():
        if file_digest(checkpoint / name) != expected:
            raise ValueError(f"Checkpoint file failed SHA256 verification: {name}")
    reference = Path(reference).resolve() if reference else checkpoint / "reference.npz"
    if file_digest(reference) != metadata["files"]["reference.npz"]:
        raise ValueError("Evaluation reference differs from the checkpoint reference")
    config = CubeGraspConfig.from_dict(metadata["config"])
    holder = DummyVecEnv([lambda: _environment(reference, config)])
    normalizer = VecNormalize.load(checkpoint / "vecnormalize.pkl", holder)
    normalizer.training = False
    normalizer.norm_reward = False
    model = PPO.load(checkpoint / "policy.zip", device="cpu")
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
    return {
        "episodes_requested": int(requested_episodes),
        "episodes_completed": completed,
        "seed_start": int(seed_start),
        "seeds": [int(record["seed"]) for record in records],
        "successes": sum(record["success"] for record in records),
        "success_rate": float(np.mean([record["success"] for record in records])) if records else None,
        "grasp_acquired_rate": float(np.mean([record.get("grasp_acquired", False) for record in records])) if records else None,
        "grasp_acquired_definition": "At least 0.1 seconds of consecutive stability under the task's contact criteria; reset may already have finger contacts.",
        "mean_duration_seconds": float(np.mean([record["duration_seconds"] for record in records])) if records else None,
        "mean_max_stable_seconds": float(np.mean([record["max_stable_seconds"] for record in records])) if records else None,
        "mean_max_displacement_m": float(np.mean([record["max_displacement_m"] for record in records])) if records else None,
        "mean_action_change_squared": float(np.mean([record["mean_action_change_squared"] for record in records])) if records else None,
        "peak_penetration_m": max((record["peak_penetration_m"] for record in records), default=0.),
        "failure_counts": dict(Counter(record["failure_reason"] or "unsuccessful_timeout" for record in records if not record["success"])),
        "complete": completed == requested_episodes,
        "records": records,
    }


def evaluate_policy(model, normalizer, reference, config, *, episodes=20,
                    seed_start=10000, baseline=None, output=None, deadline=None,
                    should_stop=None, trajectories=False):
    """Evaluate independent, seeded native episodes without touching training stats."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    env = _environment(reference, config)
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
            observation, info = env.reset(seed=seed)
            initial_contacts = int(info.get("finger_contacts", 0))
            acquired = False
            if trajectory:
                row = {"episode": episode, "seed": seed, "step": 0, "sim_time": 0.,
                       "event": "reset", "observation": observation.tolist(), "info": _json_value(info)}
                if hasattr(env, "data"):
                    row.update(qpos=env.data.qpos.tolist(), qvel=env.data.qvel.tolist())
                trajectory.write(json.dumps(row, allow_nan=False) + "\n")
            rng = np.random.default_rng(seed + 1_000_000)
            previous = np.zeros(env.action_space.shape, dtype=np.float32)
            action_changes = []
            stable = displacement = penetration = 0.
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
                acquired = acquired or bool(info.get("grasp_acquired", False))
                stable = max(stable, float(info.get("max_stable_seconds", info.get("stable_seconds", 0.))))
                displacement = max(displacement, float(info.get("cube_displacement", 0.)))
                penetration = max(penetration, float(info.get("peak_penetration", 0.)))
                if trajectory:
                    row = {"episode": episode, "seed": seed, "step": step + 1,
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
            records.append({"seed": seed, "success": bool(info.get("success", False)),
                            "grasp_acquired": acquired, "initial_finger_contacts": initial_contacts,
                            "failure_reason": str(info.get("failure_reason") or ""),
                            "steps": step + 1, "duration_seconds": float(info.get("elapsed_seconds", (step + 1) * ctrl_dt)),
                            "max_stable_seconds": stable, "max_displacement_m": displacement,
                            "peak_penetration_m": penetration,
                            "mean_action_change_squared": float(np.mean(action_changes)),
                            "final_info": info})
    finally:
        env.close()
        if trajectory:
            trajectory.close()
    result = summarize_episodes(records, episodes, seed_start)
    result.update(baseline=baseline, interrupted=interrupted,
                  elapsed_seconds=time.monotonic() - start,
                  task="upright_rigid_rubik_cube_grasp", engine="native_mujoco")
    if output:
        atomic_json(output / "evaluation.json", result)
    return result


def record_video(model, normalizer, reference, config, destination, *, seconds=30., seed_start=30000):
    """Real simulation time; episode resets are labelled and listed in a sidecar."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    if seconds <= 0:
        raise ValueError("Video duration must be positive")
    env = _environment(reference, config, render_mode="rgb_array")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fps = 20
    frame_count = int(round(seconds * fps))
    ctrl_dt = float(getattr(config, "ctrl_dt", .05))
    simulation_time = 0.
    episode_time = 0.
    observation, info = env.reset(seed=seed_start)
    episode = 0
    resets = [{"video_seconds": 0., "episode": 0, "seed": seed_start}]
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
                resets_before = len(resets)
                while simulation_time < (frame + 1) / fps - 1e-9:
                    if terminal:
                        episode += 1
                        observation, info = env.reset(seed=seed_start + episode)
                        episode_time = 0.
                        resets.append({"video_seconds": simulation_time, "episode": episode,
                                       "seed": seed_start + episode, "previous_outcome": terminal_info})
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
                    raise RuntimeError("CubeGraspEnv.render() did not return an RGB image")
                canvas = Image.fromarray(np.asarray(frame_rgb, dtype=np.uint8))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, canvas.width, 82), fill=(20, 24, 31))
                outcome = "SUCCESS" if info.get("success") else str(info.get("failure_reason") or "GRASPING")
                reset_label = " | RESET" if len(resets) != resets_before or frame == 0 else ""
                draw.text((12, 8), f"Upright cube grasp | Episode {episode + 1} | {outcome}{reset_label}", font=font, fill="white")
                draw.text((12, 36), f"Real time {video_time:05.2f}s | Stable {info.get('stable_seconds', 0.):.2f}s | Multi-episode replay", font=font, fill="white")
                writer.append_data(np.asarray(canvas))
    finally:
        env.close()
    metadata = {"seconds": frame_count / fps, "simulation_seconds": simulation_time,
                "sampling_tolerance_seconds": ctrl_dt, "fps": fps, "frames": frame_count,
                "episodes_shown": episode + 1, "resets": resets,
                "note": "Resets are separate attempts; this is not a continuous successful hold."}
    atomic_json(destination.with_suffix(".json"), metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--video", type=float, default=0., metavar="SECONDS")
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    model, normalizer, config, reference, metadata = load_checkpoint(args.checkpoint, args.reference)
    try:
        report = evaluate_policy(model, normalizer, reference, config,
                                 episodes=args.episodes, seed_start=args.seed_start,
                                 output=args.output, trajectories=True)
        report.update(checkpoint=str(Path(args.checkpoint).resolve()),
                      training_steps=metadata["training_steps"],
                      checkpoint_reference_sha256=metadata["files"]["reference.npz"])
        if args.video:
            report["video"] = record_video(model, normalizer, reference, config,
                                           Path(args.output) / "replay.mp4", seconds=args.video)
        atomic_json(Path(args.output) / "evaluation.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    finally:
        normalizer.close()


if __name__ == "__main__":
    main()
