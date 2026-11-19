"""Bind a new 24-hour run to the verified grasp and preserved source weights."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import tempfile

SOURCE_RUN = Path(__file__).resolve().parents[1] / "cube_two_driver_v3/artifacts/thumb_pinky_20261117"
EXPERIMENTS = Path(__file__).resolve().parents[1]
PROTOCOL_SOURCES = (
    "cube_strict_x_v4/engine.py", "cube_strict_x_v4/env.py", "cube_strict_x_v4/contact_history.py",
    "cube_strict_x_v4/policy.py", "cube_two_driver_v3/env.py", "cube_two_driver_v3/roles.py",
    "cube_contact_recovery_v2/env.py", "cube_adaptive_60h/config.py", "cube_adaptive_60h/env.py",
    "cube_adaptive_60h/shaping.py", "cube_adaptive_60h/fast_contacts.py", "cube_adaptive_60h/native_contacts.c",
    "cube_adaptive_60h/policy.py", "cube_contact_ppo/env.py", "cube_contact_ppo/task.py",
    "cube_quarter_turn_v2/task.py", "cube_quarter_turn/env.py", "cube_quarter_turn/task.py",
)


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reward_protocol(profile):
    return {
        "units": "Raw environment rewards before PPO reward normalization. Rate terms multiply elapsed seconds.",
        "support": {
            "fingers": ["index", "middle", "ring"], "force_range_N_inclusive": [0.1, 12.0],
            "sample_ms": 10, "first_violation_latches": True,
            "eligible_peak": "Maximum positive signed X angle up to but excluding the first raw-force violation, at physically valid samples.",
        },
        "progress": {
            "episode_cap": 20, "fraction": "clip(min(signed_angle/goal, 1 - full_pose_error/goal), 0, 1)",
            "credit": "20 * positive increase above the episode high-water mark. Requires eligibility, increasing signed angle and decreasing 3D error.",
            "high_water": "Updated every active sample, including ineligible ones. Reversing then regaining the same angle earns no extra credit.",
            "regression": "-20 * increase_in_3D_error_deg / 90, independent of positive progress credit, while eligible.",
        },
        "driver": {
            "fingers": ["thumb", "pinky"], "episode_cap": 8,
            "torque_scale_Nm": 0.005, "tangential_speed_scale_m_s": 0.005,
            "active": "Requires a new eligible progress increment and all five forces at or below upper limits. Each driver must meet its own minimum to contribute.",
            "quality_per_driver": "clip(axis_moment/0.005, 0, 1) * clip(tangential_speed/0.005, 0, 1)",
            "credit": "8 * eligible_fraction_gain * (0.5 * mean(qualities) + 0.5 * min(qualities)) * warning_factor",
            "warning_factor_if_any": 0.5, "warning_factor_otherwise": 1.0,
        },
        "cap_scope": "Each reset starts a fresh episode at one curriculum goal. The 20 and 8 point caps apply separately per episode; there is no cross-episode stage pool or whole-return cap.",
        "support_shaping": {
            "potential": "10 * min(fraction) + 5 * mean(fraction) + 3 * mean(proximity), over the three support fingers",
            "fraction": "clip(force_N/0.1, 0, 1)", "proximity": "exp(-(max(tip_gap_m,0)/0.02)^2)",
            "reward": "Change in support potential after its first sample, minus dt * (12 * mean((1-fraction)^2) + 8 * max((1-fraction)^2)).",
        },
        "after_first_violation": {
            "recovery_window_s": 0.15, "success_eligible": False,
            "learning_signal": "Only support potential change and three-finger deficit cost inside recovery. Stop on recovery, expiry or inherited failure. Unrecovered termination costs 20.",
            "control_cycle": "Action slew and mount proximity penalties are cancelled on cycles ending after violation. Earlier valid samples in the same cycle retain their rewards.",
        },
        "other_eligible_terms": {
            "success_bonus": 50, "failure_cost": 20, "hold_potential": "20 * change_in_unbroken_hold_seconds / 1.0",
            "coverage": "mean(clip(force/lower_limit, 0, 1)) over five fingers", "coverage_change_weight": 2,
            "time_rate": "-1 - 2*(3D_error_deg/90)^2 + 0.5*coverage",
            "motion_rate": "-0.05*min(off_axis_angular_speed^2,100) - 0.1*min((linear_speed/0.1)^2,25)",
            "excess_force_rate": "-0.2*mean((max(force-upper_limit,0)/upper_limit)^2)",
            "capture_hold_rate": "-mean(five_finger_fractional_deficit^2) during capture (3D error <=20 deg) or hold",
            "near_goal_rate": "2*exp(-(3D_error/tolerance)^2)*coverage^2 for positive goals with valid rotation support and three named fingers",
            "tip_rate": "0.25*mean(exp(-(max(tip_gap_m,0)/0.02)^2)) with valid rotation support",
            "rotation_support": "Thumb and at least two others reach their lower limits.",
            "five_finger_deficit_rate_weight": -profile["contact_weight"],
            "per_finger_overdue_rate_weight": -profile["loss_weight"], "overdue_grace_s": profile["loss_grace"],
            "overdue": "mean(clip(seconds_below_lower_limit - grace_s, 0, 1))",
            "pressure_potential_gain": profile["pressure_potential_gain"], "pressure_saturation_N": [1, 0.5, 0.5, 0.5, 0.5],
            "pressure": "Reward change in gain * mean(clip(force/saturation,0,1)), starting at the second sample.",
            "stall_rate": "-stall_weight * clip((stagnation_seconds - 0.5)/4,0,1)", "stall_weight": profile["stall_weight"],
            "action_slew": "-0.01*mean((applied_action-previous_applied_action)^2) per control cycle",
            "mount_proximity_training_only": "-elapsed_s * 2 * clip((0.005 - clearance_m)/0.005,0,2)^2",
        },
    }


def protocol(root, job, source, metadata, preserved, y):
    return {
        "schema": "strict_x_v4_protocol_1",
        "job_sha256": checksum(root / "job.json"),
        "clock_utc": [job["started_at"], job["deadline_at"]],
        "source_sha256_relative_to_experiments": {name: checksum(EXPERIMENTS / name) for name in PROTOCOL_SOURCES},
        "x_transfer": {
            "path": str(source), "axis": metadata["axis"], "task": metadata["task"], "steps": metadata["training_steps"],
            "metadata_sha256": checksum(source / "metadata.json"), "files_sha256": metadata["files"],
            "init": "Copy legacy normalization and hidden features into both arms. Zero the value head; reinitialize identical action heads and log std. Six new V4 input columns are zero.",
            "action_mean": job["initial_action_mean"], "action_std": job["initial_std"], "max_std": job["max_std"],
        },
        "preserved_y": {
            "manifest": str(preserved), "manifest_sha256": job["preserved_y_manifest_sha256"],
            "original_source": y["source"], "preserved_checkpoint": y["checkpoint"],
            "steps": y["training_steps"], "files_sha256": y["sha256"], "training_allocation": y["training_allocation"],
        },
        "observations": {
            "version": "v3_143_plus_index_middle_ring_margin_then_warning",
            "dims": job["observation_dims"], "dtype": "float32", "prefix": "Unchanged V3 features including previous applied action.",
            "appended_order": ["index_margin", "middle_margin", "ring_margin", "index_warning", "middle_warning", "ring_warning"],
            "force_predictor": {
                "sample_ms": 10, "history_samples": 6, "lookback_ms": 50, "prediction_horizon_ms": 50,
                "prediction_N": "2 * current_force_N - force_50ms_ago_N; use current force until history is full.",
                "margin": "clip((predicted_force_N - 0.1)/0.1, -1, 1)",
                "warn": "Full history and current force <0.3 N and prediction <0.1 N on two consecutive samples.",
                "clear": "Current force >=0.2 N and prediction >=0.1 N for five consecutive samples.",
                "scoring": "Raw forces determine eligibility. For warning statistics a low-force crossing within 50 ms is a hit; pending onsets at termination are censored.",
            },
        },
        "acceptance": {
            "environment_version": "x_strict_three_support_v4", "cube_edge_m": 0.057, "mount_deg": 20,
            "axis": "Positive initial cube-face X, aligned to mount after warmup; moment arms originate at the moving cube COM.",
            "warmup_s": 0.25, "control_period_s": 0.05, "physics_step_per_worker_s": job["physics_timesteps"],
            "action_dims": 7, "max_motor_target_increment_fraction": job["profile"]["action_fraction"],
            "curriculum_deg": [5, 10, 20, 30, 45, 60, 75, 90], "tolerance_deg": "min(5, goal/4)",
            "hold_requires": "Signed angle error and full 3D error each within tolerance; all five raw forces in bounds; unbroken eligible hold.",
            "finger_order": ["thumb", "index", "middle", "ring", "pinky"], "lower_N": [0.2, 0.1, 0.1, 0.1, 0.1],
            "upper_N": [20, 12, 12, 12, 12], "linear_speed_strictly_below_m_s": 0.03,
            "angular_speed_strictly_below_rad_s": 0.5, "hold_s": 1.0, "forbidden_support_terminates_immediately": True,
            "max_penetration_m": 0.001, "max_displacement_m": 0.08, "penetration_or_workspace_grace_s": 0.1,
            "min_total_hand_force_N": 0.02, "max_unsupported_s": 0.2, "insufficient_rotation_support_grace_s": 0.15,
            "stagnation_s": 8, "episode_timeout_s_after_warmup": 12,
        },
        "stage_limits": {
            "overall_wall_hours": 24, "freeze_reserve_s": job["final_seconds"],
            "pilot_max_s_each": job["pilot_arm_seconds"], "pilot_max_steps_each": job["pilot_max_steps"],
            "pilot_steps_compared": job["pilot_milestones"], "pilot_hard_stop_hours_since_start": 6,
            "curriculum_time": "Unused pilot time transfers to curriculum. Stages share the time until freeze; each stage has no fixed training-hour allotment.",
            "checkpoint_every_s": job["save_seconds"], "regular_validation_every_s": job["eval_seconds"],
            "validation_trials_per_physics_step": 20, "successes_required_per_step": 16,
            "consecutive_regular_rounds_required": 2, "plateau_search_max_s_once_per_stage": 600,
            "final_trials_per_step_if_90deg_qualified": 100, "final_successes_required_per_step": 100,
        },
        "reward": reward_protocol(job["profile"]),
    }


def ensure_protocol(path, data):
    """Publish a complete read-only record once. Never replace an existing record."""
    expected = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if not path.exists():
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".protocol-") as temp:
            temp.write(expected)
            temp.flush()
            os.fchmod(temp.fileno(), 0o444)
            os.fsync(temp.fileno())
            try:
                os.link(temp.name, path)
            except FileExistsError:
                pass  # A concurrent preparation can publish the same record.
    if path.is_symlink() or path.read_bytes() != expected or path.stat().st_mode & 0o222:
        raise ValueError(f"Experiment protocol differs or is writable: {path}")


def prepare(root, source_run=SOURCE_RUN):
    root, source_run = Path(root).resolve(), Path(source_run).resolve()
    previous = json.loads((source_run / "job.json").read_text())
    source = Path(previous["source_checkpoint"])
    metadata = json.loads((source / "metadata.json").read_text())
    for name, expected in metadata["files"].items():
        if checksum(source / name) != expected:
            raise ValueError(f"Source checkpoint mismatch: {name}")
    inputs = source_run / "inputs"
    reference_hash = checksum(inputs / "reference.npz")
    if reference_hash != checksum(source / "reference.npz"):
        raise ValueError("Source policy and prepared grasp differ")
    root.mkdir(parents=True, exist_ok=True)
    clock_path = root / "clock.json"
    if not clock_path.exists():
        now = datetime.now(timezone.utc)
        with clock_path.open("x") as stream:
            json.dump(dict(started_at=now.isoformat(), deadline_at=(now + timedelta(hours=24)).isoformat(),
                           source=str(source), policy_sha256=checksum(source/"policy.zip"), reference_sha256=reference_hash), stream, indent=2)
    clock = json.loads(clock_path.read_text())
    if (clock["reference_sha256"],clock["policy_sha256"]) != (reference_hash,checksum(source/"policy.zip")):
        raise ValueError("Inputs changed after the experiment clock started")
    for name in ("reference.npz", "prepare_report.json"):
        dest = root / "inputs" / name
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            shutil.copy2(inputs / name, dest)
        if checksum(dest) != checksum(inputs / name):
            raise ValueError(f"Experiment input mismatch: {name}")
    preserved = Path(previous["preserved_y_manifest"])
    y = json.loads(preserved.read_text())
    for name, expected in y["sha256"].items():
        if checksum(preserved.parent / "checkpoint" / name) != expected:
            raise ValueError(f"Preserved Y mismatch: {name}")
    job = {k: previous[k] for k in ("profile","physics_timesteps","rollout_steps","initial_action_mean","initial_std","max_std")}
    job.update(clock, task="strict_x_v4",root=str(root),source_checkpoint=str(source),preserved_y_manifest=str(preserved),
               preserved_y_manifest_sha256=checksum(preserved),seed=7100000,validation_seed=7200000,review_seed=7300000,
               diagnostic_seed=7400000,final_seed=8100000,initial_stage=5.,pilot_max_steps=32768,
               pilot_milestones=[0,4096,8192,16384,32768],pilot_arm_seconds=7200,eval_seconds=1800,
               save_seconds=300,final_seconds=7200,observation_dims={"v3":143,"v4":149})
    path = root / "job.json"
    if path.exists():
        if json.loads(path.read_text()) != job:
            raise ValueError("The prepared experiment is immutable")
    else:
        with path.open("x") as stream:
            json.dump(job,stream,indent=2)
    ensure_protocol(root / "protocol.json", protocol(root, job, source, metadata, preserved, y))
    return job


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",required=True)
    args=p.parse_args();print(json.dumps(prepare(args.root),indent=2))
