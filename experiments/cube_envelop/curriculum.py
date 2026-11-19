"""Run bounded five-finger holding, independent verification, then XYZ turns.

The parent deadline includes training, baseline evaluation, and the independent
hold check. Deadline/SIGTERM stops the active child before any further stage;
only bounded cleanup time is allowed afterward to finish checkpoint writes.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


@dataclass(frozen=True)
class Budget:
    start: float
    seconds: float
    steps: int

    @property
    def deadline(self):
        return self.start + self.seconds

    def remaining_seconds(self, now):
        return max(0., self.deadline - now)

    def remaining_steps(self, consumed):
        return max(0, self.steps - int(consumed))

    def stage_seconds(self, now, cap_seconds=None):
        remaining = self.remaining_seconds(now)
        return remaining if cap_seconds is None else min(remaining, cap_seconds)


def hold_training_passed(status):
    """Promotion requires the two-check decision and its final complete result."""
    if not isinstance(status, dict):
        return False
    result = status.get("latest_validation") or {}
    return bool(
        status.get("status") == "completed"
        and status.get("stop_reason") == "hold_two_validation_checks_at_least_90_percent"
        and result.get("validation_complete") is True
        and (result.get("consecutive_axis_passes") or {}).get("hold", 0) >= 2
        and (result.get("group_success_rates") or {}).get("hold", 0.) >= .9
    )


def independent_hold_passed(report):
    """Exactly 100 held-out attempts must finish; capture alone is insufficient."""
    if not isinstance(report, dict):
        return False
    return bool(
        report.get("task_mode") == "hold"
        and report.get("complete") is True
        and report.get("episodes_requested") == 100
        and report.get("episodes_completed") == 100
        and report.get("successes", 0) >= 90
        and report.get("success_rate", 0.) >= .9
        and report.get("seed_start") == 40000
    )


class CurriculumRunner:
    def __init__(self, reference, output, *, hours=2., steps=500000, envs=4,
                 seed=0, baseline_episodes=5, hold_minutes=30., config=None,
                 clock=time.monotonic, sleeper=time.sleep,
                 popen=subprocess.Popen, poll_seconds=.2, grace_seconds=30.):
        if not math.isfinite(hours) or not 0 < hours <= 2:
            raise ValueError("Curriculum budget must be positive and at most 2 hours")
        if not 0 < steps <= 500000 or not 1 <= envs <= 4:
            raise ValueError("Curriculum is limited to 500000 steps and four environments")
        if not math.isfinite(hold_minutes) or not 0 < hold_minutes <= 30:
            raise ValueError("Holding-stage cap must be positive and at most 30 minutes")
        if baseline_episodes < 0 or poll_seconds <= 0 or grace_seconds < 0:
            raise ValueError("Invalid baseline/poll/cleanup setting")
        self.reference = Path(reference).resolve()
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "status.json").exists():
            raise ValueError("Curriculum output already contains a run; choose a new directory")
        if not self.reference.is_file():
            raise FileNotFoundError(self.reference)
        self.config_path = Path(config).resolve() if config else None
        if self.config_path and not self.config_path.is_file():
            raise FileNotFoundError(self.config_path)
        self.clock, self.sleeper, self.popen = clock, sleeper, popen
        self.poll_seconds, self.grace_seconds = poll_seconds, grace_seconds
        self.envs, self.seed = envs, seed
        self.baseline_episodes, self.hold_minutes = baseline_episodes, hold_minutes
        self.budget = Budget(clock(), hours * 3600., steps)
        self.stop_reason = None
        self.child = None
        self.status = {
            "task": "cube_envelope_hold_then_xyz90_v1", "status": "initializing",
            "pid": os.getpid(), "started_at": _now(), "budget_hours": hours,
            "deadline_at": (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat(),
            "total_step_budget": steps, "hold_stage_minutes": hold_minutes,
            "reference": str(self.reference), "parallel_environments": envs,
            "config": str(self.config_path) if self.config_path else None,
            "active_stage": None, "training_steps": 0, "stages": {},
            "deadline_policy": "At the total deadline send SIGTERM to the active child; allow at most 30 seconds solely for checkpoint cleanup; never launch a further stage.",
            "paths": {name: str(self.output / name) for name in ("hold", "verify_hold", "reorient")},
        }
        self._write_status()

    def _write_status(self):
        self.status.update(elapsed_seconds=self.clock() - self.budget.start,
                           remaining_seconds=self.budget.remaining_seconds(self.clock()))
        _atomic_json(self.output / "status.json", self.status)

    def request_stop(self, signum=signal.SIGTERM):
        self.stop_reason = signal.Signals(signum).name
        if self.child is not None and self.child.poll() is None:
            # Only signal the trainer first: its workers must stay alive while
            # the trainer saves and closes the vector environment gracefully.
            self.child.send_signal(signum)

    def _finish(self, reason, status="stopped"):
        self.status.update(status=status, stop_reason=reason, active_stage=None,
                           child_pid=None, finished_at=_now())
        self._write_status()
        return self.status

    def _allowed(self):
        return self.stop_reason is None and self.budget.remaining_seconds(self.clock()) > 0

    def _train_command(self, output, mode, hours, steps, eval_every, init=None, config=None):
        command = [sys.executable, "-m", "experiments.cube_envelop.train",
                   "--reference", str(self.reference), "--output", str(output),
                   "--task-mode", mode, "--hours", str(hours), "--steps", str(steps),
                   "--envs", str(self.envs), "--seed", str(self.seed),
                   "--eval-every", str(eval_every), "--eval-episodes", "20",
                   "--baseline-episodes", str(self.baseline_episodes if mode == "hold" else 0)]
        if init:
            command += ["--init-checkpoint", str(init)]
        if config:
            command += ["--config", str(config)]
        return command

    def _run_child(self, name, command, *, stage_deadline):
        """Run one child; save live paths and enforce both stage and parent limits."""
        stage_output = self.output / name
        stage_output.mkdir(parents=True, exist_ok=True)
        log_path = self.output / (name + ".log")
        started = self.clock()
        completed_stage_steps = int(self.status.get("training_steps", 0))
        stage = {"status": "running", "started_at": _now(), "command": command,
                 "log": str(log_path), "status_file": str(stage_output / "status.json"),
                 "progress_file": str(stage_output / "progress.jsonl"),
                 "output": str(stage_output), "budget_seconds": max(0., stage_deadline-started)}
        self.status["stages"][name] = stage
        self.status.update(status="running", active_stage=name, current_stage_training_steps=0)
        self._write_status()
        termination_at = None
        termination_reason = None
        forced = False
        last_update = 0.
        with log_path.open("w") as log:
            self.child = self.popen(command, stdout=log, stderr=subprocess.STDOUT,
                                    cwd=str(Path(__file__).resolve().parents[2]),
                                    start_new_session=True)
            stage["pid"] = self.child.pid
            self.status["child_pid"] = self.child.pid
            self._write_status()
            try:
                while self.child.poll() is None:
                    now = self.clock()
                    reason = self.stop_reason
                    if reason is None and now >= self.budget.deadline:
                        reason = "total_wall_clock_budget"
                    if reason is None and now >= stage_deadline:
                        reason = "stage_wall_clock_budget"
                    if reason is not None and termination_at is None:
                        termination_at, termination_reason = now, reason
                        self.child.send_signal(signal.SIGTERM)
                        stage.update(status="stopping", termination_reason=reason,
                                     termination_requested_at=_now())
                        self._write_status()
                    if termination_at is not None and now - termination_at >= self.grace_seconds:
                        try:
                            os.killpg(self.child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        self.child.wait()
                        forced = True
                        break
                    if now - last_update >= 5.:
                        child_status = _read_json(stage_output / "status.json")
                        if child_status:
                            stage["live"] = {key: child_status.get(key) for key in
                                             ("status", "training_steps", "elapsed_seconds", "latest_checkpoint", "latest_validation")}
                            if name in ("hold", "reorient"):
                                current_steps = int(child_status.get("training_steps", 0))
                                self.status.update(current_stage_training_steps=current_steps,
                                                   training_steps=completed_stage_steps + current_steps)
                        self._write_status()
                        last_update = now
                    self.sleeper(self.poll_seconds)
                returncode = self.child.wait()
            except BaseException:
                if self.child.poll() is None:
                    self.child.send_signal(signal.SIGTERM)
                    try:
                        self.child.wait(timeout=self.grace_seconds)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(self.child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        self.child.wait()
                raise
            finally:
                self.child = None
        stage.update(status="finished" if returncode == 0 else "error", returncode=returncode,
                     elapsed_seconds=self.clock()-started, finished_at=_now(),
                     forced_kill=forced, termination_reason=termination_reason)
        self.status["child_pid"] = None
        self._write_status()
        return returncode

    def run(self):
        try:
            if not self._allowed():
                return self._finish(self.stop_reason or "total_wall_clock_budget")
            hold_output = self.output / "hold"
            seconds = self.budget.stage_seconds(self.clock(), self.hold_minutes * 60.)
            code = self._run_child("hold", self._train_command(
                hold_output, "hold", seconds / 3600., self.budget.steps, 4096,
                config=self.config_path),
                stage_deadline=min(self.budget.deadline, self.clock()+seconds))
            hold_status = _read_json(hold_output / "status.json")
            held_steps = int((hold_status or {}).get("training_steps", 0))
            self.status.update(training_steps=held_steps, hold_training_steps=held_steps)
            self.status["hold_training_result"] = hold_status
            if not self._allowed():
                return self._finish(self.stop_reason or "total_wall_clock_budget")
            if code != 0:
                return self._finish("hold_child_failed", "error")
            if not hold_training_passed(hold_status):
                return self._finish("hold_validation_gate_not_met")
            remaining_steps = self.budget.remaining_steps(held_steps)
            if remaining_steps == 0:
                return self._finish("total_step_budget")

            checkpoint = hold_output / "latest"
            verify_output = self.output / "verify_hold"
            command = [sys.executable, "-m", "experiments.cube_envelop.evaluate",
                       "--checkpoint", str(checkpoint), "--reference", str(self.reference),
                       "--output", str(verify_output), "--episodes", "100", "--seed-start", "40000"]
            code = self._run_child("verify_hold", command, stage_deadline=self.budget.deadline)
            report = _read_json(verify_output / "evaluation.json")
            self.status["hold_independent_evaluation"] = {
                key: (report or {}).get(key) for key in
                ("task_mode", "complete", "episodes_completed", "successes", "success_rate", "capture_rate", "seed_start")}
            self.status["hold_independent_evaluation"]["report"] = str(verify_output / "evaluation.json")
            if not self._allowed():
                return self._finish(self.stop_reason or "total_wall_clock_budget")
            if code != 0:
                return self._finish("hold_evaluation_child_failed", "error")
            if not independent_hold_passed(report):
                return self._finish("hold_independent_100_episode_gate_not_met")

            remaining = self.budget.remaining_seconds(self.clock())
            reorient_output = self.output / "reorient"
            initialization = _read_json(checkpoint / "metadata.json")
            if not initialization or initialization.get("task_mode") != "hold":
                return self._finish("hold_initialization_checkpoint_missing_or_wrong_mode", "error")
            config_path = self.output / "reorient_init_config.json"
            _atomic_json(config_path, initialization["config"])
            self.status["reorient_initialization"] = {
                "checkpoint": str(checkpoint.resolve()),
                "source_training_steps": initialization.get("training_steps"),
                "source_reference_sha256": (initialization.get("files") or {}).get("reference.npz"),
                "config": str(config_path),
                "semantics": "Copy hold policy and observation statistics; initialize a new PPO optimizer and reset stage training steps to zero.",
            }
            code = self._run_child("reorient", self._train_command(
                reorient_output, "reorient", remaining / 3600., remaining_steps, 32768,
                init=checkpoint, config=config_path),
                stage_deadline=self.budget.deadline)
            reorient_status = _read_json(reorient_output / "status.json")
            self.status["reorient_training_result"] = reorient_status
            self.status["training_steps"] = held_steps + int((reorient_status or {}).get("training_steps", 0))
            if not self._allowed():
                return self._finish(self.stop_reason or "total_wall_clock_budget")
            if code != 0:
                return self._finish("reorient_child_failed", "error")
            if (reorient_status or {}).get("stop_reason") == "all_axes_two_validation_checks_at_least_90_percent":
                return self._finish("reorient_validation_goal_reached", "completed")
            return self._finish("reorient_stage_stopped")
        except BaseException as exc:
            if self.child is not None and self.child.poll() is None:
                self.child.send_signal(signal.SIGTERM)
            self.status["error"] = f"{type(exc).__name__}: {exc}"
            self._finish("curriculum_error", "error")
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hours", type=float, default=2.)
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline-episodes", type=int, default=5)
    parser.add_argument("--hold-minutes", type=float, default=30.)
    parser.add_argument("--config", help="Optional task config for the initial hold stage")
    args = parser.parse_args(argv)
    runner = CurriculumRunner(**vars(args))
    prior = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in prior:
        signal.signal(sig, lambda signum, frame: runner.request_stop(signum))
    try:
        result = runner.run()
        print(json.dumps(result, indent=2), flush=True)
        return 1 if result["status"] == "error" else 0
    finally:
        for sig, handler in prior.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
