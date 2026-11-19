"""The automatic curriculum must never promote on capture or incomplete tests."""
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

from experiments.cube_envelop.curriculum import (
    Budget, CurriculumRunner, hold_training_passed, independent_hold_passed,
)


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def passed_hold():
    return {"status": "completed", "task_mode": "hold", "training_steps": 8192,
            "stop_reason": "hold_two_validation_checks_at_least_90_percent",
            "latest_validation": {"validation_complete": True, "validation_episodes": 20,
                                  "consecutive_axis_passes": {"hold": 2},
                                  "group_success_rates": {"hold": .9}}}


def hold_evaluation(successes=90, complete=True):
    return {"task_mode": "hold", "complete": complete, "episodes_requested": 100,
            "episodes_completed": 100 if complete else 99,
            "successes": successes, "success_rate": successes/100,
            "seed_start": 40000, "capture_rate": 1.}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


class BudgetTests(unittest.TestCase):
    def test_hold_cap_and_total_remaining_budget(self):
        budget = Budget(100., 7200., 500000)
        self.assertEqual(budget.stage_seconds(100., 1800.), 1800.)
        self.assertEqual(budget.stage_seconds(7100., 1800.), 200.)
        self.assertEqual(budget.remaining_seconds(7400.), 0.)
        self.assertEqual(budget.remaining_steps(8192), 491808)
        self.assertEqual(budget.remaining_steps(500001), 0)

    def test_capture_and_incomplete_test_cannot_promote(self):
        status = passed_hold()
        self.assertTrue(hold_training_passed(status))
        status["stop_reason"] = "wall_clock_budget"
        self.assertFalse(hold_training_passed(status))
        self.assertTrue(independent_hold_passed(hold_evaluation()))
        self.assertFalse(independent_hold_passed(hold_evaluation(89)))
        self.assertFalse(independent_hold_passed(hold_evaluation(99, complete=False)))
        same_validation_seeds = hold_evaluation()
        same_validation_seeds["seed_start"] = 10000
        self.assertFalse(independent_hold_passed(same_validation_seeds))


class CurriculumTests(unittest.TestCase):
    def make_runner(self, root, **kwargs):
        reference = root / "reference.npz"
        reference.write_bytes(b"mock reference")
        clock = Clock()
        runner = CurriculumRunner(reference, root / "run", clock=clock, sleeper=clock.advance,
                                  **kwargs)
        return runner, clock

    def fake_stage(self, runner, clock, calls, hold=None, evaluation=None,
                   hold_seconds=1., evaluation_seconds=1.):
        def child(name, command, *, stage_deadline):
            calls.append((name, command, stage_deadline))
            if name == "hold":
                clock.advance(hold_seconds)
                write_json(runner.output / name / "status.json", hold or passed_hold())
                write_json(runner.output / name / "latest" / "metadata.json", {
                    "task_mode": "hold", "training_steps": 8192,
                    "config": {"task_mode": "hold", "cube_size": .057},
                    "files": {"reference.npz": "mocksha"}})
            elif name == "verify_hold":
                clock.advance(evaluation_seconds)
                write_json(runner.output / name / "evaluation.json", evaluation or hold_evaluation())
            else:
                clock.advance(.1)
                write_json(runner.output / name / "status.json", {
                    "status": "stopped", "stop_reason": "step_budget", "training_steps": 1024})
            return 0
        return child

    def test_hold_failure_stops_before_independent_test_and_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, clock = self.make_runner(Path(directory))
            calls = []
            failed = passed_hold()
            failed.update(status="stopped", stop_reason="wall_clock_budget")
            failed["latest_validation"]["group_success_rates"]["hold"] = .25
            with mock.patch.object(runner, "_run_child", side_effect=self.fake_stage(runner, clock, calls, hold=failed)):
                result = runner.run()
            self.assertEqual([call[0] for call in calls], ["hold"])
            self.assertEqual(result["stop_reason"], "hold_validation_gate_not_met")

    def test_independent_hold_failure_never_starts_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, clock = self.make_runner(Path(directory))
            calls = []
            with mock.patch.object(runner, "_run_child", side_effect=self.fake_stage(
                    runner, clock, calls, evaluation=hold_evaluation(89))):
                result = runner.run()
            self.assertEqual([call[0] for call in calls], ["hold", "verify_hold"])
            self.assertEqual(result["stop_reason"], "hold_independent_100_episode_gate_not_met")

    def test_passed_hold_uses_only_remaining_time_steps_and_explicit_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, clock = self.make_runner(Path(directory))
            calls = []
            with mock.patch.object(runner, "_run_child", side_effect=self.fake_stage(runner, clock, calls)):
                result = runner.run()
            self.assertEqual([call[0] for call in calls], ["hold", "verify_hold", "reorient"])
            hold_command = calls[0][1]
            self.assertEqual(float(hold_command[hold_command.index("--hours")+1]), .5)
            command = calls[2][1]
            self.assertAlmostEqual(float(command[command.index("--hours")+1]), 7198/3600)
            self.assertEqual(int(command[command.index("--steps")+1]), 500000-8192)
            self.assertIn("--init-checkpoint", command)
            self.assertNotIn("--resume", command)
            config = json.loads(Path(command[command.index("--config")+1]).read_text())
            self.assertEqual(config["task_mode"], "hold")
            self.assertEqual(command[command.index("--task-mode")+1], "reorient")
            self.assertEqual(result["training_steps"], 8192+1024)
            self.assertIn("new PPO optimizer", result["reorient_initialization"]["semantics"])

    def test_total_deadline_during_hold_verification_blocks_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, clock = self.make_runner(Path(directory), hours=.001)
            calls = []
            with mock.patch.object(runner, "_run_child", side_effect=self.fake_stage(
                    runner, clock, calls, hold_seconds=1., evaluation_seconds=3.)):
                result = runner.run()
            self.assertEqual([call[0] for call in calls], ["hold", "verify_hold"])
            self.assertEqual(result["stop_reason"], "total_wall_clock_budget")

    def test_stage_timeout_signals_trainer_and_bounds_cleanup(self):
        class Process:
            pid = 987654
            returncode = None

            def __init__(self):
                self.signals = []

            def poll(self):
                return self.returncode

            def send_signal(self, signum):
                self.signals.append(signum)

            def wait(self, timeout=None):
                self.returncode = -9
                return self.returncode

        with tempfile.TemporaryDirectory() as directory:
            process = Process()
            runner, clock = self.make_runner(Path(directory), popen=lambda *a, **k: process,
                                            grace_seconds=.3, poll_seconds=.1)
            start = clock()
            with mock.patch("experiments.cube_envelop.curriculum.os.killpg") as kill:
                code = runner._run_child("hold", ["mock"], stage_deadline=start+.2)
            self.assertEqual(code, -9)
            self.assertEqual(process.signals, [signal.SIGTERM])
            kill.assert_called_once_with(process.pid, signal.SIGKILL)
            self.assertLessEqual(clock()-start, .7)
            self.assertTrue(runner.status["stages"]["hold"]["forced_kill"])
            self.assertIsNone(runner.child)

    def test_live_parent_steps_include_previous_and_current_stage(self):
        class Process:
            pid = 987654

            def __init__(self):
                self.polls = 0

            def poll(self):
                self.polls += 1
                return None if self.polls == 1 else 0

            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory() as directory:
            runner, clock = self.make_runner(Path(directory), popen=lambda *a, **k: Process())
            runner.status["training_steps"] = 8192
            write_json(runner.output / "reorient" / "status.json", {
                "status": "training", "training_steps": 1024, "elapsed_seconds": 1.})
            code = runner._run_child("reorient", ["mock"], stage_deadline=clock()+10.)
            self.assertEqual(code, 0)
            self.assertEqual(runner.status["current_stage_training_steps"], 1024)
            self.assertEqual(runner.status["training_steps"], 8192+1024)


if __name__ == "__main__":
    unittest.main()
