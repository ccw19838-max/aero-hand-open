"""The 30/60/90 amendment must preserve the training boundary and reset qualification."""
from copy import deepcopy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from experiments.cube_adaptive_60h.storage import atomic_json, digest
from .curriculum import ORIGINAL_DEGREES, REVISED_DEGREES, read, record_amendment
from .evaluation import summarize, TIMESTEPS
from .run import Experiment


def validation(target, counts):
    metrics = summarize([])
    metrics.update(median_valid_peak_degrees=target, median_eligible_peak_degrees=target,
                   mean_best_error=1., maximum_stable_seconds=1.)
    return dict(complete=True, target_degrees=target, success_rate=min(counts)/20, **metrics,
                groups={str(dt): dict(complete=True, trials=20, requested_trials=20, successes=count, **metrics)
                        for dt, count in zip(TIMESTEPS, counts)})


class AmendmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)/"experiments"/"strict_x"/"artifacts"/"run"
        self.checkpoint = self.root/"arms"/"v4"/"checkpoints"/"step_040960"
        self.checkpoint.mkdir(parents=True)
        self.latest = self.root/"arms"/"v4"/"latest"
        self.latest.symlink_to(self.checkpoint)
        self.job = dict(profile={}, task="strict_x", validation_seed=123, final_seconds=7200,
                        deadline_at="2099-01-02T00:00:00+00:00", started_at="2099-01-01T00:00:00+00:00",
                        eval_seconds=1800, preserved_y_manifest="/preserved-y")
        atomic_json(self.root/"job.json", self.job)
        self.original = Experiment(self.root).state
        old_result = dict(steps=39936, target=5, counts=[20, 20], rate=1., median=5.)
        self.original.update(phase="curriculum", stage=5, streak=1, steps={"v3":32768, "v4":40960},
                             history={"5":[old_result]}, best={"5":old_result}, qualified={"5":old_result},
                             practice={"5":{"qualified":True}}, milestones={"v3":{"0":old_result}, "v4":{"0":old_result}},
                             last_evaluation=old_result, last_v4_checkpoint=str(self.checkpoint),
                             next_regular={"v4":time.time() + 1800})
        atomic_json(self.root/"state.json", self.original)
        policy = self.checkpoint/"policy.zip"
        policy.write_bytes(b"fixture checkpoint")
        meta = dict(arm="v4", stage=5, training_steps=40960, ppo_training_epochs=160,
                    job_sha256=self.original["job_sha256"], files={"policy.zip":digest(policy)})
        atomic_json(self.checkpoint/"metadata.json", meta)
        record_amendment(self.root, self.original, self.job)
        self.manifest = self.root/"amendment.json"
        self.archive = self.root/"state_before_30_60_90.json"

    def test_boundary_and_original_report_survive_transition(self):
        exp = Experiment(self.root)
        self.assertEqual(read(self.archive), self.original)
        manifest = read(self.manifest)
        self.assertEqual((manifest["training_steps"], manifest["ppo_training_epochs"]), (40960,160))
        self.assertEqual(manifest["previous_curriculum_degrees"], list(ORIGINAL_DEGREES))
        self.assertEqual(manifest["curriculum_degrees"], list(REVISED_DEGREES))
        self.assertEqual(manifest["deadline_at"], exp.job["deadline_at"])
        self.assertEqual(manifest["policy_sha256"], digest(self.checkpoint/"policy.zip"))
        self.assertEqual(manifest["previous_state_sha256"], digest(self.archive))
        for field in ("steps", "milestones", "last_v4_checkpoint", "next_regular", "job_sha256"):
            self.assertEqual(exp.state[field], self.original[field], field)
        self.assertEqual((exp.state["stage"], exp.state["streak"], exp.state["final_quorum"]), (30, 0, False))
        self.assertIsNone(exp.state["last_evaluation"])
        self.assertTrue(all(exp.state[field] == {} for field in ("best","qualified","history","practice")))

    def test_repeated_projection_and_restart_do_not_rewind_new_work(self):
        exp = Experiment(self.root)
        initial = deepcopy(exp.state)
        original_manifest = self.manifest.read_bytes()
        # Crash between immutable manifest publication and state commit.
        self.assertEqual(Experiment(self.root).state, initial)
        exp.state.update(stage=60, streak=1, qualified={"30":{"steps":43008}})
        exp.state["steps"]["v4"] += 3072
        atomic_json(self.root/"state.json", exp.state)
        record_amendment(self.root, exp.state, exp.job)
        self.assertEqual(self.manifest.read_bytes(), original_manifest)
        self.assertEqual(Experiment(self.root).state, exp.state)
        self.assertEqual(read(self.archive), self.original)

    def test_changed_archive_at_first_transition_is_rejected(self):
        self.archive.write_bytes(self.archive.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            Experiment(self.root)

    def test_changed_source_policy_at_first_transition_is_rejected(self):
        (self.checkpoint/"policy.zip").write_bytes(b"replaced")
        with self.assertRaises(ValueError):
            Experiment(self.root)

    def test_changed_manifest_counter_at_first_transition_is_rejected(self):
        manifest = read(self.manifest)
        manifest["ppo_training_epochs"] += 4
        atomic_json(self.manifest, manifest)
        with self.assertRaises(ValueError):
            Experiment(self.root)

    def test_changed_archive_after_resume_is_rejected(self):
        exp = Experiment(self.root)
        atomic_json(self.root/"state.json", exp.state)
        self.archive.write_bytes(self.archive.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            Experiment(self.root)

    def test_original_policy_and_checkpoint_metadata_are_verified_on_resume(self):
        exp = Experiment(self.root)
        atomic_json(self.root/"state.json", exp.state)
        meta_path = self.checkpoint/"metadata.json"
        meta = read(meta_path)
        for field, altered in (("arm", "v3"), ("stage", 60), ("job_sha256", "wrong"),
                               ("training_steps", 41984), ("ppo_training_epochs", 164)):
            with self.subTest(field=field):
                atomic_json(meta_path, {**meta, field:altered})
                with self.assertRaises(ValueError):
                    Experiment(self.root)
        atomic_json(meta_path, meta)
        (self.checkpoint/"policy.zip").write_bytes(b"new source")
        with self.assertRaises(ValueError):
            Experiment(self.root)

    def test_changed_or_missing_manifest_after_resume_is_rejected(self):
        exp = Experiment(self.root)
        atomic_json(self.root/"state.json", exp.state)
        self.manifest.write_bytes(self.manifest.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            Experiment(self.root)
        self.manifest.unlink()
        with self.assertRaises(ValueError):
            Experiment(self.root)

    def test_only_two_distinct_new_qualifying_rounds_promote_each_stage(self):
        exp = Experiment(self.root)
        engine = exp.engine = Mock()
        engine.steps = exp.state["steps"]["v4"]
        engine.epochs = 160
        engine.save.return_value = self.checkpoint
        initial = exp.state["steps"]["v4"]

        def assess(kind, counts, advance=True):
            if advance:
                engine.steps += 1024
                engine.epochs += 4
                exp.state["steps"]["v4"] = engine.steps
            target = exp.state["stage"]
            with patch("experiments.cube_strict_x_v4.run.evaluate", return_value=validation(target,counts)):
                return exp.assess("v4", target, float("inf"), kind)

        assess("entry", (20,20), advance=False)
        self.assertEqual(exp.state["history"], {})
        self.assertEqual(exp.state["streak"], 0)
        for target, next_target in ((30,60),(60,90),(90,90)):
            with self.subTest(target=target):
                self.assertEqual(exp.state["stage"], target)
                # High pooled success still fails if either integrator is below 16/20.
                assess("regular", (16,16))
                self.assertEqual(exp.state["streak"], 1)
                assess("regular", (16,16), advance=False)
                self.assertEqual(exp.state["streak"], 1)
                assess("regular", (20,15))
                self.assertEqual(exp.state["streak"], 0)
                assess("regular", (16,20))
                self.assertEqual(exp.state["stage"],target)
                self.assertFalse(exp.state["final_quorum"])
                assess("regular", (20,16))
                self.assertIn(str(target), exp.state["qualified"])
                self.assertEqual(exp.state["streak"], 0)
                self.assertEqual(exp.state["stage"], next_target)
                self.assertEqual(len(exp.state["history"][str(target)]), 4)
                self.assertEqual(exp.state["final_quorum"], target == 90)
        self.assertEqual(engine.steps, initial + 12 * 1024)
        self.assertEqual([call.args[0] for call in engine.set_stage.call_args_list],[60,90])
        self.assertEqual(set(exp.state["qualified"]), {"30","60","90"})


if __name__ == "__main__":
    unittest.main()
