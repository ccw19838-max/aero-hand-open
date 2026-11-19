"""Isolated final-delivery regression tests. No experiment initial states are opened."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from experiments.cube_adaptive_60h.storage import atomic_json, digest
from . import finalize as delivery


def completed(dt, seed, success):
    info = dict(strict_success=success, first_violation=None, max_strict_hold_seconds=float(success),
                eligible_peak_degrees=89. if success else 2., max_progress_degrees=90. if success else 3.,
                best_angle_error_degrees=0. if success else 88., failure_reason='' if success else 'timeout',
                warning_lead_ms=[], warning_count=0, warning_false_count=0, warning_censored_count=0)
    return dict(seed=seed, dt=dt, success=success, interrupted=False, final=info)


class FinalizeTests(unittest.TestCase):
    def setUp(self):
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        self.root = Path(cache.name)
        self.source = self.root / 'candidate'
        self.source.mkdir()
        (self.source / 'policy.zip').write_bytes(b'synthetic test fixture')
        deadline = datetime.now(timezone.utc) + timedelta(seconds=180)
        self.deadline = deadline.timestamp()
        self.seed = 31000
        atomic_json(self.root / 'job.json', dict(deadline_at=deadline.isoformat(), final_seed=self.seed,
                    profile=dict(name='test')))
        self.state = dict(stage=5, best={}, qualified={}, final_quorum=False, last_v4_checkpoint=str(self.source))

    def run_mock(self, state, succeeds, stop=lambda: False):
        def env(_, profile, target, dt, arm):
            return types.SimpleNamespace(dt=dt, target=target, close=lambda: None)

        def episode(environment, model, seed, **_):
            return completed(environment.dt, seed, succeeds(environment, seed))

        def record(model, inputs, profile, target, row, output, checkpoint, **kwargs):
            kind = kwargs['kind']
            path = output / f'{kind}.mp4'
            path.write_bytes(b'synthetic media')
            (output / f'{kind}_states.npz').write_bytes(b'synthetic states')
            atomic_json(output / f'{kind}_trajectory.json', row)
            return dict(video=str(path), video_sha256=digest(path), full_decode_passed=True,
                        policy_sha256=digest(Path(checkpoint) / 'policy.zip'))

        with (patch('experiments.cube_strict_x_v4.engine.load_for_eval', return_value=(None, dict(arm='v4', profile=dict(name='test')))),
              patch.object(delivery.evaluation, 'make_env', side_effect=env),
              patch.object(delivery.evaluation, 'episode', side_effect=episode),
              patch('experiments.cube_strict_x_v4.replay.record_case', side_effect=record)):
            return delivery.finalize(self.root, state, self.deadline, stop)

    def test_resumed_review_preserves_seeds_and_candidates(self):
        stage5 = self.root / 'passed5'
        stage5.mkdir()
        (stage5 / 'policy.zip').write_bytes(b'passed stage 5')
        self.state.update(stage=10, qualified={'5': {'checkpoint': str(stage5)}})
        run = []
        def trial(env, seed):
            run.append((env.dt, seed, env.target))
            return env.target == 5

        partial = self.run_mock(self.state, trial, stop=lambda: len(run) >= 7)
        self.assertFalse(partial['complete'])
        self.assertEqual(len(run), 7)
        # Changing the live selection must not change already frozen candidates.
        self.state['qualified']['5']['checkpoint'] = str(self.source)
        done = self.run_mock(self.state, trial)
        self.assertTrue(done['complete'])
        self.assertFalse(done['verified'])
        self.assertEqual(len(set(run)), 80)
        self.assertEqual(len(run), 80)
        best = done['artifacts']['best']
        failure = done['artifacts']['typical_failure']
        self.assertEqual((best['pin']['target'], failure['pin']['target']), (5, 10))
        self.assertEqual(best['pin']['policy_sha256'], digest(stage5 / 'policy.zip'))
        self.assertEqual(failure['pin']['policy_sha256'], digest(self.source / 'policy.zip'))
        self.assertTrue(Path(done['report']).exists())

    def test_no_checkpoint_leaves_actionable_report(self):
        self.state['last_v4_checkpoint'] = None
        result = self.run_mock(self.state, lambda *_: False)
        self.assertFalse(result['verified'] or result['complete'])
        self.assertIn('没有可用的 V4 检查点', Path(result['report']).read_text())

    def test_acceptance_requires_all_independent_trials_at_each_step(self):
        self.state.update(stage=90, final_quorum=True, qualified={'90': {'checkpoint': str(self.source)}})
        done = self.run_mock(self.state, lambda env, seed: not (env.dt == .001 and seed == self.seed))
        self.assertTrue(done['complete'])
        self.assertFalse(done['verified'])
        groups = delivery.read(done['reports'][0])['groups']
        self.assertEqual([g['successes'] for g in groups.values()], [99, 100])
        self.assertEqual(done['artifacts']['typical_failure']['pin']['seed'], self.seed)

    def test_all_success_has_no_fabricated_failure(self):
        self.state.update(stage=90, final_quorum=True, qualified={'90': {'checkpoint': str(self.source)}})
        done = self.run_mock(self.state, lambda *_: True)
        self.assertTrue(done['complete'] and done['verified'])
        self.assertEqual(done['failure_status'], 'none_observed')
        self.assertNotIn('typical_failure', done['artifacts'])

    def test_all_success_fallback_has_historical_provenance(self):
        self.state.update(stage=90, final_quorum=True, qualified={'90': {'checkpoint': str(self.source)}})
        previous = self.root / 'historical'
        previous.mkdir()
        (previous / 'policy.zip').write_bytes(b'prior V4 weights')
        archived = self.root / 'evaluations' / 'v4.json'
        rows = [completed(dt, 30, False) for dt in delivery.evaluation.TIMESTEPS]
        report = delivery.evaluation.summarize_report(rows, 10, 1, 30)
        report['checkpoint'] = str(previous)
        atomic_json(archived, report)
        done = self.run_mock(self.state, lambda *_: True)
        self.assertTrue(done['complete'] and done['verified'])
        pin = done['artifacts']['typical_failure']['pin']
        self.assertEqual(pin['purpose'], 'archived_validation')
        self.assertEqual(pin['policy_sha256'], digest(previous / 'policy.zip'))
        self.assertEqual(pin['source_report'], str(archived))

    def test_three_stage_course_diagnoses_30_with_only_legacy_qualifications(self):
        self.state.update(stage=30, curriculum_degrees=[30, 60, 90],
                          qualified={'5': {'checkpoint': str(self.source)}, '10': {'checkpoint': str(self.source)}})
        result = self.run_mock(self.state, lambda *_: False)
        self.assertTrue(result['complete'])
        self.assertEqual(result['highest_qualified_degrees'], 0)
        self.assertIn('30° 课程尚未连续两轮达到晋级要求', Path(result['report']).read_text())
        manifest = delivery.read(result['frozen'])
        self.assertEqual([t['name'] for t in manifest['tasks']], ['challenge_30'])

    def test_three_stage_freeze_ignores_retired_targets_between_active_stages(self):
        self.state['curriculum_degrees'] = [30, 60, 90]
        for stage, passed, targets in [(60, (5, 30, 45), [30, 60]), (90, (10, 30, 60, 75), [60, 90])]:
            with self.subTest(stage=stage):
                self.state.update(stage=stage, qualified={str(t): {'checkpoint': str(self.source)} for t in passed})
                tasks = delivery.freeze_plan(self.state, self.seed)
                self.assertEqual([t['target'] for t in tasks], targets)
                self.assertEqual([t['purpose'] for t in tasks], ['qualified_stage_review', 'next_stage_diagnostic'])
                self.assertTrue(all(t['seed'] != self.seed for t in tasks))

    def test_three_stage_course_opens_independent_acceptance_after_90_qualifies(self):
        self.state.update(stage=90, final_quorum=True, curriculum_degrees=[30, 60, 90],
                          qualified={str(t): {'checkpoint': str(self.source)} for t in (30, 60, 90)})
        task, = delivery.freeze_plan(self.state, self.seed)
        self.assertEqual((task['name'], task['episodes'], task['seed']), ('acceptance_90', 100, self.seed))

    def test_unqualified_90_attempt_does_not_open_acceptance_seeds(self):
        self.state.update(stage=90, qualified={'75': {'checkpoint': str(self.source)}})
        tasks = delivery.freeze_plan(self.state, self.seed)
        self.assertEqual([t['target'] for t in tasks], [75, 90])
        self.assertTrue(all(t['episodes'] == 20 and t['seed'] != self.seed for t in tasks))


if __name__ == '__main__':
    unittest.main()
