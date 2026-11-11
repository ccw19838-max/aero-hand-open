"""Behavioral checks for preparation gates, real policy bundles and CPU budgets."""
from dataclasses import replace
import json
import contextlib
import io
from pathlib import Path
import tempfile
import time
import unittest

import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics, specs
from flax import struct

from experiments.pen_spin.config import PenSpinConfig
from experiments.pen_spin.model import model_fingerprint, task_fingerprint
from experiments.pen_spin.policy import (digest_file, inference_function, load_checkpoint,
                                         network_factory, require_prepared, save_checkpoint)
from experiments.pen_spin.train import CurriculumMonitor, run_guarded, stage_budget_seconds
from experiments.pen_spin.evaluate import _episode, _parallel_episodes, _row, main as evaluate_main


@struct.dataclass
class ToyData:
    time: jax.Array


@struct.dataclass
class ToyState:
    data: ToyData
    obs: dict
    reward: jax.Array
    done: jax.Array
    metrics: dict
    info: dict


class ToyEnv:
    """Different seed-dependent episode lengths exercise batch termination."""
    backend='mjx'
    config=replace(PenSpinConfig(),episode_seconds=.15)
    def reset(self,key):
        target=(key[1]%3+1).astype(jp.float32)
        obs={'state':jp.array([0.,target,0.]),'privileged_state':jp.array([0.,target,0.,0.])}
        metrics={name:jp.array(0.) for name in ('success','failure','phase','theta','phi','valid_progress_radians')}
        return ToyState(ToyData(jp.array(0.)),obs,jp.array(0.),jp.array(False),metrics,{})
    def step(self,state,action):
        count=state.obs['state'][0]+1
        target=state.obs['state'][1]
        done=count>=target
        obs={'state':jp.array([count,target,0.]),'privileged_state':jp.array([count,target,0.,0.])}
        metrics=dict(success=done.astype(jp.float32),failure=jp.array(0.),phase=jp.array(1.),
                     theta=count,phi=count,valid_progress_radians=count)
        return state.replace(data=ToyData(count*.05),obs=obs,reward=jp.array(1.),done=done,metrics=metrics)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = replace(PenSpinConfig(), hidden_layers=(8,8))
        self.reference = self.directory/'reference.npz'
        provenance = dict(config_digest=self.config.digest,
                          task_digest=task_fingerprint(self.config),
                          model_digest=model_fingerprint(self.config))
        # A synthetic state tests only the bundle gate, never task feasibility.
        np.savez(self.reference,qpos=np.zeros(23),qvel=np.zeros(22),ctrl=np.zeros(7),
                 n0=np.array([1.,0.,0.]),c0=np.zeros(3),**provenance)
        self.report = dict(passed=True,status='passed',
                           reference_sha256=digest_file(self.reference),**provenance)
        self.write_report()
        self.config.save(self.directory/'config.json')

    def write_report(self):
        (self.directory/'prepare_report.json').write_text(json.dumps(self.report))

    def test_preparation_requires_unambiguous_pass_and_bound_physics(self):
        require_prepared(self.reference,self.config)
        require_prepared(self.reference,replace(self.config,budget_hours=1,learning_rate=1e-4))
        for passed,status in [(False,'passed'),(True,'ready'),(1,'passed')]:
            self.report.update(passed=passed,status=status);self.write_report()
            with self.assertRaises(ValueError):require_prepared(self.reference,self.config)
        self.report.update(passed=True,status='passed');self.write_report()
        with self.assertRaises(ValueError):
            require_prepared(self.reference,replace(self.config,sim_dt=.001))
        with self.assertRaises(ValueError):
            require_prepared(self.reference,replace(self.config,contact_force=.1))
        self.reference.write_bytes(self.reference.read_bytes()+b'altered')
        with self.assertRaises(ValueError):require_prepared(self.reference,self.config)

    def test_checkpoint_restores_normalizer_and_exact_deterministic_inference(self):
        sizes={'state':3,'privileged_state':4}
        normalizer=running_statistics.init_state({name:specs.Array((size,),jp.float32)
                                                 for name,size in sizes.items()})
        net=network_factory(self.config)(observation_size=sizes,action_size=7,
                                        preprocess_observations_fn=running_statistics.normalize)
        params=(normalizer,net.policy_network.init(jax.random.PRNGKey(1)),
                net.value_network.init(jax.random.PRNGKey(2)))
        obs={'state':jp.array([.1,-.2,.3]),'privileged_state':jp.array([1.,2.,3.,4.])}
        key=jax.random.PRNGKey(3)
        expected=inference_function(params,self.config)(params,obs,key)
        bundle=self.directory/'bundle'
        save_checkpoint(bundle,params,self.config,self.reference,0.,123,batch_deadline_unix=time.time()-1)
        actual,config,reference,metadata=load_checkpoint(bundle)
        np.testing.assert_array_equal(expected,inference_function(actual,config)(actual,obs,key))
        self.assertEqual(metadata['training_steps'],123)
        self.assertTrue((bundle/'prepare_report.json').is_file())
        require_prepared(reference,config)
        # Authoritative payload, not editable JSON sidecars, controls inference.
        (bundle/'metadata.json').write_text('{}')
        self.assertEqual(load_checkpoint(bundle)[3]['training_steps'],123)
        evaluation=self.directory/'expired_evaluation'
        with contextlib.redirect_stdout(io.StringIO()):
            code=evaluate_main(['--checkpoint',str(bundle),'--output',str(evaluation),
                                '--episodes','100','--video-seconds','0'])
        self.assertEqual(code,2)
        self.assertEqual(json.loads((evaluation/'evaluation.json').read_text())['completed_episodes'],0)
        (bundle/'reference.npz').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):load_checkpoint(bundle)

    def test_curriculum_needs_two_successful_checks_and_stops_stagnation(self):
        monitor=CurriculumMonitor(self.config)
        self.assertEqual(monitor.update(.8,.1),'continue')
        self.assertEqual(monitor.update(.7,.2),'continue')
        self.assertEqual(monitor.update(.8,.3),'continue')
        self.assertEqual(monitor.update(.8,.3),'promote')
        monitor=CurriculumMonitor(self.config)
        self.assertEqual(monitor.update(0.,0.),'continue')
        for _ in range(3):self.assertEqual(monitor.update(0.,0.),'continue')
        self.assertEqual(monitor.update(0.,0.),'stagnant')

    def test_phase_budgets_preserve_plan_and_total(self):
        np.testing.assert_allclose(stage_budget_seconds(replace(self.config,task_mode='grasp',budget_hours=2)),[7200.])
        np.testing.assert_allclose(stage_budget_seconds(self.config),np.array([2,4,6,12])*3600)
        short=replace(self.config,budget_hours=.1)
        self.assertAlmostEqual(sum(stage_budget_seconds(short)),360.)
        with self.assertRaises(ValueError):replace(self.config,budget_hours=25)

    def test_active_grasp_reset_gate_cannot_authorize_rotation(self):
        cfg=replace(self.config,task_mode='grasp',schema_version=2)
        provenance=dict(config_digest=cfg.digest,model_digest=model_fingerprint(cfg),task_digest=task_fingerprint(cfg))
        np.savez(self.reference,qpos=np.zeros(23),qvel=np.zeros(22),ctrl=np.zeros(7),
                 n0=np.array([1.,0.,0.]),c0=np.zeros(3),**provenance)
        self.report.update(**provenance,reference_sha256=digest_file(self.reference),
                           status='grasp_reset_ready',scope='grasp_acquisition_only',gates={'model':{'passed':True}})
        self.write_report()
        require_prepared(self.reference,cfg)
        with self.assertRaises(ValueError):require_prepared(self.reference,replace(cfg,task_mode='spin'))
        self.report['gates']['model']['passed']=False;self.write_report()
        with self.assertRaises(ValueError):require_prepared(self.reference,cfg)

    def test_outer_watchdog_bounds_startup_without_a_callback(self):
        output=self.directory/'timed'
        config=replace(self.config,budget_hours=.00005)  # 0.18s: before heavy import/JIT.
        request=dict(config=config.to_dict(),reference=str(self.reference),output=str(output))
        start=time.monotonic()
        self.assertEqual(run_guarded(request,output),0)
        self.assertLess(time.monotonic()-start,3.)
        report=json.loads((output/'training_report.json').read_text())
        self.assertEqual(report['status'],'wallclock_stopped')
        self.assertTrue(report['full_skill_success_not_assumed'])
        self.assertFalse((output/'stages/hold/last/checkpoint.pkl').exists())

    def test_four_way_evaluation_matches_sequential_with_partial_final_batch(self):
        env=ToyEnv()
        apply=lambda params,obs,key:jp.zeros(obs['state'].shape[:-1]+(7,))
        seeds=list(range(1000,1005))
        sequential=[_episode(env,None,apply,seed)[0] for seed in seeds]
        parallel=[result[0] for result in _parallel_episodes(env,None,apply,seeds)]
        self.assertEqual(sequential,parallel)
        self.assertEqual([r['steps'] for r in parallel],[seed%3+1 for seed in seeds])

    def test_substep_export_discards_terminal_padding(self):
        state=ToyEnv().reset(jax.random.PRNGKey(0))
        trace=np.zeros((25,24),dtype=np.float32)
        trace[:,0]=.004;trace[0,0]=.002
        state=state.replace(info={'substep_trace':jp.asarray(trace)})
        row=_row(state,jp.zeros(7),1,True)
        self.assertEqual(len(row['substeps']),2)


if __name__=='__main__':
    unittest.main()
