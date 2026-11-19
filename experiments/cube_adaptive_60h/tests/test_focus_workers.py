from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,tempfile,unittest
from unittest.mock import patch
from experiments.cube_adaptive_60h.config import Profile
from experiments.cube_adaptive_60h.controller import Controller
from experiments.cube_adaptive_60h.training_parameters import worker_timesteps


class FocusWorkers(unittest.TestCase):
    def test_default_and_four_workers_balance_validated_integrators(self):
        self.assertEqual(worker_timesteps({}),(.001,.002))
        self.assertEqual(worker_timesteps({'physics_timesteps':[.001,.002,.001,.002]}),(.001,.002,.001,.002))
        for bad in ([.001],[.001,.002,.001],[.001]*4,[.001,.003],[True,.002]):
            with self.assertRaises(ValueError):worker_timesteps({'physics_timesteps':bad})

    def test_x_only_controller_never_prepares_or_spawns_y(self):
        now=datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp)
            goal=dict(active_axes=['x'],allocation={'x':4,'y':0,'z':0},hours=24,
                started_at=now.isoformat(),deadline_at=(now+timedelta(hours=24)).isoformat(),
                next_review_at=(now+timedelta(hours=3)).isoformat(),previous_run=tmp,reference=str(root/'reference.npz'),
                verification_seed_base=200000000)
            entry=dict(segment=0,profile=Profile(bc_weight=0.).dictionary(),initial_target=90.,
                source_checkpoint=str(root/'source'),physics_timesteps=[.001,.002,.001,.002],
                history=[],best_by_target={},crashes=0,verified=False)
            (root/'goal.json').write_text(json.dumps(goal))
            (root/'controller_state.json').write_text(json.dumps(dict(axes={'x':entry},status='prepared',reviews=[],
                last_review_index=0,goal_complete=False)))
            c=Controller(root);c.reference=goal['reference'];c.prepare_retention()
            self.assertFalse((root/'retention_y.npz').exists())
            path=c.ensure_job('x');job=json.loads(path.read_text())
            self.assertEqual(job['axis'],'x');self.assertEqual(worker_timesteps(job),(.001,.002,.001,.002))
            self.assertEqual(job['verification_seed_base'],200000000)
            self.assertFalse((root/'y').exists())
            spawned=[];stopped=[]
            def spawn(axis):spawned.append(axis);c.stop=True
            with patch.object(c,'stop_previous'),patch.object(c,'spawn',side_effect=spawn),\
                 patch.object(c,'stop_axis',side_effect=stopped.append),patch.object(c,'collect',return_value=None):
                c.run()
            self.assertEqual(spawned,['x']);self.assertEqual(stopped,['x'])
            self.assertFalse((root/'y').exists())


if __name__=='__main__':unittest.main()
