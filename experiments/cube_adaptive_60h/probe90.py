"""Early unassisted neural90deg probe; leaves the review timetable unchanged."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,time,json
import torch
from .evaluate import evaluate,compact
from .config import Profile
from .storage import load,atomic_json


def probe(goal,episodes=4):
    torch.set_num_threads(1);root=Path(goal);state=json.loads((root/'controller_state.json').read_text())
    report=dict(observed_at=datetime.now(timezone.utc).isoformat(),formal_review=False,
        scheduled_review_unchanged=state['next_review_at'],scope='Early unassisted target90 probe on new selected seeds; not final100-case proof',axes={})
    deadline=min(time.time()+240.,datetime.fromisoformat(state['deadline_at']).timestamp())
    for axis,key in [('x','10'),('y','30')]:
        checkpoint=state['axes'][axis]['best_by_target'][key]['path'];model,meta=load(checkpoint)
        result=evaluate(model,root/'sources/x/reference.npz',axis,Profile(**meta['profile']),90.,episodes,4000000,deadline)
        report['axes'][axis]=dict(checkpoint=checkpoint,source_training_target=float(key),final90=result)
        print(json.dumps(dict(checkpoint=checkpoint,**compact(result)),ensure_ascii=False),flush=True)
    atomic_json(root/'reviews/000_probe_review.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--goal',required=True);args=p.parse_args();probe(args.goal)
