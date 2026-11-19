"""Evaluate/replay V2 through exactly the same observation and action code."""
import argparse,json,os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import torch
from experiments.cube_adaptive_60h.config import Profile
from .io import load,atomic_json
from .evaluation import evaluate,compact


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--target',type=float,choices=(0.,90.),default=90.)
    parser.add_argument('--episodes',type=int,default=20);parser.add_argument('--seed',type=int,default=5800000)
    parser.add_argument('--output',required=True);parser.add_argument('--trace',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.episodes<1:parser.error('episodes must be positive')
    checkpoint=Path(args.checkpoint).resolve();model,meta=load(checkpoint)
    # The copied reference report and prefix bank travel with the checkpoint.
    report=evaluate(model,checkpoint,Profile(**meta['profile']),args.target,args.episodes,args.seed,trace=args.trace)
    atomic_json(args.output,report);print(json.dumps(compact(report),indent=2))
