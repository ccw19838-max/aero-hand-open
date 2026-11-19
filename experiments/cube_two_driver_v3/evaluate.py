import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/aero-cube-cache')
import argparse,json
from pathlib import Path
import torch
from experiments.cube_adaptive_60h.config import Profile
from .env import LADDER
from .io import load,atomic_json
from .evaluation import evaluate,compact


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--target',type=float,choices=LADDER,default=90.)
    p.add_argument('--episodes',type=int,default=20);p.add_argument('--seed',type=int,default=6700000);p.add_argument('--output',required=True);p.add_argument('--trace',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    if args.episodes<1:p.error('episodes must be positive')
    path=Path(args.checkpoint).resolve();model,meta=load(path)
    report=evaluate(model,path,Profile(**meta['profile']),args.target,args.episodes,args.seed,trace=args.trace)
    atomic_json(args.output,report);print(json.dumps(compact(report),indent=2))
