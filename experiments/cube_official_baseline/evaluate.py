"""Evaluate the official-objective CPU baseline without declaring 90-degree success."""
import argparse
import torch
from .train import load,evaluate,atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--episodes',type=int,default=100);parser.add_argument('--seed-start',type=int,default=50000000)
    args=parser.parse_args();torch.set_num_threads(1)
    model,metadata=load(args.checkpoint)
    report=evaluate(model,args.episodes,args.seed_start)
    report.update(checkpoint=args.checkpoint,checkpoint_training_steps=metadata['training_steps'])
    atomic_json(args.output,report)
    print({k:v for k,v in report.items() if k!='records'})


if __name__=='__main__':main()
