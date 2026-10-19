"""Run the current immutable source-index vision training bridge."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rinkspace.strict_source_dataset_training import train

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(train(args.config),ensure_ascii=False),flush=True)
