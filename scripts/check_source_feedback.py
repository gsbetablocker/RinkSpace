"""Recheck source-backed feedback regressions; no training or fitting."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rinkspace.feedback_regression import verify_feedback_regressions
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=verify_feedback_regressions(args.event,output=args.output)
    print(json.dumps(dict(receipt=result['receipt'],learned_cases=len(result['learned_procedure_regression']),
        visual_improvement_verified=False),ensure_ascii=False))
