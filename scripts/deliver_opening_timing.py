"""Deliver the quiet-opening phase correction through the existing player."""
from pathlib import Path
import argparse
import json
import sys

project = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(project/'src'), str(project)]
from rinkspace.core.native import NativeCore
core = NativeCore('D:/CodexData/RinkSpace_Data/build/solver_continue_1102/rinkspace_core.dll')
from calibration.program_tracking.body_tracking.opening_timing import build, TimingProblem
from calibration.program_tracking.body_tracking.opening_motion_delivery import run

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(project, args.root, args.output, builder=build, problem_type=TimingProblem,
        stem='opening_timing_1103', variant_id='timing1103',
        label='新候选：22.90秒后下探、扶腰后展开'), ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
