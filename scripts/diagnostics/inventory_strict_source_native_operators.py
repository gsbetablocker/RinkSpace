"""Write the current all-source native operator routing inventory."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from calibration.solver_runtime.strict_source_operator_inventory import inventory, reference

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = inventory(reference(args.dataset))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(report, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    print(json.dumps(dict(report_reference=reference(args.output), counts=report['counts'],
        kinds=report['finite_points_by_target_kind'],
        operators=report['finite_points_by_declared_curve_operator'],
        groups=len(report['records']))), flush=True)
