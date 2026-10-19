"""Run an authenticated immutable source pending exporter snapshot."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.solver_runtime import install
install()
from calibration.program_tracking.strict_model_scope import authenticate_source_reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module-reference',nargs=2,required=True)
    parser.add_argument('--pack',nargs=2,required=True)
    parser.add_argument('--filter',nargs=2,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reference=dict(path=args.module_reference[0],sha256=args.module_reference[1])
    authenticate_source_reference(reference)
    spec=importlib.util.spec_from_file_location(
        'calibration.program_tracking.immutable_pending_exporter',reference['path'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.export_exact_named_identity_pending_pack(
        dict(path=args.pack[0],sha256=args.pack[1]),
        dict(path=args.filter[0],sha256=args.filter[1]),args.output)
    authenticate_source_reference(reference)
    print(json.dumps({key:result[key]for key in ('path','sha256','points','rows',
        'pending_named_points','receipt_reference','pending_evidence_bank_reference')},ensure_ascii=False))


if __name__=='__main__':main()
