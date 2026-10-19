"""Freeze current source packs and exact admission indices, without a solver."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.program_tracking.strict_source_fit_dataset import build_logical_dataset,reference


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--logical-inputs',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    started=time.perf_counter()
    inputs=json.loads(args.logical_inputs.read_text(encoding='utf8'))
    if inputs['schema']!='rinkspace.logical-current-qualified-source-inputs.v1':
        raise ValueError('Current reviewed logical source bindings required')
    records=[dict(source_pack=r['absolute_inputs'],pending_filter_references=[
        f['filter_reference'] for f in r['current_original_pending_filters']]) for r in inputs['records']]
    ref=build_logical_dataset(records,args.output_dir,
        model_reference=inputs['model_identity_reference'],
        coverage_reference=inputs['original_native_and_retirement_coverage_reference'],
        duplicate_audit_reference=inputs['operator_and_duplicate_preflight_reference'],
        authorities=[reference(args.logical_inputs),*inputs['original_authorities'].values(),
            reference(__file__),reference(ROOT/'calibration/program_tracking/strict_source_fit_dataset.py')])
    print(json.dumps(dict(dataset_reference=ref,seconds=time.perf_counter()-started,
        solver_started=False),separators=(',',':')),flush=True)


if __name__=='__main__':main()
