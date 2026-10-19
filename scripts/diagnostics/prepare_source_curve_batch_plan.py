"""Prepare a lightweight immutable source batching plan; never execute FIT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.program_tracking.compact_source_header_inventory import read_minimal_compact_source_headers
from calibration.program_tracking.source_curve_batch_plan import compile_source_curve_batch_plan
from calibration.program_tracking.strict_model_scope import authenticate_source_reference,file_sha


def main():
    parser=argparse.ArgumentParser(description='Retain all exact source slices and paths in an uninstalled primitive reuse plan')
    parser.add_argument('--pack',type=Path,required=True);parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--model-identity',type=Path,required=True);parser.add_argument('--model-identity-sha256',required=True)
    parser.add_argument('--production-check',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve an earlier source plan')
    source_ref=dict(path=str(args.pack.resolve()),sha256=args.expected_sha256)
    model_ref=dict(path=str(args.model_identity.resolve()),sha256=args.model_identity_sha256)
    authenticate_source_reference(source_ref);authenticate_source_reference(model_ref)
    identity=json.loads(args.model_identity.read_text(encoding='utf8'))
    started=time.perf_counter();headers,offsets=read_minimal_compact_source_headers(args.pack)
    with np.load(args.pack,allow_pickle=False)as z:
        sigma=z['sigma_px'].copy();metadata=json.loads(str(z['metadata_json'].item()))
    plan=compile_source_curve_batch_plan(headers,offsets,sigma,source_reference=source_ref,
        model_reference=model_ref,model_q_sha256=identity['q_sha256'],source_metadata=metadata)
    if args.production_check:
        check=json.loads(args.production_check.read_text(encoding='utf8'))
        if check.get('schema')!='rinkspace.actual-production-lossless-exact-curve-source-batching-probe.v1' or check['q_sha256']!=identity['q_sha256']:
            raise ValueError('Actual production check belongs to another model or operator')
        plan['partial_actual_production_check_reference']=dict(path=str(args.production_check.resolve()),sha256=file_sha(args.production_check))
        plan['partial_check_does_not_qualify_complete_world_or_full_source_pack']=True
    if file_sha(args.pack)!=source_ref['sha256'] or file_sha(args.model_identity)!=model_ref['sha256']:
        raise ValueError('Bound source/model changed during read-only preparation')
    plan['preparation_seconds']=time.perf_counter()-started
    plan['implementation_reference']=dict(path=str(Path(__file__).resolve()),sha256=file_sha(__file__))
    plan['plan_implementation_reference']=dict(path=str(ROOT/'calibration/program_tracking/source_curve_batch_plan.py'),
        sha256=file_sha(ROOT/'calibration/program_tracking/source_curve_batch_plan.py'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8')as f:json.dump(plan,f,ensure_ascii=False,separators=(',',':'))
    print(json.dumps(dict(output=str(args.output),source_points=plan['original_source_points'],source_rows=plan['original_source_rows'],
        prepared_groups=plan['prepared_groups'],finite_sigma_points=plan['finite_positive_working_sigma_points'],
        missing_sigma_points=plan['missing_working_sigma_points'],status_rows=plan['status_row_counts'],
        points_removed=0,execution_ready=False,fit_started=False)),flush=True)


if __name__=='__main__':main()
