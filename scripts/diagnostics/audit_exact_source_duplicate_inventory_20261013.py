"""Read-only exact duplicate audit using minimal headers and numeric source UV."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.program_tracking.compact_source_header_inventory import read_minimal_compact_source_headers
from calibration.program_tracking.strict_model_scope import file_sha,authenticate_source_reference


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pack',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior exact source audit')
    source_ref=dict(path=str(args.pack.resolve()),sha256=args.expected_sha256)
    authenticate_source_reference(source_ref)
    started=time.perf_counter();headers,offsets=read_minimal_compact_source_headers(args.pack)
    with np.load(args.pack,allow_pickle=False)as z:
        uv=z['uv'].copy();sigma=z['sigma_px'].copy();original=z['original_point_indices'].copy()
    choices={};duplicate_records=[];operator_duplicate_count=0;missing_replaced=0;split_conflicts=0;operator_conflicts=0
    by_material=Counter();groups={};split_points=Counter();operator_points=Counter();operator_rows=Counter()
    for ri,h in enumerate(headers):
        physical=(h['actual_pts'],h['scope'],h['physical_material_id'],
            json.dumps(h['model_target'],sort_keys=True,separators=(',',':')),h['observation_semantics'])
        if physical not in groups:groups[physical]=len(groups)
        gi=groups[physical]
        operator=json.dumps({k:h.get(k) for k in ('curve_observation_kind','point_correspondence',
            'must_preserve_upper_branch','scanline_branch','scanline_fixed_axis','extent_side',
            'clip_boundary_x','clip_blur_allowance_px','source_operator_array_fields_present')},sort_keys=True,separators=(',',':'))
        lo,hi=map(int,offsets[ri:ri+2])
        split_points[h['split']]+=hi-lo
        operator_points[operator]+=hi-lo
        operator_rows[operator]+=1
        for pi in range(lo,hi):
            key=(gi,np.asarray(uv[pi],np.float64).tobytes())
            finite=bool(np.isfinite(sigma[pi]) and sigma[pi]>0)
            record=(pi,ri,finite,h['split'],operator)
            if key in choices:
                old=choices[key];same_operator=old[4]==operator
                operator_duplicate_count+=int(same_operator)
                operator_conflicts+=int(not same_operator)
                split_conflicts+=int(old[3]!=h['split'])
                by_material[h['physical_material_id']]+=1
                if finite and not old[2]:
                    missing_replaced+=1;choices[key]=record
                duplicate_records.append(dict(actual_pts=h['actual_pts'],material=h['physical_material_id'],
                    model_target=h['model_target'],observation_semantics=h['observation_semantics'],
                    first_global_point_index=old[0],first_source_row_index=old[1],
                    repeated_global_point_index=pi,repeated_source_row_index=ri,
                    original_candidate_index=int(original[pi]),first_original_candidate_index=int(original[old[0]]),
                    uv=uv[pi].tolist(),same_operator=same_operator,
                    first_split=old[3],repeated_split=h['split'],
                    first_sigma_finite=old[2],repeated_sigma_finite=finite,
                    ordinary_public_merge_would_prefer_new_measured_support=finite and not old[2]))
            else:choices[key]=record
    if file_sha(args.pack)!=args.expected_sha256:raise ValueError('Source bytes changed during audit')
    report=dict(schema='rinkspace.readonly-exact-source-pixel-target-operator-duplicate-inventory.v1',
        source_reference=source_ref,points=len(uv),rows=len(headers),physical_material_exposure_target_groups=len(groups),
        ordinary_public_merge_exact_identity='actual_pts + scope + physical_material_id + exact model_target JSON + observation_semantics + float64 UV bytes',
        duplicate_source_pixels_and_physical_targets=len(duplicate_records),
        duplicate_source_pixels_targets_and_same_operator=operator_duplicate_count,
        same_pixel_target_with_conflicting_operator=operator_conflicts,
        same_pixel_target_with_conflicting_fit_held_membership=split_conflicts,
        missing_support_duplicates_with_later_finite_support=missing_replaced,
        unique_exact_pixel_target_count=len(choices),finite_positive_working_sigma_points=int((np.isfinite(sigma)&(sigma>0)).sum()),
        duplicates_by_material=dict(by_material),duplicate_records=duplicate_records,
        original_split_point_counts=dict(split_points),
        original_operator_flag_inventory=[dict(operator=json.loads(k),rows=operator_rows[k],points=v)
            for k,v in operator_points.items()],
        original_point_order_values_proofs_and_roles_changed=False,pixels_removed=0,solver_started=False,
        source_scope_or_fit_qualified_by_this_audit=False,read_seconds=time.perf_counter()-started,
        implementation_reference=dict(path=str(Path(__file__).resolve()),sha256=file_sha(__file__)))
    with args.output.open('x',encoding='utf8')as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('duplicate_records','duplicates_by_material')},ensure_ascii=True),flush=True)


if __name__=='__main__':main()
