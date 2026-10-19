"""Audit final source lineage and exact pending exclusion without fitting.

Read minimal headers and numeric arrays; never expand every proof/row of the
large programme pack into a second full JSON object graph.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.program_tracking.compact_source_header_inventory import read_minimal_compact_source_headers
from calibration.program_tracking.compact_source_json import unpack_source_json
from calibration.program_tracking.strict_model_scope import authenticate_source_reference,file_sha


def reference(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=file_sha(path))


def _uv_key(uv):
    p=np.asarray(uv,np.float64).copy()
    p[p==0]=0.
    return p.tobytes()


def audit(assembly_reference,pending_manifest_reference,output):
    for ref in (assembly_reference,pending_manifest_reference):authenticate_source_reference(ref)
    assembly=json.loads(Path(assembly_reference['path']).read_text(encoding='utf8'))
    pool_ref={k:assembly['final_absolute_source_pack'][k] for k in ('path','sha256')}
    coverage_ref=assembly['original_native_and_retirement_coverage_reference']
    for ref in (pool_ref,coverage_ref):authenticate_source_reference(ref)
    pending_manifest=json.loads(Path(pending_manifest_reference['path']).read_text(encoding='utf8'))
    headers,offsets=read_minimal_compact_source_headers(pool_ref['path'])
    with np.load(pool_ref['path'],allow_pickle=False) as z:
        metadata=json.loads(str(z['metadata_json'].item()))
        uv=z['uv'];sigma=z['sigma_px']
        frames={int(p):(int(n),bytes(h).decode('ascii')) for p,n,h in zip(
            z['source_frame_actual_pts'],z['source_frame_native_pts'],z['source_frame_BGR_sha256'])}
    with np.load(coverage_ref['path'],allow_pickle=False) as z:
        canonical=np.flatnonzero(z['representative'])
        expected={int(z['actual_pts'][i]):(int(z['native_pts'][i]),bytes(z['BGR_sha256'][i]).decode('ascii')) for i in canonical}
        native_count=len(z['native_pts']);retired=len(z['old_pixel_disposition'])
        if not bool(z['native_material_measurement_attempted'].all()):raise ValueError('Original native source processing incomplete')
    if frames!=expected or sorted(frames)!=metadata['processed_actual_pts']:
        raise ValueError('Final actual/native/RGB binding changed original canonical coverage')
    if not metadata['all_exposures_processed'] or not metadata['all_exposures_exact_native_BGR_bound']:
        raise ValueError('Final source clock incomplete')
    if not np.isfinite(uv).all() or len(uv)!=len(sigma) or int(offsets[-1])!=len(uv):
        raise ValueError('Final finite source values and header offsets disagree')
    for ref in metadata['original_references']:authenticate_source_reference(ref)
    full_keys=set();fit_keys=set();by_material=Counter();by_split=Counter();duplicates=0
    for ri,row in enumerate(headers):
        actual=int(row['actual_pts'])
        if actual not in frames or (int(row['native_pts']),row['decoded_BGR_sha256'])!=frames[actual]:
            raise ValueError('A final row changed its own original exposure binding')
        if row['scope'] not in ('ice','board'):raise ValueError('Model-external source entered the pool')
        material=row['physical_material_id']
        operator=json.dumps([row['model_target'],row['observation_semantics']],sort_keys=True,separators=(',',':'))
        lo,hi=map(int,offsets[ri:ri+2]);by_material[material]+=hi-lo;by_split[row['split']]+=hi-lo
        for point in uv[lo:hi]:
            pixel=_uv_key(point);key=(material,actual,operator,pixel)
            duplicates+=key in full_keys;full_keys.add(key)
            if row['split']=='fit':fit_keys.add((material,actual,pixel))
    if duplicates:raise ValueError('Final source pool contains exact named operator duplicates')
    pending_points=0;pending_records=[]
    for record in pending_manifest['current_effective_pending_banks']:
        bank_ref=record['pending_evidence_bank_reference']
        filter_ref=record['pending_filter_reference']
        derived_ref=record['derived_pack_reference']
        for ref in (bank_ref,filter_ref,derived_ref):authenticate_source_reference(ref)
        with np.load(bank_ref['path'],allow_pickle=False) as z:
            values=z['uv'];original=z['original_pack_point_indices'];group=unpack_source_json(z['pending_filter_json'])
        lookup={int(i):point for i,point in zip(original,values)}
        current=0
        for entry in group['entries']:
            actual=int(entry['source_binding']['actual_pts']);material=entry['physical_material_id']
            for index in entry['pack_point_indices']:
                if index not in lookup:raise ValueError('Effective pending bank lost an original source point')
                if (material,actual,_uv_key(lookup[index])) in fit_keys:
                    raise ValueError('Effective pending named pixel was resurrected in final FIT-declared rows')
                current+=1
        if current!=len(values) or current!=group['pending_named_point_count']:
            raise ValueError('Effective pending bank and exact filter point count disagree')
        pending_points+=current
        pending_records.append(dict(**record,pending_points=current,absent_from_final_named_FIT_declared_rows=True))
    finite=np.isfinite(sigma)&(sigma>0)
    result=dict(schema='rinkspace.final-measured-source-pool-integrity-audit.v1',
        assembly_reference=assembly_reference,pending_manifest_reference=pending_manifest_reference,
        final_absolute_source_pack_reference=pool_ref,original_coverage_reference=coverage_ref,
        points=len(uv),rows=len(headers),source_points_by_split=dict(by_split),source_points_by_material=dict(by_material),
        finite_source_working_sigma_points=int(finite.sum()),missing_source_working_sigma_points=int((~finite).sum()),
        native_frames=native_count,actual_exposures=len(frames),native_alias_frames=native_count-len(frames),
        all_original_actual_native_RGB_bindings_bitexact=True,all_source_authority_bytes_authenticated=True,
        exact_named_operator_duplicates=duplicates,effective_pending_named_points=pending_points,
        current_effective_pending_banks=pending_records,retired_original_points_not_appended=retired,
        source_processing_is_not_manual_visual_completion=True,numeric_integrity_is_not_pixel_accuracy=True,
        whole_programme_source_fit_ready=False,accuracy_accepted=False,training_started=False,solver_started=False)
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve earlier source integrity audit')
    with output.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,separators=(',',':'))
    return reference(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--assembly',type=Path,required=True)
    parser.add_argument('--pending-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(reference(args.assembly),reference(args.pending_manifest),args.output)),flush=True)
