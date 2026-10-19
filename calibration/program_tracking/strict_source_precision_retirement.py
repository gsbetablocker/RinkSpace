"""Retain old source pixels only by exact new source approval.

Missing approval means pending/superseded evidence, not proof that the old
physical material failed. Original human coordinates, roles and archives are
preserved; this index does not union old detector pixels into new active input.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .compact_source_json import unpack_source_json
from .strict_model_scope import authenticate_source_reference, file_sha


def index_current_source_precision(old_reference, current_reference, output_path):
    """Compact source-pixel locators with exact readout/target and HELD checks."""
    for reference in (old_reference,current_reference):authenticate_source_reference(reference)
    output_path=Path(output_path)
    if output_path.exists():raise ValueError('Preserve earlier source precision dispositions')
    def load(reference):
        with np.load(reference['path'],allow_pickle=False) as saved:
            metadata=json.loads(str(saved['metadata_json'].item()))
            return metadata,unpack_source_json(saved['rows_json']),saved['uv'],saved['row_offsets']
    old,oldrows,oldUV,oldoffset=load(old_reference)
    new,newrows,newUV,newoffset=load(current_reference)
    if old['source']!=new['source'] or new['physically_compacted_approved_pixels_only'] is not True:
        raise ValueError('Current precision readouts require the same exact source')
    exact={};current_entities=set();held_entities=set()
    for row in oldrows:
        if row['split']=='held':held_entities.add((row['actual_pts'],row['feature_id']))
    def descriptor(row):
        return (row['actual_pts'],row['physical_material_id'],json.dumps(row['model_target'],sort_keys=True,separators=(',',':')))
    for row,start,stop in zip(newrows,newoffset[:-1],newoffset[1:]):
        if row['split']=='fit' and (row['actual_pts'],row['feature_id']) in held_entities:
            raise ValueError('New source readout bypassed an existing calibration HELD frame/entity')
        key=descriptor(row);current_entities.add(key)
        for index in range(int(start),int(stop)):
            identity=key+(newUV[index].astype(np.float64).tobytes(),)
            prior=exact.get(identity)
            if prior is not None and prior[1]!=row['split']:
                raise ValueError('One exact current source point has competing calibration roles')
            exact[identity]=(index,row['split'])
    disposition=np.full(len(oldUV),3,np.uint8);retained=np.full(len(oldUV),-1,np.int64)
    processed=set(new['processed_actual_pts'])
    for row,start,stop in zip(oldrows,oldoffset[:-1],oldoffset[1:]):
        key=descriptor(row)
        if row['actual_pts'] not in processed:continue
        for index in range(int(start),int(stop)):
            current=exact.get(key+(oldUV[index].astype(np.float64).tobytes(),))
            if current is not None:
                if row['split']=='held' and current[1]!='held':
                    raise ValueError('Exact old HELD source pixel became FIT')
                disposition[index]=0;retained[index]=current[0]
            else:disposition[index]=1 if key in current_entities else 2
    labels={'0':'exact_current_source_approval_retained',
        '1':'superseded_by_current_actual_readout_not_a_physical_failure_claim',
        '2':'missing_current_approved_material_evidence_pending',
        '3':'current_source_exposure_unprocessed'}
    metadata=dict(schema='rinkspace.strict-source-precision-retirement.v1',source=new['source'],
        old_source_pack_reference=old_reference,current_source_pack_reference=current_reference,
        disposition_labels=labels,new_active_contains_no_old_fallback=True,
        original_human_source_pixels_authorities_and_roles_unchanged=True,
        missing_exact_match_is_not_physical_failure_truth=True,
        tolerance_or_model_residual_used_for_pixel_retention=False,solver_started=False)
    np.savez_compressed(output_path,old_point_disposition=disposition,
        exact_retained_current_point_index=retained,
        metadata_json=json.dumps(metadata,ensure_ascii=False,separators=(',',':')))
    return dict(path=str(output_path.resolve()),sha256=file_sha(output_path),
        counts=np.bincount(disposition,minlength=4).tolist())


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--old-pack',nargs=2,required=True)
    parser.add_argument('--current-pack',nargs=2,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    ref=lambda v:dict(path=v[0],sha256=v[1])
    print(json.dumps(index_current_source_precision(ref(args.old_pack),ref(args.current_pack),args.output)))


if __name__=='__main__':main()
