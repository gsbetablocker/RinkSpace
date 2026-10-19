"""Export exact pending indices from the completed 37-exposure source review."""
from __future__ import annotations
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from calibration.program_tracking.compact_source_json import unpack_source_json
from calibration.program_tracking.source_nested_material_identity_audit import inherited_nested_material_review_problem
from calibration.program_tracking.strict_model_scope import authenticate_source_reference,file_sha
from calibration.program_tracking.strict_source_named_identity_filter_with_exclusions import current_row_source_binding,_pending_indices
from calibration.program_tracking.strict_source_fit_dataset import _named_pending_indices

OUT=Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012/current_source_pool_joint_qualified_20261013_v1/source_scope_accuracy_review_v1')


def ref(path):return dict(path=str(Path(path).resolve()),sha256=file_sha(path))


def main():
    inventory_path=OUT/'034_F3_nested_current_point_inventory.json'
    sequence_path=OUT/'034_F3_all_37_exact_source_exposures/exact_selected_source_sequence.json'
    inventory=json.loads(inventory_path.read_text(encoding='utf8'))
    sequence=json.loads(sequence_path.read_text(encoding='utf8'))
    source=inventory['source_pack_reference'];authenticate_source_reference(source)
    with np.load(source['path'],allow_pickle=False)as z:
        rows=unpack_source_json(z['rows_json']);templates=unpack_source_json(z['source_proof_templates_json'])
        meta=json.loads(str(z['metadata_json'].item()))
        values={k:z[k]for k in ('row_offsets','uv','source_proof_template_index','source_frame_actual_pts','source_frame_native_pts','source_frame_BGR_sha256')}
    frames={int(p):dict(native_pts=int(n),decoded_BGR_sha256=s.decode())for p,n,s in zip(values['source_frame_actual_pts'],values['source_frame_native_pts'],values['source_frame_BGR_sha256'])if n>=0 and s}
    pack=SimpleNamespace(rows=rows,values=values,proof_templates=templates,metadata=meta,frame_bindings=frames)
    reviewed={r['actual_pts']:r for r in sequence['records']}
    records=[];entries=[];indices=[]
    reason='Specific inner paint material remains unverified in the original source sequence. User-confirmed F3 outer carrier is preserved; alternating RGB gradients and the outer annotation do not establish each claimed middle/inner/center layer. Retain these exact readouts pending, not as a FIT/training positive or a negative.'
    for record in inventory['records']:
        ri=record['original_source_row_index'];row=rows[ri];points=record['point_indices']
        if row['physical_material_id']=='F3:outer' or row['split']!='fit':raise ValueError('Confirmed outer/HELD cannot enter this pending review')
        for pi in points:
            proof=dict(templates[int(values['source_proof_template_index'][pi])],uv=values['uv'][pi].tolist())
            if not inherited_nested_material_review_problem(row,proof,point_uv=values['uv'][pi]):
                raise ValueError('Do not quarantine an independently reviewed inner-material point')
        evidence=reviewed[row['actual_pts']]
        if evidence['decoded_BGR_sha256']!=row['decoded_BGR_sha256']:raise ValueError('Real source review does not match the frozen exposure')
        binding=current_row_source_binding(pack,ri)
        entry=dict(pack_row_index=ri,physical_material_id=row['physical_material_id'],
            source_binding=binding,pack_point_indices=points,
            exclude_from_final_named_FIT=True,status='source_material_measured_named_identity_pending',reason=reason)
        for key in ('original_candidate_indices','source_query_indices'):
            if key in row:entry[key]=[row[key][i]for i in record['point_local_indices']]
        entries.append(entry)
        records.append(dict(original_source_row_index=ri,physical_material_id=row['physical_material_id'],
            source_binding=binding,original_pack_global_point_indices=points,uv=record['uv'],
            exact_raw_crop_reference=evidence['exact_raw_crop_reference'],
            exact_point_pointer_reference=evidence['point_pointer_reference'],
            same_model_full_grid_reference=evidence['same_model_full_grid_reference'],
            specific_named_inner_material_verified=False,decision='pending_not_negative'))
        indices.extend(points)
    indices=sorted(indices)
    if len(indices)!=len(set(indices))or len(indices)!=190:raise ValueError('Current reviewed exact point inventory changed')
    early=json.loads((OUT/'034_F3_229333248_exact_nested_identity_pending.json').read_text(encoding='utf8'))
    if not set(early['original_pack_global_point_indices']).issubset(indices):raise ValueError('Earlier exact 18-point review differs')
    fact=dict(schema='rinkspace.exact-original-source-material-identity-pending.v1',source_pack_reference=source,
        frozen_dataset_reference=inventory['frozen_dataset_reference'],inventory_reference=ref(inventory_path),
        complete_original_source_sequence_reference=ref(sequence_path),records=records,
        original_pack_global_point_indices=indices,pending_points=len(indices),actual_exposures=len(reviewed),
        material_counts=inventory['exact_original_FIT_points_by_material'],reason=reason,
        supersedes_18_point_pending_reference=ref(OUT/'034_F3_229333248_exact_nested_identity_pending.json'),
        confirmed_outer_and_other_entities_preserved=True,unknown_specific_material_is_not_negative=True,
        named_reflection_asserted=False,original_UV_sigma_or_evidence_modified=False,
        frozen_dataset_modified=False,training_started=False,full_programme_accuracy_accepted=False,
        implementation_references=[ref(__file__),ref(ROOT/'calibration/program_tracking/source_nested_material_identity_audit.py')])
    fact_path=OUT/'034_F3_all_37_exposures_190_exact_nested_identity_pending.json'
    if fact_path.exists():raise FileExistsError('Preserve completed scope facts')
    fact_path.write_text(json.dumps(fact,separators=(',',':')),encoding='utf8')
    group=dict(source_pack_reference=source,pending_named_point_count=len(indices),
        source_review_witnesses=[ref(fact_path),ref(sequence_path),ref(inventory_path)],entries=entries)
    filtered=dict(schema='rinkspace.exact-source-named-identity-pending-filter.v1',
        retain_pending_source_readouts_full_literal_evidence=True,source_pack_groups=[group],
        supersedes_filter_reference=ref(OUT/'034_F3_229333248_exact_nested_identity_pending_filter.json'),
        unknown_specific_material_is_not_negative=True,confirmed_outer_source_retained=True,
        source_material_UV_sigma_or_clock_modified=False,review_accuracy_accepted=False)
    path=OUT/'034_F3_all_37_exposures_190_exact_nested_identity_pending_filter.json'
    if path.exists():raise FileExistsError('Preserve completed pending filters')
    path.write_text(json.dumps(filtered,separators=(',',':')),encoding='utf8')
    physical=_pending_indices(pack,group)
    if sorted(i for s in physical.values()for i in s)!=indices:raise ValueError('Physical filter changed exact point indices')
    selected=_named_pending_indices(rows,values['row_offsets'],source,meta['source'],
        {k:(v['native_pts'],v['decoded_BGR_sha256'])for k,v in frames.items()},[ref(path)])
    if not np.array_equal(selected,indices):raise ValueError('Logical filter changed exact point indices')
    print(json.dumps(dict(fact_reference=ref(fact_path),filter_reference=ref(path),pending_points=len(indices),
        public_physical_and_logical_exact_filters_checked=True)),flush=True)


if __name__=='__main__':main()
