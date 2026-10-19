"""Freeze the exact 038/039 material questions after actual-source review."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from calibration.program_tracking.compact_source_json import unpack_source_json
from calibration.program_tracking.strict_model_scope import authenticate_source_reference,file_sha
from calibration.program_tracking.strict_source_fit_dataset import _named_pending_indices
from calibration.program_tracking.strict_source_named_identity_filter_with_exclusions import current_row_source_binding,_pending_indices

OUT=Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012/current_source_pool_joint_qualified_20261013_v1/source_scope_accuracy_review_v1')


def ref(p):return dict(path=str(Path(p).resolve()),sha256=file_sha(p))


def main():
    inventory_path=OUT/'terminal_nine_specific_inner_seed_chain_inventory.json'
    source_sequence=OUT/'038_039_inherited_inner_exact_source_review/exact_038_039_selected_inner_source_sequence.json'
    neighbor=OUT/'038_039_inherited_inner_exact_source_review/seed_native_neighbors/specific_inner_seed_neighbors_source_receipt.json'
    inventory=json.loads(inventory_path.read_text(encoding='utf8'))
    sequence=json.loads(source_sequence.read_text(encoding='utf8'))
    reviewed={r['native_pts']:r for r in sequence['records']}
    selected=[r for r in inventory['specific_inner_source_review_needed_records']if r['cut']in ('038','039')]
    reason=('Actual original RGB, source seed neighbors and the same 8a28 all-grid navigation were reviewed. '
        'Real outer rings/paint material remain confirmed, but these precise named inner-layer locations cannot '
        'be independently verified. Some claimed center locations sit on visibly different paint transitions. '
        'The saved alternating-gradient sequence does not certify each specific middle/inner/center material. '
        'Keep the exact readouts and all original proofs pending; do not use as FIT/training positives or negatives.')
    groups=[];facts=[];all_counts=Counter()
    for cut in ('038','039'):
        current=[r for r in selected if r['cut']==cut];source=current[0]['source_pack_reference']
        authenticate_source_reference(source)
        with np.load(source['path'],allow_pickle=False)as z:
            rows=unpack_source_json(z['rows_json']);templates=unpack_source_json(z['source_proof_templates_json'])
            meta=json.loads(str(z['metadata_json'].item()))
            values={k:z[k]for k in ('row_offsets','source_proof_template_index','source_frame_actual_pts','source_frame_native_pts','source_frame_BGR_sha256')}
        frames={int(p):dict(native_pts=int(n),decoded_BGR_sha256=s.decode())for p,n,s in zip(values['source_frame_actual_pts'],values['source_frame_native_pts'],values['source_frame_BGR_sha256'])if n>=0 and s}
        pack=SimpleNamespace(rows=rows,values=values,proof_templates=templates,metadata=meta,frame_bindings=frames)
        entries=[];count=Counter();indices=[]
        for record in current:
            ri=record['original_source_row_index'];row=rows[ri];points=record['additional_review_needed_indices']
            if not points:continue
            if row['split']!='fit' or row['physical_material_id'].endswith(':outer'):raise ValueError('Confirmed outer and all HELD are preserved')
            e=reviewed[row['native_pts']]
            if e['source_binding']['decoded_native_BGR_sha256']!=row['decoded_BGR_sha256']:raise ValueError('Actual source review bytes differ')
            binding=current_row_source_binding(pack,ri);lo=int(values['row_offsets'][ri]);local=np.asarray(points,np.int64)-lo
            entry=dict(pack_row_index=ri,physical_material_id=row['physical_material_id'],source_binding=binding,
                pack_point_indices=points,exclude_from_final_named_FIT=True,status='source_material_measured_named_identity_pending',reason=reason)
            for key in ('original_candidate_indices','source_query_indices'):
                if key in row:entry[key]=np.asarray(row[key])[local].tolist()
            entries.append(entry);indices.extend(points);count[row['physical_material_id']]+=len(points)
            facts.append(dict(cut=cut,source_pack_reference=source,original_source_row_index=ri,
                physical_material_id=row['physical_material_id'],source_binding=binding,
                original_pack_global_point_indices=points,uv=record['uv'],
                exact_raw_crop_reference=e['exact_raw_crop_reference'],same_model_grid_reference=e['same_model_full_grid_reference'],
                point_pointer_reference=e['point_pointer_reference'],original_outer_annotation=record['seed_annotation_id'],
                original_seed_authority=record['source_seed_authority'],
                original_measured_nested_RGB_topology_and_all_proofs_retained=True,
                this_specific_named_inner_material_location_verified=False,decision='pending_not_negative'))
        if len(set(indices))!=len(indices):raise ValueError('An original pixel was listed twice')
        all_counts.update(count)
        groups.append(dict(source_pack_reference=source,pending_named_point_count=len(indices),entries=entries,
            _validation_pack=pack,_source=meta['source'],_frames=frames))
    if sum(g['pending_named_point_count']for g in groups)!=1380:raise ValueError('Reviewed original source point inventory changed')
    fact=dict(schema='rinkspace.exact-original-source-material-identity-pending.v1',
        terminal_inner_seed_inventory_reference=ref(inventory_path),
        complete_selected_original_source_review_reference=ref(source_sequence),
        specific_source_seed_and_native_neighbors_reference=ref(neighbor),records=facts,
        points_by_material=dict(all_counts),pending_points=1380,selected_reviewed_actual_exposures=156,
        source_reason=reason,source_viewed_false_flag_alone_caused_rejection=False,
        real_outer_paint_and_human_confirmed_points_preserved=True,
        original_RGB_topology_support_and_literal_UV_sigma_proofs_retained=True,
        new_negative_or_reflection_labels_created=False,old_frozen_pack_or_dataset_changed=False,
        existing_F1_current_profile_qualification_covers='house_connection_F1_R1:segment0, not these distinct F1/R1 rings',
        physically_measured_paint_is_not_automatically_this_named_inner_layer=True,
        independently_verified_specific_inner_source_points_removed=0,
        no_new_primitive_or_fit_started=True,full_programme_accuracy_accepted=False,implementation_reference=ref(__file__))
    fact_path=OUT/'038_039_exact_specific_inner_material_pending_facts.json'
    if fact_path.exists():raise FileExistsError('Preserve completed earlier scope facts')
    fact_path.write_text(json.dumps(fact,separators=(',',':')),encoding='utf8')
    witnesses=[ref(fact_path),ref(source_sequence),ref(neighbor),ref(inventory_path)]
    public_groups=[{k:v for k,v in g.items()if not k.startswith('_')}for g in groups]
    for g in public_groups:g['source_review_witnesses']=witnesses
    filtered=dict(schema='rinkspace.exact-source-named-identity-pending-filter.v1',
        retain_pending_source_readouts_full_literal_evidence=True,source_pack_groups=public_groups,
        confirmed_outer_source_retained=True,unknown_specific_material_is_not_negative=True,
        source_UV_sigma_or_clock_modified=False,source_layer_scope_not_certified_from_projection=True)
    path=OUT/'038_039_exact_specific_inner_material_pending_filter.json'
    if path.exists():raise FileExistsError('Preserve completed earlier scope filters')
    path.write_text(json.dumps(filtered,separators=(',',':')),encoding='utf8')
    for g in groups:
        physical=_pending_indices(g['_validation_pack'],g)
        expected=sorted(i for e in g['entries']for i in e['pack_point_indices'])
        if sorted(i for s in physical.values()for i in s)!=expected:raise ValueError('Physical filter changed exact indices')
        pack=g['_validation_pack']
        logical=_named_pending_indices(pack.rows,pack.values['row_offsets'],g['source_pack_reference'],g['_source'],
            {k:(v['native_pts'],v['decoded_BGR_sha256'])for k,v in g['_frames'].items()},[ref(path)])
        if not np.array_equal(logical,expected):raise ValueError('Logical filter changed exact indices')
    print(json.dumps(dict(fact_reference=ref(fact_path),filter_reference=ref(path),pending_points=1380,
        points_by_material=dict(all_counts),public_physical_and_logical_filters_exact_checked=True)),flush=True)


if __name__=='__main__':main()
