"""Read nine immutable terminal seed chains; identify exact inner review needs.

An automated review flag is not a negative or a new quarantine. Specific
source evidence must be checked before any additional point is withheld.
"""
from __future__ import annotations
from collections import Counter
import gc
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from calibration.program_tracking.compact_source_json import unpack_source_json
from calibration.program_tracking.source_nested_material_identity_audit import inherited_nested_material_review_problem
from calibration.program_tracking.strict_model_scope import authenticate_source_reference,file_sha

WORK=Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012')
OUT=WORK/'current_source_pool_joint_qualified_20261013_v1/source_scope_accuracy_review_v1'


def ref(path):return dict(path=str(Path(path).resolve()),sha256=file_sha(path))


def main():
    started=time.perf_counter()
    plan_path=WORK/'nine_cut_terminal_core_coverage_20261013_v1/terminal_nine_cut_plan.json'
    plan=json.loads(plan_path.read_text(encoding='utf8'))
    dp=ROOT/'events/ankara_2026/programmes/okada_mei_sp/source_datasets/current_review_20261013/fit_dataset.json'
    dataset=json.loads(dp.read_text(encoding='utf8'));lookup={c['source_pack']['sha256']:c for c in dataset['cuts']}
    confirmed_pending_path=OUT/'034_F3_all_37_exposures_190_exact_nested_identity_pending_filter.json'
    confirmed=json.loads(confirmed_pending_path.read_text(encoding='utf8'))
    known={g['source_pack_reference']['sha256']:{i for e in g['entries']for i in e['pack_point_indices']}for g in confirmed['source_pack_groups']}
    counts=Counter();cut_records=[];needs=[];direct=[];auth_cache={}
    for entry in plan['entries']:
        source=entry['absolute_inputs'];authenticate_source_reference(source)
        cut=entry['shot_id'].rsplit('_',1)[-1]
        selection=lookup[source['sha256']]
        with np.load(selection['support_index']['path'],allow_pickle=False)as z:fit=z['fit_point_indices']
        with np.load(source['path'],allow_pickle=False)as z:
            rows=unpack_source_json(z['rows_json']);proofs=unpack_source_json(z['source_proof_templates_json'])
            uv=z['uv'];offsets=z['row_offsets'];proof_ids=z['source_proof_template_index']
        local=Counter()
        for ri,row in enumerate(rows):
            material=row['physical_material_id'];parts=material.rsplit(':',1)
            if len(parts)!=2 or parts[1]not in ('middle','inner','center'):continue
            lo,hi=map(int,offsets[ri:ri+2]);indices=fit[(fit>=lo)&(fit<hi)]
            if not len(indices):continue
            local['current_finite_inner_points']+=len(indices)
            flagged=[];independent=[]
            for pi in indices:
                proof=dict(proofs[int(proof_ids[pi])],uv=uv[pi].tolist())
                issue=inherited_nested_material_review_problem(row,proof,point_uv=uv[pi])
                if issue:flagged.append(int(pi))
                else:independent.append(int(pi))
            seed=row.get('source_seed_reference',{})
            authority=row.get('source_seed_authority',seed.get('authority'))
            if isinstance(authority,dict)and authority.get('path')and authority.get('sha256'):
                key=(authority['path'],authority['sha256'])
                if key not in auth_cache:
                    authenticate_source_reference(authority);auth_cache[key]=authority
            shared=dict(source_pack_reference=source,cut=cut,original_source_row_index=ri,
                physical_material_id=material,native_pts=row['native_pts'],actual_pts=row['actual_pts'],
                decoded_BGR_sha256=row['decoded_BGR_sha256'],
                seed_curve_id=seed.get('curve_id'),seed_native_pts=seed.get('native_pts'),
                seed_annotation_id=seed.get('source_annotation_id'),
                source_seed_authority=authority,
                physical_identity_basis=seed.get('physical_identity_basis'),
                has_saved_actual_nested_RGB_sequence=bool(seed.get('same_source_house_layer_identity_evidence')),
                temporal_kind=row.get('temporal_evidence',{}).get('kind'),
                same_shot_RGB_tracking=row.get('temporal_evidence',{}).get('source_RGB_tracking_confirmed'),
                declared_final_RGB_precision_checked=row.get('final_current_RGB_pixel_precision_checked'))
            if flagged:
                local['outer_inherited_specific_inner_review_needed_points']+=len(flagged)
                pending=sorted(set(flagged)&known.get(source['sha256'],set()))
                rest=sorted(set(flagged)-set(pending))
                local['already_original_RGB_reviewed_pending_points']+=len(pending)
                local['additional_source_review_needed_not_automatically_isolated_points']+=len(rest)
                needs.append(dict(shared,original_pack_global_point_indices=flagged,uv=uv[flagged].tolist(),
                    already_reviewed_exact_pending_indices=pending,additional_review_needed_indices=rest,
                    physical_negative_asserted=False,automatic_quarantine_created=False))
            if independent:
                local['not_flagged_by_narrow_outer_inheritance_guard_points']+=len(independent)
                direct.append(dict(shared,original_pack_global_point_indices=independent,
                    static_or_manual_false_flag_alone_did_not_cause_rejection=True))
        counts.update(local);cut_records.append(dict(cut=cut,source_pack_reference=source,counts=dict(local)))
        print(json.dumps(cut_records[-1],separators=(',',':')),flush=True)
        del rows,proofs,uv,offsets,proof_ids,fit;gc.collect()
    result=dict(schema='rinkspace.terminal-nine-source-specific-inner-seed-chain-audit.v1',
        terminal_plan_reference=ref(plan_path),frozen_finite_v1_dataset_reference=ref(dp),
        existing_specific_source_review_pending_filter_reference=ref(confirmed_pending_path),
        records=cut_records,counts=dict(counts),specific_inner_source_review_needed_records=needs,
        retained_records_not_flagged_by_narrow_inheritance_guard=direct,
        authenticated_seed_authorities=list(auth_cache.values()),
        scope_proof_manual_false_alone_used_as_rejection=False,
        new_pixels_isolated_without_actual_source_review=False,
        model_projection_or_training_output_used_for_scope=False,
        original_source_or_confirmed_carrier_modified=False,
        full_programme_accuracy_accepted=False,elapsed_seconds=time.perf_counter()-started,
        implementation_references=[ref(__file__),ref(ROOT/'calibration/program_tracking/source_nested_material_identity_audit.py')])
    path=OUT/'terminal_nine_specific_inner_seed_chain_inventory.json'
    if path.exists():raise FileExistsError('Preserve completed earlier inventories')
    path.write_text(json.dumps(result,separators=(',',':')),encoding='utf8')
    print(json.dumps(dict(receipt=ref(path),counts=dict(counts))),flush=True)


if __name__=='__main__':main()
