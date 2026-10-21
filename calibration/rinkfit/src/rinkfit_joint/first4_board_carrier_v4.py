"""Bind source-reviewed board contexts to the same existing continuous carrier.

A context label is retained as provenance. It does not split the common board
or supply a measured world segment. No source pixel, geometry, or admission is
changed. Candidate parts are existing finite search domains, not new evidence.
"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
from collections import Counter
from rinkfit_start.native_scene import NativeScene
from .first4_material_paths_v3 import context_board_union
from .first4_material_paths_v2 import BOARD_SEGMENTS, _ref
from .io_utils import read_json, write_json, file_sha256
from .source_generation_input_v2 import QualifiedSourceGenerationInput


def repair_single_branch_contexts(model,mapping,rows,review):
    result=copy.deepcopy(mapping);rows=list(rows);scene=NativeScene(model);delta=[];checks=[]
    if review.get('review_origin') != 'Codex_actual_original_RGB_and_frozen_reader_self_review':
        raise ValueError('actual_source_review_required')
    frames={f['native_pts']:f for f in review['frames']}
    if review['source_policy_limits']['outside_existing_candidate_carrier_evidence_found'] is not False:
        raise ValueError('outside_existing_carrier_domain_requires_new_source_review')
    for role in ('top','contact'):
        uid='board.'+role+'.right_end|board|'+('board_upper_material_edge|top' if role=='top' else 'ice_board_interface|contact')
        desc=result['semantic_to_material'][uid];members=[r for r in rows if r.get('joint_material_id')==uid]
        if not members:raise ValueError('source_reviewed_context_rows_missing')
        old=copy.deepcopy(desc);mi=int(review['explicit_material_carriers'][role]['material_index'])
        canonical=model['materials'][mi];height=1. if role=='top' else 0.
        expected_operator='board_upper_material_edge' if role=='top' else 'ice_board_interface'
        physical_role='blue_band_UPPER_outer_edge' if role=='top' else 'yellow_band_LOWER_outer_edge'
        for r in members:
            f=frames.get(r['native_pts']);d=r.get('typed_material_descriptor') or {}
            if not f or f['native_png']['sha256'] != r['native_png_sha256']:
                raise ValueError('unreviewed_actual_source_exposure')
            definition=r['measurement_definition'];physical=definition['expected_physical_definition']
            if r['physical_surface']!='board' or d.get('kind')!='board_edge' or d.get('role')!=role:
                raise ValueError('source_context_surface_or_edge_role_conflict')
            if r['measurement_operator_resolved']!=expected_operator or definition.get('conflict') is not False or physical.get('physical_role')!=physical_role:
                raise ValueError('source_context_physical_definition_conflict')
            if r.get('point_correspondence'):
                raise ValueError('physical_fixed_entity_cannot_become_sliding_context')
        allowed,carrier=context_board_union(canonical,[r['typed_material_descriptor'] for r in members],scene,mi)
        if allowed is None:raise ValueError('same_role_carrier_has_no_explicit_finite_domain')
        if carrier['fixed_height_fraction']!=height:raise ValueError('same_carrier_height_conflict')
        # The original contextual segment supplies only a deterministic seed.
        seeds=[i for i in allowed if canonical['paths'][i]['boundary_segment']==BOARD_SEGMENTS['right_end']]
        if len(seeds)!=1:raise ValueError('existing_context_seed_missing_from_carrier')
        seed=seeds[0]
        proof=dict(basis='actual_original_source_same_common_board_carrier_role_and_height',
            original_part_is_context_not_hard_correspondence=True,
            original_context_target=desc['target'],original_material_index=old['material_index'],
            shared_existing_carrier_material_index=mi,fixed_height_fraction=height,
            source_physical_role=physical_role,source_operator=expected_operator,
            source_review_origin=review['review_origin'],human_acceptance_created=False,
            existing_finite_carrier_domain=carrier,source_global_station_measured=False,
            same_brand_or_screen_angle_used_for_identity=False,screen_nearest_branch_used=False,
            historical_candidate_parts_are_not_new_source_observations=True,
            opposite_parts_excluded_by_new_source_measurement=False)
        desc.update(material_index=mi,path_index=seed,allowed_paths=allowed,path_domain_version=4,
            path_domain_proof=proof,finite_station_interval=[0.,1.],geometry_ready=True,
            basis='source_reviewed_same_common_board_carrier_role_and_height')
        for r in members:
            rid=r['source_row_id'];previous=copy.deepcopy(mapping.get('row_to_material',{}).get(rid))
            if previous:raise ValueError('single_branch_context_has_unreviewed_row_override')
            override=copy.deepcopy(desc);override.update(source_row_id=rid,source_uv=r['source_uv'],
                native_pts=r['native_pts'],shot_id=r['shot_id'],source_packet_sha256=r['source_packet_sha256'])
            result.setdefault('row_to_material',{})[rid]=override
            delta.append(dict(source_row_id=rid,joint_material_id=uid,source_uv=r['source_uv'],
                current_uv=r.get('current_uv'),shot_id=r['shot_id'],native_pts=r['native_pts'],
                source_packet_sha256=r['source_packet_sha256'],source_admitted=r['source_admitted'],
                old_material_index=old['material_index'],old_path_index=old['path_index'],old_allowed_paths=old['allowed_paths'],
                new_material_index=mi,new_path_index=seed,new_allowed_paths=allowed))
        checks.append(dict(semantic_id=uid,rows=len(members),admitted=sum(r['source_admitted'] for r in members),
            old=old,new=copy.deepcopy(desc)))
    changed={c['semantic_id'] for c in checks}
    if any(result['semantic_to_material'][u]!=v for u,v in mapping['semantic_to_material'].items() if u not in changed):
        raise ValueError('unrelated_semantic_material_changed')
    counts=dict(context_semantic_ids_bound=len(checks),context_rows_bound=len(delta),
        admitted=sum(r['source_admitted'] for r in delta),not_admitted_retained=sum(not r['source_admitted'] for r in delta))
    report=dict(schema='rinkfit.first4.single_branch_context_carriers.v4',counts=counts,
        checks=checks,row_binding_delta=delta,source_review=review,
        model_geometry_paths_and_parameters_unchanged=True,source_UV_ID_PTS_and_admission_unchanged=True,
        S01_prefix_unchanged=True,global_physical_station_not_measured=True,
        no_new_path_or_private_board=True,existing_old_materials_retained=True)
    result['schema']='rinkfit.first4_actual_materials.board_carriers_v4'
    result['board_single_branch_carrier_repair_v4']={k:v for k,v in report.items() if k!='row_binding_delta'}
    return result,report


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--mapping',required=True)
    p.add_argument('--binding',required=True);p.add_argument('--review',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);audit=out.with_name('board_carrier_audit_v4.json')
    if out.exists() or audit.exists():raise ValueError('immutable carrier output already exists')
    q=QualifiedSourceGenerationInput(a.binding);review=read_json(a.review)
    for f in review['frames']:
        if file_sha256(f['native_png']['path'])!=f['native_png']['sha256']:
            raise ValueError('reviewed_original_native_PNG_changed')
    result,report=repair_single_branch_contexts(read_json(a.model),read_json(a.mapping),q.rows(),review)
    refs=dict(model=_ref(a.model),base_map=_ref(a.mapping),source_binding=_ref(a.binding),
              source_review=_ref(a.review),compiler=_ref(__file__))
    report['input_references']=refs;result['board_single_branch_carrier_repair_v4']['input_references']=refs
    write_json(out,result);write_json(audit,report)
    print(json.dumps(dict(mapping=_ref(out),audit=_ref(audit),counts=report['counts'])))

if __name__=='__main__':main()
