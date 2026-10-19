"""Summarize already-read source hypotheses; perform no inference or admission."""
from __future__ import annotations
import argparse,gzip,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np

def digest(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def write(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    opt=parser.parse_args();run=opt.run.resolve()
    if not run.is_relative_to(Path('D:/CodexData/RinkSpace_Data')):raise ValueError('Data input required')
    manifest=json.loads((run/'run_manifest.json').read_text(encoding='utf8'))
    old=json.loads(Path(manifest['old_row_index']).read_text(encoding='utf8'))
    if digest(manifest['old_row_index'])!=manifest['old_row_index_sha256']:raise ValueError('Old index changed')
    old_by_index={r['compact_packet_row_index']:r for r in old['records']}
    full_meta_path=Path(manifest['authority_path']).parent/'human_registered_full_stroke_hard_guard_v1.json'
    full_meta=json.loads(full_meta_path.read_text(encoding='utf8'));truth={}
    if digest(full_meta_path)!='c5db59096204de9b6fd253521736a2255316170339d535fe34812f527beae868':raise ValueError('Full human metadata identity changed')
    if digest(full_meta['guard_arrays_reference']['path'])!='5abd423e11a6e1d28f0d69ac41498c8336300b1bad20fd6ef885354d6add6e51':raise ValueError('Full human numeric identity changed')
    with np.load(full_meta['guard_arrays_reference']['path'],allow_pickle=False)as arrays:
        for record in full_meta['records']:
            truth[record['id']]=arrays[record['array_prefix']+'_uv'].copy()
    counts=Counter();rows=[];shot_reports=[];seen=set();human_material_checks=[];fresh_checks=0
    for shot in manifest['shots']:
        if not shot.get('complete_actual_exposure_coverage'):continue
        sname=shot['shot'];measurements=run/(sname+'_measurements.jsonl.gz');pts=set();display_count=0;scount=Counter()
        with gzip.open(measurements,'rt',encoding='utf8')as stream:
            for line in stream:
                frame=json.loads(line)
                if frame['phase']=='later_reference_backward_navigation_preparation':continue
                p=frame['actual_pts']
                if p in pts:raise ValueError('Duplicate main actual exposure')
                pts.add(p);display_count+=len(frame['display_pts'])
                for feature in frame['features']:
                    scount['feature_queries']+=len(feature['navigation_uv'])
                    scount['outframe_feature_queries']+=len(feature['outframe_query_indices'])
                    scount['generic_RGB_missing_queries']+=len(feature['missing_or_occluded_query_indices'])
                    support=feature.get('flow_supported')
                    if support is not None:scount['immediate_flow_unsupported_queries']+=sum(not x for x in support)
                    if not feature['direct_human_GT_input']:continue
                    # Launch v1 stored a mutable navigation alias. The physical
                    # GT comes directly from authenticated full human arrays,
                    # never from a measured/updated search hint in that output.
                    nav=truth[feature['id']][np.asarray(feature['original_query_indices'],int)]
                    entry=None;lower=None;unique=None
                    if 'yellow_material_order'in feature:
                        mat=feature['yellow_material_order'];lower=np.asarray(mat['lower_exit_uv'],float)
                        unique=np.asarray(mat['unique_lower_role'],bool)&~np.asarray(mat['missing_same_band'],bool)
                        delta=np.linalg.norm(lower-nav,axis=1)
                        human_material_checks.append(dict(shot=sname,actual_pts=p,feature_id=feature['id'],role='yellow_contact',
                            query_count=len(nav),material_unique_queries=int(unique.sum()),
                            measured_exit_minus_direct_human_uv_px=delta.tolist(),
                            material_unique_mask=unique.tolist(),
                            median_unique_distance_px=float(np.median(delta[unique]))if unique.any()else None,
                            maximum_unique_distance_px=float(delta[unique].max())if unique.any()else None,
                            human_GT_replaced=False,independent_test=False,
                            status='development human source comparison; unique material does not certify physical edge'))
                    if 'blue_TOP_material_order'in feature:
                        mat=feature['blue_TOP_material_order'];values=[];unique_values=[]
                        for entry,ev,good in zip(mat['entry_uv'],mat['evidence'],mat['unique_entry_role']):
                            seed=np.asarray(ev['original_source_navigation_seed_uv'],float)
                            d=float(np.linalg.norm(np.asarray(entry)-seed));values.append(d)
                            if good:unique_values.append(d)
                        human_material_checks.append(dict(shot=sname,actual_pts=p,feature_id=feature['id'],role='blue_TOP',
                            query_count=len(nav),material_unique_queries=len(unique_values),
                            measured_entry_minus_direct_human_uv_px=values,
                            median_unique_distance_px=float(np.median(unique_values))if unique_values else None,
                            maximum_unique_distance_px=max(unique_values)if unique_values else None,
                            human_GT_replaced=False,independent_test=False,
                            status='development human source comparison; unique material does not certify physical edge'))
                for observed in frame['old_source_checks']:
                    index=observed['compact_packet_row_index']
                    if index in seen:raise ValueError('Old source row checked more than once')
                    seen.add(index);original=old_by_index[index]
                    if observed['original_full_uv_float64_sha256']!=original['original_full_uv_float64_sha256']:raise ValueError('Original UV identity changed')
                    if observed['original_sigma']!=original['original_sigma']or observed['original_uv']!=original['original_uv']:raise ValueError('Original UV/sigma changed')
                    original_uv=np.asarray(original['original_uv'],dtype='<f8')
                    if hashlib.sha256(original_uv.tobytes()).hexdigest()!=original['original_full_uv_float64_sha256']:raise ValueError('Original UV bytes digest mismatch')
                    for key in ('compact_packet_row_index','block','group_order','row_order','id','collection_role','semantic_role'):
                        if observed[key]!=original[key]:raise ValueError('Original row field differs: '+key)
                    if observed['actual_pts']!=original['group_actual_pts']or observed['source_native_pts']!=original.get('source_native_pts'):raise ValueError('Original clock differs')
                    # Recompute recorded deltas from the independently retained
                    # material output and frozen old UV, never mutable guides.
                    if observed['semantic_role']in('ice_contact','ice_board_contact'):
                        mat=observed['yellow_material_order']
                        for point in observed['points']:
                            i=point['original_uv_index']
                            delta=np.asarray(mat['lower_exit_uv'][i],float)-original_uv[i]
                            upper=np.asarray(mat['entry_uv'][i],float)-original_uv[i]
                            if not np.array_equal(delta,point['fresh_edge_minus_old_uv'])or not np.array_equal(upper,point['yellow_upper_minus_old_uv']):raise ValueError('Contact fresh output/delta differs')
                            fresh_checks+=1
                    elif observed['semantic_role']=='board_top':
                        pending=np.arange(len(original_uv));expected={}
                        for mat in observed['blue_TOP_material_order_layers']:
                            selected=np.asarray(mat['queried_original_indices'],int)
                            measured={int(ev['source_query_index']):(entry,exit_)for entry,exit_,ev in zip(mat['entry_uv'],mat['internal_exit_uv'],mat['evidence'])}
                            for query,local in enumerate(selected):expected[int(pending[local])]=measured.get(query)
                            pending=np.delete(pending,selected)
                        if len(pending):raise ValueError('Incomplete blue material layers')
                        for point in observed['points']:
                            i=point['original_uv_index'];value=expected[i]
                            if value is None:
                                if point['fresh_edge_minus_old_uv']is not None or point['blue_internal_exit_minus_old_uv']is not None:raise ValueError('Missing blue output fabricated')
                            elif not np.array_equal(np.asarray(value[0],float)-original_uv[i],point['fresh_edge_minus_old_uv'])or not np.array_equal(np.asarray(value[1],float)-original_uv[i],point['blue_internal_exit_minus_old_uv']):raise ValueError('Blue fresh output/delta differs')
                            fresh_checks+=1
                    rec={key:original.get(key)for key in ('compact_packet_row_index','shot','block','group_actual_pts','group_order','row_order',
                        'collection_role','original_pts','source_native_pts','id','original_split','original_source_split','semantic_role',
                        'original_physical_identity','original_world_scene_binding','physical_facet_status','original_explicit_board_side',
                        'source_origin','original_full_uv_float64_sha256','original_uv','original_sigma','point_count')}
                    rec.update(actual_measurement_pts=p,measurement_source_binding=frame['source_binding'],
                        measurement_decoded_RGB_sha256=frame['decoded_RGB_sha256'],points=observed['points'],
                        material_evidence_reference=dict(path=str(measurements),actual_pts=p,compact_packet_row_index=index),
                        physical_facet_identity=observed.get('physical_facet_identity'),
                        binding_association_pending=bool(observed.get('facet_quarantine_candidate',False)),
                        union_binding_without_explicit_facet_is_not_itself_wrong=True,
                        source_material_quarantine_does_not_include_binding_uncertainty=True,
                        admission='main source review required; raw historical packet unchanged')
                    if rec['semantic_role']in('board_top','ice_board_contact','ice_contact'):
                        if [r['original_uv_index']for r in rec['points']]!=list(range(rec['point_count'])):raise ValueError('Incomplete original UV query coverage')
                        counts['TOP_contact_rows']+=1;counts['TOP_contact_points']+=len(rec['points'])
                        counts['binding_association_pending_rows']+=int(rec['binding_association_pending'])
                        for point in rec['points']:
                            counts['wrong_interface_diagnostic_candidates']+=int(point['retire_candidate'])
                            counts['material_quarantine_candidates']+=int(point['quarantine_candidate'])
                            counts['unique_material_hypotheses']+=int(point['unique_material_support'])
                            counts['outframe_old_points']+=int(point['visibility']=='outframe')
                    else:counts['other_raw_source_rows_indexed_without_new_operator']+=1
                    rows.append(rec);scount['all_old_rows_checked']+=1
        if len(pts)!=shot['actual_exposures']or display_count!=shot['native_presentations']:raise ValueError('Actual/native coverage mismatch')
        if scount['all_old_rows_checked']!=shot['expected_old_source_rows']:raise ValueError('Old row coverage mismatch')
        shot_reports.append(dict(shot=sname,actual_exposure_count=len(pts),native_presentation_count=display_count,
            coverage=dict(scount),measurement_sha256=digest(measurements),
            native_clock_ledger_sha256=digest(run/(sname+'_all_presentations.jsonl.gz'))))
    rows.sort(key=lambda x:(x['shot'],x['group_actual_pts'],x['group_order'],x['row_order']))
    lineage=dict(source_run=str(run),source_tool_sha256=manifest['source_code_sha256'],authority_sha256=manifest['authority_sha256'],
        old_row_index_sha256=manifest['old_row_index_sha256'],source_video_sha256=manifest['source_sha256'],
        model_sha256=manifest['model_sha256'],clock_proof_sha256=manifest['clock']['proof_sha256'])
    snapshot=run/'source_tool_at_launch.py'
    launch_hash=next(value for key,value in manifest['source_code_sha256'].items()if key.endswith('read_registered_human_source_shots.py'))
    if digest(snapshot)!=launch_hash:raise ValueError('Actual launch source snapshot differs')
    lineage['actual_launch_source_snapshot']=dict(path=str(snapshot),sha256=launch_hash)
    output=run/'old_TOP_contact_candidate_review_index.json'
    write(output,dict(schema='rinkspace.original-source-material-candidate-review.v1',lineage=lineage,
        original_packets_modified=False,automatic_active_source_retirement=False,
        wrong_interface_candidates_require_RGB_and_facet_review=True,
        missing_competing_or_outframe_are_quarantine_candidates_not_proven_wrong=True,records=rows))
    summary=dict(schema='rinkspace.registered-source-complete-shot-QA-summary.v1',lineage=lineage,status=manifest['status'],
        complete_requested_scope=manifest.get('complete_all_requested_shots',False),shots=shot_reports,counts=dict(counts),
        indexed_original_rows=len(rows),candidate_review_index=str(output),candidate_review_index_sha256=digest(output),
        direct_human_development_material_comparisons=human_material_checks,
        material_uniqueness_is_not_physical_identity_or_facet_proof=True,
        union_binding_without_explicit_facet_is_not_itself_wrong=True,
        outframe_query_flags_describe_navigation_bounds_not_proven_feature_visibility=True,
        missing_human_facet_does_not_trigger_automatic_reidentification_of_new_visible_facets=True,
        launch_v1_navigation_output_aliasing_detected=True,
        feature_pre_measurement_navigation_output_status='invalid_review_output; mutable launch alias cannot certify later source position or GT',
        later_derived_navigation_to_GT_alignment_not_calculated=True,
        human_comparisons_use_original_full_authority_arrays_not_mutable_output_navigation=True,
        raw_old_point_RGB_results_not_affected_by_navigation_output_aliasing=True,
        original_old_row_UV_sigma_id_index_clock_role_hash_all_exact=True,
        fresh_material_fields_vs_frozen_old_UV_delta_exact_checked_points=fresh_checks,
        occlusion_not_confirmed_from_this_reader=True,derived_source_hypotheses_promoted_to_GT=False,
        painted_baseline_and_Lane_output_is_generic_gradient_not_verified_center_axis=True,
        no_camera_projection_navigation=True,old_FIT_HELD_and_sigma_preserved=True)
    write(run/'continuous_source_QA_summary.json',summary)
    write(run/'source_QA_integrity_receipt.json',dict(status='passed_old_point_fresh_material_integrity_with_isolated_feature_navigation_defect',
        lineage=lineage,actual_exposures=sum(r['actual_exposure_count']for r in shot_reports),
        native_presentations=sum(r['native_presentation_count']for r in shot_reports),
        original_rows=len(rows),old_TOP_contact_points=fresh_checks,
        old_original_UV_sigma_id_index_clock_role_hash_exact=True,
        fresh_material_delta_vs_original_UV_exact=True,
        actual_world_facet_binding_review_separate_from_source_RGB_material_review=True,
        launch_feature_pre_measurement_navigation_field='invalid_review_output; mutable array alias',
        reference_human_GT_recovered_only_at_exact_reference_PTS_from_original_full_NPZ=True,
        subsequent_derived_feature_GT_alignment_not_asserted=True,
        old_material_outputs_use_independent_arrays_not_the_mutable_feature_navigation=True,
        raw_packets_modified=False,active_source_admission=False,
        completed_scope=summary['complete_requested_scope'],summary_sha256=digest(run/'continuous_source_QA_summary.json'),
        candidate_index_sha256=digest(output)))
    print(json.dumps(dict(status=summary['status'],complete=summary['complete_requested_scope'],shots=[dict(shot=r['shot'],actual=r['actual_exposure_count'],native=r['native_presentation_count'])for r in shot_reports],counts=dict(counts),indexed_original_rows=len(rows))))

if __name__=='__main__':main()
