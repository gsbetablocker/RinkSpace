"""Combine already passed same-q source audits for the actual fresh startup."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SYS=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def ref(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    oldpath=SYS/'same_q27fa_public_all11_source_and_exact_old71_caps_audit_v1.json';old=read(oldpath)
    latestpath=SYS/'same_q27fa_latest_user3_truth85_audit.json';latest=read(latestpath)
    values85path=SYS/'current85_caps_values_bound_to_exact27fa_v1.json';current=read(values85path)
    protectedpath=SYS/'current82_caps_values_bound_to_exact27fa_v1.json';protected=read(protectedpath)
    originalproof=SYS/'original_absolute_active_compaction_complete_layout_verification_20261010.json';compact=read(originalproof)
    qualified=SYS/'source_pixel_review_additional/native_ice_qualified_parent82_plus_five_RGB_lines_231678464_FIT.json'
    packet=read(qualified);qualification=read(packet['source_only_review_qualification']['path']);rawref=qualification['original_packet'];raw=read(rawref['path'])
    if ref(rawref['path'])!=rawref:raise ValueError('Final NativeIce raw parent bytes changed')
    for audit in (old,latest):
        if audit['passed']is not True or audit['complete_q_sha256']!=old['complete_q_sha256']:raise ValueError('Same-q audits need exact passed identity')
    if compact['passed']is not True or compact['source_compaction']['removed_active_scalar_rows']!=2934:raise ValueError('Actual original compaction proof required')
    if current['audit']!=ref(latestpath):raise ValueError('Latest85 values audit binding changed')
    old_confirmed_ref=latest['new_source_options']['--user-confirmed-source-truth-authority']
    old_confirmed=read(old_confirmed_ref['path']);confirmed=deepcopy(old_confirmed);rebindings=[]
    for record in confirmed['records']:
        if record['family']!='native_ice_curve':continue
        old_packet_ref=record['packet_reference']
        if ref(old_packet_ref['path'])!=old_packet_ref:raise ValueError('Original confirmed ice packet bytes changed')
        original_packet=read(old_packet_ref['path']);i=record['original_row_index']
        original_row=original_packet['rows'][i];current_row=packet['rows'][i]
        canonical=lambda row:json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
        if (canonical(original_row)!=canonical(current_row)
            or hashlib.sha256(canonical(current_row)).hexdigest()!=record['source_row_canonical_json_sha256']
            or current_row['id']!=record.get('source_row_id',record['id'])
            or any(not np.array_equal(record[k],current_row[k])for k in ('uv','sigma_px'))):
            raise ValueError('Final87 confirmed source row index/hash/UV/sigma must remain exact')
        rebindings.append(dict(id=record['id'],original_row_index=i,source_row_canonical_json_sha256=record['source_row_canonical_json_sha256'],
            old_packet_reference=old_packet_ref,new_packet_reference=ref(qualified),row_fields_byte_canonical_exact=True))
        record['packet_reference']=ref(qualified)
    confirmed.update(parent_authority_reference=old_confirmed_ref,
        final_native_ice_packet_reference=ref(qualified),
        source_binding_revision=dict(rule='Only packet_reference of unchanged confirmed native_ice_curve rows rebinds to final87',
            old30_group_ids_UV_sigma_supports_and_physical_identity_unchanged=True,records=rebindings))
    confirmedpath=SYS/'current_user_ice_corrections/truth_append/confirmed_same30_rebound_final87.json';write(confirmedpath,confirmed)
    rebindproof=SYS/'current_user_ice_corrections/truth_append/final87_confirmed_same30_packet_binding_proof.json'
    write(rebindproof,dict(schema='rinkspace.final87-confirmed-same30-exact-source-row-packet-rebinding.v1',passed=True,
        parent_authority=old_confirmed_ref,derived_authority=ref(confirmedpath),final_packet=ref(qualified),
        total_records=len(confirmed['records']),rebound_native_ice_records=rebindings,
        all_non_ice_records_unchanged=[r for r in confirmed['records']if r['family']!='native_ice_curve']==[r for r in old_confirmed['records']if r['family']!='native_ice_curve'],
        UV_sigma_original_rowindex_rowhash_groupid_supports_and_qualification_unchanged=True,
        original_authority_packet_receipt_and_old82_new3_source_files_modified=False,solver_started=False))
    audit=deepcopy(old);audit.update(schema='rinkspace.same-complete-q-current-user-source-and-exact-parent-caps-append-audit.v1',
        protected_same_q_caps=ref(protectedpath),new_group_count=85,
        new_source_options=deepcopy(latest['new_source_options']),families=deepcopy(current['families']),
        combined_same_q_actual_audits=[ref(oldpath),ref(latestpath)],
        source_compaction_verification=ref(originalproof),
        final87_confirmed_authority_exact_row_rebinding=ref(rebindproof),
        final_five_RGB_source_admission=ref(SYS/'source_pixel_review_additional/five_RGB_ice_lines_actual_NativeIce_append_receipt.json'),
        fresh_complete_source_profiles_J_H_required=True,old_complete_source_cache_reuse_authorized=False)
    audit['new_source_options']['--additional-ice-source']=rawref
    audit['new_source_options']['--user-confirmed-source-truth-authority']=ref(confirmedpath)
    for family,values in audit['families'].items():
        parent_n=old['families'][family]['old_count'];existing=protected['families'][family];m=len(existing['group_ids'])
        if any(values[k][:m]!=existing[k]for k in ('group_ids','limits','uncertainties')):raise ValueError('Protected82 exact prefix changed')
        values['old_count']=parent_n
    # Source-only observed point inventories; no camera/J recomputation.
    sparse_latest=sum(2*len(r['uv'])for r in raw['rows'][79:82] if r['split']=='fit')
    five=sum(2*len(r['uv'])for r in raw['rows'][82:] if r['split']=='fit')
    excluded=sum(2*len(r['excluded_point_indices'])for r in qualification['records'])
    if (sparse_latest,five,excluded)!=(196,488,30):raise ValueError('Actual final87 source row inventory differs')
    audit['added_direct_manual_centre_scalars']=old['added_direct_manual_centre_scalars']+sparse_latest
    audit['total_added_source_scalars']=old['total_added_source_scalars']+2*sparse_latest+five-excluded-2934
    audit['new_complete_source_scalar_count']=old['new_complete_source_scalar_count']+2*sparse_latest+five-excluded-2934
    audit['source_row_change_breakdown']=dict(original71_to_review82_add=2240,latest_user3_native_and_centres=2*sparse_latest,
        five_RGB_source_scalars=five,qualified_native_points_physically_removed=excluded,
        original_bad_and_dense_absolute_scalars_physically_removed=2934)
    auditpath=SYS/'same_q27fa_review85_final87_compact_startup_audit.json';write(auditpath,audit)
    values=deepcopy(current);values.update(schema='rinkspace.exact-parent-plus-current-source-cap-values.v1',audit=ref(auditpath),families=audit['families'])
    valuepath=SYS/'current85_final87_compact_caps_bound_to_exact27fa.json';write(valuepath,values)
    configpath=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review85_corrected_compact_20261010.json'
    config=read(ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_speed_candidate_20261010.json');options=config['source_options']
    for key,reference in latest['new_source_options'].items():options[key]=reference['path']
    options['--user-confirmed-source-truth-authority']=str(confirmedpath)
    options['--additional-ice-source']=str(qualified)
    seed=Path(old['complete_state_reference']['path']);truth=Path(old['old_caps_reference']['path'])
    with np.load(seed,allow_pickle=False)as z:
        damping=float(z['damping']);next_alpha=float(z['next_line_search_fraction']);qcount=len(z['q'])
    options['--camera-seed']=str(seed);options['--truth-restoration-state']=str(truth);options['--initial-damping']=damping
    options['--original-absolute-input-compaction-source']=str(SYS/'nearboard49_original_source_mass_coarsening_evidence_20261010.json')
    scope=dict(schema='rinkspace.reviewed-current-manual-append-startup-binding.v1',same_q_public_audit=ref(auditpath),
        exact_caps_values=ref(valuepath),protected_same_q_caps=ref(protectedpath),
        source_qualified_native_packet_sources={'additional_real_native_ice_curves':ref(qualified)},admitted_operator_sources={})
    fields={'static_motion_source_role_retirement':'--static-motion-source-role-retirement-source',
        'original_motion_far_board_role_correction':'--original-motion-far-board-role-source',
        'original_ice_pixel_correction':'--original-ice-pixel-correction-source',
        'original_source_point_qualification':'--original-source-point-qualification-source',
        'original_absolute_input_compaction':'--original-absolute-input-compaction-source'}
    for field,option in fields.items():scope['admitted_operator_sources'][field]=dict(reference=ref(options[option]))
    scopepath=SYS/'same_q27fa_review85_final87_compact_startup_binding.json';write(scopepath,scope)
    options['--truth-restoration-reviewed-manual-append-audit']=str(scopepath)
    for key in list(options):
        if key.startswith('--reuse-') and key not in ('--reuse-source-jacobian',):options.pop(key)
    config['continuation']=dict(complete_accepted_state=str(seed),checkpoint_file_sha256=old['complete_state_reference']['sha256'],
        truth_restoration_state=str(truth),truth_restoration_state_sha256=old['old_caps_reference']['sha256'],
        state=dict(q_sha256=old['complete_q_sha256'],parameters=qcount,damping=damping,next_line_search_fraction=next_alpha,fields=77,profiles=18),
        previous_input_configuration=str(ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_speed_candidate_20261010.json'),
        current85_caps_source=ref(valuepath),same_q_actual_source_audits=[ref(oldpath),ref(latestpath)],startup_binding=ref(scopepath),
        same_q_original71_and_review82_caps_uncertainties_exact=True,latest_user3_full_truth_appended=True,
        original66_corrected_source_pixels=66,original_unqualified_absolute_points_physically_removed=12,
        nearboard49_active_points_before=1850,nearboard49_active_points_after=395,
        new_five_RGB_FIT_points=244,unqualified_native_points_physically_removed=15,
        expected_complete_active_source_scalars=audit['new_complete_source_scalar_count'],
        next_start_requires_fresh_source_J=True,no_cached_J_H_g_or_prefix_imported=True,
        numerical_startup_prepared=True,solver_run_started=False,accuracy_accepted=False)
    write(configpath,config)
    print(json.dumps(dict(config=str(configpath),source_rows=audit['new_complete_source_scalar_count'],
        source_delta=audit['total_added_source_scalars'],damping=damping,next_alpha=next_alpha,startup_binding=str(scopepath))))

if __name__=='__main__':main()
