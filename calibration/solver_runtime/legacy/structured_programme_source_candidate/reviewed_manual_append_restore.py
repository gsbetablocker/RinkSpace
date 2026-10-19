"""Authenticate reviewed source appends at the actual complete startup state.

This prepares a standard restoration packet. It never creates a source graph,
changes observations, relaxes an old cap, or authorizes reuse of a normal/J.
"""
from pathlib import Path
import copy,hashlib,json
import numpy as np

SCHEMA='rinkspace.reviewed-current-manual-append-startup-binding.v1'
BASE_AUDIT_SCHEMA='rinkspace.stopped-complete-q-public-current11-source-and-old71-cap-exact-append-audit.v1'
CURRENT_AUDIT_SCHEMA='rinkspace.same-complete-q-current-user-source-and-exact-parent-caps-append-audit.v1'
CURRENT_VALUES_SCHEMA='rinkspace.exact-parent-plus-current-source-cap-values.v1'

def _sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def _read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def _plain(value):return json.loads(json.dumps(value,allow_nan=False,default=lambda v:v.tolist()if isinstance(v,np.ndarray)else v.item()if isinstance(v,np.generic)else str(v)))
def _ref(path):return dict(path=str(Path(path).resolve()),sha256=_sha(path))
def _check_ref(reference):
    path=Path(reference['path']).resolve()
    if _sha(path)!=reference['sha256']:raise ValueError('Reviewed append reference bytes changed: '+str(path))
    return path

def _checked_truth_operator_revision(scope,audit,parent,old,manifest,factors):
    """Admit only the evidenced same-q change of finite support distance."""
    revision=scope.get('admitted_truth_support_operator_revision')
    if revision is None:
        if _plain(manifest.get('human_truth_support_operator'))!=_plain(old.get('human_truth_support_operator')):
            raise ValueError('Changed finite truth support operator requires explicit same-q semantic migration')
        return None
    proof_path=_check_ref(revision['actual_same_q_public_audit']);proof=_read(proof_path)
    from .human_stroke_truth import human_support_operator_contract
    contract=human_support_operator_contract()
    if(proof.get('schema')!='rinkspace.same-q-original86-finite-support-rectangle-semantic-revision-audit.v1'
        or proof.get('passed')is not True or proof.get('complete_q_sha256')!=audit['complete_q_sha256']
        or revision.get('cap_policy')!='minimum_parent_cap_and_actual_revised_same_q_violation'
        or _plain(revision.get('contract'))!=_plain(contract)
        or _plain(proof.get('operator_contract'))!=_plain(contract)
        or _plain(manifest.get('human_truth_support_operator'))!=_plain(contract)
        or _plain(manifest['source_contract'].get('human_truth_support_operator'))!=_plain(contract)
        or proof.get('source_UV_normal_normal_support_tangent_support_bytes_unchanged')is not True
        or proof.get('original_parameter_and_camera_world_and_circle_layer_identity_unchanged')is not True
        or proof.get('new_hard_operator_target_and_fresh_J_required')is not True
        or _check_ref(proof['original_restoration_caps'])!=_check_ref(audit['old_caps_reference'])
        or _check_ref(proof['original_runtime_manifest'])!=_check_ref(audit['parent_manifest_reference'])
        or _check_ref(proof['complete_state_reference'])!=_check_ref(audit['complete_state_reference'])
        or audit['added_direct_manual_centre_scalars']!=0
        or scope.get('protected_same_q_caps')is not None):
        raise ValueError('Finite truth operator revision lacks its exact unchanged-support same-q certificate')
    if audit['total_added_source_scalars']!=0:
        # A soft-source append is independent of this hard-support semantic
        # migration. Admit only its measured, same-q protected-source proof;
        # the original complete source count gate below remains authoritative.
        append=scope.get('admitted_truth_preserving_native_source_append')
        if not isinstance(append,dict):
            raise ValueError('Combined finite truth revision needs the actual native soft-source append certificate')
        append_proof=_read(_check_ref(append['actual_same_q_public_audit']))
        bound=manifest['additional_real_native_board_edges'];previous=old['additional_real_native_board_edges']
        observed_delta=int(bound['scalar_rows'])-int(previous['scalar_rows'])
        expected=audit['new_source_options']['--additional-board-source']
        if(append_proof.get('schema')!='rinkspace.actual-native-board185-full-base-J-and-curve-budget.v1'
            or append_proof.get('passed')is not True or append_proof.get('passed_R_J')is not True
            or append_proof.get('complete_q_sha256')!=audit['complete_q_sha256']
            or append_proof.get('original109_UV_sigma_support_roles_splits_exact')is not True
            or append_proof.get('old15_HELD_R_bitexact')is not True
            or append_proof.get('original54_human_primary_full_all_queries_bounds_carriers_bitexact')is not True
            or append_proof.get('original30_confirmed_all_queries_bounds_carriers_bitexact')is not True
            or append_proof.get('new_authority_actual_hash_validated')is not True
            or append_proof.get('FD_max_error',np.inf)>append_proof.get('FD_existing_allowed',-np.inf)
            or _check_ref(append_proof['candidate'])!=_check_ref(expected)
            or Path(bound['packet_path']).resolve()!=_check_ref(expected)
            or bound['packet_sha256']!=expected['sha256']
            or append_proof.get('scalar_rows')!=bound['scalar_rows']
            or observed_delta<=0 or observed_delta!=audit['total_added_source_scalars']
            or append.get('source_scalar_delta')!=observed_delta
            or manifest['additional_real_native_ice_curves']['new_scalar_rows']!=old['additional_real_native_ice_curves']['new_scalar_rows']):
            raise ValueError('Combined finite truth revision differs from the actual protected native soft-source append')
        protected_sources=_read(_check_ref(append_proof['legal_source_refs_handoff']))['source_options']
        for key in ('--direct-human-truth-authority','--direct-human-full-stroke-guard',
            '--direct-human-stroke-cap-support','--practical-human-location-envelope',
            '--user-confirmed-source-truth-authority'):
            if _check_ref(audit['new_source_options'][key])!=Path(protected_sources[key]).resolve():
                raise ValueError('Combined native source append differs from its actual protected authority: '+key)
    for reference in contract['implementation_sources']:_check_ref(reference)
    for family in ('native','confirmed'):
        if _plain(factors[family].contract.get('human_truth_support_operator'))!=_plain(contract):
            raise ValueError('Actual truth factor support operator differs: '+family)
    if proof.get('group_count')!=sum(len(v['limits'])for v in parent['families'].values()):
        raise ValueError('Finite truth semantic revision cannot append or remove manual groups')
    for family,saved in parent['families'].items():
        row=proof['families'][family]
        old=np.asarray(row['legacy_actual_same_q_violation_px'],float)
        new=np.asarray(row['corrected_actual_same_q_violation_px'],float)
        limits=np.asarray(saved['limits'],float)
        expected=np.minimum(limits,np.where(new<=1e-8,0.,new))
        if(row['group_ids']!=saved['group_ids'] or row['original_caps_px']!=saved['limits']
            or row['original_uncertainties_exact']!=saved['uncertainties']
            or old.shape!=limits.shape or new.shape!=limits.shape
            or not np.isfinite(old).all()or not np.isfinite(new).all()or np.any(new<0.)
            or np.any(old>limits+1e-8)or np.any(new>old+1e-8)
            or np.any(new[limits==0.]>1e-8)
            or not np.array_equal(np.asarray(row['proposed_ratchet_only_corrected_semantic_caps_px']),expected)
            or (family!='native'and not np.array_equal(old,new))):
            raise ValueError('Finite truth semantic revision would change or expand original support/caps: '+family)
    return proof_path,proof

def prepare_reviewed_manual_append_state(audit_path,old_state_path,manifest_path,manifest,q,factors,*,
        group_ids,violation_buffer,uncertainties,source_target,role_target,write):
    scope=_read(audit_path)
    if scope.get('schema')!=SCHEMA:raise ValueError('Explicit reviewed manual append startup binding required')
    audit_file=_check_ref(scope['same_q_public_audit']);audit=_read(audit_file)
    values_file=_check_ref(scope['exact_caps_values']);values=_read(values_file)
    if(audit.get('schema')not in (BASE_AUDIT_SCHEMA,CURRENT_AUDIT_SCHEMA) or audit.get('passed')is not True
        or values.get('schema')not in ('rinkspace.exact-old71-plus-current11-cap-values.v1',CURRENT_VALUES_SCHEMA)
        or _check_ref(values['audit'])!=audit_file):raise ValueError('Actual same-q public append audit required')
    if _check_ref(audit['old_caps_reference'])!=Path(old_state_path).resolve():raise ValueError('Reviewed append belongs to another parent cap state')
    parent=_read(old_state_path);old_manifest_file=_check_ref(audit['parent_manifest_reference']);old=_read(old_manifest_file)
    if(_check_ref(parent['source_input_manifest_reference'])!=old_manifest_file
        or parent['source_target']!=source_target(old)
        or parent['complete_q_sha256']!=audit['complete_q_sha256']
        or values['complete_q_sha256']!=audit['complete_q_sha256']):raise ValueError('Parent complete source/q target differs')
    seed=_check_ref(audit['complete_state_reference'])
    if _check_ref(parent['complete_state_reference'])!=seed:raise ValueError('Parent complete checkpoint reference differs')
    with np.load(seed,allow_pickle=False)as z:
        if not np.array_equal(z['q'],q):raise ValueError('Reviewed manual append complete q differs')
    if manifest.get('physical_domain_initial_q_sha256')!=audit['complete_q_sha256']:raise ValueError('Actual startup q differs from reviewed source state')
    truth_revision=_checked_truth_operator_revision(scope,audit,parent,old,manifest,factors)
    if(manifest['full_parameters']!=old['full_parameters']or manifest['source_rows']!=audit['new_complete_source_scalar_count']
        or manifest['source_rows']-old['source_rows']!=audit['total_added_source_scalars']):raise ValueError('Actual append parameter/source row inventory differs')
    prior=manifest.get('principal_point_engineering_prior')
    prior_scope=scope.get('admitted_engineering_prior_append')
    if prior is not None or prior_scope is not None:
        if prior is None or prior_scope is None:raise ValueError('Engineering prior append needs its exact same-q source certificate')
        prior_file=_check_ref(prior_scope['actual_same_q_public_audit']);prior_proof=_read(prior_file)
        if(prior_proof.get('passed')is not True or prior_proof.get('rows')!=6
            or prior_proof.get('complete_q_sha256')!=audit['complete_q_sha256']
            or _plain(prior)!=_plain(prior_proof.get('contract'))
            or _plain(manifest['source_contract'].get('principal_point_engineering_prior'))!=_plain(prior)
            or prior.get('parameter_count')!=manifest['full_parameters']
            or prior.get('engineering_sigma_native_px')!=32.
            or prior.get('current_native_principal_target_px')!=[960.,540.]
            or prior.get('radial_rows_admitted')is not False
            or prior.get('new_geometry_or_camera_parameters')!=0
            or manifest['source_contract'].get('original_source_scalar_prefix_preserved')is not True
            or manifest['source_contract'].get('original_source_scalars_before_PP6')!=manifest['source_rows']-6):
            raise ValueError('Actual engineering PP6 source contract or complete row append differs')
        for reference in prior['implementation_sources']:_check_ref(reference)
    manual=audit['new_source_options']
    for reference in manual.values():_check_ref(reference)
    for field,option in (('full_stroke_guard','--direct-human-full-stroke-guard'),('source_2D_stroke_cap_support','--direct-human-stroke-cap-support'),('practical_manual_location_envelope','--practical-human-location-envelope')):
        bound=factors['native'].contract[field];expected=manual[option]
        if Path(bound['path']).resolve()!=Path(expected['path']).resolve()or bound['sha256']!=expected['sha256']:raise ValueError('Actual full source support identity differs: '+field)
    plan_keys=('additional_real_native_board_edges','additional_real_native_ice_curves')
    for family,option in zip(plan_keys,('--additional-board-source','--additional-ice-source')):
        bound=manifest[family];expected=manual[option]
        if(Path(bound['packet_path']).resolve()!=Path(expected['path']).resolve()or bound['packet_sha256']!=expected['sha256']):
            # Source-only eligibility can retire erroneous assistant pixels
            # while keeping every parent row, UV, sigma, role and source slot.
            # It is explicitly scoped; it cannot admit another source packet.
            qualified=scope.get('source_qualified_native_packet_sources',{}).get(family)
            if family!='additional_real_native_ice_curves' or qualified is None:
                raise ValueError('Actual admitted native source identity differs')
            qualified_path=_check_ref(qualified)
            if(Path(bound['packet_path']).resolve()!=qualified_path or bound['packet_sha256']!=qualified['sha256']):
                raise ValueError('Actual qualified native source differs from the explicit startup binding')
            from calibration.solver_runtime.source_point_qualification import load_source_point_qualification
            qualification=load_source_point_qualification(qualified_path,_read(qualified_path),source_family='additional_native_ice_curves')
            if(qualification is None or _plain(bound.get('source_only_review_qualification'))!=_plain(qualification.contract)
                or _check_ref(qualification.contract['original_packet'])!=_check_ref(expected)
                or qualification.contract['original_packet']['sha256']!=expected['sha256']):
                raise ValueError('Qualified native source changed its authenticated parent or eligibility contract')
    for key,entry in scope['admitted_operator_sources'].items():
        path=_check_ref(entry['reference']);bound=manifest[key]
        if key=='static_motion_source_role_retirement':
            compound=_read(path);append=compound['source_reviewed_dynamic_validity_append'];_check_ref(append)
            parent_ui=old[key]
            continuing_append=parent_ui.get('schema')=='rinkspace.static-motion-source-role-retirement.v1.source-reviewed-dynamic-append'
            if continuing_append and parent_ui.get('source_reviewed_dynamic_validity_append')!=append:
                raise ValueError('Existing dynamic source validity append identity changed')
            expected_ui=copy.deepcopy(parent_ui['original_UI_retirement_contract'] if continuing_append else parent_ui)
            compact=manifest.get('original_absolute_input_compaction')
            compact_ref=scope['admitted_operator_sources'].get('original_absolute_input_compaction',{}).get('reference')
            if compact is not None:
                if(compact_ref is None or compact.get('schema')!='rinkspace.original-absolute-active-input-compaction.v1'
                    or Path(compact['packet_path']).resolve()!=_check_ref(compact_ref)
                    or compact['packet_sha256']!=compact_ref['sha256']):
                    raise ValueError('Retirement index composition needs the exact admitted absolute source compaction')
                composition=compact.get('original_retirement_low_block_binding_composition')
                parent_compact=old.get('original_absolute_input_compaction')
                parent_composition=None if parent_compact is None else parent_compact.get('original_retirement_low_block_binding_composition')
                if parent_composition is not None:
                    if(not isinstance(composition,dict)
                        or composition.get('before')!=parent_composition.get('before')
                        or parent_composition.get('after')!=expected_ui['programme_block']
                        or parent_ui['programme_block']!=expected_ui['programme_block']):
                        raise ValueError('Existing UI source qualification differs across exact absolute layout composition')
                elif not isinstance(composition,dict)or composition.get('before')!=expected_ui['programme_block']:
                    raise ValueError('Original UI source range differs before absolute compaction')
                low=[r for r in compact['blocks']if r['block']=='low']
                if len(low)!=1:raise ValueError('One actual absolute-compacted low source block required')
                row=low[0];after=dict(family='programme:low',original_start=row['compact_start'],
                    original_stop=row['compact_start']+row['compact_rows'],motion_point_count=row['motion_point_count'])
                if composition.get('after')!=after or bound.get('programme_block')!=after:
                    raise ValueError('Actual current UI/dynamic source range disagrees with absolute compaction')
                expected_ui['programme_block']=after
            if(bound.get('schema')!='rinkspace.static-motion-source-role-retirement.v1.source-reviewed-dynamic-append'
                or bound.get('original_UI_retirement_contract')!=expected_ui
                or bound.get('source_reviewed_dynamic_validity_append')!=append
                or bound.get('source_scalar_slots_removed')!=0 or bound.get('new_scalar_rows')!=0):raise ValueError('Actual reviewed dynamic source validity differs')
        elif key in ('original_board_curve_integration','original_ice_curve_integration','native_base_curve_integration','original_wide_curve_integration'):
            source_contract=manifest['source_contract']
            if key in ('original_board_curve_integration','original_ice_curve_integration','original_wide_curve_integration'):
                source_contract=source_contract.get('evaluation_source_contract',{})
            if(bound is None or _plain(bound.get('policy_reference'))!=_plain(entry['reference'])
                or _plain(source_contract.get(key))!=_plain(bound)
                or bound.get('fresh_source_J_and_profiles_required')is not True):
                raise ValueError('Actual admitted curve integration policy differs: '+key)
            for reference in bound['implementation_sources']:_check_ref(reference)
        elif(Path(bound['packet_path']).resolve()!=path or bound['packet_sha256']!=entry['reference']['sha256']):raise ValueError('Actual reviewed role source differs: '+key)
    direct_old=old['source_contract']['direct_manual_centre_secondary'];direct_new=manifest['source_contract']['direct_manual_centre_secondary']
    expected=copy.deepcopy(direct_old);expected['authority']=manual['--direct-human-truth-authority'];expected['added_scalar_rows']+=audit['added_direct_manual_centre_scalars']
    if direct_new!=expected:raise ValueError('Direct manual centre append changed more than admitted source rows')
    protected=None
    if scope.get('protected_same_q_caps') is not None:
        protected_file=_check_ref(scope['protected_same_q_caps']);protected=_read(protected_file)
        protected_audit=_read(_check_ref(protected['audit']))
        if(protected.get('schema')!='rinkspace.exact-old71-plus-current11-cap-values.v1'
            or protected.get('complete_q_sha256')!=audit['complete_q_sha256']
            or protected_audit.get('complete_q_sha256')!=audit['complete_q_sha256']
            or protected_audit.get('passed')is not True):
            raise ValueError('Protected existing caps need the exact passed same-q source audit')
    packet=copy.deepcopy(parent);families={}
    for family,factor in factors.items():
        if factor is None:continue
        actual_ids=group_ids(factor,family);frozen=values['families'][family];saved=parent['families'][family];n=frozen['old_count']
        authority=manual['--direct-human-truth-authority'if family=='native'else'--user-confirmed-source-truth-authority']if family!='wide'else saved['authority']
        if _plain(factor.contract['authority'])!=authority or _check_ref(authority)!=Path(factor.contract['authority']['path']).resolve():raise ValueError('Actual manual authority differs: '+family)
        expected_limits=saved['limits']
        if truth_revision is not None:
            revised=truth_revision[1]['families'][family]
            expected_limits=revised['proposed_ratchet_only_corrected_semantic_caps_px']
            if(n!=len(saved['group_ids'])or len(frozen['group_ids'])!=n
                or frozen['violations_px']!=revised['corrected_actual_same_q_violation_px']):
                raise ValueError('Same-q finite truth semantic revision differs from actual cap values: '+family)
        if(actual_ids!=frozen['group_ids']or actual_ids[:n]!=saved['group_ids']
            or frozen['limits'][:n]!=expected_limits or frozen['uncertainties'][:n]!=saved['uncertainties']):raise ValueError('Parent manual group/cap prefix differs: '+family)
        current_uncertainty=np.asarray(uncertainties(factor,family),float);current_violation=np.asarray(violation_buffer(factor,q,family=family),float)
        if not np.array_equal(current_uncertainty,np.asarray(frozen['uncertainties'],float)):raise ValueError('Actual manual uncertainties differ: '+family)
        if not np.isfinite(current_violation).all()or not np.allclose(current_violation,np.asarray(frozen['violations_px'],float),rtol=0.,atol=1e-8):raise ValueError('Actual same-q source violations differ: '+family)
        limits=np.asarray(frozen['limits'],float).copy()
        limits[n:]=np.where(current_violation[n:]<=1e-8,0.,current_violation[n:]+current_uncertainty[n:])
        if protected is not None:
            existing=protected['families'][family];m=len(existing['group_ids'])
            if(any(existing[key]!=protected_audit['families'][family][key]for key in ('group_ids','limits','uncertainties'))
                or actual_ids[:m]!=existing['group_ids'] or frozen['limits'][:m]!=existing['limits']
                or frozen['uncertainties'][:m]!=existing['uncertainties']
                or not np.array_equal(current_uncertainty[:m],np.asarray(existing['uncertainties'],float))
                or np.any(current_violation[:m]>np.asarray(existing['limits'],float)+1e-8)):
                raise ValueError('Protected existing same-q group/cap/uncertainty prefix changed')
            # Existing82 remains exact; only genuinely appended groups receive
            # an initial recovery limit from their actual startup operator.
            limits[:m]=existing['limits']
        if not np.isfinite(limits).all()or np.any(limits<0.)or np.any(current_violation>limits+1e-8):raise ValueError('Actual source would exceed reviewed recovery cap: '+family)
        if np.any(limits>np.where(current_violation<=1e-8,0.,current_violation+current_uncertainty)):raise ValueError('Reviewed cap would expand actual fresh limits: '+family)
        families[family]=dict(authority=authority,group_ids=actual_ids,limits=limits.tolist(),uncertainties=frozen['uncertainties'])
    if sum(len(v['limits'])for v in families.values())!=audit['new_group_count']:raise ValueError('Complete reviewed manual group inventory differs')
    packet.update(source_input_manifest_reference=_ref(manifest_path),source_target=source_target(manifest),source_role_operator_target=role_target(manifest),families=families,
        authenticated_reviewed_manual_append=dict(startup_binding=_ref(audit_path),same_q_public_audit=_ref(audit_file),parent_truth_state=_ref(old_state_path),
            original71_limits_uncertainties_and_group_prefix_exact=True,new11_caps_only_actual_same_q_violation=True,
            engineering_hard_band_expansion_for_new11_px=0.,actual_new_source_rows=manifest['source_rows'],old_source_J_H_or_profile_reuse_authorized=False))
    if protected is not None:
        packet['authenticated_reviewed_manual_append'].update(protected_same_q_caps=_ref(protected_file),
            protected_existing_group_limits_and_uncertainties_exact=True,
            protected_group_count=sum(len(v['group_ids'])for v in protected['families'].values()))
    packet['authenticated_reviewed_manual_append'].update(
        actual_same_q_group_count=sum(len(v['group_ids'])for v in families.values()),
        new_groups_receive_only_actual_source_violation_recovery_limits=True)
    if truth_revision is not None:
        packet['authenticated_reviewed_manual_append'].update(
            original71_limits_uncertainties_and_group_prefix_exact=False,
            original_group_ids_uncertainties_and_source_support_arrays_exact=True,
            admitted_truth_support_operator_revision=_plain(scope['admitted_truth_support_operator_revision']),
            actual_same_q_truth_semantic_audit=_ref(truth_revision[0]),
            all_previous_caps_only_ratcheted_to_actual_revised_same_q_violation=True,
            old_query_caps_not_claimed_semantically_equivalent=True)
    target=Path(manifest_path).parent/'reviewed_manual_append_truth_restoration_state.json'
    if target.exists():raise ValueError('Reviewed restoration output already exists; preserve prior result')
    write(target,packet)
    return target
