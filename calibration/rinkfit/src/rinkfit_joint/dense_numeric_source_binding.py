"""Rebind unchanged dense source qualification and budgets to a numeric overlay.

No source pixels are read here. The exact prior raw packet remains the parent;
only a separately pinned observed-state numeric initialization is introduced.
Native entry independently verifies the source facts and the overlay boundary.
"""
from pathlib import Path
from copy import deepcopy
import json
import numpy as np
from .io_utils import read_json,file_sha256
from .dense_joint_packet import _array
from .dense_joint_proofs import _ref,_checked,_write


def bind_numeric_packet(*,prior_packet,numeric_packet,prior_source_qualification,
                        prior_engineering_weights,overlay,output):
    prior_path=Path(prior_packet).resolve();next_path=Path(numeric_packet).resolve()
    qual_path=Path(prior_source_qualification).resolve();weights_path=Path(prior_engineering_weights).resolve()
    overlay_path=Path(overlay).resolve();out=Path(output).resolve()
    before=read_json(prior_path);raw=json.loads(next_path.read_text(encoding='utf-8-sig'))
    prior_ref=_ref(prior_path);next_ref=_ref(next_path);oldqual_ref=_ref(qual_path);oldweights_ref=_ref(weights_path);overlay_ref=_ref(overlay_path)
    if raw.get('dense_parent_packet')!=prior_ref or raw.get('dense_numeric_auxiliary_overlay')!=overlay_ref:raise ValueError('Numeric packet does not inherit the exact qualified raw packet and overlay')
    allowed={'schema','dense_parent_packet','dense_parent_append','dense_numeric_auxiliary_overlay','arrays'}
    if set(raw)-allowed:raise ValueError('Numerical-only packet contains an unsupported raw metadata override')
    if raw.get('schema')!=before['schema']:raise ValueError('Numeric packet ABI differs')
    if 'arrays'in raw and raw['arrays']!=before['arrays']:raise ValueError('A numeric overlay must not change any of the eighteen raw arrays')
    append=raw.get('dense_parent_append',{})
    if set(append)-{'source_manifests'}:raise ValueError('Numeric overlay cannot change source IDs, Query/W, paths or dependence groups')
    if 'source_manifests'in append:
        for ref in append['source_manifests']:_checked(ref,next_path.parent)
    priorqual=read_json(qual_path);priorweights=read_json(weights_path);overlaydoc=read_json(overlay_path)
    if priorqual['source_packet']!=prior_ref or priorweights['source_packet']!=prior_ref or priorweights['dense_source_qualification']!=oldqual_ref:raise ValueError('Prior qualification and engineering budget bind another raw packet')
    schema=overlaydoc.get('schema')
    if schema not in ('rinkfit.observed_row_numeric_auxiliary_overlay.v1','rinkfit.segmented_native_numeric_auxiliary_overlay.v1')or overlaydoc.get('source_packet')!=prior_ref or overlaydoc.get('source_geometry_qualification')!=oldqual_ref:raise ValueError('Overlay source provenance differs from current qualification')
    required_flags={'numerical_initialization_only':True,'true_RS_verified':False,'source_UV_ID_PTS_qualification_Query_W_unchanged':True,'old_AL_duals_imported':False}
    if any(overlaydoc.get(k)is not v for k,v in required_flags.items()):raise ValueError('Overlay omitted its numeric-only, fresh true-RS boundary')
    n=int(before['row_count'])
    if int(overlaydoc['raw_rows'])!=n or int(priorqual['raw_rows'])!=n or int(priorweights['raw_rows'])!=n:raise ValueError('Numeric overlay raw denominator differs')
    dependencies=[_ref(__file__),prior_ref,oldqual_ref,overlay_ref]
    def verify_columns(specs,offset,count):
        if set(specs)!={'station','row','alpha','path'}:raise ValueError('Exactly four numeric auxiliary columns are required')
        columns={}
        for name,ref in specs.items():
            shape=ref.get('shape')
            if not isinstance(shape,list)or len(shape)!=1 or offset<0 or offset+count>int(shape[0]):raise ValueError('Numeric piece source slice is outside its actual column')
            columns[name]=_array(ref,overlay_path.parent,dtype='<u4'if name=='path'else'<f8',shape=shape)[offset:offset+count]
        if any(not np.isfinite(columns[name]).all()for name in ('station','row','alpha'))or np.any(columns['station']<0)or np.any(columns['station']>1):raise ValueError('Numeric station/row/alpha are outside their finite initialization domain')
    if schema=='rinkfit.observed_row_numeric_auxiliary_overlay.v1':
        verify_columns(overlaydoc['arrays'],0,n)
        if not isinstance(overlaydoc.get('complete_state_token'),str)or not overlaydoc['complete_state_token']:raise ValueError('Actual complete observed-state token is required')
        for key in ('candidate_model','observed_checkpoint','observed_state_binding','explicit_nuisance_manifest','issuer'):
            _checked(overlaydoc[key],overlay_path.parent);dependencies.append(overlaydoc[key])
    else:
        if overlaydoc.get('full_current_checkpoint_claimed')is not False:raise ValueError('Segmented initialization cannot claim a complete current checkpoint')
        pieces=overlaydoc.get('pieces',[]);n0=int(before['dense_source_binding']['row_start'])
        if len(pieces)!=2:raise ValueError('This dense-six boundary requires one complete prefix and one raw numeric suffix')
        cursor=0
        for index,piece in enumerate(pieces):
            start,count,offset=int(piece['start']),int(piece['count']),int(piece['source_offset'])
            if start!=cursor or count<=0:raise ValueError('Numeric pieces must form an exact nonoverlapping raw-row partition')
            verify_columns(piece['arrays'],offset,count);cursor+=count
            if index==0:
                if piece.get('kind')!='complete_native_checkpoint_prefix'or start!=0 or count!=n0 or offset!=0 or piece['source_packet']!=priorqual['parent_packet']:raise ValueError('Complete fitted prefix does not match the inherited native source packet')
                for key in ('source_packet','physical_model','complete_checkpoint','state_binding','explicit_nuisance_manifest','complete_state_verification'):
                    _checked(piece[key],overlay_path.parent);dependencies.append(piece[key])
                check=read_json(_checked(piece['complete_state_verification'],overlay_path.parent))
                if check.get('passed')is not True or int(check['raw_rows'])!=count or check['state_token']!=piece.get('complete_state_token')or check['source_identity']!=piece['source_packet']['sha256']:raise ValueError('Actual fitted complete-prefix receipt differs')
            else:
                if start!=n0 or offset!=n0 or count!=n-n0:raise ValueError('Numeric suffix does not cover the exact raw suffix')
                if piece.get('kind')=='raw_packet_numeric_seed':
                    if piece['source_packet']!=prior_ref or piece['arrays']!={key:before['arrays'][key]for key in ('station','row','alpha','path')}:raise ValueError('Numeric suffix arrays are not the current original raw packet specs')
                elif piece.get('kind')=='complete_native_checkpoint_tail':
                    if piece.get('current_source_packet')!=prior_ref:raise ValueError('Captured numeric tail is not rebound to the current source packet')
                    for key in ('source_packet','physical_model','complete_checkpoint','state_binding','explicit_nuisance_manifest','complete_state_verification','source_subset_verification'):
                        _checked(piece[key],overlay_path.parent);dependencies.append(piece[key])
                    verified=read_json(_checked(piece['complete_state_verification'],overlay_path.parent))
                    if verified.get('passed')is not True or int(verified['source_rows'])!=n or verified['complete_state_token']!=piece.get('complete_state_token')or verified['source_packet_sha256']!=piece['source_packet']['sha256']or verified['complete_checkpoint_sha256']!=piece['complete_checkpoint']['sha256']or verified.get('all_four_raw_numeric_columns_bitexact_to_complete_state')is not True or verified.get('complete_physical_model_exact_to_a0')is not True:raise ValueError('Actual complete captured-tail checkpoint receipt differs')
                    subset=read_json(_checked(piece['source_subset_verification'],overlay_path.parent))
                    refs={'old_source_packet':piece['source_packet'],'current_source_packet':prior_ref,'current_model':overlaydoc['candidate_model'],'tail_model':piece['physical_model'],'current_qualification':oldqual_ref,'complete_state_verification':piece['complete_state_verification'],'complete_checkpoint':piece['complete_checkpoint'],'numeric_array_refs':piece['arrays']}
                    if subset.get('schema')!='rinkfit.actual_captured_tail_source_subset.v1'or any(subset.get(key)!=value for key,value in refs.items()):raise ValueError('Captured-tail source subset identity differs')
                    if subset.get('all_six_physical_cameras_and_scene_JSON_exact')is not True or subset.get('original_source_identity_and_raw_16_arrays_exact')is not True or subset.get('current_active_is_strict_subset')is not True or int(subset['newly_admitted'])!=0 or int(subset['current_active'])!=int(priorweights['active_rows'])or int(subset['tail_active'])!=int(priorweights['new_active_rows']):raise ValueError('Captured-tail source subset does not preserve current geometry and qualification')
                    status=_array(subset['RS_status'],overlay_path.parent,dtype='|u1',shape=[n]);dependencies.append(subset['RS_status'])
                    active_tail=_array(priorweights['files']['active.u8'],weights_path.parent,dtype='|u1',shape=[n])[n0:]!=0
                    if int(np.count_nonzero((status[n0:]!=1)&active_tail))!=int(subset['tail_failed_RS_retained']):raise ValueError('Captured-tail invalid denominator was hidden or changed')
                    if subset.get('source_search_repeated')is not False or subset.get('old_AL_or_physical_state_imported')is not False:raise ValueError('Captured-tail boundary imported unsupported state')
                else:raise ValueError('Unsupported numeric suffix kind')
        if cursor!=n:raise ValueError('Numeric overlay omits part of the full raw denominator')
        closure=read_json(_checked(overlaydoc['projection_closure'],overlay_path.parent))
        if closure.get('clock_rigs_materials_scene_and_first4_metadata_exact')is not True or closure.get('all_other_blocks_exact')is not True or closure['prefix_model']!=pieces[0]['physical_model']or closure['current_model']!=overlaydoc['candidate_model']:raise ValueError('Fitted-prefix physical projection closure differs')
        for key in ('candidate_model','projection_closure','issuer'):
            _checked(overlaydoc[key],overlay_path.parent);dependencies.append(overlaydoc[key])
    # This verifies the byte-level numeric transfer only. The production native
    # gate verifies its complete checkpoint, physical values and allowed paths.
    unknown=int(priorqual.get('unknown_new_rows',0));active=int(priorweights['active_rows'])
    for key,expected in priorqual['membership_array_sha256'].items():
        if before['arrays'][key]['sha256']!=expected:raise ValueError('Prior qualified raw array differs: '+key)
    for key in ('weights.f64','active.u8','information_class.u8'):_checked(priorweights['files'][key],weights_path.parent)
    out.mkdir(exist_ok=False,parents=True)
    qual=deepcopy(priorqual);qual.update(source_packet=next_ref,issuer=_ref(__file__),source_geometry_qualification=oldqual_ref,
        numerical_auxiliary_overlay=overlay_ref,source_geometry_qualification_inherited_exact=True,
        actual_input_references=priorqual['actual_input_references']+dependencies,
        original_raw_source_columns_unchanged=True,numerical_auxiliary_overlay_does_not_issue_source_truth=True)
    qual_ref=_write(out/'source_qualification.json',qual)
    weights=deepcopy(priorweights);weights.update(source_packet=next_ref,dense_source_qualification=qual_ref,issuer=_ref(__file__),
        inherited_engineering_weights=oldweights_ref,numerical_auxiliary_overlay=overlay_ref,
        all_engineering_weight_class_and_active_binary_refs_unchanged=True,engineering_budget_policy_unchanged=True)
    weights_ref=_write(out/'engineering_weights.json',weights)
    result=dict(schema='rinkfit.numeric_overlay_source_qualification_rebind.v1',passed=True,packet=next_ref,
        source_qualification=qual_ref,engineering_weights=weights_ref,source_geometry_qualification=oldqual_ref,
        numerical_auxiliary_overlay=overlay_ref,prior_engineering_weights=oldweights_ref,
        raw_rows=n,active_rows=active,unknown_new_rows=unknown,source_identity_digest=qual['actual_column_identity_sha256'],
        original_eighteen_raw_array_specs_unchanged=True,source_raw_UV_ID_PTS_qualification_Query_W_and_path_authority_unchanged=True,
        current_source_semantic_generation_unchanged=True,engineering_weight_files_and_component_budgets_unchanged=True,
        source_pixels_decoded=0,old_source_guard_or_pixel_review_reexecuted=False,old_AL_duals_imported=False,
        true_RS_verified=False,fresh_trueRS_v3_required=True,native_gate_required=True,physical_quality_accepted=False)
    _write(out/'receipt.json',result);return result
