"""Bind the user's C identity correction to existing exact RGB readouts.

The old A1-labelled search was pending material data. Its pixel measurements
remain unchanged; only the subsequently user-confirmed search identity becomes
C. Neither the user's arrows nor the old navigation positions become UV truth.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_json(directory, stem, payload):
    encoded=json.dumps(payload,ensure_ascii=False,separators=(',', ':')).encode('utf8')
    digest=hashlib.sha256(encoded).hexdigest();path=directory/f'{stem}_{digest[:16]}.json'
    if path.exists() and path.read_bytes()!=encoded:raise ValueError('Immutable source evidence changed')
    if not path.exists():path.write_bytes(encoded)
    return dict(path=str(path.resolve()),sha256=digest)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workroot',type=Path,required=True)
    args=parser.parse_args();work=args.workroot.resolve();pts=230066432
    pending=work/'actual_red_material_full_span_pending_230066432.npz'
    with np.load(pending,allow_pickle=False) as saved:
        arrays={k:saved[k].copy() for k in saved.files if k!='metadata_json'}
        original_metadata=json.loads(str(saved['metadata_json']))
    old=original_metadata['frames'][0];binding=old['source_binding']
    original=work/'source_feedback/right_end_wide_actual_source_230066432.png'
    bgr=cv2.imread(str(original));rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    if (hashlib.sha256(bgr.tobytes()).hexdigest()!=binding['decoded_native_BGR_sha256']
            or hashlib.sha256(rgb.tobytes()).hexdigest()!=old['decoded_native_RGB_sha256']):
        raise ValueError('User identity frame and pending material data differ in actual original pixels')
    manifest=json.loads((work/'source_pixels/source_measurements.json').read_text(encoding='utf8'))
    indices={v['curve_id']:i for i,v in enumerate(manifest['inventory'])}
    if binding['source_pts']!=pts or binding['native_pts']!=pts:raise ValueError('Exact user source clock required')
    authority=immutable_json(work,'direct_user_C_identity_230066432',dict(
        schema='rinkspace.direct-user-original-source-material-identity.v1',
        authority_kind='direct_user_original_source_identity_confirmation',
        source_binding=binding,decoded_native_RGB_sha256=old['decoded_native_RGB_sha256'],
        original_source_image=dict(path=str(original),sha256=sha(original)),
        confirmation_received_in_root_task='private-thread-confirmation-01',
        user_confirmation=dict(curve_id='C:segment0',feature_id='C',
            statement_paraphrase='用户明确确认箭头A与B所指两段薄红漆带均为当前模型C中线。',
            arrow_tip_navigation_uv=[[340,660],[1553,684]],
            source_visibility_navigation_spans=[dict(x=[104,568],y=[655,665]),dict(x=[1312,1888],y=[676,689])],
            arrow_positions_are_exact_pixel_ground_truth=False,
            A1_or_A2_current_identity_confirmed=False),
        unmodeled_source_material=dict(navigation_uv=[1250,792],
            description='用户箭头C所指前景红标识不属于当前模型',
            model_membership='outside_current_model',fit_admission_allowed=False),
        ordinary_paint_measurement_operator='midpoint_of_two_current_actual_RGB_material_interfaces',
        current_centerline_UV_must_come_from_actual_profiles=True,
        superseded_pending_search_identity='A1:outer',
        pending_actual_readouts=dict(path=str(pending),sha256=sha(pending)),
        original_failed_A1_history_preserved=True))
    authority['kind']='direct_user_original_source_identity_confirmation'
    seed=dict(curve_id='C:segment0',physical_id='C',actual_pts=pts,native_pts=pts,
        source_sha256=binding['source_sha256'],time_base=binding['time_base'],shot_id=binding['shot_id'],
        decoded_native_RGB_sha256=old['decoded_native_RGB_sha256'],
        decoded_native_BGR_sha256=binding['decoded_native_BGR_sha256'],
        navigation_uv=[[104.,655.],[568.,665.],[1312.,677.],[1888.,689.]],
        source_material_class='static_painted_ice',surface_and_physical_identity_confirmed=True,
        authority=authority,is_exact_centerline_pixel_ground_truth=False,
        original_source_image_actually_reviewed=True,physical_identity_from_model_projection=False)
    guide=dict(seed,navigation_uv=[[8.,646.],[1908.,692.]],sampling_spacing_px=8,
        normal_search_half_extent_px=128.,source_entity_identity_status='direct_user_confirmed_C')
    packet=immutable_json(work,'ice_source_guides_C_230066432',dict(
        schema='rinkspace.reviewed-original-source-guide-packet.v1',source_sha256=binding['source_sha256'],
        time_base=binding['time_base'],guides=[guide],identity_seeds=[seed],
        source_guide_UV_is_not_exact_ground_truth=True,source_guides_are_not_fit_admission=True))
    # These are the very same RGB measurements, including competitors and
    # exclusion flags. The explicit user authority changes the pending search
    # identity; it does not turn any candidate into an admitted observation.
    arrays[f'{pts}_query_curve'][:]=indices['C:segment0']
    receipt=json.loads(json.dumps(old['source_guided_queries']))
    receipt['guides']=[guide];receipt['query_curve_ids']=['C:segment0']*len(arrays[f'{pts}_query_curve'])
    for evidence in receipt['guide_evidence']:
        evidence.update(curve_id='C:segment0',authority=authority)
    receipt['subsequent_direct_user_search_identity_correction']=dict(
        from_pending_search_curve='A1:outer',to_curve='C:segment0',authority=authority,
        original_pending_measurement_reference=dict(path=str(pending),sha256=sha(pending)),
        original_measured_UV_sides_quantiles_changed=False,candidate_flags_changed=False)
    metadata=dict(original_metadata,schema='rinkspace.direct-user-corrected-source-query-overlay.v1')
    frame=dict(old,source_guided_queries=receipt,selected_rows=[],source_identity_authority=authority,
        current_pending_query_identity='C:segment0',original_pending_search_identity='A1:outer',
        observed_center_UV_from_user_arrows=False)
    metadata['frames']=[frame]
    overlay=work/'actual_C_source_overlay_user_identity_230066432.npz'
    arrays['metadata_json']=np.asarray(json.dumps(metadata,ensure_ascii=False,separators=(',', ':')))
    if overlay.exists():
        with np.load(overlay,allow_pickle=False) as existing:
            if set(existing.files)!=set(arrays) or any(not np.array_equal(existing[k],v,equal_nan=True) if v.dtype.kind in 'fc' else not np.array_equal(existing[k],v) for k,v in arrays.items()):
                raise ValueError('Immutable C actual-source overlay changed')
    else:np.savez_compressed(overlay,**arrays)
    registry_path=work/'ice_source_identity_seeds.json';registry=json.loads(registry_path.read_text(encoding='utf8'))
    if not any(r['curve_id']==seed['curve_id'] and r['actual_pts']==pts for r in registry['records']):
        registry['records'].append(seed)
        registry['records'].sort(key=lambda r:(r['actual_pts'],r['curve_id']))
        registry_path.write_text(json.dumps(registry,ensure_ascii=False,separators=(',', ':')),encoding='utf8')
    print(json.dumps(dict(authority=authority,packet=packet,overlay=dict(path=str(overlay),sha256=sha(overlay)),
        source_identity_curves=len({r['curve_id'] for r in registry['records']}),source_identity_records=len(registry['records']),
        actual_source_queries=len(arrays[f'{pts}_query_curve']),actual_pending_readouts=len(arrays[f'{pts}_candidate_uv']),
        A1_A2_registered=False,fit_admission_done=False),ensure_ascii=False))


if __name__=='__main__':main()
