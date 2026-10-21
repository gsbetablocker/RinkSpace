"""Verify complete source-only wide-context affine corrections and unknowns."""
import hashlib
from collections import Counter
import numpy as np
from .io_utils import read_json, read_jsonl


def source_wide_readout_rows(reference, verify):
    g = read_json(verify(reference))
    if g['schema'] != 'rinkfit.source_current_wide_context_affine_observation_generation.v1':
        raise ValueError('Explicit wide-context actual source generation required')
    for key in ('input_contract','original_generation','original_source_qualification','original_typed_source_rows',
                'reader','full_affine_support_reader','full_affine_support_QA','pre_support_readout_records',
                'source_rows','readout_records','frame_receipt','correction_patch','source_lineage_QA','semantic_visual_review'):
        verify(g[key])
    for r in g['reader_helpers']: verify(r)
    c = read_json(verify(g['input_contract']))
    for key in ('original_source_generation','original_physical_source_qualification','original_typed_source_rows',
                'old_packet','selected_original_rows','source_config','reader','reader_execution_bytes',
                'full_affine_support_reader','full_affine_support_QA','pre_support_readout_records'):
        verify(c[key])
    for r in c['reader_helpers']: verify(r)
    if (c['reader']['sha256'] != g['reader']['sha256'] or c['reader_execution_bytes']['sha256'] != g['reader']['sha256']
        or c['model_or_projection_input'] is not None or not g['no_model_or_interpolated_UV']
        or g['statistical_covariance_issued'] or not g['full_batch_not_success_subset']):
        raise ValueError('Wide source readout changed its actual reader or introduced model UV/precision')
    old_all = list(read_jsonl(verify(c['original_typed_source_rows'])))
    old = {r['source_row_id']:r for r in old_all if r['shot_id']=='s04' and r['surface_instance_id']=='Ad2'}
    rows = list(read_jsonl(verify(g['source_rows']))); records = list(read_jsonl(verify(g['readout_records'])))
    recs = {r['source_row_id']:r for r in records}; patch = read_json(verify(g['correction_patch']))
    deltas = {r['source_row_id']:r for r in patch['rows']}; lineage = read_json(verify(g['source_lineage_QA']))
    frames = {r['native_pts']:r for r in read_json(verify(g['frame_receipt']))}
    support = read_json(verify(g['full_affine_support_QA']))
    if (len(old)!=c['row_count'] or len(rows)!=len(old) or len(records)!=len(old)
        or len(recs)!=len(records) or len(deltas)!=len(old) or set(recs)!=set(old) or set(deltas)!=set(old)
        or len({r['source_row_id']for r in rows})!=len(old) or g['all_source_rows']!=len(old)
        or support['original_total_rows']!=len(old)):
        raise ValueError('Wide correction lost a source row or selected a successful subset')
    params=c['parameters'];half=params['template_half'];shape=params['template_shape']
    if shape != [97,97] or half!=48 or not params['fixed_birth_template']:
        raise ValueError('Unexpected actual wide-context source operator')
    f64hash=lambda v:hashlib.sha256(np.asarray(v,dtype='<f8').tobytes()).hexdigest()
    counters=Counter(); tracks=set(); seen=set()
    for row in rows:
        rid=row['source_row_id'];prior=old[rid];rec=recs[rid];d=deltas[rid];seen.add(rid)
        if (row['physical_track_id']!=prior['physical_track_id'] or rec['track_id']!=prior['physical_track_id']
            or row['native_pts']!=prior['native_pts'] or row['current_rgb_sha256']!=prior['current_rgb_sha256']
            or rec['current_RGB_sha256']!=prior['current_rgb_sha256'] or rec['native_pts']!=prior['native_pts']
            or row['surface_instance_id']!='Ad2' or row['shot_id']!='s04' or rec['model_or_world_projection_used']
            or d['old_source_uv_binary64_sha256']!=f64hash(prior['source_uv'])
            or rec['original_uv_binary64_sha256']!=f64hash(prior['source_uv'])
            or row['original_source_uv']!=prior['source_uv'] or rec['original_uv']!=prior['source_uv']
            or row['source_readout_has_calibrated_precision'] or row['new_human_review']):
            raise ValueError('Wide readout original UV/clock/RGB/physical-source lineage differs')
        if row['source_mean_changed']!=rec['source_UV_changed'] or d['source_UV_changed']!=rec['source_UV_changed']:
            raise ValueError('Final patch and final affine support gate disagree')
        status=rec['status'];counters[status]+=1
        if status=='wide_context_direct_source_supported':
            F=rec['forward'];B=rec['reverse'];A=np.asarray(F['warp']);Z=np.asarray(B['warp'])
            fuv=np.asarray(F['search_origin'])+A@np.array([half,half,1.])
            buv=np.asarray(B['search_origin'])+Z@np.array([half,half,1.])
            corners=np.array([[-half,-half],[-half,half],[half,-half],[half,half]],float)
            cycle=np.linalg.norm(corners@(Z[:,:2]@A[:,:2]-np.eye(2)).T+buv-np.asarray(rec['birth_uv']),axis=1)
            if (F['status']!='registered' or B['status']!='registered' or fuv.tolist()!=F['registered_uv']
                or buv.tolist()!=B['registered_uv'] or min(F['registered_NCC'],B['registered_NCC'])<params['NCC_min']
                or cycle.max()>params['all_four_template_corner_cycle_max_px']
                or abs(cycle.max()-rec['affine_patch_cycle_max_px'])>1e-10):
                raise ValueError('Wide UV lacks actual reciprocal affine centre and complete four-corner cycle')
            supports={x['direction']:x for x in rec['full_affine_context_support']}
            if set(supports)!={'forward','reverse'}:raise ValueError('Both actual affine mask supports required')
            for direction,warp in [('forward',F),('reverse',B)]:
                margin=warp['translation_peak_NCC']-warp['other_branch_peak_NCC']-(1-warp['translation_peak_NCC'])
                if (margin<=0 or abs(margin-rec[direction+'_branch_identity_margin'])>1e-10
                    or warp['branch_exclusion_halfwidth_xy']!=params['branch_exclusion_halfwidth_xy']):
                    raise ValueError('Repeated letter branch is not uniquely supported by wider actual context')
                C=np.array([[0,0,1],[96,0,1],[96,96,1],[0,96,1]])@np.asarray(warp['warp']).T+np.asarray(warp['search_origin'])
                proof=supports[direction];lo=np.floor(C.min(0)-1).astype(int);hi=np.ceil(C.max(0)+1).astype(int)
                exposure=rec['native_pts']if direction=='forward'else rec['birth_native_pts'];fr=frames[exposure]
                if (proof['native_pts']!=exposure or proof['actual_RGB_sha256']!=fr['RGB_sha256']
                    or proof['mask_sha256']!=fr['mask_sha256'] or proof['excluded_pixels_in_bbox']!=0
                    or proof['status']!='pass_full_affine_context_support'
                    or proof['conservative_affine_support_bbox_plus_1px_halo']!=[lo.tolist(),hi.tolist()]
                    or np.max(np.abs(C-np.asarray(proof['registered_corners_native_uv'])))>1e-10):
                    raise ValueError('New readout affine support leaves actual native mask/graphics evidence')
            if rec['source_UV_changed']:
                if fuv.tolist()!=row['current_uv'] or np.linalg.norm(fuv-np.asarray(prior['source_uv']))<=1:
                    raise ValueError('Applied wide readout mean is not the stored direct affine centre')
                tracks.add(rec['track_id'])
            elif f64hash(row['current_uv'])!=f64hash(prior['source_uv']):
                raise ValueError('Unchanged supported mean lost original binary64 source UV')
        elif rec['source_UV_changed']:
            raise ValueError('Unknown or masked readout cannot issue a new precise UV')
        elif f64hash(row['current_uv'])!=f64hash(prior['source_uv']):
            raise ValueError('Unknown readout must preserve original source UV')
        if (row['source_uv']!=row['current_uv'] or row['current_uv']!=rec['current_uv'] or row['current_uv']!=d['current_uv']
            or row['source_mean_binary64_sha256']!=f64hash(row['current_uv'])
            or d['current_uv_binary64_sha256']!=f64hash(row['current_uv'])
            or row['source_readout_generation_id']!=g['observation_generation_id']
            or row['source_readout_status']!=status or row['source_readout_unknown_reason']!=rec['unknown_reason']):
            raise ValueError('Production readout row is not this exact final observation generation')
    changed=sum(r['source_UV_changed']for r in records)
    if (changed!=g['source_uv_updated_rows'] or len(tracks)!=g['source_tracks_updated']
        or lineage['source_mean_changed_rows']!=changed or not lineage['all_original_IDs_once']):
        raise ValueError('Actual wide source correction effect differs from its complete generation')
    return {r['source_row_id']:r for r in rows}
