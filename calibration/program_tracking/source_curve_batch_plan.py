"""Lightweight, uninstalled exact-curve batching plans for strict source packs."""
from __future__ import annotations

from collections import OrderedDict, Counter
import hashlib
import json
import numpy as np

from .strict_source_representatives import _declared_paths


_OPERATOR_FIELDS = ('curve_observation_kind', 'point_correspondence',
    'must_preserve_upper_branch', 'scanline_branch', 'scanline_fixed_axis',
    'extent_side', 'clip_boundary_x', 'clip_blur_allowance_px',
    'source_operator_array_fields_present')
_FINITE_TARGETS = frozenset(('current_scene_ice_curve', 'advertisement_boundary',
    'current_scene_board_print_contour', 'current_scene_board_edge'))


def compile_source_curve_batch_plan(headers, offsets, sigma, *, source_reference,
        model_reference, model_q_sha256, source_metadata):
    """Keep exact source slices/paths; group candidates without FIT admission.

    Missing sigma and every typed/unknown operator remain singleton entries.
    The complete 100-world/shared-optics route must be qualified separately
    before any plan can become an executable solver adapter.
    """
    offsets=np.asarray(offsets)
    sigma=np.asarray(sigma)
    if (offsets.shape!=(len(headers)+1,) or offsets.dtype.kind not in 'iu'
            or offsets[0]!=0 or offsets[-1]!=len(sigma) or np.any(np.diff(offsets)<0)
            or sigma.ndim!=1 or np.isinf(sigma).any() or np.any(sigma[np.isfinite(sigma)]<=0)):
        raise ValueError('Original source row offsets and finite-positive-or-missing sigma required')
    expected=source_metadata['expected_actual_pts']
    processed_list=source_metadata['processed_actual_pts']
    if (any(type(p) is not int for p in expected+processed_list)
            or len(set(expected))!=len(expected) or len(set(processed_list))!=len(processed_list)
            or not set(processed_list).issubset(expected)
            or source_metadata['all_exposures_processed'] is not (set(processed_list)==set(expected))):
        raise ValueError('Immutable expected/processed source clock is inconsistent')
    processed=set(processed_list)
    finite=np.isfinite(sigma)&(sigma>0)
    prefix=np.r_[0,np.cumsum(finite,dtype=np.int64)]
    groups=OrderedDict();split_points=Counter();status_rows=Counter();status_points=Counter()
    payload_identity=hashlib.sha256(b'rinkspace.immutable-source-batch-plan-point-order.v1')
    payload_identity.update(offsets.tobytes());payload_identity.update(sigma.tobytes())
    for ri,h in enumerate(headers):
        lo,hi=map(int,offsets[ri:ri+2]);n=hi-lo
        if (h.get('original_source_row_index',ri)!=ri or h['point_count']!=n
                or h['actual_pts'] not in processed or h['split'] not in ('fit','held')):
            raise ValueError('Exact original row order, cut clock and explicit FIT/HELD required')
        target=h['model_target']
        if 'scene_q_sha256' in target and target['scene_q_sha256']!=model_q_sha256:
            raise ValueError('Source print descriptor belongs to another frozen model state')
        operator={k:h.get(k) for k in _OPERATOR_FIELDS}
        typed=(h.get('point_correspondence') is True or h.get('must_preserve_upper_branch') is True
            or h.get('curve_observation_kind') not in (None,'measured_transition_location')
            or any(h.get(k) is not None for k in ('scanline_branch','scanline_fixed_axis',
                'extent_side','clip_boundary_x','clip_blur_allowance_px'))
            or bool(h.get('source_operator_array_fields_present')))
        complete_support=int(prefix[hi]-prefix[lo])==n
        if not complete_support:status='singleton_pending_original_source_sigma'
        elif typed:status='singleton_original_typed_operator'
        elif target.get('kind') not in _FINITE_TARGETS:status='singleton_unknown_original_operator'
        else:status='candidate_exact_finite_primitive_reuse_requires_producer_qualification'
        context=dict(shot_id=h['shot_id'],actual_pts=h['actual_pts'],native_pts=h['native_pts'],
            decoded_BGR_sha256=h['decoded_BGR_sha256'],scope=h['scope'],physical_material_id=h['physical_material_id'],
            model_target=target,observation_semantics=h['observation_semantics'],split=h['split'],original_operator=operator)
        identity=json.dumps(context,sort_keys=True,separators=(',',':'),ensure_ascii=False)
        payload_identity.update(identity.encode('utf8'))
        key=(status,identity) if status.startswith('candidate_') else ('singleton',ri)
        if key not in groups:
            groups[key]=dict(source_context=context,preparation_status=status,original_rows=[],points=0,
                solver_FIT_admitted=False,execution_ready=False)
        # The public path helper reads only len(uv) and source station indices.
        # No duplicate UV array is loaded, made up or stored by this plan.
        paths,basis=_declared_paths(dict(h,uv=range(n)))
        rows=dict(original_source_row_index=ri,original_global_point_range=[lo,hi],
            source_header_and_proof_locator=dict(source_pack_reference='source_reference',source_row_index=ri),
            original_point_count=n,original_finite_sigma_points=int(prefix[hi]-prefix[lo]),
            independent_visible_source_path_basis=basis,independent_visible_source_paths=paths,
            original_operator=operator,original_split=h['split'],actual_pts=h['actual_pts'],native_pts=h['native_pts'])
        groups[key]['original_rows'].append(rows);groups[key]['points']+=n
        split_points[h['split']]+=n;status_rows[status]+=1;status_points[status]+=n
    records=list(groups.values())
    for i,g in enumerate(records):
        g['prepared_group_index']=i;g['original_row_count']=len(g['original_rows'])
    if sum(g['points'] for g in records)!=len(sigma) or sum(len(g['original_rows']) for g in records)!=len(headers):
        raise ValueError('Prepared plan lost original points or rows')
    return dict(schema='rinkspace.uninstalled-exact-source-curve-batching-plan.v1',
        source_reference=dict(source_reference),frozen_model_reference=dict(model_reference),model_q_sha256=model_q_sha256,
        original_source_points=len(sigma),original_source_rows=len(headers),prepared_groups=len(records),
        original_split_point_counts=dict(split_points),finite_positive_working_sigma_points=int(finite.sum()),
        missing_working_sigma_points=int((~finite).sum()),status_row_counts=dict(status_rows),status_point_counts=dict(status_points),
        grouped_candidates_reduce_calls_only_not_points=True,source_points_removed=0,
        original_expected_actual_pts=source_metadata['expected_actual_pts'],original_processed_actual_pts=source_metadata['processed_actual_pts'],
        all_source_exposures_processed=source_metadata['all_exposures_processed'],
        original_row_order_offsets_sigma_and_context_sha256=payload_identity.hexdigest(),
        original_UV_sides_quantiles_and_proofs_remain_in_immutable_source_slices=True,
        repeated_large_numeric_arrays_stored=False,independent_source_paths_joined=False,
        original_sigma_or_loss_weights_changed=False,source_scope_or_manual_accuracy_qualified=False,
        guards=dict(complete_world100_primitive_producer_verified=False,
            every_active_shared_optics_mount_and_view_derivative_verified=False,
            full_pack_same_q_and_nonlinear_trial_qualification_verified=False,
            complete_iteration_speed_verified=False,full_solver_consumer_installed=False,
            source_FIT_admission_authorized=False),
        groups=records,fit_started=False)
