"""Measure every native presentation that shares a canonical exposure.

The original exposure clock stays unchanged. Re-encoded presentation pixels
can differ, so a repeated-exposure marker cannot replace reading that frame's
RGB. These readouts are evidence checks, never additional FIT observations.
"""
from __future__ import annotations

import hashlib
import copy

import cv2
import numpy as np

from .programme_source_measurement_cache import (
    restore_exact_navigation_queries, supplement_selected_profile_quantiles,
    recheck_source_candidate_exclusions)


def compare_material_candidates(canonical, current, inventory):
    """Compare actual interfaces at identical source search stations.

    The matching allowance is the measured interface readout blur plus native
    pixel quantization. Paint width and projected geometry supply no allowance.
    Physical identity still belongs to the separate source identity review.
    """
    for name in ('query_curve', 'query_uv', 'query_normal', 'query_interval'):
        left, right = np.asarray(canonical[name]), np.asarray(current[name])
        if left.shape != right.shape or not np.array_equal(left.astype(right.dtype), right):
            raise ValueError('Native presentation changed the canonical source search stations: '+name)
    count = len(canonical['candidate_uv'])
    status = np.zeros(count, np.uint8)
    uv = np.full((count, 2), np.nan)
    sides = np.full((count, 2, 2), np.nan)
    own_indices = np.full(count, -1, np.int32)
    current_queries = np.asarray(current['candidate_query'])
    canonical_quantiles = np.asarray(canonical.get('candidate_side_quantiles_uv',
        np.full((count, 2, 3, 2), np.nan)), float)
    quantiles = np.asarray(current['candidate_side_quantiles_uv'], float)
    native_quantization = .5*(float(canonical['source_readout_quantization_step_px'])
        +float(current['source_readout_quantization_step_px']))
    groups = {}
    for i, query in enumerate(current_queries):
        groups.setdefault(int(query), []).append(i)
    for index, query in enumerate(canonical['candidate_query']):
        query = int(query)
        if canonical['candidate_flags'][index] & 2:
            status[index] = 2
            continue
        if (canonical['candidate_flags'][index] & 1
                or inventory[int(canonical['query_curve'][query])]['operator'] == 'unresolved'
                or not np.isfinite(canonical_quantiles[index]).all()):
            continue
        matches = np.asarray(groups.get(query, ()), np.int32)
        if not len(matches):
            status[index] = 3
            continue
        # Both sides of a stroke must independently agree. A material-interface
        # candidate has the same single observed interface in its two slots.
        source_sides = np.asarray(canonical['candidate_sides'][index], float)
        current_sides = np.asarray(current['candidate_sides'][matches], float)
        blur_left = np.linalg.norm(canonical_quantiles[index, :, 2]
            -canonical_quantiles[index, :, 0], axis=-1)
        blur_right = np.linalg.norm(quantiles[matches, :, 2]
            -quantiles[matches, :, 0], axis=-1)
        allowance = .5*(blur_left[None]+blur_right)+native_quantization
        distances = np.linalg.norm(current_sides-source_sides[None], axis=-1)
        supported = np.isfinite(allowance).all(1) & (distances <= allowance).all(1)
        corresponding = matches[supported]
        if not len(corresponding):
            status[index] = 3
            continue
        clean = corresponding[(np.asarray(current['candidate_flags'])[corresponding] & 3) == 0]
        if len(clean) != 1:
            if not len(clean) and np.all(np.asarray(current['candidate_flags'])[corresponding] & 2):
                status[index] = 2
            else:
                status[index] = 4
            continue
        own = int(clean[0])
        status[index] = 1
        own_indices[index] = own
        # The output uses the current presentation's actual interface q50,
        # never the canonical UV or the projected search station.
        sides[index] = quantiles[own, :, 1]
        uv[index] = sides[index].mean(axis=0)
    # Two canonical candidates cannot claim the same current source interface.
    supported = np.flatnonzero(status == 1)
    if len(supported):
        values, counts = np.unique(own_indices[supported], return_counts=True)
        repeated = supported[np.isin(own_indices[supported], values[counts > 1])]
        status[repeated] = 4
        uv[repeated] = np.nan
        sides[repeated] = np.nan
    return dict(canonical_candidate_indices=np.arange(count, dtype=np.int32),
        canonical_candidate_status=status, measured_native_uv=uv,
        measured_native_sides_uv=sides,matched_native_candidate_indices=own_indices)


def check_native_alias_paint_precision(checked,current,rgb,inventory):
    """Check matched interfaces in this presentation without advancing tracks."""
    from .vision_ice_model_context_identity import source_paint_pixel_accuracy_check
    result=dict(checked)
    for key in ('canonical_candidate_status','measured_native_uv','measured_native_sides_uv'):
        result[key]=np.asarray(checked[key]).copy()
    own=np.asarray(checked['matched_native_candidate_indices'])
    query=np.asarray(current['candidate_query']);curves=np.asarray(current['query_curve'])
    field=np.asarray(rgb,np.float32);receipts=[]
    for ci,definition in enumerate(inventory):
        if definition['operator']!='paint_center':continue
        canonical_indices=np.flatnonzero(result['canonical_candidate_status']==1)
        indices=own[canonical_indices]
        take=curves[query[indices]]==ci
        canonical_indices,indices=canonical_indices[take],indices[take]
        if not len(indices):continue
        receipt=source_paint_pixel_accuracy_check(rgb,current,indices,definition,
            native_rgb_field=field)
        accepted=np.asarray(receipt['accepted_candidate_indices'])
        held=canonical_indices[~np.isin(indices,accepted)]
        result['canonical_candidate_status'][held]=3
        result['measured_native_uv'][held]=np.nan
        result['measured_native_sides_uv'][held]=np.nan
        receipts.append(dict(curve_id=definition['curve_id'],
            actual_current_candidate_indices=indices.tolist(),
            accepted_actual_current_candidate_indices=accepted.tolist(),
            pending=receipt['pending'],own_native_source_binding=receipt['own_native_source_binding']))
    result['native_paint_pixel_accuracy_checks']=receipts
    return result


def review_native_alias(*, frame, canonical_measurement, exclusion_mask,
        source_ice_exclusion_mask, source_ice_scope_mask, fresh_board_measurement,
        navigation_reader, canonical_old_uv, canonical_old_uv_start_index,
        canonical_raw_bgr=None, canonical_board_rows=(),
        candidate_local_clearance=None, primary_person_components=None,
        canonical_ice_source_rows=(),canonical_exclusion_mask=None,
        canonical_person_components=None,profile_reader_class=None):
    binding = frame['source_binding']
    raw = np.asarray(frame['raw_bgr'])
    digest = hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
    if digest != binding['decoded_native_BGR_sha256']:
        raise ValueError('Native alias review RGB differs from its original source binding')
    if frame['representative'] or int(frame['native_pts']) == int(frame['actual_pts']):
        raise ValueError('This reader checks native aliases, not new canonical exposures')
    canonical = canonical_measurement
    if int(canonical['source_binding']['actual_pts']) != int(frame['actual_pts']):
        raise ValueError('Native alias and canonical exposure identity differ')
    restore_exact_navigation_queries(canonical, navigation_reader, frame['camera25'])
    effective_inventory = canonical.get('effective_source_role_inventory', navigation_reader.inventory)
    if canonical_raw_bgr is not None:
        canonical_digest = hashlib.sha256(np.ascontiguousarray(canonical_raw_bgr).tobytes()).hexdigest()
        if canonical_digest != canonical['source_binding']['decoded_native_BGR_sha256']:
            raise ValueError('Held representative RGB is not this alias canonical source')
        # Original early cache chunks predate gradient-quantile storage. Read
        # their own original RGB support instead of inventing a tolerance.
        complete = np.flatnonzero((np.asarray(canonical['candidate_flags']) & 1) == 0)
        supplement_selected_profile_quantiles(cv2.cvtColor(canonical_raw_bgr, cv2.COLOR_BGR2RGB),
            canonical, complete, effective_inventory, backend=navigation_reader.profile_backend)
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    current = navigation_reader.read(rgb,
        int(frame['native_pts']), frame['camera25'], source_ice_exclusion_mask,
        shot_id=binding['shot_id'])
    current['source_binding'] = dict(binding, source_pts=int(frame['native_pts']),
        native_pts=int(frame['native_pts']))
    role_evidence = canonical.get('source_role_override_evidence', {})
    if role_evidence:
        from .vision_ice_source_role_override import remeasure_confirmed_source_roles
        roles = {r['curve_id']:r for r in effective_inventory}
        overrides = {name:dict(confirmed_measurement_operator=roles[name]['operator'],
            confirmed_material_signal=roles[name]['material_signal'],
            confirmed_material_transition=roles[name].get('material_transition'),
            semantic_definition=roles[name].get('source_role_authority'),
            source_role_evidence=authority) for name,authority in role_evidence.items()}
        restore_exact_navigation_queries(current, navigation_reader, frame['camera25'])
        current, effective_inventory = remeasure_confirmed_source_roles(
            rgb, current, source_ice_exclusion_mask,
            navigation_reader.inventory, overrides, profile_backend=navigation_reader.profile_backend)
    guided=canonical.get('source_guided_queries')
    if guided is not None:
        from .vision_ice_source_guides import remeasure_actual_source_guided_queries
        current=remeasure_actual_source_guided_queries(rgb,current,source_ice_exclusion_mask,
            effective_inventory,guided,profile_backend=navigation_reader.profile_backend)
    local_proofs = {}
    if candidate_local_clearance is not None:
        if primary_person_components is None:
            raise ValueError('Native local clearance requires its own measured person contours')
        # The canonical state supplies static-material navigation only. Each
        # alias must independently prove visibility in its own current RGB.
        # Its temporary state must never advance the canonical exposure clock.
        own_clearance = copy.copy(candidate_local_clearance)
        own_clearance.inventory = effective_inventory
        cleared = own_clearance.consume(rgb, current, source_ice_exclusion_mask,
            source_ice_scope_mask=source_ice_scope_mask,
            known_source_rows=canonical_ice_source_rows,
            **{key:primary_person_components[key] for key in
                ('primary_box_mask','measured_person_mask','other_person_boxes_mask')})
        current = cleared['measurement']
        source_ice_exclusion_mask = cleared['exclusion_mask']
        local_proofs = cleared['candidate_local_source_clearance']
    # Recheck the comparison context against the alias's own effective source
    # mask without overwriting the canonical presentation's original flags.
    canonical = dict(canonical)
    recheck_source_candidate_exclusions(canonical, source_ice_exclusion_mask)
    checked = compare_material_candidates(canonical, current, effective_inventory)
    checked = check_native_alias_paint_precision(checked,current,rgb,effective_inventory)
    board_checks = dict(board_material_check_pending=True,
        canonical_board_row_count=len(canonical_board_rows), native_rows=[])
    try:
        from .vision_board_native_alias import review_board_native_alias
    except ModuleNotFoundError as error:
        if error.name != 'rinkspace.vision_board_native_alias':
            raise
    else:
        board_checks = review_board_native_alias(canonical_raw_bgr=canonical_raw_bgr,
            canonical_board_rows=canonical_board_rows, raw_bgr=raw,
            exclusion_mask=exclusion_mask, source_binding=binding,
            precomputed_measurement=fresh_board_measurement,
            canonical_exclusion_mask=canonical_exclusion_mask,
            canonical_person_components=canonical_person_components,
            person_components=primary_person_components,
            **({'profile_reader_class':profile_reader_class} if profile_reader_class is not None else {}))
    return dict(native_pts=int(frame['native_pts']), actual_pts=int(frame['actual_pts']),
        decoded_native_BGR_sha256=digest, native_RGB_actually_measured=True,
        canonical_UV_checked_against_current_native_material=True,
        new_FIT_observations_created=False, physical_identity_verified_by_alias_match=False,
        board_native_checks=board_checks,
        native_candidate_local_source_clearance=local_proofs, **checked)
