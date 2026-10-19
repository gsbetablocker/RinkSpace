"""Actual original-image boundary reread on a native alias, with no tracking."""
from __future__ import annotations

import hashlib
import cv2
import numpy as np

from calibration.program_tracking.source_board_measurements import measure_board_source_frame
from calibration.program_tracking.far_rail_source_components import source_exclusion
from .vision_programme_board_review import BoardProgrammeSourceReader, _sample, SOURCE_SAMPLING_OPERATOR


def review_board_native_alias(*, canonical_raw_bgr, canonical_board_rows,
                              raw_bgr, exclusion_mask, source_binding,
                              precomputed_measurement=None,
                              canonical_exclusion_mask=None,
                              canonical_person_components=None,
                              person_components=None,
                              profile_reader_class=None):
    """Keep canonical identity; obtain every output UV from this native RGB.

    Canonical UV only navigates a local source read. No canonical tracker,
    source clock, solver, camera, fitted residual, or array is modified.
    Missing or competing native transitions remain explicit pending rows.
    """
    canonical = np.asarray(canonical_raw_bgr)
    current = np.asarray(raw_bgr)
    if (canonical.shape != (1080, 1920, 3) or current.shape != canonical.shape
            or canonical.dtype != np.uint8 or current.dtype != np.uint8):
        raise ValueError('Both actual original native BGR images are required')
    current_digest = hashlib.sha256(current.tobytes()).hexdigest()
    canonical_digest = hashlib.sha256(canonical.tobytes()).hexdigest()
    expected = source_binding.get('decoded_native_BGR_sha256', source_binding.get('decoded_BGR_sha256'))
    if expected is not None and expected != current_digest:
        raise ValueError('Native alias RGB differs from its source binding')
    current_exclusion = source_exclusion(current, np.asarray(exclusion_mask, bool))
    if precomputed_measurement is None:
        measurement = measure_board_source_frame(current, exclusion_mask, source_binding)
        bands = measurement['boundary_candidates']
    elif isinstance(precomputed_measurement, dict):
        measured_binding = precomputed_measurement.get('source', {})
        for key, alternate in (('native_pts', 'source_pts'),
                               ('actual_pts', 'actual_pts'),
                               ('decoded_BGR_sha256', 'decoded_native_BGR_sha256')):
            expected = source_binding.get(key, source_binding.get(alternate))
            measured = measured_binding.get(key, measured_binding.get(alternate))
            if expected is not None and measured is not None and expected != measured:
                raise ValueError('Precomputed board read belongs to a different native source frame')
        bands = precomputed_measurement['boundary_candidates']
    else:
        bands = precomputed_measurement
    reader_class = BoardProgrammeSourceReader if profile_reader_class is None else profile_reader_class
    if not isinstance(reader_class, type) or not issubclass(reader_class, BoardProgrammeSourceReader):
        raise TypeError('The native profile reader must preserve the public board source-reader contract')
    reader = reader_class.__new__(reader_class)
    local_clearance, context_xy = None, None
    if (canonical_exclusion_mask is not None and canonical_person_components is not None
            and person_components is not None and canonical_board_rows):
        clocks = {(int(r['native_pts']), int(r['actual_pts']), r.get('shot_id'))
                  for r in canonical_board_rows if r.get('native_pts') is not None}
        if len(clocks) != 1:
            raise ValueError('Canonical board rows must belong to one actual source image')
        native, actual, shot = next(iter(clocks))
        own_native = int(source_binding.get('native_pts', source_binding.get('source_pts')))
        if (actual == int(source_binding['actual_pts']) and native < own_native
                and shot == source_binding.get('shot_id')):
            from .vision_board_local_clearance import KnownBoardMaterialBoxClearance
            local_clearance = KnownBoardMaterialBoxClearance(allow_native_alias_same_exposure=True)
            canonical_binding = dict(source_binding, source_pts=native, native_pts=native,
                actual_pts=actual, decoded_native_BGR_sha256=canonical_digest,
                decoded_BGR_sha256=canonical_digest)
            local_clearance.begin(canonical, canonical_exclusion_mask,
                                  canonical_binding, canonical_person_components)
            local_clearance.finish()
            local_clearance.begin(current, exclusion_mask, source_binding, person_components)
    results = []
    pending = not bool(canonical_board_rows)
    for row_number, row in enumerate(canonical_board_rows):
        uv = np.asarray(row.get('uv', ()), np.float64).reshape(-1, 2)
        normal = np.asarray(row.get('normal', ()), np.float64).reshape(-1, 2)
        permitted = (row.get('static_material_confirmed') is True
                     and row.get('pixel_location_confirmed') is True
                     and row.get('actual_source_RGB_measured') is True
                     and row.get('model_projection_or_fit_residual_used_for_identity') is False)
        result = dict(canonical_row_index=row_number,
            physical_material_id=row.get('physical_material_id'),
            canonical_point_count=len(uv), native_pts=int(source_binding.get('native_pts', source_binding.get('source_pts'))),
            actual_pts=int(source_binding['actual_pts']), uv=[], normal=[],
            actual_source_RGB_measured=False, canonical_UV_used_as_output=False,
            source_sampling_operator=SOURCE_SAMPLING_OPERATOR,
            missing_or_competing_canonical_point_indices=list(range(len(uv))))
        if row.get('curve', {}).get('kind') == 'shared_board_pattern_material_points':
            from .vision_board_material_point_native_alias import review_discrete_board_material_point_alias
            checked = review_discrete_board_material_point_alias(canonical_raw_bgr=canonical,
                row=row, raw_bgr=current, canonical_exclusion_mask=canonical_exclusion_mask,
                exclusion_mask=current_exclusion, source_binding=source_binding, permitted=permitted)
            result.update(checked)
            pending |= len(checked['uv']) != len(uv)
            results.append(result)
            continue
        if not permitted or normal.shape != uv.shape or not len(uv):
            result['status'] = 'pending_canonical_physical_boundary_or_source_normal'
            pending = True
            results.append(result)
            continue
        normal = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        curve = row.get('curve', {})
        circle = curve.get('kind') in ('board_circle', 'shared_board_pattern_circle_curve')
        radius = 1.5 if circle else 6. if 'vertical' in curve.get('edge', '') else 3.
        entity = row.get('source_entity_context_uv')
        if (profile_reader_class is not None and curve.get('extent_scope') == 'internal_graphic_only'
                and entity is not None):
            # The narrow interface must not use the opposite border to define
            # its source chroma direction. This extent is navigation only.
            bounds = np.asarray(entity, float)
            lo, hi = bounds.min(0), bounds.max(0)
            vertices = np.asarray([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]])
            span = np.ptp(vertices @ normal.T, axis=0)
            radius = np.maximum(1.5, np.minimum(3., .3*span))
        source_distances = np.asarray(radius)*np.array([-1., 1.]) if np.ndim(radius) == 0 else np.asarray(radius)[:, None]*np.array([-1., 1.])
        sides = _sample(canonical, uv[:, None]+normal[:, None]*source_distances[..., None])
        contrast = sides[:, 1]-sides[:, 0]
        strength = np.linalg.norm(contrast, axis=1)
        contrast /= np.maximum(strength[:, None], 1e-9)
        template = dict(anchor=dict(curve=curve, edge=curve.get('edge', row.get('edge', '')),
                                    physical_material_id=row['physical_material_id'],
                                    source_annotation_reference=row.get('source_identity_authority', row.get('source_annotation_reference')),
                                    model_target=row.get('model_target', {}),
                                    source_component_correspondence=row.get('source_component_correspondence')),
                        source_contrast=strength, contrast=contrast)
        track = dict(template=template, search_uv=uv.copy(), normals=normal.copy())
        if entity is not None:
            track['source_entity_context_uv'] = np.asarray(entity, float)
        membership = None
        same_source_exposure = (int(row.get('actual_pts', -1)) == int(source_binding['actual_pts'])
            and row.get('shot_id') == source_binding.get('shot_id')
            and int(row.get('native_pts', -1)) < int(source_binding.get('native_pts', source_binding.get('source_pts'))))
        if same_source_exposure and row.get('source_identity_confirmed') is True:
            membership = dict(kind='known_original_source_printed_board_material',
                physical_material_id=row['physical_material_id'], source_identity_confirmed=True,
                source_identity_authority=row['source_identity_authority'],
                current_actual_pts=int(source_binding['actual_pts']),
                current_native_pts=int(source_binding.get('native_pts', source_binding.get('source_pts'))),
                decoded_native_BGR_sha256=current_digest,
                canonical_native_pts=int(row['native_pts']),
                canonical_decoded_native_BGR_sha256=canonical_digest,
                actual_original_source_printed_board_membership_confirmed=True,
                source_entity_identity_method='same_actual_exposure_native_alias_source_material_profile_confirmation_required',
                unknown_rail_column_does_not_invent_rail_UV=True,
                model_projection_or_fit_residual_used_for_identity=False)
        points = reader._read_edge(current.astype(np.float32), current_exclusion, bands, track,
            known_source_board_material=membership)
        local_proofs = {}
        if local_clearance is not None and membership is not None:
            if context_xy is None:
                context_mask = np.zeros((1080, 1920), np.uint8)
                for context_row in canonical_board_rows:
                    entity = context_row.get('source_entity_context_uv')
                    if entity is None or context_row.get('source_identity_confirmed') is not True:
                        continue
                    entity = np.asarray(entity, float)
                    lo = np.maximum(0, np.floor(entity.min(0))).astype(int)
                    hi = np.minimum([1920, 1080], np.ceil(entity.max(0)+1)).astype(int)
                    context_mask[lo[1]:hi[1], lo[0]:hi[0]] = 255
                context_mask[local_clearance.current['hard']] = 0
                keys = cv2.SIFT_create(nfeatures=2000, contrastThreshold=.014).detect(
                    local_clearance.current['gray'], context_mask)
                context_xy = np.asarray([key.pt for key in keys], np.float32).reshape(-1, 2)
            entity = row.get('source_entity_context_uv')
            if entity is not None:
                entity = np.asarray(entity, float)
                take = (context_xy >= entity.min(0)).all(1)&(context_xy <= entity.max(0)).all(1)
                track.update(support_xy=context_xy[take], source_entity_context_uv=entity)
            else:
                track['support_xy'] = np.empty((0, 2), np.float32)
            extra, proof = local_clearance.additional_points(reader, track,
                physical_identity_confirmed=True, known_source_board_material=membership)
            existing = {int(point[0]) for point in points}
            added = [point for point in extra if int(point[0]) not in existing]
            points = sorted([*points, *added], key=lambda point: point[0])
            local_proofs = {int(point[0]): proof[int(point[0])] for point in added}
        if membership is not None:
            from .vision_board_local_clearance import _descriptor, _clear_context
            extent = 6. if row.get('model_target', {}).get('kind') == 'current_scene_board_print_contour' else 24. if 'vertical' in template['anchor']['edge'] else 10.
            confirmed, material_support = [], {}
            for point in points:
                index = int(point[0])
                if np.linalg.norm(point[1]-uv[index]) > 3.:
                    continue
                current_mask = (local_clearance.current['hard'] if index in local_proofs else current_exclusion)
                canonical_mask = (local_clearance.previous['hard'] if local_clearance is not None
                    else np.asarray(canonical_exclusion_mask, bool) if canonical_exclusion_mask is not None
                    else np.zeros((1080, 1920), bool))
                if (not _clear_context(dict(hard=current_mask), point[1], point[2], extent)
                        or not _clear_context(dict(hard=canonical_mask), uv[index], normal[index], extent)):
                    continue
                a, a_strength = _descriptor(canonical, uv[index], normal[index], extent)
                b, b_strength = _descriptor(current, point[1], point[2], extent)
                correlation = float(a@b)
                if min(a_strength, b_strength) < 20. or correlation < .90:
                    continue
                confirmed.append(point)
                material_support[index] = dict(actual_original_source_material_context_correlation=correlation,
                    canonical_actual_RGB_interface_uv=uv[index].tolist(),
                    current_actual_RGB_interface_uv=point[1].tolist(),
                    full_current_profile_retained_hard_person_exclusion_clear=True,
                    original_conservative_person_union_clear=index not in local_proofs,
                    current_source_interface_actually_remeasured=True,
                    canonical_UV_not_output=True)
            points = confirmed
            result.update(source_board_material_membership_evidence=membership,
                          actual_native_material_profile_confirmation=material_support)
            if local_proofs:
                kept = {int(point[0]) for point in points}
                result['candidate_local_source_clearance'] = {index: proof for index, proof in local_proofs.items() if index in kept}
                result['historical_person_union_modified'] = False
        indices = [p[0] for p in points]
        result.update(uv=[p[1].tolist() for p in points],
            normal=[p[2].tolist() for p in points],
            actual_source_RGB_measured=bool(points),
            canonical_point_indices=indices,
            actual_readout_blur_interval_px=[p[4].tolist() for p in points],
            normal_interval_px=[p[4].tolist() for p in points],
            quantization_step_px=1.,
            source_native_pixel_quantization_step_px=1.,
            source_interpolation_grid_px=1./32., profile_sampling_step_px=.5,
            missing_or_competing_canonical_point_indices=[i for i in range(len(uv)) if i not in set(indices)],
            status=('actual_native_alias_visible_boundary_read' if len(indices) == len(uv)
                    else 'pending_missing_or_competing_actual_native_boundary'),
            identity_origin='same_known_physical_boundary_current_original_native_RGB_readout',
            source_profile_reader_class=f'{reader_class.__module__}.{reader_class.__name__}',
            measurement_method=row.get('source_readout_uncertainty', {}).get('measurement_method'),
            source_board_instance_identity_inherited_from_canonical_review=True,
            normal_measurement_origin='canonical_source_normal_navigation_only',
            model_projection_or_fit_residual_used_for_identity=False)
        if row.get('model_target', {}).get('kind') == 'current_scene_board_print_contour':
            result.update(model_target=dict(row['model_target']),
                          observation_semantics='board_print_feature')
        pending |= len(indices) != len(uv)
        results.append(result)
    return dict(board_material_check_pending=bool(pending),
        canonical_board_row_count=len(canonical_board_rows), native_rows=results)
