"""Two-source-image, named-board-boundary support inside a coarse person box.

No person archive or global exclusion is changed. A source-confirmed board
curve is reread in both original images, retaining measured person contours,
all other actors, overlays, real upper/lower board readouts and static texture.
"""
from __future__ import annotations

import cv2
import numpy as np

from calibration.program_tracking.far_rail_source_components import source_exclusion
from calibration.program_tracking.source_board_measurements import measure_board_source_frame, board_bracket_candidates


def _pixels(point, normal, extent):
    tangent = np.array([-normal[1], normal[0]])
    return np.rint(point[None, None]+np.arange(-extent, extent+.1)[:, None, None]*normal
                   + np.arange(-3., 3.1)[None, :, None]*tangent).astype(int).reshape(-1, 2)


def _clear_context(state, point, normal, extent):
    pixels = _pixels(point, normal, extent)
    return ((pixels >= 0).all() and (pixels < [1920, 1080]).all()
            and not state['hard'][pixels[:, 1], pixels[:, 0]].any())


def _descriptor(image, point, normal, extent):
    pixels = _pixels(point, normal, extent)
    values = image[pixels[:, 1], pixels[:, 0]].astype(float)
    values -= values.mean(axis=0)
    scale = np.linalg.norm(values)
    return values.ravel()/max(scale, 1e-9), float(scale)


class KnownBoardMaterialBoxClearance:
    def __init__(self, actual_successor=None, *, allow_native_alias_same_exposure=False):
        self.actual_successor = dict(actual_successor or {})
        self.allow_native_alias_same_exposure = bool(allow_native_alias_same_exposure)
        self.previous = None
        self.current = None

    def begin(self, image, conservative_mask, binding, components):
        primary = np.asarray(components['primary_box_mask'], bool)
        hard = (np.asarray(components['measured_person_mask'], bool)
                | np.asarray(components['other_person_boxes_mask'], bool)
                | (np.asarray(conservative_mask, bool)&~primary))
        hard = source_exclusion(image, hard)
        hard = cv2.dilate(hard.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
        self.current = dict(image=image, gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            primary=primary, hard=hard, binding=dict(binding), board=None,
            primary_box_index=components.get('primary_box_index'),
            primary_box_SAM2_coverage=components.get('primary_box_SAM2_coverage'))

    def _board(self, state):
        if state['board'] is None:
            state['board'] = measure_board_source_frame(
                state['image'], state['hard'], state['binding'])
        return state['board']['boundary_candidates']

    def _source_context_navigation(self, track, known_source_board_material=None):
        """Use actual named-board texture when a straight edge has no corner.

        This is source-image navigation. Both actual boundary profiles still
        have to be measured; neither matrix nor transported UV is emitted.
        """
        current, previous = self.current, self.previous
        points = np.asarray(track.get('support_xy', ()), np.float32).reshape(-1, 2)
        independent_scope = bool(known_source_board_material
            and known_source_board_material.get('source_identity_confirmed') is True
            and known_source_board_material.get('actual_original_source_printed_board_membership_confirmed') is True)
        paired = (np.zeros(len(points), bool) if independent_scope else
            board_bracket_candidates(points, self._board(current), current['hard']) >= 0)
        take = paired.copy()
        entity = track.get('source_entity_context_uv', track['template']['anchor'].get('source_entity_context_uv'))
        if known_source_board_material is not None and entity is not None:
            extent = np.asarray(entity, float)
            take |= ((points >= extent.min(0)).all(1)&(points <= extent.max(0)).all(1))
        points = points[take]
        if len(points) < 8:
            return None, None
        kwargs = dict(winSize=(31, 31), maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS, 40, .01))
        back, valid, _ = cv2.calcOpticalFlowPyrLK(current['gray'], previous['gray'], points[:, None], None, **kwargs)
        if back is None:
            return None, None
        forward, reverse, _ = cv2.calcOpticalFlowPyrLK(previous['gray'], current['gray'], back, None, **kwargs)
        if forward is None:
            return None, None
        back = back[:, 0]
        errors = np.linalg.norm(forward[:, 0]-points, axis=1)
        pixels = np.rint(back).astype(int)
        bounded = np.clip(pixels, [0, 0], [1919, 1079])
        good = (valid.ravel().astype(bool)&reverse.ravel().astype(bool)
                & np.isfinite(back).all(1)&(errors <= 1.)
                & (pixels >= 0).all(1)&(pixels < [1920, 1080]).all(1)
                & ~previous['hard'][bounded[:, 1], bounded[:, 0]])
        if good.sum() < 8:
            return None, None
        matrix, inliers = cv2.findHomography(points[good], back[good], cv2.RANSAC, 2.)
        if matrix is None or inliers is None or inliers.sum() < 8:
            return None, None
        from .vision_programme_board_review import _source_orientation_preserved
        keep = np.flatnonzero(good)[inliers.ravel().astype(bool)]
        if (np.ptp(points[keep], axis=0).min() < 8.
                or not _source_orientation_preserved(matrix, points[keep])):
            return None, None
        return matrix, dict(kind='known_board_material_context_actual_RGB_LK_navigation',
            actual_static_board_context_inliers=len(keep),
            source_context_forward_backward_LK_error_min_median_max_px=np.quantile(errors[keep], [0, .5, 1]).tolist(),
            source_context_previous_uv=back[keep].tolist(), source_context_current_uv=points[keep].tolist(),
            source_context_inside_known_original_printed_board_entity=True,
            source_material_scope_independent_of_missing_rail_columns=known_source_board_material is not None,
            model_projection_or_fit_residual_used=False,
            transported_source_coordinates_not_output_UV=True)

    def additional_points(self, reader, track, *, physical_identity_confirmed,
                          known_source_board_material=None):
        current, previous = self.current, self.previous
        if current is None or previous is None or not physical_identity_confirmed:
            return [], {}
        now, before = current['binding'], previous['binding']
        actual, previous_actual = int(now['actual_pts']), int(before['actual_pts'])
        native_alias = (self.allow_native_alias_same_exposure and actual == previous_actual
            and int(now.get('native_pts', now.get('source_pts'))) > int(before.get('native_pts', before.get('source_pts')))
            and now.get('source_sha256', now.get('video_sha256')) == before.get('source_sha256', before.get('video_sha256'))
            and now.get('time_base') == before.get('time_base'))
        if (now.get('shot_id') != before.get('shot_id') or (actual <= previous_actual and not native_alias)
                or (self.actual_successor and not native_alias and self.actual_successor.get(previous_actual) != actual)
                or current['primary_box_index'] is None):
            return [], {}
        guesses = np.asarray(track['search_uv'], float)
        xy = np.rint(guesses).astype(int)
        inside = (xy >= 0).all(1)&(xy < [1920, 1080]).all(1)
        columns = np.clip(xy[:, 0], 0, 1919)
        if not (inside & current['primary'][:, columns].any(axis=0)).any():
            return [], {}
        template = track['template']
        anchor = template['anchor']
        printed = anchor.get('model_target', {}).get('kind') == 'current_scene_board_print_contour'
        extent = 6. if printed else 24. if 'vertical' in anchor.get('edge', '') else 10.
        independently_known_board = bool(known_source_board_material
            and known_source_board_material.get('physical_material_id') == anchor['physical_material_id']
            and known_source_board_material.get('source_identity_confirmed') is True
            and known_source_board_material.get('actual_original_source_printed_board_membership_confirmed') is True)
        current_bands = [] if independently_known_board else self._board(current)
        candidates = reader._read_edge(current['image'].astype(np.float32), current['hard'],
            current_bands, track, known_source_board_material=known_source_board_material)
        candidates = [p for p in candidates
            if current['primary'][:, int(round(p[1][0]))].any()
            and _clear_context(current, p[1], p[2], extent)]
        if not candidates:
            return [], {}
        # Batch every interface and its two visible material sides for this
        # curve; no per-point flow call or background worker is introduced.
        support = np.asarray([p[1]+np.array([-5., 0., 5.])[:, None]*p[2]
                              for p in candidates], np.float32).reshape(-1, 1, 2)
        kwargs = dict(winSize=(31, 31), maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS, 40, .01))
        back, valid, _ = cv2.calcOpticalFlowPyrLK(current['gray'], previous['gray'], support, None, **kwargs)
        if back is None:
            return [], {}
        forward, reverse, _ = cv2.calcOpticalFlowPyrLK(previous['gray'], current['gray'], back, None, **kwargs)
        if forward is None:
            return [], {}
        error = np.linalg.norm(forward[:, 0]-support[:, 0], axis=1).reshape(-1, 3)
        flow_good = (valid.reshape(-1, 3).all(1)&reverse.reshape(-1, 3).all(1)
                     & np.isfinite(back.reshape(-1, 3, 2)).all((1, 2))
                     & (error.max(1) <= 2.))
        previous_support = back.reshape(-1, 3, 2)
        old_points = previous_support[:, 1].astype(float)
        old_normal = previous_support[:, 2]-previous_support[:, 0]
        old_normal /= np.maximum(np.linalg.norm(old_normal, axis=1, keepdims=True), 1e-9)
        context_navigation, context_evidence = None, None
        use_context = np.zeros(len(candidates), bool)
        if not flow_good.all():
            context_navigation, context_evidence = self._source_context_navigation(track, known_source_board_material)
            if context_navigation is not None:
                ids = np.flatnonzero(~flow_good)
                source_points = np.asarray([candidates[i][1] for i in ids], np.float32)
                source_ends = np.asarray([candidates[i][1]+candidates[i][2]*5. for i in ids], np.float32)
                old_points[ids] = cv2.perspectiveTransform(source_points[None], context_navigation)[0]
                transported_ends = cv2.perspectiveTransform(source_ends[None], context_navigation)[0]
                old_normal[ids] = transported_ends-old_points[ids]
                old_normal[ids] /= np.maximum(np.linalg.norm(old_normal[ids], axis=1, keepdims=True), 1e-9)
                use_context[ids] = True
        old_template = dict(template,
            source_contrast=template['source_contrast'][[p[0] for p in candidates]],
            contrast=template['contrast'][[p[0] for p in candidates]])
        old_track = dict(template=old_template, search_uv=old_points, normals=old_normal)
        old_membership = (dict(known_source_board_material, current_actual_pts=previous_actual,
            current_native_pts=int(before.get('native_pts', before.get('source_pts'))),
            decoded_native_BGR_sha256=before.get('decoded_native_BGR_sha256', before.get('decoded_BGR_sha256')),
            source_entity_membership_navigation='same_shot_actual_RGB_LK_pending_two_profile_confirmation')
            if known_source_board_material is not None else None)
        old_reads = {p[0]: p for p in reader._read_edge(previous['image'].astype(np.float32),
            previous['hard'], [] if independently_known_board else self._board(previous),
            old_track, known_source_board_material=old_membership)}
        accepted, proof = [], {}
        for j, point in enumerate(candidates):
            old = old_reads.get(j)
            if (not (flow_good[j] or use_context[j]) or old is None
                    or np.linalg.norm(old[1]-old_points[j]) > 3.
                    or not _clear_context(previous, old[1], old[2], extent)):
                continue
            descriptor, strength = _descriptor(current['image'], point[1], point[2], extent)
            old_descriptor, old_strength = _descriptor(previous['image'], old[1], old[2], extent)
            correlation = float(descriptor@old_descriptor)
            if min(strength, old_strength) < 20. or correlation < .90:
                continue
            index = int(point[0])
            from .vision_programme_board_review import _sample
            current_sides = point[1]+np.array([-5., 5.])[:, None]*point[2]
            previous_sides = old[1]+np.array([-5., 5.])[:, None]*old[2]
            accepted.append(point)
            proof[index] = dict(kind='known_static_board_boundary_visible_inside_primary_detector_box',
                physical_material_id=anchor['physical_material_id'],
                source_identity_authority=anchor['source_annotation_reference'],
                source_identity_confirmed=True, actual_pts=actual,
                same_actual_exposure_native_alias_pair=bool(native_alias),
                native_pts=int(now.get('native_pts', now.get('source_pts'))), shot_id=now.get('shot_id'),
                decoded_native_BGR_sha256=now.get('decoded_native_BGR_sha256', now.get('decoded_BGR_sha256')),
                previous_actual_pts=previous_actual,
                previous_native_pts=int(before.get('native_pts', before.get('source_pts'))),
                previous_decoded_native_BGR_sha256=before.get('decoded_native_BGR_sha256', before.get('decoded_BGR_sha256')),
                current_actual_RGB_interface_uv=point[1].tolist(),
                previous_actual_RGB_interface_uv=old[1].tolist(),
                current_actual_RGB_material_sides_uv=current_sides.tolist(),
                previous_actual_RGB_material_sides_uv=previous_sides.tolist(),
                current_actual_BGR_material_sides=_sample(current['image'], current_sides).reshape(2, 3).tolist(),
                previous_actual_BGR_material_sides=_sample(previous['image'], previous_sides).reshape(2, 3).tolist(),
                source_forward_backward_LK_error_px=error[j].tolist() if flow_good[j] else None,
                source_boundary_point_direct_LK_available=bool(flow_good[j]),
                source_named_board_context_tracking=context_evidence if use_context[j] else None,
                actual_material_RGB_context_correlation=correlation,
                current_actual_transition_interval_px=point[4].tolist(),
                previous_actual_transition_interval_px=old[4].tolist(),
                source_board_material_membership_evidence=known_source_board_material,
                current_and_previous_complete_source_board_material_profile_measured=True,
                rail_uv_interpolation_or_geometry_used_as_source_scope=False,
                measured_person_dilated_contour_and_other_person_boxes_excluded=True,
                only_primary_box_context_exclusion_removed=True,
                primary_box_index=current['primary_box_index'],
                primary_box_SAM2_coverage=current['primary_box_SAM2_coverage'],
                historical_person_union_modified=False,
                model_projection_or_fit_residual_used_for_identity=False)
        return accepted, proof

    def finish(self):
        if self.current is not None:
            self.current['image'] = self.current['image'].copy()
        self.previous, self.current = self.current, None
