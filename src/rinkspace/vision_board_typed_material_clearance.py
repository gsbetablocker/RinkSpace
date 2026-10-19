"""Two-dimensional discrete-print clearance without changing frozen readers.

The generic contour clearance has a one-dimensional normal and can transport
only already measured points. A discrete corner instead needs its original
three-role navigation and two actual printed borders in both source images.
"""
from __future__ import annotations

import numpy as np

from .vision_board_local_clearance import KnownBoardMaterialBoxClearance
from .vision_board_material_point_profiles import (
    BoardMaterialPointSourceReader, MATERIAL_ROLES, read_printed_material_corners,
)
from .vision_board_material_point_native_alias import _navigation, _context


class TypedMaterialBoardBoxClearance(KnownBoardMaterialBoxClearance):
    def additional_points(self, reader, track, *, physical_identity_confirmed,
                          known_source_board_material=None):
        anchor = track['template']['anchor']
        if anchor['curve'].get('kind') != 'shared_board_pattern_material_points':
            return super().additional_points(reader, track,
                physical_identity_confirmed=physical_identity_confirmed,
                known_source_board_material=known_source_board_material)
        current, previous = self.current, self.previous
        if not physical_identity_confirmed or current is None or previous is None:
            return [], {}
        if (tuple(anchor.get('material_roles', ())) != MATERIAL_ROLES
                or anchor.get('point_correspondence') is not True):
            return [], {}
        now, before = current['binding'], previous['binding']
        actual, previous_actual = int(now['actual_pts']), int(before['actual_pts'])
        if (now.get('shot_id') != before.get('shot_id')
                or self.actual_successor.get(previous_actual) != actual):
            return [], {}
        # The current original source was already associated with this named
        # static printed board. No ambiguous colour component gains identity.
        if (not known_source_board_material
                or known_source_board_material.get('physical_material_id') != anchor['physical_material_id']
                or known_source_board_material.get('source_identity_confirmed') is not True
                or known_source_board_material.get('actual_original_source_printed_board_membership_confirmed') is not True):
            return [], {}
        guides = np.asarray(track['search_uv'], float)
        if guides.shape != (3, 2):
            return [], {}
        current_bands = self._board(current)
        candidates = reader._read_edge(current['image'].astype(np.float32), current['hard'],
            current_bands, track, known_source_board_material=known_source_board_material)
        candidates = [point for point in candidates
            if current['primary'][int(round(point[1][1])), int(round(point[1][0]))]]
        if not candidates:
            return [], {}
        navigation = _navigation(current['image'], previous['image'],
            current['hard'], previous['hard'], guides)
        if navigation is None:
            return [], {}
        previous_guides, registration = navigation
        # Registration transports only NAV, including the unmeasured tip.
        # The previous source's actual two borders create its own vertices.
        previous_reads = read_printed_material_corners(
            previous['image'], previous['hard'], previous_guides)
        registration = dict(registration,
            kind='same_shot_named_print_actual_RGB_navigation_for_two_dimensional_material_corner',
            current_actual_pts=actual, previous_actual_pts=previous_actual)
        accepted, proofs = [], {}
        for point in candidates:
            index = int(point[0])
            if index not in previous_reads:
                continue
            old = previous_reads[index]
            first = _context(current['image'], current['hard'], point[1])
            second = _context(previous['image'], previous['hard'], old['uv'])
            if first is None or second is None:
                continue
            correlation = float(first[0] @ second[0])
            if correlation < .90:
                continue
            current_digest = now.get('decoded_native_BGR_sha256', now.get('decoded_BGR_sha256'))
            readout = reader._material_point_readouts[(current_digest, anchor['physical_material_id'])][index]
            accepted.append(point)
            proofs[index] = dict(
                kind='known_static_discrete_printed_board_corner_visible_inside_primary_detector_box',
                physical_material_id=anchor['physical_material_id'], material_role=MATERIAL_ROLES[index],
                source_identity_authority=anchor['source_annotation_reference'],
                source_identity_confirmed=True, actual_pts=actual,
                native_pts=int(now.get('native_pts', now.get('source_pts'))),
                shot_id=now['shot_id'], decoded_native_BGR_sha256=current_digest,
                previous_actual_pts=previous_actual,
                previous_native_pts=int(before.get('native_pts', before.get('source_pts'))),
                previous_decoded_native_BGR_sha256=before.get('decoded_native_BGR_sha256', before.get('decoded_BGR_sha256')),
                current_actual_RGB_material_point_uv=point[1],
                previous_actual_RGB_material_point_uv=old['uv'],
                current_actual_two_border_material_point_readout=readout,
                previous_actual_two_border_material_point_readout=old,
                current_and_previous_complete_source_board_material_profile_measured=True,
                actual_material_RGB_context_correlation=correlation,
                source_named_board_context_tracking=registration,
                full_original_three_role_guides_retained_as_navigation_only=True,
                unmeasured_bottom_tip_not_used_as_pixel_truth=True,
                source_board_material_membership_evidence=known_source_board_material,
                measured_person_dilated_contour_and_other_person_boxes_excluded=True,
                only_primary_box_context_exclusion_removed=True,
                primary_box_index=current['primary_box_index'],
                primary_box_SAM2_coverage=current['primary_box_SAM2_coverage'],
                normal_is_not_defined_for_discrete_material_point=True,
                rail_uv_interpolation_or_geometry_used_as_source_scope=False,
                historical_person_union_modified=False,
                model_projection_or_fit_residual_used_for_identity=False)
        return accepted, proofs


class TypedMaterialBoardSourceReader(BoardMaterialPointSourceReader):
    """Frozen-implementation successor with a separate typed clearance path."""
    def _read_edge(self, image, exclusion, bands, track, *, known_source_board_material=None):
        if (track['template']['anchor']['curve'].get('kind') == 'shared_board_pattern_material_points'
                and np.asarray(track['search_uv']).shape != (3, 2)):
            # A generic normal-based caller cannot supply this role topology.
            # In particular a measured single vertex cannot invent two guides.
            return []
        return super()._read_edge(image, exclusion, bands, track,
            known_source_board_material=known_source_board_material)

    def read(self, image, *args, **kwargs):
        if kwargs.get('person_components') is not None and not isinstance(
                getattr(self, 'box_clearance', None), TypedMaterialBoardBoxClearance):
            self.box_clearance = TypedMaterialBoardBoxClearance(self.actual_successor)
        return super().read(image, *args, **kwargs)
