"""Current RGB readout of existing discrete printed-outline corners.

Source guides navigate the two visible borders. Their actual transition
samples locate the corner and its two-dimensional readout bounds. A rounded
lower outline cannot supply an exact fixed tip merely from its silhouette.
"""
from __future__ import annotations

import hashlib
import cv2
import numpy as np

from calibration.program_tracking.source_board_measurements import board_bracket_candidates
from .vision_programme_board_review import _sample
from .vision_board_print_outer_profiles import PrintedContourBoardSourceReader


MATERIAL_ROLES = ('upper_left_printed_outline_corner',
                  'upper_right_printed_outline_corner',
                  'bottom_printed_outline_tip')


def _transition(image, exclusion, guess, outward):
    offsets = np.arange(-8., 8.001, .5)
    tangent = np.asarray([-outward[1], outward[0]])
    xy = guess + offsets[:, None, None]*outward + np.asarray([-1., 0., 1.])[None, :, None]*tangent
    pixels = np.rint(xy).astype(int)
    if ((pixels < 0).any() or (pixels >= [1920, 1080]).any()
            or exclusion[pixels[..., 1], pixels[..., 0]].any()):
        return None
    colors = np.median(_sample(image, xy), axis=1)
    values = colors.mean(1)
    delta = values[4:] - values[:-4]
    peaks = np.flatnonzero((delta[1:-1] >= delta[:-2]) & (delta[1:-1] >= delta[2:])
                           & (delta[1:-1] >= 8.)) + 1
    peaks = peaks[np.abs(offsets[peaks+2]) <= 4.]
    if not len(peaks):
        return None
    # The named upper printed border lies below a separate printed label.
    # Read its nearby interface; a farther, stronger label is another feature.
    peak = int(min(peaks, key=lambda k: (abs(offsets[k+2]), -delta[k])))
    centre = peak+2
    start = max(0, centre-6)
    valley = start+int(np.argmin(values[start:centre+1]))
    high_index = centre+int(np.argmax(values[centre:min(len(values), centre+7)]))
    low, high = float(values[valley]), float(values[high_index])
    if (high-low < 24. or colors[valley, 0]-colors[valley, 1:].mean() < 8.):
        return None
    quantiles = []
    for fraction in (.25, .5, .75):
        level = low+fraction*(high-low)
        crossings = np.flatnonzero((values[:-1] <= level) & (values[1:] > level))
        crossings = crossings[(crossings >= valley) & (crossings < high_index)]
        if len(crossings) != 1:
            return None
        k = int(crossings[0])
        quantiles.append(offsets[k]+.5*(level-values[k])/(values[k+1]-values[k]))
    if abs(quantiles[1]) > 4.:
        return None
    return dict(uv=guess+quantiles[1]*outward, outward=outward,
                blur=np.asarray([quantiles[0]-quantiles[1], quantiles[2]-quantiles[1]]),
                actual_transition_gradient=float(delta[peak]),
                actual_inside_BGR=colors[valley], actual_outside_BGR=colors[high_index])


def _line(readings):
    if len(readings) < 4:
        return None
    points = np.asarray([r['uv'] for r in readings], np.float32)
    # Eight-pixel guide spacing can differ by a fraction after the actual
    # transition reread; retain the profile's half-pixel sampling resolution.
    if np.ptp(points, axis=0).max() < 7.5:
        return None
    fitted = cv2.fitLine(points, cv2.DIST_HUBER, 0, .01, .01).ravel()
    direction, origin = fitted[:2].astype(float), fitted[2:].astype(float)
    normal = np.asarray([-direction[1], direction[0]])
    residual = (points-origin) @ normal
    if np.max(np.abs(residual)) > 2.:
        return None
    # These are actual blur, source quantization and measured line residuals,
    # rather than the eight-pixel navigation search extent.
    projected = np.asarray([r['blur']*(r['outward'] @ normal) for r in readings])
    blur = np.sort(projected, axis=1)
    lo = float(blur[:, 0].min()-np.max(np.abs(residual))-.5)
    hi = float(blur[:, 1].max()+np.max(np.abs(residual))+.5)
    return dict(origin=origin, normal=normal, offset_bounds=[lo, hi], readings=readings)


def read_printed_material_corners(image, exclusion, source_guides):
    """Return actual upper corners; preserve an unobserved fixed tip as unknown."""
    guides = np.asarray(source_guides, float)
    if guides.shape != (3, 2) or not np.isfinite(guides).all():
        raise ValueError('The three existing named material guides are required in their original role order')
    left, right, bottom = guides
    horizontal = right-left
    horizontal /= max(np.linalg.norm(horizontal), 1e-9)
    vertical = bottom-(left+right)*.5
    vertical /= max(np.linalg.norm(vertical), 1e-9)
    upper_outward = np.asarray([horizontal[1], -horizontal[0]])
    if upper_outward @ vertical > 0:
        upper_outward *= -1
    top = [_transition(image, exclusion, left+(right-left)*fraction, upper_outward)
           for fraction in np.linspace(.15, .85, 15)]
    top = _line([r for r in top if r is not None])
    if top is None:
        return {}
    found = {}
    for index, (corner, outward) in enumerate(((left, -horizontal), (right, horizontal))):
        readings = [_transition(image, exclusion, corner+distance*vertical, outward)
                    for distance in np.arange(4., 18., 2.)]
        side = _line([r for r in readings if r is not None])
        if side is None:
            continue
        matrix = np.asarray([top['normal'], side['normal']])
        if abs(np.linalg.det(matrix)) < .6:
            continue
        constant = np.asarray([top['normal'] @ top['origin'], side['normal'] @ side['origin']])
        uv = np.linalg.solve(matrix, constant)
        if np.linalg.norm(uv-corner) > 8.:
            continue
        vertices = [np.linalg.solve(matrix, constant+[a, b])
                    for a in top['offset_bounds'] for b in side['offset_bounds']]
        bounds = np.asarray([np.min(vertices, axis=0), np.max(vertices, axis=0)])
        low, high = np.floor(bounds[0]).astype(int), np.ceil(bounds[1]).astype(int)+1
        if ((low < 0).any() or (high > [1920, 1080]).any()
                or exclusion[low[1]:high[1], low[0]:high[0]].any()):
            continue
        found[index] = dict(uv=uv, actual_2d_readout_bounds_xy_px=bounds,
            source_sigma_xy_px=None, measurement_method='actual_original_RGB_two_visible_printed_border_intersection',
            actual_source_RGB_measured=True, source_guides_used_as_output=False,
            separate_upper_label_used_as_corner=False,
            visible_border_readings=[top['readings'], side['readings']],
            current_source_pixel_context_person_exclusion_clear=True,
            fitted_scene_or_camera_used_for_readout=False)
    return found


class BoardMaterialPointSourceReader(PrintedContourBoardSourceReader):
    """Delta reader for existing three-role material points and existing curves."""
    def _read_edge(self, image, exclusion, bands, track, *, known_source_board_material=None):
        anchor = track['template']['anchor']
        if anchor['curve'].get('kind') != 'shared_board_pattern_material_points':
            return super()._read_edge(image, exclusion, bands, track,
                                     known_source_board_material=known_source_board_material)
        if (tuple(anchor.get('material_roles', ())) != MATERIAL_ROLES
                or anchor.get('point_correspondence') is not True):
            raise ValueError('Discrete printed features must retain their exact existing material correspondence')
        found = read_printed_material_corners(image, exclusion, track['search_uv'])
        independent = bool(known_source_board_material
            and known_source_board_material.get('physical_material_id') == anchor['physical_material_id']
            and known_source_board_material.get('source_identity_confirmed') is True
            and known_source_board_material.get('actual_original_source_printed_board_membership_confirmed') is True)
        found = {i: e for i, e in found.items()
                 if independent or board_bracket_candidates(e['uv'][None], bands, exclusion)[0] >= 0}
        digest = hashlib.sha256(np.asarray(image, np.uint8).tobytes()).hexdigest()
        if not hasattr(self, '_material_point_readouts'):
            self._material_point_readouts = {}
        self._material_point_readouts[(digest, anchor['physical_material_id'])] = found
        return [(i, evidence['uv'], np.zeros(2),
                 min(r['actual_transition_gradient'] for edge in evidence['visible_border_readings'] for r in edge),
                 np.asarray([np.nan, np.nan])) for i, evidence in found.items()]

    def read(self, image, *args, **kwargs):
        self._material_point_readouts = {}
        result = super().read(image, *args, **kwargs)
        digest = hashlib.sha256(np.asarray(image, np.uint8).tobytes()).hexdigest()
        for row in result['advertisement_rows']:
            if row['curve'].get('kind') != 'shared_board_pattern_material_points':
                continue
            if not len(row['uv']):
                row.update(source_unmeasured_discrete_material_roles=list(MATERIAL_ROLES),
                           discrete_material_point_measurement_pending=True)
                continue
            if len(row['uv']) != 1:
                raise ValueError('Separated discrete material features cannot become a source contour')
            index = int(row['anchor_point_indices'][0])
            evidence = self._material_point_readouts[(digest, row['physical_material_id'])][index]
            track = self.tracks.get(row['physical_material_id'])
            if track is not None:
                row.update(source_navigation_material_guides=np.asarray(track['search_uv'], float).copy(),
                    source_navigation_material_guide_roles=list(MATERIAL_ROLES),
                    source_navigation_material_guides_are_not_pixel_truth=True,
                    source_anchor_reference=dict(track['template']['anchor']['source_annotation_reference']))
            physical = f"Ad{row['advertisement_number']}.material_point.{MATERIAL_ROLES[index]}"
            row['physical_material_id'] = physical
            row['temporal_evidence'] = dict(row['temporal_evidence'], physical_material_id=physical)
            if row.get('source_board_material_membership_evidence'):
                row['source_board_material_membership_evidence'] = dict(
                    row['source_board_material_membership_evidence'], physical_material_id=physical)
            row.update(material_roles=[MATERIAL_ROLES[index]], point_correspondence=True,
                observation_semantics='board_print_feature', source_material_point_readout=evidence,
                normal=np.empty((0, 2)), normal_measurement_origin='not_applicable_discrete_material_point',
                source_neighbour_adjacency_confirmed=False,
                actual_2d_readout_bounds_xy_px=evidence['actual_2d_readout_bounds_xy_px'][None],
                source_sigma_xy_px=None)
            row['source_readout_uncertainty'].update(
                measurement_method=evidence['measurement_method'], normal_interval_px=None,
                actual_readout_blur_interval_px=None,
                actual_2d_readout_bounds_xy_px=evidence['actual_2d_readout_bounds_xy_px'][None])
        return result
