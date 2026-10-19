"""Read an authenticated printed contour against its actual white board side.

An ornamental inner band can create several transitions before the exterior
interface. A measured, neutral, stable exterior plateau distinguishes those
bands without using scene geometry. Unclear exterior material stays pending.
"""
from __future__ import annotations

import numpy as np

from calibration.program_tracking.source_board_measurements import board_bracket_candidates
from .vision_programme_board_review import _sample
from .vision_board_narrow_graphic_profiles import NarrowGraphicBoardSourceReader


def read_printed_outer_source_interface(image, exclusion, bands, track, *,
                                      known_source_board_material=None):
    template, found = track['template'], []
    anchor = template['anchor']
    if anchor.get('model_target', {}).get('kind') != 'current_scene_board_print_contour':
        raise ValueError('A named existing printed-component contour is required')
    if not anchor.get('source_component_correspondence'):
        return []
    independent = bool(known_source_board_material
        and known_source_board_material.get('physical_material_id') == anchor['physical_material_id']
        and known_source_board_material.get('source_identity_confirmed') is True
        and known_source_board_material.get('actual_original_source_printed_board_membership_confirmed') is True
        and known_source_board_material.get('model_projection_or_fit_residual_used_for_identity') is False)
    offsets = np.arange(-6., 6.001, .5)
    for index, (guess, normal) in enumerate(zip(track['search_uv'], track['normals'])):
        if template['source_contrast'][index] < 12.:
            continue
        tangent = np.asarray([-normal[1], normal[0]])
        xy = guess+offsets[:, None, None]*normal+np.asarray([-2., 0., 2.])[None, :, None]*tangent
        pixels = np.rint(xy).astype(int)
        if ((pixels < 0).any() or (pixels >= [1920, 1080]).any()
                or exclusion[pixels[..., 1], pixels[..., 0]].any()):
            continue
        colors = np.median(_sample(image, xy), axis=1)
        outside = colors[-5:]
        if (outside.std(0).max() > 6. or np.ptp(outside.mean(0)) > 12.
                or outside.mean(0).min() < 80.):
            continue
        values = colors @ template['contrast'][index]
        delta = values[4:]-values[:-4]
        local = np.flatnonzero((delta[1:-1] >= delta[:-2])&(delta[1:-1] >= delta[2:])
            & (delta[1:-1] >= 8.))+1
        local = local[np.abs(offsets[local+2]) <= 3.]
        if not len(local):
            continue
        # The source-confirmed exterior side is uniform actual board material.
        # Earlier ornamental transitions therefore are not this exterior edge.
        peak = int(local[-1]);centre = peak+2
        valley_start = max(0, centre-6)
        valley = valley_start+int(np.argmin(values[valley_start:centre+1]))
        low, high = float(values[valley]), float(np.median(values[-5:]))
        if high-low < 24.:
            continue
        readings = []
        for fraction in (.25, .5, .75):
            level = low+fraction*(high-low)
            crossings = np.flatnonzero((values[:-1] <= level)&(values[1:] > level))
            crossings = crossings[crossings >= valley]
            if len(crossings) != 1:
                break
            k = int(crossings[0])
            readings.append(offsets[k]+.5*(level-values[k])/(values[k+1]-values[k]))
        if (len(readings) != 3 or abs(readings[1]) > 3.
                or abs(readings[1]-offsets[centre]) > 3.):
            continue
        uv = guess+readings[1]*normal
        if board_bracket_candidates(uv[None], bands, exclusion)[0] < 0 and not independent:
            continue
        found.append((index, uv, normal, float(delta[peak]),
                      np.asarray([readings[0]-readings[1], readings[2]-readings[1]])))
    return found


class PrintedContourBoardSourceReader(NarrowGraphicBoardSourceReader):
    """Separate source-delta reader; the running board implementation is intact."""
    def _read_edge(self, image, exclusion, bands, track, *, known_source_board_material=None):
        if track['template']['anchor'].get('model_target', {}).get('kind') == 'current_scene_board_print_contour':
            return read_printed_outer_source_interface(image, exclusion, bands, track,
                known_source_board_material=known_source_board_material)
        return super()._read_edge(image, exclusion, bands, track,
            known_source_board_material=known_source_board_material)

    def read(self, *args, **kwargs):
        result = super().read(*args, **kwargs)
        for row in result['advertisement_rows']:
            if row.get('model_target', {}).get('kind') == 'current_scene_board_print_contour':
                row['source_readout_uncertainty']['measurement_method'] = 'actual_outer_print_interface_q25_q50_q75_to_measured_stable_white_board_side'
                row['actual_original_RGB_stable_neutral_exterior_side_measured'] = bool(len(row['uv']))
                row['inner_ornamental_band_used_as_outer_boundary'] = False
        return result
