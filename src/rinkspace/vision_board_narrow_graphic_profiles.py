"""Actual single-interface readout for named narrow internal board graphics.

The registered source entity extent selects a read window only. Every UV and
blur interval comes from the current original RGB; no model projection, drawn
stroke, opposite border or interpolated rail is emitted as a pixel observation.
This reader is a separate delta entry point; it does not change a running reader.
"""
from __future__ import annotations

import numpy as np

from calibration.program_tracking.source_board_measurements import board_bracket_candidates
from .vision_programme_board_review import BoardProgrammeSourceReader, _sample


def read_narrow_internal_graphic_interface(image, exclusion, bands, track, *,
                                           known_source_board_material=None):
    template = track['template']
    anchor = template['anchor']
    if anchor['curve'].get('extent_scope') != 'internal_graphic_only':
        raise ValueError('The narrow profile cannot change an exterior boundary role')
    entity = track.get('source_entity_context_uv', anchor.get('source_entity_context_uv'))
    if entity is None:
        return []
    entity = np.asarray(entity, float)
    low, high = entity.min(0), entity.max(0)
    vertices = np.asarray([low, [high[0], low[1]], high, [low[0], high[1]]])
    independent = bool(known_source_board_material
        and known_source_board_material.get('physical_material_id') == anchor['physical_material_id']
        and known_source_board_material.get('source_identity_confirmed') is True
        and known_source_board_material.get('actual_original_source_printed_board_membership_confirmed') is True
        and known_source_board_material.get('model_projection_or_fit_residual_used_for_identity') is False)
    observed = []
    for index, (guess, normal) in enumerate(zip(track['search_uv'], track['normals'])):
        if template['source_contrast'][index] < 12.:
            continue
        span = float(np.ptp(vertices @ normal))
        # A source guide can bound a crop; it cannot supply its two border UVs.
        # Stay within one interface instead of crossing the narrow opposite side.
        extent = min(8., max(3., .4*span))
        extent = np.floor(extent*2)/2
        offsets = np.arange(-extent, extent+.001, .5)
        tangent = np.asarray([-normal[1], normal[0]])
        xy = guess+offsets[:, None, None]*normal+np.asarray([-2., 0., 2.])[None, :, None]*tangent
        pixels = np.rint(xy).astype(int)
        if ((pixels < 0).any() or (pixels >= [1920, 1080]).any()
                or exclusion[pixels[..., 1], pixels[..., 0]].any()):
            continue
        values = np.median(_sample(image, xy), axis=1) @ template['contrast'][index]
        delta = values[4:]-values[:-4]
        peak = int(np.argmax(delta))
        strength = float(delta[peak])
        if strength < 8. or peak in (0, len(delta)-1):
            continue
        competing = np.flatnonzero(delta > .8*strength)
        if len(competing) and np.ptp(competing)*.5 > 4.:
            continue
        before, after = float(np.median(values[:3])), float(np.median(values[-3:]))
        if after-before < 24.:
            continue
        positions = []
        for fraction in (.25, .5, .75):
            level = before+fraction*(after-before)
            crossing = np.flatnonzero((values[:-1] <= level)&(values[1:] > level))
            if len(crossing) != 1:
                break
            k = int(crossing[0])
            positions.append(offsets[k]+.5*(level-values[k])/(values[k+1]-values[k]))
        if len(positions) != 3 or abs(positions[1]-offsets[peak+2]) > 3.:
            continue
        uv = guess+positions[1]*normal
        if board_bracket_candidates(uv[None], bands, exclusion)[0] < 0 and not independent:
            continue
        observed.append((index, uv, normal, strength,
                         np.asarray([positions[0]-positions[1], positions[2]-positions[1]])))
    return observed


class NarrowGraphicBoardSourceReader(BoardProgrammeSourceReader):
    """Existing board reader with a source-bounded internal-graphic profile."""
    def _read_edge(self, image, exclusion, bands, track, *, known_source_board_material=None):
        if track['template']['anchor']['curve'].get('extent_scope') == 'internal_graphic_only':
            return read_narrow_internal_graphic_interface(image, exclusion, bands, track,
                known_source_board_material=known_source_board_material)
        return super()._read_edge(image, exclusion, bands, track,
            known_source_board_material=known_source_board_material)

    def read(self, *args, **kwargs):
        result = super().read(*args, **kwargs)
        for row in result['advertisement_rows']:
            if row['curve'].get('extent_scope') == 'internal_graphic_only':
                row['source_readout_uncertainty']['measurement_method'] = 'actual_internal_graphic_single_interface_q25_q50_q75'
                row['source_profile_extent_is_navigation_only'] = True
                row['opposite_graphic_border_used_as_output'] = False
        return result
