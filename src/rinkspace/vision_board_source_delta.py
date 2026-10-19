"""Consume an immutable, actually reread board source delta on the same scene."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from calibration.program_tracking.strict_absolute_source import (
    normalize_selected_board_rows, board_print_target_inventory,
    board_material_point_target_inventory,
)
from calibration.program_tracking.strict_model_scope import authenticate_source_reference
from .vision_board_material_point_profiles import MATERIAL_ROLES


def normalize_exact_board_source_delta(path, *, expected_sha256, scene):
    """Preserve current UV/provenance; add only explicit source-array indices.

    No source image is decoded, pixel moved, identity inferred, or weight
    invented. The public consumer checks exact UV, target and source evidence.
    """
    path, _ = authenticate_source_reference(dict(path=str(path), sha256=expected_sha256))
    content = json.loads(Path(path).read_text(encoding='utf8'))
    if content.get('schema') != 'rinkspace.known-actual-board-material-new-source-delta.v1':
        raise ValueError('An immutable actual board material delta is required')
    authority_path, _ = authenticate_source_reference(content['source_identity_authority'])
    authority = json.loads(Path(authority_path).read_text(encoding='utf8'))
    scene_path, _ = authenticate_source_reference(authority['scene_reference'])
    if json.loads(Path(scene_path).read_text(encoding='utf8')) != scene:
        raise ValueError('Board delta must use the same frozen scene as its navigation')
    result = []
    for measurement in content['measurements']:
        measurement = deepcopy(measurement)
        rows = measurement['advertisement_rows']
        for row in rows:
            row['uv'] = np.asarray(row['uv'], float).reshape(-1, 2)
            row['normal'] = np.asarray(row.get('normal', ()), float).reshape(-1, 2)
            uncertainty = row['source_readout_uncertainty']
            if row['curve']['kind'] == 'shared_board_pattern_material_points':
                role = row['material_roles'][0]
                row.setdefault('anchor_point_indices', np.asarray([MATERIAL_ROLES.index(role)], np.int32))
            elif row.get('normal_measurement_origin') == 'same_current_source_visible_contour_adjacent_measured_UV':
                # The public consumer independently recomputes this normal
                # from these exact adjacent actual UV before giving support.
                row['normal_from_actual_source_neighbour_pixels'] = True
                uncertainty['source_native_pixel_quantization_step_px'] = 1.
                uncertainty['actual_readout_blur_interval_px'] = np.asarray(
                    uncertainty['actual_readout_blur_interval_px'], float)
                uncertainty['normal_interval_px'] = np.asarray(uncertainty['normal_interval_px'], float)
        normalized = normalize_selected_board_rows(rows, measurement,
            source_binding=measurement['source'],
            board_print_targets=board_print_target_inventory(scene),
            board_material_point_targets=board_material_point_target_inventory(scene),
            reviewer='same_model_navigation_actual_source_board_delta')
        result.append(dict(source=measurement['source'], **normalized))
    return result
