"""Review-only metre grids on unchanged native geometry and camera coordinates.

One post-geometry closest60x30 conversion supplies both ice and board spacing.
The previous working-unit ice grid is retained explicitly for comparisons.
No fitted material, source observation or camera coordinate is converted here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np

from rinkspace.metric_scale import SQUARED_DIMENSION_ERROR, boundary_span, estimate_uniform_scale
from .render_result import clip_polygon_halfspace


FIELD = 'review_metric_grid_conversion'
LEGACY_FIELD = 'legacy_working_ice_grid'
SCHEMA = 'rinkspace.review-uniform-metric-grids.v1'


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf8')).hexdigest()


def scene_geometry_sha(scene):
    """Hash every scene field except the two display grids and their receipt."""
    return json_sha({k: v for k, v in scene.items()
                     if k not in ('ice_grid', FIELD, LEGACY_FIELD)})


def _ice_grid(scene, scale):
    outline = np.asarray(scene['ice_boundary']['points_xyz'], np.float64)
    low, high = outline[:, :2].min(axis=0), outline[:, :2].max(axis=0)
    spacing = 1. / scale
    columns, rows = map(int, np.ceil((high-low) / spacing))
    cells = []
    for col in range(columns):
        left, right = low[0]+col*spacing, min(low[0]+(col+1)*spacing, high[0])
        for row in range(rows):
            bottom, top = low[1]+row*spacing, min(low[1]+(row+1)*spacing, high[1])
            polygon = outline
            for axis, limit, greater in ((0, left, True), (0, right, False),
                                         (1, bottom, True), (1, top, False)):
                polygon = clip_polygon_halfspace(polygon, axis, limit, greater)
                if len(polygon) < 3:
                    break
            if len(polygon) < 3:
                continue
            area = abs(float(np.sum(polygon[:, 0]*np.roll(polygon[:, 1], -1)
                                    - polygon[:, 1]*np.roll(polygon[:, 0], -1)))) / 2.
            if area <= 64*np.finfo(float).eps*spacing*spacing:
                continue
            cells.append(dict(id=f'grid_{col:03d}_{row:03d}', column=col, row=row,
                              polygon_xyz=polygon.tolist(), area_scene_units_squared=area,
                              clipped_at_rink_boundary=bool(area < (right-left)*(top-bottom)-1e-10)))
    return dict(type='review_one_metre_orthogonal_cells_clipped_to_actual_boundary',
                origin_xyz=[float(low[0]), float(low[1]), 0.],
                column_axis_xyz=[1., 0., 0.], row_axis_xyz=[0., 1., 0.],
                spacing_xy=[spacing, spacing], spacing_xy_m=[1., 1.],
                columns=columns, rows=rows, cells=cells,
                label='1米正交审阅格；几何后统一换算接近60×30米',
                unit_status='metric_assumed; one uniform post-geometry conversion; not surveyed metres')


def prepare_metric_review_scene(scene):
    """Return a display copy, preserving every non-grid scene field literally."""
    if FIELD in scene:
        validate_metric_review_scene(scene)
        return scene
    before = scene_geometry_sha(scene)
    scale = estimate_uniform_scale(boundary_span(scene))
    result = dict(scene)
    legacy = deepcopy(scene['ice_grid'])
    legacy.update(label='历史工作单位网格，仅供对照；不是最终1米格',
                  display_policy='legacy_working_grid_comparison_only_not_final_metres')
    result[LEGACY_FIELD] = legacy
    result['ice_grid'] = _ice_grid(scene, scale)
    result[FIELD] = dict(schema=SCHEMA, policy=SQUARED_DIMENSION_ERROR,
                         formula='s=(60*L+30*W)/(L*L+W*W)',
                         metres_per_scene_unit=scale,
                         source_span_xy_scene_units=boundary_span(scene).tolist(),
                         converted_span_xy_m=(boundary_span(scene)*scale).tolist(),
                         ice_spacing_m=1., board_station_spacing_m=1.,
                         one_uniform_scale_for_both_axes_ice_and_board=True,
                         scene_geometry_sha256=before,
                         metric_ice_grid_sha256=json_sha(result['ice_grid']),
                         legacy_source_ice_grid_sha256=json_sha(scene['ice_grid']),
                         legacy_working_ice_grid_sha256=json_sha(legacy),
                         legacy_grid_policy=legacy['display_policy'],
                         actual_material_camera_source_and_solver_unchanged=True,
                         verified_metre_scale=False)
    validate_metric_review_scene(result)
    return result


def validate_metric_review_scene(scene):
    """Authenticate the conversion against this exact unchanged scene."""
    receipt = scene[FIELD]
    scale = estimate_uniform_scale(boundary_span(scene))
    if (receipt.get('schema') != SCHEMA or receipt.get('policy') != SQUARED_DIMENSION_ERROR
            or receipt.get('scene_geometry_sha256') != scene_geometry_sha(scene)
            or receipt.get('metric_ice_grid_sha256') != json_sha(scene['ice_grid'])
            or receipt.get('legacy_working_ice_grid_sha256') != json_sha(scene[LEGACY_FIELD])
            or not math.isclose(receipt['metres_per_scene_unit'], scale, rel_tol=1e-14)
            or not np.allclose(np.asarray(scene['ice_grid']['spacing_xy'])*scale,
                               [1., 1.], rtol=1e-14, atol=0.)
            or receipt.get('one_uniform_scale_for_both_axes_ice_and_board') is not True
            or receipt.get('verified_metre_scale') is not False):
        raise ValueError('Review metric-grid conversion differs from its bound scene')
    return receipt
