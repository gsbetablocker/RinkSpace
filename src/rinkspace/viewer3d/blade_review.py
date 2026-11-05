"""Saved shoe/blade projections and source evidence on the exact native PTS."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .source import native_pts, region_outline


class BladeReview:
    """Read display-only assets; never interpolate observations or fit geometry."""
    def __init__(self, path, *, assets=None, time_base=None):
        path = Path(path)
        self.document = json.loads(path.read_text(encoding='utf8'))
        if self.document.get('schema') != 'rinkspace.blade-player-review.v1':
            raise ValueError('Unsupported blade player review schema')
        if time_base is not None and 'time_base' in self.document and tuple(self.document['time_base']) != tuple(time_base):
            raise ValueError('Blade review time base differs from player source')
        projection = self.document['projection_npz']
        projection = (assets.resolve(projection, base=path.parent, must_exist=True) if assets is not None
                      else path.parent / projection)
        self.variants = self.document['variants']
        if not isinstance(self.variants, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not v
                                                      for k, v in self.variants.items()):
            raise ValueError('Invalid blade review variant mapping')
        with np.load(projection, allow_pickle=False) as saved:
            pts = saved['pts']
            if pts.ndim != 1 or pts.dtype.kind not in 'iu' or not len(pts) or np.any(np.diff(pts.astype(np.int64)) <= 0):
                raise ValueError('Blade review requires ordered unique native PTS')
            self.index = {native_pts(value): index for index, value in enumerate(pts)}
            self.models = {}
            for variant, prefix in self.variants.items():
                model = {}
                for suffix, shape in [('blade_uv', (len(pts), 2, 3, 2)),
                                      ('boot_uv', (len(pts), 2, 2, 2)),
                                      ('axis_rate_rad_s', (len(pts), 2))]:
                    values = np.asarray(saved[prefix + '_' + suffix], dtype=float)
                    if values.shape != shape or np.isinf(values).any():
                        raise ValueError('Invalid blade review ' + prefix + '_' + suffix)
                    if suffix == 'axis_rate_rad_s' and not np.isnan(values[0]).all():
                        raise ValueError('First blade direction rate must remain unknown')
                    model[suffix] = values.copy()
                for suffix, shape in [('pitch_degrees', (len(pts), 2)),
                                      ('roll_degrees', (len(pts), 2)),
                                      ('runner_min_x_fraction', (len(pts), 2)),
                                      ('station_height_m', (len(pts), 2, 3)),
                                      ('runner_minus_pick_m', (len(pts), 2))]:
                    key = prefix + '_' + suffix
                    if key not in saved:continue
                    values = np.asarray(saved[key], dtype=float)
                    if values.shape != shape or not np.isfinite(values).all():
                        raise ValueError('Invalid blade review ' + key)
                    model[suffix] = values.copy()
                self.models[variant] = model
        self.rows = {}
        for key in ('source_boot_axes', 'source_regions', 'source_visibility', 'source_steel_segments', 'axis_metrics'):
            by_pts = {}
            for row in self.document.get(key, []):
                pts = native_pts(row['pts'])
                allowed_feet = (None, 0, 1) if key == 'source_regions' else (0, 1)
                if isinstance(row.get('foot'), bool) or row.get('foot') not in allowed_feet:
                    raise ValueError('Invalid blade review source foot')
                if key == 'source_regions':
                    region_outline(row)
                elif key == 'source_boot_axes':
                    for point in ('heel_uv', 'toe_uv'):
                        _points(row[point], (2,), 'source boot ' + point)
                    for extent in ('heel_half_extent_uv', 'toe_half_extent_uv'):
                        values = _points(row[extent], (2,), 'source boot ' + extent)
                        if np.any(values < 0):raise ValueError('Negative source boot uncertainty extent')
                elif key == 'source_steel_segments':
                    points = np.asarray(row['points'], float)
                    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
                        raise ValueError('Invalid visible steel subsegment')
                by_pts.setdefault(pts, []).append(row)
            self.rows[key] = by_pts

    def at(self, variant, pts):
        pts = native_pts(pts)
        frame = self.index.get(pts)
        model = self.models.get(variant)
        current = dict(pts=pts, model=None)
        if model is not None and frame is not None:
            current['model'] = {key: values[frame] for key, values in model.items()}
        current.update({key: rows.get(pts, []) for key, rows in self.rows.items()})
        current['axis_metrics'] = [row for row in current['axis_metrics'] if row.get('variant') == variant]
        return current


def _points(value, shape, label):
    values = np.asarray(value, float)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError('Invalid ' + label)
    return values


def overlay_primitives(current):
    """Return native-pixel lines/points/labels for the existing source overlay."""
    green = (.2, 1., .4, 1.)
    orange = (1., .54, .12, 1.)
    blue, yellow, white = (.3, .45, 1., 1.), (1., .86, .15, 1.), (.94, .98, 1., 1.)
    result = []

    def line(points, color, width=2.):
        points = np.asarray(points, float)
        if np.isfinite(points).all():result.append(dict(kind='line', points=points, color=color, width=width))

    def label(point, text, color):
        point = np.asarray(point, float)
        if np.isfinite(point).all():result.append(dict(kind='text', points=point[None], color=color, text=text))

    model = current['model']
    if model is not None:
        for foot in range(2):
            boot, blade = model['boot_uv'][foot], model['blade_uv'][foot]
            line(boot, orange)
            label(boot[0], 'M heel %d' % foot, orange)
            label(boot[1], 'M toe %d' % foot, orange)
            line(blade[[0, 2]], blue)
            for index, (name, color) in enumerate(zip(('R', 'M', 'F'), (blue, yellow, white))):
                if np.isfinite(blade[index]).all():
                    result.append(dict(kind='point', points=blade[index:index+1], color=color))
                    label(blade[index], '%s%d' % (name, foot), color)
    for row in current['source_regions']:
        line(region_outline(row), green)
        label(row['centre_uv'], 'S %s foot=%s' % (row.get('role', 'unknown'), row.get('foot')), green)
    for row in current['source_boot_axes']:
        line([row['heel_uv'], row['toe_uv']], green)
        for endpoint in ('heel', 'toe'):
            point = np.asarray(row[endpoint + '_uv'], float)
            extent = np.asarray(row[endpoint + '_half_extent_uv'], float)
            line(point + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * extent, green, 1.)
            label(point, 'S %s %d %s' % (endpoint, row['foot'], row.get('role', 'unknown')), green)
    for row in current['source_steel_segments']:
        line(row['points'], green, 3.)
        label(row['points'][0], 'S visible steel subsegment %d' % row['foot'], green)
    for foot in range(2):
        visibility = next((row for row in current['source_visibility'] if row['foot'] == foot), None)
        if visibility is None or visibility.get('blade_axis_observable') is not True:
            label([12., 28. + 20. * foot], 'S blade axis %d: unknown' % foot, green)
    if model is None:label([12., 8.], 'Model shoe/blade projection: unknown', orange)
    return result


def diagnostic_lines(current):
    """Report only saved values; a missing PTS/foot stays explicitly unknown."""
    def number(row, key):
        value = row.get(key) if row is not None else None
        try:value = float(value)
        except (TypeError, ValueError):return 'unknown'
        return 'unknown' if not np.isfinite(value) else '%.2f' % value

    lines = []
    for foot in range(2):
        metric = next((row for row in current['axis_metrics'] if row['foot'] == foot), None)
        model = current['model']
        rate = None if model is None else {'rate': model['axis_rate_rad_s'][foot]}
        lines.append('足 %d | 鞋轴差 %s° | 条件刀轴差 %s° | 模型20ms方向速率 %s rad/s' %
                     (foot, number(metric, 'boot_axis_error_deg'), number(metric, 'conditional_blade_axis_error_deg'), number(rate, 'rate')))
        visibility = next((row for row in current['source_visibility'] if row['foot'] == foot), None)
        observable = visibility is not None and visibility.get('blade_axis_observable') is True
        reason = visibility.get('orientation_unknown_reason') if visibility else None
        lines.append('足 %d | 原片刀轴：%s%s' %
                     (foot, '可辨' if observable else 'unknown', '' if observable else '；' + str(reason or '本PTS无可辨方向观测')))
        if model is not None:
            angles = []
            for key, label in [('pitch_degrees', '刀纵倾'), ('roll_degrees', '刀侧倾')]:
                if key in model:angles.append('%s %.2f°' % (label, model[key][foot]))
            if 'runner_min_x_fraction' in model:
                angles.append('runner最低点 %.3f（0后1前）' % model['runner_min_x_fraction'][foot])
            if angles:lines.append('足 %d | 模型 %s' % (foot, ' | '.join(angles)))
            heights = []
            if 'station_height_m' in model:
                heights.append('原始参考面高度 rear/mid/front %.2f/%.2f/%.2f cm' %
                               tuple(model['station_height_m'][foot] * 100.))
            if 'runner_minus_pick_m' in model:
                heights.append('runner-pick相对高度 %.2f cm（runner减pick）' %
                               (model['runner_minus_pick_m'][foot] * 100.))
            if heights:
                lines.append('足 %d | 模型 %s（原始诊断，非真实穿冰判据）' % (foot, ' | '.join(heights)))
    return lines
