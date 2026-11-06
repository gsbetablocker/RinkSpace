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
                for suffix in ('forward_world', 'lateral_world', 'up_world', 'ice_normal_world',
                               'fore_aft_pitch_degrees', 'edge_bank_degrees', 'total_lean_degrees',
                               'plane_normal_to_ice_normal_degrees', 'runner_u', 'edge_mode',
                               'contact_mode', 'mode_conflict'):
                    key = prefix + '_' + suffix
                    if key not in saved:continue
                    values = np.asarray(saved[key])
                    shape = (len(pts), 2, 3) if suffix.endswith('_world') else (len(pts), 2)
                    valid = values.shape == shape
                    if suffix in ('edge_mode', 'contact_mode'):
                        valid &= values.dtype.kind == 'U' and not np.any(np.char.strip(values) == '')
                    elif suffix == 'mode_conflict':
                        valid &= values.dtype.kind == 'b'
                    else:
                        valid &= values.dtype.kind in 'fiu' and np.isfinite(values).all()
                    if not valid:raise ValueError('Invalid blade review ' + key)
                    model[suffix] = values.copy()
                for edge_suffix in ('runner_edges_uv','runner_upper_edges_uv'):
                    edge_key = prefix + '_' + edge_suffix
                    if edge_key in saved:
                        edge_values = np.asarray(saved[edge_key], dtype=float)
                        if (edge_values.ndim != 5 or edge_values.shape[:3] != (len(pts), 2, 2)
                                or edge_values.shape[-1] != 2 or edge_values.shape[-2] < 2
                                or not np.isfinite(edge_values).all()):
                            raise ValueError('Invalid saved actual runner projection')
                        model[edge_suffix] = edge_values.copy()
                self.models[variant] = model
        self.rows = {}
        for key in ('source_boot_axes', 'source_regions', 'source_visibility', 'source_steel_segments', 'axis_metrics',
                    'source_structures', 'source_plane_locations', 'source_ice_grid'):
            by_pts = {}
            for row in self.document.get(key, []):
                pts = native_pts(row['pts'])
                allowed_feet = (None, 0, 1) if key in ('source_regions','source_structures') else (0, 1)
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
                elif key == 'source_structures':
                    if pts not in self.index or not isinstance(row.get('features'), dict):
                        raise ValueError('Invalid exact-PTS source structures')
                    for feature in row['features'].values():
                        points = np.asarray(feature.get('points_uv', []), float)
                        if points.size and (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()):
                            raise ValueError('Invalid source structure native points')
                elif key == 'source_plane_locations':
                    _points(row['xy_conditional_m'], (2,), 'source ice-plane location')
                elif key == 'source_ice_grid':
                    for grid in row['lines']:
                        points = np.asarray(grid['points_uv'], float)
                        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                            raise ValueError('Invalid source ice grid projection')
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

    def model_array(self, variant, key, pts):
        """Read a saved model array on exact PTS; missing samples stay unknown."""
        model = self.models.get(variant)
        if model is None or key not in model:return None
        indices = [self.index.get(native_pts(value)) for value in pts]
        if any(index is None for index in indices):return None
        return model[key][indices].copy()


def _points(value, shape, label):
    values = np.asarray(value, float)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError('Invalid ' + label)
    return values


def skate_axis_primitives(state, *, unit_ratio=1., ice_normal_world=None):
    """Arrow glyphs from saved skate transforms; no geometry or contact fit.

    The existing exported diagnostic axes are 0.08 candidate world units long.
    Their cache unit reexpression is also applied to these display-only glyphs.
    """
    rotation = np.asarray(state['skate_R'])
    if rotation.ndim != 4 or rotation.shape[1:] != (2, 3, 3) or not np.isfinite(rotation).all():
        raise ValueError('Invalid saved skate axis rotation')
    origin = _points(state['skate_origin'], (len(rotation), 2, 3), 'saved skate axis origin')
    if not np.isfinite(unit_ratio) or unit_ratio <= 0:raise ValueError('Invalid skate axis unit ratio')
    length = .08 * unit_ratio
    if 'axes' in state:
        saved = np.asarray(state['axes'])
        if saved.ndim != 5 or saved.shape[:2] != (len(rotation), 2) or saved.shape[-2:] != (2, 3) or not np.isfinite(saved).all():
            raise ValueError('Invalid saved diagnostic axes')
        lengths = np.linalg.norm(saved[..., 1, :] - saved[..., 0, :], axis=-1)
        if np.any(lengths <= 0):raise ValueError('Degenerate saved diagnostic axis length')
        length = np.median(lengths, axis=-1)
    lengths = np.broadcast_to(length, origin.shape[:2])
    vectors = [('F', rotation[..., 0], (.95, .25, .25, 1.)),
               ('L', rotation[..., 1], (.3, .95, .3, 1.)),
               ('U', rotation[..., 2], (.3, .6, 1., 1.))]
    if ice_normal_world is not None:
        normal = _points(ice_normal_world, origin.shape, 'saved ice normal')
        vectors.append(('iceN', normal, (1., .86, .15, 1.)))
    result = []
    for label, direction, color in vectors:
        # Perpendicular arrowheads affect only the drawing, never saved axes.
        norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        if np.any(norm <= 0):raise ValueError('Degenerate saved skate direction')
        unit = direction / norm
        side = rotation[..., 0] - np.sum(rotation[..., 0] * unit, axis=-1, keepdims=True) * unit
        fallback = rotation[..., 1] - np.sum(rotation[..., 1] * unit, axis=-1, keepdims=True) * unit
        side = np.where(np.linalg.norm(side, axis=-1, keepdims=True) > 1e-8, side, fallback)
        side /= np.linalg.norm(side, axis=-1, keepdims=True)
        tip = origin + lengths[..., None] * direction
        base = tip - lengths[..., None] * .22 * unit
        delta = lengths[..., None] * .12 * side
        points = np.stack((origin, tip, tip, base + delta, tip, base - delta), axis=2)
        for foot in range(2):
            result.append(dict(points=points[:, foot], color=color, foot=foot, label=label,
                               name='Model skate %s foot %d; not source contact' % (label, foot)))
    return result


def overlay_primitives(current, *, include_model_reference=True, include_model_contours=False,
                       include_source_structures=True, include_source_legacy=True, include_ice_grid=False):
    """Native primitives; source structures and model diagnostics are explicit layers."""
    green = (.2, 1., .4, 1.);orange = (1., .54, .12, 1.)
    blue, yellow, white = (.3, .45, 1., 1.), (1., .86, .15, 1.), (.94, .98, 1., 1.)
    result = []
    def line(points, color, width=2., **metadata):
        points=np.asarray(points,float)
        if len(points)>1 and np.isfinite(points).all():result.append(dict(kind='line',points=points,color=color,width=width,**metadata))
    def label(point,text,color):
        point=np.asarray(point,float)
        if np.isfinite(point).all():result.append(dict(kind='text',points=point[None],color=color,text=text))
    model=current['model']
    if model is not None and include_model_reference:
        for foot in range(2):
            boot,blade=model['boot_uv'][foot],model['blade_uv'][foot]
            line(boot,orange);label(boot[0],'M heel %d'%foot,orange);label(boot[1],'M toe %d'%foot,orange)
            line(blade[[0,2]],blue)
            for index,(name,color) in enumerate(zip(('R','M','F'),(blue,yellow,white))):
                if np.isfinite(blade[index]).all():
                    result.append(dict(kind='point',points=blade[index:index+1],color=color));label(blade[index],'%s%d'%(name,foot),color)
    if model is not None and include_model_contours:
        for suffix,color,width in [('runner_edges_uv',(1.,.15,.65,1.),2.),('runner_upper_edges_uv',(1.,.45,.75,1.),1.5)]:
            if suffix not in model:continue
            for foot in range(2):
                for edge in model[suffix][foot]:
                    line(edge,color,width,evidence_kind='model_actual_runner',foot=foot,model_curve_role=suffix)
    if include_ice_grid:
        for row in current.get('source_ice_grid',[]):
            for grid in row['lines']:line(grid['points_uv'],(.65,.65,.65,.55),1.,evidence_kind='auxiliary_ice_coordinates')
    if include_source_legacy:
        for row in current['source_regions']:
            line(region_outline(row),green);label(row['centre_uv'],'S %s foot=%s'%(row.get('role','unknown'),row.get('foot')),green)
        for row in current['source_boot_axes']:
            line([row['heel_uv'],row['toe_uv']],green)
            for endpoint in ('heel','toe'):
                point=np.asarray(row[endpoint+'_uv'],float);extent=np.asarray(row[endpoint+'_half_extent_uv'],float)
                line(point+np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])*extent,green,1.)
                label(point,'S %s %d %s'%(endpoint,row['foot'],row.get('role','unknown')),green)
        for row in current['source_steel_segments']:
            line(row['points'],green,3.);label(row['points'][0],'S visible steel subsegment %d'%row['foot'],green)
        for foot in range(2):
            visibility=next((row for row in current['source_visibility'] if row['foot']==foot),None)
            if visibility is None or visibility.get('blade_axis_observable') is not True:
                label([12.,28.+20.*foot],'S blade axis %d: unknown'%foot,green)
        if model is None and include_model_reference:label([12.,8.],'Model shoe/blade projection: unknown',orange)
    if include_source_structures:
        colors={'boot_outline':(.12,.65,.8,1.),'boot_sole':(.88,.49,.08,1.),'heel_block':(.71,.31,.06,1.),
                'blade_assembly_outline':(.53,.25,.71,1.),'runner_lower_edge':(.08,.69,.22,1.),
                'runner_upper_edge':(.08,.45,.25,1.),'toe_pick':(.86,.16,.18,1.),'rear_extension':(.2,.31,.84,1.),
                'front_stanchion':(.63,.25,.69,1.),'rear_stanchion':(.63,.25,.69,1.),
                'diagonal_brace':(.67,.25,.55,1.),'ice_interface':(.75,.57,0.,1.)}
        for row in current.get('source_structures',[]):
            for key,feature in row['features'].items():
                points=np.asarray(feature.get('points_uv',[]),float)
                if points.size==0:continue
                if feature.get('closed'):points=np.r_[points,points[:1]]
                color=colors.get(key,(.37,.37,.37,1.));status=feature.get('status','unknown')
                metadata=dict(evidence_kind='source_structure',feature=key,source_status=status,foot=row['foot'])
                if status=='observed':line(points,color,2.,**metadata)
                else:
                    for a,b in zip(points[:-1],points[1:]):
                        count=max(1,int(np.ceil(np.linalg.norm(b-a)/3.)))
                        for j in range(0,count,2):line([a+(b-a)*j/count,a+(b-a)*min(j+1,count)/count],color,2.,**metadata)
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
                     (foot, '可辨' if observable else 'unknown', '' if observable else '；' + str(reason or '本PTS无可辨方向观测')) +
                     '；源edge/contact：unknown/unknown')
        if model is not None:
            angles = []
            for key, label in [('pitch_degrees', '旧pitch诊断'), ('roll_degrees', '旧roll诊断')]:
                if key in model:angles.append('%s %.2f°' % (label, model[key][foot]))
            if 'runner_min_x_fraction' in model:
                angles.append('runner最低点 %.3f（0后1前）' % model['runner_min_x_fraction'][foot])
            if angles:lines.append('足 %d | 模型旧诊断 %s（非钢片角度测量）' % (foot, ' | '.join(angles)))
            vectors = []
            for key, label in [('forward_world', 'F'), ('lateral_world', 'L'), ('up_world', 'U'), ('ice_normal_world', 'iceN')]:
                if key in model:vectors.append('%s [%s]' % (label, ','.join('%.3f' % value for value in model[key][foot])))
            if vectors:lines.append('足 %d | 模型世界向量（非源测量）%s' % (foot, ' | '.join(vectors)))
            angles = []
            for key, label in [('fore_aft_pitch_degrees', 'fore-aft pitch'), ('edge_bank_degrees', 'edge bank'),
                               ('total_lean_degrees', 'total lean'), ('plane_normal_to_ice_normal_degrees', 'plane-normal / ice-normal')]:
                if key in model:angles.append('%s %.2f°' % (label, model[key][foot]))
            if angles:lines.append('足 %d | 模型向量角（非源测量）%s' % (foot, ' | '.join(angles)))
            modes = []
            if 'runner_u' in model:modes.append('runner_u %.6f（0后1前）' % model['runner_u'][foot])
            for key in ('edge_mode', 'contact_mode'):
                if key in model:modes.append('%s %s' % (key, model[key][foot]))
            if 'mode_conflict' in model:modes.append('mode_conflict %s' % str(bool(model['mode_conflict'][foot])).lower())
            if modes:lines.append('足 %d | 模型模式（非源接触）%s' % (foot, ' | '.join(modes)))
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
