"""One source-derived text-content chart, independent of banner corners.

The input consists only of photographed glyph contours. Shared cap/baseline
and straight vertical strokes supply an initialization homography. Constructed
content-frame bounds are never registered as observed panel corners. No font,
camera inverse projection, per-glyph warp or background pattern enters it.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np


def _contours(paths):
    if isinstance(paths, Mapping):
        records = paths.items()
    else:
        records = []
        for record in paths:
            label = record.get('key', record.get('label', record.get('id')))
            if not isinstance(label, str):
                raise ValueError('glyph contour needs a stable key or label')
            records.append((label.rsplit('.print.', 1)[-1], record))
    result = {}
    for key, record in records:
        points = record.get('source_native_uv') if isinstance(record, Mapping) else record
        q = np.asarray(points, dtype=np.float64)
        if key in result or q.ndim != 2 or q.shape[1] != 2 or len(q) < 3 or not np.isfinite(q).all():
            raise ValueError('unique finite source-native glyph contours are required')
        result[key] = q
    if not result:
        raise ValueError('text glyph contours are required')
    return result


def _robust_line(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    use = np.triu(np.abs(dx) > max(1e-10, np.ptp(x)*.05), 1)
    if not use.any():
        raise ValueError('line support is degenerate')
    slope = float(np.median(dy[use]/dx[use]))
    intercept = float(np.median(y-slope*x))
    error = y-(slope*x+intercept)
    return slope, intercept, error


def derive_text_chart(paths, cap_keys, vertical_keys, *, isotropic_at_content_midpoint=True):
    """Return ``{'H': ..., 'metadata': ...}`` without changing source records.

    Input keys identify *flat-cap/flat-baseline* glyphs and glyphs with straight
    outer vertical stems, respectively. Rounded S/O/U overshoot is deliberately
    not used as a flat-cap constraint, and all their points subsequently pass
    unchanged through this same H. H maps source UV to v-down content units
    with cap height one and its left content origin zero.

    A nearly parallel source vertical family is estimated from the middle 70%
    of each named glyph's outer envelopes. The cap/baseline convergence is then
    removed globally. The optional horizontal gauge preserves local isotropic
    source-pixel scale at the content midpoint. It does not establish the true
    physical aspect, metric dimensions, banner extent or camera calibration.
    """
    contours = _contours(paths)
    cap_keys, vertical_keys = tuple(cap_keys), tuple(vertical_keys)
    if len(set(cap_keys)) < 3 or len(set(vertical_keys)) < 1:
        raise ValueError('three flat glyphs and a vertical-stem glyph are required')
    if (set(cap_keys) | set(vertical_keys))-set(contours):
        raise ValueError('selected text-chart support glyph is missing')

    vertical = []
    for key in vertical_keys:
        q = contours[key]
        height = float(np.ptp(q[:, 1]))
        if height <= 0:
            raise ValueError('vertical glyph has no native height')
        ymin, ymax = q[:, 1].min()+.15*height, q[:, 1].max()-.15*height
        q = q[(q[:, 1] >= ymin) & (q[:, 1] <= ymax)]
        bins = max(8, min(48, int(round(.7*height))))
        edges = np.linspace(ymin, ymax+1e-9, bins+1)
        rows = []
        for low, high in zip(edges[:-1], edges[1:]):
            p = q[(q[:, 1] >= low) & (q[:, 1] < high)]
            if len(p) >= 2:
                rows.append([float(np.median(p[:, 1])), float(p[:, 0].min()), float(p[:, 0].max())])
        if len(rows) < 6:
            continue
        rows = np.asarray(rows)
        for side, col in (('left', 1), ('right', 2)):
            slope, intercept, error = _robust_line(rows[:, 0], rows[:, col])
            if abs(slope) > .25:
                continue
            vertical.append({'glyph':key, 'side':side, 'x_per_y':slope,
                'intercept':intercept, 'rms_native_px':float(np.sqrt(np.mean(error**2))),
                'row_support':len(rows)})
    if not vertical:
        raise ValueError('no supported near-vertical glyph edges')
    shear = float(np.median([line['x_per_y'] for line in vertical]))

    supports = {}
    lines = {}
    for side in ('cap', 'baseline'):
        samples = []
        for key in cap_keys:
            q = contours[key]
            extreme = q[:, 1].min() if side == 'cap' else q[:, 1].max()
            band = max(.75, .018*float(np.ptp(q[:, 1])))
            p = q[np.abs(q[:, 1]-extreme) <= band]
            x = float(np.median(p[:, 0]-shear*p[:, 1]))
            y = float(np.median(p[:, 1]))
            samples.append({'glyph':key, 'x_sheared':x, 'source_y':y})
        slope, intercept, error = _robust_line([p['x_sheared'] for p in samples],
                                              [p['source_y'] for p in samples])
        lines[side] = np.asarray([slope, intercept])
        for sample, residual in zip(samples, error):
            sample['line_residual_native_px'] = float(residual)
        supports[side] = samples

    all_points = np.concatenate(list(contours.values()))
    xs = all_points[:, 0]-shear*all_points[:, 1]
    left, right = float(xs.min()), float(xs.max())
    delta = lines['baseline']-lines['cap']
    hleft, hright = delta[0]*left+delta[1], delta[0]*right+delta[1]
    if min(hleft, hright) <= 0 or right <= left:
        raise ValueError('text cap/baseline geometry is invalid on source support')
    hmid = (hleft+hright)/2
    horizontal = hmid/hleft if isotropic_at_content_midpoint else 1.
    m, b = lines['cap']
    dm, db = delta
    H = np.asarray([[horizontal, -horizontal*shear, -horizontal*left],
                    [-m, 1+m*shear, -b], [dm, -dm*shear, db]])
    if abs(H[2, 2]) < 1e-12 or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError('text-content homography is degenerate')
    H /= H[2, 2]
    return {'H':H.tolist(), 'metadata':{
        'coordinate_unit':'source_text_cap_height_equals_one',
        'v_direction':'down', 'content_left_origin':0., 'physical_aspect_verified':False,
        'content_frame_is_observed_banner_rectangle':False, 'source_corners_created':False,
        'source_observations_created':0, 'individual_glyph_transform':False,
        'camera_projection_used':False, 'background_pattern_used':False,
        'isotropic_source_scale_at_content_midpoint':bool(isotropic_at_content_midpoint),
        'horizontal_scale_gauge':float(horizontal),
        'source_vertical_family':'estimated_nearly_parallel_text_stems',
        'vertical_x_per_y':shear, 'vertical_line_supports':vertical,
        'cap_keys':list(cap_keys), 'vertical_keys':list(vertical_keys),
        'cap_line_sheared':lines['cap'].tolist(),
        'baseline_line_sheared':lines['baseline'].tolist(),
        'line_supports':supports, 'source_cap_height_left_right_px':[float(hleft),float(hright)]}}
