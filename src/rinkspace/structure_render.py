"""Render supplied whole-rink hypotheses, without fitting or validating them.

All arcs/patterns are caller-supplied sampled paths. No rink dimensions, curve
radii, physical identities, support neighbourhoods or metres are inferred here.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


STYLES = {
    'observed': {'rgb': (67, 215, 180), 'dash': None, 'zh': '观测', 'en': 'observed'},
    'inferred': {'rgb': (134, 174, 255), 'dash': (3., 8.), 'zh': '推断', 'en': 'inferred'},
    'assumed': {'rgb': (255, 186, 83), 'dash': (10., 9.), 'zh': '假设', 'en': 'assumed'},
}
CLOSED_KINDS = {'boundary', 'polygon', 'circle', 'ring', 'closed_curve'}
FRONT_BRANCH_EPSILON = 1e-9


def _points(value, minimum=2):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum or not np.isfinite(points).all():
        raise ValueError(f'Expected at least {minimum} finite model/image XY points')
    return points


def _primitive(value, *, boundary=False):
    if not isinstance(value, dict) or not value.get('id'):
        raise ValueError('Every primitive needs an explicit ID')
    status = value.get('status')
    if status not in STYLES:
        raise ValueError('Every primitive needs observed, inferred or assumed status')
    kind = 'boundary' if boundary else value.get('kind', 'polyline')
    points = _points(value.get('points_model'), 3 if boundary else 1 if kind == 'point' else 2)
    closed = boundary or bool(value.get('closed')) or kind in CLOSED_KINDS
    if closed and len(points) < 3:
        raise ValueError('A closed primitive needs at least three points')
    anchor = value.get('label_anchor_model')
    if anchor is not None:
        anchor = _points([anchor], 1)[0]
    else:
        anchor = points[0] if kind in {'boundary', 'point'} else points[len(points)//2]
    return {**value, 'kind': kind, 'status': status, 'points': points, 'closed': closed,
            'label_anchor': anchor, 'label': str(value.get('label') or value['id'])}


def _scene(model):
    structure = model.get('structure')
    if not isinstance(structure, dict) or not isinstance(structure.get('boundary'), dict):
        raise ValueError('structure.boundary must declare a closed contour with status and ID')
    boundary = _primitive(structure['boundary'], boundary=True)
    primitives = [boundary, *[_primitive(p) for p in structure.get('primitives', [])]]
    ids = [p['id'] for p in primitives]
    if len(ids) != len(set(ids)):
        raise ValueError('Structure primitive IDs must be unique, including boundary')
    grid = model.get('grid', {})
    bounds = structure.get('bounds', grid.get('bounds'))
    if bounds is not None:
        bounds = np.asarray(bounds, float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all() or not (bounds[0] < bounds[2] and bounds[1] < bounds[3]):
            raise ValueError('Bounds must be finite [xmin,ymin,xmax,ymax]')
    # Grid intervals are drawn only when supplied; bounds alone never invent
    # metre or equal-physical-spacing grid values.
    if bounds is not None:
        x0, y0, x1, y1 = bounds
        status = grid.get('status', 'assumed')
        if status not in STYLES:
            raise ValueError('Grid status must be observed, inferred or assumed')
        for axis, values in [('x', grid.get('x_values', [])), ('y', grid.get('y_values', []))]:
            if not isinstance(values, (list, tuple)) or len(values) > 1000 or not np.isfinite(values).all():
                raise ValueError('Grid values must contain at most 1000 finite coordinates per axis')
            for index, value in enumerate(values):
                points = [[value, y0], [value, y1]] if axis == 'x' else [[x0, value], [x1, value]]
                primitive = _primitive({'id': f'__grid_{axis}_{index}', 'kind': 'line',
                    'points_model': points, 'status': status, 'label': ''})
                primitive.update(label='', grid=True)
                primitives.append(primitive)
    return structure, primitives, bounds


def _matrix(value):
    matrix = np.asarray(value, float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.any(matrix):
        raise ValueError('Projection must be a finite nonsingular 3x3 H_image_from_model')
    matrix = matrix / np.max(np.abs(matrix))
    if abs(np.linalg.det(matrix)) < 1e-14:
        raise ValueError('Projection matrix is singular or numerically degenerate')
    return matrix


def _homogeneous(matrix, point):
    vector = np.r_[point, 1.]
    vector /= max(1., np.max(np.abs(vector)))
    result = matrix @ vector
    scale = np.max(np.abs(result))
    return result / scale if scale > 0 and np.isfinite(result).all() else None


def _front_branch_segment(left, right, sign):
    """Keep one signed homogeneous half-space, without joining path vertices.

    Endpoint normalization uses positive scales, so a positive linear blend
    stays on the same original projective segment. Clip before perspective
    division; otherwise a back-facing endpoint can reappear inside the image.
    """
    left, right = left*sign, right*sign
    left_front, right_front = left[2] > FRONT_BRANCH_EPSILON, right[2] > FRONT_BRANCH_EPSILON
    if not left_front and not right_front:
        return None, None, 'behind_plane', False
    if left_front and right_front:
        return left, right, None, False
    target_w = np.nextafter(FRONT_BRANCH_EPSILON, np.inf)
    t = (target_w-left[2])/(right[2]-left[2])
    clipped = left+(right-left)*t
    # Eliminate cancellation at the near-plane intersection, not geometric
    # extrapolation; the other homogeneous components remain interpolated.
    clipped[2] = target_w
    if left_front:
        right = clipped
    else:
        left = clipped
    return left, right, None, True


def _visible_segment(matrix, a, b, width, height, front_branch_sign=None, branch_counts=None):
    """Clip in homogeneous image coordinates; never connect across infinity."""
    left, right = _homogeneous(matrix, a), _homogeneous(matrix, b)
    if left is None or right is None:
        return None, 'nonfinite'
    if front_branch_sign is not None:
        left, right, reason, clipped = _front_branch_segment(left, right, front_branch_sign)
        if reason:
            return None, reason
        if clipped and branch_counts is not None:
            branch_counts['clipped_branch_segments'] += 1
    elif left[2] == 0 or right[2] == 0 or np.sign(left[2]) != np.sign(right[2]):
        return None, 'projective_horizon'
    if left[2] < 0:
        left, right = -left, -right
    lo, hi = 0., 1.
    # x>=0, y>=0, x<width, y<height; normalized endpoints avoid integer
    # overflow even when a valid projected endpoint lies far outside the image.
    for plane in ([1, 0, 0], [0, 1, 0], [-1, 0, width-1], [0, -1, height-1]):
        f0, f1 = float(np.dot(plane, left)), float(np.dot(plane, right))
        if f0 < 0 and f1 < 0:
            return None, 'outside_image'
        if f0 < 0:
            lo = max(lo, f0 / (f0-f1))
        elif f1 < 0:
            hi = min(hi, f0 / (f0-f1))
        if lo > hi:
            return None, 'outside_image'
    points = [left+(right-left)*t for t in (lo, hi)]
    if any(p[2] <= 0 for p in points):
        return None, 'projective_horizon'
    result = np.array([p[:2]/p[2] for p in points])
    if not np.isfinite(result).all():
        return None, 'nonfinite'
    return np.clip(result, [0, 0], [width-1, height-1]), None


def _distortion_parameters(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('Distortion must supply k1, centre_image and scale_image')
    centre = _points([value.get('centre_image')], 1)[0]
    k1, scale = value.get('k1'), value.get('scale_image')
    if not isinstance(k1, (int, float)) or not np.isfinite(k1) or not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Candidate radial k1 must be finite and scale_image positive')
    return {'k1': float(k1), 'centre_image': centre.tolist(), 'scale_image': float(scale)}


def _distort(xy, distortion):
    if distortion is None:
        return xy
    centre = np.asarray(distortion['centre_image'], float)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        normalized = (xy-centre)/distortion['scale_image']
        r2 = float(np.dot(normalized, normalized))
        # This candidate radial model is used only on its invertible central
        # branch. Distant points must not fold back into the source image.
        if 1+3*distortion['k1']*r2 <= 0:
            return None
        result = centre+(xy-centre)*(1+distortion['k1']*r2)
    return result if np.isfinite(result).all() else None


def _distorted_segments(matrix, a, b, width, height, distortion, front_branch_sign=None, branch_counts=None):
    # Sample before projecting/distorting: a straight model line can become a
    # curved image path. Adjacent intervals crossing H's horizon stay separate.
    points = np.linspace(a, b, 33)
    for left, right in zip(points, points[1:]):
        ha, hb = _homogeneous(matrix, left), _homogeneous(matrix, right)
        if ha is None or hb is None:
            yield None, 'nonfinite'
            continue
        if front_branch_sign is not None:
            ha, hb, reason, clipped = _front_branch_segment(ha, hb, front_branch_sign)
            if reason:
                yield None, reason
                continue
            if clipped and branch_counts is not None:
                branch_counts['clipped_branch_segments'] += 1
        elif ha[2] == 0 or hb[2] == 0 or np.sign(ha[2]) != np.sign(hb[2]):
            yield None, 'projective_horizon'
            continue
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            pa, pb = _distort(ha[:2]/ha[2], distortion), _distort(hb[:2]/hb[2], distortion)
        if pa is None or pb is None or not np.isfinite([pa, pb]).all():
            yield None, 'nonfinite'
            continue
        # Clip only after distortion; an unwarped offscreen endpoint does not
        # imply that the warped curve is offscreen.
        yield _visible_segment(np.eye(3), pa, pb, width, height)


def _image_point(matrix, point, width, height, distortion=None, front_branch_sign=None):
    p = _homogeneous(matrix, point)
    if p is None or p[2] == 0 or (front_branch_sign is not None and front_branch_sign*p[2] <= FRONT_BRANCH_EPSILON):
        return None
    xy = _distort(p[:2]/p[2], distortion)
    if xy is None:
        return None
    return xy if np.isfinite(xy).all() and 0 <= xy[0] < width and 0 <= xy[1] < height else None


def _stroke(layer, alpha, segment, style, thickness, phase):
    a, b = segment
    length = float(np.linalg.norm(b-a))
    color = tuple(reversed(style['rgb']))
    def line(p, q):
        first, last = tuple(np.rint(p).astype(int)), tuple(np.rint(q).astype(int))
        cv2.line(layer, first, last, color, thickness, cv2.LINE_AA)
        cv2.line(alpha, first, last, 255, thickness, cv2.LINE_AA)
    if style['dash'] is None:
        line(a, b)
        return phase
    on, off = style['dash']; period = on+off
    position = 0.
    while position < length:
        offset = (phase+position) % period
        step = min(length-position, (on-offset) if offset < on else (period-offset))
        if step < 1e-7:
            step = min(length-position, 1e-5)
        if offset < on and length > 0:
            line(a+(b-a)*position/length, a+(b-a)*(position+step)/length)
        position += step
    return (phase+length) % period


@lru_cache(maxsize=16)
def _font(size):
    for name, cjk in [('C:/Windows/Fonts/msyh.ttc', True), ('C:/Windows/Fonts/simhei.ttf', True),
        ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', True),
        ('/System/Library/Fonts/PingFang.ttc', True), ('DejaVuSans.ttf', False)]:
        try:
            return ImageFont.truetype(name, size), name, cjk
        except OSError:
            pass
    return ImageFont.load_default(), 'Pillow default', False


def _allowed_mask(shape, ice_mask, person_polygons):
    height, width = shape
    allowed = np.full(shape, 255, np.uint8)
    if ice_mask is not None:
        mask = np.asarray(ice_mask)
        if mask.shape != shape or not np.isfinite(mask).all():
            raise ValueError('Ice mask must match the full input image pixel domain')
        allowed[mask <= 0] = 0
    for polygon in person_polygons or []:
        points = _points(polygon, 3)
        if np.max(np.abs(points)) > 1e7:
            raise ValueError('Person polygon exceeds safe image-coordinate range')
        # Image-domain polygons may extend past the image edges; OpenCV safely
        # raster-clips them. Each polygon subtracts from the union separately.
        cv2.fillPoly(allowed, [np.rint(points).astype(np.int32)], 0)
    return allowed


def _draw_geometry(base, primitives, matrix, allowed, distortion=None, front_branch_sign=None):
    height, width = base.shape[:2]
    layer, alpha = np.zeros_like(base), np.zeros((height, width), np.uint8)
    labels, records = [], []
    font, font_name, cjk = _font(max(11, min(19, round(width/70))))
    for primitive in primitives:
        points = primitive['points']; style = STYLES[primitive['status']]
        segments = list(zip(points, points[1:]))
        if primitive['closed'] and not np.array_equal(points[0], points[-1]):
            segments.append((points[-1], points[0]))
        record = {'id': primitive['id'], 'kind': primitive['kind'], 'status': primitive['status'],
            'closed': primitive['closed'], 'visible_segments': 0, 'rejected_projective_horizon_segments': 0,
            'rejected_behind_plane_segments': 0, 'clipped_branch_segments': 0,
            'outside_image_segments': 0, 'nonfinite_segments': 0}
        phase = 0.
        thickness = 1 if primitive.get('grid') else 3 if primitive['kind'] == 'boundary' else 2
        for a, b in segments:
            candidates = (_distorted_segments(matrix,a,b,width,height,distortion,front_branch_sign,record)
                          if distortion is not None else [_visible_segment(matrix,a,b,width,height,front_branch_sign,record)])
            for segment, reason in candidates:
                if reason:
                    key = {'projective_horizon': 'rejected_projective_horizon_segments',
                           'behind_plane': 'rejected_behind_plane_segments',
                           'outside_image': 'outside_image_segments', 'nonfinite': 'nonfinite_segments'}[reason]
                    record[key] += 1
                    continue
                record['visible_segments'] += 1
                phase = _stroke(layer, alpha, segment, style, thickness, phase)
        if primitive['kind'] == 'point':
            p = _image_point(matrix, points[0], width, height, distortion, front_branch_sign)
            if p is not None:
                for delta in (np.array([4., 0.]), np.array([0., 4.])):
                    _stroke(layer, alpha, [p-delta, p+delta], style, 2, 0)
                record['visible_segments'] += 2
        if primitive['kind'] == 'direction' and len(points) >= 2:
            tip = _image_point(matrix, points[-1], width, height, distortion, front_branch_sign)
            if distortion is None:
                shaft, reason = _visible_segment(matrix, points[-2], points[-1], width, height, front_branch_sign)
            else:
                shaft, reason = list(_distorted_segments(matrix,points[-2],points[-1],width,height,distortion,front_branch_sign))[-1]
            if tip is not None and shaft is not None and not reason:
                direction = shaft[1]-shaft[0]; distance = np.linalg.norm(direction)
                if distance > 1e-6:
                    unit = direction/distance; normal = np.array([-unit[1], unit[0]])
                    for side in (-1, 1):
                        _stroke(layer, alpha, [tip, tip-unit*12+normal*side*5], style, 2, 0)
        anchor = _image_point(matrix, primitive['label_anchor'], width, height, distortion, front_branch_sign)
        if primitive['label'] and anchor is not None:
            text = primitive.get('display_label',primitive['label'])
            labels.append((anchor, text if cjk else str(primitive['id']), style['rgb']))
        records.append(record)
    alpha = cv2.bitwise_and(alpha, allowed)
    ratio = alpha.astype(float)/255
    # OpenCV antialias colors on a black layer are already premultiplied.
    combined = np.clip(base.astype(float)*(1-ratio[..., None])+layer*(allowed[..., None]/255), 0, 255).astype(np.uint8)
    result = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)).convert('RGBA')
    text_layer = Image.new('RGBA', result.size)
    draw = ImageDraw.Draw(text_layer)
    for anchor, text, color in labels:
        if not text:
            continue
        bounds=draw.textbbox((0,0),text,font=font)
        text_width=bounds[2]-bounds[0]
        position=(max(6,min(float(anchor[0])+6,width-text_width-6)),max(6,float(anchor[1])-20))
        draw.text(position, text, font=font, fill=(*color, 255),
                  stroke_width=2, stroke_fill=(10, 19, 26, 230))
    text_alpha = np.asarray(text_layer.getchannel('A')).astype(np.uint16)*allowed//255
    text_layer.putalpha(Image.fromarray(text_alpha.astype(np.uint8)))
    result = Image.alpha_composite(result, text_layer).convert('RGB')
    return result, {'primitives': records, 'font': font_name, 'cjk_font_available': cjk,
        'rejected_behind_plane_segments': sum(p['rejected_behind_plane_segments'] for p in records),
        'clipped_branch_segments': sum(p['clipped_branch_segments'] for p in records),
        'geometry_drawn_pixels_before_labels': int(np.count_nonzero(alpha))}


def _decorate(image, *, projection, unit, y_down=False, distortion=None):
    width, height = image.size
    size = max(10, min(20, round(width/60)))
    font, _, cjk = _font(size)
    small, _, _ = _font(max(9, size-3))
    draw = ImageDraw.Draw(image)
    header = size*3+9+(size+4 if distortion is not None else 0)
    draw.rectangle((0, 0, width, header), fill=(14, 24, 34))
    title = '整场条件草模投影' if projection else '整场条件草模'
    subtitle = '比例与角部含假设 / 非实测尺寸 / 仅绘制给定坐标'
    if not cjk:
        title = 'Whole-rink conditional projection' if projection else 'Whole-rink conditional sketch'
        subtitle = 'Proportions/corners include assumptions; not measured dimensions'
    draw.text((12, 6), title, font=font, fill=(240, 246, 252))
    draw.text((12, size+12), subtitle, font=small, fill=(200, 216, 228))
    if distortion is not None:
        note = f'候选径向畸变（未标定） k1={distortion["k1"]:g}' if cjk else f'Candidate radial distortion, NOT CALIBRATED: k1={distortion["k1"]:g}'
        draw.text((12, size*2+17), note, font=small, fill=STYLES['assumed']['rgb'])
    footer = max(42, size*2+18)
    draw.rectangle((0, height-footer, width, height), fill=(14, 24, 34))
    for index, (status, style) in enumerate(STYLES.items()):
        x = 12+index*width/3; y = height-footer+size/2+7
        if style['dash'] is None:
            draw.line((x, y, x+27, y), fill=style['rgb'], width=2)
        else:
            step = 5 if status == 'inferred' else 12
            for offset in range(0, 27, step):
                draw.line((x+offset, y, x+min(offset+step/2, 27), y), fill=style['rgb'], width=2)
        label = f'{style["zh"]} {status}' if cjk else status
        draw.text((x+33, y-size/2-1), label, font=small, fill=style['rgb'])
    note = f'坐标单位：{unit or "未注明"}；全图均为候选草模，颜色不代表独立验收通过。' if cjk else 'Given coordinates only; status colors do not certify independent geometric accuracy.'
    draw.text((12, height-size-6), note, font=small, fill=(194, 211, 222))
    if not projection:
        origin = (25, height-footer-18)
        draw.line([origin, (origin[0]+35, origin[1])], fill=(200, 210, 224), width=2)
        draw.line([origin, (origin[0], origin[1]+(28 if y_down else -28))], fill=(200, 210, 224), width=2)
        draw.text((origin[0]+38, origin[1]-8), 'x', font=small, fill=(220, 230, 240))
        draw.text((origin[0]-5, origin[1]+(29 if y_down else -44)), 'y', font=small, fill=(220, 230, 240))
    return [(0, 0, width, header), (0, height-footer, width, height)]


def _save(image, out_path, model, metadata):
    path = Path(out_path)
    if path.suffix.lower() != '.png':
        raise ValueError('Structure render output must be a PNG')
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format='PNG')
    return {'image_path': str(path.resolve()), 'image_width': image.width, 'image_height': image.height,
        'model_id': model.get('model_id'), 'model_revision': model.get('revision'),
        'geometry_level': model.get('geometry_level'), 'unit': model.get('unit'),
        'status': 'conditional_structure_visualization_only', 'independent_geometry_validated': False,
        'status_legend': {key: value['zh'] for key, value in STYLES.items()}, **metadata}


def render_structure_overview(model, out_path, *, size=(1200, 900), y_down=False):
    """Draw the complete supplied contour/parts, with equal model-unit scaling.

    structure.bounds is preferred over grid.bounds. The viewport expands when
    supplied paths extend beyond those bounds; coordinates are never modified.
    Equal drawing scale does not assert real-world Euclidean or metric truth.
    """
    width, height = size
    if not isinstance(width, int) or not isinstance(height, int) or width < 320 or height < 240:
        raise ValueError('Overview size must be integer pixels, at least 320x240')
    structure, primitives, declared = _scene(model)
    all_points = np.vstack([p['points'] for p in primitives]+[p['label_anchor'][None] for p in primitives])
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    if declared is not None:
        lo, hi = np.minimum(lo, declared[:2]), np.maximum(hi, declared[2:])
    span = hi-lo
    if np.any(span <= 0):
        raise ValueError('Whole-structure bounds must have positive X and Y spans')
    lo -= span*.06; hi += span*.06; span=hi-lo
    padding_x, top, bottom = 58., 105., 95.
    scale = min((width-2*padding_x)/span[0], (height-top-bottom)/span[1])
    if scale <= 0:
        raise ValueError('Overview viewport is too small for annotations')
    offset_x = (width-span[0]*scale)/2
    offset_y = top+(height-top-bottom-span[1]*scale)/2
    matrix = np.array([[scale, 0, offset_x-scale*lo[0]],
        [0, scale if y_down else -scale, offset_y-scale*lo[1] if y_down else offset_y+scale*hi[1]], [0, 0, 1.]])
    base = np.full((height, width, 3), (34, 26, 18), np.uint8)
    image, info = _draw_geometry(base, primitives, _matrix(matrix), np.full((height, width), 255, np.uint8))
    captions = _decorate(image, projection=False, unit=model.get('unit'), y_down=y_down)
    return _save(image, out_path, model, {**info, 'view': 'whole_structure_overview',
        'H_overview_from_model': matrix.tolist(), 'y_axis_display_direction': 'down' if y_down else 'up',
        'declared_bounds': declared.tolist() if declared is not None else None,
        'rendered_bounds': [*lo.tolist(), *hi.tolist()], 'boundary_closed': True,
        'primitive_status_counts': dict(Counter(p['status'] for p in primitives if not p.get('grid'))),
        'caption_pixel_regions': captions})


def render_structure_projection(image_bgr, model, H_image_from_model, out_path, *,
                                ice_mask=None, person_polygons_image=None, distortion=None,
                                front_branch_sign=None):
    """Project given whole-structure paths into unchanged input pixel geometry.

    Only an explicit full-image ice mask and supplied person polygons restrict
    the overlay. Legacy landmark neighbourhood/support polygons are not read.
    By default, segments crossing the projective horizon are rejected. An
    explicit +1/-1 front_branch_sign retains only s*w>epsilon, clipping each
    original segment before division; rejected vertices are never reconnected.
    Its sign refers to the supplied H, not a silently chosen matrix gauge.
    Optional radial distortion is a caller-supplied candidate, never a camera
    calibration: c+(Hxy-c)*(1+k1*sum(((Hxy-c)/scale_image)**2)).
    """
    if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError('Expected uint8 BGR source image')
    if image_bgr.shape[0] < 120 or image_bgr.shape[1] < 240:
        raise ValueError('Projection image must be at least 240x120 for readable provenance labels')
    if (front_branch_sign is not None and
            (isinstance(front_branch_sign, (bool, np.bool_)) or
             not isinstance(front_branch_sign, (int, float, np.integer, np.floating)) or
             front_branch_sign not in (-1, 1))):
        raise ValueError('front_branch_sign must be None, +1 or -1 in the supplied H gauge')
    _, primitives, _ = _scene(model)
    matrix = _matrix(H_image_from_model)
    distortion = _distortion_parameters(distortion)
    allowed = _allowed_mask(image_bgr.shape[:2], ice_mask, person_polygons_image)
    image, info = _draw_geometry(image_bgr.copy(), primitives, matrix, allowed, distortion, front_branch_sign)
    captions = _decorate(image, projection=True, unit=model.get('unit'), distortion=distortion)
    return _save(image, out_path, model, {**info, 'view': 'whole_structure_projection',
        'H_image_from_model': np.asarray(H_image_from_model, float).tolist(),
        'image_coordinate_domain': 'input_image_pixels', 'image_resized': False,
        'distortion': {**distortion, 'status': 'candidate_not_calibrated'} if distortion is not None else None,
        'samples_per_input_segment': 32 if distortion is not None else 1,
        'front_branch_sign': int(front_branch_sign) if front_branch_sign is not None else None,
        'front_branch_epsilon': FRONT_BRANCH_EPSILON if front_branch_sign is not None else None,
        'clipping': {'ice_mask_supplied': ice_mask is not None,
            'person_polygon_count': len(person_polygons_image) if person_polygons_image is not None else 0,
            'allowed_image_pixels': int(np.count_nonzero(allowed)),
            'legacy_landmark_neighbourhoods_used': False}, 'caption_pixel_regions': captions,
        'primitive_status_counts': dict(Counter(p['status'] for p in primitives if not p.get('grid')))})
