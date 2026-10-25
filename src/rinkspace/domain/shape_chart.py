"""Exact axis-aligned material quotient charts for the common parameter layout.

Only compilation/encoding lives here. Forward geometry and station inversion
consume the existing native shape evaluator; this is not a second optimizer.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json

import numpy as np

SCHEMA = 'rinkspace.shape_local_chart.v1'


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def shape_group_definition(template, group, *, fixed_material_points=()):
    """Include the complete declared group and reject a split parent/hole."""
    ids = set(group['path_ids'])
    paths = {p['id']: p for p in template['paths']}
    if not ids or not ids <= paths.keys():
        raise ValueError('Missing shape group path')
    for p in paths.values():
        parent = p.get('parent_contour_key')
        if parent is not None and ((p['id'] in ids) != (parent in ids)):
            raise ValueError('Shape chart splits parent/hole ownership: '+p['id'])
    result = deepcopy(group)
    result['paths'] = [{k:deepcopy(p[k]) for k in
        ('id', 'material_uv', 'closed', 'role', 'parent_contour_key') if k in p}
        for identity in sorted(ids) for p in [paths[identity]]]
    result['fixed_material_points'] = deepcopy(list(fixed_material_points))
    return result


def _vertices(path):
    vertices = np.asarray(path['material_uv'], dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 2 or not np.isfinite(vertices).all():
        raise ValueError('Invalid material polyline: '+path['id'])
    if path.get('closed') and not np.array_equal(vertices[0], vertices[-1]):
        vertices = np.vstack([vertices, vertices[0]])
    return vertices


def _constraints(path):
    vertices = _vertices(path)
    edges = np.diff(vertices, axis=0)
    if np.any(np.count_nonzero(edges, axis=1) > 1):
        raise ValueError('Non-axis-aligned edge: '+path['id'])
    # Remove duplicate drawing vertices for the proof, without changing the
    # original vertex-index station layout stored in the chart.
    points = vertices[np.r_[True, np.any(edges != 0, axis=1)]]
    closed = bool(path.get('closed'))
    if closed and np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 2:
        raise ValueError('Degenerate shape path: '+path['id'])
    keep = []
    for i, p in enumerate(points):
        if not closed and i in (0, len(points)-1):
            keep.append(p)
            continue
        before, after = p-points[i-1], points[(i+1) % len(points)]-p
        # A reversal is an extremal endpoint of the material set, not an
        # ordinary collinear drawing subdivision.
        if np.dot(before, after) <= 0:
            keep.append(p)
    return np.asarray(keep, dtype=float)


def _axis(q):
    q = sorted(set(float(x) for x in q))
    a = np.clip(4*np.asarray(q)[:, None]-np.arange(4), 0., 1.)
    # Exact rational row elimination selects the rank. No SVD threshold can
    # silently turn a nearly aligned segment or a narrow feature into a gauge.
    pivots, selected = {}, []
    for row in a:
        exact = list(map(Fraction.from_float, row))
        mean = sum(exact)/4
        reduced = [x-mean for x in exact]
        for pivot, basis in pivots.items():
            scale = reduced[pivot]
            reduced = [x-scale*y for x, y in zip(reduced, basis)]
        pivot = next((i for i, x in enumerate(reduced) if x), None)
        if pivot is not None:
            scale = reduced[pivot]
            pivots[pivot] = [x/scale for x in reduced]
            selected.append(np.asarray([float(x-mean) for x in exact]))
    basis = []
    for row in selected:
        for previous in basis:
            row = row-np.dot(row, previous)*previous
        row = row/np.linalg.norm(row)
        first = np.flatnonzero(abs(row) > 1e-14)[0]
        if row[first] < 0:
            row = -row
        basis.append(row)
    return {'q':q, 'A':a.tolist(), 'B':np.asarray(basis).reshape(-1, 4).tolist(),
            'rank':len(basis), 'rank_with_constant':len(basis)+1}


def compile_shape_local_chart(group_definition, coordinate_gauge):
    """Compile a stable maximum-entropy representative of an exact polyline.

    group_definition contains all group paths and any genuinely fixed material
    coordinates. Ordinary contour observations are not material landmarks.
    """
    group = deepcopy(group_definition)
    if group.get('parameterization') != 'monotone_axis_warp.v1':
        raise ValueError('Only monotone_axis_warp.v1 has this chart')
    paths = group['paths']
    if len({p['id'] for p in paths}) != len(paths) or set(group['path_ids']) != {p['id'] for p in paths}:
        raise ValueError('Chart requires every group path exactly once')
    ids = set(group['path_ids'])
    if any(p.get('parent_contour_key') is not None and p['parent_contour_key'] not in ids for p in paths):
        raise ValueError('Chart omits a parent contour')
    box = np.asarray(group['reference_bounds'], dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError('Invalid reference bounds')
    points = [_constraints(p) for p in paths]
    for point in group.get('fixed_material_points', []):
        if not isinstance(point.get('id'), str) or point.get('path_id') not in ids:
            raise ValueError('Fixed material point needs an identity and group path')
        uv = np.asarray(point['material_uv'], dtype=float)
        vertices = _vertices(next(p for p in paths if p['id'] == point['path_id']))
        on_edge = any(np.all(uv >= np.minimum(a,b)) and np.all(uv <= np.maximum(a,b))
            and np.all(uv[a == b] == a[a == b]) for a,b in zip(vertices[:-1],vertices[1:]))
        if uv.shape != (2,) or not np.isfinite(uv).all() or not on_edge:
            raise ValueError('Fixed material point is outside its actual path')
        points.append(uv[None, :])
    points = np.vstack(points)
    q = (points-box[:2])/(box[2:]-box[:2])
    if np.any(q < 0) or np.any(q > 1):
        raise ValueError('Shape constraints outside reference bounds')
    gauge = deepcopy(coordinate_gauge)
    fixed = []
    if gauge.get('kind') == 'reference_group_layout':
        if gauge['parameter_block_id'] == group['parameter_block_id']:
            fixed = list(gauge['fixed_local_indices'])
    elif gauge.get('kind') != 'unit_panel_chart':
        raise ValueError('Unknown coordinate gauge')
    if len(set(fixed)) != len(fixed) or any(type(i) is not int or not 0 <= i < 4 for i in fixed):
        raise ValueError('Unsupported affine gauge')
    axes = [_axis(q[:, i]) for i in (0, 1)]
    affine = [i for i in range(4) if i not in fixed]
    size = len(affine)+sum(axis['rank'] for axis in axes)
    expansion = np.zeros((10, size))
    for col, row in enumerate(affine):
        expansion[row, col] = 1.
    offset = len(affine)
    for start, axis in zip((4, 7), axes):
        b = np.asarray(axis['B']).reshape(-1, 4)
        expansion[start:start+3, offset:offset+len(b)] = (b[:, :3]-b[:, 3:]).T
        offset += len(b)
    chart = {'schema':SCHEMA, 'group_id':group['id'], 'parameter_block_id':group['parameter_block_id'],
        'path_ids':sorted(ids), 'ambient_size':10, 'tangent_size':size,
        'u_basis':axes[0]['B'], 'v_basis':axes[1]['B'], 'axes':axes,
        'fixed_affine_indices':fixed, 'affine_indices':affine,
        'd_ambient_d_local':expansion.tolist(), 'ambient_origin':[0.]*10,
        'definition':group, 'coordinate_gauge':gauge,
        'proof':'strict_axis_edges_complete_turns_open_endpoints_holes_and_fixed_material_points',
        'basis_convention':'sorted_q_exact_rank_ordered_gram_schmidt_first_nonzero_positive'}
    chart['revision'] = _hash(chart)
    return chart


def validate_shape_local_chart(chart):
    rebuilt = compile_shape_local_chart(chart['definition'], chart['coordinate_gauge'])
    if rebuilt != chart:
        raise ValueError('Shape local chart content/revision differs from its proof')


def expand_shape_state(chart, local_values):
    local = np.asarray(local_values, dtype=float)
    if local.shape != (chart['tangent_size'],) or not np.isfinite(local).all():
        raise ValueError('Invalid shape local values')
    ambient = np.asarray(chart['ambient_origin'])+np.asarray(chart['d_ambient_d_local']) @ local
    if not np.isfinite(ambient).all():
        raise ValueError('Nonfinite shape expansion')
    return ambient


def _softmax(values):
    values = np.asarray(values, dtype=float)
    weights = np.exp(values-np.max(values))
    weights /= weights.sum()
    if not np.all(weights > 0):
        raise ValueError('Shape weights reach the numerical simplex boundary')
    return weights


def _encode_axis(basis, weights):
    b = np.asarray(basis, dtype=float).reshape(-1, 4)
    if not len(b):
        return np.empty(0)
    target = b @ weights
    local = np.zeros(len(b))
    # Convex dual moment match. Iteration limits are numerical guards only;
    # failure is explicit and never clips a state to a new shape domain.
    for _ in range(100):
        w = _softmax(b.T @ local)
        gradient = b @ w-target
        if np.max(abs(gradient)) <= 4e-15:
            return local
        hessian = (b*w) @ b.T-np.outer(b @ w, b @ w)
        try:
            direction = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError('Shape moment match is numerically singular') from exc
        before = np.linalg.norm(gradient)
        for power in range(50):
            candidate = local-direction*(.5**power)
            candidate_gradient = b @ _softmax(b.T @ candidate)-target
            if np.linalg.norm(candidate_gradient) < before:
                local = candidate
                break
        else:
            raise ValueError('Shape moment matching did not decrease its residual')
    raise ValueError('Shape moment matching did not converge')


def encode_shape_state(chart, ambient_shape10):
    ambient = np.asarray(ambient_shape10, dtype=float)
    if ambient.shape != (10,) or not np.isfinite(ambient).all():
        raise ValueError('Invalid ambient shape')
    fixed = chart['fixed_affine_indices']
    if fixed and np.any(ambient[fixed] != np.asarray(chart['ambient_origin'])[fixed]):
        raise ValueError('Existing reference affine gauge changed')
    values = [ambient[chart['affine_indices']]]
    for start, name in ((4, 'u_basis'), (7, 'v_basis')):
        values.append(_encode_axis(chart[name], _softmax(np.r_[ambient[start:start+3], 0.])))
    return np.concatenate(values)


def migrate_shape_stations(chart, path_id, stations, old_values, new_values, core):
    """Preserve native material points on the same original path/seed edge."""
    path = next((p for p in chart['definition']['paths'] if p['id'] == path_id), None)
    if path is None:
        raise ValueError('Station path is outside shape chart')
    vertices = _vertices(path)
    old = np.asarray(stations, dtype=float)
    if old.ndim != 1 or not np.isfinite(old).all() or np.any((old < 0) | (old > 1)):
        raise ValueError('Invalid finite station support')
    if not len(old) or np.array_equal(old_values, new_values):
        return old.copy()
    scaled = old*(len(vertices)-1)
    edge = np.minimum(np.floor(scaled).astype(int), len(vertices)-2)
    fraction = scaled-edge
    first, last = vertices[edge], vertices[edge+1]
    seeds = first+(last-first)*fraction[:, None]
    bounds = chart['definition']['reference_bounds']
    target = core.deform_print_points(seeds, bounds, old_values)
    if np.any(target['status']):
        raise ValueError('Native old material state invalid')
    target = np.asarray(target['material_uv'])
    axis = np.argmax(abs(last-first), axis=1)
    rows = np.arange(len(old))
    increasing = last[rows, axis] >= first[rows, axis]
    low, high = np.zeros(len(old)), np.ones(len(old))
    for _ in range(56):
        middle = (low+high)/2
        candidate = core.deform_print_points(first+(last-first)*middle[:, None], bounds, new_values)
        if np.any(candidate['status']):
            raise ValueError('Native canonical material state invalid')
        actual = np.asarray(candidate['material_uv'])
        below = (actual[rows, axis] < target[rows, axis]) == increasing
        low = np.where(below, middle, low)
        high = np.where(below, high, middle)
    migrated = (edge+(low+high)/2)/(len(vertices)-1)
    at_vertex = fraction == 0
    at_end = old == 1
    migrated[at_vertex | at_end] = old[at_vertex | at_end]
    final_seed = first+(last-first)*(migrated*(len(vertices)-1)-edge)[:, None]
    actual = core.deform_print_points(final_seed, bounds, new_values)
    if np.any(actual['status']) or not np.allclose(actual['material_uv'], target, rtol=0., atol=3e-14):
        raise ValueError('Station migration cannot preserve the original material point on its edge')
    return migrated



def load_shape_local_charts(path):
    """Read an explicit proven chart selection for prepare/SceneParameterModel."""
    from pathlib import Path
    document = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if document.get('schema') != 'rinkspace.shape_local_charts.v1':
        raise ValueError('Unknown shape chart collection')
    charts = {}
    for chart in document['charts']:
        validate_shape_local_chart(chart)
        identity = chart['parameter_block_id']
        if identity in charts:
            raise ValueError('Duplicate shape chart block')
        charts[identity] = chart
    return charts
