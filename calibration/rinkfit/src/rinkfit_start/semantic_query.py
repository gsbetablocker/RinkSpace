"""Restore immutable real native query curves and certify their interpolation.

The historical navigation camera defines the *image sampling operator* only.
It does not constrain the new optimized world or camera. Original normal-
coverage scalar intervals retain their source units. In this event the frozen
navigation optics has only k1, so each query is an exact cubic/cubic rational
pixel curve. Bernstein convex hull bounds cover every finite interval and
every exported interpolation chord, without sampled error certification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from numpy.polynomial import Polynomial as Poly

from .data import sha256_file
from .runtime import atomic_json
from .semantic_snapshot import read_reference
from .semantics import _write_array


def _reference(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha256_file(path)}


def _project(camera, xy):
    camera = np.asarray(camera, float)
    points = np.c_[np.asarray(xy, float).reshape(-1, 2), np.zeros(len(np.asarray(xy).reshape(-1, 2)))]
    R = cv2.Rodrigues(camera[3:6])[0]
    K = np.array([[np.exp(camera[6]), 0., camera[8]], [0., np.exp(camera[7]), camera[9]], [0., 0., 1.]])
    uv = cv2.projectPoints(points, camera[3:6], -R@camera[:3], K, camera[11:25])[0].reshape(-1, 2)
    uv[:, 0] += camera[10]*(uv[:, 1]-camera[9])/np.exp(camera[7])
    return uv


def _ground(camera, query):
    """Exact original pure inverse reader; no source owner is constructed."""
    from calibration.solver_runtime import install
    install()
    from calibration.program_tracking.projection import ground_xy
    xy, valid = ground_xy(camera, np.asarray(query, float)[None])
    if not valid[0]:
        raise ValueError('Frozen navigation query cannot be restored onto its original sampling plane')
    return xy[0]


def query_rational(camera, query_uv, axis_xy, original_scalar_step):
    camera = np.asarray(camera, float); query = np.asarray(query_uv, float); axis = np.asarray(axis_xy, float)
    if camera.shape != (25,) or np.any(camera[12:] != 0.):
        raise ValueError('This exact query compiler requires the actual frozen k1-only navigation camera')
    if not np.isclose(np.linalg.norm(axis), 1., atol=1e-12, rtol=0):
        raise ValueError('Original physical section axis must be unit length')
    origin = _ground(camera, query)
    probe = _project(camera, origin+np.array([-.01, .01])[:, None]*axis)
    derivative = (probe[1]-probe[0])/.02; scale = float(np.linalg.norm(derivative))
    if not np.isfinite(scale) or scale <= 0 or original_scalar_step <= 0:
        raise ValueError('Invalid original physical query sampling scale')
    R = cv2.Rodrigues(camera[3:6])[0]
    start = R@(np.r_[origin, 0.]-camera[:3])
    direction = R@np.r_[axis*(original_scalar_step/scale), 0.]
    A, B, D = [Poly([start[i], direction[i]]) for i in range(3)]
    radius = A*A+B*B
    xd = A*D*D+camera[11]*A*radius
    yd = B*D*D+camera[11]*B*radius
    denominator = D**3
    U = camera[8]*denominator+np.exp(camera[6])*xd+camera[10]*yd
    V = camera[9]*denominator+np.exp(camera[7])*yd
    norm = float(denominator.coef[0])
    if norm <= 0:
        raise ValueError('Original sampling query is not in front of its navigation camera')
    return U/norm, V/norm, denominator/norm, D/(norm**(1/3)), {
        'query_world_xy_original_sampling_only': origin.tolist(),
        'physical_width_axis_xy_original_sampling_only': axis.tolist(),
        'local_pixels_per_original_world_unit': scale,
        'original_scalar_to_local_profile_pixels': original_scalar_step,
        'restored_query_tangent_unit': (derivative/scale).tolist()}


def _bernstein_abs(polynomial, lo, hi):
    """Convex hull bound for all real polynomial values on [lo, hi]."""
    local = polynomial(Poly([float(lo), float(hi-lo)]))
    a = np.asarray(local.coef); degree = len(a)-1
    coefficients = np.array([sum(a[j]*math.comb(k, j)/math.comb(degree, j)
        for j in range(k+1)) for k in range(degree+1)])
    # Explicit outward envelope for the small-degree affine conversion and
    # Bernstein sums. This is numerical arithmetic allowance, not pixel slack.
    padding = 512*np.finfo(float).eps*max(1., float(np.sum(np.abs(a))))
    return float(np.max(np.abs(coefficients))+padding)


def rational_bound(numerator, denominator, depth, lo, hi):
    depth_lower = min(float(depth(lo)), float(depth(hi)))
    if depth_lower <= 0:
        raise ValueError('Original finite query support crosses a navigation-camera pole')
    return _bernstein_abs(numerator, lo, hi)/(depth_lower**3)


def evaluate_curve(U, V, denominator, alpha):
    alpha = np.asarray(alpha, float)
    return np.stack((U(alpha)/denominator(alpha), V(alpha)/denominator(alpha)), axis=-1)


def certified_polyline(U, V, denominator, depth, lo, hi, *, tolerance_px=.001):
    """Subdivide until every full chord has a rational Bernstein certificate."""
    queue = [(float(lo), float(hi))]; accepted = []
    while queue:
        a, b = queue.pop(); endpoints = evaluate_curve(U, V, denominator, [a, b])
        line_u = Poly([endpoints[0, 0], (endpoints[1, 0]-endpoints[0, 0])/(b-a)])(Poly([-a, 1.]))
        line_v = Poly([endpoints[0, 1], (endpoints[1, 1]-endpoints[0, 1])/(b-a)])(Poly([-a, 1.]))
        bx = rational_bound(U-line_u*denominator, denominator, depth, a, b)
        by = rational_bound(V-line_v*denominator, denominator, depth, a, b)
        bound = float(np.hypot(bx, by))+1e-9
        if bound <= tolerance_px:
            accepted.append((a, b, bound))
        else:
            middle = (a+b)/2
            if middle == a or middle == b:
                raise ValueError('Floating interval cannot certify original query interpolation')
            queue.append((middle, b)); queue.append((a, middle))
    accepted.sort()
    alpha = np.array([accepted[0][0]]+[item[1] for item in accepted])
    return alpha, evaluate_curve(U, V, denominator, alpha), accepted


def export_query_packet(semantic_manifest, navigation_camera_archive, output_dir, *, tolerance_px=.001):
    manifest_path = Path(semantic_manifest); semantic_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    spec = manifest['assets']['profiles.jsonl']
    if sha256_file(semantic_root/spec['file']) != spec['sha256']:
        raise ValueError('Original full profile evidence changed')
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        raise ValueError('Query packet is already frozen; use a new version directory')
    camera_path = Path(navigation_camera_archive)
    with np.load(camera_path, allow_pickle=False) as saved:
        cameras = {int(i): c.copy() for i, c in zip(saved['programme_indices'], saved['camera25'])}
    records = [json.loads(line) for line in (semantic_root/spec['file']).read_text(encoding='utf-8').splitlines()]
    native_cache = {}; source_rows = []; offsets = [0]; alpha_values = []; uv_values = []; intervals = []
    poly_u = []; poly_v = []; poly_d = []; certificates = []; origins = Counter()
    max_chord = max_transverse = max_scalar_change = max_query_origin = max_measured_curve_error = 0.
    for record in records:
        source_ref = record['native_readout_packet']; key = source_ref['path'], source_ref['sha256']
        if key not in native_cache:
            native_cache[key] = read_reference(source_ref)
        packet = native_cache[key]; pts = int(packet['source']['native_pts']); frame = (pts-256256)//256+1
        if pts != 256256+(frame-1)*256 or frame not in cameras:
            raise ValueError('Original profile query does not match the navigation camera clock')
        camera = cameras[frame]; evidence = record['source_evidence']; query = np.asarray(evidence['query_uv'], float)
        normal = np.asarray(evidence['profile_normal'], float); bounds = evidence['complete_band_native_interval_px']
        step = evidence.get('physical_width_profile_native_sampling_step_px')
        if step is None:
            alignment = evidence.get('transverse_profile_image_normal_coverage_scale')
            if alignment is None:
                raise ValueError('Original coverage-to-pixel query scale is missing')
            step = 1/float(alignment)
        U, V, D, depth, provenance = query_rational(camera, query, evidence['physical_width_axis_xy'], step)
        if np.linalg.norm(np.asarray(provenance['restored_query_tangent_unit'])-normal) > 1e-10:
            raise ValueError('Restored original sampling tangent differs from saved profile direction')
        lo, hi = map(float, bounds)
        tangent = np.array([-normal[1], normal[0]])
        h = tangent[0]*(U-query[0]*D)+tangent[1]*(V-query[1]*D)
        transverse = rational_bound(h, D, depth, lo, hi)
        pixel_alpha = normal[0]*(U-query[0]*D)+normal[1]*(V-query[1]*D)
        scalar_change = rational_bound(pixel_alpha-Poly([0., 1.])*D, D, depth, lo, hi)
        alpha, uv, chords = certified_polyline(U, V, D, depth, lo, hi, tolerance_px=tolerance_px)
        origin_error = float(np.linalg.norm(evaluate_curve(U, V, D, 0.)-query))
        observed_world = np.asarray(evidence['actual_measured_midpoint_ground_xy'])
        original_world = np.asarray(provenance['query_world_xy_original_sampling_only'])
        original_axis = np.asarray(evidence['physical_width_axis_xy'])
        measured_alpha = float((observed_world-original_world)@original_axis)*provenance['local_pixels_per_original_world_unit']/step
        curve_midpoint = evaluate_curve(U, V, D, measured_alpha)
        measured_projection = _project(camera, observed_world)[0]
        measured_curve_error = float(np.linalg.norm(curve_midpoint-measured_projection))
        max_query_origin = max(max_query_origin, origin_error)
        max_measured_curve_error = max(max_measured_curve_error, measured_curve_error)
        source_rows.append(record['source_row']); offsets.append(offsets[-1]+len(alpha))
        alpha_values.extend(alpha); uv_values.extend(uv); intervals.append([lo, hi])
        pad = lambda p: np.pad(p.coef, (0, 4-len(p.coef)))
        poly_u.append(pad(U)); poly_v.append(pad(V)); poly_d.append(pad(D))
        max_chord = max(max_chord, max(c[2] for c in chords)); max_transverse = max(max_transverse, transverse)
        max_scalar_change = max(max_scalar_change, scalar_change); origins[evidence['measurement_origin']] += 1
        certificates.append({'source_row': record['source_row'], 'frame': frame,
            'original_readout_packet': source_ref, 'query_index': record['query_index'],
            'query_uv': query.tolist(), 'profile_normal': normal.tolist(),
            'original_parameter_units': record['query_parameter_units'], 'original_finite_interval': [lo, hi],
            'local_scalar_to_pixel_alpha_difference_bound_px': scalar_change,
            'legacy_straight_query_max_transverse_error_bound_px': transverse,
            'query_origin_roundtrip_error_px': origin_error, 'measured_midpoint_distance_to_original_curve_px': measured_curve_error,
            'chord_certificates': [{'alpha_min': a, 'alpha_max': b, 'max_error_px_bound': e} for a, b, e in chords],
            **provenance})
    arrays = {}
    for key, filename, values, dtype in [
        ('source_rows', 'source_rows.u64', source_rows, '<u8'), ('offsets', 'offsets.u64', offsets, '<u8'),
        ('alpha', 'alpha.f64', alpha_values, '<f8'), ('uv', 'uv.f64', uv_values, '<f8'),
        ('bounds', 'bounds.f64', intervals, '<f8'), ('numerator_u', 'numerator_u.f64', poly_u, '<f8'),
        ('numerator_v', 'numerator_v.f64', poly_v, '<f8'), ('denominator', 'denominator.f64', poly_d, '<f8')]:
        arrays[key] = _write_array(output/filename, values, dtype)
    atomic_json(output/'certificates.json', certificates)
    result = {'schema': 'rinkfit.immutable_native_query_curve_packet.v1', 'profiles': len(records),
        'semantic_manifest': _reference(manifest_path), 'original_navigation_camera': _reference(camera_path),
        'arrays': arrays, 'certificates': _reference(output/'certificates.json'),
        'query_curve_definition': 'immutable historical Brown projection of original physical transverse sampling section',
        'parameter_definition': 'original normal-equivalent coverage coordinate; not Euclidean image alpha',
        'rational_coefficients': 'ascending powers; pixel=(numerator_u, numerator_v)/denominator',
        'curve_sampling': 'piecewise linear at alpha/uv; every chord certified by rational Bernstein convex hull',
        'maximum_interpolation_error_bound_px': max_chord, 'frozen_interpolation_tolerance_px': tolerance_px,
        'legacy_straight_query_max_transverse_error_bound_px': max_transverse,
        'maximum_original_scalar_to_pixel_alpha_difference_bound_px': max_scalar_change,
        'maximum_query_origin_roundtrip_error_px': max_query_origin,
        'maximum_measured_midpoint_distance_to_original_curve_px': max_measured_curve_error,
        'measurement_origins': dict(origins), 'original_finite_intervals_retained_without_unit_relabeling': True,
        'source_UV_ID_material_W_changed': False, 'old_scene_or_camera_is_new_world_truth': False,
        'exporter': _reference(__file__)}
    atomic_json(output/'manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-manifest', required=True, type=Path)
    parser.add_argument('--navigation-camera', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    result = export_query_packet(args.semantic_manifest, args.navigation_camera, args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k != 'arrays'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
