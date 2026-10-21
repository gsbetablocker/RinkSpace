"""Explicit screen -> provisional physical initialization, with transfer losses.

This module never imports historical K/R/C. It consumes the audited H state,
uses carrier/conic operators (not equal curve parameters) to reconcile charts,
derives provisional pinhole poses, and fits the actual native head/motion model.
The output remains a candidate: numerical transfer is not source calibration.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .io_utils import read_json, write_json, file_sha256, object_sha256
from .joint_model import as_event_view
from .native_baseline_init import Motion, _angles, _physical, _head, _free
from rinkfit_start.native_scene import NativeScene, camera_from_event


def _stats(x):
    x = np.asarray(x, float).reshape(-1)
    finite = np.isfinite(x)
    a = x[finite]
    return dict(denominator=len(x), finite=int(finite.sum()), unknown=int((~finite).sum()),
                rms=float(np.sqrt(np.mean(a*a))) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                p95=float(np.percentile(a, 95)) if len(a) else None,
                max=float(a.max()) if len(a) else None,
                min=float(a.min()) if len(a) else None)


def _project(H, xy):
    q = np.einsum('...ij,...kj->...ki', H, np.concatenate((xy, np.ones(xy.shape[:-1]+(1,))), -1))
    with np.errstate(divide='ignore', invalid='ignore'):
        return q[..., :2]/q[..., 2:3]


def _closest_reference(curve, xy, *, finite=True):
    """Entity operator in reference coordinates; no point identity is invented."""
    if curve['current_path_kind'] == 0:
        a, b = np.asarray(curve['old_endpoints_L'])
        d = b-a
        station = (xy-a)@d/(d@d)
        if finite:
            station = np.clip(station, 0., 1.)
        return a+station[..., None]*d
    c = np.asarray(curve['old_center_L'])
    d = xy-c
    length = np.linalg.norm(d, axis=-1)
    if np.any(length < 1e-12):
        raise ValueError('Circle-center query has no unique nearest entity point')
    return c+curve['old_radius_L']*d/length[..., None]


def _chart_from_entities(model, audit_file):
    audit = read_json(audit_file)
    curves = []
    coordinates = []
    for curve in audit['curves']:
        matches = [m for m in model['materials'] if m['label'] == curve['current_material_label']]
        if len(matches) != 1 or len(matches[0]['paths']) != 1:
            raise ValueError('Ambiguous current template entity: '+curve['curve_id'])
        path = matches[0]['paths'][0]
        if path['kind'] != curve['current_path_kind'] or path['kind'] not in (0, 1):
            raise ValueError('Template entity operator changed: '+curve['curve_id'])
        block = model['blocks'][path['geometry_block']]
        q = np.asarray(block['values'])
        if path['kind'] == 0:
            expected = np.asarray(curve['current_endpoints_L'])
            if not np.allclose(q[:4].reshape(2, 2), expected, rtol=0, atol=1e-13):
                raise ValueError('Template audit does not describe this current line')
            u = np.linspace(0., 1., 17)
            xy = expected[0]+u[:, None]*(expected[1]-expected[0])
        else:
            ri = path.get('radius_index', 2)
            if not np.allclose(q[:2], curve['current_center_L'], rtol=0, atol=1e-13) or abs(q[ri]-curve['current_radius_L']) > 1e-13:
                raise ValueError('Template audit does not describe this current conic')
            angle = np.linspace(0., 2*np.pi, 32, endpoint=False)
            xy = q[:2]+q[ri]*np.c_[np.cos(angle), np.sin(angle)]
        curves.append(curve)
        coordinates.append(xy)

    def matrix(v):
        # A projective warp can hide nonuniform template changes by destroying
        # the Euclidean metric needed for calibrated pose decomposition. Use a
        # similarity gauge and expose all irreducible geometry differences.
        scale = np.exp(v[0])
        c, s = np.cos(v[1]), np.sin(v[1])
        return np.array([[scale*c, -scale*s, v[2]], [scale*s, scale*c, v[3]], [0., 0., 1.]])

    def residual(v):
        T = matrix(v)
        out = []
        for curve, xy in zip(curves, coordinates):
            z = _project(T, xy)
            # Existing template endpoints were not measured as point identities.
            # Reconcile line carriers here; retain the actual finite extents in
            # later screen loss. Do not force mismatched terminal extents to fit.
            nearest = _closest_reference(curve, z, finite=False)
            # Each named entity has equal total chart weight. This is navigation,
            # not a pixel covariance or a same-station/angle correspondence.
            out.append((z-nearest).ravel()/np.sqrt(len(z)))
        return np.concatenate(out)

    result = least_squares(residual, np.zeros(4), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    T = matrix(result.x)
    if abs(np.linalg.det(T)) < 1e-10:
        raise ValueError('Entity reconciliation produced a singular chart')
    per_entity = []
    for curve, xy in zip(curves, coordinates):
        z = _project(T, xy)
        per_entity.append(dict(curve_id=curve['curve_id'],
                               reference_plane_error_L=_stats(np.linalg.norm(z-_closest_reference(curve, z), axis=1))))
    return T, curves, coordinates, dict(
        audit=dict(path=str(Path(audit_file).resolve()), sha256=file_sha256(audit_file)),
        current_to_reference_H=T.tolist(), entity_count=len(curves), per_entity=per_entity,
        chart_family='Euclidean_similarity; nonuniform_template_difference_exposed',
        operator='line_carrier_normal_or_circle_radial_distance; finite_extent_loss_separately_reported',
        equal_station_or_equal_angle_correspondences=False, ellipse_center_correspondences=False,
        geometry_equality_certified=False, source_qualification_granted=False,
        reason='Two nonuniformly changed templates cannot generally be reconciled by a single projective chart.')


def _estimate_principal(H):
    """Shared PP from all H orthogonality constraints, with variable frame f."""
    z = H[:, 2, :2]
    u = np.c_[z[:, 0]*z[:, 1], z[:, 0]**2-z[:, 1]**2]
    def residual(pp):
        A = H[:, :2, :2]-pp[None, :, None]*z[:, None, :]
        v = np.c_[np.sum(A[:, :, 0]*A[:, :, 1], axis=1),
                   np.sum(A[:, :, 0]**2-A[:, :, 1]**2, axis=1)]
        denominator = np.linalg.norm(v, axis=1)*np.linalg.norm(u, axis=1)
        if np.any(denominator <= 1e-30):
            raise ValueError('Degenerate H cannot constrain provisional principal point')
        return (v[:, 0]*u[:, 1]-v[:, 1]*u[:, 0])/denominator
    result = least_squares(residual, [960., 540.],
                           bounds=([-1920., -1080.], [3840., 2160.]),
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    singular = np.linalg.svd(result.jac, compute_uv=False)
    return result.x, dict(principal_point=result.x.tolist(), constraint_rms=float(np.sqrt(np.mean(result.fun**2))),
                          Jacobian_singular_values=singular.tolist(), assumptions=['square_pixels', 'zero_skew', 'shared_PP_with_variable_f'],
                          estimated_from_H=True, imported_historical_K=False, hardware_calibration_claimed=False)


def _pinhole_from_H(H, xy, pp=None):
    """Estimate f from two orthogonality equations and explicit provisional PP."""
    if pp is None:
        pp = np.broadcast_to(np.array([960., 540.]), (len(H), 2))
    pp = np.asarray(pp, float)
    if pp.shape == (2,):
        pp = np.broadcast_to(pp, (len(H), 2))
    A = H[:, :2, :2]-pp[:, :, None]*H[:, 2, :2][:, None, :]
    z = H[:, 2, :2]
    v = np.c_[np.sum(A[:, :, 0]*A[:, :, 1], axis=1),
               np.sum(A[:, :, 0]**2-A[:, :, 1]**2, axis=1)]
    u = np.c_[z[:, 0]*z[:, 1], z[:, 0]**2-z[:, 1]**2]
    invf2 = -np.sum(v*u, axis=1)/np.sum(v*v, axis=1)
    if not np.isfinite(invf2).all() or np.any(invf2 <= 0):
        raise ValueError('H has no positive provisional square-pixel focal solution; retain ambiguity')
    f = 1/np.sqrt(invf2)
    K = np.broadcast_to(np.eye(3), (len(H), 3, 3)).copy()
    K[:, 0, 0] = f
    K[:, 1, 1] = f
    K[:, :2, 2] = pp
    B = np.linalg.solve(K, H)
    scale = 2/(np.linalg.norm(B[:, :, 0], axis=1)+np.linalg.norm(B[:, :, 1], axis=1))
    branches = []
    X = np.concatenate((xy, np.zeros(xy.shape[:-1]+(1,))), -1)
    for sign in (1., -1.):
        r1 = sign*scale[:, None]*B[:, :, 0]
        r2 = sign*scale[:, None]*B[:, :, 1]
        M = np.stack((r1, r2, np.cross(r1, r2)), -1)
        U, _, Vt = np.linalg.svd(M)
        D = np.broadcast_to(np.eye(3), M.shape).copy()
        D[:, 2, 2] = np.linalg.det(U@Vt)
        R = U@D@Vt
        t = sign*scale[:, None]*B[:, :, 2]
        C = -np.einsum('nji,nj->ni', R, t)
        Y = np.einsum('nij,nkj->nki', R, X-C[:, None, :])
        pred = np.einsum('nij,nkj->nki', K, Y)
        uv = pred[:, :, :2]/pred[:, :, 2:3]
        branches.append((R, C, uv, Y[:, :, 2]))
    select = (branches[0][1][:, 2] > 0)
    R = np.where(select[:, None, None], branches[0][0], branches[1][0])
    C = np.where(select[:, None], branches[0][1], branches[1][1])
    uv = np.where(select[:, None, None], branches[0][2], branches[1][2])
    depth = np.where(select[:, None], branches[0][3], branches[1][3])
    if np.any(C[:, 2] <= 0):
        raise ValueError('No above-ice pose branch for a native H')
    return R, C, np.log(f), uv, depth


def audit_optical_coverage(model_file, compact_seed_file, source_array_file, source_metadata_file,
                          independent_native_summary_file, output_file):
    """List real source exposures and audited CP rays per rig without fitting optics.

    The focal and ray ranges are marginal ranges, not a certified rectangular
    calibration domain. Source rows retain their original qualification; this
    function does not admit additional pixels or infer camera hardware specs.
    """
    model = read_json(model_file)
    seed = read_json(compact_seed_file)
    source = np.load(source_array_file, allow_pickle=False)
    metadata = read_json(source_metadata_file)
    summary = read_json(independent_native_summary_file)
    frame_focal = {}
    source_focal = {}
    per_shot = {}
    for i, sid in enumerate(metadata['shot_ids']):
        shot = model['shots'][sid]
        camera = camera_from_event(as_event_view(model, sid))
        frame_pts = np.arange(shot['scope']['first_pts'], shot['scope']['last_pts']+1, 256, dtype=np.int64)
        focal = np.exp(camera.spline((frame_pts-shot['time_origin_pts'])/12800.)[:, 2])
        frame_focal[sid] = focal
        selected = source['shot_index'] == i
        pts = source['native_pts'][selected]
        source_focal[sid] = np.exp(camera.spline((pts-shot['time_origin_pts'])/12800.)[:, 2])
        rays = summary['stages']['initial_state']['shots'][sid]['ray_radius']
        per_shot[sid] = dict(rig=shot['rig_id'], native_frame_focal_px=_stats(focal),
                             active_source_rows=int(selected.sum()), active_source_exposures=len(np.unique(pts)),
                             active_source_focal_px=_stats(source_focal[sid]),
                             audited_CP_ray_radius=rays,
                             declared_radius=camera.optics['radius_max'])
    rigs = {}
    for rid in model['rigs']:
        sids = [sid for sid, s in model['shots'].items() if s['rig_id'] == rid]
        minimum = min(per_shot[s]['audited_CP_ray_radius']['min'] for s in sids)
        maximum = max(per_shot[s]['audited_CP_ray_radius']['max'] for s in sids)
        rigs[rid] = dict(shots=sids, native_frame_focal_px=_stats(np.concatenate([frame_focal[s] for s in sids])),
                         active_source_focal_px=_stats(np.concatenate([source_focal[s] for s in sids])),
                         active_source_audited_CP_ray_radius_range=[minimum, maximum],
                         rectangular_focal_ray_coverage_certified=False,
                         new_wide_optical_calibration_completed=False)
    first_min = min(frame_focal['s01'])
    first_max = max(frame_focal['s01'])
    wide_lines = []
    material_map = {m['material_index']:m for m in metadata['material_map']}
    for sid in ('s04', 's07', 's09'):
        shot_index = metadata['shot_ids'].index(sid)
        selected = source['shot_index'] == shot_index
        rows = np.flatnonzero(selected)
        rows = rows[source_focal[sid] < first_min]
        groups = {}
        for row in rows:
            material = material_map[int(source['material_index'][row])]
            if not material['label'].startswith('ice.') or material['paths'][0]['kind'] != 0:
                continue
            key = (int(source['native_pts'][row]), int(source['material_index'][row]))
            groups.setdefault(key, []).append(row)
        counts = np.asarray([len(v) for v in groups.values()])
        wide_lines.append(dict(shot=sid, exposures=len({k[0] for k in groups}), line_groups=len(groups),
                               source_line_rows=int(counts.sum()),
                               groups_with_only_one_or_two_UV_points=int((counts <= 2).sum()),
                               groups_with_at_least_three_UV_points=int((counts >= 3).sum()),
                               these_rows_alone_certify_wide_distortion=False))
    report = dict(schema='rinkfit.actual_rig_optical_support_audit.v1',
                  input_model=dict(path=str(Path(model_file).resolve()), sha256=file_sha256(model_file)),
                  sources=dict(arrays_sha256=file_sha256(source_array_file),
                               metadata_sha256=file_sha256(source_metadata_file),
                               independent_CP_summary_sha256=file_sha256(independent_native_summary_file)),
                  per_shot=per_shot, rigs=rigs,
                  preserved_S01_focal_interval_px=[float(first_min), float(first_max)],
                  wide_line_capture=wide_lines, calibration_truth_claimed=False,
                  conclusion='A larger ray disk does not extend Brown calibration. Sparse independent UV groups do not by themselves identify wide line curvature.',
                  required_consumers=['Python NativeCamera', 'C++ physical/event model',
                                      'raw GPU exact math and derivative ABI',
                                      'domain/RS certificates', 'replay and serialization'])
    write_json(output_file, report)
    return report


def _shared_fit(motion, h, o, meta, X, target, mask, targetq, static_indices):
    """Fit actual shared head and native cubic motion to screen UV witnesses."""
    # Solve increments from existing allocated coordinates. Solving absolute
    # coordinates with a minimum-norm least-squares solver sets unsupported
    # control columns to zero (including log f), corrupting held intervals.
    from scipy.optimize import lsq_linear
    x0 = motion.x.copy()
    for axis, D in enumerate(motion.D):
        columns = np.flatnonzero(motion.axis == axis)
        A = D[:, columns]
        current = motion.constant[:, axis]+D@x0
        delta = lsq_linear(A, targetq[:, axis]-current,
                           bounds=(motion.lo[columns]-x0[columns], motion.hi[columns]-x0[columns]),
                           tol=1e-11, lsmr_tol=1e-12).x
        x0[columns] += delta
    ids = np.asarray(static_indices, int)
    hb = motion.model['blocks'][motion.head_block]
    initial = np.r_[h[ids], x0]
    lower = np.r_[np.asarray(hb['lower'])[ids], motion.lo]
    upper = np.r_[np.asarray(hb['upper'])[ids], motion.hi]
    scales = np.r_[np.asarray(hb['scales'])[ids], motion.scale]
    width = X.shape[1]*2

    def unpack(x):
        hh = h.copy()
        hh[ids] = x[:len(ids)]
        return motion.values(x[len(ids):]), hh

    def local(q, hh):
        uv, *_ = _physical(q, hh, o, meta, X)
        return np.where(mask[..., None], uv-target, 0.).reshape(len(X), -1)

    def fun(x):
        return local(*unpack(x)).ravel()

    def jac(x):
        from scipy import sparse
        q, hh = unpack(x)
        base = local(q, hh)
        eps = 2e-7
        parts = []
        if len(ids):
            columns = []
            for j in ids:
                trial = hh.copy()
                trial[j] += eps
                columns.append(((local(q, trial)-base)/eps).ravel())
            parts.append(sparse.csr_matrix(np.column_stack(columns)))
        rr, cc, vv = [], [], []
        for axis, D in enumerate(motion.coo):
            trial = q.copy()
            trial[:, axis] += eps
            derivative = (local(trial, hh)-base)/eps
            rr.append((D.row[:, None]*width+np.arange(width)).ravel())
            cc.append(np.broadcast_to(D.col[:, None], (len(D.data), width)).ravel())
            vv.append((D.data[:, None]*derivative[D.row]).ravel())
        parts.append(sparse.csr_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                                      shape=(len(X)*width, len(motion.x))))
        return sparse.hstack(parts, format='csr')

    # The initializer has only hundreds of coordinates, with sparse local cubic
    # blocks plus 15 global head coordinates. Directly solve that small damped
    # normal system. LSMR on every trust-region subproblem was needlessly slow
    # on this nearly gauge-degenerate *initialization* problem. Production Ceres
    # /GPU factors and their statistical objective remain unchanged.
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    x = initial.copy()
    residual = fun(x)
    before = residual.copy()
    cost = float(residual@residual/2)
    damping = 1e-3
    nfev = 1
    accepted = 0
    message = ''
    success = False
    while True:
        J = jac(x).multiply(scales).tocsr()
        normal = (J.T@J).tocsc()
        gradient = np.asarray(J.T@residual).ravel()
        diagonal = np.maximum(normal.diagonal(), 1e-20)
        projected = gradient.copy()
        projected[(x <= lower+1e-12)&(projected > 0)] = 0.
        projected[(x >= upper-1e-12)&(projected < 0)] = 0.
        normalized_gradient = np.max(abs(projected)/np.sqrt(diagonal))/(1+np.linalg.norm(residual))
        if normalized_gradient < 1e-9:
            message = 'projected scaled gradient converged'
            success = True
            break
        while True:
            delta = spsolve(normal+sparse.diags(damping*diagonal), -gradient)
            if not np.isfinite(delta).all():
                damping *= 10.
                continue
            trial = np.clip(x+scales*delta, lower, upper)
            actual_step = (trial-x)/scales
            if np.max(abs(actual_step)) <= 1e-10*(1+np.max(abs(x/scales))):
                message = 'no further representable improving step in declared parameter box'
                success = normalized_gradient < 1e-6
                break
            candidate = fun(trial)
            nfev += 1
            trial_cost = float(candidate@candidate/2)
            predicted = float(-gradient@actual_step-.5*actual_step@(normal@actual_step))
            ratio = (cost-trial_cost)/predicted if predicted > 0 else -np.inf
            if np.isfinite(trial_cost) and trial_cost < cost and ratio > 0:
                improvement = cost-trial_cost
                old_cost = cost
                x, residual, cost = trial, candidate, trial_cost
                accepted += 1
                damping = max(1e-12, damping*(.3 if ratio > .75 else 1.))
                print(json.dumps(dict(seed_head_step=accepted, nfev=nfev, cost=cost,
                                      improvement=improvement, gain_ratio=ratio)), flush=True)
                if improvement <= 1e-8*(1+old_cost) and ratio > .25:
                    message = 'relative accepted screen cost converged'
                    success = True
                break
            damping *= 10.
        if message:
            break
    q, hh = unpack(x)
    motion.store(x[len(ids):])
    return hh, dict(cost_before=float(before@before/2), cost_after=cost, nfev=nfev,
                    accepted_steps=accepted, message=message, solver_success=success,
                    numerical_screen_fit_only=True,
                    initialization_linear_solver='direct_sparse_damped_normal_system')


def _initialize_head_from_poses(hb, h, meta, R, C, logf):
    """Recover the shared physical axes and moving optical center from H poses.

    The center is not collapsed into one median tripod location. P, lever,
    perpendicular offset and zoom pupil displacement retain their existing
    physical meanings. All observations here are explicitly numerical H poses.
    """
    from scipy.optimize import lsq_linear
    h = h.copy()
    optics = np.zeros(12)
    ids = np.asarray([j for j in range(6) if _free(hb, j)], int)
    def unpack(v):
        hh = h.copy()
        hh[ids] = v
        return hh
    def fun(v):
        hh = unpack(v)
        q = np.c_[_angles(R, hh, meta), logf]
        actual, _ = _physical(q, hh, optics, meta)
        return (actual-R).ravel()
    rotation_fit = least_squares(fun, h[ids],
                                 bounds=(np.asarray(hb['lower'])[ids], np.asarray(hb['upper'])[ids]),
                                 ftol=1e-12, xtol=1e-12, gtol=1e-12)
    h[ids] = rotation_fit.x
    _, n, b = _head(h, meta)
    q = np.c_[_angles(R, h, meta), logf]
    actual_R, _ = _physical(q, h, optics, meta)
    Rt = np.transpose(actual_R, (0, 2, 1))
    normal = np.cross(n, np.einsum('nij,j->ni', Rt, b))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    z = logf-np.log(meta['f_ref'])
    A = np.concatenate((np.broadcast_to(np.eye(3), (len(R), 3, 3)), Rt,
                        normal[:, :, None], Rt[:, :, 2:3]*z[:, None, None],
                        Rt[:, :, 2:3]*z[:, None, None]**2), axis=2)
    center_fit = lsq_linear(A.reshape(-1, 9), C.ravel(),
                            bounds=(hb['lower'][6:15], hb['upper'][6:15]), tol=1e-13)
    h[6:15] = center_fit.x
    _, actual_C = _physical(q, h, optics, meta)
    return h, dict(rotation_nfev=rotation_fit.nfev,
                    rotation_matrix_rms=float(np.sqrt(np.mean(rotation_fit.fun**2))),
                    optical_center_error_L=_stats(np.linalg.norm(actual_C-C, axis=1)),
                    center_rank=int(np.linalg.matrix_rank(A.reshape(-1, 9))),
                    actual_moving_optical_center_model=True, copied_historical_pose=False)


def build_native_seed_from_screen(model_file, screen_seed_file, template_audit_file, outdir, *,
                                  compact_reader_file=None):
    """Build a real 3-rig/9-shot model without copying historical camera poses.

    S01's scene, main head/optics and every existing S01 coordinate are held
    exactly. Other main shots use that one shared head. Therefore incompatibility
    is reported rather than hidden in a private fourth rig or second scene.
    Board/advertisements remain their native 3D surfaces and are never passed
    through the ice H. Those surfaces need source refinement after this seed.
    """
    start = time.perf_counter()
    out = Path(outdir).resolve()
    if out.exists():
        raise FileExistsError('Use a new immutable physical seed generation')
    model = read_json(model_file)
    parent = copy.deepcopy(model)
    screen_document = read_json(screen_seed_file)
    state = None
    if screen_document.get('schema') == 'rinkfit.screen_recovery_state.v1':
        from .screen_recovery_state import ScreenRecoveryState
        state = ScreenRecoveryState(screen_seed_file)
        seed = state.seed
        compact_reader_file = Path(__file__).with_name('screen_recovery_state.py')
    else:
        if compact_reader_file is None:
            compact_reader_file = Path(screen_seed_file).parent.parent/'tools/compact_screen_seed.py'
        spec = importlib.util.spec_from_file_location('_audited_compact_screen_reader', compact_reader_file)
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        seed = reader.ScreenSeed(screen_seed_file)
    allowed_navigation = np.asarray(seed.a['navigation_role_allowed'], dtype=bool)
    if set(model['rigs']) != {'main', 'secondary', 'end'} or set(model['shots']) != {f's{i:02d}' for i in range(1, 10)}:
        raise ValueError('Expected one scene, three rigs and nine shots')
    if seed.meta['source_video_sha256'] != model['source_clock']['source_video_sha256']:
        raise ValueError('Source video mismatch')
    T, curves, coordinates, chart = _chart_from_entities(model, template_audit_file)
    pts = seed.pts
    numbers = seed.shots
    H_reference = (np.stack([state.matrix(int(p), f's{int(n):02d}') for p, n in zip(pts, numbers)])
                   if state is not None else np.asarray(seed.a['H_reference_xy_to_native']))
    H = H_reference@T
    # Sensor-grid intersections are numerical witnesses, never source features.
    pixels = np.asarray([(u, v) for v in (180., 360., 540., 720., 900.)
                         for u in (180., 440., 700., 960., 1220., 1480., 1740.)])
    pixels = np.broadcast_to(pixels, (len(pts), len(pixels), 2)).copy()
    xy = _project(np.linalg.inv(H), pixels)
    scene = NativeScene(as_event_view(model, 's01'))
    mask = (scene.containment(xy) < -.001) & np.isfinite(xy).all(-1)
    original_den = np.einsum('fi,fki->fk', H[:, 2], np.concatenate((xy, np.ones(xy.shape[:-1]+(1,))), -1))
    mask &= seed.a['front_sign'][:, None]*original_den > 0
    pp = np.empty((len(pts), 2))
    principal_reports = {}
    for rid in ('main', 'secondary', 'end'):
        sids = [int(sid[1:]) for sid, s in model['shots'].items() if sid != 's01' and s['rig_id'] == rid]
        selected = np.isin(numbers, sids)
        allowed = selected & allowed_navigation
        value, principal_reports[rid] = _estimate_principal(H[allowed])
        pp[selected] = value
    R, C, logf, pin_uv, depth = _pinhole_from_H(H, xy, pp)
    mask &= depth > 0
    if np.any(mask.sum(1) < 4):
        raise ValueError('A frame lacks four finite supported plane witnesses; cannot initialize a successful subset')
    X = np.concatenate((xy, np.zeros(xy.shape[:-1]+(1,))), -1)
    pin_error = np.linalg.norm(pin_uv-pixels, axis=-1)
    template_error = np.full(len(pts), np.nan)
    template_sum = np.zeros(len(pts))
    template_count = np.zeros(len(pts), np.int64)
    # Batched by entity: no full UV cache and no 300,000 Python frame calls.
    for curve, current in zip(curves, coordinates):
        transformed = _project(T, current)
        nearest = _closest_reference(curve, transformed)
        uv = _project(H_reference, transformed)
        other = _project(H_reference, nearest)
        denominator = H_reference[:, 2]@np.c_[transformed, np.ones(len(transformed))].T
        visible = ((uv[..., 0] >= 0)&(uv[..., 0] < 1920)&(uv[..., 1] >= 0)&(uv[..., 1] < 1080)
                   &np.isfinite(uv).all(-1)&np.isfinite(other).all(-1)
                   &(seed.a['front_sign'][:, None]*denominator > 0))
        error2 = np.sum((uv-other)**2, axis=-1)
        template_sum += np.sum(np.where(visible, error2, 0.), axis=1)
        template_count += visible.sum(1)
    known_template = template_count > 0
    template_error[known_template] = np.sqrt(template_sum[known_template]/template_count[known_template])
    protected = {model['scene_block'], model['rigs']['main']['head_block'], model['rigs']['main']['optics_block']}
    protected.update(model['shots']['s01']['control_blocks'])
    protected.add(model['shots']['s01']['reference_angles_block'])
    protected.update(c['block'] for c in model['shots']['s01'].get('control_coordinates', []))
    protected_sha = object_sha256([(i, model['blocks'][i]) for i in sorted(protected)])
    reports = {}
    out.mkdir(parents=True)
    for rid in ('secondary', 'end', 'main'):
        sids = [sid for sid, shot in model['shots'].items() if sid != 's01' and shot['rig_id'] == rid]
        select = np.isin(numbers, [int(s[1:]) for s in sids])
        fit = select & allowed_navigation
        rig = model['rigs'][rid]
        hb = model['blocks'][rig['head_block']]
        ob = model['blocks'][rig['optics_block']]
        h = np.asarray(hb['values']).copy()
        o = np.asarray(ob['values']).copy()
        meta = rig['camera']
        pose_head_report = None
        if rid != 'main':
            # New provisional physical calibration from H, not from old K/R/C.
            idx = np.flatnonzero(fit)
            first = idx[numbers[idx] == int(rig['reference_shot'][1:])][0]
            meta['R0_seed'] = R[first].reshape(-1).tolist()
            meta['n_seed'] = [0., 0., 1.]
            meta['n_tangent1'] = [0., 1., 0.]
            meta['n_tangent2'] = [-1., 0., 0.]
            axis = np.array([1., 0., 0.])
            normal = R[first]@np.array(meta['n_seed'])
            axis -= normal*(axis@normal)
            axis /= np.linalg.norm(axis)
            meta['b_seed'] = axis.tolist()
            h[:] = 0.
            h[6:9] = np.median(C[fit], axis=0)
            # Pinhole is an explicitly provisional initial value. The Brown
            # coefficients remain physical variables for real source refinement.
            o[:11] = list(pp[first])+[0., 0., 0., 0., 0., 0., 0., 0., 0.]
            h, pose_head_report = _initialize_head_from_poses(hb, h, meta, R[fit], C[fit], logf[fit])
            if np.any(h < hb['lower']) or np.any(h > hb['upper']) or np.any(o < ob['lower']) or np.any(o > ob['upper']):
                raise ValueError('Source-derived provisional calibration is outside declared physical parameter box')
            ob['values'] = o.tolist()
        motion = Motion(model, sids, numbers[fit], pts[fit])
        motion.head_block = rig['head_block']
        targetq = np.c_[_angles(R[fit], h, meta), logf[fit]]
        for sid in sids:
            subset = numbers[fit] == int(sid[1:])
            targetq[subset, :2] = np.unwrap(targetq[subset, :2], axis=0)
        static = [j for j in range(15) if rid != 'main' and _free(hb, j)]
        h, fit_report = _shared_fit(motion, h, o, meta, X[fit], pixels[fit], mask[fit], targetq, static)
        hb['values'] = h.tolist()
        full_motion = Motion(model, sids, numbers[select], pts[select])
        q = full_motion.values(full_motion.x)
        uv, dep, radius, *_ = _physical(q, h, o, meta, X[select])
        error = np.linalg.norm(uv-pixels[select], axis=-1)
        reports[rid] = dict(shots=sids, fit=fit_report, pose_head_initialization=pose_head_report,
                            all_native_frames=int(select.sum()),
                            source_truth_created=False, provisional_pinhole_optics=(rid != 'main'),
                            old_main_full_dependency_preserved=(rid == 'main'),
                            screen_transfer_px=_stats(error[mask[select]]),
                            focal_native_px=_stats(np.exp(q[:, 2])),
                            ray_radius=_stats(radius[mask[select]]),
                            negative_depth=int((mask[select] & (dep <= 0)).sum()),
                            outside_declared_ray_radius=int((mask[select] & (radius > meta['radius_max'])).sum()))
        write_json(out/(rid+'_fit.json'), reports[rid])
        print(json.dumps(dict(rig=rid, report=reports[rid])), flush=True)
    if object_sha256([(i, model['blocks'][i]) for i in sorted(protected)]) != protected_sha:
        raise ValueError('Protected S01 physical dependency changed')
    if parent['materials'] != model['materials'] or parent['scene_block'] != model['scene_block']:
        raise ValueError('Ice reconciliation changed shared scene/material data')
    for block in model['blocks']:
        value = np.asarray(block['values'])
        if not np.isfinite(value).all() or np.any(value < block['lower']) or np.any(value > block['upper']):
            raise ValueError('Seed parameter outside declared domain: '+block['name'])
    full_error = np.full(mask.shape, np.nan)
    rs_error = np.full(mask.shape, np.nan)
    invalid = np.zeros(mask.shape, bool)
    production_delta = 0.
    for sid in sorted(model['shots']):
        if sid == 's01':
            continue
        selected = numbers == int(sid[1:])
        shot = model['shots'][sid]
        rig = model['rigs'][shot['rig_id']]
        motion = Motion(model, [sid], numbers[selected], pts[selected])
        q = motion.values(motion.x)
        h = np.asarray(model['blocks'][rig['head_block']]['values'])
        o = np.asarray(model['blocks'][rig['optics_block']]['values'])
        predicted, *_ = _physical(q, h, o, rig['camera'], X[selected])
        cam = camera_from_event(as_event_view(model, sid))
        from rinkfit_start.native_camera import rotation
        angles = model['blocks'][shot['reference_angles_block']]['values']
        cam.R0 = rotation(cam.b, angles[1])@cam.R0@rotation(cam.n, angles[0])
        t = (pts[selected]-shot['time_origin_pts'])/12800.
        native = cam.at(X[selected], t[:, None])
        production_delta = max(production_delta, float(np.max(abs(native['uv']-predicted))))
        replay = cam.project_rs(X[selected], t[:, None])
        full_error[selected] = np.linalg.norm(predicted-pixels[selected], axis=-1)
        rs_error[selected] = np.linalg.norm(replay['uv']-pixels[selected], axis=-1)
        invalid[selected] = ~replay['valid']
    if production_delta > 2e-6:
        raise ValueError('Actual production NumPy forward disagrees with seed fitting evaluator')
    per_shot = []
    for sn in range(2, 10):
        selected = numbers == sn
        mm = mask[selected]
        per_shot.append(dict(shot=f's{sn:02d}', frames=int(selected.sum()),
                             template_conversion_screen_px=_stats(template_error[selected]),
                             H_to_provisional_pinhole_px=_stats(pin_error[selected][mm]),
                             pose_to_Brown_shared_head_midtime_px=_stats(full_error[selected][mm]),
                             pose_to_Brown_RS_shared_head_px=_stats(rs_error[selected][mm]),
                             witness_denominator=int(mm.sum()), invalid_domain_RS=int((mm & invalid[selected]).sum()),
                             estimated_K_focal_range_px=[float(np.exp(logf[selected]).min()), float(np.exp(logf[selected]).max())]))
    provenance = dict(schema='rinkfit.screen_to_native_physical_seed.v1', accepted=False,
                      input_model=dict(path=str(Path(model_file).resolve()), sha256=file_sha256(model_file)),
                      screen_seed=dict(path=str(Path(screen_seed_file).resolve()), sha256=file_sha256(screen_seed_file)),
                      screen_provider_consumed=(state is not None),
                      supported_screen_corrections_consumed=bool(state is not None and state.corrections),
                      compact_reader=dict(path=str(Path(compact_reader_file).resolve()), sha256=file_sha256(compact_reader_file)),
                      historical_K_R_C_imported=False, source_likelihood_modified=False,
                      screen_witnesses_are_source_observations=False, source_refinement_required=True,
                      S01_complete_dependency_sha256=protected_sha, scene_count=1, rig_count=3, shot_count=9,
                      provisional_K_assumptions=dict(principal_points=principal_reports, aspect=1., skew=0.,
                                                     f='least_squares_H_column_orthogonality', hardware_calibration_claimed=False),
                      provisional_secondary_end_Brown='zero initial values only; uncalibrated and free for source refinement',
                      main_wide_optics_extension_completed=False,
                      ice_H_used_for_board_or_advertisement=False,
                      unresolved=['No source-calibrated cross-template equivalence',
                                  'Main wide Brown extrapolation remains to be replaced with fully consumed optical representation',
                                  'Board/advertisement 3D source refinement not performed by planar initialization',
                                  'Shared main state still requires source-evidenced unlock including S01 regression',
                                  'Physical source-pixel and static-tracklet refinement required'])
    model['screen_to_native_seed'] = provenance
    write_json(out/'model.json', model)
    np.savez_compressed(out/'transfer.npz', native_pts=pts, shot_number=numbers,
                        provisional_R=R, provisional_C=C, provisional_logf=logf,
                        witness_mask=mask, template_conversion_rms_px=template_error,
                        H_to_pinhole_error_px=pin_error,
                        physical_midtime_error_px=full_error, physical_RS_error_px=rs_error,
                        invalid_domain_RS=invalid)
    report = dict(schema='rinkfit.screen_to_native_physical_seed_report.v1', seconds=time.perf_counter()-start,
                  model=dict(path=str(out/'model.json'), sha256=file_sha256(out/'model.json')),
                  physical_seed_generated=True, physical_fit_accepted=False,
                  template=chart, per_shot=per_shot, rigs=reports,
                  source_accuracy_claimed=False, measurements_created=0,
                  independent_production_midtime_max_abs_px=production_delta, provenance=provenance)
    write_json(out/'report.json', report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--screen-seed', required=True)
    p.add_argument('--template-audit', required=True)
    p.add_argument('--compact-reader')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    report = build_native_seed_from_screen(a.model, a.screen_seed, a.template_audit, a.out,
                                           compact_reader_file=a.compact_reader)
    print(json.dumps(dict(model=report['model'], per_shot=report['per_shot'])), flush=True)


if __name__ == '__main__':
    main()
