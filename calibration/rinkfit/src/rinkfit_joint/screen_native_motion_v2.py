"""Independent motion-only screen lift with exact cardinal refinement.

Screen witnesses remain numerical navigation. Only source-role allowed frames
enter fitting; protected frames are replayed. Existing head, Brown/wide, readout,
scene and S01 are immutable. No packet, checkpoint or source precision is issued.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import spsolve
from .io_utils import read_json, write_json, file_sha256
from .compact_screen_seed import ScreenSeed
from .screen_to_native_seed import _project, _chart_from_entities
from .native_baseline_init import Motion, _physical
from .joint_model import as_event_view
from rinkfit_start.native_scene import camera_from_event, NativeScene
from rinkfit_start.native_camera import rotation
from rinkfit_start.camera_spline_chart import cardinal_refinement, cubic_design

EPS = np.finfo(float).eps
TOL = np.sqrt(EPS)


def _camera(model, sid):
    shot = model['shots'][sid]
    camera = camera_from_event(as_event_view(model, sid))
    ref = model['blocks'][shot['reference_angles_block']]['values']
    camera.R0 = rotation(camera.b, ref[1]) @ camera.R0 @ rotation(camera.n, ref[0])
    return camera


def _stats(x, valid=None):
    a = np.asarray(x, float).ravel()
    good = np.isfinite(a)
    if valid is not None:
        good &= np.asarray(valid, bool).ravel()
    z = a[good]
    return dict(denominator=len(a), valid=int(good.sum()), invalid=int((~good).sum()),
                full_rms=float(np.sqrt(np.mean(a*a))) if good.all() and len(a) else None,
                finite_diagnostic_rms=float(np.sqrt(np.mean(z*z))) if len(z) else None,
                p95=float(np.percentile(z, 95)) if len(z) else None,
                max=float(z.max()) if len(z) else None)


def _frame_forward(q, h, o, meta, world, row_delta):
    shape = world.shape[:-1]
    rows = (q[:, None, :]+row_delta).reshape(-1, 3)
    uv, depth, radius, *_ = _physical(rows, h, o, meta, world.reshape(-1, 1, 3))
    return uv.reshape(shape+(2,)), ((depth[:, 0] > 0) &
        (radius[:, 0] <= float(meta['radius_max'])) &
        np.isfinite(uv[:, 0]).all(-1)).reshape(shape)


def fit_frame_coordinates(h, o, meta, world, target_uv, initial_q, *, mask=None,
                          row_delta=None, lower=None, upper=None):
    """Batched independent 3q GN diagnostic/teacher, starting from actual q.

    row_delta optionally preserves the original motion's target-row-time offset
    from frame center. It is fixed during this diagnostic, not a private state
    for each sensor row. This function neither fits protected frames implicitly
    nor creates a physical trajectory. The caller supplies its qualified scope.
    """
    X = np.asarray(world, float)
    target = np.asarray(target_uv, float)
    q = np.asarray(initial_q, float).copy()
    if X.ndim != 3 or X.shape[-1] != 3 or target.shape != X.shape[:-1]+(2,) or q.shape != (len(X), 3):
        raise ValueError('Expected F x K worlds/UV and F x 3 actual initial q')
    m = np.ones(X.shape[:-1], bool) if mask is None else np.asarray(mask, bool)
    delta = np.zeros(X.shape) if row_delta is None else np.asarray(row_delta, float)
    if delta.shape != X.shape or np.any(m.sum(1) < 2):
        raise ValueError('Each supplied frame needs finite navigation support')
    lo = np.broadcast_to([-20., -20., np.log(100.)] if lower is None else lower, q.shape)
    hi = np.broadcast_to([20., 20., np.log(1e6)] if upper is None else upper, q.shape)
    if np.any(q < lo) or np.any(q > hi):
        raise ValueError('Actual initial q lies outside supplied original coordinate box')
    uv, valid = _frame_forward(q, h, o, meta, X, delta)
    if np.any(m & ~valid):
        raise ValueError('Frame GN cannot silently remove an invalid original witness')
    residual = np.where(m[..., None], uv-target, 0.)
    initial_error = np.linalg.norm(uv-target, axis=-1)
    cost = np.sum(residual*residual, axis=(1, 2))/2
    damping = np.full(len(q), 1e-3)
    done = np.zeros(len(q), bool)
    accepted = np.zeros(len(q), int)
    messages = np.full(len(q), '', object)
    scales = np.array([.1, .1, 1.])
    while not done.all():
        ids = np.flatnonzero(~done)
        qq = q[ids]
        base = residual[ids]
        J = np.empty(base.shape+(3,))
        for axis in range(3):
            step = np.cbrt(EPS)*(1+abs(qq[:, axis]))
            plus, minus = qq.copy(), qq.copy()
            plus[:, axis] += step
            minus[:, axis] -= step
            pu, _ = _frame_forward(plus, h, o, meta, X[ids], delta[ids])
            mu, _ = _frame_forward(minus, h, o, meta, X[ids], delta[ids])
            J[..., axis] = np.where(m[ids, ..., None], (pu-mu)/(2*step[:, None, None]), 0.)*scales[axis]
        jj = J.reshape(len(ids), -1, 3)
        rr = base.reshape(len(ids), -1)
        normal = np.einsum('fki,fkj->fij', jj, jj)
        gradient = np.einsum('fki,fk->fi', jj, rr)
        diagonal = np.maximum(np.diagonal(normal, axis1=1, axis2=2), np.finfo(float).tiny)
        pg = gradient.copy()
        pg[(qq <= lo[ids]+EPS*(1+abs(qq))) & (pg > 0)] = 0.
        pg[(qq >= hi[ids]-EPS*(1+abs(qq))) & (pg < 0)] = 0.
        optimality = np.max(abs(pg)/np.sqrt(diagonal), axis=1)/(1+np.linalg.norm(rr, axis=1))
        converged = optimality <= TOL
        done[ids[converged]] = True
        messages[ids[converged]] = 'scaled projected gradient'
        active = ~converged
        if not active.any():
            continue
        ids, qq = ids[active], qq[active]
        normal, gradient, diagonal = normal[active], gradient[active], diagonal[active]
        trial_normal = normal.copy()
        trial_normal[:, np.arange(3), np.arange(3)] += damping[ids, None]*diagonal
        step = np.linalg.solve(trial_normal, -gradient[..., None])[..., 0]
        trial = np.clip(qq+step*scales, lo[ids], hi[ids])
        actual = (trial-qq)/scales
        tiny = np.max(abs(actual), axis=1) <= TOL*(1+np.max(abs(qq/scales), axis=1))
        tu, tv = _frame_forward(trial, h, o, meta, X[ids], delta[ids])
        tr = np.where(m[ids, ..., None], tu-target[ids], 0.)
        tc = np.sum(tr*tr, axis=(1, 2))/2
        legal = np.all(tv | ~m[ids], axis=1) & np.isfinite(tc)
        improve = legal & (tc < cost[ids])
        before = cost[ids].copy()
        q[ids[improve]] = trial[improve]
        residual[ids[improve]] = tr[improve]
        cost[ids[improve]] = tc[improve]
        accepted[ids[improve]] += 1
        damping[ids[improve]] = np.maximum(EPS, damping[ids[improve]]*.3)
        damping[ids[~improve]] *= 10.
        small_gain = improve & ((before-tc) <= TOL*(1+before))
        stopped = tiny | small_gain | ~np.isfinite(damping[ids])
        done[ids[stopped]] = True
        messages[ids[stopped]] = 'representable step or relative objective limit'
    uv, valid = _frame_forward(q, h, o, meta, X, delta)
    error = np.linalg.norm(uv-target, axis=-1)
    return q, dict(frame_count=len(q), before=_stats(initial_error[m]), after=_stats(error[m], valid[m]),
                   accepted_steps=accepted.tolist(), stop_messages=messages.tolist(),
                   row_motion_offsets_fixed=row_delta is not None,
                   source_observations_created=False, candidate_trajectory_created=False)


def refine_shot_exact(model, sid, multiple):
    """Append one new native uniform storage with identical old cubic function.

    cardinal_refinement is reused in its fine-from-coarse direction. The fine
    domain endpoints equal the complete old allocated endpoints exactly; neither
    readout halo is trimmed. Existing physical/reference/gauge values survive.
    """
    if sid == 's01':
        raise ValueError('S01 refinement is outside this motion-only initializer')
    out = copy.deepcopy(model)
    shot = out['shots'][sid]
    if shot.get('control_coordinates'):
        raise ValueError('Existing packed chart needs an explicit matching lift')
    old = copy.deepcopy(shot['camera'])
    m = int(multiple)
    if m != multiple or m < 2:
        raise ValueError('Integer cardinal subdivision required')
    first, dt, n = old['first_control_time'], old['dt'], old['n_controls']
    fine_dt = dt/m
    fine_first = first+dt-fine_dt
    fine_n = m*(n-3)+3
    R, coarse = cardinal_refinement(fine_first, fine_dt, fine_n, m)
    if R.shape != (fine_n, n) or abs(coarse['first_control_time']-first) > 1e-12:
        raise ValueError('Cardinal endpoint/gauge lattice mismatch')
    original = camera_from_event(as_event_view(model, sid)).controls
    fine = R@original
    gauge = cubic_design([0.], fine_first, fine_dt, fine_n).toarray()[0]
    inds = np.flatnonzero(gauge)
    dep = int(np.argmax(abs(gauge)))
    if np.max(abs(gauge@fine[:, :2])) > 2e-10:
        raise ValueError('Refinement did not preserve angle function gauge')
    oldids = shot['control_blocks']
    exemplar = model['blocks'][oldids[0]]
    for bi in oldids:
        block = model['blocks'][bi]
        if any(block[k] != exemplar[k] for k in ('lower', 'upper', 'scales')):
            raise ValueError('Heterogeneous original control box needs explicit mapping')
        out['blocks'][bi]['constant'] = True
    ids = []
    for j, value in enumerate(fine):
        block = copy.deepcopy(exemplar)
        block.update(name=f'shot/{sid}/motion_v2/{fine_dt:.17g}/{j}',
                     values=value.tolist(), constant=False, fixed_indices=[0, 1] if j == dep else [])
        if j == dep:
            block['values'][:2] = [0., 0.]
        if np.any(np.array(block['values']) < block['lower']) or np.any(np.array(block['values']) > block['upper']):
            raise ValueError('Refined native storage exceeds original coordinate box')
        ids.append(len(out['blocks']))
        out['blocks'].append(block)
    shot['control_blocks'] = ids
    shot['camera'].update(first_control_time=fine_first, dt=fine_dt, n_controls=fine_n,
                          dependent_control=dep, reference_indices=inds.tolist(), reference_weights=gauge[inds].tolist())
    actual = camera_from_event(as_event_view(out, sid)).controls
    coefficient_error = float(np.max(abs(actual-fine)))
    if coefficient_error > 2e-10:
        raise ValueError('Stored dependent-control expansion differs from refinement')
    # Cubic equality is checked at four distinct points of every new cell;
    # value equality there identifies the polynomial. Derivatives/RS halo follow.
    edges = fine_first+np.arange(1, fine_n-1)*fine_dt
    t = (edges[:-1, None]+np.array([.125, .375, .625, .875])[None, :]*fine_dt).ravel()
    a = cubic_design(t, first, dt, n)@original
    b = cubic_design(t, fine_first, fine_dt, fine_n)@actual
    err = float(np.max(abs(a-b)))
    if err > 2e-10:
        raise ValueError('Whole original-domain cubic equivalence failed')
    return out, dict(shot=sid, multiple=m, original_camera=old, new_camera=copy.deepcopy(shot['camera']),
                     old_control_blocks_frozen_unused=oldids, new_control_blocks=ids,
                     allocated_domain_before=[first+dt, first+(n-2)*dt],
                     allocated_domain_after=[fine_first+fine_dt, fine_first+(fine_n-2)*fine_dt],
                     entire_domain_polynomial_check_points=len(t), max_q_function_difference=err,
                     max_expanded_coefficient_error=coefficient_error, both_RS_halos_retained=True,
                     primitive_and_GPU_ABI_unchanged=True)


def _linear_teacher(motion, target, permitted=None):
    x = motion.x.copy()
    reports = []
    for axis, D in enumerate(motion.D):
        cols = np.flatnonzero(motion.axis == axis)
        A = D[:, cols]
        supported = np.asarray(abs(A).sum(axis=0)).ravel() > 0
        if permitted is not None:
            supported &= permitted[cols]
        cols, A = cols[supported], A[:, supported]
        current = motion.constant[:, axis]+D@x
        if not len(cols):
            reports.append(dict(axis=axis, supported_coordinates=0, unsupported_coordinates_unchanged=int((~supported).sum())))
            continue
        result = lsq_linear(A, target[:, axis]-current,
            bounds=(motion.lo[cols]-x[cols], motion.hi[cols]-x[cols]), tol=TOL, lsmr_tol=EPS**.75)
        x[cols] += result.x
        reports.append(dict(axis=axis, cost=float(result.cost), optimality=float(result.optimality),
                            unsupported_coordinates_unchanged=int((~supported).sum()), success=bool(result.success)))
    return x, reports


def _coordinate_support(model, sids, refs, intervals):
    """Only increments whose complete cardinal support lies in real allowed runs.

    This is a source-role support rule, not a smoothness/SV prior. Unsupported
    intervals and every protected exposure's entire allowed RS halo retain the
    previous function. Gauge-elimination spill into the dependent basis is also
    included. Reference-angle shifts have global support and remain unchanged.
    """
    lookup = {}
    details = {}
    for sid in sids:
        shot = model['shots'][sid]; cam = shot['camera']
        first, dt = cam['first_control_time'], cam['dt']
        gauge = dict(zip(cam['reference_indices'], cam['reference_weights']))
        dep = cam['dependent_control']
        def inside(j):
            support = (first+(j-2)*dt, first+(j+2)*dt)
            # Strict inclusion keeps zero-support boundary roundoff from
            # leaking into a protected sensor interval.
            return any(support[0] >= a and support[1] <= b for a,b in intervals[sid])
        for j,bi in enumerate(shot['control_blocks']):
            local = inside(j)
            for axis in range(3):
                lookup[(bi,axis)] = local and (axis == 2 or not gauge.get(j,0.) or inside(dep))
        for axis in range(2):
            lookup[(shot['reference_angles_block'],axis)] = False
        details[sid] = dict(allowed_continuous_source_intervals=intervals[sid],
            full_basis_support_seconds=4*dt, reference_angle_global_increment_frozen=True)
    return np.array([lookup.get(tuple(r), False) for r in refs]), details


def _fit_row_motion(model, sids, numbers, row_pts, X, target, teacher, intervals):
    motion = Motion(model, sids, numbers, row_pts)
    if not len(motion.x):
        raise ValueError('No existing free motion coordinates')
    supported = np.zeros(len(motion.x), bool)
    for D in motion.D:
        supported |= np.asarray(abs(D).sum(axis=0)).ravel() > 0
    permitted, support_report = _coordinate_support(model, sids, motion.refs, intervals)
    supported &= permitted
    active = np.flatnonzero(supported)
    if not len(active):
        rig = model['rigs'][model['shots'][sids[0]]['rig_id']]
        h = np.array(model['blocks'][rig['head_block']]['values']); o = np.array(model['blocks'][rig['optics_block']]['values'])
        q = motion.values(motion.x); uv, dep, rad, *_ = _physical(q,h,o,rig['camera'],X[:,None,:])
        valid = (dep[:,0]>0)&(rad[:,0]<=rig['camera']['radius_max'])&np.isfinite(uv[:,0]).all(-1)
        if not valid.all():
            raise ValueError('Original fitting rows invalid at immutable supported scope')
        cost = float(np.sum((uv[:,0]-target)**2)/2)
        return dict(cost_before=cost,cost_after=cost,accepted_steps=0,nfev=1,stop='No complete basis support inside allowed runs at this spacing',
            all_original_fitting_rows_retained=len(X),unsupported_coordinates_unchanged=len(motion.x),source_role_support=support_report)
    if teacher is None:
        x, linear = motion.x.copy(), []
    else:
        x, linear = _linear_teacher(motion, teacher, permitted=supported)
    rig = model['rigs'][model['shots'][sids[0]]['rig_id']]
    h = np.array(model['blocks'][rig['head_block']]['values'])
    o = np.array(model['blocks'][rig['optics_block']]['values'])
    meta = rig['camera']
    D = [d[:, active].tocsr() for d in motion.D]
    scales, lo, hi = motion.scale[active], motion.lo[active], motion.hi[active]
    fixed = motion.constant+np.column_stack([d@x for d in motion.D])
    origin = x[active].copy()
    def state(v):
        return fixed+np.column_stack([d@(v-origin) for d in D])
    def forward(v):
        q = state(v)
        uv, dep, rad, *_ = _physical(q, h, o, meta, X[:, None, :])
        uv = uv[:, 0]
        valid = (dep[:, 0] > 0) & (rad[:, 0] <= meta['radius_max']) & np.isfinite(uv).all(-1)
        return q, uv-target, valid
    v = origin.copy()
    q, residual, valid = forward(v)
    original_x = motion.x.copy()
    original_q = motion.values(original_x)
    original_uv, original_dep, original_radius, *_ = _physical(original_q, h, o, meta, X[:, None, :])
    original_valid = (original_dep[:,0]>0)&(original_radius[:,0]<=meta['radius_max'])&np.isfinite(original_uv[:,0]).all(-1)
    original_cost = float(np.sum((original_uv[:,0]-target)**2)/2) if original_valid.all() else np.inf
    teacher_cost = float(np.sum(residual*residual)/2) if valid.all() else np.inf
    if teacher_cost > original_cost or not valid.all():
        # A teacher never authorizes worsening or invalidating the old function.
        x = original_x; v = x[active].copy(); origin = v.copy()
        fixed = motion.values(x)
        q, residual, valid = forward(v)
    if not valid.all():
        raise ValueError('Original row-time fitting scope has invalid witnesses; no subset fitting')
    cost = float(np.sum(residual*residual)/2)
    before = cost
    damping = 1e-3
    accepted, nfev = 0, 1
    while True:
        parts = []
        for axis in range(3):
            step = np.cbrt(EPS)*(1+abs(q[:, axis]))
            plus, minus = q.copy(), q.copy()
            plus[:, axis] += step; minus[:, axis] -= step
            pu, *_ = _physical(plus, h, o, meta, X[:, None, :])
            mu, *_ = _physical(minus, h, o, meta, X[:, None, :])
            deriv = (pu[:, 0]-mu[:, 0])/(2*step[:, None])
            parts.append(sparse.vstack((sparse.diags(deriv[:, 0])@D[axis], sparse.diags(deriv[:, 1])@D[axis]), format='csr'))
        J = sum(parts).multiply(scales).tocsr()
        rr = np.r_[residual[:, 0], residual[:, 1]]
        normal = (J.T@J).tocsc()
        gradient = np.asarray(J.T@rr).ravel()
        diagonal = np.maximum(normal.diagonal(), np.finfo(float).tiny)
        pg = gradient.copy()
        pg[(v <= lo+EPS*(1+abs(v))) & (pg > 0)] = 0.
        pg[(v >= hi-EPS*(1+abs(v))) & (pg < 0)] = 0.
        optimality = float(np.max(abs(pg)/np.sqrt(diagonal))/(1+np.linalg.norm(rr)))
        if optimality <= TOL:
            message = 'scaled projected gradient'; break
        while True:
            step = spsolve(normal+sparse.diags(damping*diagonal), -gradient)
            if not np.isfinite(step).all():
                damping *= 10.; continue
            trial = np.clip(v+scales*step, lo, hi)
            actual = (trial-v)/scales
            if np.max(abs(actual)) <= TOL*(1+np.max(abs(v/scales))):
                message = 'representable step limit'; break
            tq, tr, tv = forward(trial); nfev += 1
            tc = float(np.sum(tr*tr)/2) if tv.all() else np.inf
            predicted = float(-gradient@actual-.5*actual@(normal@actual))
            ratio = (cost-tc)/predicted if predicted > 0 else -np.inf
            if np.isfinite(tc) and tc < cost and ratio > 0:
                improvement = cost-tc; previous = cost
                v, q, residual, cost = trial, tq, tr, tc
                accepted += 1
                damping = max(EPS, damping*(.3 if ratio > .75 else 1.))
                if improvement <= TOL*(1+previous):
                    message = 'relative navigation objective limit'
                else:
                    message = ''
                break
            damping *= 10.
        if message:
            break
    final = motion.x.copy()
    final[active] = v
    motion.store(final)
    return dict(cost_before=before, cost_after=cost, accepted_steps=accepted, nfev=nfev,
                optimality=optimality, stop=message, linear_teacher=linear,
                unsupported_coordinates_unchanged=int((~supported).sum()), source_role_support=support_report,
                source_row_time_in_objective=True, all_original_fitting_rows_retained=len(X))


def _evaluate(model, pts, numbers, X, target, mask, allowed, sids):
    report, arrays = {}, {}
    for sid in sids:
        select = numbers == int(sid[1:]); m = mask[select]
        camera = _camera(model, sid); shot = model['shots'][sid]
        t = (pts[select]-shot['time_origin_pts'])/12800.
        rows = t[:, None]+camera.tau*(target[select, :, 1]/1079.-.5)
        at = camera.at(X[select], rows)
        rs = camera.project_rs(X[select], t[:, None])
        error = np.linalg.norm(at['uv']-target[select], axis=-1)
        re = np.linalg.norm(rs['uv']-target[select], axis=-1)
        nav = allowed[select]
        unique_pts, frame_index = np.unique(pts[select], return_inverse=True)
        numerator = np.bincount(frame_index, weights=np.sum(np.where(m, re*re, 0.), axis=1), minlength=len(unique_pts))
        denominator = np.bincount(frame_index, weights=m.sum(1), minlength=len(unique_pts))
        frame = np.sqrt(numerator/denominator)
        width = min(10, len(frame)); roll = np.convolve(frame*frame, np.ones(width), 'valid')/width
        j = int(np.argmax(roll))
        worst_records = (frame_index>=j)&(frame_index<j+width)
        report[sid] = dict(frames=len(unique_pts), numerical_record_rows=int(select.sum()), row_time_all=_stats(error[m], at['valid'][m]),
            row_time_navigation=_stats(error[nav][m[nav]], at['valid'][nav][m[nav]]),
            row_time_protected=_stats(error[~nav][m[~nav]], at['valid'][~nav][m[~nav]]),
            trueRS_all=_stats(re[m], rs['valid'][m]),
            trueRS_navigation=_stats(re[nav][m[nav]], rs['valid'][nav][m[nav]]),
            trueRS_protected=_stats(re[~nav][m[~nav]], rs['valid'][~nav][m[~nav]]),
            worst10=dict(first_pts=int(unique_pts[j]), last_pts=int(unique_pts[j+width-1]),
                finite_diagnostic_rms=float(np.sqrt(roll[j])),
                invalid=int((m[worst_records] & ~rs['valid'][worst_records]).sum())),
            optical_domain=camera.optical_domain_certificate())
        arrays[sid+'_row_uv'] = at['uv']; arrays[sid+'_row_valid'] = at['valid']
        arrays[sid+'_RS_uv'] = rs['uv']; arrays[sid+'_RS_valid'] = rs['valid']
        arrays[sid+'_RS_root_equality'] = rs['root_equality_px']
    return report, arrays


def build_motion_candidate(model_file, screen_seed_file, transfer_file, report_file, outdir, *,
                           rigs=('secondary', 'end'), template_audit_file=None, external_witness_npz=None):
    out = Path(outdir).resolve()
    if out.exists():
        raise FileExistsError('Use a new immutable motion candidate directory')
    original = read_json(model_file); model = copy.deepcopy(original)
    seed = ScreenSeed(screen_seed_file); transfer = np.load(transfer_file, allow_pickle=False)
    previous = read_json(report_file)
    pts, numbers = seed.pts, seed.shots
    if not np.array_equal(pts, transfer['native_pts']) or not np.array_equal(numbers, transfer['shot_number']):
        raise ValueError('Exact native PTS/shot transfer identity required')
    allowed = seed.a['navigation_role_allowed'].astype(bool)
    chart = None
    surface_ids = None
    if external_witness_npz is not None:
        witness = np.load(external_witness_npz, allow_pickle=False)
        def field(*names):
            for name in names:
                if name in witness:
                    return witness[name]
            raise ValueError('External numerical witness field missing: '+str(names))
        if 'row_frame_index' in witness and 'row_world_index' in witness:
            frame_pts = np.asarray(field('native_pts', 'pts'))
            frame_numbers = np.asarray(field('shot_number', 'numbers'))
            fi = np.asarray(field('row_frame_index'), np.int64)
            wi = np.asarray(field('row_world_index'), np.int64)
            worlds = np.asarray(field('world_xyz'), float)
            if np.any(fi<0) or np.any(fi>=len(frame_pts)) or np.any(wi<0) or np.any(wi>=len(worlds)):
                raise ValueError('Invalid de-duplicated numerical navigation index')
            pts = frame_pts[fi]
            numbers = frame_numbers[fi]
            X = worlds[wi, None, :]
            target = np.asarray(field('target_uv'), float)[:, None, :]
            mask = np.ones(X.shape[:-1], bool)
            requested = np.asarray(field('row_navigation_allowed'), bool)
            surface_ids = np.asarray(field('row_entity'))
        else:
            pts = np.asarray(field('native_pts', 'pts'))
            numbers = np.asarray(field('shot_number', 'numbers'))
            X = np.asarray(field('world'), float)
            target = np.asarray(field('target_uv'), float)
            mask = np.asarray(field('witness_mask', 'mask'), bool)
            requested = np.asarray(field('navigation_role_allowed', 'allowedrole'), bool)
            surface_ids = np.asarray(field('surface_id'))
        if pts.dtype.kind not in 'iu' or X.ndim != 3 or X.shape[-1] != 3 or target.shape != X.shape[:-1]+(2,) or mask.shape != X.shape[:-1]:
            raise ValueError('External witnesses require native integer PTS, N x K x 3/2 worlds/UV and matching mask')
        if len(pts) != len(X) or numbers.shape != pts.shape or requested.shape != pts.shape:
            raise ValueError('External frame scope shape mismatch')
        rolemap = {int(p):(int(n), bool(a)) for p,n,a in zip(seed.pts, seed.shots, seed.a['navigation_role_allowed'])}
        allowed = np.zeros(len(pts), bool)
        for j,(p,n) in enumerate(zip(pts, numbers)):
            if int(n) == 1:
                if requested[j]:
                    raise ValueError('S01 external targets remain validation only')
                continue
            if int(p) not in rolemap or rolemap[int(p)][0] != int(n):
                raise ValueError('External native PTS/shot has no exact inherited role')
            allowed[j] = rolemap[int(p)][1] and requested[j]
            if requested[j] and not rolemap[int(p)][1]:
                raise ValueError('External witnesses cannot promote a protected frame')
        if not np.isfinite(X).all() or not np.isfinite(target).all():
            raise ValueError('Use finite numerical padding plus an explicit false mask')
        chart = dict(external_numerical_3D_witnesses=True, path=str(Path(external_witness_npz).resolve()),
                     sha256=file_sha256(external_witness_npz), ice_plane_assumption=False,
                     surface_id_values=np.unique(surface_ids).tolist(), source_observations=False)
    else:
        target = np.broadcast_to(np.array([(u, v) for v in (180., 360., 540., 720., 900.)
            for u in (180., 440., 700., 960., 1220., 1480., 1740.)]), (len(pts), 35, 2)).copy()
        if template_audit_file is not None:
            T, _, _, chart = _chart_from_entities(model, template_audit_file)
        else:
            T = np.array(previous['template']['current_to_reference_H'])
            chart = copy.deepcopy(previous['template'])
        H = seed.a['H_reference_xy_to_native']@T
        xy = _project(np.linalg.inv(H), target)
        X = np.concatenate((xy, np.zeros(xy.shape[:-1]+(1,))), -1)
        if template_audit_file is None:
            mask = transfer['witness_mask'].astype(bool)
        else:
            scene = NativeScene(as_event_view(model, 's01'))
            denominator = np.einsum('fi,fki->fk', H[:, 2], np.concatenate((xy, np.ones(xy.shape[:-1]+(1,))), -1))
            mask = (scene.containment(xy) < -.001) & np.isfinite(xy).all(-1)
            mask &= seed.a['front_sign'][:, None]*denominator > 0
        if np.any(mask.sum(1) < 2):
            raise ValueError('A full native frame lacks finite numerical plane support')
    selected = [s for s in sorted(model['shots']) if s != 's01' and model['shots'][s]['rig_id'] in rigs]
    if set(rigs)-set(model['rigs']) or not selected:
        raise ValueError('Unknown or empty actual rig scope')
    out.mkdir(parents=True)
    before, ba = _evaluate(model, pts, numbers, X, target, mask, allowed, selected)
    write_json(out/'before.json', before); np.savez_compressed(out/'before_arrays.npz', **ba)
    levels = []
    for rid in rigs:
        sids = [s for s in selected if model['shots'][s]['rig_id'] == rid]
        fitframes = np.isin(numbers, [int(s[1:]) for s in sids]) & allowed
        ids = np.flatnonzero(fitframes)
        rig = model['rigs'][rid]
        h = np.array(model['blocks'][rig['head_block']]['values']); o = np.array(model['blocks'][rig['optics_block']]['values']); meta = rig['camera']
        row_pts = pts[ids, None]+12800.*o[11]*(target[ids, :, 1]/1079.-.5)
        rownumbers = np.broadcast_to(numbers[ids, None], row_pts.shape)
        keep = mask[ids]
        fitX = X[ids][keep]; fitUV = target[ids][keep]
        fitN = rownumbers[keep]; fitP = row_pts[keep]
        if X.shape[1] < 2 or np.any(mask[ids].sum(1)<2):
            # Ragged de-duplicated records share one physical Motion at repeated
            # PTS. Never allocate a private independent 3q per record.
            fitQ = None
            tq = dict(independent_frame_teacher_skipped=True,
                reason='Ragged single-record rows go directly to the shared cubic sparse GN; repeated PTS share exactly the same shot trajectory',
                numerical_navigation_rows=len(fitX), source_observations_created=False)
        else:
            mid = Motion(model, sids, numbers[ids], pts[ids])
            qmid = mid.values(mid.x)
            mr = Motion(model, sids, rownumbers.ravel(), row_pts.ravel())
            qrow = mr.values(mr.x).reshape(len(ids), X.shape[1], 3)
            delta = qrow-qmid[:, None, :]
            teacher, tq = fit_frame_coordinates(h, o, meta, X[ids], target[ids], qmid, mask=mask[ids], row_delta=delta)
            fitQ = (teacher[:, None, :]+delta)[keep]
            np.savez_compressed(out/(rid+'_frame_q.npz'), native_pts=pts[ids], shot_number=numbers[ids], q_before=qmid, q_fit=teacher, original_row_delta=delta)
        write_json(out/(rid+'_frame_q_report.json'), tq)
        gaps = np.concatenate([np.diff(pts[(numbers == int(s[1:])) & allowed]) for s in sids])/12800.
        cadence = float(np.min(gaps[gaps > 0]))
        intervals = {}
        half = float(meta['tau_bound'])/2
        for sid in sids:
            stamps = np.unique(pts[(numbers == int(sid[1:])) & allowed])
            cut = np.r_[0, np.flatnonzero(np.diff(stamps) != int(seed.meta['native_step_pts']))+1, len(stamps)]
            origin = model['shots'][sid]['time_origin_pts']
            intervals[sid] = [[float((stamps[a]-origin)/12800.-half),float((stamps[b-1]-origin)/12800.+half)]
                for a,b in zip(cut[:-1],cut[1:]) if b>a]
        prior_cost = np.inf
        while True:
            fit = _fit_row_motion(model, sids, fitN, fitP, fitX, fitUV, fitQ, intervals)
            stage, arrays = _evaluate(model, pts, numbers, X, target, mask, allowed, sids)
            step = dict(rig=rid, dt={s:model['shots'][s]['camera']['dt'] for s in sids},
                        actual_navigation_cadence_seconds=cadence, fit=fit, per_shot=stage)
            levels.append(step)
            label = f'level_{len(levels):02d}_{rid}'
            write_json(out/(label+'.json'), step)
            np.savez_compressed(out/(label+'_arrays.npz'), **arrays)
            print(json.dumps(dict(stage=label, dt=step['dt'], fit=fit, per_shot=stage)), flush=True)
            cost = fit['cost_after']
            resolution = min(model['shots'][s]['camera']['dt'] for s in sids)
            if np.isfinite(prior_cost) and cost > prior_cost+TOL*(1+prior_cost):
                raise ValueError('Equivalent refinement worsened actual navigation objective')
            if cost <= EPS*EPS*(1+np.sum(fitUV*fitUV)):
                step['refinement_stop'] = 'Navigation transfer at arithmetic precision'
                break
            ratio = int(round(resolution/cadence))
            if ratio <= 1:
                step['refinement_stop'] = 'Reached actual native navigation support cadence; no finer temporal evidence added'
                break
            if abs(resolution/cadence-ratio) > 1e-9:
                raise ValueError('Native cadence is not an integer nested cardinal lattice')
            factor = next(d for d in range(2, ratio+1) if ratio % d == 0)
            prior_cost = cost
            refinement = []
            for sid in sids:
                model, proof = refine_shot_exact(model, sid, factor)
                refinement.append(proof)
            step['next_refinement'] = refinement
            write_json(out/(label+'.json'), step)
    after, aa = _evaluate(model, pts, numbers, X, target, mask, allowed, selected)
    write_json(out/'after.json', after); np.savez_compressed(out/'after_arrays.npz', **aa)
    # Only selected shot motion/reference values and their old/new storage flags
    # may differ. Every shared physical value and every S01 record remains exact.
    motionids = set()
    for sid in selected:
        motionids.update(original['shots'][sid]['control_blocks'])
        motionids.add(original['shots'][sid]['reference_angles_block'])
    for i, block in enumerate(original['blocks']):
        if i not in motionids and block != model['blocks'][i]:
            raise ValueError('Non-motion original dependency changed')
    for key in ('materials', 'rigs', 'scene_block'):
        if original[key] != model[key]:
            raise ValueError('Shared physical representation changed')
    if original['shots']['s01'] != model['shots']['s01']:
        raise ValueError('Protected S01 shot changed')
    model['screen_native_motion_v2'] = dict(accepted=False, motion_only=True, rigs=list(rigs),
        source_images_read=False, source_precision_created=False, original_CP_rebound=False,
        input_model_sha256=file_sha256(model_file), numerical_screen_witnesses_are_not_source=True)
    write_json(out/'model.json', model)
    result = dict(schema='rinkfit.screen_native_motion_candidate.v2', accepted=False,
        model=dict(path=str(out/'model.json'), sha256=file_sha256(out/'model.json')),
        inputs={k:dict(path=str(Path(p).resolve()), sha256=file_sha256(p)) for k,p in dict(
            model=model_file, screen_seed=screen_seed_file, transfer=transfer_file, report=report_file).items()},
        rigs=list(rigs), before=before, after=after, levels=levels,
        S01_and_all_shared_scene_head_optics_unchanged=True, protected_frames_fit=False,
        source_row_time_in_objective=True, new_CP_or_source_packet_issued=False, witness_chart=chart,
        conclusion='Independent numerical motion initializer; source accuracy and full-nine material alignment remain separate obligations.')
    for name,pin in (('template_audit', template_audit_file), ('external_witness_npz', external_witness_npz)):
        if pin is not None:
            result['inputs'][name] = dict(path=str(Path(pin).resolve()), sha256=file_sha256(pin))
    write_json(out/'report.json', result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True); p.add_argument('--screen-seed', required=True)
    p.add_argument('--transfer', required=True); p.add_argument('--report', required=True)
    p.add_argument('--out', required=True); p.add_argument('--rigs', nargs='+', default=['secondary', 'end'])
    p.add_argument('--template-audit'); p.add_argument('--external-witness-npz')
    a = p.parse_args()
    result = build_motion_candidate(a.model, a.screen_seed, a.transfer, a.report, a.out, rigs=tuple(a.rigs),
        template_audit_file=a.template_audit, external_witness_npz=a.external_witness_npz)
    print(json.dumps(dict(model=result['model'], after=result['after'])), flush=True)


if __name__ == '__main__':
    main()
