"""Constrained Gauss--Newton proposal for a verified continuous-motion scope."""
import json

import numpy as np
from scipy.optimize import Bounds, minimize
from scipy.sparse import csc_matrix, csr_matrix, diags, eye, triu, vstack


def conic_step(solver, hessian, gradient, lower, upper, matrix, slack, balls):
    """Use the same convex quadratic and exact affine norm cones directly."""
    dimension=len(gradient)
    I=eye(dimension,format='csc')
    pieces=[-I,I,matrix]
    bounds=[-lower,upper,slack]
    cones=[solver.NonnegativeConeT(2*dimension+len(slack))]
    for j,offset,cap,_ in balls:
        pieces.append(vstack((csr_matrix((1,dimension)),-csr_matrix(j)),format='csc'))
        bounds.append(np.r_[cap,offset])
        cones.append(solver.SecondOrderConeT(len(offset)+1))
    settings=solver.DefaultSettings()
    settings.verbose=False
    settings.max_threads=1
    settings.tol_feas=1.e-10
    settings.tol_gap_abs=1.e-12
    settings.tol_gap_rel=1.e-12
    problem=solver.DefaultSolver(triu(csc_matrix(hessian),format='csc'),gradient,
                                 vstack(pieces,format='csc'),np.concatenate(bounds),cones,settings)
    solution=problem.solve()
    q=np.asarray(solution.x)
    if q.shape!=gradient.shape or not np.isfinite(q).all():
        raise ArithmeticError('Nonfinite conic motion proposal')
    excess=max(float(np.max(matrix@q-slack,initial=0.)),float(np.max(lower-q,initial=0.)),
               float(np.max(q-upper,initial=0.)),
               max((float(np.linalg.norm(offset+j@q)-cap) for j,offset,cap,_ in balls),default=0.))
    diagnostic={}
    if str(solution.status)=='PrimalInfeasible':
        # Clarabel's positive dual certificate isolates the simultaneous
        # affine faces causing infeasibility. Keep indices in solver order.
        dual=np.asarray(solution.z)
        rows=dual[2*dimension:2*dimension+len(slack)]
        diagnostic['infeasible_halfspace_indices']=[int(i) for i in np.argsort(rows)[-12:][::-1] if rows[i]>1.e-8]
        i=2*dimension+len(slack)
        cones=[]
        for _,offset,_,label in balls:
            cones.append((float(dual[i]),label))
            i+=len(offset)+1
        diagnostic['infeasible_norm_labels']=[label for weight,label in sorted(cones,reverse=True)[:12] if weight>1.e-8]
    return q,dict(method='sparse second-order cone',message=str(solution.status),
                  success=str(solution.status) in ('Solved','AlmostSolved') and excess<=1.e-8,
                  iterations=int(solution.iterations),halfspaces=len(slack),norms=len(balls),
                  max_constraint_excess=excess,**diagnostic)


def metric_step(hessian, gradient, lower, upper, matrix, slack, balls, *, initial=None):
    """Minimize the cost quadratic over the trust box and affine guard domains.

    Add violated rows to the working set, including whole residual norms. The
    returned direction is checked against *all* supplied domains. No cone
    tangent at the origin stands in for an entire residual norm.
    """
    try:
        import clarabel
    except ImportError:
        clarabel=None
    if clarabel is not None:
        return conic_step(clarabel,hessian,gradient,lower,upper,matrix,slack,balls)
    q = np.zeros_like(gradient) if initial is None else np.clip(initial, lower, upper)
    active = np.zeros(len(slack), bool)
    norm_active = np.zeros(len(balls), bool)
    count = 0
    initial_fit = initial is not None
    while True:
        excess = matrix@q - slack
        norm_excess = np.array([np.linalg.norm(offset + j@q)-cap for j, offset, cap, _ in balls])
        maximum = max(excess.max(initial=0.), norm_excess.max(initial=0.))
        if initial_fit and maximum <= 1.e-8:
            return q, dict(success=bool(fit.success) if count else True, working_solves=count,
                           halfspaces=int(active.sum()), norms=int(norm_active.sum()),
                           max_constraint_excess=float(maximum))
        add = excess > 1.e-9
        add_norms = norm_excess > 1.e-9
        if initial_fit and not (np.any(add & ~active) or np.any(add_norms & ~norm_active)):
            return q, dict(success=False, message=str(fit.message), working_solves=count,
                           halfspaces=int(active.sum()), norms=int(norm_active.sum()),
                           max_constraint_excess=float(maximum))
        active |= add
        norm_active |= add_norms
        a, b = matrix[active].toarray(), slack[active]
        chosen = [part for part, keep in zip(balls, norm_active, strict=True) if keep]

        def inequalities(z):
            normals = [cap - np.linalg.norm(offset + j@z) for j, offset, cap, _ in chosen]
            return np.r_[b-a@z, normals]

        def normal_jacobian(z):
            rows = [-a]
            for j, offset, _, _ in chosen:
                moved = offset+j@z
                rows.append(-(moved@j)[None]/max(np.linalg.norm(moved), 1.e-13))
            return np.concatenate(rows, axis=0)

        def objective(z):
            change = hessian@z
            return float(z@(gradient+change/2)), gradient+change

        constraints = ([dict(type='ineq', fun=inequalities, jac=normal_jacobian)]
                       if active.any() or norm_active.any() else [])
        fit = minimize(objective, q, method='SLSQP', jac=True, bounds=Bounds(lower, upper),
                       constraints=constraints, options=dict(ftol=2.e-10, maxiter=600))
        q = fit.x
        if not np.isfinite(q).all():
            raise ArithmeticError('Nonfinite constrained motion proposal')
        count += 1
        initial_fit = True


def local_coordinates(problem, columns, relative=True):
    state = problem.body.state
    knot, axis = divmod(columns-17, 69)
    track = np.where(axis < 3, -1, (axis-3)//3)
    source = problem.reference.p
    num, den = source['time_base']
    time = state.knots if relative else state.knots + source['pts'][0]*num/den
    return [dict(column=int(c), knot=int(k), track=int(t), axis=int(a%3),
                 support_seconds=[float(time[k]), float(time[k+4])],
                 unit='solver metre' if t == -1 else 'right SO(3) control radian')
            for c, k, t, a in zip(columns, knot, track, axis, strict=True)]


def source_projection(problem, x, frames, joints):
    from .rebuild_exposure_body import moving_projection

    if not len(frames):
        return np.empty((0,2))
    p = problem.body
    unique, frame_index = np.unique(frames,return_inverse=True)
    vocabulary, joint_index = np.unique(joints,return_inverse=True)
    tau = p.projector.readout_times(unique)
    times = (problem.reference.times[unique, None] + tau[:, None]*np.linspace(-.5, .5, 5)).ravel()
    world = problem.reference.world(problem.reference.at(times), p.state.increments(x[17:], times))['world']
    nodes = world.reshape(len(unique), 5, -1, 3)[:, :, p.layout.body_landmarks[vocabulary]]
    return moving_projection(p.projector,nodes,unique)['uv'][frame_index,joint_index]


def skate_projection(problem, x, rows, *, jacobian=False):
    """Project named outsole/runner points through the saved native exposure."""
    from .programme_skate_evidence import exposure_times
    from .programme_whole_skate import native_point_derivative, chain
    from .rebuild_exposure_body import moving_projection
    from .skate_physics import fk_state
    from .skate_template import profile_height

    state=problem.body.state
    source=problem.reference.p['pts']
    binding=problem.binding.packet()
    chord=np.array([[0.,0.,binding['floor'][0]],[binding['xs'][-1],0.,binding['floor'][-1]]])
    parameters=binding['parameters']
    runner_x=np.array([-parameters.runner_rear_extension_m,
                       parameters.sole_length_m-parameters.runner_front_inset_m])
    runner=np.column_stack((runner_x,np.zeros(2),profile_height(runner_x,parameters)))
    # The upper/sole seam and the bottom of the solid heel block are distinct
    # material locations. A visible heel-block corner cannot target the insole.
    heel_block=np.array([[0.,0.,parameters.rear_mount_height_m]])
    local=np.r_[chord,chord.mean(axis=0)[None],runner,heel_block,np.zeros((1,3))]
    feature={'heel':0,'toe':1,'sole_middle':2,'runner_rear':3,'runner_front':4,
             'heel_block_rear':5,'cuff_center':6}
    if any(row['feature'] not in feature or row['foot'] not in (0,1) for row in rows):
        raise ValueError('Unsupported native shoe material point or physical foot')
    uv=np.empty((len(rows),2))
    matrices=[];positions=[]
    for foot in (0,1):
        picks=[i for i,row in enumerate(rows) if row['foot']==foot]
        if not picks:
            continue
        stamps=[rows[i]['pts'] for i in picks]
        frames=np.searchsorted(source,stamps)
        np.testing.assert_array_equal(source[frames],stamps)
        point=local[[feature[rows[i]['feature']] for i in picks]]
        cuff=np.array([rows[i]['feature']=='cuff_center' for i in picks])
        times=exposure_times(problem,frames)
        base=problem.reference.at(times)
        delta=state.increments(x[17:],times)

        def vertices(pose):
            origins=pose['O'][:,foot].reshape(-1,5,3)
            axes=pose['R'][:,foot].reshape(-1,5,3,3)
            points=np.broadcast_to(point[:,None],origins.shape)
            if cuff.any():
                # The visible mouth of the boot is distinct from an inferred
                # anatomical ankle. Use the centre of its actual deformed
                # collar, retaining the same cuff geometry as the renderer.
                b=problem.binding
                alpha=b.deformation(pose['native'],pose['frames'],
                                    skate_relative_R=pose['skate_relative_R'])
                collar=binding['upper'][np.unique(binding['collar'])]
                centre=b.deform(collar,alpha)[:,foot].mean(axis=1).reshape(-1,5,3)
                points=points.copy();points[cuff]=centre[cuff]
            return (origins+np.einsum('nfij,nfj->nfi',axes,points))[:,:,None]

        pose=fk_state(problem.reference,base,delta)
        value=moving_projection(problem.camera,vertices(pose),frames,jacobian=jacobian)
        uv[picks]=value['uv'][:,0]
        if jacobian:
            body,shoe=native_point_derivative(problem,base,delta,vertices,foot,value['pixel_jac_nodes'])
            matrices.append(chain(problem,delta,times,body[:,0],shoe[:,0]))
            positions.extend((2*np.asarray(picks)[:,None]+np.arange(2)).ravel())
    derivative=vstack(matrices,format='csr')[np.argsort(positions)] if jacobian else None
    return uv,derivative


def projection_planes(uv,jacobian,bounds):
    lo,hi=np.swapaxes(np.asarray(bounds),0,1)
    return vstack((jacobian,-jacobian),format='csr'),np.r_[(hi-uv).ravel(),(uv-lo).ravel()]


def relative_rotations(state, x, times, tracks, *, jacobian=False):
    from .absolute_motion_curve import _log
    from .absolute_motion_state import evaluate_curve, inverse_right_jacobian

    ts, source_index = np.unique(np.asarray(times).ravel(), return_inverse=True)
    source_index = source_index.reshape(-1, 2)
    increment = x[17:17+state.body_size].reshape(state.m, 69)[:, 3:].reshape(-1, 22, 3)
    sample = evaluate_curve(state.current_curve(x[17:]), ts, with_blocks=jacobian,
                            control_increments=increment)
    ra, rb = (sample['rotations'][source_index[:, j], tracks] for j in range(2))
    relative = ra.swapaxes(-1, -2)@rb
    vector = _log(relative)
    if not jacobian:
        return vector, None
    end = inverse_right_jacobian(vector)
    maps = (-end@relative.swapaxes(-1, -2), end)
    rows, columns, values = [], [], []
    for j in range(2):
        i = source_index[:, j]
        ids = 17 + 69*sample['control_indices'][i, :, None] + 3 + 3*tracks[:, None, None] + np.arange(3)
        d = maps[j][:, None]@sample['rotation_right_blocks'][i, :, tracks]
        rows.append(np.broadcast_to(np.arange(vector.size).reshape(-1, 1, 3, 1), d.shape).ravel())
        columns.append(np.broadcast_to(ids[:, :, None], d.shape).ravel())
        values.append(d.ravel())
    matrix = csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                        shape=(vector.size, len(x)))
    matrix.eliminate_zeros()
    return vector, matrix


def target_observations(problem, spec):
    pts = problem.reference.p['pts']
    body = problem.body
    rows = spec['image_regions']
    frames = np.searchsorted(pts, [r['pts'] for r in rows])
    np.testing.assert_array_equal(pts[frames], [r['pts'] for r in rows])
    joints = np.array([r['joint'] for r in rows])
    points, keep = [], []
    for k, (f, j) in enumerate(zip(frames, joints, strict=True)):
        take = np.flatnonzero((body.frames[body.of] == f) & (body.model_oj == j))
        if len(take) != 1:
            if rows[k].get('fit', True) or rows[k].get('protect_projection'):
                raise ValueError(f'Unbound fitting target: source PTS {pts[f]}, joint {j}')
            # A manually visible holdout needs no admitted detector row. The
            # saved-pose comparison projects it directly on the same camera.
            continue
        keep.append(k)
        points.append(take[0])
    return frames[keep], joints[keep], points, [rows[k] for k in keep]


def target_terms(problem, first, x, spec, emit):
    from .programme_guard_proposals import recover_body_pixel_jacobian

    pts = problem.reference.p['pts']
    body = problem.body
    frames, joints, points, rows = target_observations(problem, spec)
    jac = recover_body_pixel_jacobian(problem, first, points)
    uv = body.cache['uv'][body.of[points], body.layout.body_landmarks[joints]]
    verify = source_projection(problem, x, frames, joints)
    np.testing.assert_allclose(verify, uv, rtol=0., atol=2.e-6)
    # Author the bounds from clean native images. Use only their centres in the
    # proposal merit, with per-axis image uncertainty. The held-out bounds and
    # original frozen guards still judge each exact nonlinear trial.
    axes = np.asarray([r.get('fit', True) for r in rows]).repeat(2)
    target = np.asarray([r['bounds_uv'] for r in rows]).reshape(-1, 2, 2)
    centre = target.mean(axis=1)
    sigma = np.asarray([r['sigma_uv'] for r in rows]).ravel()
    j_target = jac.multiply((1/sigma)[:, None]).tocsr()[axes]
    r_target = ((uv-centre).ravel()/sigma)[axes]
    rotation = spec['rotation_steps']
    ends = np.searchsorted(pts, np.asarray([r['pts_pair'] for r in rotation]))
    np.testing.assert_array_equal(pts[ends], [r['pts_pair'] for r in rotation])
    track = np.array([r['track'] for r in rotation])
    ts = problem.reference.times[ends]
    vector, rotation_j = relative_rotations(body.state, x, ts, track, jacobian=True)
    scale = 1/np.deg2rad([r['scale_deg'] for r in rotation])
    factor = np.asarray([r['relative_target'] for r in rotation])
    r_spin = (scale[:, None]*vector*(1-factor[:, None])).ravel()
    j_spin = rotation_j.multiply(scale.repeat(3)[:, None]).tocsr()
    emit('scoped_source_targets', image_pts=pts[frames].tolist(), view_pixel_before=uv.tolist(),
         bounds_uv=target.tolist(), local_rotation_step_before_deg=np.rad2deg(np.linalg.norm(vector,axis=1)).tolist(),
         image_rows_used_in_proposal=int(axes.sum()), same_source_role=True)
    return (np.r_[r_target, r_spin], vstack((j_target, j_spin), format='csr'),
            (frames, joints, uv, jac, ts, track, vector, rotation_j), rows)


def derivative_receipt(problem, x, measurements, selected):
    frames, joints, uv, jac, ts, track, vector, rotation_j = measurements
    # Include anatomical downstream rows when the knee and ankle rotate about
    # their own centres and therefore have little effect on the knee pixel.
    col_ids = np.unique(jac.indices)
    lanes = []
    for tracks in ((-1,), (0,), (1,), (2,), (4,), (5,), (7,), (8,)):
        cols = col_ids[np.isin(((col_ids-17)%69-3)//3, tracks)]
        cols = np.intersect1d(cols, selected)
        if not len(cols):
            continue
        strongest = np.asarray(abs(jac[:, cols]).max(axis=0).toarray()).ravel()
        lanes.append(cols[int(strongest.argmax())])
    if rotation_j.nnz:
        keep = np.intersect1d(rotation_j.indices, selected)
        if len(keep):
            strongest = abs(rotation_j[:, keep]).max(axis=0).toarray().ravel()
            lanes.append(keep[int(strongest.argmax())])
    checks = []
    for col in np.unique(lanes):
        h = 2.e-6
        plus, minus = x.copy(), x.copy()
        plus[col] += h
        minus[col] -= h
        native = (source_projection(problem, plus, frames, joints)
                  - source_projection(problem, minus, frames, joints)).ravel()/(2*h)
        spin = (relative_rotations(problem.body.state, plus, ts, track)[0]
                - relative_rotations(problem.body.state, minus, ts, track)[0]).ravel()/(2*h)
        expected = np.r_[jac[:, [col]].toarray().ravel(), rotation_j[:, [col]].toarray().ravel()]
        got = np.r_[native, spin]
        np.testing.assert_allclose(got, expected, rtol=3.e-5, atol=8.e-5)
        k = int(np.argmax(abs(expected)))
        checks.append(dict(**local_coordinates(problem, np.array([col]), relative=False)[0],
                           finite_difference_step=h, analytic=float(expected[k]), finite_difference=float(got[k]),
                           max_abs_error=float(np.max(abs(got-expected))),
                           source_pts=int(problem.reference.p['pts'][frames[k//2]]) if k<len(native) else None))
    return checks


def reduce_domains(margins, selected, scale, lower, upper, *, reserve_at_flight_boundary=True):
    matrix, slack, labels = margins
    box = np.maximum(abs(lower), abs(upper))
    A = matrix[:, selected].multiply(scale).tocsr()
    A.eliminate_zeros()
    # Per-point reserves that approximate a robust percentile can prohibit
    # movement even when its actual percentile improves. Keep the percentile's
    # affine gradient, and require the original percentile on the exact trial.
    aggregate = np.asarray(['/observation_' not in k for k in labels],dtype=bool)
    keep = aggregate & (np.diff(A.indptr)>0) & (abs(A)@box > slack)
    A = A[keep]
    b = slack[keep]
    normalization = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    A = diags(1./normalization)@A
    balls = []
    for item in margins.image_balls:
        if item['source_id'].startswith('__body_group__/'):
            continue
        mat = item['jacobian'].tocsr()[:, selected].multiply(scale).tocsr()
        mat.eliminate_zeros()
        offset = item['offset']
        potential = (abs(mat)@box).reshape(offset.shape)
        bound = np.linalg.norm(abs(offset)+potential, axis=1)
        for k in np.flatnonzero((bound > item['radius']) & (potential > 0).any(axis=1)):
            size = offset.shape[1]
            local = mat[k*size:(k+1)*size]
            active = np.diff(local.indptr)>0
            radius = item['radius'][k]
            fixed = np.linalg.norm(offset[k, ~active])
            cap = np.sqrt(max((radius-fixed)*(radius+fixed), 0.))
            # Keep a small model-agreement reserve in the *proposal* only for
            # flight norms enforced at machine-level historical tolerances.
            if item['labels'][k].startswith(('__flight__/', '__flight_dense__/')):
                reserve=0.002*cap
                if not reserve_at_flight_boundary:
                    reserve=min(reserve,max(0.,cap-np.linalg.norm(offset[k,active]))/2.)
                cap = max(0.,cap-reserve)
            norm = max(cap, np.linalg.norm(offset[k, active]), 1.e-4)
            balls.append((local[active].toarray()/norm, offset[k, active]/norm, cap/norm, item['labels'][k]))
    return A.tocsr(), b/normalization, [k for k, keep in zip(labels, keep, strict=True) if keep], balls


def scoped_proposals(problem, first, x, guards, scoped_guard, mask, spec, output):
    """Build alternate directions on one verified native programme state."""
    from .programme_guard_proposals import ProgrammeGuardProposals
    from .programme_source_step import SourceMargins, all_source_margins, solve_bounded_quadratic
    from .scoped_motion_domains import ScopedMotionDomains, ScopedSurfaceDomains, local_surface_regressions
    from .fullbody_round import write_json

    def emit(event, **detail):
        problem.log.write(json.dumps(dict(event=event, **detail))+'\n')

    coords = local_coordinates(problem, problem.active_parameter_indices, relative=False)
    groups = spec['control_tracks']
    holds = spec.get('held_track_at_source_seconds',{})
    selected = np.array([k['column'] for k in coords if str(k['track']) in groups
                        and groups[str(k['track'])][0] <= np.mean(k['support_seconds']) <= groups[str(k['track'])][1]
                        and not any(k['support_seconds'][0]<stamp<k['support_seconds'][1]
                                    for stamp in holds.get(str(k['track']),[]))])
    if not len(selected):
        raise ValueError('The reviewed span selects no active controls')
    coeffs = local_coordinates(problem, selected, relative=False)
    emit('scoped_metric_setup', active_columns=len(selected), target_sha256=spec['sha256'])
    t_r, t_j, measurements, target_rows = target_terms(problem, first, x, spec, emit)
    phase = None
    if phase_spec := spec.get('phase_quality'):
        from .scoped_phase_quality import ScopedPhaseQuality
        phase = ScopedPhaseQuality(scoped_guard,first,x,phase_spec,selected,skate_projection,
                                   source_projection,guards.cavity_trial_guard)
        phase_margins, phase_r, phase_j = phase.blocks(x,proposal_merit=True)
        t_r = np.r_[t_r,phase_r]
        t_j = vstack((t_j,phase_j),format='csr')
        write_json(output/'phase_derivatives.json',dict(checks=phase.fd_checks,phase_merit_rows=len(phase_r)))
    fd = derivative_receipt(problem, x, measurements, selected)
    write_json(output/'native_derivative_check.json', dict(columns=fd,
        conditional_m_per_solver_m=problem.binding.ratio, analytic='actual weighted body rows with robust transform inverted and five rolling-exposure nodes; exact right SO(3) spline chain'))
    full = first['jac'][:, selected].tocsr()
    full.eliminate_zeros()
    used = np.diff(full.indptr)>0
    j, r = full[used], first['residual'][used]
    del full
    weighted_columns = np.sqrt(np.asarray(j.multiply(j).sum(axis=0)).ravel())
    scale = 1./np.maximum(weighted_columns, 1.e-6)
    j = j.multiply(scale).tocsr()
    target_jac = t_j[:, selected].multiply(scale).tocsr()
    unit_box = np.where((selected-17)%69 < 3, spec['root_box_m']/problem.binding.ratio,
                        np.deg2rad(spec['rotation_box_deg'])) / scale
    rot = (selected-17)%69 >= 3
    lower = np.where(rot, np.maximum((-np.pi-x[selected])/scale, -unit_box), -unit_box)
    upper = np.where(rot, np.minimum((np.pi-x[selected])/scale, unit_box), unit_box)
    h = (j.T@j).toarray()
    h.flat[::len(h)+1] += .07**2
    g = j.T@r
    fit_h = h + (target_jac.T@target_jac).toarray()
    fit_g = g + target_jac.T@t_r
    raw, raw_report = solve_bounded_quadratic(lambda t: (float(t@(g+h@t/2)), g+h@t),
                                              np.zeros(len(g)), lower, upper)
    hints, _ = solve_bounded_quadratic(lambda t: (float(t@(fit_g+fit_h@t/2)), fit_g+fit_h@t),
                                       raw, lower, upper)
    margins = all_source_margins(problem, first, guards.baseline, guards.turn_before, guards.torso_before, mask)
    context = ProgrammeGuardProposals(problem, first, margins, guards.check, guards.joint_peaks,
        guards.surface_peak, signed_joint_domains=True, flight_guard=guards.flight_guard,
        dense_flight_guard=guards.dense_flight_guard, cavity_trial_guard=guards.cavity_trial_guard)
    a, b, names, balls = reduce_domains(context.margins, selected, scale, lower, upper,
        reserve_at_flight_boundary=not spec.get('refresh_nonlinear_tangents',False))
    del margins, context
    emit('scoped_metric_domains', affected_residual_rows=int(used.sum()), linear_constraints=len(b),
         residual_norm_constraints=len(balls), inverse_column_scaling=[float(scale.min()), float(scale.max())])
    # Diagnose the previous proposal domains on the same target quadratic.
    if phase is None or spec.get('run_comparison_trials',True):
        euclid, report_euclid = metric_step(np.eye(len(h)), -hints, lower, upper, a, b, balls, initial=hints)
        global_only, report_global = metric_step(fit_h, fit_g, lower, upper, a, b, balls, initial=hints)
    else:
        euclid=global_only=raw
        report_euclid=report_global=None

    active = ScopedMotionDomains(scoped_guard,selected)
    exact_domains,_ = active.blocks(x,relative_rotations)
    local_a, local_b, local_names, local_balls = reduce_domains(exact_domains,selected,scale,lower,upper)
    a, b, names = vstack((a,local_a),format='csr'), np.r_[b,local_b], names+local_names
    balls.extend(local_balls)
    if phase is not None:
        phase_a, phase_b, phase_names, phase_balls = reduce_domains(phase_margins,selected,scale,lower,upper)
        a,b,names=vstack((a,phase_a),format='csr'),np.r_[b,phase_b],names+phase_names
        balls.extend(phase_balls)
    held_images = [i for i,row in enumerate(target_rows) if row.get('protect_projection')]
    if held_images:
        frames,joints,coordinates,pixel_j=measurements[:4]
        frames,joints=frames[held_images],joints[held_images]
        points=coordinates[held_images].ravel()
        body_J=pixel_j[(2*np.asarray(held_images)[:,None]+np.arange(2)).ravel()]
        bounds=np.array([target_rows[i]['bounds_uv'] for i in held_images])
        limits=np.stack((bounds[:,0]+.15,bounds[:,1]-.15))
        j_hold=vstack((body_J,-body_J),format='csr')
        b_hold=np.r_[limits[1].ravel()-points,points-limits[0].ravel()]
        hold_names=[f'protected_source_projection/{target_rows[i]["pts"]}/joint_{target_rows[i]["joint"]}/{side}/{axis}'
                    for side in ('high','low') for i in held_images for axis in ('u','v')]
        held = SourceMargins(j_hold,b_hold,hold_names,[])
        img_a,img_b,img_names,_=reduce_domains(held,selected,scale,lower,upper)
        a,b,names=vstack((a,img_a),format='csr'),np.r_[b,img_b],names+img_names
        emit('scoped_held_image_regions',names=hold_names, measured_source_pixels=True)
    shoe_rows=spec.get('protected_skate_regions',[])
    if shoe_rows:
        shoe_bounds=np.asarray([row['bounds_uv'] for row in shoe_rows],float)
        shoe_uv,shoe_j=skate_projection(problem,x,shoe_rows,jacobian=True)
        shoe_a,shoe_b=projection_planes(shoe_uv,shoe_j,shoe_bounds)
        shoe_names=[f'protected_skate_projection/{row["pts"]}/foot_{row["foot"]}/{row["feature"]}/{side}/{axis}'
                    for side in ('high','low') for row in shoe_rows for axis in ('u','v')]
        mA,mB,mnames,_=reduce_domains(SourceMargins(shoe_a,shoe_b,shoe_names,[]),selected,scale,lower,upper)
        a,b,names=vstack((a,mA),format='csr'),np.r_[b,mB],names+mnames
        emit('scoped_skate_projection_domains',names=shoe_names,baseline_uv=shoe_uv.tolist())
    surface=None
    before_surface=(a,b,names)
    sample_caps=spec.get('local_surface_caps',[])
    candidates=[]

    def refreshed_surface_domains(candidates):
        checked=ScopedSurfaceDomains(problem,x,candidates,sample_caps,
                                     selected=selected if phase is not None else None)
        s_a,s_b,s_names,_=reduce_domains(checked.blocks(x),selected,scale,lower,upper)
        early_a,early_b,early_names=before_surface
        emit('scoped_refreshed_surface_domains',halfspaces=len(s_b),**checked.freshness)
        return checked,vstack((early_a,s_a),format='csr'),np.r_[early_b,s_b],early_names+s_names

    if spec.get('surface_proposal_from_rejected'):
        from .rebuild_prepare import digest, load
        from .scoped_candidate import proposal_vector
        rejected=spec['surface_proposal_from_rejected']
        for item in rejected if isinstance(rejected,list) else [rejected]:
            if digest(item['path'])!=item['sha256']:
                raise ValueError('Surface refinement trial identity changed')
            if item.get('format')=='scoped_vector':
                packet=load(item['path'])
                np.testing.assert_array_equal(packet['parent_x'],x)
                prior=packet['x']
                if prior.shape!=x.shape or not np.isfinite(prior).all():
                    raise ValueError('Invalid scoped surface trial')
                changed=np.flatnonzero(prior!=x)
                if np.any(~np.isin(changed,problem.active_parameter_indices)):
                    raise ValueError('Surface refinement trial changed fixed columns')
            else:
                prior=proposal_vector(problem,item['path'],spec['parent_sha256'],parent_vector=x)
            candidates.append(prior)
        surface,a,b,names=refreshed_surface_domains(candidates)
    elif phase is not None and problem.scoped_surface_trial.current_scan is not problem.scoped_surface_trial.scan:
        surface,a,b,names=refreshed_surface_domains([])
    if guards.cavity_trial_guard is not None:
        cavity = guards.cavity_trial_guard
        term = cavity._term(first)
        # The old tangent leaves quadratic motion, and a change of active
        # foot witness, free. Keep each historical AL envelope as a norm.
        cavity_radius=[]
        for i, kind in enumerate(cavity.kinds):
            cap = cavity.caps[i] + 1.e-7*max(1.,cavity.caps[i])
            reserve = float(spec.get('cavity_proposal_reserve',{}).get(kind,0.))
            cavity_radius.append(dict(source_id=cavity.prefix+kind,offset=cavity.start[:,i][None],
                radius=[max(0.,cap-reserve)],jacobian=term.jacobian[i::len(cavity.kinds)],labels=[cavity.prefix+kind]))
        _,_,_,cavity_balls=reduce_domains(SourceMargins(csr_matrix((0,problem.size)),np.empty(0),[],cavity_radius),
                                        selected,scale,lower,upper)
        balls.extend(cavity_balls)
    emit('scoped_motion_interior_constraints', local_halfspaces=len(local_b),
         local_norms=len(local_balls), total_halfspaces=len(b),total_norms=len(balls))

    def refine(domains):
        A, B = a, b
        evidence = []
        current_cuts = {}
        refresh = spec.get('refresh_nonlinear_tangents',False)
        for _ in range(20 if phase is not None else 12):
            rows=list(current_cuts.values())
            trial_A=vstack([A,*[m for m,_ in rows]],format='csr') if rows else A
            trial_B=np.r_[B,*[d for _,d in rows]] if rows else B
            q,status = metric_step(fit_h,fit_g,lower,upper,trial_A,trial_B,domains,initial=hints)
            if status['max_constraint_excess']>1.e-7:
                domain_names=names+[key for key,(matrix,_) in current_cuts.items() for j in range(matrix.shape[0])]
                status['infeasible_halfspace_labels']=[domain_names[i] if i<len(domain_names) else 'retained_trial_tangent'
                                                        for i in status.get('infeasible_halfspace_indices',[])]
                emit('scoped_local_linear_domain_rejected',iteration=len(evidence),**status)
                return q,status,evidence
            trial=x.copy();trial[selected]+=q*scale
            J,excess,labels=active.cuts(trial,relative_rotations)
            if phase is not None:
                P,extra,phase_labels=phase.cuts(trial)
                J=vstack((J,P),format='csr');excess=np.r_[excess,extra]
                labels.extend(phase_labels)
            if held_images:
                uv=source_projection(problem,trial,frames,joints).ravel()
                over=np.r_[uv-limits[1].ravel(),limits[0].ravel()-uv]
                take=np.flatnonzero(over>1.e-5)
                J=vstack((J,j_hold[take]),format='csr');excess=np.r_[excess,over[take]]
                labels.extend(hold_names[k] for k in take)
            if shoe_rows:
                uv,pixel_j=skate_projection(problem,trial,shoe_rows,jacobian=True)
                S,slack=projection_planes(uv,pixel_j,shoe_bounds)
                take=np.flatnonzero(slack<2.e-4)
                J=vstack((J,S[take]),format='csr');excess=np.r_[excess,2.e-4-slack[take]]
                labels.extend(shoe_names[k] for k in take)
            if surface is not None:
                S,slack,s_names=surface.blocks(trial)
                affected=np.flatnonzero(slack<-5.e-7)
                J=vstack((J,S[affected]),format='csr');excess=np.r_[excess,-slack[affected]]
                labels.extend(s_names[k] for k in affected)
            emit('scoped_local_guard_linearization',iteration=len(evidence),
                 maximum_constraint_excess=status['max_constraint_excess'],
                 nonlinear_excess_rows=len(excess),norms=len(domains),
                 largest_nonlinear_excess=[dict(factor=labels[i],excess=float(excess[i]))
                                           for i in np.argsort(excess)[-5:][::-1]])
            if not labels:
                return q,status,evidence
            D=J[:,selected].multiply(scale).tocsr()
            magnitude=np.sqrt(np.asarray(D.multiply(D).sum(axis=1)).ravel())
            if np.any(magnitude==0):
                failed=[labels[i] for i in np.flatnonzero(magnitude==0)]
                emit('scoped_local_no_selected_derivative',labels=failed)
                raise ArithmeticError('A violated local constraint has no selected derivative: '+', '.join(failed))
            cuts=diags(1./magnitude)@D
            rhs=cuts@q-(excess+5.e-6)/magnitude
            if phase is None:
                A=vstack((A,cuts),format='csr');B=np.r_[B,rhs]
            else:
                # Keep the latest local tangent when ranks, triangle normals
                # or the axis of a residual norm move between trial points.
                families=np.asarray([label.split('/affected_row_',1)[0]
                    if refresh or label.startswith('phase/source/') else '' for label in labels])
                hard=families==''
                A=vstack((A,cuts[hard]),format='csr');B=np.r_[B,rhs[hard]]
                for key in np.unique(families[~hard]):
                    current=families==key
                    current_cuts[str(key)]=(cuts[current],rhs[current])
            record=[]
            for i,label in enumerate(labels):
                j=D[i]
                order=np.argsort(abs(j.data))[-6:][::-1]
                record.append(dict(factor=label,actual_excess=float(excess[i]),
                                   dominant_columns=selected[j.indices[order]].tolist()))
            evidence.append(record)
        status['success']=False
        status['nonlinear_local_refinement_exhausted']=True
        return q,status,evidence

    def proposal_receipt(z):
        norm = np.array([np.linalg.norm(offset + jac@z)-cap for jac,offset,cap,_ in balls])
        def rejected(values, labels):
            order = np.argsort(values)[-20:][::-1]
            return {labels[k]:float(values[k]) for k in order if values[k]>1.e-8}
        return dict(delta=np.asarray(scale*z).tolist(), model_cost_change=float(z@(g+h@z/2)),
                    proposal_merit_change=float(z@(fit_g+fit_h@z/2)),
                    affine_guard_excess=rejected(a@z-b, names),
                    affine_norm_excess=rejected(norm, [ball[-1] for ball in balls]))

    constrained, report, cuts = refine(balls)
    sample_replays=[]
    if (sample_caps or phase is not None) and report['max_constraint_excess']<=1.e-7:
        guard=problem.scoped_surface_trial
        for _ in range(4):
            trial=x.copy();trial[selected]+=constrained*scale
            failures=[]
            if phase is not None:
                failures,coverage=guard.screen_proposal(trial)
            if sample_caps:
                scan,coverage=guard.scan_trial(trial)
                failures.extend(local_surface_regressions(scan,guard.surface.geometry,sample_caps))
            sample_replays.append(dict(failures=failures,**coverage))
            emit('scoped_proposal_surface_refresh',failures=failures,**coverage)
            if not failures or report['max_constraint_excess']>1.e-7:
                break
            candidates.append(trial)
            surface,a,b,names=refreshed_surface_domains(candidates)
            constrained,report,new_cuts=refine(balls)
            cuts.extend(new_cuts)
        write_json(output/('phase_surface_trials.json' if phase is not None else 'scoped_collar_trials.json'),sample_replays)
    directions = dict(normal_unconstrained=raw, source_unconstrained=hints, euclidean_projection=euclid,
                      incomplete_metric=global_only, constrained_metric=constrained)
    report_ablation=None
    ablation_cuts=[]
    if spec.get('run_comparison_trials',True):
        ablation_balls = [entry for entry in balls if not entry[-1].startswith('__flight__/episode/entry_flight_core/')]
        ablation,report_ablation,ablation_cuts=refine(ablation_balls)
        directions['diagnostic_without_entry_flight']=ablation
    receipt = dict(target_sha256=spec['sha256'], physical_columns=coeffs, column_scale=scale.tolist(),
                   trials={k:proposal_receipt(z) for k,z in directions.items()},
                   cavity_proposal_reserve=spec.get('cavity_proposal_reserve',{}),
                   solvers=dict(normal=raw_report, euclidean=report_euclid, incomplete_metric=report_global,
                                constrained=report,diagnostic_without_entry_flight=report_ablation),
                   nonlinear_local_constraints=dict(constrained=cuts, diagnostic_without_entry_flight=ablation_cuts))
    write_json(output/'scoped_directions.json', receipt)
    np.savez_compressed(output/'scoped_linearization.npz', x=x, columns=selected, weighted_columns=weighted_columns,
                       hessian=h, gradient=g, target_hessian=fit_h, target_gradient=fit_g,
                       column_scale=scale, residual_row_numbers=np.flatnonzero(used))
    emit('scoped_metric_solved', **report)
    if report['max_constraint_excess'] > 1.e-7 or (phase is not None and not report['success']):
        raise ArithmeticError('Scoped metric solver could not satisfy the proposal domains')
    return selected, directions, scale, measurements


def trial_receipt(problem, base, selected, directions, scale, measurements, output, spec):
    """Use production nonlinear factors on the same revised observations."""
    from .absolute_motion_curve import RotationChartDomainError
    from .rebuild_exposure_body import MovingBodyProjectionDomainError
    from .fullbody_round import write_json
    from .runner_interior import RunnerCoordinateDomainError

    x = problem.cache_x.copy()
    frames, joints, uv, jac, ts, track, vector, rotation_j = measurements
    base_cost = float(base['residual']@base['residual']/2)
    original = {b.name:b.residual.copy() for b in base['weighted_blocks']
                if b.name.startswith('episode/entry_flight_core/')}
    returned = []
    for key in ('constrained_metric', 'normal_unconstrained', 'euclidean_projection', 'diagnostic_without_entry_flight'):
        if key!='constrained_metric' and 'diagnostic_without_entry_flight' not in directions:
            continue
        z = x.copy()
        z[selected] += scale*directions[key]
        problem.log.write(json.dumps(dict(event='scoped_exact_trial_start', trial=key))+'\n')
        try:
            trial = problem.evaluate(z, False)
        except (RotationChartDomainError, MovingBodyProjectionDomainError, RunnerCoordinateDomainError) as error:
            record = dict(domain_error=str(error), domain=type(error).__name__)
        else:
            guard = problem.additional_trial_quality(trial, .05)
            guard.update(problem.full_candidate_guard(z, trial))
            failures = {k:v for k,v in guard.items() if v['rejected']}
            groups = {k:v for k,v in guard.items() if k.startswith('__flight__/episode/entry_flight_core/')}
            current = {b.name:b.residual for b in trial['weighted_blocks'] if b.name in original}
            contributions = {name: sorted(enumerate((value*value-original[name]**2).tolist()),
                                           key=lambda e: abs(e[1]), reverse=True)[:12]
                             for name,value in current.items()}
            record = dict(cost_before=base_cost, cost_after=float(trial['residual']@trial['residual']/2),
                          source_pts=problem.reference.p['pts'][frames].tolist(),
                          projected_uv=source_projection(problem,z,frames,joints).tolist(),
                          rotation_step_deg=np.rad2deg(np.linalg.norm(relative_rotations(
                              problem.body.state,z,ts,track)[0],axis=1)).tolist(),
                          failed_guards=failures, entry_flight_norms=groups, flight_row_squared_changes=contributions)
            shoe_regions=spec.get('protected_skate_regions',[])
            violation=False
            if shoe_regions:
                uv,_=skate_projection(problem,z,shoe_regions)
                lo,hi=np.asarray([r['bounds_uv'] for r in shoe_regions]).swapaxes(0,1)
                excess=np.linalg.norm(np.maximum(np.maximum(lo-uv,uv-hi),0.),axis=1)
                violation=bool(np.any(excess>1.e-5))
                record['protected_skate_projections']=[dict(**row,projected_uv=point.tolist(),excess_px=float(e))
                    for row,point,e in zip(shoe_regions,uv,excess,strict=True)]
            limits=spec.get('local_surface_caps',[])
            if limits and key=='constrained_metric':
                from .scoped_motion_domains import local_surface_regressions
                surface=problem.scoped_surface_trial
                scan,_=surface.scan_trial(z)
                regressions=local_surface_regressions(scan,surface.surface.geometry,limits)
                record['reviewed_collar_failures']=regressions
                violation=violation or bool(regressions)
            if key == 'constrained_metric':
                if not violation:
                    returned.append((key, z))
                np.savez_compressed(output/'scoped_proposal_vector.npz', x=z, parent_x=x, columns=selected)
        problem.log.write(json.dumps(dict(event='scoped_exact_trial', trial=key, **record))+'\n')
        write_json(output/('exact_'+key+'.json'), record)
    if not returned:
        raise ArithmeticError('Scoped metric proposal did not reach the nonlinear factor domain')
    return tuple(returned)
