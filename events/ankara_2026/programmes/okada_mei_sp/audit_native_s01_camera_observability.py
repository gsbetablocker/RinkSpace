"""Read-only, undamped camera sensitivity after profiling nuisance coordinates.

Uses every actual native row Jacobian, without modifying the solver or its state.
The CPU eigendecompositions below are an engineering diagnostic, not a fit.
Physical inequality boundaries are deliberately not asserted to be measurements.
"""
import time
import numpy as np


CAMERA_NAMES = (
    'R0_x', 'R0_y', 'R0_z', 'pan_axis_tangent1', 'pan_axis_tangent2',
    'tilt_axis', 'pivot_x', 'pivot_y', 'pivot_z', 'lever_x', 'lever_y',
    'lever_z', 'axis_spacing', 'pupil1', 'pupil2', 'cx', 'cy', 'eta',
    'cx_zoom', 'cy_zoom', 'k10', 'k11', 'k20', 'k21', 'p1', 'p2',
    'readout', 'wide_k1', 'wide_k2', 'wide_p1', 'wide_p2')


def symmetric(a):
    return (a+a.T)*.5


def profile_camera(normal, camera_columns, names):
    """Column-normalized undamped Gram matrix and its nuisance Schur complement."""
    scale = np.sqrt(np.maximum(np.diag(normal), 0.))
    live = scale > 0
    divisor = np.where(live, scale, 1.)
    gram = symmetric(normal/divisor[:, None]/divisor[None, :])
    nuisance = np.setdiff1d(np.flatnonzero(live), camera_columns)
    cc = gram[np.ix_(camera_columns, camera_columns)]
    nn = gram[np.ix_(nuisance, nuisance)]
    nc = gram[np.ix_(nuisance, camera_columns)]
    values, vectors = np.linalg.eigh(nn)
    cutoff = np.finfo(float).eps*max(nn.shape)*max(float(values[-1]), 1.)*16
    usable = values > cutoff
    if values[0] < -cutoff:
        raise ValueError('Undamped nuisance Gram matrix is not positive semidefinite')
    solution = vectors[:, usable] @ ((vectors[:, usable].T@nc)/values[usable, None])
    profiled = symmetric(cc-nc.T@solution)
    eigenvalues, eigenvectors = np.linalg.eigh(profiled)
    if eigenvalues[0] < -cutoff*4:
        raise ValueError('Profiled camera information has material negative curvature')
    return dict(scale=scale, gram=gram, nuisance=nuisance, solution=solution,
                profiled=profiled, eigenvalues=eigenvalues, eigenvectors=eigenvectors,
                nuisance_eigenvalues=values,nuisance_eigenvectors=vectors,
                nuisance_usable=usable,
                summary=dict(nuisance_columns=len(nuisance),
                    nuisance_numerical_null_count=int((~usable).sum()),
                    numerical_eigenvalue_cutoff=float(cutoff),
                    nuisance_min_max_eigenvalues=[float(values[0]), float(values[-1])],
                    camera_profile_eigenvalues=eigenvalues.tolist(),
                    individual_camera_columns=[dict(parameter=name,
                        raw_derivative_l2=float(scale[col]),
                        remaining_squared_derivative_fraction=float(profiled[i,i]),
                        cancellation_fraction=float(1-profiled[i,i]))
                        for i,(col,name) in enumerate(zip(camera_columns, names))]))


def audit(resident, observations, evaluation, motion_columns, camera_size,
          column_names, out, write, emit, context=None):
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    q = resident._q
    normal = np.zeros((q,q))
    raw_normal = np.zeros((q,q))
    gradient = np.zeros(q)
    zero_station_derivatives = 0
    endpoint_rows = 0
    pts = observations['pts']
    frames = np.unique(pts)
    camera_columns = np.arange(motion_columns,motion_columns+camera_size)
    camera_names = CAMERA_NAMES[:camera_size]
    max_symmetry = 0.
    for frame_index, pt in enumerate(frames):
        rows = np.flatnonzero(pts==pt)
        raw = resident.probe_rows(rows)
        jac = raw[:,:,:q].copy()
        tangent = raw[:,:,q]
        norm2 = np.einsum('ij,ij->i',tangent,tangent)
        curve = observations['kind'][rows]==1
        zero_station_derivatives += int(np.count_nonzero(curve & (norm2==0)))
        # Every curve station is profiled as locally free for this information
        # audit, even at an endpoint. The actual fit retains its finite bounds.
        free = curve & (norm2>0)
        station = observations['numeric_stations'][rows]
        endpoint_rows += int(np.count_nonzero(curve & ((station<=0)|(station>=1))))
        columns = np.flatnonzero(np.any(jac!=0,axis=(0,1)))
        small = jac[:,:,columns]
        flat = small.reshape(-1,len(columns))
        raw_normal[np.ix_(columns,columns)] += flat.T@flat
        dot = np.einsum('ij,ijk->ik',tangent[free],small[free])
        small[free] -= tangent[free,:,None]*(dot/norm2[free,None])[:,None,:]
        flat = small.reshape(-1,len(columns))
        piece = flat.T@flat
        max_symmetry=max(max_symmetry,float(np.max(abs(piece-piece.T))))
        normal[np.ix_(columns,columns)] += piece
        gradient[columns] += flat.T@evaluation['values'][rows,9:11].ravel()
        if frame_index%100==0 or frame_index==len(frames)-1:
            emit('camera_information_rows',frames_done=frame_index+1,
                 frames_total=len(frames),audit_elapsed_seconds=time.monotonic()-started)
    result = profile_camera(normal,camera_columns,camera_names)
    modes = []
    for k in range(min(8,camera_size)):
        vector = result['eigenvectors'][:,k]
        order = np.argsort(abs(vector))[::-1][:8]
        cancel = -result['solution']@vector
        nuisance_order = np.argsort(abs(cancel))[::-1][:8]
        modes.append(dict(eigenvalue=float(result['eigenvalues'][k]),
            camera=[dict(parameter=camera_names[i],coefficient=float(vector[i])) for i in order],
            compensating_nuisance=[dict(parameter=column_names[result['nuisance'][i]],
                coefficient=float(cancel[i])) for i in nuisance_order]))
    np.savez(out/'undamped_information.npz',normal=normal,raw_normal=raw_normal,
        gradient=gradient,camera_columns=camera_columns,column_scale=result['scale'],
        nuisance_columns=result['nuisance'],nuisance_solution=result['solution'],
        profiled_camera=result['profiled'],eigenvalues=result['eigenvalues'],
        eigenvectors=result['eigenvectors'])
    report=dict(schema='rinkspace.native_camera_observability.v1',
        context=context or {},
        rows=len(pts),frames=len(frames),retained_columns=q,camera_columns=camera_size,
        all_rows_native_jacobian=True,optimizer_steps=0,damping=0.,
        observation_weights=False,parameter_priors=False,
        inequality_boundaries_used_as_measurements=False,
        curve_station_endpoint_rows_profiled_as_free=endpoint_rows,
        zero_curve_station_derivative_rows=zero_station_derivatives,
        native_corner_station_tangent_selection_retained=True,
        method='Column-normalized data-only Gauss-Newton Gram; nonzero native curve tangents removed; all motion and scene columns profiled by symmetric pseudoinverse. Native local polygon-kink minima may return a zero station tangent, which this audit retains.',
        limitation='Local first-order ambiguity diagnostic at the current native station-side choices, not a global optimum, calibrated covariance, or proof that each compensating direction satisfies physical bounds.',
        column_names=column_names,soft_camera_modes=modes,
        zero_information_columns=[column_names[i] for i in np.flatnonzero(result['scale']==0)],
        nuisance_numerical_null_modes=[dict(eigenvalue=float(result['nuisance_eigenvalues'][k]),
            components=[dict(parameter=column_names[result['nuisance'][i]],
                coefficient=float(result['nuisance_eigenvectors'][i,k]))
                for i in np.argsort(abs(result['nuisance_eigenvectors'][:,k]))[-8:][::-1]])
            for k in np.flatnonzero(~result['nuisance_usable'])],
        max_accumulation_asymmetry=max_symmetry,elapsed_seconds=time.monotonic()-started,
        **result['summary'])
    write(out/'report.json',report)
    emit('camera_information_complete',report=str(out/'report.json'),
         audit_elapsed_seconds=time.monotonic()-started,
         camera_profile_eigenvalues=result['eigenvalues'].tolist())
    return report
