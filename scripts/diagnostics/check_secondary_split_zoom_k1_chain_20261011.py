"""Read-only split source/target k1 tangent check; stdout, no fit or caches.

The prospective coefficient is k1_i = k1 + slope *
(native_logf_i - log(1920) - original_q_reference).  This script checks its
raw fixed-chart derivative, then the original frozen-profile Schur projection.
It does not implement or claim a new nonlinear depth-profiling kernel.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import time

import replay_cut6_F3_existing_RF4_source_response_20261011 as s
from calibration.solver_runtime.legacy.secondary_optics_candidate.z190_motion_optics_extension import (
    _line_geometry, _raw_optics, _weights, native_motion_with_optics,
)
from z190_moving_center_motion import _inverse_brown
from z190_motion_native import native_motion
from z190_fixed_lever_xyz_source_bundle import FixedLeverXYZSourceBundle

np, cv2 = s.np, s.cv2
Q = 'f9d83084051c0c5498ed01646f2fd4b8ef64dd15e46f8ec471eb5fba5ec97496'
DIRECTORY = s.R / 'okada_review86_q091f_coupled_pivot_source_SOC_20261011/accepted_states' / Q


def original_inputs(state, value, data, border):
    """Same original RF4 cache and complete 65536-point input batch."""
    block = state.bypts[s.PTS]
    calibration = block.bundle.calibration
    if any(calibration.axial):
        raise ValueError('This original fixed-XYZ cache requires zero axial')
    local = np.r_[np.asarray(calibration.pivot_xyz)/100., 0.,
                  value[block.start:block.start+4*len(data['stamps'])]]
    owner = SimpleNamespace(calibration=calibration, _check_shared=lambda q: None)
    vv, rr, dr, cc, ds, dv = FixedLeverXYZSourceBundle.cache(owner, local, True)
    cameras = np.asarray([s.camera(state, value, int(p)) for p in data['stamps']])
    np.testing.assert_array_equal(cc, cameras[:, :3])
    np.testing.assert_array_equal(vv[:, 3], cameras[:, 6])
    np.testing.assert_array_equal(rr, np.asarray([cv2.Rodrigues(c[3:6])[0] for c in cameras]))
    edge = border[:-1] if np.array_equal(border[0], border[-1]) else border
    planes = np.zeros((len(data['a']), 4))
    planes[data['finite_mask'] & ~data['board']] = [0., 0., 1., 0.]
    for view in np.unique(data['ia'][data['board']]):
        ids = np.flatnonzero(data['board'] & (data['ia'] == view))
        selector = SimpleNamespace(_facet_course_cache=None, records=[None]*len(ids),
            source_row_groups={0: np.arange(len(ids))}, source_cameras={0: cameras[view]},
            source_uv=data['a'][ids], exact_cache_statistics=dict(facet_group_hits=0, facet_group_misses=0))
        course = SimpleNamespace(sample_perimeter=lambda q, with_jacobian=False: (border,))
        facets = s.OriginalMotionFarBoardRoleBinding._current_source_facets(
            selector, course, None, ('fixed_current_course', s.token(border)))
        zero = np.zeros((3, 0))
        planes[ids] = [s.vertical_plane_value_jacobian(edge[i], edge[(i+1)%len(edge)], zero, zero)[0]
                       for i in facets]
    ia, ib = data['ia'], data['ib']
    inputs = dict(a=data['a'], b=data['b'], fa=np.exp(vv[ia, 3]), fb=np.exp(vv[ib, 3]),
        ra=rr[ia], rb=rr[ib], da=dr[ia], db=dr[ib], ca=cc[ia], cb=cc[ib],
        dca_shared=ds[ia], dcb_shared=ds[ib], dca_view=dv[ia], dcb_view=dv[ib],
        pp=cameras[0, 8:10], k1=cameras[0, 11], planes=planes,
        sigma=data['motion_sigma'].reshape(-1, 2), profile_weight=data['profile'], jacobian=True)
    x = vv[:, 3] - np.log(1920.) - calibration.q_reference
    return inputs, x[ia], x[ib]


def physical_branches(k, trace, floor):
    known = trace[:, 12] != 0
    interior = trace[:, 9] != 0
    # A failed final Newton search can leave an unknown, nonstationary
    # interior point.  It is still the same line chart, never rho=0.
    physical = known | (trace[:, 10] != 0) | (trace[:, 11] != 0)
    floor_rows = (~known) & (trace[:, 10] != 0) & (trace[:, 0] > 1e-14)
    if floor_rows.any() and floor <= 0:
        raise ValueError('Variable upper positive-depth limit is not a fixed physical rho')
    # Here floor > 0, qz > 0, so a finite positive upper endpoint is the
    # original source-depth floor. Infinity/unidentified rows use rho=0.
    return known, interior, physical, floor_rows


def split_raw(k, trace, wa, wb, floor):
    geometry = _line_geometry(k, trace[:, 0])
    target = geometry['fb'][:, None] * geometry['point'] * geometry['radius2'][:, None]
    shared = _raw_optics(k, trace[:, 0], geometry)[:, :, 2]
    raw = (shared-target)*wa[:, None] + target*wb[:, None]
    known, interior, physical, floor_rows = physical_branches(k, trace, floor)
    na = geometry['na']; r2 = np.sum(na*na, axis=1); coeff = float(k['k1'])
    a = 1.+coeff*r2+2.*coeff*na[:, 0]**2
    b = 2.*coeff*na[:, 0]*na[:, 1]
    c = 1.+coeff*r2+2.*coeff*na[:, 1]**2
    determinant = a*c-b*b
    dna = -np.c_[c*na[:, 0]-b*na[:, 1], a*na[:, 1]-b*na[:, 0]]*r2[:, None]/determinant[:, None]
    world = np.einsum('ni,nij->nj', np.c_[na, np.ones(len(na))], k['ra'])
    dworld = np.einsum('ni,nij->nj', np.c_[dna, np.zeros(len(na))], k['ra'])
    dq = np.einsum('nij,nj->ni', k['rb'], dworld)
    pd = -k['planes'][:, 3]-np.sum(k['planes'][:, :3]*k['ca'], axis=1)
    rho = np.zeros(len(na)); drho = np.zeros(len(na))
    rho[known] = np.sum(k['planes'][known, :3]*world[known], axis=1)/pd[known]
    drho[known] = np.sum(k['planes'][known, :3]*dworld[known], axis=1)/pd[known]
    rho[floor_rows] = 1./floor
    Y = geometry['q'] + rho[:, None]*geometry['t']
    n = Y[:, :2]/Y[:, 2, None]
    source_Y = (dq+drho[:, None]*geometry['t'])*wa[:, None]
    dn = (source_Y[:, :2]-n*source_Y[:, 2, None])/Y[:, 2, None]
    r2 = np.sum(n*n, axis=1)
    source = geometry['fb'][:, None]*((1.+coeff*r2)[:, None]*dn
        + 2.*coeff*np.sum(n*dn, axis=1)[:, None]*n)
    target = geometry['fb'][:, None]*n*r2[:, None]*wb[:, None]
    raw[physical] = (source+target)[physical]
    return raw, geometry, interior


def profiled_split_column(k, trace, wa, wb, floor, tangent):
    """Original Schur/implicit/physical dispatch, including mixed curvature."""
    raw, geometry, interior = split_raw(k, trace, wa, wb, floor)
    W = _weights(k, len(wa))
    column = raw.copy()
    factor = np.sum(W*tangent*raw, axis=1)/np.maximum(trace[:, 3], 1e-30)
    column[interior] -= (tangent*factor[:, None])[interior]
    implicit = ((trace[:, 12] == 0) & (trace[:, 10] == 0)
                & (trace[:, 11] == 0) & (trace[:, 4] > 1e-22) & ~interior)
    if implicit.any():
        na = geometry['na']; coeff = float(k['k1']); source_r2 = np.sum(na*na, axis=1)
        a = 1.+coeff*source_r2+2.*coeff*na[:, 0]**2
        b = 2.*coeff*na[:, 0]*na[:, 1]
        c = 1.+coeff*source_r2+2.*coeff*na[:, 1]**2
        dna = -np.c_[c*na[:, 0]-b*na[:, 1], a*na[:, 1]-b*na[:, 0]]*source_r2[:, None]/(a*c-b*b)[:, None]
        dw = np.einsum('ni,nij->nj', np.c_[dna, np.zeros(len(na))], k['ra'])*wa[:, None]
        dq = np.einsum('nij,nj->ni', k['rb'], dw)
        nq, u, q = geometry['nq'], geometry['u'], geometry['q']
        dnq = (dq[:, :2]-nq*dq[:, 2, None])/q[:, 2, None]
        dh = -dnq*geometry['t'][:, 2, None]
        dl = np.sum(u*dh, axis=1)
        du = (dh-u*dl[:, None])/np.maximum(geometry['length'], 1e-30)[:, None]
        dn = dnq+trace[:, 0, None]*du
        n = geometry['point']; D = geometry['D']
        mixed = geometry['fb'][:, None]*(np.einsum('nij,nj->ni', D, du)
            + 2.*coeff*(np.sum(n*dn, axis=1)[:, None]*u
                        +np.sum(n*u, axis=1)[:, None]*dn
                        +np.sum(u*dn, axis=1)[:, None]*n)
            +wb[:, None]*(np.sum(n*n, axis=1)[:, None]*u+2.*np.sum(n*u, axis=1)[:, None]*n))
        residual = (geometry['fb'][:, None]*n*geometry['radial'][:, None]+k['pp']-k['b'])
        dlam = -(np.sum(W*tangent*raw, axis=1)+np.sum(W*residual*mixed, axis=1))/np.maximum(trace[:, 4], 1e-30)
        column[implicit] += (tangent*dlam[:, None])[implicit]
    return column, dict(schur_rows=int(interior.sum()), nonstationary_original_implicit_rows=int(implicit.sum()),
                       no_unknown_failed_row_reinterpreted_as_physical_rho_zero=True)


def fixed_chart_value(k, trace, wa, wb, amount, floor, *, with_line_tangent=False):
    """Physical FD: original full batch inverse; nominal lambda or true rho.

    The existing NumPy inverse accepts per-row coefficients by broadcasting.
    No observed UV, sigma, profile or source-role selection is transformed.
    """
    n = len(k['a']); ka = float(k['k1'])+amount*wa; kb = float(k['k1'])+amount*wb
    na, _ = _inverse_brown(k['a'], k['fa'], k['pp'], ka)
    world = np.einsum('ni,nij->nj', np.c_[na, np.ones(n)], k['ra'])
    q = np.einsum('nij,nj->ni', k['rb'], world)
    t = np.einsum('nij,nj->ni', k['rb'], k['ca']-k['cb'])
    nq = q[:, :2]/q[:, 2, None]; h = t[:, :2]-nq*t[:, 2, None]
    length = np.sqrt(h[:, 0]**2+h[:, 1]**2); u = h/np.maximum(length, 1e-30)[:, None]
    point = nq+trace[:, 0, None]*u
    known, interior, physical, floor_rows = physical_branches(k, trace, floor)
    rho = np.zeros(n); pd = -k['planes'][:, 3]-np.sum(k['planes'][:, :3]*k['ca'], axis=1)
    rho[known] = np.sum(k['planes'][known, :3]*world[known], axis=1)/pd[known]
    rho[floor_rows] = 1./floor
    Y = q+rho[:, None]*t
    point[physical] = (Y[:, :2]/Y[:, 2, None])[physical]
    radius2 = np.sum(point*point, axis=1)
    residual = k['fb'][:, None]*point*(1.+kb*radius2)[:, None]+k['pp']-k['b']
    if with_line_tangent:
        tangent = k['fb'][:, None]*((1.+kb*radius2)[:, None]*u
            +2.*kb[:, None]*np.sum(point*u, axis=1)[:, None]*point)
        return residual, tangent
    return residual


def main():
    started = time.perf_counter()
    options = s.read(s.CONFIG)['source_options']; manifest = s.read(DIRECTORY.parent.parent/'input_manifest.json')
    state, q, identity, error = s.public.load_state(DIRECTORY, Path(options['--world-init']), manifest)
    if error != 0 or identity['q_sha256'] != Q:
        raise ValueError('Exact f9 public complete state required')
    with np.load(DIRECTORY/'complete_state_UNACCEPTED.npz', allow_pickle=False) as z:
        names = z['aux__block_names'].tolist()
        keys = ['aux__block_names', 'aux__block_lengths', 'aux__canonical_pts',
                'aux__current_depth_and_material_domain_contract_json', 'profile__weights_'+str(names.index('secondary'))]
        saved = {key:z[key].copy() for key in keys}
    data = s.secondary_inputs(manifest, saved)
    course, cq, _ = state.geometry.print_course_binding(q[state.geometry_parameter_indices])
    border = np.asarray(course.sample_perimeter(cq, with_jacobian=False)[0])
    k, wa, wb = original_inputs(state, q, data, border)
    result, check, trace, nuisance = native_motion(**k, minimum_source_depth_scene_units=data['floor'], return_nuisance=True)
    if '--failed-depth-only' in sys.argv:
        bad = (trace[:, 8] == 0) | (trace[:, 7] != 0); ids = np.flatnonzero(bad)
        geometry = _line_geometry(k, trace[:, 0]); W = _weights(k, len(wa))
        qz, tz, L = geometry['q'][:, 2], geometry['t'][:, 2], geometry['length']
        rho_max = 1./data['floor']
        upper = np.minimum(np.where(tz > 0, L/np.where(tz > 0, tz, 1.), np.inf),
            np.where(qz+rho_max*tz > 0, rho_max*L/np.where(qz+rho_max*tz > 0, qz+rho_max*tz, 1.), np.inf))
        shot = s.read(s.ROOT/'calibration/program_tracking/source_shots.json')['shots'][1:10][5]
        cut = (((data['pa'] >= shot['start_pts']) & (data['pa'] < shot['end_pts']))
               | ((data['pb'] >= shot['start_pts']) & (data['pb'] < shot['end_pts'])))
        metadata = dict(q_sha256=Q, complete_state_path=str(DIRECTORY/'complete_state_UNACCEPTED.npz'),
            complete_state_sha256=s.sha(DIRECTORY/'complete_state_UNACCEPTED.npz'),
            script_sha256=s.sha(__file__), profile_sha256=s.token(data['profile']),
            original_39679_batch_reused=True, observed_UV_sigma_roles_unchanged=True)
        path = Path(str(DIRECTORY.parent.parent)+'.bootstrap')/'secondary_profile_failed_rows_f9_readonly.npz'
        np.savez_compressed(path, metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
            original_point_indices=ids, pa=data['pa'][ids], pb=data['pb'][ids], cut6=cut[ids],
            trace=trace[ids], raw=result[0].reshape(-1,2)[ids], a=k['a'][ids], b=k['b'][ids],
            nq=geometry['nq'][ids], u=geometry['u'][ids], q=geometry['q'][ids], t=geometry['t'][ids],
            line_length=L[ids], lambda_upper=upper[ids], fb=geometry['fb'][ids], pp=k['pp'],
            k1=np.array(k['k1']), W=W[ids], sigma=k['sigma'][ids], profile_weight=data['profile'][ids],
            minimum_source_depth_scene_units=np.array(data['floor']))
        s.emit('failed_profile_reuse', path=str(path), bytes=path.stat().st_size, sha256=s.sha(path),
            bad_points=len(ids), cut6_bad_points=int(cut[ids].sum()), no_source_images_written=True)
        return
    actual, original_optics_audit, _ = native_motion_with_optics(**k, minimum_source_depth_scene_units=data['floor'])
    np.testing.assert_array_equal(actual[0], result[0])
    raw_constant, geometry, interior = split_raw(k, trace, np.ones(len(wa)), np.ones(len(wb)), data['floor'])
    raw_slope, _, _ = split_raw(k, trace, wa, wb, data['floor'])
    tangent = nuisance['profile_nuisance_tangent_native']; weights = _weights(k, len(wa))
    projected, dispatch = profiled_split_column(k, trace, np.ones(len(wa)), np.ones(len(wb)), data['floor'], tangent)
    good = (trace[:, 8] != 0) & (trace[:, 7] == 0)
    legacy = actual[-1].reshape(-1, 2, 3)[:, :, 2]/.001
    shot = s.read(s.ROOT/'calibration/program_tracking/source_shots.json')['shots'][1:10][5]
    inside_a = (data['pa'] >= int(shot['start_pts'])) & (data['pa'] < int(shot['end_pts']))
    inside_b = (data['pb'] >= int(shot['start_pts'])) & (data['pb'] < int(shot['end_pts']))
    touching = inside_a | inside_b
    pairs = sorted(set(zip(data['pa'][touching & ~good].tolist(), data['pb'][touching & ~good].tolist())))
    s.emit('cut6_nominal_depth_eligibility', touching_points=int(touching.sum()),
        nonconverged_or_search_failed=int((touching & ~good).sum()),
        finite=int((touching & (trace[:, 12] != 0)).sum()),
        interior=int((touching & (trace[:, 9] != 0)).sum()),
        boundary=int((touching & (trace[:, 10] != 0)).sum()),
        unidentified=int((touching & (trace[:, 11] != 0)).sum()),
        nonconverged_directed_pairs=[dict(pa=pa,pb=pb,points=int(((data['pa']==pa)&(data['pb']==pb)&~good).sum())) for pa,pb in pairs],
        no_original_motion_rows_removed=True)
    s.emit('split_k1_original_constant_chain', q_sha256=Q, current_k1=float(k['k1']),
        complete_points=len(wa), public_cache_camera_bitexact=True,
        original_sigma_profile_and_finite_roles_preserved=True,
        constant_projected_vs_existing_column_max_px=float(abs(projected[good]-legacy[good]).max()),
        original_implicit_branch=dispatch,
        original_optics_fd_branch_changes=original_optics_audit['optics_fd_branch_changes'],
        converged_good_points=int(good.sum()), unqualified_nonconverged_points=int((~good).sum()),
        source_target_logf_coordinate_range=[float(min(wa.min(),wb.min())),float(max(wa.max(),wb.max()))])
    for label, aa, bb, derivative in [('constant',np.ones(len(wa)),np.ones(len(wb)),raw_constant),('k1_logf',wa,wb,raw_slope)]:
        for h in (1e-6, 5e-7):
            fd = (fixed_chart_value(k, trace, aa, bb, h, data['floor'])
                  -fixed_chart_value(k, trace, aa, bb, -h, data['floor']))/(2*h)
            s.emit('split_k1_fixed_chart_FD', direction=label, h=h,
                good_points=int(good.sum()), max_native_pixel_error=float(abs(fd[good]-derivative[good]).max()),
                response_max_px=float(abs(derivative[good]).max()),
                nominal_interior_lambda_fixed=True, finite_and_floor_use_original_physical_rho=True)
    s.emit('split_k1_scope', source_images_written=0, arrays_written=0, fit=False,
        nonlinear_variable_k1_profile_kernel_not_implemented=True,
        no_residual_correlation_or_identifiability_claim=True, elapsed_seconds=time.perf_counter()-started)


if __name__ == '__main__':
    main()
