"""One axial(logf) or PPy(logf) motion column, frozen state, stdout only.

The coefficient is working metres per log-focal unit. Both source and target
optical centres move through the public world-to-camera convention. Every
trial reuses the full original secondary batch and profiles its bounded depth
again. No parameter is estimated, and no image, matrix or state is written.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import time

import check_secondary_split_zoom_k1_chain_20261011 as split
from calibration.solver_runtime.source_loss import source_irls_weights, weighted_source_cost
from z190_motion_native import native_motion

np, s = split.np, split.s
Q = '11c84ac8596e4e14f42268051b68d3128d8bc9c0a909bf4366bc52171988b36b'
DIRECTORY = s.R / 'okada_review86_q091f_photo_mount_box_release_20261011/accepted_states' / Q


def emit(stage, **values):
    s.emit(stage, **values)


def centre_derivatives(inputs, source_logf_coordinate, target_logf_coordinate, scene_unit_m):
    """dC/da = R.T e_z * x / scene_unit_m, for BOTH directed endpoints."""
    unit = float(scene_unit_m)
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError('Positive original working metres per scene unit required')
    n = len(inputs['a'])
    xa, xb = [np.asarray(x, float) for x in (source_logf_coordinate, target_logf_coordinate)]
    if xa.shape != (n,) or xb.shape != (n,) or not np.isfinite(xa).all() or not np.isfinite(xb).all():
        raise ValueError('Every original directed endpoint needs its own log-focal coordinate')
    # Row 2 of the world->camera R is the world direction R.T @ e_z.
    da = np.asarray(inputs['ra'], float)[:, 2, :] * xa[:, None] / unit
    db = np.asarray(inputs['rb'], float)[:, 2, :] * xb[:, None] / unit
    return da, db


def load_axial_motion_context(directory=DIRECTORY, *, world_init=None):
    """Small public camera/world owner + authenticated original 39679 arrays.

    This does not construct the programme source graph or any native pixel
    source factor. The saved accepted camera is the numerical authority.
    """
    directory = Path(directory).resolve()
    manifest = s.read(directory.parent.parent / 'input_manifest.json')
    if world_init is None:
        world_init = Path(s.read(s.CONFIG)['source_options']['--world-init'])
    world_init = Path(world_init).resolve()
    state, value, identity, camera_error = s.public.load_state(directory, world_init, manifest)
    if camera_error != 0:
        raise ValueError('Exact accepted complete state/public camera required')
    path = directory / 'complete_state_UNACCEPTED.npz'
    if s.sha(path) != identity['state_npz_sha256']:
        raise ValueError('Accepted complete numerical state changed')
    with np.load(path, allow_pickle=False) as archive:
        if s.token(archive['q']) != identity['q_sha256'] or not np.array_equal(
                state.evaluation_q(archive['q']), value):
            raise ValueError('Canonical accepted q and actual evaluation permutation differ')
        ordinal = archive['aux__block_names'].tolist().index('secondary')
        keys = ('aux__block_names', 'aux__block_lengths', 'aux__canonical_pts',
                'aux__current_depth_and_material_domain_contract_json', 'profile__weights_' + str(ordinal))
        saved = {key: archive[key].copy() for key in keys}
    data = s.secondary_inputs(manifest, saved)
    course, cq, _ = state.geometry.print_course_binding(value[state.geometry_parameter_indices])
    border = np.asarray(course.sample_perimeter(cq, with_jacobian=False)[0])
    inputs, xa, xb = split.original_inputs(state, value, data, border)
    block = state.bypts[s.PTS]
    if any(block.bundle.calibration.axial):
        raise ValueError('Nominal accepted model must have zero original axial coefficients')
    unit = float(state.geometry.scene_unit_m)
    dca, dcb = centre_derivatives(inputs, xa, xb, unit)
    shots = s.read(s.ROOT / 'calibration/program_tracking/source_shots.json')['shots'][1:10]
    cut_masks = {}
    for number, shot in enumerate(shots, 1):
        inside_a = (data['pa'] >= int(shot['start_pts'])) & (data['pa'] < int(shot['end_pts']))
        inside_b = (data['pb'] >= int(shot['start_pts'])) & (data['pb'] < int(shot['end_pts']))
        if np.any(inside_a | inside_b):
            cut_masks[str(number)] = inside_a | inside_b
    covered = np.logical_or.reduce(list(cut_masks.values()))
    if not covered.all() or len(data['a']) != 39679:
        raise ValueError('Every original secondary point must belong to the existing shot inventory')
    context = SimpleNamespace(state=state, q=value, identity=identity, manifest=manifest,
        directory=directory, state_path=path, world_init=world_init, inputs=inputs, data=data,
        xa=xa, xb=xb, dca=dca, dcb=dcb, unit=unit, cut_masks=cut_masks, border=border,
        cameras=np.asarray([s.camera(state, value, int(p)) for p in data['stamps']]),
        q_reference=float(block.bundle.calibration.q_reference), parameter_units='working_metres_per_logf')
    planes, facets = _axial_source_planes(context, 0.)
    np.testing.assert_array_equal(planes, inputs['planes'])
    context.nominal_facets = facets
    for key in ('a', 'b'):
        context.inputs[key].setflags(write=False)
    context.original_uv_tokens = {key: s.token(inputs[key]) for key in ('a', 'b')}
    return context


def _axial_source_planes(context, amount):
    """Original dynamic source-ray facet selection with the actual trial C."""
    c = context
    cameras = c.cameras.copy()
    x = cameras[:, 6] - np.log(1920.) - c.q_reference
    rotations = np.asarray([s.cv2.Rodrigues(row[3:6])[0] for row in cameras])
    cameras[:, :3] += float(amount) * rotations[:, 2, :] * x[:, None] / c.unit
    return _source_planes_for_cameras(c, cameras)


def _source_planes_for_cameras(context, cameras):
    """Use ORIGINAL source UV and actual current camera25, never shifted UV."""
    c = context
    planes = c.inputs['planes'].copy()
    facets = np.full(len(c.data['a']), -1, int)
    edge = c.border[:-1] if np.array_equal(c.border[0], c.border[-1]) else c.border
    zero = np.zeros((3, 0))
    for view in np.unique(c.data['ia'][c.data['board']]):
        ids = np.flatnonzero(c.data['board'] & (c.data['ia'] == view))
        selector = SimpleNamespace(_facet_course_cache=None, records=[None] * len(ids),
            source_row_groups={0: np.arange(len(ids))}, source_cameras={0: cameras[view]},
            source_uv=c.data['a'][ids], exact_cache_statistics=dict(facet_group_hits=0, facet_group_misses=0))
        course = SimpleNamespace(sample_perimeter=lambda q, with_jacobian=False: (c.border,))
        selected = s.OriginalMotionFarBoardRoleBinding._current_source_facets(
            selector, course, None, ('fixed_current_course', s.token(c.border)))
        facets[ids] = selected
        planes[ids] = [s.vertical_plane_value_jacobian(edge[i], edge[(i + 1) % len(edge)], zero, zero)[0]
                       for i in selected]
    return planes, facets


def ppy_model_coordinates(context, amount):
    """Model-coordinate equivalence of PPy_i = PPy_0 + amount * x_i.

    The returned a/b are disposable coordinate copies. Original observations
    and their source identities stay unchanged, including in facet selection.
    """
    c = context
    amount = float(amount)
    if not np.isfinite(amount):
        raise ValueError('Finite native px/logf PPy coefficient required')
    a, b = c.inputs['a'].copy(), c.inputs['b'].copy()
    a[:, 1] -= amount * c.xa
    b[:, 1] -= amount * c.xb
    cameras = c.cameras.copy()
    x = cameras[:, 6] - np.log(1920.) - c.q_reference
    cameras[:, 9] += amount * x
    return a, b, cameras


def check_ppy_model_coordinate_equivalence(context, *, amount=1.):
    """Independent public camera25 ray and forward-projection crosscheck."""
    from calibration.program_tracking.projection import world_rays, project_central_valid
    c = context
    a, b, cameras = ppy_model_coordinates(c, amount)
    np.testing.assert_array_equal(c.cameras[:, 8:10], np.broadcast_to(c.inputs['pp'], (len(c.cameras), 2)))
    np.testing.assert_array_equal(c.cameras[:, 6], c.cameras[:, 7])
    if np.any(c.cameras[:, 10] != 0.) or np.any(c.cameras[:, 12:] != 0.):
        raise ValueError('Original square-focal, zero-skew, k1-only model required')
    ray_max = residual_max = forward_shift_max = roundtrip_max = 0.
    checked = 0
    for original, model, view_ids, xpoint in ((c.inputs['a'], a, c.data['ia'], c.xa),
                                             (c.inputs['b'], b, c.data['ib'], c.xb)):
        for view in np.unique(view_ids):
            ids = np.flatnonzero(view_ids == view)
            camera, trial = c.cameras[view], cameras[view]
            rays, valid = world_rays(trial, original[ids])
            model_rays, model_valid = world_rays(camera, model[ids])
            np.testing.assert_array_equal(valid, model_valid)
            if not valid.all():
                raise ValueError('Public inverse domain failed: no original point may be dropped')
            ray_max = max(ray_max, float(abs(rays - model_rays).max()))
            xyz = camera[:3] + 10. * rays
            actual_uv, _, good = project_central_valid(trial, xyz)
            model_uv, _, model_good = project_central_valid(camera, xyz)
            if not good.all() or not model_good.all():
                raise ValueError('Public central forward projection failed')
            residual_max = max(residual_max, float(abs(
                (actual_uv - original[ids]) - (model_uv - model[ids])).max()))
            expected = np.zeros((len(ids), 2)); expected[:, 1] = float(amount) * xpoint[ids]
            forward_shift_max = max(forward_shift_max, float(abs(actual_uv - model_uv - expected).max()))
            roundtrip_max = max(roundtrip_max, float(abs(actual_uv - original[ids]).max()))
            checked += len(ids)
    if ray_max > 1e-10 or residual_max > 1e-8 or forward_shift_max > 1e-8:
        raise ValueError('Independent PPy model-coordinate equivalence did not hold')
    emit('ppy_public_coordinate_equivalence', coefficient=float(amount), directed_endpoints_checked=checked,
        max_world_ray_difference=ray_max, max_original_residual_difference_px=residual_max,
        max_forward_shift_difference_px=forward_shift_max,
        public_inverse_roundtrip_max_px=roundtrip_max, public_inverse_original_tolerance_px=.005,
        source_and_target_checked=True, original_measurement_uv_unchanged=True,
        finite_facets_use_original_UV_and_modified_public_camera25=True)


def ppy_motion_value(context, amount):
    """Whole original batch, PPy model coordinates and newly profiled depth."""
    c = context
    a, b, cameras = ppy_model_coordinates(c, amount)
    planes, facets = _source_planes_for_cameras(c, cameras)
    inputs = dict(c.inputs, a=a, b=b, planes=planes, jacobian=False)
    result, checks, trace = native_motion(**inputs,
        minimum_source_depth_scene_units=c.data['floor'], return_trace=True)
    for key in ('a', 'b'):
        if c.inputs[key].flags.writeable or s.token(c.inputs[key]) != c.original_uv_tokens[key]:
            raise ValueError('Original observation coordinates must remain read-only and exact')
    sigma = np.asarray(inputs['sigma'])
    return dict(r=result / sigma.ravel(), raw_native_px=result.reshape(-1, 2), trace=trace, checks=checks,
        source_facets=facets, changed_source_facets=facets != c.nominal_facets,
        max_principal_y_shift_native_px=abs(float(amount)) * float(max(abs(c.xa).max(), abs(c.xb).max())))


def ppy_motion_column(context, *, fd_step=.05):
    """ONE actual profiled PPy column from whole-batch h and h/2 stencils.

    This is a numerical derivative of the actual original depth-profile value,
    not an analytic Schur/GN column. Failure rows and changed branches remain.
    """
    c = context
    h = float(fd_step)
    if not np.isfinite(h) or h <= 0:
        raise ValueError('Positive finite native px/logf FD width required')
    nominal = ppy_motion_value(c, 0.)
    original = axial_motion_value(c, 0.)
    np.testing.assert_array_equal(nominal['r'], original['r'])
    np.testing.assert_array_equal(nominal['trace'], original['trace'])
    masks = _masks(c, nominal['trace'])
    derivatives = []
    stable = []
    stencil_receipts = []
    for width in (h, h / 2):
        plus, minus = ppy_motion_value(c, width), ppy_motion_value(c, -width)
        derivative = (plus['raw_native_px'] - minus['raw_native_px']) / (2. * width)
        changed = np.any(plus['trace'][:, 9:13] != nominal['trace'][:, 9:13], axis=1) | np.any(
            minus['trace'][:, 9:13] != nominal['trace'][:, 9:13], axis=1) | plus['changed_source_facets'] | minus['changed_source_facets']
        quality = np.any(plus['trace'][:, 7:9] != nominal['trace'][:, 7:9], axis=1) | np.any(
            minus['trace'][:, 7:9] != nominal['trace'][:, 7:9], axis=1)
        derivatives.append(derivative); stable.append(~changed)
        stencil_receipts.append(dict(coefficient_step=width,
            source_facet_changes_plus_minus=[int(item['changed_source_facets'].sum()) for item in (plus, minus)],
            dispatch_changes=int(changed.sum()), quality_flag_changes=int(quality.sum()),
            all_original_points_retained=True))
    delta = derivatives[1] - derivatives[0]
    for name, mask in masks.items():
        emit('ppy_profile_FD_two_scale', group=name, points=int(mask.sum()),
            h=h, h2=h/2, all_rows_derivative=_stats(derivatives[1], mask),
            all_rows_two_scale_difference=_stats(delta, mask),
            stable_dispatch_two_scale_difference=_stats(delta, mask & stable[0] & stable[1]))
    emit('ppy_profile_FD_stencils', stencils=stencil_receipts,
        original_UV_read_only=True, all_failure_rows_retained=True,
        exact_profile_value_derivative_not_Schur_GN=True)
    sigma = np.asarray(c.inputs['sigma'])
    quadratic = np.zeros(len(nominal['r']), bool)
    return dict(**nominal, Jppy=(derivatives[1]/sigma).ravel(), Jppy_native_px=derivatives[1],
        Jppy_h_native_px=derivatives[0], sigma=sigma, profile=c.data['profile'],
        inner_weights_native=c.data['profile']/(sigma*sigma),
        outer_weights=source_irls_weights(nominal['r'], quadratic, None), quadratic=quadratic,
        source_pts=c.data['pa'], target_pts=c.data['pb'], cut_masks=c.cut_masks,
        finite_mask=c.data['finite_mask'], board_mask=c.data['board'],
        coefficient_units='native_px_per_logf', stencil_receipts=stencil_receipts,
        all_original_rows_retained=True)


def check_ppy_response(context, *, fd_step=.05):
    c = context
    check_ppy_model_coordinate_equivalence(c)
    column = ppy_motion_column(c, fd_step=fd_step)
    emit('ppy_original_motion_column', q_sha256=c.identity['q_sha256'],
        diagnostic_script=dict(path=str(Path(__file__).resolve()), sha256=s.sha(Path(__file__))),
        coefficient_units=column['coefficient_units'], original_q_reference=c.q_reference,
        original_sigma_sha256=s.token(c.inputs['sigma']), profile_sha256=s.token(c.data['profile']),
        original_points=len(c.data['a']), original_production_chunk_size=65536,
        original_branch_counts={k:int(v.sum()) for k,v in _masks(c,column['trace']).items()},
        original_source_target_model_coordinate_transform_not_measurement_revision=True)
    groups = dict(all=np.ones(len(c.data['a']), bool), **{'shot_' + k: v for k,v in c.cut_masks.items()})
    for name, mask in groups.items():
        scalar = np.repeat(mask,2); r=column['r'][scalar]; w=column['outer_weights'][scalar]; j=column['Jppy'][scalar]
        emit('ppy_profile_tangent', group=name, points=int(mask.sum()),
            original_source_cost=weighted_source_cost(r,np.zeros(len(r),bool),None),
            weighted_r_dot_J=float((w*r)@j), weighted_J_norm_squared=float((w*j)@j))
    for amount in (-1.,0.,1.):
        actual=ppy_motion_value(c,amount); masks=_masks(c,actual['trace'])
        costs={name:dict(cost=weighted_source_cost(actual['r'][np.repeat(mask,2)],np.zeros(2*int(mask.sum()),bool),None),
            source_facet_changes=int((actual['changed_source_facets']&mask).sum()),
            branch_counts={k:int((v&mask).sum()) for k,v in masks.items()}) for name,mask in groups.items()}
        emit('ppy_actual_cost', coefficient=amount,
            max_principal_y_shift_native_px=actual['max_principal_y_shift_native_px'],costs=costs,
            fixed_other_parameters=True, all_original_points_retained=True,
            dynamic_source_facets=True, whole_batch_fresh_depth_profile=True,
            complete_truth_and_finite_extent_guards_not_audited=True)
    return column


def axial_motion_column(context, *, schur=True):
    """Return original normalized r and ONE normalized axial column in RAM.

    `profile` is the frozen inner-depth profile, NOT an outer source weight.
    `outer_weights` are the original component Huber2 IRLS weights, applied
    after the once-only sigma normalization. All unknown/failure rows remain.
    The original Schur column and the exact implicit-profile column are distinct.
    """
    c = context
    n = len(c.inputs['a'])
    a_shared = np.zeros((n, 3, 4))
    b_shared = np.zeros_like(a_shared)
    a_shared[:, :, 0], b_shared[:, :, 0] = c.dca, c.dcb
    inputs = dict(c.inputs, jacobian=True, da=np.zeros((3, 3, 3)), db=np.zeros((3, 3, 3)),
        dca_shared=a_shared, dcb_shared=b_shared,
        dca_view=np.zeros((3, 4)), dcb_view=np.zeros((3, 4)))
    kwargs = dict(minimum_source_depth_scene_units=c.data['floor'], schur=bool(schur))
    if schur:
        result, checks, trace, nuisance = native_motion(**inputs, **kwargs, return_nuisance=True)
    else:
        result, checks, trace = native_motion(**inputs, **kwargs, return_trace=True)
        nuisance = None
    raw = result[0].reshape(n, 2)
    column = result[3][:, 0].reshape(n, 2)
    sigma = np.asarray(inputs['sigma'], float)
    normalized = (raw / sigma).ravel()
    normalized_column = (column / sigma).ravel()
    quadratic = np.zeros(len(normalized), bool)
    return dict(r=normalized, Jaxial=normalized_column, raw_native_px=raw,
        Jaxial_native_px=column, sigma=sigma, profile=c.data['profile'],
        inner_weights_native=c.data['profile'] / (sigma * sigma),
        outer_weights=source_irls_weights(normalized, quadratic, None), quadratic=quadratic,
        trace=trace, nuisance=nuisance, checks=checks,
        source_pts=c.data['pa'], target_pts=c.data['pb'], finite_mask=c.data['finite_mask'],
        board_mask=c.data['board'], cut_masks=c.cut_masks,
        parameter_units=c.parameter_units, all_original_rows_retained=True)


def axial_motion_value(context, amount):
    """Finite coefficient response: same complete batch, newly profiled depth."""
    amount = float(amount)
    if not np.isfinite(amount):
        raise ValueError('Finite working-metre/logf coefficient required')
    planes, facets = _axial_source_planes(context, amount)
    inputs = dict(context.inputs, jacobian=False, planes=planes,
        ca=np.asarray(context.inputs['ca']) + amount * context.dca,
        cb=np.asarray(context.inputs['cb']) + amount * context.dcb)
    result, checks, trace = native_motion(**inputs,
        minimum_source_depth_scene_units=context.data['floor'], return_trace=True)
    sigma = np.asarray(inputs['sigma'])
    return dict(r=result / sigma.ravel(), raw_native_px=result.reshape(-1, 2), trace=trace, checks=checks,
        source_facets=facets, changed_source_facets=facets != context.nominal_facets)


def _masks(context, trace):
    known = trace[:, 12] != 0
    interior = trace[:, 9] != 0
    boundary = trace[:, 10] != 0
    unidentified = trace[:, 11] != 0
    implicit = ~known & ~boundary & ~unidentified & (trace[:, 4] > 1e-22) & ~interior
    return dict(all=np.ones(len(trace), bool), finite_board=context.data['board'],
        finite_ice=context.data['finite_mask'] & ~context.data['board'],
        unknown_schur_interior=interior, unknown_nonstationary_implicit=implicit,
        unknown_inverse_depth_zero_endpoint=~known & boundary & (trace[:, 0] <= 1e-14),
        unknown_positive_upper_endpoint=~known & boundary & (trace[:, 0] > 1e-14),
        unknown_unidentified=unidentified,
        unconverged_or_search_failed=(trace[:, 8] == 0) | (trace[:, 7] != 0))


def _stats(values, mask):
    values = np.asarray(values)[mask]
    if not len(values):
        return dict(points=0)
    norm = np.linalg.norm(values, axis=1)
    return dict(points=len(values), max_component=float(abs(values).max()),
        norm_p50_p95_max=np.quantile(norm, [.5, .95, 1.]).tolist(), L2=float(np.linalg.norm(values)))


def check_axial_response(context, *, fd_step=1e-3):
    """Print bounded-profile FD comparisons; return the reusable column dict."""
    h = float(fd_step)
    if not np.isfinite(h) or h <= 0:
        raise ValueError('Positive finite working-metre/logf FD step required')
    c = context
    schur = axial_motion_column(c, schur=True)
    implicit = axial_motion_column(c, schur=False)
    nominal = axial_motion_value(c, 0.)
    np.testing.assert_array_equal(schur['r'], implicit['r'])
    np.testing.assert_array_equal(schur['r'], nominal['r'])
    np.testing.assert_array_equal(schur['trace'][:, :13], implicit['trace'][:, :13])
    np.testing.assert_array_equal(schur['trace'][:, :13], nominal['trace'][:, :13])
    masks = _masks(c, schur['trace'])
    if sum(masks[name].sum() for name in ('finite_board', 'finite_ice', 'unknown_schur_interior',
            'unknown_nonstationary_implicit', 'unknown_inverse_depth_zero_endpoint',
            'unknown_positive_upper_endpoint', 'unknown_unidentified')) != len(c.data['a']):
        raise ValueError('Original branch partition must account for every point, including failures')
    raw_fixed = schur['Jaxial_native_px'].copy()
    interior = masks['unknown_schur_interior']
    nuisance = schur['nuisance']
    raw_fixed[interior] += (nuisance['profile_nuisance_tangent_native'] *
        nuisance['fixed_lambda_projection_factors'][:, 2, 0, None])[interior]
    emit('axial_original_motion_column', q_sha256=c.identity['q_sha256'],
        state=dict(path=str(c.state_path), sha256=s.sha(c.state_path)),
        diagnostic_script=dict(path=str(Path(__file__).resolve()), sha256=s.sha(Path(__file__))),
        original_working_scene_unit_m=c.unit, coefficient_units=c.parameter_units,
        original_q_reference=c.q_reference,
        source_x_range=[float(c.xa.min()), float(c.xa.max())],
        target_x_range=[float(c.xb.min()), float(c.xb.max())],
        batch_points=len(c.data['a']), native_production_chunk_size=65536,
        original_sigma_sha256=s.token(c.inputs['sigma']), profile_sha256=s.token(c.data['profile']),
        branch_counts={key: int(value.sum()) for key, value in masks.items()},
        same_public_nominal_R_C_f_bitexact=True, both_source_target_C_derivatives_present=True,
        old_calibration_and_cache_not_mutated=True, inner_profile_not_outer_counted=True,
        all_rows_in_returned_r_and_Jaxial=True,
        Schur_column_is_original_GN_projection_not_exact_profile_residual_derivative=True)
    for cut, cut_mask in c.cut_masks.items():
        emit('axial_shot_column', shot=cut, directed_pairs=len(set(zip(c.data['pa'][cut_mask], c.data['pb'][cut_mask]))),
            source_point_count=int(cut_mask.sum()), branch_counts={k: int((v & cut_mask).sum()) for k, v in masks.items()},
            native_column_by_branch={k: _stats(schur['Jaxial_native_px'], v & cut_mask) for k, v in masks.items()},
            original_source_cost=weighted_source_cost(schur['r'].reshape(-1, 2)[cut_mask].ravel(),
                np.zeros(2 * int(cut_mask.sum()), bool), None))
    emit('axial_unknown_depth_absorption',
        Schur_interior_fixed_lambda_L2=float(np.linalg.norm(raw_fixed[interior])),
        Schur_interior_remaining_L2=float(np.linalg.norm(schur['Jaxial_native_px'][interior])),
        at_infinite_depth_column=_stats(schur['Jaxial_native_px'], masks['unknown_inverse_depth_zero_endpoint']),
        no_unknown_source_classified_as_known_by_failure_flag=True,
        nonstationary_original_implicit_rows_retained=int(masks['unknown_nonstationary_implicit'].sum()))
    previous_fd = None
    for width in (h, h / 2):
        plus, minus = axial_motion_value(c, width), axial_motion_value(c, -width)
        fd = (plus['raw_native_px'] - minus['raw_native_px']) / (2 * width)
        changed = np.any(plus['trace'][:, 9:13] != schur['trace'][:, 9:13], axis=1) | np.any(
            minus['trace'][:, 9:13] != schur['trace'][:, 9:13], axis=1) | plus['changed_source_facets'] | minus['changed_source_facets']
        quality_changed = np.any(plus['trace'][:, 7:9] != schur['trace'][:, 7:9], axis=1) | np.any(
            minus['trace'][:, 7:9] != schur['trace'][:, 7:9], axis=1)
        records = {}
        for key, mask in masks.items():
            records[key] = dict(all_rows_profile_FD_vs_original_implicit=_stats(fd - implicit['Jaxial_native_px'], mask),
                all_rows_profile_FD_vs_original_Schur=_stats(fd - schur['Jaxial_native_px'], mask),
                same_branch_profile_FD_vs_implicit=_stats(fd - implicit['Jaxial_native_px'], mask & ~changed),
                dispatch_branch_changes=int((changed & mask).sum()), depth_quality_flag_changes=int((quality_changed & mask).sum()))
        emit('axial_whole_batch_profile_FD', coefficient_step=width, branches=records,
            dynamic_source_facet_changes_plus_minus=[int(v['changed_source_facets'].sum()) for v in (plus, minus)],
            dynamic_source_ray_facet_selection_recomputed=True,
            max_C_change_scene_units=float(max(abs(width * c.dca).max(), abs(width * c.dcb).max())),
            normalized_source_cost_plus_minus=[weighted_source_cost(v['r'], schur['quadratic'], None) for v in (plus, minus)],
            all_original_points_retained=True, finite_depths_reprofiled=True,
            two_scale_FD_max_delta=None if previous_fd is None else float(abs(fd - previous_fd).max()),
            complete_truth_and_finite_extent_guards_not_audited=True,
            changed_branches_are_reported_not_dropped_or_relabelled=True)
        previous_fd = fd
    return schur


def compact_axial_cost_response(context, *, amounts=(-.01, 0., .01)):
    """Same existing column and actual fixed-other values, without estimating a."""
    c = context
    schur = axial_motion_column(c, schur=True)
    implicit = axial_motion_column(c, schur=False)
    np.testing.assert_array_equal(schur['r'], implicit['r'])
    emit('axial_compact_identity', q_sha256=c.identity['q_sha256'],
        diagnostic_script=dict(path=str(Path(__file__).resolve()), sha256=s.sha(Path(__file__))),
        coefficient_units=c.parameter_units, profile_sha256=s.token(c.data['profile']),
        original_sigma_sha256=s.token(c.inputs['sigma']), fixed_other_parameters=True)
    groups = dict(all=np.ones(len(c.data['a']), bool), **{'shot_' + k: v for k, v in c.cut_masks.items()})
    for group, point_mask in groups.items():
        mask = np.repeat(point_mask, 2)
        residual, weights = schur['r'][mask], schur['outer_weights'][mask]
        records = {}
        for name, item in (('original_Schur', schur), ('actual_profile_implicit', implicit)):
            column = item['Jaxial'][mask]
            records[name] = dict(weighted_r_dot_J=float((weights * residual) @ column),
                weighted_J_norm_squared=float((weights * column) @ column))
        emit('axial_compact_tangent', group=group, points=int(point_mask.sum()),
            original_source_cost=weighted_source_cost(residual, np.zeros(len(residual), bool), None),
            columns=records, sigma_normalization_once=True, profile_inner_only=True,
            original_outer_IRLS_at_nominal=True)
    for amount in amounts:
        actual = axial_motion_value(c, amount)
        if amount == 0:
            np.testing.assert_array_equal(actual['r'], schur['r'])
        branch_masks = _masks(c, actual['trace'])
        costs = {}
        for group, point_mask in groups.items():
            mask = np.repeat(point_mask, 2)
            costs[group] = dict(cost=weighted_source_cost(actual['r'][mask], np.zeros(int(mask.sum()), bool), None),
                branch_counts={k: int((v & point_mask).sum()) for k, v in branch_masks.items()},
                changed_source_facets=int((actual['changed_source_facets'] & point_mask).sum()))
        max_shift_m = abs(float(amount)) * c.unit * max(
            float(np.linalg.norm(c.dca, axis=1).max()), float(np.linalg.norm(c.dcb, axis=1).max()))
        emit('axial_compact_actual', amount=float(amount), max_optical_C_displacement_working_m=max_shift_m,
            costs=costs, all_original_points_retained=True, dynamic_source_facets=True,
            whole_original_batch_fresh_depth_profile=True,
            complete_truth_and_finite_extent_guards_not_audited=True)
    return schur


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=DIRECTORY)
    parser.add_argument('--world-init', type=Path)
    parser.add_argument('--mode', choices=('axial', 'ppy'), default='axial')
    parser.add_argument('--fd-step', type=float)
    parser.add_argument('--compact-cost-only', action='store_true')
    options = parser.parse_args()
    started = time.perf_counter()
    context = load_axial_motion_context(options.state, world_init=options.world_init)
    if options.mode == 'ppy':
        check_ppy_response(context, fd_step=.05 if options.fd_step is None else options.fd_step)
    elif options.compact_cost_only:
        compact_axial_cost_response(context)
    else:
        check_axial_response(context, fd_step=1e-3 if options.fd_step is None else options.fd_step)
    emit('complete_zoom_motion_response', mode=options.mode,seconds=time.perf_counter() - started,
        source_images_decoded=0, source_images_written=0,
        fitted_parameters_or_new_q=0, matrix_or_report_files_written=0,
        full_source_J_or_normal_constructed=False, original_production_code_modified=False)


if __name__ == '__main__':
    main()
