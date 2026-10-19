"""One frozen finite RF4 counterfactual; stdout only, no J or fitted state.

The secondary original 39679-point batch is evaluated intact, with original
profiles and newly profiled bounded depths, before incident rows are selected.
The four-coordinate direction is fixed by the earlier small camera experiment.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'RINKSPACE_MOTION_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts/diagnostics')]
sys.dont_write_bytecode = True

import numpy as np
import cv2
from calibration.solver_runtime import install
install()
import audit_current_user_ice_truth as public
from joint_absolute_source_context_20261011 import existing_material_source_byte_alias
from calibration.solver_runtime.curve_source_grouping import SourceWithNativeBaseCurveIntegration
from calibration.solver_runtime.source_loss import SourceLossWeights, weighted_source_cost
from structured_programme_source_candidate.original_motion_far_board_roles import OriginalMotionFarBoardRoleBinding
from shared_board_course_candidate.board_course_material import vertical_plane_value_jacobian
from installation_mount_candidate.mount_graph import pivot_outside_domain
from z190_motion_native import native_motion

R = public.RUNS
H = public.HERE
Q = '091f7278c5c601208c168fdbf9b560f227503d2d2dfd6fd0f3a10a7568fb16c8'
DIRECTORY = R / 'okada_review86_q8_all_saved_location_estimators_fix_20261011/accepted_states' / Q
CONFIG = ROOT / 'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_q091f_original_saved_locations_and_F3_fix_20261011.json'
PTS = 230507776
DELTA = np.array([-.0008778056420903737, .0003355374288158857,
                  -.0009384162102046441, -.0229512907659278])


def emit(stage, **record):
    print(json.dumps(dict(stage=stage, **record), ensure_ascii=False, allow_nan=False,
          default=lambda value: value.tolist() if isinstance(value, np.ndarray)
          else value.item() if isinstance(value, np.generic) else str(value)), flush=True)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def token(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def camera(state, value, pts):
    block = state.bypts[int(pts)]
    return block.bundle.camera(block.local(value[:state.structured_programme_coordinate_chart.core_count]), int(pts))


def secondary_inputs(manifest, saved):
    coverage = read(H / 'current_motion_role_whole_programme_exact_947_20261011.json')
    reference = coverage['actual_runtime_manifest']
    if sha(reference['path']) != reference['sha256']:
        raise ValueError('Frozen original motion array authority changed')
    old = read(reference['path'])
    for key in ('original_motion_far_board_role_correction', 'original_motion_edge_counts',
                'original_motion_rows', 'actual_exposures', 'native_presentations'):
        if manifest[key] != old[key]:
            raise ValueError('Current original motion identity changed: ' + key)
    wanted = old['persistent_source_replay_identity']['actual_original_source_input_arrays']
    references = read(H / 'actual_pan_pairs_original_FIT_source_endpoints_699_readonly_20261011.json')['source_input_authorities']
    source = next(r for r in references if 'source_ray_matches.npz' in r['path'])
    if sha(source['path']) != source['sha256']:
        raise ValueError('Original secondary observed source pixels changed')
    with np.load(source['path'], allow_pickle=False) as z:
        pairs, counts = z['pair_pts'].copy(), np.diff(z['offset'])
        a, b = z['source_uv'].astype(float), z['target_uv'].astype(float)
    ordinal = saved['aux__block_names'].tolist().index('secondary')
    lengths = saved['aux__block_lengths'] // 4
    start = int(sum(lengths[:ordinal]))
    stamps = saved['aux__canonical_pts'][start:start + int(lengths[ordinal])]
    index = {int(p): i for i, p in enumerate(stamps)}
    ia = np.repeat([index[int(p)] for p in pairs[:, 0]], counts)
    ib = np.repeat([index[int(p)] for p in pairs[:, 1]], counts)
    board = np.zeros(len(a), bool)
    board[manifest['original_motion_far_board_role_correction']['original_global_bundle_indices_by_block']['secondary']] = True
    finite = board.copy()
    finite_refs = [
        Path('D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00009/RinkSpace_tmp/z190_finite_motion_source_inventory_v1/secondary_fullray_2_ice_source_admission.json'),
        public.DATA / 'history/work_sessions/20261002_private-thread-00008/source_solver_audit/secondary_core_ice_current_review_v1/source_qualified_original_ice_motion_rows.json']
    for path in finite_refs:
        packet = read(path)
        ids = ([r['original_global_index'] for r in packet['rows']] if 'rows' in packet else
               [r['original_global_bundle_index'] for r in packet['records'] if r['status'] == 'qualified_ice_z0'])
        finite[ids] = True
    values = dict(a=a, b=b, ia=ia, ib=ib, motion_sigma=np.full(2 * len(a), 2.), finite_mask=finite,
                  stamps=stamps, course_board_indices=np.flatnonzero(board),
                  static_noncourse_finite_point_indices=np.flatnonzero(finite & ~board))
    for name, value in values.items():
        expected = wanted['programme:secondary:' + name]
        value = np.ascontiguousarray(value, dtype=np.dtype(expected['dtype']))
        actual = dict(dtype=value.dtype.str, shape=list(value.shape), sha256=token(value))
        if actual != expected:
            raise ValueError('Original secondary array differs: ' + name)
        values[name] = value
    profile = saved['profile__weights_' + str(ordinal)]
    if profile.shape != (len(a), 2):
        raise ValueError('Complete original secondary frozen profile required')
    domain = json.loads(str(saved['aux__current_depth_and_material_domain_contract_json'].item()))
    floor = next(r['candidate_minimum_source_Z_m'] for r in domain['blocks'] if r['block'] == 'secondary') / domain['scene_unit_metres']
    selected = (stamps[ia] == PTS) | (stamps[ib] == PTS)
    if len(a) != 39679 or len(a) > 65536:
        raise ValueError('Original single secondary production batch changed')
    emit('secondary_original_array_identity', points=len(a), incident_points=int(selected.sum()),
         original_complete_array_hashes_exact=True, profile_sha256=token(profile),
         minimum_source_depth_scene_units=floor, source_reference=source,
         original_production_chunk_size=65536, original_direction_preserved=True)
    return dict(**values, profile=profile, floor=floor, board=board, incident=selected,
                pa=stamps[ia], pb=stamps[ib])


def motion_value(state, value, data, border):
    cams = np.array([camera(state, value, int(p)) for p in data['stamps']])
    rotations = np.array([cv2.Rodrigues(c[3:6])[0] for c in cams])
    planes = np.zeros((len(data['a']), 4))
    planes[data['finite_mask'] & ~data['board']] = [0., 0., 1., 0.]
    edge = border[:-1] if np.array_equal(border[0], border[-1]) else border
    for view in np.unique(data['ia'][data['board']]):
        rows = np.flatnonzero(data['board'] & (data['ia'] == view))
        course = SimpleNamespace(sample_perimeter=lambda q, with_jacobian=False: (border,))
        selector = SimpleNamespace(_facet_course_cache=None, records=[None] * len(rows),
            source_row_groups={0: np.arange(len(rows))}, source_cameras={0: cams[view]},
            source_uv=data['a'][rows], exact_cache_statistics=dict(facet_group_hits=0, facet_group_misses=0))
        facets = OriginalMotionFarBoardRoleBinding._current_source_facets(
            selector, course, None, ('public_current_border', token(border)))
        zero = np.zeros((3, 0))
        planes[rows] = np.array([vertical_plane_value_jacobian(edge[i], edge[(i + 1) % len(edge)], zero, zero)[0] for i in facets])
    ca, cb = cams[data['ia']], cams[data['ib']]
    sigma = data['motion_sigma'].reshape(-1, 2)
    raw, checks, trace = native_motion(a=data['a'], b=data['b'], fa=np.exp(ca[:, 6]), fb=np.exp(cb[:, 6]),
        ra=rotations[data['ia']], rb=rotations[data['ib']], da=None, db=None,
        ca=ca[:, :3], cb=cb[:, :3], dca_shared=None, dcb_shared=None, dca_view=None, dcb_view=None,
        pp=cams[0, 8:10], k1=cams[0, 11], planes=planes, sigma=sigma,
        profile_weight=data['profile'], minimum_source_depth_scene_units=data['floor'],
        jacobian=False, return_trace=True)
    return raw.reshape(-1, 2), checks, trace


def motion_cost(raw, data, mask):
    normalized = (raw / data['motion_sigma'].reshape(-1, 2))[mask].ravel()
    # Frozen profile is used by the depth nuisance kernel only.  It must not
    # be multiplied into the original outer Huber source objective again.
    return weighted_source_cost(normalized, np.zeros(len(normalized), bool), None)


def main():
    options = read(CONFIG)['source_options']
    manifest = read(DIRECTORY.parent.parent / 'input_manifest.json')
    state, q, identity, error = public.load_state(DIRECTORY, Path(options['--world-init']), manifest)
    if error != 0. or identity['q_sha256'] != Q:
        raise ValueError('One exact explicitly selected source state is required')
    identity['_accepted_directory'] = str(DIRECTORY)
    state.current_user_pivot_outside_constraint_contract = manifest['camera_pivot_exterior_constraints']
    block = state.bypts[PTS]
    start = block.start + 4 * block.bundle.index[PTS]
    trial = q.copy()
    trial[start:start + 4] += DELTA
    ca, cb = camera(state, q, PTS), camera(state, trial, PTS)
    ra, rb = [cv2.Rodrigues(c[3:6])[0] for c in (ca, cb)]
    angle = np.degrees(np.arccos(np.clip((np.trace(rb @ ra.T) - 1.) / 2., -1., 1.)))
    emit('fixed_RF4_counterfactual', complete_parent_q_sha256=Q, actual_pts=PTS,
         RF4_evaluation_columns=list(range(start, start + 4)), delta_RF4=DELTA,
         actual_SO3_change_degrees=angle, camera_C_change_scene_units=cb[:3] - ca[:3],
         camera_C_change_working_m=(cb[:3] - ca[:3]) * state.geometry.scene_unit_m,
         C_change_from_original_lever_pose_only=True, focal_ratio=np.exp(cb[6] - ca[6]),
         actual_focal_before_after=np.exp([ca[6], cb[6]]),
         frozen_profile_token=identity['frozen_profile_token'], script_sha256=sha(__file__))
    with np.load(DIRECTORY / 'complete_state_UNACCEPTED.npz', allow_pickle=False) as z:
        names = z['aux__block_names'].tolist()
        keys = ['aux__block_names', 'aux__block_lengths', 'aux__canonical_pts',
                'aux__current_depth_and_material_domain_contract_json',
                'profile__weights_' + str(names.index('secondary'))]
        saved = {key: z[key].copy() for key in keys}
    data = secondary_inputs(manifest, saved)
    course, cq, _ = state.geometry.print_course_binding(q[state.geometry_parameter_indices])
    border = np.asarray(course.sample_perimeter(cq, with_jacobian=False)[0])
    if not np.array_equal(border, np.asarray(read(DIRECTORY / 'scene_UNACCEPTED.json')['ice_boundary']['points_xyz'])):
        raise ValueError('Public full finite course differs from saved source scene')
    before, before_check, before_trace = motion_value(state, q, data, border)
    after, after_check, after_trace = motion_value(state, trial, data, border)
    selected = data['incident']
    for kind, mask in [('all_incident', selected), ('finite_board', selected & data['board']),
                       ('finite_ice', selected & data['finite_mask'] & ~data['board']),
                       ('profiled_unknown', selected & ~data['finite_mask'])]:
        x, y = motion_cost(before, data, mask), motion_cost(after, data, mask)
        emit('incident_original_motion_cost', kind=kind, points=int(mask.sum()), before=x, after=y,
             change=y - x, before_raw_p95=float(np.quantile(np.linalg.norm(before[mask], axis=1), .95)) if mask.any() else None,
             after_raw_p95=float(np.quantile(np.linalg.norm(after[mask], axis=1), .95)) if mask.any() else None,
             unknown_rho_zero_before_after=[int(((tr[:, 0] == 0.) & mask & ~data['finite_mask']).sum()) for tr in (before_trace, after_trace)])
    for pa, pb in sorted(set(zip(data['pa'][selected].tolist(), data['pb'][selected].tolist()))):
        mask = (data['pa'] == pa) & (data['pb'] == pb)
        x, y = motion_cost(before, data, mask), motion_cost(after, data, mask)
        emit('incident_directed_pair', source_actual_pts=pa, target_actual_pts=pb,
             points=int(mask.sum()), finite_board=int((mask & data['board']).sum()),
             finite_ice=int((mask & data['finite_mask'] & ~data['board']).sum()), before=x, after=y, change=y-x)
    emit('profiled_motion_domain', before=before_check, after=after_check,
         untouched_rows_max_raw_change=float(abs(after[~selected]-before[~selected]).max()),
         newly_profiled_bounded_depth=True, sigma_applied_once=True, frozen_profile_not_outer_multiplied=True)
    board = public.NativeBoardEdgeFactor(state, options['--additional-board-source'], len(q))
    ice = public.NativeIceCurveFactor(state, options['--additional-ice-source'], len(q))
    integration = SourceWithNativeBaseCurveIntegration(SimpleNamespace(contract={},
        native_board_factor=board, native_ice_factor=ice), options['--native-base-curve-integration-policy'])
    for name, factor in [('additional_real_native_board_edges', board), ('additional_real_native_ice_curves', ice)]:
        active = factor if factor is board else factor._qualified_active_fit_view()
        payload = integration._payloads[name]
        cursor = 0
        for row in active.rows:
            stop = cursor + 2 * len(row['uv'])
            if row['actual_pts'] == PTS:
                loss = SourceLossWeights(payload.integration_weights[cursor:stop],
                    None if payload.working_normalization_multiplier is None else payload.working_normalization_multiplier[cursor:stop])
                r0, r1 = [factor._residual(v, [row]) for v in (q, trial)]
                costs = [weighted_source_cost(r, factor.quadratic[cursor:stop], loss) for r in (r0, r1)]
                emit('same_exposure_native_FIT', family=name, id=row['id'], points=len(row['uv']),
                     before_cost=costs[0], after_cost=costs[1], change=costs[1]-costs[0],
                     raw_typed_before=(r0*np.repeat(row['sigma'], 2)).reshape(-1, 2),
                     raw_typed_after=(r1*np.repeat(row['sigma'], 2)).reshape(-1, 2))
            cursor = stop
        if cursor != factor.row_count:
            raise ValueError('Qualified original native scalar layout differs')
    human = public.bind_human(state, board, ice, options, len(q))
    wide = public.wide_factor(state, identity, options, len(q), options['--practical-human-location-envelope'])
    with existing_material_source_byte_alias() as aliases:
        circle, material, placement = public.existing_LOW_source_owners(state, q, DIRECTORY, options,
            read(DIRECTORY / 'scene_UNACCEPTED.json'))
    confirmed = public.UserConfirmedSourceTruth(state, options['--user-confirmed-source-truth-authority'], board, ice,
        parameter_count=len(q), circle_factor=circle, material_factor=material, placement_source=placement)
    snapshots = [public.values(human, wide, confirmed, v) for v in (q, trial)]
    caps = read(DIRECTORY / 'truth_restoration_state.json')
    count = 0
    for family, display in [('native', 'primary'), ('wide', 'wide'), ('confirmed', 'confirmed_details')]:
        ids = [r['id'] for r in snapshots[0][display]]
        authority = caps['families'][family]
        if ids != authority['group_ids']:
            raise ValueError('Every original truth identity must be bound: ' + family)
        limits = np.asarray(authority['limits'])
        b, a = [v['buffers'][family] for v in snapshots]
        if np.max(b-limits) > 1e-8:
            raise ValueError('Original q091 truth baseline does not meet its existing caps')
        changed = np.flatnonzero(a != b)
        count += len(ids)
        emit('original_truth_family', family=family, groups=len(ids), changed_groups=len(changed),
             exceeds_original_limits=int((a > limits + 1e-8).sum()),
             maximum_original_cap_excess_px=float(np.max(a-limits)),
             changed=[dict(id=ids[i], before=float(b[i]), after=float(a[i]), original_limit=float(limits[i])) for i in changed])
    if count != 86:
        raise ValueError('All original 86 protected groups must be evaluated')
    pivots = [pivot_outside_domain(state, value) for value in (q, trial)]
    if pivots[0] != pivots[1]:
        raise ValueError('Only RF4 changed but mechanical-pivot domain differs')
    state.geometry.validate_parameters(q[state.geometry_parameter_indices])
    emit('domain_scope', all86_actual_queries_bound=True, all86_primary_and_full_strokes_checked=True,
         mechanical_pivot_domain=pivots[1]['valid'], unchanged_full_physical_geometry=True,
         native_motion_Brown_source_target_cheirality_and_bounded_depth_valid=True,
         original_focal_domain_100_to_50000_valid=bool(100. <= np.exp(cb[6]) <= 50000.),
         complete_original_bundle_finite_extent_guard_not_reconstructed=True,
         source_images_decoded=0, source_images_written=0, new_q_J_fit_or_report_files_created=False,
         exact_existing_material_byte_aliases=aliases)


if __name__ == '__main__':
    main()
