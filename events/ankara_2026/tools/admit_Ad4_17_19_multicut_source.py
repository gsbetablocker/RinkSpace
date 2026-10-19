"""Append reviewed RGB print edges to the existing native board-edge packet.

Uses the existing finite-material/normal-interval operator and physical-camera
archive adapter. No source point is duplicated to meet a row-size restriction,
and Ad18's detached ink is not admitted as a rectangular border.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
sys.dont_write_bytecode = True

import cv2
import numpy as np

SOURCE_SHA = 'f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'
REVIEWED_PTS = frozenset((230022144, 230469376, 230474752, 230797568, 230925312))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def source_review(candidate):
    """Check frozen actual RGB and preserve every measured field exactly."""
    if (candidate.get('source_video_sha256') != SOURCE_SHA
            or candidate.get('time_base') != '1/15360'
            or candidate.get('model_parameters_or_old_source_sigma_UV_roles_modified') is not False
            or candidate.get('existing_HELD_not_promoted') is not True):
        raise ValueError('Preserved same-video RGB-only candidate required')
    frames = {}
    rows = candidate['rows']
    if set(int(r['pts']) for r in rows) != REVIEWED_PTS:
        raise ValueError('Only the five independently viewed native sources are admitted')
    for row in rows:
        curve = row['curve']
        n = curve.get('advertisement_number')
        if (n not in (4, 17, 19) or curve.get('kind') != 'advertisement_boundary'
                or curve.get('extent_scope') != 'external_border'
                or curve.get('edge') not in ('left_vertical', 'right_vertical', 'top_edge', 'bottom_edge')
                or row.get('point_correspondence') is not False
                or row.get('curve_observation_kind') != 'normal_interval'
                or row.get('model_projection_used_to_generate_or_select_uv') is not False
                or row.get('source_video_sha256') != SOURCE_SHA
                or row.get('time_base') != '1/15360'):
            raise ValueError('Only unchanged actual external print-boundary queries are admitted')
        path = Path(row['source_raw_frame']).resolve()
        if sha(path) != row['source_file_sha256']:
            raise ValueError('Frozen raw native source bytes changed')
        if path not in frames:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape != (1080, 1920, 3):
                raise ValueError('Exact 1920 by 1080 native RGB required')
            frames[path] = hashlib.sha256(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
        if frames[path] != row['source_native_RGB_sha256']:
            raise ValueError('Frozen source RGB pixels changed')
        measurements = row['measured_RGB_sources']
        if (len(measurements) != len(row['uv'])
                or row['uv'] != [m['actual_source_uv'] for m in measurements]
                or row['normal'] != [m['source_search_normal'] for m in measurements]
                or row['normal_interval_px'] != [m['observed_normal_interval_px'] for m in measurements]
                or any(m['disposition'] != 'measured_material_transition' for m in measurements)):
            raise ValueError('Actual source measurement UV, normal or optical support changed')
        expected_role = ('complete_event_name_group_external_border' if n in (4, 17)
                         else 'printed_colour_panel_external_border')
        if row['geometry_role'] != expected_role:
            raise ValueError('Internal print component cannot replace the complete external group')
    counts = {split: sum(len(r['uv']) for r in rows if r['split'] == split)
              for split in ('fit', 'holdout')}
    if counts != dict(fit=31, holdout=11):
        raise ValueError('Frozen 31 FIT and 11 HELD source points must remain intact')
    return frames, counts


def build(opt):
    candidate, parent = read(opt.candidate), read(opt.parent)
    _, counts = source_review(candidate)
    if (parent.get('schema') != 'rinkspace.additional-native-board-boundary-source.v1'
            or parent.get('status') != 'explicit_new_fit_target_authorized'
            or parent.get('source_video_sha256') != SOURCE_SHA):
        raise ValueError('The current admitted same-source parent board packet is required')
    old_ids = {r['id'] for r in parent['rows']}
    if old_ids.intersection(r['id'] for r in candidate['rows']):
        raise ValueError('Source rows already present in the parent')
    identity = read(opt.state.parent/'state_identity.json')
    if sha(identity['camera_path']) != identity['camera_sha256']:
        raise ValueError('Actual complete camera archive changed')
    with np.load(identity['camera_path'], allow_pickle=False) as cameras:
        aliases = dict(zip(map(int, cameras['pts']), map(int, cameras['exposure_representative_pts'])))
    shots = read(ROOT/'calibration/program_tracking/source_shots.json')['shots'][1:10]
    new_rows = deepcopy(candidate['rows'])
    exposures = {}
    for row in new_rows:
        native = int(row['pts'])
        if native not in aliases:
            raise ValueError('Native RGB source lacks an authenticated exposure alias')
        shot = next((s for s in shots if int(s['start_pts']) <= native < int(s['end_pts'])), None)
        if shot is None:
            raise ValueError('Source is outside the nine-shot programme')
        row.update(actual_pts=aliases[native], source_shot_id=shot['id'],
            source_identity_reinspection=dict(
                raw_native_image_viewed=True,
                complete_event_name_external_extent_checked=row['curve']['advertisement_number'] in (4, 17),
                internal_magenta_white_blue_component_joins_not_external_boundaries=True,
                red_Oriental_Bio_background_checked=row['curve']['advertisement_number'] == 19,
                physical_number_authority=ref(ROOT/'configs/rink_model/ankara_2026_advertisements_user_confirmed_20260924.json'),
                actual_exposure_alias_authority=ref(identity['camera_path']),
                native_pts=native, actual_pts=aliases[native], time_base='1/15360',
                missing_edges_not_invented=True, new_direct_user_pixel_annotation_claimed=False),
            frozen_before_fit=True)
        exposures[native] = dict(actual_pts=aliases[native], shot_id=shot['id'], camera_id=shot['camera_id'])
    fit_actual = {r['actual_pts'] for r in new_rows if r['split'] == 'fit'}
    held_actual = {r['actual_pts'] for r in new_rows if r['split'] == 'holdout'}
    if fit_actual & held_actual:
        raise ValueError('New FIT and HELD pixels must use separate actual exposures')
    packet = deepcopy(parent)
    packet['rows'] += new_rows
    packet.update(
        current_target_version='parent_plus_Ad4_17_19_actual_RGB_multicut_v1',
        parent_packet=ref(opt.parent),
        appended_source_ids=[r['id'] for r in new_rows],
        appended_native_pts=sorted(exposures),
        source_append_Ad4_17_19=dict(candidate=ref(opt.candidate),
            new_FIT_points=counts['fit'], new_HELD_points=counts['holdout'],
            source_native_RGB_and_roles_verified=True, source_exposures=exposures,
            parent_all_rows_exact_prefix_preserved=True,
            candidate_all_UV_sigma_normal_support_split_fields_preserved=True,
            same_source_NEVER_old_projection_selected=True,
            candidate_points_not_duplicated_to_make_two_point_rows=True,
            Ad18_not_admitted='Detached MEIWA ink has no verified rectangular exterior; existing Ad5/18 material roles need separate evidence.',
            shared_width_height_or_geometry_constraints_added=False,
            source_accuracy_accepted=False),
        fit_observation_count=sum(r['split'] == 'fit' for r in packet['rows']),
        held_observation_count=sum(r['split'] == 'holdout' for r in packet['rows']),
        current_fit_scalar_count=2*sum(len(r['uv']) for r in packet['rows'] if r['split'] == 'fit'),
        current_primary_held_scalar_count=2*sum(len(r['uv']) for r in packet['rows'] if r['split'] == 'holdout'),
        original_parent_rows_UV_sigma_support_and_roles_unchanged=True,
        geometry_parameters_or_shared_sizes_changed=False,
        solver_target_updated_here=False, accuracy_accepted=False)
    if packet['rows'][:len(parent['rows'])] != parent['rows']:
        raise AssertionError('Parent source rows changed')
    for before, after in zip(candidate['rows'], new_rows):
        if any(after[key] != value for key, value in before.items()):
            raise AssertionError('Candidate measurement changed')
    write_new(opt.output, packet)
    print(json.dumps(dict(packet=ref(opt.output), added_FIT_points=31, added_HELD_points=11,
                          parent_rows_preserved=len(parent['rows']), added_rows=len(new_rows)), ensure_ascii=False), flush=True)


def verify(opt):
    from scripts.diagnostics.verify_current_native_primitives_20261007 import state_camera_adapter
    from rinkspace.vision_camera_initialization import current_complete_source_geometry
    arrays = dict(np.load(opt.state, allow_pickle=False))
    identity = read(opt.state.parent/'state_identity.json')
    cameras = dict(np.load(identity['camera_path'], allow_pickle=False))
    provider, world, proof = current_complete_source_geometry(opt.state, opt.auxiliary)
    state, camera_error = state_camera_adapter(arrays, cameras, provider)
    from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
    q = arrays['q'].copy()
    parent = NativeBoardEdgeFactor(state, opt.parent, len(q))
    complete = NativeBoardEdgeFactor(state, opt.packet, len(q))
    baseline, block = parent.block(q, True), complete.block(q, True)
    prefix = len(baseline.r)
    if (not np.array_equal(baseline.r, block.r[:prefix])
            or not np.array_equal(baseline.scales, block.scales[:prefix])
            or not np.array_equal(baseline.quadratic, block.quadratic[:prefix])
            or (baseline.J != block.J[:prefix]).nnz):
        raise AssertionError('Parent actual residual, raw weights or full-q Jacobian changed')
    new_fit = complete.rows[len(parent.rows):]
    new_held = complete.held[len(parent.held):]
    if sum(len(r['uv']) for r in new_fit) != 31 or sum(len(r['uv']) for r in new_held) != 11:
        raise AssertionError('Source FIT/HELD roles changed during admission')
    singleton = [r['id'] for r in new_fit if len(r['uv']) == 1]
    if singleton != ['actual_multicut_native230797568_Ad4_top_edge_source']:
        raise AssertionError('The single observed source query was duplicated or dropped')
    columns = np.unique(block.J.indices)
    direction = np.zeros(len(q))
    direction[columns] = np.random.default_rng(230022144).normal(size=len(columns))
    direction /= max(float(abs(block.J @ direction).max()), 1.)
    checks = []
    cuts = np.r_[0, np.cumsum([2*len(r['uv']) for r in complete.rows])]
    for h in (1e-4, 1e-5, 1e-6):
        fd = (complete.block(q+h*direction, False).r-complete.block(q-h*direction, False).r)/(2*h)
        delta = abs(fd-block.J@direction)
        worst = int(delta.argmax())
        ri = int(np.searchsorted(cuts, worst, side='right')-1)
        checks.append(dict(h=h, maximum=float(delta.max()),
            appended_maximum=float(delta[prefix:].max()), worst_scalar=worst,
            worst_source_row=complete.rows[ri]['id']))
    print(json.dumps(dict(full_q_FD_step_diagnostics=checks)), flush=True)
    error = checks[0]['maximum']
    appended_error = checks[0]['appended_maximum']
    if any(c['maximum'] >= 2e-3 for c in checks):
        raise AssertionError(('Actual full-q camera/world directional FD failed', checks))
    scene = read(opt.state.parent/'scene_UNACCEPTED.json')
    from calibration.solver_runtime.legacy.run_current_named66_full_programme import check_current_board_edge_exports
    exported = check_current_board_edge_exports(complete, q, scene)
    review = complete.review(q)
    previous_review_exact = None
    if opt.previous_verification is not None:
        previous_review_exact = read(opt.previous_verification)['review']['rows'] == review['rows']
        if not previous_review_exact:
            raise AssertionError('Fixing support derivatives changed actual source pixel residuals')
    report = dict(schema='rinkspace.Ad4-Ad17-Ad19-real-RGB-native-board-admission-check.v1',
        state=ref(opt.state), parent=ref(opt.parent), packet=ref(opt.packet),
        physical_camera_archive_max_error=camera_error, current_world_proof=proof,
        J_shape=list(block.J.shape), active_full_q_columns=len(columns),
        full_q_directional_FD_max_error=error,
        appended_full_q_directional_FD_max_error=appended_error,
        full_q_FD_step_diagnostics=checks,
        inherited_parent_FD_error_present=error >= 2e-3,
        finite_singleton_query_ids=singleton,
        parent_all_source_rows_exact_prefix_preserved=True,
        parent_actual_residual_scales_quadratic_full_q_J_exact_prefix_preserved=True,
        previous_v1_actual_source_pixel_residuals_exact=previous_review_exact,
        all31_FIT_and11_HELD_points_preserved=True, no_geometry_parameters_added=True,
        exported_finite_model_edges_checked=exported, review=review,
        Ad18_admitted=False, complete_source_or_motion_graph_not_rebuilt=True,
        optimizer_or_video_not_run=True, source_accuracy_accepted=False)
    write_new(opt.output, report)
    print(json.dumps(dict(report=ref(opt.output), J_shape=list(block.J.shape),
        full_q_directional_FD_max_error=error,
        appended_full_q_directional_FD_max_error=appended_error, singleton_ids=singleton,
        new_source_review=[r for r in review['rows'] if r['id'] in {s['id'] for s in new_fit+new_held}]), ensure_ascii=False), flush=True)


def probe(opt):
    from scripts.diagnostics.verify_current_native_primitives_20261007 import state_camera_adapter
    from rinkspace.vision_camera_initialization import current_complete_source_geometry
    arrays = dict(np.load(opt.state, allow_pickle=False))
    identity = read(opt.state.parent/'state_identity.json')
    cameras = dict(np.load(identity['camera_path'], allow_pickle=False))
    provider, world, proof = current_complete_source_geometry(opt.state, opt.auxiliary)
    state, camera_error = state_camera_adapter(arrays, cameras, provider)
    from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
    from z190_source_camera_diagnostic import absolute_value_jac15
    factor = NativeBoardEdgeFactor(state, opt.packet, len(arrays['q']))
    q = arrays['q'].copy()
    block = factor.block(q, True)
    cuts = np.r_[0, np.cumsum([2*len(r['uv']) for r in factor.rows])]
    ri = next(i for i, r in enumerate(factor.rows) if r['id'] == opt.row)
    row = factor.rows[ri]
    sl = slice(int(cuts[ri]), int(cuts[ri+1]))
    matrix = block.J[sl].toarray()
    columns = np.unique(block.J.indices)
    direction = np.zeros(len(q))
    direction[columns] = np.random.default_rng(230022144).normal(size=len(columns))
    direction /= max(float(abs(block.J@direction).max()), 1.)
    camera = factor._cameras(q)[row['actual_pts']]
    derived = factor._source_rows(q, row, camera)
    raw_rows = [{k:v for k,v in part.items() if k != 'normal_interval'} for part in derived]
    raw = absolute_value_jac15(camera, raw_rows, False).reshape(-1, 2)*row['sigma'][:, None]
    signed = np.einsum('ni,ni->n', raw, row['normal'])
    normal_bounds = row['normal_interval']
    column_checks = []
    active = np.flatnonzero(abs(matrix).max(axis=0) > 1e-8)
    def value(v):
        return factor._residual(v, [row])
    for col in active:
        h = factor.camera_steps.get(int(col), 1e-6)
        checks = []
        for step in (h, h*.1, h*.01):
            plus, minus = q.copy(), q.copy()
            plus[col] += step; minus[col] -= step
            fd = (value(plus)-value(minus))/(2*step)
            checks.append(dict(step=step, FD=fd.tolist(),
                error=float(abs(fd-matrix[:, col]).max())))
        column_checks.append(dict(column=int(col),
            kind='world' if col in factor.active_world else 'camera',
            q_value=float(q[col]), direction_component=float(direction[col]),
            J=matrix[:, col].tolist(), checks=checks))
    direction_checks = []
    view_start = state.bypts[row['actual_pts']].start+4*state.bypts[row['actual_pts']].bundle.index[row['actual_pts']]
    groups = dict(world=factor.active_world,
        view=np.arange(view_start, view_start+4),
        mount_pivot_optics=np.setdiff1d(factor.camera_columns, np.arange(view_start, view_start+4)),
        all=columns)
    for name, slots in groups.items():
        d = np.zeros(len(q)); d[slots] = direction[slots]
        predicted = matrix@d
        for h in (1e-4, 1e-5, 1e-6):
            fd = (value(q+h*d)-value(q-h*d))/(2*h)
            error = abs(fd-predicted)
            direction_checks.append(dict(group=name, step=h, FD=fd.tolist(),
                predicted=predicted.tolist(), maximum_error=float(error.max()),
                maximum_error_native_px=float((error*np.repeat(row['sigma'],2)).max()),
                relative_to_full_source_direction=float(error.max()/max(float(abs(block.J@direction).max()), 1e-30))))
    report = dict(schema='rinkspace.native-board-inherited-J-point-column-probe.v1',
        state=ref(opt.state), packet=ref(opt.packet), source_row_id=row['id'],
        UV=row['uv'].tolist(), sigma=row['sigma'].tolist(),
        raw_native_residual_px=raw.tolist(), signed_normal_px=signed.tolist(),
        normal_interval_px=normal_bounds.tolist(),
        distance_from_normal_support_boundary_px=np.minimum(abs(signed-normal_bounds[:,0]),abs(signed-normal_bounds[:,1])).tolist(),
        supported_residual=block.r[sl].tolist(),
        column_checks=column_checks, direction_checks=direction_checks,
        source_or_weights_changed=False)
    write_new(opt.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    build_parser = sub.add_parser('build')
    build_parser.add_argument('--candidate', required=True, type=Path)
    build_parser.add_argument('--parent', required=True, type=Path)
    build_parser.add_argument('--state', required=True, type=Path)
    build_parser.add_argument('--output', required=True, type=Path)
    verify_parser = sub.add_parser('verify')
    for name in ('state', 'auxiliary', 'parent', 'packet', 'output'):
        verify_parser.add_argument('--'+name, required=True, type=Path)
    verify_parser.add_argument('--previous-verification', type=Path)
    probe_parser = sub.add_parser('probe')
    for name in ('state', 'auxiliary', 'packet', 'output'):
        probe_parser.add_argument('--'+name, required=True, type=Path)
    probe_parser.add_argument('--row', required=True)
    opt = p.parse_args()
    dict(build=build, verify=verify, probe=probe)[opt.mode](opt)


if __name__ == '__main__':
    main()
