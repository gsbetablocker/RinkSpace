"""Freeze the root-reviewed outer interface, then load only its typed factor.

This validates the current circle input path, not a fit or an accuracy pass.
The complete checkpoint supplies a thin structural context only; no motion
graph, profiles, residual, Jacobian, model projection or new camera is built.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
OUT = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/native_board_material_circle_RGB_review_20261012')
RUN = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_ce235f_wide_curve_budget_resume_20261012')
Q = '917c0fa45f80af8e16c74d0f2d56486fdb500b591357add3dc2b240b3de64cdf'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def ref(path):
    path = Path(path)
    with path.open('rb') as file:
        return dict(path=str(path), sha256=hashlib.file_digest(file, 'sha256').hexdigest())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    return ref(path)


def main():
    started = time.perf_counter()
    manifest = read(RUN / 'input_manifest.json')
    old_contract = manifest['additional_real_native_board_circles']
    parent_ref = dict(path=old_contract['packet_path'], sha256=old_contract['packet_sha256'])
    if ref(parent_ref['path']) != parent_ref:
        raise ValueError('Original q32 circle packet changed')
    parent = read(parent_ref['path'])
    evidencepath = OUT / 'C001_actual_outer_stroke_public_reader_candidates.json'
    evidence = read(evidencepath)
    if evidence['current_packet'] != parent_ref or evidence['packet_row'] != 1:
        raise ValueError('Actual source evidence belongs to another input')
    original = parent['rows'][1]
    newuv = [r['actual_outer_interface_candidate_uv'] for r in evidence['outermost_visible_material_interface_candidates']]
    flags = [r['interface_candidate']['source_flags'] for r in evidence['outermost_visible_material_interface_candidates']]
    review = dict(schema='rinkspace.native-circle-reviewed-outer-interface-uv-revision.v1',
        source_video_sha256=parent['source_video_sha256'], time_base=parent['time_base'],
        original_packet=parent_ref, original_packet_row=1, row_id=original['id'],
        original_uv=original['uv'], original_sigma_px=original['sigma_px'], corrected_uv=newuv,
        physical_role='outermost_visible_circular_print_stroke',
        authority=dict(kind='assistant_and_root_actual_native_RGB_outer_interface_review',
            basis='Root independently opened all10 current-vs-outer RAW crops and confirmed the exterior black circle/background interface. Lettering, white flanks and the black-stroke midpoint are distinct',
            exact_user_pixel_ground_truth=False, role_review_is_not_calibrated_subpixel_accuracy=True),
        public_reader_evidence=ref(evidencepath),
        actual_RAW_outer_interface_review_image=ref(OUT / 'C001_actual_native_current_vs_OUTER_candidates_all10.png'),
        original_RGBA_source=evidence['source_RGB'], actual_native_RGB_sha256=evidence['actual_native_RGB_sha256'],
        original_source_row_float64_UV_sha256=hashlib.sha256(np.asarray(original['uv'], np.float64).tobytes()).hexdigest(),
        unresolved_interface_direction_flags=[i for i, f in enumerate(flags) if f['sided_material_direction_unverified']],
        all_current_interface_flags=flags,
        public_reader_flags_and_wider_material_competition_preserved=True,
        black_stroke_center_not_used_as_outer_interface=True,
        model_or_interpolated_UV_used=False, automatic_localization_precision_certified=False,
        source_weights_world_ring_ID_clock_roles_and_other15_rows_unchanged=True,
        nominal_optical_support_or_sampling_budget_not_changed=True)
    review_ref = write(OUT / 'C001_outer_interface_actual_RAW_role_review_ROOT_CONFIRMED.json', review)
    revised = deepcopy(parent)
    row = revised['rows'][1]
    row['uv'] = newuv
    row['historical_source_support_before_20261012_outer_interface_revision'] = deepcopy(original['source_support'])
    support = []
    for i, (old, value) in enumerate(zip(original['source_support'], evidence['outermost_visible_material_interface_candidates'])):
        reading = value['interface_candidate']
        support.append(dict(side=old['side'], axis=old['axis'], fixed=old['fixed'],
            uv=newuv[i], sigma_px=original['sigma_px'][i],
            native_search_support=evidence['visible_outer_stroke_search_intervals_absolute_native_px'][i],
            normal_interval_px=reading['observed_normal_interval_px'],
            current_outer_interface_candidate_evidence=dict(reference=ref(evidencepath), point_index=i),
            public_reader_source_flags=reading['source_flags'],
            source_RGB_branch_manually_reviewed_by_assistant_and_root=True,
            diagnostic_support_only_not_an_active_source_optical_operator=True,
            exact_user_pixel_ground_truth=False, automatic_localization_precision_certified=False))
    row['source_support'] = support
    row['source_reviewed_uv_revision'] = review_ref
    row['source_review_authority'] = '2026-10-12 assistant/root actual RAW outer circular print/background interface role review; signed interface candidates selected by visible physical role; original uncertainty and all unresolved public flags retained'
    revised.update(reviewed_source_uv_revision=review_ref,
        current_target_version='circle16_C001_outer_interface_actual_RAW_role_revision_20261012',
        all_existing_FIT_and_holdout_rows_unchanged=False, source_installed_in_solver=False)
    fit = [r for r in revised['rows'] if r['split'] == 'fit']
    held = [r for r in revised['rows'] if r['split'] == 'holdout']
    revised.update(observation_count=len(revised['rows']), fit_observation_count=len(fit),
        holdout_observation_count=len(held), fit_uv_count=sum(len(r['uv']) for r in fit),
        fit_scalar_count=2 * sum(len(r['uv']) for r in fit),
        current_fit_scalar_count=2 * sum(len(r['uv']) for r in fit),
        current_primary_held_scalar_count=2 * sum(len(r['uv']) for r in held))
    revised_ref = write(OUT / 'native_circle16_C001_outer_interface_RAW_reviewed_UV_revision_NOT_FITTED.json', revised)
    for i, (a, b) in enumerate(zip(parent['rows'], revised['rows'])):
        if i != 1 and a != b:
            raise ValueError('Another current FIT/HELD row changed')
    assert revised['geometry_contract'] == parent['geometry_contract']
    # Load the normal public typed-factor constructor with exact current
    # geometry and camera dependency indices, without evaluating its values.
    from calibration.solver_runtime import install
    install()
    from rinkspace.vision_camera_initialization import current_complete_source_geometry
    from structured_programme_source_candidate.paired_programme_chart import CompleteProgrammeParameterLayout, PairedProgrammeCoordinateChart
    from structured_programme_source_candidate.native_board_circle_source import NativeBoardCircleFactor
    identity = read(RUN / 'accepted_states' / Q / 'state_identity.json')
    checkpoint = RUN / 'accepted_states' / Q / 'complete_state_UNACCEPTED.npz'
    assert ref(checkpoint)['sha256'] == identity['state_npz_sha256']
    fields = ['q', 'aux__canonical_pts', 'aux__block_start', 'aux__block_lengths', 'aux__block_names',
        'aux__installation_xyz_slots', 'aux__all_installation_optics_installation_order',
        'aux__current_shared_mount_parameter_indices', 'aux__current_all_installation_optics_parameter_indices',
        'aux__current_complete_parameter_layout_json']
    with np.load(checkpoint, allow_pickle=False) as arrays:
        saved = {key: arrays[key].copy() for key in fields}
    assert hashlib.sha256(saved['q'].tobytes()).hexdigest() == Q
    config = read(ROOT / 'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_ce235f_wide_curve_budget_20261012.json')
    provider, world, geometry_proof = current_complete_source_geometry(checkpoint, config['source_options']['--world-init'])
    declaration = json.loads(str(saved['aux__current_complete_parameter_layout_json'].item()))
    layout = CompleteProgrammeParameterLayout(declaration['original_complete_parameter_count'], declaration['additional_world_parameter_count'])
    chart = PairedProgrammeCoordinateChart(len(world) - 66)
    q = layout.evaluation_q(saved['q'])
    assert np.array_equal(q[np.asarray(chart.world_indices)], world)
    assert ref(identity['camera_path'])['sha256'] == identity['camera_sha256']
    with np.load(identity['camera_path'], allow_pickle=False) as arrays:
        camera = {key: arrays[key].copy() for key in ('pts', 'exposure_representative_pts', 'installation_id')}
    aliases = dict(zip(map(int, camera['pts']), map(int, camera['exposure_representative_pts'])))
    installation = {}
    for i, actual in enumerate(camera['exposure_representative_pts']):
        installation.setdefault(int(actual), str(camera['installation_id'][i]))
    order = saved['aux__all_installation_optics_installation_order'].tolist()
    optics = saved['aux__current_all_installation_optics_parameter_indices'].reshape(len(order), 3)
    mounts = saved['aux__current_shared_mount_parameter_indices'].reshape(len(order), 3)
    bypts, cursor = {}, 0
    for name, start, length in zip(saved['aux__block_names'], saved['aux__block_start'], saved['aux__block_lengths']):
        stamps = saved['aux__canonical_pts'][cursor:cursor + int(length)//4]
        owners = {installation[int(p)] for p in stamps}
        if len(owners) != 1:
            raise ValueError('Original physical camera block installation changed')
        owner = order.index(owners.pop())
        block = NS(name=str(name), start=int(start), xyz_slots=saved['aux__installation_xyz_slots'][owner],
            optics_chart=NS(indices=optics[owner], mount_indices=mounts[owner]),
            bundle=NS(index={int(p): i for i, p in enumerate(stamps)}))
        bypts.update({int(p): block for p in stamps})
        cursor += len(stamps)
    state = NS(geometry=provider, geometry_parameter_indices=np.asarray(chart.world_indices),
        structured_programme_coordinate_chart=chart, original=NS(aliases=aliases), bypts=bypts)
    old = NativeBoardCircleFactor(state, parent_ref['path'], old_contract['prefix_parameter_count'])
    current = NativeBoardCircleFactor(state, revised_ref['path'], old_contract['prefix_parameter_count'])
    for name in ('scales', 'quadratic', 'circle_indices', 'active', 'camera_columns', 'active_world'):
        assert np.array_equal(getattr(old, name), getattr(current, name)), name
    assert old.row_count == current.row_count == 218
    assert current.contract['reviewed_source_uv_revision']['row_id'] == original['id']
    actual = next(r for r in current.rows if r['id'] == original['id'])
    assert actual['_source_optical_support'] is None
    assert [r['_source_optical_support'] is not None for r in old.rows] == [r['_source_optical_support'] is not None for r in current.rows]
    assert np.array_equal(actual['uv'], np.asarray(newuv))
    assert np.array_equal(actual['sigma'], np.asarray(original['sigma_px']))
    receipt = dict(schema='rinkspace.native-circle-reviewed-UV-public-installation-check.v1',
        revised_packet=revised_ref, RGB_role_review=review_ref, original_q32_packet=parent_ref,
        public_loader='NativeBoardCircleFactor', current_geometry_proof=geometry_proof,
        native_circle_source_SHA=ref(ROOT / 'calibration/solver_runtime/legacy/structured_programme_source_candidate/native_board_circle_source.py')['sha256'],
        exact_q32_geometry_and_parameter_chart_restored=True,
        fit_rows=len(current.rows), held_rows=len(current.held), scalar_rows=current.row_count,
        current_source_sigma_quadratic_all_dependency_columns_exact=True,
        only_C001_UV_and_its_diagnostic_provenance_changed=True,
        no_active_optical_support_added=True, new_public_contract=current.contract,
        residual_J_solver_or_projection_evaluated=False, q32_state_not_changed=True,
        production_config_not_changed=True, elapsed_seconds=time.perf_counter() - started)
    receipt_ref = write(OUT / 'native_circle16_C001_outer_interface_public_installation_receipt.json', receipt)
    print(json.dumps(dict(revised_packet=revised_ref, public_installation_receipt=receipt_ref,
        elapsed_seconds=receipt['elapsed_seconds'], fit_rows=len(current.rows), scalar_rows=current.row_count), ensure_ascii=False))


if __name__ == '__main__':
    main()
