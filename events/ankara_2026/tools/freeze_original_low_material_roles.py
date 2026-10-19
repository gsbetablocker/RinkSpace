"""Freeze complete native RGB material reviews using existing source-only helpers.

No camera, geometry, solver, residual, guard, graph, pickle or runtime is loaded.
The historical cut5 builder and its frozen packet bytes remain unchanged.
"""
from copy import deepcopy
from pathlib import Path
import argparse, json
from freeze_original_cut5_9288_material_roles import (
    BASE, SOURCE_SHA, arrays, check_ref, np, read, ref, write_immutable)

CANDIDATE_SHA = '5fbb9c6d4c9aed5857c82b1f1cdc9a7e42ecc83ac2c1c07fd0f83616b5be94f2'
RECEIPT_SHA = 'c93a9db6104b1bbbcc0e88edf567908632d38d9584d77b48b58f495df48ef930'
PARENT_SHA = '2ce1e3711cb66540bb11bd014b03f8ed2c1ea350e5e8b596c148613c41edcc6d'
REVIEWS = (
    ('independent_pages000_041_material_review_v1.json', '9798414b94d3848ef20348bdb037906f9924bfb4f54d897fe3c429689b66a48a', range(0, 42)),
    ('independent_pages042_083_material_review_v1.json', 'c8f01ed43a039187c5b87396e920471dc93fe27db3f944ece0293d51b00095e8', range(42, 84)),
    ('root_pages084_098_all_original_points_actual_RGB_material_QUALIFIED_v1.json', 'dbad03302a655aac51e306c95c11f5db033e035a453d38fe740a521c8ffeaf77', range(84, 99)),
    ('independent_pages099_123_material_review_v1.json', 'e3800f8063900f102ca253617252a463df85ffb9f2800dbc57066025b087234e', range(99, 124)),
)
NUMERIC_KEYS = ('shot_index', 'original_raw_point_index', 'original_raw_pair_index',
    'original_raw_pair_local_index', 'original_selected_row', 'original_full_low_bundle_row',
    'current_original_prefix_bundle_index', 'source_pts', 'target_pts',
    'source_native_pair_pts', 'source_uv', 'target_uv', 'source_sigma_px')
PAGE_KEYS = ('original_candidate_ordinal', 'sheet_index', 'pair_review_ordinal',
             'pair_point_ordinal', 'sheet_pair_row')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--authority', required=True)
    a = ap.parse_args()
    assert a.authority.strip()
    directory = BASE / 'full_cut8_region_actual_RGB_review'
    cp = BASE / 'full_cut8_source_only_overlap/original_low_cut8_full_source_overlap_candidates_v1.json'
    rp = directory / 'actual_original_cut8_6337_region_candidates_RGB_receipt_v1.json'
    pp = BASE / 'secondary686_plus_low9341_original_FIT_far_board_role_ADMITTED_v1.json'
    candidate, receipt, parent = read(cp, CANDIDATE_SHA), read(rp, RECEIPT_SHA), read(pp, PARENT_SHA)
    assert candidate['shot_index'] == receipt['shot_index'] == 8
    assert candidate['source_video_sha256'] == parent['source_video_sha256'] == SOURCE_SHA
    assert candidate['time_base'] == parent['time_base'] == '1/15360'
    assert receipt['source_candidates'] == ref(cp)
    assert candidate['source_measurements_roles_weights_modified'] is False
    assert candidate['HELD_promoted'] is False and candidate['new_source_rows'] == 0
    assert len(candidate['records']) == 8226 and receipt['selected_original_points'] == 6337
    assert receipt['actual_source_pairs'] == 372 and receipt['actual_source_exposures'] == 373
    assert parent['original_unknown_motion_points_retyped'] == 10027 and len(parent['source_packets']) == 5
    keys = ('original_source_ledger', 'original_selected_rows', 'original_raw_motion',
            'clock_alias_authority', 'clock_state', 'static_UI_retirement',
            'original_typed_prefix_contract', 'existing_role_packet')
    for key in keys:
        if key == 'clock_state':
            assert candidate[key]['q_or_camera_values_read'] is False
            continue  # Original clock authority is an arrays-only reference.
        check_ref(candidate[key])
    assert candidate['existing_role_packet'] == ref(pp)
    for child in parent['source_packets']:
        check_ref(child)
    check_ref(receipt['source_region_masks']); check_ref(receipt['source_mask_archive'])
    masks = arrays(receipt['source_mask_archive']['path'], ('original_source_index',
        'point_endpoint_reason_bits', 'point_both_endpoint_region_candidate'))
    both = np.all(masks['point_endpoint_reason_bits'] == 0, axis=1)
    assert np.array_equal(both, masks['point_both_endpoint_region_candidate'])
    assert np.array_equal(masks['original_source_index'],
        [r['current_original_prefix_bundle_index'] for r in candidate['records']])
    original = {r['current_original_prefix_bundle_index']: r for r in candidate['records']}
    point_map = {r['current_original_prefix_bundle_index']: r for r in receipt['point_to_pair_sheet']}
    expected = {candidate['records'][i]['current_original_prefix_bundle_index'] for i in np.flatnonzero(both)}
    assert len(original) == 8226 and len(point_map) == len(expected) == 6337 and set(point_map) == expected
    sheets = {r['sheet_index']: r for r in receipt['sheets']}
    assert set(sheets) == set(range(124))
    for sheet in sheets.values():
        check_ref(sheet)
    human_refs, reviewed_pages, human_by_index = [], set(), {}
    for ordinal, (name, digest, scope) in enumerate(REVIEWS):
        path = directory / name
        human = read(path, digest); human_refs.append(ref(path))
        assert human['source_RGB_receipt'] == ref(rp)
        if human['schema'] == 'rinkspace.original-low-FIT-actual-RGB-source-only-material-review.v1':
            assert human['status'] == 'complete_source_only_material_review'
            assert not human['uncertain_source_only_rows'] and not human['remaining_sheet_indices']
            assert not human.get('rejected_source_only_rows', [])
            reviewed, rows = human['reviewed_sheets'], human['qualified_source_only_rows']
            qualified = lambda row: row['material_identity_review'] == 'actual_both_endpoints_static_far_board_printed_material'
        else:
            assert human['schema'] == 'rinkspace.original-low-cut8-human-source-material-qualification.v1'
            assert human['status'] == 'actual_RGB_material_qualified_SOURCE_ONLY_not_runtime_admitted'
            assert not any(human[k] for k in ('unknown_material_points', 'nearfield_equipment_points', 'people_or_UI_or_ice_points'))
            reviewed, rows = human['actual_RGB_pages_reviewed'], human['records']
            assert human['original_source_row_UV_sigma_PTS_split_weights_changed'] is False
            qualified = lambda row: (row['both_end_material_reviewed']
                and row['material_identity_review'] == 'qualified_far_board_print_both_actual_RGB_ends'
                and row['material_family'] == 'static_far_board_printed_ISU_OrientalBio_or_white_board_face'
                and row['subpixel_correspondence_certified'] is False)
        pages = {s['sheet_index'] for s in reviewed}
        assert pages == set(scope) and not pages & reviewed_pages
        reviewed_pages |= pages
        for sheet in reviewed:
            actual = sheets[sheet['sheet_index']]
            assert sheet['path'] == actual['path'] and sheet['sha256'] == actual['sha256']
        wanted = {i for i, r in point_map.items() if r['sheet_index'] in pages}
        assert {r['current_original_prefix_bundle_index'] for r in rows} == wanted
        for row in rows:
            i = row['current_original_prefix_bundle_index']
            assert i not in human_by_index and qualified(row)
            assert all(row[k] == original[i][k] for k in NUMERIC_KEYS)
            assert all(row[k] == point_map[i][k] for k in PAGE_KEYS)
            human_by_index[i] = dict(human_material_review_ref_index=ordinal)
    assert reviewed_pages == set(range(124)) and set(human_by_index) == expected
    source = arrays(candidate['original_source_ledger']['path'], ('active_original_global_index',
        'source_uv_a', 'source_uv_b', 'source_endpoint_index_a', 'source_endpoint_index_b', 'original_sigma_xy'))
    raw = arrays(candidate['original_raw_motion']['path'], ('pair_pts', 'offsets', 'xy0', 'xy1'))
    selected = arrays(candidate['original_selected_rows']['path'], ('original_global_row_indices',))['original_global_row_indices']
    aliases_raw = arrays(candidate['clock_alias_authority']['path'], ('pts', 'exposure_representative_pts'))
    aliases = dict(zip(map(int, aliases_raw['pts']), map(int, aliases_raw['exposure_representative_pts'])))
    clock = arrays(candidate['clock_state']['path'], ('aux__block_names', 'aux__block_lengths',
        'aux__canonical_pts', 'aux__qualified_original_motion_ice_contract_json'))
    bi = list(clock['aux__block_names']).index('low'); lengths = clock['aux__block_lengths'] // 4
    lo = int(lengths[:bi].sum()); stamps = clock['aux__canonical_pts'][lo:lo + int(lengths[bi])]
    assert bi == 3 and len(stamps) == 1851 and len(selected) == 83453
    assert len(source['active_original_global_index']) == 434991
    assert candidate['full_original_cut_offset'] == 351555
    typed = {r['global_bundle_index'] for r in read(candidate['original_typed_prefix_contract']['path'])}
    ice3 = set(json.loads(str(clock['aux__qualified_original_motion_ice_contract_json'].item()))['global_bundle_indices'])
    retired = {r['current_bundle_motion_row'] for r in read(candidate['static_UI_retirement']['path'])['records']}
    old_low = set()
    for child in parent['source_packets']:
        packet = read(child['path'])
        if packet['schema'] in ('rinkspace.original-low-motion-source-qualified-far-board-role-admission.v1',
                                'rinkspace.original-low-FIT-source-qualified-far-board-role-admission.v1'):
            old_low |= {r['current_original_prefix_bundle_index'] for r in packet.get('admitted_source_only_rows', packet.get('records', []))}
    assert len(old_low) == 9341 and len(typed) == 274 and len(ice3) == 3 and len(retired) == 89
    assert not expected & (old_low | typed | ice3 | retired)
    qualified_rows = []
    for mapped in receipt['point_to_pair_sheet']:
        i = mapped['current_original_prefix_bundle_index']; row = original[i]
        full = int(row['original_full_low_bundle_row']); g = int(row['original_raw_point_index'])
        j = int(row['original_raw_pair_index']); s = int(row['original_selected_row'])
        native = tuple(map(int, raw['pair_pts'][j]))
        assert full == 351555 + s == int(source['active_original_global_index'][i])
        assert g == int(selected[s]) == int(raw['offsets'][j]) + int(row['original_raw_pair_local_index'])
        assert native == tuple(row['source_native_pair_pts'])
        assert (aliases[native[0]], aliases[native[1]]) == (row['source_pts'], row['target_pts'])
        assert int(stamps[source['source_endpoint_index_a'][full]]) == row['source_pts']
        assert int(stamps[source['source_endpoint_index_b'][full]]) == row['target_pts']
        assert row['source_pts'] != row['target_pts']
        for values, key in ((raw['xy0'][g], 'source_uv'), (raw['xy1'][g], 'target_uv'),
            (source['source_uv_a'][full], 'source_uv'), (source['source_uv_b'][full], 'target_uv'),
            (source['original_sigma_xy'][full], 'source_sigma_px')):
            assert np.array_equal(values, row[key])
        r = deepcopy(row); r.update({k: mapped[k] for k in PAGE_KEYS})
        r.update(split='fit', material_role_review='confirmed_static_current_far_board_printed_material',
            both_end_original_native_RGB_material_reviewed=True,
            source_only_material_description='Stationary printed far inner board or its white board face, explicitly viewed at both actual native RGB endpoints; original matching precision and sigma retained.',
            actual_original_RGB_material_certificate=dict(**human_by_index[i],
                actual_source_and_target_endpoint_visually_reviewed=True,
                selected_point_on_people_ice_UI_near_equipment=False,
                original_UV_sigma_PTS_and_correspondence_reestimated=False,
                material_role_not_inferred_from_navigation_or_model_error=True))
        assert all(r[k] == row[k] for k in row)
        qualified_rows.append(r)
    pending = [r['current_original_prefix_bundle_index'] for r in candidate['records'] if r['current_original_prefix_bundle_index'] not in expected]
    assert len(qualified_rows) == 6337 and len(pending) == 1889
    qualification = dict(schema='rinkspace.original-low-FIT-source-only-static-board-material-qualification.v1',
        status='actual_original_RGB_material_qualified_SOURCE_ONLY', shot_index=8,
        source_video_sha256=SOURCE_SHA, time_base='1/15360', source_candidate_packet=deepcopy(receipt['source_candidates']),
        actual_RGB_receipt=ref(rp), human_material_reviews=human_refs,
        source_qualified_far_board_count=6337, source_qualified_actual_pair_count=372,
        actual_original_RGB_pages_reviewed=124, actual_source_exposures=373,
        new_query_or_scalar_rows=0, original_UV_PTS_sigma_and_correspondence_unchanged=True,
        HELD_or_diagnostic_tracks_not_promoted_to_FIT=True,
        subpixel_correspondence_accuracy_or_sigma_not_reestimated=True,
        navigation_mask_not_material_identity=True, material_role_not_selected_by_model_error=True,
        clock_state_container_receipt=ref(candidate['clock_state']['path']),
        source_region_masks=deepcopy(receipt['source_region_masks']), source_mask_archive=deepcopy(receipt['source_mask_archive']),
        remaining_unqualified_source_candidate_count=1889, remaining_unqualified_source_candidate_prefix_indices=pending,
        remaining_unqualified_candidates_keep_unknown_depth_role=True, whole_low_original_motion_not_reclassified=True,
        uniqueness_and_existing_role_disjoint_check=dict(qualified_unique_prefix_indices=6337,
            qualified_unique_original_raw_point_indices=len({r['original_raw_point_index'] for r in qualified_rows}),
            overlap_existing_low9341=0, overlap_existing_typed274=0, overlap_ice3=0, overlap_retired89=0),
        admitted_source_only_rows=qualified_rows, authority=a.authority,
        source_operator_residual_invariance_not_claimed=True,
        runtime_roles_weights_rows_or_target_not_modified_by_this_source_only_tool=True,
        script=ref(Path(__file__).resolve()))
    for key in keys:
        qualification[key] = deepcopy(candidate[key])
    qref = write_immutable(directory / 'cut8_original_FIT6337_actual_RGB_far_board_material_QUALIFIED_SOURCE_ONLY_v1.json', qualification)
    admitted = deepcopy(qualification)
    admitted.update(schema='rinkspace.original-low-FIT-source-qualified-far-board-role-admission.v1',
        status='explicit_original_motion_source_role_revision_authorized', new_source_scalar_rows=0,
        admission=dict(authority=a.authority, parent_candidate=qref, unknown_depth_operator_changed_explicitly=True,
            source_r_invariance_not_claimed=True, source_scalar_rows_added=0))
    aref = write_immutable(directory / 'cut8_original_FIT6337_actual_RGB_far_board_role_ADMITTED_v1.json', admitted)
    combined = deepcopy(parent)
    combined.update(source_packets=deepcopy(parent['source_packets']) + [aref], authority=a.authority,
        original_unknown_motion_points_retyped=16364, historical_10027_parent_packet=ref(pp),
        inherited_10027_children_preserved_in_order=True, additional_original_low_cut8_material_roles=6337,
        original_motion_material_role_count_by_block=dict(secondary=686, low=15678),
        planned_finite_point_count_by_block=[13, 0, 692, 16149, 0, 0],
        remaining_unqualified_cut8_candidate_count=1889, remaining_unqualified_cut5_candidate_count=5930,
        whole_low_original_motion_not_reclassified=True, source_r_invariance_not_claimed=True)
    cref = write_immutable(BASE / 'secondary686_plus_low15678_original_FIT_far_board_role_ADMITTED_v1.json', combined)
    print(json.dumps(dict(source_qualification=qref, admitted_low6337=aref, combined16364=cref,
        actual_RGB_qualified_points=6337, unknown_cut8_candidate_points=1889,
        inherited_roles=10027, source_rows_added=0, source_UV_sigma_PTS_bitexact=True)), flush=True)


if __name__ == '__main__':
    main()
