"""Freeze explicitly RGB-reviewed original cut5 material roles, without runtime.

The original motion measurements remain unchanged. This tool authenticates the
numeric source ledgers and three actual RGB reviews; it does not load a camera,
scene, residual, guard, graph, pickle or solver.
"""
from pathlib import Path
from copy import deepcopy
import argparse, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
sys.dont_write_bytecode = True
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import numpy as np

BASE = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/external_expert_20261007/physical_geometry_unlock_20261008/low_original_Ad4_nearby_source_roles')
SOURCE_SHA = 'f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'
CANDIDATE_SHA = 'cac37b8c14d72bf46421b38a845736f5535668deeab6e41598396835a6e16954'
RECEIPT_SHA = 'baf8a32a2bedc7a9aaf4e3c4e2a9260242091398be99e1dff42fe00f5311ee07'
OLD739_SHA = '7c609bf13ccadb5425f8f573f452e0f6666d707ad36ed037bcae19af4876877c'
REVIEW_NAMES = (
    ('independent_pages000_086_material_review_v1.json', '5e1d81963c20046451ab6000d90c8cd4d017e8d7343a822b13d2c43b082e2708', 'ad_evidence', range(0, 87)),
    ('independent_middle_087_173_material_review_SOURCE_ONLY_v1.json', 'ec9fc38d288d2a3383256d6f7d12b9510f06c3d5ece65f5c0a0b6619048cc8b6', 'source_review', range(87, 174)),
    ('root_pages174_259_all_original_points_actual_RGB_material_QUALIFIED_v1.json', '545e645cf75d595c93eccb572414fcbfdcc7a58fdfae510bf46b3c5d798fb4c1', 'root', range(174, 260)),
)
NUMERIC_KEYS = ('original_raw_cut5_point_index', 'original_raw_pair_index',
    'original_raw_pair_local_index', 'original_selected_cut5_row',
    'original_full_low_bundle_row', 'current_original_prefix_bundle_index',
    'source_pts', 'target_pts', 'source_native_pair_pts', 'source_uv',
    'target_uv', 'source_sigma_px')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read(path, expected=None):
    if expected is not None and sha(path) != expected:
        raise ValueError(f'Frozen input changed: {path}')
    return json.loads(Path(path).read_text(encoding='utf8'))


def check_ref(r):
    if sha(r['path']) != r['sha256']:
        raise ValueError(f'Frozen source reference changed: {r["path"]}')


def arrays(path, keys):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in keys}


def write_immutable(path, data):
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text(encoding='utf8') != text:
            raise ValueError(f'Preserve frozen existing packet: {path}')
    else:
        path.write_text(text, encoding='utf8')
    return ref(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--authority', required=True)
    a = ap.parse_args()
    if not a.authority.strip():
        raise ValueError('Explicit original-source material correction authority required')
    review_dir = BASE / 'full_cut5_region9288_actual_RGB_review'
    candidate_path = BASE / 'full_cut5_source_only_overlap/original_low_cut5_full_source_overlap_candidates_v2.json'
    receipt_path = review_dir / 'actual_original_cut5_9288_region_candidates_RGB_receipt_v1.json'
    old_path = BASE / 'secondary686_plus_low53_original_FIT_far_board_role_ADMITTED_v1.json'
    candidate = read(candidate_path, CANDIDATE_SHA)
    receipt = read(receipt_path, RECEIPT_SHA)
    old = read(old_path, OLD739_SHA)
    assert candidate['source_video_sha256'] == SOURCE_SHA and candidate['time_base'] == '1/15360'
    assert receipt['source_candidates'] == ref(candidate_path)
    assert old['original_unknown_motion_points_retyped'] == 739
    assert old['source_video_sha256'] == SOURCE_SHA and len(old['source_packets']) == 4
    for r in old['source_packets']:
        check_ref(r)
    for k in ('original_source_ledger', 'original_selected_rows', 'original_raw_motion',
              'clock_alias_authority', 'original_ice_source', 'static_UI_retirement',
              'original_typed_prefix_contract', 'existing_low53_role_packet'):
        check_ref(candidate[k])
    check_ref(receipt['source_region_masks'])
    check_ref(receipt['source_mask_archive'])
    region = read(receipt['source_region_masks']['path'])
    masks = arrays(receipt['source_mask_archive']['path'], ('original_source_index',
        'point_endpoint_reason_bits', 'point_both_endpoint_region_candidate'))
    both = np.all(masks['point_endpoint_reason_bits'] == 0, axis=1)
    assert np.array_equal(both, masks['point_both_endpoint_region_candidate'])
    assert len(candidate['records']) == 15218 and int(both.sum()) == 9288
    assert np.array_equal(masks['original_source_index'],
        [r['current_original_prefix_bundle_index'] for r in candidate['records']])
    original = {r['current_original_prefix_bundle_index']: r for r in candidate['records']}
    point_map = {r['current_original_prefix_bundle_index']: r for r in receipt['point_to_pair_sheet']}
    assert len(original) == 15218 and len(point_map) == 9288
    expected = {candidate['records'][i]['current_original_prefix_bundle_index'] for i in np.flatnonzero(both)}
    assert set(point_map) == expected
    sheets = {r['sheet_index']: r for r in receipt['sheets']}
    assert set(sheets) == set(range(260))
    # Hash each original page once; no repeated decode or image copies.
    for r in sheets.values():
        check_ref(r)
    human_refs, human_by_index, reviewed_pages = [], {}, set()
    for ordinal, (name, digest, reviewer, scope) in enumerate(REVIEW_NAMES):
        p = review_dir / name
        human = read(p, digest)
        h_ref = ref(p)
        human_refs.append(h_ref)
        if ordinal == 0:
            assert human['status'] == 'complete_source_only_material_review'
            assert not human['uncertain_source_only_rows'] and not human['remaining_sheet_indices']
            reviewed = human['reviewed_sheets']; rows = human['qualified_source_only_rows']
            assert human['source_RGB_receipt'] == ref(receipt_path)
        elif ordinal == 1:
            assert human['assignment']['reviewed_every_selected_source_and_target_mark']
            assert not human['unknown_points'] and not human['rejected_points']
            reviewed = human['sheets']; rows = human['qualified_material_points']
            assert human['source_receipt_sha256'] == RECEIPT_SHA
        else:
            assert human['status'] == 'actual_RGB_material_qualified_SOURCE_ONLY_not_runtime_admitted'
            assert human['unknown_material_points'] == 0 and human['people_or_UI_or_ice_points'] == 0
            assert human['nearfield_equipment_points'] == 0
            reviewed = human['actual_RGB_pages_reviewed']; rows = human['records']
            assert human['source_RGB_receipt'] == ref(receipt_path)
        pages = {r['sheet_index'] for r in reviewed}
        assert pages == set(scope) and not pages & reviewed_pages
        reviewed_pages |= pages
        for page in reviewed:
            index = page['sheet_index']
            assert page['path'] == sheets[index]['path'] and page['sha256'] == sheets[index]['sha256']
        wanted = {i for i, r in point_map.items() if r['sheet_index'] in pages}
        assert {r['current_original_prefix_bundle_index'] for r in rows} == wanted
        for row in rows:
            i = row['current_original_prefix_bundle_index']
            assert i not in human_by_index
            assert all(row[k] == original[i][k] for k in NUMERIC_KEYS)
            assert all(row[k] == point_map[i][k] for k in ('original_candidate_ordinal',
                'sheet_index', 'pair_review_ordinal', 'pair_point_ordinal', 'sheet_pair_row'))
            if ordinal == 0:
                assert row['material_identity_review'] == 'actual_both_endpoints_static_far_board_printed_material'
            elif ordinal == 1:
                cert = row['independent_source_material_review']
                assert cert['actual_source_and_target_endpoint_visually_reviewed']
                assert not cert['selected_point_on_people_ice_UI_near_equipment']
                assert cert['sheet_sha256'] == sheets[row['sheet_index']]['sha256']
            else:
                assert row['both_end_material_reviewed'] and row['material_role'] == 'board_face_far'
            human_by_index[i] = dict(reviewer=reviewer, human_material_review_ref_index=ordinal)
    assert reviewed_pages == set(range(260)) and set(human_by_index) == expected
    source = arrays(candidate['original_source_ledger']['path'], ('active_original_global_index',
        'source_uv_a', 'source_uv_b', 'source_endpoint_index_a', 'source_endpoint_index_b', 'original_sigma_xy'))
    raw = arrays(candidate['original_raw_motion']['path'], ('pair_pts', 'offsets', 'xy0', 'xy1'))
    selected = arrays(candidate['original_selected_rows']['path'], ('original_global_row_indices',))['original_global_row_indices']
    az = arrays(candidate['clock_alias_authority']['path'], ('pts', 'exposure_representative_pts'))
    aliases = dict(zip(map(int, az['pts']), map(int, az['exposure_representative_pts'])))
    clock = arrays(candidate['clock_state']['path'], ('aux__block_names', 'aux__block_lengths',
        'aux__canonical_pts', 'aux__qualified_original_motion_ice_contract_json'))
    bi = list(clock['aux__block_names']).index('low'); lengths = clock['aux__block_lengths'] // 4
    lo = int(lengths[:bi].sum()); stamps = clock['aux__canonical_pts'][lo:lo+int(lengths[bi])]
    assert bi == 3 and len(stamps) == 1851 and len(selected) == 319568
    assert len(source['active_original_global_index']) == 434991
    old53 = read(candidate['existing_low53_role_packet']['path'])
    old53_indices = {r['current_original_prefix_bundle_index'] for r in old53['records']}
    typed = {r['global_bundle_index'] for r in read(candidate['original_typed_prefix_contract']['path'])}
    ice3 = set(json.loads(str(clock['aux__qualified_original_motion_ice_contract_json'].item()))['global_bundle_indices'])
    retired = {r['current_bundle_motion_row'] for r in read(candidate['static_UI_retirement']['path'])['records']}
    known_ice = {r['original_global_index'] for r in read(candidate['original_ice_source']['path'])['qualified_rows'] if r['split'] == 'fit'}
    assert len(old53_indices) == 53 and len(typed) == 274 and len(ice3) == 3 and len(retired) == 89
    assert not expected & (old53_indices | typed | ice3 | retired)
    qualified = []
    for mapped in receipt['point_to_pair_sheet']:
        i = int(mapped['current_original_prefix_bundle_index']); row = original[i]
        full = int(row['original_full_low_bundle_row']); g = int(row['original_raw_cut5_point_index'])
        j = int(row['original_raw_pair_index']); local = int(row['original_raw_pair_local_index'])
        s = int(row['original_selected_cut5_row']); native = tuple(map(int, raw['pair_pts'][j]))
        assert full == 31987+s == int(source['active_original_global_index'][i])
        assert g == int(selected[s]) == int(raw['offsets'][j])+local and g not in known_ice
        assert native == tuple(row['source_native_pair_pts'])
        assert (aliases[native[0]], aliases[native[1]]) == (row['source_pts'], row['target_pts'])
        assert int(stamps[source['source_endpoint_index_a'][full]]) == row['source_pts']
        assert int(stamps[source['source_endpoint_index_b'][full]]) == row['target_pts']
        assert row['source_pts'] != row['target_pts']
        for values, key in ((raw['xy0'][g], 'source_uv'), (raw['xy1'][g], 'target_uv'),
            (source['source_uv_a'][full], 'source_uv'), (source['source_uv_b'][full], 'target_uv'),
            (source['original_sigma_xy'][full], 'source_sigma_px')):
            assert np.array_equal(values, row[key])
        r = deepcopy(row)
        r.update({k: mapped[k] for k in ('original_candidate_ordinal', 'sheet_index',
            'pair_review_ordinal', 'pair_point_ordinal', 'sheet_pair_row')})
        r.update(split='fit', material_role_review='confirmed_static_current_far_board_printed_material',
            both_end_original_native_RGB_material_reviewed=True,
            source_only_material_description='Stationary printed far inner board or its white board face, independently viewed at both actual native RGB endpoints; original matching precision and sigma retained.',
            actual_original_RGB_material_certificate=dict(**human_by_index[i],
                actual_source_and_target_endpoint_visually_reviewed=True,
                selected_point_on_people_ice_UI_near_equipment=False,
                original_UV_sigma_PTS_and_correspondence_reestimated=False,
                material_role_not_inferred_from_navigation_or_model_error=True))
        assert all(r[k] == row[k] for k in row)  # Preserve all candidate source/history fields.
        qualified.append(r)
    pending = [r['current_original_prefix_bundle_index'] for r in candidate['records'] if r['current_original_prefix_bundle_index'] not in expected]
    assert len(qualified) == 9288 and len(pending) == 5930
    qualification = dict(schema='rinkspace.original-low-FIT-source-only-static-board-material-qualification.v1',
        status='actual_original_RGB_material_qualified_SOURCE_ONLY', source_video_sha256=SOURCE_SHA,
        time_base='1/15360', source_candidate_packet=deepcopy(receipt['source_candidates']),
        actual_RGB_receipt=ref(receipt_path), human_material_reviews=human_refs,
        source_qualified_far_board_count=9288, source_qualified_actual_pair_count=780,
        actual_original_RGB_pages_reviewed=260, actual_source_exposures=845,
        new_query_or_scalar_rows=0, original_UV_PTS_sigma_and_correspondence_unchanged=True,
        HELD_or_diagnostic_tracks_not_promoted_to_FIT=True,
        subpixel_correspondence_accuracy_or_sigma_not_reestimated=True,
        navigation_mask_not_material_identity=True, material_role_not_selected_by_model_error=True,
        source_region_masks=deepcopy(receipt['source_region_masks']),
        source_mask_archive=deepcopy(receipt['source_mask_archive']),
        remaining_unqualified_source_candidate_count=5930,
        remaining_unqualified_source_candidate_prefix_indices=pending,
        remaining_unqualified_candidates_keep_unknown_depth_role=True,
        whole_low_original_motion_not_reclassified=True,
        uniqueness_and_existing_role_disjoint_check=dict(qualified_unique_prefix_indices=9288,
            qualified_unique_original_raw_point_indices=len({r['original_raw_cut5_point_index'] for r in qualified}),
            overlap_old_low53=0, overlap_existing_typed274=0, overlap_ice3=0, overlap_retired89=0,
            overlap_original_known_ice=0),
        admitted_source_only_rows=qualified, authority=a.authority,
        source_operator_residual_invariance_not_claimed=True,
        runtime_roles_weights_rows_or_target_not_modified_by_this_source_only_tool=True,
        script=ref(Path(__file__).resolve()))
    for k in ('original_source_ledger', 'original_selected_rows', 'original_raw_motion',
              'clock_alias_authority', 'clock_state', 'original_ice_source', 'static_UI_retirement',
              'original_typed_prefix_contract', 'existing_low53_role_packet'):
        qualification[k] = deepcopy(candidate[k])
    q_ref = write_immutable(review_dir / 'cut5_original_FIT9288_actual_RGB_far_board_material_QUALIFIED_SOURCE_ONLY_v1.json', qualification)
    admitted = deepcopy(qualification)
    admitted.update(schema='rinkspace.original-low-FIT-source-qualified-far-board-role-admission.v1',
        status='explicit_original_motion_source_role_revision_authorized', new_source_scalar_rows=0,
        admission=dict(authority=a.authority, parent_candidate=q_ref,
            unknown_depth_operator_changed_explicitly=True, source_r_invariance_not_claimed=True,
            source_scalar_rows_added=0))
    a_ref = write_immutable(review_dir / 'cut5_original_FIT9288_actual_RGB_far_board_role_ADMITTED_v1.json', admitted)
    combined = deepcopy(old)
    combined.update(source_packets=deepcopy(old['source_packets']) + [a_ref],
        authority=a.authority, original_unknown_motion_points_retyped=10027,
        historical_739_parent_packet=ref(old_path),
        inherited_739_children_preserved_in_order=True, additional_original_low_material_roles=9288,
        original_motion_material_role_count_by_block=dict(secondary=686, low=9341),
        planned_finite_point_count_by_block=[13, 0, 692, 9812, 0, 0],
        remaining_unqualified_candidate_count=5930, whole_low_original_motion_not_reclassified=True,
        source_r_invariance_not_claimed=True)
    c_ref = write_immutable(BASE / 'secondary686_plus_low9341_original_FIT_far_board_role_ADMITTED_v1.json', combined)
    print(json.dumps(dict(source_qualification=q_ref, admitted_low9288=a_ref, combined10027=c_ref,
        actual_RGB_qualified_points=9288, unknown_candidate_points=5930, inherited_roles=739,
        source_rows_added=0, source_UV_sigma_PTS_bitexact=True), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
