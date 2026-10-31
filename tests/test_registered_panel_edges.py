import cv2
import numpy as np
import pytest

from rinkspace.session.registered_panel_edges import read_registered_panel_edges


def _scene():
    rgb = np.full((360, 640, 3), 235, np.uint8)
    ramp = np.linspace(0., 38., 400)
    rgb[80:280, 120:520] = np.rint(np.array([18., 42., 110.])+ramp[:, None]).astype('uint8')
    # A genuine local dark-to-light transition in blue ink is not an exterior.
    rgb[80:280, 225:250] += 35
    # The enclosed glyph is a competing white source interface, not board white.
    rgb[120:245, 190:211] = 235
    rgb = cv2.blur(rgb, (9, 1))
    corners = np.array([[150., 80.], [520., 80.], [520., 280.], [270., 280.]])
    paths = [np.array([corners[i], corners[(i+1) % 4]]) for i in range(4)]
    binding = dict(source_sha256='a'*64, source_pts=256, time_base='1/12800',
        shot_id='synthetic', independent_source_pattern_match_verified=True)
    return rgb, paths, binding


def _read(rgb, paths, binding, exclusion=None, **kwargs):
    if exclusion is None: exclusion = np.zeros(rgb.shape[:2], bool)
    return read_registered_panel_edges(rgb, paths, exclusion, binding=binding,
        candidate_material_validator=lambda row: True, **kwargs)


def test_tilted_interior_navigation_recovers_actual_exterior_and_rejects_glyph():
    rgb, paths, binding = _scene()
    before = rgb.copy(); paths_before = [p.copy() for p in paths]
    proof = {}
    rows, record = _read(rgb, paths, binding, proof_sink=lambda stage, arrays: proof.update({stage: arrays}))
    left = [r for r in rows if r['curve'] == 3]
    assert len(left) >= 6
    np.testing.assert_allclose(np.array([r['uv'] for r in left])[:, 0], 119.5, atol=.75)
    assert np.ptp(np.array([r['uv'] for r in left])[:, 1]) >= 140.
    raw = [r for r in record['candidate_records'] if r['stage'] == 'registered_navigation_audit']
    interior = [r for r in raw if 170. < r['uv'][0] < 270. and 100. < r['uv'][1] < 250.]
    assert interior
    assert all('outer_quantile_guard_not_component_exterior_board_white' in r['pending_reasons'] for r in interior)
    glyph = [r for r in interior if 182. < r['uv'][0] < 215.]
    assert glyph
    assert all(r['disposition'] == 'pending' for r in glyph)
    assert record['source_component_used_for_topology_only']
    assert record['source_search_chart']['source_contour_UV_used_as_measurement'] is False
    assert set(proof) == {'source_topology', 'registered_navigation_audit', 'actual_source_envelope_search'}
    for row in rows:
        np.testing.assert_array_equal(row['uv'], np.array(row['actual_source_quantiles'])[0, 1])
        assert not row['model_UV_synthesized']
    np.testing.assert_array_equal(rgb, before)
    for path, original in zip(paths, paths_before): np.testing.assert_array_equal(path, original)


def test_source_exclusion_still_blocks_the_complete_actual_quantile_span():
    rgb, paths, binding = _scene()
    mask = np.zeros(rgb.shape[:2], bool)
    mask[145:205, 95:145] = True
    rows, record = _read(rgb, paths, binding, mask)
    left = np.array([r['uv'] for r in rows if r['curve'] == 3])
    assert len(left)
    assert not ((left[:, 1] >= 145) & (left[:, 1] < 205)).any()
    assert any('actual_transition_span_crosses_source_exclusion' in r['pending_reasons']
               for r in record['candidate_records'])


def test_pattern_identity_is_a_required_caller_precondition():
    rgb, paths, binding = _scene()
    binding['independent_source_pattern_match_verified'] = False
    with pytest.raises(ValueError, match='Independent source pattern'):
        _read(rgb, paths, binding)
def test_old_row_audit_requires_unchanged_unmasked_complete_interior_source_guard():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp/audit_registered_exterior_rows.py'
    spec = importlib.util.spec_from_file_location('registered_exterior_audit', path)
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    rgb, paths, binding = _scene(); proof = {}
    _, record = _read(rgb, paths, binding, proof_sink=lambda stage, arrays: proof.update({stage: arrays}))
    raw = [c for c in record['candidate_records'] if c['stage'] == 'registered_navigation_audit']
    rows = [(i, dict(c, path_id='Ad3.left')) for i, c in enumerate(raw)]
    mask = np.zeros(rgb.shape[:2], bool)
    findings = audit.audit_rows(rgb, proof['source_topology'], mask, rows,
        record['candidate_records'], source_hash_matches=True)
    bad = [f for f in findings if f['withdrawal_authorized_by_this_audit']]
    assert bad
    assert all(f['entire_guard_inside_source_material'] and f['entire_guard_unexcluded'] for f in bad)
    assert all('outer_quantile_guard_not_component_exterior_board_white' in f['candidate_rejection_reasons'] for f in bad)
    assert any(not f['withdrawal_authorized_by_this_audit'] for f in findings)
    changed = audit.audit_rows(rgb, proof['source_topology'], mask, rows,
        record['candidate_records'], source_hash_matches=False)
    assert not any(f['withdrawal_authorized_by_this_audit'] for f in changed)


@pytest.mark.parametrize('bridge', [False, True])
def test_parallel_same_colour_rail_cannot_join_panel_exterior(bridge):
    rgb, _, binding = _scene()
    rgb[20:60, 105:535] = [30, 48, 125]
    if bridge:
        # A narrow leak from an imperfect source occluder joins the two dark
        # entities. Filling its raw external contour would include board white.
        rgb[55:90, 500:504] = [30, 48, 125]
    corners = np.array([[120., 25.], [520., 25.], [520., 280.], [120., 280.]])
    paths = [np.array([corners[i], corners[(i+1) % 4]]) for i in range(4)]
    rows, record = _read(rgb, paths, binding)
    top = np.array([r['uv'] for r in rows if r['curve'] == 0])
    assert len(top) >= 7
    np.testing.assert_allclose(top[:, 1], 79.5, atol=.75)
    rail = [c for c in record['candidate_records'] if 15 < c['uv'][1] < 65]
    assert rail
    assert not any(c['candidate_id'] in record['selected_candidate_ids'] for c in rail)
    assert any('inner_quantile_guard_not_actual_source_material' in c['pending_reasons'] for c in rail)
    assert any(not c['selected'] for c in record['source_topology']['source_core_components'])


def test_occluded_panel_piece_below_half_navigation_overlap_keeps_true_left_edge():
    rgb, paths, binding = _scene()
    mask = np.zeros(rgb.shape[:2], bool)
    mask[65:295, 230:285] = True
    proof = {}
    rows, record = _read(rgb, paths, binding, mask,
        proof_sink=lambda stage, arrays: proof.update({stage: arrays}))
    left = np.array([r['uv'] for r in rows if r['curve'] == 3])
    assert len(left) >= 6
    np.testing.assert_allclose(left[:, 0], 119.5, atol=.75)
    assert any(c['selected'] and c['registered_overlap_px'] < .5*c['area_px']
               for c in record['source_topology']['source_core_components'])
    topology = proof['source_topology']
    for row in rows:
        np.testing.assert_array_equal(row['uv'], np.array(row['actual_source_quantiles'])[0, 1])
        assert not row['model_UV_synthesized']
    assert not topology['actual_material'][mask[topology['lo'][1]:topology['hi'][1],
                                               topology['lo'][0]:topology['hi'][0]]].any()


@pytest.mark.parametrize('colour', [(120, 30, 25), (120, 30, 115)])
def test_two_dimensional_panel_topology_also_preserves_red_and_purple_materials(colour):
    rgb, paths, binding = _scene()
    rgb[80:280, 120:520] = colour
    rgb[120:245, 190:211] = 235
    rows, _ = _read(rgb, paths, binding)
    assert {r['curve'] for r in rows} == {0, 1, 2, 3}
    for row in rows:
        np.testing.assert_array_equal(row['uv'], np.array(row['actual_source_quantiles'])[0, 1])


def test_isolated_rail_without_two_dimensional_panel_remains_unresolved():
    rgb, paths, binding = _scene()
    rgb[:] = 235
    rgb[100:125, 120:520] = [30, 48, 125]
    rows, record = _read(rgb, paths, binding)
    assert not rows
    assert record['source_topology']['status'] in {
        'no_registered_two_dimensional_panel_core',
        'source_core_does_not_span_registered_panel_cross_section'}


def test_short_panel_uses_its_visible_source_scale_and_keeps_all_four_edges():
    rgb = np.full((150, 360, 3), 235, np.uint8)
    rgb[50:90, 40:320] = [45, 30, 110]
    corners = np.array([[40., 50.], [320., 50.], [320., 90.], [40., 90.]])
    paths = [np.array([corners[i], corners[(i+1) % 4]]) for i in range(4)]
    binding = _scene()[2]
    rows, _ = _read(rgb, paths, binding, spacing=12.)
    assert {r['curve'] for r in rows} == {0, 1, 2, 3}


def test_narrow_mask_hides_physical_edge_even_when_wide_search_has_clear_pixels():
    rgb,paths,binding=_scene()
    mask=np.zeros(rgb.shape[:2],bool);mask[120:240,95:145]=True
    rows,record=_read(rgb,paths,binding,mask,search_radius=64.,spacing=12.)
    left=np.array([r['uv']for r in rows if r['curve']==3])
    assert len(left)>=3
    assert (left[:,1]<120).any() and (left[:,1]>=240).any()
    assert not ((left[:,1]>=120)&(left[:,1]<240)).any()
    report=next(v for v in record['reports']if v['curve']==3)
    assert report['source_line_masked_queries']
    assert report['source_visibility_initial_report']['disposition']=='insufficient_distributed_source_line_span'
    for row in rows:
        np.testing.assert_array_equal(row['uv'],np.asarray(row['actual_source_quantiles'])[0,1])
