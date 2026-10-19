"""Rebuilt mesh/attachment consistency; fixtures are synthetic, not video QA."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location('joint_export', Path(__file__).resolve().parents[1] / 'scripts/export_joint_scene_model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def synthetic_scene():
    return dict(schema_version='rinkspace.scene3d.v1', model_id='synthetic', revision='source',
        coordinate_system={'unit': 'synthetic', 'metres_per_unit': None},
        parameters=dict(bounds=[-10., -5., 10., 5.], corner_radius=1.,
            corner_radii=dict(upper_left=1.2, upper_right=1.8, lower_left=1., lower_right=2.),
            grid_spacing=2., arc_steps_per_quarter=12, board_height=1.1, board_thickness=.1),
        ice_features=[dict(id='circle', center_xyz=[-5., 0., 0.], radius=1.5),
                      dict(id='line', segments_xyz=[[[9.5, 5., 0.], [9.5, -5., 0.]]]),
                      dict(id='outside_circle', center_xyz=[11., 0., 0.], radius=1.)],
        advertisements=[dict(id='original_advert', attachment_s=987., board_panel_id='old_panel',
            anchor_xyz=[0., 5., None], reference_schematic_xy=[0., 6.],
            source={'original_source': 'synthetic'}, placement_status='schematic',
            height_above_ice=None, height=None, width_along_board=None, use_as_ice_point=False)],
        meshes=[], source={})


def test_independent_corners_mesh_grid_and_ads_rebuilt():
    original = synthetic_scene()
    scene, geometry, checks = module.rebuild_scene(original)
    assert checks['mesh_and_grid_checks_passed']
    assert geometry['checks']['thick_board_mesh_watertight']
    assert scene['parameters']['corner_radii'] == original['parameters']['corner_radii']
    assert checks['semantic_checks']['features_outside_rink_or_invalid'] == ['outside_circle']
    assert scene['ice_features'][2] == original['ice_features'][2]
    assert scene['ice_features'][1]['joint_export_boundary_clipped']
    assert scene['ice_features'][1]['joint_unclipped_segments_xyz'] == original['ice_features'][1]['segments_xyz']
    ad = scene['advertisements'][0]
    assert ad['previous_geometry_attachment']['attachment_s'] == 987.
    assert not ad['attachment_reconstruction']['old_s_reused']
    assert ad['source'] == original['advertisements'][0]['source']
    assert ad['anchor_xyz'][2] is None
    panel = next(p for p in scene['board_surface']['panels'] if p['id'] == ad['board_panel_id'])
    assert panel['s_range'][0] <= ad['attachment_s'] <= panel['s_range'][1]
    n = len(scene['ice_boundary']['points_xyz'])
    ice = scene['meshes'][0]; boards = scene['meshes'][1]
    np.testing.assert_array_equal(np.asarray(ice['positions'])[:n], np.asarray(boards['positions'])[:n])
    assert not scene['joint_refinement']['boundary_mesh_rebuild_required']


def test_valid_unequal_corners_can_exceed_half_width_when_neighbours_fit():
    radii = dict(upper_left=4., lower_left=1., upper_right=1., lower_right=1.)
    boundary = module.rounded_boundary([-10., -3., 10., 3.], radii, .5, 24)
    np.testing.assert_allclose(boundary.min(axis=0), [-10., -3.])
    np.testing.assert_allclose(boundary.max(axis=0), [10., 3.])
    edges = np.roll(boundary, -1, axis=0) - boundary
    assert np.all(np.linalg.norm(edges, axis=1) > 1e-9)
    cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    assert np.min(cross) >= -1e-9
    radii['lower_left'] = 3.
    with pytest.raises(ValueError, match='adjacent corner'):
        module.rounded_boundary([-10., -3., 10., 3.], radii, .5, 24)


def test_export_preserves_existing_output_revision(tmp_path):
    dest = tmp_path / 'new_revision'
    result = module.export(None, None, dest, semantic=synthetic_scene())
    assert result['mesh_and_grid_checks_passed']
    loaded = module.load(dest / 'scene.json')
    assert loaded['source']['geometry_json_sha256'] == module.sha(dest / 'scene_geometry.json')
    digest = module.sha(dest / 'scene.json')
    with pytest.raises(FileExistsError):
        module.export(None, None, dest, semantic=synthetic_scene())
    assert module.sha(dest / 'scene.json') == digest


def test_user_confirmed_advertisement_ring_survives_export_without_schematic_reattachment(tmp_path):
    semantic = synthetic_scene()
    advertisements = []
    for number in range(1, 27):
        partner = number + 13 if number <= 13 else number - 13
        advertisements.append(dict(
            id=f'advert_{number:02}', label=f'content_{(number - 1) % 13}',
            user_number=number, visual_content_id=f'content_{(number - 1) % 13}',
            qualitative_side='upper_long_side' if number <= 13 else 'lower_long_side',
            perimeter_order_counterclockwise=number,
            same_content_partner_number=partner,
            same_content_partner_id=f'advert_{partner:02}',
            relative_position_status='user_confirmed_side_order_and_schematic_layout',
            physical_position_status='unresolved_requires_real_video_fit',
            attachment_s=None, board_panel_id=None, anchor_xyz=[None, None, None],
            width_along_board=None, height=None, height_above_ice=None,
            reference_schematic_xy=[float(number), 6.] if number % 2 else None,
            historical_schematic_attachment={'attachment_s': float(number),
                                             'anchor_xyz': [float(number), 5., None]},
        ))
    semantic['advertisements'] = advertisements
    semantic['advertisement_topology'] = {'source': 'user-edited PPT',
                                         'pair_rule': '1-13 correspond to 14-26'}

    dest = tmp_path / 'confirmed_ring'
    module.export(None, None, dest, semantic=semantic)
    rebuilt = module.load(dest / 'scene.json')
    assert rebuilt['advertisements'] == advertisements
    assert rebuilt['advertisement_topology'] == semantic['advertisement_topology']
    assert 'historical schematic attachments are provenance only' in rebuilt['uncertainty']['advertisements']


def test_user_confirmed_physical_attachment_needs_explicit_update():
    semantic = synthetic_scene()
    semantic['advertisements'][0]['relative_position_status'] = 'user_confirmed_side_order_and_schematic_layout'
    with pytest.raises(ValueError, match='explicit video-backed geometry update'):
        module.rebuild_scene(semantic)


def test_real_high_view_corner_evidence_survives_next_scene_rebuild():
    root = Path(__file__).resolve().parents[1]
    current = module.load(root / 'outputs/rink_model/ankara_2026_shared/v0.23-high-view-ad-corners/scene.json')
    rebuilt, _, checks = module.rebuild_scene(current)
    assert checks['mesh_and_grid_checks_passed']
    assert rebuilt['advertisement_topology'] == current['advertisement_topology']
    assert rebuilt['advertisement_high_view_update'] == current['advertisement_high_view_update']
    assert rebuilt['advertisements'] == current['advertisements']
    assert rebuilt['source']['high_view_ad_corner_observations_sha256'] == current['source']['high_view_ad_corner_observations_sha256']


def test_export_retains_latent_details_nonconvergence_and_alternatives():
    semantic = synthetic_scene()
    semantic['board_landmarks'] = [{'id': 'observed_detail', 'estimated_xyz': [0., 5., .4],
        'parameter_status': 'conditional', 'physical_xyz_verified': False,
        'degeneracy_warnings': ['single_installation_only']}]
    fit = {'optimizer': {'success': False, 'nfev': 220, 'message': 'budget'},
           'local_numerical_identifiability': {'rank_at_relative_1e-6': 3, 'parameter_count': 4},
           'frames': [], 'installations': []}
    comparison = {'holdout_read_or_used': False, 'ambiguous_feature': 'synthetic_circle'}
    scene, _, _ = module.rebuild_scene(semantic, fit, comparison=comparison)
    assert scene['board_landmarks'] == semantic['board_landmarks']
    assert scene['joint_refinement']['optimizer'] == fit['optimizer']
    assert scene['joint_refinement']['training_candidate_comparison'] == comparison
    assert 'did_not_report_convergence' in scene['uncertainty']['numerical_solution']
