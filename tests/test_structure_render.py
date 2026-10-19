"""Disposable drawing contracts; synthetic scenes are not rink observations."""
import copy

import cv2
import numpy as np
from PIL import Image
import pytest

from rinkspace.structure_render import (
    _matrix, _visible_segment, _distort, _distorted_segments,
    render_structure_overview, render_structure_projection,
)


def fixture_model():
    angles = np.linspace(0, 2*np.pi, 49)
    ring = np.c_[np.cos(angles)*13, np.sin(angles)*13].tolist()
    return {'model_id': 'synthetic-structure-fixture', 'revision': 'test-only',
        'geometry_level': 'euclidean_conditional', 'unit': 'normalized_non_metric',
        'structure': {'bounds': [-100, -50, 100, 50], 'boundary': {
            'id': 'B', 'kind': 'boundary', 'status': 'assumed', 'label': '假设完整边界',
            'points_model': [[-90, -40], [90, -40], [100, -30], [100, 30], [90, 40],
                             [-90, 40], [-100, 30], [-100, -30]]},
            'primitives': [
                {'id': 'M', 'kind': 'direction', 'status': 'inferred', 'label': '主方向',
                 'points_model': [[-70, 0], [70, 0]], 'label_anchor_model': [0, 3]},
                {'id': 'R1', 'kind': 'circle', 'status': 'observed', 'label': '给定图案', 'points_model': ring},
                {'id': 'E1', 'kind': 'line', 'status': 'inferred', 'label': '端部编号',
                 'points_model': [[75, -25], [75, 25]]},
                {'id': 'C1', 'kind': 'point', 'status': 'assumed', 'label': '角部编号', 'points_model': [[90, 40]]},
                {'id': 'A1', 'kind': 'arc', 'status': 'assumed', 'label': '给定角弧',
                 'points_model': [[-90+10*np.cos(t), 30+10*np.sin(t)] for t in np.linspace(np.pi/2, np.pi, 24)]},
            ]},
        'grid': {'bounds': [-4, -4, 4, 4], 'x_values': [-80, -40, 0, 40, 80],
                 'y_values': [-30, 0, 30], 'status': 'assumed'},
        # Deliberately tiny legacy support must never shrink the whole scene.
        'support_polygon_image': [[300, 170], [305, 170], [305, 175], [300, 175]]}


def test_overview_keeps_whole_boundary_explicit_structure_bounds_and_statuses(tmp_path):
    model = fixture_model(); before = copy.deepcopy(model)
    result = render_structure_overview(model, tmp_path/'overview.png', size=(1000, 650))
    assert Image.open(result['image_path']).size == (1000, 650)
    assert result['declared_bounds'] == [-100, -50, 100, 50]
    assert result['boundary_closed'] and not result['independent_geometry_validated']
    boundary = next(p for p in result['primitives'] if p['id'] == 'B')
    assert boundary['visible_segments'] == 8 and boundary['closed']
    assert set(result['status_legend']) == {'observed', 'inferred', 'assumed'}
    assert result['primitive_status_counts'] == {'assumed': 3, 'inferred': 2, 'observed': 1}
    assert model == before
    h = np.array(result['H_overview_from_model'])
    assert abs(h[0, 0]) == abs(h[1, 1]) and h[1, 1] < 0


def test_projection_uses_full_ice_and_person_union_not_old_ring_support(tmp_path):
    model = fixture_model(); image = np.full((360, 640, 3), 100, np.uint8)
    original = image.copy()
    matrix = np.array([[2.5, 0, 320], [0, 2.5, 180], [0, 0, 1.]])
    ice = np.zeros(image.shape[:2], np.uint8);ice[75:285] = 255
    people = [[[275, 115], [330, 115], [330, 240], [275, 240]],
              [[305, 120], [365, 120], [365, 240], [305, 240]]]
    result = render_structure_projection(image, model, matrix, tmp_path/'projection.png',
        ice_mask=ice, person_polygons_image=people)
    rendered = cv2.imread(result['image_path'])
    # Two overlapping exclusions remain excluded, with no even-odd reopening.
    assert np.array_equal(rendered[125:235, 280:360], image[125:235, 280:360])
    assert np.array_equal(rendered[62:74], image[62:74])
    assert np.array_equal(image, original)
    assert result['clipping']['person_polygon_count'] == 2
    assert result['clipping']['legacy_landmark_neighbourhoods_used'] is False
    assert not result['image_resized']
    # Strong visible structure far from the irrelevant old four-ring support.
    assert np.count_nonzero(np.any(rendered[80:280, 60:250] != 100, axis=2)) > 500


def test_projection_horizon_rejects_crossing_and_safely_clips_extreme_endpoints():
    horizon = _matrix([[1, 0, 100], [0, 1, 100], [.02, 0, 1]])
    segment, reason = _visible_segment(horizon, [-100, 0], [100, 0], 640, 360)
    assert segment is None and reason == 'projective_horizon'
    # Very distant points on the same finite side still clip to image bounds.
    segment, reason = _visible_segment(np.eye(3), [-1e12, 180], [1e12, 180], 640, 360)
    assert reason is None and np.isfinite(segment).all()
    np.testing.assert_allclose(segment, [[0, 180], [639, 180]], atol=.001)
    # An overall negative matrix represents the same projection.
    other, reason = _visible_segment(-np.eye(3), [-100, 180], [900, 180], 640, 360)
    assert reason is None
    np.testing.assert_allclose(other, [[0, 180], [639, 180]], atol=1e-8)


def test_horizon_rejections_are_reported_and_do_not_bridge_image(tmp_path):
    model = fixture_model()
    result = render_structure_projection(np.full((360, 640, 3), 60, np.uint8), model,
        [[1, 0, 320], [0, 1, 180], [.02, 0, 1]], tmp_path/'horizon.png')
    assert sum(row['rejected_projective_horizon_segments'] for row in result['primitives']) > 0
    assert result['status'] == 'conditional_structure_visualization_only'


def test_assumed_path_dash_phase_survives_dense_arc_sampling(tmp_path):
    model = fixture_model()
    # Dense samples must not reset every short dash into one misleading solid line.
    model['structure']['primitives'] = [{'id':'dense', 'kind':'arc', 'status':'assumed',
        'points_model': [[x, 5] for x in np.linspace(-80, 80, 401)], 'label_anchor_model':[-95,-35]}]
    model['grid'].update(x_values=[], y_values=[])
    result = render_structure_projection(np.full((360, 640, 3), 40, np.uint8), model,
        [[2,0,320],[0,2,180],[0,0,1]], tmp_path/'dashed.png')
    image = cv2.imread(result['image_path'])
    changed = np.any(image[190, 170:470] != 40, axis=1)
    assert changed.any() and (~changed).sum() > 35


@pytest.mark.parametrize('change', ['unknown_status', 'nonfinite_point', 'bad_bounds'])
def test_invalid_or_unlabelled_structure_is_not_silently_drawn(tmp_path, change):
    model = fixture_model()
    if change == 'unknown_status':model['structure']['boundary']['status'] = 'passed'
    if change == 'nonfinite_point':model['structure']['boundary']['points_model'][0][0] = float('nan')
    if change == 'bad_bounds':model['structure']['bounds'] = [100, -50, -100, 50]
    with pytest.raises(ValueError):render_structure_overview(model, tmp_path/'bad.png')


def test_projection_rejects_mismatched_mask_or_singular_H(tmp_path):
    model = fixture_model(); image = np.zeros((360,640,3),np.uint8)
    with pytest.raises(ValueError, match='mask'):
        render_structure_projection(image,model,np.eye(3),tmp_path/'bad.png',ice_mask=np.ones((10,10)))
    with pytest.raises(ValueError, match='singular'):
        render_structure_projection(image,model,np.zeros((3,3)),tmp_path/'bad.png')


def test_candidate_radial_formula_samples_curve_and_clips_after_warp():
    np.testing.assert_allclose(_distort(np.array([100.,0]),
        {'k1':.2,'centre_image':[0,0],'scale_image':100}),[120,0])
    distortion={'k1':.6,'centre_image':[320,180],'scale_image':320}
    segments=list(_distorted_segments(np.eye(3),[100,100],[540,100],640,360,distortion))
    assert len(segments)==32 and all(reason is None for _,reason in segments)
    points=np.vstack([line for line,_ in segments])
    assert np.ptp(points[:,1])>20  # a straight model path becomes curved
    inward={**distortion,'k1':-.2}
    segments=list(_distorted_segments(np.eye(3),[-80,180],[720,180],640,360,inward))
    assert segments[0][1] is None and 0<segments[0][0][0,0]<100
    assert segments[-1][1] is None and 540<segments[-1][0][-1,0]<640
    crossing=list(_distorted_segments(_matrix([[1,0,320],[0,1,180],[.02,0,1]]),
        [-100,0],[100,0],640,360,distortion))
    assert any(reason=='projective_horizon' for _,reason in crossing)
    for segment,reason in crossing:
        if reason is None:
            assert np.isfinite(segment).all() and (segment>=0).all()


def test_radial_render_is_explicit_candidate_and_retains_foreground_exclusion(tmp_path):
    image=np.full((360,640,3),100,np.uint8)
    distortion={'k1':-.12,'centre_image':[320,180],'scale_image':320}
    result=render_structure_projection(image,fixture_model(),[[2,0,320],[0,2,180],[0,0,1]],
        tmp_path/'radial.png',distortion=distortion,
        person_polygons_image=[[[280,120],[360,120],[360,240],[280,240]]])
    rendered=cv2.imread(result['image_path'])
    assert np.array_equal(rendered[125:235,285:355],image[125:235,285:355])
    assert result['distortion']['status']=='candidate_not_calibrated'
    assert result['distortion']['k1']==-.12 and result['samples_per_input_segment']==32
    assert not result['independent_geometry_validated']
    with pytest.raises(ValueError,match='scale_image'):
        render_structure_projection(image,fixture_model(),np.eye(3),tmp_path/'bad.png',
            distortion={**distortion,'scale_image':0})


@pytest.mark.parametrize('radial', [False, True])
def test_explicit_front_branch_rejects_back_projected_geometry_points_and_labels(tmp_path, radial):
    image=np.full((360,640,3),60,np.uint8)
    # The negative homogeneous branch projects into the image mathematically.
    # With an explicit positive front gauge, none of its lines, points, arrows
    # or labels may be painted. None keeps the legacy projective convention.
    matrix=-np.array([[2.,0,320],[0,2.,180],[0,0,1.]])
    distortion={'k1':.05,'centre_image':[320,180],'scale_image':320} if radial else None
    result=render_structure_projection(image,fixture_model(),matrix,tmp_path/'back.png',
        distortion=distortion,front_branch_sign=1)
    rendered=cv2.imread(result['image_path'])
    assert result['geometry_drawn_pixels_before_labels']==0
    assert result['rejected_behind_plane_segments']>0
    assert result['clipped_branch_segments']==0
    assert all(p['visible_segments']==0 for p in result['primitives'])
    assert np.array_equal(rendered[90:290],image[90:290])
    assert result['status']=='conditional_structure_visualization_only'
    assert result['independent_geometry_validated'] is False
    # Selecting the actual negative front gauge restores the same finite scene.
    front=render_structure_projection(image,fixture_model(),matrix,tmp_path/'front.png',
        distortion=distortion,front_branch_sign=-1)
    assert front['geometry_drawn_pixels_before_labels']>100
    assert front['rejected_behind_plane_segments']==0


@pytest.mark.parametrize('radial', [False, True])
def test_crossing_segment_keeps_only_visible_positive_ray_without_back_branch(radial):
    # y=180 on both branches; the back branch would falsely reappear at x<36.
    # The legitimate positive side enters from the right edge and ends at x=140.
    matrix=_matrix([[1,0,320],[3.6,1,180],[.02,0,1]])
    counts={'clipped_branch_segments':0}
    if radial:
        candidates=list(_distorted_segments(matrix,[-1000,0],[100,0],640,360,
            {'k1':0.,'centre_image':[320,180],'scale_image':320},
            front_branch_sign=1,branch_counts=counts))
        assert any(reason=='behind_plane' for _,reason in candidates)
    else:
        candidates=[_visible_segment(matrix,[-1000,0],[100,0],640,360,
            front_branch_sign=1,branch_counts=counts)]
        np.testing.assert_allclose(candidates[0][0],[[639,180],[140,180]],atol=1e-6)
    visible=[segment for segment,reason in candidates if reason is None]
    assert visible and counts['clipped_branch_segments']==1
    points=np.vstack(visible)
    assert np.isfinite(points).all() and points[:,0].min()>=140-1e-6
    assert points[:,0].max()<=639 and np.allclose(points[:,1],180)


def test_radial_candidate_cannot_fold_distant_geometry_back_into_image():
    from rinkspace.structure_render import _distort
    distortion={'k1':-.1,'centre_image':[320.,180.],'scale_image':320.}
    assert _distort(np.array([340.,180.]),distortion) is not None
    # At sqrt(10) radii the polynomial maps a distant point exactly onto its
    # centre, but that lies beyond the radial turning point and is not valid.
    assert _distort(np.array([320.+320*np.sqrt(10),180.]),distortion) is None
