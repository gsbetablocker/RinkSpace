"""Physical observation and shared-fit contracts for the conditional whole rink."""
from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip('scipy')

from rinkspace.geometry import project
from rinkspace.whole_rink import (INVALID_PROJECTION_RESIDUAL, distorted_project,
    fit_whole_rink, infer_visible_branch_sign, observation_residual,
    undistort_observed_points, valid_projected_points)


def _line_spec(first_x=37., second_x=37., *, prior_only=False):
    """Two independent fixed image poses see the same physical separator."""
    parameters = {'separator_x': {'initial': 25., 'bounds': [10., 80.]}}
    if prior_only:
        parameters['unseen_radius'] = {'initial': 8., 'bounds': [2., 20.], 'prior_sigma': 2.}
    h1 = np.array([[3., 0., 70.], [0., 2., 25.], [0., 0., 1.]])
    h2 = np.array([[-2., .4, 620.], [.25, 3., 40.], [0., 0., 1.]])
    feature = {'id': 'separator', 'kind': 'axis_line', 'axis': 'x',
               'coordinate': 'separator_x', 'extent': [0., 100.]}
    observations = []
    for name, h, x in [('view_a', h1, first_x), ('view_b', h2, second_x)]:
        observations.append({'id': name+'_separator', 'view_id': name, 'feature_id': 'separator',
            'type': 'line', 'role': 'fit', 'sigma_image_px': 2.,
            'points_image': project(h, [[x, 10.], [x, 35.], [x, 65.], [x, 90.]]).tolist()})
    return {'parameters': parameters, 'features': [feature],
            'views': [{'id': 'view_a', 'pose_fixed': True, 'H_image_from_model_initial': h1.tolist()},
                      {'id': 'view_b', 'pose_fixed': True, 'H_image_from_model_initial': h2.tolist()}],
            'observations': observations, 'structure_bounds': [0., 0., 200., 100.]}


def test_visible_arc_residual_does_not_penalize_unseen_continuation():
    observation = {'type': 'curve', 'points_image': [[2., 1.], [5., 1.], [8., 1.]]}
    nearby_loop = {'kind': 'polyline', 'points': [[0., 0.], [10., 0.], [10., 10.], [0., 10.], [0., 0.]]}
    far_continuation = {'kind': 'polyline', 'points': [[0., 0.], [10., 0.], [1000., 1000.], [-1000., 1000.], [0., 0.]]}
    # The only measured arc is the lower segment. Neither extra coverage nor
    # thousands of units of hidden model curve may add reverse-distance terms.
    first = observation_residual(observation, nearby_loop, {}, np.eye(3))
    second = observation_residual(observation, far_continuation, {}, np.eye(3))
    np.testing.assert_allclose(first, [1., 1., 1.], atol=1e-12)
    np.testing.assert_allclose(second, first, atol=1e-12)
    assert len(second) == len(observation['points_image'])


def test_known_projective_pose_has_correct_point_and_line_normal_residuals():
    h = np.array([[3.1, .35, 80.], [-.15, 2.5, 45.], [.001, -.0006, 1.]])
    points = {'kind': 'point', 'points': [[12., 7.], [55., 73.]]}
    image = project(h, points['points'])
    ob = {'type': 'point', 'points_image': (image + [3., -4.]).tolist()}
    np.testing.assert_allclose(observation_residual(ob, points, {}, h).reshape(-1, 2),
                               [[-3., 4.], [-3., 4.]], atol=1e-10)
    line = {'kind': 'line', 'points': [[20., 0.], [20., 100.]]}
    a, b = project(h, line['points'])
    normal = np.array([a[1]-b[1], b[0]-a[0]])
    normal /= np.linalg.norm(normal)
    # Position along an identified straight line is not a point identity.
    samples = a + np.array([-.2, .15, .65, 1.3])[:, None]*(b-a) + 2.5*normal
    residual = observation_residual({'type': 'line', 'points_image': samples.tolist()}, line, {}, h)
    np.testing.assert_allclose(residual, 2.5, atol=1e-10)


def test_zero_distortion_preserves_supporting_line_semantics_outside_endpoints():
    feature={'kind':'line','points':[[0.,10.],[10.,10.]]}
    h=np.array([[3.,0.,150.],[0.,2.,120.],[0.,0.,1.]])
    observed=project(h,[[-20.,10.],[5.,10.],[40.,10.]])+[0.,3.]
    ob={'type':'line','points_image':observed.tolist()}
    raw=observation_residual(ob,feature,{},h)
    zero=observation_residual(ob,feature,{},h,
        {'k1':0.,'centre_image':[480.,270.],'scale_image':480.})
    np.testing.assert_allclose(zero,raw,atol=1e-10)
    np.testing.assert_allclose(np.abs(zero),3.,atol=1e-10)


def test_radial_line_uses_raw_normal_error_without_invented_endpoint_ties():
    feature={'kind':'line','points':[[0.,10.],[10.,10.]]}
    h=np.array([[3.,0.,150.],[0.,2.,120.],[0.,0.,1.]])
    d={'k1':-.06,'centre_image':[480.,270.],'scale_image':480.}
    pts=np.array([[-20.,10.],[5.,10.],[40.,10.]])
    observed=distorted_project(h,pts,d)
    residual=observation_residual({'type':'line','points_image':observed.tolist()},feature,{},h,d)
    np.testing.assert_allclose(residual,0.,atol=1e-8)
    raw_tangent=(distorted_project(h,pts+[1e-3,0],d)-
                 distorted_project(h,pts-[1e-3,0],d))
    normal=np.c_[-raw_tangent[:,1],raw_tangent[:,0]]
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    shifted=observed+3*normal
    residual,domain=observation_residual({'type':'line','points_image':shifted.tolist()},
        feature,{},h,d,return_domain=True)
    np.testing.assert_allclose(np.abs(residual),3.,atol=1e-6)
    assert domain['penalized_residual_count']==0


def test_known_projective_pose_matches_fresh_visible_circle_samples():
    h = np.array([[3.1, .35, 80.], [-.15, 2.5, 45.], [.001, -.0006, 1.]])
    feature = {'kind': 'circle', 'centre': [40., 50.], 'radius': 18.}
    # Fresh physical angles do not coincide with the implementation's tessellation.
    angles = np.linspace(.173, 1.213, 13)
    measured = np.c_[40.+18.*np.cos(angles), 50.+18.*np.sin(angles)]
    observation = {'type': 'curve', 'points_image': project(h, measured).tolist()}
    error = observation_residual(observation, feature, {}, h)
    assert len(error) == 13
    assert np.max(error) < .05  # Only finite polyline approximation remains.


def test_radial_distortion_preserves_two_identified_points_as_two_points():
    h = np.array([[3., .2, 80.], [-.1, 2., 50.], [.001, -.0004, 1.]])
    feature = {'kind': 'point', 'points': [[15., 20.], [130., 70.]]}
    distortion = {'k1': -.08, 'centre_image': [480., 270.], 'scale_image': 960.}
    image = project(h, feature['points'])
    centre = np.asarray(distortion['centre_image'])
    radius_squared = np.sum(((image-centre)/960.)**2, axis=1)
    observed = centre + (image-centre)*(1-.08*radius_squared)[:, None]
    residual = observation_residual({'type': 'point', 'points_image': observed.tolist()},
                                    feature, {}, h, distortion)
    assert residual.shape == (4,)
    np.testing.assert_allclose(residual, 0., atol=1e-10)


def test_check_observations_cannot_pull_the_shared_fit_or_claim_fit_support():
    spec = _line_spec()
    baseline = fit_whole_rink(spec)
    checked = copy.deepcopy(spec)
    checked['views'].append({'id': 'held_out', 'pose_fixed': True,
                             'H_image_from_model_initial': np.eye(3).tolist()})
    checked['observations'].append({'id': 'held_out_wrong_line', 'view_id': 'held_out',
        'feature_id': 'separator', 'type': 'line', 'role': 'check', 'sigma_image_px': .01,
        'points_image': [[75., 10.], [75., 90.]]})
    result = fit_whole_rink(checked)
    assert result['parameter_estimates']['separator_x']['value'] == pytest.approx(
        baseline['parameter_estimates']['separator_x']['value'], abs=1e-8)
    diagnostics = result['fitting_diagnostics']
    assert diagnostics['fit_observation_count'] == 2
    assert diagnostics['check_observation_count'] == 1
    held = next(d for d in diagnostics['observation_diagnostics'] if d['id'] == 'held_out_wrong_line')
    assert held['max_image_px'] > 30.
    assert not held['independent_validation']
    primitive = result['structure']['primitives'][0]
    assert primitive['source_view_ids'] == ['view_a', 'view_b']


def test_two_views_share_one_geometry_parameter_and_both_influence_it():
    exact = fit_whole_rink(_line_spec())
    assert exact['parameter_estimates']['separator_x']['value'] == pytest.approx(37., abs=1e-7)
    assert max(d['max_image_px'] for d in exact['fitting_diagnostics']['observation_diagnostics']) < 1e-6
    # Equalize image-normal sensitivity so two conflicting views have equal
    # statistical weight; changing either changes the one common separator.
    conflicting = _line_spec(30., 50.)
    for ob, view in zip(conflicting['observations'], conflicting['views']):
        h = np.asarray(view['H_image_from_model_initial'])
        direction = h[:2, 1]
        normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        ob['sigma_image_px'] = abs(float(normal @ h[:2, 0]))
    first = fit_whole_rink(conflicting)
    assert first['parameter_estimates']['separator_x']['value'] == pytest.approx(40., abs=1e-5)
    moved = copy.deepcopy(conflicting)
    h2 = np.asarray(moved['views'][1]['H_image_from_model_initial'])
    moved['observations'][1]['points_image'] = project(h2, [[60., y] for y in [10.,35.,65.,90.]]).tolist()
    second = fit_whole_rink(moved)
    assert second['parameter_estimates']['separator_x']['value'] == pytest.approx(45., abs=1e-4)
    # Both emitted source projections must use that same changed geometry.
    primitive = second['structure']['primitives'][0]
    assert np.asarray(primitive['points_model'])[:, 0] == pytest.approx([45.,45.], abs=1e-4)


def test_a_prior_does_not_hide_an_unobserved_physical_parameter():
    result = fit_whole_rink(_line_spec(prior_only=True))
    diagnostic = result['fitting_diagnostics']
    assert diagnostic['solver_converged']
    assert diagnostic['unknown_count'] == 2
    assert diagnostic['jacobian_rank_with_priors'] == 2
    assert diagnostic['image_only_jacobian_rank'] == 1
    assert diagnostic['image_underconstrained']
    assert diagnostic['image_only_condition'] is None
    assert result['parameter_estimates']['unseen_radius']['value'] == pytest.approx(8.)
    assert not diagnostic['independent_accuracy_validated']
    assert result['status'] == 'conditional_candidate'


def test_one_line_cannot_determine_a_free_projective_camera():
    spec = _line_spec()
    spec['parameters']['separator_x']['fixed'] = True
    spec['views'] = [spec['views'][0]]
    spec['views'][0]['pose_fixed'] = False
    spec['observations'] = [spec['observations'][0]]
    # Many points on that one line still constrain only its two line degrees
    # of freedom, even though there are more residual rows than camera unknowns.
    h = np.asarray(spec['views'][0]['H_image_from_model_initial'])
    spec['observations'][0]['points_image'] = project(
        h, np.c_[np.full(50, 37.), np.linspace(0., 100., 50)]).tolist()
    result = fit_whole_rink(spec, max_nfev=80)
    diagnostic = result['fitting_diagnostics']
    assert diagnostic['unknown_count'] == 8
    assert diagnostic['image_only_jacobian_rank'] < diagnostic['unknown_count']
    assert diagnostic['image_underconstrained']
    assert diagnostic['image_only_condition'] is None
    assert not diagnostic['independent_accuracy_validated']


def test_seed_points_on_one_line_are_rejected_before_camera_fitting():
    spec = _line_spec()
    view = spec['views'][0]
    del view['H_image_from_model_initial']
    view['initial_correspondences'] = {'points_model': [[0.,0.],[10.,0.],[20.,0.],[30.,0.]],
                                     'points_image': [[20.,40.],[40.,40.],[60.,40.],[80.,40.]]}
    with pytest.raises(ValueError, match='Invalid view initialisation'):
        fit_whole_rink(spec)


def test_a_fully_fixed_model_still_returns_actual_projection_diagnostics():
    spec = _line_spec(25., 25.)
    spec['parameters']['separator_x']['fixed'] = True
    result = fit_whole_rink(spec)
    diagnostic = result['fitting_diagnostics']
    assert diagnostic['unknown_count'] == 0
    assert diagnostic['iterations'] == 0
    assert diagnostic['image_only_jacobian_rank'] == 0
    assert not diagnostic['image_underconstrained']
    assert max(d['max_image_px'] for d in diagnostic['observation_diagnostics']) < 1e-8
    assert not diagnostic['independent_accuracy_validated']


def test_locally_exact_observations_do_not_hide_a_horizon_across_the_whole_rink():
    spec = _line_spec(25., 25.)
    spec['parameters']['separator_x']['fixed'] = True
    h = np.array([[2., 0., 30.], [0., 3., 40.], [-.01, 0., 1.]])
    spec['views'] = [{'id': 'view_a', 'pose_fixed': True, 'H_image_from_model_initial': h.tolist()}]
    spec['observations'] = [spec['observations'][0]]
    spec['observations'][0]['points_image'] = project(h, [[25., 10.], [25., 90.]]).tolist()
    result = fit_whole_rink(spec)
    assert result['fitting_diagnostics']['observation_diagnostics'][0]['max_image_px'] < 1e-8
    projection = result['view_poses'][0]['full_extent_projection']
    assert projection['horizon_crosses_full_extent']
    assert not projection['whole_projection_valid']


def _negative_branch_camera():
    # Actual three-contact-line 7265 diagnostic H, normalized to H22=1.
    # The displayed source pixels lie on its negative branch. Inverting these
    # pixels supplies a numeric fixture, not independently measured landmarks.
    h=np.array([[-20.760057336266602,7.8216053375038745,1122.6447231790742],
                [-5.928313155622334,-.9355148878030363,156.3632788375159],
                [-.031110531881666903,-.005305983307842121,1.]])
    observed=np.array([[565.,319.5],[630.,319.],[695.,321.]])
    world=project(np.linalg.inv(h),observed)
    return h,observed,world


def test_actual_normalized_camera_can_have_a_valid_negative_observation_branch():
    h,observed,world=_negative_branch_camera()
    assert np.all((np.c_[world,np.ones(len(world))]@h.T)[:,2]<0)
    assert infer_visible_branch_sign(h,observed)==-1
    assert infer_visible_branch_sign(-h,observed)==1
    feature={'kind':'point','points':world.tolist()}
    observation={'type':'point','points_image':observed.tolist()}
    for matrix in (h,-h):
        residual=observation_residual(observation,feature,{},matrix)
        np.testing.assert_allclose(residual,0.,atol=1e-9)
    rejected=observation_residual(observation,feature,{},h,visible_branch_sign=1)
    assert rejected.shape==(6,)
    assert np.all(rejected==INVALID_PROJECTION_RESIDUAL)


def test_view_branch_uses_fit_pixels_and_checks_cannot_flip_it():
    h,observed,world=_negative_branch_camera()
    spec={'parameters':{},'features':[{'id':'points','kind':'point','points':world.tolist()}],
          'views':[{'id':'real_camera','pose_fixed':True,'H_image_from_model_initial':h.tolist()}],
          'observations':[{'id':'fit','view_id':'real_camera','feature_id':'points','type':'point',
                           'role':'fit','sigma_image_px':2.,'points_image':observed.tolist()}],
          'structure_bounds':[-110.,-50.,110.,50.]}
    # Many inconsistent check pixels on the opposite image branch must not
    # choose the branch used for the single real fit observation.
    for index in range(12):
        spec['observations'].append({'id':f'check_{index}','view_id':'real_camera','feature_id':'points',
            'type':'point','role':'check','sigma_image_px':1.,'points_image':[[0.,0.],[1.,0.],[2.,0.]]})
    result=fit_whole_rink(spec)
    assert result['view_poses'][0]['visible_branch_sign']==-1
    assert result['view_poses'][0]['visible_branch_source']=='fit_observation_inverse_h'
    diagnostics=result['fitting_diagnostics']
    assert diagnostics['fit_projection_domain_valid']
    assert diagnostics['observation_diagnostics'][0]['max_image_px']<1e-9
    assert all(d['projection_domain']['penalized_residual_count']==6
               for d in diagnostics['observation_diagnostics'][1:])


@pytest.mark.parametrize('feature,kind',[
    ({'kind':'circle','centre':[480.,270.],'radius':1500.},'curve'),
    ({'kind':'line','points':[[1980.,240.],[1980.,300.]]},'line')])
def test_wide_radial_polynomial_fold_cannot_create_an_observed_ice_curve(feature,kind):
    # k=-.098 is the observed end-reseed failure regime: radius 1500 maps
    # back close to the image centre although its radial derivative is negative.
    distortion={'k1':-.098,'centre_image':[480.,270.],'scale_image':480.}
    fake=distorted_project(np.eye(3),[[1980.,270.]],distortion)
    assert 480.<fake[0,0]<600.
    observation={'type':kind,'points_image':fake.tolist()}
    residual,domain=observation_residual(observation,feature,{},np.eye(3),distortion,return_domain=True)
    assert residual.shape==(1,)
    assert residual[0]==INVALID_PROJECTION_RESIDUAL
    assert domain['valid_adjacent_segment_count']==0
    assert domain['penalized_residual_count']==1


def test_a_folded_identified_point_is_penalized_without_changing_vector_length():
    distortion={'k1':-.098,'centre_image':[480.,270.],'scale_image':480.}
    world=np.array([[480.,270.],[1980.,270.]])
    fake=distorted_project(np.eye(3),world,distortion)
    residual=observation_residual({'type':'point','points_image':fake.tolist()},
        {'kind':'point','points':world.tolist()},{},np.eye(3),distortion)
    np.testing.assert_allclose(residual[:2],0.,atol=1e-12)
    np.testing.assert_array_equal(residual[2:],np.full(2,INVALID_PROJECTION_RESIDUAL))
    assert residual.shape==(4,)


def test_invalid_radial_gap_is_not_bridged_between_surviving_curve_vertices():
    distortion={'k1':-1.,'centre_image':[0.,0.],'scale_image':1.}
    feature={'kind':'polyline','points':[[-.2,.2],[-.1,.2],[1.,0.],[.1,.2],[.2,.2]]}
    residual,domain=observation_residual({'type':'curve','points_image':[[0.,.19]]},
        feature,{},np.eye(3),distortion,return_domain=True)
    # Removing the invalid middle vertex and joining the rest would invent a
    # horizontal chord through the observation and incorrectly return zero.
    assert residual[0]==pytest.approx(.095,abs=1e-12)
    assert domain['valid_model_vertex_count']==4
    assert domain['valid_adjacent_segment_count']==2


def test_radial_inverse_uses_monotonic_root_and_rejects_unreachable_raw_pixels():
    distortion={'k1':-1.,'centre_image':[0.,0.],'scale_image':1.}
    valid_world=np.array([[.2,.1],[-.3,.05]])
    raw=distorted_project(np.eye(3),valid_world,distortion)
    restored,valid=undistort_observed_points(raw,distortion)
    assert valid.all()
    np.testing.assert_allclose(restored,valid_world,atol=1e-11)
    restored,valid=undistort_observed_points([[.5,0.]],distortion)
    assert not valid[0]  # Maximum monotonic distorted radius is about .3849.
    assert np.isnan(restored).all()


def test_projection_domain_does_not_join_across_a_camera_horizon():
    h=np.array([[1.,0.,0.],[0.,1.,0.],[.1,0.,1.]])
    # The third point belongs to the other branch, with valid vertices on both
    # sides of it in the supplied polyline sequence.
    world=np.array([[0.,1.],[2.,1.],[-20.,0.],[3.,1.],[4.,1.]])
    projected,valid=valid_projected_points(h,world,visible_branch_sign=1)
    assert valid.tolist()==[True,True,False,True,True]
    assert np.isnan(projected[2]).all()
