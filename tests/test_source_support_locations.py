"""Moving real-steel geometry, finite source ranges and explicit fit admission."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.fullbody_fit import skew
from calibration.program_tracking.body_tracking.skate_template import SkateParameters, build_skate
from calibration.program_tracking.body_tracking.source_support_locations import (
    SourceSupportLocationFactors, _polygon_terms, _runner_witness,
)


def test_exact_runner_witness_includes_longitudinal_motion_derivative():
    template = build_skate(SkateParameters())
    binding = SimpleNamespace(packet=lambda: dict(template=template))
    R = Rotation.from_euler('xyz', [[12., 2., 31.], [-9., 7., -13.], [17., 20., 10.]], degrees=True).as_matrix()
    O = np.array([[.1, -.2, .03], [-.2, .3, -.01], [0., 0., .1]])
    witness = _runner_witness(binding, R, O)
    direction = np.array([[.2, -.3, .1], [-.1, .2, .4], [.3, .1, -.2]])
    h = 1e-6
    a = _runner_witness(binding, R @ Rotation.from_rotvec(h*direction).as_matrix(), O)
    b = _runner_witness(binding, R @ Rotation.from_rotvec(-h*direction).as_matrix(), O)
    np.testing.assert_array_equal(a['feature_index'], witness['feature_index'])
    np.testing.assert_array_equal(b['feature_index'], witness['feature_index'])
    fixed = -R @ skew(witness['local'])
    moving = R @ witness['derivative_normal'] @ skew(R[:, 2, :])
    analytic = np.einsum('nij,nj->ni', fixed+moving, direction)
    np.testing.assert_allclose((a['world']-b['world'])/(2*h), analytic, rtol=2e-7, atol=2e-9)
    assert np.linalg.norm(moving) > .1
    assert witness['interior'].any() and not witness['interior'].all()
    # Every prediction remains on its actual selected finite steel arc.
    arcs = [a for a in template['contact_arcs'] if a['semantic'] == 'runner']
    for i, index in enumerate(witness['feature_index']):
        arc = arcs[index]; theta = witness['theta'][i]
        assert arc['theta_start_rad'] <= theta <= arc['theta_end_rad']
        expected = np.asarray(arc['center'])+arc['radius_m']*(np.cos(theta)*np.asarray(arc['axis_cos'])+np.sin(theta)*np.asarray(arc['axis_sin']))
        np.testing.assert_allclose(witness['local'][i], expected, atol=1e-14)


def test_finite_source_range_has_no_midpoint_attraction_and_correct_derivatives():
    polygon = np.array([[-3., -1.], [4., -1.], [4., 2.], [-3., 2.]])
    for point in [[-2., -.5], [3., 1.5]]:
        residual, derivative, _, kind = _polygon_terms(np.array(point), polygon)
        np.testing.assert_array_equal(residual, np.zeros(2))
        np.testing.assert_array_equal(derivative, np.zeros((2, 2)))
        assert kind == 'inside'
    direction = np.array([.17, -.31]); h = 1e-6
    for point in [np.array([5., .4]), np.array([5., 3.])]:
        residual, derivative, _, _ = _polygon_terms(point, polygon)
        numeric = (_polygon_terms(point+h*direction, polygon)[0]-_polygon_terms(point-h*direction, polygon)[0])/(2*h)
        np.testing.assert_allclose(derivative @ direction, numeric, rtol=1e-8, atol=1e-9)
        assert np.linalg.norm(residual) > 0


def source_case():
    uv = np.array([[10., 20.], [14., 20.]])
    jac = np.array([[.1, .02], [-.01, .03]])
    xy = uv @ jac.T
    angle = 2*np.pi*np.arange(32)/32
    boundary = xy[:, None]+np.c_[np.cos(angle), np.sin(angle)][None] @ jac.T
    mapping = dict(points_uv=uv.tolist(), xyz_conditional_m=np.c_[xy, np.zeros(2)].tolist(),
        xy_jacobian_conditional_m_per_px=np.repeat(jac[None], 2, axis=0).tolist(),
        pixel_uncertainty_radius_px=[1., 1.], valid=[True, True],
        sampled_pixel_disk_boundary_xy_conditional_m=boundary.tolist(),
        sampled_pixel_disk_boundary_valid=np.ones((2, 32), bool).tolist(),
        sampled_xy_aabb_conditional_m=np.stack((boundary.min(axis=1), boundary.max(axis=1)), axis=1).tolist(),
        frames=[0, 0], camera_row_time_seconds=[.005, .005])
    feature = dict(source_status='ambiguous', mapping_role='source_ice_interface_candidate',
        source_annotation=dict(points_uv=uv.tolist(), status='ambiguous'),
        source_frame=dict(role='development_review', optimization_enabled=False), mapping=mapping)
    document = dict(schema='rinkspace.source-skate-plane.v1', time_base=[1, 12800],
        no_body_pose_used=True, diagnostic_footprint_only=False, metres_per_scene_unit=54.,
        camera_binding=dict(metres_per_scene_unit=54., solver_metres_per_scene_unit=60., length_export_ratio=.9),
        frames=[dict(pts=256, foot=1, features=dict(ice_interface=feature))])
    ref = SimpleNamespace(p=dict(pts=np.array([256], np.int64)), times=np.array([0.]), at=lambda t: dict(times=t))
    state = SimpleNamespace(shoe_chain=lambda *args: None)
    problem = SimpleNamespace(reference=ref, binding=SimpleNamespace(ratio=.9, x=np.zeros(17)),
        camera=SimpleNamespace(scale=60., p=dict(motion_times=np.array([0.]))),
        body=SimpleNamespace(size=100, state=state), size=117, optimize_shared=False)
    evidence = [dict(pts=256, foot=1, role='fit', optimization_enabled=True,
        phase='continuous right support', support_identity_evidence='surrounding source frames retain the support-foot identity',
        context_source_pts=[0, 256, 512])]
    return problem, document, evidence


def test_ambiguous_range_admits_only_current_source_support_and_checks_export_ratio():
    problem, document, evidence = source_case()
    original = copy.deepcopy(document)
    factor = SourceSupportLocationFactors(problem, document, support_evidence=evidence)
    assert len(factor.rows) == 1
    assert factor.rows[0]['feature']['source_status'] == 'ambiguous'
    assert factor.rows[0]['feature']['source_frame']['optimization_enabled'] is False
    assert factor.rows[0]['time'] == .005
    assert factor.weight+factor.curve_frame_weight == factor.frame_weight_budget
    np.testing.assert_allclose(factor.rows[0]['objective_weight'], .005)
    assert document == original
    empty = SourceSupportLocationFactors(problem, document, support_evidence=[])
    out = empty.evaluate(np.zeros(problem.body.size), jacobian=True)
    assert out['jac'].shape == (0, problem.size) and not len(out['residual'])
    with pytest.raises(ValueError, match='surrounding native PTS'):
        bad = copy.deepcopy(evidence); bad[0]['context_source_pts'] = [256]
        SourceSupportLocationFactors(problem, document, support_evidence=bad)
    problem.binding.ratio = 1.
    with pytest.raises(ValueError, match='length units differ'):
        SourceSupportLocationFactors(problem, document, support_evidence=evidence)
