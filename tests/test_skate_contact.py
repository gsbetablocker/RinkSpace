"""Focused reference-kernel risks; no reconstruction or native solver."""
import numpy as np
from calibration.program_tracking.body_tracking.skate_contact import Plane, SupportBand, feature_minimum, frame_residuals, rigid_transform_features, trapezoid_time_weights, contact_jacobian, constraint_jacobian, template_world_features

def rotation(v):
    v = np.asarray(v, float)
    a = np.linalg.norm(v)
    if a == 0:
        return np.eye(3)
    x, y, z = v / a
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)

def arc(fid='runner', y=0.0):
    return dict(kind='arc', id=fid, semantic='runner', center=np.array([0.0, y, 2.095]), axis_cos=np.array([0.0, 0.0, -1.0]), axis_sin=np.array([1.0, 0.0, 0.0]), radius_m=2.1, theta_start_rad=-0.08, theta_end_rad=0.08)

def test_arc_interior_below_ice_despite_positive_endpoints():
    a = arc()
    minimum = feature_minimum(a)
    endpoint_z = a['center'][2] - a['radius_m'] * np.cos(0.08)
    assert endpoint_z > 0 and abs(minimum['height_m'] + 0.005) < 1e-12

def test_bezier_interior_and_control_derivative():
    epsilon = 1e-07
    b = dict(kind='bezier', id='quadratic', control_points=[[-1.0, 0.0, 0.01], [0.0, 0.0, -0.03], [1.0, 0.0, 0.01]])
    witness = feature_minimum(b)
    assert abs(witness['parameter'] - 0.5) < 1e-12 and abs(witness['height_m'] + 0.01) < 1e-12
    epsilon = 1e-07
    for k in range(3):
        values = []
        for sign in (1.0, -1.0):
            copy = dict(b, control_points=np.array(b['control_points']))
            copy['control_points'][k, 2] += sign * epsilon
            values.append(feature_minimum(copy)['height_m'])
        np.testing.assert_allclose((values[0] - values[1]) / (2 * epsilon), witness['height_partials']['control_points'][k, 2], atol=1e-08)

def test_unknown_is_not_forced_to_ice():
    a = arc()
    elevated = rigid_transform_features([a], np.eye(3), [0.0, 0.0, 0.05])
    unknown = frame_residuals({'left': elevated}, nonpenetration_sigma_m=0.005)
    support = SupportBand((('left', 'runner'),), -0.001, 0.001, 0.005)
    standing = frame_residuals({'left': elevated}, nonpenetration_sigma_m=0.005, supports=[support])
    assert len(unknown['residuals']) == 1 and unknown['residuals'][0] == 0
    assert len(standing['residuals']) == 2 and standing['residuals'][1] > 0

def test_edge_support_preserves_roll_and_airborne_other_edge():
    edges = [arc('edge_a', -0.002), arc('edge_b', 0.002)]
    tilted = rigid_transform_features(edges, rotation([0.45, 0, 0]), [0.0, 0.0, 0.0])
    dz = -min((feature_minimum(f)['height_m'] for f in tilted))
    tilted = rigid_transform_features(tilted, np.eye(3), [0.0, 0.0, dz])
    band = SupportBand((('left', 'edge_a'), ('left', 'edge_b')), -1e-10, 1e-10, 0.005)
    result = frame_residuals({'left': tilted}, nonpenetration_sigma_m=0.005, supports=[band], allowed_penetration_m=1e-10)
    assert np.max(abs(result['residuals'])) < 1e-08
    assert max((feature_minimum(f)['height_m'] for f in tilted)) > 0.001

def test_solid_vertex_guard_and_separate_pick_tolerance():
    a = arc()
    elevated = rigid_transform_features([a], np.eye(3), [0.0, 0.0, 0.05])
    guard = dict(kind='points', id='lower_solid', semantic='solid_guard', points=[[0.0, 0.0, -0.003], [0.0, 0.1, 0.1], [0.1, 0.0, 0.1]])
    pick = dict(kind='points', id='pick', semantic='toe_pick', points=[[0.0, 0.0, -0.002]])
    guarded = frame_residuals({'left': elevated + [guard, pick]}, nonpenetration_sigma_m=0.005, allowed_penetration_m={'runner': 0.001, 'toe_pick': 0.004, 'solid_guard': 0.0})
    assert guarded['rows'][0]['feature_id'] == 'lower_solid'
    np.testing.assert_allclose(guarded['residuals'], [0.6])

def test_no_softmin_or_vertex_count_bias():
    a = arc()
    guard = dict(kind='points', id='lower_solid', semantic='solid_guard', points=[[0.0, 0.0, -0.003], [0.0, 0.1, 0.1], [0.1, 0.0, 0.1]])
    whole = frame_residuals({'left': [a]}, nonpenetration_sigma_m=0.005)['residuals']
    pieces = [dict(a, id='s0', theta_end_rad=-0.02), dict(a, id='s1', theta_start_rad=-0.02, theta_end_rad=0.02), dict(a, id='s2', theta_start_rad=0.02)]
    split = frame_residuals({'left': pieces}, nonpenetration_sigma_m=0.005)['residuals']
    np.testing.assert_allclose(whole, split, atol=1e-12)
    duplicated = dict(guard, points=guard['points'] * 5)
    c0 = frame_residuals({'left': [guard]}, nonpenetration_sigma_m=0.005)['residuals']
    c1 = frame_residuals({'left': [duplicated]}, nonpenetration_sigma_m=0.005)['residuals']
    np.testing.assert_array_equal(c0, c1)

def test_constant_signal_cost_invariant_to_50_vs_100_hz():
    a = arc()
    costs = []
    for n in (51, 101):
        weights = trapezoid_time_weights(np.linspace(0, 1, n))
        costs.append(sum((float(frame_residuals({'left': [a]}, nonpenetration_sigma_m=0.005, time_weight_s=w)['residuals'] @ frame_residuals({'left': [a]}, nonpenetration_sigma_m=0.005, time_weight_s=w)['residuals'] / 2) for w in weights)))
    np.testing.assert_allclose(costs[0], costs[1], atol=1e-12)

def test_translation_rotation_chain_derivative():
    a = arc()
    epsilon = 1e-07
    support = SupportBand((('left', 'runner'),), -0.001, 0.001, 0.005)
    R0 = rotation([0.08, 0.03, 0.02])
    t0 = np.array([0.02, 0.01, -0.05])

    def evaluate(q):
        geometry = rigid_transform_features([a], rotation(q[3:]) @ R0, t0 + q[:3])
        return frame_residuals({'left': geometry}, nonpenetration_sigma_m=0.005, supports=[support])
    result = evaluate(np.zeros(6))

    def point_jac(row):
        radial = row['point'] - t0
        return np.c_[np.eye(3), np.stack([np.cross(axis, radial) for axis in np.eye(3)], axis=1)]
    analytic = contact_jacobian(result, point_jac)
    numeric = np.empty_like(analytic)
    for k in range(6):
        d = np.zeros(6)
        d[k] = epsilon
        numeric[:, k] = (evaluate(d)['residuals'] - evaluate(-d)['residuals']) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numeric, rtol=3e-06, atol=3e-06)
    raw = constraint_jacobian(result, point_jac)
    np.testing.assert_allclose(raw[:, :3], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], atol=1e-12)
    assert all((row['violation_m'] >= 0 for row in result['rows']))

def test_real_template_adapter_guard_tolerances_and_support_whitelist():
    from calibration.program_tracking.body_tracking.skate_template import build_skate
    template = build_skate()
    allowed = {f['semantic']: 0.003 for f in template['contact_arcs']}
    allowed.update(runner=0.003, toe_pick=0.004, solid_guard=0.0)
    features = template_world_features(template, np.eye(3), [0.0, 0.0, -0.002])
    geometry_result = frame_residuals({'left': features}, nonpenetration_sigma_m=0.005, allowed_penetration_m=allowed)
    assert geometry_result['residuals'][0] == 0.0
    glide = template['support_feature_ids']['glide']
    band = SupportBand(tuple((('left', fid) for fid in glide)), -0.003, 0.001, 0.005)
    geometry_result = frame_residuals({'left': features}, nonpenetration_sigma_m=0.005, allowed_penetration_m=allowed, supports=[band])
    assert np.max(abs(geometry_result['residuals'])) < 1e-09
    guard_id = next((f['id'] for f in features if f.get('support_eligible') is False))
    try:
        frame_residuals({'left': features}, nonpenetration_sigma_m=0.005, supports=[SupportBand((('left', guard_id),), 0.0, 0.0, 0.005)])
    except ValueError:
        pass
    else:
        raise AssertionError('Solid guard was incorrectly admitted as contact support')

def test_canonical_blade_frame_fix_preserves_world_geometry():
    pitch = 0.23
    z = np.linspace(-0.13, 0.13, 17)
    y = 0.012 * (z / 0.13) ** 2 + z * np.tan(pitch)
    B = np.column_stack([np.zeros_like(z), y, z])
    centre = B.mean(axis=0)
    oldR = rotation([pitch, 0, 0])
    newR = rotation([-pitch, 0, 0])
    oldL = (B - centre) @ oldR
    newL = (B - centre) @ newR
    WB = rotation([0.1, 0.2, 0.3])
    origin = np.array([1.0, 2.0, 3.0])
    oldW = oldL @ (WB @ oldR).T + origin + WB @ centre
    newW = newL @ (WB @ newR).T + origin + WB @ centre
    np.testing.assert_allclose(oldW, newW, atol=1e-14)
    assert abs(newL[-1, 1] - newL[0, 1]) < 1e-14
    assert abs(oldL[-1, 1] - oldL[0, 1]) > 0.1
