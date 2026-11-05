"""Focused mathematical checks; no RinkSpace fit or network is executed."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.centroidal_physics_diagnostics import (
    source_seconds, flight_projection, discrete_segment_rates,
    centroidal_momentum, phase_safe_flight_diagnostics,
    convex_uniform_volume_properties, same_phase_stencil_valid,
    flight_position_operator, angular_impulse_operator, ground_force_torque_diagnostics,
)


class CentroidalChecks(unittest.TestCase):
    def test_nonuniform_known_gravity_and_affine_gauge(self):
        t = np.array([0., .017, .043, .062, .08])
        g = np.array([0., 0., -9.81])
        C = np.array([3., -7., 1.2]) + t[:, None] * [2., 1., 3.] + .5*t[:, None]**2*g
        out = flight_projection(t + 81.36, C, g)
        np.testing.assert_allclose(out['position_residual_m'], 0., atol=3e-14)
        np.testing.assert_allclose(out['acceleration_world_mps2'], np.broadcast_to(g, (3, 3)), atol=1e-10)
        modified = C + [5., 6., 8.] + t[:, None] * [2., -1., 4.]
        other = flight_projection(t, modified, g)
        np.testing.assert_allclose(other['position_residual_m'], 0., atol=3e-14)
        self.assertFalse(out['acceptance_gate'])

    def test_discrete_rates_use_nonuniform_time_and_world_so3(self):
        t = np.array([0., .013, .039, .063, .091])
        X = (np.array([2., 3., 1.]) + t[:, None]*[1., -2., 3.] + t[:, None]**2*[2., 1., 4.])[:, None]
        omega = np.array([.5, .7, 2.])
        base = Rotation.from_euler('xyz', [20., 30., 40.], degrees=True).as_matrix()
        R = (Rotation.from_rotvec(t[:, None]*omega).as_matrix() @ base)[:, None]
        out = discrete_segment_rates(t, X, R)
        expected = (np.array([1., -2., 3.]) + 2*t[1:-1, None]*[2., 1., 4.])[:, None]
        np.testing.assert_allclose(out['segment_velocity_world_mps'], expected, atol=1e-12)
        np.testing.assert_allclose(out['segment_omega_world_radps'], np.broadcast_to(omega, (3, 1, 3)), atol=1e-12)
        np.testing.assert_array_equal(source_seconds(np.array([10000, 10013, 10039]), (1, 1000)), [0., .013, .039])

    def test_orbital_and_intrinsic_momentum_and_world_frame_invariance(self):
        # Two extended bodies orbit at radius 2 and rotate about their own centres.
        # Per-total-mass orbital H_z = r^2 * omega = 12; intrinsic H_z = .4*3=1.2.
        X = np.array([[[2., 0., 0.], [-2., 0., 0.]]])
        V = np.array([[[0., 6., 0.], [0., -6., 0.]]])
        R = np.broadcast_to(np.eye(3), (1, 2, 3, 3)).copy()
        W = np.broadcast_to([0., 0., 3.], (1, 2, 3)).copy()
        I = np.broadcast_to(np.diag([.2, .3, .4]), (2, 3, 3)).copy()
        args = dict(centres_world_m=X, velocity_world_mps=V,
                    rotation_world_from_segment=R, omega_world_radps=W,
                    segment_relative_masses=[1., 1.], unit_mass_inertia_segment_m2=I)
        out = centroidal_momentum(**args)
        np.testing.assert_allclose(out['specific_orbital_momentum_world_m2ps'], [[0., 0., 12.]])
        np.testing.assert_allclose(out['specific_intrinsic_momentum_world_m2ps'], [[0., 0., 1.2]])
        np.testing.assert_allclose(out['specific_angular_momentum_world_m2ps'], [[0., 0., 13.2]])
        S = Rotation.from_euler('xyz', [25., 71., 16.], degrees=True).as_matrix()
        transformed = dict(args, centres_world_m=X @ S.T + [21., -11., 7.],
                           velocity_world_mps=V @ S.T + [4., 9., -2.],
                           rotation_world_from_segment=S @ R, omega_world_radps=W @ S.T)
        rotated = centroidal_momentum(**transformed)
        np.testing.assert_allclose(rotated['specific_angular_momentum_world_m2ps'],
                                   out['specific_angular_momentum_world_m2ps'] @ S.T, atol=3e-14)

    def test_phase_gap_is_not_silently_bridged(self):
        t = np.arange(9) * .02
        C = np.column_stack((t, t*0, 1.-4.905*t*t))
        mask = np.array([True, True, True, False, True, True, True, True, False])
        valid = np.array([False, True, False, False, False, True, True, False, False])
        reports = phase_safe_flight_diagnostics(t, C, mask, [0., 0., -9.81],
                  specific_H_world_m2ps=np.tile([0., 0., 1.], (9, 1)), H_rate_stencil_valid=valid)
        np.testing.assert_array_equal(reports[0]['source_indices'], [0, 1, 2])
        np.testing.assert_array_equal(reports[1]['source_indices'], [4, 5, 6, 7])
        np.testing.assert_array_equal(reports[1]['specific_angular_impulse_source_pairs'], [[5, 6]])
        self.assertNotIn('specific_angular_impulse_source_pairs', reports[0])

    def test_covariance_propagation_does_not_assume_independent_d2_rows(self):
        t = np.arange(5)*.02
        C = np.column_stack((t, t*0, 1.-4.905*t*t))
        out = flight_projection(t, C, [0., 0., -9.81], temporal_covariance_m2=np.eye(5)*.001**2)
        covariance = out['acceleration_noise_covariance_m2_s4']
        np.testing.assert_allclose(np.diag(covariance), 37.5)
        self.assertAlmostEqual(covariance[0, 1]/covariance[0, 0], -2/3)
        self.assertAlmostEqual(out['gravity_departure_from_endpoint_chord_m'], .007848)

    def test_convex_box_inertia_about_its_own_com(self):
        X = np.array([[x, y, z] for x in (0., 2.) for y in (0., 4.) for z in (0., 6.)])
        out = convex_uniform_volume_properties(X + [70., -10., 20.])
        self.assertAlmostEqual(out['volume_m3'], 48.)
        np.testing.assert_allclose(out['centre_m'], [71., -8., 23.])
        np.testing.assert_allclose(out['unit_mass_inertia_centre_m2'], np.diag([(16+36)/12, (4+36)/12, (4+16)/12]), atol=1e-13)


if __name__ == '__main__':
    unittest.main()



def test_phase_stencil_rejects_unknown_and_changes_without_dropping_source_rows():
    t = np.array([0., .013, .039, .063, .091, .114, .145])
    phase = np.array(['flight', 'flight', 'flight', 'unknown', 'support', 'support', 'support'])
    confirmed = phase != 'unknown'
    centres = np.column_stack((t, t*t, t*0))[:, None]
    rotations = np.broadcast_to(np.eye(3), (len(t), 1, 3, 3))
    result = discrete_segment_rates(t, centres, rotations, phase_ids=phase, confirmed_phase=confirmed)
    np.testing.assert_array_equal(result['source_indices'], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(result['derivative_stencil_valid'], [True, False, False, False, True])
    np.testing.assert_array_equal(result['interval_stencil_valid'], [True, True, False, False, True, True])
    # An explicitly changed phase also cuts otherwise confirmed adjacent runs.
    assert not same_phase_stencil_valid(['left', 'right'], [[0, 1]], confirmed=np.ones(2, bool))[0]
    assert not discrete_segment_rates(t, centres, rotations)['derivative_stencil_valid'].any()


def test_normalized_fraction_path_uses_declared_weights_unchanged():
    fractions = np.array([.3, .7])
    args = dict(centres_world_m=np.array([[[1., 0., 0.], [-2., 0., 0.]]]),
        velocity_world_mps=np.array([[[0., 2., 0.], [0., -1., 0.]]]),
        rotation_world_from_segment=np.broadcast_to(np.eye(3), (1, 2, 3, 3)),
        omega_world_radps=np.zeros((1, 2, 3)),
        unit_mass_inertia_segment_m2=np.broadcast_to(np.eye(3), (2, 3, 3)))
    result = centroidal_momentum(**args, segment_mass_fractions=fractions)
    np.testing.assert_array_equal(result['normalized_mass_fractions'], fractions)
    relative = centroidal_momentum(**args, segment_relative_masses=fractions*17.)
    np.testing.assert_allclose(result['specific_angular_momentum_world_m2ps'], relative['specific_angular_momentum_world_m2ps'])
    import pytest
    with pytest.raises(ValueError, match='already sum to one'):
        centroidal_momentum(**args, segment_mass_fractions=[3., 7.])
    with pytest.raises(ValueError, match='either'):
        centroidal_momentum(**args, segment_mass_fractions=fractions, segment_relative_masses=[1., 2.])


def test_flight_position_operator_matches_projection_with_correlated_covariance():
    t = np.array([0., .017, .043, .062, .08])
    gravity = np.array([0., 0., -9.81])
    C = [3., -7., 1.2] + t[:, None]*[2., 1., 3.] + .5*t[:, None]**2*gravity
    C[2] += [.001, -.002, .003]
    for covariance in [None, .001**2*(np.eye(5)+.3*np.ones((5, 5)))]:
        result = flight_projection(t, C, gravity, temporal_covariance_m2=covariance)
        op = flight_position_operator(t, gravity, temporal_covariance_m2=covariance)
        expected = result['position_residual_m'] if covariance is None else result['whitened_residual']
        np.testing.assert_allclose(op['operator'] @ (C-op['gravity_position_m']), expected, atol=1e-11)
        np.testing.assert_allclose(op['operator'] @ np.c_[np.ones(len(t)), t], 0., atol=1e-10)


def test_delta_H_operator_and_ground_torque_respect_every_nested_stencil():
    t = np.array([0., .013, .039, .063, .091, .114, .145, .168, .193])
    support = np.array([True, True, True, False, True, True, True, True, True])
    phase = np.where(support, 'support', 'unknown')
    hvalid = np.array([False, True, False, False, False, True, True, True, False])
    H = np.array([1., 2., 3.]) + t[:, None]*[.3, -.2, 1.7]
    impulse = angular_impulse_operator(t, phase, confirmed_phase=support, derivative_stencil_valid=hvalid)
    np.testing.assert_array_equal(impulse['source_pairs'], [[5, 6], [6, 7]])
    np.testing.assert_allclose(impulse['operator'] @ H, np.diff(t[5:8])[:, None]*[.3, -.2, 1.7])
    acceleration = np.array([.3, -.4, .5])
    gravity = np.array([0., 0., -9.81])
    C = [3., -2., 4.] + t[:, None]*[1., 2., 3.] + .5*t[:, None]**2*acceleration
    result = ground_force_torque_diagnostics(t, C, H, support, gravity,
        H_rate_stencil_valid=hvalid, phase_ids=phase)
    np.testing.assert_array_equal(result['force_source_indices'], [1, 5, 6, 7])
    np.testing.assert_array_equal(result['torque_source_indices'], [6])
    np.testing.assert_allclose(result['specific_total_external_force_world_mps2'],
                               np.broadcast_to(acceleration-gravity, (4, 3)), atol=1e-10)
    np.testing.assert_allclose(result['specific_external_torque_com_world_m2ps2'], [[.3, -.2, 1.7]], atol=1e-13)
    assert result['support_feasibility_enforced'] is False
