import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from calibration.program_tracking.body_tracking.runner_support_factors import runner_mode_terms, runner_tangent_limits, source_runner_samples
from calibration.program_tracking.body_tracking.skate_template import SkateParameters


def test_runner_normal_fan_allows_bank_without_forcing_flat_blade():
    rear, front, scale = runner_tangent_limits(SkateParameters())
    normal = Rotation.from_rotvec(np.c_[np.deg2rad([0, 25, -35]), np.zeros((3, 2))]).as_matrix()[:, 2]
    residual, _, _ = runner_mode_terms(normal, np.full(3, -.01), rear, front, scale, .012, np.ones(3))
    np.testing.assert_array_equal(residual, np.zeros((3, 3)))
    tilted = Rotation.from_rotvec(np.c_[np.zeros(2), np.deg2rad([40, -40]), np.zeros(2)]).as_matrix()[:, 2]
    residual, _, _ = runner_mode_terms(tilted, np.zeros(2), rear, front, scale, .012, np.ones(2))
    assert np.count_nonzero(residual[:, :2]) == 2
    assert np.all(residual[:, :2].sum(1) > 0)


def test_finite_mode_kernel_derivatives_and_toe_pick_preference():
    rear, front, scale = runner_tangent_limits(SkateParameters())
    v = np.array([[.7, .1, .5], [-.6, -.2, .7]])
    gap = np.array([.008, -.002]); weights = np.array([.04, .08])
    residual, D, _ = runner_mode_terms(v, gap, rear, front, scale, .012, weights)
    assert residual[0, 2] > 0 and residual[1, 2] == 0
    direction = np.array([[.3, -.1, .2], [-.1, .2, .3]]); dg = np.array([.002, -.003]); h = 1e-6
    a = runner_mode_terms(v+h*direction, gap+h*dg, rear, front, scale, .012, weights)[0]
    b = runner_mode_terms(v-h*direction, gap-h*dg, rear, front, scale, .012, weights)[0]
    exact = D*np.c_[direction@rear, -direction@front, dg]
    np.testing.assert_allclose((a-b)/(2*h), exact, rtol=1e-8, atol=1e-9)


def test_native_interval_gate_never_invents_contact_or_boundary_pts():
    pts = np.arange(1000, 5000, 256)
    base = dict(id='runner', foot=1, start_pts=1000, end_pts=4840, mode='runner_interior', state='supported', role='fit', optimization_enabled=True, source_evidence_pts=[1000])
    policy = dict(time_base=[1, 12800], objective_weight=20., quadrature_stride=2, intervals=[base])
    selected = source_runner_samples(pts, (1500/12800., 4100/12800.), policy)
    assert selected and all(1500 <= pts[i] <= 4100 for i, _, _, _ in selected)
    assert sum(w for _, _, w, _ in selected) < 20*(4100-1500)/12800.
    for state in ['airborne', 'unknown']:
        policy['intervals'] = [dict(base, state=state)]
        assert source_runner_samples(pts, (0, 1), policy) == []
    policy['intervals'] = [base, base]
    with pytest.raises(ValueError, match='duplicate'):
        source_runner_samples(pts, (0, 1), policy)
