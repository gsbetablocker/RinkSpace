"""Actual ice/contact assembly on tiny fake states; no assets or fitting."""
import io
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.program_tracking.body_tracking import skate_integration as skate
from calibration.program_tracking.body_tracking.ice_soft import IceSoftPolicy


# Toy objective settings exercise arithmetic, not production pass criteria.
POLICY = IceSoftPolicy(penetration_deadzone_m=.004, penetration_scale_m=.03, penetration_weight=.5,
    support_hover_deadzone_m=.006, support_scale_m=.04, support_weight=.75)


class PointBinding:
    ratio = 1.7
    collar_constraints_enabled = False
    sample_ids = np.array([0])
    neutral_native = neutral_frames = neutral_samples = None

    def __init__(self):
        self.x = np.zeros(17)
        self.surface = SimpleNamespace(sample_vertices=lambda *args, **kwargs: np.zeros((1, 1, 3)))

    def packet(self):
        return dict(template=dict(contact_arcs=[], contact_point_groups=[dict(id='runner',
            points=np.zeros((1, 3)), semantic='runner', support_eligible=True)],
            support_feature_ids=dict(glide=['runner'], pick=['runner'])))

    def compatibility_rows(self, *args):return np.zeros(32)


class HeightState:
    active = np.array([2])

    def basis(self, times):return np.ones((len(times), 1))

    def increments(self, x, times):
        delta = np.zeros((len(times), 69))
        delta[:, 2] = x[0]
        return delta


class Reference:
    def at(self, times):return {}


def make_joint(monkeypatch, *, mode='glide', heights=(-.04, .05), optimize_shared=True):
    """Root-Z moves both feet; shared[0] moves left/right with rates 2/-3."""
    binding = PointBinding()
    reference = Reference()
    reference.binding = binding
    reference.p = dict(pts=np.array([256]))

    def fk_state(ref, packet, delta):
        n = len(delta)
        origins = np.zeros((n, 2, 3))
        origins[:, :, 2] = (np.asarray(heights)/ref.binding.ratio + delta[:, 2, None]
            + np.array([2., -3.])*ref.binding.x[0])
        return dict(P=np.zeros((n, 3)), Q=np.tile(np.eye(3), (n, 1, 1)),
            frames=np.tile(np.eye(3), (n, 22, 1, 1)), R=np.tile(np.eye(3), (n, 2, 1, 1)),
            O=origins, native_world=np.zeros((n, 22, 3)))

    monkeypatch.setattr(skate, 'fk_state', fk_state)
    world = dict(P=np.zeros((5, 3)), Q=np.tile(np.eye(3), (5, 1, 1)),
        native=np.zeros((5, 1, 3)), frames=np.tile(np.eye(3), (5, 22, 1, 1)))
    base = dict(residual=np.zeros(32), jac=csr_matrix((32, 1)), parts={}, world=world,
        compatibility_residual=np.zeros(32), compatibility_jacobian=csr_matrix((32, 1)))
    joint = skate.JointContactProblem.__new__(skate.JointContactProblem)
    joint.binding = binding
    joint.reference = reference
    joint.body = SimpleNamespace(n=1, size=1, frames=np.array([0]), state=HeightState(),
        evaluate=lambda x, jacobian=False: base,
        execution=SimpleNamespace(map_columns=lambda function, columns: [function(c) for c in columns]))
    joint.source_motion = SimpleNamespace(evaluate=lambda x, jacobian=False:
        dict(residual=np.empty(0), jac=csr_matrix((0, 1)), parts={}, metrics={}))
    joint.size = 18
    joint.compatibility_width = 32
    joint.cavity_labels = ['left_foot_cavity', 'right_foot_cavity']
    joint.cavity_rows_per_frame = 2
    joint.ice_policy = POLICY
    joint.optimize_shared = optimize_shared
    joint.origin = 0.
    joint.evidence = [dict(id='source_support_toy', pts=256, feet=[1], mode=mode, source_support_weight_s=.08)]
    joint.start = time.perf_counter()
    joint.outer = 0
    joint.log = io.StringIO()
    # Isolate ice shape-column assembly from unrelated pixel/footwear replay.
    joint.shared_replay = lambda *args, **kwargs: dict(residual=np.zeros(32), compatibility_residual=np.zeros(32))
    joint.set_physics_times([.02])
    return joint


def ice_slice(result):
    # Fake body data (32), neutral compatibility (32), shared prior (17).
    return slice(81, 81+len(result['ice']['indices']))


@pytest.mark.parametrize('optimize_shared', [False, True])
def test_ice_stays_in_finite_data_while_al_and_feasibility_have_only_cavity_rows(monkeypatch, optimize_shared):
    joint = make_joint(monkeypatch, optimize_shared=optimize_shared)
    result = joint.evaluate(np.zeros(18), True)
    assert len(joint.al.multiplier) == result['gate']['tested_rows'] == 4
    assert joint.row_labels == joint.foot_row_labels
    assert all('cavity' in name for _, name in joint.row_labels)
    assert not set(result['raw_ice']['labels']) & set(joint.row_labels)
    assert result['raw_ice']['margin_m'].min() < -.03
    assert result['gate']['feasible'] and not result['gate']['ice_is_delivery_gate']
    np.testing.assert_array_equal(result['contact']['g'], np.zeros(4))
    np.testing.assert_array_equal(result['residual'][-4:], np.zeros(4))
    assert result['parts']['contact_AL'] == 0.
    assert result['parts']['ice_penetration'] > 0. and result['parts']['ice_support_hover'] > 0.
    np.testing.assert_allclose(result['residual'][ice_slice(result)], result['ice']['residual'])
    np.testing.assert_array_equal(result['raw_ice']['g'], result['raw_ice']['margin_m']/skate.NORMALIZATION)


@pytest.mark.parametrize('column', [0, 17])
def test_soft_body_and_shared_columns_match_independent_scale_sign_and_finite_difference(monkeypatch, column):
    joint = make_joint(monkeypatch)
    x = np.zeros(18)
    result = joint.evaluate(x, True)
    chosen = ice_slice(result)
    penetration_factor = np.sqrt(.02*POLICY.penetration_weight)/POLICY.penetration_scale_m
    hover_factor = np.sqrt(.08*POLICY.support_weight)/POLICY.support_scale_m
    rates = [2., -3.] if column == 0 else [1., 1.]
    expected = joint.binding.ratio*np.array([-rates[0]*penetration_factor, 0., rates[1]*hover_factor])
    analytic = result['jac'][chosen, column].toarray().ravel()
    np.testing.assert_allclose(analytic, expected, rtol=1e-9, atol=1e-11)
    step = 1e-7
    plus = x.copy();plus[column] += step
    minus = x.copy();minus[column] -= step
    numeric = (joint.evaluate(plus)['residual'][chosen]-joint.evaluate(minus)['residual'][chosen])/(2*step)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-8, atol=1e-10)
    if column == 17:
        assert analytic[0] < 0 < analytic[2]


@pytest.mark.parametrize('mode', ['unknown', 'airborne'])
def test_unknown_and_airborne_keep_universal_penetration_without_support_or_hover(monkeypatch, mode):
    joint = make_joint(monkeypatch, mode=mode)
    result = joint.evaluate(np.zeros(18), True)
    assert result['raw_ice']['support_groups'] == []
    assert np.isnan(result['raw_ice']['support_height']).all()
    assert all(name.endswith('_nonpenetration') for _, name in result['raw_ice']['labels'])
    assert result['ice']['diagnostics']['kinds'] == ['ice_penetration', 'ice_penetration']
    assert result['parts']['ice_support_hover'] == 0.
    assert result['parts']['ice_penetration'] > 0.
    assert len(result['ice']['indices']) == 2
    assert result['gate']['feasible'] and len(joint.al.multiplier) == 4
