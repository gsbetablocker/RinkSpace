"""Exact reuse must preserve the real Brown finite-curve kernel and FD chart."""
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.solver_runtime import install

install()
from calibration.solver_runtime.source_loss import SourceLossWeights
from calibration.program_tracking.first_shot_physical_state import GeometryDomainError
from calibration.program_tracking import static_venue_exact as exact
from calibration.program_tracking.static_venue_source import StaticCameraPolicy, StaticVenueFactor


def make_factors(*, columns=(0, 1), step=1e-5, guard=None):
    def scene(q):
        heights = (-1. + q[0], q[1], 1. + q[0], 2.)
        return {'ice_features': [dict(id=name, kind='polyline', segments_xyz=[
            [[-2., float(y), 10.], [2., float(y), 10.]]])
            for name, y in zip('ABCD', heights)]}

    rows = tuple(dict(id=name, target=dict(kind='line_segment', feature_id=name),
        pixels_uv=[[900. + 20.*i, 460. + 80.*k + i] for i in range(n)],
        sigma_px=[.7 + .2*i + .1*k for i in range(n)])
        for k, (name, n) in enumerate(zip('ABCD', (2, 3, 1, 2))))
    scales = np.concatenate([np.repeat(row['sigma_px'], 2) for row in rows])
    packet = SimpleNamespace(rows=rows, scales=scales,
        quadratic=np.zeros(len(scales), bool),
        loss_weights=SourceLossWeights(np.full(len(scales), .6)),
        ledger={'family': 'unchanged-test-source'}, identity_sha256='test-source',
        frame={'PTS': 13, 'time_base': '1/25'})
    camera = np.zeros(25)
    camera[6:8] = np.log(800.)
    camera[8:10] = [960., 540.]
    camera[11:15] = [-.01, .0001, .00001, -.00002]
    kwargs = dict(programme_count=3, geometry_columns=np.asarray(columns),
        scene_at=scene, geometry_guard=guard or (lambda q: True),
        geometry_fd_step=step, samples=8)
    policy = StaticCameraPolicy('brown13', camera)
    old = StaticVenueFactor(packet, policy, **kwargs)
    new = exact.ExactStaticVenueFactor(packet, policy, **kwargs)
    return old, new, new.initial_state(np.zeros(3))


def spy_real_kernel(monkeypatch):
    """Observe skipped projections while still executing the actual kernel."""
    calls = []
    original = exact.absolute_value_jac15

    def observed(camera, rows, jacobian=True):
        calls.append(tuple(row['world'].copy() for row in rows))
        return original(camera, rows, jacobian)

    monkeypatch.setattr(exact, 'absolute_value_jac15', observed)
    return calls


def assert_same_block(old, new, q):
    expected, actual = old.block(q, True), new.block(q, True)
    np.testing.assert_array_equal(actual.r, expected.r)
    np.testing.assert_array_equal(actual.J.toarray(), expected.J.toarray())
    assert actual.scales is expected.scales
    assert actual.quadratic is expected.quadratic
    assert actual.row_ledger is expected.row_ledger
    assert actual.loss_weights is expected.loss_weights
    assert new.last_geometry_stencils == old.last_geometry_stencils
    return actual


def test_brown13_changed_subsets_scatter_back_in_source_order(monkeypatch):
    calls = spy_real_kernel(monkeypatch)
    old, new, q = make_factors(columns=(0, 1))
    assert_same_block(old, new, q)
    # Noncontiguous A/C, then B, with unequal source row sizes.
    assert [len(rows) for rows in calls] == [2, 2, 1, 1]
    assert new.last_exact_reuse['unchanged_group_projections_reused'] == 10
    calls.clear()
    old.geometry_columns = new.geometry_columns = np.array([1, 0])
    q[0], q[1] = .03, -.02
    assert_same_block(old, new, q)
    assert [len(rows) for rows in calls] == [1, 1, 2, 2]
    # Repeating the original order must not retain a stale subset layout.
    calls.clear()
    old.geometry_columns = new.geometry_columns = np.array([0, 1])
    assert_same_block(old, new, q)
    assert [len(rows) for rows in calls] == [2, 2, 1, 1]


def test_identical_world_reuses_residual_without_sample_projection(monkeypatch):
    calls = spy_real_kernel(monkeypatch)
    old, new, q = make_factors(columns=(2,))
    block = assert_same_block(old, new, q)
    assert calls == []
    assert block.J[:, 2].nnz == 0
    assert new.last_exact_reuse['unchanged_group_projections_reused'] == 8
    assert new.last_geometry_stencils[0]['method'] == 'central_original_step'


@pytest.mark.parametrize('allowed_sign,method', [
    (1., 'forward_original_step'), (-1., 'backward_original_step')])
def test_one_sided_geometry_guard_preserves_original_step(allowed_sign, method):
    checked = []

    def guard(q):
        checked.append(q.copy())
        if allowed_sign*q[0] < 0.:
            raise GeometryDomainError('literal test boundary')

    old, new, q = make_factors(columns=(0,), step=2.5e-5, guard=guard)
    assert_same_block(old, new, q)
    assert len(checked) == 6  # Each factor: base, original minus, original plus.
    np.testing.assert_array_equal([row[0] for row in checked],
        [0., -new.fd_step, new.fd_step]*2)
    stencil = new.last_geometry_stencils[0]
    assert stencil['method'] == method and stencil['step'] == 2.5e-5
    failure = 'minus' if allowed_sign > 0 else 'plus'
    assert stencil[failure+'_geometry_domain_failure'] == 'literal test boundary'


def test_tiny_nonzero_world_change_is_projected_without_tolerance(monkeypatch):
    calls = spy_real_kernel(monkeypatch)
    old, new, q = make_factors(columns=(1,), step=1e-12)
    baseline = exact.current_material_curve(new.scene_at(q[:3]),
        new.packet.rows[1]['target'], samples=new.samples)
    sample = q[:3].copy()
    sample[1] += new.fd_step
    changed = exact.current_material_curve(new.scene_at(sample),
        new.packet.rows[1]['target'], samples=new.samples)
    assert np.allclose(baseline, changed) and baseline.tobytes() != changed.tobytes()
    block = assert_same_block(old, new, q)
    assert [len(rows) for rows in calls] == [1, 1]
    assert block.J[:, 1].nnz > 0
    assert new.last_exact_reuse['unchanged_group_projections_reused'] == 6
