"""Shadow policies alter only original positive-source progress targets."""
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.solver_runtime.physical_signed_margin_SOC import attach_original_signed_margin_rows
from calibration.solver_runtime.trial_fraction_source_domain import attach_trial_fraction_domain
from scripts.diagnostics.truth_progress_domain_probe_20261011 import build_truth_progress_probe_domain


@pytest.fixture
def parent():
    matrix = np.array([[1., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., 1.], [-1., 1.]])
    original = np.array([-.1, 2., 1., -0., 4., -.5])
    owners = np.array([-1, 0, 0, 1, 2, -1])
    previous = np.array([2., 0., 4.])
    limits = np.array([3., 0., 5.])
    coefficient = np.array([0., 2., 2., 0., 4., 0.])
    rho = .8
    targets = previous * rho
    record = dict(uv=np.array([[1., 2.]]), factor=SimpleNamespace(name='original source owner'))
    context = dict(record=record, nominal_finite_segments=np.array([0]))
    responses = dict(
        native=dict(row_start=1, row_stop=4, dynamic_group_limits=targets[:2].copy(),
                    dynamic_row_group_index=np.array([0, 0, 1]), nominal_carrier=np.array([1., 2.]),
                    native_nominal_query_chart=dict(contexts=[context], nominal_carrier=np.array([1., 2.])),
                    finite_station_contexts=[dict(record=record, group=0, source_vertex=0,
                                                 segment=0, station=.5, coefficients=np.array([.5, .5]))],
                    row_carrier_projection=csr_matrix(np.eye(2))),
        confirmed=dict(row_start=4, row_stop=6, dynamic_group_limits=targets[2:].copy(),
                       dynamic_row_group_index=np.array([0, -1]), nominal_carrier=np.array([3.])))
    domain = dict(matrix=matrix, lower=original-rho*coefficient, curvature_responses=responses,
                  report=dict(truth_feasibility_priority=dict(restoration_required=True,
                      demanded_common_original_violation_fraction=rho,
                      current_original_group_violations_px=previous.tolist(),
                      demanded_group_violations_px=targets.tolist())))
    domain = attach_trial_fraction_domain(domain, original_constraint_lower=original,
        row_group_index=owners, previous_group_violations=previous, common_fraction=rho)
    domain = attach_original_signed_margin_rows(domain, row_indices=np.array([0]),
        nominal_margins=np.array([.1]), row_ids=['physical_pivot'])
    return SimpleNamespace(domain=domain, original=original, owners=owners, previous=previous,
                           limits=limits, coefficient=coefficient, record=record)


def probe(parent, policy='preserve_current_violation', **kwargs):
    return build_truth_progress_probe_domain(parent.domain,
        current_group_violations=parent.previous, existing_group_limits=parent.limits,
        policy=policy, original_constraint_lower=parent.original, row_group_index=parent.owners,
        **kwargs)


@pytest.mark.parametrize('policy', ['preserve_current_violation', 'current_nonworsening', 'existing_caps'])
def test_exact_original_targets_without_changing_physical_rows(parent, policy):
    result = probe(parent, policy)
    targets = parent.limits if policy == 'existing_caps' else parent.previous
    expected = parent.domain['lower'].copy()
    owned_positive = parent.coefficient != 0.
    expected[owned_positive] = parent.original[owned_positive]-targets[parent.owners[owned_positive]]
    assert result['matrix'] is parent.domain['matrix']
    assert result['lower'].tobytes() == expected.tobytes()
    assert result['lower'][~owned_positive].tobytes() == parent.domain['lower'][~owned_positive].tobytes()
    np.testing.assert_array_equal(np.concatenate([r['dynamic_group_limits']
        for r in result['curvature_responses'].values()]), targets)
    note = result['report']['truth_progress_policy_probe']
    assert note['exact_original_constraint_lower_tapped']
    assert note['original_support_reference_bitexact']
    assert not note['historical_caps_changed']
    assert note['no_fake_common_fraction_used']


@pytest.mark.parametrize('policy', ['preserve_current_violation', 'existing_caps'])
def test_all_trial_fractions_keep_same_parent_targets_without_full_SOC_cuts(parent, policy):
    result = probe(parent, policy)
    pristine = result['lower'].copy()
    for alpha in (1., .5, 1./256.):
        child = result['at_trial_fraction'](alpha)
        assert child['matrix'] is result['matrix']
        assert child['lower'].tobytes() == pristine.tobytes()
        assert child['original_signed_margin_rows']['row_ids'] == ('physical_pivot',)
        assert child['original_signed_margin_rows']['nominal_margins'].tobytes() == np.array([.1]).tobytes()
        assert child['report']['trial_fraction_domain']['targets_constant_at_every_actual_trial_fraction']
    result['lower'][:] += 100.
    result['curvature_responses']['native']['dynamic_group_limits'][:] += 100.
    assert result['at_trial_fraction'](.5)['lower'].tobytes() == pristine.tobytes()


def test_mutable_metadata_isolated_but_canonical_source_record_identity_retained(parent):
    original_lower = parent.domain['lower'].copy()
    original_carrier = parent.domain['curvature_responses']['native']['nominal_carrier'].copy()
    result = probe(parent)
    first = result['at_trial_fraction'](.5)
    native = first['curvature_responses']['native']
    assert native['native_nominal_query_chart']['contexts'][0]['record'] is parent.record
    assert native['finite_station_contexts'][0]['record'] is parent.record
    assert native['row_carrier_projection'] is not parent.domain['curvature_responses']['native']['row_carrier_projection']
    native['row_carrier_projection'].data[:] = 12.
    native['nominal_carrier'][:] = 12.
    native['native_nominal_query_chart']['contexts'][0]['nominal_finite_segments'][:] = 12
    native['finite_station_contexts'][0]['coefficients'][:] = 12.
    first['original_signed_margin_rows']['nominal_margins'][:] = 12.
    first['report']['truth_feasibility_priority']['current_original_group_violations_px'][0] = 12.
    second = result['at_trial_fraction'](.5)
    second_native = second['curvature_responses']['native']
    np.testing.assert_array_equal(second_native['nominal_carrier'], original_carrier)
    np.testing.assert_array_equal(second_native['row_carrier_projection'].toarray(), np.eye(2))
    np.testing.assert_array_equal(second_native['native_nominal_query_chart']['contexts'][0]['nominal_finite_segments'], [0])
    assert second_native['finite_station_contexts'][0]['record'] is parent.record
    assert parent.domain['lower'].tobytes() == original_lower.tobytes()
    np.testing.assert_array_equal(parent.previous, [2., 0., 4.])
    np.testing.assert_array_equal(parent.limits, [3., 0., 5.])
    np.testing.assert_array_equal(parent.owners, [-1, 0, 0, 1, 2, -1])


def test_actual_native_and_station_interfaces_accept_original_record_identity(parent, monkeypatch):
    # Exercise the production gate and its subsequent query call without a
    # graph, projection Jacobian, source decode or changed observation.
    from calibration.solver_runtime.native_truth_nominal_chart import native_nominal_query_carriers
    from calibration.solver_runtime.finite_rectangle_projection_fd import _contexts
    package = ModuleType('structured_programme_source_candidate'); package.__path__ = []
    human = ModuleType('structured_programme_source_candidate.human_stroke_truth')
    camera = ModuleType('structured_programme_source_candidate.native_camera_local')
    calls = []
    def query(*args, **kwargs):
        calls.append(kwargs)
        return dict(signed_normal_px=np.array([.2]), coverage_tangent_px=np.array([.3]))
    human.finite_stroke_queries = query
    camera.native_camera_local = lambda block, core: core
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, human.__name__, human)
    monkeypatch.setitem(sys.modules, camera.__name__, camera)
    parent.record.update(row=dict(curve=dict(kind='line'), actual_pts=123),
        normals=np.array([[0., 1.]]), bounds=np.array([[-1., 1.]]),
        factor=SimpleNamespace(_world=lambda q, curve: np.array([[0., 0., 0.], [1., 0., 0.]])))
    native_context = parent.domain['curvature_responses']['native']['native_nominal_query_chart']['contexts'][0]
    for name in ('nominal_finite_segment_categories', 'nominal_source_support_faces', 'nominal_source_station_branches'):
        native_context[name] = None
    truth = SimpleNamespace(parameter_count=2, height_columns=np.array([1]), world_indices=np.array([0]),
        full_stroke_records=[parent.record], blocks={123: SimpleNamespace(bundle=SimpleNamespace(camera=lambda q, pts: q))},
        state=SimpleNamespace(geometry=SimpleNamespace(validate_parameters=lambda q: None),
            structured_programme_coordinate_chart=SimpleNamespace(core_count=2)))
    native = probe(parent)['curvature_responses']['native']
    value = native_nominal_query_carriers(truth, np.array([0., 1.]), native['native_nominal_query_chart'])
    np.testing.assert_array_equal(value, [.2, .3])
    assert len(calls) == 1
    assert _contexts(truth, native['finite_station_contexts'])[0]['record'] is parent.record


def test_fallback_reconstruction_cannot_claim_exact_original_support(parent):
    result = build_truth_progress_probe_domain(parent.domain, current_group_violations=parent.previous)
    note = result['report']['truth_progress_policy_probe']
    assert note['numeric_reconstruction_used']
    assert not note['original_support_reference_bitexact']
    assert note['original_b0_reconstruction_error_not_observable_without_tap']
    unchanged = parent.coefficient == 0.
    assert result['lower'][unchanged].tobytes() == parent.domain['lower'][unchanged].tobytes()


def test_accumulated_SOC_domain_is_rejected_without_silently_reusing_cuts(parent):
    parent.domain['lower'][1] += .1
    with pytest.raises(ValueError, match='contracted domain'):
        probe(parent)
    with pytest.raises(ValueError, match='pristine original domain'):
        build_truth_progress_probe_domain(parent.domain, current_group_violations=parent.previous)


@pytest.mark.parametrize('alpha', [0., -.1, 1.01, np.nan, np.inf])
def test_invalid_alpha_rejected(parent, alpha):
    with pytest.raises(ValueError):
        probe(parent)['at_trial_fraction'](alpha)


@pytest.mark.parametrize('bad', [-2, 2, .5])
def test_invalid_family_ownership_rejected(parent, bad):
    local = parent.domain['curvature_responses']['native']['dynamic_row_group_index'].astype(float)
    local[0] = bad
    parent.domain['curvature_responses']['native']['dynamic_row_group_index'] = local
    with pytest.raises(ValueError):
        probe(parent)


def test_missing_group_and_mismatched_global_ownership_rejected(parent):
    bad_owners = parent.owners.copy(); bad_owners[1] = 2
    with pytest.raises(ValueError, match='ownership'):
        build_truth_progress_probe_domain(parent.domain, current_group_violations=parent.previous,
            original_constraint_lower=parent.original, row_group_index=bad_owners)
    parent.domain['curvature_responses'].pop('confirmed')
    with pytest.raises(ValueError, match='All original truth groups'):
        probe(parent)


@pytest.mark.parametrize('limits', [[1., 0., 5.], [3., -1., 5.], [3., 1., 5.], [3., 0.], [3., 0., np.nan]])
def test_existing_caps_must_be_original_admissible_and_keep_source_zero(parent, limits):
    with pytest.raises(ValueError):
        build_truth_progress_probe_domain(parent.domain, current_group_violations=parent.previous,
            existing_group_limits=limits, policy='existing_caps', original_constraint_lower=parent.original,
            row_group_index=parent.owners)


def test_missing_exact_tap_half_or_caps_or_fake_common_fraction_rejected(parent):
    with pytest.raises(ValueError, match='Tap both'):
        build_truth_progress_probe_domain(parent.domain, current_group_violations=parent.previous,
            original_constraint_lower=parent.original)
    with pytest.raises(ValueError, match='explicitly supplied'):
        build_truth_progress_probe_domain(parent.domain, current_group_violations=parent.previous,
            policy='existing_caps')
    with pytest.raises(ValueError, match='original domain'):
        probe(parent, policy='common_contraction')
    parent.domain['report']['truth_feasibility_priority']['demanded_common_original_violation_fraction'] = 1.
    with pytest.raises(ValueError, match='strict common fraction'):
        probe(parent)
