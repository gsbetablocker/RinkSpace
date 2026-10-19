"""The preserved real failure distinguishes source kinks from a bad Jacobian.

The optional Data fixture is the exact preflight failure, never a fit input or
an edited residual. Missing historical Data skips this local regression only.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix


_PROBE = Path(
    'D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/'
    'solver_acceleration_20261003/strict_direction_probes/'
    'failed_strict_direction_4ca01bd439b8f647df81.npz')
_SHA = 'a411a7536f7732832b8ec3d2c05053c9805825a7607a5c712a52bcf9f57856ce'


@pytest.fixture(scope='module')
def probe():
    if not _PROBE.is_file():
        pytest.skip('Preserved actual source-direction diagnostic is unavailable')
    assert hashlib.sha256(_PROBE.read_bytes()).hexdigest() == _SHA
    with np.load(_PROBE, allow_pickle=False) as archive:
        values = {key: np.array(archive[key], copy=True) for key in archive.files}
    assert json.loads(str(values['metadata_json']))['actual_pts'] == 229672192
    return values


def _prediction(probe):
    matrix = csr_matrix((probe['matrix_data'], probe['matrix_indices'],
                         probe['matrix_indptr']), shape=tuple(probe['matrix_shape']))
    return matrix @ probe['direction']


def _qualified(probe, *, jacobian_scale=1.):
    from calibration.solver_runtime import install
    install()
    from calibration.solver_runtime.source_direction_probe import (
        qualify_source_direction_records)
    records = {
        label: {key[len(label) + 2:]: value
                for key, value in probe.items() if key.startswith(label + '__')}
        for label in ('nominal', 'plus', 'minus')}
    matrix = csr_matrix((jacobian_scale * probe['matrix_data'], probe['matrix_indices'],
                         probe['matrix_indptr']), shape=tuple(probe['matrix_shape']))
    original_indices = json.loads(str(probe['metadata_json']))[
        'selected_original_group_point_indices']
    return qualify_source_direction_records(
        records, matrix, probe['direction'], float(probe['step']),
        original_group_point_indices=original_indices)


def _relative(measured, predicted):
    return float(np.abs(measured - predicted).max()
                 / max(1., float(np.abs(predicted).max())))


def _fixed_residuals(probe):
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate.native_ice_curve_source import (
        _curve_error_on_association)
    association = (probe['nominal__selected_segment'],
                   probe['nominal__selected_raw_t'])
    sigma = np.repeat(probe['nominal__sigma'], 2)
    result = []
    for label in ('nominal', 'plus', 'minus'):
        raw, _ = _curve_error_on_association(
            probe[label + '__camera25'], probe[label + '__world'],
            probe['nominal__uv'], association)
        result.append(raw / sigma)
    return result


def _stable_queries(probe):
    def region(raw_t):
        return np.where(raw_t <= 0., -1, np.where(raw_t >= 1., 1, 0))
    segment = probe['nominal__selected_segment']
    clamp = region(probe['nominal__selected_raw_t'])
    stable = np.ones(len(segment), bool)
    for label in ('plus', 'minus'):
        stable &= probe[label + '__selected_segment'] == segment
        stable &= region(probe[label + '__selected_raw_t']) == clamp
    return stable


def test_original_fixed_branch_qualifies_every_query_without_changing_nominal(probe):
    before = {key: value.tobytes() for key, value in probe.items()}
    nominal, plus, minus = _fixed_residuals(probe)
    assert np.array_equal(nominal, probe['nominal__residual'])
    measured = (plus - minus) / (2 * float(probe['step']))
    assert _relative(measured, _prediction(probe)) < 1e-4
    assert before == {key: value.tobytes() for key, value in probe.items()}


def test_actual_unrestricted_trial_gate_still_qualifies_exactly_stable_queries(probe):
    predicted = _prediction(probe)
    measured = ((probe['plus__residual'] - probe['minus__residual'])
                / (2 * float(probe['step'])))
    stable = _stable_queries(probe)
    assert np.flatnonzero(~stable).tolist() == [8, 9, 10, 11, 12, 15]
    assert np.count_nonzero(stable) == 10
    scalar_stable = np.repeat(stable, 2)
    # Use the stable subset's own denominator, so another query cannot hide
    # an incorrect derivative on the genuinely differentiable source branch.
    assert _relative(measured[scalar_stable], predicted[scalar_stable]) < 1e-4
    assert _relative(measured, predicted) > 1e-4


def test_fixed_branch_gate_rejects_a_scaled_jacobian_even_at_actual_source_kink(probe):
    _, plus, minus = _fixed_residuals(probe)
    measured = (plus - minus) / (2 * float(probe['step']))
    assert _relative(measured, 1.01 * _prediction(probe)) > 1e-4


def test_current_preflight_records_all_changed_source_queries_without_retirement(probe):
    receipt = _qualified(probe)
    assert receipt['passed'] is True
    assert receipt['original_relative_threshold'] == 1e-4
    assert receipt['complete_original_branch_J_passed'] is True
    assert receipt['unrestricted_stable_queries_passed'] is True
    assert receipt['unrestricted_stable_queries'] == 10
    assert receipt['unrestricted_changed_queries'] == 6
    assert receipt['source_FIT_pixels_removed'] == 0
    assert receipt['all_FIT_source_residuals_J_and_nonlinear_trials_unchanged'] is True
    assert [entry['original_group_point_index']
            for entry in receipt['changed_segment_or_clamp_queries']] == [17, 19, 21, 23, 25, 32]


def test_current_preflight_still_rejects_actual_complete_J_scale_error(probe):
    receipt = _qualified(probe, jacobian_scale=1.01)
    assert receipt['passed'] is False
    assert receipt['complete_original_branch_J_passed'] is False
