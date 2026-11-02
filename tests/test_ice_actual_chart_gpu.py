"""Mechanical actual-ice CUDA regression; no footage or saved run is required.

Execution requires an explicitly selected G4/G3/CORE CUDA + cuDSS library.
An absent library is a skip, never evidence of a passed CUDA regression.
"""
import os

import numpy as np
import pytest

from rinkspace.core.gpu import ResidentResidualProblem
from rinkspace.core.native import NativeCore
from rinkspace.domain.ice_actual_chart import (
    ACTUAL_RADIUS_INDICES,
    ACTUAL_X_INDICES,
    NOMINAL_WITNESS_INDICES,
    actual_ice_chart_configuration,
)
from rinkspace.domain.ice_tolerance import DEVIATION_COUNT

import test_g2_gpu as fixture
import test_ice_tolerance as ice_fixture


CODES = np.array([
    *range(100, 132), *range(200, 204), *range(300, 312),
    *range(320, 324), *range(400, 412), *range(420, 424),
], dtype=np.int32)
OLD_ACTIVE_NOMINAL = np.array([
    0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    19, 20, 24, 25, 27, 29, 30, 33, 34, 35,
])


@pytest.fixture
def native_context(monkeypatch):
    library = next((os.environ[name] for name in (
        'RINKSPACE_G4_LIBRARY', 'RINKSPACE_G3_LIBRARY', 'RINKSPACE_CORE_LIBRARY',
    ) if os.environ.get(name)), None)
    if library is None:
        pytest.skip('explicit G4/G3/CORE CUDA + cuDSS library required; CUDA regression not run')
    core = NativeCore(library)
    assert hasattr(core._lib, 'rs_gpu_ice_actual_chart_version'), (
        'explicitly selected library lacks actual ice chart support')
    monkeypatch.setenv('RINKSPACE_G2_LIBRARY', str(core.library_path))
    seed = fixture.G2GPU()
    seed.setUp()
    return core, seed


def _scene():
    values = np.r_[
        [.5, .25, .1, .12, 1.85, .02],
        ice_fixture.nominal(),
        .0012 * np.sin(1.7 * np.arange(DEVIATION_COUNT) + .23),
    ]
    # Different F/R widths and signed transverse/right tilt are essential:
    # a zero or constant-width fixture misses the nonlinear chart path.
    values[45 + 48:45 + 52] = [-.001, .0015, -.0008, .0011]
    values[45 + 52:45 + 56] = [.0013, -.0009, .0014, -.0012]
    values[45 + 60:45 + 64] = [.0014, -.0011, .0017, -.0013]
    values[45 + 68:45 + 72] = [-.0012, .0011, -.0014, .0015]

    legacy_charts = np.zeros(values.size, np.int32)
    legacy_charts[[1, 2, 3, 5]] = 1
    legacy_charts[4] = 2
    positive_nominal = [3, 4, 9, 19, 20, 24, 25, 27, 29, 30, 33, 34]
    legacy_charts[6 + np.array(positive_nominal)] = 1
    legacy_charts[11:14] = [3, 4, 5]
    old_active = np.r_[6 + OLD_ACTIVE_NOMINAL, 45 + np.arange(DEVIATION_COUNT)]
    assert len(old_active) == 134
    configuration = actual_ice_chart_configuration(values, legacy_charts, old_active)
    assert configuration is not None
    active = configuration['active']
    assert len(active) == 128
    assert not np.isin(NOMINAL_WITNESS_INDICES, active).any()
    directions = np.zeros((values.size, len(active)))
    # Match the existing runner's positive/ordered .01 and additive .001
    # chart scales; actual quantities receive their physical-unit overrides.
    for column, index in enumerate(active):
        step = .01 if legacy_charts[index] > 0 else .001
        directions[index, column] = configuration['steps'].get(int(index), step)
    return values, configuration['charts'], active, directions


def _native_paths(core, values):
    assert values.shape == (152,)
    assert np.isfinite(values).all()
    assert np.max(np.abs(values[45:152])) <= .01
    paths = []
    # Each original shared_geometry call runs the complete 206-relation,
    # strict-topology/carrier oracle as well as the inclusive deviation box.
    for code in CODES:
        kind, parameters = core.shared_geometry(values[:6], values[6:], int(code), options=127)
        assert np.isfinite(parameters).all()
        paths.append(dict(kind=kind, parameters=parameters))
    assert len(paths) == 68
    return paths


def _problem(core, seed, values, *, observed=None, charts=None, directions=None):
    paths = _native_paths(core, values)
    rows = len(CODES) * 3
    kw = {
        **seed.kw, 'board': values[:6], 'layout': values[6:], 'options': 127,
        'codes': np.repeat(np.arange(len(CODES), dtype=np.int32), 3),
        'stations': np.tile([.17, .51, .83], len(CODES)),
        'times': np.linspace(0., .1, rows),
        'observed': np.zeros((rows, 2)) if observed is None else observed,
    }
    problem = ResidentResidualProblem(core, **kw)
    try:
        problem.configure_geometry(paths)
        if charts is not None:
            records = np.zeros((len(CODES), 12), np.int32)
            records[:, 0] = 1
            records[:, 1] = CODES
            problem.configure_scene_parameters(
                values=values, records=records, data=np.zeros(1),
                directions=directions, charts=charts)
            q = directions.shape[1]
            problem.configure_pixel_linearization(
                directions=np.zeros((len(seed.kw['controls']), 3, q)),
                local_ids=np.full(rows, -1, np.int32), local_scales=np.zeros(0))
    except BaseException:
        problem.close()
        raise
    return problem


def _reference_retract(values, charts, delta):
    """Independent physical update, followed by interior legacy witness export.

    This Python reference does not call a native chart or differentiate the
    min/max witness choice. All test directions remain well inside +/-1%, so
    midpoint witnesses are representable without a boundary repair.
    """
    actual_x = np.repeat(values[[6, 7]], 4) + values[9] * values[ACTUAL_X_INDICES]
    nominal_radii = values[10] * np.r_[values[11:14], 1.]
    actual_radius = (np.tile(nominal_radii, 8) *
                     (1. + values[ACTUAL_RADIUS_INDICES])).reshape(8, 4)
    actual_x = actual_x + delta[ACTUAL_X_INDICES]
    actual_radius = actual_radius * np.exp(delta[ACTUAL_RADIUS_INDICES].reshape(8, 4))

    result = values.copy()
    for index, code in enumerate(charts):
        if code == 0:
            result[index] += delta[index]
        elif code == 1:
            result[index] *= np.exp(delta[index])
        elif code == 2:
            result[index] = 1. + (values[index] - 1.) * np.exp(delta[index])
        else:
            assert code in (10, 11, 12, 13), 'ordered radii must be witness outputs'

    pitch = result[9]
    for group in range(2):
        segment = actual_x[group * 4:(group + 1) * 4]
        low = segment.max() - .01 * pitch
        high = segment.min() + .01 * pitch
        assert low <= high
        centre = low + .5 * (high - low)
        result[6 + group] = centre
        indices = ACTUAL_X_INDICES[group * 4:(group + 1) * 4]
        result[indices] = (segment - centre) / pitch
    lower = actual_radius.max(axis=0) / 1.01
    upper = actual_radius.min(axis=0) / .99
    assert np.all(lower <= upper)
    radius = lower + .5 * (upper - lower)
    result[10] = radius[3]
    result[11:14] = radius[:3] / radius[3]
    # Stable subtraction inverse avoids r/R-1 cancellation.
    result[ACTUAL_RADIUS_INDICES] = ((actual_radius - radius) / radius).ravel()
    assert np.max(np.abs(result[45:152])) < .009
    return result


def _uv(problem, backend='cuda'):
    evaluated = problem.evaluate(backend)
    np.testing.assert_array_equal(evaluated['status'], 0)
    return evaluated['values'][:, 3:5]


def test_actual_chart_all_128_cuda_columns_and_independent_fd(native_context):
    core, seed = native_context
    values, charts, active, directions = _scene()
    rows = np.arange(len(CODES) * 3, dtype=np.int32)
    with _problem(core, seed, values, charts=charts, directions=directions) as problem:
        assert problem.linearization_info['q'] == 128
        assert problem.linearization_info['local_count'] == 0
        np.testing.assert_allclose(_uv(problem), _uv(problem, 'cpu'), rtol=1e-12, atol=3e-11)
        problem.linearize('cuda')
        cuda = problem.probe_rows(rows)
        problem.linearize('cpu')
        cpu = problem.probe_rows(rows, 'cpu')
        np.testing.assert_allclose(cuda[:, :, :128], cpu[:, :, :128], rtol=1e-8, atol=3e-10)
        np.testing.assert_array_equal(cuda[:, :, 128], 0.)
        assert np.isfinite(cuda).all()
        assert np.all(np.linalg.norm(cuda[:, :, :128], axis=(0, 1)) > 0.)

        # F/R actual X, each of the four radius layers (different houses),
        # pitch, nominal width, row gap, taper and signed right tilt.
        for ambient in [49, 52, 53, 56, 61, 66, 71, 76, 9, 15, 45, 93, 105]:
            column = int(np.flatnonzero(active == ambient)[0])
            epsilon = .002
            samples = []
            for sign in (-1., 1.):
                state = _reference_retract(values, charts, sign * epsilon * directions[:, column])
                with _problem(core, seed, state) as sample:
                    samples.append(_uv(sample))
            difference = (samples[1] - samples[0]) / (2. * epsilon)
            np.testing.assert_allclose(
                cuda[:, :, column], difference, rtol=3e-4, atol=3e-8,
                err_msg=f'actual ice ambient index {ambient}')


def test_one_actual_joint_lm_trial_descends_and_exports_legal_legacy_state(native_context, monkeypatch):
    core, seed = native_context
    monkeypatch.setenv('RINKSPACE_SMALL_DUAL_BACKEND', 'cpu')
    monkeypatch.setenv('RINKSPACE_STATIONARITY_BACKEND', 'cpu')
    values, charts, active, directions = _scene()
    planted_step = .03 * np.sin(.43 * np.arange(len(active)) + .2)
    for number, ambient in enumerate(ACTUAL_X_INDICES):
        planted_step[np.flatnonzero(active == ambient)[0]] = .2 * np.cos(number + .3)
    for number, ambient in enumerate(ACTUAL_RADIUS_INDICES):
        planted_step[np.flatnonzero(active == ambient)[0]] = .16 * np.sin(.7 * number + .4)
    planted_step[np.flatnonzero(active == 9)[0]] = .04
    planted = _reference_retract(values, charts, directions @ planted_step)
    # Targets use the same native geometry and camera with independently
    # retracted actual quantities; no Jacobian-generated target offsets.
    with _problem(core, seed, planted) as target:
        observed = _uv(target).copy()
    with _problem(core, seed, values, observed=observed, charts=charts, directions=directions) as problem:
        old_cost = .5 * np.sum((_uv(problem) - observed) ** 2)
        assert old_cost > 0
        problem.configure_solver(
            lower=np.zeros(0), upper=np.zeros(0),
            device_budget_bytes=problem.creation['free_after_alloc_bytes'],
            host_budget_bytes=2**30, damping_metric='jacobi', initial_lambda=.001)
        summary = problem.lm_step()['summary']
        assert summary['accepted'], summary
        assert summary['trial_domain_failures'] == 0, summary
        assert summary['trial_cost'] < summary['old_cost'], summary
        np.testing.assert_allclose(summary['old_cost'], old_cost, rtol=1e-9, atol=1e-13)
        final_uv = _uv(problem)
        final_cost = .5 * np.sum((final_uv - observed) ** 2)
        assert final_cost < old_cost
        np.testing.assert_allclose(final_cost, summary['trial_cost'], rtol=1e-8, atol=1e-13)
        np.testing.assert_allclose(final_uv, _uv(problem, 'cpu'), rtol=1e-12, atol=3e-11)

        state = problem.export_state()
        assert state['layout'].shape == (146,)
        assert state['scene_values'].shape == (152,)
        np.testing.assert_array_equal(state['scene_values'][:6], state['board'])
        np.testing.assert_array_equal(state['scene_values'][6:], state['layout'])
        np.testing.assert_array_equal(state['controls'], seed.kw['controls'])
        np.testing.assert_array_equal(state['stations'], np.tile([.17, .51, .83], len(CODES)))
        _native_paths(core, state['scene_values'])
        assert np.max(np.abs(state['layout'][39:])) <= .01
        problem.linearize('cuda')
        diagnostic = problem.stationarity_diagnostics(include_constraints=True)
        assert diagnostic['invalid'] == 0
        assert diagnostic['physical_feasibility_violation'] < 1e-10


@pytest.mark.parametrize('domain_cache', ['0', '1'])
def test_actual_radius_spread_is_repaired_before_original_domain_acceptance(native_context, monkeypatch, domain_cache):
    core, seed = native_context
    monkeypatch.setenv('RINKSPACE_SMALL_DUAL_BACKEND', 'cpu')
    monkeypatch.setenv('RINKSPACE_STATIONARITY_BACKEND', 'cpu')
    monkeypatch.setenv('RINKSPACE_SHARED_DOMAIN_CACHE', domain_cache)
    values, charts, _, _ = _scene()
    directions = np.zeros((len(values), 1))
    directions[61, 0] = .001
    directions[65, 0] = -.001
    with _problem(core, seed, values, charts=charts, directions=directions) as source:
        uv = _uv(source)
        source.linearize('cuda')
        jac = source.probe_rows(np.arange(len(uv)))[:, :, 0]
    # The unconstrained log-radius spread is about 20%, while the common
    # nominal witness allows only log(1.01/.99). A NaN separation point used
    # to reject this trial before the affine group face could repair it.
    observed = uv + 100. * jac
    with _problem(core, seed, values, observed=observed, charts=charts, directions=directions) as problem:
        problem.configure_solver(lower=np.zeros(0), upper=np.zeros(0),
            device_budget_bytes=problem.creation['free_after_alloc_bytes'],
            host_budget_bytes=2**30, damping_metric='jacobi', initial_lambda=.001)
        summary = problem.lm_step()['summary']
        assert summary['accepted'], summary
        assert summary['trial_domain_failures'] == 0
        assert summary['actual_decrease'] > 0.
        assert problem.performance_metrics()['constraint_projections'] > 0
        state = problem.export_state()
        _native_paths(core, state['scene_values'])
        assert np.max(np.abs(state['layout'][39:])) <= .01
        initial_r0 = values[10] * values[11] * (1. + values[61])
        initial_r1 = values[10] * values[11] * (1. + values[65])
        final_values = state['scene_values']
        final_r0 = final_values[10] * final_values[11] * (1. + final_values[61])
        expected_step = (np.log(1.01 / .99) - np.log(initial_r0 / initial_r1)) / .002
        actual_step = np.log(final_r0 / initial_r0) / .001
        np.testing.assert_allclose(actual_step, expected_step, rtol=0., atol=2e-9)
        # An accepted inclusive boundary must remain feasible at the next
        # zero step, rather than trigger endless boundary/damping retries.
        result = problem.solve()
        assert result['numerically_converged'], result
        assert result['trials'] == 0, result
