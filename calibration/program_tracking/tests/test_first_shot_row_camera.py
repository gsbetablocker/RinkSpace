"""Independent native SO(3), timing and conditioned-chain checks."""
from dataclasses import replace

import numpy as np
import pytest

from calibration.program_tracking.first_shot_row_camera import (
    RowCameraDomainError, prepare_frame_row_camera,
)
from calibration.program_tracking.z190_physical_camera import SharedCalibration, ViewState


class _State:
    frame_count = 5
    scene_unit_working_m = .31354
    row_readout_index = 147
    count = 148
    pts = np.arange(256256, 257281, 256, dtype=np.int64)
    source_clock = {"time_base": "1/12800", "fps": "50/1"}
    base = SharedCalibration("test-native-shot", (10., 20., 5.),
        lever_xyz=(.5, -.2, .25), principal_x_fraction=(.48,),
        principal_y_fraction=(.53,), k1=(.08,))

    def _q(self, q):
        assert q.shape == (self.count,)
        return q

    def view_columns(self, n):
        return np.arange(127 + 4 * (n - 1), 131 + 4 * (n - 1))

    def calibration(self, q):
        d = q[118:127]
        return replace(self.base,
            pivot_xyz=tuple(np.asarray(self.base.pivot_xyz) + d[:3] / self.scene_unit_working_m),
            lever_xyz=tuple(np.asarray(self.base.lever_xyz) + d[3:6] / self.scene_unit_working_m),
            principal_x_fraction=(self.base.principal_x_fraction[0] + d[6] / 1920.,),
            principal_y_fraction=(self.base.principal_y_fraction[0] + d[7] / 1080.,),
            k1=(self.base.k1[0] + .001 * d[8],))


def _fixture(tau):
    state = _State(); q = np.zeros(state.count)
    for n in range(1, 6):
        q[state.view_columns(n)] = [1.58 + .002 * n, -.3 + .008 * n,
                                    .26 - .007 * n, 1.2 - .006 * n]
    q[-1] = tau
    return state, q


@pytest.mark.parametrize("n", [1, 3, 5])
def test_zero_readout_replays_the_exact_native_camera(n):
    state, q = _fixture(0.)
    ctx = prepare_frame_row_camera(state, q, n)
    expected = state.calibration(q).camera25(ViewState.from_state4(q[state.view_columns(n)]))
    for row in [0., 300., 539.5, 800., 1079.]:
        value, columns, chain = ctx.camera15_mapping(row, jacobian=False)
        assert np.array_equal(value, expected)
        assert chain is None
        assert len(columns) == (22 if n == 3 else 18)
    assert np.array_equal(ctx.camera25_at_rows(np.array([0., 1079.])),
                          np.tile(expected, (2, 1)))


@pytest.mark.parametrize("tau", [10.46, -13.])
@pytest.mark.parametrize("n", [1, 3, 5])
def test_every_installation_neighbour_and_readout_column_matches_independent_fd(tau, n):
    state, q = _fixture(tau)
    ctx = prepare_frame_row_camera(state, q, n)
    condition = np.r_[np.full(3, .01), np.ones(5), np.full(3, .001), np.ones(4)]
    for row in [0., 300., 800., 1079.]:
        _, columns, chain = ctx.camera15_mapping(row)
        for j, column in enumerate(columns):
            h = 1e-5 if column == state.row_readout_index else 1e-6
            high, low = q.copy(), q.copy()
            high[column] += h; low[column] -= h
            hp = prepare_frame_row_camera(state, high, n).camera15_mapping(row, jacobian=False)[0]
            lp = prepare_frame_row_camera(state, low, n).camera15_mapping(row, jacobian=False)[0]
            finite = (hp[:15] - lp[:15]) * condition / (2. * h)
            np.testing.assert_allclose(chain[:, j], finite, atol=1e-7, rtol=1e-5)


def test_batch_scalar_trajectory_and_cache_identity():
    state, q = _fixture(10.46)
    ctx = prepare_frame_row_camera(state, q, 3)
    rows = np.linspace(0., 1079., 91)
    scalar = np.asarray([ctx.camera15_mapping(y, jacobian=False)[0] for y in rows])
    np.testing.assert_allclose(ctx.camera25_at_rows(rows), scalar, atol=1e-12, rtol=1e-12)
    changed = q.copy(); changed[-1] += 1.
    assert ctx.cache_key != prepare_frame_row_camera(state, changed, 3).cache_key
    # A neighbour can change while the centre camera remains exactly unchanged.
    changed = q.copy(); changed[state.view_columns(2)[1]] += .001
    assert ctx.cache_key != prepare_frame_row_camera(state, changed, 3).cache_key


def test_timing_guards_and_terminal_extrapolation_are_explicit():
    state, q = _fixture(20.)
    ctx = prepare_frame_row_camera(state, q, 1)
    assert np.isfinite(ctx.camera25_at_row(0.)).all()
    with pytest.raises(ValueError):
        ctx.camera25_at_row(-1.)
    with pytest.raises(RowCameraDomainError):
        ctx.camera25_at_row(-2000., geometry_navigation=True)
    q[-1] = 20.001
    with pytest.raises(RowCameraDomainError):
        prepare_frame_row_camera(state, q, 1)


def test_tiny_native_rotation_differences_do_not_round_to_a_stationary_camera():
    state, q = _fixture(10.46)
    for n in range(1, 6):
        q[state.view_columns(n)[:3]] = [1.65 + 2e-8 * n, .2 + 3e-8 * n, -.19 - 1e-8 * n]
    ctx = prepare_frame_row_camera(state, q, 3)
    condition = np.r_[np.full(3, .01), np.ones(5), np.full(3, .001), np.ones(4)]
    for row in [0., 1079.]:
        _, columns, chain = ctx.camera15_mapping(row)
        for j, column in enumerate(columns):
            if column not in np.r_[state.view_columns(2)[:3], state.view_columns(3)[:3], state.view_columns(4)[:3]]:
                continue
            for h in [1e-7, 1e-6, 1e-5]:
                high, low = q.copy(), q.copy(); high[column] += h; low[column] -= h
                hp = prepare_frame_row_camera(state, high, 3).camera15_mapping(row, jacobian=False)[0]
                lp = prepare_frame_row_camera(state, low, 3).camera15_mapping(row, jacobian=False)[0]
                finite = (hp[:15] - lp[:15]) * condition / (2. * h)
                np.testing.assert_allclose(chain[:, j], finite, atol=1e-7, rtol=1e-5)


def test_small_absolute_rotations_keep_scalar_and_batch_on_the_same_short_arc():
    state, q = _fixture(10.46)
    direction = np.array([2e-8, 3e-8, -1e-8])
    for n in range(1, 6):
        q[state.view_columns(n)[:3]] = n * direction
    ctx = prepare_frame_row_camera(state, q, 3)
    rows = np.array([0., 300., 539.5, 800., 1079.])
    expected = (3. + q[-1] * .001 * (rows / 1079. - .5) / .02)[:, None] * direction
    scalar = np.asarray([ctx.camera25_at_row(y) for y in rows])
    np.testing.assert_allclose(scalar[:, 3:6], expected, atol=1e-20, rtol=1e-12)
    np.testing.assert_allclose(ctx.camera25_at_rows(rows), scalar, atol=1e-12, rtol=1e-12)
