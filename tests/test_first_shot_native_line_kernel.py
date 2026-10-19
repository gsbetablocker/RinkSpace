"""Native source queries retain their mean, coordinate and implicit-row chain."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from calibration.solver_runtime import install
install()
from calibration.program_tracking.first_shot_source_kernel import _interfaces, evaluate_projected_targets
from calibration.program_tracking.first_shot_native_line_kernel import (
    NativeLineQueryError, evaluate_native_line_queries, evaluate_row_native_line_queries,
)
from calibration.program_tracking.first_shot_row_camera import FrameRowCamera
from calibration.program_tracking.first_shot_row_source import _observations, evaluate_row_timed_frame, row_timed_raw
from calibration.program_tracking.z190_physical_camera import SharedCalibration


def central_case():
    camera = np.zeros(25)
    camera[3:6] = [.07, -.11, .03]
    camera[6:8] = np.log([900., 870.])
    camera[8:11] = [960., 540., 1.2]
    camera[11:15] = [.035, -.003, .001, -.0007]
    origin, vector = np.array([-2., -.4, 7.]), np.array([4., 1.1, .5])
    t = np.linspace(0., 1., 17)
    world = origin+t[:, None]*vector
    true = _interfaces()[0](camera, origin+np.array([.17, .43, .82])[:, None]*vector)[0]
    direction = np.array([[.6, .8], [-.3, np.sqrt(.91)], [.15, np.sqrt(.9775)]])
    row = dict(_target_key='line', target=dict(kind='ice_curve'), point_correspondence=False,
        _continuous_world_line=True, _line_end_extent_geometry_columns=[0],
        uv=true+2.*direction, native_line_queries=dict(query_uv=true-3.*direction,
            profile_normal=direction, query_ids=['q0', 'q1', 'q2']))
    derivative = np.zeros(world.shape+(3,))
    derivative[:, :, 0] = t[:, None]*vector
    derivative[:, 1, 1] = 1.
    derivative[:, 2, 2] = t
    return camera, row, world, derivative, true


def test_fixed_query_intersection_is_not_nearest_foot_or_observation_selected():
    camera, row, world, _, true = central_case()
    before = deepcopy(row)
    r, _, _ = evaluate_native_line_queries(camera, row, world, jacobian=False)
    np.testing.assert_allclose(r.reshape(-1, 2)+row['uv'], true, atol=2e-10, rtol=0.)
    moved = dict(row, uv=row['uv']+[[19., -30.], [-11., 7.], [23., 17.]])
    changed = evaluate_native_line_queries(camera, moved, world, jacobian=False)[0]
    np.testing.assert_allclose(changed.reshape(-1, 2)+moved['uv'], true, atol=2e-10, rtol=0.)
    old_row = dict(row); old_row.pop('native_line_queries')
    nearest = evaluate_projected_targets(camera, [old_row], {'line':world}, jacobian=False)[0]
    assert np.linalg.norm(nearest-r) > .1
    np.testing.assert_array_equal(row['uv'], before['uv'])
    np.testing.assert_array_equal(row['native_line_queries']['query_uv'], before['native_line_queries']['query_uv'])


def test_camera15_and_world_j_include_the_implicit_query_station():
    camera, row, world, derivative, _ = central_case()
    r, jcam, jworld = evaluate_native_line_queries(camera, row, world, derivative)
    np.testing.assert_array_equal(jworld[:, 0], 0.)
    h = 1e-6
    scale = np.r_[np.full(3, 100.), np.ones(5), np.full(3, 1000.), np.ones(4)]
    for col in range(15):
        plus, minus = camera.copy(), camera.copy()
        plus[col] += h*scale[col]; minus[col] -= h*scale[col]
        rp = evaluate_native_line_queries(plus, row, world, jacobian=False)[0]
        rm = evaluate_native_line_queries(minus, row, world, jacobian=False)[0]
        np.testing.assert_allclose((rp-rm)/(2*h), jcam[:, col], rtol=3e-7, atol=3e-5)
    for col in range(3):
        rp = evaluate_native_line_queries(camera, row, world+h*derivative[:, :, col], jacobian=False)[0]
        rm = evaluate_native_line_queries(camera, row, world-h*derivative[:, :, col], jacobian=False)[0]
        np.testing.assert_allclose((rp-rm)/(2*h), jworld[:, col], rtol=3e-7, atol=3e-5)
    dispatched = evaluate_projected_targets(camera, [row], {'line':world}, {'line':derivative})
    for a, b in zip(dispatched, (r, jcam, jworld), strict=True):np.testing.assert_array_equal(a, b)


def test_finite_world_support_is_not_clamped_to_an_endpoint():
    camera, row, world, _, _ = central_case()
    outside = _interfaces()[0](camera, (world[0]+1.4*(world[-1]-world[0]))[None])[0]
    saved = dict(query_uv=outside, profile_normal=[[0., 1.]], query_ids=['outside'])
    with pytest.raises(NativeLineQueryError, match='found 0'):
        evaluate_native_line_queries(camera, dict(row, uv=outside, native_line_queries=saved), world, jacobian=False)


def test_multiple_physical_roots_and_tangent_queries_fail_closed():
    camera = np.zeros(25); camera[6:8] = np.log([900., 900.]); camera[8:10] = [960., 540.]; camera[11] = 2.
    world = np.array([[-2., 1., 6.], [2., 1., 6.]])
    y0 = _interfaces()[0](camera, np.array([[0., 1., 6.]]))[0][0, 1]
    row = dict(uv=np.array([[960., y0+2.]]), native_line_queries=dict(
        query_uv=[[960., y0+2.]], profile_normal=[[1., 0.]], query_ids=['ambiguous']))
    with pytest.raises(NativeLineQueryError, match='found 2'):
        evaluate_native_line_queries(camera, row, world, jacobian=False)
    row['uv'][0, 1] = y0; row['native_line_queries']['query_uv'][0][1] = y0
    with pytest.raises(NativeLineQueryError):
        evaluate_native_line_queries(camera, row, world, jacobian=False)


class SyntheticState:
    frame_count = 3
    pts = np.array([0, 256, 512])
    source_clock = dict(time_base='1/12800', fps='50/1')
    row_readout_index = 139
    count = 140
    scene_unit_working_m = 1.
    world_parameter_indices = np.arange(3)
    zoom_parameter_indices = ()
    base_calibration = SharedCalibration('test', (0., 0., 0.))

    def _q(self, q):return np.asarray(q)
    def view_columns(self, index):return np.arange(127+4*(index-1), 131+4*(index-1))
    def calibration(self, q):
        return replace(self.base_calibration, pivot_xyz=tuple(q[118:121]), lever_xyz=tuple(q[121:124]),
            principal_x_fraction=(q[124]/1920.,), principal_y_fraction=(q[125]/1080.,), k1=(.001*q[126],))


def rolling_case():
    state = SyntheticState(); q = np.zeros(state.count)
    q[121:124] = [.01, -.2, .12]; q[124:127] = [960., 540., 20.]
    q[127:139] = np.array([
                  [.01, -.02, .003, np.log(900./1920.)], [.03, .02, .005, np.log(930./1920.)],
                  [.052, .06, .004, np.log(970./1920.)]]).ravel()
    q[139] = 12.
    context = FrameRowCamera(state, q, 2)
    world = np.array([[-2., -1.7, 7.], [2., 2.1, 7.5]])
    pts = world[0]+np.array([.2, .75])[:, None]*(world[-1]-world[0])
    uv = _interfaces()[0](context.camera25_at_row(539.5), pts)[0]
    direction = np.array([[.3, np.sqrt(.91)], [-.4, np.sqrt(.84)]])
    row = dict(_target_key='line', target=dict(kind='ice_curve'), point_correspondence=False,
        _continuous_world_line=True, uv=uv+2.*direction,
        native_line_queries=dict(query_uv=uv-3.*direction, profile_normal=direction, query_ids=['a', 'b']))
    dw = np.zeros(world.shape+(3,))
    dw[:, :, :3] = np.eye(3)[None]
    return state, q, context, row, world, dw


def test_exact_row_camera_velocity_includes_rotation_focal_and_moving_centre():
    state, q, context, _, _, _ = rolling_case()
    scale = np.r_[np.full(3, .01), np.ones(5), np.full(3, .001), np.ones(4)]
    for y in [200., 530., 800.]:
        h = .001
        numeric = (context.camera25_at_row(y+h)[:15]-context.camera25_at_row(y-h)[:15])/(2*h)*scale
        np.testing.assert_allclose(context.camera15_row_derivative(y), numeric, atol=2e-10, rtol=1e-6)
    # At the native knot the existing piecewise trajectory owns its right
    # derivative; the two adjacent velocities need not be equal.
    h = .0001; y = 539.5
    numeric = (context.camera25_at_row(y+h)[:15]-context.camera25_at_row(y)[:15])/h*scale
    np.testing.assert_allclose(context.camera15_row_derivative(y), numeric, atol=2e-10, rtol=1e-5)
    q[state.row_readout_index] = 0.
    np.testing.assert_array_equal(FrameRowCamera(state, q, 2).camera15_row_derivative(100.), np.zeros(15))


def test_self_row_two_variable_chain_keeps_all_neighbour_rf4_and_readout():
    state, q, context, row, world, dw = rolling_case()
    r, jc, jg, columns = evaluate_row_native_line_queries(context, row, world, dw)
    base = q.copy()
    def value(qq):
        changed = world+np.einsum('ijk,k->ij', dw, qq[:3]-base[:3])
        return evaluate_row_native_line_queries(FrameRowCamera(state, qq, 2), row, changed, jacobian=False)[0]
    for col in np.r_[np.arange(3), columns]:
        h = 1e-5 if col in (124, 125, 126, 139) else 1e-6
        plus, minus = q.copy(), q.copy(); plus[col] += h; minus[col] -= h
        numeric = (value(plus)-value(minus))/(2*h)
        expected = jg[:, col] if col < 3 else jc[:, np.flatnonzero(columns == col)[0]]
        np.testing.assert_allclose(numeric, expected, rtol=8e-6, atol=3e-5)
    frame = dict(index=2, source_point_count=2, rows=[row])
    whitening = type('W', (), dict(apply=staticmethod(lambda x:x)))()
    public_r, public_j = evaluate_row_timed_frame(state, q, frame, whitening, {'line':world}, {'line':dw})
    np.testing.assert_array_equal(public_r, r)
    np.testing.assert_array_equal(public_j[:, columns].toarray(), jc)
    np.testing.assert_array_equal(public_j[:, :3].toarray(), jg)
    assert np.linalg.norm(jc[:, np.isin(columns, state.view_columns(1))]) > 0.
    assert np.linalg.norm(jc[:, np.isin(columns, state.view_columns(3))]) > 0.


def test_native_query_metadata_is_sliced_in_source_order():
    _, _, _, row, world, dw = rolling_case()
    samples = list(_observations(dict(rows=[row]), {'line':world}, {'line':dw}))
    assert [s[0]['native_line_queries']['query_ids'] for s in samples] == [['a'], ['b']]
    for i, (sample, _, _) in enumerate(samples):
        np.testing.assert_array_equal(sample['native_line_queries']['query_uv'], np.asarray(row['native_line_queries']['query_uv'])[i:i+1])


def test_prediction_may_leave_sensor_without_clamping_the_source_or_prediction():
    state, q, context, row, _, _ = rolling_case()
    world = np.array([[-2., 4.6, 7.], [2., 4.6, 7.]])
    row['uv'] = np.array([[960., 1070.]])
    row['native_line_queries'] = dict(query_uv=[[960., 1070.]],
        profile_normal=[[0., 1.]], query_ids=['outside_prediction'])
    raw = evaluate_row_native_line_queries(context, row, world, jacobian=False)[0]
    predicted = raw+row['uv'][0]
    assert predicted[1] > 1079.
    with pytest.raises(ValueError):
        context.camera15_mapping(float(predicted[1]))
    assert np.isfinite(context.camera15_mapping(float(predicted[1]), geometry_navigation=True)[0]).all()
    h = .001
    scale = np.r_[np.full(3, .01), np.ones(5), np.full(3, .001), np.ones(4)]
    numeric = (context.camera25_at_row(predicted[1]+h, geometry_navigation=True)[:15]
        -context.camera25_at_row(predicted[1]-h, geometry_navigation=True)[:15])/(2*h)*scale
    np.testing.assert_allclose(context.camera15_row_derivative(predicted[1], geometry_navigation=True), numeric,
        rtol=1e-6, atol=2e-10)
    np.testing.assert_array_equal(row['uv'], [[960., 1070.]])
    terminal = FrameRowCamera(state, q, 3)
    world[:, 1] = 8.3
    with pytest.raises(NativeLineQueryError, match='native/terminal time support'):
        evaluate_row_native_line_queries(terminal, row, world, jacobian=False)


def test_safe_frame_packet_replays_original_implicit_query_and_correlated_W(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from calibration.program_tracking import first_shot_fit as fit
    from calibration.program_tracking.first_shot_safe_frame_export import export_source_frames, load_source_frames
    from calibration.program_tracking.source_evidence_whitening import SourceWhiteningOperator
    from calibration.program_tracking.first_shot_source_transition import reference
    state, q, _, row, world, _ = rolling_case()
    row['sigma'] = np.array([.7, 1.3])
    row['source_point_ids'] = ['original-a', 'original-b']
    operator = SourceWhiteningOperator(np.array([[1., 2., -.5, .25], [.2, .1, .3, .7]]),
        tuple(row['source_point_ids']), 'original-joint-readout', 'original-material-normal')
    frame = dict(index=2, native_pts=256, source_point_count=2, rows=[row],
        whitening_groups=[dict(point_indices=[0, 1], operator=operator)])
    w = fit.FrozenWhitening(frame)
    ref = reference(__file__)
    class Camera:
        def __init__(self, original):
            assert original.binding_reference == ref
        def __getstate__(self):
            return dict(pts=state.pts.copy(), source_clock=state.source_clock)
    monkeypatch.setattr(fit, '_CameraState', Camera)
    owner = SimpleNamespace(start=q, state=state, world=lambda *a: ({'line':world}, None),
        bundle=SimpleNamespace(frames=[frame], whitening=[w], ledgers=[{'native_pts':256}],
            state=SimpleNamespace(binding_reference=ref)),
        formal_offsets=np.array([0, w.count]), scales=w.scales, quadratic=np.zeros(w.count, bool),
        model_reference=ref, association_reference=ref, source_producers=[ref],
        checkpoint_arrays=lambda: {'native_pts':state.pts}, checkpoint_identity=lambda: {'original':True})
    output = tmp_path / 'selected'
    export_source_frames(owner, output_directory=output, frame_indices=(2,))
    payload, receipt = load_source_frames(output / 'frames.json')
    restored = payload['frames'][0]
    expected = row_timed_raw(state, q, frame, {'line':world})
    actual = row_timed_raw(state, payload['q'], restored['frame'], restored['worlds'])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(restored['raw_pixel_residual'], expected)
    np.testing.assert_array_equal(fit.FrozenWhitening(restored['frame']).apply(actual), w.apply(expected))
    np.testing.assert_array_equal(restored['whitened_residual'], w.apply(expected))
    assert restored['frame']['rows'][0]['native_line_queries']['query_ids'] == ['a', 'b']
    assert restored['frame']['rows'][0]['source_point_ids'] == ['original-a', 'original-b']
    assert receipt['pickle_deserialization'] is False
    assert len(actual) == 4 and len(restored['whitened_residual']) == 2
