"""Synthetic root priors, checkpoint replay and camera proxies; no real fit."""
import io
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking import fullbody_fit as fullbody
from calibration.program_tracking.body_tracking.action_integration import ClipReference
from calibration.program_tracking.body_tracking.camera_bridge import project_world, rays
from calibration.program_tracking.body_tracking.fullbody_state import matrices
from calibration.program_tracking.body_tracking.ice_soft import IceSoftPolicy
from calibration.program_tracking.body_tracking.root_trajectory import (
    image_height_given_xy, proxy_root_samples, smooth_root_samples,
)
from calibration.program_tracking.body_tracking.skate_integration import (
    ContinuousPoseWindow, ContinuousRootWindow, JointContactProblem, base_layer_packet, checkpoint_state_kind,
    restore_checkpoint_layers, state_from_packet, window_factory,
)


def curved_reference(root):
    times = np.arange(13)/10
    initial = dict(P=np.array([.3, -.2, .9])+times[:, None]*[.2, .1, -.1]
        + times[:, None]**2*[.4, -.3, .2],
        Q=matrices(times[:, None]*[.1, -.2, .3]),
        local_R=matrices(np.broadcast_to(times[:, None, None]*np.array([.03, -.02, .01]), (13, 21, 3))))
    p = dict(pts=np.arange(13)*1280, fit_time_seconds=times)
    binding = SimpleNamespace(x=np.zeros(17), collar_constraints_enabled=False)
    reference = ClipReference(SimpleNamespace(binding=binding), root, p, initial)
    reference.footwear_observations = {}
    reference.fixed_checkpoint_identity = {}

    def world(state, delta=None):
        delta = np.zeros((len(state['P']), 69)) if delta is None else delta
        P = state['P']+delta[:, :3]
        Q = state['Q']@matrices(delta[:, 3:6])
        local = state['local_R']@matrices(delta[:, 6:].reshape(-1, 21, 3))
        # A small articulated world consumer makes pose preservation observable.
        relative = np.einsum('njab,b->nja', local, [.1, .2, .3])
        return dict(P=P, Q=Q, local_R=local, world=P[:, None]+np.einsum('nab,njb->nja', Q, relative))

    reference.world = world
    return reference


def test_actual_root_acceleration_offset_cancels_and_has_correct_jacobian(tmp_path, monkeypatch):
    reference = curved_reference(tmp_path)
    ledger = dict(pts=reference.p['pts'], use=np.zeros((13, 17), bool))
    monkeypatch.setattr(fullbody, 'load', lambda path: ledger)
    monkeypatch.setattr(fullbody, 'get_body_execution', lambda ref: SimpleNamespace())
    projector = SimpleNamespace(p=dict(rig=np.zeros(38)))
    problem = fullbody.FullBodyProblem(reference, projector, 'in-memory-ledger', (0., 1.2), 'full',
        state_factory=ContinuousRootWindow)
    matrix = dict(problem.prior_matrices)['actual_root_acceleration']
    zero = np.zeros(problem.size)
    initial = dict(problem.prior_rows(zero))['actual_root_acceleration']
    expected = np.broadcast_to(np.array([.8, -.6, .4])/(15*np.sqrt(2)), (13, 3)).ravel()
    np.testing.assert_allclose(initial, expected, atol=1e-12)
    assert np.linalg.norm(initial) > 0.
    assert 'correction_acceleration' not in dict(problem.prior_rows(zero))
    controls = np.linalg.lstsq(problem.state.basis(reference.times), -reference.at(reference.times)['P'], rcond=None)[0].ravel()
    np.testing.assert_allclose(dict(problem.prior_rows(controls))['actual_root_acceleration'], 0., atol=1e-12)
    direction = np.random.default_rng(703).normal(size=problem.size)
    step = 1e-6
    plus = dict(problem.prior_rows(controls+step*direction))['actual_root_acceleration']
    minus = dict(problem.prior_rows(controls-step*direction))['actual_root_acceleration']
    np.testing.assert_allclose((plus-minus)/(2*step), matrix@direction, rtol=1e-9, atol=1e-10)


def test_continuous_root_checkpoint_roundtrip_preserves_complete_world_and_derivative(tmp_path):
    reference = curved_reference(tmp_path)
    state = ContinuousRootWindow(reference, (0., 1.2))
    rng = np.random.default_rng(141)
    previous, current = rng.normal(0, .02, (2, state.size))
    reference.new_refinements.append((state, previous))
    problem = JointContactProblem.__new__(JointContactProblem)
    problem.reference = reference
    problem.saved_base_layers = base_layer_packet(reference)
    problem.body = SimpleNamespace(state=state)
    problem.physics_times = reference.times
    problem.binding = reference.binding
    problem.al = SimpleNamespace(multiplier=np.zeros(4), penalty=10.)
    problem.state_kind = 'continuous_root'
    problem.ice_policy = IceSoftPolicy(.004, .03, .5, .006, .04, .75)
    payload = problem.checkpoint_fields(np.r_[reference.binding.x, current])
    stream = io.BytesIO()
    np.savez_compressed(stream, **payload)
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as packet:
        saved = {key:packet[key].copy() for key in packet.files}
    restored = curved_reference(tmp_path)
    assert checkpoint_state_kind(saved) == 'continuous_root'
    assert str(saved['base_layer_0_state_kind']) == 'continuous_root'
    replay_state = state_from_packet(restored, saved)
    np.testing.assert_array_equal(replay_state.active, [0, 1, 2])
    assert replay_state.d == 3 and replay_state.size == replay_state.m*3
    t = np.linspace(0, 1.2, 37)
    expected = state.state_at(current, t)
    restore_checkpoint_layers(restored, saved, include_primary=True)
    actual = restored.world(restored.at(t))
    for key in ('P', 'Q', 'local_R', 'world'):
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-14)
    np.testing.assert_allclose(restored.root_derivative(t, 2),
        reference.root_derivative(t, 2)+state.basis(t, 2)@current.reshape(state.m, 3), atol=1e-13)
    np.testing.assert_array_equal(state.increments(current, t)[:, 3:], 0.)
    assert all(isinstance(layer, ContinuousRootWindow) for layer, _ in restored.new_refinements)


def test_root_base_and_pose_primary_roundtrip_fixes_root_and_preserves_changed_rotations(tmp_path):
    reference = curved_reference(tmp_path)
    root = ContinuousRootWindow(reference, (0., 1.2))
    rng = np.random.default_rng(402)
    reference.new_refinements.append((root, rng.normal(0., .02, root.size)))
    assert window_factory('continuous_pose') is ContinuousPoseWindow
    pose = window_factory('continuous_pose')(reference, (0., 1.2))
    np.testing.assert_array_equal(pose.active, np.arange(3, 69))
    assert pose.d == 66 and pose.size == pose.m*66
    controls = rng.normal(0., .03, pose.size)
    t = np.linspace(0., 1.2, 37)
    fixed = reference.at(t)
    expected = pose.state_at(controls, t)
    np.testing.assert_array_equal(pose.increments(controls, t)[:, :3], 0.)
    np.testing.assert_array_equal(expected['P'], fixed['P'])
    assert not np.allclose(expected['Q'], fixed['Q'])
    assert not np.allclose(expected['local_R'], fixed['local_R'])
    payload = base_layer_packet(reference)
    payload.update(x=np.r_[reference.binding.x, controls], knots=pose.knots,
        active=pose.active, seconds=np.array(pose.seconds), state_kind=np.array('continuous_pose'))
    stream = io.BytesIO()
    np.savez_compressed(stream, **payload)
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as packet:
        saved = {key:packet[key].copy() for key in packet.files}
    assert str(saved['base_layer_0_state_kind']) == 'continuous_root'
    assert checkpoint_state_kind(saved) == 'continuous_pose'
    restored = curved_reference(tmp_path)
    assert isinstance(state_from_packet(restored, saved), ContinuousPoseWindow)
    restore_checkpoint_layers(restored, saved, include_primary=True)
    actual = restored.world(restored.at(t))
    np.testing.assert_array_equal(actual['P'], fixed['P'])
    for key in ('P', 'Q', 'local_R', 'world'):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0., atol=1e-14)
    np.testing.assert_allclose(restored.root_derivative(t, 2), reference.root_derivative(t, 2), rtol=0., atol=1e-14)
    # The same pose primary must retain its type when promoted to a base layer.
    layers = base_layer_packet(restored)
    assert str(layers['base_layer_0_state_kind']) == 'continuous_root'
    assert str(layers['base_layer_1_state_kind']) == 'continuous_pose'
    np.testing.assert_array_equal(layers['base_layer_1_active'], np.arange(3, 69))
    replay = curved_reference(tmp_path)
    restore_checkpoint_layers(replay, layers)
    np.testing.assert_allclose(replay.world(replay.at(t))['world'], expected['world'], rtol=0., atol=1e-14)


def camera_case():
    scale = 2.3
    camera = np.zeros(25)
    camera[:3] = [.2, -.4, 2.5]
    camera[3:6] = Rotation.from_matrix(np.diag([1., -1., -1.])).as_rotvec()
    camera[6:8] = np.log([900., 1000.])
    camera[8:11] = [960., 540., .05]
    camera[11:15] = [.01, -.001, .0002, -.0001]
    cameras = np.tile(camera, (4, 1))
    root = np.array([[.5+i*.2, 1.2+i*.1, .9] for i in range(4)])
    relative = np.random.default_rng(928).normal(0., .2, (4, 17, 3))
    relative[:, 15] = [.08, .02, -.75]
    relative[:, 16] = [-.08, -.02, -.70]
    uv = []
    for cam, centre, body in zip(cameras, root, relative):
        pixels, valid = project_world((centre+body)/scale, cam)
        assert valid.all()
        uv.append(pixels)
    return cameras, np.asarray(uv), np.ones((4, 17), bool), relative, root, scale


def test_ankle_proxy_keeps_anatomical_height_and_excludes_unknown_flight_and_missing_pixels():
    cameras, uv, qualified, relative, root, scale = camera_case()
    modes = np.array(['glide', 'unknown', 'airborne', 'glide'])
    candidates = np.isin(modes, ['glide', 'pick'])
    qualified[3, 15:] = False
    runner_z = np.tile([-.9, -.8], (4, 1))
    samples, foot = proxy_root_samples(cameras, uv, qualified, relative, runner_z,
        candidates, np.array([0, -1, 1, 0]), scale)
    np.testing.assert_allclose(samples[0], root[0], atol=1e-10)
    np.testing.assert_array_equal(foot, [0, -1, -1, -1])
    assert np.isnan(samples[1:]).all()
    # The ankle is 0.15 m above the supporting runner, never a blade pixel at Z=0.
    assert root[0, 2]+relative[0, 15, 2] == pytest.approx(.15)
    ray, valid = rays(uv[0, 15][None], cameras[0])
    assert valid[0]
    centre = cameras[0, :3]*scale
    wrong_ground_intersection = centre-centre[2]/ray[0, 2]*ray[0]-relative[0, 15]
    assert np.linalg.norm(samples[0]-wrong_ground_intersection) > .1


def test_image_height_given_known_xy_recovers_synthetic_root_and_keeps_unavailable():
    cameras, uv, qualified, relative, root, scale = camera_case()
    qualified[3] = False
    qualified[3, :3] = True
    heights = image_height_given_xy(cameras, uv, qualified, relative, root[:, :2], scale)
    np.testing.assert_allclose(heights[:3], root[:3, 2], atol=1e-9)
    assert np.isnan(heights[3])


def test_smoothing_preserves_linear_trajectory_across_missing_samples_and_requires_evidence():
    times = np.linspace(0., 1., 11)
    expected = np.array([.2, -.1, .9])+times[:, None]*[.3, .2, -.1]
    samples = expected.copy()
    samples[[2, 5, 8]] = np.nan
    np.testing.assert_allclose(smooth_root_samples(times, samples), expected, atol=1e-12)
    samples[3:] = np.nan
    with pytest.raises(ValueError, match='four'):
        smooth_root_samples(times, samples)


def test_budget_reservation_stops_before_a_known_expensive_jacobian():
    from calibration.program_tracking.body_tracking.skate_run_policy import Monitor, RunPolicy
    monitor=Monitor(RunPolicy(wall_seconds=80),now=100.)
    assert not monitor.reserve_evaluation(now=101.,estimated_seconds=30.)['stop']
    status=monitor.reserve_evaluation(now=160.,estimated_seconds=30.)
    assert status['stop'] and not status['quality_failed']
    assert status['reasons'][0]['code']=='evaluation_budget_reserve'
    assert status['reasons'][0]['remaining_seconds']==20.
    assert monitor.record()['accepted']==0
