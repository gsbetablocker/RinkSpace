"""Source-timed quiet-body contracts, independent of real assets or fitting."""
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.opening_motion import MotionProblem, MotionWindow
from calibration.program_tracking.body_tracking.opening_timing import (
    TimingReference, TimingProblem, quiet_state_prior, QUIET_END, QUIET_BODY_JOINTS,
    START, END,
)


def rotations(angle, axis=2):
    vectors = np.zeros((len(np.atleast_1d(angle)), 3))
    vectors[:, axis] = np.atleast_1d(angle)
    return Rotation.from_rotvec(vectors).as_matrix()


def reference():
    seconds = np.array([START-.10, START, 22.50, QUIET_END, 23.20, 24.0, 24.70, END+.10])
    local = np.empty((len(seconds), 21, 3, 3))
    for joint in range(1, 22):
        local[:, joint-1] = rotations(.02*joint+(.1+.01*joint)*(seconds-START), joint % 3)
    packet = dict(raw_official_pts=np.rint(seconds*12800).astype(np.int64),
                  raw_legacy_pts=np.arange(len(seconds)),
                  source_time_base=np.array([1, 12800]), local_R=local,
                  Q_world=rotations(.2+.3*(seconds-START)))

    def parent_at(t):
        t = np.atleast_1d(t)
        return dict(P=np.c_[1+.3*t, 2-.07*t, .9+.01*t], Q=rotations(.1*t),
                    local_R=np.broadcast_to(np.eye(3), (len(t), 21, 3, 3)).copy(), times=t)

    def world(state, delta=None):
        flex = np.linalg.norm(Rotation.from_matrix(state['local_R'][:, 3]).as_rotvec(), axis=1)
        gear = np.zeros((len(flex), 2, 48, 3))
        gear[..., 2] = state['P'][:, None, None, 2]-np.cos(flex)[:, None, None]
        return dict(gear=gear)

    source = np.rint(np.array([START, 22.50, QUIET_END, 23.10, 24.0, END])*12800).astype(np.int64)
    parent = SimpleNamespace(at=parent_at, world=world, p=dict(pts=source), times=source/12800.-START)
    preparation = local[1].copy()
    for joint in (1, 2, 4, 5, 7, 8, 10, 11):
        preparation[joint-1] = preparation[joint-1]@rotations([.15], joint % 3)[0]
    old = SimpleNamespace(parent=parent, preparation=dict(local_R=preparation), at=parent_at)
    return TimingReference(old, None, packet)


def test_body_stays_constant_until_2290_while_arms_head_and_wrist_keep_raw_motion():
    ref = reference()
    source = np.array([START, 22.30, 22.50, QUIET_END])
    state = ref.at(source-START)
    for key in ('P', 'Q'):
        np.testing.assert_allclose(state[key], np.broadcast_to(state[key][0], state[key].shape), atol=2e-14)
    for joint in QUIET_BODY_JOINTS:
        local = state['local_R'][:, joint-1]
        np.testing.assert_allclose(local, np.broadcast_to(local[0], local.shape), atol=2e-14)
    # These joints are deliberately outside the quiet-body observation. Their
    # motion must survive even while the torso and legs remain stationary.
    for joint in (12, 15, 16, 17, 18, 19, 21):
        expected = rotations(.02*joint+(.1+.01*joint)*(source-START), joint % 3)
        np.testing.assert_allclose(state['local_R'][:, joint-1], expected, atol=2e-14)
        assert Rotation.from_matrix(expected[0].T@expected[-1]).magnitude() > .1


def test_lowering_and_translation_are_free_after_the_quiet_boundary():
    ref = reference()
    source = np.array([QUIET_END, 23.20, 24.0])
    state = ref.at(source-START)
    assert np.ptp(state['P'][:, 0]) > .1
    assert np.ptp(state['P'][:, 2]) > .01
    assert Rotation.from_matrix(state['Q'][0].T@state['Q'][-1]).magnitude() > .1
    knee = state['local_R'][:, 3]
    assert Rotation.from_matrix(knee[0].T@knee[-1]).magnitude() > .1


def test_quiet_prior_has_a_free_common_state_but_detects_early_body_drift():
    ref = reference()
    window = MotionWindow(ref, (START, END), 'full')
    times = np.array([0., .2, .4, QUIET_END-START])
    matrix = quiet_state_prior(window, times)
    constant = np.zeros((window.m, window.d))
    for column, active in enumerate(window.active):
        if active < 6:
            constant[:, column] = .05*(active+1)
    # Common root displacement/orientation is fitted freely, not pinned to a
    # world location or to zero. Only differences across source times matter.
    np.testing.assert_allclose(matrix@constant.ravel(), 0., atol=2e-12)
    initial = ref.at(times)
    ref.refinements.append((window, constant.ravel()))
    corrected = ref.at(times)
    np.testing.assert_allclose(corrected['P']-initial['P'],
                               np.broadcast_to([.05, .10, .15], corrected['P'].shape), atol=2e-14)
    greville = np.array([np.mean(window.knots[i+1:i+4]) for i in range(window.m)])
    drift = np.zeros_like(constant)
    drift[:, np.flatnonzero(window.active == 0)[0]] = greville
    assert np.linalg.norm(matrix@drift.ravel()) > 1.


def test_quiet_prior_excludes_arms_hands_neck_and_head_columns():
    ref = reference()
    window = MotionWindow(ref, (START, END), 'full')
    matrix = quiet_state_prior(window, [0., .2, QUIET_END-START])
    body_columns = np.r_[np.arange(6),
        [6+(joint-1)*3+axis for joint in QUIET_BODY_JOINTS for axis in range(3)]]
    selected = np.isin(window.active, body_columns)
    rng = np.random.default_rng(2290)
    controls = np.zeros((window.m, window.d))
    controls[:, ~selected] = rng.normal(size=(window.m, np.count_nonzero(~selected)))
    np.testing.assert_array_equal(matrix@controls.ravel(), np.zeros(matrix.shape[0]))
    actual_active = window.active[np.unique(matrix.indices) % window.d]
    assert set(actual_active).issubset(set(body_columns))
    assert {66, 67, 68}.issubset(set(window.active[~selected]))


def test_quiet_prior_adds_no_rows_for_exposures_after_2290():
    ref = reference()
    window = MotionWindow(ref, (START, END), 'full')
    quiet = np.array([0., .2, QUIET_END-START])
    first = quiet_state_prior(window, quiet)
    appended = quiet_state_prior(window, np.r_[quiet, QUIET_END-START+.02, 1.4, END-START])
    np.testing.assert_array_equal(first.toarray(), appended.toarray())
    later = quiet_state_prior(window, [QUIET_END-START+.02, 1.4, END-START])
    assert later.shape == (0, window.size)
    assert later.nnz == 0


def test_timing_problem_connects_quiet_prior_to_the_actual_exposure_times(monkeypatch):
    ref = reference()

    def minimal_motion_init(self, reference, camera, observations, log=None):
        self.frames = np.arange(len(reference.times))
        self.state = MotionWindow(reference, (START, END), 'full')
        self.mid_times = reference.times.copy()
        self.prior_matrices = []

    monkeypatch.setattr(MotionProblem, '__init__', minimal_motion_init)
    problem = TimingProblem(ref, None, None)
    assert [name for name, _ in problem.prior_matrices] == ['source_confirmed_quiet_body']
    np.testing.assert_array_equal(problem.quiet, ref.p['pts']/12800. <= QUIET_END)
    expected = quiet_state_prior(problem.state, ref.times)
    np.testing.assert_array_equal(problem.prior_matrices[0][1].toarray(), expected.toarray())


def test_timing_reference_preserves_exact_parent_after_25_and_c2_attachment():
    ref = reference()
    window = MotionWindow(ref, (START, END), 'full')
    ref.refinements.append((window, np.full(window.size, .015)))
    end = END-START
    for order in (0, 1, 2):
        np.testing.assert_allclose(window.basis([end], order), 0., atol=1e-12)
    for key in ('P', 'Q', 'local_R'):
        np.testing.assert_array_equal(ref.at([end, end+.02, 10.])[key],
                                       ref.parent.at([end, end+.02, 10.])[key])
    errors = []
    for h in (.002, .001):
        current, parent = ref.at([end-h]), ref.parent.at([end-h])
        errors.append(sum(np.linalg.norm(current[key]-parent[key]) for key in ('P', 'Q', 'local_R')))
    assert 0 < errors[1] < errors[0]/6