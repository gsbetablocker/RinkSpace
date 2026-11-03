"""Right-arm-only support and state-preservation contracts without real assets."""
import numpy as np
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.opening_hand_polish import (
    HandWindow, RIGHT_ARM_JOINTS, START, END, HAND_END,
)
from calibration.program_tracking.body_tracking.opening_motion import MotionWindow
from test_opening_timing import reference as timing_reference


def reference():
    # Include a nonzero saved timing correction, so the hand overlay must
    # preserve a fitted trajectory rather than just an unmodified raw pose.
    ref = timing_reference()
    state = MotionWindow(ref, (START, END), 'full')
    controls = np.random.default_rng(2340).normal(scale=.007, size=state.size)
    ref.refinements.append((state, controls))
    return ref


def test_hand_window_contains_exactly_the_nine_right_arm_angles():
    ref = reference()
    state = HandWindow(ref)
    expected = [6+(joint-1)*3+axis for joint in (17, 19, 21) for axis in range(3)]
    np.testing.assert_array_equal(state.active, expected)
    assert RIGHT_ARM_JOINTS == (17, 19, 21)
    assert state.d == 9 and state.size == state.m*9
    assert state.seconds == (START, HAND_END)


def test_all_nine_initial_angles_are_free_on_top_of_the_fitted_timing_state():
    ref = reference()
    state = HandWindow(ref)
    before = ref.at([0.])
    initial_angles = np.array([[.08, -.03, .04], [-.02, .07, .03], [.05, .02, -.06]])
    controls = np.tile(initial_angles.ravel(), (state.m, 1))
    ref.refinements.append((state, controls.ravel()))
    after = ref.at([0.])
    corrections = Rotation.from_rotvec(initial_angles).as_matrix()
    for index, joint in enumerate(RIGHT_ARM_JOINTS):
        np.testing.assert_allclose(after['local_R'][0, joint-1],
            before['local_R'][0, joint-1]@corrections[index], atol=2e-14)


def test_hand_overlay_keeps_root_and_all_other_local_states_exact():
    ref = reference()
    state = HandWindow(ref)
    times = np.array([0., .38, .78, 1.08, HAND_END-START, 24.0-START, END-START, 10.])
    before = ref.at(times)
    controls = np.random.default_rng(17).normal(scale=.12, size=state.size)
    ref.refinements.append((state, controls))
    after = ref.at(times)
    for key in ('P', 'Q'):
        np.testing.assert_array_equal(after[key], before[key])
    frozen = np.ones(21, bool)
    frozen[np.asarray(RIGHT_ARM_JOINTS)-1] = False
    np.testing.assert_array_equal(after['local_R'][:, frozen], before['local_R'][:, frozen])
    assert np.linalg.norm(after['local_R'][:4]-before['local_R'][:4]) > .01
    outside = times >= HAND_END-START
    for key in ('P', 'Q', 'local_R'):
        np.testing.assert_array_equal(after[key][outside], before[key][outside])


def test_hand_overlay_has_a_c2_return_to_timing_at_2340():
    ref = reference()
    state = HandWindow(ref)
    end = HAND_END-START
    for derivative in (0, 1, 2):
        np.testing.assert_allclose(state.basis([end], derivative), 0., atol=1e-12)
    times = np.array([end-.002, end-.001])
    before = ref.at(times)['local_R']
    ref.refinements.append((state, np.full(state.size, .03)))
    after = ref.at(times)['local_R']
    errors = np.linalg.norm((after-before).reshape(len(times), -1), axis=1)
    assert 0 < errors[1] < errors[0]/6