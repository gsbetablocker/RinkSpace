"""Source-clock and articulated-motion contracts without programme assets."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.opening_motion import (
    MotionReference, MotionWindow, interpolate_motion, surface_contact_vectors, START, END,
)
from calibration.program_tracking.body_tracking.opening_repair import OpeningWindow


def rotations(angle, axis=2):
    angle = np.atleast_1d(angle)
    vectors = np.zeros((len(angle), 3))
    vectors[:, axis] = angle
    return Rotation.from_rotvec(vectors).as_matrix()


def packet(dynamic_leg=True):
    # Genuine source stamps deliberately have a nonzero official origin.
    stamps = np.array([START-.10, START, 22.80, 23.60, 24.44, 24.70, END+.10])
    local = np.broadcast_to(np.eye(3), (len(stamps), 21, 3, 3)).copy()
    local[:, 20] = rotations(.35*(stamps-START), 0)
    if dynamic_leg:
        local[:, 3] = rotations(.40*(stamps-START), 0)
    return dict(raw_official_pts=np.rint(stamps*12800).astype(np.int64),
                raw_legacy_pts=np.arange(len(stamps)),
                source_time_base=np.array([1, 12800]), local_R=local,
                Q_world=rotations(.20*(stamps-START)))


def reference(dynamic_leg=True):
    def parent_at(t):
        t = np.atleast_1d(t)
        return dict(P=np.c_[1+.07*t, 2-.02*t, .9+.01*t],
                    Q=rotations(.10*t),
                    local_R=np.broadcast_to(np.eye(3), (len(t), 21, 3, 3)).copy(),
                    times=t)

    def world(state, delta=None):
        # A minimal hinged support proxy: flexing the knee raises the blade
        # relative to the pelvis. It tests the height/pose connection, not SMPL.
        flex = Rotation.from_matrix(state['local_R'][:, 3]).as_rotvec()[:, 0]
        gear = np.zeros((len(flex), 2, 48, 3))
        gear[..., 2] = state['P'][:, None, None, 2]-np.cos(flex)[:, None, None]
        return dict(gear=gear)

    parent = SimpleNamespace(at=parent_at, world=world,
                             p=dict(pts=np.array([round(START*12800)])))
    preparation = np.broadcast_to(np.eye(3), (21, 3, 3)).copy()
    preparation[3] = rotations([.25], 0)[0]
    old = SimpleNamespace(parent=parent, preparation=dict(local_R=preparation), at=parent_at)
    return MotionReference(old, None, packet(dynamic_leg))


def test_wrist_tracks_official_source_time_including_interpolated_exposures():
    ref = reference()
    source_seconds = np.array([START, 22.46, 23.26, 24.30])
    local = ref.at(source_seconds-START)['local_R']
    expected = rotations(.35*(source_seconds-START), 0)
    np.testing.assert_allclose(local[:, 20], expected, atol=2e-14)
    assert Rotation.from_matrix(local[0, 20].T@local[-1, 20]).magnitude() > .5


def test_preparation_does_not_replace_dynamic_lower_limb_changes():
    dynamic, static = reference(True), reference(False)
    source_seconds = np.array([START, 22.80, 23.60, 24.44])
    moving = dynamic.at(source_seconds-START)['local_R'][:, 3]
    preparation_only = static.at(source_seconds-START)['local_R'][:, 3]
    # The same initial preparation is applied to both trajectories. Its gradual
    # release must not erase the independently supplied raw knee excursion.
    relative = preparation_only.transpose(0, 2, 1)@moving
    expected_change = rotations(.40*(source_seconds-START), 0)
    np.testing.assert_allclose(relative, expected_change, atol=2e-14)
    np.testing.assert_allclose(moving[0], dynamic.preparation['local_R'][3], atol=2e-14)


def test_flexing_legs_changes_support_seed_height_without_freezing_translation():
    ref = reference()
    times = np.array([0., .68, 1.48])
    state = ref.at(times)
    # The inferred support seed may follow articulation rather than hold the
    # pelvis at a single height; existing lateral motion remains available.
    assert np.ptp(state['P'][:, 2]) > .03
    np.testing.assert_allclose(state['P'][:, :2], ref.old.at(times)['P'][:, :2])
    support = ref.world(state)['gear'][..., 2]
    np.testing.assert_allclose(support, 0., atol=2e-14)


@pytest.mark.parametrize('side', [-1, 1])
def test_raw_motion_never_extrapolates_even_by_one_source_tick(side):
    raw = packet()
    edge = raw['raw_official_pts'][0 if side < 0 else -1]/12800.
    with pytest.raises(ValueError, match='bracket'):
        interpolate_motion(raw, [edge+side/12800.])


def test_raw_motion_includes_exact_bracket_endpoints():
    raw = packet()
    Q, local = interpolate_motion(raw, raw['raw_official_pts'][[0, -1]]/12800.)
    np.testing.assert_allclose(Q, raw['Q_world'][[0, -1]], atol=2e-14)
    np.testing.assert_allclose(local, raw['local_R'][[0, -1]], atol=2e-14)


def test_first_source_frame_can_move_and_after_25_is_exact_parent():
    ref = reference()
    state = MotionWindow(ref, (START, END), 'full')
    initial = ref.at([0.])['P'].copy()
    controls = np.zeros((state.m, state.d))
    controls[:, np.flatnonzero(state.active == 0)[0]] = .12
    ref.refinements.append((state, controls.ravel()))
    np.testing.assert_allclose(ref.at([0.])['P'], initial+[[.12, 0., 0.]], atol=2e-14)
    # Far-future queries must use the parent directly, without asking the raw
    # packet to extrapolate beyond its final genuine exposure.
    times = np.array([END-START, END-START+.02, 10.])
    for key in ('P', 'Q', 'local_R'):
        np.testing.assert_array_equal(ref.at(times)[key], ref.parent.at(times)[key])


def test_parent_attachment_has_vanishing_value_velocity_and_acceleration():
    ref = reference()
    state = MotionWindow(ref, (START, END), 'full')
    ref.refinements.append((state, np.full(state.size, .015)))
    end = END-START
    for order in (0, 1, 2):
        np.testing.assert_allclose(state.basis([end], order), 0., atol=1e-12)
    # A C2 attachment's residual to a smooth parent falls cubically. Check the
    # complete articulated reference, including its changing raw orientations.
    errors = []
    for h in (.002, .001):
        current, parent = ref.at([end-h]), ref.parent.at([end-h])
        errors.append(sum(np.linalg.norm(current[k]-parent[k]) for k in ('P', 'Q', 'local_R')))
    assert 0 < errors[1] < errors[0]/6

def test_contact_vectors_follow_rigid_motion_without_world_point_pinning():
    palms = np.array([[[.10, .02, .03], [.14, -.01, .02]],
                      [[-.06, .08, .04], [.02, .10, -.03]]])
    waist = np.array([[[.08, 0., 0.], [.13, .01, .01], [.18, -.01, .02]],
                      [[-.07, .06, .01], [-.02, .09, -.01], [.03, .11, -.02]]])
    R = Rotation.from_rotvec([[.7, -.2, .3], [-.4, .8, .1]]).as_matrix()
    shift = np.array([[4., -3., 2.], [-2., 5., -.7]])
    before = surface_contact_vectors(palms, waist)
    after = surface_contact_vectors(np.einsum('nij,nkj->nki', R, palms)+shift[:, None],
                                    np.einsum('nij,nkj->nki', R, waist)+shift[:, None])
    np.testing.assert_allclose(after, np.einsum('nij,nkj->nki', R, before), atol=2e-14)
    np.testing.assert_allclose(np.sum(after**2, axis=(1, 2)),
                               np.sum(before**2, axis=(1, 2)), atol=2e-14)


def test_colocated_skin_and_palm_samples_have_zero_contact_residual():
    positions = np.array([[.2, -.3, .4], [7., 2., -1.]])
    palms = np.repeat(positions[:, None], 3, axis=1)
    waist = positions[:, None]
    np.testing.assert_array_equal(surface_contact_vectors(palms, waist), np.zeros_like(palms))


def test_contact_may_slide_to_a_different_nearest_surface_sample():
    # Two separated samples make the nearest location unambiguous. The palm
    # visits each sample with the same 2cm normal gap; no waist ID is pinned.
    waist = np.broadcast_to([[0., 0., 0.], [1., 0., 0.]], (2, 2, 3)).copy()
    palms = np.array([[[0., .02, 0.]], [[1., .02, 0.]]])
    residual = surface_contact_vectors(palms, waist)
    np.testing.assert_allclose(residual, np.broadcast_to([0., .02, 0.], palms.shape), atol=1e-14)
    # Reordering skin vertices cannot change which physical location is used.
    np.testing.assert_allclose(surface_contact_vectors(palms, waist[:, ::-1]), residual, atol=1e-14)
    moving_gap = np.array([[[0., .01, 0.]], [[1., .04, 0.]]])
    np.testing.assert_allclose(np.linalg.norm(surface_contact_vectors(moving_gap, waist), axis=-1),
                               [[.01], [.04]], atol=1e-14)


def test_motion_window_adds_only_three_right_wrist_columns():
    ref = reference()
    previous = OpeningWindow(ref, (START, END), 'full')
    state = MotionWindow(ref, (START, END), 'full')
    assert len(np.unique(state.active)) == state.d
    assert set(state.active)-set(previous.active) == {66, 67, 68}
    assert set(previous.active).issubset(state.active)
    assert state.size == state.m*state.d
    np.testing.assert_array_equal(state.knots, previous.knots)


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_right_wrist_has_independent_free_initial_rotation_and_c2_parent_return(axis):
    ref = reference()
    state = MotionWindow(ref, (START, END), 'full')
    uncorrected = ref.at([0.])
    controls = np.zeros((state.m, state.d))
    controls[:, np.flatnonzero(state.active == 66+axis)[0]] = .20
    ref.refinements.append((state, controls.ravel()))
    initial = ref.at([0.])
    expected = rotations([.20], axis)[0]
    np.testing.assert_allclose(initial['local_R'][0, 20],
                               uncorrected['local_R'][0, 20]@expected, atol=2e-14)
    np.testing.assert_array_equal(initial['local_R'][:, :20], uncorrected['local_R'][:, :20])
    np.testing.assert_array_equal(initial['P'], uncorrected['P'])
    end = END-START
    for derivative in (0, 1, 2):
        np.testing.assert_allclose(state.basis([end], derivative), 0., atol=1e-12)
    times = np.array([end, end+.02, 10.])
    for key in ('P', 'Q', 'local_R'):
        np.testing.assert_array_equal(ref.at(times)[key], ref.parent.at(times)[key])