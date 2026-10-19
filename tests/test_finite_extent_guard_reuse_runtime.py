"""Small original-validator fixtures; no programme or motion graph is loaded."""
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.solver_runtime import install
install()
from z190_fixed_lever_xyz_source_bundle import FixedLeverXYZSourceBundle
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle
from calibration.solver_runtime import finite_extent_guard_reuse_runtime as runtime


class TinyGeometry:
    pass


class TinyBundle:
    cache = FixedLeverXYZSourceBundle.cache
    _check_shared = FixedLeverXYZSourceBundle._check_shared
    validate_finite_extents = MovingPivotSourceBundle.validate_finite_extents

    def __init__(self):
        self.calibration = SimpleNamespace(lever_xyz=(0., 0., 0.),
            principal_x_fraction=(.5,), principal_y_fraction=(.5,), k1=(0.,))
        self.a = np.array([[960., 540.]])
        self.ia = np.array([0])
        self.finite_planes = np.array([[0., 0., 1., 0.]])
        self.finite_records = [dict(global_bundle_index=0, surface_kind='ice_plane',
                                    original_source_clock='1/15360', original_source_pts=123)]
        self.scene = dict(ice_boundary=dict(points_xyz=[[-1000., -1000., 0.],
            [1000., -1000., 0.], [1000., 1000., 0.], [-1000., 1000., 0.]]),
            board_surface=dict(top_edge_xyz=[[0., 0., 200.]]))
        self.profile_weights = None
        self.robust_phase = False


class TinyBlock:
    def __init__(self, name):
        self.name = name
        self.bundle = TinyBundle()
        self.geometry = TinyGeometry()
        self.geometry_parameter_indices = np.array([8, 9])
        self.optics_chart = SimpleNamespace(coordinate_chart=SimpleNamespace(core_q=np.asarray))
        self._course_current_geometry_key = None
        self.local_calls = 0
        self.camera_calls = 0

    def local(self, q):
        self.local_calls += 1
        self._course_current_geometry_key = np.asarray(q[8:]).tobytes()
        return np.asarray(q[:8])

    def camera_local(self, q):
        self.camera_calls += 1
        return np.asarray(q[:8])


@pytest.fixture
def tiny(monkeypatch):
    runtime._BUNDLES.clear()
    calls = []
    original = MovingPivotSourceBundle.validate_finite_extents

    def counted(bundle, cache):
        calls.append(bundle)
        return original(bundle, cache)

    monkeypatch.setattr(TinyBundle, 'cache', runtime._wrap_cache(FixedLeverXYZSourceBundle.cache))
    monkeypatch.setattr(TinyBundle, 'validate_finite_extents', runtime._wrap_validator(counted))
    monkeypatch.setattr(TinyBlock, 'local', runtime._wrap_local(TinyBlock.local))
    q = np.array([0., 0., 1., 0., np.pi, 0., 0., np.log(1000.), 0., 200.])
    blocks = tuple(TinyBlock(str(i)) for i in range(6))
    actors = SimpleNamespace(blocks=blocks, delegate=SimpleNamespace(_busy=False,
        closed=False, _actors=[]), start_calls=0, last_stats=[])

    def ordered_original(manager, value):
        manager.start_calls += 1
        manager.delegate._actors = list(blocks)
        for block in blocks:
            local = block.local(value)
            f = local[7::4]
            if np.any((f < np.log(100.)) | (f > np.log(50000.))):
                raise ValueError('Physical focal domain')
            block.bundle.validate_finite_extents(block.bundle.cache(local, False))

    monkeypatch.setattr(runtime, '_ORIGINAL_ORDERED_GUARD', ordered_original)
    monkeypatch.setattr(runtime, '_REPORT_DIRECTORY', None)

    def prime():
        for block in blocks:
            local = block.local(q)
            block.bundle.validate_finite_extents(block.bundle.cache(local, True))

    return q, blocks, actors, calls, prime


def test_original_six_passes_reused_across_profile_change_without_actor_start(tiny):
    q, blocks, actors, calls, prime = tiny
    prime()
    assert len(calls) == 6
    for block in blocks:
        block.bundle.profile_weights = np.array([[.25, .5]])
        block.bundle.robust_phase = True
    assert runtime._ordered_guard_with_reuse(actors, q) is None
    assert actors.start_calls == 0 and actors.delegate._actors == []
    assert len(calls) == 6
    assert [runtime._state(b.bundle)['hits'] for b in blocks] == [1]*6
    assert [b.local_calls for b in blocks] == [2]*6
    # Exact PASS does not swallow the original source/truth suffix.
    suffix = []
    def full_guard():
        runtime._ordered_guard_with_reuse(actors, q)
        suffix.append('truth')
        raise ValueError('original truth suffix')
    with pytest.raises(ValueError, match='original truth suffix'):
        full_guard()
    assert suffix == ['truth'] and len(calls) == 6


@pytest.mark.parametrize('changed', ['local', 'world', 'calibration', 'source_metadata', 'source_uv', 'scene'])
def test_any_complete_input_or_source_fact_change_uses_original_route(tiny, changed):
    q, blocks, actors, calls, prime = tiny
    prime()
    trial = q.copy()
    if changed == 'local':
        trial[7] += .01
    elif changed == 'world':
        trial[9] += 1.
    elif changed == 'calibration':
        blocks[0].bundle.calibration.k1 = (.0001,)
    elif changed == 'source_metadata':
        blocks[0].bundle.finite_records[0]['original_source_pts'] += 1
    elif changed == 'source_uv':
        blocks[0].bundle.a[0, 0] += 1.
    else:
        blocks[0].bundle.scene['board_surface']['top_edge_xyz'][0][2] += 1.
    runtime._ordered_guard_with_reuse(actors, trial)
    assert actors.start_calls == 1
    assert len(calls) > 6


def test_original_first_failure_and_repeated_failure_not_cached(tiny):
    q, blocks, actors, calls, prime = tiny
    prime()
    bad = q.copy()
    bad[4] = 0.  # Original source ray points upwards: negative finite depth.
    for expected_calls in (7, 8):
        with pytest.raises(ValueError, match='Qualified finite source negative depth'):
            runtime._ordered_guard_with_reuse(actors, bad)
        assert len(calls) == expected_calls
    assert actors.start_calls == 2
    assert runtime._state(blocks[0].bundle)['hits'] == 0


def test_mutated_original_cache_values_do_not_reuse_stale_local_pass(tiny):
    q, blocks, actors, calls, prime = tiny
    prime()
    bundle = blocks[0].bundle
    cache = bundle.cache(q[:8], False)
    cache[3][0, 0] = 3000.  # Source intersection is now outside original polygon.
    with pytest.raises(ValueError, match='Qualified ice point outside hard ice boundary'):
        bundle.validate_finite_extents(cache)
    assert len(calls) == 7


def test_certificates_are_external_to_bundle_pickle_and_unknown_cache_falls_back(tiny):
    q, blocks, actors, calls, prime = tiny
    before = [pickle.dumps(b.bundle, protocol=5) for b in blocks]
    prime()
    assert [pickle.dumps(b.bundle, protocol=5) for b in blocks] == before
    bundle = blocks[0].bundle
    cache = bundle.cache(q[:8], False)
    bundle.validate_finite_extents(tuple(value.copy() for value in cache))
    assert len(calls) == 7


def test_original_focal_failure_not_hidden_by_certificate(tiny):
    q, blocks, actors, calls, prime = tiny
    prime()
    bad = q.copy()
    bad[7] = np.log(50.)
    with pytest.raises(ValueError, match='Physical focal domain'):
        runtime._ordered_guard_with_reuse(actors, bad)
    assert len(calls) == 6 and actors.start_calls == 1
