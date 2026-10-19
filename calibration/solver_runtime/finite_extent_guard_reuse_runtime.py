"""Reuse successful original finite-extent checks at identical physical inputs.

Only the finite-extent validator is memoized. Original local/material admission,
focal checks and every subsequent source/truth guard remain in their order.
Certificates are process-local and never become part of an actor pickle.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import time
import weakref

import numpy as np


_BUNDLES = weakref.WeakKeyDictionary()
_RECEIPT = None
_REPORT_DIRECTORY = None
_ORIGINAL_ORDERED_GUARD = None


def _array(value):
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.fields is not None:
        raise TypeError('Original finite-extent numeric input required')
    return array.dtype.str, array.shape, array.tobytes()


def _state(bundle):
    state = _BUNDLES.get(bundle)
    if state is None:
        state = dict(world=None, pending=None, certificate=None,
                     original_calls=0, hits=0, key_fallbacks=0)
        _BUNDLES[bundle] = state
    return state


def _world(block, q):
    core = block.optics_chart.coordinate_chart.core_q(q)
    return _array(np.asarray(core)[block.geometry_parameter_indices])


def _facts(bundle):
    """All original validator source inputs; no bundle/material graph pickle.

    The validator reads only finite-record-selected motion rows. Retain full
    record literals as well as every selected index, UV and plane byte. The
    original scene arrays and full calibration literal retain source facts even
    when a change happens to leave the eventual numeric bound unchanged.
    """
    records = bundle.finite_records
    indices = np.asarray([row['global_bundle_index'] for row in records], dtype=int)
    # Selection and serialization failures are not validator authority. Caller
    # falls back to the original validator and its original first exception.
    return (pickle.dumps(records, protocol=5), _array(indices),
            _array(np.asarray(bundle.ia)[indices]),
            _array(np.asarray(bundle.a)[indices]),
            _array(np.asarray(bundle.finite_planes)[indices]),
            None if bundle.scene is None else _array(np.asarray(bundle.scene['ice_boundary']['points_xyz'])),
            None if bundle.scene is None else _array(np.asarray(bundle.scene['board_surface']['top_edge_xyz'])),
            pickle.dumps(bundle.calibration, protocol=5))


def _key(bundle, local, world, cache):
    return world, local, _facts(bundle), tuple(_array(value) for value in cache)


def _wrap_local(original):
    @functools.wraps(original)
    def local(block, q):
        result = original(block, q)
        state = _state(block.bundle)
        try:
            state['world'] = (_world(block, q), weakref.ref(block.geometry))
        except Exception:
            # Original local() already succeeded; bookkeeping cannot change it.
            state['world'] = None
            state['key_fallbacks'] += 1
        return result
    return local


def _wrap_cache(original):
    @functools.wraps(original)
    def cache(bundle, q, jacobian):
        result = original(bundle, q, jacobian)
        state = _state(bundle)
        try:
            # Keep only the immediately produced original cache. The identity
            # check prevents an unrelated/adapted cache tuple using stale q.
            state['pending'] = (result, _array(q), state['world'])
        except Exception:
            state['pending'] = None
            state['key_fallbacks'] += 1
        return result
    return cache


def _wrap_validator(original):
    @functools.wraps(original)
    def validate(bundle, cache):
        state = _state(bundle)
        pending = state['pending']
        key = None
        if pending is not None and pending[0] is cache:
            try:
                world = None if pending[2] is None else pending[2][0]
                key = _key(bundle, pending[1], world, cache)
            except Exception:
                state['key_fallbacks'] += 1
        previous = state['certificate']
        if key is not None and previous is not None and previous['key'] == key:
            state['hits'] += 1
            return previous['result']
        state['original_calls'] += 1
        # Failures, including BaseException control stops, are never cached.
        result = original(bundle, cache)
        if key is not None and (result is None or type(result) is bool):
            state['certificate'] = dict(key=key, result=result,
                                        geometry=None if pending[2] is None else pending[2][1])
        else:
            state['certificate'] = None
        return result
    return validate


def _exact_parent_pass(block, q):
    """Read-only proof search. Original full local() still runs on delivery."""
    state = _BUNDLES.get(block.bundle)
    if state is None or state['certificate'] is None:
        return False
    certificate = state['certificate']
    owner = certificate['geometry']
    if owner is None or owner() is not block.geometry:
        return False
    world = _world(block, q)
    if world != certificate['key'][0]:
        return False
    # A same-world certificate must refer to the parent's currently bound
    # original motion geometry, not camera_local's deliberately smaller graph.
    if getattr(block, '_course_current_geometry_key', None) != world[2]:
        return False
    from .staged_source_guard import preserve_programme_camera_bindings
    pending = state['pending']
    try:
        with preserve_programme_camera_bindings((block,)):
            local = block.camera_local(q)
            cache = block.bundle.cache(local, False)
            return _key(block.bundle, _array(local), world, cache) == certificate['key']
    finally:
        state['pending'] = pending


def _write_report(event):
    if _REPORT_DIRECTORY is not None:
        try:
            with (_REPORT_DIRECTORY/'finite_extent_guard_reuse.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False)+'\n')
        except OSError:
            # A receipt cannot change an original source/control exception.
            pass


def _ordered_guard_with_reuse(actors, q):
    began = time.perf_counter()
    blocks = actors.blocks
    can_reuse = False
    if not actors.delegate._busy and not actors.delegate.closed:
        try:
            can_reuse = all(_exact_parent_pass(block, q) for block in blocks)
        except Exception:
            # Unsupported evidence/bookkeeping never becomes physical PASS.
            can_reuse = False
    before = [(_state(block.bundle)['hits'], _state(block.bundle)['original_calls']) for block in blocks]
    succeeded = False
    try:
        if not can_reuse:
            result = _ORIGINAL_ORDERED_GUARD(actors, q)
        else:
            # Deliver the same six local/focal/cache/validator calls as the
            # original serial branch. Every source suffix executes afterwards.
            for block in blocks:
                local = block.local(q)
                focal = local[7::4]
                if np.any((focal < np.log(100.)) | (focal > np.log(50000.))):
                    raise ValueError('Physical focal domain')
                block.bundle.validate_finite_extents(block.bundle.cache(local, False))
            result = None
            actors.last_stats = [dict(block=block.name, status='ok', operation='finite_extent_guard',
                                     original_complete_bundle_guard=True,
                                     exact_process_local_original_PASS_reused=True) for block in blocks]
            actors.last_guard_seconds = time.perf_counter()-began
        succeeded = True
        return result
    finally:
        after = [(_state(block.bundle)['hits'], _state(block.bundle)['original_calls']) for block in blocks]
        _write_report(dict(schema='rinkspace.process-local-exact-finite-extent-guard-reuse.v1',
            q_sha256=hashlib.sha256(np.asarray(q).tobytes()).hexdigest(),
            parent_same_world_complete_local_and_source_facts_verified=can_reuse,
            route='parent_original_guard_exact_PASS' if can_reuse else 'original_resident_guard',
            finite_PASS_hits=sum(a[0]-b[0] for a,b in zip(after,before)),
            original_parent_finite_calls=sum(a[1]-b[1] for a,b in zip(after,before)),
            resident_workers=len(actors.delegate._actors), succeeded=succeeded,
            seconds=time.perf_counter()-began, every_other_original_guard_retained=True,
            profile_weights_not_part_of_finite_physical_operator=True,
            failures_cached=False, certificates_process_local_and_not_pickled=True))


def install_finite_extent_guard_reuse_runtime(*, report_directory=None):
    """Install after parallel guard; same call also installs child-local memo.

    A file actor initializer can call this before its existing original entry.
    Child processes without the parent parallel-guard class hook only install
    their local cache/validator wrappers; no actor protocol/entry is replaced.
    """
    global _RECEIPT, _REPORT_DIRECTORY, _ORIGINAL_ORDERED_GUARD
    if _RECEIPT is not None:
        return _RECEIPT
    from . import install
    install()
    from z190_moving_pivot_source_bundle import MovingPivotSourceBundle
    from z190_fixed_lever_xyz_source_bundle import FixedLeverXYZSourceBundle
    from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock
    from structured_programme_source_candidate.current_full_source import CurrentProgrammeSource
    methods = dict(local=StructuredProgrammeBlock.local,
                   moving_cache=MovingPivotSourceBundle.cache,
                   fixed_cache=FixedLeverXYZSourceBundle.cache,
                   validator=MovingPivotSourceBundle.validate_finite_extents)
    if FixedLeverXYZSourceBundle.validate_finite_extents is not methods['validator']:
        raise TypeError('Expected original inherited finite-extent validator')
    expected = 'z190_moving_pivot_source_bundle'
    if methods['validator'].__module__ != expected:
        raise TypeError('Original finite-extent validator identity changed')
    directory = None if report_directory is None else Path(report_directory).resolve()
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    StructuredProgrammeBlock.local = _wrap_local(methods['local'])
    MovingPivotSourceBundle.cache = _wrap_cache(methods['moving_cache'])
    FixedLeverXYZSourceBundle.cache = _wrap_cache(methods['fixed_cache'])
    MovingPivotSourceBundle.validate_finite_extents = _wrap_validator(methods['validator'])
    guard_globals = CurrentProgrammeSource.guard.__globals__
    dispatch = guard_globals.get('_rinkspace_ordered_source_guards')
    if dispatch is not None:
        _ORIGINAL_ORDERED_GUARD = dispatch
        guard_globals['_rinkspace_ordered_source_guards'] = _ordered_guard_with_reuse
    _REPORT_DIRECTORY = directory
    _RECEIPT = dict(schema='rinkspace.process-local-exact-finite-extent-guard-reuse.v1',
        helper_path=str(Path(__file__).resolve()),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_methods_sha256={key: hashlib.sha256(inspect.getsource(value).encode()).hexdigest()
                                 for key,value in methods.items()},
        original_parallel_guard_dispatch_retained_on_any_certificate_miss=True,
        parent_parallel_guard_dispatch_hook_installed=dispatch is not None,
        successful_original_PASS_only=True, original_failures_and_control_stops_not_cached=True,
        exact_complete_local_and_world_and_selected_source_facts_bytes_required=True,
        full_calibration_records_selected_planes_source_UV_scene_retained=True,
        original_local_focal_cache_and_other_guard_order_retained=True,
        process_local_certificate_storage_outside_actor_pickle=True,
        no_actor_start_or_close_on_exact_parent_PASS=True,
        resident_actor_original_start_required_for_original_residual_profile_commands=True,
        registered_source_files_modified=False, actual_complete_cycle_speedup_pending=True)
    if directory is not None:
        (directory/'finite_extent_guard_reuse_receipt.json').write_text(json.dumps(_RECEIPT, indent=2)+'\n', encoding='utf8')
    return _RECEIPT
