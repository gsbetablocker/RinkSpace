"""Retain the unmodified trial-cost raw values for the same-state review.

The first real hit runs the original full raw evaluation and requires uint64
identity. No residual, Jacobian, whitening, source row, or projection is changed.
"""
from collections import OrderedDict
import hashlib
from pathlib import Path
import types
import numpy as np

_CAPTURE = None
_ORIGINAL_RAW = None


def _sha(q):
    return hashlib.sha256(np.asarray(q, dtype=float).tobytes()).hexdigest()


def _capture_raw(state, q, frame, worlds, derivatives=None):
    result = _ORIGINAL_RAW(state, q, frame, worlds, derivatives)
    if _CAPTURE is not None and derivatives is None:
        raw = np.asarray(result[0], dtype=float)
        expected = int(frame['source_point_count'])
        if raw.shape != (2 * expected,) or not np.isfinite(raw).all():
            raise ValueError('Raw retention changed original native point inventory')
        index = int(frame['index']) - 1
        if index in _CAPTURE:
            raise ValueError('Repeated frame in one source-cost chunk')
        _CAPTURE[index] = (_sha(q), np.linalg.norm(raw.reshape(-1, 2), axis=1))
    return result


def worker_init(*args):
    import native_transverse_bundle as original
    from . import first_shot_row_source as row
    global _ORIGINAL_RAW
    original.init_worker(*args)
    _ORIGINAL_RAW = row._raw_and_local
    row._raw_and_local = _capture_raw


def source_chunk(task):
    import native_transverse_bundle as original
    global _CAPTURE
    q, _, derivatives, frames, expected = task
    _CAPTURE = {} if derivatives is None else None
    try:
        r, j, receipt = original.source_chunk(task)
        if _CAPTURE is not None:
            if list(_CAPTURE) != list(frames):
                raise ValueError('Retained raw frames differ from original source chunk')
            if any(token != expected for token, _ in _CAPTURE.values()):
                raise ValueError('Retained raw belongs to a different complete state')
            receipt['_same_state_cost_raw'] = np.concatenate([a for _, a in _CAPTURE.values()])
            receipt['same_state_cost_raw_producer'] = 'original _raw_and_local, before unchanged whitening'
        return r, j, receipt
    finally:
        _CAPTURE = None


def install(owner):
    """Install after seed authentication, before the first source pool opens."""
    import native_transverse_bundle as runtime
    if owner.pool is not None:
        raise ValueError('Install raw retention before source workers start')
    if getattr(runtime, '_raw_retention_installed', False):
        raise ValueError('Raw retention can only be installed once in a process')
    producer = dict(path=str(Path(__file__).resolve()),
                    sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    owner.source_producers.append(producer)
    runtime.init_worker = worker_init
    runtime.source_chunk = source_chunk
    runtime._raw_retention_installed = True
    original_blocks, original_raw = owner.source_blocks, owner.raw
    original_identity = owner.checkpoint_identity
    cache = OrderedDict()
    proof = dict(schema='rinkspace.same-state-original-cost-raw-retention.v1',
                 producer=producer, projection_or_residual_math_changed=False,
                 first_actual_hit_full_original_raw_bitwise_verification_required=True,
                 verified=False, cache_hits=0, full_original_shadow_evaluations=0,
                 changed_q_never_reused=True, retained_states=2)

    def check():
        if hashlib.sha256(Path(producer['path']).read_bytes()).hexdigest() != producer['sha256']:
            raise ValueError('Raw retention producer changed during run')

    def blocks(adapter, q, jacobian, **kwargs):
        check()
        before = len(adapter.worker_receipts)
        yield from original_blocks(q, jacobian, **kwargs)
        receipts = adapter.worker_receipts[before:]
        captures = []
        for receipt in receipts:
            raw = receipt.pop('_same_state_cost_raw', None)
            if raw is not None:
                if receipt['augmented_q_sha256'] != _sha(q):
                    raise ValueError('Source-cost raw receipt has wrong complete q')
                raw = np.asarray(raw, float)
                frames = list(receipt['frames_0based'])
                if len(raw) != sum(adapter.bundle.frames[i]['source_point_count'] for i in frames):
                    raise ValueError('Source-cost raw receipt point count differs')
                receipt['same_state_cost_raw_sha256'] = _sha(raw)
                captures.append((frames, raw))
        if jacobian or kwargs.get('indices') is not None:
            return
        if not captures:
            if receipts:
                raise ValueError('Source workers returned no original trial-cost raw')
            return  # Authenticated initial r replay does not run a source worker.
        frames = [i for ids, _ in captures for i in ids]
        if frames != list(range(len(adapter.bundle.frames))):
            raise ValueError('Full source-cost raw does not cover every native frame in order')
        value = np.concatenate([raw for _, raw in captures])
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite original source-cost raw')
        token = np.asarray(q, float).tobytes()
        cache[token] = value
        cache.move_to_end(token)
        while len(cache) > 2:
            cache.popitem(last=False)

    def raw(adapter, q, **kwargs):
        check()
        if kwargs.get('indices') is not None:
            return original_raw(q, **kwargs)
        token = np.asarray(q, float).tobytes()
        value = cache.get(token)
        if value is None:
            return original_raw(q, **kwargs)
        if kwargs.get('physical_guard', True):
            adapter.guard(q)
        if not proof['verified']:
            actual = np.asarray(original_raw(q, **kwargs), float)
            proof['full_original_shadow_evaluations'] += 1
            if actual.shape != value.shape or not np.array_equal(actual.view(np.uint64), value.view(np.uint64)):
                raise ValueError('Original trial-cost raw and full original review raw differ')
            proof.update(verified=True, verified_qaug_sha256=_sha(q),
                         verified_raw_sha256=_sha(actual), verified_native_points=len(actual),
                         maximum_pixel_difference=0.0)
            return actual
        proof['cache_hits'] += 1
        proof['last_hit_qaug_sha256'] = _sha(q)
        return value.copy()

    def identity(adapter):
        result = original_identity()
        result['same_state_original_cost_raw_retention'] = dict(proof)
        return result

    owner.source_blocks = types.MethodType(blocks, owner)
    owner.raw = types.MethodType(raw, owner)
    owner.checkpoint_identity = types.MethodType(identity, owner)
    owner.raw_retention_receipt = proof
    return proof
