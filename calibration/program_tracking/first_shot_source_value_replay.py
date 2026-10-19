"""Replay complete original non-J source values only at an identical state.

Install after seed/execution/raw wrappers and before obtaining solver_kwargs.
The caller must keep update_profiles=None. The first real cache hit exhausts
and compares the original factory before returning any values; later hits keep
the original physical guard and all downstream lagged/Huber/normal algebra.
This is value retention, never a claim that another projection or J was made.
"""
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np

from calibration.solver_runtime.source_loss import SourceLossWeights


def _reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _hash_value(digest, value):
    """Hash frozen source inputs without JSON-expanding their large arrays."""
    if isinstance(value, SourceLossWeights):
        digest.update(b'immutable-source-loss-weights')
        _hash_value(digest, (value.integration_weights, value.working_normalization_multiplier))
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('Frozen source value arrays cannot contain objects')
        digest.update(b'array' + value.dtype.str.encode() + str(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, np.generic):
        _hash_value(digest, np.asarray(value))
    elif isinstance(value, Mapping):
        digest.update(b'mapping')
        for key in sorted(value, key=lambda item: str(item)):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode() + str(len(value)).encode())
        for item in value:
            _hash_value(digest, item)
    elif isinstance(value, slice):
        digest.update(b'slice')
        _hash_value(digest, (value.start, value.stop, value.step))
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(type(value).__name__.encode())
        digest.update(json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf8'))
    else:
        raise TypeError('Frozen source inputs need explicit arrays/metadata: ' + type(value).__name__)


def _input_digest(owner, algebra):
    digest = hashlib.sha256(b'rinkspace.original-source-value-inputs.v1')
    _hash_value(digest, dict(parameter_count=owner.state.count, workers=owner.workers,
        quadratic=owner.quadratic, scales=owner.scales, formal_offsets=owner.formal_offsets,
        loss_weights=getattr(owner, 'loss_weights', None),
        source_loss_weights=getattr(owner, 'source_loss_weights', None),
        profile_weights=getattr(owner, 'profile_weights', None)))
    algebra._ledger_digest(digest, owner.bundle.ledgers)
    _hash_value(digest, dict(binding=getattr(getattr(owner.bundle, 'state', None), 'binding_reference', None),
        model_reference=getattr(owner, 'model_reference', None),
        association_reference=getattr(owner, 'association_reference', None)))
    for frame, whitening in zip(owner.bundle.frames, owner.bundle.whitening, strict=True):
        # These are the actual row-source inputs and the complete frozen apply()
        # data, including correlated operators, native sigma and ordinary rows.
        _hash_value(digest, dict(index=frame['index'], source_point_count=frame['source_point_count'],
                                rows=frame['rows'], whitening=vars(whitening)))
    return digest.hexdigest()


def _immutable_array(value):
    value = np.asarray(value)
    if value.dtype.hasobject:
        raise TypeError('Source replay cannot retain object arrays')
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def install(owner, *, update_profiles=None, retained_states=2):
    """Retain original complete False blocks; install last among value wrappers.

    source_blocks(..., True), subsets and unrecognized producer options always
    delegate. The returned solver kwargs explicitly keep update_profiles=None.
    Changing any frozen source input or producer is an error, never a cache hit.
    """
    from calibration.solver_runtime import install as install_runtime
    install_runtime()
    import z190_streaming_source_normal as algebra
    import z190_joint_source_solver as solver
    from calibration.solver_runtime import source_loss

    if hasattr(owner, 'source_value_replay_receipt'):
        raise ValueError('Same-state source value replay is already installed')
    if update_profiles is not None or owner.solver_kwargs().get('update_profiles') is not None:
        raise ValueError('Original source value replay requires update_profiles=None')
    if not isinstance(retained_states, int) or isinstance(retained_states, bool) or retained_states < 1:
        raise ValueError('Positive explicit source-value retention count required')
    original_blocks = owner.source_blocks
    original_identity = owner.checkpoint_identity
    original_solver_kwargs = owner.solver_kwargs
    producer = _reference(__file__)
    active_producers = deepcopy(owner.source_producers) + [producer]
    helper_refs = [_reference(module.__file__) for module in (algebra, solver, source_loss)]
    code = getattr(getattr(original_blocks, '__func__', original_blocks), '__code__', None)
    if code is not None:
        helper_refs.append(_reference(code.co_filename))
    profile_token = solver._frozen_profile_token(None)
    parameter_count = int(owner.state.count)
    row_count = len(owner.scales)
    input_digest = _input_digest(owner, algebra)
    cache = OrderedDict()
    contract = None
    busy = False
    proof = dict(schema='rinkspace.same-state-original-source-value-replay.v1', producer=producer,
        certification_producers=helper_refs, update_profiles_is_None=True,
        frozen_profile_token=profile_token, frozen_source_inputs_sha256=input_digest,
        retained_states=retained_states, original_values_only=True, jacobians_retained=False,
        source_projection_claimed_on_replay=False, physical_guard_policy_unchanged=True,
        original_rows_order_masks_scales_ledgers_and_loss_weights_unchanged=True,
        first_actual_hit_full_original_factory_bitwise_shadow_required=True,
        verified=False, full_original_shadow_evaluations=0, cache_hits=0, cache_misses=0,
        changed_complete_q_never_reused=True)
    receipts = []

    def check():
        if owner.source_producers != active_producers:
            raise ValueError('Same-state source value producer inventory changed')
        for ref in active_producers + helper_refs:
            if _reference(ref['path']) != ref:
                raise ValueError('Same-state source value producer changed: ' + ref['path'])
        if (getattr(owner, 'update_profiles', None) is not None
                or solver._frozen_profile_token(None) != profile_token
                or _input_digest(owner, algebra) != input_digest):
            raise ValueError('Frozen source profiles, arrays or original ledger changed')

    def state_key(q):
        value = np.asarray(q)
        if value.dtype != np.dtype(np.float64):
            raise ValueError('Same-state source value replay requires the original float64 complete q')
        value, digest = algebra._state_digest(value, parameter_count)
        return value.tobytes(), digest

    def loss_copy(value):
        if value is None or isinstance(value, source_loss.SourceLossWeights):
            return value  # SourceLossWeights already has immutable bytes backing.
        return _immutable_array(value)

    def block_digest(block):
        digest = hashlib.sha256()
        for array in (block.r, block.quadratic, block.scales):
            algebra._array_digest(digest, array)
        algebra._ledger_digest(digest, block.row_ledger)
        if block.loss_weights is None:
            digest.update(b'no-source-loss-weights')
        else:
            digest.update(type(block.loss_weights).__name__.encode())
            if isinstance(block.loss_weights, source_loss.SourceLossWeights):
                digest.update(block.loss_weights.identity_digest)
            else:
                algebra._array_digest(digest, block.loss_weights)
        return digest.hexdigest()

    def same_array(left, right):
        left, right = np.asarray(left), np.asarray(right)
        return (left.dtype == right.dtype and left.shape == right.shape
                and np.array_equal(np.ascontiguousarray(left).view(np.uint8),
                                   np.ascontiguousarray(right).view(np.uint8)))

    def same_values(left, right):
        if not all(same_array(getattr(left, name), getattr(right, name))
                   for name in ('r', 'quadratic', 'scales')):
            return False
        encode = dict(sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'), default=algebra._json_default)
        if json.dumps(left.row_ledger, **encode) != json.dumps(right.row_ledger, **encode):
            return False
        a, b = left.loss_weights, right.loss_weights
        if a is None or b is None:
            return a is None and b is None
        if isinstance(a, source_loss.SourceLossWeights) or isinstance(b, source_loss.SourceLossWeights):
            if not isinstance(a, source_loss.SourceLossWeights) or not isinstance(b, source_loss.SourceLossWeights):
                return False
            if not same_array(a.integration_weights, b.integration_weights):
                return False
            a, b = a.working_normalization_multiplier, b.working_normalization_multiplier
            return (a is None and b is None) if a is None or b is None else same_array(a, b)
        return same_array(a, b)

    def freeze(supplied, ordinal, cursor, expected):
        if isinstance(supplied, Mapping):
            supplied = algebra.SourceRowBlock(**supplied)
        r, _, mask, scales, signature = algebra._checked_block(supplied, parameter_count, False)
        if supplied.J is not None:
            raise ValueError('Complete non-J source producer returned a Jacobian')
        if expected is not None:
            algebra._check_replay(signature, ordinal, expected, False)
        stop = cursor + len(r)
        if (stop > row_count or not np.array_equal(mask, np.asarray(owner.quadratic)[cursor:stop])
                or not np.array_equal(scales.view(np.uint64),
                                      np.asarray(owner.scales, np.float64)[cursor:stop].view(np.uint64))):
            raise ValueError('Source value replay changed original contiguous masks/scales/rows')
        copied = algebra.SourceRowBlock(_immutable_array(supplied.r), None,
            _immutable_array(supplied.quadratic), _immutable_array(supplied.scales),
            deepcopy(supplied.row_ledger), loss_copy(supplied.loss_weights))
        return copied, signature, stop, block_digest(copied)

    def output(block):
        # Arrays cannot be made writable; ledger ownership remains per iterator.
        return algebra.SourceRowBlock(block.r, None, block.quadratic, block.scales,
                                      deepcopy(block.row_ledger), block.loss_weights)

    def consume(q, kwargs, expected):
        blocks, signatures, digests = [], [], []
        cursor = 0
        for ordinal, supplied in enumerate(original_blocks(q, False, **kwargs)):
            frozen, signature, cursor, digest = freeze(supplied, ordinal, cursor, expected)
            blocks.append(frozen); signatures.append(signature); digests.append(digest)
            yield supplied
        if not blocks or cursor != row_count:
            raise ValueError('Source value replay did not include every original row')
        current = algebra.SourceBlockContract(parameter_count, cursor, tuple(signatures), profile_token)
        if expected is not None:
            algebra._finish_replay(len(blocks), cursor, expected)
        return tuple(blocks), tuple(digests), current

    def blocks(adapter, q, jacobian, **kwargs):
        nonlocal busy, contract
        if (jacobian or kwargs.get('indices') is not None
                or set(kwargs) - {'indices', 'physical_guard'}):
            yield from original_blocks(q, jacobian, **kwargs)
            return
        if busy:
            raise RuntimeError('Same-state source value replay requires a non-reentrant complete factory')
        busy = True
        try:
            check()
            token, q_sha = state_key(q)
            entry = cache.get(token)
            if entry is None:
                proof['cache_misses'] += 1
                iterator = consume(q, kwargs, contract)
                while True:
                    try:
                        yield next(iterator)
                    except StopIteration as completed:
                        retained, digests, current = completed.value
                        break
                if state_key(q)[0] != token:
                    raise ValueError('Complete q changed while producing source values')
                check()
                if contract is None:
                    contract = current
                cache[token] = (retained, digests)
                cache.move_to_end(token)
                while len(cache) > retained_states:
                    cache.popitem(last=False)
                receipts.append(dict(qaug_sha256=q_sha, mode='original full value production retained',
                    original_source_rows=row_count, original_source_blocks=len(retained),
                    source_projection_claimed=False, original_factory_executed=True))
                return
            retained, digests = entry
            for ordinal, (block, expected_digest) in enumerate(zip(retained, digests, strict=True)):
                _, _, _, _, signature = algebra._checked_block(block, parameter_count, False)
                algebra._check_replay(signature, ordinal, contract, False)
                if block_digest(block) != expected_digest:
                    raise ValueError('Retained original source values or ledger were modified')
            algebra._finish_replay(len(retained), sum(len(block.r) for block in retained), contract)
            if not proof['verified']:
                # Exhaust the original wrapper (including raw receipt extraction)
                # before exposing any shadow-confirmed values to the solver.
                proof['full_original_shadow_evaluations'] += 1
                iterator = consume(q, kwargs, contract)
                while True:
                    try:
                        next(iterator)
                    except StopIteration as completed:
                        actual, actual_digests, _ = completed.value
                        break
                if (actual_digests != digests or len(actual) != len(retained)
                        or not all(same_values(a, b) for a, b in zip(actual, retained, strict=True))):
                    raise ValueError('Original same-q source values failed full bitwise shadow comparison')
                if state_key(q)[0] != token:
                    raise ValueError('Complete q changed during original source shadow evaluation')
                check()
                proof.update(verified=True, verified_qaug_sha256=q_sha,
                    verified_source_rows=row_count, verified_source_blocks=len(retained),
                    verified_original_block_sha256=list(digests), maximum_residual_difference=0.0,
                    verification='all array bytes equal; canonical original ledger equal; full source contract')
                receipts.append(dict(qaug_sha256=q_sha, mode='full original bitwise shadow matched',
                    original_source_rows=row_count, original_source_blocks=len(actual),
                    source_projection_claimed=False, original_factory_executed=True))
                yield from (output(block) for block in actual)
                return
            if kwargs.get('physical_guard', True):
                adapter.guard(q)
            check()
            if state_key(q)[0] != token:
                raise ValueError('Complete q changed during original physical guard')
            cache.move_to_end(token)
            proof['cache_hits'] += 1
            proof['last_hit_qaug_sha256'] = q_sha
            receipts.append(dict(qaug_sha256=q_sha, mode='same-state original values replayed',
                original_source_rows=row_count, original_source_blocks=len(retained),
                source_projection_claimed=False, original_factory_executed=False))
            yield from (output(block) for block in retained)
        finally:
            busy = False

    def identity(adapter):
        result = original_identity()
        result['same_state_original_source_value_replay'] = deepcopy(proof)
        return result

    def solver_kwargs(adapter):
        result = original_solver_kwargs()
        if result.get('update_profiles') is not None:
            raise ValueError('Source value replay cannot run with a profile update callback')
        result['update_profiles'] = None
        return result

    owner.source_producers.append(producer)
    owner.source_blocks = MethodType(blocks, owner)
    owner.checkpoint_identity = MethodType(identity, owner)
    owner.solver_kwargs = MethodType(solver_kwargs, owner)
    owner.source_value_replay_receipt = proof
    owner.source_value_replay_evaluations = receipts
    return proof
