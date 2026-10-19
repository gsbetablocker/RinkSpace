"""Streaming algebra for the unchanged normalized joint-source objective.

This opt-in helper owns no camera, admission, profile, guard, optimizer, cache,
or filesystem. The caller supplies a *replayable* factory in original source
row order, with profiles frozen before evaluation. Consume source/kernel-sized
blocks directly; wrapping an already concatenated six-cut J saves no producer
memory. By default one source block and its weighted J are live. Optional
same-state retention keeps only generated r/CSR blocks for one native replay.

Each block supplies r, J, quadratic, scales and an original row ledger. r/J
are already normalized exactly as in the existing solver. scales only convert
J @ step back to native pixels; they never change the objective. No row is
removed, no sigma is inferred, and Huber2 applies to precisely ~quadratic.

Floating summation is block-associated, so equivalence to a concatenated J is
mathematical plus floating roundoff, not a bit-exact assertion. Frozen source
row contracts and linearization digests detect incomplete or changed replays.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class SourceRowBlock:
    """One original, ordered row block; arrays remain caller-owned/read-only.

    J may be None only during trial-cost replay. row_ledger is original source
    identity/order metadata, not residual-derived row selection. A compact
    {family, original_start, original_stop, source_sha256, ...} ledger is valid;
    a list of existing row records is also valid and is hashed incrementally.
    """
    r: object
    J: object
    quadratic: object
    scales: object
    row_ledger: object


@dataclass(frozen=True)
class BlockSignature:
    rows: int
    inventory_sha256: str
    linearization_sha256: str | None


@dataclass(frozen=True)
class SourceBlockContract:
    parameter_count: int
    row_count: int
    signatures: tuple[BlockSignature, ...]
    frozen_profile_token: str


@dataclass
class StreamingSourceNormal:
    normal: csc_matrix
    gradient: np.ndarray
    cost: float
    contract: SourceBlockContract
    report: dict
    native_replay_cache: object | None = None


@dataclass(frozen=True)
class FrozenNativeReplayCache:
    """One linearization's original normalized blocks, never a source cache.

    Retention uses the existing r/CSR arrays; no six-block prepared graph is
    copied. Every replay revalidates the state, profile token, original block
    inventory and r/J digest. The caller releases this after native GN checks.
    """
    source_state: np.ndarray
    source_state_sha256: str
    contract: SourceBlockContract
    blocks: tuple[SourceRowBlock, ...]
    report: dict


def _state_digest(state,parameter_count):
    value=np.asarray(state,dtype=np.float64)
    if value.shape!=(parameter_count,)or not np.isfinite(value).all():
        raise ValueError('Native replay cache requires the exact finite source state')
    digest=hashlib.sha256();_array_digest(digest,value)
    return value,digest.hexdigest()


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Source ledger needs explicit JSON metadata: {type(value).__name__}")


def _array_digest(hasher, array):
    value = np.ascontiguousarray(array)
    hasher.update(str(value.shape).encode("ascii"))
    hasher.update(value.dtype.str.encode("ascii"))
    if value.size:
        hasher.update(memoryview(value).cast("B"))


def _ledger_digest(hasher, ledger):
    if not isinstance(ledger, (Mapping, list, tuple)):
        raise TypeError("Original row ledger must be a mapping or row-record sequence")
    encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=False, allow_nan=False,
                               separators=(",", ":"), default=_json_default)
    for chunk in encoder.iterencode(ledger):
        hasher.update(chunk.encode("utf8"))


def _checked_block(value, parameter_count, with_jacobian):
    if isinstance(value, Mapping):
        value = SourceRowBlock(**value)
    if not isinstance(value, SourceRowBlock):
        raise TypeError("Factory must yield SourceRowBlock or its exact field mapping")
    r = np.asarray(value.r, dtype=np.float64)
    quadratic = np.asarray(value.quadratic, dtype=bool)
    scales = np.asarray(value.scales, dtype=np.float64)
    if r.ndim != 1 or not len(r) or quadratic.shape != r.shape or scales.shape != r.shape:
        raise ValueError("Nonempty original block row masks/scales must match residuals")
    if not np.isfinite(r).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Finite original residuals and unchanged positive native scales required")
    inventory = hashlib.sha256()
    _array_digest(inventory, quadratic)
    _array_digest(inventory, scales)
    _ledger_digest(inventory, value.row_ledger)
    jac = None
    linearization = None
    if with_jacobian:
        if value.J is None:
            raise ValueError("Jacobian replay requires the original normalized sparse J")
        jac = csr_matrix(value.J, dtype=np.float64)
        if jac.shape != (len(r), parameter_count) or not np.isfinite(jac.data).all():
            raise ValueError("Original sparse source Jacobian inventory/finite values changed")
        # Never canonicalize an input matrix in-place. Duplicate source-column
        # entries are summed in the same CSR algebra before hashing/replaying.
        if not jac.has_canonical_format:
            jac = jac.copy()
            jac.sum_duplicates()
            jac.sort_indices()
        frozen = hashlib.sha256(inventory.digest())
        for array in (r, jac.data, jac.indices, jac.indptr):
            _array_digest(frozen, array)
        linearization = frozen.hexdigest()
    signature = BlockSignature(len(r), inventory.hexdigest(), linearization)
    return r, jac, quadratic, scales, signature


def _factory(factory):
    if not callable(factory):
        raise TypeError("A fresh iterator factory is required, not a retained block list/generator")
    return iter(factory())


def _check_replay(signature, ordinal, contract, require_linearization):
    if ordinal >= len(contract.signatures):
        raise ValueError("Source replay appended an uncontracted block")
    expected = contract.signatures[ordinal]
    if (signature.rows != expected.rows or
            signature.inventory_sha256 != expected.inventory_sha256):
        raise ValueError(f"Source block {ordinal} row order/ledger/mask/native scales changed")
    if require_linearization and signature.linearization_sha256 != expected.linearization_sha256:
        raise ValueError(f"Source block {ordinal} changed frozen r/J/profile during step replay")


def _finish_replay(count, row_count, contract):
    if count != len(contract.signatures) or row_count != contract.row_count:
        raise ValueError("Source replay omitted original blocks or rows")


def _block_cost(r, quadratic):
    # Same normalized Huber2 and quadratic formula as source_cost(). No use
    # of scales here: sigma is already part of the source r/J normalization.
    robust = np.abs(r[~quadratic])
    return (float(np.where(robust <= 2., .5*robust*robust,
                           2.*robust-2.).sum()) +
            .5*float(r[quadratic] @ r[quadratic]))


def eliminate_fixed_parameters(normal, gradient, fixed_parameter_indices=()):
    """The solver's identical algebraic zero-step elimination, without a prior."""
    n = normal.shape[0]
    raw = np.asarray(fixed_parameter_indices)
    if raw.ndim != 1 or (raw.size and not np.issubdtype(raw.dtype, np.integer)):
        raise ValueError("Explicit fixed indices must be a one-dimensional integer array")
    fixed = raw.astype(np.int64, copy=False)
    if np.any(fixed < 0) or np.any(fixed >= n) or len(np.unique(fixed)) != len(fixed):
        raise ValueError("Explicit fixed indices must be distinct and in range")
    if not len(fixed):
        return normal, gradient
    active = np.ones(n, bool)
    active[fixed] = False
    free = diags(active.astype(float), format="csc")
    result = (free @ normal @ free + diags((~active).astype(float))).tocsc()
    g = np.asarray(gradient, float).copy()
    g[fixed] = 0.
    return result, g


def accumulate_source_normal(block_factory: Callable[[], Iterable[SourceRowBlock]],
                             parameter_count: int, *, expected_row_count: int,
                             frozen_profile_token: str,
                             fixed_parameter_indices=(), expected_contract=None,
                             retain_native_replay=False,source_state=None):
    """Return the complete unchanged IRLS H, g, cost, plus a replay contract.

    Call after the existing once-per-linearization profile update. The caller
    must freeze those profiles across this pass, step replay, and trial costs.
    frozen_profile_token identifies that snapshot, not a newly estimated prior.
    H=sum Jb.T Wb Jb and g=sum Jb.T Wb rb; numerical damping/ridge stay outside.
    Existing whole-block evaluators can first yield one cut at a time; kernel
    streaming can later yield each original motion chunk without changing rows.
    Optional retention is solely for this same state's native step checks; it
    never authorizes evaluating a different state or changing profile weights.
    """
    if not isinstance(parameter_count, (int, np.integer)) or parameter_count <= 0:
        raise ValueError("Positive original parameter count required")
    n = int(parameter_count)
    if not isinstance(expected_row_count, (int, np.integer)) or expected_row_count <= 0:
        raise ValueError("Positive original source row count required")
    if not isinstance(frozen_profile_token, str) or not frozen_profile_token:
        raise ValueError("Explicit frozen profile snapshot token required")
    cached_blocks=[]if retain_native_replay else None
    cached_state=state_digest=None
    if retain_native_replay:
        source_state_value,state_digest=_state_digest(source_state,n)
        cached_state=source_state_value.copy();cached_state.setflags(write=False)
    if expected_contract is not None and (expected_contract.parameter_count != n or
            expected_contract.frozen_profile_token != frozen_profile_token):
        raise ValueError("Original parameter inventory or frozen profile token changed")
    normal = csc_matrix((n, n), dtype=np.float64)
    gradient = np.zeros(n, np.float64)
    signatures = []
    costs = []
    rows = nnz = peak_rows = peak_j_bytes = 0
    iterator = _factory(block_factory)
    for ordinal, supplied in enumerate(iterator):
        r, jac, quadratic, scales, signature = _checked_block(supplied, n, True)
        if expected_contract is not None:
            _check_replay(signature, ordinal, expected_contract, True)
        weights = np.minimum(1., 2./np.maximum(np.abs(r), 1e-12))
        weights[quadratic] = 1.
        # Literal existing sqrt-weight Gram, but only for this current block.
        weighted = jac.multiply(np.sqrt(weights)[:, None]).tocsr()
        contribution = (weighted.T @ weighted).tocsc()
        normal = normal + contribution
        gradient += np.asarray(jac.T @ (weights*r)).ravel()
        costs.append(_block_cost(r, quadratic))
        signatures.append(signature)
        if cached_blocks is not None:
            cached_blocks.append(SourceRowBlock(r,jac,quadratic,scales,supplied.row_ledger
                if isinstance(supplied,SourceRowBlock)else supplied['row_ledger']))
        rows += len(r)
        nnz += jac.nnz
        peak_rows = max(peak_rows, len(r))
        peak_j_bytes = max(peak_j_bytes, jac.data.nbytes+jac.indices.nbytes+jac.indptr.nbytes)
        # Drop the consumer's references before requesting the next producer
        # block. A yielding producer must likewise not retain previous arrays.
        del supplied, r, jac, quadratic, scales, weights, weighted, contribution
    if not signatures:
        raise ValueError("Original source iterator is empty")
    if rows != int(expected_row_count):
        raise ValueError("Streaming iterator did not include every original admitted source row")
    contract = SourceBlockContract(n, rows, tuple(signatures), frozen_profile_token)
    if expected_contract is not None:
        _finish_replay(len(signatures), rows, expected_contract)
    normal.sum_duplicates()
    normal.eliminate_zeros()
    normal, gradient = eliminate_fixed_parameters(normal, gradient, fixed_parameter_indices)
    if not np.isfinite(normal.data).all() or not np.isfinite(gradient).all():
        raise RuntimeError("Nonfinite complete streaming source normal/gradient")
    result=StreamingSourceNormal(normal, gradient, math.fsum(costs), contract, dict(
        schema="rinkspace.streaming-normalized-source-algebra.v1",
        parameters=n, original_source_rows=rows, original_source_blocks=len(signatures),
        consumed_jacobian_nonzeros=nnz, maximum_single_block_rows=peak_rows,
        maximum_single_input_csr_bytes=peak_j_bytes, normal_nonzeros=normal.nnz,
        fixed_parameter_indices=np.asarray(fixed_parameter_indices, dtype=int).tolist(),
        source_loss="unchanged Huber2, quadratic mask rows unchanged",
        weighted_gram="same sqrt(weight) Gram per original block",
        source_rows_sigmas_scales_ledgers_unchanged=True,
        full_source_residual_or_J_vstack_retained=False,
        floating_reduction_order="per block normal/gradient; fsum block costs",
        profile_update_source_evaluation_guard_line_search_not_owned=True))
    if cached_blocks is not None:
        csr_bytes=sum(b.J.data.nbytes+b.J.indices.nbytes+b.J.indptr.nbytes for b in cached_blocks)
        residual_bytes=sum(b.r.nbytes for b in cached_blocks)
        cache_report=dict(schema='rinkspace.same-state-frozen-native-replay-retention.v1',
            source_state_sha256=state_digest,frozen_profile_token=frozen_profile_token,
            original_source_blocks=len(cached_blocks),original_source_rows=rows,
            retained_CSR_payload_bytes=csr_bytes,retained_residual_payload_bytes=residual_bytes,
            retained_r_J_payload_bytes=csr_bytes+residual_bytes,
            copied_state_bytes=cached_state.nbytes,r_J_copied_for_retention=False,
            mask_scales_ledger_kept_as_original_references=True,
            prepared_source_graph_copied=False,retention_bytes_are_not_measured_process_peak=True,
            lifetime='one linearization until GN/native reference check, then release',
            wrong_state_profile_contract_or_linearization_rejected=True)
        result.native_replay_cache=FrozenNativeReplayCache(cached_state,state_digest,contract,
            tuple(cached_blocks),cache_report)
        result.report['same_state_native_replay_retention']=cache_report
    return result


def replay_source_cost(block_factory, contract: SourceBlockContract, *, frozen_profile_token: str):
    """Same trial objective, with unchanged ledger/mask/scales and all rows."""
    if frozen_profile_token != contract.frozen_profile_token:
        raise ValueError("Trial-cost profile snapshot changed")
    costs = []
    rows = count = 0
    for ordinal, supplied in enumerate(_factory(block_factory)):
        r, jac, mask, scales, signature = _checked_block(supplied, contract.parameter_count, False)
        _check_replay(signature, ordinal, contract, False)
        costs.append(_block_cost(r, mask))
        rows += len(r)
        count += 1
        del supplied, r, jac, mask, scales
    _finish_replay(count, rows, contract)
    return math.fsum(costs)


def replay_native_step_maxima(block_factory, steps, contract: SourceBlockContract, *,
                              frozen_profile_token: str,native_replay_cache=None,
                              source_state=None):
    """Replay original J to compute max(abs((J @ step)*scales)) per step.

    steps has shape (n,) or (k,n). GN and Schur-minus-GN evidence can share one
    replay by passing both rows. Default replay rebuilds original blocks;
    opt-in retention reuses this same q/profile/contract's checked blocks.
    Exact r/J digests detect mutable profile or camera state inside a factory.
    """
    if frozen_profile_token != contract.frozen_profile_token:
        raise ValueError("Native-step frozen profile snapshot changed")
    if native_replay_cache is not None:
        cache=native_replay_cache
        if not isinstance(cache,FrozenNativeReplayCache)or cache.contract!=contract:
            raise ValueError('Native replay cache source contract changed')
        state,digest=_state_digest(source_state,contract.parameter_count)
        if digest!=cache.source_state_sha256 or not np.array_equal(state,cache.source_state):
            raise ValueError('Native replay cache source state changed')
        if cache.contract.frozen_profile_token!=frozen_profile_token:
            raise ValueError('Native replay cache frozen profile changed')
        source_blocks=iter(cache.blocks)
    else:source_blocks=_factory(block_factory)
    steps = np.asarray(steps, np.float64)
    single = steps.ndim == 1
    if single:
        steps = steps[None, :]
    if (steps.ndim != 2 or steps.shape[1] != contract.parameter_count or
            not len(steps) or not np.isfinite(steps).all()):
        raise ValueError("Finite steps in complete original parameter order required")
    maxima = np.zeros(len(steps))
    rows = count = 0
    for ordinal, supplied in enumerate(source_blocks):
        r, jac, mask, scales, signature = _checked_block(supplied, contract.parameter_count, True)
        _check_replay(signature, ordinal, contract, True)
        native = np.asarray(jac @ steps.T)*scales[:, None]
        maxima = np.maximum(maxima, np.max(np.abs(native), axis=0))
        rows += len(r)
        count += 1
        del supplied, r, jac, mask, scales, native
    _finish_replay(count, rows, contract)
    return float(maxima[0]) if single else maxima


def sparse_gn_step(accumulated: StreamingSourceNormal, *, numerical_ridge_factor: float = 1e-8):
    """Existing sparse GN evidence and declared ridge; not an optimizer step."""
    ridge = float(numerical_ridge_factor)
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("Explicit finite nonnegative numerical ridge required")
    diagonal = np.maximum(accumulated.normal.diagonal(), 1.)
    step = spsolve(accumulated.normal + diags(ridge*diagonal),
                   -accumulated.gradient, permc_spec="MMD_AT_PLUS_A")
    if not np.isfinite(step).all():
        raise RuntimeError("Nonfinite complete streaming sparse GN evidence")
    return np.asarray(step), diagonal
