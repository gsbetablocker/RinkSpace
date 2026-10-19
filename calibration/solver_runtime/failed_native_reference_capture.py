"""Failure-only capture of the original complete source GN/Schur gate.

This module does not solve, change a derivative or relax a gate. It writes the
already consumed numerical state and sparse source blocks before their RAM
replay cache is released. The saved blocks replay without a programme graph.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError('Original source diagnostic metadata is not JSON: ' + type(value).__name__)


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                              indent=2, default=_json_default), encoding='utf8')


def _file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def capture_failed_native_reference(output, *, q, normal, gradient, diagonal,
        gn, reference_step, shared_indices, fixed_indices, cache, metadata):
    """Stream complete original blocks to a new owned failure directory.

    Uncompressed NPZ preserves array dtype and bytes without spending failure
    recovery time compressing hundreds of MiB. No full J is concatenated.
    ``capture_complete.json`` is written only after every block is present.
    """
    from calibration.solver_runtime.source_loss import source_loss_checkpoint_arrays
    from z190_streaming_source_normal import FrozenNativeReplayCache

    if not isinstance(cache, FrozenNativeReplayCache):
        raise ValueError('Complete original RAM replay cache required for failure capture')
    q = np.asarray(q, np.float64)
    normal = csc_matrix(normal, copy=False)
    if (q.shape != (cache.contract.parameter_count,)
            or normal.shape != (len(q), len(q))
            or not np.array_equal(q, cache.source_state)):
        raise ValueError('Failure capture cannot combine different complete numerical states')
    for name, value in (('gradient', gradient), ('diagonal', diagonal),
                        ('gn', gn), ('reference_step', reference_step)):
        value = np.asarray(value)
        if value.shape != q.shape or not np.isfinite(value).all():
            raise ValueError('Finite complete original ' + name + ' required')
    q_sha = hashlib.sha256(q.tobytes()).hexdigest()
    destination = Path(output) / 'failed_native_reference' / (q_sha[:16] + '_' + str(time.time_ns()))
    destination.mkdir(parents=True, exist_ok=False)
    blocks_directory = destination / 'source_blocks'
    blocks_directory.mkdir()
    details = dict(metadata)
    details.update(schema='rinkspace.failed-complete-source-native-GN-Schur-reference.v1',
        q_sha256=q_sha, cache_source_state_sha256=cache.source_state_sha256,
        source_contract=asdict(cache.contract), native_replay_cache_report=cache.report,
        all_original_r_J_scales_masks_and_ledgers_saved=True,
        numerical_equations_source_operator_and_threshold_unchanged=True,
        normal_pass_does_not_capture=True, complete=False)
    # Preserve a partial inventory if filesystem failure interrupts capture;
    # a partial capture is never admitted by the offline replay API.
    _write_json(destination / 'capture_metadata.json', details)
    np.savez(destination / 'numerical_state.npz', q=q,
        normal_data=normal.data, normal_indices=normal.indices,
        normal_indptr=normal.indptr, normal_shape=np.asarray(normal.shape),
        gradient=np.asarray(gradient), diagonal=np.asarray(diagonal), gn=np.asarray(gn),
        reference_step=np.asarray(reference_step),
        shared_parameter_indices=np.asarray(shared_indices, dtype=np.int64),
        fixed_parameter_indices=np.asarray(fixed_indices, dtype=np.int64),
        numerical_ridge_factor=np.array(1e-8))
    rows = 0
    inventory = []
    for ordinal, block in enumerate(cache.blocks):
        matrix = csr_matrix(block.J, copy=False)
        name = str(ordinal).zfill(3)
        np.savez(blocks_directory / (name + '.npz'), r=block.r,
            J_data=matrix.data, J_indices=matrix.indices, J_indptr=matrix.indptr,
            J_shape=np.asarray(matrix.shape), quadratic=block.quadratic,
            scales=block.scales,
            **source_loss_checkpoint_arrays(block.loss_weights, len(block.r)))
        _write_json(blocks_directory / (name + '.ledger.json'), block.row_ledger)
        inventory.append(dict(ordinal=ordinal, file=name + '.npz',
                              ledger=name + '.ledger.json', rows=len(block.r), nnz=matrix.nnz,
                              file_sha256=_file_sha(blocks_directory / (name + '.npz')),
                              ledger_sha256=_file_sha(blocks_directory / (name + '.ledger.json'))))
        rows += len(block.r)
    if rows != cache.contract.row_count or len(inventory) != len(cache.contract.signatures):
        raise ValueError('Failure capture did not contain every original source block')
    details.update(complete=True, original_source_rows=rows,
                   original_source_blocks=len(inventory), source_blocks=inventory)
    _write_json(destination / 'capture_metadata.json', details)
    _write_json(destination / 'capture_complete.json', dict(
        schema='rinkspace.complete-native-reference-capture-seal.v1',
        complete=True, q_sha256=q_sha, source_blocks=len(inventory), source_rows=rows,
        numerical_state_sha256=_file_sha(destination / 'numerical_state.npz'),
        metadata_sha256=hashlib.sha256((destination / 'capture_metadata.json').read_bytes()).hexdigest()))
    return dict(path=str(destination.resolve()), complete=True, q_sha256=q_sha,
                source_blocks=len(inventory), source_rows=rows,
                derivative_model=details.get('derivative_model'),
                reference_error_native_px=details.get('reference_error_native_px'))


def load_failed_native_reference(path):
    """Load small numerical state; source blocks remain an offline iterator."""
    path = Path(path)
    seal = json.loads((path / 'capture_complete.json').read_text(encoding='utf8'))
    metadata_bytes = (path / 'capture_metadata.json').read_bytes()
    metadata = json.loads(metadata_bytes)
    if (seal.get('complete') is not True or metadata.get('complete') is not True
            or hashlib.sha256(metadata_bytes).hexdigest() != seal['metadata_sha256']):
        raise ValueError('Complete original failure capture and metadata seal required')
    if _file_sha(path / 'numerical_state.npz') != seal['numerical_state_sha256']:
        raise ValueError('Failure capture numerical arrays changed')
    for record in metadata['source_blocks']:
        if (_file_sha(path / 'source_blocks' / record['file']) != record['file_sha256']
                or _file_sha(path / 'source_blocks' / record['ledger']) != record['ledger_sha256']):
            raise ValueError('Failure capture source block or ledger changed')
    with np.load(path / 'numerical_state.npz', allow_pickle=False) as stored:
        numeric = {key: np.array(stored[key], copy=True) for key in stored.files}
    if hashlib.sha256(numeric['q'].tobytes()).hexdigest() != seal['q_sha256']:
        raise ValueError('Failure capture complete q bytes changed')
    numeric['normal'] = csc_matrix((numeric.pop('normal_data'), numeric.pop('normal_indices'),
                                   numeric.pop('normal_indptr')), shape=tuple(numeric.pop('normal_shape')))
    return metadata, numeric


def iter_failed_native_reference_blocks(path):
    """Yield each original saved CSR without rebuilding or modifying its J."""
    from calibration.solver_runtime.source_loss import source_loss_from_checkpoint
    from z190_streaming_source_normal import SourceRowBlock

    path = Path(path)
    metadata, _ = load_failed_native_reference(path)
    for record in metadata['source_blocks']:
        with np.load(path / 'source_blocks' / record['file'], allow_pickle=False) as stored:
            matrix = csr_matrix((stored['J_data'], stored['J_indices'], stored['J_indptr']),
                                shape=tuple(stored['J_shape']))
            ledger = json.loads((path / 'source_blocks' / record['ledger']).read_text(encoding='utf8'))
            yield SourceRowBlock(stored['r'], matrix, stored['quadratic'], stored['scales'],
                                 ledger, source_loss_from_checkpoint(stored))


def replay_failed_native_reference(path):
    """Recompute the exact original two-column native step difference gate."""
    from z190_streaming_source_normal import (
        BlockSignature, SourceBlockContract, replay_native_step_maxima)

    metadata, numeric = load_failed_native_reference(path)
    saved = metadata['source_contract']
    contract = SourceBlockContract(saved['parameter_count'], saved['row_count'],
        tuple(BlockSignature(**record) for record in saved['signatures']), saved['frozen_profile_token'])
    steps = np.stack((numeric['gn'], numeric['reference_step'] - numeric['gn']))
    maxima = replay_native_step_maxima(lambda: iter_failed_native_reference_blocks(path), steps,
                                     contract, frozen_profile_token=contract.frozen_profile_token)
    return dict(maximum_GN_native_step_px=float(maxima[0]),
                reference_error_native_px=float(maxima[1]),
                derivative_model=metadata.get('derivative_model'),
                original_threshold_native_px=metadata.get('original_threshold_native_px'),
                original_complete_source_operator_replayed=True)
