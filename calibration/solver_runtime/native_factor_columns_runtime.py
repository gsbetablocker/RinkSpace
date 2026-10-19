"""Schedule complete original ice/board FD columns on private numeric owners.

Only the dependency FD helper is replaced. The original factor block still
owns residuals, nominal nearest-segment/support branches, scales, CSR, source
ledger, and final nominal camera restoration. No programme graph is copied.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import time
from types import CodeType, SimpleNamespace

import numpy as np

from .confirmed_constraint_columns_runtime import (
    _small_context, _hydrate, _copy_fields, _private_material_owner,
    _PrivateContextAdmissionError, _PrivateSourceIdentityError, _UnsupportedContext,
)

_CONTEXTS = {}
_CURRENT_Q = None
_RECEIPT = None
_REPORT_DIRECTORY = None


class _WorkerSourceIdentityError(_PrivateSourceIdentityError):
    """Altered original evidence is not an execution-integration fallback."""


def _original_fd():
    from . import native_factor_dependency_fd as module
    return getattr(module, '_rinkspace_original_native_factor_dependency_jacobian',
                   module.native_factor_dependency_jacobian)


def _closure_context(factor, q, rows, active, height_columns, evaluate):
    from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
    from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
    if type(factor) not in (NativeIceCurveFactor, NativeBoardEdgeFactor):
        raise _UnsupportedContext('Only original typed native ice/board factors are scheduled')
    block = inspect.unwrap(type(factor).block)
    codes = [value for value in block.__code__.co_consts if isinstance(value, CodeType)
             and value.co_name == '<lambda>']
    operation = '_derivative_residual'
    if not inspect.isfunction(evaluate) or evaluate.__code__ not in codes or operation not in evaluate.__code__.co_names:
        raise _UnsupportedContext('Original factor block raw-FD callback identity is required')
    values = inspect.getclosurevars(evaluate).nonlocals
    if (values.get('self') is not factor or rows is not factor.rows
            or np.asarray(values.get('q')).dtype != q.dtype
            or np.asarray(values.get('q')).shape != q.shape
            or np.asarray(values.get('q')).tobytes() != q.tobytes()):
        raise _UnsupportedContext('Original factor/row/current-q callback binding changed')
    camera_columns = values['fit_cameras'] if operation == '_derivative_residual' else factor.camera_columns
    replay = dict(mode=operation, cameras=values['cameras'],
                  camera_columns=np.asarray(camera_columns, int).copy(),
                  associations=values.get('associations'))
    carrier = SimpleNamespace(state=factor.state, blocks=factor.blocks,
        records=[dict(factor=factor)], placement_source=None,
        parameter_count=factor.parameter_count, world_indices=factor.world_indices,
        height_columns=height_columns, active=active, steps=factor.camera_steps)
    small, pool, blocks, factors, pattern = _small_context(carrier)
    private = factors[0]
    _copy_fields(factor, private, ('rows', 'camera_steps', 'camera_columns', 'scales'))
    return ((small, blocks, factors, pattern), replay), pool


def _raw_evaluator(factor, q, replay):
    cameras, columns = replay['cameras'], replay['camera_columns']
    if replay['mode'] == '_derivative_residual':
        return lambda value, groups: factor._derivative_residual(value,
            cameras=cameras if np.array_equal(value[columns], q[columns]) else factor._cameras(value),
            associations=replay['associations'], apply_optical_support=False, row_indices=groups)
    if replay['mode'] == '_residual':
        return lambda value, groups: factor._residual(value,
            [factor.rows[i] for i in groups],
            cameras=cameras if np.array_equal(value[columns], q[columns]) else None,
            apply_optical_support=False)
    raise _PrivateContextAdmissionError('Unknown original native raw-FD callback mode')


def evaluate_native_columns_worker(job):
    """Independent job protocol; reuse one material owner per existing worker."""
    global _CURRENT_Q
    from .absolute_geometry_fd_pool_candidate import initialize_absolute_geometry_worker
    from .constraint_sample_reuse_runtime import install_constraint_sample_reuse_runtime
    from .tail_world_memo_runtime import install_tail_world_memo_runtime, _scopes
    initialize_absolute_geometry_worker()
    install_constraint_sample_reuse_runtime()
    install_tail_world_memo_runtime()
    snapshot = job['material_owner']
    material_key = (snapshot['path'], snapshot['sha256'])
    geometry = _private_material_owner(snapshot)
    payload = job['context_payload']
    digest = hashlib.sha256(payload).hexdigest()
    if digest != job['context_sha256']:
        raise _WorkerSourceIdentityError('Original private native context changed during transport')
    q = np.asarray(job['q'], float)
    if hashlib.sha256(q.tobytes()).hexdigest() != job['q_sha256']:
        raise _WorkerSourceIdentityError('Complete original q changed during native transport')
    if _CURRENT_Q != job['q_sha256']:
        _CONTEXTS.clear()
        _CURRENT_Q = job['q_sha256']
    context_key = (material_key, digest, job['q_sha256'])
    if context_key not in _CONTEXTS:
        try:
            context, replay = pickle.loads(payload)
            owner = _hydrate(context, geometry)
            _CONTEXTS[context_key] = (owner.records[0]['factor'], replay)
        except Exception as error:
            raise _PrivateContextAdmissionError('Private native hydrate failed: '
                + type(error).__name__ + ': ' + str(error)) from error
    factor, replay = _CONTEXTS[context_key]
    evaluate = _raw_evaluator(factor, q, replay)
    scopes = _scopes()
    if id(factor) in scopes:
        raise _PrivateContextAdmissionError('Private native world memo already has an active owner')
    scope = dict(factor=factor, values={}, hits=0, misses=0, stored_array_bytes=0)
    scopes[id(factor)] = scope
    try:
        # Compare like-for-like fixed nominal association replay. The FD base
        # remains the original caller's raw residual, never this check value.
        try:
            nominal = np.asarray(evaluate(q, np.arange(len(factor.rows), dtype=int)))
        except Exception as error:
            raise _PrivateContextAdmissionError('Private native nominal admission failed: '
                + type(error).__name__ + ': ' + str(error)) from error
        reference = job['nominal_replay']
        if nominal.shape != reference.shape or nominal.dtype != reference.dtype or nominal.tobytes() != reference.tobytes():
            raise _PrivateContextAdmissionError('Private original native raw carrier differs from caller nominal replay')
        results = []
        original = _original_fd()
        active = np.asarray(job['active'], int)
        for position in job['positions']:
            try:
                value = original(factor, q, factor.rows, active[position:position + 1],
                    job['material_columns'], job['base'], evaluate, job['height_columns'])
                context = factor._source_dependency_fd_report['contexts'][0]
                results.append((position, value[:, 0], context, None))
            except Exception as error:
                # Numerical/physical errors retain the original column order.
                # Control BaseException is not converted to a FD failure.
                results.append((position, None, None, error))
        return dict(context_sha256=digest, q_sha256=job['q_sha256'],
            material_sha256=snapshot['sha256'], pid=os.getpid(), results=results,
            private_world_memo_hits=scope['hits'], private_world_producer_calls=scope['misses'])
    finally:
        scopes.pop(id(factor), None)
        scope['values'].clear()


def _pool_available(pool):
    from .absolute_geometry_fd_pool_candidate import AbsoluteGeometryGroupPool
    if (type(pool) is not AbsoluteGeometryGroupPool or pool.closed or not pool.owns_executor
            or pool.workers <= 1 or getattr(pool, '_rinkspace_active_geometry_group_iterators', 0)
            or getattr(pool, '_rinkspace_force_parent_activation_FD', False)
            or any(getattr(pool, name, False) for name in (
                '_rinkspace_confirmed_columns_active', '_rinkspace_native_columns_active',
                '_rinkspace_shared_material_bank_precompute_active'))
            or any(int(getattr(pool, name, 0)) != 0 for name in (
                '_rinkspace_confirmed_columns_pending_jobs', '_rinkspace_native_columns_pending_jobs',
                '_rinkspace_shared_material_bank_pending_jobs'))):
        return False
    if pool.executor is None:
        return True
    last = getattr(pool, 'last_completion_scheduling', {})
    absolute_idle = last.get('active_futures_at_exit') == 0 and last.get('buffered_batches_at_exit') == 0
    private_idle = any(receipt.get('executor_identity') == id(pool.executor) and receipt.get('pending_jobs') == 0
        for receipt in (getattr(pool, name, {}) for name in (
            '_rinkspace_confirmed_columns_idle_receipt', '_rinkspace_native_columns_idle_receipt',
            '_rinkspace_shared_material_bank_idle_receipt')))
    return absolute_idle or private_idle


def _event(factor, **fields):
    event = dict(stage='native_factor_column_route', factor_class=type(factor).__name__, **fields)
    print(json.dumps(event, ensure_ascii=False), flush=True)
    if _REPORT_DIRECTORY is not None:
        with (_REPORT_DIRECTORY / 'actual_native_private_column_execution.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')


def _parallel_native_fd(factor, q, rows, active, material_columns, base, evaluate, height_columns):
    original = _original_fd()
    q, active, base = np.asarray(q, float), np.asarray(active, int), np.asarray(base, float)
    args = (factor, q, rows, active, material_columns, base, evaluate, height_columns)
    if len(active) <= 1:
        return original(*args)
    try:
        context, pool = _closure_context(factor, q, rows, active, height_columns, evaluate)
        payload = pickle.dumps(context, protocol=5)
    except Exception as error:
        _event(factor, route='original_serial', reason='Light native context packing failed: '
               + type(error).__name__ + ': ' + str(error))
        return original(*args)
    if not _pool_available(pool):
        _event(factor, route='original_serial', reason='Original owned material pool lacks idle execution admission')
        return original(*args)
    # This original callback must succeed independently; its mathematical
    # failures are never mistaken for a worker/hydrate integration failure.
    nominal = np.asarray(evaluate(q, np.arange(len(rows), dtype=int)))
    digest = hashlib.sha256(payload).hexdigest()
    q_digest = hashlib.sha256(q.tobytes()).hexdigest()
    try:
        executor = pool._executor()
    except Exception as error:
        pool.release_idle()
        _event(factor, route='original_serial', reason='Original owned executor startup failed: '
               + type(error).__name__ + ': ' + str(error))
        return original(*args)
    if executor is None:
        _event(factor, route='original_serial', reason='Original pool memory admission selected parent execution')
        return original(*args)
    pool._rinkspace_native_columns_active = True
    pool._rinkspace_native_columns_pending_jobs = 0
    futures, failures, outcomes, worker_pids = [], [], {}, set()
    succeeded = False
    started = time.perf_counter()
    result = None
    try:
        _event(factor, route='original_FD_private_columns', original_active_columns=len(active),
            source_groups=len(rows), requested_workers=pool.workers,
            small_context_serialized_bytes=len(payload), current_q_sha256=q_digest)
        common = dict(material_owner=dict(pool.material_owner), context_payload=payload,
            context_sha256=digest, q=q, q_sha256=q_digest, active=active,
            material_columns=np.asarray(material_columns, int), base=base,
            nominal_replay=nominal, height_columns=height_columns)
        for worker in range(min(pool.workers, len(active))):
            positions = list(range(worker, len(active), pool.workers))
            try:
                future = executor.submit(evaluate_native_columns_worker, dict(common, positions=positions))
            except Exception as error:
                failures.append('Native column job submission failed: ' + type(error).__name__ + ': ' + str(error))
                break
            futures.append((positions, future))
            pool._rinkspace_native_columns_pending_jobs += 1
        for positions, future in futures:
            try:
                returned = future.result()
            except _PrivateSourceIdentityError:
                raise
            except Exception as error:
                failures.append('Native column transport/admission failed: ' + type(error).__name__ + ': ' + str(error))
                continue
            finally:
                pool._rinkspace_native_columns_pending_jobs -= 1
            if (returned['context_sha256'] != digest or returned['q_sha256'] != q_digest
                    or returned['material_sha256'] != pool.material_owner['sha256']
                    or [item[0] for item in returned['results']] != positions):
                raise ValueError('Original native column identity/order changed during transport')
            worker_pids.add(returned['pid'])
            for position, derivative, report, error in returned['results']:
                if position in outcomes:
                    raise ValueError('An original native column was returned twice')
                outcomes[position] = (derivative, report, error)
        # Known original mathematical failures escape before any serial
        # integration fallback, using their original active-column order.
        for position in range(len(active)):
            if position in outcomes and outcomes[position][2] is not None:
                raise outcomes[position][2]
        if failures:
            raise _PrivateContextAdmissionError(' | '.join(dict.fromkeys(failures)))
        if set(outcomes) != set(range(len(active))):
            raise ValueError('An original active native column was omitted')
        result = np.zeros((len(base), len(active)), float)
        contexts = []
        for position in range(len(active)):
            derivative, report, _ = outcomes[position]
            derivative = np.asarray(derivative)
            if derivative.shape != base.shape or derivative.dtype != result.dtype or not np.isfinite(derivative).all():
                raise ValueError('Original native derivative shape/type/domain changed in worker')
            if report['column'] != int(active[position]):
                raise ValueError('Original native dependency report column changed')
            result[:, position] = derivative
            contexts.append(report)
        # Retain original final sample binding and authenticate one real
        # transported complete column against the unchanged caller callback.
        last = original(factor, q, rows, active[-1:], material_columns, base, evaluate, height_columns)
        if last[:, 0].tobytes() != result[:, -1].tobytes():
            raise ValueError('Private final native FD column differs from original caller execution')
        factor._source_dependency_fd_report = dict(
            schema='rinkspace.original-native-factor-camera-dependency-FD.v1',
            complete_scalar_rows=len(base), original_active_columns=len(active),
            original_sample_scalar_rows=len(base) * len(active),
            actual_sample_scalar_rows=sum(item['evaluated_scalar_rows'] for item in contexts),
            material_columns=np.asarray(material_columns, int).tolist(), contexts=contexts,
            seconds=time.perf_counter() - started,
            original_source_pixels_support_sigma_roles_FD_and_full_state_checks_preserved=True,
            accuracy_accepted=False)
        succeeded = True
    except _PrivateContextAdmissionError as error:
        failures = [str(error)]
    finally:
        try:
            if succeeded:
                pool._rinkspace_native_columns_idle_receipt = dict(
                    executor_identity=id(executor), pending_jobs=0, current_q_sha256=q_digest)
            else:
                for _, future in futures:
                    if not future.done():
                        future.cancel()
                pool.release_idle()
                pool.__dict__.pop('_rinkspace_native_columns_idle_receipt', None)
        finally:
            pool._rinkspace_native_columns_pending_jobs = 0
            pool._rinkspace_native_columns_active = False
        try:
            _event(factor, route='original_FD_private_columns_result', succeeded=succeeded,
                actual_worker_pids=sorted(worker_pids), current_q_sha256=q_digest,
                original_active_columns=len(active), source_groups=len(rows),
                small_context_serialized_bytes=len(payload), private_context_execution_fallback_reasons=failures,
                successful_idle_executor_retained_for_original_next_stage=succeeded,
                pending_jobs=0, executor_identity=id(executor) if succeeded else None,
                material_owner_loaded_once_per_worker=True, full_programme_graph_transmitted=False,
                original_last_complete_column_byte_checked=succeeded, actual_speedup_pending=True)
        except Exception:
            if succeeded:
                raise
    if failures:
        _event(factor, route='original_serial', reason='All owned native jobs released: ' + ' | '.join(failures))
        return original(*args)
    return result


def install_native_factor_columns_runtime(*, report_directory=None):
    """Install after camera/world reuse; only replace imported native FD helper."""
    global _RECEIPT, _REPORT_DIRECTORY
    destination = None if report_directory is None else Path(report_directory).resolve()
    if _RECEIPT is not None:
        if _RECEIPT['report_directory'] != (None if destination is None else str(destination)):
            raise RuntimeError('Native column scheduling already belongs to another run')
        return _RECEIPT
    from . import native_factor_dependency_fd as module
    original = module.native_factor_dependency_jacobian
    module._rinkspace_original_native_factor_dependency_jacobian = original
    module.native_factor_dependency_jacobian = _parallel_native_fd
    helper = Path(__file__).resolve()
    _REPORT_DIRECTORY = destination
    _RECEIPT = dict(schema='rinkspace.private-original-native-FD-columns.v1',
        helper_path=str(helper), helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        report_directory=None if destination is None else str(destination),
        original_dependency_fd_source_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
        original_per_column_positive_pivot_FD_helper_and_dependency_rows_retained=True,
        original_factor_residual_nearest_segment_support_scales_CSR_ledger_and_camera_restore_retained=True,
        caller_raw_FD_closure_identity_and_worker_nominal_bytes_checked=True,
        worker_private_material_camera_calibration_and_block_world_memos=True,
        existing_owned_material_pool_and_snapshot_reused=True,
        execution_admission_failure_releases_jobs_then_complete_original_serial_FD=True,
        original_mathematical_physical_and_control_exceptions_propagate=True,
        successful_idle_executor_retained_for_original_J_release_and_overall_finally=True,
        full_programme_graph_profiles_motion_arrays_or_CSR_transmitted=False,
        actual_production_equivalence_and_speedup_pending=True, accuracy_accepted=False)
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'native_factor_columns_runtime_receipt.json').write_text(
            json.dumps(_RECEIPT, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return _RECEIPT
