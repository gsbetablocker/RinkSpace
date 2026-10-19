"""Run original finite-station FD columns on the existing private material pool.

Only the column execution is scheduled. The caller retains the nominal
station chart, source witnesses, physical certificates and face assembly.
Installing this helper does not start a pool or construct a source graph.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

import numpy as np

_ORIGINAL = None
_RECEIPT = None
_REPORT_DIRECTORY = None
_CONTEXTS = {}
_CALLER_COLUMN_QUALIFICATION_Q = None


def _projection_module():
    from calibration.solver_runtime import finite_rectangle_projection_fd
    return finite_rectangle_projection_fd


def _small_station_context(truth, contexts):
    from calibration.solver_runtime.confirmed_constraint_columns_runtime import _small_context
    carrier = SimpleNamespace(state=truth.state, blocks=truth.blocks,
        records=truth.full_stroke_records, placement_source=None,
        parameter_count=truth.parameter_count, world_indices=truth.world_indices,
        height_columns=truth.height_columns, active=truth.active, steps=truth.steps)
    small, pool, blocks, factors, pattern = _small_context(carrier)
    small.full_stroke_records = small.records
    # Record identity is intentionally reconstructed inside the same private
    # tuple; transporting original record/factor pointers copies the live graph.
    descriptors = [{key: value for key, value in item.items() if key != 'record'}
                   for item in contexts]
    payload = pickle.dumps(((small, blocks, factors, pattern), descriptors), protocol=5)
    return payload, pool


def _nominal_arrays(rows):
    count = len(rows)
    return dict(projected_foot_xy=np.asarray([r['foot'] for r in rows]).reshape(count, 2),
        station_screen_tangent=np.asarray([r['tangent'] for r in rows]).reshape(count, 2),
        nominal_physical_station=np.asarray([r['station'] for r in rows]),
        physical_endpoints_xy=np.asarray([r['endpoints'] for r in rows]).reshape(count, 2, 2),
        endpoint_valid=np.asarray([r['endpoint_valid'] for r in rows], bool).reshape(count, 2))


def _exact(actual, expected):
    return (actual.shape == expected.shape and actual.dtype == expected.dtype
            and actual.tobytes() == expected.tobytes())


def evaluate_station_columns_worker(job):
    """Independent job protocol, reusing the same private geometry owner."""
    from calibration.solver_runtime.absolute_geometry_fd_pool_candidate import initialize_absolute_geometry_worker
    from calibration.solver_runtime.confirmed_constraint_columns_runtime import (
        _hydrate, _private_material_owner, _PrivateContextAdmissionError, _PrivateSourceIdentityError)
    initialize_absolute_geometry_worker()
    projection = _projection_module()
    payload, snapshot = job['context_payload'], job['material_owner']
    digest = hashlib.sha256(payload).hexdigest()
    if digest != job['context_sha256']:
        raise _PrivateSourceIdentityError('Original station numerical context changed during transport')
    q = np.asarray(job['q'], float)
    if hashlib.sha256(q.tobytes()).hexdigest() != job['q_sha256']:
        raise _PrivateSourceIdentityError('Complete original station q changed during transport')
    key = (snapshot['path'], snapshot['sha256'], digest, job['q_sha256'])
    if key not in _CONTEXTS:
        _CONTEXTS.clear()
        geometry = _private_material_owner(snapshot)
        try:
            context, descriptors = pickle.loads(payload)
            truth = _hydrate(context, geometry)
            contexts = [dict(item, record=truth.full_stroke_records[int(item['group'])])
                        for item in descriptors]
            contexts = projection._contexts(truth, contexts)
            _CONTEXTS[key] = truth, contexts
        except Exception as error:
            raise _PrivateContextAdmissionError('Private station hydrate failed: '
                + type(error).__name__ + ': ' + str(error)) from error
    truth, contexts = _CONTEXTS[key]
    helpers = projection._helpers()
    try:
        nominal = projection._project(truth, q, contexts, helpers)
    except Exception as error:
        raise _PrivateContextAdmissionError('Private nominal station admission differs from caller: '
            + type(error).__name__ + ': ' + str(error)) from error
    arrays = _nominal_arrays(nominal)
    for name, value in arrays.items():
        if not _exact(value, job['nominal_arrays'][name]):
            raise _PrivateContextAdmissionError('Private original station nominal differs: ' + name)
    if [r['proof'] for r in nominal] != job['projection_validation']:
        raise _PrivateContextAdmissionError('Original station physical projection proof differs in worker')
    dependencies = helpers[2](truth, truth.full_stroke_records)
    if len(dependencies) != len(job['dependencies']) or any(
            not _exact(a, b) for a, b in zip(dependencies, job['dependencies'])):
        raise _PrivateContextAdmissionError('Original station camera/world dependencies differ in worker')
    dependency_sets = [set(map(int, dependencies[c['group']])) for c in contexts]
    selected = np.asarray(job['selected'], int)
    results = []
    try:
        for position in job['positions']:
            column = int(selected[position])
            involved = [i for i, deps in enumerate(dependency_sets) if column in deps]
            subset = [contexts[i] for i in involved]
            # Use the same caller-authenticated base and exact arithmetic order.
            base = np.asarray([np.r_[nominal[i]['foot'], nominal[i]['endpoints'].ravel()]
                               for i in involved]).ravel()
            evaluations = 0
            def evaluate(value):
                nonlocal evaluations
                evaluations += 1
                rows = projection._project(truth, value, subset, helpers,
                    endpoint_masks=[nominal[i]['endpoint_valid'] for i in involved])
                return np.asarray([np.r_[r['foot'], r['endpoints'].ravel()] for r in rows]).ravel()
            try:
                d = helpers[1](evaluate, q, column, truth.steps.get(column, 1e-6),
                               truth.height_columns, base=base).reshape(-1, 6)
                results.append((position, involved, d, evaluations, None))
            except Exception as error:
                # Preserve the earliest original numerical failure in caller
                # column order, irrespective of worker completion order.
                results.append((position, involved, None, evaluations, error))
                break
    finally:
        projection._project(truth, q, contexts, helpers)
    return dict(context_sha256=digest, q_sha256=job['q_sha256'],
        material_sha256=snapshot['sha256'], pid=os.getpid(), results=results,
        complete_nominal_arrays_and_physical_proof_byte_checked=True)


def _record(**event):
    event.update(stage='finite_station_column_route', wall_time_ns=time.time_ns())
    print(json.dumps(event, ensure_ascii=False), flush=True)
    if _REPORT_DIRECTORY is not None:
        with (_REPORT_DIRECTORY / 'actual_station_private_column_execution.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')


def _parallel_station_contexts(truth, q, contexts, selected):
    global _CALLER_COLUMN_QUALIFICATION_Q
    from calibration.solver_runtime.confirmed_constraint_columns_runtime import (
        _PrivateContextAdmissionError, _PrivateSourceIdentityError)
    from calibration.solver_runtime.native_factor_columns_runtime import _pool_available
    projection = _projection_module()
    original = _ORIGINAL
    q = projection._state(truth, q)
    contexts = projection._contexts(truth, contexts)
    raw_selected = np.asarray(selected)
    if raw_selected.ndim != 1 or raw_selected.dtype.kind not in 'iu':
        return original(truth, q, contexts, selected)
    selected = np.asarray(raw_selected, int)
    if (len(np.unique(selected)) != len(selected) or np.any(selected < 0)
            or np.any(selected >= truth.parameter_count)):
        return original(truth, q, contexts, selected)
    try:
        payload, pool = _small_station_context(truth, contexts)
    except Exception as error:
        _record(route='original_serial', reason='Light typed station context packing failed: '
                + type(error).__name__ + ': ' + str(error))
        return original(truth, q, contexts, selected)
    if not _pool_available(pool):
        _record(route='original_serial', reason='Existing owned material pool lacks idle execution admission')
        return original(truth, q, contexts, selected)
    helpers = projection._helpers()
    nominal = projection._project(truth, q, contexts, helpers)
    arrays = _nominal_arrays(nominal)
    dependencies = helpers[2](truth, truth.full_stroke_records)
    dependency_sets = [set(map(int, dependencies[c['group']])) for c in contexts]
    positions = [i for i, column in enumerate(selected)
                 if any(int(column) in deps for deps in dependency_sets)]
    if len(positions) <= 1:
        return original(truth, q, contexts, selected)
    digest = hashlib.sha256(payload).hexdigest()
    q_digest = hashlib.sha256(q.tobytes()).hexdigest()
    try:
        executor = pool._executor()
    except Exception as error:
        pool.release_idle()
        _record(route='original_serial', reason='Existing executor startup failed: '
                + type(error).__name__ + ': ' + str(error))
        return original(truth, q, contexts, selected)
    if executor is None:
        return original(truth, q, contexts, selected)
    # The existing native private-column flag/receipt is the compatible idle
    # execution protocol understood by every existing pool-lifetime consumer.
    pool._rinkspace_native_columns_active = True
    pool._rinkspace_native_columns_pending_jobs = 0
    futures, failures, outcomes, worker_pids = [], [], {}, set()
    succeeded = False
    caller_column_checked = False
    started = time.perf_counter()
    try:
        _record(route='original_FD_private_columns', requested_workers=pool.workers,
            actual_FD_columns=len(positions), selected_columns=len(selected),
            station_points=len(contexts), small_context_serialized_bytes=len(payload),
            current_q_sha256=q_digest)
        common = dict(material_owner=dict(pool.material_owner), context_payload=payload,
            context_sha256=digest, q=q, q_sha256=q_digest, selected=selected,
            nominal_arrays=arrays, dependencies=dependencies,
            projection_validation=[r['proof'] for r in nominal])
        for worker in range(min(pool.workers, len(positions))):
            group = positions[worker::pool.workers]
            try:
                future = executor.submit(evaluate_station_columns_worker, dict(common, positions=group))
            except Exception as error:
                failures.append('Station submission failed: ' + type(error).__name__ + ': ' + str(error))
                break
            futures.append((group, future))
            pool._rinkspace_native_columns_pending_jobs += 1
        for group, future in futures:
            try:
                returned = future.result()
            except _PrivateSourceIdentityError:
                raise
            except Exception as error:
                failures.append('Station transport/admission failed: ' + type(error).__name__ + ': ' + str(error))
                continue
            finally:
                pool._rinkspace_native_columns_pending_jobs -= 1
            if (returned['context_sha256'] != digest or returned['q_sha256'] != q_digest
                    or returned['material_sha256'] != pool.material_owner['sha256']):
                raise ValueError('Original station context/state/material identity changed during transport')
            worker_pids.add(returned['pid'])
            returned_results = returned['results']
            indices = [item[0] for item in returned_results]
            if (not returned_results or indices != group[:len(indices)]
                    or (indices != group and returned_results[-1][-1] is None)):
                raise ValueError('Original station column order/count changed during transport')
            for position, involved, d, evaluations, error in returned['results']:
                if position in outcomes:
                    raise ValueError('An original station column was returned twice')
                outcomes[position] = involved, d, evaluations, error
        for position in positions:
            if position in outcomes and outcomes[position][-1] is not None:
                raise outcomes[position][-1]
        if failures:
            raise _PrivateContextAdmissionError(' | '.join(dict.fromkeys(failures)))
        if set(outcomes) != set(positions):
            raise ValueError('An original station column was omitted')
        count, width = len(contexts), len(selected)
        foot_j = np.zeros((count, 2, width))
        endpoint_j = np.zeros((count, 2, 2, width))
        evaluations = 0
        for position in positions:
            involved, d, evaluated, _ = outcomes[position]
            expected = [i for i, deps in enumerate(dependency_sets) if int(selected[position]) in deps]
            d = np.asarray(d)
            if involved != expected or d.shape != (len(involved), 6) or d.dtype != foot_j.dtype or not np.isfinite(d).all():
                raise ValueError('Original station derivative dependency/shape/type/domain changed in worker')
            foot_j[involved, :, position] = d[:, :2]
            endpoint_j[involved, :, :, position] = d[:, 2:].reshape(-1, 2, 2)
            evaluations += evaluated
        # Qualify this new interface once in the installing process, during a
        # real production call. Every later q still checks all private nominal
        # arrays/proofs and runs the original full guards; it does not claim a
        # fresh serial derivative comparison on that later state.
        if _CALLER_COLUMN_QUALIFICATION_Q is None:
            last = positions[-1]
            serial_last = original(truth, q, contexts, selected[last:last + 1])
            if (not _exact(serial_last['fixed_foot_jacobian'][:, :, 0], foot_j[:, :, last])
                    or not _exact(serial_last['endpoint_jacobian'][:, :, :, 0], endpoint_j[:, :, :, last])):
                raise ValueError('Private final station FD column differs from original caller execution')
            caller_column_checked = True
        result = dict(arrays, fixed_foot_jacobian=foot_j, endpoint_jacobian=endpoint_j,
            report=dict(schema='rinkspace.finite-rectangle-projected-station-FD.v1',
                selected_columns=selected.tolist(), actual_FD_columns=[int(selected[i]) for i in positions],
                real_FD_evaluations=evaluations,
                dependent_selected_columns=[[int(c) for c in selected if int(c) in deps]
                                            for deps in dependency_sets],
                projection_validation=[r['proof'] for r in nominal],
                complete_q_world_and_positive_pivot_validation_per_sample=True,
                original_coordinate_steps_and_positive_pivot_FD_retained=True,
                source_UV_normals_world_role_and_dynamic_guard_unchanged=True,
                projected_segment_interpolation_not_world_point_projection=True))
        succeeded = True
        if caller_column_checked:
            _CALLER_COLUMN_QUALIFICATION_Q = q_digest
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
            projection._project(truth, q, contexts, helpers)
        _record(route='original_FD_private_columns_result', succeeded=succeeded,
            actual_worker_pids=sorted(worker_pids), current_q_sha256=q_digest,
            original_FD_columns=len(positions), station_points=len(contexts),
            seconds=time.perf_counter() - started, pending_jobs=0,
            private_context_execution_fallback_reasons=failures,
            complete_nominal_arrays_and_physical_proof_byte_checked=succeeded,
            original_caller_last_column_byte_checked=caller_column_checked,
            caller_column_qualification_q_sha256=_CALLER_COLUMN_QUALIFICATION_Q,
            later_q_derivatives_not_claimed_serial_rechecked=True,
            original_same_pool_retained_for_next_stage=succeeded,
            source_values_supports_guards_and_FD_math_changed=False,
            whole_cycle_speedup_pending=True, accuracy_accepted=False)
    if failures:
        _record(route='original_serial', reason='All owned station jobs released: ' + ' | '.join(failures))
        return original(truth, q, contexts, selected)
    return result


def install_station_columns_runtime(*, report_directory=None):
    """Install before original finite-station calls; start no worker or source."""
    global _ORIGINAL, _RECEIPT, _REPORT_DIRECTORY
    destination = None if report_directory is None else Path(report_directory).resolve()
    if _RECEIPT is not None:
        if _RECEIPT['report_directory'] != (None if destination is None else str(destination)):
            raise RuntimeError('Station column scheduling already belongs to another run')
        return _RECEIPT
    projection = _projection_module()
    _ORIGINAL = projection.projected_station_contexts
    source = inspect.getsource(_ORIGINAL)
    projection.projected_station_contexts = functools.wraps(_ORIGINAL)(_parallel_station_contexts)
    _REPORT_DIRECTORY = destination
    _RECEIPT = dict(schema='rinkspace.private-original-finite-station-FD-columns.v1',
        report_directory=None if destination is None else str(destination),
        helper_path=str(Path(__file__).resolve()), helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_projection_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_project_physical_certificates_endpoint_masks_and_positive_pivot_FD_reused=True,
        original_typed_camera_context_and_existing_owned_material_pool_required=True,
        workers_receive_only_assigned_columns_and_small_numeric_context=True,
        no_source_graph_or_new_executor_manager_constructed=True,
        complete_nominal_arrays_and_proof_checked_on_every_production_call=True,
        one_real_caller_column_checked_only_on_first_successful_production_route=True,
        actual_full_column_equivalence_and_complete_cycle_speedup_pending=True,
        accuracy_accepted=False)
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'station_columns_runtime_receipt.json').write_text(
            json.dumps(_RECEIPT, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return _RECEIPT
