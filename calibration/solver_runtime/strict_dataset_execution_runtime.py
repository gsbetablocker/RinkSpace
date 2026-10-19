"""Execute strict ice material FD on the existing private material pool.

Each perturbed world produces the complete original ice inventory once for
all requested targets. Pixels, camera projection, source ordering and Gram
remain with the strict dataset factor in the parent process.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, wait
import hashlib
import math
from pathlib import Path
import time
from types import MethodType

import numpy as np

_ICE_VIEWS = {}


def _digest(array):
    array = np.asarray(array)
    return (array.dtype.str, array.shape, hashlib.sha256(array.tobytes()).hexdigest())


def ice_target_worlds(ice, world, targets):
    """The factor's unchanged producer, shared across target lookups."""
    from rinkspace.vision_programme_ice_review import ice_curve_inventory
    features = ice.features(world)
    inventory = ice_curve_inventory(dict(ice_features=list(features.values())))
    lookup = {}
    for record in inventory:
        lookup.setdefault(record['curve_id'], []).append(record)
    result = []
    for target in targets:
        matches = lookup.get(target['curve_id'], [])
        if (len(matches) != 1 or matches[0]['feature_id'] != target['feature_id']
                or matches[0]['world_curve_role'] != target['world_curve_role']):
            raise ValueError('Exact current modeled ice material curve missing')
        result.append(np.asarray(matches[0]['xyz'], float))
    return result


def evaluate_strict_ice_columns_worker(job):
    from .absolute_geometry_fd_pool_candidate import initialize_absolute_geometry_worker
    from .confirmed_constraint_columns_runtime import _private_material_owner
    from structured_programme_source_candidate.current_ice_material_scene import CurrentIceMaterialView
    initialize_absolute_geometry_worker()
    snapshot = job['material_owner']
    token = (snapshot['path'], snapshot['sha256'])
    if token not in _ICE_VIEWS:
        _ICE_VIEWS.clear()
        owner = _private_material_owner(snapshot)
        _ICE_VIEWS[token] = CurrentIceMaterialView(owner.current)
    ice = _ICE_VIEWS[token]
    world = np.asarray(job['world'], float)
    if _digest(world) != job['world_identity']:
        raise ValueError('Strict FD job changed complete current world bytes')
    nominal = ice_target_worlds(ice, world, job['targets'])
    if [_digest(value) for value in nominal] != job['nominal_identities']:
        raise ValueError('Private strict ice producer differs from parent nominal world')
    started = time.perf_counter()
    values = [np.empty((*value.shape, len(job['positions']))) for value in nominal]
    failures = []
    for offset, column in enumerate(job['positions']):
        plus, minus = world.copy(), world.copy()
        plus[column] += 1e-6
        minus[column] -= 1e-6
        try:
            wp = ice_target_worlds(ice, plus, job['targets'])
            wm = ice_target_worlds(ice, minus, job['targets'])
            for ordinal, (p, m, original) in enumerate(zip(wp, wm, nominal)):
                if p.shape != original.shape or m.shape != original.shape:
                    raise ValueError('Current material derivative changed finite curve knots')
                values[ordinal][:, :, offset] = (p - m) / (2e-6)
        except Exception as error:
            failures.append((int(column), error))
    return dict(positions=job['positions'], values=values, failures=failures,
                world_identity=job['world_identity'], seconds=time.perf_counter()-started)


def _existing_pool(factor):
    from .native_factor_columns_runtime import _pool_available
    pools = {id(pool): pool for block in factor.state.blocks
             if (pool := getattr(block, '_absolute_geometry_fd_pool', None)) is not None}
    if len(pools) != 1:
        return None
    pool = next(iter(pools.values()))
    if not _pool_available(pool) or not pool._headroom('strict_ice_FD_existing_pool'):
        return None
    if pool.executor is not None:
        queued = getattr(pool.executor, '_pending_work_items', None)
        if not isinstance(queued, dict) or queued:
            return None
    return pool


def _prepare_all_ice_derivatives(factor, q):
    from .strict_source_dataset_adapter import _key
    pool = _existing_pool(factor)
    if pool is None:
        return False
    targets = list(OrderedDict((_key(row['model_target']), row['model_target'])
                   for row in factor.rows if row['model_target']['kind'] == 'current_scene_ice_curve').values())
    if not targets:
        return False
    executor = pool._executor()
    if executor is None:
        return False
    keys = [_key(target) for target in targets]
    nominal = []
    for key, target in zip(keys, targets):
        if key not in factor._world_cache:
            factor._world_cache[key] = factor._produce_world(q, target)
        value, analytical = factor._world_cache[key]
        if analytical is not None:
            raise ValueError('Strict ice FD execution cannot replace an analytical producer')
        nominal.append(np.asarray(value))
    columns = np.asarray(factor.world_indices, int)
    world = q[columns].copy()
    width = max(1, math.ceil(len(columns) / (2 * pool.workers)))
    batches = [list(range(start, min(start+width, len(columns))))
               for start in range(0, len(columns), width)]
    iterator = iter(batches)
    pending = {}
    derivatives = [np.empty((*value.shape, len(columns))) for value in nominal]
    failures = []
    started = time.perf_counter()
    def submit():
        positions = next(iterator, None)
        if positions is None:
            return False
        job = dict(material_owner=pool.material_owner, world=world,
                   world_identity=_digest(world), targets=targets,
                   nominal_identities=[_digest(value) for value in nominal], positions=positions)
        pending[executor.submit(evaluate_strict_ice_columns_worker, job)] = positions
        return True
    submitted = []
    original_submit = submit
    def tracked_submit():
        previous = set(pending)
        added = original_submit()
        if added:
            submitted.extend(set(pending) - previous)
            pool._rinkspace_native_columns_pending_jobs += 1
        return added
    succeeded = False
    pool._rinkspace_native_columns_active = True
    pool._rinkspace_native_columns_pending_jobs = 0
    pool.__dict__.pop('_rinkspace_native_columns_idle_receipt', None)
    try:
        for _ in range(pool.max_inflight):
            if not tracked_submit():
                break
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                positions = pending.pop(future)
                try:
                    result = future.result()
                    if result['positions'] != positions or result['world_identity'] != _digest(world):
                        raise ValueError('Strict ice FD original column identity changed')
                    failures.extend(result['failures'])
                    for derivative, values in zip(derivatives, result['values']):
                        derivative[:, :, positions] = values
                except Exception as error:
                    failures.append((positions[0], error))
                finally:
                    pool._rinkspace_native_columns_pending_jobs -= 1
                tracked_submit()
        if failures:
            raise min(failures, key=lambda record: record[0])[1]
        # Qualify the private producer against a complete original parent column
        # for every target before using any worker derivative at this current q.
        for target, derivative in zip(targets, derivatives):
            active = np.flatnonzero(np.any(derivative != 0, axis=(0, 1)))
            if len(active):
                position = int(active[0]); column = columns[position]
                plus, minus = q.copy(), q.copy()
                plus[column] += 1e-6; minus[column] -= 1e-6
                expected = (np.asarray(factor._produce_world(plus, target)[0])
                            - np.asarray(factor._produce_world(minus, target)[0])) / (2e-6)
                if not np.array_equal(expected, derivative[:, :, position]):
                    raise ValueError('Private strict ice column differs from original parent arithmetic')
        if (pending or pool._rinkspace_native_columns_pending_jobs != 0
                or not all(future.done() for future in submitted)):
            raise RuntimeError('Strict ice FD cannot certify an executor with pending jobs')
        queued = getattr(executor, '_pending_work_items', None)
        if pool.executor is not executor or not isinstance(queued, dict) or queued:
            raise RuntimeError('Strict ice FD executor still has queued or foreign jobs')
        for key, derivative in zip(keys, derivatives):
            active = np.any(derivative != 0, axis=(0, 1))
            factor._derivative_cache[key] = (columns[active], derivative[:, :, active])
        succeeded = True
    finally:
        try:
            if succeeded:
                pool._rinkspace_native_columns_idle_receipt = dict(
                    executor_identity=id(executor), pending_jobs=0,
                    current_q_sha256=hashlib.sha256(q.tobytes()).hexdigest())
            else:
                # A failed submission, result, qualification or cancellation
                # must leave no unobserved work behind a fabricated idle flag.
                for future in pending:
                    future.cancel()
                for future in pending:
                    try:
                        future.result()
                    except BaseException:
                        pass
                pool.release_idle()
                pool.__dict__.pop('_rinkspace_native_columns_idle_receipt', None)
        finally:
            pool._rinkspace_native_columns_pending_jobs = 0
            pool._rinkspace_native_columns_active = False
    factor.strict_execution_last_ice_FD = dict(
        seconds=time.perf_counter()-started, targets=len(targets), world_columns=len(columns),
        existing_pool_workers=pool.workers, jobs=len(batches),
        original_complete_world_bytes_checked=True,
        all_parent_nominal_worlds_byte_verified=True,
        original_parent_column_per_target_byte_verified=True,
        original_step_and_numerical_producers_retained=True,
        new_executor_manager_created=False, source_pixels_or_rows_changed=False)
    factor.strict_execution_last_ice_FD.update(
        existing_native_private_column_idle_protocol_retained=True,
        all_submitted_futures_consumed=True, pending_jobs=0,
        same_executor_retained_for_confirmed_and_station_columns=True)
    return True


def install_strict_dataset_execution(factor):
    """Only this factor receives the hook; existing loaded modules stay intact."""
    if hasattr(factor, 'strict_execution_receipt'):
        return factor.strict_execution_receipt
    original_world = factor._world
    def scheduled_world(self, q, target, jacobian):
        from .strict_source_dataset_adapter import _key
        if self._evaluation_q is not q:
            self._prepare_evaluation(q)
        if (jacobian and target['kind'] == 'current_scene_ice_curve'
                and _key(target) not in self._derivative_cache):
            _prepare_all_ice_derivatives(self, q)
        return original_world(q, target, jacobian)
    factor._world = MethodType(scheduled_world, factor)
    factor.strict_execution_receipt = dict(
        schema='rinkspace.strict-dataset-existing-pool-ice-FD-execution.v1',
        helper_path=str(Path(__file__).resolve()),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        complete_current_world_and_all_target_parent_checks_required=True,
        absent_existing_material_pool_uses_original_serial_route=True,
        derivative_step_m=1e-6, source_rows_sigma_operator_and_chart_unchanged=True,
        original_camera_chain_and_projection_retained=True,
        candidate_camera_mapping_enabled=False,
        candidate_camera_mapping_disabled_reason=(
            'Candidate failed original complete Schur versus sparse-GN native-pixel consistency gate; '
            'original camera-coordinate FD retained'),
        numerical_equivalence_and_speedup_require_actual_execution=True)
    return factor.strict_execution_receipt
