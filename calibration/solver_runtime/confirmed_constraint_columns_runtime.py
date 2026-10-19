"""Original confirmed-source FD columns on private, small worker contexts.

Only filling J is scheduled. Original constraints still assemble source bounds,
retained faces, curvature carriers and reports in their original row order.
Workers load the existing material snapshot once; no programme graph, motion
observations, actor buffers, profiles or source CSR is serialized here.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import textwrap
from types import MethodType, SimpleNamespace

import numpy as np


_MATERIALS = {}
_CONTEXTS = {}
_RECEIPT = None
_REPORT_DIRECTORY = None


class _UnsupportedContext(TypeError):
    pass


class _PrivateContextAdmissionError(RuntimeError):
    """Execution integration failure before any original FD column is run."""


class _PrivateSourceIdentityError(ValueError):
    """Altered original evidence is never an execution fallback."""


def _private_material_owner(snapshot):
    # All jobs on one worker execute serially. Reuse the same already-private
    # original owner across material, confirmed and native protocols, retaining
    # the original absolute helper's per-owner certificate set.
    from . import absolute_geometry_fd_pool_candidate as source
    key = (snapshot['path'], snapshot['sha256'])
    existing = source._PRIVATE_GEOMETRIES.get(key)
    if existing is not None:
        _MATERIALS[key] = existing[0]
        return existing[0]
    if key not in _MATERIALS:
        try:
            payload = Path(snapshot['path']).read_bytes()
        except Exception as error:
            raise _PrivateContextAdmissionError('Private material read failed: '
                + type(error).__name__ + ': ' + str(error)) from error
        if hashlib.sha256(payload).hexdigest() != snapshot['sha256']:
            raise _PrivateSourceIdentityError('Original private material snapshot changed before worker load')
        try:
            _MATERIALS[key] = pickle.loads(payload)
        except Exception as error:
            raise _PrivateContextAdmissionError('Private material unpickle failed: '
                + type(error).__name__ + ': ' + str(error)) from error
    source._PRIVATE_GEOMETRIES[key] = (_MATERIALS[key], set())
    return _MATERIALS[key]


def _copy_fields(source, target, names):
    for name in names:
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))


def _check_pattern_numerical_closure(factors, geometries):
    from secondary_optics_candidate.secondary_optics_graph import OpticsPatternFactor
    from z190_shared_board_pattern_source import SharedBoardPatternSourceFactor, SharedPerimeterBoardPatternGeometry
    from z190_external_board_pattern_source import ExternalBoardPatternDistanceSourceFactor
    from shared_board_course_candidate.shared_rink_course_numeric_geometry import SharedRinkCourseNumericGeometry
    visited = set()
    def factor(value):
        if value is None or id(value) in visited:
            return
        visited.add(id(value))
        if type(value) is OpticsPatternFactor:
            if set(vars(value)) != {'factor', 'steps'}:
                raise _UnsupportedContext('Pattern optics factor has additional unreviewed payload fields')
            factor(value.factor)
        elif type(value) in (SharedBoardPatternSourceFactor, ExternalBoardPatternDistanceSourceFactor):
            if set(vars(value)) != {'geometry', 'samples', 'theta'}:
                raise _UnsupportedContext('Pattern factor has additional unreviewed payload fields')
            geometry(value.geometry)
        else:
            raise _UnsupportedContext('Unreviewed typed pattern factor cannot enter private confirmed context')
    def geometry(value):
        if value is None or id(value) in visited:
            return
        visited.add(id(value))
        if not isinstance(value, (SharedPerimeterBoardPatternGeometry, SharedRinkCourseNumericGeometry)):
            # The course's original ice model is numeric, but is not needed
            # by pattern material functions and is not traversed here.
            raise _UnsupportedContext('Unreviewed pattern geometry linkage')
        graph_fields = {'state', 'bundle', 'blocks', 'bypts', 'actors', 'frames', 'fit', 'held',
                        'profile_update', 'motion_kernel', 'source', 'base_source', 'owner'}
        if graph_fields.intersection(vars(value)):
            raise _UnsupportedContext('Pattern geometry contains a programme/source graph backreference')
        if isinstance(value, SharedPerimeterBoardPatternGeometry):
            for field in ('original', 'course'):
                if field in vars(value):
                    geometry(vars(value)[field])
    for value in factors:
        factor(value)
    for value in geometries:
        geometry(value)


def _small_context(owner):
    """Whitelist original numerical inputs, omitting every full-state backref."""
    from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock
    from z190_fixed_lever_xyz_source_bundle import FixedLeverXYZSourceBundle
    from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
    from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
    from structured_programme_source_candidate.native_board_circle_source import NativeBoardCircleFactor
    from structured_programme_source_candidate.native_board_material_point_source import NativeBoardMaterialPointFactor
    from structured_programme_source_candidate.course_patterns_independent_placements import (
        SourceWithIndependentPatternPlacements, _bind_independent_current_geometry)

    state = SimpleNamespace(geometry=None,
        structured_programme_coordinate_chart=owner.state.structured_programme_coordinate_chart,
        geometry_parameter_indices=np.asarray(owner.state.geometry_parameter_indices).copy())
    block_clones = {}
    factor_clones = {}
    pool = None

    def block(original):
        nonlocal pool
        if id(original) in block_clones:
            return block_clones[id(original)]
        if type(original) is not StructuredProgrammeBlock or type(original.bundle) is not FixedLeverXYZSourceBundle:
            raise _UnsupportedContext('Only original typed programme camera/fixed-XYZ bundle contexts are supported')
        current_pool = getattr(original, '_absolute_geometry_fd_pool', None)
        if current_pool is None or (pool is not None and pool is not current_pool):
            raise _UnsupportedContext('All confirmed cameras must share the original owned material pool')
        pool = current_pool
        result = type(original).__new__(type(original))
        _copy_fields(original, result, ('xyz_slots', 'start', 'q0', 'optics_chart',
                                       'optics_base_calibration', 'geometry_parameter_indices'))
        result.geometry = None
        result.bundle = type(original.bundle).__new__(type(original.bundle))
        _copy_fields(original.bundle, result.bundle, ('calibration', 'index'))
        result._native_camera_valid_worlds = OrderedDict()
        block_clones[id(original)] = result
        return result

    def pattern(original):
        # The original binding changes geometry/factor/external_factor locally.
        # Preserve those typed numerical objects, but no fit/held/frame/state graph.
        if getattr(original.bind_current_geometry, '__func__', None) is not _bind_independent_current_geometry:
            raise _UnsupportedContext('The reviewed original independent-placement binding is required')
        _check_pattern_numerical_closure(
            [original.factor, original.external_factor, original._original_course_pattern_factor,
             original._original_course_external_pattern_factor],
            [original.geometry, original._original_course_pattern_geometry])
        result = type(original).__new__(type(original))
        _copy_fields(original, result, ('geometry', 'factor', 'external_factor',
            '_original_course_pattern_geometry', '_original_course_pattern_factor',
            '_original_course_external_pattern_factor', '_independent_pair_offsets'))
        result._independent_pair_offsets = original._independent_pair_offsets.copy()
        result.state = state
        result.course_world_geometry = None
        # Bound MethodType pickles through its function name, whereas the
        # original override is installed under bind_current_geometry. Bind it
        # after unpickling, keeping only original typed numerical fields here.
        return result

    supported = (NativeBoardEdgeFactor, NativeIceCurveFactor, NativeBoardCircleFactor, NativeBoardMaterialPointFactor)
    private_pattern = None

    def factor(original):
        nonlocal private_pattern
        if id(original) in factor_clones:
            return factor_clones[id(original)]
        if type(original) not in supported:
            raise _UnsupportedContext('Unreviewed confirmed native factor cannot enter the lightweight worker')
        result = type(original).__new__(type(original))
        _copy_fields(original, result, ('parameter_count', 'core_count', 'world_indices',
            'world_parameter_count', 'unit', 'packet_path', '_physical_course_scope_overrides',
            'physical_course_scope_authority', '_printed_contour_materials',
            'active_world', 'circle_indices', 'active_circle_indices', 'radius_shared',
            'pattern_start', 'pattern_columns'))
        result.state = state
        result.blocks = {exposure: block(value) for exposure, value in original.blocks.items()}
        result.geometry = None
        result._cache = OrderedDict()
        if isinstance(original, NativeIceCurveFactor):
            result._ice_material_view = None
        if isinstance(original, NativeBoardMaterialPointFactor):
            if private_pattern is None:
                private_pattern = pattern(original.patterns)
            if owner.placement_source is None or original.patterns is not owner.placement_source.patterns:
                raise _UnsupportedContext('One original independent pattern/material placement owner is required')
            result.patterns = private_pattern
        factor_clones[id(original)] = result
        return result

    result = type(owner).__new__(type(owner))
    _copy_fields(owner, result, ('parameter_count', 'world_indices', 'height_columns', 'active', 'steps'))
    result.state = state
    result.blocks = {exposure: block(value) for exposure, value in owner.blocks.items()}
    result.records = [dict(item, factor=factor(item['factor'])) for item in owner.records]
    result.placement_source = None
    if private_pattern is not None:
        original = owner.placement_source
        method = getattr(original, 'pattern_placement_context', None)
        placement_owner = getattr(method, '__self__', None)
        if (type(placement_owner) is not SourceWithIndependentPatternPlacements
                or getattr(method, '__func__', None) is not SourceWithIndependentPatternPlacements.pattern_placement_context
                or placement_owner.patterns is not original.patterns
                or placement_owner.material_factor is not original.material_factor):
            raise _UnsupportedContext('Original typed independent placement bound context required')
        # The complete source delegates this original bound context through
        # native-track, static-role and persistent-ice wrappers. Only its typed
        # numerical owner is needed privately; retain every live wrapper.
        placement = type(placement_owner).__new__(type(placement_owner))
        _copy_fields(placement_owner, placement, ('count', 'placement_count', 'placement_indices'))
        placement.patterns = private_pattern
        placement.material_factor = factor(placement_owner.material_factor)
        result.placement_source = placement
    return result, pool, list(block_clones.values()), list(factor_clones.values()), private_pattern


def _hydrate(context, geometry):
    owner, blocks, factors, pattern = context
    owner.state.geometry = geometry
    for block in blocks:
        block.geometry = geometry
    for factor in factors:
        factor.geometry = geometry.current
    if pattern is not None:
        pattern.course_world_geometry = geometry
        from structured_programme_source_candidate.course_patterns_independent_placements import _bind_independent_current_geometry
        pattern.bind_current_geometry = MethodType(_bind_independent_current_geometry, pattern)
    return owner


def _evaluate_columns(owner, q, positions, selected, base):
    from structured_programme_source_candidate.human_stroke_truth import _truth_record_memo_nominal
    from structured_programme_source_candidate.fixed_pivot_positive_domain_fd import positive_pivot_coordinate_derivative
    _truth_record_memo_nominal(owner, q)
    memo = {}
    # Original nominal admission authenticates every camera/source in this worker.
    try:
        current = owner._carriers(q, record_memo=memo)[0]
    except Exception as error:
        raise _PrivateContextAdmissionError(
            'Private nominal admission failed after original caller nominal accepted: '
            + type(error).__name__ + ': ' + str(error)) from error
    if current.shape != base.shape or current.dtype != base.dtype or current.tobytes() != base.tobytes():
        raise _PrivateContextAdmissionError('Private original confirmed carrier differs from the caller nominal carrier')
    results = []
    for position in positions:
        column = int(selected[position])
        try:
            derivative = positive_pivot_coordinate_derivative(
                lambda value: owner._carriers(value, record_memo=memo)[0],
                q, column, owner.steps.get(column, 1e-6), owner.height_columns, base=base)
            results.append((int(position), derivative, None))
        except Exception as error:
            # Return original numeric errors by column, so the parent chooses
            # the first error in the original column order. BaseException is
            # never changed into a numerical outcome or fallback.
            results.append((int(position), None, error))
    return results


def evaluate_confirmed_columns_worker(job):
    """New protocol; does not wrap or alter the absolute-material job protocol."""
    from .absolute_geometry_fd_pool_candidate import initialize_absolute_geometry_worker
    from .constraint_sample_reuse_runtime import install_constraint_sample_reuse_runtime, _with_literal_construction
    initialize_absolute_geometry_worker()
    install_constraint_sample_reuse_runtime()
    snapshot = job['material_owner']
    material_key = (snapshot['path'], snapshot['sha256'])
    geometry = _private_material_owner(snapshot)
    payload = job['context_payload']
    digest = hashlib.sha256(payload).hexdigest()
    if digest != job['context_sha256']:
        raise ValueError('Private confirmed numerical context changed during transport')
    context_key = (material_key, digest, job['q_sha256'])
    if context_key not in _CONTEXTS:
        # One current q per worker, not a growing sequence of state clones.
        _CONTEXTS.clear()
        try:
            _CONTEXTS[context_key] = _hydrate(pickle.loads(payload), geometry)
        except Exception as error:
            raise _PrivateContextAdmissionError('Private confirmed hydrate failed: ' + type(error).__name__ + ': ' + str(error)) from error
    q = np.asarray(job['q'], float)
    if hashlib.sha256(q.tobytes()).hexdigest() != job['q_sha256']:
        raise ValueError('Complete original q changed during confirmed column transport')
    owner = _CONTEXTS[context_key]
    results = _with_literal_construction(_evaluate_columns, None)(
        owner, q, job['positions'], np.asarray(job['selected'], int), np.asarray(job['base'], float))
    return dict(context_sha256=digest, q_sha256=job['q_sha256'],
                material_sha256=snapshot['sha256'], pid=os.getpid(), results=results)


def _serial_columns(owner, q, selected, base, J, active, memo):
    from structured_programme_source_candidate.fixed_pivot_positive_domain_fd import positive_pivot_coordinate_derivative
    for index, column in enumerate(selected):
        if int(column) in active:
            J[:, index] = positive_pivot_coordinate_derivative(
                lambda value: owner._carriers(value, record_memo=memo)[0],
                q, int(column), owner.steps.get(int(column), 1e-6), owner.height_columns, base=base)


def _original_route(owner, reason):
    event = dict(stage='confirmed_constraint_column_route', route='original_serial',
                 reason=reason, source_records=len(owner.records))
    if _REPORT_DIRECTORY is not None:
        with (_REPORT_DIRECTORY / 'actual_confirmed_private_column_execution.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')
    print(json.dumps(event, ensure_ascii=False), flush=True)


def _parallel_columns(owner, q, selected, base, J, active, memo):
    from .absolute_geometry_fd_pool_candidate import AbsoluteGeometryGroupPool
    try:
        small, pool, blocks, factors, pattern = _small_context(owner)
    except Exception as error:
        # Packing invokes no original numeric producer. Missing lightweight
        # fields must not abort an otherwise valid complete source solve.
        _original_route(owner, 'Light context packing failed: ' + type(error).__name__ + ': ' + str(error))
        return _serial_columns(owner, q, selected, base, J, active, memo)
    if (type(pool) is not AbsoluteGeometryGroupPool or pool.closed or not pool.owns_executor
            or pool.workers <= 1 or getattr(pool, '_rinkspace_active_geometry_group_iterators', 0)
            or getattr(pool, '_rinkspace_confirmed_columns_active', False)
            or getattr(pool, '_rinkspace_native_columns_active', False)
            or getattr(pool, '_rinkspace_shared_material_bank_precompute_active', False)
            or any(int(getattr(pool, name, 0)) != 0 for name in (
                '_rinkspace_confirmed_columns_pending_jobs', '_rinkspace_native_columns_pending_jobs',
                '_rinkspace_shared_material_bank_pending_jobs'))
            or getattr(pool, '_rinkspace_force_parent_activation_FD', False)):
        _original_route(owner, 'Original owned material pool is not available and idle for private column jobs')
        return _serial_columns(owner, q, selected, base, J, active, memo)
    if pool.executor is not None:
        last = getattr(pool, 'last_completion_scheduling', {})
        absolute_idle = last.get('active_futures_at_exit') == 0 and last.get('buffered_batches_at_exit') == 0
        private_idle = any(receipt.get('executor_identity') == id(pool.executor) and receipt.get('pending_jobs') == 0
            for receipt in (getattr(pool, name, {}) for name in (
                '_rinkspace_confirmed_columns_idle_receipt', '_rinkspace_native_columns_idle_receipt',
                '_rinkspace_shared_material_bank_idle_receipt')))
        if not (absolute_idle or private_idle):
            _original_route(owner, 'Existing owned material executor has no complete zero-pending-job receipt')
            return _serial_columns(owner, q, selected, base, J, active, memo)
    positions = [i for i, column in enumerate(selected) if int(column) in active]
    if not positions:
        return
    try:
        payload = pickle.dumps((small, blocks, factors, pattern), protocol=5)
    except Exception as error:
        _original_route(owner, 'Light context serialization failed: ' + type(error).__name__ + ': ' + str(error))
        return _serial_columns(owner, q, selected, base, J, active, memo)
    digest = hashlib.sha256(payload).hexdigest()
    q_digest = hashlib.sha256(np.asarray(q, float).tobytes()).hexdigest()
    executor = pool._executor()
    if executor is None:
        _original_route(owner, 'Original material pool memory admission returned parent execution')
        return _serial_columns(owner, q, selected, base, J, active, memo)
    pool._rinkspace_confirmed_columns_active = True
    futures = []
    worker_pids = set()
    succeeded = False
    integration_failures = []
    try:
        print(json.dumps(dict(stage='confirmed_constraint_column_route', route='original_FD_private_columns',
            requested_workers=pool.workers, original_active_columns=len(positions),
            small_context_serialized_bytes=len(payload), source_records=len(owner.records),
            current_q_sha256=q_digest), ensure_ascii=False), flush=True)
        common = dict(material_owner=dict(pool.material_owner), context_payload=payload,
                      context_sha256=digest, q=np.asarray(q, float), q_sha256=q_digest,
                      selected=np.asarray(selected, int), base=np.asarray(base, float))
        for worker in range(min(pool.workers, len(positions))):
            group = positions[worker::pool.workers]
            if group:
                futures.append((group, executor.submit(evaluate_confirmed_columns_worker, dict(common, positions=group))))
        outcomes = {}
        for group, future in futures:
            try:
                result = future.result()
            except _PrivateContextAdmissionError as error:
                integration_failures.append(str(error))
                continue
            if (result['context_sha256'] != digest or result['q_sha256'] != q_digest
                    or result['material_sha256'] != pool.material_owner['sha256']
                    or [r[0] for r in result['results']] != group):
                raise ValueError('Original confirmed column identity/order changed during transport')
            worker_pids.add(result['pid'])
            for position, derivative, error in result['results']:
                if position in outcomes:
                    raise ValueError('A confirmed source column was returned twice')
                outcomes[position] = (derivative, error)
        if integration_failures:
            # Drain every submitted job first. A real original FD error in an
            # admitted worker is still a mathematical failure, not a reason
            # to hide it behind a context fallback.
            for position in positions:
                if position in outcomes and outcomes[position][1] is not None:
                    raise outcomes[position][1]
            raise _PrivateContextAdmissionError(' | '.join(dict.fromkeys(integration_failures)))
        if set(outcomes) != set(positions):
            raise ValueError('An original active confirmed source column was omitted')
        for position in positions:
            derivative, error = outcomes[position]
            if error is not None:
                raise error
            derivative = np.asarray(derivative)
            if derivative.shape != base.shape or derivative.dtype != J.dtype or not np.isfinite(derivative).all():
                raise ValueError('Original confirmed derivative shape/type/domain changed in worker')
            J[:, position] = derivative
        # The serial constraints producer leaves camera/calibration bindings at
        # the final active column's final original sample. Replay that exact
        # last column on the caller to preserve that binding and independently
        # verify one real worker column, retaining original print-domain retries.
        last = positions[-1]
        serial_last = np.zeros_like(J[:, last:last + 1])
        _serial_columns(owner, q, selected[last:last + 1], base, serial_last, active, memo)
        if serial_last[:, 0].tobytes() != J[:, last].tobytes():
            raise ValueError('Private final confirmed FD column differs from its original caller execution')
        succeeded = True
    except _PrivateContextAdmissionError as error:
        integration_failures = [str(error)]
    finally:
        try:
            if succeeded:
                # All futures were consumed; keep this same idle executor for
                # the next original source stage. Its original J release and
                # driver's overall finally remain the cleanup owners.
                pool._rinkspace_confirmed_columns_idle_receipt = dict(
                    executor_identity=id(executor), pending_jobs=0, current_q_sha256=q_digest)
            else:
                # Only an execution admission failure may return to serial.
                # Original numerical/control errors also clean up, then escape.
                for _, future in futures:
                    if not future.done():
                        future.cancel()
                pool.release_idle()
                pool.__dict__.pop('_rinkspace_confirmed_columns_idle_receipt', None)
        finally:
            pool._rinkspace_confirmed_columns_active = False
        if _REPORT_DIRECTORY is not None:
            event = dict(schema='rinkspace.private-original-confirmed-FD-columns.v1',
                succeeded=succeeded, requested_workers=pool.workers, actual_worker_pids=sorted(worker_pids),
                original_active_columns=len(positions), source_records=len(owner.records),
                complete_scalar_carrier_rows=len(base), current_q_sha256=q_digest,
                small_context_serialized_bytes=len(payload), original_material_snapshot_bytes=pool.material_owner['serialized_bytes'],
                original_column_math_bounds_faces_group_order_and_queries_retained=True,
                original_caller_final_column_and_camera_binding_replayed=succeeded,
                successful_idle_executor_retained_for_original_next_stage=succeeded,
                private_context_execution_fallback_reasons=integration_failures,
                full_programme_graph_profiles_motion_arrays_or_CSR_transmitted=False,
                material_owner_loaded_once_per_worker=True, actual_speedup_pending=True)
            try:
                with (_REPORT_DIRECTORY / 'actual_confirmed_private_column_execution.jsonl').open('a', encoding='utf8') as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            except OSError:
                if succeeded:
                    raise
    if integration_failures:
        _original_route(owner, 'Private context admission failed; all owned jobs released: ' + ' | '.join(integration_failures))
        # No parallel column has been installed when an admission fails.
        # Recompute the complete original selected-column loop, not a subset.
        J.fill(0.)
        return _serial_columns(owner, q, selected, base, J, active, memo)


def install_confirmed_constraint_columns_runtime(*, report_directory=None):
    """Install after sample/literal reuse, before constructing this run sources."""
    global _RECEIPT, _REPORT_DIRECTORY
    destination = None if report_directory is None else Path(report_directory).resolve()
    if _RECEIPT is not None:
        if _RECEIPT['report_directory'] != (None if destination is None else str(destination)):
            raise RuntimeError('Confirmed column scheduling is already installed for another run')
        return _RECEIPT
    from structured_programme_source_candidate.human_stroke_truth import UserConfirmedSourceTruth
    original = inspect.unwrap(UserConfirmedSourceTruth.constraints)
    if original.__code__.co_freevars:
        raise TypeError('Original source-owned confirmed constraints function required')
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    target = tree.body[0]
    expected_loop = ast.parse('for index,column in enumerate(selected):\n    pass').body[0]
    loops = [(i, n) for i, n in enumerate(target.body) if isinstance(n, ast.For)
             and ast.dump(n.target) == ast.dump(expected_loop.target)
             and ast.dump(n.iter) == ast.dump(expected_loop.iter)]
    if len(loops) != 1:
        raise TypeError('Exactly one original selected confirmed FD column loop is required')
    target.body[loops[0][0]] = ast.parse('_rinkspace_parallel_confirmed_columns(self,q,selected,value,J,active,record_memo)').body[0]
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, original.__code__.co_firstlineno - 1)
    original.__globals__['_rinkspace_parallel_confirmed_columns'] = _parallel_columns
    namespace = {}
    exec(compile(tree, inspect.getsourcefile(original), 'exec'), original.__globals__, namespace)
    changed = namespace[target.name]
    # Retain the already-installed outer literal-source construction wrapper.
    original.__code__ = changed.__code__
    helper = Path(__file__).resolve()
    _REPORT_DIRECTORY = destination
    _RECEIPT = dict(schema='rinkspace.private-original-confirmed-FD-columns.v1',
        report_directory=None if destination is None else str(destination),
        helper_path=str(helper), helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        original_constraints_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        ast_change='Only replace original selected-column J-fill loop; retain installed outer literal wrapper',
        existing_owned_material_executor_and_snapshot_reused=True,
        worker_job_protocol_separate_from_original_absolute_material_protocol=True,
        original_complete_derivative_including_physical_print_retry_preserved=True,
        original_matrix_face_mask_lower_group_report_and_curvature_assembly_retained=True,
        worker_private_geometry_camera_calibration_pattern_placement_and_memos=True,
        private_context_admission_failure_releases_jobs_then_complete_original_serial_columns=True,
        original_FD_mathematical_and_physical_domain_exceptions_propagate_in_column_order=True,
        successful_idle_executor_retained_for_original_J_release_and_overall_finally=True,
        full_programme_graph_motion_profile_or_CSR_arrays_transmitted=False,
        original_small_truth_UV_normals_bounds_and_metadata_transmitted=True,
        actual_production_equivalence_and_speedup_pending=True, accuracy_accepted=False)
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'confirmed_constraint_columns_runtime_receipt.json').write_text(
            json.dumps(_RECEIPT, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return _RECEIPT
