"""Reuse an already produced role binding across index-only source compaction.

The original compaction body and its complete numerical checks still execute.
This hook changes its sole cache-clear assignment only when the original
binding's complete physics and source inputs remain equal. It never certifies
compaction and never carries a result across a changed parameter state.
"""
from __future__ import annotations

import ast
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import textwrap
import time

import numpy as np

_ACTIVE = ContextVar('rinkspace_compaction_role_reindex', default=None)
_RECEIPT = None
_REPORT_DIRECTORY = None
_EVENTS = []
_INDEX_FIELDS = frozenset(('active_global_bundle_index',
    'duplicate_runtime_representative_original_global_bundle_index'))
_ARRAY_FIELDS = ('planes', 'plane_jacobian', 'extent_world',
    'extent_world_jacobian', 'z_range_scene', 'z_range_world_jacobian',
    'selected_course_facets')
_BOUND_FIELDS = frozenset((*_ARRAY_FIELDS, 'records'))
_RECORD_FIELDS = frozenset(('global_bundle_index', 'surface_kind',
    'extent_start_xy', 'extent_end_xy', 'physical_surface_identity',
    'source_role_binding_id', 'source_role_block', 'source_native_pair_pts',
    'original_source_role', 'material_identity', 'actual_source_record',
    'current_course_facet_index', 'facet_index_is_current_geometry_not_measured_location'))


def _array_identity(value):
    a = np.asarray(value)
    if a.dtype.hasobject:
        raise ValueError('Object arrays are not eligible for role reindex reuse')
    return (a.shape, a.dtype.str, hashlib.sha256(a.tobytes(order='C')).hexdigest())


def _digest(value):
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _source_rows(rows):
    return [{key: value for key, value in row.items() if key not in _INDEX_FIELDS}
        for row in rows]


def _physics(state, role):
    geometry = state.geometry
    world = np.asarray(state.q)[state.geometry_parameter_indices]
    course, cq, chain = geometry.current.print_course_binding(world)
    values = course.sample_perimeter(cq, with_jacobian=True)
    # The original binding reads precisely bottom/top/Jbottom/Jtop. The complete
    # world, course coordinates, chain and all cameras are also bound here.
    original_key = (float(geometry.scene_unit_m), np.asarray(cq, float).tobytes(),
        np.asarray(chain, float).tobytes(), b''.join(np.asarray(role.source_cameras[stamp], float).tobytes()
            for stamp in sorted(role.source_row_groups)))
    return dict(original_key=original_key, q=_array_identity(state.q),
        indices=_array_identity(state.geometry_parameter_indices),
        world=_array_identity(world), geometry=id(geometry), current=id(geometry.current),
        course=id(course), parameter_count=int(geometry.parameter_count),
        scene_unit_m=float(geometry.scene_unit_m), cq=_array_identity(cq),
        chain=_array_identity(chain), perimeter=tuple(_array_identity(v) for v in values[:4]),
        cameras=tuple((stamp, _array_identity(camera))
            for stamp, camera in sorted(role.source_cameras.items())),
        source_uv=_array_identity(role.source_uv),
        groups=tuple((stamp, _array_identity(rows))
            for stamp, rows in sorted(role.source_row_groups.items())),
        source_stamps=_digest(role.source_stamps_by_block), binding_id=role.binding_id,
        methods=tuple(id(getattr(type(role), name)) for name in
            ('current_binding', '_build_current_binding', '_current_source_facets')),
        calibration=tuple((block.name, _digest(block.bundle.calibration.manifest()))
            for block in state.blocks))


def _bound_identity(bound):
    if set(bound) != _BOUND_FIELDS:
        raise ValueError('Unknown original role binding fields')
    if any(set(record) != _RECORD_FIELDS for record in bound['records']):
        raise ValueError('Unknown original role record fields')
    return tuple((key, _array_identity(bound[key])) for key in _ARRAY_FIELDS)


def _capture(state):
    role = getattr(state, 'original_motion_far_board_roles', None)
    cache = getattr(role, '_current_binding_cache', None)
    if role is None or cache is None:
        return dict(state=state, role=role, eligible=False, reason='no_existing_original_binding')
    try:
        key, bound = cache
        if len(key) != 4 or len(bound['records']) != len(role.records):
            raise ValueError('Incomplete original binding')
        if _digest(_source_rows([record['actual_source_record'] for record in bound['records']])) != _digest(_source_rows(role.records)):
            raise ValueError('Existing binding source rows do not match original role')
        if any(int(record['global_bundle_index']) != int(row.get('active_global_bundle_index', row['original_global_bundle_index']))
               for record, row in zip(bound['records'], role.records)):
            raise ValueError('Existing binding indices do not match original role')
        physics = _physics(state, role)
        if key != physics['original_key']:
            raise ValueError('Existing binding is not at the original current physics key')
        return dict(state=state, role=role, eligible=True, cache=cache, key=key,
            bound=bound, physics=physics, numeric=_bound_identity(bound),
            sources=_digest(_source_rows(role.records)), injected=None)
    except Exception as error:
        return dict(state=state, role=role, eligible=False,
            reason='capture_ineligible:' + type(error).__name__ + ': ' + str(error))


def _reindexed_binding(role):
    scope = _ACTIVE.get()
    if scope is None or not scope['eligible'] or role is not scope['role']:
        return None
    try:
        if role._current_binding_cache is not scope['cache']:
            raise ValueError('Original binding cache changed during compaction')
        if _physics(scope['state'], role) != scope['physics']:
            raise ValueError('Full binding physics input changed')
        if _digest(_source_rows(role.records)) != scope['sources']:
            raise ValueError('Qualified source metadata changed beyond runtime indices')
        if _bound_identity(scope['bound']) != scope['numeric']:
            raise ValueError('Original binding numeric arrays changed')
        old = scope['bound']
        records = []
        for previous, row in zip(old['records'], role.records):
            record = {}
            for key, value in previous.items():
                if key == 'global_bundle_index':
                    record[key] = int(row.get('active_global_bundle_index', row['original_global_bundle_index']))
                elif key == 'actual_source_record':
                    record[key] = deepcopy(row)
                elif key == 'material_identity':
                    # This is the original producer's direct source reference.
                    record[key] = row['physical_material']
                else:
                    record[key] = deepcopy(value)
            records.append(record)
        bound = dict(old)
        bound['records'] = records
        result = (scope['key'], bound)
        scope['injected'] = result
        scope['reason'] = 'same_full_physics_and_qualified_source_bytes_index_only'
        return result
    except Exception as error:
        scope['reason'] = 'reindex_ineligible:' + str(error)
        return None


def _transform(original):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    sites = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == 'role'
                and node.targets[0].attr == '_current_binding_cache'
                and isinstance(node.value, ast.Constant) and node.value.value is None):
            node.value = ast.copy_location(ast.Call(
                func=ast.Name(id='_rinkspace_compaction_reindexed_binding', ctx=ast.Load()),
                args=[ast.Name(id='role', ctx=ast.Load())], keywords=[]), node.value)
            sites += 1
    if sites != 1:
        raise RuntimeError('Expected exactly one original role cache-clear assignment')
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace['_rinkspace_compaction_reindexed_binding'] = _reindexed_binding
    exec(compile(tree, original.__code__.co_filename, 'exec'), namespace)
    return namespace[original.__name__]


def _wrap(original):
    transformed = _transform(original)

    @wraps(original)
    def wrapped(state, *args, **kwargs):
        started = time.perf_counter()
        scope = _capture(state)
        token = _ACTIVE.set(scope)
        success = False
        try:
            result = transformed(state, *args, **kwargs)
            success = True
            return result
        finally:
            _ACTIVE.reset(token)
            # Never leave an injected cache behind after any original later
            # mask, retirement, q, local/physical or control failure.
            if not success and scope.get('injected') is not None:
                scope['role']._current_binding_cache = None
            event = dict(stage='compaction_role_binding_reindex',
                original_compaction_completed=success,
                reused_existing_numeric_binding=success and scope.get('injected') is not None,
                reason=scope.get('reason', 'original_cache_clear_retained'),
                role_records=len(scope['role'].records) if scope['role'] is not None else 0,
                original_diagnostics_executed=True, diagnostic_PASS_reused=False,
                elapsed_with_original_compaction_seconds=time.perf_counter()-started)
            _EVENTS.append(event)
            try:
                if _REPORT_DIRECTORY is not None:
                    with (_REPORT_DIRECTORY / 'compaction_binding_reindex_execution.jsonl').open('a', encoding='utf8') as stream:
                        stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            except Exception:
                pass  # Reporting must not mask an original failure.

    wrapped._rinkspace_compaction_binding_reindex = True
    return wrapped


def install_compaction_binding_reindex_runtime(*, report_directory=None):
    global _RECEIPT, _REPORT_DIRECTORY
    from calibration.solver_runtime import runtime
    runtime.install()
    from structured_programme_source_candidate import original_motion_input_compaction as module
    if report_directory is not None:
        _REPORT_DIRECTORY = Path(report_directory)
        _REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if _RECEIPT is None:
        original = module.compact_original_motion_inputs
        module.compact_original_motion_inputs = _wrap(original)
        _RECEIPT = dict(schema='rinkspace.compaction-binding-index-only-reuse.v1',
            helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            original_source_sha256=hashlib.sha256(Path(inspect.getsourcefile(original)).read_bytes()).hexdigest(),
            original_function_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
            original_cache_clear_sites_replaced=1, registered_source_modified=False,
            original_compaction_and_numeric_checks_preserved=True,
            complete_q_course_chain_camera_source_inputs_checked=True,
            records_preserve_original_order_and_alias_multiplicity=True,
            historical_diagnostic_PASS_reused=False)
    return dict(_RECEIPT)


def snapshot_compaction_binding_reindex_runtime():
    return dict(installation=None if _RECEIPT is None else dict(_RECEIPT), events=list(_EVENTS))
