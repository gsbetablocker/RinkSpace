"""Seed an owned actor's role view from the complete original current binding.

No planes or source facts are inferred. Only already produced numeric rows
are copied, after exact source metadata/camera-key validation. The view's
unchanged current_binding still checks its current course/chain/camera key.
"""
from __future__ import annotations

import hashlib
import multiprocessing
from pathlib import Path

import numpy as np

_ROW_ARRAYS = ('planes', 'plane_jacobian', 'extent_world',
    'extent_world_jacobian', 'selected_course_facets')
_GLOBAL_ARRAYS = ('z_range_scene', 'z_range_world_jacobian')
_BOUND_FIELDS = frozenset((*_ROW_ARRAYS, *_GLOBAL_ARRAYS, 'records'))
_RECORD_FIELDS = frozenset(('global_bundle_index', 'surface_kind',
    'extent_start_xy', 'extent_end_xy', 'physical_surface_identity',
    'source_role_binding_id', 'source_role_block', 'source_native_pair_pts',
    'original_source_role', 'material_identity', 'actual_source_record',
    'current_course_facet_index', 'facet_index_is_current_geometry_not_measured_location'))


def _same(value, other):
    if isinstance(value, np.ndarray) or isinstance(other, np.ndarray):
        return (type(value) is np.ndarray and type(other) is np.ndarray
            and value.shape == other.shape and value.dtype == other.dtype
            and not value.dtype.hasobject and value.tobytes() == other.tobytes())
    if isinstance(value, dict) or isinstance(other, dict):
        return (type(value) is dict and type(other) is dict and value.keys() == other.keys()
            and all(_same(item, other[key]) for key, item in value.items()))
    if isinstance(value, (tuple, list)) or isinstance(other, (tuple, list)):
        return (type(value) is type(other) and len(value) == len(other)
            and all(_same(a, b) for a, b in zip(value, other)))
    return type(value) is type(other) and bool(value == other)


def _camera_bytes(role, stamps):
    values = []
    for stamp in sorted(stamps):
        camera = np.asarray(role.source_cameras[stamp], float)
        if camera.shape != (25,) or not np.isfinite(camera).all():
            raise ValueError('Complete finite original camera25 required')
        values.append(camera.tobytes())
    return b''.join(values)


def seed_owned_role_view_cache(role, view, indices):
    """Return a receipt; on rejection leave the original fresh view unchanged.

Only an owned ``rink-source-`` child releases the unused complete numeric
cache after success. A parent invocation can validate/copy but cannot clear
its complete role cache. This function performs no geometry calculation.
"""
    receipt = dict(schema='rinkspace.owned-role-original-binding-cache-seed.v1',
        seeded=False, parent_or_source_records_modified=False,
        numeric_arrays_are_independent_copies=True,
        original_current_binding_key_gate_retained=True,
        full_worker_numeric_cache_released=False)
    try:
        from structured_programme_source_candidate.original_motion_far_board_roles import OriginalMotionFarBoardRoleBinding
        if type(role) is not OriginalMotionFarBoardRoleBinding or type(view) is not type(role) or role is view:
            raise ValueError('Original complete role and separate original-class view required')
        if view._current_binding_cache is not None:
            raise ValueError('Original fresh view required before cache seeding')
        owned = np.asarray(indices)
        if (owned.ndim != 1 or owned.dtype.kind not in 'iu' or len(owned) == 0
                or np.any(owned < 0) or np.any(owned >= len(role.records))
                or np.any(owned[1:] <= owned[:-1])):
            raise ValueError('Owned row indices must preserve complete original order')
        cache = role._current_binding_cache
        if type(cache) is not tuple or len(cache) != 2:
            raise ValueError('Complete original current binding cache required')
        key, bound = cache
        if type(key) is not tuple or len(key) != 4 or set(bound) != _BOUND_FIELDS:
            raise ValueError('Original binding schema/key required')
        n = len(role.records)
        if len(bound['records']) != n:
            raise ValueError('Full cache must cover every qualified original role')
        if not _same(role.source_uv, np.asarray([row['source_uv'] for row in role.records], float)):
            raise ValueError('Complete original source UV values no longer match qualified records')
        all_rows = []
        for stamp, rows in role.source_row_groups.items():
            a = np.asarray(rows)
            if a.ndim != 1 or a.dtype.kind not in 'iu':
                raise ValueError('Original integer source ray groups required')
            for index in a:
                i = int(index)
                if not 0 <= i < n or int(role.records[i]['source_pts']) != int(stamp):
                    raise ValueError('Complete original source ray group identity changed')
                all_rows.append(i)
        if sorted(all_rows) != list(range(n)):
            raise ValueError('Full original ray groups must cover every source row exactly once')
        if (not isinstance(key[0], float) or not np.isfinite(key[0])
                or any(type(part) is not bytes for part in key[1:])
                or len(key[1]) != 6 * np.dtype(float).itemsize
                or key[3] != _camera_bytes(role, role.source_row_groups)):
            raise ValueError('Full original camera/course cache key changed')
        arrays = {name: bound[name] for name in (*_ROW_ARRAYS, *_GLOBAL_ARRAYS)}
        if any(type(a) is not np.ndarray or a.dtype.hasobject for a in arrays.values()):
            raise ValueError('Original numeric ndarray fields required')
        J = arrays['plane_jacobian']
        if J.ndim != 3 or J.shape[:2] != (n, 4):
            raise ValueError('Full original plane Jacobian shape required')
        ng = J.shape[2]
        shapes = dict(planes=(n, 4), plane_jacobian=(n, 4, ng),
            extent_world=(n, 2, 3), extent_world_jacobian=(n, 2, 3, ng),
            selected_course_facets=(n,), z_range_scene=(2,),
            z_range_world_jacobian=(2, ng))
        if (len(key[2]) != 6 * ng * np.dtype(float).itemsize
                or any(a.shape != shapes[name] or not np.isfinite(a).all()
                    or (name != 'selected_course_facets' and a.dtype != np.dtype(float))
                    for name, a in arrays.items())
                or arrays['selected_course_facets'].dtype.kind not in 'iu'):
            raise ValueError('Incomplete original numeric binding arrays/chain')
        for i, (cached, source) in enumerate(zip(bound['records'], role.records)):
            if (set(cached) != _RECORD_FIELDS
                    or not _same(cached['actual_source_record'], source)
                    or cached['material_identity'] is not source['physical_material']
                    or int(cached['global_bundle_index']) != int(source.get('active_global_bundle_index', source['original_global_bundle_index']))
                    or cached['source_role_binding_id'] != role.binding_id
                    or cached['source_role_block'] != source['source_role_block']
                    or cached['source_native_pair_pts'] != [source['source_pts'], source['target_pts']]
                    or int(cached['current_course_facet_index']) != int(arrays['selected_course_facets'][i])
                    or not _same(cached['extent_start_xy'], arrays['extent_world'][i, 0, :2].tolist())
                    or not _same(cached['extent_end_xy'], arrays['extent_world'][i, 1, :2].tolist())):
                raise ValueError('Original cached role metadata/index no longer matches qualified source')
        if (len(view.records) != len(owned)
                or any(view.records[local] is not role.records[int(index)] for local, index in enumerate(owned))
                or not _same(view.source_uv, np.asarray(role.source_uv)[owned])
                or not _same(view.indices, np.asarray(role.indices)[owned])
                or view.binding_id != role.binding_id):
            raise ValueError('Owned view source row identity changed')
        remap = {int(index): local for local, index in enumerate(owned)}
        expected_groups = {}
        for stamp, source_rows in role.source_row_groups.items():
            members = [int(i) in remap for i in source_rows]
            if not any(members):
                continue
            if not all(members):
                raise ValueError('Original source ray group must not be split')
            expected_groups[stamp] = np.asarray([remap[int(i)] for i in source_rows], dtype=np.asarray(source_rows).dtype)
        if (sum(len(rows) for rows in expected_groups.values()) != len(owned)
                or not _same(view.source_row_groups, expected_groups)
                or _camera_bytes(view, expected_groups) != _camera_bytes(role, expected_groups)):
            raise ValueError('Owned original per-stamp source/camera groups changed')
        projected = dict(bound)
        for name in _ROW_ARRAYS:
            projected[name] = np.take(arrays[name], owned, axis=0)
        for name in _GLOBAL_ARRAYS:
            projected[name] = arrays[name].copy(order='K')
        projected['records'] = [bound['records'][int(i)] for i in owned]
        view._current_binding_cache = (key[:3] + (_camera_bytes(view, expected_groups),), projected)
        if multiprocessing.current_process().name.startswith('rink-source-'):
            role._current_binding_cache = None
            receipt['full_worker_numeric_cache_released'] = True
        receipt.update(seeded=True, full_original_records=n, owned_records=len(owned),
            copied_numeric_bytes=sum(projected[name].nbytes for name in (*_ROW_ARRAYS, *_GLOBAL_ARRAYS)),
            source_record_multiplicity_order_and_material_alias_preserved=True)
    except Exception as error:
        receipt['reason'] = type(error).__name__ + ': ' + str(error)
    view._rinkspace_role_binding_seed_receipt = receipt
    return receipt


def role_view_binding_seed_identity():
    return dict(helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        registered_source_modified=False, geometry_or_source_values_recomputed=False)
