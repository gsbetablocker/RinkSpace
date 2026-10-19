"""Explicit same-value execution reuse for the official first-shot interfaces.

No residual, Jacobian, guard, target, knot or row-clock formula is replaced.
World calls share one ORIGINAL complete ice inventory within that call. Native
row constructors retain their original body except that the complete same-q
RF4 column/view preparation is shared. Importing does not install any patch.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import textwrap
import threading
import weakref

import numpy as np

_LOCK = threading.RLock()
_LOCAL = threading.local()
_VIEWS = weakref.WeakKeyDictionary()
_API = None
_ORIGINAL_ROW_INIT = None
_ORIGINAL_WORLD_PATHS = None
_ORIGINAL_FEATURE_WORLD = None
_ORIGINAL_VIEW_COLUMNS = None
_GENERATED_ROW_INIT = None
_COUNTERS = dict(complete_ice_inventories=0, scoped_ice_target_queries=0,
                 same_q_view_preparations=0, same_q_view_cache_hits=0,
                 unsupported_chart_uncached_preparations=0)


def _plain_views(state, values):
    columns = tuple(np.asarray(state.view_columns(i + 1), int)
                    for i in range(int(state.frame_count)))
    views = np.asarray([values[c] for c in columns])
    views.flags.writeable = False
    return columns, views


def _prepared_views(state, values):
    """Complete numeric bytes, never object identity or a subset of q."""
    method = getattr(state.view_columns, '__func__', None)
    if method is not _ORIGINAL_VIEW_COLUMNS:
        _COUNTERS['unsupported_chart_uncached_preparations'] += 1
        return _plain_views(state, values)
    # The original method depends on precisely frame_count and view_start.
    # state._q(values), every PTS/readout check and calibration construction
    # remain in the unmodified surrounding row constructor on every call.
    token = (values.dtype.str, values.shape, values.tobytes(), int(state.count),
             int(state.frame_count), int(state.view_start), method)
    try:
        with _LOCK:
            previous = _VIEWS.get(state)
            if previous is not None and previous[0] == token:
                _COUNTERS['same_q_view_cache_hits'] += 1
                return previous[1], previous[2]
            columns, views = _plain_views(state, values)
            for columns_at_frame in columns:
                columns_at_frame.flags.writeable = False
            _VIEWS[state] = (token, columns, views)
            _COUNTERS['same_q_view_preparations'] += 1
            return columns, views
    except TypeError:
        # Unknown non-weak-referenceable state types use the original work.
        _COUNTERS['unsupported_chart_uncached_preparations'] += 1
        return _plain_views(state, values)


@contextmanager
def _inventory_scope():
    scopes = getattr(_LOCAL, 'inventories', None)
    if scopes is None:
        scopes = _LOCAL.inventories = []
    scopes.append({})
    try:
        yield
    finally:
        scopes.pop()


def _scoped_world_paths(self, shared_geometry, descriptors):
    with _inventory_scope():
        return _ORIGINAL_WORLD_PATHS(self, shared_geometry, descriptors)


def _scoped_feature_world(scene, target_descriptor, templates=None):
    scopes = getattr(_LOCAL, 'inventories', ())
    if not scopes or not isinstance(target_descriptor, dict) or target_descriptor.get('kind') != 'ice_curve':
        return _ORIGINAL_FEATURE_WORLD(scene, target_descriptor, templates)
    cache = scopes[-1]
    token = id(scene)
    entry = cache.get(token)
    if entry is None or entry[0] is not scene:
        from rinkspace.vision_programme_ice_review import ice_curve_inventory
        # This is the original full producer and all its scene-wide identity
        # checks. No selected-feature shortcut can hide an unrelated error.
        inventory = ice_curve_inventory(scene)
        lookup = {}
        for record in inventory:
            lookup.setdefault(record['curve_id'], []).append(record)
        entry = cache[token] = (scene, lookup)
        _COUNTERS['complete_ice_inventories'] += 1
    _COUNTERS['scoped_ice_target_queries'] += 1
    target = target_descriptor
    matches = entry[1].get(target.get('curve_id'), [])
    if len(matches) != 1:
        raise ValueError('Missing or ambiguous named ice curve: ' + str(target.get('curve_id')))
    return _API['framewise']._xyz(matches[0]['xyz'], label=target['curve_id'])


def _compile_row_constructor(original):
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef) or function.name != '__init__':
        raise ValueError('The public row constructor source changed')
    locations = {}
    for i, node in enumerate(function.body):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == 'self'):
            name = node.targets[0].attr
            if name in ('_view_columns', '_views'):
                if name in locations:
                    raise ValueError('Ambiguous original row view preparation')
                locations[name] = i
    if set(locations) != {'_view_columns', '_views'}:
        raise ValueError('The original complete row view preparation is missing')
    first, second = locations['_view_columns'], locations['_views']
    if second != first + 1:
        raise ValueError('The original row view preparation order changed')
    expected_columns = 'tuple(np.asarray(state.view_columns(i + 1), int) for i in range(self.frame_count))'
    expected_views = 'np.asarray([values[columns] for columns in self._view_columns])'
    if (ast.dump(function.body[first].value) != ast.dump(ast.parse(expected_columns, mode='eval').body)
            or ast.dump(function.body[second].value) != ast.dump(ast.parse(expected_views, mode='eval').body)):
        raise ValueError('The original RF4 column or view arithmetic changed')
    replacement = ast.parse('self._view_columns, self._views = _value_prepared_views(state, values)').body[0]
    function.body[first:second+1] = [replacement]
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace['_value_prepared_views'] = _prepared_views
    exec(compile(tree, __file__ + '::original_FrameRowCamera_init', 'exec'), namespace)
    generated = namespace['__init__']
    generated.__doc__ = original.__doc__
    return generated


def statistics():
    with _LOCK:
        return dict(_COUNTERS, prepared_camera_states=len(_VIEWS))


def clear_preparations():
    """Drop execution-only reuse; no q, scene or original model cache changes."""
    with _LOCK:
        _VIEWS.clear()
        for key in _COUNTERS:
            _COUNTERS[key] = 0


def install():
    """Install explicitly in each participating parent/worker Python process."""
    global _API, _ORIGINAL_ROW_INIT, _ORIGINAL_WORLD_PATHS
    global _ORIGINAL_FEATURE_WORLD, _ORIGINAL_VIEW_COLUMNS, _GENERATED_ROW_INIT
    from ..solver_runtime import install as install_runtime
    install_runtime()
    from . import first_shot_row_camera as row
    from . import first_shot_physical_state as physical
    from . import framewise_model as framewise
    if _API is None:
        _ORIGINAL_ROW_INIT = row.FrameRowCamera.__init__
        _ORIGINAL_WORLD_PATHS = physical.FirstShotPhysicalState.world_paths
        _ORIGINAL_FEATURE_WORLD = framewise.feature_world
        _ORIGINAL_VIEW_COLUMNS = physical.FirstShotPhysicalState.view_columns
        _GENERATED_ROW_INIT = _compile_row_constructor(_ORIGINAL_ROW_INIT)
        _API = dict(row=row, physical=physical, framewise=framewise, retained={
            'camera15_mapping': row.FrameRowCamera.camera15_mapping,
            'camera25_at_rows': row.FrameRowCamera.camera25_at_rows,
            'camera15_row_derivative': row.FrameRowCamera.camera15_row_derivative,
            'guard': physical.FirstShotPhysicalState.guard,
            'guard_geometry': physical.FirstShotPhysicalState.guard_geometry,
            'ice_material_containment_status': physical.FirstShotPhysicalState.ice_material_containment_status})
    if (row.FrameRowCamera.__init__ not in (_ORIGINAL_ROW_INIT, _GENERATED_ROW_INIT)
            or physical.FirstShotPhysicalState.world_paths not in (_ORIGINAL_WORLD_PATHS, _scoped_world_paths)
            or framewise.feature_world not in (_ORIGINAL_FEATURE_WORLD, _scoped_feature_world)):
        raise RuntimeError('An unknown first-shot execution override is already installed')
    row.FrameRowCamera.__init__ = _GENERATED_ROW_INIT
    physical.FirstShotPhysicalState.world_paths = _scoped_world_paths
    framewise.feature_world = _scoped_feature_world
    from .official_static_surface_dataset import reference
    return dict(schema='rinkspace.first-shot-same-value-execution.v1', producer=reference(__file__),
                complete_inventory_once_per_world_paths_call=True,
                inventory_scope_includes_all_scene_errors=True,
                same_complete_numeric_q_views_prepared_once_per_state=True,
                every_original_q_PTS_readout_neighbour_check_retained=True,
                row_initializer_replaced_nodes=['_view_columns preparation', '_views preparation'],
                residual_J_guard_or_source_data_changed=False,
                single_point_and_non_ice_queries_use_original_producer=True,
                applies_to_current_process_only=True,
                original_producers=[reference(fn.__code__.co_filename) for fn in
                    (_ORIGINAL_ROW_INIT, _ORIGINAL_WORLD_PATHS, _ORIGINAL_FEATURE_WORLD)],
                original_row_constructor_source_sha256=hashlib.sha256(inspect.getsource(_ORIGINAL_ROW_INIT).encode()).hexdigest(),
                retained_methods={name: (getattr(row.FrameRowCamera, name, None) is fn
                    if name.startswith('camera') else getattr(physical.FirstShotPhysicalState, name, None) is fn)
                    for name, fn in _API['retained'].items()},
                view_cache_identity=dict(state='weak physical-state instance',
                    numerical_q='dtype,shape,and every received q byte; recomputed after original state._q on every constructor',
                    chart='original view_columns function, state.count, frame_count, view_start',
                    entries='one current complete q preparation per live state',
                    unsupported_chart='original uncached preparation',
                    columns_and_views='immutable original formula outputs',
                    native_PTS_and_readout='original validation and snapshot on every frame, never cached',
                    calibration='original state.calibration(values) on every frame, never cached',
                    outer_augmented_q_authentication_modified=False),
                inventory_cache_identity=dict(lifetime='one original world_paths call in the current thread',
                    scene='strong object identity within that call; no reuse across q or shared5 changes',
                    producer='original complete ice_curve_inventory(scene), with every duplicate/unknown feature check',
                    descriptors='unchanged original key and target lookup errors',
                    outputs='original _xyz validation/copy; all original knots, closure and order',
                    single_point='all non-ice queries, including ad_corner N=1, go directly to original feature_world'),
                real_complete_q_r_J_and_guard_validation_pending=True)


def uninstall():
    if _API is None:
        return
    row, physical, framewise = _API['row'], _API['physical'], _API['framewise']
    if (row.FrameRowCamera.__init__ not in (_ORIGINAL_ROW_INIT, _GENERATED_ROW_INIT)
            or physical.FirstShotPhysicalState.world_paths not in (_ORIGINAL_WORLD_PATHS, _scoped_world_paths)
            or framewise.feature_world not in (_ORIGINAL_FEATURE_WORLD, _scoped_feature_world)):
        raise RuntimeError('Cannot restore over an unknown first-shot execution override')
    row.FrameRowCamera.__init__ = _ORIGINAL_ROW_INIT
    physical.FirstShotPhysicalState.world_paths = _ORIGINAL_WORLD_PATHS
    framewise.feature_world = _ORIGINAL_FEATURE_WORLD
    clear_preparations()
