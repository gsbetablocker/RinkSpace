"""Reuse identical original values only inside one current role binding.

Every source row, ray admission, selected-facet comparison, full world chain
and source record remains original. Explicit uncached replay builds never
reuse rays. Plane/J reuse belongs only to this build's identical facet inputs.
"""
from __future__ import annotations

import ast
from contextvars import ContextVar
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

import numpy as np

_SCOPE = ContextVar('rinkspace_single_role_binding_values', default=None)
_RECEIPT = None
_REPORT_DIRECTORY = None


def _record(value):
    if _REPORT_DIRECTORY is not None:
        with (_REPORT_DIRECTORY / 'actual_role_binding_value_reuse.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


class _BindingScope:
    def __enter__(self):
        self.value = dict(rays={}, hits=0, misses=0, build_reuse=False)
        self.token = _SCOPE.set(self.value)
        return self

    def __exit__(self, exc_type, exc, tb):
        _SCOPE.reset(self.token)
        value = self.value
        value['rays'].clear()
        # Logging cannot mask an original domain or control failure.
        if exc_type is None and (value['hits'] or value['misses']):
            _record(dict(stage='original_role_ray_value_reuse', original_calls=value['misses'],
                eliminated_identical_calls=value['hits'], cache_entries_after_scope=0,
                full_camera_UV_and_border_bytes_checked=True, cross_call_or_q_reuse=False))
        return False


def _ray_value(original, phase, camera, uv, border):
    scope = _SCOPE.get()
    if scope is None or (phase == 'build' and not scope['build_reuse']):
        return original(camera, uv, border)
    arrays = tuple(np.ascontiguousarray(value, dtype=float) for value in (camera, uv, border))
    key = tuple((value.shape, value.dtype.str, value.tobytes()) for value in arrays)
    if key in scope['rays']:
        scope['hits'] += 1
        return scope['rays'][key].copy()
    # Original successful computation includes its original current ray and
    # forward finite-course admission. Failed calls are never cached.
    result = original(camera, uv, border)
    scope['rays'][key] = result.copy()
    scope['misses'] += 1
    return result


def _current_binding_build(original, *args):
    scope = _SCOPE.get()
    if scope is None:
        return original(*args)
    prior = scope['build_reuse']
    scope['build_reuse'] = True
    try:
        return original(*args)
    finally:
        scope['build_reuse'] = prior


def _facet_plane(original, cache, facet, start, end, jstart, jend):
    # bottom/world_j remain unchanged within the original build. The same
    # facet indexes therefore name identical original ordered numeric inputs.
    # Call at its first original row, preserving original failure priority.
    if facet not in cache:
        cache[facet] = original(start, end, jstart, jend)
    return tuple(value.copy() for value in cache[facet])


def _finish_planes(cache, result):
    count = len(result['records'])
    _record(dict(stage='original_role_facet_plane_value_J_reuse', original_source_records=count,
        distinct_current_facets=len(cache), original_plane_J_calls=len(cache),
        eliminated_identical_plane_J_calls=count-len(cache), cache_entries_after_return=0,
        original_plane_J_producer_and_order_retained=True,
        every_original_source_record_and_deepcopy_retained=True, cross_call_or_q_reuse=False))
    cache.clear()
    return result


def _patch(function, mode):
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    target = tree.body[0]
    sites = []
    class Replace(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if (mode in ('selection', 'build') and isinstance(node.func, ast.Name)
                    and node.func.id == 'current_far_board_points'):
                if len(node.args) != 3 or node.keywords:
                    raise ValueError('Original source ray producer arguments changed')
                sites.append('original_ray_value')
                node.args = [node.func, ast.Constant(mode), *node.args]
                node.func = ast.Name('_rinkspace_role_ray_value', ast.Load())
            elif (mode == 'scope' and isinstance(node.func, ast.Attribute)
                    and node.func.attr == '_build_current_binding'):
                if len(node.args) != 5 or node.keywords:
                    raise ValueError('Original current role build arguments changed')
                sites.append('current_binding_build')
                node.args = [node.func, *node.args]
                node.func = ast.Name('_rinkspace_role_current_binding_build', ast.Load())
            elif mode == 'build' and isinstance(node.func, ast.Name) and node.func.id == 'vertical_plane_value_jacobian':
                if ast.unparse(node) != 'vertical_plane_value_jacobian(ends[0], ends[1], jac[0], jac[1])':
                    raise ValueError('Original repeated facet plane/J inputs changed')
                sites.append('original_facet_plane_J')
                node.args = [node.func, ast.Name('_rinkspace_facet_planes', ast.Load()), ast.Name('i', ast.Load()), *node.args]
                node.func = ast.Name('_rinkspace_role_facet_plane', ast.Load())
            return node

        def visit_Return(self, node):
            self.generic_visit(node)
            if mode == 'build':
                sites.append('original_build_return')
                node.value = ast.Call(ast.Name('_rinkspace_role_finish_planes', ast.Load()),
                    [ast.Name('_rinkspace_facet_planes', ast.Load()), node.value], [])
            return node

    tree = Replace().visit(tree)
    expected = {'scope': ['current_binding_build'], 'selection': ['original_ray_value'],
                'build': ['original_ray_value', 'original_facet_plane_J', 'original_build_return']}[mode]
    if sites != expected or function.__code__.co_freevars:
        raise ValueError('Original closure-free role value sites changed: ' + str(sites))
    first = 1 if isinstance(target.body[0], ast.Expr) and isinstance(target.body[0].value, ast.Constant) and isinstance(target.body[0].value.value, str) else 0
    if mode == 'scope':
        scope = ast.parse('with _rinkspace_role_binding_value_scope():\n    pass').body[0]
        scope.body = target.body[first:]
        target.body[first:] = [scope]
    elif mode == 'build':
        target.body.insert(first, ast.parse('_rinkspace_facet_planes = {}').body[0])
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, function.__code__.co_firstlineno - 1)
    function.__globals__.update(_rinkspace_role_ray_value=_ray_value,
        _rinkspace_role_current_binding_build=_current_binding_build,
        _rinkspace_role_facet_plane=_facet_plane, _rinkspace_role_finish_planes=_finish_planes,
        _rinkspace_role_binding_value_scope=_BindingScope)
    namespace = {}
    exec(compile(tree, inspect.getsourcefile(function), 'exec'), function.__globals__, namespace)
    # Preserve both OriginalMotion... and ExistingFinite... aliases.
    function.__code__ = namespace[function.__name__].__code__
    return dict(function=function.__qualname__, original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                changes=sites)


def install_role_binding_value_reuse_runtime(*, report_directory=None):
    global _RECEIPT, _REPORT_DIRECTORY
    if _RECEIPT is not None:
        return _RECEIPT
    from structured_programme_source_candidate.original_motion_far_board_roles import OriginalMotionFarBoardRoleBinding, ExistingFinitePerimeterBinding
    cls = OriginalMotionFarBoardRoleBinding
    for name in ('current_binding', '_current_source_facets', '_build_current_binding'):
        if getattr(cls, name) is not getattr(ExistingFinitePerimeterBinding, name):
            raise ValueError('Original role/existing17 shared producer aliases required')
    changes = [_patch(cls.current_binding, 'scope'), _patch(cls._current_source_facets, 'selection'),
               _patch(cls._build_current_binding, 'build')]
    path = Path(__file__).resolve()
    _REPORT_DIRECTORY = None if report_directory is None else Path(report_directory).resolve()
    if _REPORT_DIRECTORY is not None:
        _REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _RECEIPT = dict(schema='rinkspace.original-role-single-binding-value-reuse.v1', helper_path=str(path),
        helper_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), original_functions=changes,
        ray_reuse_scope='one original current_binding; full camera/UV/border bytes; successful values only',
        plane_J_reuse_scope='one original build; identical facet endpoints and full world-J inputs',
        original_uncached_replay_build_rays_retained=True,
        independent_returned_ray_and_plane_arrays=True, all_original_source_records_and_deepcopy_retained=True,
        original_selected_facet_comparison_full_world_chain_and_domain_gates_retained=True,
        no_cross_call_or_q_cache=True, registered_source_files_modified=False, accuracy_accepted=False)
    if _REPORT_DIRECTORY is not None:
        (_REPORT_DIRECTORY / 'role_binding_value_reuse_receipt.json').write_text(
            json.dumps(_RECEIPT, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return _RECEIPT
