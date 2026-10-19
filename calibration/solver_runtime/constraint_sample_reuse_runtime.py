"""Reuse only unchanged camera-local and literal keys inside original samples.

Every original complete-state/world admission, exposure camera and native query
still executes. The registered source files and their manifests are unchanged.
No mutable local result is retained beyond one complete sample; literal source
bytes live for one confirmed constraints construction and are checked at exit.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from contextvars import ContextVar
import functools
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import pickle
import textwrap

import numpy as np


_CAMERA_SAMPLE = ContextVar('rinkspace_original_constraint_camera_sample', default=None)
_LITERAL_CONSTRUCTION = ContextVar('rinkspace_original_confirmed_literal_construction', default=None)
_RECEIPT = None


@contextmanager
def _camera_sample_scope():
    token = _CAMERA_SAMPLE.set({})
    try:
        yield
    finally:
        _CAMERA_SAMPLE.reset(token)


def _original_exposure_cameras(blocks, core, native_camera_local):
    """Original exposure order/camera calls, one original local per safe block.

    The cache is valid only for the reviewed original programme camera-local
    implementation. Unknown adapters and changed calibration bindings execute
    their original access on every exposure. Different exact core inputs do
    not share a local, including nested factor calls with a different chart.
    """
    from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock

    cache = _CAMERA_SAMPLE.get()
    if cache is None:
        cache = {}
    core_array = np.asarray(core)
    core_key = (core_array.shape, core_array.dtype.str, core_array.tobytes())
    construction = _LITERAL_CONSTRUCTION.get()
    cameras = {}
    for exposure, block in blocks.items():
        safe = (isinstance(block, StructuredProgrammeBlock)
                and getattr(type(block), 'camera_local', None) is StructuredProgrammeBlock.camera_local
                and getattr(type(block), 'local', None) is StructuredProgrammeBlock.local
                and 'camera_local' not in vars(block) and 'local' not in vars(block))
        key = (id(block), core_key)
        previous = cache.get(key) if safe else None
        if (previous is not None and previous[0] is block
                and previous[1] is block.bundle
                and previous[2] is block.bundle.calibration):
            local = previous[3]
            if construction is not None:
                construction['camera_local_reuses'] += 1
        else:
            local = native_camera_local(block, core)
            if safe:
                cache[key] = (block, block.bundle, block.bundle.calibration, local)
            if construction is not None:
                construction['camera_local_calls'] += 1
        # This remains an original camera admission for EVERY exposure, even
        # when the subsequent per-record carrier/query memo has an exact hit.
        cameras[exposure] = block.bundle.camera(local, exposure)
        if construction is not None:
            construction['original_exposure_camera_calls'] += 1
    return cameras


def _source_descriptor(item):
    # Match the original dict order and protocol, including historical metadata.
    return dict(row=item['row'], radius=item['radius'], normals=item['normals'],
                bounds=item['bounds'], original_support_operator=item.get('original_support_operator'))


def _literal_source_bytes(owner, item, source):
    construction = _LITERAL_CONSTRUCTION.get()
    if construction is None or construction['owner'] is not owner:
        return pickle.dumps(source, protocol=5)
    cached = construction['sources'].get(id(item))
    if cached is None:
        literal = pickle.dumps(source, protocol=5)
        construction['sources'][id(item)] = (item, literal)
        construction['source_serializations'] += 1
        return literal
    if cached[0] is not item:
        raise RuntimeError('A confirmed record identity changed inside its constraints construction')
    construction['literal_source_reuses'] += 1
    return cached[1]


def _with_camera_sample(original):
    @functools.wraps(original)
    def sampled(*args, **kwargs):
        with _camera_sample_scope():
            return original(*args, **kwargs)
    return sampled


def _with_literal_construction(original, report_directory):
    @functools.wraps(original)
    def construction(owner, *args, **kwargs):
        records = tuple(owner.records)
        context = dict(owner=owner, sources={}, source_serializations=0,
                       literal_source_reuses=0, camera_local_calls=0,
                       camera_local_reuses=0, original_exposure_camera_calls=0)
        token = _LITERAL_CONSTRUCTION.set(context)
        succeeded = False
        try:
            result = original(owner, *args, **kwargs)
            # Original producers only read these source descriptors. Verify all
            # complete literal bytes before exposing any resulting constraints.
            if len(owner.records) != len(records) or any(a is not b for a, b in zip(owner.records, records)):
                raise RuntimeError('Confirmed source inventory changed during constraints construction')
            for item, literal in context['sources'].values():
                if pickle.dumps(_source_descriptor(item), protocol=5) != literal:
                    raise RuntimeError('Confirmed literal source inputs changed during constraints construction')
            succeeded = True
            return result
        finally:
            # In particular, do not catch or replace SourceControlStop, which
            # deliberately inherits BaseException rather than Exception.
            _LITERAL_CONSTRUCTION.reset(token)
            if report_directory is not None:
                event = dict(schema='rinkspace.constraint-sample-local-and-literal-reuse.v1',
                             succeeded=succeeded,
                             source_records_serialized=context['source_serializations'],
                             literal_source_reuses=context['literal_source_reuses'],
                             camera_local_calls=context['camera_local_calls'],
                             camera_local_reuses=context['camera_local_reuses'],
                             original_exposure_camera_calls=context['original_exposure_camera_calls'],
                             complete_source_bytes_authenticated_at_successful_exit=succeeded,
                             original_q_world_camera_query_and_FD_checks_retained=True)
                # Diagnostics must not mask a numerical/control exception.
                try:
                    with (report_directory / 'actual_confirmed_constraint_sample_reuse.jsonl').open('a', encoding='utf8') as stream:
                        stream.write(json.dumps(event, ensure_ascii=False) + '\n')
                except OSError:
                    if succeeded:
                        raise
    return construction


class _CameraComprehension(ast.NodeTransformer):
    def __init__(self):
        self.changed = 0

    def visit_DictComp(self, node):
        # Match only the literal original {p: b.bundle.camera(
        # native_camera_local(b, core), p) for p,b in owner.blocks.items()}.
        if len(node.generators) != 1:
            return self.generic_visit(node)
        gen = node.generators[0]
        if (gen.ifs or gen.is_async or not isinstance(gen.target, ast.Tuple)
                or len(gen.target.elts) != 2 or not all(isinstance(x, ast.Name) for x in gen.target.elts)):
            return self.generic_visit(node)
        exposure, block = [x.id for x in gen.target.elts]
        expected_value = ast.parse(f'{block}.bundle.camera(native_camera_local({block},core),{exposure})', mode='eval').body
        if (not isinstance(node.key, ast.Name) or node.key.id != exposure
                or ast.dump(node.value) != ast.dump(expected_value)
                or not isinstance(gen.iter, ast.Call) or gen.iter.args or gen.iter.keywords
                or not isinstance(gen.iter.func, ast.Attribute) or gen.iter.func.attr != 'items'
                or not isinstance(gen.iter.func.value, ast.Attribute) or gen.iter.func.value.attr != 'blocks'):
            return self.generic_visit(node)
        self.changed += 1
        return ast.copy_location(ast.Call(ast.Name('_rinkspace_original_exposure_cameras', ast.Load()),
            [gen.iter.func.value, ast.Name('core', ast.Load()), ast.Name('native_camera_local', ast.Load())], []), node)


class _LiteralSourcePickle(ast.NodeTransformer):
    def __init__(self):
        self.changed = 0

    def visit_Call(self, node):
        expected = ast.parse('pickle.dumps(source,protocol=5)', mode='eval').body
        if ast.dump(node) == ast.dump(expected):
            self.changed += 1
            return ast.copy_location(ast.Call(ast.Name('_rinkspace_literal_source_bytes', ast.Load()),
                [ast.Name('owner', ast.Load()), ast.Name('item', ast.Load()), ast.Name('source', ast.Load())], []), node)
        return self.generic_visit(node)


def _transform(original, *, camera=False, literal=False, nested_sample=False):
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise TypeError('Expected one original source-owned function')
    if camera:
        rewrite = _CameraComprehension()
        tree = rewrite.visit(tree)
        if rewrite.changed != 1:
            raise TypeError('Expected exactly one original exposure camera comprehension')
    if literal:
        rewrite = _LiteralSourcePickle()
        tree = rewrite.visit(tree)
        if rewrite.changed != 1:
            raise TypeError('Expected exactly one original literal source pickle expression')
    if nested_sample:
        nested = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'memoized_queries']
        if len(nested) != 1:
            raise TypeError('Expected the original complete FD sample closure')
        nested[0].body = [ast.With([ast.withitem(ast.Call(ast.Name('_rinkspace_camera_sample_scope', ast.Load()), [], []), None)], nested[0].body)]
    ast.fix_missing_locations(tree)
    original.__globals__.update(
        _rinkspace_original_exposure_cameras=_original_exposure_cameras,
        _rinkspace_literal_source_bytes=_literal_source_bytes,
        _rinkspace_camera_sample_scope=_camera_sample_scope)
    namespace = {}
    exec(compile(tree, inspect.getsourcefile(original), 'exec'), original.__globals__, namespace)
    changed = namespace[tree.body[0].name]
    return changed, dict(original_qualified_name=f'{original.__module__}.{original.__qualname__}',
                         original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                         camera_comprehensions_replaced=int(camera), literal_pickle_expressions_replaced=int(literal),
                         original_complete_sample_scope_added=bool(nested_sample))


def install_constraint_sample_reuse_runtime(*, report_directory=None):
    """Install before source construction; parent integrates this single API."""
    global _RECEIPT
    destination = None if report_directory is None else Path(report_directory).resolve()
    if _RECEIPT is not None:
        if _RECEIPT['report_directory'] != (None if destination is None else str(destination)):
            raise RuntimeError('Constraint sample reuse is already installed for another run')
        return _RECEIPT
    human = importlib.import_module('structured_programme_source_candidate.human_stroke_truth')
    fd = importlib.import_module('structured_programme_source_candidate.human_constraint_dependency_fd')
    changes = []
    for name in ('_confirmed_record_memo_key', '_confirmed_pure_query_memo'):
        changed, receipt = _transform(getattr(human, name), literal=True)
        setattr(human, name, changed)
        changes.append(receipt)
    for owner, name in ((human.UserConfirmedSourceTruth, '_carriers'), (human.HumanStrokeTruth, '_queries_for_records')):
        changed, receipt = _transform(getattr(owner, name), camera=True)
        setattr(owner, name, _with_camera_sample(changed))
        changes.append(receipt)
    changed, receipt = _transform(fd.dependent_carrier_jacobian, camera=True, nested_sample=True)
    fd.dependent_carrier_jacobian = changed
    changes.append(receipt)
    for module_name, class_name in (
            ('native_board_edge_source', 'NativeBoardEdgeFactor'),
            ('native_board_circle_source', 'NativeBoardCircleFactor'),
            ('native_board_material_point_source', 'NativeBoardMaterialPointFactor'),
            ('native_ice_curve_source', 'NativeIceCurveFactor')):
        module = importlib.import_module('structured_programme_source_candidate.' + module_name)
        owner = getattr(module, class_name)
        changed, receipt = _transform(owner._cameras, camera=True)
        owner._cameras = changed
        changes.append(receipt)
    human.UserConfirmedSourceTruth.constraints = _with_literal_construction(human.UserConfirmedSourceTruth.constraints, destination)
    helper = Path(__file__).resolve()
    _RECEIPT = dict(schema='rinkspace.constraint-sample-local-and-literal-reuse.v1',
                    report_directory=None if destination is None else str(destination),
                    helper_path=str(helper), helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
                    source_function_changes=changes,
                    original_world_admission_per_FD_sample_retained=True,
                    original_bundle_camera_per_exposure_in_original_order_retained=True,
                    original_query_and_exact_q_camera_world_dynamic_keys_retained=True,
                    literal_source_bytes_authenticated_at_successful_construction_exit=True,
                    registered_source_files_and_manifest_changed=False,
                    local_mutable_cache_crosses_FD_sample_or_thread=False,
                    actual_speedup_pending=True, accuracy_accepted=False)
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'constraint_sample_reuse_runtime_receipt.json').write_text(
            json.dumps(_RECEIPT, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return _RECEIPT
