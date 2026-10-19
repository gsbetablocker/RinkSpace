"""Authenticated final source-graph inputs, never a saved derivative model.

The supported capture boundary is after original role installation and motion
input compaction, before CurrentProgrammeSource/actors are constructed. The
caller keeps all current checkpoint, truth and numerical continuation gates.
This module deliberately provides no main/central hook or cloudpickle escape.
"""
from __future__ import annotations

from collections import OrderedDict
import ast
import copy
import functools
import hashlib
import json
from pathlib import Path
import pickle
import time
import uuid

import numpy as np


SCHEMA = 'rinkspace.final-original-source-state-skeleton.v1'

# These exact predecessors produced the existing immutable static payload.
# Compatibility authenticates their archived bytes and the unchanged writer
# contract; it never substitutes a historical source/input file for today.
_STATIC_READER_PREDECESSORS = {
    'source_state_skeleton_cache_runtime.py': '9c3b771bfeda5a11791f2cd5f7fe23a6c321d060b301f2b617b36ab4e9f15d80',
    'named66_source_skeleton_adapter_runtime.py': '4d464d998927b9bd91d992bd70046799418094cdf2a31dabc0a33059e55d4b26',
    'execution_acceleration.py': 'baf1c9d2d728392d8c4ab4f7891210f309bc07c8d4421783d7c59c73b9ccc402',
    'residual_actor_lifecycle.py': 'c50fe7cf33cd832a57053d597766b61a739fba77f68d3a9067a8e5741550e278',
    'complete_source_pool_lifetime_runtime.py': 'f1948f5ddc0bcec9e2c19790b1e77aa8c7df7863a42de66be5dcb2a5e22cc899',
}
# Exact successful capture immediately before the measured-location local
# restore fix. Both archived files already exist; no cache/input is rewritten.
# Their writer AST is independently compared below, as for the older bridge.
_MEASURED_LOCATION_READER_PREDECESSORS = {
    'source_state_skeleton_cache_runtime.py': '79fe3818816fdce5905553dc8f823aecfd15fda0a429b69f272a9d8ad3cd07d8',
    'named66_source_skeleton_adapter_runtime.py': '6c36eefbef0de32032c7098d2698008b6a98755317702a0738f6ef97d148512b',
}
_MEASURED_LOCATION_READER_SNAPSHOT = 'okada_q091_same_normal_truth_progress_shadow_bound_entry_20261011.bootstrap'
_PURE_EXECUTION_READER_UPDATES = {
    'residual_actor_lifecycle.py': dict(
        current_sha256='5785c6def3cec14a3a994222c90944bbc47d8a70f330a59414a7c2f85502d299',
        installed_target='ResidualOnlySourceActors.ordered_evaluations',
        invocation_boundary='Only after CurrentProgrammeSource/actors construction; not original static source assembly',
        cached_graph_constructor_or_source_fields_changed=False),
    'complete_source_pool_lifetime_runtime.py': dict(
        current_sha256='dda21dcfe4528d660a8723acb94211ba9d49102c0dfdfaba1bb9591e0cfed96e',
        installed_targets=['AbsoluteGeometryGroupPool.release_idle',
            'CanonicalCompleteProgrammeSource.blocks', 'SourceWithDirectManualCentres.blocks'],
        invocation_boundary='Only post-assembly full source/FD execution; only this version receipt wording changed',
        cached_graph_constructor_or_source_fields_changed=False),
}
_POST_ASSEMBLY_CSR_SOURCE_SHA = {
    'programme_csr_three_span_merge_runtime.py': 'ac3a953ff8c5bb0f8608359071a936981f5e68c4637aa2f3e1afbe3296fbde5d',
    'native/programme_csr_three_span_merge.c': '03e5e1693638f028789acd49073e2dbf846961173fb98c0f8c25e4d2af815d1b',
}
_POST_ASSEMBLY_WIDE_SOURCE_SHA = {
    'wide_source_scene_runtime.py': '87888186815c56fb61e32f5b883028ced46cc9049626cf3665af286b860577c8',
}
_WRITER_NAMES = frozenset(('SCHEMA', 'CAPTURE_STATIC_LOCAL_NAMES',
    'RESTORED_DRIVER_LOCAL_NAMES', '_EMPTY_CACHES', '_INVALIDATED',
    '_DERIVED_ARRAYS', '_FORBIDDEN_EXECUTION', '_json_literal', '_filtered_state',
    '_reducer_description', '_original_two_item_reduction', '_OriginalSkeletonPickler',
    '_HashingOutput', 'save_final_source_state_skeleton'))
_ADAPTER_WRITER_NAMES = frozenset(('_MISSING_LOCALS', '_INPUT_READ_SCOPE',
    '_AUDIT_INSTALLED', '_AssemblyInputReads', '_input_open_audit',
    '_install_input_read_audit', '_literal', '_declared_references',
    '_scope_references', '_runtime_implementation_references', '_qualified_priors',
    '_structural_options'))


def _contract_nodes(contents, names, methods=()):
    tree = ast.parse(contents)
    result = []
    found = set()
    for node in tree.body:
        name = getattr(node, 'name', None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            result.append(node); found.add(name)
        if isinstance(node, ast.ClassDef) and node.name == 'Named66SourceSkeletonAdapter':
            for method in node.body:
                if getattr(method, 'name', None) in methods:
                    result.append(method); found.add(method.name)
    if found != set(names) | set(methods):
        raise SkeletonCacheRejected('Static writer contract definitions changed')
    return ast.dump(ast.Module(body=result, type_ignores=[]), include_attributes=False)


def _central_static_contract(contents):
    """Ignore only the complete LP backend's execution installation/receipt."""
    tree = ast.parse(contents)
    class RemoveExecutionLP(ast.NodeTransformer):
        def __init__(self):
            self.csr_imports = self.csr_installs = self.csr_snapshots = 0
            self.wide_imports = self.wide_installs = self.wide_snapshots = 0

        def visit_ImportFrom(self, node):
            if node.level == 1 and node.module == 'full_priority_clarabel_runtime':
                if [(item.name, item.asname) for item in node.names] != [('install_full_priority_clarabel_runtime', None)]:
                    raise SkeletonCacheRejected('Unexpected LP execution import')
                return None
            if node.level == 1 and node.module == 'programme_csr_three_span_merge_runtime':
                if [(item.name, item.asname) for item in node.names] != [('install_programme_csr_three_span_merge_runtime', None)]:
                    raise SkeletonCacheRejected('Unexpected post-assembly CSR import')
                self.csr_imports += 1
                return None
            if node.level == 1 and node.module == 'wide_source_scene_runtime':
                if [(item.name, item.asname) for item in node.names] != [('install_wide_source_scene_runtime', None)]:
                    raise SkeletonCacheRejected('Unexpected post-assembly wide scene import')
                self.wide_imports += 1
                return None
            return node

        def visit_Assign(self, node):
            if len(node.targets) == 1:
                target = node.targets[0]
                if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                        and target.value.id == 'receipt' and isinstance(target.slice, ast.Constant)
                        and target.slice.value == 'programme_CSR_three_span_merge'):
                    expected = ast.parse("install_programme_csr_three_span_merge_runtime(report_directory=destination / 'programme_csr_three_span_merge_execution')", mode='eval').body
                    if ast.dump(node.value, include_attributes=False) != ast.dump(expected, include_attributes=False):
                        raise SkeletonCacheRejected('Post-assembly CSR installation arguments changed')
                    self.csr_installs += 1
                    return None
                if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                        and target.value.id == 'receipt' and isinstance(target.slice, ast.Constant)
                        and target.slice.value == 'wide_numeric_source_scene'):
                    expected = ast.parse("install_wide_source_scene_runtime(driver, report_directory=destination / 'wide_numeric_source_scene_execution')", mode='eval').body
                    if ast.dump(node.value, include_attributes=False) != ast.dump(expected, include_attributes=False):
                        raise SkeletonCacheRejected('Post-assembly wide scene installation arguments changed')
                    self.wide_installs += 1
                    return None
                if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                        and target.value.id == 'receipt' and isinstance(target.slice, ast.Constant)
                        and target.slice.value == 'full_priority_Clarabel'):
                    if not (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                            and node.value.func.id in ('install_full_priority_clarabel_runtime', 'dict')):
                        raise SkeletonCacheRejected('Unexpected LP execution receipt')
                    if node.value.func.id == 'dict':
                        values = {item.arg:item.value for item in node.value.keywords}
                        if (node.value.args or set(values) != {'enabled', 'reason'}
                                or not isinstance(values['enabled'], ast.Constant)
                                or values['enabled'].value is not False
                                or not isinstance(values['reason'], ast.Constant)
                                or not isinstance(values['reason'].value, str)):
                            raise SkeletonCacheRejected('LP removal must be a literal disabled execution receipt')
                    else:
                        expected = ast.parse("install_full_priority_clarabel_runtime(report_directory=destination / 'full_priority_Clarabel_execution')", mode='eval').body
                        if ast.dump(node.value, include_attributes=False) != ast.dump(expected, include_attributes=False):
                            raise SkeletonCacheRejected('LP installation arguments changed')
                    return None
            return self.generic_visit(node)

        def visit_AugAssign(self, node):
            if isinstance(node.target, ast.Name) and node.target.id == 'helper_names' and isinstance(node.value, ast.Tuple):
                removed = [item for item in node.value.elts if isinstance(item, ast.Constant)
                    and item.value in (_POST_ASSEMBLY_CSR_SOURCE_SHA | _POST_ASSEMBLY_WIDE_SOURCE_SHA)]
                self.csr_snapshots += sum(item.value in _POST_ASSEMBLY_CSR_SOURCE_SHA for item in removed)
                self.wide_snapshots += sum(item.value in _POST_ASSEMBLY_WIDE_SOURCE_SHA for item in removed)
                node.value.elts = [item for item in node.value.elts if item not in removed]
            return self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == 'install_residual_actor_lifecycle':
                for keyword in node.keywords:
                    if keyword.arg == 'retain_idle_residual_actors':
                        if not isinstance(keyword.value, ast.Constant) or keyword.value.value is not True:
                            raise SkeletonCacheRejected('Only the audited original residual retention execution policy is compatible')
                node.keywords = [item for item in node.keywords if item.arg != 'retain_idle_residual_actors']
            return self.generic_visit(node)
    normalizer = RemoveExecutionLP()
    result = normalizer.visit(tree)
    if any((normalizer.csr_imports, normalizer.csr_installs, normalizer.csr_snapshots)):
        if (normalizer.csr_imports, normalizer.csr_installs, normalizer.csr_snapshots) != (1, 1, 2):
            raise SkeletonCacheRejected('Complete explicit post-assembly CSR execution contract required')
        root = Path(__file__).resolve().parent
        for relative, expected in _POST_ASSEMBLY_CSR_SOURCE_SHA.items():
            if expected is None or _sha(root/relative) != expected:
                raise SkeletonCacheRejected('Audited post-assembly CSR execution source changed: '+relative)
    if any((normalizer.wide_imports, normalizer.wide_installs, normalizer.wide_snapshots)):
        if (normalizer.wide_imports, normalizer.wide_installs, normalizer.wide_snapshots) != (1, 1, 1):
            raise SkeletonCacheRejected('Complete explicit post-assembly wide scene execution contract required')
        root = Path(__file__).resolve().parent
        for relative, expected in _POST_ASSEMBLY_WIDE_SOURCE_SHA.items():
            if _sha(root/relative) != expected:
                raise SkeletonCacheRejected('Audited post-assembly wide scene source changed: '+relative)
    return ast.dump(result, include_attributes=False)


def compatible_static_reader(path, expected, actual=None):
    """Certify one exact old writer/installer against the current implementation.

    Exact archived predecessors must retain the same writer contract. Original
    actual reads/registered producers never invoke this compatibility route.
    """
    path = Path(path).resolve()
    root = Path(__file__).resolve().parent
    measured_location_fix = _MEASURED_LOCATION_READER_PREDECESSORS.get(path.name) == expected
    if path.parent != root or not (measured_location_fix or _STATIC_READER_PREDECESSORS.get(path.name) == expected):
        return None
    from .config import data_root
    snapshot = (_MEASURED_LOCATION_READER_SNAPSHOT if measured_location_fix
                else 'cold_start_candidate_20261010_1433.bootstrap')
    previous = data_root()/snapshot/'execution_sources'/path.name
    if not previous.is_file() or _sha(previous) != expected:
        raise SkeletonCacheRejected('Certified predecessor source snapshot changed: '+str(previous))
    before, after = previous.read_text(encoding='utf8'), path.read_text(encoding='utf8')
    if path.name in _PURE_EXECUTION_READER_UPDATES:
        approved = _PURE_EXECUTION_READER_UPDATES[path.name]
        current = actual if actual is not None else _sha(path)
        if current != approved['current_sha256']:
            raise SkeletonCacheRejected('Audited post-assembly execution implementation changed: '+str(path))
        old_contract = new_contract = json.dumps(approved, sort_keys=True)
        if path.name == 'complete_source_pool_lifetime_runtime.py':
            class ReceiptWording(ast.NodeTransformer):
                def visit_keyword(self, node):
                    if node.arg in ('original_residual_actor_release_before_J_retained',
                                    'residual_actor_lifetime_owned_by_separate_lifecycle_policy'):
                        if not isinstance(node.value, ast.Constant) or node.value.value is not True:
                            raise SkeletonCacheRejected('Unexpected pool lifecycle receipt wording')
                        node.arg = 'residual_actor_lifetime_owned_by_separate_lifecycle_policy'
                    return node
            old_nodes = ast.dump(ReceiptWording().visit(ast.parse(before)), include_attributes=False)
            new_nodes = ast.dump(ReceiptWording().visit(ast.parse(after)), include_attributes=False)
            if old_nodes != new_nodes:
                raise SkeletonCacheRejected('Pool lifecycle changed beyond its audited receipt wording')
    elif path.name == 'source_state_skeleton_cache_runtime.py':
        old_contract = _contract_nodes(before, _WRITER_NAMES)
        new_contract = _contract_nodes(after, _WRITER_NAMES)
    elif path.name == 'named66_source_skeleton_adapter_runtime.py':
        methods = ('capture_input_reads', 'record', 'location', 'save', '_save')
        old_contract = _contract_nodes(before, _ADAPTER_WRITER_NAMES, methods)
        new_contract = _contract_nodes(after, _ADAPTER_WRITER_NAMES, methods)
        def current_options(contents):
            assignment = next(node for node in ast.parse(contents).body
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == '_CURRENT_OPTIONS' for target in node.targets))
            if not (isinstance(assignment.value, ast.Call) and isinstance(assignment.value.func, ast.Name)
                    and assignment.value.func.id == 'frozenset' and len(assignment.value.args) == 1):
                raise SkeletonCacheRejected('Explicit current execution options required')
            return frozenset(ast.literal_eval(assignment.value.args[0]))
        if current_options(after) != current_options(before) | {'accepted_steps',
                'normal_workers', 'geometry_fd_workers', 'reuse_source_jacobian'}:
            raise SkeletonCacheRejected('Certified current execution option classification changed')
    else:
        old_contract, new_contract = _central_static_contract(before), _central_static_contract(after)
    if old_contract != new_contract:
        raise SkeletonCacheRejected('Certified static writer/installer contract changed: '+str(path))
    return dict(path=str(path), predecessor_sha256=expected,
                current_sha256=actual if actual is not None else _sha(path),
                predecessor_source_snapshot=str(previous),
                measured_location_reader_local_restore=measured_location_fix,
                unchanged_static_contract_sha256=hashlib.sha256(old_contract.encode('utf8')).hexdigest(),
                original_read_and_registered_producer_SHA_exempted=False)

# Names are an integration contract: no driver locals()/closure is pickled.
# The numerical seed-dependent names are returned by the restore callback.
CAPTURE_STATIC_LOCAL_NAMES = (
    'state', 'args', 'factorold', 'template', 'parent', 'physical_geometry_initializer',
    'initial', 'mount_working_constraints', 'mount_parameter_bounds',
    'pivot_outside_constraints', 'ice_pixel_correction', 'far_board_role_correction',
    'motion_input_compaction',
)
RESTORED_DRIVER_LOCAL_NAMES = (
    'q', 'oldfull', 'inherited_current_world', 'inherited_geometry_tail',
    'inherited_circle_suffix', 'inherited_pattern_suffix', 'inherited_checkpoint_q9',
    'continued_damping', 'continued_damping_identity', 'continued_line_search_fraction',
    'inherited_adaptive_line_search', 'profiles', 'pending_compacted_profiles',
    'inherited', 'core_count', 'world_indices', 'evaluation_base_full_count',
)

_EMPTY_CACHES = frozenset(('_scene_cache', '_value_cache', '_successful_world_validation',
    '_source_camera_map_cache', 'source_cameras', 'source_camera_signatures',
    '_native_camera_valid_worlds', '_absolute_geometry_fd_stencil_cache',
    '_derived_course_center_cache'))
_INVALIDATED = frozenset(('_course_current_geometry_key', '_course_camera_cache_key',
    '_course_camera_cache', '_current_binding_cache', '_facet_course_cache',
    'source_camera_signature'))
_DERIVED_ARRAYS = frozenset(('course_current_plane_values', 'course_current_plane_jacobian',
    'structured_current_extent_jacobian'))
_FORBIDDEN_EXECUTION = frozenset(('source_actors', '_absolute_geometry_fd_pool',
    'after_programme_jacobian', 'after_shared_pattern_evaluation', 'runtime_report_path'))


class SkeletonCacheRejected(ValueError):
    pass


def _sha(path):
    # Every load reads bytes again; stat is never source authority across runs.
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def _json_literal(value):
    def encode(item):
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise TypeError('Numeric source structure required')
            return dict(dtype=item.dtype.str, shape=list(item.shape), bytes=item.tobytes().hex())
        if isinstance(item, np.generic):
            return encode(np.asarray(item))
        if isinstance(item, Path):
            return str(item.resolve())
        raise TypeError('Explicit literal source structure required: '+type(item).__name__)
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False, default=encode))


def authenticate_source_skeleton_inputs(*, source_references, structural_options):
    """Authenticate all caller-declared source refs and the registered closure.

    source_references must be the complete source/input closure already owned
    by the driver, including child role/RGB/admission/clock/compaction refs.
    Numerical camera-seed q/profile, output and worker settings are separate.
    No path walk can safely infer this semantic distinction on the caller's
    behalf; completeness is an explicit caller contract recorded in the key.
    """
    from . import runtime
    root = Path(__file__).resolve().parent
    actual_files = {}
    def actual_sha(path):
        path = Path(path).resolve()
        if path not in actual_files:
            actual_files[path] = _sha(path)
        return actual_files[path]
    registry = []
    for name in ('source_manifest.json', 'event_factories_manifest.json', 'legacy_dependencies.json'):
        path = root/name
        manifest = json.loads(path.read_text(encoding='utf8'))
        registry.append(dict(path=str(path), sha256=actual_sha(path)))
        rows = list(manifest.get('modules', {}).values()) + list(manifest.get('native_sources', ()))
        for row in rows:
            target = (root/row['path']).resolve()
            actual = actual_sha(target)
            if actual != row['sha256']:
                raise SkeletonCacheRejected('Registered original producer changed: '+str(target))
            registry.append(dict(path=str(target), sha256=actual))
    # Preserve the same event-factory alias/registration authentication too.
    runtime.verify_legacy_dependencies()
    references = []
    seen = {}
    for reference in source_references:
        path = Path(reference['path']).resolve()
        expected = reference['sha256']
        if str(path) in seen and seen[str(path)] != expected:
            raise SkeletonCacheRejected('Conflicting source authority: '+str(path))
        actual = actual_sha(path)
        if actual != expected:
            if (reference.get('role') != 'installed_static_construction_runtime'
                    or compatible_static_reader(path, expected, actual) is None):
                raise SkeletonCacheRejected('Original source/input bytes changed: '+str(path))
        seen[str(path)] = expected
        references.append(dict(path=str(path), sha256=expected,
                               role=reference.get('role', 'original_source_input')))
    if not references or not structural_options:
        raise SkeletonCacheRejected('Complete explicit source references and structural options required')
    return dict(schema=SCHEMA,
        registered_sources=sorted({(r['path'], r['sha256']) for r in registry}),
        original_source_references=sorted(references, key=lambda r:(r['path'], r['role'])),
        complete_structural_options=_json_literal(structural_options),
        registered_and_source_file_actual_SHA_required_each_load=True)


def _filtered_state(obj, state, root_state):
    if not isinstance(state, dict):
        return state
    result = dict(state)
    for key in _FORBIDDEN_EXECUTION:
        if key in result:
            value = result[key]
            if value is not None:
                raise SkeletonCacheRejected('Snapshot boundary contains live execution field: '+key)
            result.pop(key)
    for key in _EMPTY_CACHES:
        if key in result:
            # These are declared derived caches, not source registrations.
            result[key] = OrderedDict() if isinstance(result[key], OrderedDict) else {}
    for key in _INVALIDATED:
        if key in result:
            result[key] = None
    for key in _DERIVED_ARRAYS:
        result.pop(key, None)
    if 'profile_weights' in result:
        result['profile_weights'] = None
        result['robust_phase'] = False
    if obj is root_state:
        result['q'] = None  # An actual checkpoint callback must restore it.
        result.pop('seed_contract', None)
    prior = result.get('priors')
    if callable(prior) and prior.__name__ == '<lambda>':
        if result.get('_isolated_cache_heightless_prior') is not True:
            raise SkeletonCacheRejected('Unauthenticated instance prior closure cannot be snapshotted')
        result.pop('priors')
    return result


def _reducer_description(obj, reduced):
    name = type(obj).__module__+'.'+type(obj).__qualname__
    if not isinstance(reduced, tuple):
        return name+'; reduction='+type(reduced).__name__
    restore = reduced[0] if reduced else None
    restore_name = getattr(restore, '__module__', type(restore).__module__)+'.'+getattr(restore, '__qualname__', type(restore).__qualname__)
    args = reduced[1] if len(reduced)>1 else None
    args_shape = [type(value).__module__+'.'+type(value).__qualname__ for value in args] if isinstance(args, tuple) else type(args).__name__
    return name+'; reduction_length='+str(len(reduced))+'; restore='+restore_name+'; argument_types='+str(args_shape)


def _original_two_item_reduction(obj, reduced, root_state):
    """Preserve certified original constructor/restore functions verbatim."""
    from shared_unmeasured_advertisement_geometry_candidate.paired_locations_shared_sizes_geometry import _restore_paired_geometry
    from shared_unmeasured_advertisement_geometry_candidate.shared_unmeasured_advertisement_geometry import SharedUnmeasuredAdvertisementGeometry
    from shared_ice_hockey_geometry_candidate.shared_ice_hockey_numeric_geometry import _restore as restore_hockey
    from shared_advertisement_pairs_candidate.shared_rink_numeric_geometry_lanes_ad_pairs import _restore_combined
    from shared_board_course_candidate.shared_rink_course_numeric_geometry import _restore as restore_course
    restore, args = reduced
    if restore is type(obj) and isinstance(obj, SharedUnmeasuredAdvertisementGeometry) and isinstance(args, tuple) and len(args) == 2:
        return reduced
    if not isinstance(args, tuple) or not args or args[0] is not type(obj):
        raise SkeletonCacheRejected('Unsupported original source object state reducer: '+_reducer_description(obj, reduced))
    if restore is _restore_paired_geometry and len(args) == 2 and args[1] is vars(obj):
        # This original restore allocates without calling a constructor, so
        # its full state dictionary must receive the same cache filtering.
        return restore, (args[0], _filtered_state(obj, args[1], root_state))
    if restore in (restore_hockey, _restore_combined, restore_course):
        # These original constructors do not serialize an instance state
        # dictionary. Their original argument tuple already omits caches.
        return reduced
    raise SkeletonCacheRejected('Unsupported original source object state reducer: '+_reducer_description(obj, reduced))


class _OriginalSkeletonPickler(pickle.Pickler):
    def __init__(self, stream, root_state):
        super().__init__(stream, protocol=5)
        self.root_state = root_state

    def reducer_override(self, obj):
        # No saved source residual/J/normal matrix is permitted in this boundary.
        if type(obj).__module__.startswith('scipy.sparse'):
            raise SkeletonCacheRejected('Sparse derivative/normal value present in source skeleton: '+type(obj).__module__+'.'+type(obj).__qualname__)
        values = getattr(obj, '__dict__', None)
        if values is None or isinstance(obj, type) or callable(obj):
            return NotImplemented
        if not any(key in values for key in (_EMPTY_CACHES | _INVALIDATED | _DERIVED_ARRAYS |
                    _FORBIDDEN_EXECUTION | {'profile_weights', 'priors'})) and obj is not self.root_state:
            return NotImplemented
        reduced = obj.__reduce_ex__(5)
        if isinstance(reduced, tuple) and len(reduced) == 2:
            return _original_two_item_reduction(obj, reduced, self.root_state)
        if not isinstance(reduced, tuple) or len(reduced) < 3:
            raise SkeletonCacheRejected('Unsupported original source object state reducer: '+_reducer_description(obj, reduced))
        state = _filtered_state(obj, reduced[2], self.root_state)
        return reduced[:2] + (state,) + reduced[3:]


class _HashingOutput:
    """Stream protocol5 bytes once, including large PickleBuffer writes."""
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.nbytes = 0

    def write(self, value):
        view = value.raw() if isinstance(value, pickle.PickleBuffer) else memoryview(value)
        try:
            count = self.stream.write(view)
            if count != view.nbytes:
                raise OSError('Original source skeleton write was incomplete')
            self.digest.update(view)
            self.nbytes += count
            return count
        finally:
            view.release()


def save_final_source_state_skeleton(destination, *, state, static_locals, capold,
        source_references, structural_options, assembled_q_sha256):
    """Save only from the next necessary real assembly, before raw/actors.

    Unknown closures/types fail explicitly. The live state is never cleared
    or mutated; the original pickle reducers receive filtered state copies.
    """
    from prepared_cache_helpers import install_mappingproxy_pickle, install_geometry_unpickle
    install_mappingproxy_pickle(); install_geometry_unpickle()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    identity = authenticate_source_skeleton_inputs(source_references=source_references,
                                                   structural_options=structural_options)
    if getattr(state, 'source_actors', None) is not None:
        raise SkeletonCacheRejected('Final source skeleton must precede original actor construction')
    missing = set(CAPTURE_STATIC_LOCAL_NAMES)-set(static_locals)
    if missing:
        raise SkeletonCacheRejected('Explicit supported assembly locals missing: '+str(sorted(missing)))
    saved_locals = {key:static_locals[key] for key in CAPTURE_STATIC_LOCAL_NAMES}
    if saved_locals['state'] is not state:
        raise SkeletonCacheRejected('Assembly state and explicit locals differ')
    # The template keeps immutable original aux/mask/scale authority, never an
    # old numerical profile to be mistaken for the currently restored one.
    saved_locals['template'] = {k:v for k,v in saved_locals['template'].items()
                                if not k.startswith('profile__')}
    payload = dict(schema=SCHEMA, static_locals=saved_locals,
                   capold_arrays={k:np.asarray(capold[k]) for k in ('quadratic', 'scales')})
    path = destination/'source_state_skeleton.pickle'
    if path.exists():
        raise SkeletonCacheRejected('Existing immutable source skeleton must not be overwritten')
    temporary = destination/('.unpublished_'+uuid.uuid4().hex+'.pickle')
    try:
        with temporary.open('xb') as stream:
            output = _HashingOutput(stream)
            _OriginalSkeletonPickler(output, state).dump(payload)
        temporary.rename(path)
    finally:
        # Only this fresh unpublished artifact is owned by this call.
        temporary.unlink(missing_ok=True)
    receipt = dict(schema=SCHEMA, identity=identity, pickle_sha256=output.digest.hexdigest(),
        serialized_bytes=output.nbytes, assembly_original_q_sha256=assembled_q_sha256,
        historical_compaction_diagnostic_q_sha256=assembled_q_sha256,
        compaction_diagnostic_current_q_PASS=False,
        source_rows_and_model_still_original=True, current_checkpoint_restore_callback_required=True,
        original_constructor_chart_q_is_static_origin_not_resume_authority=True,
        cached_current_q_profiles_truth_limits_damping_used_as_resume_authority=False,
        source_residual_J_H_or_actors_cached=False,
        helper_sha256=_sha(__file__), static_local_names=list(CAPTURE_STATIC_LOCAL_NAMES))
    (destination/'source_state_skeleton_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    return receipt


def load_final_source_state_skeleton(directory, *, source_references, structural_options,
        restore_checkpoint, output_directory):
    """Return explicit locals only AFTER a current checkpoint has been restored."""
    from . import install
    install()
    from prepared_cache_helpers import install_mappingproxy_pickle, install_geometry_unpickle
    from benchmark_harness import restore_local_priors
    install_mappingproxy_pickle(); install_geometry_unpickle()
    directory = Path(directory).resolve()
    receipt = json.loads((directory/'source_state_skeleton_receipt.json').read_text(encoding='utf8'))
    identity = authenticate_source_skeleton_inputs(source_references=source_references,
                                                   structural_options=structural_options)
    # JSON turns registry tuple pairs into lists; compare the canonical literal.
    if receipt.get('schema') != SCHEMA or _json_literal(identity) != receipt.get('identity'):
        raise SkeletonCacheRejected('Source skeleton original producer/input/structure identity differs')
    if (receipt.get('helper_sha256') != _sha(__file__)
            and compatible_static_reader(__file__, receipt.get('helper_sha256')) is None):
        raise SkeletonCacheRejected('Source skeleton loader implementation changed')
    path = directory/'source_state_skeleton.pickle'
    if _sha(path) != receipt['pickle_sha256']:
        raise SkeletonCacheRejected('Authenticated source skeleton pickle bytes changed')
    with path.open('rb') as stream:
        payload = pickle.load(stream)
    if payload.get('schema') != SCHEMA:
        raise SkeletonCacheRejected('Original source skeleton payload schema changed')
    values = payload['static_locals']
    state = values['state']
    restore_local_priors(state)
    if state.q is not None or getattr(state, 'source_actors', None) is not None:
        raise SkeletonCacheRejected('Snapshot supplied a current q or actor instead of original source inputs')
    if not callable(restore_checkpoint):
        raise SkeletonCacheRejected('Current original checkpoint restore callback required')
    restored = restore_checkpoint(state)
    if not isinstance(restored, dict) or set(RESTORED_DRIVER_LOCAL_NAMES)-set(restored):
        raise SkeletonCacheRejected('Current checkpoint restore omitted original continuation locals')
    values.update(restored)
    values['capold'] = payload['capold_arrays']
    state.output = Path(output_directory).resolve()
    values['args'].output = state.output
    values['geometry'] = state.geometry.current
    values['contract'] = state.structured_programme_contract
    values['existing17_check'] = None
    # Keep old diagnostic facts byte-identical for the existing semantic gate,
    # but record their history outside it. They are never current guard/J PASS.
    values['skeleton_cache_provenance'] = dict(receipt_path=str(directory/'source_state_skeleton_receipt.json'),
        pickle_sha256=receipt['pickle_sha256'], assembly_original_q_sha256=receipt['assembly_original_q_sha256'],
        compaction_retained_r_J_diagnostics_historical=True,
        current_complete_q_sha256=restored['current_complete_q_sha256'],
        original_postrole_assembly_q_sha256=hashlib.sha256(np.asarray(values['q']).tobytes()).hexdigest(),
        current_source_guard_and_fresh_source_J_required=True,
        source_semantic_compaction_literal_not_rewritten=True)
    return values


def restore_final_skeleton_checkpoint(state, checkpoint_arrays, *, current_seed_parts,
        circle_enabled, placement_columns, adaptive_source_backtracking=False):
    """Restore any current same-structure seed through the ORIGINAL charts.

    No len(old bundle.a) compatibility branch is used on the compacted graph.
    Appended/changed physical layouts reject explicitly and take original cold
    assembly. Truth limits, damping identity gates and fresh guard/J stay caller
    owned and execute after this callback as in the original driver.
    """
    from structured_programme_source_candidate.paired_programme_chart import CompleteProgrammeParameterLayout, CORE_COUNT, OLD
    from z190_joint_source_solver import make_profile_update, _frozen_profile_token
    saved = checkpoint_arrays
    manifest = json.loads(str(np.asarray(saved['aux__current_complete_parameter_layout_json']).item()))
    extra = state.geometry.parameter_count-66
    if int(manifest['additional_world_parameter_count']) != extra:
        raise SkeletonCacheRejected('Current checkpoint requires a changed/append physical source layout')
    original_count = int(manifest['original_complete_parameter_count'])
    layout = CompleteProgrammeParameterLayout(original_count, extra)
    if _json_literal(layout.manifest()) != _json_literal(manifest):
        raise SkeletonCacheRejected('Current checkpoint complete canonical layout differs')
    canonical = np.asarray(saved['q'], float).copy()
    evaluation = layout.evaluation_q(canonical)
    chart = state.structured_programme_coordinate_chart
    if chart.world_parameter_count != state.geometry.parameter_count or chart.core_count != CORE_COUNT+extra:
        raise SkeletonCacheRejected('Final source skeleton physical chart differs')
    placement_columns = int(placement_columns)
    if placement_columns not in (0, 2, 4):
        raise SkeletonCacheRejected('Explicit original placement suffix size required')
    prefix = canonical[:original_count]
    placement = prefix[-placement_columns:].copy() if placement_columns else None
    if placement_columns:
        prefix = prefix[:-placement_columns]
    legacy, world66, circle = current_seed_parts(prefix, circle_enabled=bool(circle_enabled))
    if world66 is None:
        raise SkeletonCacheRejected('Original current_seed_parts did not recognize the complete prefix')
    tail = canonical[original_count:].copy()
    world = np.r_[world66, tail]
    reconstructed = chart.initialize_full_q(legacy, world)
    if circle is not None:
        reconstructed = np.r_[reconstructed, circle]
    if placement is not None:
        reconstructed = np.r_[reconstructed, placement]
    if not np.array_equal(reconstructed, evaluation):
        raise SkeletonCacheRejected('Original seed-parts/chart mapping changed complete checkpoint bytes')
    state.q = chart.core_q(evaluation[:chart.core_count]).copy()
    profiles = {key[9:]:np.asarray(value) for key,value in saved.items() if key.startswith('profile__')}
    updater = make_profile_update(state.blocks)
    updater.restore_checkpoint(profiles)
    inherited = _frozen_profile_token(updater)
    expected_profiles = {key:np.asarray(value) for key,value in updater.iter_checkpoint_arrays()}
    if any(not np.array_equal(expected_profiles[key], value) for key,value in profiles.items()):
        raise SkeletonCacheRejected('Original profile restore did not retain current checkpoint arrays')
    damping = float(np.asarray(saved['damping']))
    if not np.isfinite(damping) or damping <= 0:
        raise SkeletonCacheRejected('Original finite continued damping required')
    fraction = 1.; adaptive = False
    if 'next_line_search_fraction' in saved:
        fraction = float(np.asarray(saved['next_line_search_fraction'])); adaptive = True
    elif adaptive_source_backtracking and 'history_json' in saved:
        from calibration.solver_runtime.checkpoint_history import read_checkpoint_history
        accepted = [record for record in read_checkpoint_history(saved)
                    if record.get('accepted')]
        if accepted and accepted[-1].get('global_schur_trial_records'):
            fraction = float(accepted[-1]['global_schur_trial_records'][-1]['whole_source_step_fraction'])
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise SkeletonCacheRejected('Original finite line-search continuation required')
    q9 = None
    if 'aux__shared_pattern_start' in saved:
        start = int(np.asarray(saved['aux__shared_pattern_start']))
        if not 0 <= start <= OLD.core_count-9:
            raise SkeletonCacheRejected('Original preserved q9 prefix is absent')
        q9 = canonical[start:start+9].copy()
    # At this exact original assembly boundary circle/placement have not yet
    # been appended. The original factories below take q[-25:] as wide camera
    # parameters, then append their separately inherited suffixes themselves.
    return dict(q=evaluation[:chart.full_count].copy(), oldfull=np.asarray(legacy).copy(),
        current_complete_q_sha256=hashlib.sha256(canonical.tobytes()).hexdigest(),
        inherited_current_world=np.asarray(world66).copy(), inherited_geometry_tail=tail,
        inherited_circle_suffix=circle, inherited_pattern_suffix=placement, inherited_checkpoint_q9=q9,
        continued_damping=damping,
        continued_damping_identity=dict(quadratic=np.asarray(saved['quadratic']).copy(),
            scales=np.asarray(saved['scales']).copy(), identity=json.loads(str(np.asarray(saved['identity_json']).item()))),
        continued_line_search_fraction=fraction, inherited_adaptive_line_search=adaptive,
        profiles=profiles, pending_compacted_profiles=None, inherited=inherited,
        core_count=chart.core_count, world_indices=np.asarray(state.geometry_parameter_indices).copy(),
        evaluation_base_full_count=CORE_COUNT+28+extra)
