"""Unused named66 integration for the existing final-source skeleton format.

Only static source assembly is replaceable. Current checkpoint reconstruction,
camera binding, truth/continuation gates, world guard and fresh J remain current.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types
import uuid

import numpy as np

from . import source_state_skeleton_cache_runtime as skeleton


# These options only apply after source assembly or select current execution.
# All other parser fields, including every structural boolean, remain keyed.
_CURRENT_OPTIONS = frozenset(('output', 'camera_seed', 'truth_restoration_state',
    'serial_source', 'serial_current_patterns', 'residual_source_workers',
    'residual_native_threads', 'initial_damping', 'evaluate_only', 'readonly_profile',
    'accepted_steps', 'normal_workers', 'geometry_fd_workers', 'reuse_source_jacobian',
    'reuse_complete_linearization', 'reuse_source_blocks_from', 'reuse_wide_linearization',
    'verify_source_blocks_from', 'verified_source_prefix_report'))

_MISSING_LOCALS = ('mount_reinitialization', 'pivot_reinitialization',
    'pose_feasibility_initializer', 'absolute_camera_initializer',
    'shared_linear_constraints', 'camera_error')

_INPUT_READ_SCOPE = ContextVar('named66_static_input_reads', default=None)
_AUDIT_INSTALLED = False


class _AssemblyInputReads:
    """Record only files actually read by the original static assembly.

    The audit hook observes opens; it never replaces the original IO call.
    Hashing happens with this scope suspended, before the original read, and
    every captured file is rehashed before publishing the static skeleton.
    Current checkpoint bytes are explicitly restored by the original chart.
    """
    def __init__(self, camera_seed):
        self.excluded = None if camera_seed is None else Path(camera_seed).resolve()
        self.files = {}
        self.absent = set()
        self.errors = []

    def opened(self, path, mode, flags):
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        if isinstance(mode, str):
            if any(letter in mode for letter in ('w', 'a', '+', 'x')):
                return
        elif (int(flags) & os.O_ACCMODE) != os.O_RDONLY:
            return
        target = Path(os.fsdecode(path)).resolve()
        if target == self.excluded:
            return
        # The installed interpreter/libraries are execution environment,
        # authenticated separately from model inputs by producer versions.
        prefixes = (Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve())
        if any(target.is_relative_to(prefix) for prefix in prefixes):
            return
        if target in self.files:
            return
        token = _INPUT_READ_SCOPE.set(None)
        try:
            if target.is_file():
                self.files[target] = _sha(target)
            else:
                self.absent.add(target)
        except Exception as exc:
            # Observability cannot turn successful original IO into failure.
            # An incomplete observation only rejects this optional capture.
            self.errors.append(type(exc).__name__ + ': ' + str(exc))
        finally:
            _INPUT_READ_SCOPE.reset(token)


def _input_open_audit(event, arguments):
    scope = _INPUT_READ_SCOPE.get()
    if scope is not None and event == 'open':
        try:
            scope.opened(*arguments)
        except Exception as exc:
            scope.errors.append(type(exc).__name__ + ': ' + str(exc))


def _install_input_read_audit():
    global _AUDIT_INSTALLED
    if not _AUDIT_INSTALLED:
        sys.addaudithook(_input_open_audit)
        _AUDIT_INSTALLED = True


def _validate_current_checkpoint_numbers(saved):
    q = np.asarray(saved['q'], float)
    if q.ndim != 1 or not np.isfinite(q).all():
        raise ValueError('Complete original finite q required')
    damping = float(np.asarray(saved['damping']))
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError('Finite continued optimizer damping required')
    if 'next_line_search_fraction' in saved:
        fraction = float(np.asarray(saved['next_line_search_fraction']))
        if not np.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError('Finite original line-search continuation required')


def _sha(path):
    return skeleton._sha(path)


def _literal(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {key:_literal(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_literal(item) for item in value]
    return value


def _structural_options(opt):
    return {key:_literal(value) for key,value in vars(opt).items() if key not in _CURRENT_OPTIONS}


def _refs_from_json(value, found):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value and isinstance(value['path'], str):
            path = Path(value['path']).resolve()
            if path.is_file():
                expected = value['sha256']
                if path in found and found[path] != expected:
                    raise skeleton.SkeletonCacheRejected('Conflicting static source reference: '+str(path))
                found[path] = expected
        for item in value.values():
            _refs_from_json(item, found)
    elif isinstance(value, list):
        for item in value:
            _refs_from_json(item, found)


def _declared_references(opt, declared_roots):
    """Hash each explicit resolved input once; no stat or history-ref keys."""
    roots = [Path(path).resolve() for path in declared_roots]
    roots += [value.resolve() for key,value in vars(opt).items()
              if key not in _CURRENT_OPTIONS and isinstance(value, Path)]
    found = {}
    for path in roots:
        if path in found:
            continue
        if not path.is_file():
            raise skeleton.SkeletonCacheRejected('Declared original static input missing: '+str(path))
        found[path] = _sha(path)
        # A source packet can retain historical path/SHA evidence. Its bytes
        # are authority, but those historical producer versions need not be
        # today's files. Current dependencies are the actual assembly reads,
        # original SHA calls and registered/runtime implementation closure.
    return [dict(path=str(path), sha256=digest, role='explicit_structural_source_input')
            for path,digest in sorted(found.items(), key=lambda pair:str(pair[0]))]


def _scope_references():
    # The existing scope recorded actual original source authentication calls.
    # Its caller-owned closure must also cover defaults not read by those six
    # SHA producers; the adapter requires explicit roots for that part.
    from .startup_source_auth_reuse_runtime import _ACTIVE
    scope = _ACTIVE.get()
    if scope is None or scope.closed:
        raise skeleton.SkeletonCacheRejected('Original authenticated construction scope required for capture')
    return [dict(path=str(path), sha256=entry['sha256'], role='original_source_build_SHA_call')
            for path,entry in sorted(scope.files.items(), key=lambda pair:str(pair[0]))]


def _runtime_implementation_references():
    # Constructor wrappers may already be imported before the read scope.
    # Their actual source bytes (including the central installer) therefore
    # belong to this static assembly's implementation closure too.
    root = Path(__file__).resolve().parent
    paths = {Path(module.__file__).resolve() for module in tuple(sys.modules.values())
             if getattr(module, '__file__', None) and Path(module.__file__).suffix == '.py'
             and Path(module.__file__).resolve().parent == root}
    return [dict(path=str(path), sha256=_sha(path), role='installed_static_construction_runtime')
            for path in sorted(paths, key=str)]


@contextmanager
def _qualified_priors(state):
    """Recognize only the original typed heightless clone's lambda."""
    from structured_programme_source_candidate.paired_programme_blocks import clone_structured_programme_block
    expected = [value for value in clone_structured_programme_block.__code__.co_consts
                if isinstance(value, types.CodeType) and value.co_name == '<lambda>']
    changed = []
    try:
        for block in (*state.blocks, *state.original.blocks):
            bundle = block.bundle
            prior = vars(bundle).get('priors')
            if not callable(prior) or getattr(prior, '__name__', None) != '<lambda>':
                continue
            if getattr(bundle, '_isolated_cache_heightless_prior', False) is True:
                continue
            if (len(expected) != 1 or prior.__closure__ is not None
                    or prior.__defaults__ is None or len(prior.__defaults__) != 1
                    or prior.__defaults__[0] is not bundle or prior.__code__ != expected[0]):
                raise skeleton.SkeletonCacheRejected('Unknown current bundle prior closure')
            existed = '_isolated_cache_heightless_prior' in vars(bundle)
            changed.append((bundle, existed, vars(bundle).get('_isolated_cache_heightless_prior')))
            bundle._isolated_cache_heightless_prior = True
        yield
    finally:
        for bundle, existed, previous in changed:
            if existed:
                bundle._isolated_cache_heightless_prior = previous
            else:
                vars(bundle).pop('_isolated_cache_heightless_prior', None)


def _bind_current_all_blocks(state):
    """Prime all original cameras before the first complete-role plane lookup."""
    from secondary_optics_candidate.secondary_optics_graph import SecondaryOpticsFixedXYZRinkBlock
    role = getattr(state.geometry, '_original_motion_far_board_role_binding', None)
    existing = getattr(state.geometry, '_existing_finite_perimeter_binding', None)
    if role is not None:
        for block in state.blocks:
            if getattr(block, '_original_motion_far_board_role_binding', None) is not role:
                continue
            local = SecondaryOpticsFixedXYZRinkBlock.local(block, state.q)
            cameras = {stamp:np.asarray(block.bundle.camera(local, stamp), float).copy()
                       for stamp in role.source_stamps_by_block[block.name]}
            role.source_cameras.update(cameras)
            if existing is not None:
                existing.source_cameras.update(cameras)
    for block in state.blocks:
        block.local(state.q)


def _restore_original_measured_location_local(values, opt):
    """Restore the installed assembly local from the authenticated source state.

    The installer already stores this exact contract on state before motion
    and absolute compaction. Reinstalling would overwrite its immutable row
    certificates; the cached state and all authority bytes have instead just
    passed the original skeleton authentication.
    """
    from .original_measured_transition_locations import CONTRACT_KEY, SCHEMA, MODE
    contract = getattr(values['state'], CONTRACT_KEY, None)
    path = opt.original_measured_transition_location_source
    if path is None:
        if contract is not None:
            raise skeleton.SkeletonCacheRejected('Unrequested original measured-location authority in source state')
    else:
        path = Path(path).resolve()
        reference = contract.get('packet_reference') if isinstance(contract, dict) else None
        if (not isinstance(reference, dict) or contract.get('schema') != SCHEMA + '.installed'
                or contract.get('observation') != MODE
                or Path(reference.get('path', '')).resolve() != path
                or reference.get('sha256') != _sha(path)):
            raise skeleton.SkeletonCacheRejected('Installed original measured-location authority differs from current option')
    values['original_measured_locations'] = contract


class Named66SourceSkeletonAdapter:
    def __init__(self, *, cache_directory, report_directory, declared_source_roots,
                 complete_static_input_closure_confirmed, current_seed_parts):
        if complete_static_input_closure_confirmed is not True or not declared_source_roots:
            raise ValueError('Caller must declare the complete original static input closure')
        self.cache = Path(cache_directory).resolve()
        self.report = Path(report_directory).resolve()
        self.report.mkdir(parents=True, exist_ok=True)
        self.roots = tuple(declared_source_roots)
        self.seed_parts = current_seed_parts
        self.sha = _sha(__file__)
        self.last_input_reads = None
        _install_input_read_audit()

    @contextmanager
    def capture_input_reads(self, opt):
        from .source_assembly_gc_runtime import defer_automatic_source_gc
        scope = _AssemblyInputReads(opt.camera_seed)
        token = _INPUT_READ_SCOPE.set(scope)
        try:
            with defer_automatic_source_gc('original_static_source_assembly', self.report):
                yield
        finally:
            _INPUT_READ_SCOPE.reset(token)
            self.last_input_reads = scope

    def record(self, **value):
        value.update(schema='rinkspace.named66-final-source-skeleton-adapter.v1',
                     wall_time_ns=time.time_ns())
        with (self.report/'named66_source_skeleton_execution.jsonl').open('a', encoding='utf8') as output:
            output.write(json.dumps(value, sort_keys=True)+'\n')

    def location(self, structural, roots):
        key = hashlib.sha256(json.dumps(dict(options=structural, roots=roots),
            sort_keys=True, separators=(',', ':')).encode('utf8')).hexdigest()
        return self.cache/key

    def load(self, opt, output):
        from .source_assembly_gc_runtime import defer_automatic_source_gc
        with defer_automatic_source_gc('authenticated_source_skeleton_load', self.report):
            return self._load(opt, output)

    def _load(self, opt, output):
        if opt.camera_seed is None:
            self.record(route='original_static_source_assembly', reason='Explicit complete current checkpoint required')
            return None
        # Invalid current numerical state is an original error, not a cache
        # integration failure to retry through source reconstruction.
        with np.load(opt.camera_seed, allow_pickle=False) as archive:
            saved = {key:archive[key].copy() for key in archive.files}
        _validate_current_checkpoint_numbers(saved)
        try:
            structural = _structural_options(opt)
            roots = _declared_references(opt, self.roots)
            directory = self.location(structural, roots)
            receipt_path = directory/'named66_adapter_receipt.json'
            if not receipt_path.is_file():
                # The first immutable capture keyed four post-assembly
                # execution settings. Admit it only through the exact old
                # writer contract, with every old source reference intact.
                matches = []
                for candidate in self.cache.glob('*/named66_adapter_receipt.json'):
                    previous = json.loads(candidate.read_text(encoding='utf8'))
                    if (previous.get('adapter_sha256') == skeleton._STATIC_READER_PREDECESSORS.get(Path(__file__).name)
                            and {key:value for key,value in previous.get('structural_options', {}).items()
                                 if key not in _CURRENT_OPTIONS} == structural
                            and previous.get('declared_root_references') == roots):
                        matches.append((candidate, previous))
                if not matches:
                    return None
                if len(matches) != 1:
                    raise skeleton.SkeletonCacheRejected('Multiple certified predecessor source captures match')
                receipt_path, receipt = matches[0]
            else:
                receipt = json.loads(receipt_path.read_text(encoding='utf8'))
            reader_bridge = None
            if receipt['adapter_sha256'] != self.sha:
                reader_bridge = skeleton.compatible_static_reader(__file__, receipt['adapter_sha256'], self.sha)
            saved_structural = receipt['structural_options']
            if (receipt['adapter_sha256'] != self.sha and reader_bridge is None
                    or {key:value for key,value in saved_structural.items() if key not in _CURRENT_OPTIONS} != structural
                    or receipt['declared_root_references'] != roots):
                raise skeleton.SkeletonCacheRejected('Current declared source roots or structural options differ')
            references = receipt['complete_source_references']
            if any(Path(path).exists() for path in receipt['original_absent_read_paths']):
                raise skeleton.SkeletonCacheRejected('An optional original assembly input appeared')
            # The existing loader independently rehashes every original input
            # and registered producer against this authenticated closure.
            placement = 0
            if opt.independent_board_pattern_placement_source is not None:
                placement = receipt['placement_columns']
            values = skeleton.load_final_source_state_skeleton(receipt['source_skeleton_directory'],
                source_references=references, structural_options=saved_structural,
                restore_checkpoint=lambda state:skeleton.restore_final_skeleton_checkpoint(state, saved,
                    current_seed_parts=self.seed_parts,
                    circle_enabled=opt.additional_board_circle_source is not None,
                    placement_columns=placement,
                    adaptive_source_backtracking=opt.adaptive_source_backtracking), output_directory=output)
            args = values['args']
            args.output = Path(output).resolve()
            args.source_processes = not opt.serial_source
            args.source_workers = 1
            args.source_actor_threads = 3
            args.compact_source_review = True
            args.residual_source_workers = opt.residual_source_workers
            args.residual_native_threads = opt.residual_native_threads
            # This field is a numerical checkpoint profile token in the
            # original installation contract, not static source authority.
            values['state'].structured_programme_contract['profile_source_token'] = values['inherited']
            _restore_original_measured_location_local(values, opt)
            for name in _MISSING_LOCALS:
                values[name] = None
            values['camera_error'] = 0.
            if opt.camera_pivot_exterior_constraint_source is not None:
                from installation_mount_candidate.mount_graph import pivot_outside_linear_step_constraints
                state = values['state']
                values['shared_linear_constraints'] = lambda value, selected:pivot_outside_linear_step_constraints(state, value, selected)
        except Exception as exc:
            # Only source authentication, pickle/structure and restore-chart
            # integration are speculative. The original full assembly remains
            # the authority after a rejection of this cache.
            self.record(route='original_static_source_assembly', reason=type(exc).__name__+': '+str(exc))
            return None
        # Deliberately outside the cache fallback: current physical/source
        # domain errors remain genuine original errors, never cold retries.
        _bind_current_all_blocks(values['state'])
        self.record(route='authenticated_final_static_source_skeleton',
            current_checkpoint_sha256=_sha(opt.camera_seed),
            current_complete_q_sha256=hashlib.sha256(np.asarray(saved['q']).tobytes()).hexdigest(),
            original_postrole_assembly_q_sha256=hashlib.sha256(np.asarray(values['q']).tobytes()).hexdigest(),
            original_all_block_camera_and_world_binding_recomputed=True,
            current_truth_damping_initializers_guard_profiles_and_fresh_J_execute=True,
            cached_derivative_or_truth_limits_used=False,
            original_static_camera_chart_equivalence_proof_reused=True,
            source_adapter_receipt=str(receipt_path),
            certified_predecessor_reader_bridge=reader_bridge,
            current_execution_options=dict(accepted_steps=opt.accepted_steps,
                normal_workers=opt.normal_workers, geometry_fd_workers=opt.geometry_fd_workers,
                reuse_source_jacobian=opt.reuse_source_jacobian),
            original_assembly_read_file_SHA_count=len(references))
        return values

    def save(self, values, opt):
        from .source_assembly_gc_runtime import defer_automatic_source_gc
        with defer_automatic_source_gc('static_source_skeleton_capture', self.report):
            return self._save(values, opt)

    def _save(self, values, opt):
        try:
            roots = _declared_references(opt, self.roots)
            reads = self.last_input_reads
            if reads is None or reads.errors:
                raise skeleton.SkeletonCacheRejected('Original complete assembly read observation unavailable: '+str(None if reads is None else reads.errors))
            references = roots+_scope_references()+_runtime_implementation_references()+[dict(path=str(path), sha256=digest,
                role='actual_original_static_assembly_read') for path,digest in sorted(reads.files.items(), key=lambda item:str(item[0]))]
            structural = _structural_options(opt)
            directory = self.location(structural, roots)
            source_key = hashlib.sha256(json.dumps(references, sort_keys=True,
                separators=(',', ':')).encode('utf8')).hexdigest()
            source_directory = directory/(source_key+'_'+uuid.uuid4().hex)
            with _qualified_priors(values['state']):
                result = skeleton.save_final_source_state_skeleton(source_directory,
                    state=values['state'], static_locals=values, capold=values['capold'],
                    source_references=references, structural_options=structural,
                    assembled_q_sha256=hashlib.sha256(np.asarray(values['q']).tobytes()).hexdigest())
            placement = len(values['inherited_pattern_suffix']) if values['inherited_pattern_suffix'] is not None else 0
            receipt = dict(adapter_sha256=self.sha, structural_options=structural,
                declared_root_references=roots, complete_source_references=references,
                placement_columns=placement, skeleton_sha256=result['pickle_sha256'],
                source_skeleton_directory=str(source_directory),
                original_absent_read_paths=sorted(map(str, reads.absent)),
                original_input_closure_observed_on_actual_original_assembly=True,
                numerical_checkpoint_bytes_excluded_and_restored_current=True,
                historical_compaction_diagnostics_are_not_current_q_PASS=True)
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory/('.unpublished_'+uuid.uuid4().hex+'.json')
            try:
                temporary.write_text(json.dumps(receipt, sort_keys=True, indent=2)+'\n', encoding='utf8')
                temporary.replace(directory/'named66_adapter_receipt.json')
            finally:
                temporary.unlink(missing_ok=True)
            self.record(route='captured_original_postrole_compacted_static_source',
                serialized_bytes=result['serialized_bytes'], source_references=len(references))
        except Exception as exc:
            self.record(route='original_static_source_assembly_capture_rejected',
                        reason=type(exc).__name__+': '+str(exc))


def attach_named66_source_skeleton_reuse(tree, namespace, *, cache_directory,
        report_directory, declared_source_roots, complete_static_input_closure_confirmed):
    """Apply before startup_source_auth wraps main; no source file is changed."""
    adapter = Named66SourceSkeletonAdapter(cache_directory=cache_directory,
        report_directory=report_directory, declared_source_roots=declared_source_roots,
        complete_static_input_closure_confirmed=complete_static_input_closure_confirmed,
        current_seed_parts=namespace['current_seed_parts'])
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    starts = [i for i,node in enumerate(main.body) if isinstance(node, ast.Assign)
              and ast.unparse(node) == 'state, args, template, parent = prepare(DEFAULT_SEED, out)']
    ends = [i for i,node in enumerate(main.body) if isinstance(node, ast.Assign)
            and ast.unparse(node).startswith("active_backend_report = restore_native_backend_contract(stage='before_source_actor_construction')")]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError('Original named66 postrole/compaction assembly boundary changed')
    start, end = starts[0], ends[0]
    assembly = main.body[start:end]
    loads = list(skeleton.CAPTURE_STATIC_LOCAL_NAMES)+list(skeleton.RESTORED_DRIVER_LOCAL_NAMES)+list(_MISSING_LOCALS)
    loads += ['capold', 'geometry', 'contract', 'original_measured_locations']
    assignments = [ast.parse(name+" = _rinkspace_final_skeleton_values["+repr(name)+']').body[0] for name in loads]
    # Imports whose names are referenced later in the original driver remain
    # local to main, but installers are not executed again on a cache hit.
    imports = []
    for node in assembly:
        imports.extend(part for part in ast.walk(node) if isinstance(part, ast.ImportFrom))
    hit = imports+assignments
    branch = ast.parse('if _rinkspace_final_skeleton_values is None:\n    pass\nelse:\n    pass').body[0]
    scope = ast.parse('with _rinkspace_capture_final_skeleton_input_reads(opt):\n    pass').body[0]
    scope.body = assembly
    branch.body = [scope, ast.parse('_rinkspace_save_final_skeleton(locals(), opt)').body[0]]
    branch.orelse = hit
    main.body[start:end] = [ast.parse('_rinkspace_final_skeleton_values = _rinkspace_load_final_skeleton(opt, out)').body[0], branch]
    namespace['_rinkspace_load_final_skeleton'] = adapter.load
    namespace['_rinkspace_save_final_skeleton'] = adapter.save
    namespace['_rinkspace_capture_final_skeleton_input_reads'] = adapter.capture_input_reads
    ast.fix_missing_locations(tree)
    return tree
