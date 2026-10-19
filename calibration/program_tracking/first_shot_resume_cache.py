"""Authenticated same-state source replay; every changed state uses its producer.

This module owns no observations, weights, physical bounds or acceptance policy.
It may only reuse a complete saved linearization after checking its source chart.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import types

import numpy as np
from scipy.sparse import load_npz, vstack


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _equal(actual, expected, message):
    if not np.array_equal(actual, expected):
        raise ValueError(message)


def certify_pool_lifetime_change(old_reference, directory):
    """Admit exactly deletion of the old per-J two-line pool shutdown."""
    directory = Path(directory)
    receipt = json.loads((directory / 'edit_receipt.json').read_text(encoding='utf8'))
    before = directory / 'first_shot_fit_before.py'
    target = Path(old_reference['path'])
    old_bytes = before.read_bytes()
    if hashlib.sha256(old_bytes).hexdigest() != old_reference['sha256']:
        raise ValueError('Pool transition snapshot is not the authenticated old producer')
    if str(target.resolve()) != str(Path(receipt['target']).resolve()):
        raise ValueError('Pool transition targets another producer')
    old_text = old_bytes.decode('utf8').replace('\r\n', '\n')
    current_bytes = target.read_bytes()
    current_text = current_bytes.decode('utf8').replace('\r\n', '\n')
    removed = ('            if self._geometry_pool is not None:\n'
               '                self._geometry_pool.shutdown(wait=True); self._geometry_pool=None\n')
    if old_text.count(removed) != 1:
        raise ValueError('Unique original per-J shutdown statement required')
    expected = old_text.replace(removed, '', 1)
    if current_text != expected:
        raise ValueError('Producer differs beyond the two-line pool lifetime repair')
    old_tree = ast.parse(old_text)
    new_tree = ast.parse(current_text)
    world = next(method for node in old_tree.body
                 if isinstance(node, ast.ClassDef) and node.name == 'FirstShotSourceBundle'
                 for method in node.body
                 if isinstance(method, ast.FunctionDef) and method.name == 'world')
    jacobian = next(node for node in world.body
                    if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                    and node.test.id == 'jacobian')
    removed_node = jacobian.body.pop()
    if not isinstance(removed_node, ast.If) or 'shutdown' not in ast.unparse(removed_node):
        raise ValueError('Unexpected geometry shutdown AST location')
    if ast.dump(old_tree, include_attributes=False) != ast.dump(new_tree, include_attributes=False):
        raise ValueError('Pool repair changed the producer AST beyond the shutdown statement')
    current = reference(target)
    if receipt.get('before_sha256') != old_reference['sha256'] or receipt.get('after_sha256') != current['sha256']:
        raise ValueError('Pool transition receipt hash mismatch')
    return dict(old=old_reference, current=current, old_snapshot=reference(before),
                edit_receipt=reference(directory / 'edit_receipt.json'),
                only_two_line_pool_shutdown_removed=True, AST_only_shutdown_removed=True,
                source_values_J_rows_and_physical_guards_unchanged=True)


def source_inventory(bundle):
    from calibration.program_tracking.first_shot_alignment_monitor import _target_label
    ids, labels, pts, frames, uv, frame_map = [], [], [], [], [], []
    offset = 0
    for frame in bundle.frames:
        count = 0
        for row in frame['rows']:
            n = len(row['uv'])
            if n != len(row['source_point_ids']):
                raise ValueError('Original source ID/UV count mismatch')
            ids.extend(map(str, row['source_point_ids']))
            labels.extend([_target_label(row['target'])] * n)
            pts.extend([frame['native_pts']] * n)
            frames.extend([frame['index']] * n)
            uv.extend(row['uv'])
            count += n
        if count != frame['source_point_count']:
            raise ValueError('Original frame source count mismatch')
        frame_map.append(dict(frame=frame['index'], native_pts=frame['native_pts'],
                              raw_start=offset, raw_count=count))
        offset += count
    return dict(source_point_ids=np.asarray(ids), source_target_labels=np.asarray(labels),
                native_pts=np.asarray(pts, np.int64), frame_indices=np.asarray(frames, np.int32),
                source_uv=np.asarray(uv, float)), frame_map


class CompleteSourceReplay:
    """Authenticate a complete q/r/J/raw snapshot and install exact-state replay."""

    def __init__(self, adapter, *, manifest, prior_report, checkpoint, pool_transition,
                 world_manifest=None):
        self.adapter = adapter
        self.manifest_reference = reference(manifest)
        self.report_reference = reference(prior_report)
        self.checkpoint_reference = reference(checkpoint)
        self.manifest = json.loads(Path(manifest).read_text(encoding='utf8'))
        self.report = json.loads(Path(prior_report).read_text(encoding='utf8'))
        self.pool_transition = Path(pool_transition)
        self.transitions = {}
        self.verified = {}
        self.verify_references(self.manifest)
        self.verify_references(self.report)
        with np.load(checkpoint, allow_pickle=False) as saved:
            self.q = saved['q'].copy()
            self.damping = float(saved['damping'])
            self.fraction = float(saved['next_line_search_fraction'])
            self.verify_references(json.loads(str(saved['identity_json'])))
            _equal(self.q, adapter.start, 'Checkpoint q differs from the complete adapter state')
            _equal(saved['scales'], adapter.scales, 'Checkpoint source scales changed')
            _equal(saved['quadratic'], adapter.quadratic, 'Checkpoint source mask changed')
            for name, value in adapter.checkpoint_arrays().items():
                _equal(saved['aux__' + name], value, 'Checkpoint chart/aux changed: ' + name)
        if self.manifest['qaug_sha256'] != array_sha(self.q) or self.report['qaug_sha256'] != array_sha(self.q):
            raise ValueError('Complete cache/report/checkpoint state hashes differ')
        if adapter.seed_receipt['initial_damping'] != self.damping or adapter.seed_receipt['initial_line_search_fraction'] != self.fraction:
            raise ValueError('Continuation changed source damping or line-search fraction')
        if self.manifest['binding'] != adapter.bundle.state.binding_reference or self.report['binding'] != self.manifest['binding']:
            raise ValueError('Complete source binding changed')
        self.scales = adapter.scales.copy()
        self.quadratic = adapter.quadratic.copy()
        current_inventory, frames = source_inventory(adapter.bundle)
        with np.load(self.manifest['source_inventory']['path'], allow_pickle=False) as saved:
            if set(saved.files) != set(current_inventory):
                raise ValueError('Original source inventory fields changed')
            for name, value in current_inventory.items():
                _equal(saved[name], value, 'Original UV/ID/PTS/label/order changed: ' + name)
        frame_map = json.loads(Path(self.manifest['source_frame_map']['path']).read_text(encoding='utf8'))
        if frame_map['frames'] != frames or frame_map['binding'] != self.manifest['binding']:
            raise ValueError('Original source frame/PTS/raw ordering changed')
        self.inventory = current_inventory
        self.frames = frames
        with np.load(self.manifest['complete_current_state_r_scales_quadratic']['path'], allow_pickle=False) as saved:
            _equal(saved['q_augmented'], self.q, 'Saved source linearization q changed')
            for name, value in adapter.complete_state_payload(self.q).items():
                _equal(saved[name], value, 'Saved source chart/state changed: ' + name)
            _equal(saved['scales'], adapter.scales, 'Saved source scales changed')
            _equal(saved['quadratic'], adapter.quadratic, 'Saved source mask changed')
            self.r = saved['r'].copy()
        with np.load(self.report['snapshot']['path'], allow_pickle=False) as saved:
            for name, value in adapter.complete_state_payload(self.q).items():
                _equal(saved[name], value, 'Saved raw snapshot chart/state changed: ' + name)
            _equal(saved['source_point_ids'], current_inventory['source_point_ids'], 'Raw source IDs changed')
            _equal(saved['native_pts_per_point'], current_inventory['native_pts'], 'Raw source PTS changed')
            self.raw = saved['point_errors_px'].copy()
        if self.r.shape != adapter.scales.shape or self.raw.shape != current_inventory['native_pts'].shape:
            raise ValueError('Saved source/raw vector shape mismatch')
        if not np.isfinite(self.r).all() or not np.isfinite(self.raw).all():
            raise ValueError('Nonfinite saved source/raw value')
        self._validate_old_blocks()
        self.parts = [np.asarray(c, int) for c in np.array_split(np.arange(len(adapter.bundle.frames)), adapter.workers) if len(c)]
        self.matrices = None
        self.replay_calls = {'source_values': 0, 'source_J': 0, 'raw': 0}
        self.world_reference = None
        if world_manifest is not None:
            self._install_world(world_manifest)
        self.receipt = dict(schema='rinkspace.authenticated-complete-current-source-resume.v1',
            manifest=self.manifest_reference, prior_report=self.report_reference, checkpoint=self.checkpoint_reference,
            qaug_sha256=array_sha(self.q), parameter_count=len(self.q), source_rows=len(self.r),
            source_points=len(self.raw), native_exposures=len(adapter.bundle.frames),
            producer_transitions=list(self.transitions.values()), exact_other_references=len(self.verified),
            source_inventory_UV_ID_sigma_whitening_PTS_order_and_chart_exact=True,
            every_frame_original_ledger_exact=True, scales_and_quadratic_exact=True,
            initial_damping=self.damping, initial_line_search_fraction=self.fraction,
            old_source_blocks=len(self.manifest['blocks']), replay_source_blocks=len(self.parts),
            source_workers=adapter.workers, geometry_workers=adapter.bundle.geometry_workers,
            block_repartition_only_original_row_order_retained=True,
            H_and_gradient_rebuilt_from_original_r_J_Huber_rows=True, old_H_or_gradient_reused=False,
            world_cache=self.world_reference, source_cost=self.report['source_objective_cost'],
            only_exact_complete_state_replayed=True, all_changed_states_fresh=True,
            same_physical_guard_source_weights_and_solver_policy=True)

    def verify_reference(self, ref):
        key = (str(Path(ref['path']).resolve()), ref['sha256'])
        if key in self.verified:
            return
        actual = reference(ref['path'])
        if actual != ref:
            target = Path(ref['path'])
            from calibration.program_tracking import first_shot_fit
            if target.resolve() != Path(first_shot_fit.__file__).resolve():
                raise ValueError('Authenticated resume producer/source reference changed: ' + str(target))
            self.transitions[key] = certify_pool_lifetime_change(ref, self.pool_transition)
        self.verified[key] = actual

    def verify_references(self, value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                self.verify_reference(value)
            for item in value.values():
                self.verify_references(item)
        elif isinstance(value, list):
            for item in value:
                self.verify_references(item)

    def _validate_old_blocks(self):
        cursor, frame_cursor = 0, 0
        for index, block in enumerate(self.manifest['blocks']):
            ledger = block['row_ledger']
            if block['block'] != index or block['row_start'] != cursor:
                raise ValueError('Saved source blocks changed order')
            old_frames = ledger['frames']
            stop = frame_cursor + len(old_frames)
            if old_frames != self.adapter.bundle.ledgers[frame_cursor:stop]:
                raise ValueError('Original source sigma/W/UV/ID/frame ledger changed')
            expected = dict(family='complete-native-PP-two-axis-transverse',
                binding=self.adapter.bundle.state.binding_reference,
                two_axis_model=self.adapter.model_reference, frames=old_frames)
            if ledger != expected:
                raise ValueError('Original source block ledger changed')
            if int(self.adapter.formal_offsets[frame_cursor]) != cursor or int(self.adapter.formal_offsets[stop]) != cursor + block['row_count']:
                raise ValueError('Original formal row boundaries changed')
            if block['J_shape'] != [block['row_count'], len(self.q)]:
                raise ValueError('Saved complete source J dimensions changed')
            cursor += block['row_count']
            frame_cursor = stop
        if cursor != len(self.r) or frame_cursor != len(self.adapter.bundle.frames):
            raise ValueError('Incomplete source cache inventory')

    def require_same_state(self, q):
        _equal(q, self.q, 'Complete cached linearization cannot be used at a changed state')
        _equal(self.adapter.scales, self.scales, 'Active source scales changed')
        _equal(self.adapter.quadratic, self.quadratic, 'Active source mask changed')
        self._validate_old_blocks()

    def _install_world(self, path):
        self.world_reference = reference(path)
        manifest = json.loads(Path(path).read_text(encoding='utf8'))
        self.verify_references(manifest)
        if manifest['qaug_sha256'] != array_sha(self.q) or manifest['binding'] != self.adapter.bundle.state.binding_reference:
            raise ValueError('Saved world derivative cache is not the exact current state')
        with np.load(manifest['cache']['path'], allow_pickle=False) as saved:
            _equal(saved['q_augmented'], self.q, 'World cache complete state changed')
            _equal(saved['world_parameter_indices'], self.adapter.state.world_parameter_indices, 'World parameter columns changed')
            geometry = self.q[self.adapter.state.world_parameter_indices]
            _equal(saved['geometry_values'], geometry, 'World cache geometry state changed')
            keys = saved['world_keys'].tolist()
            if keys != list(self.adapter.bundle.descriptors):
                raise ValueError('World target order changed')
            descriptors = [json.dumps(self.adapter.bundle.descriptors[k], sort_keys=True) for k in keys]
            _equal(saved['target_descriptors_json'], descriptors, 'World target descriptors changed')
            world = {k: saved[f'world_{i:04d}'].copy() for i, k in enumerate(keys)}
            derivative = {k: saved[f'D_{i:04d}'].copy() for i, k in enumerate(keys)}
        self.adapter.transverse_world_cache[np.asarray(geometry, float).tobytes()] = world, derivative
        self.adapter._world_delta = self.adapter.state.deviations(self.q).copy()

    def _load_matrices(self):
        if self.matrices is not None:
            return
        pieces = [[] for _ in self.parts]
        bounds = [(int(self.adapter.formal_offsets[c[0]]), int(self.adapter.formal_offsets[c[-1] + 1])) for c in self.parts]
        for old in self.manifest['blocks']:
            matrix = load_npz(old['J']['path']).tocsr()
            if list(matrix.shape) != old['J_shape'] or matrix.nnz != old['J_nonzeros'] or not np.isfinite(matrix.data).all():
                raise ValueError('Saved original CSR dimensions/nonzeros/values changed')
            start, stop = old['row_start'], old['row_start'] + old['row_count']
            for parts, (left, right) in zip(pieces, bounds, strict=True):
                a, b = max(left, start), min(right, stop)
                if a < b:
                    parts.append(matrix[a-start:b-start])
        self.matrices = [vstack(parts, format='csr') for parts in pieces]
        if sum(matrix.nnz for matrix in self.matrices) != self.manifest['J_nonzeros']:
            raise ValueError('Repartition changed complete source J nonzero inventory')

    def blocks(self, q, jacobian):
        from z190_streaming_source_normal import SourceRowBlock
        self.require_same_state(q)
        if jacobian:
            self._load_matrices()
        for index, frames in enumerate(self.parts):
            left, right = int(self.adapter.formal_offsets[frames[0]]), int(self.adapter.formal_offsets[frames[-1]+1])
            yield SourceRowBlock(self.r[left:right], self.matrices[index] if jacobian else None,
                self.adapter.quadratic[left:right], self.adapter.scales[left:right],
                dict(family='complete-native-PP-two-axis-transverse', binding=self.adapter.bundle.state.binding_reference,
                     two_axis_model=self.adapter.model_reference, frames=[self.adapter.bundle.ledgers[i] for i in frames]))

    def install(self):
        adapter = self.adapter
        original_source = adapter.source_blocks
        original_raw = adapter.raw
        original_identity = adapter.checkpoint_identity
        active_producers = deepcopy(adapter.source_producers)
        producer = reference(__file__)

        def source_blocks(owner, q, jacobian, **kwargs):
            if kwargs.get('indices') is not None or not np.array_equal(q, self.q):
                yield from original_source(q, jacobian, **kwargs)
                return
            for ref in active_producers + [producer]:
                if reference(ref['path']) != ref:
                    raise ValueError('Active resume source producer changed')
            if kwargs.get('physical_guard', True):
                owner.guard(q)
            blocks = []
            for block in self.blocks(q, jacobian):
                blocks.append(block)
                yield block
            self.replay_calls['source_J' if jacobian else 'source_values'] += 1
            if jacobian:
                from z190_streaming_source_normal import accumulate_source_normal
                from mount_source_metric_direction import ConditionalMountDirection
                accumulation = accumulate_source_normal(lambda: iter(blocks), owner.state.count,
                    expected_row_count=len(owner.scales), frozen_profile_token='same-parent-head-and-common-five-geometry',
                    fixed_parameter_indices=owner.fixed)
                damping = owner._damping_for(q)
                owner.metric = ConditionalMountDirection(owner, q, accumulation.normal, accumulation.gradient, damping,
                    columns=[121, 122, 123, int(owner.state.original.zoom_parameter_indices[3]), owner.state.axis_index],
                    mount_support=owner.model.head.envelope_support, mount_values=owner.model.head._mount_values)
                owner.metric_receipts.append(dict(qaug_sha256=array_sha(q), source_rows=len(owner.scales),
                    source_cost=accumulation.cost, damping=damping, exact_complete_saved_source_J_replayed=True,
                    H_and_gradient_rebuilt=True, all_original_source_W_Huber_and_gauge_unchanged=True))

        def raw(owner, q, **kwargs):
            if kwargs.get('indices') is not None or not np.array_equal(q, self.q):
                return original_raw(q, **kwargs)
            self.require_same_state(q)
            if kwargs.get('physical_guard', True):
                owner.guard(q)
            self.replay_calls['raw'] += 1
            return self.raw.copy()

        def identity(owner):
            value = original_identity()
            value['authenticated_complete_source_resume'] = dict(self.receipt, producer=producer)
            return value

        adapter.source_blocks = types.MethodType(source_blocks, adapter)
        adapter.raw = types.MethodType(raw, adapter)
        adapter.checkpoint_identity = types.MethodType(identity, adapter)
        return self
