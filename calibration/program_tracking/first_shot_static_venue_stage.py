"""A new full state/source stage with persistent static venue observations.

The programme owner keeps its original parameter chart, source rows and worker
pools. The outer state appends the independently conditioned wide camera and
appends its original material rows on every source evaluation. No old r/J or
frozen-world continuation is claimed for this expanded objective.
"""
from copy import deepcopy
from collections import OrderedDict
import hashlib
import math
import sys
from types import MethodType

import numpy as np
from scipy.sparse import hstack, issparse, csr_matrix, vstack

from .first_shot_source_transition import reference, _digest


def apply_complete_group_quarantine(programme, specifications, *, manifest_reference):
    """Install an authenticated whole-estimator source revision before pools.

    Authentication/admission belongs to the caller's explicit manifest. The
    original source frames are retained as evidence by the quarantine utility;
    every active whitening operator is rebuilt on the exact retained mapping.
    """
    from . import first_shot_group_quarantine as quarantine
    from .first_shot_fit import FrozenWhitening
    if (programme.pool is not None or programme.bundle._source_pool is not None
            or programme.bundle._geometry_pool is not None):
        raise ValueError('Complete-group source transition must precede every worker')
    if reference(manifest_reference['path']) != manifest_reference:
        raise ValueError('Explicit complete-group quarantine manifest changed')
    if getattr(programme, 'initial_J_manifest', None) is not None:
        raise ValueError('Source quarantine cannot borrow an initial source Jacobian')
    old_frames = programme.bundle.frames
    frames, proof = quarantine.quarantine_complete_groups(old_frames, specifications,
        evidence_reference=manifest_reference)
    coverage = lambda current: [r for f in current for r in f['rows'] if r.get('_continuous_world_line')]
    if _digest(coverage(old_frames)) != _digest(coverage(frames)):
        raise ValueError('Whole-group revision requires a separate native-line coverage rebuild')
    indices = proof['retained_formal_indices']
    whitening = [FrozenWhitening(frame) for frame in frames]
    scales = np.concatenate([w.scales for w in whitening])
    if not np.array_equal(scales, programme.scales[indices]):
        raise ValueError('Complete group transition changed retained formal scales')
    mask = programme.quadratic[indices].copy()
    descriptors = OrderedDict()
    ledgers = []
    for frame, w, previous in zip(frames, whitening, programme.bundle.ledgers, strict=True):
        digest = hashlib.sha256()
        import json
        for row in frame['rows']:
            descriptors.setdefault(row['_target_key'], row['target'])
            digest.update(json.dumps(row['source_point_ids'], separators=(',', ':')).encode())
            digest.update(row['uv'].tobytes()); digest.update(row['sigma'].tobytes())
        for group in frame['whitening_groups']:
            digest.update(group['operator'].matrix.tobytes())
        ledger = dict(previous, source_points=frame['source_point_count'], formal_rows=w.count,
            source_inventory_sha256=digest.hexdigest())
        if frame.get('complete_group_quarantine_revision') == manifest_reference:
            ledger['complete_group_quarantine_revision'] = deepcopy(manifest_reference)
        ledgers.append(ledger)
    bundle = programme.bundle
    bundle.frames, bundle.whitening, bundle.ledgers = frames, whitening, ledgers
    bundle.scales, bundle.quadratic = scales, mask
    bundle.descriptors, bundle.keys = descriptors, tuple(descriptors)
    bundle.target_descriptors = tuple(descriptors.values())
    programme.scales, programme.quadratic = scales.copy(), mask.copy()
    programme.formal_offsets = np.r_[0, np.cumsum([w.count for w in whitening])]
    if programme.formal_offsets[-1] != len(scales):
        raise ValueError('Revised complete-source frame offsets changed')
    for owner, name in ((bundle, '_world_cache'), (bundle, '_finite_coverage_cache'),
                        (programme, 'transverse_world_cache'), (programme, 'guard_cache')):
        getattr(owner, name).clear()
    programme.metric = None
    programme._initial_complete_J_reuse = False
    receipt = {key: deepcopy(value) for key, value in proof.items() if key != 'retained_formal_indices'}
    receipt.update(manifest=deepcopy(manifest_reference), producer=reference(quarantine.__file__),
        entry=reference(__file__), original_native_line_coverage_inventory_exact=True,
        original_source_datasets_and_uncertain_evidence_preserved=True,
        new_active_source_r_J_and_all_caches_required=True, old_source_r_J_reused=False)
    programme.complete_group_quarantine_receipt = receipt
    bundle.complete_group_quarantine_receipt = receipt
    previous_identity = programme.checkpoint_identity
    def identity(owner):
        value = previous_identity()
        value['explicit_complete_group_source_quarantine'] = deepcopy(receipt)
        value['initial_exact_complete_current_state_full_J_reuse'] = False
        return value
    programme.checkpoint_identity = MethodType(identity, programme)
    for ref in (reference(__file__), reference(quarantine.__file__), manifest_reference):
        if ref not in programme.source_producers:
            programme.source_producers.append(deepcopy(ref))
    return receipt


class ExpandedState:
    def __init__(self, owner):
        self._owner = owner
        self.count = owner.parameter_count
        self.initial_q = owner.start.copy()
        self.world_parameter_indices = owner.programme.state.world_parameter_indices.copy()
        self.world_parameter_count = len(self.world_parameter_indices)
        self.global_shared_indices = owner.shared.copy()

    def _q(self, q):
        value = np.asarray(q, dtype=np.float64)
        if value.shape != (self.count,) or not np.isfinite(value).all():
            raise ValueError('Complete finite programme plus venue-camera state required')
        return value

    def __getattr__(self, name):
        value = getattr(self._owner.programme.state, name)
        if not callable(value):
            return value
        def call(*args, **kwargs):
            if args and isinstance(args[0], np.ndarray) and args[0].shape == (self.count,):
                args = (args[0][:self._owner.programme_count], *args[1:])
            return value(*args, **kwargs)
        return call


class StaticVenueStage:
    """Expand one authenticated programme owner without changing its source.

    Callers authenticate the saved accepted prefix before construction. Any
    common-scale gauge must be added only after a separate exact similarity
    audit; this wrapper never invents a fixed geometry value or numeric box.
    """
    def __init__(self, programme, factor, *, parent_reference, entry_reference=None,
                 source_references=()):
        if hasattr(programme, 'fixed_geometry_receipt'):
            raise ValueError('Construct the new stage before installing any previous fixed-world wrapper')
        self.programme = programme
        self.factor = factor
        self.programme_count = len(programme.start)
        if (factor.programme_count != self.programme_count
                or not np.array_equal(factor.geometry_columns, programme.state.world_parameter_indices)):
            raise ValueError('Static factor must use the exact complete original world chart')
        self.parameter_count = factor.parameter_count
        self.start = factor.initial_state(programme.start)
        if (self.start.shape != (self.parameter_count,)
                or not np.array_equal(self.start[:self.programme_count].view(np.uint64),
                                      programme.start.view(np.uint64))):
            raise ValueError('Appended static-camera initializer changed the accepted programme prefix')
        self.programme_shared = programme.shared.copy()
        self.shared = np.r_[self.programme_shared, factor.camera_columns].astype(int)
        self.fixed = programme.fixed.copy()
        if np.isin(self.fixed, programme.state.world_parameter_indices).any():
            raise ValueError('Previous fixed geometry must be explicitly released for the new wide stage')
        self.scales = np.r_[programme.scales, factor.scales]
        self.quadratic = np.r_[programme.quadratic, factor.quadratic]
        self.source_producers = programme.source_producers
        self.recovery_references = programme.recovery_references
        self.parent_reference = deepcopy(parent_reference)
        self.stage_references = [reference(__file__),
            reference(sys.modules[type(factor).__module__].__file__)]
        if entry_reference is not None:
            self.stage_references.append(deepcopy(entry_reference))
        self.stage_references.extend(deepcopy(tuple(source_references)))
        for ref in self.stage_references:
            if reference(ref['path']) != ref:
                raise ValueError('New venue stage producer changed')
            if ref not in self.source_producers:
                self.source_producers.append(ref)
        self.state = ExpandedState(self)
        self._initial_complete_J_reuse = False
        self.stage_receipt = dict(schema='rinkspace.programme-plus-persistent-static-venue-stage.v1',
            accepted_programme_parent=deepcopy(parent_reference),
            programme_parameter_count=self.programme_count, parameter_count=self.parameter_count,
            programme_q_sha256=hashlib.sha256(programme.start.tobytes()).hexdigest(),
            programme_prefix_all_parameters_preserved=True,
            previous_fixed_world_coordinates_released=True,
            fixed_parameter_indices=self.fixed.tolist(),
            world_parameter_indices=programme.state.world_parameter_indices.tolist(),
            static_camera_columns=factor.camera_columns.tolist(),
            static_venue_factor=deepcopy(factor.identity),
            producers_and_admission_references=deepcopy(self.stage_references),
            programme_formal_rows=len(programme.scales),
            static_venue_formal_rows=len(factor.scales),
            combined_formal_rows=len(self.scales),
            static_venue_source_identity_sha256=factor.packet.identity_sha256,
            static_venue_ledger_sha256=_digest(factor.packet.ledger),
            old_new_source_objectives_are_different=True,
            initial_source_r_J_world_cache_reuse=False,
            static_venue_rows_recomputed_in_every_source_evaluation=True,
            independent_static_clock_not_inserted_in_programme_native_PTS=True,
            pure_common_scale_gauge_selected=False,
            geometry_fixed_or_numeric_bounds_invented=False)

    def __getattr__(self, name):
        return getattr(self.programme, name)

    def split_state(self, q):
        value = self.state._q(q)
        return value[:self.programme_count], value[self.factor.camera_columns]

    def guard(self, q):
        prefix, _ = self.split_state(q)
        self.programme.guard(prefix)
        if hasattr(self.factor, 'camera_guard'):
            result = self.factor.camera_guard(q)
            if result is not None and not bool(result):
                raise ValueError('Static venue camera source domain rejected')
        else:
            self.factor.block(q, False)
        return True

    def world(self, q, jacobian=False):
        prefix, _ = self.split_state(q)
        return self.programme.world(prefix, jacobian)

    def source_blocks(self, q, jacobian):
        self.state._q(q)
        cursor, wide_blocks = 0, 0
        programme_rows = len(self.programme.scales)
        for block in self.factor.append_source_blocks(self.programme.source_blocks, q, bool(jacobian)):
            stop = cursor + len(block.r)
            if (stop > len(self.scales)
                    or not np.array_equal(block.scales, self.scales[cursor:stop])
                    or not np.array_equal(block.quadratic, self.quadratic[cursor:stop])):
                raise ValueError('Persistent combined source scales/mask/count changed')
            if jacobian and block.J.shape != (len(block.r), self.parameter_count):
                raise ValueError('Persistent combined source Jacobian columns changed')
            if stop > programme_rows:
                if (cursor != programme_rows or len(block.r) != len(self.factor.scales)
                        or _digest(block.row_ledger) != self.stage_receipt['static_venue_ledger_sha256']):
                    raise ValueError('Complete admitted static venue rows must be appended exactly once')
                from calibration.solver_runtime.source_loss import loss_weight_arrays
                actual_weights = loss_weight_arrays(block.loss_weights, len(block.r))
                original_weights = loss_weight_arrays(self.factor.packet.loss_weights, len(block.r))
                if not all(np.array_equal(a, b) for a, b in zip(actual_weights, original_weights, strict=True)):
                    raise ValueError('Original static venue Huber integration weights changed')
                wide_blocks += 1
            yield block
            cursor = stop
        if cursor != len(self.scales) or wide_blocks != 1:
            raise ValueError('Persistent static venue source block is absent or duplicated')

    def evaluate(self, q, jacobian=False):
        blocks = list(self.source_blocks(q, jacobian))
        residual = np.concatenate([block.r for block in blocks])
        return (residual, vstack([block.J for block in blocks], format='csr')) if jacobian else residual

    def cost(self, q):
        from z190_joint_source_solver import source_cost
        return math.fsum(source_cost(block.r, block.quadratic, block.loss_weights)
                         for block in self.source_blocks(q, False))

    def raw_programme(self, q, **kwargs):
        prefix, _ = self.split_state(q)
        return self.programme.raw(prefix, **kwargs)

    def raw_static(self, q):
        block = self.factor.block(q, False)
        pixel_vectors = np.asarray(block.r) * self.factor.scales
        return np.linalg.norm(pixel_vectors.reshape(-1, 2), axis=1)

    def raw(self, q, **kwargs):
        """Programme point distances followed by independent static distances.

        The two clocks remain separate in their ledgers; this concatenation is
        diagnostic inventory order and never inserts a wide frame in native PTS.
        """
        return np.r_[self.raw_programme(q, **kwargs), self.raw_static(q)]

    def linear_constraints(self, q, selected):
        prefix, _ = self.split_state(q)
        if not np.array_equal(selected, self.shared):
            raise ValueError('Exact new-stage shared chart required')
        original = self.programme.linear_constraints(prefix, self.programme_shared.copy())
        matrix = original['matrix']
        if matrix.shape != (len(original['lower']), len(self.programme_shared)):
            raise ValueError('Original programme shared-domain dimensions changed')
        zeros = np.zeros((matrix.shape[0], len(self.factor.camera_columns)))
        expanded = (hstack((matrix, csr_matrix(zeros)), format='csr')
                    if issparse(matrix) else np.c_[matrix, zeros])
        return dict(original, matrix=expanded,
            report=dict(original.get('report', {}), independent_static_camera_zero_domain_columns=
                self.factor.camera_columns.tolist(), original_rows_and_current_lower_preserved=True))

    def recover(self, parent, proposed):
        old, _ = self.split_state(parent)
        trial, camera = self.split_state(proposed)
        result = self.programme.recover(old, trial)
        value = np.r_[result['q'], camera]
        return dict(q=value, report=dict(result.get('report', {}),
            static_venue_camera_proposal_preserved=True,
            static_camera_numeric_clipping_or_new_bounds=False,
            complete_expanded_guard_and_true_combined_source_cost_required=True))

    def checkpoint_arrays(self):
        return dict(self.programme.checkpoint_arrays(),
            programme_parameter_count=np.array([self.programme_count], np.int64),
            static_venue_camera_columns=self.factor.camera_columns.copy(),
            static_venue_initial_camera_state=self.start[self.factor.camera_columns].copy(),
            static_venue_world_columns=self.factor.geometry_columns.copy())

    def complete_state_payload(self, q):
        prefix, camera = self.split_state(q)
        result = dict(self.programme.complete_state_payload(prefix))
        result['programme_q_augmented'] = prefix.copy()
        result['q_augmented'] = np.asarray(q).copy()
        result['static_venue_camera_state'] = camera.copy()
        result['static_venue_camera25'] = self.factor.camera_policy.decode(camera)
        result.update({'aux__' + key: value for key, value in self.checkpoint_arrays().items()})
        return result

    def checkpoint_identity(self):
        result = deepcopy(self.programme.checkpoint_identity())
        result.update(schema='rinkspace.complete-programme-and-independent-venue-camera.v1',
            parameter_count=self.parameter_count, programme_parameter_count=self.programme_count,
            shared_dense_count=len(self.shared),
            persistent_static_venue_stage=deepcopy(self.stage_receipt),
            initial_exact_complete_current_state_full_J_reuse=False,
            source_basis='original native programme plus persistent independent wide material controls',
            previous_accepted_checkpoint_used_as_parameter_initializer_only=True,
            previous_fixed_geometry_reused_as_constraint=False)
        return result

    def solver_kwargs(self):
        result = dict(self.programme.solver_kwargs())
        result.update(source_block_factory=self.source_blocks,
            fixed_parameter_indices=self.fixed.copy(),
            global_schur_count=len(self.shared), global_schur_indices=self.shared.copy(),
            shared_linear_constraints=self.linear_constraints, trial_feasibility_recovery=self.recover,
            checkpoint_arrays=self.checkpoint_arrays(), checkpoint_identity=self.checkpoint_identity())
        return result

    def close(self):
        return self.programme.close()
