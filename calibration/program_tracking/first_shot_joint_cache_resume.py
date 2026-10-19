"""Authenticate combined T/static handoffs without applying their chain twice."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(ref):
    if reference(ref['path']) != ref:
        raise ValueError('Frozen handoff reference changed: ' + ref['path'])
    return json.loads(Path(ref['path']).read_text(encoding='utf8'))


def equal(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    if (a.dtype != b.dtype or a.shape != b.shape or a.dtype.hasobject
            or a.tobytes() != b.tobytes()):
        raise ValueError('Exact handoff mismatch: ' + label)


def qsha(q):
    return hashlib.sha256(np.ascontiguousarray(q, dtype=np.float64).tobytes()).hexdigest()


def verify_refs(value):
    if isinstance(value, dict):
        if {'path', 'sha256'} <= set(value):
            ref = {k:value[k] for k in ('path', 'sha256')}
            if reference(ref['path']) != ref:
                raise ValueError('Frozen handoff dependency changed: ' + ref['path'])
        for item in value.values():
            verify_refs(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            verify_refs(item)


def load_parent(config):
    """Every final numerical array comes from the normal handoff save."""
    result = read(config['completed_parent_branch'])
    stop = result.get('stop_reason', {})
    if result.get('status') != 'stopped_by_control' or not stop.get('current_linearization_cache'):
        raise ValueError('A normally completed explicit fresh-current-J handoff is required')
    handoff = stop['current_linearization_cache']
    if (handoff.get('schema') != 'rinkspace.fresh-current-complete-source-J-handoff.v1'
            or not handoff.get('complete_current_r_J_scales_mask_and_aux_verified')
            or not handoff.get('public_solver_forces_same_q_fresh_J_before_handoff')):
        raise ValueError('Original owning writer did not certify the fresh complete current cache')
    if handoff['source_manifest'] != config['parent_source_cache']:
        raise ValueError('The selected cache is not the actual final handoff cache')
    verify_refs(config['parent_checkpoint'])
    with np.load(config['parent_checkpoint']['path'], allow_pickle=False) as saved:
        parent = dict(q=saved['q'].copy(),
            aux={k[5:]:saved[k].copy() for k in saved.files if k.startswith('aux__')},
            profiles={k[9:]:saved[k].copy() for k in saved.files if k.startswith('profile__')},
            damping=float(saved['damping']), fraction=float(saved['next_line_search_fraction']),
            scales=saved['scales'].copy(), quadratic=saved['quadratic'].copy(),
            identity=json.loads(str(saved['identity_json'])), checkpoint=deepcopy(config['parent_checkpoint']))
    if (qsha(parent['q']) != result['q_sha256'] or qsha(parent['q']) != handoff['qaug_sha256']
            or parent['q'].dtype != np.float64 or parent['q'].shape != (7700,)
            or not np.isfinite(parent['q']).all() or not 0 < parent['fraction'] <= 1
            or not np.isfinite(parent['damping']) or parent['damping'] <= 0):
        raise ValueError('Final checkpoint/cache/branch q or numerical state differs')
    if parent['profiles']:
        raise ValueError('This candidate requires the observed no-profile-callback 8d execution')
    payload = result['final_complete_state']
    if payload != config['parent_complete_state']:
        raise ValueError('Final complete payload differs from the normal completed branch')
    verify_refs(payload)
    with np.load(payload['path'], allow_pickle=False) as saved:
        equal(saved['q_augmented'], parent['q'], 'final complete canonical q')
        for key, value in parent['aux'].items():
            equal(saved['aux__'+key], value, 'final complete aux ' + key)
        parent['physical_q'] = saved['physical_q_augmented'].copy()
        parent['expanded_q'] = saved['native_time_expanded_base_canonical_q'].copy()
    parent['q_sha256'] = qsha(parent['q'])
    return parent


class ExactJointCache:
    """Only original same-q unweighted CSR; fresh live r and all row metadata."""
    def __init__(self, owner, config, parent):
        self.owner, self.parent = owner, parent
        self.manifest_ref = deepcopy(config['parent_source_cache'])
        self.manifest = read(self.manifest_ref)
        m = self.manifest
        if (m.get('schema') != 'rinkspace.complete-canonical-effective-joint-source-J.v1'
                or m['qaug_sha256'] != parent['q_sha256']
                or m['J_shape'] != [len(owner.scales), len(parent['q'])]
                or not m.get('source_J_before_Huber_weights') or not m.get('full_static_rows_always_present')):
            raise ValueError('Complete canonical T plus static same-q cache required')
        verify_refs(m['source_producers'])
        verify_refs(m['source_identity'])
        verify_refs(m['complete_current_state_r_scales_quadratic'])
        with np.load(m['complete_current_state_r_scales_quadratic']['path'], allow_pickle=False) as saved:
            equal(saved['q_augmented'], parent['q'], 'cache canonical q')
            equal(saved['physical_q_augmented'], parent['physical_q'], 'cache physical q')
            equal(saved['native_time_expanded_base_canonical_q'], parent['expanded_q'], 'cache expanded T q')
            equal(saved['scales'], owner.scales, 'cache row scales')
            equal(saved['quadratic'], owner.quadratic, 'cache row masks')
            for key, value in parent['aux'].items():
                equal(saved['aux__'+key], value, 'cache all parent aux ' + key)
            self.r = saved['r'].copy()
        equal(owner.scales, parent['scales'], 'parent all source scales')
        equal(owner.quadratic, parent['quadratic'], 'parent all source masks')
        verify_refs(m['block_loss_weights'])
        with np.load(m['block_loss_weights']['path'], allow_pickle=False) as saved:
            self.loss_arrays = {k:saved[k].copy() for k in saved.files}
        self.matrices = None
        self.shadow_blocks = None
        self.shadow_receipt = None
        self.calls = 0

    def same_q(self, q):
        return (np.asarray(q).dtype == self.parent['q'].dtype and
                np.asarray(q).shape == self.parent['q'].shape and
                np.asarray(q).tobytes() == self.parent['q'].tobytes())

    def verify_live_shadow(self, blocks, *, update_profiles=None):
        """No derivative execution: all current source values and W/UV/PTS ledgers."""
        if update_profiles is not None:
            raise ValueError('A new/stateless or stateful profile callback cannot borrow this cache')
        from calibration.solver_runtime.source_loss import source_loss_checkpoint_arrays
        current = list(blocks)
        if len(current) != len(self.manifest['blocks']):
            raise ValueError('Current full-source block count changed')
        cursor = 0
        checked = []
        for index, (block, saved) in enumerate(zip(current, self.manifest['blocks'], strict=True)):
            n = len(block.r)
            if (saved['block'] != index or saved['row_start'] != cursor
                    or saved['row_count'] != n or saved['J_shape'] != [n, len(self.parent['q'])]):
                raise ValueError('Full-source row partition/order changed')
            equal(block.r, self.r[cursor:cursor+n], 'live original residual block ' + str(index))
            equal(block.scales, self.owner.scales[cursor:cursor+n], 'live scales ' + str(index))
            equal(block.quadratic, self.owner.quadratic[cursor:cursor+n], 'live masks ' + str(index))
            if json.dumps(block.row_ledger, sort_keys=True) != json.dumps(saved['row_ledger'], sort_keys=True):
                raise ValueError('Original complete UV/PTS/sigma/W row ledger changed')
            loss = source_loss_checkpoint_arrays(block.loss_weights, n)
            expected = {k.split('__block_')[0]:v for k,v in self.loss_arrays.items()
                        if k.endswith('__block_'+str(index))}
            if set(loss) != set(expected):
                raise ValueError('Current source integration/working normalization inventory changed')
            for key in loss:
                equal(loss[key], expected[key], 'live loss weights ' + str(index) + '/' + key)
            checked.append(dict(block=index, rows=n, current_r_W_masks_scales_ledger_loss_bytes_exact=True))
            cursor += n
        if cursor != len(self.owner.scales):
            raise ValueError('Full-source shadow missed rows')
        self.shadow_blocks = current
        self.shadow_receipt = dict(passed=True, source_manifest=self.manifest_ref,
            qaug_sha256=self.parent['q_sha256'], rows=cursor, blocks=checked,
            complete_current_original_source_values_evaluated=True,
            no_J_or_world_FD_recomputed=True, no_profile_callback=True)
        return deepcopy(self.shadow_receipt)

    def blocks(self, q):
        if not self.same_q(q) or self.shadow_receipt is None:
            raise ValueError('Saved complete source J requires exact q and a complete live shadow')
        from z190_streaming_source_normal import SourceRowBlock
        if self.matrices is None:
            self.matrices = []
            for block in self.manifest['blocks']:
                verify_refs(block['J'])
                matrix = load_npz(block['J']['path']).tocsr()
                if (list(matrix.shape) != block['J_shape'] or matrix.nnz != block['J_nonzeros']
                        or not np.isfinite(matrix.data).all()):
                    raise ValueError('Saved complete source CSR shape/nnz/value contract changed')
                self.matrices.append(matrix)
            if sum(j.nnz for j in self.matrices) != self.manifest['J_nonzeros']:
                raise ValueError('Saved complete J nonzero count changed')
        self.calls += 1
        for live, matrix in zip(self.shadow_blocks, self.matrices, strict=True):
            yield SourceRowBlock(live.r, matrix, live.quadratic, live.scales, live.row_ledger, live.loss_weights)


class ExactTimeContinuation:
    """Identity wrapper: original source and original T chart, final saved start."""
    def __init__(self, base, parent, *, manifest_reference, entry_reference, config):
        self.base, self._parent = base, parent
        self.start = parent['q'].copy()
        self._refs = [reference(__file__), deepcopy(entry_reference), deepcopy(manifest_reference)]
        for ref in self._refs:
            verify_refs(ref)
        existing = base.checkpoint_arrays()
        if set(existing) != set(parent['aux']):
            raise ValueError('Reconstructed original 8d chart auxiliary inventory changed')
        for key in existing:
            equal(existing[key], parent['aux'][key], 'reconstructed complete aux ' + key)
        if parent['identity'].get('native_time_stage') != base.native_time_receipt:
            raise ValueError('Original T basis/history/recipe changed')
        expanded, _ = base.expanded_q_and_chain(self.start)
        physical, _ = base.effective_q_and_chain(self.start)
        equal(expanded, parent['expanded_q'], 'original T expansion at final accepted q')
        equal(physical, parent['physical_q'], 'original physical expansion at final accepted q')
        self.scales, self.quadratic = base.scales, base.quadratic
        self.fixed, self.shared = base.fixed.copy(), base.shared.copy()
        self.source_producers = deepcopy(base.source_producers)
        self.recovery_references = deepcopy(base.recovery_references)
        for ref in self._refs:
            if ref not in self.source_producers:
                self.source_producers.append(deepcopy(ref))
            if ref not in self.recovery_references:
                self.recovery_references.append(deepcopy(ref))
        self._initial_complete_J_reuse = True
        self.continuation_receipt = dict(
            schema='rinkspace.exact-current-T-source-cache-continuation.v1',
            manifest=deepcopy(manifest_reference), producers=deepcopy(self._refs),
            parent_checkpoint=parent['checkpoint'], parent_complete_state=config['parent_complete_state'],
            completed_parent_branch=config['completed_parent_branch'],
            source_cache=config['parent_source_cache'],
            parent_q_sha256=parent['q_sha256'], all_parent_auxiliary_count=len(parent['aux']),
            initial_damping=parent['damping'], initial_line_search_fraction=parent['fraction'],
            original_T_recipe_and_initializer_are_history_only=True,
            current_q_is_accepted_parent_not_original_T_initializer=True,
            all_parent_aux_byte_exact_no_recursive_duplicate_archive=True,
            source_parameters_domain_and_native_clock_unchanged=True,
            source_J_is_already_pulled_to_T_canonical_columns=True,
            second_chain_multiplication=False, exact_same_q_cache_only=True,
            new_q_uses_original_public_source_factory=True)
        self.exact_cache = ExactJointCache(self, config, parent)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _check(self):
        for ref in self._refs:
            verify_refs(ref)

    def _canonical_q(self, q):
        self._check()
        return self.base._canonical_q(q)

    def source_blocks(self, q, jacobian):
        q = self._canonical_q(q)
        if jacobian and self.exact_cache.same_q(q):
            yield from self.exact_cache.blocks(q)
        else:
            yield from self.base.source_blocks(q, bool(jacobian))

    def checkpoint_arrays(self):
        self._check()
        current = self.base.checkpoint_arrays()
        if set(current) != set(self._parent['aux']):
            raise ValueError('Original auxiliary inventory changed during continuation')
        for key in current:
            equal(current[key], self._parent['aux'][key], 'current all inherited aux ' + key)
        return {key:value.copy() for key,value in self._parent['aux'].items()}

    def checkpoint_identity(self):
        self._check()
        value = deepcopy(self.base.checkpoint_identity())
        value.update(source_producers=deepcopy(self.source_producers),
            exact_current_cache_continuation=deepcopy(self.continuation_receipt),
            initial_exact_complete_current_state_full_J_reuse=True)
        return value

    def complete_state_payload(self, q):
        q = self._canonical_q(q)
        result = self.base.complete_state_payload(q)
        for key,value in self.checkpoint_arrays().items():
            result['aux__'+key] = value
        # Explicit lineage metadata is not a numerical reparameterization.
        result['exact_current_cache_continuation_json'] = np.asarray(
            json.dumps(self.continuation_receipt, sort_keys=True))
        return result

    def solver_kwargs(self):
        self._check()
        result = dict(self.base.solver_kwargs())
        if result.get('update_profiles') is not None:
            raise ValueError('Only the authenticated no-profile-callback parent can use this adapter')
        result.update(initial_damping=self._parent['damping'],
            initial_line_search_fraction=self._parent['fraction'],
            checkpoint_arrays=self.checkpoint_arrays(), checkpoint_identity=self.checkpoint_identity(),
            source_block_factory=self.source_blocks, reuse_source_jacobian=False)
        return result
