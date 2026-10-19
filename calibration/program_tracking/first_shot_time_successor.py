"""Explicit source successor of an unchanged accepted native-time chart.

The original T initializer and recipe remain historical construction inputs.
The outer stage starts at an authenticated complete accepted native-control q.
Only separately reviewed source metadata may change before any pool opens.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .first_shot_source_transition import reference, read
from .first_shot_joint_initializer import _same
from .first_shot_source_snapshot import _assert_same, _frame_contract, coverage_to_record
from .first_shot_derived_venue_stage import DerivedVenueState

SCHEMA = 'rinkspace.native-time-source-successor.v1'
DELTA_SCHEMA = 'rinkspace.reviewed-native-source-successor-delta.v1'


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def authenticate_references(value):
    if isinstance(value, dict):
        if {'path', 'sha256'} <= set(value):
            if reference(value['path']) != {key:value[key] for key in ('path', 'sha256')}:
                raise ValueError('Frozen source-successor reference changed: '+value['path'])
        for item in value.values():
            authenticate_references(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            authenticate_references(item)


def read_accepted_parent(manifest):
    """Read q/ALL aux/numerical state, never previous source r or J."""
    for key in ('accepted_parent_checkpoint', 'accepted_parent_payload', 'accepted_parent_report'):
        authenticate_references(manifest[key])
    with np.load(manifest['accepted_parent_checkpoint']['path'], allow_pickle=False) as saved:
        q = saved['q'].copy()
        aux = {key[5:]:saved[key].copy() for key in saved.files if key.startswith('aux__')}
        damping = float(saved['damping']); fraction = float(saved['next_line_search_fraction'])
        identity = json.loads(str(saved['identity_json']))
    if (q.dtype != np.float64 or q.shape != (7700,) or not np.isfinite(q).all()
            or not np.isfinite(damping) or damping < 0 or not 0 < fraction <= 1
            or not aux or any(value.dtype.hasobject for value in aux.values())):
        raise ValueError('Complete finite native-time accepted parent and numerical state required')
    report = read(manifest['accepted_parent_report'])
    completed = read(manifest['completed_parent_branch'])
    if (report.get('kind') != 'accepted' or report.get('qaug_sha256') != _sha(q)
            or completed.get('q_sha256') != _sha(q)):
        raise ValueError('Explicit complete accepted parent is not the completed branch state')
    with np.load(manifest['accepted_parent_payload']['path'], allow_pickle=False) as saved:
        _same(saved['q_augmented'], q, 'accepted native-time canonical payload')
        for key, value in aux.items():
            _same(saved['aux__'+key], value, 'every accepted native-time parent aux: '+key)
        physical = saved['physical_q_augmented'].copy()
        expanded = saved['native_time_expanded_base_canonical_q'].copy()
    return dict(q=q, aux=aux, damping=damping, fraction=fraction, physical_q=physical,
                expanded_q=expanded, identity=identity, q_sha256=_sha(q),
                checkpoint=deepcopy(manifest['accepted_parent_checkpoint']))


def _layers(base):
    result = []; pending = [base]; seen = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value)); result.append(value)
        for name in ('base', 'physical_stage'):
            child = vars(value).get(name)
            if child is not None:
                pending.append(child)
    return result


def apply_reviewed_source_delta(base_owner, delta_reference, expected_counts, *, producer_references):
    """Use the existing exact edit APIs before pools, r/J or raw caches exist."""
    from .first_shot_ordinary_delta import apply_ordinary_delta
    from .first_shot_group_quarantine import quarantine_complete_groups
    from .first_shot_fit import FrozenWhitening
    programme = base_owner.programme; bundle = programme.bundle
    if any(value is not None for value in (programme.pool, bundle._source_pool, bundle._geometry_pool)):
        raise ValueError('Source-successor edits require all source/geometry pools unopened')
    if programme.worker_receipts or programme.evaluation_receipts:
        raise ValueError('Source successor cannot change metadata after a source evaluation')
    delta = read(delta_reference); authenticate_references(delta)
    if (delta.get('schema') != DELTA_SCHEMA or not delta.get('evidence_references')
            or delta.get('original_source_snapshot') != bundle.safe_source_snapshot_receipt['snapshot']):
        raise ValueError('Reviewed successor delta must bind this exact authenticated source snapshot')
    ordinary = delta.get('ordinary_delta'); groups = delta.get('quarantine_complete_groups', [])
    if ordinary is None and not groups:
        raise ValueError('A source successor requires an explicit nonempty reviewed source change')
    before = bundle.frames; original_contract = _frame_contract(before)
    finite_inventory = bundle.finite_coverage_inventory
    finite_record = coverage_to_record(finite_inventory)
    native_before = [row for frame in before for row in frame['rows'] if 'native_line_queries' in row]
    revised = before; ordinary_proof = group_proof = None
    if ordinary is not None:
        revised, ordinary_proof = apply_ordinary_delta(revised, ordinary, bundle.state.binding, policy=bundle.policy)
    if groups:
        revised, group_proof = quarantine_complete_groups(revised, groups, evidence_reference=delta_reference)
    native_after = [row for frame in revised for row in frame['rows'] if 'native_line_queries' in row]
    _assert_same(native_before, native_after, 'all continuous/native-query rows untouched by successor')
    contract = _frame_contract(revised)
    if (contract['frame_count'] != original_contract['frame_count']
            or contract['native_pts_sha256'] != original_contract['native_pts_sha256']):
        raise ValueError('Source successor changed the native exposure inventory/clock')
    bundle.frames = revised; bundle.descriptors = OrderedDict()
    for frame in revised:
        for row in frame['rows']:
            bundle.descriptors.setdefault(row['_target_key'], deepcopy(row['target']))
    bundle.keys = tuple(bundle.descriptors); bundle.target_descriptors = tuple(bundle.descriptors.values())
    bundle.initial_target_proof = bundle.state.validate_initial_targets(bundle.target_descriptors)
    bundle.whitening = [FrozenWhitening(frame) for frame in revised]
    bundle.scales = np.concatenate([w.scales for w in bundle.whitening])
    bundle.quadratic = np.zeros(len(bundle.scales), bool)
    bundle.ledgers = []
    for frame, w, digest in zip(revised, bundle.whitening, contract['frame_inventory_sha256'], strict=True):
        bundle.ledgers.append(dict(family='accepted_official_static_source', programme_index=frame['index'],
            native_pts=frame['native_pts'], time_base=bundle.state.source_clock['time_base'],
            source_points=frame['source_point_count'], formal_rows=w.count, source_inventory_sha256=digest,
            source_packet=frame['source_packet'], association_policy=deepcopy(bundle.policy),
            complete_source_snapshot=delta['original_source_snapshot'], reviewed_successor_delta=delta_reference))
    programme.scales = bundle.scales.copy(); programme.quadratic = bundle.quadratic.copy()
    programme.formal_offsets = np.r_[0, np.cumsum([w.count for w in bundle.whitening])]
    if programme.formal_offsets[-1] != len(programme.scales):
        raise ValueError('Successor formal offsets do not cover the complete new source')
    for obj, name in ((bundle, '_world_cache'), (bundle, '_finite_coverage_cache'),
                      (programme, 'transverse_world_cache'), (programme, 'guard_cache')):
        getattr(obj, name).clear()
    programme._world_delta = programme._geometry_pool_delta = None
    _assert_same(finite_record, coverage_to_record(bundle.finite_coverage_inventory), 'unchanged finite coverage metadata')
    if bundle.finite_coverage_inventory is not finite_inventory:
        raise ValueError('Unchanged native-query coverage object was unexpectedly rebuilt')
    # The retained raw-value wrapper has not captured any values yet. Verify its
    # first actual new-source hit rather than calling old evidence a new audit.
    programme.raw_retention_receipt.update(verified=False, cache_hits=0, full_original_shadow_evaluations=0,
        first_actual_hit_full_original_raw_bitwise_verification_required=True,
        new_source_transition=deepcopy(delta_reference))
    combined_scales = np.r_[programme.scales, base_owner.factor.scales]
    combined_quadratic = np.r_[programme.quadratic, base_owner.factor.quadratic]
    refs = [reference(__file__), reference(apply_ordinary_delta.__code__.co_filename),
            reference(quarantine_complete_groups.__code__.co_filename), delta_reference, *producer_references]
    authenticate_references(refs)
    for layer in [programme, *_layers(base_owner)]:
        if layer is not programme:
            layer.scales = combined_scales; layer.quadratic = combined_quadratic
            receipt = vars(layer).get('stage_receipt')
            if receipt is not None:
                receipt.update(source_successor_delta=deepcopy(delta_reference))
                for key, count in (('programme_formal_rows', len(programme.scales)),
                                   ('combined_formal_rows', len(combined_scales))):
                    if key in receipt:
                        receipt[key] = count
        for name in ('source_producers', '_references'):
            values = vars(layer).get(name)
            if values is not None:
                for ref in refs:
                    if ref not in values:
                        values.append(deepcopy(ref))
    actual = dict(source_points=contract['source_points'], programme_formal_rows=len(programme.scales),
        wide_formal_rows=len(base_owner.factor.scales), complete_formal_rows=len(combined_scales))
    if actual != expected_counts:
        raise ValueError('Explicit source-successor counts differ from actual complete metadata: '+str(actual))
    proof = dict(schema='rinkspace.pre-pool-reviewed-source-successor-application.v1', passed=True,
        delta=deepcopy(delta_reference), old_contract=original_contract, new_contract=contract,
        ordinary_edit_proof=ordinary_proof, complete_group_quarantine_proof=group_proof,
        all_native_queries_and_finite_coverage_byte_exact=True, finite_coverage_original_object_retained=True,
        descriptors_whitening_ledgers_scales_offsets_rebuilt=True, source_only_caches_cleared=True,
        geometry_or_camera_parameters_modified=False, source_or_geometry_pool_opened=False,
        source_residual_J_or_old_numerical_cache_reused=False, actual_counts=actual)
    return proof


def export_successor_inventory(programme, inherited_media_reference, output, *, transition_reference):
    """Reuse authenticated media records; rebuild only actual source/W maps."""
    from .first_shot_resume_cache import source_inventory
    output = Path(output)
    old = read(inherited_media_reference)
    if old.get('render_worker_must_verify_image_and_mask_bytes_before_use') is not True:
        raise ValueError('Authenticated inherited media must retain render-time image verification')
    if len(old['frames']) != len(programme.bundle.frames):
        raise ValueError('Successor media does not cover every original native exposure')
    records = []
    for frame, ledger, record in zip(programme.bundle.frames, programme.bundle.ledgers, old['frames'], strict=True):
        if (record['index'] != frame['index'] or record['native_pts'] != frame['native_pts']
                or record['actual_pts'] != frame['native_pts']
                or record['time_base'] != programme.state.source_clock['time_base']):
            raise ValueError('Successor changed inherited media/native PTS correspondence')
        _assert_same(record['source_packet'], frame['source_packet'], 'inherited native media source packet')
        records.append(dict(record, source_ledger=deepcopy(ledger)))
    inventory, frame_map = source_inventory(programme.bundle)
    np.savez_compressed(output/'immutable_programme_source_inventory.npz', **inventory)
    np.savez_compressed(output/'source_sigma_and_formal_offsets.npz',
        sigma_px=np.concatenate([np.asarray(row['sigma'],float) for frame in programme.bundle.frames for row in frame['rows']]),
        formal_rows=np.asarray([w.count for w in programme.bundle.whitening],np.int64),
        formal_offsets=programme.formal_offsets, scales=programme.scales, quadratic=programme.quadratic)
    result = dict(old, frames=records, source_frame_map=frame_map,
        source_inventory=reference(output/'immutable_programme_source_inventory.npz'),
        formal_inventory=reference(output/'source_sigma_and_formal_offsets.npz'),
        inherited_authenticated_media=inherited_media_reference, reviewed_source_transition=transition_reference,
        unchanged_source_packet_index_and_PTS_every_exposure_verified=True,
        image_mask_and_original_media_metadata_retained=True,
        legacy_PNG_bytes_rehashed_and_decoded_BGR_authenticated=False,
        inherited_legacy_PNG_authentication_not_replayed=True,
        all_other_image_and_mask_bytes_rehashed_by_this_metadata_export=False,
        producer=reference(__file__))
    path = output/'media_manifest.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return inventory, reference(path)


class NativeTimeSourceSuccessor:
    """Identity chart over a frozen T stage with an explicit accepted start."""
    def __init__(self, base, parent, *, manifest_reference, source_transition_reference, entry_reference):
        self.base = base
        self.parameter_count, self.programme_count = base.parameter_count, base.programme_count
        self.programme, self.physical_stage, self.factor = base.programme, base.physical_stage, base.factor
        self.start = parent['q'].copy(); self.fixed = base.fixed.copy(); self.shared = base.shared.copy()
        self.scales, self.quadratic = base.scales, base.quadratic
        base._canonical_q(self.start)
        if parent['identity'].get('native_time_stage') != base.native_time_receipt:
            raise ValueError('Accepted parent native-time basis/chart identity differs from construction')
        existing = base.checkpoint_arrays()
        if set(existing) != set(parent['aux']):
            raise ValueError('Accepted native-time parent auxiliary inventory differs from the reconstructed chart')
        for key, value in existing.items():
            _same(value, parent['aux'][key], 'complete reconstructed native-time auxiliary: '+key)
        expanded, _ = base.expanded_q_and_chain(self.start)
        physical, _ = base.effective_q_and_chain(self.start)
        _same(expanded, parent['expanded_q'], 'accepted parent expanded T q')
        _same(physical, parent['physical_q'], 'accepted parent derived physical q')
        self._parent = parent; self._initial_complete_J_reuse = False
        self._references = deepcopy(base._references)
        for ref in (reference(__file__), entry_reference, manifest_reference, source_transition_reference):
            authenticate_references(ref)
            if ref not in self._references:
                self._references.append(deepcopy(ref))
        self.source_producers = deepcopy(base.source_producers)
        for ref in self._references:
            if ref not in self.source_producers:
                self.source_producers.append(deepcopy(ref))
        self.recovery_references = deepcopy(base.recovery_references)
        self.native_time_receipt = deepcopy(base.native_time_receipt)
        self.source_successor_receipt = dict(schema=SCHEMA, manifest=deepcopy(manifest_reference),
            source_transition=deepcopy(source_transition_reference), accepted_parent=parent['checkpoint'],
            accepted_parent_q_sha256=parent['q_sha256'], complete_parent_auxiliary_count=len(parent['aux']),
            parent_damping=parent['damping'], parent_next_line_search_fraction=parent['fraction'],
            original_T_recipe_and_initializer_kept_as_history=True, identical_T_basis_fixed_history_and_parameter_chart=True,
            all_accepted_q_and_aux_preserved_byte_exact=True, old_source_r_J_or_cost_reused=False,
            predictions_of_new_source_not_claimed_equivalent=True, fresh_full_source_J_required=True)
        self.state = DerivedVenueState(self)

    def __getattr__(self, name):
        value = getattr(self.base, name)
        if callable(value):
            raise AttributeError('Source successor requires an explicit adapter for '+name)
        return value

    def _check(self):
        authenticate_references(self._references)

    def _canonical_q(self, q):
        self._check(); return self.base._canonical_q(q)

    def expanded_q_and_chain(self, q):
        return self.base.expanded_q_and_chain(self._canonical_q(q))

    def effective_q_and_chain(self, q):
        return self.base.effective_q_and_chain(self._canonical_q(q))

    def chart_identity(self):
        return self.base.chart_identity()

    def state_identity(self, q):
        return dict(self.base.state_identity(self._canonical_q(q)), source_successor=deepcopy(self.source_successor_receipt))

    def split_state(self, q):
        return self.base.split_state(self._canonical_q(q))

    def guard(self, q):
        return self.base.guard(self._canonical_q(q))

    def source_blocks(self, q, jacobian):
        yield from self.base.source_blocks(self._canonical_q(q), bool(jacobian))

    def evaluate(self, q, jacobian=False):
        return self.base.evaluate(self._canonical_q(q), bool(jacobian))

    def world(self, q, jacobian=False):
        return self.base.world(self._canonical_q(q), bool(jacobian))

    def raw_programme(self, q, **kwargs):
        return self.base.raw_programme(self._canonical_q(q), **kwargs)

    def raw_static(self, q):
        return self.base.raw_static(self._canonical_q(q))

    def raw(self, q, **kwargs):
        return self.base.raw(self._canonical_q(q), **kwargs)

    def cost(self, q):
        return self.base.cost(self._canonical_q(q))

    def linear_constraints(self, q, selected):
        return self.base.linear_constraints(self._canonical_q(q), selected)

    def recover(self, parent, proposed):
        return self.base.recover(self._canonical_q(parent), self._canonical_q(proposed))

    def checkpoint_arrays(self):
        self._check()
        result = {key:value.copy() for key,value in self._parent['aux'].items()}
        result.update({'source_successor_parent__'+key:value.copy() for key,value in self._parent['aux'].items()})
        result.update(source_successor_initial_q=self.start.copy(),
            source_successor_parent_damping=np.asarray(self._parent['damping']),
            source_successor_parent_next_line_search_fraction=np.asarray(self._parent['fraction']),
            source_successor_identity_json=np.asarray(json.dumps(self.source_successor_receipt,ensure_ascii=False,sort_keys=True)))
        return result

    def complete_state_payload(self, q):
        q = self._canonical_q(q); result = self.base.complete_state_payload(q)
        result.update({'aux__'+key:value for key,value in self.checkpoint_arrays().items()})
        result['canonical_physical_state_identity_json'] = np.asarray(json.dumps(self.state_identity(q),sort_keys=True))
        return result

    def checkpoint_identity(self):
        self._check(); result = deepcopy(self.base.checkpoint_identity())
        result.update(source_successor=deepcopy(self.source_successor_receipt),
                      source_producers=deepcopy(self.source_producers),
                      initial_exact_complete_current_state_full_J_reuse=False)
        return result

    def solver_kwargs(self):
        result = dict(self.base.solver_kwargs())
        result.update(initial_damping=self._parent['damping'],
            initial_line_search_fraction=self._parent['fraction'], source_block_factory=self.source_blocks,
            checkpoint_arrays=self.checkpoint_arrays(), checkpoint_identity=self.checkpoint_identity(),
            shared_linear_constraints=self.linear_constraints, trial_feasibility_recovery=self.recover,
            reuse_source_jacobian=False)
        return result

    def close(self):
        return self.base.close()
