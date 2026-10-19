"""Restore one full public programme owner from authenticated source metadata.

Every source equation, whitening and parallel sampler is the public producer.
No historical constructor, residual, Jacobian, or numerical cache is replayed.
The restored legacy camera chart is used only as an explicit parameter parent;
the caller installs the declared new camera model before any evaluation.
"""
from collections import OrderedDict
from copy import deepcopy
from types import MethodType
import numpy as np

from .first_shot_source_transition import reference, read
from .first_shot_source_snapshot import restore_source_bundle, _assert_same


def restore_programme_owner(snapshot_reference, parent, *, model_reference,
        association_reference, geometry_workers, source_workers, entry_reference):
    import native_transverse_bundle as runtime
    bundle = restore_source_bundle(snapshot_reference, geometry_workers=geometry_workers, source_workers=1)
    policy = read(association_reference)
    _assert_same(bundle.policy, policy, 'restored_current_association')
    model = runtime.load_model(model_reference)
    state = model.TransverseNativePrincipalState(bundle.state)
    if state.count != 7674 or np.asarray(parent['q']).shape != (7687,):
        raise ValueError('Explicit complete accepted parent chart required before camera transition')
    for key, value in dict(native_pts=bundle.state.pts, parent_q=bundle.state.parent_q,
            parent_shared_slots=bundle.state.parent_shared_slots,
            original_8a_initial_q=bundle.state.initial_q).items():
        _assert_same(np.asarray(value), parent['aux'][key], 'parent_chart.' + key)
    owner = runtime.CompleteTransverseBundle.__new__(runtime.CompleteTransverseBundle)
    owner.model_reference = deepcopy(model_reference); owner.model = model
    owner.setup = owner.setup_receipt = None
    owner.bundle = bundle; owner.association_reference = deepcopy(association_reference); owner.state = state
    owner.workers = int(source_workers); owner.pool = None
    owner.worker_receipts = []; owner.evaluation_receipts = []
    owner.shared = state.global_shared_indices.copy()
    owner.quadratic = bundle.quadratic.copy(); owner.scales = bundle.scales.copy()
    owner.formal_offsets = np.r_[0, np.cumsum([w.count for w in bundle.whitening])]
    if owner.formal_offsets[-1] != len(owner.scales):
        raise ValueError('Restored complete source formal offsets changed')
    owner.source_producers = deepcopy(bundle.safe_source_snapshot_receipt['source_producers'])
    for ref in (model_reference, reference(runtime.__file__), reference(__file__), entry_reference):
        if reference(ref['path']) != ref:
            raise ValueError('Restored programme execution producer changed')
        if ref not in owner.source_producers:
            owner.source_producers.append(deepcopy(ref))
    owner.guard_cache = OrderedDict(); owner.guard_evaluations = 0
    owner.start = np.asarray(parent['q'][:state.count], float).copy()
    state._q(owner.start)
    owner.fixed = np.asarray(parent['aux']['native_PP_gauge_fixed_columns'], int).copy()
    owner.anchors = np.asarray(parent['aux']['native_PP_anchor_frames_1based'], int).copy()
    owner.seed_receipt = dict(schema='rinkspace.restored-source-explicit-parameter-parent.v1',
        accepted_complete_parent=deepcopy(parent['reference']), source_snapshot=deepcopy(snapshot_reference),
        initial_damping=float(parent['values']['damping']),
        initial_line_search_fraction=float(parent['values']['next_line_search_fraction']),
        complete_parent_aux_count=len(parent['aux']), parameter_parent_only=True,
        old_camera_not_evaluated_and_not_claimed_new_source_accepted=True,
        source_residual_J_and_world_value_caches_loaded=False)
    owner.seed_aux = {'aux__' + key: value.copy() for key, value in parent['aux'].items()}
    owner.bundle.state = model.make_unshifted_world_state(state)
    owner.transverse_world_cache = OrderedDict(); owner._world_delta = owner._geometry_pool_delta = None
    owner.metric = None; owner.initial_J_manifest = None; owner._initial_complete_J_reuse = False
    owner.recovery_references = []  # Replaced by the explicit new physical transition before evaluation.
    owner.last_guard_report = None
    stored=bundle.safe_source_snapshot_receipt
    # Complete historical proofs (including numerical index maps) already live
    # in the authenticated safe snapshot. The JSON/checkpoint identity binds
    # that exact document, rather than duplicating ndarray objects into JSON.
    source_receipt=dict(snapshot=deepcopy(snapshot_reference),producer=deepcopy(stored['producer']),
        source_producers=deepcopy(stored['source_producers']),
        parent_authentication=deepcopy(stored['parent_authentication']),
        source_transition=deepcopy(stored['source_transition']),
        original_revision_proof_names=list(stored['original_revision_proofs']),
        complete_original_revision_proofs_stored_in_authenticated_snapshot=True,
        complete_public_bundle_initialized=True,old_residual_J_or_value_caches_loaded=False,
        all_execution_pools_initially_closed=True)
    __import__('json').dumps(source_receipt,allow_nan=False)
    receipt = dict(schema='rinkspace.complete-public-programme-source-restore.v1',
        source_snapshot=deepcopy(snapshot_reference), source=source_receipt,
        accepted_complete_parent=deepcopy(parent['reference']), model_parameter_parent=model_reference,
        producers=[reference(__file__), entry_reference], complete_public_bundle_fields_initialized=True,
        original_public_world_source_workers_and_persistent_pool_protocol_retained=True,
        previous_r_J_or_numerical_caches_reused=False,
        old_source_authentication_receipt_referenced_not_reexecuted=True)
    def identity(adapter):
        return dict(schema='rinkspace.restored-source-parameter-parent.v1',
            binding=adapter.bundle.state.binding_reference, association=adapter.association_reference,
            seed=deepcopy(adapter.seed_receipt), source_producers=deepcopy(adapter.source_producers),
            parameter_count=state.count, source_restore=deepcopy(receipt),
            previous_accepted_complete_identity=deepcopy(parent['identity']),
            initial_exact_complete_current_state_full_J_reuse=False)
    owner.checkpoint_identity = MethodType(identity, owner)
    def source_blocks(adapter, q, jacobian, **kwargs):
        beginning=len(adapter.evaluation_receipts)
        yield from runtime.CompleteTransverseBundle.source_blocks(adapter,q,jacobian,**kwargs)
        for item in adapter.evaluation_receipts[beginning:]:
            item['full_parameter_columns']=len(q)
            item['source_parameter_chart']='explicit restored-source current physical camera model'
    owner.source_blocks=MethodType(source_blocks,owner)
    owner.source_restore_receipt = receipt
    return owner, receipt
