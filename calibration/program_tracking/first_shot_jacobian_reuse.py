"""Use the public modified-Newton source model without changing its gates.

Only the public solver decides refreshes, trial acceptance and convergence.
A retained Jacobian is approximate away from its recorded anchor. Current
source values, profiles, Huber weights, normals and true trial costs are fresh.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=float).tobytes()).hexdigest()


def _verify(ref):
    if reference(ref['path']) != ref:
        raise ValueError('Complete current source cache reference changed: ' + ref['path'])


def _same(actual, expected, label):
    if not np.array_equal(actual, expected):
        raise ValueError('Complete current source cache differs: ' + label)


def current_linearization_receipt(owner, q, accumulated, cache_directory):
    """Reference the existing owning writer; never re-export J or choose old q."""
    q = np.asarray(q, float)
    retained = accumulated.native_replay_cache
    if retained is None:
        raise ValueError('A complete fresh current-source replay cache is required for handoff')
    _same(retained.source_state, q, 'public retained source state')
    candidates = [Path(cache_directory) / 'latest.json']
    initial = owner.checkpoint_identity().get('authenticated_complete_source_resume', {})
    if initial.get('qaug_sha256') == _sha(q):
        candidates.append(Path(initial['manifest']['path']))
    selected = None
    for path in candidates:
        if path.exists():
            manifest = json.loads(path.read_text(encoding='utf8'))
            if manifest['qaug_sha256'] == _sha(q):
                selected = path, manifest
                break
    if selected is None:
        raise ValueError('Owning source writer has no exact-current-q complete J manifest')
    path, manifest = selected
    if manifest['J_shape'] != [len(owner.scales), len(q)]:
        raise ValueError('Owning source writer has incomplete current J dimensions')
    if not manifest.get('source_J_before_Huber_weights'):
        raise ValueError('Handoff requires the original unweighted source Jacobian')
    state_ref = manifest['complete_current_state_r_scales_quadratic']
    _verify(state_ref)
    with np.load(state_ref['path'], allow_pickle=False) as saved:
        _same(saved['q_augmented'], q, 'saved full state')
        _same(saved['scales'], owner.scales, 'source scales')
        _same(saved['quadratic'], owner.quadratic, 'source mask')
        _same(saved['r'], np.concatenate([block.r for block in retained.blocks]), 'current residuals')
        for key, value in owner.complete_state_payload(q).items():
            _same(saved[key], value, 'state/aux ' + key)
    if sum(block.J.nnz for block in retained.blocks) != manifest['J_nonzeros']:
        raise ValueError('Handoff complete source J nonzero inventory changed')
    for block in manifest['blocks']:
        _verify(block['J'])
    world_path = Path(cache_directory).parent / 'world_J_cache/latest.json'
    world = None
    if world_path.exists():
        info = json.loads(world_path.read_text(encoding='utf8'))
        if info.get('qaug_sha256') == _sha(q):
            _verify(info['cache'])
            world = reference(world_path)
    if world is None and initial.get('qaug_sha256') == _sha(q):
        world = initial.get('world_cache')
    return dict(schema='rinkspace.fresh-current-complete-source-J-handoff.v1',
        qaug_sha256=_sha(q), source_manifest=reference(path), world_manifest=world,
        complete_current_r_J_scales_mask_and_aux_verified=True,
        public_solver_forces_same_q_fresh_J_before_handoff=True,
        original_owning_writer_used=True, duplicate_source_J_export=False,
        world_manifest_same_full_q=world is not None,
        source_rows=len(owner.scales), parameter_count=len(q))


def install_source_jacobian_reuse(owner, *, cache_directory, entry_reference=None, reuse_source_jacobian=True):
    """Install after authenticated replay and other source execution wrappers."""
    if hasattr(owner, 'source_jacobian_reuse_policy'):
        raise ValueError('Source Jacobian reuse is already installed')
    if not hasattr(owner, '_unmeasured_mount_domain'):
        raise ValueError('This continuation requires the released-domain recover independent of stale H')
    import z190_joint_source_solver as solver
    from structured_programme_source_candidate import lagged_jacobian_source as lagged
    original_kwargs = owner.solver_kwargs
    original_identity = owner.checkpoint_identity
    refs = [reference(__file__), reference(solver.__file__), reference(lagged.__file__)]
    if entry_reference is not None:
        refs.append(deepcopy(entry_reference))
    for ref in refs:
        if ref not in owner.recovery_references:
            owner.recovery_references.append(ref)
    policy = dict(schema='rinkspace.public-modified-newton-source-execution.v1',
        producers=refs, public_solver_reuse_source_jacobian=bool(reuse_source_jacobian),
        retained_J_away_from_anchor_is_approximate=bool(reuse_source_jacobian),
        fresh_current_source_J_every_new_state=not reuse_source_jacobian,
        current_r_profiles_Huber_weights_H_and_gradient_recomputed=True,
        all_original_physical_guards_and_true_nonlinear_source_cost_required=True,
        source_inventory_UV_PTS_sigma_whitening_masks_and_parameter_domain_unchanged=True,
        existing_public_refresh_policy_and_fresh_J_convergence_gate_retained=True,
        existing_default_prediction_refresh_ratio_unchanged=True,
        finite_endpoint_extensions_are_guard_recovery_not_open_endpoint_events=True,
        released_mount_recovery_does_not_use_previous_source_metric=True,
        exact_iteration_anchor_and_current_q_in_public_source_jacobian_reuse_report=True,
        explicit_current_cache_handoff_forces_fresh_same_q_J=True,
        no_added_iteration_time_or_reuse_age_cap=True)

    def check():
        for ref in refs:
            if reference(ref['path']) != ref:
                raise ValueError('Frozen source Jacobian execution producer changed')

    def identity(adapter):
        result = original_identity()
        result['source_J_execution_policy'] = deepcopy(policy)
        return result

    def save_current(q, accumulated):
        check()
        return current_linearization_receipt(owner, q, accumulated, cache_directory)

    def solver_kwargs(adapter):
        check()
        result = original_kwargs()
        if not result.get('cache_native_source_replay') or not callable(result.get('source_block_factory')):
            raise ValueError('Public modified Newton requires existing full-source retained blocks')
        if result.get('save_current_linearization_cache') is not None:
            raise ValueError('Current complete-source writer callback is already installed')
        result['reuse_source_jacobian'] = bool(reuse_source_jacobian)
        result['save_current_linearization_cache'] = save_current
        result['checkpoint_identity']['source_J_execution_policy'] = deepcopy(policy)
        return result

    owner.checkpoint_identity = MethodType(identity, owner)
    owner.solver_kwargs = MethodType(solver_kwargs, owner)
    owner.source_jacobian_reuse_policy = policy
    return deepcopy(policy)


def annotate_current_report(report, owner):
    """Update before the owning writer hashes its result, never rewrite artifacts."""
    if not isinstance(report, dict) or 'fresh_current_source_J_every_new_state' not in report:
        return report
    report['fresh_current_source_J_every_new_state'] = owner.source_jacobian_reuse_policy['fresh_current_source_J_every_new_state']
    if getattr(owner, '_initial_complete_J_reuse', True) is False:
        report['initial_exact_complete_current_state_full_J_reuse'] = False
        report['initial_source_J_reuse'] = False
    report['source_J_execution_policy'] = deepcopy(owner.source_jacobian_reuse_policy)
    actual = report.get('event', {}).get('source_jacobian_reuse')
    report['direction_source_J'] = deepcopy(actual) if actual is not None else dict(
        derivative_model='initial_state_no_direction_yet',
        initial_complete_current_J_authenticated=getattr(owner, '_initial_complete_J_reuse', True))
    if hasattr(owner, 'raw_retention_receipt'):
        report['same_state_original_cost_raw_retention'] = deepcopy(owner.raw_retention_receipt)
    return report
