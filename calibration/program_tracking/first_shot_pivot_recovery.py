"""Opt-in exact outside-boundary retraction after the original trial recovery.

Use the existing nearest-course helper with its coordinate-FD reserve disabled.
No positive clearance, source/prior row or per-frame camera shift is added.
The original recovery, complete guard and true source-cost acceptance remain.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import threading
from types import MethodType, SimpleNamespace

import numpy as np

_ERROR = 'Fixed physical tripod pivot is inside the current actual board course'


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _sha(value):
    value = np.asarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def install_pivot_boundary_recovery(owner, *, entry_reference=None, qa_references=()):
    """Install after existing mount/finite recovery; never replace their math."""
    from installation_mount_candidate import mount_graph
    from .first_shot_physical_state import INSTALLATION
    if hasattr(owner, 'pivot_boundary_recovery_receipt'):
        raise ValueError('Exact pivot boundary recovery is already installed')
    state = owner.state
    original = state.original
    pivot = np.arange(original.installation_slice.start, original.installation_slice.start + 3)
    if state.count != 7674 or original.count != 5172 or not np.array_equal(pivot, [118, 119, 120]):
        raise ValueError('Explicit original 7674/5172 first-shot pivot chart required')
    if not hasattr(owner, 'pivot_outside_step_receipt'):
        raise ValueError('Install the original pivot outside shared-QP bridge first')
    scale = float(original.scene_unit_working_m)
    contract = deepcopy(original.pivot_contract)
    relation = deepcopy(contract['by_installation'][INSTALLATION])
    if (relation['relation'] != 'pivot_outside_or_on_current_actual_board_course'
            or scale != contract['scene_unit_working_m'] or scale != float(mount_graph.SCENE_UNIT_M)):
        raise ValueError('Original outside-or-on fact and inherited working scale required')
    relation['pivot_parameter_indices'] = pivot.tolist()
    contract['world_parameter_indices'] = list(range(100))
    contract['by_installation'] = {INSTALLATION: relation}
    provider = original.provider.current
    # The helper's final validate_mounts sees no invented facade mount chart.
    # Its domain is real; all real mounts are checked by owner.guard afterwards.
    facade = SimpleNamespace(geometry=SimpleNamespace(current=provider),
        current_user_pivot_outside_constraint_contract=contract, installation_optics={})
    previous_recover = owner.recover
    previous_identity = owner.checkpoint_identity
    references = [reference(__file__), reference(mount_graph.__file__),
                  {key: contract['facts'][key] for key in ('path', 'sha256')}]
    if entry_reference is not None:
        references.append(deepcopy(entry_reference))
    references.extend(deepcopy(list(qa_references)))
    for ref in references:
        if reference(ref['path'])['sha256'] != ref['sha256']:
            raise ValueError('Frozen original pivot recovery producer/fact changed')
        if ref not in owner.recovery_references:
            owner.recovery_references.append(deepcopy(ref))
    receipt = dict(schema='rinkspace.original-pivot-outside-boundary-trial-recovery.v1',
        producers=references, helper='installation_mount_candidate.mount_graph.reinitialize_invalid_pivots_outside',
        original_recovery_executes_first=True, only_actual_guard_failed_candidate_may_be_retracted=True,
        require_coordinate_fd_margin=False, added_physical_clearance_buffer_prior_or_source_rows=False,
        only_pivot_XY_retracted_columns=pivot[:2].tolist(), per_frame_translation_added=False,
        full_original_recovery_and_guard_after_retraction_required=True,
        original_source_cost_and_solver_acceptance_still_required=True,
        valid_candidates_pass_unchanged=True, repaired_candidates=0, valid_passthroughs=0)
    lock = threading.Lock()

    def check():
        if original.provider.current is not provider:
            raise ValueError('Actual pivot course producer identity changed')
        for ref in references:
            if reference(ref['path'])['sha256'] != ref['sha256']:
                raise ValueError('Frozen original pivot boundary recovery producer/fact changed')

    def evaluation(q):
        q = np.asarray(q, np.float64)
        if q.shape != (7674,) or not np.isfinite(q).all():
            raise ValueError('Finite complete original source state required for pivot retraction')
        old, _ = state.split(q)
        value = q.copy()
        value[pivot] = np.asarray(original.calibration(old).pivot_xyz, float) * scale
        return value

    def retract(q):
        q = np.asarray(q, np.float64)
        before = evaluation(q)
        domain = mount_graph.pivot_outside_domain(facade, before)
        if domain['valid']:
            return q, None
        after, proof = mount_graph.reinitialize_invalid_pivots_outside(
            facade, before, require_coordinate_fd_margin=False)
        result = q.copy()
        result[pivot[:2]] += after[pivot[:2]] - before[pivot[:2]]
        changed = np.flatnonzero(result.view(np.uint64) != q.view(np.uint64))
        if not set(changed).issubset(set(pivot[:2])):
            raise ValueError('Original pivot helper changed a nonpivot coordinate')
        actual = mount_graph.pivot_outside_domain(facade, evaluation(result))
        if not actual['valid']:
            raise ValueError('Mapped nearest-course pivot remains outside the original domain')
        return result, dict(original_helper=proof, actual_mapped_after_domain=actual,
            changed_columns=changed.tolist(), pivot_delta_working_length=(result[pivot]-q[pivot]).tolist(),
            all_nonpivot_complete_q_bytes_unchanged=True, added_physical_clearance=0.)

    def recover(adapter, parent, proposed):
        check()
        if not lock.acquire(blocking=False):
            raise RuntimeError('Pivot trial recovery requires a synchronous non-reentrant parent call')
        original_guard = adapter.guard
        captured = []
        thread = threading.get_ident()

        def observe_guard(target, q):
            try:
                return original_guard(q)
            except ValueError as error:
                if threading.get_ident() == thread and str(error) == _ERROR:
                    captured.append((error, np.asarray(q, np.float64).copy()))
                raise

        try:
            adapter.guard = MethodType(observe_guard, adapter)
            failed_error = None
            first = None
            try:
                first = previous_recover(parent, proposed)
            except ValueError as error:
                match = [q for observed, q in captured if observed is error]
                if str(error) != _ERROR or len(match) != 1:
                    raise
                candidate = match[0]
                failed_error = error
            finally:
                adapter.guard = original_guard
            if first is not None:
                candidate = np.asarray(first['q'], np.float64)
            corrected, proof = retract(candidate)
            if proof is None:
                if failed_error is not None:
                    raise failed_error
                receipt['valid_passthroughs'] += 1
                return first
            # Re-enter exactly the existing chain, so finite endpoints and all
            # its own reports/guards keep their established policy and evidence.
            result = previous_recover(parent, corrected)
            if not isinstance(result, dict) or 'q' not in result:
                raise TypeError('Original recovery must return its complete candidate and report')
            result = dict(result)
            adapter.guard(result['q'])
            current = np.asarray(result['q'], np.float64)
            proof.update(candidate_before_retraction_sha256=_sha(candidate),
                retracted_candidate_sha256=_sha(corrected), final_candidate_sha256=_sha(current),
                preceding_original_recovery_guard_error=None if failed_error is None else str(failed_error),
                subsequent_original_recovery_changed_columns=np.flatnonzero(
                    current.view(np.uint64) != corrected.view(np.uint64)).tolist(),
                full_original_guard_passed=True, source_cost_not_evaluated_by_this_helper=True,
                acceptance_not_claimed=True)
            result['pivot_boundary_recovery'] = proof
            receipt['repaired_candidates'] += 1
            receipt['last_recovery'] = deepcopy(proof)
            return result
        finally:
            adapter.guard = original_guard
            lock.release()

    def identity(adapter):
        result = previous_identity()
        result['original_pivot_boundary_trial_recovery'] = deepcopy(receipt)
        return result

    owner.recover = MethodType(recover, owner)
    owner.checkpoint_identity = MethodType(identity, owner)
    owner.pivot_boundary_recovery_receipt = receipt
    return receipt
