"""Exact same-source continuation after the public fresh-current-J handoff."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
import numpy as np
import fit_time_source as source
from calibration.program_tracking.first_shot_joint_cache_resume import (
    ExactTimeContinuation, equal, load_parent, qsha, read, reference, verify_refs)
from calibration.program_tracking.first_shot_joint_stage_runtime import JointStageRunner

construction = source.construction
_original_direction_QA = construction.verify_native_control_directions
_CONFIG = None
_CONFIG_REF = None
_PARENT = None


def load_continuation(path):
    global _CONFIG, _CONFIG_REF, _PARENT
    _CONFIG_REF = reference(path); _CONFIG = read(_CONFIG_REF)
    if (_CONFIG.get('schema') != 'rinkspace.exact-current-native-time-continuation.v1'
            or _CONFIG.get('configuration_status') != 'ready'
            or _CONFIG.get('entry') != reference(__file__)):
        raise ValueError('A ready immutable exact-current continuation manifest is required')
    verify_refs(_CONFIG)
    _PARENT = load_parent(_CONFIG)
    manifest, ref = source.load_successor(_CONFIG['reconstruct_source_manifest']['path'])
    if ref != _CONFIG['reconstruct_source_manifest']:
        raise ValueError('Original 8d source/T construction manifest changed')
    # Preserve this exact historical construction identity. The outer wrapper
    # carries the new final start and explicitly names its new manifest.
    return manifest, ref


def require_stopped(manifest):
    check = deepcopy(manifest)
    check['native_time_source_successor']['parent_owner_shutdown'] = _CONFIG['parent_owner_shutdown']
    check['native_time_source_successor']['completed_parent_branch'] = _CONFIG['completed_parent_branch']
    return source.require_all_parents_stopped(check)


def restore_same_source(base_owner, manifest, manifest_ref, output):
    inventory, media = source.restore_then_revise_source(base_owner, manifest, manifest_ref, output)
    with np.load(_CONFIG['parent_source_inventory']['path'], allow_pickle=False) as saved:
        if set(saved.files) != set(inventory):
            raise ValueError('Same-source continuation point inventory fields changed')
        for key in inventory:
            equal(inventory[key], saved[key], 'all original UV/ID/PTS/labels ' + key)
    old_media, new_media = read(_CONFIG['parent_media_manifest']), read(media)
    if (old_media['frames'] != new_media['frames']
            or old_media['source_frame_map'] != new_media['source_frame_map']
            or old_media['native_pts_sha256'] != new_media['native_pts_sha256']):
        raise ValueError('Actual 8d media, original W ledgers or complete native clock changed')
    for key in ('formal_inventory',):
        with np.load(old_media[key]['path'], allow_pickle=False) as old, np.load(new_media[key]['path'], allow_pickle=False) as new:
            if set(old.files) != set(new.files):
                raise ValueError('Same-source formal inventory fields changed')
            for field in old.files:
                equal(new[field], old[field], 'all source formal metadata ' + field)
    reconstructed = manifest['actual_source_successor_transition']
    original = _PARENT['identity']['source_successor']['source_transition']
    verify_refs(original)
    construction.write(output/'same_source_reconstruction.json', dict(passed=True,
        original_8d_source_transition=original, reconstruction_transition=reconstructed,
        original_UV_ID_PTS_W_sigma_masks_formal_order_and_media_all_equal=True,
        parent_cache_not_yet_replayed=True, producer=reference(__file__)))
    # Source facts are exactly the original 8d facts, not a second source delta.
    manifest['actual_source_successor_transition'] = original
    return inventory, media


def make_stage(base_owner, manifest):
    historical = source.make_accepted_time_successor(base_owner, manifest)
    owner = ExactTimeContinuation(historical, _PARENT, manifest_reference=_CONFIG_REF,
        entry_reference=reference(__file__), config=_CONFIG)
    return owner


def numerical_QA(owner, numerical, historical_parent):
    history = source.verify_successor_numerical_state(owner.base, numerical, historical_parent)
    kwargs = owner.solver_kwargs()
    if kwargs['initial_damping'] != _PARENT['damping'] or kwargs['initial_line_search_fraction'] != _PARENT['fraction']:
        raise ValueError('Actual final accepted lambda/fraction changed')
    equal(owner.start, _PARENT['q'], 'actual final accepted canonical start')
    for key,value in owner.checkpoint_arrays().items():
        equal(value, _PARENT['aux'][key], 'all final accepted aux ' + key)
    return dict(original_T_construction_history_QA=history,
        complete_actual_final_q_aux_lambda_fraction_byte_exact=True,
        exact_current_continuation=owner.continuation_receipt)


def verify_current_directions(owner, historical_configuration, output):
    """Both current-q local camera FD steps use the untouched public criterion."""
    output = Path(output)
    shadow = owner.exact_cache.verify_live_shadow(
        owner.base.source_blocks(owner.start, False), update_profiles=owner.solver_kwargs().get('update_profiles'))
    checks = []
    for label, ref in zip(('1e-8', '1e-9'), _CONFIG['current_direction_configurations'], strict=True):
        config = read(ref)
        expected = float(label)
        if any(float(config['central_steps'][k]) != expected for k in ('pan', 'tilt', 'logf')):
            raise ValueError('Explicit current local FD steps changed')
        checks.append(_original_direction_QA(owner, ref, output.with_name('native_direction_'+label+'.json')))
    construction.write(output, dict(schema='rinkspace.current-handoff-full-source-and-local-direction-QA.v1',
        passed=True, canonical_q_sha256=qsha(owner.start), original_public_thresholds_unchanged=True,
        current_q_FD_1e_minus_8_and_1e_minus_9=checks,
        full_original_source_shadow=shadow, historical_q_FD_not_used_as_current=True,
        complete_world_J_not_recomputed=True, producer=reference(__file__)))
    return reference(output)


class CacheRunner(source.SuccessorRunner):
    def prepare(self, **extra):
        receipt = JointStageRunner.prepare(self, **extra)
        receipt.update(initial_J_or_old_source_cache_borrowed=True,
            old_source_r_J_or_cost_loaded=True, initializer_predictions_changed_by_explicit_time_subspace=False,
            fresh_full_world171_and_time_control_J_required=False,
            complete_exact_current_handoff_J_authenticated=True,
            original_T_initialization_not_reexecuted=True,
            complete_parent_checkpoint=_CONFIG['parent_checkpoint'],
            exact_current_continuation=self.owner.continuation_receipt,
            complete_current_original_source_shadow=self.owner.exact_cache.shadow_receipt,
            no_optimizer_trial_or_current_complete_world_J_recomputed=True)
        construction.write(self.output/'append_QA.json', receipt)
        return receipt

    def source_blocks(self, q, jacobian):
        cache = self.owner.exact_cache
        if not jacobian or not cache.same_q(q):
            yield from JointStageRunner.source_blocks(self, q, jacobian)
            return
        bank = self.cache/('generation_%06d'%self.generation)
        bank.mkdir(parents=True, exist_ok=False)
        # Reference the owning parent's existing CSR files, without copying or
        # multiplying their already canonical native-time Jacobian columns.
        for block in self.owner.source_blocks(q, True):
            yield block
        np.savez(bank/'linearization_state.npz', **self.owner.complete_state_payload(q),
            r=cache.r, scales=self.owner.scales, quadratic=self.owner.quadratic)
        manifest = dict(schema=cache.manifest['schema'], generation=self.generation,
            qaug_sha256=qsha(q), J_shape=cache.manifest['J_shape'], J_nonzeros=cache.manifest['J_nonzeros'],
            complete_current_state_r_scales_quadratic=reference(bank/'linearization_state.npz'),
            blocks=deepcopy(cache.manifest['blocks']), block_loss_weights=cache.manifest['block_loss_weights'],
            source_J_before_Huber_weights=True, source_producers=self.owner.source_producers,
            source_identity=self.owner.checkpoint_identity(), full_static_rows_always_present=True,
            exact_same_q_parent_handoff_cache=cache.manifest_ref,
            complete_live_original_source_shadow=cache.shadow_receipt,
            duplicate_CSR_export=False, second_T_chain_multiplication=False)
        construction.write(bank/'manifest.json', manifest)
        construction.write(self.cache/'latest.json', manifest)
        self.generation += 1
        print(json.dumps(dict(stage='exact_current_combined_source_J_referenced', q_sha256=qsha(q),
            rows=len(self.owner.scales), original_CSR_files_not_copied=True)), flush=True)


def main():
    construction.load_manifest = load_continuation
    construction.require_stopped_parent = require_stopped
    construction.restore_review_inputs = restore_same_source
    construction.make_time_stage = make_stage
    construction.verify_execution_numerical_state = numerical_QA
    construction.verify_native_control_directions = verify_current_directions
    construction.JointStageRunner = CacheRunner
    construction.main()


if __name__ == '__main__':
    main()
