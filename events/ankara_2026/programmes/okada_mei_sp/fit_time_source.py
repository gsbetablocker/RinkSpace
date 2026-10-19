"""Reviewed source successor of a complete accepted native-time fit.

The frozen construction entry still authenticates j24 and constructs its one
original T chart. Explicit pre-pool callbacks change only reviewed source
metadata, then wrap that chart with the complete accepted native-time parent.
No T reinitialization, previous residual/J replay or producer edit is performed.
"""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
import fit_fixed_head_time as construction
import numpy as np
from calibration.program_tracking.first_shot_source_transition import reference, read
from calibration.program_tracking.first_shot_joint_initializer import _same
from calibration.program_tracking.first_shot_safe_frame_export import _encode
from calibration.program_tracking.first_shot_joint_stage_runtime import JointStageRunner
from calibration.program_tracking.first_shot_time_successor import (
    SCHEMA, authenticate_references, read_accepted_parent, apply_reviewed_source_delta,
    export_successor_inventory, NativeTimeSourceSuccessor)
from calibration.program_tracking import first_shot_time_successor as successor_adapter

_load_construction = construction.load_manifest
_require_construction_stopped = construction.require_stopped_parent
_restore_construction_inputs = construction.restore_review_inputs
_make_construction_time_stage = construction.make_time_stage
_verify_construction_numerical_state = construction.verify_execution_numerical_state


class SuccessorRunner(JointStageRunner):
    """Use the public runner with the exact accepted numerical search seed."""
    def prepare(self, **extra):
        extra.update(complete_parent_checkpoint=self.owner._parent['checkpoint'],
            source_successor=self.owner.source_successor_receipt,
            accepted_native_time_parameters_loaded_byte_exact=True,
            original_T_initialization_not_reexecuted=True,
            old_source_r_J_or_cost_loaded=False, fresh_full_world171_and_time_control_J_required=True)
        return super().prepare(**extra)

    def run(self, *, wait_for_control=True):
        import z190_joint_source_solver as solver
        from calibration.program_tracking.first_shot_jacobian_reuse import install_source_jacobian_reuse
        from calibration.program_tracking.first_shot_prediction_refresh import install_prediction_refresh
        owner = self.owner
        install_source_jacobian_reuse(owner, cache_directory=self.cache, entry_reference=reference(__file__))
        driver = SimpleNamespace(solve_joint_source=solver.solve_joint_source, solver_runtime=solver)
        install_prediction_refresh(owner, driver=driver, entry_reference=reference(__file__),
            minimum_prediction_ratio=.25, reset_initial_fresh_source_fraction=False)
        kwargs = owner.solver_kwargs(); kwargs['source_block_factory'] = self.source_blocks
        if (kwargs['initial_damping'] != owner._parent['damping']
                or kwargs['initial_line_search_fraction'] != owner._parent['fraction']):
            raise ValueError('Accepted parent damping/fraction changed after all execution wrappers')
        kwargs['checkpoint_identity'].update(explicit_stage_manifest=self.manifest_reference,
            source_inventory=reference(self.output/'immutable_programme_source_inventory.npz'),
            media_manifest=self.media, completed_append_QA=reference(self.output/'append_QA.json'))
        construction.write(self.output/'run_identity.json', kwargs['checkpoint_identity'])
        if wait_for_control:
            while True:
                path = self.output/'stage_control.json'
                if path.exists():
                    control = read(reference(path))
                    if control.get('action') == 'close_owner':
                        return None
                    if control.get('action') == 'start_fit':
                        if (control.get('append_QA') != reference(self.output/'append_QA.json')
                                or control.get('q_sha256') != construction.sha(owner.start)):
                            raise ValueError('Source successor start must bind the actual new-source QA')
                        break
                time.sleep(2.)
        construction.write(self.output/'status.json', dict(stage='formal_fit_running',
            pid=__import__('os').getpid(), fit_started=True))
        result = driver.solve_joint_source(owner.evaluate, owner.start, owner.quadratic, owner.scales,
            self.output/'public_solver', owner.guard, checkpoint_callback=self.checkpoint,
            source_stage_timing=True, verbose=True, **kwargs)
        np.savez_compressed(self.output/'final_complete_state.npz', **owner.complete_state_payload(result.q))
        construction.write(self.output/'result.json', dict(status=result.status, stop_reason=result.stop_reason,
            q_sha256=construction.sha(result.q), final_complete_state=reference(self.output/'final_complete_state.npz'),
            accepted_iterations=sum(record['kind']=='accepted' for record in self.records),
            source_successor=owner.source_successor_receipt, accuracy_accepted=False, video_generated=False))
        return result


def load_successor(path):
    ref = reference(path); successor = read(ref)
    if (successor.get('schema') != SCHEMA or successor.get('configuration_status') != 'ready'
            or successor.get('entry') != reference(__file__)
            or successor.get('construction_entry') != reference(construction.__file__)
            or successor.get('successor_adapter') != reference(successor_adapter.__file__)):
        raise ValueError('A ready explicit source-successor manifest and frozen construction entry are required')
    for key in ('construction_manifest', 'accepted_parent_checkpoint', 'accepted_parent_payload',
                'accepted_parent_report', 'completed_parent_branch', 'parent_owner_shutdown', 'source_delta'):
        if not isinstance(successor.get(key), dict) or not {'path','sha256'} <= set(successor[key]):
            raise ValueError('Explicit frozen source-successor reference required: '+key)
    authenticate_references(successor)
    manifest, construction_ref = _load_construction(successor['construction_manifest']['path'])
    if construction_ref != successor['construction_manifest']:
        raise ValueError('Complete original T construction manifest changed')
    manifest['native_time_source_successor'] = successor
    manifest['native_time_source_successor_reference'] = ref
    if successor.get('source_direction_configuration') is not None:
        manifest['native_time_direction_configuration'] = successor['source_direction_configuration']
    read_accepted_parent(successor)  # No owner, source rows or pool is constructed.
    return manifest, ref


def require_all_parents_stopped(manifest):
    result = _require_construction_stopped(manifest)
    successor = manifest['native_time_source_successor']
    stopped = read(successor['parent_owner_shutdown'])
    completed = read(successor['completed_parent_branch'])
    if (stopped.get('passed') is not True
            or stopped.get('all_observed_owner_launcher_and_workers_absent') is not True
            or Path(stopped['result']).resolve() != Path(successor['completed_parent_branch']['path']).resolve()
            or completed.get('status') != 'stopped_by_control'):
        raise ValueError('Normal completed native-time parent shutdown receipt required')
    pids = [int(pid) for pid in stopped['checked_pid_list']]
    if not pids or len(set(pids)) != len(pids) or any(pid <= 0 for pid in pids):
        raise ValueError('Complete distinct native-time parent observed PID inventory required')
    absent_at = datetime.fromisoformat(stopped['checked_at_utc'].replace('Z', '+00:00'))
    if absent_at.tzinfo is None:
        raise ValueError('Native-time parent absence proof requires a timezone-qualified timestamp')
    current = [construction._process_identity(row) for row in construction._current_windows_processes(pids)]
    reused = []
    for row in current:
        created = datetime.fromisoformat(row['CreationDate'])
        if row['ProcessId'] not in pids or created <= absent_at:
            raise ValueError('Visible recorded PID is not proved newer than authenticated parent absence: '
                             +str(row['ProcessId']))
        reused.append(dict(pid=row['ProcessId'], current_instance=row,
                           created_strictly_after_authenticated_parent_absence=True))
    return dict(result, accepted_native_time_parent_shutdown=successor['parent_owner_shutdown'],
        accepted_native_time_parent_instances_proved_exited=pids,
        trusted_parent_absence_at_utc=absent_at.astimezone(timezone.utc).isoformat(),
        current_PID_reuse_after_proved_parent_absence=reused, no_process_terminated=True)


def restore_then_revise_source(base_owner, manifest, manifest_ref, output):
    _, old_media = _restore_construction_inputs(base_owner, manifest, manifest_ref, output)
    successor = manifest['native_time_source_successor']
    (output/'source_successor_entry_snapshot.py').write_bytes(Path(__file__).read_bytes())
    # Preserve the just-authenticated old inventory beside the current one.
    (output/'immutable_programme_source_inventory.npz').replace(output/'construction_source_inventory.npz')
    proof = apply_reviewed_source_delta(base_owner, successor['source_delta'], successor['expected_source_counts'],
        producer_references=(reference(__file__), manifest_ref))
    arrays = {}; encoded = _encode(proof, arrays)
    array_path = output/'source_successor_transition_arrays.npz'
    np.savez_compressed(array_path, **arrays)
    proof_path = output/'source_successor_transition.json'
    construction.write(proof_path, dict(schema='rinkspace.safe-reviewed-source-successor-transition.v1',
        passed=True, payload=encoded, arrays=reference(array_path),
        delta=successor['source_delta'], parent=successor['accepted_parent_checkpoint'],
        source_or_geometry_pool_opened=False, producer=reference(__file__)))
    manifest['actual_source_successor_transition'] = reference(proof_path)
    return export_successor_inventory(base_owner.programme, old_media, output,
                                      transition_reference=manifest['actual_source_successor_transition'])


def make_accepted_time_successor(base_owner, manifest):
    original_time = _make_construction_time_stage(base_owner, manifest)
    successor = manifest['native_time_source_successor']
    parent = read_accepted_parent(successor)
    owner = NativeTimeSourceSuccessor(original_time, parent,
        manifest_reference=manifest['native_time_source_successor_reference'],
        source_transition_reference=manifest['actual_source_successor_transition'], entry_reference=reference(__file__))
    owner.programme.seed_receipt.update(initial_damping=parent['damping'],
        initial_line_search_fraction=parent['fraction'], complete_native_time_source_parent=parent['checkpoint'],
        source_successor_accepted_q_sha256=parent['q_sha256'])
    return owner


def verify_successor_numerical_state(owner, numerical, construction_parent):
    inherited = _verify_construction_numerical_state(owner.base, numerical, construction_parent)
    kwargs = owner.solver_kwargs(); aux = owner.checkpoint_arrays(); parent = owner._parent
    if (kwargs['initial_damping'] != parent['damping']
            or kwargs['initial_line_search_fraction'] != parent['fraction']):
        raise ValueError('Actual source-successor numerical arguments differ from complete accepted parent')
    for key, value in parent['aux'].items():
        _same(aux[key], value, 'all current native-time parent auxiliary arrays: '+key)
        _same(aux['source_successor_parent__'+key], value, 'all archived native-time parent auxiliaries: '+key)
    return dict(construction_historical_numerical_QA=inherited,
        actual_successor_parent_damping=parent['damping'], actual_successor_parent_fraction=parent['fraction'],
        accepted_parent_numerical_arguments_and_all_aux_byte_exact=True,
        original_recipe_initialization_not_claimed_to_be_accepted_parent=True)


def main():
    construction.load_manifest = load_successor
    construction.require_stopped_parent = require_all_parents_stopped
    construction.restore_review_inputs = restore_then_revise_source
    construction.make_time_stage = make_accepted_time_successor
    construction.verify_execution_numerical_state = verify_successor_numerical_state
    construction.JointStageRunner = SuccessorRunner
    construction.main()


if __name__ == '__main__':
    main()
