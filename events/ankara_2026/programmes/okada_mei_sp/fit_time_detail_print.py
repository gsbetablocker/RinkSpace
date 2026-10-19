"""Explicit complete-parent time-detail and print-shape model successor."""
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import subprocess
import os

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
import fit_time_source as source
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from calibration.program_tracking.first_shot_source_transition import reference, read
from calibration.program_tracking.first_shot_time_successor import read_accepted_parent, authenticate_references
from calibration.program_tracking.first_shot_joint_stage_runtime import JointStageRunner
from calibration.program_tracking.first_shot_time_detail_stage import NativeTimeDetailVenueStage
from calibration.program_tracking import first_shot_disjoint_print as print_runtime

construction = source.construction
SCHEMA = 'rinkspace.native-time-detail-print-successor.v1'
_CONFIG = _CONFIG_REF = _PARENT = None
_original_camera = construction.install_camera
_original_directions = construction.verify_native_control_directions


def load_configuration(path):
    global _CONFIG, _CONFIG_REF, _PARENT
    _CONFIG_REF = reference(path)
    _CONFIG = read(_CONFIG_REF)
    if (_CONFIG.get('schema') != SCHEMA or _CONFIG.get('configuration_status') != 'ready'
            or _CONFIG.get('entry') != reference(__file__)
            or _CONFIG.get('print_runtime') != reference(print_runtime.__file__)
            or _CONFIG.get('detail_runtime') != reference(sys.modules[NativeTimeDetailVenueStage.__module__].__file__)):
        raise ValueError('Ready explicit same-parent model successor required')
    authenticate_references(_CONFIG)
    for key in ('detail_protocol_QA', 'print_protocol_QA'):
        if read(_CONFIG[key]).get('passed') is not True:
            raise ValueError('Passed real source/model protocol QA required: '+key)
    _PARENT = read_accepted_parent(_CONFIG)
    if _PARENT['q_sha256'] != _CONFIG['parent_q_sha256']:
        raise ValueError('Sealed complete final accepted parent differs')
    manifest, old_ref = source.load_successor(_CONFIG['reconstruct_source_manifest']['path'])
    if old_ref != _CONFIG['reconstruct_source_manifest']:
        raise ValueError('Historical original source reconstruction changed')
    return manifest, _CONFIG_REF


def require_stopped(manifest):
    # A recycled PID can now belong to a protected Windows service. CIM creation
    # time proves that reuse without opening the protected process or inventing
    # an unavailable command line. Historical receipt authentication stays intact.
    historic = read(manifest['previous_owner_shutdown'])
    inventory = read(historic['process_inventory'])
    thresholds = {int(row['ProcessId']): datetime.fromisoformat(row['CreationDate'].replace('Z', '+00:00'))
        for row in inventory['processes']}
    for shutdown in (manifest['native_time_source_successor']['parent_owner_shutdown'],
                     _CONFIG['parent_owner_shutdown']):
        stopped = read(shutdown)
        after = datetime.fromisoformat(stopped['checked_at_utc'].replace('Z', '+00:00'))
        for pid in stopped['checked_pid_list']:
            pid = int(pid)
            thresholds[pid] = max(after, thresholds.get(pid, after))
    pwsh = 'C:/Users/REDACTED/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe'
    query = 'Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Depth 4 -Compress'
    rows = json.loads(subprocess.check_output([pwsh, '-NoProfile', '-Command', query], text=True, encoding='utf-8'))
    if not isinstance(rows, list):
        rows = [rows]
    def this_entry_launcher(row):
        command = (row.get('CommandLine') or '').replace('\\', '/').casefold()
        return (int(row['ProcessId']) == os.getppid()
            and all(str(path).replace('\\', '/').casefold() in command for path in
                (__file__, _CONFIG_REF['path'], _CONFIG['planned_output']))
            and Path(row.get('ExecutablePath') or '').resolve() == (ROOT/'.venv/Scripts/python.exe').resolve())
    newer = []
    for row in rows:
        pid = int(row['ProcessId'])
        command = row.get('CommandLine') or ''
        if (row.get('Name') or '').lower().startswith('python') and any(token in command for token in
                ('fit_time_cache.py', 'fit_time_source.py', 'fit_fixed_head_time.py', 'fit_time_detail_print.py')):
            # This preparation/entry process is the only permitted new instance.
            if pid != os.getpid() and not this_entry_launcher(row):
                raise ValueError('Another formal time-fit instance is visible: '+repr(row))
        if pid not in thresholds:
            continue
        created = datetime.fromisoformat(row['CreationDate'].replace('Z', '+00:00'))
        if created.tzinfo is None or created <= thresholds[pid]:
            raise ValueError('Historical parent PID is not proved to be a newer instance: '+repr(row))
        newer.append(dict(pid=pid, current_instance=row,
            historical_or_absence_threshold_utc=thresholds[pid].astimezone(timezone.utc).isoformat(),
            current_instance_created_strictly_later=True))
    def certified_parent_instances(pids):
        if not set(map(int, pids)) <= set(thresholds):
            raise ValueError('Historical caller requested an unauthenticated PID')
        return []  # All present requested PIDs were separately proved newer above.
    original_query = construction._current_windows_processes
    try:
        construction._current_windows_processes = certified_parent_instances
        result = source.require_all_parents_stopped(manifest)
    finally:
        construction._current_windows_processes = original_query
    proof = read(_CONFIG['parent_owner_shutdown'])
    completed = read(_CONFIG['completed_parent_branch'])
    if (proof.get('passed') is not True or proof.get('full_windows_process_domain') is not True
            or proof.get('all_observed_owner_launcher_and_workers_absent') is not True
            or proof.get('no_time_fit_owner_visible') is not True
            or proof.get('no_process_terminated') is not True
            or completed.get('status') not in ('no_accepted_trial', 'stopped_by_control')
            or proof.get('terminal_status') != completed.get('status')
            or Path(proof['result']).resolve() != Path(_CONFIG['completed_parent_branch']['path']).resolve()
            or proof.get('final_q_sha256') != _PARENT['q_sha256']
            or proof.get('complete_state') != completed.get('final_complete_state')
            or proof.get('accepted_parent_checkpoint') != _CONFIG['accepted_parent_checkpoint']
            or proof.get('final_public_accepted_q_aux_damping_fraction_byte_exact') is not True):
        raise ValueError('Actual complete terminal parent and full Windows absence proof required')
    pids = list(map(int, proof['checked_pid_list']))
    absent = datetime.fromisoformat(proof['checked_at_utc'].replace('Z', '+00:00'))
    if not pids or len(set(pids)) != len(pids) or min(pids) <= 0 or absent.tzinfo is None:
        raise ValueError('Observed distinct parent PID inventory and qualified absence time required')
    reused = []
    for row in certified_parent_instances(pids):
        row = construction._process_identity(row)
        if datetime.fromisoformat(row['CreationDate']) <= absent:
            raise ValueError('Parent process instance has not exited')
        reused.append(row)
    return dict(result, detail_print_parent_shutdown=_CONFIG['parent_owner_shutdown'],
        complete_parent_terminal_status=completed['status'], current_newer_PID_reuse=newer,
        full_windows_CIM_inventory_checked=True, current_process_count=len(rows),
        protected_process_commandline_null_preserved=True)


def install_camera_and_material(programme, **kwargs):
    result = _original_camera(programme, **kwargs)
    result['explicit_print_model_successor'] = print_runtime.install_successor(programme,
        configuration_reference=_CONFIG['print_configuration'], entry_reference=reference(__file__),
        transform_source_targets=False)
    return result


def restore_inputs(base_owner, manifest, manifest_ref, output):
    with print_runtime.historical_source_templates(base_owner.programme):
        inventory, media = source.restore_then_revise_source(base_owner, manifest, manifest_ref, output)
    # Old source-delta identity checks have finished before kind/alias metadata changes.
    proof = print_runtime.refresh_source_targets(base_owner.programme)
    construction.write(Path(output)/'print_source_transition.json', proof)
    return inventory, media


def camera_detail_support(probe):
    """Only exact structural source support; this does not certify identifiability."""
    from native_transverse_bundle import source_chunk
    q = probe.start
    physical, _ = probe.effective_q_and_chain(q)
    programme = probe.programme
    worlds, _ = programme.world(physical[:probe.programme_count], False)
    nworld = len(programme.state.world_parameter_indices)
    zeros = {key: np.zeros(value.shape+(nworld,)) for key, value in worlds.items()}
    frame_groups = [group.tolist() for group in np.array_split(np.arange(1248), 6)]
    tasks = [(physical[:probe.programme_count], worlds, zeros, group,
              construction.sha(physical[:probe.programme_count])) for group in frame_groups]
    matrices = []
    receipts = []
    rows = 0
    for group, (residual, native_j, worker) in zip(frame_groups,
            programme._pool().map(source_chunk, tasks, chunksize=1), strict=True):
        partial = probe.pullback_native_camera_partials(native_j,
            input_basis='physical_native_camera', q=q)
        matrices.append(csr_matrix(partial))
        rows += len(residual)
        receipts.append(dict(frames=[group[0]+1, group[-1]+1], rows=len(residual), worker=worker))
    if rows != len(programme.scales):
        raise ValueError('Native camera support omitted original formal source rows')
    candidate = vstack(matrices, format='csr')
    counts = np.asarray(candidate.getnnz(axis=0)).ravel()
    if candidate.shape != (rows, 468):
        raise ValueError('Explicit complete candidate detail inventory required')
    selected = np.flatnonzero(counts).astype(np.int64)
    return selected, dict(schema='rinkspace.current-source-native-time-detail-support.v1',
        passed=True, canonical_q_sha256=construction.sha(q), candidate_details=468,
        supported_detail_indices=selected.tolist(), exact_nonzero_rows_by_detail=counts.tolist(),
        programme_formal_rows=rows, complete_native_exposures=1248,
        original_source_UV_W_RS_and_clock_used=True, world_derivatives_zero_only_for_camera_support=True,
        shared_world_or_static_J_not_computed_or_reused=True,
        original_derived_chain_applied_once=True, data_identifiability_or_accuracy_certified=False,
        detail_basis_choice_is_a_model_candidate=True, workers=receipts)


def make_stage(base_owner, manifest):
    common = dict(parent=_PARENT, entry_reference=reference(__file__), transition_reference=_CONFIG_REF,
        allowed_rebuilt_aux=['print_H8_QP_domain_identity_json'],
        live_model_reference=_CONFIG['print_configuration'])
    probe = NativeTimeDetailVenueStage(base_owner, active_details=[], **common)
    if _CONFIG.get('detail_selection_policy') != 'all_exactly_source_supported_candidate_details':
        raise ValueError('Explicit candidate source-support policy required')
    selected, proof = camera_detail_support(probe)
    owner = NativeTimeDetailVenueStage(base_owner, plan=probe.plan, active_details=selected, **common)
    owner.detail_support_receipt = proof
    owner.source_successor_receipt = dict(schema=SCHEMA, parent=_CONFIG['accepted_parent_checkpoint'],
        parent_q_sha256=_PARENT['q_sha256'], unchanged_source_inventory=True,
        original_T_initialization_not_reexecuted=True, print_configuration=_CONFIG['print_configuration'],
        detail_model=owner.detail_receipt, all_prior_corrections_and_aux_preserved=True,
        old_r_J_or_cost_reused=False, new_shape_predictions_not_claimed_equivalent=True)
    return owner


def verify_numerical(owner, numerical, historical_parent):
    from calibration.program_tracking.first_shot_joint_initializer import _same
    kwargs = owner.solver_kwargs()
    if (kwargs['initial_damping'] != _PARENT['damping']
            or kwargs['initial_line_search_fraction'] != _PARENT['fraction']):
        raise ValueError('Actual full accepted numerical state changed')
    _same(owner.start, _PARENT['q'], 'all final accepted canonical parameters')
    aux = owner.checkpoint_arrays()
    for key, value in _PARENT['aux'].items():
        _same(aux[key], value, 'all accepted auxiliary state '+key)
    return dict(passed=True, parent_q_sha256=_PARENT['q_sha256'],
        complete_parent_q_and_aux_byte_exact=True, parent_aux_count=len(_PARENT['aux']),
        initial_damping=_PARENT['damping'], initial_fraction=_PARENT['fraction'],
        historical_construction_numeric_seed_not_used_for_new_execution=True,
        print_template_changes_are_explicit_separate_model_facts=True)


def verify_directions(owner, historical_configuration, output):
    output = Path(output)
    saved = owner.control_slots
    checks = []
    try:
        for label, ref in zip(('1e-8', '1e-9'), _CONFIG['current_direction_configurations'], strict=True):
            owner.control_slots = saved
            checks.append(_original_directions(owner, ref, output.with_name('coarse_direction_'+label+'.json')))
            selected = set(map(int, owner.active_details))
            detail_slots = {}
            for axis, kind in enumerate(('pan', 'tilt', 'logf')):
                detail_slots[kind] = np.asarray([owner.plan.slots[axis, j] for j in range(156)
                    if axis*156+j in selected], np.int64)
            if any(not len(v) for v in detail_slots.values()):
                raise ValueError('Every detail camera axis needs actual source support')
            owner.control_slots = detail_slots
            checks.append(_original_directions(owner, ref, output.with_name('detail_direction_'+label+'.json')))
    finally:
        owner.control_slots = saved
    construction.write(output, dict(schema='rinkspace.actual-time-detail-print-source-QA.v1',
        passed=True, canonical_q_sha256=construction.sha(owner.start), checks=checks,
        candidate_detail_support=owner.detail_support_receipt,
        original_camera_source_thresholds_and_complete_W_unchanged=True,
        geometry_print_H8_real_source_protocol=_CONFIG['print_protocol_QA'],
        all_source_J_must_be_fresh_for_new_model=True, whole_shot_accuracy_accepted=False))
    return reference(output)


class DetailPrintRunner(source.SuccessorRunner):
    def prepare(self, **extra):
        extra.update(complete_parent_checkpoint=_CONFIG['accepted_parent_checkpoint'],
            explicit_model_successor=owner_receipt(self.owner),
            original_T_initialization_not_reexecuted=True, old_source_r_J_or_cost_loaded=False,
            fresh_full_world_and_detail_camera_J_required=True)
        return JointStageRunner.prepare(self, **extra)


def owner_receipt(owner):
    return dict(source_model_successor=owner.source_successor_receipt,
        time_detail_model=owner.detail_receipt, current_source_support=owner.detail_support_receipt)


def main():
    construction.load_manifest = load_configuration
    construction.require_stopped_parent = require_stopped
    construction.install_camera = install_camera_and_material
    construction.restore_review_inputs = restore_inputs
    construction.make_time_stage = make_stage
    construction.verify_execution_numerical_state = verify_numerical
    construction.verify_native_control_directions = verify_directions
    construction.JointStageRunner = DetailPrintRunner
    construction.main()


if __name__ == '__main__':
    main()
