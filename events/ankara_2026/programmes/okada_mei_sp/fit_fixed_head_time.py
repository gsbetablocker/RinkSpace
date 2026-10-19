"""Explicit native-time subspace successor of a complete fixed-head joint fit.

The supplied recipe chooses the time basis. This entry chooses no capacity,
knots, speed limits or observations. A draft configuration cannot construct an
owner. A ready configuration requires a stopped parent, its complete accepted
state, and a separately authenticated control initializer.
"""
from pathlib import Path
import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import inspect
import json
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(r'D:\CodexData\RinkSpace_Data')
Z = DATA/'events/ankara_2026/programmes/okada_mei_sp/results/shot1_physical_fit_20261014/zoom_poly_72ac'
sys.path[:0] = [str(ROOT), str(Z), str(Z.parent/'head_axis_audit'), str(Path(__file__).parent)]
from calibration.solver_runtime import install
install()
import numpy as np
from scipy.sparse import csr_matrix, hstack
import fit_fixed_head_print_qp as previous
from calibration.program_tracking.first_shot_source_transition import reference, read
from calibration.program_tracking.first_shot_joint_initializer import read_complete_parent, _same
from calibration.program_tracking.first_shot_joint_stage_runtime import JointStageRunner, export_effective_review, sha, write
from calibration.program_tracking.first_shot_restored_owner import restore_programme_owner
from calibration.program_tracking.first_shot_fixed_head_transition import expand_derived_chart, install as install_camera
from calibration.program_tracking.first_shot_derived_inactive import DerivedInactiveChart
from calibration.program_tracking.first_shot_derived_venue_stage import DerivedVenueStage
from calibration.program_tracking.first_shot_static_venue_stage import StaticVenueStage
from calibration.program_tracking.static_venue_source import StaticVenueFactor, StaticCameraPolicy
from calibration.program_tracking.static_venue_admission import VerifiedStaticMaterialPacket
from calibration.program_tracking.first_shot_native_time_stage import NativeTimeVenueStage, recipe_contract
from native_transverse_bundle import load_model

SCHEMA = 'rinkspace.restored-fixed-head-native-time-stage.v1'
REQUIRED_OVERRIDES = (
    'saved_camera_checkpoint', 'saved_camera_payload', 'saved_camera_report',
    'saved_camera_parent_kind', 'saved_camera_q_sha256', 'constant_principal_policy',
    'previous_completed_branch', 'previous_owner_shutdown', 'native_time_stage',
    'native_time_protocol_QA', 'native_time_recipe', 'native_time_basis_arrays',
    'native_time_initializer', 'native_time_direction_configuration')


def load_manifest(path):
    ref = reference(path); current = read(ref)
    if (current.get('schema') != SCHEMA or current.get('configuration_status') != 'ready'
            or current.get('entry') != reference(__file__)):
        raise ValueError('A ready, explicit native-time entry manifest is required; draft cannot start')
    for name in REQUIRED_OVERRIDES:
        if name not in current or current[name] is None:
            raise ValueError('Native-time transition must explicitly replace '+name)
        if name not in ('saved_camera_parent_kind', 'saved_camera_q_sha256'):
            value = current[name]
            if not isinstance(value, dict) or not {'path', 'sha256'} <= set(value):
                raise ValueError('Explicit immutable transition reference required: '+name)
    inherited = read(current['inherited_configuration'])
    if inherited.get('entry') != reference(previous.__file__):
        raise ValueError('Frozen complete-parent construction entry changed')
    manifest = dict(inherited, **current)
    for value in manifest.values():
        if isinstance(value, dict) and {'path', 'sha256'} <= set(value):
            if reference(value['path']) != {k:value[k] for k in ('path', 'sha256')}:
                raise ValueError('Explicit source/model/transition reference changed: '+value['path'])
    return manifest, ref


def preflight_time_files(manifest):
    """Validate actual safe initializer/recipe/checkpoint metadata without an owner."""
    with np.load(manifest['saved_camera_checkpoint']['path'], allow_pickle=False) as saved:
        parent = saved['q'].copy()
        aux = {key:saved[key].copy() for key in saved.files if key.startswith('aux__')}
        damping = float(saved['damping']); fraction = float(saved['next_line_search_fraction'])
    if (parent.dtype != np.float64 or parent.shape != (7700,) or not np.isfinite(parent).all()
            or sha(parent) != manifest['saved_camera_q_sha256']
            or not np.isfinite(damping) or damping < 0 or not 0 < fraction <= 1):
        raise ValueError('Complete finite saved native-time parent and numerical state required')
    with np.load(manifest['saved_camera_payload']['path'], allow_pickle=False) as saved:
        _same(saved['q_augmented'], parent, 'complete saved canonical payload before T')
        for key, value in aux.items():
            _same(saved[key], value, 'all complete saved auxiliary arrays before T: '+key)
    recipe = read(manifest['native_time_recipe'])
    with np.load(manifest['native_time_basis_arrays']['path'], allow_pickle=False) as saved:
        basis = csr_matrix((saved['data'], saved['indices'], saved['indptr']), shape=tuple(saved['shape']))
        pts = saved['native_pts'].copy()
    with np.load(manifest['native_time_initializer']['path'], allow_pickle=False) as saved:
        initial = saved['q'].copy(); _same(saved['parent_q'], parent, 'explicit time initializer parent')
    contract = recipe_contract(basis, pts, recipe['control_slots'], initial, parent)
    if any(recipe.get(key) != value for key, value in contract.items()) or basis.shape[0] != 1248:
        raise ValueError('Actual basis/PTS/slots/initializer hashes differ from explicit recipe')
    if 'aux__native_pts' not in aux:
        raise ValueError('Actual saved parent lacks its native PTS auxiliary inventory')
    _same(pts, aux['aux__native_pts'], 'time-basis native PTS and complete accepted parent')
    controls = np.concatenate([np.asarray(recipe['control_slots'][kind], int) for kind in ('pan', 'tilt', 'logf')])
    other = np.setdiff1d(np.arange(7700), controls)
    _same(initial[other], parent[other], 'all non-control complete parent coordinates before T')
    policy = read(manifest['constant_principal_policy'])
    if (policy['complete_saved_initializer'] != manifest['saved_camera_checkpoint']
            or policy['canonical_q_sha256'] != sha(parent)
            or policy['fixed_zero_parameter_indices'] != [5168, 5169, 7675, 7676]):
        raise ValueError('Explicit current-parent constant-principal policy changed')
    direction = read(manifest['native_time_direction_configuration'])
    frames = np.asarray(direction['frame_indices_0based'])
    if (frames.dtype.kind not in 'iu' or len(frames) == 0 or len(np.unique(frames)) != len(frames)
            or np.any(frames < 0) or np.any(frames >= 1248)
            or any(not np.isfinite(float(direction['central_steps'][kind]))
                   or float(direction['central_steps'][kind]) <= 0 for kind in ('pan', 'tilt', 'logf'))):
        raise ValueError('Actual native source-direction QA configuration is invalid')
    policy = manifest['execution_numerical_initialization']
    import z190_joint_source_solver as solver
    defaults = inspect.signature(solver.solve_joint_source).parameters
    if (policy['basis'] != 'fresh_new_model_public_solver_defaults'
            or policy['initial_damping'] != defaults['initial_damping'].default
            or policy['initial_line_search_fraction'] != defaults['initial_line_search_fraction'].default
            or policy['solver_producer'] != reference(solver.__file__)
            or policy['parent_damping_retained_in_history_and_aux'] != damping
            or policy['parent_next_fraction_retained_in_history_and_aux'] != fraction
            or policy['parent_prediction_equivalence_not_claimed'] is not True):
        raise ValueError('Explicit new-model numerical initialization differs from frozen public defaults')
    result = dict(schema='rinkspace.native-time-files-pre-owner-preflight.v1', passed=True,
        canonical_parent_q_sha256=sha(parent), canonical_initial_q_sha256=sha(initial),
        all_complete_parent_aux_payload_byte_exact=True, complete_auxiliary_count=len(aux),
        basis_shape=list(basis.shape), literal_recipe_contract=contract,
        fixed_parameter_count=3784+3*(1248-basis.shape[1]), shared_count=172,
        parent_damping=damping, parent_fraction=fraction,
        execution_numerical_initialization=policy, old_r_J_loaded=False,
        source_owner_or_pool_constructed=False, actual_source_direction_QA_completed=False)
    json.dumps(result, allow_nan=False)
    return result


def _process_identity(value):
    """CIM creation time, executable name and command line identify an instance."""
    pid = int(value['ProcessId'])
    created = datetime.fromisoformat(value['CreationDate'].replace('Z', '+00:00'))
    if (pid <= 0 or created.tzinfo is None or not isinstance(value['Name'], str)
            or not value['Name'] or not isinstance(value['CommandLine'], str)
            or not value['CommandLine']):
        raise ValueError('Complete timezone-qualified process instance identity required')
    return dict(ProcessId=pid, CreationDate=created.astimezone(timezone.utc).isoformat(),
                Name=value['Name'].casefold(), CommandLine=value['CommandLine'])


def _current_windows_processes(pids):
    """Read only the recorded PIDs; do not terminate or attach to any process."""
    if os.name != 'nt':
        raise ValueError('This programme entry requires the actual Windows process domain')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    native = ctypes.WinDLL('ntdll')
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel.GetProcessTimes.argtypes = [ctypes.c_void_p, *([ctypes.POINTER(ctypes.c_uint64)]*4)]
    kernel.QueryFullProcessImageNameW.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                               ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    native.NtQueryInformationProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
                                               ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    native.NtQueryInformationProcess.restype = ctypes.c_long
    class UnicodeString(ctypes.Structure):
        _fields_ = [('Length', ctypes.c_ushort), ('MaximumLength', ctypes.c_ushort),
                    ('Buffer', ctypes.c_void_p)]
    epoch = datetime(1601, 1, 1, tzinfo=timezone.utc)
    rows = []
    for pid in pids:
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            if ctypes.get_last_error() == 87:
                continue
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            code = ctypes.c_uint32()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            if code.value != 259:
                continue
            times = [ctypes.c_uint64() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
                raise ctypes.WinError(ctypes.get_last_error())
            length = ctypes.c_uint32(32768); image = ctypes.create_unicode_buffer(length.value)
            if not kernel.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(length)):
                raise ctypes.WinError(ctypes.get_last_error())
            size = ctypes.c_uint32()
            native.NtQueryInformationProcess(handle, 60, None, 0, ctypes.byref(size))
            if size.value < ctypes.sizeof(UnicodeString):
                raise OSError('Cannot read recorded PID command line size: '+str(pid))
            buffer = ctypes.create_string_buffer(size.value)
            status = native.NtQueryInformationProcess(handle, 60, buffer, len(buffer), ctypes.byref(size))
            if status < 0:
                raise OSError('Cannot read recorded PID command line: '+str((pid, hex(status & 0xffffffff))))
            command = UnicodeString.from_buffer(buffer)
            command_line = ctypes.wstring_at(command.Buffer, command.Length//2)
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            if code.value == 259:
                rows.append(dict(ProcessId=int(pid), Name=Path(image.value).name, CommandLine=command_line,
                    CreationDate=(epoch+timedelta(microseconds=times[0].value//10)).isoformat()))
        finally:
            kernel.CloseHandle(handle)
    return rows


def require_stopped_parent(manifest):
    """Reject the recorded live instances, allowing demonstrably recycled PIDs."""
    stopped = read(manifest['previous_owner_shutdown'])
    result = read(manifest['previous_completed_branch'])
    if (stopped.get('passed') is not True or stopped.get('all_owner_and_workers_exited') is not True
            or stopped.get('completed_branch') != manifest['previous_completed_branch']
            or result.get('q_sha256') != manifest['saved_camera_q_sha256']):
        raise ValueError('Exact completed-parent and owner/pool shutdown receipt required')
    pids = [int(stopped['owner_pid']), *map(int, stopped['worker_pids'])]
    if len(set(pids)) != len(pids) or any(pid <= 0 for pid in pids):
        raise ValueError('Distinct actual parent owner and worker PIDs required')
    inventory = read(stopped['process_inventory'])
    recorded = [_process_identity(row) for row in inventory['processes']]
    by_pid = {row['ProcessId']:row for row in recorded}
    if (inventory.get('owner_pid') != stopped['owner_pid'] or len(by_pid) != len(recorded)
            or set(by_pid) != set(pids)):
        raise ValueError('Authenticated historical owner/worker instance inventory differs from shutdown')
    current = [_process_identity(row) for row in _current_windows_processes(pids)]
    if len({row['ProcessId'] for row in current}) != len(current):
        raise ValueError('Current Windows instance inventory repeats a PID')
    reused = []
    for row in current:
        pid = row['ProcessId']
        if pid not in by_pid:
            raise ValueError('Current Windows query returned an unrequested PID')
        old = by_pid[pid]
        if row['CreationDate'] == old['CreationDate']:
            if row != old:
                raise ValueError('Parent creation time matches but process identity is inconsistent: '+str(pid))
            raise ValueError('Recorded parent owner/worker instance is still active: '+str(pid))
        reused.append(dict(pid=pid, historical_instance=old, current_instance=row))
    return dict(recorded_parent_processes_rechecked_exited=pids,
                shutdown_receipt=manifest['previous_owner_shutdown'],
                authenticated_process_inventory=stopped['process_inventory'],
                process_identity_fields=['ProcessId', 'CreationDate', 'Name', 'CommandLine'],
                current_PID_reuse_not_parent_instances=reused,
                checked_at_utc=datetime.now(timezone.utc).isoformat(), no_process_terminated=True)


def restore_review_inputs(base_owner, manifest, manifest_ref, output):
    """Authenticate the unchanged source and complete parent BEFORE applying T."""
    initial = read(manifest['inherited_complete_source_QA'])
    proof = read(manifest['inherited_current_source_QA'])
    old = read(initial['manifest'])
    for key in ('complete_source_snapshot', 'fixed_head_model', 'feasible_complete_initializer',
                'feasible_initializer_proof', 'static_material_admission', 'derived_recipe',
                'accepted_parent_report', 'physical_camera_initializer'):
        if manifest[key] != old[key]:
            raise ValueError('Authenticated source/model input changed: '+key)
    for name in ('actual_new_model_trial_raw_full_shadow_verified',
                 'new_full_source_parent_trial_parent_exact',
                 'all_current_active_world_canonical_effective_byte_exact',
                 'all15_parent_aux_history_byte_exact'):
        if proof.get(name) is not True:
            raise ValueError('Incomplete inherited producer/source equivalence: '+name)
    accepted = read(manifest['saved_camera_report'])
    if (initial.get('passed') is not True or proof.get('passed') is not True
            or accepted['kind'] != manifest['saved_camera_parent_kind']
            or accepted['qaug_sha256'] != sha(base_owner.start)
            or initial['source_rows'] != len(base_owner.scales)
            or initial['programme_rows'] != len(base_owner.programme.scales)
            or initial['wide_rows'] != len(base_owner.factor.scales)):
        raise ValueError('Complete source and actual saved parent authentication failed')
    media = initial['media_manifest']; media_data = read(media)
    inventory_ref = media_data['source_inventory']
    if reference(inventory_ref['path']) != inventory_ref:
        raise ValueError('Authenticated immutable source inventory changed')
    with np.load(inventory_ref['path'], allow_pickle=False) as saved:
        inventory = {key:saved[key].copy() for key in saved.files}
    (output/'immutable_programme_source_inventory.npz').write_bytes(Path(inventory_ref['path']).read_bytes())
    write(output/'inherited_source_authentication.json', dict(
        schema='rinkspace.pre-time-map-complete-parent-source-authentication.v1', passed=True,
        manifest=manifest_ref, canonical_parent_q_sha256=sha(base_owner.start),
        complete_parent_checkpoint=manifest['saved_camera_checkpoint'],
        complete_parent_report=manifest['saved_camera_report'],
        inherited_original_source_QA=manifest['inherited_current_source_QA'],
        source_snapshot=manifest['complete_source_snapshot'], all_parent_aux_preserved=True,
        old_r_J_or_cost_loaded_for_new_objective=False,
        source_inventory_and_producer_raw_equivalence_inherited_without_full_replay=True,
        time_initializer_predictions_and_cost_not_claimed_equivalent=True))
    return inventory, media


def make_time_stage(base_owner, manifest):
    from calibration.program_tracking import first_shot_native_time_stage as module
    if reference(module.__file__) != manifest['native_time_stage']:
        raise ValueError('Explicit native-time producer changed')
    if read(manifest['native_time_protocol_QA']).get('passed') is not True:
        raise ValueError('Passed native-time protocol QA required')
    recipe = read(manifest['native_time_recipe'])
    with np.load(manifest['native_time_basis_arrays']['path'], allow_pickle=False) as saved:
        basis = csr_matrix((saved['data'], saved['indices'], saved['indptr']), shape=tuple(saved['shape']))
        pts = saved['native_pts'].copy()
    with np.load(manifest['native_time_initializer']['path'], allow_pickle=False) as saved:
        initial = saved['q'].copy()
        _same(saved['parent_q'], base_owner.start, 'complete parent of time-control initializer')
    owner = NativeTimeVenueStage(base_owner, basis=basis, native_pts=pts,
        control_slots=recipe['control_slots'], canonical_start=initial,
        recipe_reference=manifest['native_time_recipe'], entry_reference=reference(__file__))
    if len(owner.fixed) != 3784+3*(1248-basis.shape[1]):
        raise ValueError('Dynamic native-time fixed-column count changed')
    return owner


def verify_execution_numerical_state(owner, numerical, resume):
    """Check the actual solver arguments and archival arrays after all wrappers."""
    kwargs = owner.solver_kwargs()
    for name in ('initial_damping', 'initial_line_search_fraction'):
        if kwargs.get(name) != numerical[name]:
            raise ValueError('Actual wrapped native-time solver numerical initialization changed: '+name)
    arrays = owner.checkpoint_arrays()
    for key, value in resume['aux'].items():
        _same(arrays['historical_saved_camera__'+key], value, 'archived complete pre-time parent aux: '+key)
    expected = dict(native_time_parent_damping=resume['damping'],
        native_time_parent_next_line_search_fraction=resume['fraction'],
        native_time_execution_initial_damping=numerical['initial_damping'],
        native_time_execution_initial_line_search_fraction=numerical['initial_line_search_fraction'])
    for key, value in expected.items():
        _same(arrays[key], np.asarray(value), 'actual saved parent and new-model numerical state: '+key)
    return dict(actual_wrapped_solver_numerical_arguments_verified=True,
                all_saved_parent_aux_archived_byte_exact=True, archived_parent_auxiliary_count=len(resume['aux']),
                numerical_state_arrays={key:float(value) for key,value in expected.items()})


def verify_native_control_directions(owner, configuration_reference, output):
    """Actual revised UV/query/W and RS derivative directions, not a full world J."""
    from native_transverse_bundle import source_chunk
    from calibration.program_tracking.first_shot_row_source import evaluate_row_timed_frame
    configuration = read(configuration_reference)
    frames = np.asarray(configuration['frame_indices_0based'], np.int64)
    if (len(frames) == 0 or len(np.unique(frames)) != len(frames)
            or np.any(frames < 0) or np.any(frames >= 1248)):
        raise ValueError('Explicit distinct actual native frames required for source direction QA')
    q = owner.start; qe, E = owner.effective_q_and_chain(q); programme = owner.programme
    worlds, _ = programme.world(qe[:owner.programme_count], False)
    count = len(programme.state.world_parameter_indices)
    zero_D = {key:np.zeros(value.shape+(count,)) for key,value in worlds.items()}
    task = (qe[:owner.programme_count], worlds, zero_D, frames.tolist(), sha(qe[:owner.programme_count]))
    residual, J, worker = next(programme._pool().map(source_chunk, [task], chunksize=1))
    direct = [evaluate_row_timed_frame(programme.state, qe[:owner.programme_count],
        programme.bundle.frames[int(i)], programme.bundle.whitening[int(i)], worlds, None)[0] for i in frames]
    _same(residual, np.concatenate(direct), 'actual source worker and row-timed UV/query/W residual')
    pulled = (hstack((J, csr_matrix((len(residual), 13))), format='csr') @ E).tocsr()
    if pulled[:, owner.fixed_native].nnz:
        raise ValueError('Newly fixed native-history columns received source derivatives')
    checks = []
    for kind in ('pan', 'tilt', 'logf'):
        slots = owner.control_slots[kind]
        direction = np.zeros(len(q)); direction[slots] = np.sin(np.arange(len(slots))+1.)
        step = float(configuration['central_steps'][kind])
        if not np.isfinite(step) or step <= 0:
            raise ValueError('Explicit positive source-direction FD step required')
        values = []
        for sign in (-1., 1.):
            sample = q + sign*step*direction; physical, _ = owner.effective_q_and_chain(sample)
            local = []
            for i in frames:
                value = evaluate_row_timed_frame(programme.state, physical[:owner.programme_count],
                    programme.bundle.frames[int(i)], programme.bundle.whitening[int(i)], worlds, None)
                local.append(value[0])
            values.append(np.concatenate(local))
        numeric = (values[1]-values[0])/(2*step); analytic = np.asarray(pulled@direction).ravel()
        maximum = float(np.max(np.abs(numeric-analytic)))
        relative = float(np.linalg.norm(numeric-analytic)/max(np.linalg.norm(numeric), 1e-30))
        if not np.any(analytic) or (relative > 1e-4 and maximum > 1e-5):
            raise ValueError('Actual native-time source/RS direction failed: '+str((kind, relative, maximum)))
        checks.append(dict(kind=kind, central_step=step, relative_L2=relative,
                           maximum_absolute=maximum, actual_source_direction_L2=float(np.linalg.norm(analytic))))
    receipt = dict(schema='rinkspace.actual-source-native-time-RS-control-directions.v1', passed=True,
        canonical_q_sha256=sha(q), physical_q_sha256=sha(qe), configuration=configuration_reference,
        native_time_recipe=owner.native_time_receipt['recipe'], native_frame_indices_0based=frames.tolist(),
        actual_source_worker=worker, directions=checks, all_three_PTZoom_control_types_checked=True,
        original_UV_queries_sigma_and_complete_W_used=True,
        world_derivatives_explicitly_zero_only_in_this_local_camera_QA=True,
        complete_world_J_cache_not_populated_or_claimed=True,
        full_programme_control_rank_or_physical_accuracy_not_claimed=True)
    write(output, receipt); return reference(output)


def main():
    global write
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage-manifest', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--geometry-workers', type=int, default=20)
    parser.add_argument('--source-workers', type=int, default=6); parser.add_argument('--serve', action='store_true')
    args = parser.parse_args(); manifest, ref = load_manifest(args.stage_manifest)
    files_preflight = preflight_time_files(manifest)
    stopped = require_stopped_parent(manifest)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    from calibration.program_tracking import first_shot_atomic_output as atomic
    write = atomic.install(expected_runtime_reference=manifest['writer_original_runtime'],
                           expected_solver_reference=manifest['writer_existing_retry_policy'])
    previous.write = write
    from native_pp_continuation import configure_native_runtime
    configure_native_runtime(stage='native-time-reviewed-source-fixed-head', native_threads=1)
    (out/'run_wrapper_executed_snapshot.py').write_bytes(Path(__file__).read_bytes())
    programme = owner = None
    try:
        write(out/'entry_files_preflight.json', files_preflight)
        write(out/'status.json', dict(stage='restoring_authenticated_source_metadata', pid=os.getpid(), fit_started=False))
        parent = read_complete_parent(manifest, pool_transition=Z/'geometry_pool_reuse')
        if sha(parent['q']) != manifest['accepted_parent_q_sha256']:
            raise ValueError('Original complete source reconstruction parent changed')
        from calibration.program_tracking import first_shot_value_execution as value, first_shot_raw_execution as raw
        from calibration.program_tracking import first_shot_fused_execution as fused, row_scene_projection_fast
        parent_installation = value.install(); row_scene_projection_fast.install()
        programme, restored = restore_programme_owner(manifest['complete_source_snapshot'], parent,
            model_reference=manifest['previous_model'], association_reference=manifest['association'],
            geometry_workers=args.geometry_workers, source_workers=args.source_workers, entry_reference=reference(__file__))
        write(out/'source_restore.json', restored)
        old_chart = DerivedInactiveChart.from_recipe(programme.state, manifest['derived_recipe']['path'],
            parameter_count=len(parent['q']), release_rows=manifest['release_rows'],
            admission_dependencies=manifest['admission_dependencies'])
        model = load_model(manifest['fixed_head_model']); state = model.TransverseNativePrincipalState(programme.state)
        with np.load(manifest['feasible_complete_initializer']['path'], allow_pickle=False) as saved:
            original = saved['q'].copy()
        resume = previous.read_saved_camera_parent(manifest, original, state,
            fixed_chart_columns=old_chart.fixed_canonical_rows)
        start = resume['q'].copy()
        numerical = manifest['execution_numerical_initialization']
        programme.seed_receipt.update(initial_damping=numerical['initial_damping'],
            initial_line_search_fraction=numerical['initial_line_search_fraction'],
            accepted_complete_camera_parent=manifest['saved_camera_checkpoint'],
            inherited_parent_damping=resume['damping'], inherited_parent_fraction=resume['fraction'],
            explicit_new_model_numerical_initialization=numerical)
        from calibration.program_tracking.first_shot_transverse_execution import install_geometry_context
        install_geometry_context(programme); raw_proof = raw.install(programme)
        inherited = read(manifest['inherited_current_source_QA'])
        if (inherited.get('actual_new_model_trial_raw_full_shadow_verified') is not True
                or manifest['raw_retention_producer_authentication'] != raw_proof['producer']
                or inherited.get('source_points') != 191068):
            raise ValueError('Original source raw-retention producer equivalence not authenticated')
        fused.install_worker_execution(programme, parent_installation=parent_installation, entry_reference=reference(__file__),
            validation_references=(reference(Z/'value_real_state_QA/report.json'), reference(Z/'pivot_step_QA/report.json')))
        from functools import partial
        import native_transverse_bundle as native_runtime
        from calibration.program_tracking.first_shot_camera_initialization import _install_execution
        cache_execution = dict(producer=manifest['installation_cache_producer'],
            QA=manifest['installation_cache_actual_source_QA'], expected_math_reference=manifest['fixed_head_math'],
            function='install_execution')
        native_runtime.init_worker = partial(atomic.cached_worker_init, execution=cache_execution,
                                           original_initializer=native_runtime.init_worker)
        programme.source_producers.extend([manifest['atomic_output_adapter'], manifest['installation_cache_producer'],
                                          manifest['installation_cache_actual_source_QA'], manifest['fixed_head_math']])
        camera = install_camera(programme, model_reference=manifest['fixed_head_model'], start=start[:state.count],
            full_parent_aux=parent['aux'], entry_reference=reference(__file__), transition_reference=ref)
        _install_execution(cache_execution, type(programme.state.fixed_head))
        raw_proof.update(verified=True, verified_native_points=191068,
            first_actual_hit_full_original_raw_bitwise_verification_required=False,
            inherited_producer_verification=manifest['inherited_current_source_QA'], old_raw_values_not_loaded=True,
            current_source_cost_raw_still_computed_fresh=True,
            original_same_implementation_and_source_inventory_equivalence_not_replayed=True)
        write(out/'camera_transition.json', camera)
        admission = read(manifest['static_material_admission']); packet_ref = admission['source_packet']
        packet = VerifiedStaticMaterialPacket(packet_ref['path'], expected_sha256=packet_ref['sha256'],
            admitted_control_ids=admission['admitted_control_ids'], reviewed_admission=manifest['static_material_admission'])
        template = parent['identity']['persistent_static_venue_stage']['static_venue_factor']['camera']['template_camera25']
        factor = StaticVenueFactor(packet, StaticCameraPolicy('brown13', template), **previous.base.factor_options(programme))
        physical = StaticVenueStage(programme, factor, parent_reference=manifest['accepted_parent_report'],
            entry_reference=reference(__file__),
            source_references=[v for v in manifest.values() if isinstance(v, dict) and {'path', 'sha256'} <= set(v)])
        chart = expand_derived_chart(old_chart, programme.state, parameter_count=len(start), camera_start=programme.start)
        physical_parent, _ = chart.effective_q_and_chain(start)
        _same(physical_parent, resume['physical_q'], 'complete actual parent physical q before T')
        dependencies = sorted({int(i) for row in packet.ledger['reviewed_material_extension']['bindings']
                               for i in row['geometry_dependencies']})
        chart.assert_source_admission(dependencies)
        aux = physical.checkpoint_arrays()
        aux['static_venue_initial_camera_state'] = parent['aux']['static_venue_initial_camera_state'].copy()
        aux.update({'historical_saved_camera__'+key:value.copy() for key,value in resume['aux'].items()})
        aux.update(native_time_parent_damping=np.asarray(resume['damping']),
            native_time_parent_next_line_search_fraction=np.asarray(resume['fraction']),
            native_time_execution_initial_damping=np.asarray(numerical['initial_damping']),
            native_time_execution_initial_line_search_fraction=np.asarray(numerical['initial_line_search_fraction']))
        base_owner = DerivedVenueStage(physical, chart, canonical_start=start, inherited_aux=aux,
            entry_reference=reference(__file__), transition_reference=ref,
            producer_references=(reference(out/'source_restore.json'), reference(out/'camera_transition.json')))
        from calibration.program_tracking import first_shot_constant_principal as optical
        if reference(optical.__file__) != manifest['constant_principal_stage']:
            raise ValueError('Constant-principal producer changed')
        base_owner = optical.ConstantPrincipalVenueStage(base_owner, entry_reference=reference(__file__),
            optical_policy_reference=manifest['constant_principal_policy'])
        policy = read(manifest['constant_principal_policy'])
        if (policy['complete_saved_initializer'] != manifest['saved_camera_checkpoint']
                or policy['canonical_q_sha256'] != sha(base_owner.start)
                or policy['fixed_zero_parameter_indices'] != [5168, 5169, 7675, 7676]):
            raise ValueError('Constant-principal policy does not bind the complete current parent')
        from calibration.program_tracking import first_shot_print_halfspaces as print_qp
        if reference(print_qp.__file__) != manifest['print_halfspace_adapter']:
            raise ValueError('Existing print-halfspace producer changed')
        print_receipt = print_qp.install(base_owner, entry_reference=reference(__file__), qa_reference=manifest['print_halfspace_QA'])
        write(out/'print_qp_transition.json', print_receipt)
        counts = dict(programme_source_points=sum(f['source_point_count'] for f in programme.bundle.frames),
            programme_formal_rows=len(programme.scales), static_group_count=len(packet.rows),
            static_formal_rows=len(factor.scales), complete_formal_rows=len(base_owner.scales),
            parameter_count=len(base_owner.start), global_shared_count=len(base_owner.shared))
        for key, value in counts.items():
            if value != manifest['expected_'+key]:
                raise ValueError('Authenticated complete source/model contract changed: '+key)
        inventory, media = restore_review_inputs(base_owner, manifest, ref, out)
        owner = make_time_stage(base_owner, manifest)
        numerical_QA = verify_execution_numerical_state(owner, numerical, resume)
        previous.metadata_preflight(owner, out/'metadata_preflight.json')
        write(out/'native_time_transition.json', dict(owner.native_time_receipt, parent_shutdown=stopped,
            base_parent_q_sha256=sha(base_owner.start), canonical_initial_q_sha256=sha(owner.start),
            parent_damping=resume['damping'], parent_fraction=resume['fraction'],
            execution_numerical_initialization=numerical,
            actual_execution_numerical_state_QA=numerical_QA,
            old_source_r_J_or_cost_reused=False, initializer_prediction_equivalence_not_claimed=True))
        actual_QA = verify_native_control_directions(owner, manifest['native_time_direction_configuration'],
                                                   out/'actual_native_time_source_QA.json')
        runner = JointStageRunner.__new__(JointStageRunner)
        runner.owner = owner; runner.output = out; runner.manifest_reference = ref
        runner.generation = 0; runner.records = []; runner.started = time.time()
        runner.cache = out/'source_J_cache'; runner.inventory = inventory; runner.media = media
        runner.review_export = lambda q, folder:export_effective_review(owner, q, folder,
                                                        physical_export=previous.base.previous.export_review)
        runner.prepare(complete_parent_checkpoint=manifest['saved_camera_checkpoint'],
            explicit_native_time_recipe=manifest['native_time_recipe'], actual_native_time_source_QA=actual_QA,
            initializer_predictions_changed_by_explicit_time_subspace=True,
            old_source_r_J_or_cost_loaded=False, inherited_producer_raw_equivalence_not_replayed=True,
            fresh_full_world171_and_time_control_J_required=True, initial_state_accuracy_accepted=False)
        if args.serve:
            runner.run(wait_for_control=False)
    except BaseException as error:
        from calibration.program_tracking.first_shot_safe_frame_export import _encode
        arrays = {}; report = _encode(getattr(error, 'report', None), arrays)
        if arrays: np.savez_compressed(out/'run_failure_arrays.npz', **arrays)
        write(out/'run_failure.json', dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
            actual_domain_or_source_report=report, previous_parent_and_failure_evidence_preserved=True))
        raise
    finally:
        if owner is not None: owner.close()
        elif programme is not None: programme.close()


if __name__ == '__main__':
    main()
