"""Continue the complete physical shot; wide images are relationship references.

The frozen grouped factory restores the full accepted state. This explicit
composition fixes its historical independent wide camera and fits only the
complete official50 programme source, preserving the common adjustable venue.
"""
from datetime import datetime
from pathlib import Path
import json
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PROGRAMME = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(PROGRAMME)]
import fit_group_seed as seed
from calibration.program_tracking.first_shot_programme_objective import ProgrammeSourceObjective
from calibration.program_tracking.first_shot_programme_runner import programme_runner
from calibration.program_tracking import first_shot_initial_prefix as initial_prefix
from calibration.program_tracking.first_shot_source_transition import reference, read
from calibration.program_tracking.first_shot_time_successor import authenticate_references

base = seed.base
SCHEMA = 'rinkspace.programme-source-fixed-head-ad7-continuation.v1'
FIT_TOKENS = seed.FIT_TOKENS+(Path(__file__).name.casefold(),)


def load_configuration(path):
    manifest, ref = seed._load(path)
    config = base._CONFIG
    authenticate_references(config)
    if (config.get('execution_entry') != reference(__file__)
            or config.get('execution_composition_schema') != SCHEMA
            or config.get('mathematical_entry') != reference(seed.__file__)
            or config.get('chart_runtime') != reference(seed.chart.__file__)
            or config.get('spawn_runtime') != reference(seed.spawn.__file__)
            or config.get('mount_scale_tangent_runtime') != reference(seed.mount_scale.__file__)
            or config.get('programme_objective_runtime') != reference(sys.modules[ProgrammeSourceObjective.__module__].__file__)
            or config.get('programme_runner_runtime') != reference(sys.modules[programme_runner.__module__].__file__)
            or config.get('detail_selection_policy') != 'restore_accepted_nonzero_B4_and_verify_current_source_support'):
        raise ValueError('Explicit frozen physical factory and programme-source composition required')
    runtime = read(config['grouped_continuation_configuration'])
    if (runtime['producer'] != config['chart_runtime']
            or runtime['entry'] != config['mathematical_entry']
            or runtime['completed_parent_binding'] != config['parent_owner_shutdown']
            or runtime['complete_parent_references'] != read(config['parent_owner_shutdown'])['complete_parent_references']):
        raise ValueError('Same authenticated final parent and frozen grouped factory required')
    qa = read(config['current_grouped_model_QA'])
    expected = dict(runtime=config['chart_runtime'], configuration=config['grouped_continuation_configuration'],
        parent_checkpoint=config['accepted_parent_checkpoint'], parent_q_sha256=config['parent_q_sha256'],
        parent_aux_count=len(base._PARENT['aux']),
        source_contract=read(config['parent_source_contract'])['full_frame_contract'])
    for key, value in expected.items():
        if qa.get(key) != value:
            raise ValueError('Actual frozen current-model QA differs: '+key)
    for key in ('passed', 'all_parent_q_aux_expanded_physical_byte_exact', 'all1248_row_cameras_byte_exact',
                'complete_source_ID_UV_PTS_W_and_raw_byte_exact', 'old_2192_logo_source_worlds_byte_exact',
                'fresh_source_and_geometry_spawn_byte_exact', 'actual_physical_and_new15_guards_passed',
                'fixed5_source_columns_zero', 'current_world_and_source8_FD_passed'):
        if qa.get(key) is not True:
            raise ValueError('Actual complete physical/model check required: '+key)
    mount = read(config['mount_scale_tangent_QA'])
    if (mount.get('adapter_matrix_vs_actual_QP_prototype_max_error') != 0.
            or mount.get('lower_byte_exact_unchanged') is not True
            or mount.get('all_nonmount_rows_byte_exact_unchanged') is not True):
        raise ValueError('Existing mount lower/domain preserving scale correction required')
    objective = read(config['programme_objective_QA'])
    for key, value in dict(passed=True, producer_runtime=config['programme_objective_runtime'],
                          parent_q_sha256=config['parent_q_sha256'], programme_formal_rows=214133,
                          retired_static_pixel_rows=4278, shared_count=154,
                          all_parent_q_and_aux_preserved=True, retired_camera13_fixed_byte_exact=True,
                          programme_r_scales_quadratic_ledger_unchanged=True,
                          original_physical_constraint_rows_and_lower_retained=True,
                          solver_dense_delivery_equal_owner_CSR=True).items():
        if objective.get(key) != value:
            raise ValueError('Actual programme-only objective check differs: '+key)
    runner = read(config['programme_runner_QA'])
    if (runner.get('passed') is not True or runner.get('runtime') != config['programme_runner_runtime']
            or runner.get('actual_frozen_parent_methods_executed') is not True
            or runner.get('last_programme_block_not_mislabelled_wide') is not True
            or runner.get('checkpoint_report_index_and_return_hashes_match') is not True):
        raise ValueError('Programme scope/cost/cache/checkpoint reporting interface QA required')
    if config.get('initial_programme_prefix_configuration') is not None:
        if config.get('initial_programme_prefix_runtime') != reference(initial_prefix.__file__):
            raise ValueError('Explicit same-q source-prefix runtime required')
        prefix = read(config['initial_programme_prefix_QA'])
        expected_prefix = dict(passed=True, runtime=config['initial_programme_prefix_runtime'],
            runner_runtime=config['programme_runner_runtime'],
            configuration=config['initial_programme_prefix_configuration'],
            parent_q_sha256=config['parent_q_sha256'],
            fresh_actual_r_scales_masks_loss_ledger_all214133_byte_exact=True,
            all_six_original_CSR_references_verified_and_reused=True,
            old_H_gradient_robust_weights_cost_or_mount_callback_reused=False,
            source_cache_scope_programme214133_wide0=True,
            complete_q7700_original402aux_preserved=True,
            fresh_payload405aux_saved=True, different_q_dispatch_protocol_passed=True)
        for key, value in expected_prefix.items():
            if prefix.get(key) != value:
                raise ValueError('Actual same-q source-prefix delivery check differs: '+key)
    base._CONFIG['actual_programme_composition_preflight'] = dict(passed=True,
        mathematical_entry=config['mathematical_entry'], current_model_QA=config['current_grouped_model_QA'],
        programme_objective_QA=config['programme_objective_QA'], all_parent_state_preserved=True,
        wide_scope='relative rink-line relationships only; wide camera and pixel factor retired')
    return manifest, ref


def process_preflight(config, config_ref):
    rows = seed.legacy_dense.windows_processes()
    own = next((row for row in rows if int(row['ProcessId']) == os.getpid()), None)
    if own is None:
        raise ValueError('Actual full Windows process domain must include this process')
    created = datetime.fromisoformat(own['CreationDate'].replace('Z', '+00:00'))
    conflicts, unreadable, launchers = [], [], []
    for row in rows:
        if not (row.get('Name') or '').casefold().startswith(('python', 'pypy')):
            continue
        command = (row.get('CommandLine') or '').replace('\\', '/').casefold()
        if not command:
            unreadable.append(row)
            continue
        pid = int(row['ProcessId'])
        if pid == os.getpid() or not any(token in command for token in FIT_TOKENS):
            continue
        before = datetime.fromisoformat(row['CreationDate'].replace('Z', '+00:00'))
        launcher = (pid == os.getppid() == int(own['ParentProcessId']) and before <= created
            and Path(row.get('ExecutablePath') or '').resolve() == (ROOT/'.venv/Scripts/python.exe').resolve()
            and all(str(value).replace('\\', '/').casefold() in command
                    for value in (__file__, config_ref['path'], config['planned_output'])))
        (launchers if launcher else conflicts).append(row)
    if conflicts or unreadable:
        raise ValueError('Conflicting fit or unreadable Python instances: '+repr(dict(conflicts=conflicts, unreadable=unreadable)))
    return dict(passed=True, full_windows_process_domain=True, current_pid=os.getpid(),
                permitted_direct_venv_parent=launchers, no_process_terminated=True)


def make_stage(base_owner, manifest):
    inner = seed.make_stage(base_owner, manifest)
    owner = ProgrammeSourceObjective(inner, entry_reference=reference(__file__), transition_reference=base._CONFIG_REF)
    config = base._CONFIG.get('initial_programme_prefix_configuration')
    return (initial_prefix.install(owner, parent=base._PARENT,
        configuration_reference=config, entry_reference=reference(__file__))
        if config is not None else owner)


def programme_owner(owner):
    return owner.base if isinstance(owner, initial_prefix.InitialProgrammePrefix) else owner


def verify_numerical(owner, numerical, historical_parent):
    result = seed.verify_numerical(programme_owner(owner).base, numerical, historical_parent)
    seed.same(owner.start, base._PARENT['q'], 'all final programme canonical coordinates')
    for key, value in base._PARENT['aux'].items():
        seed.same(owner.checkpoint_arrays()[key], value, 'complete historical auxiliary '+key)
    if (len(owner.shared) != 154 or len(owner.scales) != 214133
            or len(owner.retired_camera_columns) != 13
            or not np.isin(owner.retired_camera_columns, owner.fixed).all()):
        raise ValueError('Exact complete programme-only source chart required')
    kwargs = owner.solver_kwargs()
    if (kwargs['initial_damping'] != base._PARENT['damping']
            or kwargs['initial_line_search_fraction'] != base._PARENT['fraction']):
        raise ValueError('Actual terminal numerical search state changed')
    return dict(result, programme_objective=owner.receipt, current_model_factory_unchanged=True)


def verify_directions(owner, historical_configuration, output):
    # Same camera/venue map and same complete programme prefix. The independent
    # retired factor has zero camera-time derivatives in these direction tests.
    inner_ref = seed.verify_directions(programme_owner(owner).base, historical_configuration, Path(output).with_name('physical_factory_directions.json'))
    base.construction.write(Path(output), dict(passed=True, execution_entry=reference(__file__),
        frozen_factory_direction_QA=inner_ref, programme_objective_QA=base._CONFIG['programme_objective_QA'],
        programme_prefix_and_all_native_camera_maps_unchanged=True,
        no_wide_camera_or_pixel_fit=True, whole_shot_accuracy_accepted=False))
    return reference(output)


def owner_receipt(owner):
    return dict(seed.owner_receipt(programme_owner(owner).base), programme_source_objective=owner.receipt,
                initial_programme_prefix=getattr(owner, 'prefix_receipt', None))


def main():
    base.load_configuration = load_configuration
    seed.process_preflight = process_preflight
    seed.FIT_TOKENS = FIT_TOKENS
    base.require_stopped = seed.require_stopped
    base.install_camera_and_material = seed.install_camera_and_material
    base.restore_inputs = seed.restore_inputs
    base.make_stage = make_stage
    base.verify_numerical = verify_numerical
    base.verify_directions = verify_directions
    base.owner_receipt = owner_receipt
    seed.legacy_dense._METADATA_MODULE.metadata_preflight = seed.legacy_dense.metadata_preflight
    base.DetailPrintRunner = programme_runner(base.DetailPrintRunner)
    base.main()


if __name__ == '__main__':
    main()
