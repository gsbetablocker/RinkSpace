"""Explicit fixed-text seed successor of a completed grouped physical parent.

Uses the public grouped-parent modules. No fit can start without
the actual completed parent, current-model QA and full Windows process proof.
"""
from datetime import datetime
from pathlib import Path
import json, os, sys
import numpy as np

ROOT=Path('D:/OneDrive/codex_projs/RinkSpace')
PROGRAMME=ROOT/'events/ankara_2026/programmes/okada_mei_sp'
sys.path[:0]=[str(ROOT),str(PROGRAMME)]
import fit_time_detail_print as base
import fit_time_grouped_compat as legacy_dense
from calibration.program_tracking import first_shot_ad7_cont_runtime as chart
from calibration.program_tracking import first_shot_mount_scale_tangent as mount_scale
from calibration.program_tracking import first_shot_disjoint_spawn as spawn
from calibration.program_tracking.first_shot_source_transition import reference,read
from calibration.program_tracking.first_shot_source_snapshot import _frame_contract
from calibration.program_tracking.first_shot_time_successor import authenticate_references
from calibration.program_tracking.first_shot_time_detail_stage import NativeTimeDetailVenueStage

_load=base.load_configuration
_install=base.install_camera_and_material
_restore=base.restore_inputs
_stopped=base.require_stopped
_numerical=base.verify_numerical
_directions=base.verify_directions
FIT_TOKENS=legacy_dense.FIT_TOKENS+(Path(__file__).name.casefold(),)


def same(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():
        raise ValueError('Full completed parent changed: '+label)


def process_preflight(config,config_ref):
    rows=legacy_dense.windows_processes()
    own=next((r for r in rows if int(r['ProcessId'])==os.getpid()),None)
    if own is None:raise ValueError('Full Windows process domain must contain current owner')
    own_created=datetime.fromisoformat(own['CreationDate'].replace('Z','+00:00'))
    conflicts=[];unreadable=[];launchers=[]
    for row in rows:
        if not(row.get('Name')or'').casefold().startswith(('python','pypy')):continue
        command=(row.get('CommandLine')or'').replace('\\','/').casefold()
        if not command:unreadable.append(row);continue
        pid=int(row['ProcessId'])
        if pid==os.getpid() or not any(token in command for token in FIT_TOKENS):continue
        created=datetime.fromisoformat(row['CreationDate'].replace('Z','+00:00'))
        launcher=(pid==os.getppid()==int(own['ParentProcessId']) and created<=own_created
            and Path(row.get('ExecutablePath')or'').resolve()==(ROOT/'.venv/Scripts/python.exe').resolve()
            and all(str(path).replace('\\','/').casefold()in command
                    for path in(__file__,config_ref['path'],config['planned_output'])))
        (launchers if launcher else conflicts).append(row)
    if conflicts or unreadable:
        raise ValueError('Another fit or unreadable Python instance is visible: '+repr(dict(conflicts=conflicts,unreadable=unreadable)))
    return dict(passed=True,full_windows_process_domain=True,current_pid=os.getpid(),
        process_count=len(rows),permitted_direct_venv_parent=launchers,no_process_terminated=True)


def load_configuration(path):
    manifest,ref=_load(path)
    config=base._CONFIG
    authenticate_references(config)
    if (config.get('execution_entry')!=reference(__file__)
            or config.get('execution_composition_schema')!='rinkspace.completed-grouped-parent-text-seed.v1'
            or config.get('chart_runtime')!=reference(chart.__file__)
            or config.get('spawn_runtime')!=reference(spawn.__file__)
            or config.get('mount_scale_tangent_runtime')!=reference(mount_scale.__file__)
            or config.get('detail_selection_policy')!='restore_accepted_nonzero_B4_and_verify_current_source_support'
            or config.get('old_source_r_J_or_trial_results_loaded')is not False):
        raise ValueError('Exact completed-parent fixed-text successor required')
    mount_qa=read(config['mount_scale_tangent_QA'])
    if (mount_qa.get('adapter_matrix_vs_actual_QP_prototype_max_error')!=0.
            or mount_qa.get('lower_byte_exact_unchanged')is not True
            or mount_qa.get('all_nonmount_rows_byte_exact_unchanged')is not True):
        raise ValueError('Exact unchanged mount-domain scale tangent QA required')
    runtime=read(config['grouped_continuation_configuration'])
    if (runtime['producer']!=config['chart_runtime'] or runtime['entry']!=config['execution_entry']
            or runtime['completed_parent_binding']!=config['parent_owner_shutdown']
            or runtime['complete_parent_references']!=read(config['parent_owner_shutdown'])['complete_parent_references']):
        raise ValueError('Runtime must bind this exact completed parent and execution')
    qa=read(config['current_grouped_model_QA'])
    expected=dict(runtime=config['chart_runtime'],configuration=config['grouped_continuation_configuration'],
        parent_checkpoint=config['accepted_parent_checkpoint'],parent_q_sha256=config['parent_q_sha256'],
        parent_aux_count=len(base._PARENT['aux']),source_contract=read(config['parent_source_contract'])['full_frame_contract'])
    for key,value in expected.items():
        if qa.get(key)!=value:raise ValueError('Actual current grouped-parent protocol differs: '+key)
    for key in ('passed','all_parent_q_aux_expanded_physical_byte_exact','all1248_row_cameras_byte_exact',
                'complete_source_ID_UV_PTS_W_and_raw_byte_exact','old_2192_logo_source_worlds_byte_exact',
                'fresh_source_and_geometry_spawn_byte_exact','actual_physical_and_new15_guards_passed',
                'fixed5_source_columns_zero','current_world_and_source8_FD_passed'):
        if qa.get(key)is not True:raise ValueError('Actual full current model QA required: '+key)
    base._CONFIG['actual_grouped_parent_composition_preflight']=dict(passed=True,
        current_model_QA=config['current_grouped_model_QA'],parent_aux_count=len(base._PARENT['aux']),
        old_model_derivatives_reused=False,text_initialization_is_not_added_source=True)
    return manifest,ref


def require_stopped(manifest):
    current=process_preflight(base._CONFIG,base._CONFIG_REF)
    return dict(_stopped(manifest),current_grouped_parent_process_domain=current)


def install_camera_and_material(programme,**kwargs):
    result=_install(programme,**kwargs)
    result['spawn_transport']=spawn.install(programme,entry_reference=reference(__file__))
    return result


def restore_inputs(base_owner,manifest,manifest_ref,output):
    inventory,media=_restore(base_owner,manifest,manifest_ref,output)
    with np.load(base._CONFIG['parent_source_inventory']['path'],allow_pickle=False)as saved:
        if set(inventory)!=set(saved.files):raise ValueError('Complete source inventory keys changed')
        for key in saved.files:same(inventory[key],saved[key],'source inventory '+key)
    current=_frame_contract(base_owner.programme.bundle.frames)
    if current!=read(base._CONFIG['parent_source_contract'])['full_frame_contract']:
        raise ValueError('Source target/ID/UV/sigma/W/native clock contract changed')
    base.construction.write(Path(output)/'group_source_immutable.json',dict(passed=True,
        parent_source_inventory=base._CONFIG['parent_source_inventory'],full_frame_contract=current,
        all_source_ID_UV_W_and_native_clock_unchanged=True,C0_UV_and_complete_W_unchanged=True))
    return inventory,media


def text_initializer_predictions_changed(config):
    runtime=read(config['grouped_continuation_configuration'])
    proposal=read(runtime['continuation'])
    initialized=proposal['initializer'] is not None
    expected_mode=('restore_and_update_fixed_text_initializer' if initialized
                   else 'restore_existing_grouped_chart')
    if proposal['mode']!=expected_mode:
        raise ValueError('Text initializer mode and explicit reference differ')
    before=np.asarray(proposal['restored_current_seed8'],float)
    after=np.asarray(proposal['evaluation_seed8'],float)
    changed=before[3:].tobytes()!=after[3:].tobytes()
    if changed and not initialized:
        raise ValueError('Changed text seed requires an explicit initializer')
    return changed


def make_stage(base_owner,manifest):
    config=base._CONFIG
    runtime_ref=config['grouped_continuation_configuration']
    installed=chart.install_successor(base_owner,parent=base._PARENT,
        configuration_reference=runtime_ref,entry_reference=reference(__file__))
    owner=NativeTimeDetailVenueStage(base_owner,parent=base._PARENT,entry_reference=reference(__file__),
        transition_reference=base._CONFIG_REF,active_details=None,
        allowed_rebuilt_aux=['print_H8_QP_domain_identity_json'],live_model_reference=config['print_configuration'])
    fixed=chart.restore_fixed_columns(owner,parent=base._PARENT,configuration_reference=runtime_ref)
    mount_tangent=mount_scale.install(owner,entry_reference=reference(__file__))
    chart.install_solver_delivery(owner)
    selected,proof=base.camera_detail_support(owner)
    absent=np.setdiff1d(owner.active_details,selected)
    if len(absent):raise ValueError('Inherited B4 details lack current actual source support: '+repr(absent.tolist()))
    proof.update(accepted_active_details_restored=True,restored_active_detail_indices=owner.active_details.tolist(),
        current_model_source_support_fresh=True,previous_source_support_or_FD_reused=False,
        release_or_reselection_performed=False)
    owner.detail_support_receipt=proof
    owner.source_successor_receipt=dict(schema='rinkspace.completed-grouped-parent-text-seed.v1',
        parent=config['accepted_parent_checkpoint'],parent_q_sha256=base._PARENT['q_sha256'],
        grouped_continuation=installed,initialization_only_fixed=fixed,
        mount_scale_tangent=mount_tangent,
        all_prior_q_aux_lambda_fraction_retained=True,source_UV_W_PTS_and_C0_unchanged=True,
        old_r_J_or_cost_reused=False,text_predictions_changed_explicitly=text_initializer_predictions_changed(config),
        whole_rink_geometry_or_camera_confirmed=False)
    return owner


def verify_numerical(owner,numerical,historical_parent):
    result=_numerical(owner,numerical,historical_parent)
    physical,_=owner.effective_q_and_chain(owner.start)
    expanded,_=owner.expanded_q_and_chain(owner.start)
    same(physical,base._PARENT['physical_q'],'effective physical q')
    same(expanded,base._PARENT['expanded_q'],'expanded canonical q')
    if len(owner.shared)!=167 or not np.isin(np.arange(5162,5167),owner.fixed).all():
        raise ValueError('Five unsupported text columns must remain fixed')
    return dict(result,full_physical_and_expanded_parent_byte_exact=True,
        text_model_seed_change_is_explicit=True,current_model_QA=base._CONFIG['current_grouped_model_QA'])


def verify_directions(owner,historical_configuration,output):
    ref=_directions(owner,historical_configuration,output)
    value=read(ref)
    value.update(execution_entry=reference(__file__),
        grouped_continuation_configuration=base._CONFIG['grouped_continuation_configuration'],
        current_model_QA=base._CONFIG['current_grouped_model_QA'],
        fixed_initialization=owner.ad7_continuation_fixed_receipt,
        current_model_camera_FD_fresh=True,previous_source_support_or_FD_reused=False,
        all_formal_world_and_camera_J_fresh_required=True)
    base.construction.write(Path(output),value)
    return reference(output)


def owner_receipt(owner):
    return dict(source_model_successor=owner.source_successor_receipt,time_detail_model=owner.detail_receipt,
        current_source_support=owner.detail_support_receipt,
        fixed_initialization=owner.ad7_continuation_fixed_receipt)


def main():
    base.load_configuration=load_configuration;base.require_stopped=require_stopped
    base.install_camera_and_material=install_camera_and_material;base.restore_inputs=restore_inputs
    base.make_stage=make_stage;base.verify_numerical=verify_numerical
    base.verify_directions=verify_directions;base.owner_receipt=owner_receipt
    # Reuse the frozen value-preserving metadata bridge and original preflight.
    legacy_dense._METADATA_MODULE.metadata_preflight=legacy_dense.metadata_preflight
    base.main()


if __name__=='__main__':main()
