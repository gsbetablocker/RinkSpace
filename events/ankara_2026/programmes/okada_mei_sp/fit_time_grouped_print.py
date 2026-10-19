"""Complete accepted B4 continuation with explicit source-supported Ad7 groups."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import json,os,subprocess,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import fit_time_detail_print as base
from calibration.program_tracking import first_shot_ad7_shape_chart as chart
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
PWSh='C:/Users/REDACTED/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe'
FIT_TOKENS=('fit_fixed_head_time.py','fit_time_source.py','fit_time_cache.py',
    'fit_time_detail_print.py','fit_time_detail_spawn.py','fit_time_grouped_print.py')


def windows_processes():
    query='Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Depth 4 -Compress'
    rows=json.loads(subprocess.check_output([PWSh,'-NoProfile','-Command',query],text=True,encoding='utf-8'))
    rows=rows if isinstance(rows,list)else[rows]
    if not rows or any(not isinstance(row,dict)or'ProcessId'not in row for row in rows):
        raise ValueError('Actual full Windows process inventory unavailable')
    return rows


def current_process_preflight(config,config_ref):
    rows=windows_processes();own=next((r for r in rows if int(r['ProcessId'])==os.getpid()),None)
    if own is None:raise ValueError('Actual full Windows domain does not contain this process')
    own_created=datetime.fromisoformat(own['CreationDate'].replace('Z','+00:00'))
    matches=[];unreadable=[];launcher=[]
    for row in rows:
        pid=int(row['ProcessId']);command=(row.get('CommandLine')or'').replace('\\','/').casefold()
        if not(row.get('Name')or'').casefold().startswith(('python','pypy')):continue
        if not command:unreadable.append(row);continue
        if not any(token in command for token in FIT_TOKENS)or pid==os.getpid():continue
        created=datetime.fromisoformat(row['CreationDate'].replace('Z','+00:00'))
        parent=(pid==os.getppid()==int(own['ParentProcessId']) and created<=own_created
            and Path(row.get('ExecutablePath')or'').resolve()==(ROOT/'.venv/Scripts/python.exe').resolve()
            and all(str(path).replace('\\','/').casefold()in command for path in
                (__file__,config_ref['path'],config['planned_output'])))
        if parent:launcher.append(row)
        else:matches.append(row)
    if matches or unreadable:raise ValueError('Another formal or unreadable Python instance is visible: '+repr(dict(matches=matches,unreadable=unreadable)))
    return dict(passed=True,full_windows_process_domain=True,process_count=len(rows),
        current_pid=os.getpid(),permitted_direct_venv_parent=launcher,no_process_terminated=True)


def validate_composition(config,config_ref):
    authenticate_references(config)
    if(config.get('execution_entry')!=reference(__file__)or config.get('chart_runtime')!=reference(chart.__file__)
            or config.get('spawn_runtime')!=reference(spawn.__file__)
            or config.get('detail_selection_policy')!='restore_accepted_nonzero_B4_and_verify_current_source_support'
            or config.get('completed_same_model_source_support_and_FD_reused')is not False):
        raise ValueError('Exact explicit grouped physical successor with fresh current-model QA required')
    group=read(config['grouped_shape_configuration']);qa=read(config['group_spawn_protocol_QA']);numeric=read(config['group_protocol_QA'])
    for key,expected in(('chart_runtime',config['chart_runtime']),('configuration',config['grouped_shape_configuration']),
            ('spawn_runtime',config['spawn_runtime']),('print_runtime',config['print_runtime']),
            ('print_configuration',config['print_configuration']),('parent_checkpoint',config['accepted_parent_checkpoint']),
            ('canonical_q_sha256',config['parent_q_sha256'])):
        if qa.get(key)!=expected:raise ValueError('Final actual grouped spawn QA differs: '+key)
    if(qa.get('passed')is not True or qa.get('complete_final_source_inventory_exact')is not True
            or qa.get('parent_q_all400aux_and_expansion_byte_exact')is not True
            or qa.get('all1248_true_RS_full_W_values')is not True
            or qa.get('full_original_raw_and_cache_hit_byte_exact')is not True
            or qa.get('full_raw_points')!=182038 or qa.get('parent_aux_count')!=400
            or qa.get('parent_fraction')!=1.0 or qa.get('parent_damping')!=2.56e-10
            or qa['source_spawn'].get('r_and_CSR_bytes_exact')is not True
            or qa['geometry_spawn'].get('full_bank_paths')!=38
            or qa['geometry_spawn'].get('all_world_path_arrays_byte_exact')is not True
            or qa['QP_callback'].get('fixed_all_text_column_reduction_coefficients_exact')is not True
            or qa['QP_callback'].get('shared_after')!=167
            or qa['fixed_initialization'].get('global_slots')!=list(range(5162,5167))):
        raise ValueError('Complete actual source/RS/fullW/geometry/raw/five-fixed protocol required')
    if(numeric.get('passed')is not True or numeric.get('module')!=config['chart_runtime']
            or numeric.get('configuration')!=config['grouped_shape_configuration']
            or numeric.get('final_required_fixed_global_columns')!=list(range(5162,5167))
            or numeric.get('final_required_shared_count')!=167):
        raise ValueError('Final exact-module numerical/world/RS/fullW grouped protocol required')
    for key,maximum in(('all_38bank_world_directions',2e-6),('source_directions',2e-4)):
        rows=numeric[key]
        if set(int(row['column'])for row in rows)!=set(range(5159,5167)):
            raise ValueError('Full eight-slot numerical derivative inventory required: '+key)
        if any(not(row['maximum']<=maximum or row['relative']<=2e-5)for row in rows):
            raise ValueError('Original grouped derivative threshold failed: '+key)
    if(group['producer']!=config['chart_runtime']or group['parent_checkpoint']!=config['accepted_parent_checkpoint']
            or group['parent_q_sha256']!=config['parent_q_sha256']or group['parent_aux_count']!=400
            or group['fixed_initialization_global_slots']!=list(range(5162,5167))
            or group['source_support_reference']!=config['group_source_support']
            or group['assigned_source_points_by_group']!={'logo':2192,'top':0,'bottom':0}):
        raise ValueError('Final complete parent and authentic logo-only support must bind grouped seed')
    return dict(passed=True,group_configuration=config['grouped_shape_configuration'],group_protocol_QA=config['group_protocol_QA'],
        group_spawn_protocol_QA=config['group_spawn_protocol_QA'],old_J_or_model_FD_reused=False)


def load_configuration(path):
    manifest,ref=_load(path)
    base._CONFIG['actual_group_composition_preflight']=validate_composition(base._CONFIG,ref)
    return manifest,ref


def require_stopped(manifest):
    current=current_process_preflight(base._CONFIG,base._CONFIG_REF)
    return dict(_stopped(manifest),current_group_entry_process_domain=current)


def install_camera_and_material(programme,**kwargs):
    result=_install(programme,**kwargs)
    result['spawn_transport']=spawn.install(programme,entry_reference=reference(__file__))
    return result


def restore_inputs(base_owner,manifest,manifest_ref,output):
    inventory,media=_restore(base_owner,manifest,manifest_ref,output)
    with np.load(base._CONFIG['parent_source_inventory']['path'],allow_pickle=False)as saved:
        if set(inventory)!=set(saved.files):raise ValueError('Complete parent source inventory keys changed')
        for key in saved.files:
            a,b=np.asarray(inventory[key]),saved[key]
            if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():
                raise ValueError('Complete current source ID/UV/nativePTS/frame/target changed: '+key)
    current=_frame_contract(base_owner.programme.bundle.frames)
    if current!=read(base._CONFIG['group_spawn_protocol_QA'])['source_contract']:
        raise ValueError('Final complete UV/sigma/W/native clock source contract changed')
    base.construction.write(Path(output)/'group_source_immutable.json',dict(passed=True,
        parent_source_inventory=base._CONFIG['parent_source_inventory'],full_frame_contract=current,
        all_source_ID_UV_W_and_native_clock_unchanged=True,C0_UV_and_complete_W_unchanged=True))
    return inventory,media


def make_stage(base_owner,manifest):
    installation=chart.install_successor(base_owner,parent=base._PARENT,
        configuration_reference=base._CONFIG['grouped_shape_configuration'],entry_reference=reference(__file__))
    owner=NativeTimeDetailVenueStage(base_owner,parent=base._PARENT,entry_reference=reference(__file__),
        transition_reference=base._CONFIG_REF,active_details=None,
        allowed_rebuilt_aux=['print_H8_QP_domain_identity_json'],live_model_reference=base._CONFIG['print_configuration'])
    fixed=chart.finalize_stage(owner,configuration_reference=base._CONFIG['grouped_shape_configuration'])
    continuity=chart.verify_parent_continuation(owner,base._PARENT)
    # Restore the accepted B4 set, then recompute its current physical source
    # support. No zero-detail probe or old-model support/derivatives are loaded.
    selected,proof=base.camera_detail_support(owner)
    absent=np.setdiff1d(owner.active_details,selected)
    if len(absent):raise ValueError('Accepted active details lack current-model structural source support: '+repr(absent.tolist()))
    proof.update(accepted_active_details_restored=True,restored_active_detail_indices=owner.active_details.tolist(),
        current_model_source_support_fresh=True,previous_source_support_or_FD_reused=False,
        release_or_reselection_performed=False)
    owner.detail_support_receipt=proof
    owner.source_successor_receipt=dict(schema='rinkspace.complete-grouped-print-B4-successor.v1',
        parent=base._CONFIG['accepted_parent_checkpoint'],parent_q_sha256=base._PARENT['q_sha256'],
        parent_continuation=continuity,group_chart=installation,initialization_only_fixed=fixed,
        all_prior_q_aux_lambda_fraction_retained=True,source_UV_W_PTS_and_C0_unchanged=True,
        old_r_J_or_cost_reused=False,new_shape_predictions_not_parent_equivalent=True)
    return owner


def verify_numerical(owner,numerical,historical_parent):
    result=_numerical(owner,numerical,historical_parent)
    result['complete_nonzero_B4_parent_continuation']=chart.verify_parent_continuation(owner,base._PARENT)
    if len(owner.shared)!=167 or not np.isin(np.arange(5162,5167),owner.fixed).all():
        raise ValueError('All five unsupported text parameters must remain fixed')
    return result


def verify_directions(owner,historical_configuration,output):
    ref=_directions(owner,historical_configuration,output)
    result=read(ref);result.update(execution_entry=reference(__file__),group_configuration=base._CONFIG['grouped_shape_configuration'],
        group_protocol_QA=base._CONFIG['group_protocol_QA'],group_spawn_protocol_QA=base._CONFIG['group_spawn_protocol_QA'],
        fixed_initialization=owner.ad7_initialization_fixed_receipt,current_model_camera_FD_fresh=True,
        previous_source_support_or_FD_reused=False,all_formal_world_and_camera_J_fresh_required=True)
    base.construction.write(Path(output),result)
    return reference(output)


def owner_receipt(owner):
    return dict(source_model_successor=owner.source_successor_receipt,time_detail_model=owner.detail_receipt,
        current_source_support=owner.detail_support_receipt,fixed_initialization=owner.ad7_initialization_fixed_receipt)


def main():
    base.load_configuration=load_configuration;base.require_stopped=require_stopped
    base.install_camera_and_material=install_camera_and_material;base.restore_inputs=restore_inputs
    base.make_stage=make_stage;base.verify_numerical=verify_numerical
    base.verify_directions=verify_directions;base.owner_receipt=owner_receipt
    base.main()


if __name__=='__main__':main()
