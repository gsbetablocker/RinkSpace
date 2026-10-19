"""Same frozen grouped model; adapt only two legacy dense-type boundaries."""
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from types import MethodType
import json,os,sys
import numpy as np
from scipy.sparse import issparse
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import fit_time_grouped_print as group
from calibration.program_tracking.first_shot_source_transition import reference,read
from calibration.program_tracking.first_shot_time_successor import authenticate_references

windows_processes=group.windows_processes
PWSh=group.PWSh
_VALIDATE=group.validate_composition
_MAKE=group.make_stage
_DIRECTIONS=group.verify_directions
_METADATA_MODULE=group.base.construction.previous
_METADATA_ORIGINAL=_METADATA_MODULE.metadata_preflight
FIT_TOKENS=group.FIT_TOKENS+(Path(__file__).name.casefold(),)


def dense_record(record):
    """A value-preserving type bridge; lower/report/other objects are untouched."""
    matrix=record['matrix']
    return dict(record,matrix=matrix.toarray())if issparse(matrix)else record


def dense_constraints(q,selected,*,original):
    return dense_record(original(q,selected))


def install_solver_delivery(owner):
    original=owner.solver_kwargs
    def kwargs(this):
        result=original();callback=result.get('shared_linear_constraints')
        if not callable(callback):raise ValueError('Existing shared-domain callback required')
        result['shared_linear_constraints']=partial(dense_constraints,original=callback)
        identity=deepcopy(result.get('checkpoint_identity',{}))
        identity['legacy_dense_constraint_delivery']=dict(entry=reference(__file__),
            owner_domain_callback_unchanged=True,solver_boundary_dense_only=True,
            same_all_halfspaces_and_lower_values=True,source_J_guard_and_thresholds_unchanged=True)
        result['checkpoint_identity']=identity
        return result
    owner.solver_kwargs=MethodType(kwargs,owner)


def metadata_preflight(owner,output):
    # The original preflight is invoked intact. Its only incompatible read is
    # the direct np.asarray(owner.linear_constraints(...)[matrix], float).
    existed='linear_constraints'in owner.__dict__
    instance=owner.__dict__.get('linear_constraints');original=owner.linear_constraints
    converted=[]
    def metadata_constraints(this,q,selected):
        record=original(q,selected)
        if issparse(record['matrix']):converted.append(dict(shape=list(record['matrix'].shape),format=record['matrix'].format))
        return dense_record(record)
    owner.linear_constraints=MethodType(metadata_constraints,owner)
    try:
        result=_METADATA_ORIGINAL(owner,output)
    finally:
        if existed:owner.linear_constraints=instance
        else:del owner.linear_constraints
    if existed and owner.__dict__.get('linear_constraints')is not instance:
        raise RuntimeError('Exact original instance constraint method was not restored')
    group.base.construction.write(Path(output).with_name('metadata_sparse_compatibility.json'),dict(
        passed=True,entry=reference(__file__),original_metadata_preflight=reference(_METADATA_MODULE.__file__),
        original_preflight_result=result,temporary_dense_conversions=converted,
        exact_original_constraint_attribute_restored=True,solver_owner_domain_still_original=True,
        all_original_checks_and_thresholds_called_without_copy=True))
    return result


def current_process_preflight(config,config_ref):
    rows=group.windows_processes();own=next((r for r in rows if int(r['ProcessId'])==os.getpid()),None)
    if own is None:raise ValueError('Actual complete Windows process domain must contain this process')
    own_created=datetime.fromisoformat(own['CreationDate'].replace('Z','+00:00'))
    matches=[];unreadable=[];launcher=[]
    for row in rows:
        pid=int(row['ProcessId']);command=(row.get('CommandLine')or'').replace('\\','/').casefold()
        if not(row.get('Name')or'').casefold().startswith(('python','pypy')):continue
        if not command:unreadable.append(row);continue
        if not any(token in command for token in FIT_TOKENS)or pid==os.getpid():continue
        created=datetime.fromisoformat(row['CreationDate'].replace('Z','+00:00'))
        parent=(pid==os.getppid()==int(own['ParentProcessId'])and created<=own_created
            and Path(row.get('ExecutablePath')or'').resolve()==(ROOT/'.venv/Scripts/python.exe').resolve()
            and all(str(path).replace('\\','/').casefold()in command for path in(__file__,config_ref['path'],config['planned_output'])))
        if parent:launcher.append(row)
        else:matches.append(row)
    if matches or unreadable:raise ValueError('Formal owner or unreadable Python instance visible: '+repr(dict(matches=matches,unreadable=unreadable)))
    return dict(passed=True,full_windows_process_domain=True,process_count=len(rows),
        current_pid=os.getpid(),permitted_direct_venv_parent=launcher,no_process_terminated=True)


def validate_failed_exit(config):
    proof=read(config['preceding_failure_shutdown']);failure=read(proof['failure'])
    if(proof.get('passed')is not True or proof.get('full_windows_process_domain')is not True
            or proof.get('all_observed_owner_launcher_and_workers_absent')is not True
            or proof.get('no_new_iteration_or_acceptance')is not True or proof.get('no_process_terminated')is not True
            or proof.get('complete_numerical_parent_binding')!=config['parent_owner_shutdown']
            or failure.get('type')!='ValueError' or "constraint_matrix=np.asarray"not in failure.get('traceback','')):
        raise ValueError('Authenticated pre-iteration CSR metadata failure and complete natural shutdown required')
    inventory=read(proof['original_instance_inventory']);original={int(row['ProcessId']):row for row in inventory['instances']}
    if set(original)!=set(map(int,proof['checked_pid_list'])):raise ValueError('Complete failed-instance PID inventory differs')
    reused=[]
    for row in group.windows_processes():
        pid=int(row['ProcessId'])
        if pid not in original:continue
        before=datetime.fromisoformat(original[pid]['CreationDate'].replace('Z','+00:00'))
        now=datetime.fromisoformat(row['CreationDate'].replace('Z','+00:00'))
        if before.tzinfo is None or now.tzinfo is None or now<=before:
            raise ValueError('Failed owner/worker instance is still visible: '+repr(row))
        reused.append(row)
    return dict(passed=True,failure_shutdown=config['preceding_failure_shutdown'],
        current_newer_PID_reuse=reused,original_instance_creation_times_checked=True,no_process_terminated=True)


def validate_composition(config,config_ref):
    authenticate_references(config)
    old=read(config['compatibility_parent_configuration'])
    if(config.get('execution_entry')!=reference(__file__)or config.get('frozen_group_entry')!=reference(group.__file__)or config.get('compatibility_kind')!='legacy_CSR_to_dense_at_metadata_and_solver_delivery_only'):
        raise ValueError('Explicit new execution-only type compatibility configuration required')
    for key,value in old.items():
        if key in('execution_entry','planned_output','generator','launcher','recorded_at_utc'):continue
        if config.get(key)!=value:raise ValueError('Frozen 4b model/source/numerical policy changed: '+key)
    qa=read(config['compatibility_QA'])
    if(qa.get('passed')is not True or qa.get('entry')!=reference(__file__)
            or qa.get('original_metadata')!=reference(_METADATA_MODULE.__file__)
            or qa.get('temporary_method_restored_on_success_and_failure')is not True
            or qa.get('original_owner_CSR_callback_unchanged')is not True
            or qa.get('active_solver_CSR_rejection_reproduced')is not True
            or qa.get('active_solver_dense_step_byte_exact')is not True):
        raise ValueError('Passed exact-entry real legacy preflight and active-solver type QA required')
    projected=deepcopy(config);projected['execution_entry']=reference(group.__file__)
    result=_VALIDATE(projected,config_ref)
    return dict(result,type_compatibility_QA=config['compatibility_QA'],preceding_failed_owner_exit=validate_failed_exit(config),
        legacy_validation_shadow_scope='Only execution_entry is projected to authenticate unchanged frozen mathematics; actual manifest/base configuration remains new',
        actual_execution_entry=config['execution_entry'],frozen_group_entry=config['frozen_group_entry'])


def make_stage(base_owner,manifest):
    owner=_MAKE(base_owner,manifest)
    install_solver_delivery(owner)
    return owner


def verify_directions(owner,historical_configuration,output):
    ref=_DIRECTIONS(owner,historical_configuration,output)
    value=read(ref);value['algorithm_execution_entry']=value['execution_entry']
    value['execution_entry']=reference(__file__);value['type_compatibility_QA']=group.base._CONFIG['compatibility_QA']
    group.base.construction.write(Path(output),value)
    return reference(output)


def main():
    group.validate_composition=validate_composition;group.current_process_preflight=current_process_preflight
    group.make_stage=make_stage;group.verify_directions=verify_directions;group.FIT_TOKENS=FIT_TOKENS
    _METADATA_MODULE.metadata_preflight=metadata_preflight
    group.main()


if __name__=='__main__':main()
