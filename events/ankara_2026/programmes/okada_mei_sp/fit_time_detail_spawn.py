"""Compose the frozen time4a model with an authenticated spawn-only repair."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json, os, subprocess, sys

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
import numpy as np
import fit_time_detail_print as base
from calibration.program_tracking import first_shot_disjoint_spawn as spawn
from calibration.program_tracking.first_shot_source_transition import reference, read
from calibration.program_tracking.first_shot_time_successor import authenticate_references

_load = base.load_configuration
_install = base.install_camera_and_material


def require_failed_startup_exited(config, config_ref):
    proof = read(config['failed_startup_shutdown'])
    failure = read(config['failed_startup'])
    if (proof.get('passed') is not True or proof.get('full_windows_process_domain') is not True
            or proof.get('failure') != config['failed_startup']
            or proof.get('all_observed_owner_launcher_and_workers_absent') is not True
            or proof.get('owner_and_spawn_parent_domain_absent') is not True
            or proof.get('no_accepted_iteration') is not True
            or failure.get('type') != 'PicklingError'):
        raise ValueError('Authenticated failed initial checkpoint and actual shutdown required')
    pwsh = 'C:/Users/REDACTED/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe'
    query = 'Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Depth 4 -Compress'
    rows = json.loads(subprocess.check_output([pwsh, '-NoProfile', '-Command', query],text=True,encoding='utf-8'))
    if not isinstance(rows, list): rows = [rows]
    own = next((row for row in rows if int(row['ProcessId']) == os.getpid()), None)
    if own is None: raise ValueError('Actual full Windows domain must contain this process')
    absent = datetime.fromisoformat(proof['checked_at_utc'])
    if absent.tzinfo is None: raise ValueError('Qualified process absence timestamp required')
    reused = []
    for row in rows:
        pid = int(row['ProcessId'])
        created = datetime.fromisoformat(row['CreationDate'])
        command = (row.get('CommandLine') or '').replace('\\','/').casefold()
        if pid in proof['checked_pid_list']:
            if created <= absent: raise ValueError('Failed startup instance still visible')
            reused.append(row)
        if int(row['ParentProcessId']) in (14776, 19256) or 'spawn_main(parent_pid=14776,' in command:
            raise ValueError('Failed startup worker domain is still visible')
        if not (row.get('Name') or '').casefold().startswith(('python','pypy')): continue
        if not command: raise ValueError('Unreadable Python instance cannot be excluded')
        if Path(__file__).name.casefold() not in command or pid == os.getpid(): continue
        parent = (pid == os.getppid() == int(own['ParentProcessId'])
            and created <= datetime.fromisoformat(own['CreationDate'])
            and Path(row.get('ExecutablePath') or '').resolve() == (ROOT/'.venv/Scripts/python.exe').resolve()
            and all(str(path).replace('\\','/').casefold() in command for path in
                (__file__, config_ref['path'], config['planned_output'])))
        if not parent: raise ValueError('Another successor owner/launcher is visible: '+repr(row))
    return dict(passed=True, full_windows_process_domain=True, failed_startup_PID_reuse=reused,
        no_old_worker_parent_domain_visible=True, no_process_terminated=True)


def validate_composition(config, config_ref):
    authenticate_references(config)
    if (config.get('execution_entry') != reference(__file__)
            or config.get('spawn_runtime') != reference(spawn.__file__)):
        raise ValueError('Explicit unchanged model plus actual spawn-repair entry required')
    qa = read(config['spawn_protocol_QA'])
    historical = base.source.load_successor(config['reconstruct_source_manifest']['path'])[0]
    if (qa.get('passed') is not True or qa.get('adapter') != config['spawn_runtime']
            or qa.get('print_runtime') != config['print_runtime']
            or qa.get('print_configuration') != config['print_configuration']
            or qa.get('model_reference') != historical['fixed_head_model']
            or qa.get('canonical_q_sha256') != config['parent_q_sha256']):
        raise ValueError('Actual raw/source/spawn QA must pass')
    old_config = read(config['completed_camera_QA_configuration'])
    qa_source = read(config['completed_camera_source_QA'])
    for key in ('entry','detail_runtime','print_runtime','print_configuration','parent_q_sha256',
                'reconstruct_source_manifest','current_direction_configurations','detail_selection_policy',
                'accepted_parent_checkpoint','accepted_parent_payload','accepted_parent_report',
                'parent_owner_shutdown','parent_source_inventory','parent_media_manifest',
                'detail_protocol_QA','print_protocol_QA'):
        if config[key] != old_config[key]: raise ValueError('Camera QA reuse model changed: '+key)
    if (qa_source.get('passed') is not True
            or qa_source['canonical_q_sha256'] != config['parent_q_sha256']
            or old_config['parent_q_sha256'] != config['parent_q_sha256']):
        raise ValueError('Only completed exact same-q/source/model camera QA may be reused')
    if config['completed_camera_direction_reports'] != qa_source['checks'] or len(qa_source['checks']) != 4:
        raise ValueError('Exact four completed same-model camera FD reports required')
    if any(read(item).get('physical_q_sha256') != qa.get('physical_q_sha256') for item in qa_source['checks']):
        raise ValueError('Spawn raw/source proof must use the identical full effective state')
    return require_failed_startup_exited(config, config_ref)


def load_configuration(path):
    manifest, ref = _load(path)
    base._CONFIG['actual_spawn_composition_preflight'] = validate_composition(base._CONFIG, ref)
    return manifest, ref


def install_camera_and_material(programme, **kwargs):
    result = _install(programme, **kwargs)
    result['spawn_transport_repair'] = spawn.install(programme, entry_reference=reference(__file__))
    return result


def reuse_detail_support(probe):
    qa = read(base._CONFIG['completed_camera_source_QA'])
    proof = deepcopy(qa['candidate_detail_support'])
    if proof['canonical_q_sha256'] != base.construction.sha(probe.start):
        raise ValueError('Completed full-source support q differs')
    physical, _ = probe.effective_q_and_chain(probe.start)
    for fd in base._CONFIG['completed_camera_direction_reports']:
        report = read(fd)
        if (report.get('passed') is not True or report['physical_q_sha256'] != base.construction.sha(physical)
                or report['canonical_q_sha256'] != base.construction.sha(probe.start)):
            raise ValueError('Same model effective q differs from completed real source FD')
    if proof['programme_formal_rows'] != len(probe.programme.scales):
        raise ValueError('Complete formal source row inventory changed')
    proof.update(reused_completed_same_q_source_support=True,
        completed_QA=base._CONFIG['completed_camera_source_QA'],
        spawn_only_repair=base._CONFIG['spawn_protocol_QA'], source_r_or_J_loaded=False)
    return np.asarray(proof['supported_detail_indices'],np.int64), proof


def reuse_directions(owner, historical_configuration, output):
    reuse_detail_support(owner)
    qa = deepcopy(read(base._CONFIG['completed_camera_source_QA']))
    qa.update(schema='rinkspace.authenticated-completed-camera-QA-spawn-successor.v1',
        actual_source_FD_not_repeated=True, completed_actual_source_QA=base._CONFIG['completed_camera_source_QA'],
        completed_direction_reports=base._CONFIG['completed_camera_direction_reports'],
        current_spawn_protocol_QA=base._CONFIG['spawn_protocol_QA'], execution_entry=reference(__file__),
        old_source_r_or_J_loaded=False, formal_joint_source_J_still_fresh=True)
    base.construction.write(Path(output),qa)
    return reference(output)


def main():
    base.load_configuration = load_configuration
    base.install_camera_and_material = install_camera_and_material
    base.camera_detail_support = reuse_detail_support
    base.verify_directions = reuse_directions
    base.main()


if __name__ == '__main__':
    main()
