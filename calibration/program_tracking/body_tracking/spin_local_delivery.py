"""Deliver a completed local S05 repair; no fitting, inference, GUI or movie.

The retained C6 and all older variants stay intact. Registration backs up the
existing refined launcher before updating that one entry to the new config.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from . import spin_continuation_delivery as previous
from . import spin_continuous_delivery as continuous
from . import spin_support_foot_delivery as shared
from .absolute_motion_state import LIFTED_SCHEMA, curve_from_packet
from .exmr_integration import read_case_identity, validate_case_identity
from .skate_checkpoint import validate_checkpoint_identity


VARIANT = 'local_281'
BASELINE = 'continuous_refined'
SOURCE_CONFIG = 'viewer3d_spin_support_continuous_1105.json'
STEM = 'spin_281_repaired_1105'
CONFIG = 'viewer3d_' + STEM + '.json'
SECONDS = (81.54, 81.56, 81.58, 81.60, 81.62, 81.64, 81.66, 81.68, 81.70, 81.72)


def _absolute(value, name):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(name + ' must be an absolute path')
    return path.resolve()


def _completed(summary, result, parameter):
    accepted = previous._accepted(summary)
    for key, expected in (('result', result), ('parameter', parameter)):
        if _absolute(summary[key], 'Completed summary ' + key) != expected:
            raise ValueError('Completed summary refers to a different ' + key)
    if summary.get('source_frames') != 951 or summary.get('source_context_seconds') != [76, 95]:
        raise ValueError('Completed summary must retain the whole 76-95s / 951-frame curve')
    return accepted


def _case(result, parameter, pts, variant):
    result, parameter = _absolute(result, variant + ' result'), _absolute(parameter, variant + ' parameter')
    stage = result.parents[3]
    if result != stage / shared.CASE / 'fitted/result.npz':
        raise ValueError('Only the final fitted/result.npz can be delivered: ' + str(result))
    for path in (result, parameter, stage / 'run_summary.json'):
        if not path.is_file():
            raise FileNotFoundError('Required completed artifact absent: ' + str(path))
    summary = shared._json(stage / 'run_summary.json')
    _completed(summary, result, parameter)
    packet, params = shared._load(result), shared._load(parameter)
    shared._clock(packet, pts, variant)
    for field in ('source_time_base', 'physics_time_base'):
        if field in packet:
            np.testing.assert_array_equal(packet[field], shared.TIME_BASE)
    validate_checkpoint_identity(params)
    if str(params.get('state_schema_version', '')) != LIFTED_SCHEMA or str(params.get('state_kind', '')) != 'absolute_motion_v1':
        raise ValueError(variant + ': expected the single absolute_motion_v2 curve')
    curve_from_packet(params)
    validate_case_identity(params, read_case_identity(packet))
    if _absolute(str(params['absolute_case_root']), 'Absolute case') != stage:
        raise ValueError(variant + ': parameter belongs to a different case')
    if _absolute(str(packet['absolute_parameter_path']), 'Result parameter') != parameter:
        raise ValueError(variant + ': result belongs to a different parameter')
    if previous._camera_identity(packet['camera_binding_json']) != previous._camera_identity(params['camera_binding_json']):
        raise ValueError(variant + ': result and parameter cameras differ')
    from rinkspace.viewer3d.data import hand_state, motion_diagnostics_state
    hand_state(packet, len(pts), dict(id=variant, prior_label='exmr_hand_nofoot'))
    motion_diagnostics_state(packet, len(pts), 1., continuous.SAVED_FIELDS)
    return dict(variant=variant, result=result, parameter=parameter, root=stage,
                packet=packet, params=params, summary=summary)


def _saved(project, base_root, run_root, parameter):
    project, base, run, parameter = [Path(value).resolve() for value in (project, base_root, run_root, parameter)]
    paths = dict(config=project/shared.PROGRAMME/CONFIG, launcher=project/'launchers'/previous.LAUNCHER,
        registration=run/'player_registration.json', preparation=run/'player_preparation.json',
        stills=run/'native_stills', backup=run/'launcher_before.cmd', backup_record=run/'launcher_before.json')
    for path in paths.values():
        shared._guard(path, project)
    old_config = project/shared.PROGRAMME/SOURCE_CONFIG
    summary_path = run/'run_summary.json'
    contract = shared._json(base/'action_integration_1103/spin_s05/contract.json')
    source = contract['source']; pts = np.arange(972800, 1216000+1, 256, dtype=np.int64)
    if (contract.get('native50_count') != len(pts) or tuple(source['time_base']) != shared.TIME_BASE
            or source['pts_range'] != pts[[0, -1]].tolist() or source['native_size_wh'] != [1920, 1080]):
        raise ValueError('Expected all 951 original native source exposures')
    if not Path(source['path']).is_file():
        raise FileNotFoundError(source['path'])
    before, summary = shared._json(old_config), shared._json(summary_path)
    if (Path(before['source_video']).resolve() != Path(source['path']).resolve()
            or tuple(before['source_time_base']) != shared.TIME_BASE):
        raise ValueError('Retained player source identity/clock differs from contract')
    ids = [row['id'] for row in before['variants']]
    if (len(ids) != len(set(ids)) or VARIANT in ids
            or any(name not in ids for name in (*previous.BASELINES, BASELINE))):
        raise ValueError('Expected unique retained A/B/C6 variants without a local replacement')
    local = _case(summary['result'], parameter, pts, VARIANT)
    if not local['root'].is_relative_to(run):
        raise ValueError('Completed local result is outside the requested run root')
    accepted = _completed(summary, local['result'], parameter)
    if previous._accepted(local['summary']) != accepted or local['summary']['retained_step'] != summary['retained_step']:
        raise ValueError('Root and stage completed selections differ')
    definition = next(row for row in before['variants'] if row['id'] == BASELINE)
    old_result = _absolute(definition['result'], 'Retained C6 result')
    old_summary = shared._json(old_result.parents[3]/'run_summary.json')
    old = _case(old_result, old_summary['parameter'], pts, BASELINE)
    validate_checkpoint_identity(local['params'], old['params'])
    camera = previous._camera_identity(local['packet']['camera_binding_json'])
    frozen = shared._json(base/'spin_continuous_1105/initialization.json')['camera']
    if camera != previous._camera_identity(old['packet']['camera_binding_json']) or camera != previous._camera_identity(frozen):
        raise ValueError('Local/C6 camera differs from the frozen gravity camera')
    identity = read_case_identity(local['packet'])
    old_identity = read_case_identity(old['packet'])
    for field in ('source_pts_sha256', 'native_input_sha256', 'source_frames', 'source_time_base'):
        if identity[field] != old_identity[field]:
            raise ValueError('Local/C6 original source identity differs: ' + field)
    if identity['source_pts_sha256'] != hashlib.sha256(pts.astype('<i8').tobytes()).hexdigest():
        raise ValueError('Case source PTS digest differs')
    stage = local['root']; case = stage/shared.CASE
    documents = dict(support=stage/'support_and_free_shoe_regions.json', relations=stage/'grasp_relations.json')
    for path, field in ((case/'input.npz', 'case_input_sha256'), (case/'compact_smplx.npz', 'compact_smplx_sha256'),
                        (documents['support'], 'support_document_sha256'), (documents['relations'], 'relation_document_sha256')):
        if shared._digest(path) != identity[field]:
            raise ValueError('Completed case document identity differs: ' + str(path))
    shared._clock(shared._load(case/'input.npz'), pts, 'local case input')
    for field, expected in (('metres_per_scene_unit', frozen['metres_per_scene_unit']),
            ('solver_metres_per_scene_unit', frozen['solver_metres_per_scene_unit']), ('body_length_ratio', frozen['length_export_ratio'])):
        np.testing.assert_allclose(local['packet'][field], expected, rtol=1e-12, atol=0)
        np.testing.assert_array_equal(local['packet'][field], old['packet'][field])
    interval = np.asarray(summary.get('local_control_seconds', [81.2, 82.0]), dtype=float)
    if interval.shape != (2,) or not np.isfinite(interval).all() or not 76 <= interval[0] < interval[1] <= 95:
        raise ValueError('local_control_seconds must be one finite interval inside 76-95s')
    evidence = dict(schema='rinkspace.spin-local-delivery.v1', variant=VARIANT, source=source,
        source_frames=len(pts), source_pts=pts.tolist(), source_time_base=list(shared.TIME_BASE),
        parameter=str(parameter), parameter_sha256=shared._digest(parameter),
        result=str(local['result']), result_sha256=shared._digest(local['result']),
        completed_summary=str(summary_path), completed_summary_sha256=shared._digest(summary_path),
        accepted_steps=accepted, retained_step=summary['retained_step'], local_control_seconds=interval.tolist(),
        preserved_c6=dict(config=str(old_config), config_sha256=shared._digest(old_config),
            result=str(old['result']), result_sha256=shared._digest(old['result']),
            parameter=str(old['parameter']), parameter_sha256=shared._digest(old['parameter'])),
        native_camera_unchanged=True, same_pts_and_world=True, no_alignment=True,
        no_new_fit=True, no_new_inference=True, no_visible_gui_opened=True, accuracy_accepted=False)
    return dict(project=project, base=base, run=run, paths=paths, before=before, pts=pts, case=case,
        documents=documents, cases={BASELINE:old, VARIANT:local}, evidence=evidence)


def _configuration(saved):
    config = copy.deepcopy(saved['before'])
    config['variants'].append(dict(id=VARIANT, label='281帧附近左右腿关联修复，待审核',
        kind='fullbody', result=str(saved['cases'][VARIANT]['result']), shell_topology='saved',
        body_reference_label='exmr_hand_nofoot', required_saved_diagnostics=list(continuous.SAVED_FIELDS),
        body_refinement_intervals_seconds=[saved['evidence']['local_control_seconds']]))
    config.update(shot=STEM, session_id=STEM, default_variant=VARIANT, initial_source_pts=1044224,
        comparison_variants=[BASELINE, VARIANT], source_run=str(saved['case']),
        support_observations=str(saved['documents']['support']), relation_document=str(saved['documents']['relations']),
        review_status='Completed local leg-association repair; source/3D accuracy not accepted')
    return config


def register(project, base_root, run_root, parameter):
    saved = _saved(project, base_root, run_root, parameter)
    paths, project = saved['paths'], saved['project']
    for key in ('config', 'registration', 'backup', 'backup_record'):
        if paths[key].exists():
            raise FileExistsError('Existing/partial delivery is preserved: ' + str(paths[key]))
    if not (project/'launchers/START_3D_RESULTS.ps1').is_file():
        raise FileNotFoundError('Shared START_3D_RESULTS.ps1 launcher is absent')
    before = paths['launcher'].read_bytes()
    if SOURCE_CONFIG.encode('ascii') not in before:
        raise ValueError('Existing refined launcher does not point to the retained C6 config')
    relative = (shared.PROGRAMME/CONFIG).as_posix().replace('/', '\\')
    after = ('@echo off\r\npowershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File '
        '"%~dp0START_3D_RESULTS.ps1" -Config "%~dp0..\\' + relative + '" %*\r\nexit /b %ERRORLEVEL%\r\n').encode('ascii')
    backup = dict(launcher=str(paths['launcher']), backup=str(paths['backup']),
        sha256=hashlib.sha256(before).hexdigest(), bytes=len(before), bytes_base64=base64.b64encode(before).decode('ascii'))
    shared._new_bytes(paths['backup'], before, project)
    shared._write_new_json(paths['backup_record'], backup, project)
    config = _configuration(saved)
    shared._write_new_json(paths['config'], config, project)
    # Compare the original bytes again immediately before replacing this entry.
    with paths['launcher'].open('r+b') as stream:
        if stream.read() != before:
            raise ValueError('Refined launcher changed during registration; backup/config retained')
        stream.seek(0); stream.write(after); stream.truncate()
    report = dict(saved['evidence'], config=str(paths['config']), config_sha256=shared._digest(paths['config']),
        old_variants_preserved=[row['id'] for row in saved['before']['variants']],
        default_variant=VARIANT, comparison_variants=[BASELINE, VARIANT],
        launcher=str(paths['launcher']), launcher_sha256=shared._digest(paths['launcher']),
        launcher_updated=True, launcher_before=backup, launcher_backup_record=str(paths['backup_record']),
        actual_hand_channel=True, saved_diagnostics=list(continuous.SAVED_FIELDS),
        cache_preparation='separate prepare action', launch_command='"' + str(paths['launcher']) + '"')
    shared._write_new_json(paths['registration'], report, project)
    return report


def prepare(project, base_root, run_root, parameter):
    saved = _saved(project, base_root, run_root, parameter)
    if shared._json(saved['paths']['config']) != _configuration(saved):
        raise ValueError('Registered local configuration differs from completed artifacts')
    caches = continuous._prepare(saved['project'], saved['base'], saved['paths']['config'], saved['pts'], saved['cases'])
    report = dict(saved['evidence'], caches=caches, prepared_variants=[BASELINE, VARIANT],
        cache_checks='all 951 PTS, actual hands, COM, blade material points and contact witnesses against source result')
    path = saved['paths']['preparation']
    if path.exists():
        if shared._json(path) != report:
            raise FileExistsError('Changed preparation report is preserved: ' + str(path))
    else:
        shared._write_new_json(path, report, saved['project'])
    return report


def stills(project, base_root, run_root, parameter):
    saved = _saved(project, base_root, run_root, parameter)
    target, cases = saved['paths']['stills'], saved['cases']
    if target.exists():
        raise FileExistsError('Existing native stills are preserved: ' + str(target))
    requested = np.array([round(value*12800) for value in SECONDS], dtype=np.int64)
    frames = np.searchsorted(saved['pts'], requested)
    np.testing.assert_array_equal(saved['pts'][frames], requested)
    for name in ['projection_and_clock.json', *(f'compare_{int(pt)}.png' for pt in requested)]:
        shared._guard(target/name, saved['project'])
    source = saved['evidence']['source']
    if shared._digest(source['path']) != source['sha256']:
        raise ValueError('Original source video SHA256 differs from contract')
    from PIL import Image, ImageFont
    from rinkspace.viewer3d.source import SourceVideo
    from .spin_continuous_refit import restore_absolute
    from .spin_support_review import project_mesh
    from .native_mesh_overlay import paint
    target.mkdir(parents=True)
    refs = {}; camera = None
    for variant, value in cases.items():
        ref, current, _, _ = restore_absolute(saved['project'], saved['base'], target/(variant+'_geometry'), value['parameter'])
        shared._clock(ref.p, saved['pts'], variant+' restored input')
        shared._check_state(ref, value['packet'], frames)
        if camera is not None:
            for key in ('rig', 'controls', 'motion_times', 'first_control_time', 'control_dt'):
                np.testing.assert_array_equal(camera.p[key], current.p[key])
        camera = current; refs[variant] = ref
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 23)
    reader = SourceVideo(source['path'], shared.TIME_BASE); records = []
    try:
        for frame, pt in zip(frames, requested):
            raw = Image.fromarray(reader.read(int(pt)))
            if raw.size != (1920, 1080):
                raise ValueError('Original source frame size differs')
            canvas = Image.new('RGB', (5760, 1080)); canvas.paste(raw, (0, 0))
            for column, (variant, label) in enumerate(zip((BASELINE, VARIANT), ('旧C6（保留）', '281帧附近关联修复（待审核）')), start=1):
                ref = refs[variant]
                shared._check_state(ref, cases[variant]['packet'], np.array([frame]))
                mesh, shoe, _, _, equipment, count = project_mesh(ref, camera, np.array([frame]))
                su = shoe[0]; parts = [su[0, :count], su[0, count:], su[1, :count], su[1, count:]]
                faces = np.asarray(equipment['template']['faces'], dtype=np.int32)
                canvas.paste(paint(raw, mesh[0], ref.binding.surface.faces, parts,
                    [faces, ref.binding.faces, faces, ref.binding.faces], label, font, int(pt)), (column*1920, 0))
            path = target/f'compare_{int(pt)}.png'
            with path.open('xb') as stream:
                canvas.save(stream, format='PNG')
            records.append(dict(source_pts=int(pt), source_seconds=int(pt)/12800., path=str(path), sha256=shared._digest(path)))
    finally:
        reader.close()
    report = dict(saved['evidence'], frames=records, panel_order=['original_source', BASELINE, VARIANT],
        exact_exposure_nodes_per_frame=5, rendered_source_pts=requested.tolist(), source_sha256_checked=True,
        parameter_result_midpoint_checks='P/Q/local_R/actual hands, every rendered frame',
        projection='same native Brown camera/implicit rolling row; five actual body exposure nodes',
        geometry='complete SMPL-X surface and both attached lower/upper skate meshes',
        display='three full 1920x1080 native panels; no crop or alignment', no_movie_generated=True)
    shared._write_new_json(target/'projection_and_clock.json', report, saved['project'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', required=True, choices=('register', 'prepare', 'stills'))
    for name in ('project', 'base-root', 'run-root', 'parameter'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = dict(register=register, prepare=prepare, stills=stills)[args.action](args.project, args.base_root, args.run_root, args.parameter)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
