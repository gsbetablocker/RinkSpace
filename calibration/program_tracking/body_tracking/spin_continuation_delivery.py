"""Register, cache, or review a completed absolute-motion continuation.

Each action consumes real final artifacts.  No fitting, network inference,
GUI launch, source-production replay, or edits to the old A/B files occur.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from . import spin_continuous_delivery as continuous
from . import spin_support_foot_delivery as shared
from .absolute_motion_state import CURVE_SCHEMAS
from .exmr_integration import read_case_identity, validate_case_identity
from .skate_checkpoint import CAMERA_LOCATION_FIELDS, validate_checkpoint_identity


VARIANT = 'continuous_refined'
BASELINES = ('continuous_source', 'continuous_flight')
SOURCE_CONFIG = 'viewer3d_spin_continuous_1105.json'
LAUNCHER = 'START_3D_SPIN_REFINED.cmd'
SECONDS = (81.4, 85.56, 85.58, 86., 88.56, 89.32, 89.36, 90.56, 91.48, 92.8, 93.2, 93.5)


def _camera_identity(value):
    value = json.loads(str(value)) if not isinstance(value, dict) else value
    return {key: item for key, item in value.items() if key not in CAMERA_LOCATION_FIELDS}


def _names(project, run, stem):
    if not stem or any(char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for char in stem):
        raise ValueError('stem must contain only ASCII letters, digits, underscore or hyphen')
    paths = dict(config=project/shared.PROGRAMME/('viewer3d_'+stem+'.json'),
        launcher=project/'launchers'/LAUNCHER, registration=run/'player_registration.json',
        preparation=run/'player_preparation.json', stills=run/'native_stills')
    for path in paths.values():
        shared._guard(path, project)
    return paths


def _accepted(summary):
    count = summary.get('accepted_steps', summary.get('accepted'))
    if type(count) is not int or count < 0:
        raise ValueError('Completed summary accepted count must be a nonnegative integer')
    retained = summary.get('retained_step')
    if type(retained) is not int or not 0 <= retained <= count:
        raise ValueError('Completed summary retained_step must identify a retained accepted state')
    return count


def _saved(project, base_root, run_root, parameter, stem):
    project, base, run, parameter = [Path(value).resolve() for value in
        (project, base_root, run_root, parameter)]
    paths = _names(project, run, stem)
    stage = run/'continuous_stage'
    case = stage/shared.CASE
    result_path = case/'fitted/result.npz'
    required = dict(summary=run/'run_summary.json', stage_summary=stage/'run_summary.json',
        result=result_path, parameter=parameter, input=case/'input.npz',
        compact=case/'compact_smplx.npz', support=stage/'support_and_free_shoe_regions.json',
        relations=stage/'grasp_relations.json', old_config=project/shared.PROGRAMME/SOURCE_CONFIG,
        contract=base/'action_integration_1103/spin_s05/contract.json',
        frozen_initialization=base/'spin_continuous_1105/initialization.json')
    # Never substitute a live checkpoint, last-accepted state or initial export.
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f'Required completed {name} artifact is absent: {path}')
    summary = shared._json(required['summary'])
    stage_summary = shared._json(required['stage_summary'])
    accepted = _accepted(summary)
    if (_accepted(stage_summary) != accepted or
            stage_summary['retained_step'] != summary['retained_step']):
        raise ValueError('Root and stage completed selections differ')
    for field, expected in (('parameter', parameter), ('result', result_path)):
        for record in (summary, stage_summary):
            value = Path(record[field])
            if not value.is_absolute() or value.resolve() != expected:
                raise ValueError('Completed summary refers to a different '+field)
    for record in (summary, stage_summary):
        if record.get('source_frames') != 951 or record.get('source_context_seconds') != [76, 95]:
            raise ValueError('Expected a completed whole 76-95s continuation')
    contract = shared._json(required['contract'])
    source = contract['source']
    pts = np.arange(76*12800, 95*12800+1, 256, dtype=np.int64)
    if (contract.get('native50_count') != len(pts) or tuple(source['time_base']) != shared.TIME_BASE
            or source['pts_range'] != pts[[0, -1]].tolist() or source['native_size_wh'] != [1920, 1080]):
        raise ValueError('Original source contract does not contain all 951 native exposures')
    if not Path(source['path']).is_file():
        raise FileNotFoundError(source['path'])
    before = shared._json(required['old_config'])
    if (Path(before['source_video']).resolve() != Path(source['path']).resolve()
            or tuple(before['source_time_base']) != shared.TIME_BASE):
        raise ValueError('Original player source identity or time base changed')
    ids = [value['id'] for value in before['variants']]
    if len(ids) != len(set(ids)) or VARIANT in ids or any(name not in ids for name in BASELINES):
        raise ValueError('Original player must retain unique A/B variants without a refined replacement')
    packet = shared._load(result_path)
    shared._clock(packet, pts, 'completed continuation')
    shared._clock(shared._load(required['input']), pts, 'completed continuation input')
    for field in ('source_time_base', 'physics_time_base'):
        if field in packet:
            np.testing.assert_array_equal(packet[field], shared.TIME_BASE)
    params = shared._load(parameter)
    validate_checkpoint_identity(params)
    if str(params.get('state_schema_version', '')) not in CURVE_SCHEMAS or str(params.get('state_kind', '')) != 'absolute_motion_v1':
        raise ValueError('Completed parameter is not an absolute-motion state')
    validate_case_identity(params, read_case_identity(packet))
    if Path(str(params['absolute_case_root'])).resolve() != stage:
        raise ValueError('Completed parameter belongs to a different case')
    if Path(str(packet['absolute_parameter_path'])).resolve() != parameter:
        raise ValueError('Completed result is bound to a different parameter')
    camera = _camera_identity(packet['camera_binding_json'])
    frozen = shared._json(required['frozen_initialization'])['camera']
    if camera != _camera_identity(params['camera_binding_json']) or camera != _camera_identity(frozen):
        raise ValueError('Completed result/parameter differ from the frozen gravity camera')
    for field, expected in (('metres_per_scene_unit', frozen['metres_per_scene_unit']),
                             ('solver_metres_per_scene_unit', frozen['solver_metres_per_scene_unit']),
                             ('body_length_ratio', frozen['length_export_ratio'])):
        np.testing.assert_allclose(packet[field], expected, rtol=1e-12, atol=0)
    identity = read_case_identity(packet)
    if shared._digest(required['support']) != identity['support_document_sha256']:
        raise ValueError('Completed source evidence document changed')
    if shared._digest(required['relations']) != identity['relation_document_sha256']:
        raise ValueError('Completed relation document changed')
    from rinkspace.viewer3d.data import hand_state, motion_diagnostics_state
    hand_state(packet, len(pts), dict(id=VARIANT, prior_label='exmr_hand_nofoot'))
    motion_diagnostics_state(packet, len(pts), 1., continuous.SAVED_FIELDS)
    evidence = dict(schema='rinkspace.spin-continuation-delivery.v1', variant=VARIANT,
        parameter=str(parameter), parameter_sha256=shared._digest(parameter),
        result=str(result_path), result_sha256=shared._digest(result_path),
        completed_summary=str(required['summary']), completed_summary_sha256=shared._digest(required['summary']),
        source=source, source_frames=len(pts), source_time_base=list(shared.TIME_BASE),
        accepted_steps=accepted, retained_step=summary['retained_step'],
        support_document=str(required['support']), support_document_sha256=shared._digest(required['support']),
        native_camera_unchanged=True, same_pts_and_world=True, no_alignment=True,
        no_new_fit=True, no_new_inference=True, no_visible_gui_opened=True,
        accuracy_accepted=False)
    return dict(project=project, base=base, run=run, stage=stage, case=case, stem=stem,
        paths=paths, required=required, result=result_path, parameter=parameter,
        summary=summary, packet=packet, params=params, before=before, pts=pts, evidence=evidence)


def _configuration(saved, default):
    default = {'A': BASELINES[0], 'B': BASELINES[1]}.get(default, default)
    if default not in (*BASELINES, VARIANT):
        raise ValueError('default must be continuous_refined, continuous_source/A, or continuous_flight/B')
    config = copy.deepcopy(saved['before'])
    config['variants'].append(dict(id=VARIANT,
        label=f"整段优化后续算（{saved['evidence']['accepted_steps']}步，待原片审核）",
        kind='fullbody', result=str(saved['result']), shell_topology='saved',
        body_reference_label='exmr_hand_nofoot', body_refinement_intervals_seconds=[[76., 95.]],
        required_saved_diagnostics=list(continuous.SAVED_FIELDS)))
    config.update(shot=saved['stem'], session_id=saved['stem'], default_variant=default,
        source_run=str(saved['case']), support_observations=str(saved['required']['support']),
        relation_document=str(saved['required']['relations']),
        comparison_variants=[*BASELINES, VARIANT],
        review_status='Completed whole-shot continuation; source/3D accuracy not accepted',
        body_window_scope='One absolute 76-95s body curve; all 951 original source exposures')
    # These display channels keep their original meanings and actual hands.
    config['initial_visible'] = dict(config.get('initial_visible', {}),
        com=True, blade_material=True, contact=True)
    return config


def register(project, base_root, run_root, parameter, stem, default=VARIANT):
    saved = _saved(project, base_root, run_root, parameter, stem)
    project, paths = saved['project'], saved['paths']
    for key in ('config', 'launcher', 'registration'):
        if paths[key].exists():
            raise FileExistsError('Existing delivery is preserved: '+str(paths[key]))
    if not (project/'launchers/START_3D_RESULTS.ps1').is_file():
        raise FileNotFoundError('Shared START_3D_RESULTS.ps1 launcher is absent')
    config = _configuration(saved, default)
    shared._write_new_json(paths['config'], config, project)
    relative = (shared.PROGRAMME/paths['config'].name).as_posix().replace('/', '\\')
    lines = ['@echo off',
        'powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_3D_RESULTS.ps1" -Config "%~dp0..\\'+relative+'" %*',
        'exit /b %ERRORLEVEL%']
    shared._new_bytes(paths['launcher'], ('\r\n'.join(lines)+'\r\n').encode('ascii'), project)
    report = dict(saved['evidence'], config=str(paths['config']), config_sha256=shared._digest(paths['config']),
        launcher=str(paths['launcher']), launcher_sha256=shared._digest(paths['launcher']),
        original_config=str(saved['required']['old_config']), original_config_sha256=shared._digest(saved['required']['old_config']),
        old_variants_preserved=[row['id'] for row in saved['before']['variants']],
        default_variant=config['default_variant'], comparison_variants=config['comparison_variants'],
        actual_hand_channel=True, saved_diagnostics=list(continuous.SAVED_FIELDS),
        cache_preparation='separate prepare action',
        launch_command=f'"{paths["launcher"]}" -Variant {config["default_variant"]}')
    shared._write_new_json(paths['registration'], report, project)
    return report


def _registered(saved):
    path = saved['paths']['config']
    if not path.is_file():
        raise FileNotFoundError('Register the completed continuation before preparing caches: '+str(path))
    config = shared._json(path)
    expected = _configuration(saved, config.get('default_variant'))
    if config != expected:
        raise ValueError('Registered continuation config differs from its completed artifacts')
    return config


def _baseline_cases(saved, *, parameters=False):
    """Read old final states only when a later explicit prepare/stills needs them."""
    cases = {}
    for variant in BASELINES:
        definition = next(row for row in saved['before']['variants'] if row['id'] == variant)
        result = Path(definition['result'])
        if not result.is_absolute() or not result.is_file():
            raise FileNotFoundError('Required retained A/B result absent: '+str(result))
        packet = shared._load(result)
        shared._clock(packet, saved['pts'], variant)
        if _camera_identity(packet['camera_binding_json']) != _camera_identity(saved['packet']['camera_binding_json']):
            raise ValueError('Retained baseline camera differs: '+variant)
        for key in ('metres_per_scene_unit', 'solver_metres_per_scene_unit', 'body_length_ratio'):
            np.testing.assert_array_equal(packet[key], saved['packet'][key])
        value = dict(variant=variant, result=result, packet=packet)
        if parameters:
            stage = result.parents[3]
            summary = shared._json(stage/'run_summary.json')
            if Path(summary['result']).resolve() != result.resolve():
                raise ValueError('Retained baseline summary/result mismatch: '+variant)
            parameter = Path(summary['parameter'])
            if not parameter.is_absolute() or not parameter.is_file():
                raise FileNotFoundError('Retained baseline parameter absent: '+str(parameter))
            value.update(parameter=parameter, root=stage)
        cases[variant] = value
    cases[VARIANT] = dict(variant=VARIANT, result=saved['result'], packet=saved['packet'],
        parameter=saved['parameter'], root=saved['stage'])
    return cases


def prepare(project, base_root, run_root, parameter, stem, default=VARIANT):
    saved = _saved(project, base_root, run_root, parameter, stem)
    _registered(saved)
    cases = _baseline_cases(saved)
    # Existing FullSurface preparation verifies the complete cached geometry,
    # actual hand rotations, saved COM/material points and contact witnesses.
    caches = continuous._prepare(saved['project'], saved['base'], saved['paths']['config'],
                                 saved['pts'], cases)
    report = dict(saved['evidence'], config=str(saved['paths']['config']), caches=caches,
        prepared_variants=[*BASELINES, VARIANT], no_visible_gui_opened=True)
    path = saved['paths']['preparation']
    if path.exists():
        if shared._json(path) != report:
            raise FileExistsError('Changed cache preparation report is preserved: '+str(path))
    else:
        shared._write_new_json(path, report, saved['project'])
    return report


def stills(project, base_root, run_root, parameter, stem, default=VARIANT):
    saved = _saved(project, base_root, run_root, parameter, stem)
    cases = _baseline_cases(saved, parameters=True)
    target = saved['paths']['stills']
    if target.exists():
        raise FileExistsError('Existing native stills are preserved: '+str(target))
    requested = np.array([round(value*12800) for value in SECONDS], dtype=np.int64)
    frames = np.searchsorted(saved['pts'], requested)
    np.testing.assert_array_equal(saved['pts'][frames], requested)
    for pt in requested:
        shared._guard(target/f'compare_{int(pt)}.png', saved['project'])
    shared._guard(target/'projection_and_clock.json', saved['project'])
    source = saved['evidence']['source']
    if shared._digest(source['path']) != source['sha256']:
        raise ValueError('Original source video SHA256 differs from the contract')
    from PIL import Image, ImageFont
    from rinkspace.viewer3d.source import SourceVideo
    from .spin_continuous_refit import restore_absolute
    from .spin_support_review import project_mesh
    from .native_mesh_overlay import paint
    target.mkdir(parents=True)
    refs = {}
    camera = None
    for variant, value in cases.items():
        ref, current_camera, _, _ = restore_absolute(saved['project'], saved['base'],
            target/(variant+'_geometry'), value['parameter'])
        shared._clock(ref.p, saved['pts'], variant+' restored input')
        shared._check_state(ref, value['packet'], frames)
        if camera is not None:
            for key in ('rig', 'controls', 'motion_times', 'first_control_time', 'control_dt'):
                np.testing.assert_array_equal(camera.p[key], current_camera.p[key])
        camera = current_camera
        refs[variant] = ref
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 23)
    reader = SourceVideo(source['path'], shared.TIME_BASE)
    labels = ('A 原图连续运动（保留）', 'B 腾空物理（保留）', '整段优化后续算（待审核）')
    records = []
    try:
        for frame, pt in zip(frames, requested):
            raw = Image.fromarray(reader.read(int(pt)))
            if raw.size != (1920, 1080):
                raise ValueError('Original source frame size differs')
            canvas = Image.new('RGB', (4*1920, 1080))
            canvas.paste(raw, (0, 0))
            for column, (variant, label) in enumerate(zip((*BASELINES, VARIANT), labels), start=1):
                ref = refs[variant]
                shared._check_state(ref, cases[variant]['packet'], np.array([frame]))
                mesh, shoe, _, _, equipment, lower_count = project_mesh(ref, camera, np.array([frame]))
                su = shoe[0]
                parts = [su[0, :lower_count], su[0, lower_count:], su[1, :lower_count], su[1, lower_count:]]
                faces = np.asarray(equipment['template']['faces'], dtype=np.int32)
                image = paint(raw, mesh[0], ref.binding.surface.faces, parts,
                    [faces, ref.binding.faces, faces, ref.binding.faces], label, font, int(pt))
                canvas.paste(image, (column*1920, 0))
            path = target/f'compare_{int(pt)}.png'
            with path.open('xb') as stream:
                canvas.save(stream, format='PNG')
            records.append(dict(source_pts=int(pt), source_seconds=int(pt)/12800.,
                path=str(path), sha256=shared._digest(path)))
    finally:
        reader.close()
    report = dict(saved['evidence'], frames=records, panel_order=['original_source', *BASELINES, VARIANT],
        exact_exposure_nodes_per_frame=5, rendered_source_pts=requested.tolist(), source_sha256_checked=True,
        parameter_result_midpoint_checks='P/Q/local_R/actual hands, every rendered frame',
        projection='same native Brown camera/implicit rolling row; five actual body exposure nodes',
        geometry='complete SMPL-X surface and both attached lower/upper skate meshes',
        display='four full 1920x1080 native panels; no crop or alignment',
        no_movie_generated=True, no_visible_gui_opened=True)
    shared._write_new_json(target/'projection_and_clock.json', report, saved['project'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', required=True, choices=('register', 'prepare', 'stills'))
    for name in ('project', 'base-root', 'run-root', 'parameter'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--stem', required=True)
    parser.add_argument('--default', default=VARIANT, choices=(VARIANT, *BASELINES, 'A', 'B'))
    args = parser.parse_args(argv)
    result = dict(register=register, prepare=prepare, stills=stills)[args.action](
        args.project, args.base_root, args.run_root, args.parameter, args.stem, args.default)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
