"""Register the two real saved 76-95s stages; no fit, replay, inference or GUI.

The independent config/launcher is created only after both final result files
and their completed-run summaries have passed the source-clock checks. The
player consumes exported midpoint state; parameter-schema replay is upstream.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from . import spin_support_foot_delivery as shared


CONFIG = 'viewer3d_spin_continuous_1105.json'
LAUNCHER = 'START_3D_SPIN_CONTINUOUS.cmd'
SOURCE_CONFIG = 'viewer3d_spin_entry_motion_1104.json'
STAGES = ('source_stage', 'flight_stage')
VARIANTS = ('continuous_source', 'continuous_flight')
SAVED_FIELDS = ('com_world_mid', 'blade_material_world_mid', 'skate_witness_world_mid')
COMPARISON = (*VARIANTS, 'entry_motion', 'support_foot')


def _saved(project, base, output=None):
    project, base = (Path(path).resolve() for path in (project, base))
    output = Path(output).resolve() if output else base / 'spin_continuous_1105'
    paths = dict(config=project / shared.PROGRAMME / SOURCE_CONFIG,
                 contract=base / 'action_integration_1103/spin_s05/contract.json')
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'Required saved {name} is absent: {path}')
    contract = shared._json(paths['contract']); source = contract['source']
    pts = np.arange(972800, 1216000 + 1, 256, dtype=np.int64)
    if (contract['native50_count'] != len(pts)
            or tuple(source['time_base']) != shared.TIME_BASE
            or source['pts_range'] != pts[[0, -1]].tolist()
            or source['native_size_wh'] != [1920, 1080]):
        raise ValueError('Expected all 951 original 76-95s source exposures')
    source_path = Path(source['path']).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    config = shared._json(paths['config'])
    if (Path(config['source_video']).resolve() != source_path
            or tuple(config['source_time_base']) != shared.TIME_BASE):
        raise ValueError('Retained player source identity/clock differs from contract')
    ids = [variant['id'] for variant in config['variants']]
    if len(set(ids)) != len(ids) or any(name in ids for name in VARIANTS):
        raise ValueError('Duplicate or already registered continuous-stage variant')
    for name in ('entry_motion', 'support_foot', 'gvhmr_fixed'):
        if name not in ids:
            raise ValueError('Required retained comparison is absent: '+name)
    stages = {}
    from rinkspace.viewer3d.data import motion_diagnostics_state, hand_state
    for stage, variant in zip(STAGES, VARIANTS):
        root = output / stage; case = root / shared.CASE
        summary_path = root / 'run_summary.json'
        result = case / 'fitted/result.npz'
        # Never substitute a checkpoint, initial export or accepted-step log.
        for path in (summary_path, result, case / 'input.npz', case / 'compact_smplx.npz'):
            if not path.is_file():
                raise FileNotFoundError(f'Required completed {stage} artifact is absent: {path}')
        summary = shared._json(summary_path)
        if type(summary.get('accepted')) is not int or summary['accepted'] < 0:
            raise ValueError(stage+': accepted must be a nonnegative integer')
        parameter = Path(summary['parameter'])
        if not parameter.is_absolute() or not parameter.is_file():
            raise ValueError(stage+': completed summary parameter must name a real absolute file')
        if not Path(summary['result']).is_absolute() or Path(summary['result']).resolve() != result:
            raise ValueError(stage+': completed summary refers to a different final result')
        packet = shared._load(result)
        shared._clock(packet, pts, stage)
        shared._clock(shared._load(case / 'input.npz'), pts, stage+' input')
        hand_state(packet, len(pts), dict(id=variant, prior_label='exmr_hand_nofoot'))
        motion_diagnostics_state(packet, len(pts), 1., SAVED_FIELDS)
        stages[stage] = dict(root=root, case=case, parameter=parameter.resolve(),
                             result=result, summary=summary, summary_path=summary_path,
                             packet=packet, variant=variant)
    evidence = dict(schema='rinkspace.spin-continuous-saved-delivery.v1', source=source,
        source_frames=len(pts), source_pts=pts.tolist(), source_time_base=list(shared.TIME_BASE),
        stages={stage: dict(variant=value['variant'], accepted_steps=value['summary']['accepted'],
                           parameter=str(value['parameter']), parameter_sha256=shared._digest(value['parameter']),
                           result=str(value['result']), result_sha256=shared._digest(value['result']),
                           summary=str(value['summary_path']), summary_sha256=shared._digest(value['summary_path']))
                for stage, value in stages.items()},
        same_pts_and_world=True, coordinates='exported state, no alignment or ground snapping',
        com='conditional-metric mass-model approximation; not measured true centre of mass',
        blade_material_order=['rear', 'middle', 'front'],
        witness='moving minimum/contact candidate; not verified ice contact',
        no_new_fit=True, no_new_inference=True, no_parameter_replay=True,
        no_visible_gui_opened=True, accuracy_accepted=False)
    return project, base, output, paths, config, pts, stages, evidence


def check(project, base, output=None):
    """Read-only final-result/source/field validation, without cache creation."""
    return _saved(project, base, output)[-1]


def _configuration(before, stages):
    config = copy.deepcopy(before)
    for stage, title in zip(STAGES, ('A 原图连续运动', 'B 连续运动＋腾空物理')):
        value = stages[stage]
        config['variants'].append(dict(id=value['variant'],
            label=f"{title}候选（{value['summary']['accepted']}步，待原片审核）",
            kind='fullbody', result=str(value['result']), shell_topology='saved',
            body_reference_label='exmr_hand_nofoot',
            body_refinement_intervals_seconds=[[76., 95.]],
            required_saved_diagnostics=list(SAVED_FIELDS)))
    flight = stages['flight_stage']
    selection_path=flight['root'].parent/'quality_selection.json'
    selection=json.loads(selection_path.read_text(encoding='utf-8')) if selection_path.is_file() else {}
    recommended=selection.get('recommended_default_variant','continuous_flight')
    if recommended not in VARIANTS:raise ValueError('Unknown recommended continuous candidate')
    config.update(shot='spin_continuous_1105', session_id='spin_continuous_1105',
        source_run=str(flight['case']), default_variant=recommended,
        comparison_variants=list(COMPARISON), initial_source_pts=1040640,
        initial_view='side', initial_wear_inspection=False,
        initial_visible=dict(com=True, blade_material=True, contact=True, trajectory=False, pelvis=False),
        body_window_scope='Single continuous absolute-motion 76-95s candidate; all 951 source exposures',
        review_status='Two completed saved stages; source/3D accuracy not accepted',
        diagnostic_semantics=dict(com='条件米制质量近似，非真实质心',
            blade_material='每只刀固定材料点：后／中／前；不是接冰点',
            witness='移动最低点／接触候选，非真实接冰'))
    # Prefer the stage's actual observation documents when they were exported;
    # otherwise retain the existing source-bound observation paths unchanged.
    for key, name in (('support_observations', 'support_and_free_shoe_regions.json'),
                      ('relation_document', 'grasp_relations.json')):
        path = flight['root'] / name
        if path.is_file():
            config[key] = str(path)
    return config


def _prepare(project, base, config_path, pts, stages):
    from rinkspace.storage import AssetStore
    from rinkspace.viewer3d.data import ResultCatalog
    from rinkspace.viewer3d.prepare import FullSurface, prepare_variant
    catalog = ResultCatalog(project=project, data_root=AssetStore(project).data, config=config_path)
    shared._clock(catalog.packet, pts, 'continuous player input')
    if tuple(catalog.time_base) != shared.TIME_BASE:
        raise ValueError('Player time base differs')
    surface = None; caches = {}
    for stage, value in stages.items():
        definition = catalog.variant(value['variant'])
        path, _ = catalog.cache_path(definition)
        if not (path / 'metadata.json').is_file():
            surface = surface or FullSurface(catalog)
            path = prepare_variant(catalog, definition, surface)
        state = shared._load(path / 'state.npz')
        shared._clock(state, pts, stage+' player cache')
        packet = value['packet']; ratio = catalog.parameters(definition)['unit_ratio']
        for source, target in (('com_world_mid', 'com_world_mid'),
                               ('blade_material_world_mid', 'blade_material_world_mid'),
                               ('skate_witness_world_mid', 'skate_witness')):
            np.testing.assert_array_equal(state[target], packet[source] * ratio)
        np.testing.assert_array_equal(state['hand_local_R_mid'], packet['hand_local_R_mid'])
        caches[stage] = str(path)
    return caches


def prepare(project, base, output=None):
    """Prepare only real exported display states after registration exists."""
    project, base, output, _, _, pts, stages, _ = _saved(project, base, output)
    config_path = project / shared.PROGRAMME / CONFIG
    if not config_path.is_file():
        raise FileNotFoundError('Run register after the two real final results exist: '+str(config_path))
    config = shared._json(config_path)
    if (config['default_variant'] not in VARIANTS
            or config['comparison_variants'] != list(COMPARISON)):
        raise ValueError('Continuous player config differs from its saved registration')
    for stage, value in stages.items():
        variant = next(row for row in config['variants'] if row['id'] == value['variant'])
        if Path(variant['result']).resolve() != value['result']:
            raise ValueError(stage+': registered result differs')
    return _prepare(project, base, config_path, pts, stages)


def register(project, base, output=None):
    """Create an independent two-stage config/launcher and bake its two caches.

    Shares the prior delivery's clock, path guard, exclusive writes and cache
    preparation contract. Old entry/foot configs and launcher bytes are kept.
    """
    project, base, output, paths, before, pts, stages, evidence = _saved(project, base, output)
    target = project / shared.PROGRAMME / CONFIG
    launcher = project / 'launchers' / LAUNCHER
    report_path = output / 'continuous_player_registration.json'
    for path in (target, launcher, report_path):
        shared._guard(path, project)
        if path.exists():
            raise FileExistsError(f'Preserve existing delivery: {path}')
    config = _configuration(before, stages)
    shared._write_new_json(target, config, project)
    caches = _prepare(project, base, target, pts, stages)
    lines = ['@echo off',
        'powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_3D_RESULTS.ps1" -Config "%~dp0..\\' +
        (shared.PROGRAMME / CONFIG).as_posix().replace('/', '\\') + '" %*',
        'exit /b %ERRORLEVEL%']
    shared._new_bytes(launcher, ('\r\n'.join(lines) + '\r\n').encode('ascii'), project)
    report = dict(evidence, config=str(target), config_sha256=shared._digest(target),
        original_config=str(paths['config']), original_config_sha256=shared._digest(paths['config']),
        old_variants_preserved=[row['id'] for row in before['variants']],
        default_variant=config['default_variant'], comparison_variants=list(COMPARISON), caches=caches,
        views=['free', 'side', 'end', 'top', 'support_top', 'support_side'],
        launch_command=f'"{launcher}" -Variant {config["default_variant"]} -Fresh',
        actual_hand_channel=True, saved_diagnostics=list(SAVED_FIELDS))
    shared._write_new_json(report_path, report, project)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'register', 'prepare'))
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--base-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    result = dict(check=check, register=register, prepare=prepare)[args.action](
        args.project, args.base_root, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
