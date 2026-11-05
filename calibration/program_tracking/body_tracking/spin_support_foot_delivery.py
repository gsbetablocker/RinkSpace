"""Explicit saved support-foot delivery; no fitting, inference or movie export."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .spin_support_delivery import (
    PROGRAMME, CASE, TIME_BASE, VIEWS, _load, _digest, _json, _clock,
    _guard, _write_new_json,
)

VARIANT = 'support_foot'
SECONDS = (84.4, 85.2, 89.56, 92.46, 93.2)
CONFIG = 'viewer3d_spin_support_foot_1104.json'
LAUNCHER = 'START_3D_SPIN_SUPPORT_FOOT.cmd'


@dataclass(frozen=True)
class DeliverySpec:
    """Saved-state bindings; defaults retain the original support-foot entry."""
    variant: str = VARIANT
    parameter_name: str = 'spin_support_foot_1104.npz'
    parameter_from_summary: bool = False
    source_config: str = 'viewer3d_spin_grasp_1104.json'
    config_name: str = CONFIG
    launcher_name: str = LAUNCHER
    baseline_variant: str = 'grasp_candidate'
    old_result: Path | None = None
    old_parameter: Path | None = None
    old_reference_case: Path | None = None
    label: str = '支撑脚位置续算（8步，待原片审核）'
    panel_labels: tuple[str, str] = ('原联合4步（失败保留）', '支撑脚位置续算（待审核）')
    seconds: tuple[float, ...] = SECONDS
    refinement_intervals: tuple[tuple[float, float], ...] = ((82., 94.8),)
    use_current_case_inputs: bool = False
    initial_source_seconds: float | None = None
    window_scope: str | None = None


def _saved(project, base, old, output, previous=None, spec=None):
    spec = spec or DeliverySpec()
    project, base, old, output = (Path(p).resolve() for p in (project, base, old, output))
    previous = Path(previous).resolve() if previous else old / 'support_joint_v2'
    old_reference = Path(spec.old_reference_case).resolve() if spec.old_reference_case else old
    paths = dict(
        parameter=project / PROGRAMME / spec.parameter_name,
        result=output / CASE / 'fitted/result.npz',
        old_result=old / CASE / 'finished/fitted/result.npz',
        active=old / 'active_candidate.json', summary=output / 'run_summary.json',
        contract=base / 'action_integration_1103/spin_s05/contract.json',
        config=project / PROGRAMME / spec.source_config,
        input=previous / CASE / 'input.npz', old_input=old_reference / CASE / 'input.npz',
    )
    summary = None
    if spec.parameter_from_summary:
        if not paths['summary'].is_file():
            raise FileNotFoundError(f"Required completed run summary is absent: {paths['summary']}")
        summary = _json(paths['summary'])
        parameter = Path(summary['parameter'])
        if not parameter.is_absolute():
            raise ValueError('Completed summary parameter path must be absolute')
        paths['parameter'] = parameter.resolve()
        if type(summary.get('accepted')) is not int or summary['accepted'] < 0:
            raise ValueError('Completed summary accepted count must be a nonnegative integer')
    if spec.old_result is not None:
        paths['old_result'] = Path(spec.old_result).resolve()
    if spec.old_parameter is not None:
        paths['old_parameter'] = Path(spec.old_parameter).resolve()
        del paths['active']
    if spec.use_current_case_inputs:
        paths.update(compact=previous / CASE / 'compact_smplx.npz',
                     support=previous / 'support_and_free_shoe_regions.json',
                     relations=previous / 'grasp_relations.json')
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'Required saved {name} is absent: {path}')
    summary = summary if summary is not None else _json(paths['summary'])
    for name in ('parameter', 'result'):
        if Path(summary[name]).resolve() != paths[name]:
            raise ValueError(f'Completed run summary refers to a different {name}')
    if 'active' in paths:
        active = _json(paths['active'])
        if Path(active['result']).resolve() != paths['old_result']:
            raise ValueError('Old active result is not the retained four-step result')
        paths['old_parameter'] = Path(active['parameter']).resolve()
    if not paths['old_parameter'].is_file():
        raise FileNotFoundError(paths['old_parameter'])
    contract = _json(paths['contract']); source = contract['source']
    pts = np.arange(972800, 1216000 + 1, 256, dtype=np.int64)
    if (contract['native50_count'] != len(pts)
            or tuple(source['time_base']) != TIME_BASE
            or source['pts_range'] != pts[[0, -1]].tolist()
            or source['native_size_wh'] != [1920, 1080]):
        raise ValueError('Expected all 951 original 76-95s source exposures')
    source_path = Path(source['path']).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    packets = {name: _load(paths[name]) for name in ('result', 'old_result')}
    for name, packet in packets.items():
        _clock(packet, pts, name)
        if 'hand_local_R_mid' not in packet:
            raise ValueError(f'{name}: actual saved hand channel missing')
    for name in ('input', 'old_input'):
        _clock(_load(paths[name]), pts, name)
    config = _json(paths['config'])
    if (Path(config['source_video']).resolve() != source_path
            or tuple(config['source_time_base']) != TIME_BASE):
        raise ValueError('Old player source identity/clock differs from contract')
    evidence = dict(
        schema='rinkspace.spin-saved-delivery.v1', variant=spec.variant, source=source,
        source_pts=pts.tolist(), source_frames=len(pts), source_time_base=list(TIME_BASE),
        same_pts_and_world=True, accepted_steps=summary.get('accepted'),
        cases={name: dict(path=str(paths[name]), sha256=_digest(paths[name]))
               for name in ('parameter', 'result', 'old_parameter', 'old_result')},
        reference_case=str(previous), old_reference_case=str(old_reference),
        no_new_fit=True, no_new_inference=True, no_alignment=True,
        native_camera_unchanged=True, accuracy_accepted=False,
    )
    return project, base, old, output, previous, paths, packets, config, pts, evidence


def _new_bytes(path, value, project):
    path = _guard(path, project)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(value)


def register(project, base, old, output, *, previous=None, spec=None):
    """Create a separate config/launcher, preserving every old player variant.

    Only the new variant's display cache is prepared. Nothing runs on import.
    The previous case defaults to old/support_joint_v2, matching the new state.
    """
    spec = spec or DeliverySpec()
    saved = _saved(project, base, old, output, previous, spec)
    project, base, old, output, previous, paths, packets, before, pts, evidence = saved
    target = project / PROGRAMME / spec.config_name; launcher = project / 'launchers' / spec.launcher_name
    report_path = output / (spec.variant + '_player_registration.json')
    for path in (target, launcher, report_path):
        _guard(path, project)
        if path.exists():
            raise FileExistsError(f'Preserve existing delivery: {path}')
    config = copy.deepcopy(before)
    ids = [v['id'] for v in config['variants']]
    if len(ids) != len(set(ids)) or spec.variant in ids:
        raise ValueError('Duplicate or already registered support-foot variant')
    original = next(v for v in config['variants'] if v['id'] == spec.baseline_variant)
    if Path(original['result']).resolve() != paths['old_result']:
        raise ValueError('Original player baseline variant differs')
    if 'gvhmr_fixed' not in ids:
        raise ValueError('Original GVHMR variant missing')
    variant = dict(id=spec.variant, label=spec.label.format(accepted=evidence['accepted_steps']),
                   kind='fullbody', result=str(paths['result']), shell_topology='saved',
                   body_reference_label='exmr_hand_nofoot',
                   body_refinement_intervals_seconds=[list(row) for row in spec.refinement_intervals])
    config['variants'].append(variant)
    config.update(default_variant=spec.variant,
                  comparison_variants=['gvhmr_fixed', spec.baseline_variant, spec.variant],
                  review_status='Saved continuation; source/3D accuracy not accepted')
    if spec.use_current_case_inputs:
        config.update(source_run=str(previous / CASE), support_observations=str(paths['support']),
                      relation_document=str(paths['relations']))
    if spec.initial_source_seconds is not None:
        config['initial_source_pts'] = round(spec.initial_source_seconds * 12800)
    if spec.window_scope is not None:
        config['body_window_scope'] = spec.window_scope
    _write_new_json(target, config, project)
    from rinkspace.storage import AssetStore
    from rinkspace.viewer3d.data import ResultCatalog
    from rinkspace.viewer3d.prepare import prepare_variant
    catalog = ResultCatalog(project=project, data_root=AssetStore(project).data, config=target)
    _clock(catalog.packet, pts, 'player input')
    if tuple(catalog.time_base) != TIME_BASE:
        raise ValueError('Player time base differs')
    cache_path, _ = catalog.cache_path(variant)
    if cache_path.exists():
        raise FileExistsError(f'Preserve existing support-foot cache: {cache_path}')
    cache = prepare_variant(catalog, variant)
    state = _load(cache / 'state.npz'); _clock(state, pts, 'player cache')
    np.testing.assert_array_equal(state['hand_local_R_mid'], packets['result']['hand_local_R_mid'])
    lines = ['@echo off',
             'powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_3D_RESULTS.ps1" -Config "%~dp0..\\' +
             (PROGRAMME / spec.config_name).as_posix().replace('/', '\\') + '" %*',
             'exit /b %ERRORLEVEL%']
    _new_bytes(launcher, ('\r\n'.join(lines) + '\r\n').encode('ascii'), project)
    report = dict(evidence, config=str(target), config_sha256=_digest(target),
                  original_config=str(paths['config']), original_config_sha256=_digest(paths['config']),
                  old_variants_preserved=ids, default_variant=spec.variant,
                  cache=str(cache), actual_hand_channel=True, views=VIEWS,
                  launch_command=f'"{launcher}" -Variant {spec.variant} -Fresh',
                  no_visible_gui_opened=True)
    _write_new_json(report_path, report, project)
    return report


def _check_state(ref, packet, frames):
    """Check only the saved states that are actually rendered."""
    state = ref.at(ref.times[frames])
    for key, actual in (
        ('P_world_mid', state['P'] * ref.binding.ratio),
        ('Q_world_mid', state['Q']), ('local_R_mid', state['local_R']),
        ('hand_local_R_mid', state['hand_local_R']),
    ):
        np.testing.assert_allclose(actual, packet[key][frames], rtol=0, atol=2e-10,
                                   err_msg=f'Restored parameter differs from saved result: {key}')


def render_stills(project, base, old, output, *, previous=None, spec=None):
    """Render only the requested full native mesh/shoe comparisons; never a movie."""
    spec = spec or DeliverySpec()
    saved = _saved(project, base, old, output, previous, spec)
    project, base, old, output, previous, paths, packets, config, pts, evidence = saved
    target = _guard(output / (spec.variant + '_native_stills'), project)
    if target.exists():
        raise FileExistsError(f'Preserve existing still delivery: {target}')
    requested = np.array([round(t * 12800) for t in spec.seconds], dtype=np.int64)
    frames = np.searchsorted(pts, requested)
    if not np.array_equal(pts[frames], requested):
        raise ValueError('Requested original source exposures absent')
    if _digest(evidence['source']['path']) != evidence['source']['sha256']:
        raise ValueError('Source video SHA256 differs from the original contract')
    from PIL import Image, ImageFont
    from rinkspace.viewer3d.source import SourceVideo
    from .support_root_placement import restore_parent
    from .spin_support_review import project_mesh
    from .native_mesh_overlay import paint
    target.mkdir(parents=True, exist_ok=False)
    parent, camera, _, _ = restore_parent(project, base, Path(evidence['old_reference_case']), target / 'old_geometry', paths['old_parameter'])
    new, new_camera, _, _ = restore_parent(project, base, previous, target / 'new_geometry', paths['parameter'])
    for key in ('rig', 'controls', 'motion_times'):
        np.testing.assert_array_equal(camera.p[key], new_camera.p[key])
    for ref, name in ((parent, 'old_result'), (new, 'result')):
        _clock(ref.p, pts, name); _check_state(ref, packets[name], frames)
    views = [project_mesh(ref, camera, frames) for ref in (parent, new)]
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 23)
    reader = SourceVideo(evidence['source']['path'], TIME_BASE); records = []
    try:
        for j, pt in enumerate(requested):
            raw = Image.fromarray(reader.read(int(pt)))
            if raw.size != (1920, 1080):
                raise ValueError('Native source frame size differs')
            canvas = Image.new('RGB', (3840, 1080))
            for column, (ref, view, label) in enumerate(zip(
                    (parent, new), views, tuple(label.format(accepted=evidence['accepted_steps'])
                                              for label in spec.panel_labels))):
                mesh, shoe, _, _, packet, nlower = view; su = shoe[j]
                parts = [su[0, :nlower], su[0, nlower:], su[1, :nlower], su[1, nlower:]]
                faces = np.asarray(packet['template']['faces'], dtype=np.int32)
                image = paint(raw, mesh[j], ref.binding.surface.faces, parts,
                              [faces, ref.binding.faces, faces, ref.binding.faces], label, font, int(pt))
                canvas.paste(image, (column * 1920, 0))
            path = target / f'compare_{int(pt)}.png'
            with path.open('xb') as stream:
                canvas.save(stream, format='PNG')
            records.append(dict(source_pts=int(pt), source_seconds=int(pt) / 12800.,
                                path=str(path), sha256=_digest(path)))
    finally:
        reader.close()
    report = dict(evidence, frames=records, exact_exposure_nodes_per_frame=5,
                  rendered_source_pts=requested.tolist(), source_sha256_checked=True,
                  parameter_result_midpoint_checks='P/Q/body/actual hands, all rendered frames',
                  projection='native moving_projection; Brown distortion and implicit rolling row; five actual body nodes',
                  geometry='complete SMPL-X surface and both attached lower/upper skate meshes',
                  display='two full 1920x1080 native panels; no crop or pose/image alignment',
                  no_movie_generated=True, no_visible_gui_opened=True)
    _write_new_json(target / 'projection_and_clock.json', report, project)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'stills'))
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--base-root', type=Path, required=True)
    parser.add_argument('--old-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous-output', type=Path)
    args = parser.parse_args(argv)
    action = register if args.action == 'register' else render_stills
    return action(args.project, args.base_root, args.old_output, args.output, previous=args.previous_output)


if __name__ == '__main__':
    main()