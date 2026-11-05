"""Deliver saved support-joint v2 states without fitting, inference or alignment.

Nothing runs on import. Metrics, player registration and the full native movie
are separate explicit calls; all require the real final result to exist.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np


PROGRAMME = Path('events/ankara_2026/programmes/okada_mei_sp')
CASE = Path('common/exmr_hand_nofoot')
TIME_BASE = (1, 12800)
PHASES = {
    'entry_84_86': (84., 86.),
    'held_86_8_92_46': (86.8, 92.46),
    'release_92_46_93_2': (92.46, 93.2),
}
VIEWS = ['free', 'support_top', 'support_side', 'grasp']


def _load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _guard(path, project):
    from rinkspace.path_policy import guard_project_write
    return guard_project_write(path, project_root=project)


def _write_new_json(path, value, project):
    path = _guard(path, project)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also preserves failed/partial earlier deliveries.
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return path


def _paths(project, base, old, output):
    project, base, old, output = map(Path, (project, base, old, output))
    paths = {
        'parent_joint4': old / CASE / 'finished/fitted/result.npz',
        'support_seed_v2': output / CASE / 'initial/fitted/result.npz',
        'support_joint_v2': output / CASE / 'fitted/result.npz',
        'parameter': project / PROGRAMME / 'spin_support_joint_v2_1104.npz',
        'observations': old / CASE / 'observations.npz',
        'new_observations': output / CASE / 'observations.npz',
        'contract': base / 'action_integration_1103/spin_s05/contract.json',
        'config': project / PROGRAMME / 'viewer3d_spin_grasp_1104.json',
    }
    # Check final first: never replace a missing final with candidate.npz,
    # the initial state, an accepted-step log or an old result.
    for name in ('support_joint_v2', 'parameter', 'parent_joint4',
                 'support_seed_v2', 'observations', 'new_observations', 'contract'):
        if not paths[name].is_file():
            raise FileNotFoundError(f'Required saved {name} is absent: {paths[name]}')
    return paths


def _clock(packet, pts, name):
    actual = np.asarray(packet['pts'])
    if actual.dtype.kind not in 'iu' or not np.array_equal(actual, pts):
        raise ValueError(f'{name}: source PTS differ')
    if 'source_pts' in packet and not np.array_equal(packet['source_pts'], pts):
        raise ValueError(f'{name}: source_pts alias differs')
    if 'time_base' in packet and not np.array_equal(packet['time_base'], TIME_BASE):
        raise ValueError(f'{name}: source time base differs')


def _saved_cases(project, base, old, output):
    paths = _paths(project, base, old, output)
    packets = {name: _load(paths[name]) for name in
               ('parent_joint4', 'support_seed_v2', 'support_joint_v2')}
    obs = _load(paths['observations'])
    pts = np.asarray(obs['pts'])
    contract = _json(paths['contract'])
    source = contract['source']
    if (pts.dtype.kind not in 'iu' or pts.shape != (951,)
            or int(contract['native50_count']) != len(pts)
            or tuple(source['time_base']) != TIME_BASE
            or source['pts_range'] != [int(pts[0]), int(pts[-1])]
            or [int(pts[0]), int(pts[-1])] != [972800, 1216000]
            or not np.all(np.diff(pts) == 256)):
        raise ValueError('Expected all 951 actual source exposures at 76-95s')
    new_obs = _load(paths['new_observations'])
    _clock(new_obs, pts, 'new observations')
    for key in ('uv', 'use'):
        if not np.array_equal(obs[key], new_obs[key], equal_nan=True):
            raise ValueError(f'Original observation {key} changed between cases')
    if obs['use'].dtype.kind != 'b' or obs['use'].shape != (951, 17):
        raise ValueError('Expected the original native COCO17 observation mask')
    for name, packet in packets.items():
        _clock(packet, pts, name)
        if 'hand_local_R_mid' not in packet:
            raise ValueError(f'{name}: actual saved hand channel is missing')
        uv = np.asarray(packet['COCO_uv'])
        if uv.shape != (951, 17, 2) or not np.isfinite(uv).all():
            raise ValueError(f'{name}: invalid native body projection')
    return paths, packets, obs, contract


def _stats(values):
    values = np.asarray(values, float).ravel()
    finite = values[np.isfinite(values)]
    return dict(count=int(len(finite)), missing=int(len(values) - len(finite)),
                median=None if not len(finite) else float(np.median(finite)),
                p95=None if not len(finite) else float(np.quantile(finite, .95)),
                max=None if not len(finite) else float(finite.max()))


def _palm_distance(packet):
    """Distance to the saved finite segment, never to a wrist or infinite line."""
    palm = np.asarray(packet['grasp_palm_world_mid'])
    segment = np.asarray(packet['grasp_segment_world_mid'])
    if palm.shape != (951, 3) or segment.shape != (951, 2, 3):
        raise ValueError('Invalid saved palm/finite-segment geometry')
    delta = segment[:, 1] - segment[:, 0]
    length2 = np.sum(delta * delta, axis=1)
    if not np.isfinite(palm).all() or not np.isfinite(segment).all() or np.any(length2 <= 0):
        raise ValueError('Nonfinite or degenerate palm/finite-segment geometry')
    coordinate = np.clip(np.sum((palm - segment[:, 0]) * delta, axis=1) / length2, 0, 1)
    distance = np.linalg.norm(palm - segment[:, 0] - coordinate[:, None] * delta, axis=1)
    np.testing.assert_allclose(distance, packet['palm_to_segment_m_mid'], rtol=0, atol=2e-10)
    return distance


def metrics(project, base, old, output):
    """Write support_phase_metrics.json for parent, v2 seed and real v2 final."""
    project, output = Path(project), Path(output)
    target = output / 'support_phase_metrics.json'
    if target.exists():
        raise FileExistsError(f'Preserve previous metrics: {target}')
    paths, packets, obs, contract = _saved_cases(project, base, old, output)
    pts = obs['pts']; cases = {}
    for name, packet in packets.items():
        error = np.linalg.norm(packet['COCO_uv'] - obs['uv'], axis=-1)
        palm = _palm_distance(packet); phases = {}
        for phase, (start, stop) in PHASES.items():
            mask = (pts >= round(start * 12800)) & (pts <= round(stop * 12800))
            root = np.asarray(packet['P_world_mid'])[mask]
            if not np.isfinite(root).all():
                raise ValueError(f'{name}: nonfinite root')
            selected = obs['use'] & mask[:, None]
            rows = dict(source_seconds=[start, stop], exposures=int(mask.sum()),
                        body_native_px=_stats(error[selected]),
                        palm_to_finite_segment_model_m=_stats(palm[mask]),
                        root_xy_span_model_m=np.ptp(root[:, :2], axis=0).tolist(),
                        root_xy_path_model_m=float(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1).sum()))
            for joint, index in (('left_ankle', 15), ('right_ankle', 16),
                                 ('left_wrist', 9), ('right_wrist', 10)):
                rows[joint + '_native_px'] = _stats(error[mask & obs['use'][:, index], index])
            # Witness Z is the raw geometry height. Semantic clearance includes
            # a policy allowance and is reported as a separate quantity.
            for key, values in (
                ('raw_model_minimum_height_m', packet['skate_witness_world_mid'][:, :, 2]),
                ('semantic_model_clearance_m', packet['skate_clearance_m_mid']),
                ('cavity_coordinate_breach_m', packet['foot_cavity_max_violation_m_mid']),
            ):
                rows[key] = {str(foot): _stats(values[mask, foot]) for foot in (0, 1)}
            rows['source_supported_raw_gap_m'] = _stats(packet['skate_support_gap_m_mid'][mask])
            for key in ('skate_contact_mode_mid', 'source_support_mode_mid'):
                if key in packet:
                    rows[key + '_unknown_foot_frames'] = int(np.count_nonzero(packet[key][mask] == 'unknown'))
                else:
                    rows[key + '_status'] = 'unknown: field absent in saved packet'
            phases[phase] = rows
        cases[name] = dict(result=str(paths[name]), sha256=_digest(paths[name]),
                           actual_hand_channel=True, phases=phases)
    report = dict(schema='rinkspace.spin-support-joint-delivery.v1',
                  source=contract['source'], source_pts=pts.tolist(), source_frames=len(pts),
                  source_time_base=list(TIME_BASE), same_951_pts=True,
                  original_observations=str(paths['observations']),
                  original_observations_sha256=_digest(paths['observations']),
                  same_original_uv_and_use_mask=True, cases=cases,
                  parameter=str(paths['parameter']), parameter_sha256=_digest(paths['parameter']),
                  interval_endpoints='inclusive; 92.46s is intentionally in held and release',
                  interpretation='Native anatomical-keypoint development residuals; conditional model metres. Palm patch to finite declared grip proxy is not measured contact. Root Pxy is model pelvis projection, not CoM or athlete displacement. Raw witness heights and approximate cavity coordinate breaches do not certify ice contact, flight, penetration or internal shoe fit.',
                  historical_failures_and_unknowns_preserved=True, accuracy_accepted=False,
                  no_new_fit=True, no_network=True, no_alignment=True)
    _write_new_json(target, report, project)
    return report


def register(project, base, old, output, *, make_default=False):
    """Append two variants and bake their real hand caches; leave default unless requested."""
    from rinkspace.storage import AssetStore
    from rinkspace.viewer3d.data import ResultCatalog
    from rinkspace.viewer3d.prepare import prepare_variant
    from .fullbody_round import write_json

    project, output = Path(project), Path(output)
    target = output / 'player_registration_v2.json'
    if target.exists():
        raise FileExistsError(f'Preserve previous registration report: {target}')
    paths, packets, obs, contract = _saved_cases(project, base, old, output)
    config_path = paths['config']; before = _json(config_path); config = copy.deepcopy(before)
    old_ids = [variant['id'] for variant in before['variants']]
    if len(set(old_ids)) != len(old_ids):
        raise ValueError('Existing player contains duplicate variant IDs')
    added = []
    for name, label, intervals in (
        ('support_seed_v2', '源支撑联合初值 v2（未验收）', []),
        ('support_joint_v2', '源支撑真实联合 v2（待原片审核）', [[82., 94.8]]),
    ):
        variant = dict(id=name, label=label, kind='fullbody', result=str(paths[name]),
                       shell_topology='saved', body_reference_label='exmr_hand_nofoot',
                       body_refinement_intervals_seconds=intervals)
        existing = next((v for v in config['variants'] if v['id'] == name), None)
        if existing is not None and existing != variant:
            raise ValueError(f'Preserve existing conflicting player variant: {name}')
        if existing is None:
            config['variants'].append(variant)
        added.append(variant)
    if make_default:
        config['default_variant'] = 'support_joint_v2'
        config['comparison_variants'] = ['grasp_candidate', 'support_seed_v2', 'support_joint_v2']
    # The old source_run, physical camera/scene, controls, evidence and failed
    # variants remain byte-for-byte equal as JSON values.
    _guard(config_path, project); _guard(config_path.with_name(config_path.name + '.pending'), project)
    write_json(config_path, config)
    catalog = ResultCatalog(project=project, data_root=AssetStore(project).data, config=config_path)
    np.testing.assert_array_equal(catalog.pts, obs['pts'])
    if tuple(catalog.time_base) != TIME_BASE:
        raise ValueError('Player source time base differs')
    caches = []
    for variant in added:
        cache = prepare_variant(catalog, variant)
        state = _load(cache / 'state.npz')
        np.testing.assert_array_equal(state['pts'], obs['pts'])
        np.testing.assert_array_equal(state['hand_local_R_mid'], packets[variant['id']]['hand_local_R_mid'])
        caches.append(dict(variant=variant['id'], path=str(cache), actual_hand_channel=True))
    report = dict(config=str(config_path), caches=caches, views=VIEWS,
                  source_run=before['source_run'], source_frames=951,
                  same_pts_and_world=True, native_camera_unchanged=True,
                  previous_default=before['default_variant'], default_variant=config['default_variant'],
                  made_default=bool(make_default), old_variants_preserved=old_ids,
                  launch_commands={v['id']: f'"{project / "launchers" / "START_3D_GRASP.cmd"}" -Variant {v["id"]}' for v in added},
                  no_visible_gui_opened=True, accuracy_accepted=False)
    _write_new_json(target, report, project)
    return report


def _source_crops(boxes, pts):
    boxes = np.asarray(boxes, float)
    if boxes.shape != (len(pts), 4):
        raise ValueError('Invalid original source_bbox shape')
    valid = np.isfinite(boxes).all(1) & np.all(boxes[:, 2:] > boxes[:, :2], axis=1)
    if not valid.any():
        raise ValueError('No valid source_bbox for shared display crop')
    centres = np.interp(pts, pts[valid], boxes[valid, [0]] .ravel() / 2 + boxes[valid, [2]].ravel() / 2)
    crops = [(round(x - 360), 0, round(x + 360), 1080) for x in centres]
    return crops, pts[~valid].tolist()


def _verify_encoded(path, source_pts):
    """Decode and compare every encoded timestamp using exact rational times."""
    import av
    rows = []
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index >= len(source_pts) or frame.pts is None or frame.time_base is None:
                raise ValueError(f'Unexpected encoded frame/PTS: {path}, index {index}')
            encoded_time = int(frame.pts) * frame.time_base
            expected = (int(source_pts[index]) - int(source_pts[0])) * Fraction(*TIME_BASE)
            if encoded_time != expected:
                raise ValueError(f'Encoded PTS differs at frame {index}: {encoded_time} != {expected}')
            rows.append(dict(index=index, encoded_pts=int(frame.pts),
                             encoded_time_base=[frame.time_base.numerator, frame.time_base.denominator],
                             source_pts=int(source_pts[index])))
    if len(rows) != len(source_pts):
        raise ValueError(f'Encoded frame count differs: {path}: {len(rows)} != {len(source_pts)}')
    return dict(path=str(path), sha256=_digest(path), frames=len(rows), every_encoded_pts_checked=True,
                source_time_offset_pts=int(source_pts[0]), encoded_pts=rows)


def native_movie(project, base, old, output):
    """Render all 951 native exposures; call explicitly only after final completion.

    Writes support_joint_v2/native_camera_overlay without overwriting previous
    MP4/PNG/report files. Uses the actual saved local_R/hands for each body.
    """
    import av
    from PIL import Image, ImageDraw, ImageFont
    from rinkspace.viewer3d.source import SourceVideo
    from .spin_support_joint import prepare
    from .spin_support_review import project_mesh
    from .support_root_placement import restore_parent
    from .skate_integration import restore_checkpoint_layers
    from .native_mesh_overlay import paint, open_encoder

    project, base, old, output = map(Path, (project, base, old, output))
    paths, packets, obs, contract = _saved_cases(project, base, old, output)
    target = output / 'native_camera_overlay'
    videos = [target / 'support_joint_v2_native_76_95.mp4',
              target / 'parent_seed_joint_native_76_95.mp4']
    proof_pts = {round(value * 12800) for value in (81.2, 84.4, 85.2, 86.8, 89.56, 92.46, 92.8, 93.2)}
    products = [*videos, target / 'projection_and_clock.json', target / 'render_failure.json']
    products += [target / f'{prefix}_{pt}.png' for pt in proof_pts for prefix in ('native', 'compare')]
    for path in products:
        _guard(path, project)
        if path.exists():
            raise FileExistsError(f'Preserve existing native delivery: {path}')
    source = contract['source']; source_path = Path(source['path'])
    if tuple(source['time_base']) != TIME_BASE or source['native_size_wh'] != [1920, 1080]:
        raise ValueError('Source contract differs from native 1920x1080 / 1:12800')
    if _digest(source_path) != source['sha256']:
        raise ValueError('Actual source video SHA256 differs from source contract')
    active = _json(old / 'active_candidate.json')
    if Path(active['result']).resolve() != paths['parent_joint4'].resolve():
        raise ValueError('Old active parent is not the retained four-step result')
    target.mkdir(parents=True, exist_ok=True)
    new, camera, info, document, relations = prepare(project, base, old, output)
    restore_checkpoint_layers(new, _load(paths['parameter']), include_primary=True)
    parent, parent_camera, parent_info, _ = restore_parent(
        project, base, old, target / 'parent_geometry', Path(active['parameter']))
    pts = obs['pts']
    for name, ref in (('parent', parent), ('joint_v2', new)):
        _clock(ref.p, pts, name)
    for key in ('rig', 'controls', 'motion_times'):
        np.testing.assert_array_equal(parent_camera.p[key], camera.p[key])
    np.testing.assert_array_equal(parent.p['source_bbox'], new.p['source_bbox'])
    crops, missing_bbox = _source_crops(new.p['source_bbox'], pts)
    # Verify restored mid-exposure geometry against both actual saved results
    # before expanding the much larger FullSurface on five exposure nodes.
    for ref, packet in ((parent, packets['parent_joint4']), (new, packets['support_joint_v2'])):
        state = ref.at(ref.times)
        np.testing.assert_allclose(state['P'] * ref.binding.ratio, packet['P_world_mid'], rtol=0, atol=2e-10)
        np.testing.assert_allclose(state['local_R'], packet['local_R_mid'], rtol=0, atol=2e-10)
        np.testing.assert_allclose(state['hand_local_R'], packet['hand_local_R_mid'], rtol=0, atol=2e-10)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 22)
    labels = ['原联合4步（失败保留）', '支撑真实联合 v2（待审核）']
    reader = None; encoders = []; clock = []; proof = {}; started = time.perf_counter()
    try:
        for path in videos:
            container, stream = open_encoder(path, 1920, 1080)
            stream.time_base = Fraction(1, 50)
            encoders.append((container, stream))
        reader = SourceVideo(source_path, TIME_BASE)

        def encode(pair, image, index):
            container, stream = pair
            frame = av.VideoFrame.from_image(image)
            frame.pts = index; frame.time_base = Fraction(1, 50)
            for packet in stream.encode(frame):
                container.mux(packet)

        for first in range(0, len(pts), 4):
            frames = np.arange(first, min(first + 4, len(pts)))
            # The two local_R trajectories differ: project each complete body,
            # its real hand channel and attached shoes independently.
            projected = [project_mesh(ref, camera, frames) for ref in (parent, new)]
            for j, frame in enumerate(frames):
                pt = int(pts[frame]); raw = Image.fromarray(reader.read(pt))
                if raw.size != (1920, 1080):
                    raise ValueError('Decoded actual source size differs')
                overlays = []
                for ref, view, label in zip((parent, new), projected, labels):
                    mesh, shoes, mids, grip, packet, lower_count = view
                    su = shoes[j]
                    parts = [su[0, :lower_count], su[0, lower_count:], su[1, :lower_count], su[1, lower_count:]]
                    lower_faces = np.asarray(packet['template']['faces'], int)
                    image = paint(raw, mesh[j], ref.binding.surface.faces, parts,
                                  [lower_faces, ref.binding.faces, lower_faces, ref.binding.faces], label, font, pt)
                    palm, a, b = grip[j]; draw = ImageDraw.Draw(image)
                    draw.line([tuple(a), tuple(b)], fill=(255, 90, 240), width=3)
                    draw.ellipse((palm[0]-5, palm[1]-5, palm[0]+5, palm[1]+5), outline=(255, 90, 240), width=3)
                    overlays.append(image)
                canvas = Image.new('RGB', (1920, 1080), (12, 16, 23))
                for col, image in enumerate([raw, *overlays]):
                    canvas.paste(image.crop(crops[frame]).resize((640, 960)), (640 * col, 50))
                draw = ImageDraw.Draw(canvas)
                for col, label in enumerate(['原片', *labels]):
                    draw.text((640 * col + 12, 10), label, font=font, fill='white')
                draw.text((12, 1024), f'原片 {pt/12800.:.2f}s PTS {pt}；同源裁切／原 native Brown 相机／五曝光节点', font=font, fill='white')
                encode(encoders[0], overlays[1], len(clock))
                encode(encoders[1], canvas, len(clock))
                if pt in proof_pts:
                    native = target / f'native_{pt}.png'; compare = target / f'compare_{pt}.png'
                    overlays[1].save(native); canvas.save(compare)
                    proof[str(pt)] = dict(native=str(native), comparison=str(compare))
                clock.append(pt)
            if first == 0 or first % 80 == 0:
                print(json.dumps(dict(event='support_joint_v2_native_overlay', frames=len(clock), total=951)), flush=True)
        for container, stream in encoders:
            for packet in stream.encode():
                container.mux(packet)
            container.close()
    except Exception as exc:
        _write_new_json(target / 'render_failure.json',
                        dict(error=repr(exc), completed_source_pts=clock,
                             partial_files_preserved=True, accuracy_accepted=False), project)
        raise
    finally:
        if reader is not None:
            reader.close()
        for container, stream in encoders:
            try:
                container.close()
            except Exception:
                pass
    if clock != pts.tolist():
        raise ValueError('Rendered exposure clock differs from all actual source PTS')
    checks = [_verify_encoded(path, pts) for path in videos]
    report = dict(schema='rinkspace.spin-support-native-overlay.v1', source=source,
                  contract=str(paths['contract']), contract_sha256=_digest(paths['contract']),
                  source_pts=clock, source_time_base=list(TIME_BASE), source_seconds=[76., 95.],
                  source_frames=951, every_source_exposure=True, encoded_hz=50, encoded=checks,
                  comparison_columns=['source', 'parent_joint4', 'support_joint_v2'],
                  comparison_crop_xyxy=[list(crop) for crop in crops], source_bbox_missing_pts=missing_bbox,
                  crop_missing_policy='Source-bbox centre interpolation for display only; missing observations remain unknown',
                  proof=proof, parameter=str(paths['parameter']), parameter_sha256=_digest(paths['parameter']),
                  result_sha256={name: _digest(paths[name]) for name in packets},
                  full_surface_vertices=len(new.binding.surface.shaped),
                  full_surface_faces=len(new.binding.surface.faces),
                  exposure_nodes_per_body_per_frame=5, independent_body_node_evaluations=2 * 5 * 951,
                  projection='spin_support_review.project_mesh: actual FullSurface/hands plus attached lower/upper skates; native Brown and original five-node implicit rolling-row camera',
                  display='native_mesh_overlay.paint triangle-union silhouette; source occluders are not segmentation truth',
                  native_camera_unchanged=True, world_alignment=False, no_new_fit=True, no_network=True,
                  historical_failures_and_unknowns_preserved=True, accuracy_accepted=False,
                  seconds=time.perf_counter()-started)
    _write_new_json(target / 'projection_and_clock.json', report, project)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--base-root', type=Path, required=True)
    parser.add_argument('--previous-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['metrics', 'register', 'movie'], required=True)
    parser.add_argument('--make-default', action='store_true', help='Only for explicit player registration')
    args = parser.parse_args(argv)
    if args.make_default and args.stage != 'register':
        parser.error('--make-default applies only to --stage register')
    from rinkspace.storage import AssetStore
    store = AssetStore(args.project)
    base, old, output = [store.resolve(value, base=store.data, must_exist=True)
                         for value in (args.base_root, args.previous_output, args.output)]
    call = {'metrics': metrics, 'register': register, 'movie': native_movie}[args.stage]
    kwargs = dict(make_default=args.make_default) if args.stage == 'register' else {}
    report = call(args.project, base, old, output, **kwargs)
    print(json.dumps(dict(event='support_delivery_ready', stage=args.stage,
                          source_frames=report.get('source_frames', 951)), ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    main()
