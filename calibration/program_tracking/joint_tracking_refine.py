"""Joint iSAM2 refinement of shared mounts and verified live-broadcast seams.

Only explicitly reviewed same-foot contacts cross a camera cut. Static image
matches remain within their original shot; all source curves remain actual
measured pixels. The rink geometry stays unchanged and outputs are candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import io
import json
from pathlib import Path
import time

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np

from backend import IncrementalCameraGraph
from ice_refine import (_closed_ring, _curve_error, _curve_vector_error,
                        finite_curve_distance, _world_curves)
from keyframe_refine import source_features, build_links
from projection import to_state, from_state, project_fast
from render_result import DEFAULT_SCENE, DEFAULT_RELEASE, DEFAULT_SHOTS, read_scene
from track_sequence import (CameraAnchor, PairConstraint, load_anchors,
                            fast_world_rays, _lens_reference, _lens_factor, _aspect_factor,
                            SurfacePlanes, _first_hit_planes, _distributed_indices,
                            SourceEdges, _add_source_edge_factors, _pair_data, prepare_ray_pair,
                            METRIC_DTYPE, _frame_quality)
from rig_motion import add_rig_equality
from robust_observations import pointwise_huber
from correlated_motion import wrap_correlated_pair
from contact_seam import ground_whitener, shared_ground_diagnostic
from observation_weighting import ObservationQuality
import acceleration

ROOT = Path(__file__).resolve().parents[2]


def _ground(camera, pixel):
    ray = fast_world_rays(camera, np.asarray(pixel).reshape(1, 2))[0]
    # The selected visible foot points are below the horizon. Reject a
    # catastrophic trial instead of silently clipping it onto the rink.
    if ray[2] >= -1e-8:
        raise ValueError('Verified support-foot ray crossed the camera horizon')
    return camera[:2] - camera[2] * ray[:2] / ray[2]


def _mount_equality(graph, pts, mount):
    derivative = np.zeros((3, 11))
    derivative[:, :3] = np.eye(3)
    graph.add_observation([pts, mount], lambda q: q[0][:3]-q[1][:3],
                          residual_dim=3, sigma=1e-7,
                          jac_fn=lambda q: [derivative, -derivative])


def _surface_links(records, cameras, planes):
    """Use actual image pairs only where both views identify one rink surface.

    Plane membership is a conditional geometric association, not a measured
    XYZ. Its source-ray intersection changes as the camera is optimized.
    Unknown-depth spectators and pixels above the boards are excluded.
    """
    from surface_pair import SurfacePairConstraint
    output = []
    for a, b, pair in records:
        count = len(pair.xy0)
        classifications = []
        for camera, pixels in ((cameras[a], pair.xy0), (cameras[b], pair.xy1)):
            point, normal = np.zeros((count, 3)), np.zeros((count, 3))
            identity = np.zeros(count, np.uint8)
            for label in (1, 2):
                p, n, valid = _first_hit_planes(planes, camera, pixels, np.full(count, label))
                point[valid], normal[valid], identity[valid] = p[valid], n[valid], label
            classifications.append((point, normal, identity))
        p0, n0, s0 = classifications[0]
        p1, n1, s1 = classifications[1]
        same = (s0 > 0) & (s0 == s1)
        same &= (np.sum(n0*n1, axis=1) > .998)
        same &= (np.abs(np.sum((p0-p1)*n0, axis=1)) < .15)
        if same.sum() < 8:
            continue
        output.append((a, b, SurfacePairConstraint(
            pair.xy0[same], pair.xy1[same], p0[same], n0[same], s0[same],
            cameras[a].copy(), False)))
    return output


def _bridge_rotation_initial(cameras, anchors, records):
    """Initialize rotations from measured native motion, keeping calibrated K.

    The derived motion remains a separate graph factor. This propagation is
    only an initializer and never upgrades a frame's evidence/acceptance state.
    """
    adjacency = {p: [] for p in cameras}
    for a, b, pair, _ in records:
        adjacency[a].append((b, pair.xy0, pair.xy1))
        adjacency[b].append((a, pair.xy1, pair.xy0))
    queue = [(0, p) for p in anchors]
    heapq.heapify(queue)
    distance = {p: 0 for p in anchors}
    for p in anchors:
        cameras[p][3:6] = anchors[p].camera[3:6]
    while queue:
        cost, a = heapq.heappop(queue)
        if cost != distance[a]:
            continue
        for b, uv0, uv1 in adjacency[a]:
            new_cost = cost + abs(b-a)
            if new_cost >= distance.get(b, np.inf):
                continue
            rays = []
            for c, uv in ((cameras[a], uv0), (cameras[b], uv1)):
                r = fast_world_rays(c, uv) @ cv2.Rodrigues(c[3:6])[0].T
                r /= np.linalg.norm(r, axis=1)[:, None]
                rays.append(r)
            u, _, vt = np.linalg.svd(rays[0].T@rays[1])
            fix = np.eye(3)
            fix[2, 2] = np.linalg.det(u@vt)
            relative = (u@fix@vt).T
            rotation = relative@cv2.Rodrigues(cameras[a][3:6])[0]
            cameras[b][3:6] = cv2.Rodrigues(rotation)[0].ravel()
            distance[b] = new_cost
            heapq.heappush(queue, (new_cost, b))
    return len(distance)


def _components(stamps, edges):
    neighbours = {int(p): set() for p in stamps}
    for a, b in edges:
        neighbours[a].add(b)
        neighbours[b].add(a)
    unseen, groups = set(neighbours), []
    while unseen:
        todo = [unseen.pop()]
        group = set(todo)
        while todo:
            for other in neighbours[todo.pop()] & unseen:
                unseen.remove(other)
                group.add(other)
                todo.append(other)
        groups.append(sorted(group))
    return groups


def _unique_pair_pixels(source, target):
    """SIFT scales/orientations at one pixel pair are one observation."""
    points = np.column_stack((source, target))
    _, indices = np.unique(points, axis=0, return_index=True)
    indices.sort()
    return source[indices], target[indices], len(points)-len(indices)


def _warm_start(directory, guesses, shots, mount_keys, lens_keys, ad_keys,
                source_hash, model_hash, templates):
    """Reuse a completed solve without moving any original prior targets.

    Called only after factors have been built. Legacy exports can reconstruct
    the shared nodes from their diagnostics; current exports save all states.
    """
    directory = Path(directory)
    report = json.loads((directory/'diagnostics.json').read_text(encoding='utf8'))
    if (report['source_sha256'] != source_hash or
            report.get('base_model_sha256', report['model_sha256']) != model_hash):
        raise ValueError('Warm start uses a different source or scene')
    states = {key: value.copy() for key, value in guesses.items()}
    for shot in shots:
        with np.load(directory/f'shot{shot["index"]:02d}_keyframes.npz', allow_pickle=False) as data:
            old_pts = np.asarray(data['pts'], np.int64)
            target_pts = np.asarray(shot['pts'], np.int64)
            if (not set(map(int, old_pts)).issubset(map(int, target_pts)) or
                    old_pts[0] != target_pts[0] or old_pts[-1] != target_pts[-1] or
                    not np.all(np.diff(old_pts) > 0) or
                    str(data['source_sha256']) != source_hash or
                    str(data['model_sha256']) != report['model_sha256']):
                raise ValueError('Warm-start native PTS/source/model differ')
            # When an additional shot is expanded to native frames, propagate
            # only an initializer correction from its already-solved keys.
            # The new states never become pose priors or image observations.
            old_states = np.asarray([to_state(c) for c in data['camera25']])
            correction = old_states-np.asarray([guesses[int(p)] for p in old_pts])
            missing = target_pts[~np.isin(target_pts, old_pts)]
            for column in range(11):
                delta = np.interp(missing, old_pts, correction[:, column])
                for p, value in zip(missing, delta):
                    states[int(p)][column] += value
            for stamp, camera in zip(data['pts'], data['camera25']):
                stamp = int(stamp)
                if not np.allclose(camera[11:25], templates[stamp][11:25], atol=1e-12, rtol=0):
                    raise ValueError('Warm start must preserve the calibrated Brown template')
                states[stamp] = to_state(camera)
    for name, key in mount_keys.items():
        values = report['camera_mounts'][name]
        states[key][:3] = np.asarray(values['pivot_xyz'])/100.
        states[key][3:6] = np.asarray(values['camera_offset_xyz'])/100.
    for name, key in lens_keys.items():
        values = report['shared_lens'][name]
        states[key][6] = np.log(values['fx_fy_ratio'])
        states[key][8:11] = np.asarray(values['cx_cy_skew_px'])/1000.
    for number, key in ad_keys.items():
        values = report['ad_geometry_candidates'][str(number)]
        states[key][:4] = np.array([
            values['boundary_arc_shift_scene_units']['left'],
            values['boundary_arc_shift_scene_units']['right'],
            values['height_shift_scene_units']['bottom'],
            values['height_shift_scene_units']['top']])/100.
    if not all(value.shape == (11,) and np.isfinite(value).all() for value in states.values()):
        raise ValueError('Warm-start states must be finite 11-vectors')
    return states


def _resume_initial_states(specification, guesses, source_hash, base_model_hash):
    """Resume solved graph vectors after all frozen likelihoods are prepared.

    This changes initial values only.  The earlier warm_start still determines
    the original reference covariance, and prior targets are never moved.
    """
    folder = Path(specification['directory'])
    paths = {'states': folder/'graph_states.npz', 'diagnostics': folder/'diagnostics.json'}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    for name, digest in hashes.items():
        if digest != specification[name+'_sha256']:
            raise ValueError('Resume graph '+name+' hash changed')
    report = json.loads(paths['diagnostics'].read_text(encoding='utf8'))
    if (report['source_sha256'] != source_hash or
            report.get('base_model_sha256', report['model_sha256']) != base_model_hash):
        raise ValueError('Resume graph source/base model differs')
    with np.load(paths['states'], allow_pickle=False) as packet:
        keys = packet['keys'].copy()
        states = packet['states'].copy()
        if (str(packet['source_sha256']) != source_hash or
                str(packet['model_sha256']) != report['model_sha256'] or
                keys.ndim != 1 or len(set(map(int, keys))) != len(keys) or
                states.shape != (len(keys), 11) or not np.isfinite(states).all() or
                set(map(int, keys)) != set(guesses)):
            raise ValueError('Resume graph vector identities or values differ')
    result = {int(key): state for key, state in zip(keys, states)}
    return result, {'directory': str(folder.resolve()), **hashes,
                    'source_sha256': source_hash, 'model_sha256': report['model_sha256'],
                    'base_model_sha256': base_model_hash,
                    'policy': 'initial_values_only_after_reference_covariance; priors_unchanged'}


def solve(config_path: Path, output: Path, *, allow_project_output: bool = False):
    total_started = time.perf_counter()
    stage_started = total_started
    stage_cpu = time.process_time()
    timings = {}
    if output.resolve().is_relative_to(ROOT) and not allow_project_output:
        raise ValueError('Joint candidate cache must remain outside OneDrive')
    output.mkdir(parents=True, exist_ok=True)
    code_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in Path(__file__).parent.rglob('*.py')}
    native_source = acceleration.SOURCE
    code_hashes[str(native_source.relative_to(ROOT))] = hashlib.sha256(native_source.read_bytes()).hexdigest()
    from backend.callback_factory import SOURCE as callback_source
    code_hashes[str(callback_source.relative_to(ROOT))] = hashlib.sha256(callback_source.read_bytes()).hexdigest()
    (output/'code_sha256_at_start.json').write_text(json.dumps(code_hashes, indent=2), encoding='utf8')

    def finish_stage(name):
        nonlocal stage_started, stage_cpu
        now, cpu = time.perf_counter(), time.process_time()
        timings[name] = {'wall_seconds': now-stage_started, 'process_cpu_seconds': cpu-stage_cpu}
        stage_started, stage_cpu = now, cpu
        (output/'timings.json').write_text(json.dumps({
            'stages': timings, 'total_wall_seconds': now-total_started,
            'acceleration': acceleration.metadata()}, indent=2), encoding='utf8')
        print(json.dumps({'stage_completed': name, **timings[name]}), flush=True)

    config = json.loads(config_path.read_text(encoding='utf8'))
    atlas_anchor_mode = config.get('atlas_anchor_mode', 'legacy_fixed_xyz')
    if atlas_anchor_mode not in ('legacy_fixed_xyz', 'initialization_only'):
        raise ValueError(f'Unknown atlas_anchor_mode: {atlas_anchor_mode!r}')
    curve_mode = config.get('curve_residual', 'supporting_line')
    if curve_mode not in ('supporting_line', 'finite_vector'):
        raise ValueError('Unknown curve_residual mode')
    finite_vector = curve_mode == 'finite_vector'
    curve_fn = _curve_vector_error if finite_vector else _curve_error
    curve_dim = 2 if finite_vector else 1
    edge_search = config.get('source_edge_candidates', 'initializer')
    if edge_search not in ('initializer', 'current_state'):
        raise ValueError('Unknown source_edge_candidates mode')
    pivot = config.get('camera_motion', 'fixed') == 'pivot'
    surface_pairs = config.get('pair_model', 'surface' if pivot else 'rays') == 'surface'
    pointwise = config.get('robust_groups', 'pointwise') == 'pointwise'
    if pivot and not surface_pairs:
        raise ValueError('Moving optical centres require actual surfaces, not pure rotation rays')
    source_file = Path(config.get('source_shots', DEFAULT_SHOTS))
    source = json.loads(source_file.read_text(encoding='utf8'))
    scene, release = read_scene(Path(config.get('scene', DEFAULT_SCENE)),
                                Path(config.get('release', DEFAULT_RELEASE)))
    model_hash, source_hash = release['model_sha256'], source['source_sha256_from_timeline']
    motion = json.loads(Path(config['motion_index']).read_text(encoding='utf8'))
    people = json.loads(Path(motion['people_boxes']).read_text(encoding='utf8'))
    contacts = json.loads(Path(config['cut_contacts']).read_text(encoding='utf8'))
    if contacts['source_sha256'] != source_hash:
        raise ValueError('Contact annotations belong to another video')
    world = _world_curves(scene)
    world['board_top'] = _closed_ring(scene['board_surface']['top_edge_xyz'])
    ring_scene = None
    if config.get('shared_rings'):
        from shared_scene_calibration import SharedRingScene
        from shared_markers import add_shared_ring_factor
        ring_scene = SharedRingScene(scene, config['shared_rings'])
    planes = SurfacePlanes.from_scene(scene)
    anchors, anchor_files, anchor_input_hashes = {}, {}, {}
    for file in config['anchor_manifests']:
        path = Path(file)
        anchor_input_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        loaded = load_anchors(path, model_sha256=model_hash,
                              accepted_statuses={'camera_candidate', 'joint_camera_candidate_unvalidated',
                                                 'accepted', 'validated', 'camera_anchor'})
        if set(loaded) & anchors.keys():
            raise ValueError('Duplicate absolute anchor PTS')
        anchors.update(loaded)
        anchor_files.update({stamp: path.parent/f'{stamp}.npz' for stamp in loaded})
        for stamp in loaded:
            asset = anchor_files[stamp]
            anchor_input_hashes[str(asset.resolve())] = hashlib.sha256(asset.read_bytes()).hexdigest()
    templates, initial, curve_groups, shots, links = {}, {}, {}, [], []
    curve_sigmas = {}
    observation_inputs = dict(anchor_input_hashes)
    source_quality = ObservationQuality(config.get('observation_quality'), source_hash,
                                        observation_inputs)
    source_precision = None
    original_source_precision = None
    if config.get('observation_uncertainty'):
        from source_precision import SourcePixelPrecision
        source_precision = SourcePixelPrecision(config['observation_uncertainty'],
                                                source_hash, observation_inputs)
        if not finite_vector:
            raise ValueError('Source normal uncertainty requires finite curve vectors')
    if config.get('original_observation_uncertainty'):
        from source_precision import SourcePixelPrecision
        original_source_precision = SourcePixelPrecision(config['original_observation_uncertainty'],
                                                         source_hash, observation_inputs)
        if not finite_vector:
            raise ValueError('Original source normal uncertainty requires finite curve vectors')
    native_links, native_reports, raw_native_links = [], [], []
    native_edge_inputs, original_keyframes = {}, set()
    duplicate_direct_pixels = 0
    reviewed_link_exclusions = {tuple(row['pts']): row
                                for row in config.get('source_link_exclusions', [])}
    applied_link_exclusions = []
    for item in config['shots']:
        index = int(item['shot_index'])
        shot = source['shots'][index]
        with np.load(item['keyframes'], allow_pickle=False) as data:
            if str(data['model_sha256']) != model_hash or str(data['source_sha256']) != source_hash:
                raise ValueError('Keyframe source/model hash differs')
            stamps, cameras = data['pts'].copy(), data['camera25'].copy()
            keyframe_stamps = (data['original_keyframe_pts'].copy()
                               if 'original_keyframe_pts' in data.files else stamps.copy())
            legacy_native = Path(item['keyframes']).with_name(f'shot{index:02d}.npz')
            if 'original_keyframe_pts' not in data.files and legacy_native.exists():
                with np.load(legacy_native, allow_pickle=False) as prior_identity:
                    if 'original_keyframe_pts' in prior_identity.files:
                        if (not np.array_equal(prior_identity['pts'], stamps) or
                                str(prior_identity['source_sha256']) != source_hash or
                                str(prior_identity['model_sha256']) != model_hash):
                            raise ValueError('Original-keyframe identity sidecar differs from candidate')
                        keyframe_stamps = prior_identity['original_keyframe_pts'].copy()
            if not set(map(int, keyframe_stamps)).issubset(map(int, stamps)):
                raise ValueError('Original keyframe PTS must occur in candidate states')
            if stamps[0] != shot['start_pts'] or stamps[-1] != shot['end_pts']-256:
                raise ValueError('Joint keyframes must include both actual cut boundary frames')
            if np.max(np.abs(cameras[:, 11:25]-cameras[0, 11:25])) > 1e-10:
                raise ValueError('Each shot must retain its one calibrated Brown template')
            template = cameras[0].copy()
            own = {p: a for p, a in anchors.items() if p in set(stamps.tolist())}
            if not own:
                raise ValueError(f'Shot {index} has no model-bound anchor')
            pair_records = []
            if {'link_pts', 'link_offsets', 'link_xy0', 'link_xy1'}.issubset(data.files):
                for i, (a, b) in enumerate(data['link_pts']):
                    left, right = data['link_offsets'][i:i+2]
                    p, q = data['link_xy0'][left:right].copy(), data['link_xy1'][left:right].copy()
                    excluded = reviewed_link_exclusions.get((int(a), int(b)))
                    if excluded is not None:
                        digest = hashlib.sha256(np.ascontiguousarray(
                            np.column_stack((p, q)), dtype=np.float64).tobytes()).hexdigest()
                        if (excluded['source_sha256'] != source_hash or
                                excluded['pixel_pairs_sha256'] != digest or
                                not excluded.get('source_evidence') or not excluded.get('reason')):
                            raise ValueError('Reviewed wrong-link exclusion does not match its original pixels')
                        applied_link_exclusions.append({**excluded, 'rows': len(p)})
                        continue
                    p, q, duplicates = _unique_pair_pixels(p, q)
                    duplicate_direct_pixels += duplicates
                    pair = PairConstraint(p, q, np.zeros((len(p), 3)), np.zeros((len(p), 3)),
                                           np.full(len(p), 3, np.uint8), template, True)
                    pair_records.append((int(a), int(b), pair))
            else:
                features = source_features(motion['shots'][index], stamps,
                                            Path(source['source_path']), people)
                features = {p: features[p] for p in map(int, stamps)}
                pair_records = build_links(features, template, own)
        original_keyframes.update(map(int, keyframe_stamps))
        raw_pairs = []
        if item.get('native_tracks'):
            if pivot:
                raise ValueError('Native fixed-mount rays require fixed optical centres')
            path = Path(item['native_tracks'])
            with np.load(path, allow_pickle=False) as native:
                if (str(native['model_sha256']) != model_hash or
                        str(native['source_sha256']) != source_hash):
                    raise ValueError('Native initializer source/model differs')
                all_stamps, all_cameras = native['pts'].copy(), native['camera25'].copy()
            expected = np.arange(shot['start_pts'], shot['end_pts'], 256, dtype=np.int64)
            if (not np.array_equal(all_stamps, expected) or
                    all_cameras.shape != (len(expected), 25) or
                    not np.isfinite(all_cameras).all()):
                raise ValueError('Native initializer must contain every finite exact shot frame')
            if np.max(np.abs(all_cameras[:, 11:25]-template[11:25])) > 1e-10:
                raise ValueError('Native initializer changed calibrated Brown optics')
            # Previous outputs are initialization only. At original keyframes,
            # preserve the same initialization and priors as the control graph.
            for stamp, camera in zip(stamps, cameras):
                all_cameras[np.searchsorted(all_stamps, stamp)] = camera
            stamps, cameras = all_stamps, all_cameras
            motion_path = Path(motion['shots'][index]['motion_npz'])
            payload = motion_path.read_bytes()
            observation_inputs[str(motion_path.resolve())] = hashlib.sha256(payload).hexdigest()
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                raw_motion = {name: data[name].copy() for name in data.files}
            expected_pairs = np.column_stack((stamps[:-1], stamps[1:]))
            if not np.array_equal(raw_motion['pair_pts'], expected_pairs):
                raise ValueError('Raw native pairs cross or miss a reviewed cut')
            for pair_index, (a, b) in enumerate(expected_pairs):
                pair = prepare_ray_pair(cameras[pair_index], *_pair_data(raw_motion, pair_index),
                                        max_matches=int(config.get('native_max_matches', 120)))
                if pair is not None:
                    raw_pairs.append((int(a), int(b), pair))
            raw_native_links.extend(raw_pairs)
            edge_path = Path(item['source_edges'])
            observation_inputs[str(edge_path.resolve())] = hashlib.sha256(edge_path.read_bytes()).hexdigest()
            native_edge_inputs[index] = SourceEdges.load(edge_path, stamps,
                source_sha256=source_hash, model_sha256=model_hash)
        for stamp, camera in zip(stamps, cameras):
            stamp = int(stamp)
            if stamp in initial or not shot['start_pts'] <= stamp < shot['end_pts']:
                raise ValueError('Duplicate PTS or camera-cut crossing')
            templates[stamp] = camera.copy()
            initial[stamp] = to_state(camera)
        bridges = []
        if config.get('native_motion_bridges', False) and not item.get('native_tracks'):
            if pivot:
                raise ValueError('Compressed pure-camera motion bridges currently require fixed optical centres')
            from native_motion_links import build_native_motion_links
            cache = Path(config['native_motion_cache'])/f'shot{index:02d}.npz'
            bridges, bridge_report = build_native_motion_links(
                Path(motion['shots'][index]['motion_npz']), stamps, template, cache_path=cache,
                keyframe_cameras=cameras if config.get('native_lens_interpolation', True) else None)
            if config.get('initialize_rotation_from_native', True):
                local = {int(p): templates[int(p)] for p in stamps}
                bridge_report['rotation_initialization_reached'] = _bridge_rotation_initial(local, own, bridges)
                for p in local:
                    initial[p] = to_state(local[p])
            native_links.extend(bridges)
            native_reports.append({'shot_index': index, **bridge_report})
        if surface_pairs:
            pair_records = _surface_links(pair_records, templates, planes)
        for spec in item.get('curves', []):
            allowed = set(spec['ids'])
            for stamp in map(int, keyframe_stamps):
                if 'only_pts' in spec and stamp not in spec['only_pts']:
                    continue
                path = Path(spec.get('overrides', {}).get(str(stamp),
                            str(Path(spec['directory'])/f'{stamp}.npz')))
                if not path.exists():
                    continue
                payload = path.read_bytes()
                observation_inputs[str(path.resolve())] = hashlib.sha256(payload).hexdigest()
                with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                    for name in sorted(allowed & set(map(str, np.unique(data['physical_id'])))):
                        if index in native_edge_inputs and name in ('board_top', 'ice_board_contact'):
                            continue  # Same native board source is added once below.
                        selected = data['physical_id'] == name
                        uv = data['observed_uv'][selected].copy()
                        if len(uv) < 6:
                            continue
                        if name not in world:
                            raise ValueError(f'Unknown physical curve {name}')
                        curve_groups.setdefault((stamp, name), []).append(uv)
                        default_sigma = float(config.get('board_sigma_px', 2.)
                            if name in ('board_top', 'ice_board_contact')
                            else config.get('marking_sigma_px', 2.))
                        if 'point_sigma_px' in data.files:
                            supplied = np.asarray(data['point_sigma_px'], float)
                            if supplied.shape != (len(data['observed_uv']),):
                                raise ValueError(f'Point uncertainty shape differs from source pixels: {path}')
                            sigma = supplied[selected].copy()
                        else:
                            sigma = np.full(len(uv), default_sigma)
                        if not np.isfinite(sigma).all() or np.any(sigma <= 0):
                            raise ValueError(f'Invalid source pixel uncertainty: {path}')
                        curve_sigmas.setdefault((stamp, name), []).append(sigma)
        shots.append({'index': index, 'camera_id': shot['camera_id'], 'pts': stamps,
                      'anchors': own, 'links': pair_records, 'native_links': bridges,
                      'raw_native_links': raw_pairs, 'keyframe_pts': keyframe_stamps,
                      'image_components': _components(stamps, [(a,b) for a,b,_ in pair_records]),
                      'all_components': _components(stamps, [(a,b) for a,b,_ in pair_records]
                                                    + [(a,b) for a,b,_,_ in bridges]
                                                    + [(a,b) for a,b,_ in raw_pairs])})
        links.extend(pair_records)
    auxiliary_links, auxiliary_reports = [], []
    shot_at = {int(p): shot['index'] for shot in shots for p in shot['pts']}
    for spec in config.get('additional_source_links', []):
        if pivot:
            raise ValueError('Additional fixed-centre source links require fixed camera motion')
        path = Path(spec['path'])
        payload = path.read_bytes()
        observation_inputs[str(path.resolve())] = hashlib.sha256(payload).hexdigest()
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            packet = {name: data[name].copy() for name in data.files}
        if str(packet['source_sha256']) != source_hash:
            raise ValueError('Additional image links use a different original video')
        metadata = json.loads(str(packet['metadata_json']))
        if (metadata.get('model_projection_used_to_make_matches') is not False or
                metadata.get('end_ad7_corners_used_to_make_matches') is not False or
                metadata.get('sam2_people_and_hud_excluded') is not True):
            raise ValueError('Additional links lack original-pixel/static-mask provenance')
        stamps = np.asarray(packet['pair_pts'], np.int64)
        offsets = np.asarray(packet['offsets'], np.int64)
        count = len(packet['xy0'])
        if (stamps.ndim != 2 or stamps.shape[1] != 2 or
                offsets.shape != (len(stamps)+1,) or offsets[0] != 0 or
                offsets[-1] != count or np.any(np.diff(offsets) < 0) or
                len({tuple(p) for p in stamps}) != len(stamps)):
            raise ValueError('Malformed additional image pair packet')
        if any(len(packet[name]) != count for name in
               ('xy1', 'fb_error_px', 'homography_inlier', 'region_hint')):
            raise ValueError('Additional image point arrays differ in length')
        accepted = []
        for pair_index, (a, b) in enumerate(stamps):
            a, b = int(a), int(b)
            if (a not in initial or b not in initial or shot_at[a] != shot_at[b]
                    or b-a < 512 or (b-a) % 256):
                raise ValueError('Additional image links must connect existing nonadjacent frames inside one cut')
            pair = prepare_ray_pair(templates[a], *_pair_data(packet, pair_index),
                                    max_matches=int(spec.get('max_matches', 120)))
            if pair is not None:
                auxiliary_links.append((a, b, pair, spec))
                accepted.append({'pts': [a, b], 'points': len(pair.xy0)})
        auxiliary_reports.append({'path': str(path.resolve()), 'metadata': metadata,
                                  'accepted': accepted})
    for key, values in curve_groups.items():
        unique_uv, indices = np.unique(np.concatenate(values), axis=0, return_index=True)
        curve_groups[key] = unique_uv
        curve_sigmas[key] = np.concatenate(curve_sigmas[key])[indices]
    ad_corners = {}
    for filename in config.get('ad_corner_observations', []):
        path = Path(filename)
        payload = path.read_bytes()
        observation_inputs[str(path.resolve())] = hashlib.sha256(payload).hexdigest()
        manifest = json.loads(payload)
        if manifest.get('source_sha256') != source_hash:
            raise ValueError(f'Advertisement observations have a different source: {path}')
        for row in manifest['observations']:
            if row.get('status') != 'measured':
                continue
            stamp, number, role = int(row['pts']), int(row['user_number']), row['role']
            if stamp not in initial:
                raise ValueError(f'Advertisement corner PTS is not a selected keyframe: {stamp}')
            uv = np.asarray(row['uv'], float)
            if uv.shape != (2,) or not np.isfinite(uv).all() or np.any(uv < 0) or np.any(uv >= [1920, 1080]):
                raise ValueError(f'Advertisement corner is outside native source pixels: {row}')
            ad_corners[stamp, number, role] = {**row, 'input_file': str(path.resolve())}
    ad_source_component_gate = None
    if config.get('ad_source_component_gate'):
        from source_component_gate import SourceComponentGate
        ad_source_component_gate = SourceComponentGate(
            config['ad_source_component_gate'], source_hash, ad_corners,
            observation_inputs)
    ad_geometry, ad_keys, ad_initials = {}, {}, {}
    if ad_corners:
        from ad_landmarks import AdGeometry
        for number in sorted({number for _, number, _ in ad_corners}):
            ad_geometry[number] = AdGeometry(scene, number)
            ad_keys[number] = 1000+number
            ad_initials[1000+number] = ad_geometry[number].initial.copy()
    if not set(anchors).issubset(initial):
        raise ValueError('Every selected model anchor must occur in its keyframe graph')
    source_quality.require_pts(initial)
    if len(applied_link_exclusions) != len(reviewed_link_exclusions):
        raise ValueError('An explicit reviewed wrong-link exclusion was not found in source inputs')
    camera_ids = sorted({s['camera_id'] for s in shots})
    unknown = set(config.get('physical_sigma_by_camera', {}))-set(camera_ids)
    if unknown:
        raise ValueError(f'Unknown camera IDs in physical priors: {sorted(unknown)}')
    mount_keys = {name: i+1 for i, name in enumerate(camera_ids)}
    shared_lens = bool(config.get('shared_lens', False))
    lens_keys = {name: 10+key for name, key in mount_keys.items()} if shared_lens else {}
    lens_initials = {}
    if shared_lens:
        from shared_lens import lens_initial, add_lens_prior, add_frame_lens
        lens_initials = {key: lens_initial() for key in lens_keys.values()}
    mount_initial = {}
    for name, key in mount_keys.items():
        selected = [templates[int(s['pts'][0])][:3] for s in shots if s['camera_id'] == name]
        mount_initial[key] = np.r_[np.median(selected, axis=0)/100., np.zeros(8)]
    extra_nodes = len(ring_scene.keys)+3 if ring_scene else 0
    graph = IncrementalCameraGraph(11, window_size=len(initial)+len(mount_keys)+len(lens_keys)+len(ad_keys)+extra_nodes+2,
                                  relinearize_threshold=1e-5,
                                  refinement_updates=int(config.get('refinement_updates', 20)),
                                  optimization=config.get('optimization', 'gauss_newton'),
                                  diagnose_cost=config.get('diagnose_cost'))
    original_calibration, original_initials, shared_ring_groups = None, {}, {}
    if ring_scene:
        from original_calibration import add_original_calibration
        ring_scene.add_priors(graph)
        original_calibration = add_original_calibration(
            graph, scene, ring_scene.keys, ad_geometries=ad_geometry, ad_keys=ad_keys,
            finite_vector=finite_vector, source_quality=source_quality,
            source_precision=original_source_precision)
        original_initials = original_calibration.camera_initial_states
    reference_links = []
    if config.get('reference_image_links'):
        if original_calibration is None:
            raise ValueError('Cross-reference texture requires original camera graph states')
        from reference_pair import load_reference_pairs
        reference_links = load_reference_pairs(config['reference_image_links'], scene,
            templates, original_calibration.camera_templates, source_hash,
            model_hash, observation_inputs)
        for a, b, pair, sigma, metadata in reference_links:
            if config.get('reference_pixel_noise'):
                # Add the correlated source-pixel model once warm cameras are known.
                continue
            residual, jacobian = source_quality.wrap(pair.residual, pair.jacobians,
                source_quality.pair_scales(a, b, pair.xy0, pair.xy1, 'reference_texture'))
            residual, jacobian = pointwise_huber(residual, jacobian,
                                                 group_size=2, delta=2.*sigma)
            graph.add_observation([a, b], residual, residual_dim=2*len(pair.xy0),
                                  sigma=sigma, jac_fn=jacobian)
    centre_sigma = np.asarray(config.get('centre_sigma_scene', [3., 3., .6]))/100.
    polygon = np.ascontiguousarray(np.asarray(scene['ice_boundary']['points_xyz'], np.float32)[:, :2])
    for key, state in lens_initials.items():
        add_lens_prior(graph, key, state, config.get('shared_lens_settings', {}))
    for number, geometry in ad_geometry.items():
        geometry.add_prior(graph, ad_keys[number])
    for (stamp, number, role), row in ad_corners.items():
        multiplier = source_quality.scales(stamp, [row['uv']], 'ad_corner')[0]
        source_sigma = float(row.get('sigma_px', 3.))
        # The color-corner extractor already includes measured edge spread in
        # these rows. Retain its full sigma without counting blur twice.
        effective_sigma = (max(source_sigma, 3.*multiplier)
                           if row.get('source_edge_quality') else source_sigma*multiplier)
        if source_precision is not None:
            # Direct source edge widths supersede the frequency-based blur
            # multiplier here, retaining every pre-existing uncertainty floor.
            effective_sigma = source_precision.ad_sigma(
                stamp, number, role, row['uv'], source_sigma)
        if ad_source_component_gate is None:
            ad_geometry[number].factor(graph, stamp, ad_keys[number], templates[stamp],
                                       role, row['uv'], sigma_px=effective_sigma)
        else:
            components = ad_source_component_gate.components(stamp, number, role, row['uv'])
            if components:
                ad_geometry[number].factor(graph, stamp, ad_keys[number], templates[stamp],
                                           role, row['uv'], sigma_px=effective_sigma,
                                           components=components)
    for name, key in mount_keys.items():
        offset_sigma = np.full(3, float(config.get('offset_sigma_scene', .6))/100.)
        mount_sigma = np.r_[centre_sigma, offset_sigma, np.full(5, 1e-6)] if pivot else np.r_[centre_sigma, np.full(8, 1e-6)]
        graph.add_prior(key, mount_initial[key], mount_sigma)
        def physical(q):
            c = q[0][:3]*100.
            inside = cv2.pointPolygonTest(polygon, tuple(c[:2]), True)
            return np.array([max(0., inside+.1), max(0., .3-c[2])])
        strength = config.get('physical_sigma_by_camera', {}).get(
            name, config.get('physical_sigma_scene', .1))
        graph.add_observation([key], physical, residual_dim=2, sigma=float(strength))
    for shot in shots:
        mount = mount_keys[shot['camera_id']]
        for stamp in map(int, shot['pts']):
            if pivot:
                add_rig_equality(graph, stamp, mount, sigma_scene=.00001)
            else:
                _mount_equality(graph, stamp, mount)
            if shared_lens:
                add_frame_lens(graph, stamp, lens_keys[shot['camera_id']],
                               config.get('shared_lens_settings', {}))
            else:
                _lens_factor(graph, stamp, _lens_reference(shot['anchors'], stamp))
                _aspect_factor(graph, stamp, shot['anchors'])
            if stamp not in original_keyframes:
                # Thousands of interpolated candidate poses are not thousands
                # of independent observations. New native states are tied by
                # actual LK pixels, actual board pixels and one shared rig.
                continue
            # Previous solved geometry is an initializer/prior, not invented
            # source evidence. Real curves, texture and seams can adjust it.
            sigma = np.r_[np.full(3, 1e3), np.full(3, .025), [.04, .04, .02, .02, .04]]
            if stamp in anchors:
                sigma[3:8] = [.01, .01, .01, .015, .015]
            if shared_lens:
                sigma[8:11] = [.3, .3, .4]
                # Preserve only the initial common zoom. Independent fx/fy
                # priors would reintroduce the old, degenerate aspect ratio
                # which the shared lens is meant to replace.
                sigma[6:8] = [.5, .5]
                zoom_reference = float(np.mean(initial[stamp][6:8]))
                zoom_jacobian = np.zeros((1, 11))
                zoom_jacobian[0, 6:8] = .5
                graph.add_observation([stamp],
                    lambda q, target=zoom_reference: np.array([np.mean(q[0][6:8])-target]),
                    residual_dim=1, sigma=.015 if stamp in anchors else .04,
                    jac_fn=lambda q, j=zoom_jacobian: [j])
            graph.add_prior(stamp, initial[stamp], sigma)
        if shot['index'] in native_edge_inputs:
            edges = native_edge_inputs[shot['index']]
            for frame_index, stamp in enumerate(map(int, shot['pts'])):
                _add_source_edge_factors(graph, stamp, frame_index, templates[stamp],
                    templates[stamp], edges, world, fixed_mount=False,
                    finite_vector=finite_vector, dynamic_candidates=edge_search == 'current_state',
                    source_quality=source_quality)
    for (stamp, name), uv in curve_groups.items():
        template, xyz = templates[stamp], world[name]
        multipliers = source_quality.scales(stamp, uv, 'curve:'+name)
        sigma = curve_sigmas[stamp, name]*(multipliers if source_precision is None else 1.)
        normal_precision = (source_precision.curve(stamp, name, uv, sigma)
                            if source_precision is not None else None)
        if ring_scene and name.endswith('_outer'):
            group = ring_scene.group(stamp, name, template, xyz, uv, sigma)
            if group is not None:
                add_shared_ring_factor(graph, stamp, ring_scene.keys[group.feature_id],
                                       template, group, finite_vector=finite_vector,
                                       source_normal_precision=normal_precision)
                shared_ring_groups[stamp, name] = group
                continue
        scalar_sigma = sigma.copy()
        sigma = np.repeat(sigma, curve_dim)
        def residual(q, template=template, xyz=xyz, uv=uv, sigma=sigma):
            return curve_fn(from_state(q[0], template), xyz, uv)/sigma
        def jacobian(q, template=template, xyz=xyz, uv=uv, sigma=sigma):
            return [curve_fn(from_state(q[0], template), xyz, uv, True)[1]/sigma[:, None]]
        if normal_precision is not None:
            residual, jacobian = source_precision.wrap_curve(
                residual, jacobian, scalar_sigma, *normal_precision)
        if pointwise:
            residual, jacobian = pointwise_huber(residual, jacobian, group_size=curve_dim, delta=2.)
        graph.add_observation([stamp], residual, residual_dim=curve_dim*len(uv), sigma=1.,
                              jac_fn=jacobian, huber_k=None if pointwise else 2.)
    for a, b, pair in links:
        sigma = float(config.get('pair_sigma_px', 1.5))
        residual, jacobian = source_quality.wrap(pair.residual, pair.jacobians,
            source_quality.pair_scales(a, b, pair.xy0, pair.xy1, 'keyframe_pair'))
        if pointwise:
            residual, jacobian = pointwise_huber(residual, jacobian, group_size=2, delta=2.*sigma)
        graph.add_observation([a, b], residual, residual_dim=2*len(pair.xy0),
                              sigma=sigma, jac_fn=jacobian, huber_k=None if pointwise else 2.)
    for a, b, pair, sigma in native_links:
        # Transported endpoints share one composed homography; they are not
        # independent source measurements. Allow their information to be
        # reduced without disconnecting fast-pan keyframes.
        sigma *= float(config.get('native_motion_sigma_scale', 1.))
        residual, jacobian = source_quality.wrap(pair.residual, pair.jacobians,
            source_quality.pair_scales(a, b, pair.xy0, pair.xy1, 'derived_motion'))
        residual, jacobian = pointwise_huber(residual, jacobian,
                                              group_size=2, delta=3.*sigma)
        graph.add_observation([a, b], residual, residual_dim=2*len(pair.xy0),
                              sigma=sigma, jac_fn=jacobian)
    native_noise_metadata = None
    for a, b, pair in raw_native_links:
        sigma = float(config.get('native_pair_sigma_px', 2.))
        residual, jacobian = source_quality.wrap(pair.residual, pair.jacobians,
            source_quality.pair_scales(a, b, pair.xy0, pair.xy1, 'native_motion'))
        residual, jacobian = pointwise_huber(residual, jacobian,
                                            group_size=2, delta=2.*sigma)
        if config.get('native_pair_similarity_noise_px') is not None:
            residual, jacobian, native_noise_metadata = wrap_correlated_pair(
                residual, jacobian, pair.xy1, sigma_point_px=sigma,
                sigma_modes_px=config['native_pair_similarity_noise_px'])
            sigma = 1.
        graph.add_observation([a, b], residual, residual_dim=2*len(pair.xy0),
                              sigma=sigma, jac_fn=jacobian)
    for a, b, pair, spec in auxiliary_links:
        sigma = float(spec.get('sigma_px', 2.))
        residual, jacobian = source_quality.wrap(pair.residual, pair.jacobians,
            source_quality.pair_scales(a, b, pair.xy0, pair.xy1, 'multiframe_motion'))
        residual, jacobian = pointwise_huber(residual, jacobian,
                                             group_size=2, delta=2.*sigma)
        if spec.get('similarity_noise_px') is not None:
            residual, jacobian, _ = wrap_correlated_pair(
                residual, jacobian, pair.xy1, sigma_point_px=sigma,
                sigma_modes_px=spec['similarity_noise_px'])
            sigma = 1.
        graph.add_observation([a, b], residual, residual_dim=2*len(pair.xy0),
                              sigma=sigma, jac_fn=jacobian)
    atlas_anchor_diagnostics = []
    for stamp, anchor in anchors.items():
        # Preserve the legacy filter on anchor-derived XYZ candidates; they
        # are not independent physical landmark measurements.
        error = np.linalg.norm(project_fast(anchor.camera, anchor.xyz)[0]-anchor.uv, axis=1)
        keep = np.isfinite(error) & (error <= 8.)
        source_npz = str(anchor_files[stamp].resolve())
        diagnostic = {'pts': int(stamp), 'source_npz': source_npz,
                      'source_npz_sha256': anchor_input_hashes[source_npz],
                      'original_points': int(len(anchor.uv)),
                      'legacy_at_most_8px_points': int(keep.sum()),
                      'selection_limit': config.get('atlas_maximum_per_anchor'),
                      'selected_points': 0, 'factor_added': False}
        atlas_anchor_diagnostics.append(diagnostic)
        if keep.sum() < 6:
            continue
        xyz, uv, template = anchor.xyz[keep], anchor.uv[keep], templates[stamp]
        maximum = int(config.get('atlas_maximum_per_anchor', len(uv)))
        if len(uv) > maximum:
            selected = _distributed_indices(uv, error[keep], maximum)
            xyz, uv = xyz[selected], uv[selected]
        diagnostic['selected_points'] = int(len(uv))
        if atlas_anchor_mode == 'initialization_only':
            continue
        atlas_sigma = float(config.get('atlas_sigma_px', 6.))
        def residual(q, xyz=xyz, uv=uv, template=template):
            return (project_fast(from_state(q[0], template), xyz)[0]-uv).ravel()
        def jacobian(q, xyz=xyz, template=template):
            return [project_fast(from_state(q[0], template), xyz, True)[2]]
        residual, jacobian = source_quality.wrap(residual, jacobian,
            source_quality.scales(stamp, uv, 'atlas_anchor'))
        if pointwise:
            residual, jacobian = pointwise_huber(residual, jacobian, group_size=2, delta=2.*atlas_sigma)
        graph.add_observation([stamp], residual, residual_dim=2*len(uv),
                              sigma=atlas_sigma, jac_fn=jacobian, huber_k=None if pointwise else 2.)
        diagnostic['factor_added'] = True
    seams = []
    for row in contacts['cuts']:
        if row['direct_ground_seam'] != 'usable_soft':
            continue
        a, b = int(row['before_pts']), int(row['after_pts'])
        if a not in initial or b not in initial:
            continue
        if b-a != 256 or row['before']['contact_state'] != 'visible_contact' or row['after']['contact_state'] != 'visible_contact':
            raise ValueError('Contact seam does not link two consecutive visible support feet')
        ia, ib = row['selected_candidate_indices']
        pa, pb = row['before']['candidates'][ia], row['after']['candidates'][ib]
        ta, tb = templates[a], templates[b]
        def residual(q, ta=ta, tb=tb, pa=pa, pb=pb):
            return _ground(from_state(q[1], tb), pb)-_ground(from_state(q[0], ta), pa)
        settings = config.get('seam_settings', {}).get(str(row['cut_index']), {})
        covariance = None
        factor = residual
        sigma = float(settings.get('sigma_scene', config.get('seam_sigma_scene', .35)))
        if config.get('seam_noise_model') == 'source_pixels':
            pixel_sigmas = np.broadcast_to(np.asarray(settings.get('sigma_px',
                config.get('seam_sigma_px', [6., 4.])), float), (2, 2)).copy()
            pixel_sigmas *= np.asarray([source_quality.scales(a, [pa], 'contact')[0],
                source_quality.scales(b, [pb], 'contact')[0]])[:, None]
            whitener, covariance = ground_whitener(
                [ta, tb], [pa, pb], pixel_sigmas,
                settings.get('motion_sigma_scene', config.get('seam_motion_sigma_scene', .15)))
            def factor(q, raw=residual, weight=whitener):
                return weight@raw(q)
            sigma = 1.
        graph.add_observation([a, b], factor, residual_dim=2, sigma=sigma,
                              huber_k=settings.get('huber_k', 3.))
        seams.append((row['cut_index'], a, b, residual, covariance, pa, pb))
    finish_stage('load_prepare_construct')
    started = time.perf_counter()
    guesses = {**initial, **mount_initial, **lens_initials, **ad_initials,
               **original_initials, **(ring_scene.initial if ring_scene else {})}
    if config.get('warm_start'):
        guesses = _warm_start(config['warm_start'], guesses, shots, mount_keys,
                              lens_keys, ad_keys, source_hash, model_hash, templates)
        if ring_scene:
            ring_scene.warm_start(Path(config['warm_start']), guesses)
            previous_file = Path(config['warm_start'])/'original_cameras.npz'
            if previous_file.exists():
                with np.load(previous_file, allow_pickle=False) as data:
                    for pts, camera in zip(data['pts'], data['camera25']):
                        if int(pts) not in original_initials:
                            raise ValueError('Warm original reference camera ID differs')
                        guesses[int(pts)] = to_state(camera)
    reference_pixel_noise = None
    if config.get('reference_pixel_noise'):
        from pixel_pair_noise import SharedSourcePixelNoise
        noise_settings = config['reference_pixel_noise']
        if not reference_links:
            raise ValueError('Reference pixel noise requires actual reference pairs')
        noise_records = []
        for a, b, pair, sigma, metadata in reference_links:
            source_scale = source_quality.scales(a, pair.xy0, 'reference_texture_source')
            target_scale = source_quality.scales(b, pair.xy1, 'reference_texture_target')
            source_sigma = np.broadcast_to(np.asarray(noise_settings.get('source_sigma_px', sigma), float),
                                            (len(pair.xy0), 2))*source_scale[:, None]
            target_sigma = np.broadcast_to(np.asarray(noise_settings.get('target_sigma_px', sigma), float),
                                            (len(pair.xy1), 2))*target_scale[:, None]
            noise_records.append((a, b, pair, source_sigma, target_sigma))
        bundle = SharedSourcePixelNoise(noise_records, guesses)
        residual, jacobian = pointwise_huber(bundle.residual, bundle.jacobians,
                                             group_size=2, delta=2.)
        graph.add_observation(bundle.keys, residual, residual_dim=bundle.dimension,
                              sigma=1., jac_fn=jacobian)
        reference_pixel_noise = {**bundle.metadata, 'settings': noise_settings,
            'camera_initialization_from': config.get('warm_start'),
            'robust_order': 'shared_source_covariance_whitening_then_component_pair_huber2'}
    reference_board_links = []
    if config.get('reference_board_links'):
        if original_calibration is None:
            raise ValueError('Board print references require independent original camera states')
        from reference_board_pair import load_reference_board_pairs
        warm_programme = {p: from_state(guesses[p], template) for p, template in templates.items()}
        warm_original = {p: from_state(guesses[p], template)
                         for p, template in original_calibration.camera_templates.items()}
        reference_board_links = load_reference_board_pairs(config['reference_board_links'],
            scene, warm_programme, warm_original, source_hash, model_hash, observation_inputs)
        for a, b, pair, sigma, metadata in reference_board_links:
            residual, jacobian = pointwise_huber(pair.residual, pair.jacobians,
                                                 group_size=2, delta=2.)
            graph.add_observation([a, b], residual, residual_dim=2*len(pair.xy0),
                                  sigma=1., jac_fn=jacobian)
            metadata['camera_initialization_from'] = config.get('warm_start')
    finish_stage('warm_start_reference_covariance')
    resumed_initial_states = None
    if config.get('initial_graph_states_from'):
        guesses, resumed_initial_states = _resume_initial_states(
            config['initial_graph_states_from'], guesses, source_hash, model_hash)
    print(json.dumps({'stage': 'joint_optimize', 'frames': len(initial),
                      'warm_start': config.get('warm_start'),
                      'initial_graph_states_from': resumed_initial_states,
                      'custom_factor_callback': getattr(graph, 'callback_metadata', None),
                      'acceleration': acceleration.metadata()}), flush=True)
    solved = graph.update(guesses)
    finish_stage('isam2_update_and_refinement')
    if config.get('batch_polish'):
        options = config['batch_polish']
        print(json.dumps({'stage': 'same_graph_gtsam_polish', 'options': options}), flush=True)
        solved = graph.batch_polish(**options)
        finish_stage('lm_polish')
    output.mkdir(parents=True, exist_ok=True)
    base_model_hash = model_hash
    effective_scene_file = None
    solved_ads = {number: geometry.solved_geometry(solved[ad_keys[number]])
                  for number, geometry in ad_geometry.items()}
    if ring_scene:
        _, model_hash, effective_scene_file = ring_scene.export(
            output, solved, base_model_hash, solved_ads)
        candidate_release = {'schema_version': 'rinkspace.programme-scene-candidate.v1',
            'version': release['version'],
            'revision': 'v0.25-programme-calibration-unvalidated',
            'model_sha256': model_hash,
            'model_path': str(effective_scene_file.resolve()),
            'status': 'programme_shared_geometry_candidate_unvalidated',
            'base_model_sha256': base_model_hash,
            'base_release': str(Path(config.get('release', DEFAULT_RELEASE)).resolve()),
            'registered_as_latest_model': False,
            'candidate_scene': str(effective_scene_file.resolve())}
        (output/'candidate_release.json').write_text(
            json.dumps(candidate_release, indent=2), encoding='utf8')
        reference_pts = sorted(original_initials)
        np.savez_compressed(output/'original_cameras.npz',
            pts=np.asarray(reference_pts, np.int64),
            camera25=np.asarray([from_state(solved[p], original_calibration.camera_templates[p])
                                 for p in reference_pts]),
            model_sha256=np.asarray(model_hash), source_sha256=np.asarray(source_hash),
            status=np.asarray('original_reference_recalibration_unvalidated'))
        (output/'original_reprojection.json').write_text(json.dumps(
            original_calibration.evaluate(solved, include_details=True), indent=2), encoding='utf8')
    np.savez_compressed(output/'graph_states.npz',
        keys=np.asarray(sorted(solved), np.int64),
        states=np.asarray([solved[key] for key in sorted(solved)]),
        source_sha256=np.asarray(source_hash), model_sha256=np.asarray(model_hash))
    output_cameras = {p: from_state(solved[p], templates[p]) for p in initial}
    for shot in shots:
        pair_records = shot['links']
        raw_pairs = shot['raw_native_links']
        derived = shot['native_links']
        np.savez_compressed(output/f'shot{shot["index"]:02d}_keyframes.npz',
            pts=shot['pts'], camera25=np.asarray([output_cameras[int(p)] for p in shot['pts']]),
            **source_quality.frame_fields(shot['pts']),
            original_keyframe_pts=shot['keyframe_pts'],
            status=np.full(len(shot['pts']), 'temporal_candidate'),
            model_sha256=np.asarray(model_hash), source_sha256=np.asarray(source_hash),
            link_pts=np.asarray([(a, b) for a, b, _ in pair_records], np.int64).reshape(-1, 2),
            link_offsets=np.r_[0, np.cumsum([len(p.xy0) for _, _, p in pair_records])],
            link_xy0=np.concatenate([p.xy0 for _, _, p in pair_records]) if pair_records else np.empty((0, 2)),
            link_xy1=np.concatenate([p.xy1 for _, _, p in pair_records]) if pair_records else np.empty((0, 2)),
            derived_motion_pts=np.asarray([(a,b) for a,b,_,_ in derived], np.int64).reshape(-1,2),
            derived_motion_sigma_px=np.asarray([sigma*float(config.get('native_motion_sigma_scale', 1.))
                                               for _,_,_,sigma in derived]),
            derived_motion_semantics=np.asarray('composed_native_frame_motion_not_direct_endpoint_correspondences'))
        if raw_pairs:
            # Native joint outputs use the same render/mapping interface as
            # shot tracking. Local pixel diagnostics never upgrade review status.
            metrics = np.zeros(len(shot['pts']), dtype=METRIC_DTYPE)
            for name in ('pair_median_px', 'pair_p95_px',
                         'anchor_median_px', 'anchor_p95_px'):
                metrics[name] = np.nan
            position = {int(p): i for i, p in enumerate(shot['pts'])}
            for a, b, pair in raw_pairs:
                _, values = _frame_quality(pair, [solved[a], solved[b]])
                for name, value in zip(METRIC_DTYPE.names[:6], values):
                    metrics[position[b]][name] = value
            np.savez_compressed(output/f'shot{shot["index"]:02d}.npz',
                pts=shot['pts'], camera25=np.asarray([output_cameras[int(p)] for p in shot['pts']]),
                **source_quality.frame_fields(shot['pts']),
                status=np.full(len(shot['pts']), 'temporal_candidate'),
                metrics=metrics,
                shot_index=np.asarray(shot['index']), camera_id=np.asarray(shot['camera_id']),
                model_sha256=np.asarray(model_hash), source_sha256=np.asarray(source_hash),
                original_keyframe_pts=shot['keyframe_pts'],
                raw_native_pair_pts=np.asarray([(a,b) for a,b,_ in raw_pairs], np.int64),
                initializer_used_as_observations=np.asarray(False))
    records = []
    for stamp, anchor in sorted(anchors.items()):
        with np.load(anchor_files[stamp], allow_pickle=False) as data:
            numeric = {name: data[name].copy() for name in data.files}
        numeric['camera'] = numeric['camera25'] = output_cameras[stamp]
        numeric['model_sha256'] = np.asarray(model_hash)
        numeric['xyz_model_sha256'] = np.asarray(base_model_hash)
        numeric['xyz_semantics'] = np.asarray(
            'legacy_base_scene_inverse_projection; not recalibrated source landmarks')
        np.savez_compressed(output/f'{stamp}.npz', **numeric)
        records.append({'pts': stamp, 'camera_id': anchor.camera_id,
                        'status': 'joint_camera_candidate_unvalidated',
                        'camera_vector25': output_cameras[stamp].tolist(),
                        'inliers': anchor.inliers, 'ice_inliers': anchor.ice_inliers})
    (output/'cameras.json').write_text(json.dumps({'model_sha256': model_hash,
        'source_sha256': source_hash, 'frames': records}, indent=2), encoding='utf8')
    curve_errors = []
    for (stamp, name), uv in sorted(curve_groups.items()):
        before = finite_curve_distance(templates[stamp], world[name], uv)
        after_world = (ring_scene.world(shared_ring_groups[stamp, name], solved)
                       if (stamp, name) in shared_ring_groups else world[name])
        after = finite_curve_distance(output_cameras[stamp], after_world, uv)
        curve_errors.append({'pts': stamp, 'physical_id': name, 'points': len(uv),
            'p95_before_px': float(np.percentile(before, 95)),
            'p95_after_px': float(np.percentile(after, 95))})
    report = {'status': 'unverified_joint_candidate', 'seconds': time.perf_counter()-started,
        'curve_residual': curve_mode, 'curve_error_metric': 'finite_curve_distance_px',
        'source_edge_candidates': edge_search,
        'observation_quality': source_quality.report(),
        'observation_uncertainty': source_precision.report() if source_precision else None,
        'reference_pixel_noise': reference_pixel_noise,
        'reference_board_links': [metadata for a, b, pair, sigma, metadata in reference_board_links],
        'reference_image_links': [dict(metadata,
            p50_after_px=float(np.median(np.linalg.norm(pair.residual([solved[a], solved[b]]).reshape(-1,2), axis=1))),
            p95_after_px=float(np.percentile(np.linalg.norm(pair.residual([solved[a], solved[b]]).reshape(-1,2), axis=1),95)))
            for a,b,pair,sigma,metadata in reference_links],
        'model_sha256': model_hash, 'source_sha256': source_hash,
        'base_model_sha256': base_model_hash,
        'effective_scene_file': None if effective_scene_file is None else str(effective_scene_file.resolve()),
        'shared_ring_candidates': ring_scene.describe(solved) if ring_scene else {},
        'original_calibration': original_calibration.diagnostics if original_calibration else None,
        'configuration_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'observation_input_sha256': observation_inputs,
        'atlas_anchor_mode': atlas_anchor_mode,
        'resumed_initial_states': resumed_initial_states,
        'atlas_anchor_xyz_semantics': 'legacy anchor-derived XYZ; not independent physical measurements',
        'atlas_anchor_diagnostics': atlas_anchor_diagnostics,
        'frames': len(initial), 'source_image_links': len(links),
        'source_image_pairs': sum(len(pair.xy0) for _, _, pair in links),
        'duplicate_direct_pixel_pairs_removed': duplicate_direct_pixels,
        'source_reviewed_link_exclusions': applied_link_exclusions,
        'raw_native_links': len(raw_native_links),
        'raw_native_pixels': sum(len(pair.xy0) for _, _, pair in raw_native_links),
        'native_pair_noise': native_noise_metadata,
        'additional_source_links': auxiliary_reports,
        'additional_source_pair_count': len(auxiliary_links),
        'additional_source_pixels': sum(len(pair.xy0) for _, _, pair, _ in auxiliary_links),
        'native_source_edge_frames': sum(len(edges.pts) for edges in native_edge_inputs.values()),
        'ad_corner_observations': len(ad_corners),
        'ad_geometry_candidates': solved_ads,
        'ad_corner_errors': [
            {'pts': stamp, 'user_number': number, 'role': role,
             'error_px': float(np.linalg.norm(project_fast(output_cameras[stamp],
                 ad_geometry[number].corner(solved[ad_keys[number]], role).reshape(1,3))[0][0]
                 -np.asarray(row['uv'])))}
            for (stamp, number, role), row in sorted(ad_corners.items())],
        'optimization_trace': graph.last_update_diagnostics,
        'optimization_summary': getattr(graph, 'last_update_summary', {}),
        'custom_factor_callback': getattr(graph, 'callback_metadata', None),
        'batch_polish': getattr(graph, 'last_batch_diagnostics', {}),
        'warm_start': config.get('warm_start'),
        'native_motion_bridges': native_reports,
        'connectivity': [{'shot_index': s['index'], 'direct_image_component_count': len(s['image_components']),
                          'all_component_count': len(s['all_components']),
                          'components_without_absolute_anchor': [g for g in s['all_components']
                                                                  if not set(g)&set(s['anchors'])]}
                         for s in shots],
        'camera_motion': 'pivot' if pivot else 'fixed',
        'pair_model': 'surface' if surface_pairs else 'rays',
        'robust_groups': 'pointwise' if pointwise else 'block',
        'shared_lens': {name: {'fx_fy_ratio': float(np.exp(solved[key][6])),
                               'cx_cy_skew_px': (solved[key][8:11]*1000.).tolist()}
                        for name, key in lens_keys.items()},
        'camera_mounts': {name: {'pivot_xyz': (solved[key][:3]*100.).tolist(),
                               'camera_offset_xyz': (solved[key][3:6]*100.).tolist()}
                          for name, key in mount_keys.items()},
        'seams': [{'cut_index': index, 'before_distance_scene': float(np.linalg.norm(fn([initial[a], initial[b]]))),
                   'after_distance_scene': float(np.linalg.norm(fn([solved[a], solved[b]]))),
                   'ground_covariance_scene2': None if cov is None else cov.tolist(),
                   **shared_ground_diagnostic([output_cameras[a], output_cameras[b]], [pa, pb])}
                  for index, a, b, fn, cov, pa, pb in seams], 'curve_errors': curve_errors}
    if ad_source_component_gate is not None:
        report['ad_source_component_gate'] = ad_source_component_gate.report(
            require_complete=True)
    (output/'diagnostics.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    finish_stage('candidate_output_and_diagnostics')
    brief = {key: value for key, value in report.items()
             if key in ('status', 'seconds', 'model_sha256', 'base_model_sha256', 'frames',
                        'source_image_links', 'optimization_summary', 'batch_polish',
                        'curve_residual', 'effective_scene_file')}
    brief['native_bridge_count'] = len(native_links)
    return brief


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(solve(args.config, args.output)), flush=True)
