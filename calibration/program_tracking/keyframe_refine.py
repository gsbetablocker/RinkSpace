"""Close within-shot drift with CUDA advertisement/static-feature long links.

Raw SIFT features are reused from the source cache. Matches span multiple
keyframes and are geometrically checked before entering a fixed-centre iSAM2
graph. Output poses remain unvalidated candidates, including interpolated
native-frame initializations. No descriptor match is labelled an ice point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

try:
    from .runtime.bootstrap import add_geometry_dependencies
except ImportError:
    from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()

import cv2
import numpy as np
import torch

try:
    from .backend import IncrementalCameraGraph
    from .ice_refine import source_frame, _closed_ring, _world_curves, _curve_error
    from .motion import analysis_mask, match_sift_descriptors
    from .projection import to_state, from_state
    from .track_sequence import (CameraAnchor, PairConstraint, _absolute_factor,
                                  _fixed_mount_factor, _distributed_indices, _lens_reference, _lens_factor, _aspect_factor)
except ImportError:
    from backend import IncrementalCameraGraph
    from ice_refine import source_frame, _closed_ring, _world_curves, _curve_error
    from motion import analysis_mask, match_sift_descriptors
    from projection import to_state, from_state
    from track_sequence import (CameraAnchor, PairConstraint, _absolute_factor,
                                _fixed_mount_factor, _distributed_indices, _lens_reference, _lens_factor, _aspect_factor)

ROOT = Path(__file__).resolve().parents[2]
TEMP = Path('D:/CodexData/RinkSpace_program_tracking_tmp')


def _root_sift(desc):
    desc = np.asarray(desc, np.float32)
    return np.sqrt(desc/(desc.sum(axis=1, keepdims=True)+1e-12))


def source_features(shot, extra_pts, source, people):
    with np.load(shot['sift_npz']) as data:
        features = {}
        for i, stamp in enumerate(data['keyframe_pts']):
            a, b = data['offsets'][i:i+2]
            features[int(stamp)] = (data['xy_native'][a:b].copy(),
                                   _root_sift(data['descriptors'][a:b]))
    missing = set(map(int, extra_pts))-set(features)
    boxes = {int(row['pts']): row['boxes'] for row in people['frames']}
    sift = cv2.SIFT_create(nfeatures=700, contrastThreshold=.025)
    for stamp in sorted(missing):
        image = cv2.resize(source_frame(source, stamp), (960, 540), interpolation=cv2.INTER_AREA)
        keys, desc = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                           analysis_mask(stamp, boxes))
        xy = np.asarray([key.pt for key in keys], np.float32).reshape(-1, 2)*2
        if desc is None:
            desc = np.empty((0, 128), np.float32)
        features[stamp] = (xy, _root_sift(desc))
    return features


def build_links(features, camera, anchor_pts):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA matching is required; use the existing GPU-enabled runtime')
    print(json.dumps({'descriptor_device': 'cuda', 'gpu': torch.cuda.get_device_name(0)}), flush=True)
    stamps = sorted(features)
    anchor_set = set(anchor_pts)
    candidate_pairs = set()
    for i, stamp in enumerate(stamps):
        for stride in (1, 2, 4, 8, 16, 32, 64):
            if i+stride < len(stamps):
                candidate_pairs.add((stamp, stamps[i+stride]))
        if stamp in anchor_set:
            candidate_pairs.update(tuple(sorted((stamp, other))) for other in stamps if other != stamp)
    links = []
    for previous, following in sorted(candidate_pairs):
        xy0, desc0 = features[previous]
        xy1, desc1 = features[following]
        a, b, distance = match_sift_descriptors(desc0, desc1, ratio=.77, device='cuda')
        if len(a) < 12:
            continue
        p, q = xy0[a].astype(float), xy1[b].astype(float)
        # Multiple SIFT orientations/scales can repeat the same physical
        # pixel correspondence; they must not increase its information.
        _, unique = np.unique(np.column_stack((p, q)), axis=0, return_index=True)
        unique.sort()
        p, q, distance = p[unique], q[unique], distance[unique]
        if len(p) < 12:
            continue
        h, mask = cv2.findHomography(p, q, cv2.USAC_MAGSAC, 3., maxIters=5000, confidence=.999)
        if h is None or mask is None:
            continue
        keep = mask.ravel().astype(bool)
        p, q, distance = p[keep], q[keep], distance[keep]
        if len(p) < 12:
            continue
        hull = cv2.convexHull(p.astype(np.float32))
        if cv2.contourArea(hull) < .004*1920*1080:
            continue
        keep = _distributed_indices(p, distance, 72)
        p, q = p[keep], q[keep]
        constraint = PairConstraint(p, q, np.zeros((len(p), 3)),
                                    np.zeros((len(p), 3)), np.full(len(p), 3, np.uint8),
                                    camera.copy(), True)
        links.append((previous, following, constraint))
    return links


def solve(shot, dense_file, anchors, output, *, motion_index, source_shots, scene_file,
          curve_observations=None, allowed_curve_ids=('ice_board_contact',)):
    output.mkdir(parents=True, exist_ok=True)
    if output.resolve().is_relative_to(ROOT):
        raise ValueError('Intermediate keyframe graph belongs outside OneDrive')
    scene_hash = hashlib.sha256(scene_file.read_bytes()).hexdigest()
    source_data = json.loads(source_shots.read_text(encoding='utf8'))
    people = json.loads(Path(motion_index['people_boxes']).read_text(encoding='utf8'))
    with np.load(dense_file) as data:
        dense_pts = data['pts'].copy()
        dense_camera = data['camera25'].copy()
    if dense_pts[0] != shot['start_pts'] or dense_pts[-1] != shot['end_pts']-256:
        raise ValueError('Dense initializer differs from source cut range')
    curves = {}
    if curve_observations is not None:
        for path in Path(curve_observations).glob('*.npz'):
            try:
                stamp = int(path.stem)
            except ValueError:
                continue
            if not shot['start_pts'] <= stamp < shot['end_pts']:
                continue
            with np.load(path) as data:
                if not {'observed_uv', 'physical_id'}.issubset(data.files):
                    continue
                batch = {str(name): data['observed_uv'][data['physical_id'] == name].copy()
                         for name in np.unique(data['physical_id']) if str(name) in allowed_curve_ids}
                batch = {name: points for name, points in batch.items() if len(points) >= 6}
                if batch:
                    curves[stamp] = batch
    features = source_features(shot, [*anchors, *curves, int(dense_pts[-1])],
                               Path(source_data['source_path']), people)
    stamps = sorted(features)
    first = max(anchors.values(), key=lambda a: (a.ice_inliers > 0, a.inliers))
    template = first.camera.copy()
    initial = {stamp: to_state(dense_camera[(stamp-int(dense_pts[0]))//256]) for stamp in stamps}
    for stamp in initial:
        initial[stamp][:3] = template[:3]/100
    for stamp, anchor in anchors.items():
        initial[stamp] = to_state(anchor.camera)
        initial[stamp][:3] = template[:3]/100
    started = time.perf_counter()
    links = build_links(features, template, anchors)
    print(json.dumps({'shot': shot['shot_index'], 'keyframes': len(stamps), 'long_links': len(links)}), flush=True)
    graph = IncrementalCameraGraph(11, window_size=len(stamps)+2,
                                  relinearize_threshold=0.00001, refinement_updates=12)
    for stamp in stamps:
        _fixed_mount_factor(graph, stamp, template[:3], .000001)
        _lens_factor(graph, stamp, _lens_reference(anchors, stamp))
        _aspect_factor(graph, stamp, anchors)
    for anchor in anchors.values():
        _absolute_factor(graph, anchor, template, fixed_mount=True)
    scene = json.loads(scene_file.read_text(encoding='utf8'))
    world_curves = _world_curves(scene)
    world_curves['board_top'] = _closed_ring(scene['board_surface']['top_edge_xyz'])
    for stamp, groups in curves.items():
        for name, uv in groups.items():
            world = world_curves[name]
            # Freeze genuine source pixels and physical identity, not a
            # camera-projected observation. Centre stays the common mount.
            def residual(states, world=world, uv=uv):
                camera = from_state(states[0], template)
                camera[:3] = template[:3]
                return _curve_error(camera, world, uv)
            def jacobian(states, world=world, uv=uv):
                camera = from_state(states[0], template)
                camera[:3] = template[:3]
                derivative = _curve_error(camera, world, uv, True)[1]
                derivative[:, :3] = 0.
                return [derivative]
            graph.add_observation([stamp], residual, residual_dim=len(uv),
                                  sigma=2., jac_fn=jacobian, huber_k=2.)
    for previous, following in zip(stamps[:-1], stamps[1:]):
        sigma = np.array([1e-7]*3 + [.04]*3 + [.08]*2 + [.04]*3)
        graph.add_between(previous, following, initial[following]-initial[previous], sigma)
    for previous, following, pair in links:
        graph.add_observation([previous, following], pair.residual,
                              residual_dim=2*len(pair.xy0), sigma=1.5,
                              jac_fn=pair.jacobians, huber_k=2.)
    solved = graph.update(initial)
    cameras = np.asarray([from_state(solved[stamp], template) for stamp in stamps])
    diagnostics = []
    for previous, following, pair in links:
        before = pair.error([initial[previous], initial[following]])
        after = pair.error([solved[previous], solved[following]])
        diagnostics.append([previous, following, len(after), float(np.median(before)),
                            float(np.median(after)), float(np.percentile(after, 95))])
    np.savez_compressed(output/f'shot{shot["shot_index"]:02d}_keyframes.npz',
                        pts=np.asarray(stamps, np.int64), camera25=cameras,
                        status=np.full(len(stamps), 'temporal_candidate'),
                        links=np.asarray(diagnostics), model_sha256=np.asarray(scene_hash),
                        link_pts=np.asarray([(a, b) for a, b, _ in links], np.int64).reshape(-1, 2),
                        link_offsets=np.r_[0, np.cumsum([len(p.xy0) for _, _, p in links])],
                        link_xy0=np.concatenate([p.xy0 for _, _, p in links]) if links else np.empty((0, 2)),
                        link_xy1=np.concatenate([p.xy1 for _, _, p in links]) if links else np.empty((0, 2)),
                        source_sha256=np.asarray(source_data['source_sha256_from_timeline']))
    record = {'shot_index': shot['shot_index'], 'camera_id': shot['camera_id'],
              'status': 'unverified_keyframe_graph_candidate', 'keyframes': len(stamps),
              'links': len(links), 'descriptor_device': 'cuda',
              'curve_observation_frames': len(curves), 'curve_ids': list(allowed_curve_ids),
              'seconds': time.perf_counter()-started,
              'link_median_before_px': float(np.median([r[3] for r in diagnostics])) if diagnostics else None,
              'link_median_after_px': float(np.median([r[4] for r in diagnostics])) if diagnostics else None,
              'source_anchor_pts': sorted(anchors)}
    (output/f'shot{shot["shot_index"]:02d}.json').write_text(json.dumps(record, indent=2), encoding='utf8')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shot', type=int, required=True)
    parser.add_argument('--tracked', type=Path, required=True)
    parser.add_argument('--anchor', type=Path, action='append', required=True,
                        help='Numeric source anchor named <native PTS>.npz; repeat for all real anchors')
    parser.add_argument('--output', type=Path, default=TEMP/'keyframe_graph')
    parser.add_argument('--motion-index', type=Path, default=Path(__file__).with_name('motion_index.json'))
    parser.add_argument('--source-shots', type=Path, default=Path(__file__).with_name('source_shots.json'))
    parser.add_argument('--scene', type=Path, default=ROOT/'calibration/deliverables/scene.json')
    parser.add_argument('--curve-observations', type=Path,
                        help='Source-pixel NPZ directory; only explicitly selected physical IDs are used')
    parser.add_argument('--curve-id', action='append',
                        help='Reviewed physical curve ID; default uses only the real board/ice contact')
    args = parser.parse_args()
    index = json.loads(args.motion_index.read_text(encoding='utf8'))
    shot = index['shots'][args.shot]
    anchors = {}
    for path in args.anchor:
        stamp = int(path.stem)
        if not shot['start_pts'] <= stamp < shot['end_pts']:
            raise ValueError('Source anchor belongs to a different camera shot')
        with np.load(path) as data:
            camera = data['camera25'] if 'camera25' in data else data['camera']
            xyz = data['xyz'] if 'xyz' in data else np.empty((0, 3))
            uv = data['uv'] if 'uv' in data else np.empty((0, 2))
            surface = data['surface'] if 'surface' in data else np.full(len(xyz), 'unknown')
        anchors[stamp] = CameraAnchor(stamp, shot['camera_id'], camera, xyz, uv, surface,
                                      len(xyz), int(np.sum(surface == 'ice')))
    print(json.dumps(solve(shot, args.tracked, anchors, args.output, motion_index=index,
                           source_shots=args.source_shots, scene_file=args.scene,
                           curve_observations=args.curve_observations,
                           allowed_curve_ids=args.curve_id or ('ice_board_contact',))), flush=True)


if __name__ == '__main__':
    main()
