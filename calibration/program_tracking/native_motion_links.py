"""Bridge adjacent keyframes with *composed* observed native-frame LK motion.

Each homography comes from a real pair of consecutive source-video frames.
The resulting keyframe endpoint pixels are transported predictions, never
claimed to be directly matched or observed at the far keyframe.  The bridge
is a camera-motion factor for zero-Brown fixed-ray initialization/optimization,
not a new ice-plane or world-point measurement.
"""
from __future__ import annotations

import json
import heapq
import hashlib
from pathlib import Path

import numpy as np

try:
    from .runtime.bootstrap import add_geometry_dependencies
except ImportError:
    from runtime.bootstrap import add_geometry_dependencies

add_geometry_dependencies()
import cv2

try:
    from .track_sequence import PairConstraint, _distributed_indices
except ImportError:
    from track_sequence import PairConstraint, _distributed_indices


SCHEMA = "rinkspace.native-motion-bridge.v1"
FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080
MIN_STEP_MATCHES = 12


class _BrownRectifier:
    """Native frame lens to/from a common zero-distortion virtual K."""

    def __init__(self, camera: np.ndarray, virtual_camera: np.ndarray | None = None):
        self.fx, self.fy = np.exp(camera[6:8])
        self.cx, self.cy, self.skew = camera[8:11]
        virtual = camera if virtual_camera is None else virtual_camera
        self.virtual_fx, self.virtual_fy = np.exp(virtual[6:8])
        self.virtual_cx, self.virtual_cy = virtual[8:10]
        self.k = np.array([[self.fx, 0., self.cx], [0., self.fy, self.cy],
                           [0., 0., 1.]], np.float64)
        self.dist = np.ascontiguousarray(camera[11:25], np.float64)

    def to_rectified(self, native: np.ndarray) -> np.ndarray:
        pixels = np.ascontiguousarray(native, np.float64).copy()
        pixels[:, 0] -= self.skew * (pixels[:, 1] - self.cy) / self.fy
        unit = cv2.undistortPoints(pixels.reshape(-1, 1, 2), self.k, self.dist).reshape(-1, 2)
        return np.ascontiguousarray(np.column_stack((self.virtual_fx*unit[:, 0]+self.virtual_cx,
                                                     self.virtual_fy*unit[:, 1]+self.virtual_cy)))

    def to_native(self, rectified: np.ndarray) -> np.ndarray:
        unit = np.column_stack(((rectified[:, 0]-self.virtual_cx)/self.virtual_fx,
                                (rectified[:, 1]-self.virtual_cy)/self.virtual_fy))
        xyz = np.ascontiguousarray(np.column_stack((unit, np.ones(len(unit)))))
        native = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), self.k,
                                   self.dist)[0].reshape(-1, 2)
        native[:, 0] += self.skew*(native[:, 1]-self.cy)/self.fy
        return native


def _homography_pixels(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)


def _step_model(source: np.ndarray, target: np.ndarray, fb: np.ndarray,
                old_inliers: np.ndarray, native_a: int, native_b: int,
                source_rectifier: _BrownRectifier | None,
                target_rectifier: _BrownRectifier | None):
    """Independent robust H with a reproducible, explicit rejection record."""
    good = (old_inliers != 0) & (fb < 2.5)
    good &= np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    p_native = np.ascontiguousarray(source[good], dtype=np.float64)
    q_native = np.ascontiguousarray(target[good], dtype=np.float64)
    if source_rectifier is None:
        p, q = p_native.astype(np.float32), q_native.astype(np.float32)
    else:
        p = source_rectifier.to_rectified(p_native).astype(np.float32)
        q = target_rectifier.to_rectified(q_native).astype(np.float32)
    score = np.asarray(fb[good], np.float64)
    record = {"a": int(native_a), "b": int(native_b), "raw_matches": int(len(source)),
              "filtered_matches": int(len(p)), "status": "rejected"}
    finite = np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1)
    p, q = p[finite], q[finite]
    p_native, q_native, score = p_native[finite], q_native[finite], score[finite]
    record['filtered_matches'] = int(len(p))
    if len(p) < MIN_STEP_MATCHES:
        record["reason"] = "fewer_than_12_actual_filtered_matches"
        return None, None, None, None, record
    h, mask = cv2.findHomography(
        p, q, cv2.USAC_MAGSAC, ransacReprojThreshold=2.5,
        maxIters=1000, confidence=.995,
    )
    if h is None or mask is None or not np.isfinite(h).all() or abs(h[2, 2]) < 1e-12:
        record["reason"] = "homography_fit_failed"
        return None, None, None, None, record
    h = h / h[2, 2]
    inlier = mask.ravel().astype(bool)
    count = int(inlier.sum())
    record["magsac_inliers"] = count
    record["inlier_fraction"] = float(count / len(p))
    if count < MIN_STEP_MATCHES or count / len(p) < .5:
        record["reason"] = "too_few_geometric_inliers"
        return None, None, None, None, record
    fitted = _homography_pixels(p, h)
    error = np.linalg.norm(fitted - q, axis=1)
    med = float(np.median(error[inlier]))
    p95 = float(np.percentile(error[inlier], 95))
    record["median_reprojection_px"] = med
    record["p95_reprojection_px"] = p95
    record["H_rectified_xy" if source_rectifier is not None else "H_native_xy"] = h.tolist()
    if source_rectifier is not None:
        native_fitted = target_rectifier.to_native(fitted.astype(np.float64))
        native_error = np.linalg.norm(native_fitted - q_native, axis=1)
        record['median_native_reprojection_px'] = float(np.median(native_error[inlier]))
        record['p95_native_reprojection_px'] = float(np.percentile(native_error[inlier], 95))
    else:
        record['median_native_reprojection_px'] = med
        record['p95_native_reprojection_px'] = p95
    if (med > 2.0 or p95 > 5.0 or record['median_native_reprojection_px'] > 2.0
            or record['p95_native_reprojection_px'] > 5.0):
        record["reason"] = "large_actual_pair_reprojection_error"
        return None, None, None, None, record
    # A single-frame TV pan/zoom is near locally affine. Detect spurious
    # projective fits without forbidding a genuine fast translation.
    probe = np.array([[960., 540.], [961., 540.], [960., 541.]], np.float64)
    warped = _homography_pixels(probe, h)
    local = np.column_stack((warped[1] - warped[0], warped[2] - warped[0]))
    singular = np.linalg.svd(local, compute_uv=False)
    centre_step = float(np.linalg.norm(warped[0] - probe[0]))
    record["centre_motion_px"] = centre_step
    record["local_scale"] = [float(v) for v in singular]
    if (not np.isfinite(warped).all() or np.linalg.det(local) <= 0
            or singular.min() < .6 or singular.max() > 1.6 or centre_step > 250):
        record["reason"] = "implausible_one_frame_warp"
        return None, None, None, None, record
    record["status"] = "observed_pair_fit"
    return h, p[inlier], p_native[inlier], score[inlier], record


def _pair(template: np.ndarray, p: np.ndarray, q: np.ndarray) -> PairConstraint:
    count = len(p)
    return PairConstraint(
        p.copy(), q.copy(), np.zeros((count, 3), np.float64),
        np.zeros((count, 3), np.float64), np.full(count, 3, np.uint8),
        template.copy(), True,
    )


def _cache_signature(motion_path: Path, keyframes: np.ndarray,
                     template: np.ndarray, lens_sequence: np.ndarray | None) -> dict:
    stat = motion_path.stat()
    return {"motion_npz": str(motion_path.resolve()), "motion_size": stat.st_size,
            "motion_mtime_ns": stat.st_mtime_ns,
            "keyframe_pts": [int(value) for value in keyframes],
            "lens_template": np.round(template[6:25], 12).tolist(),
            "keyframe_lens_sha256": (None if lens_sequence is None else hashlib.sha256(
                np.ascontiguousarray(lens_sequence[:, 6:25], np.float64).tobytes()).hexdigest())}


def _save_cache(path: Path, signature: dict, records: list, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.r_[0, np.cumsum([len(pair.xy0) for _, _, pair, _ in records])]
    xy0 = np.concatenate([pair.xy0 for _, _, pair, _ in records], axis=0) if records else np.empty((0, 2))
    xy1 = np.concatenate([pair.xy1 for _, _, pair, _ in records], axis=0) if records else np.empty((0, 2))
    writing = path.with_name(path.stem + '.writing.npz')
    np.savez_compressed(
        writing,
        signature_json=np.asarray(json.dumps(signature, separators=(',', ':'))),
        metadata_json=np.asarray(json.dumps(metadata, separators=(',', ':'))),
        bridge_pts=np.asarray([(a, b) for a, b, _, _ in records], np.int64).reshape(-1, 2),
        sigma_px=np.asarray([sigma for _, _, _, sigma in records], np.float64),
        offsets=offsets.astype(np.int64),
        xy0=xy0.astype(np.float32), xy1=xy1.astype(np.float32),
    )
    writing.replace(path)


def _load_cache(path: Path, signature: dict, template: np.ndarray):
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if json.loads(str(data['signature_json'])) != signature:
                return None
            metadata = json.loads(str(data['metadata_json']))
            pairs = data['bridge_pts'].copy()
            sigma = data['sigma_px'].copy()
            offsets = data['offsets'].copy()
            source = data['xy0'].copy()
            target = data['xy1'].copy()
        if (metadata.get('schema') != SCHEMA or len(pairs) != len(sigma)
                or len(offsets) != len(pairs) + 1 or int(offsets[-1]) != len(source)
                or len(source) != len(target)):
            return None
        records = [(int(a), int(b), _pair(template, source[offsets[i]:offsets[i+1]],
                                          target[offsets[i]:offsets[i+1]]), float(sigma[i]))
                   for i, (a, b) in enumerate(pairs)]
        return records, metadata
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def build_native_motion_links(motion_npz: str | Path, keyframe_pts: np.ndarray,
                              camera_template: np.ndarray,
                              cache_path: str | Path | None = None,
                              *, keyframe_cameras: np.ndarray | None = None,
                              ) -> tuple[list[tuple[int, int, PairConstraint, float]], dict]:
    """Return one derived motion factor for each successfully bridged keyframe gap.

    Each link uses the actual start frame's inlier LK pixel support and its
    projection through a chain of individually checked native-frame Hs.
    ``metadata['steps']`` retains every real PTS and fit quality; rejected
    steps cause their containing keyframe gap to be omitted, never guessed.
    Brown-distorted shots use a common undistorted virtual K.  When
    ``keyframe_cameras`` is supplied, each native frame's K is interpolated
    from the original keyframes before rectification and endpoint rewarp;
    otherwise the supplied template K is an explicit approximation.  Actual
    native-pixel reprojection quality is measured at every step.
    """
    path = Path(motion_npz)
    keys = np.asarray(keyframe_pts, np.int64).reshape(-1)
    template = np.asarray(camera_template, np.float64).reshape(-1)
    if len(keys) < 2 or np.any(np.diff(keys) <= 0):
        raise ValueError('keyframe_pts must be strictly increasing with at least two PTS')
    if template.shape != (25,) or not np.isfinite(template).all():
        raise ValueError('camera_template must be a finite Brown25 vector')
    lens_sequence = None if keyframe_cameras is None else np.asarray(keyframe_cameras, np.float64)
    if lens_sequence is not None and (lens_sequence.shape != (len(keys), 25)
                                      or not np.isfinite(lens_sequence).all()):
        raise ValueError('keyframe_cameras must be finite Brown25 rows matching keyframe_pts')
    signature = _cache_signature(path, keys, template, lens_sequence)
    brown = np.max(np.abs(template[11:25])) > 1e-12
    rectifier_cache = {}
    def rectifier_at(stamp: int):
        if not brown:
            return None
        if stamp not in rectifier_cache:
            camera = template.copy()
            if lens_sequence is not None:
                camera[6:11] = [np.interp(stamp, keys, lens_sequence[:, j])
                                 for j in range(6, 11)]
            rectifier_cache[stamp] = _BrownRectifier(camera, template)
        return rectifier_cache[stamp]
    if cache_path is not None:
        cached = _load_cache(Path(cache_path), signature, template)
        if cached is not None:
            return cached
    with np.load(path, allow_pickle=False) as data:
        pair_pts = data['pair_pts'].copy()
        offsets = data['offsets'].copy()
        xy0 = data['xy0'].copy()
        xy1 = data['xy1'].copy()
        fb = data['fb_error_px'].copy()
        prior = data['homography_inlier'].copy()
    if (pair_pts.ndim != 2 or pair_pts.shape[1] != 2 or len(offsets) != len(pair_pts)+1
            or offsets[0] != 0 or offsets[-1] != len(xy0)
            or not all(len(x) == len(xy0) for x in (xy1, fb, prior))):
        raise ValueError('Native LK archive has inconsistent pair/offset arrays')
    lookup = {int(a): i for i, (a, b) in enumerate(pair_pts)}
    if len(lookup) != len(pair_pts) or any(int(a) >= int(b) for a, b in pair_pts):
        raise ValueError('Native LK PTS pairs are duplicated or reversed')
    if any(int(k) not in lookup for k in keys[:-1]) or int(keys[-1]) not in set(map(int,pair_pts[:,1])):
        raise ValueError('Keyframe PTS do not fall on this native LK chain')
    records, bridge_meta, step_meta = [], [], []
    for a, b in zip(keys[:-1], keys[1:]):
        a, b = int(a), int(b)
        index = lookup[a]
        chain = np.eye(3, dtype=np.float64)
        initial_support = None
        chain_start = len(step_meta)
        rejection = None
        while int(pair_pts[index, 0]) < b:
            native_a, native_b = map(int, pair_pts[index])
            left, right = offsets[index:index+2]
            h, p, p_native, score, report = _step_model(
                xy0[left:right], xy1[left:right], fb[left:right],
                prior[left:right], native_a, native_b,
                rectifier_at(native_a), rectifier_at(native_b),
            )
            step_meta.append(report)
            if h is None:
                rejection = report['reason']
                break
            if initial_support is None:
                initial_support = (p, p_native, score)
            chain = h @ chain
            if abs(chain[2, 2]) < 1e-12 or not np.isfinite(chain).all():
                rejection = 'composed_homography_singular'
                break
            chain /= chain[2, 2]
            if native_b == b:
                break
            index += 1
            if index >= len(pair_pts) or int(pair_pts[index, 0]) != native_b:
                rejection = 'native_pts_discontinuity'
                break
            if native_b > b:
                rejection = 'keyframe_not_on_native_pair_boundary'
                break
        actual_end = int(pair_pts[index, 1])
        if rejection is None and actual_end != b:
            rejection = 'keyframe_not_on_native_pair_boundary'
        item = {"a": a, "b": b, "step_start": chain_start,
                "step_end": len(step_meta), "n_steps": len(step_meta)-chain_start,
                "status": "rejected" if rejection else "derived_motion_bridge"}
        if rejection:
            item['reason'] = rejection
            bridge_meta.append(item)
            continue
        p, p_native, scores = initial_support
        chosen = _distributed_indices(p, scores, 96)
        support = np.ascontiguousarray(p_native[chosen], np.float64)
        transported = _homography_pixels(np.ascontiguousarray(p[chosen], np.float64), chain)
        if brown:
            transported = rectifier_at(b).to_native(transported)
        sensible = np.isfinite(transported).all(axis=1)
        sensible &= (np.abs(transported[:, 0]) < FRAME_WIDTH*5)
        sensible &= (np.abs(transported[:, 1]) < FRAME_HEIGHT*5)
        support, transported = support[sensible], transported[sensible]
        if len(support) < MIN_STEP_MATCHES:
            item.update(status='rejected', reason='fewer_than_12_finite_transport_supports')
            bridge_meta.append(item)
            continue
        residual_scale = np.sqrt(sum(step['p95_native_reprojection_px']**2
                                     for step in step_meta[chain_start:]))
        sigma = float(max(1.5, 3.0*np.sqrt(item['n_steps']/30.0),
                          .75*residual_scale,
                          4.0*np.sqrt(item['n_steps']/30.0) if brown else 0.))
        records.append((a, b, _pair(template, support, transported), sigma))
        item.update(sigma_px=sigma, support_count=int(len(support)),
                    H_coordinate_domain=('common_virtual_rectified_px' if brown
                                         else 'native_source_px'),
                    H_rectified_xy=chain.tolist() if brown else None,
                    H_native_xy=chain.tolist() if not brown else None,
                    endpoint_pixels_are='composed_prediction_not_direct_match')
        bridge_meta.append(item)
    metadata = {"schema": SCHEMA,
                "kind": "observed_pixel_derived_composed_camera_motion",
                "coordinate_system": "native_source_pixels_1920x1080",
                "homography_domain": ('common_virtual_rectified_px' if brown
                                       else 'native_source_px'),
                "brown_handling": (('interpolated_keyframe_K_undistort_and_redistort'
                                    if lens_sequence is not None else
                                    'fixed_template_K_undistort_and_redistort')
                                   if brown else 'zero_Brown_native_pixels'),
                "signature": signature,
                "status": "candidate_motion_only",
                "steps": step_meta, "bridges": bridge_meta,
                "accepted_bridges": len(records), "rejected_bridges": len(bridge_meta)-len(records)}
    if cache_path is not None:
        _save_cache(Path(cache_path), signature, records, metadata)
    return records, metadata


def _camera_rays(camera: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Invert Brown/skew at the keyframe using its own original K and lens."""
    fx, fy = np.exp(camera[6:8])
    native = np.ascontiguousarray(pixels, np.float64).copy()
    native[:, 0] -= camera[10] * (native[:, 1]-camera[9]) / fy
    if np.max(np.abs(camera[11:25])) <= 1e-12:
        xy = np.column_stack(((native[:, 0]-camera[8])/fx,
                              (native[:, 1]-camera[9])/fy))
    else:
        k = np.array([[fx, 0., camera[8]], [0., fy, camera[9]], [0., 0., 1.]])
        xy = cv2.undistortPoints(native.reshape(-1, 1, 2), k,
                                 np.ascontiguousarray(camera[11:25])).reshape(-1, 2)
    rays = np.column_stack((xy, np.ones(len(xy))))
    return rays / np.linalg.norm(rays, axis=1)[:, None]


def _relative_rotation(pair: PairConstraint, source_camera: np.ndarray,
                       target_camera: np.ndarray) -> tuple[np.ndarray, float]:
    """Kabsch alignment of transported pixel rays in the two true keyframe Ks."""
    a = _camera_rays(source_camera, pair.xy0)
    b = _camera_rays(target_camera, pair.xy1)
    finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    a, b = a[finite], b[finite]
    if len(a) < MIN_STEP_MATCHES:
        raise ValueError('fewer than 12 finite motion-bridge rays')
    u, singular, vt = np.linalg.svd(b.T @ a)
    if singular[1] / singular[0] < 1e-5:
        raise ValueError('motion-bridge ray support is angularly degenerate')
    rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    predicted = a @ rotation.T
    angular = np.arccos(np.clip(np.einsum('ni,ni->n', predicted, b), -1, 1))
    return rotation, float(np.median(np.degrees(angular)))


def propagate_rotation_initializers(
    records: list[tuple[int, int, PairConstraint, float]],
    keyframe_pts: np.ndarray,
    camera25: np.ndarray,
    model_anchor_pts: np.ndarray | list[int] | set[int],
) -> tuple[np.ndarray, dict]:
    """Propagate camera rotations from model anchors through LK-derived Hs.

    Only the world-to-camera Rodrigues entries 3:6 change; the original K,
    Brown, centre and other camera entries stay untouched.  Multiple anchors
    seed a nearest-PTS graph traversal.  This is an initializer, not source
    truth or a substitute for the absolute model anchors.
    """
    pts = np.asarray(keyframe_pts, np.int64).reshape(-1)
    cameras = np.asarray(camera25, np.float64)
    if cameras.shape != (len(pts), 25) or np.any(np.diff(pts) <= 0):
        raise ValueError('camera25 must correspond to sorted keyframe PTS')
    index = {int(t): i for i, t in enumerate(pts)}
    anchors = set(map(int, model_anchor_pts)) & set(index)
    if not anchors:
        raise ValueError('At least one model anchor must seed the rotation bridge')
    adjacency: dict[int, list] = {int(t): [] for t in pts}
    edge_reports = []
    for a, b, pair, sigma in records:
        if a not in index or b not in index or a >= b:
            raise ValueError('Motion bridge endpoints are not ordered keyframe PTS')
        try:
            relative, angular = _relative_rotation(pair, cameras[index[a]], cameras[index[b]])
        except ValueError as exc:
            edge_reports.append({'a': a, 'b': b, 'status': 'rejected', 'reason': str(exc)})
            continue
        adjacency[a].append((b, relative, b-a))
        adjacency[b].append((a, relative.T, b-a))
        edge_reports.append({'a': a, 'b': b, 'status': 'ray_Kabsch',
                             'median_ray_angle_deg': angular, 'sigma_px': float(sigma)})
    best_cost = {key: 0 for key in anchors}
    source_anchor = {key: key for key in anchors}
    rotations = {key: cv2.Rodrigues(np.ascontiguousarray(cameras[index[key], 3:6]))[0]
                 for key in anchors}
    queue = [(0, key) for key in sorted(anchors)]
    heapq.heapify(queue)
    while queue:
        cost, old = heapq.heappop(queue)
        if cost != best_cost[old]:
            continue
        for new, relative, gap in adjacency[old]:
            if new in anchors:
                continue
            next_cost = cost + gap
            if next_cost >= best_cost.get(new, float('inf')):
                continue
            best_cost[new] = next_cost
            rotations[new] = relative @ rotations[old]
            source_anchor[new] = source_anchor[old]
            heapq.heappush(queue, (next_cost, new))
    output = cameras.copy()
    for stamp, rotation in rotations.items():
        if stamp not in anchors:
            output[index[stamp], 3:6] = cv2.Rodrigues(rotation)[0].ravel()
    metadata = {'kind': 'LK_composed_motion_rotation_initializer_not_observation',
                'model_anchor_pts': sorted(anchors),
                'seed_for_pts': {str(k): int(v) for k, v in source_anchor.items()},
                'propagated_count': len(rotations)-len(anchors),
                'unreached_pts': [int(k) for k in pts if int(k) not in rotations],
                'ray_Kabsch_edges': edge_reports}
    return output, metadata
