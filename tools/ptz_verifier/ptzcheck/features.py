"""Image-only correspondences, track-disjoint holdout and train-only homographies."""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict, deque
import hashlib, json
import cv2
import numpy as np
from .geometry import homography_points
from .io import write_json

DEFAULT_MASKS = [[0., 0., .32, .16], [.84, .84, 1., 1.], [.34, .12, .68, 1.]]
# Conservative S01 broadcast overlay + central skater exclusion, NOT semantic segmentation.


def detect(frame: dict, width: int, height: int, max_width=1440,
           masks=None, max_features=2600) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(frame['image'], cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (height, width): raise ValueError('Missing/wrong-size frame')
    nw = min(max_width, width); nh = int(round(height*nw/width))
    sx, sy = nw/width, nh/height
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    mask = np.full((nh, nw), 255, np.uint8)
    mask[:5] = 0; mask[-5:] = 0; mask[:, :5] = 0; mask[:, -5:] = 0
    for x0, y0, x1, y1 in (DEFAULT_MASKS if masks is None else masks):
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1): raise ValueError('Invalid normalized mask rectangle')
        mask[int(y0*nh):int(np.ceil(y1*nh)), int(x0*nw):int(np.ceil(x1*nw))] = 0
    if not hasattr(cv2, 'SIFT_create'): raise RuntimeError('OpenCV with SIFT required')
    sift = cv2.SIFT_create(nfeatures=max_features*2, contrastThreshold=0.018, edgeThreshold=12)
    keypoints, desc = sift.detectAndCompute(image, mask)
    if desc is None: return np.zeros((0, 2)), np.zeros((0, 128), np.float32)
    order = sorted(range(len(keypoints)), key=lambda k: (-keypoints[k].response, k))
    # One node per location: SIFT multiple orientations must not leak across train/test.
    kept, occupied, cells = [], set(), defaultdict(int)
    for k in order:
        x, y = keypoints[k].pt
        pixel = (int(round(x)), int(round(y)))
        if any((pixel[0]+a, pixel[1]+b) in occupied for a in (-1, 0, 1) for b in (-1, 0, 1)): continue
        cell = (min(7, int(x/nw*8)), min(4, int(y/nh*5)))
        if cells[cell] >= 100: continue
        kept.append(k); occupied.add(pixel); cells[cell] += 1
        if len(kept) >= max_features: break
    xy = np.array([keypoints[k].pt for k in kept], np.float64) / [sx, sy]
    d = desc[kept].astype(np.float32)
    # RootSIFT descriptor normalization; this is independent of all camera estimates.
    d /= np.maximum(d.sum(axis=1, keepdims=True), 1e-10)
    return xy, np.sqrt(d)


def mutual_matches(a: np.ndarray, b: np.ndarray, ratio=.72) -> list[tuple[int, int, float]]:
    if len(a) < 2 or len(b) < 2: return []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    ab, ba = bf.knnMatch(a, b, k=2), bf.knnMatch(b, a, k=2)
    valid_b = {m.queryIdx: m.trainIdx for pair in ba if len(pair) == 2
               for m, n in [pair] if m.distance < ratio*n.distance}
    return [(m.queryIdx, m.trainIdx, float(m.distance)) for pair in ab if len(pair) == 2
            for m, n in [pair] if m.distance < ratio*n.distance and valid_b.get(m.trainIdx) == m.queryIdx]


class Tracks:
    """Union-find, forbidding two different keypoints in a frame in one track."""
    def __init__(self): self.parent = {}; self.frames = {}; self.members = {}
    def add(self, node):
        if node not in self.parent:
            self.parent[node] = node; self.frames[node] = {node[0]}; self.members[node] = {node}
    def root(self, n):
        self.add(n)
        if self.parent[n] != n: self.parent[n] = self.root(self.parent[n])
        return self.parent[n]
    def union(self, a, b):
        a, b = self.root(a), self.root(b)
        if a == b: return True
        if self.frames[a] & self.frames[b]: return False
        if len(self.members[a]) < len(self.members[b]): a, b = b, a
        self.parent[b] = a; self.frames[a] |= self.frames.pop(b); self.members[a] |= self.members.pop(b)
        return True


def partition_track(key: str, seed: int) -> bool:
    """True is holdout. Deterministic whole-track split before homography estimation."""
    h = hashlib.sha256(f'{seed}:{key}'.encode()).digest()
    return int.from_bytes(h[:8], 'big') % 10 < 3


def coverage(xy: np.ndarray, width: int, height: int) -> dict:
    if len(xy) < 3: return dict(hull_fraction=0., x_span=0., y_span=0., cells=0)
    hull = cv2.convexHull(np.asarray(xy, np.float32))
    cells = {(min(5, int(x/width*6)), min(3, int(y/height*4))) for x, y in xy}
    return dict(hull_fraction=float(cv2.contourArea(hull)/(width*height)),
                x_span=float(np.ptp(xy[:, 0])/width), y_span=float(np.ptp(xy[:, 1])/height), cells=len(cells))


def graph_components(n, pairs):
    adj = [[] for _ in range(n)]
    for i, j in pairs: adj[i].append(j); adj[j].append(i)
    seen = set(); out = []
    for start in range(n):
        if start in seen: continue
        todo = [start]; seen.add(start); component = []
        while todo:
            i = todo.pop(); component.append(i)
            for j in adj[i]:
                if j not in seen: seen.add(j); todo.append(j)
        out.append(sorted(component))
    return sorted(out, key=lambda x: (-len(x), x[0]))


def spatial_cap(rows, xy, width, height, limit=160):
    # Limit repetitions in one lettering/texture region. Applied separately to train and test.
    cells = defaultdict(list)
    for r in rows:
        p = xy[r]
        c = (min(7, int(p[0]/width*8)), min(4, int(p[1]/height*5)))
        cells[c].append(r)
    result = []
    while cells and len(result) < limit:
        for c in sorted(list(cells)):
            result.append(cells[c].pop(0))
            if not cells[c]: del cells[c]
            if len(result) == limit: break
    return result


def build_graph(frames, width, height, out: Path, *, seed=20261027,
                max_width=1440, masks=None, max_per_edge=160, offsets=(1, 2, 4, 8), component='largest') -> dict:
    cv2.setNumThreads(2); cv2.setRNGSeed(seed % 2147483647)
    points, descriptors = [], []
    for n, f in enumerate(frames):
        cache = out/'features'; cache.mkdir(exist_ok=True)
        fp = cache/f'{f["pts"]}.npz'
        fingerprint = hashlib.sha256(json.dumps(dict(max_width=max_width, masks=masks, width=width, height=height, source_image_hash=f.get("decoded_bgr_sha256"), version=2), sort_keys=True).encode()).hexdigest()
        if fp.is_file():
            with np.load(fp, allow_pickle=False) as z:
                if str(z['fingerprint']) != fingerprint: raise ValueError('Feature cache settings changed')
                p, d = z['p'], z['d']
        else:
            p, d = detect(f, width, height, max_width=max_width, masks=masks)
            np.savez_compressed(fp, p=p, d=d, fingerprint=fingerprint)
        points.append(p); descriptors.append(d)
        print(f'features {n+1}/{len(frames)}: {len(p)} at PTS {f["pts"]}', flush=True)
    candidates = []
    for i in range(len(frames)):
        for dt in offsets:
            j = i+dt
            if j >= len(frames): continue
            for a, b, score in mutual_matches(descriptors[i], descriptors[j]):
                candidates.append((score, i, a, j, b))
    tracks = Tracks()
    for _, i, a, j, b in sorted(candidates): tracks.union((i, a), (j, b))
    roots = {n: tracks.root(n) for n in list(tracks.parent)}
    track_keys = {r: ';'.join(f'{frames[i]["pts"]}:{k}' for i, k in sorted(members))
                  for r, members in tracks.members.items()}
    is_test = {r: partition_track(key, seed) for r, key in track_keys.items()}
    track_id = {r: k for k, r in enumerate(sorted(track_keys))}
    pairs = defaultdict(list)
    for score, i, a, j, b in sorted(candidates):
        ra, rb = roots[(i, a)], roots[(j, b)]
        if ra != rb or len(tracks.frames[ra]) < 3: continue
        pairs[(i, j)].append((a, b, is_test[ra], track_id[ra], score))
    edge_log, accepted = [], []
    for (i, j), matches in sorted(pairs.items()):
        a = np.asarray([points[i][m[0]] for m in matches]); b = np.asarray([points[j][m[1]] for m in matches])
        test = np.array([m[2] for m in matches], bool)
        log = dict(i=i, j=j, source_pts=[frames[i]['pts'], frames[j]['pts']], candidates=len(matches),
                   training_candidates=int((~test).sum()), holdout_candidates=int(test.sum()))
        edge_log.append(log)
        if (~test).sum() < 14:
            log['rejected'] = 'too_few_training_tracks'; continue
        H, _ = cv2.findHomography(a[~test], b[~test], cv2.RANSAC, 5., maxIters=4000, confidence=.999)
        if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
            log['rejected'] = 'homography_failure'; continue
        forward = np.linalg.norm(homography_points(H, a)-b, axis=1)
        backward = np.linalg.norm(homography_points(np.linalg.inv(H), b)-a, axis=1)
        e = np.sqrt((forward*forward+backward*backward)/2)
        # Keep broad gate to reduce bias toward ideal pinhole homographies; report rejects explicitly.
        gate = e <= 8.
        keep_train = np.flatnonzero(~test & gate)
        if len(keep_train) < 14:
            log['rejected'] = 'too_few_consistent_training_tracks'; continue
        cov_a, cov_b = coverage(a[keep_train], width, height), coverage(b[keep_train], width, height)
        log.update(training_inliers=len(keep_train), holdout_gated=int((test & gate).sum()),
                   holdout_rejected=int((test & ~gate).sum()), coverage_a=cov_a, coverage_b=cov_b)
        if any(c['x_span'] < .25 or c['y_span'] < .07 or c['hull_fraction'] < .015 or c['cells'] < 3 for c in (cov_a, cov_b)):
            log['rejected'] = 'insufficient_background_spread'; continue
        H_refit, _ = cv2.findHomography(a[keep_train], b[keep_train], 0)
        if H_refit is not None and np.isfinite(H_refit).all(): H = H_refit
        chosen = spatial_cap(keep_train, a, width, height, max_per_edge)
        chosen += spatial_cap(np.flatnonzero(test), a, width, height, max_per_edge)
        log['accepted'] = True
        accepted.append(dict(i=i, j=j, H=H, a=a[chosen], b=b[chosen], test=test[chosen],
            screened=gate[chosen], track=np.array([matches[k][3] for k in chosen], np.int64)))
    components = graph_components(len(frames), [(e['i'], e['j']) for e in accepted])
    if component == 'all':
        active_components = [c for c in components if len(c) >= 4]
    elif component == 'largest':
        active_components = [components[0]]
    else:
        k = int(component)
        if k < 0 or k >= len(components): raise ValueError('Component index out of range')
        active_components = [components[k]]
    selected = sorted({i for c in active_components for i in c})
    if len(selected) < 4 or any(len(c) < 4 for c in active_components):
        write_json(out/'graph_diagnostics.json', dict(edges=edge_log, components=components))
        raise ValueError('No connected background track graph with >=4 frames; see graph_diagnostics.json')
    index = {old: new for new, old in enumerate(selected)}
    arrays = {k: [] for k in ('i', 'j', 'a', 'b', 'test', 'screened', 'track', 'edge')}
    edges = []
    for e in accepted:
        if e['i'] not in index or e['j'] not in index: continue
        edge_index = len(edges); n = len(e['a'])
        edges.append(dict(i=index[e['i']], j=index[e['j']], H=e['H']))
        for k in ('a', 'b', 'test', 'screened', 'track'): arrays[k].append(e[k])
        arrays['i'].append(np.full(n, index[e['i']], np.int64))
        arrays['j'].append(np.full(n, index[e['j']], np.int64))
        arrays['edge'].append(np.full(n, edge_index, np.int64))
    data = {k: np.concatenate(v) for k, v in arrays.items()}
    train = ~data['test'] & data['screened']
    if np.intersect1d(data['track'][train], data['track'][data['test']]).size:
        raise RuntimeError('Track split leakage')
    meta = dict(schema='rinkspace.ptz-track-graph.v1', width=width, height=height, frames=[frames[i] for i in selected],
        candidate_frame_count=len(frames), selected_original_indices=selected,
        anchors=[index[c[0]] for c in active_components],
        active_components=[[index[i] for i in c] for c in active_components],
        component_policy=component,
        excluded_frames=[f for i, f in enumerate(frames) if i not in index], components=components,
        training_matches=int(train.sum()), holdout_matches=int(data['test'].sum()),
        training_tracks=int(len(np.unique(data['track'][train]))),
        holdout_tracks=int(len(np.unique(data['track'][data['test']]))),
        descriptor_track_split='sha256 whole-track; >=3 distinct frames; before geometry',
        seed=seed, max_per_edge=max_per_edge, masks=DEFAULT_MASKS if masks is None else masks,
        model_selection_uses_holdout=False, edges=edges)
    np.savez_compressed(out/'matches.npz', **data)
    write_json(out/'graph.json', meta)
    write_json(out/'graph_diagnostics.json', dict(edges=edge_log, components=components))
    return dict(data=data, meta=meta)


def load_graph(directory: Path) -> dict:
    meta = json.loads((directory/'graph.json').read_text(encoding='utf-8'))
    with np.load(directory/'matches.npz', allow_pickle=False) as z: data = {k: z[k] for k in z.files}
    for edge in meta['edges']: edge['H'] = np.asarray(edge['H'])
    return dict(data=data, meta=meta)
