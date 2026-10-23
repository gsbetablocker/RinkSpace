"""Resolve repaired print identities from original images, with no camera model.

The same original seed identity is reused only when supported consistently by
literal source vertices or independent source-image registrations. A current
pixel always needs its own visible boundary and local appearance evidence.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, time
import av, cv2, numpy as np
from .compact_identity_revision import _array
from .compact_identity_withhold_v2 import _source_identities
from .dense_joint_proofs import _ref


def distances(uv, poly):
    delta = poly[1:] - poly[:-1]
    rel = uv[:, None, :] - poly[None, :-1, :]
    t = np.clip(np.einsum('bni,ni->bn', rel, delta) /
                np.maximum(np.sum(delta * delta, axis=1), 1e-20), 0., 1.)
    d = np.linalg.norm(rel - t[:, :, None] * delta, axis=2)
    j = np.argmin(d, axis=1)
    i = np.arange(len(uv))
    return d[i, j], (j + t[i, j]) / (len(poly) - 1)


def foot(poly, station):
    z = float(station) * (len(poly) - 1)
    j = min(int(z), len(poly) - 2)
    return poly[j] + (z - j) * (poly[j + 1] - poly[j])


def inside(hull, q):
    return cv2.pointPolygonTest(hull, tuple(map(float, q)), False) >= 0


def correlation(reference, gray, H, center):
    lo = np.maximum([0, 0], np.floor(center - 15)).astype(int)
    hi = np.minimum([reference.shape[1], reference.shape[0]], np.ceil(center + 16)).astype(int)
    x, y = lo; X, Y = hi
    if X <= x or Y <= y:
        return -1.
    T = np.array([[1., 0., -x], [0., 1., -y], [0., 0., 1.]])
    M = T @ np.linalg.inv(H)
    warp = cv2.warpPerspective(gray, M, (X-x, Y-y)).astype(float)
    orig = reference[y:Y, x:X].astype(float)
    valid = cv2.warpPerspective(np.ones(gray.shape, np.uint8), M, (X-x, Y-y)) > 0
    a, b = orig[valid], warp[valid]
    if len(a) <= 20 or a.std() <= 5 or b.std() <= 5:
        return -1.
    return float(np.corrcoef(a, b)[0, 1])


def repair(packet_path, model, source_config, groups, workers=8):
    cv2.setNumThreads(1)
    start = time.perf_counter()
    packet = json.loads(Path(packet_path).read_text())
    arrays = {k: _array(packet['arrays'][k]) for k in
              ['uv', 'native_pts', 'material_index', 'admission', 'path']}
    uv, pts, mat, admit, path = [arrays[k] for k in
                               ['uv', 'native_pts', 'material_index', 'admission', 'path']]
    labels = [x['label'] for x in model['materials']]
    indices = {s: i for i, s in enumerate(labels)}
    cfg = json.loads(Path(source_config).read_text())
    materials = set()
    for g in groups:
        g['polys'] = [np.asarray(p['source_native_uv'], np.float64) for p in g['paths']]
        g['lookup'] = {(e['material_index'], e['path']): e['canonical'] for e in g['aliases']}
        g['name_lookup'] = {(e['label'], e['path']): e['canonical'] for e in g['aliases']}
        g['reference'] = cv2.imread(g['source_image']['file'], 0)
        assert g['reference'] is not None
        g['registrations'] = {int(e['native_pts']): e['candidates'][0]
            for e in json.loads(Path(g['registration_file']).read_text()) if e['candidates']}
        for r in g['registrations'].values():
            assert r['correlation'] >= .65 and r['registration_p95_px'] <= 3.
        g['rows'] = np.flatnonzero((admit == 1) & np.isin(mat, [e['material_index'] for e in g['aliases']]))
        materials.update(e['material_index'] for e in g['aliases'])
    wanted = np.flatnonzero((admit == 1) & np.isin(mat, list(materials)))
    source_ids, _ = _source_identities(Path(packet_path), wanted.tolist())
    row_seed = {}
    for i, sid in source_ids.items():
        if sid.startswith('native_named_print_1022/') and '/seed' in sid:
            row_seed[i] = int(sid.rsplit('/seed', 1)[1])
    # Pre-existing registrations of the same original seed are witnesses, not a
    # fitted-camera nearest-curve assignment. Conflicting witnesses stay unresolved.
    for g in groups:
        witnessed = defaultdict(lambda: defaultdict(list))
        for file in g.get('witness_files', []):
            doc = json.loads(Path(file).read_text())
            for e in doc['rows']:
                sid = e.get('source_row_id', '')
                key = (e['new_label'], e.get('new_path', 0))
                if key not in g['name_lookup'] or not sid.startswith('native_named_print_1022/') or '/seed' not in sid:
                    continue
                si = int(sid.rsplit('/seed', 1)[1]); j = g['name_lookup'][key]
                witnessed[si][j].append((e['raw_row'], str(file)))
        g['known'] = {}
        gseeds = {row_seed[int(i)] for i in g['rows'] if int(i) in row_seed}
        for si in gseeds:
            s = cfg['seeds'][si]; candidates = set(witnessed[si]); proof = {}
            if int(s['native_pts']) == int(g['source_pts']):
                d = np.array([distances(np.asarray(s['uv'], float)[None], p)[0][0] for p in g['polys']])
                exact = np.flatnonzero(d <= 1e-7)
                if len(exact) == 1:
                    candidates.add(int(exact[0]))
                    proof['literal_source_boundary_distance_px'] = float(d[exact[0]])
            if len(candidates) == 1:
                j = next(iter(candidates)); proof.update(seed_index=si, source_uv=s['uv'], native_pts=s['native_pts'],
                    source_annotation=cfg['source_anchors'][s['source_index']]['source'],
                    witness_rows=[q[0] for q in witnessed[si].get(j, [])],
                    witness_files=sorted({q[1] for q in witnessed[si].get(j, [])}))
                g['known'][si] = (j, proof)
        g['by_pts'] = {}
        for pt in np.unique(pts[g['rows']]):
            g['by_pts'][int(pt)] = g['rows'][pts[g['rows']] == pt]
        g['seed_by_pts'] = defaultdict(list)
        for si in gseeds:
            g['seed_by_pts'][int(cfg['seeds'][si]['native_pts'])].append(si)
    tasks = defaultdict(list)
    for gi, g in enumerate(groups):
        for pt in set(g['by_pts']) | set(g['seed_by_pts']):
            if pt in g['registrations']:
                tasks[pt].append(gi)
    # Decode each native exposure once per pass, shared by every visible print.
    def get_frame(pt):
        with av.open(cfg['source_video']) as con:
            st = con.streams.video[0]; st.thread_count = 1
            con.seek(int(pt), stream=st, backward=True)
            for f in con.decode(st):
                if f.pts == pt: return f.to_ndarray(format='gray')
                if f.pts > pt: break
        raise ValueError('missing exact native exposure '+str(pt))
    def work(pt, seed_pass):
        gray = get_frame(pt); result = []
        for gi in tasks[pt]:
            g = groups[gi]; reg = g['registrations'][pt]
            H = np.asarray(reg['H']); hull = np.asarray(reg['source_convex_hull'], np.float32)
            polys = [cv2.perspectiveTransform(p[None], H)[0] for p in g['polys']]
            ids = g['seed_by_pts'].get(pt, []) if seed_pass else g['by_pts'].get(pt, [])
            if not len(ids): continue
            measured = np.asarray([cfg['seeds'][i]['uv'] for i in ids], float) if seed_pass else np.asarray(uv[ids], float)
            at = cv2.perspectiveTransform(measured[None], np.linalg.inv(H))[0]
            pairs = [distances(measured, p) for p in polys]
            dd = np.array([q[0] for q in pairs]).T; ss = np.array([q[1] for q in pairs]).T
            order = np.argsort(dd, axis=1); stats = Counter(); changes = []; confirmations = []
            for k, ii in enumerate(ids):
                i = int(ii); best, second = map(int, order[k, :2]); old = None
                if not inside(hull, at[k]): stats['outside_registered_source_support'] += 1; continue
                direct = dd[k, best] <= 3 and dd[k, second] - dd[k, best] >= 3
                proof = None; chosen = best if direct else None
                if not seed_pass:
                    old = g['lookup'].get((int(mat[i]), int(path[i])))
                    seed = row_seed.get(i); known = g['known'].get(seed)
                    if chosen is None and known is not None and dd[k, known[0]] <= 3:
                        chosen, proof = known
                if chosen is None: stats['unresolved_current_pixel_identity'] += 1; continue
                f = foot(g['polys'][chosen], ss[k, chosen])
                if not inside(hull, f): stats['boundary_outside_registered_support'] += 1; continue
                corr = correlation(g['reference'], gray, H, f)
                if not np.isfinite(corr) or corr < .65:
                    stats['current_pixel_appearance_unconfirmed'] += 1; continue
                if seed_pass:
                    confirmations.append((i, chosen, dict(seed_index=i, source_uv=cfg['seeds'][i]['uv'], native_pts=pt,
                        source_annotation=cfg['source_anchors'][cfg['seeds'][i]['source_index']]['source'],
                        identity_anchor='original_seed_native_image_registration', nearest_px=float(dd[k, chosen]),
                        next_boundary_px=float(dd[k, second]), local_source_pixel_correlation=corr)))
                    continue
                if old == chosen: stats['same_repaired_contour_confirmed'] += 1; continue
                if old is None: stats['unmapped_existing_path'] += 1; continue
                target = g['paths'][chosen]
                e = dict(raw_row=i, source_row_id=source_ids[i], native_pts=pt, uv=uv[i].tolist(),
                    old_label=labels[int(mat[i])], old_material_index=int(mat[i]), old_path=int(path[i]),
                    new_label=target['label'], new_material_index=indices[target['label']], new_path=target['path'],
                    new_station=float(ss[k, chosen]), source_station_is_numeric_initializer=True,
                    nearest_px=float(dd[k, chosen]), next_boundary_px=float(min(dd[k,j] for j in range(len(polys)) if j!=chosen)),
                    old_boundary_px=float(dd[k, old]), registration_kind=reg['kind'],
                    registration_correlation=reg['correlation'], local_source_pixel_correlation=corr,
                    identity_basis='current_original_image_unique_boundary' if direct else 'original_seed_and_current_visible_boundary')
                if proof is not None: e['original_source_seed'] = proof
                changes.append(e); stats['repaired_identity'] += 1
            result.append((gi, pt, confirmations, changes, dict(stats), len(ids)))
        return result
    # First resolve all source anchors, then apply their consistent identities.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, pt, True) for pt in tasks if any(groups[i]['seed_by_pts'].get(pt) for i in tasks[pt])]
        for future in as_completed(futures):
            for gi, pt, confirmations, _, _, _ in future.result():
                g = groups[gi]
                for si, j, proof in confirmations:
                    old = g['known'].get(si)
                    if old is not None and old[0] != j:
                        g['known'].pop(si); continue
                    g['known'][si] = (j, proof)
    print(json.dumps(dict(phase='source_identities_resolved', groups={g['name']:len(g['known']) for g in groups}, seconds=time.perf_counter()-start)), flush=True)
    changes = []; coverage = []; complete = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, pt, False) for pt in tasks if any(groups[i]['by_pts'].get(pt) is not None for i in tasks[pt])]
        for future in as_completed(futures):
            for gi, pt, _, rows, stats, n in future.result():
                changes.extend(rows); coverage.append(dict(group=groups[gi]['name'], native_pts=pt, rows=n, **stats))
            complete += 1
            if complete % 100 == 0:
                print(json.dumps(dict(phase='current_native_exposures', frames=complete, rebound=len(changes), seconds=time.perf_counter()-start)), flush=True)
    unique = {}
    for e in changes:
        if e['raw_row'] in unique and (e['new_material_index'], e['new_path']) != (unique[e['raw_row']]['new_material_index'], unique[e['raw_row']]['new_path']):
            raise ValueError('inconsistent literal source groups at row '+str(e['raw_row']))
        unique[e['raw_row']] = e
    missing = {g['name']:dict(frames=sum(pt not in g['registrations'] for pt in g['by_pts']),
        rows=sum(len(ids) for pt, ids in g['by_pts'].items() if pt not in g['registrations'])) for g in groups}
    return dict(rows=sorted(unique.values(), key=lambda e:e['raw_row']), coverage=coverage,
                missing_current_image_registration=missing, seconds=time.perf_counter()-start,
                raw_source_pixels_changed=False, fitted_camera_used=False)
