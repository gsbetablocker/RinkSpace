"""Portable audit of the 1103 private supplement. No project imports or solver.

Standard-library integrity check; --metrics/--restore need NumPy, --pixels PyAV.
Array overlays replace rows or XOR bytes, never add floating-point deltas.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import io
import json
import sys
import zipfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Store:
    def __init__(self, root=None, archive=None):
        self.root = Path(root or Path(__file__).resolve().parent)
        self.archive = zipfile.ZipFile(archive) if archive else None
        self.cache = {}
        self.index = self.json('ARRAY_INDEX.json')

    def read(self, name):
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
            raise ValueError('Unsafe member: ' + name)
        return self.archive.read(name) if self.archive else (self.root / name).read_bytes()

    def json(self, name):
        return json.loads(self.read(name).decode('utf-8-sig'))

    def npy(self, name):
        import numpy as np
        value = np.load(io.BytesIO(self.read(name)), allow_pickle=False)
        encoding = self.index.get('blob_encodings', {}).get(name)
        if encoding:
            if encoding['encoding'] != 'byte_planes':
                raise ValueError('Unknown blob encoding')
            value = np.ascontiguousarray(value.T).view(np.dtype(encoding['dtype'])).reshape(encoding['shape'])
        return value

    def array(self, dataset, key):
        import numpy as np
        identity = (dataset, key)
        if identity in self.cache:
            return self.cache[identity]
        row = self.index['datasets'][dataset]['arrays'][key]
        storage = row['storage']
        if storage == 'npy':
            value = self.npy(row['member'])
        elif storage == 'alias':
            value = self.array(*row['target'])
        elif storage == 'slice':
            source = self.array(*row['target'])
            value = source[tuple(slice(*x) for x in row['slices'])]
        elif storage == 'rows':
            value = self.array(*row['base']).copy()
            indices = self.npy(row['indices'])
            if indices.dtype.kind not in 'iu' or not np.all(np.diff(indices) > 0):
                raise ValueError('Invalid replacement row indices')
            if len(indices) and (indices[0] < 0 or indices[-1] >= len(value)):
                raise ValueError('Replacement rows out of range')
            value[indices] = self.npy(row['values'])
        elif storage == 'xor_bytes':
            base = self.array(*row['base'])
            planes = np.frombuffer(base.tobytes(order='C'), np.uint8).reshape(-1, base.dtype.itemsize).T
            decoded = np.bitwise_xor(planes, self.npy(row['mask']))
            value = np.ascontiguousarray(decoded.T).view(base.dtype).reshape(base.shape)
        else:
            raise ValueError('Unknown storage: ' + storage)
        if value.dtype.str != row['dtype'] or list(value.shape) != row['shape']:
            raise ValueError('Array metadata mismatch: ' + str(identity))
        if sha(value.tobytes(order='C')) != row['raw_sha256']:
            raise ValueError('Restored array hash mismatch: ' + str(identity))
        self.cache[identity] = value
        return value

    def dataset(self, name):
        return {key: self.array(name, key) for key in self.index['datasets'][name]['arrays']}


def integrity(store):
    manifest = store.json('MANIFEST.json')
    names = [row['path'] for row in manifest['files']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate manifest path')
    if store.archive:
        actual = store.archive.namelist()
        if len(actual) != len(set(actual)) or set(actual) != set(names) | {'MANIFEST.json'}:
            raise ValueError('Unlisted or duplicate archive member')
        bad = store.archive.testzip()
        if bad:
            raise ValueError('ZIP CRC failure: ' + bad)
    for row in manifest['files']:
        raw = store.read(row['path'])
        if len(raw) != row['bytes'] or sha(raw) != row['sha256']:
            raise ValueError('Member hash/size mismatch: ' + row['path'])
    return dict(files=len(names), all_hashes_and_sizes_match=True,
                zip_crc_checked=bool(store.archive), no_network_or_author_paths_used=True)


def stats(value):
    import numpy as np
    a = np.asarray(value, float).ravel()
    finite = a[np.isfinite(a)]
    return dict(n=len(a), finite_n=len(finite), missing_n=len(a)-len(finite),
                median=float(np.median(finite)) if len(finite) else None,
                p95=float(np.percentile(finite, 95)) if len(finite) else None,
                max=float(np.max(finite)) if len(finite) else None)


def compare(actual, expected, label):
    import numpy as np
    for key, value in actual.items():
        if key not in expected:
            continue
        target = expected[key]
        if isinstance(value, dict):
            compare(value, target, label + '/' + key)
        elif value is None or isinstance(value, (bool, str)):
            if value != target:
                raise ValueError(label + '/' + key + ' differs')
        else:
            np.testing.assert_allclose(value, target, rtol=1e-9, atol=1e-8,
                                       err_msg=label + '/' + key)


def masks(pts, windows):
    import numpy as np
    result = {f'w{i+1}': (pts >= round(a*12800)) & (pts <= round(b*12800))
              for i, (a, b) in enumerate(windows)}
    result['all_windows'] = np.logical_or.reduce(list(result.values()))
    return result


def nearest(source, vertices, edges):
    import numpy as np
    source, vertices, edges = np.asarray(source), np.asarray(vertices), np.asarray(edges, int)
    if not len(edges):
        return np.full(len(source), np.nan)
    a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
    v = b-a
    alpha = np.clip(np.einsum('mkc,kc->mk', source[:, None]-a[None], v) /
                    np.maximum(np.sum(v*v, axis=1), 1e-16), 0, 1)
    squared = np.sum((a[None]+alpha[:, :, None]*v[None]-source[:, None])**2, axis=2)
    return np.sqrt(np.min(squared, axis=1))


def hull_indices(points):
    """Monotone hull of the same float32 coordinates used in the old report."""
    import numpy as np
    p = np.asarray(points, np.float32)
    ordered = sorted(range(len(p)), key=lambda i: (float(p[i, 0]), float(p[i, 1])))
    def cross(i, j, k):
        a, b, c = p[i].astype(float), p[j].astype(float), p[k].astype(float)
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    halves = []
    for ids in (ordered, ordered[::-1]):
        h = []
        for i in ids:
            while len(h) > 1 and cross(h[-2], h[-1], i) <= 0:
                h.pop()
            h.append(i)
        halves.extend(h[:-1])
    return halves


def body_metrics(store, reports, packets):
    import numpy as np
    obs = store.dataset('observations_fullbody')
    inp = store.dataset('body_input')
    pts = obs['pts']
    np.testing.assert_array_equal(pts, np.arange(283136, 575489, 256))
    report = reports['fullbody']
    windows = masks(pts, report['window_seconds'])
    summary = {}
    checked = 0
    for name in ('reference', 'legacy', 'full'):
        packet = packets[name]
        np.testing.assert_array_equal(packet['pts'], pts)
        for denominator, observed, use in (
            ('adopted', obs['uv'], obs['use']),
            ('original_same_denominator', obs['raw_coco_uv'], obs['use']),
            ('original_original_denominator', obs['raw_coco_uv'], inp['source_qualified'])):
            error = np.linalg.norm(packet['COCO_uv']-observed, axis=-1)
            for scope, mask in windows.items():
                for group, ids in report['groups'].items():
                    actual = stats(error[:, ids][use[:, ids].astype(bool) & mask[:, None]])
                    expected = report['body_pixels'][name][denominator][scope]['groups'][group]
                    compare(actual, expected, f'fullbody/{name}/{denominator}/{scope}/{group}')
                    checked += 1
                    if denominator == 'adopted' and scope == 'all_windows' and group == 'all':
                        summary[name] = actual
    all_mask = windows['all_windows']
    minimum = packets['full']['gear_world_mid'][all_mask, :, 14:, 2].min(axis=(1, 2))
    plane = dict(source_frames=int(all_mask.sum()),
                 frames_more_than_1cm_below=int((minimum < -.01).sum()),
                 lowest_m=float(minimum.min()))
    outside = {}
    for name in ('legacy', 'full'):
        outside[name] = {}
        for key in ('P_world_mid', 'Q_world', 'local_R_mid'):
            equal = np.array_equal(packets[name][key][~all_mask], packets['reference'][key][~all_mask])
            if not equal:
                raise ValueError(f'{name}/{key}: outside-window state differs')
            outside[name][key] = bool(equal)
    return dict(pixel_statistic_groups_checked=checked, adopted_same_points_px=summary,
                blade_plane=plane, outside_window_state_exact=outside)


def boot_metrics(store, report, packets):
    import numpy as np
    obs = store.dataset('observations_boot')
    old, new = packets['full'], packets['boot']
    pts = old['pts']
    m = masks(pts, ((35.2, 36.8), (40, 41.5)))
    scopes = dict(all_s01=np.ones(len(pts), bool), body_refined=m['all_windows'],
                  body_unchanged=~m['all_windows'], w1=m['w1'], w2=m['w2'])
    body_groups = dict(all=list(range(17)), ankle_proxy=[15, 16],
                       upper_limb=[7, 8, 9, 10], lower_limb=[13, 14, 15, 16])
    for mode, packet in (('open', old), ('fitted', new)):
        error = np.linalg.norm(packet['COCO_uv']-obs['uv'], axis=-1)
        for scope, mask in scopes.items():
            for group, ids in body_groups.items():
                actual = stats(error[:, ids][obs['use'][:, ids].astype(bool) & mask[:, None]])
                compare(actual, report['body_pixels'][mode][scope][group], f'boot/{mode}/{scope}/{group}')
    equipment = report['equipment_pixels']
    visible = {mode: {(r['pts'], r['foot']): r['edges'] for r in rows}
               for mode, rows in equipment['visibility'].items()}
    recomputed = []
    for row in equipment['records']:
        source = np.asarray(row['source_uv'], float)
        fi = int(np.searchsorted(pts, row['pts']))
        assert pts[fi] == row['pts']
        values = {}
        for mode, packet in (('closed', old), ('open', old), ('fitted', new)):
            vertices = packet['equipment_uv'][fi, row['foot']]
            if row['kind'] == 'outer_boot_contour':
                if mode == 'closed':
                    h = hull_indices(vertices[:14])
                    edges = list(zip(h, h[1:]+h[:1]))
                else:
                    edges = visible[mode][(row['pts'], row['foot'])]
                distance = nearest(source, vertices[:14], edges)
            elif row['kind'] == 'outer_boot_heel_toe':
                distance = np.array([nearest(source[i:i+1], vertices, [edge])[0]
                                     for i, edge in enumerate(((0, 7), (1, 8)))])
            else:
                cumulative = np.r_[0, np.cumsum(np.linalg.norm(np.diff(source, axis=0), axis=1))]
                targets = np.linspace(0, cumulative[-1], 24)
                sampled = np.stack([np.interp(targets, cumulative, source[:, i]) for i in (0, 1)], axis=1)
                candidates = (vertices[14:31], vertices[31:48])
                edges = list(zip(range(16), range(1, 17)))
                tests = [nearest(sampled, line, edges) for line in candidates]
                selected = int(np.argmin([d@d for d in tests]))
                distance = nearest(source, candidates[selected], edges)
                assert row['modes'][mode]['selected_candidate'] == ('blade_left', 'blade_right')[selected]
            expected = np.asarray([np.nan if x is None else x for x in row['modes'][mode]['residual_px']])
            np.testing.assert_allclose(distance, expected, rtol=1e-9, atol=1e-8, equal_nan=True)
            values[mode] = distance
        recomputed.append((row, values))
    equipment_summary = {}
    for scope, mask in scopes.items():
        allowed = set(map(int, pts[mask]))
        for kind, expected in equipment['groups'][scope].items():
            for mode in ('closed', 'open', 'fitted'):
                values = [v for row, out in recomputed if row['pts'] in allowed and row['kind'] == kind for v in out[mode]]
                result = stats(values)
                compare(result, expected['modes'][mode], f'boot/{scope}/{kind}/{mode}')
                if scope == 'all_s01':
                    equipment_summary.setdefault(kind, {})[mode] = result
    diagnostics = store.dataset('boot_baseline_diagnostics')
    internal = {}
    for scope, mask in scopes.items():
        for key in ('cavity_penetration_mid', 'heel_drift_mid', 'forefoot_error_mid'):
            for mode, packet in (('x0', diagnostics), ('fitted', new)):
                result = stats(packet[key][mask])
                compare(result, report['internal_fit']['groups'][scope][key][mode]['combined'], f'boot/{key}/{scope}/{mode}')
                if scope == 'all_s01':
                    internal.setdefault(key, {})[mode] = result
    plane = {}
    for mode, packet in (('open', old), ('fitted', new)):
        minimum = packet['gear_world_mid'][:, :, 14:, 2].min(axis=-1)
        plane[mode] = dict(below_z0_samples=int((minimum < 0).sum()),
                           below_z0_depth_m=stats(np.maximum(-minimum, 0)))
        compare(plane[mode], report['state']['groups']['all_s01']['blade_ice_height'][mode], 'boot/blades/'+mode)
    return dict(original_source_records_recomputed=len(recomputed), equipment_px=equipment_summary,
                internal_diagnostic_summary_m=internal, blade_plane=plane,
                visibility_edges='saved finite edge selection; source distances independently recomputed',
                cavity_scope='statistics of saved approximate-model diagnostics; no full-surface collision replay')


def opening_metrics(store, report, packets):
    import numpy as np
    obs = store.dataset('observations_opening')
    pts = obs['pts']
    output = {}
    for scope, end in (('opening_117_frames', 312832), ('complete_refinement_145_frames', 320000)):
        mask = pts <= end
        output[scope] = {}
        for mode, packet in (('parent', packets['boot']), ('fitted', packets['opening'])):
            xy = packet['P_world_mid'][mask, :2]
            native = packet['native22_world_mid'][mask]
            torso = native[:, 12]-native[:, 0]
            angles = np.rad2deg(np.arccos(np.clip(torso[:, 2]/np.linalg.norm(torso, axis=1), -1, 1)))
            minimum = packet['gear_world_mid'][mask, :, 14:, 2].min(axis=(1, 2))
            result = dict(root_XY=dict(frames=len(xy), finite=bool(np.isfinite(xy).all()),
                         net_displacement_m=float(np.linalg.norm(xy[-1]-xy[0])),
                         maximum_pairwise_span_m=float(np.linalg.norm(xy[:, None]-xy[None], axis=-1).max()),
                         path_length_m=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())),
                         torso_from_world_up_deg=stats(angles),
                         blade_minimum_z_m=dict(summary=stats(minimum), range=[float(minimum.min()), float(minimum.max())],
                                                exposures_below_plane=int((minimum < 0).sum())),
                         upper_body_source_px={})
            error = np.linalg.norm(packet['COCO_uv']-obs['uv'], axis=-1)
            for name, ids in (('head_shoulders', list(range(7))), ('arms', list(range(7, 11)))):
                result['upper_body_source_px'][name] = stats(error[:, ids][obs['use'][:, ids].astype(bool) & mask[:, None]])
            horizontal = packet['Q_world'][mask, :2, 2]
            valid = np.isfinite(horizontal).all(axis=1) & (np.linalg.norm(horizontal, axis=1) > 1e-8)
            if valid.all():
                heading = np.rad2deg(np.unwrap(np.arctan2(horizontal[:, 1], horizontal[:, 0])))
                result['root_heading'] = dict(change_deg=float(heading[-1]-heading[0]), valid_exposures=int(valid.sum()))
            compare(result, report[scope][mode], f'opening/{scope}/{mode}')
            output[scope][mode] = result
    after = pts >= 320000
    exact = {}
    for key in ('P_world_mid', 'Q_world', 'local_R_mid', 'native22_world_mid', 'gear_world_mid'):
        exact[key] = packets['boot'][key][after].tobytes() == packets['opening'][key][after].tobytes()
        if not exact[key]:
            raise ValueError('Opening attachment differs: '+key)
    output['from_25_seconds_physical_state_bitexact'] = exact
    output['projection_rows_are_not_assumed_identical'] = True
    return output


def metrics(store):
    import numpy as np
    checked = 0
    for name in store.index['datasets']:
        checked += len(store.dataset(name))
    reports = {name: store.json(path) for name, path in store.json('REVIEW_INDEX.json')['reports'].items()}
    packets = {name: store.dataset(name) for name in ('reference', 'legacy', 'full', 'boot', 'opening')}
    obs = store.dataset('observations_fullbody')
    boot = store.dataset('observations_boot')
    opening = store.dataset('observations_opening')
    for key in obs:
        if key != 'sigma_px':
            assert obs[key].tobytes() == boot[key].tobytes(), key
        assert boot[key].tobytes() == opening[key].tobytes(), key
    expected_sigma = obs['sigma_px'].copy()
    expected_sigma[:, 15:17] = 24
    np.testing.assert_array_equal(boot['sigma_px'], expected_sigma)
    assert int(obs['lower_body_inferred'].sum()) == 125
    return dict(arrays_restored_and_sha_verified=checked, source_exposures=1143,
                lower_body_inferred=125, ankle_sigma_change=dict(before=7, after=24, points=2286),
                fullbody=body_metrics(store, reports, packets),
                boot=boot_metrics(store, reports['boot'], packets),
                opening=opening_metrics(store, reports['opening'], packets),
                independent_3d_accuracy_verified=False)


def pixels(store):
    import av
    receipt = store.json('media/source_clip_receipt.json')
    raw = store.read('media/source_s01.mp4')
    packet_rows, frames = [], {}
    with av.open(io.BytesIO(raw)) as c:
        s = c.streams.video[0]
        assert str(s.time_base) == receipt['time_base']
        for p in c.demux(s):
            if p.pts is not None:
                packet_rows.append(dict(pts=p.pts, dts=p.dts, bytes=p.size,
                                        sha256=sha(bytes(p)), keyframe=p.is_keyframe))
    assert packet_rows == receipt['packet_identity']
    with av.open(io.BytesIO(raw)) as c:
        s = c.streams.video[0]
        s.codec_context.thread_count = 2
        for frame in c.decode(s):
            if receipt['target_first_pts'] <= frame.pts <= receipt['target_last_pts_inclusive']:
                frames[str(frame.pts)] = sha(frame.to_ndarray(format='bgr24').tobytes())
    expected = receipt['pixel_hashes_by_pts']
    mismatch = [key for key in expected if frames.get(key) != expected[key]]
    if mismatch or set(frames) != set(expected):
        raise ValueError(json.dumps(dict(packet_identity_matches=True, decoded_pixel_mismatches=len(mismatch),
            first_mismatched_pts=mismatch[:5], pyav_version=av.__version__,
            library_versions=av.library_versions,
            guidance='Byte-identical source packets passed. Strict BGR hashes were verified with PyAV 18.1.0; PyAV 13.0.0 differs. Use the recorded decoder/conversion environment.'), ensure_ascii=False))
    return dict(packets=len(packet_rows), packet_identity_matches=True,
                source_frames=len(frames), all_decoded_original_pixel_hashes_match=True,
                slice_sha256=sha(raw), receipt_sha256=sha(store.read('media/source_clip_receipt.json')),
                pyav_version=av.__version__, library_versions=av.library_versions)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path)
    ap.add_argument('--zip', type=Path)
    ap.add_argument('--metrics', action='store_true')
    ap.add_argument('--pixels', action='store_true')
    ap.add_argument('--restore', type=Path, help='New directory for complete reconstructed NPZ datasets')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    store = Store(args.root, args.zip)
    result = dict(schema='rinkspace.private-body-supplement-audit.v1', integrity=integrity(store))
    if args.metrics:
        result['metrics'] = metrics(store)
    if args.pixels:
        result['pixels'] = pixels(store)
    if args.restore:
        import numpy as np
        args.restore.mkdir(parents=True, exist_ok=False)
        for name in store.index['datasets']:
            if not name.replace('_', '').isalnum():
                raise ValueError('Unsafe dataset name')
            np.savez_compressed(args.restore/(name+'.npz'), **store.dataset(name))
        result['restored_datasets'] = list(store.index['datasets'])
        result['restore_note'] = 'Every array bit preserved; ZIP/NPZ container bytes are newly written.'
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text, encoding='utf8')


if __name__ == '__main__':
    main()
