"""Authenticate finite source-supported board search charts for unknown points.

The captured chart is an initialization domain on the existing continuous board.
It is neither a measured world rectangle nor a replacement for the old glyphs.
"""
import copy
from collections import defaultdict
from .io_utils import read_json, read_jsonl

SCHEMA = 'rinkfit.source_supported_body_chart_binding_contract.v1'


def _source_rows(path):
    if str(path).endswith(('.jsonl', '.jsonl.gz')):
        return list(read_jsonl(path))
    rows = read_json(path)
    return rows['rows'] if isinstance(rows, dict) else rows


def load_body_chart_contract(reference, previous, current, verify):
    import numpy as np
    from .joint_model import graph_from_joint, as_event_view
    from .camera_graph import CameraGraph
    from .static_board_tracklet_append import _ray_surface_seed
    from rinkfit_start.native_scene import NativeScene

    contract = read_json(verify(reference))
    if contract.get('schema') != SCHEMA:
        raise ValueError('Explicit captured body-chart binding contract required')
    if read_json(verify(contract['previous_model'])) != previous:
        raise ValueError('Body-chart contract uses a different physical parent')
    cg = CameraGraph(graph_from_joint(previous))
    scene = NativeScene(as_event_view(previous, 's01'))
    names = {b['name']: i for i, b in enumerate(current['blocks'])}
    old_names = {b['name']: i for i, b in enumerate(previous['blocks'])}
    labels = {m['label']: i for i, m in enumerate(current['materials'])}
    old_labels = {m['label']: i for i, m in enumerate(previous['materials'])}
    patches = {}; row_ids = set(); allowed_blocks = set(); allowed_materials = set()
    for entry in contract['patches']:
        patch = read_json(verify(entry['patch']))
        if read_json(verify(patch['base_model'])) != previous:
            raise ValueError('Captured body patch is not relative to this immutable parent')
        domain_ref = patch['domain']; domain = read_json(verify(domain_ref))
        if domain.get('schema') != 'rinkfit.source_supported_continuous_board_texture_chart_domain.v1':
            raise ValueError('Captured body domain has no authenticated continuous-board definition')
        parent_name = domain['existing_advertisement_block_name']
        adid = names[parent_name]; oldid = old_names[parent_name]
        if (previous['blocks'][oldid] != current['blocks'][adid]
            or domain['existing_advertisement_block'] != oldid
            or read_json(verify(domain['base_model'])) != previous
            or domain.get('physical_accuracy_accepted') is not False
            or domain.get('source_uv_changed') is not False
            or domain.get('unknown_world_outline_rows_activated') is not False):
            raise ValueError('Body domain changed its parent or promoted source rays to measured world truth')
        for key in ('ray_producer', 'finite_board_ray_reader', 'old_composite_world_QA'):
            if key in domain: verify(domain[key])
        relation = domain.get('captured_composite_relation', {})
        if relation:
            if relation['raw_parent_values'] != previous['blocks'][oldid]['values']:
                raise ValueError('Captured composite relation names a different raw parent')
            if 'original_capture_overlay' in relation: verify(relation['original_capture_overlay'])
        review = read_json(verify(domain['source_identity_review']))
        if 'source_video' in review:
            verify(dict(file=review['source_video'].get('file', review['source_video'].get('path')), sha256=review['source_video']['sha256']))
        for frame in review.get('reviewed_frames', []):
            for key in ('native_PNG', 'original_source_feature'):
                if key in frame:
                    verify(frame[key])
        source = _source_rows(verify(domain['source_outline_rows']))
        source_by_id = defaultdict(list)
        for r in source:
            source_by_id[(r['source_row_id'], int(r['native_pts']))].append(r)
        diagnostic = read_json(verify(domain['actual_observed_row_finite_board_ray_diagnostic']))
        observed = diagnostic['rows']; seen = set(); coords = []
        ad = np.asarray(previous['blocks'][oldid]['values'], float)
        search = copy.deepcopy(previous)
        search['blocks'][oldid]['values'] = [0., scene.perimeter, 0., scene.height]
        for row in observed:
            key = (row['source_row_id'], int(row['native_pts']))
            candidates = source_by_id.get(key, [])
            if key in seen or not candidates or not any(r.get('current_uv', r['source_uv']) == row['source_uv'] for r in candidates):
                raise ValueError('Body envelope redirects or duplicates a real source outline UV')
            seen.add(key)
            sid = row.get('shot_id', domain.get('shot_id', entry.get('shot_id', 's02')))
            camera = cg.native_camera(sid)
            t = float(cg.local_seconds(sid, np.asarray([row['native_pts']], np.int64))[0])
            q, hits = _ray_surface_seed(search, scene, camera, t, row['source_uv'], oldid)
            sh = np.asarray(q)*[scene.perimeter, scene.height]
            uv = (sh-ad[[0, 2]])/[ad[1], ad[3]-ad[2]]
            if (not np.isfinite(uv).all()
                or np.max(np.abs(uv-np.asarray(row['local_ad_uv']))) > 1e-9
                or np.max(np.abs(sh-np.asarray(row['surface_s_h']))) > 1e-10):
                raise ValueError('Body envelope is not the current observed-row finite inward ray intersection')
            coords.append(uv)
        if source and all('source_admitted' in r for r in source):
            complete = {(r['source_row_id'], int(r['native_pts'])) for r in source if r['source_admitted']}
        elif source and all('admission' in r for r in source):
            complete = {(r['source_row_id'], int(r['native_pts'])) for r in source if r['admission'] == 1}
        else:
            complete = {(r['source_row_id'], int(r['native_pts'])) for r in source}
        if seen != complete:
            raise ValueError('Body envelope omitted eligible older source rows or used a rejected row')
        coords = np.asarray(coords)
        bounds = np.asarray(domain['material_uv_bounds'], float)
        actual = np.asarray([coords.min(axis=0), coords.max(axis=0)])
        if (bounds.shape != (2, 2) or not np.all(bounds[0] < bounds[1])
            or np.max(np.abs(actual-bounds)) > 1e-9
            or np.max(np.abs(actual-np.asarray(diagnostic['bounds']))) > 1e-9):
            raise ValueError('Body chart bounds add unsupported padding or omit source ray extrema')
        ids = patch['replace_only_these_source_row_ids']
        if len(ids) != len(set(ids)) or row_ids.intersection(ids):
            raise ValueError('Body-chart patch duplicates source identities')
        row_ids.update(ids)
        point_rows = list(read_jsonl(verify(patch['source_rows'])))
        if set(ids) != {r['source_row_id'] for r in point_rows}:
            raise ValueError('Body-chart patch does not retain its whole source cohort')
        by_track = defaultdict(list)
        for r in point_rows:
            by_track[r['physical_track_id']].append(r)
        if 'numeric_initialization' in patch:
            verify(patch['numeric_initialization'])
        for item in patch['tracks']:
            track = item['track_id']; block = item['geometry_block']; name = block['name']
            bi = names[name]; obi = old_names[name]
            mi = labels[item['target_id']]; omi = old_labels[item['target_id']]
            if (track in patches or item['expected_geometry_block'] != previous['blocks'][obi]
                or not by_track[track] or current['blocks'][bi] != block):
                raise ValueError('Captured body patch changed a different track or its numerical seed')
            if (block['constant'] or block.get('fixed_indices') != [0, 1, 3, 4]
                or [block['values'][j] for j in (0, 1, 3, 4)] != [1., 0., 0., 1.]):
                raise ValueError('Body track must keep precisely two unknown material coordinates')
            lo = np.asarray(block['lower'])[[2, 5]]; hi = np.asarray(block['upper'])[[2, 5]]
            q = np.asarray(block['values'])[[2, 5]]
            if not (np.array_equal(lo, bounds[0]) and np.array_equal(hi, bounds[1])
                    and np.all(q >= lo) and np.all(q <= hi)):
                raise ValueError('Unknown point bounds do not consume the authenticated body domain')
            expected = copy.deepcopy(item['material'])
            expected['paths'][0]['geometry_block'] = bi
            expected['paths'][0]['advertisement_block'] = adid
            old_mat = copy.deepcopy(previous['materials'][omi])
            old_mat['paths'][0]['source_chart_domain_ref'] = domain_ref
            if (current['materials'][mi] != expected or expected != old_mat
                or expected['paths'][0].get('source_chart_domain_ref') != domain_ref):
                raise ValueError('Body patch redirects the original point operator or source chart')
            binding = item['binding']
            if (binding['source_chart_domain_ref'] != domain_ref
                or binding['material_uv_bounds'] != domain['material_uv_bounds']
                or binding['board_instance_id'] != domain['physical_installation']):
                raise ValueError('Point binding does not read the same physical body-chart contract')
            patches[track] = dict(item=item, domain_ref=domain_ref, source_rows=by_track[track])
            allowed_blocks.add(bi); allowed_materials.add(mi)
    if (len(previous['blocks']) != len(current['blocks'])
        or len(previous['materials']) != len(current['materials'])
        or any(a != b for i, (a, b) in enumerate(zip(previous['blocks'], current['blocks'])) if i not in allowed_blocks)
        or any(a != b for i, (a, b) in enumerate(zip(previous['materials'], current['materials'])) if i not in allowed_materials)
        or any(previous.get(k) != current.get(k) for k in ('rigs', 'shots', 'scene'))):
        raise ValueError('Captured body binding changed a shared camera, scene, old glyph, or unrelated point')
    return patches, row_ids
