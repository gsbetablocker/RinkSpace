"""Revalidate cached RGB link pairs after source-shape qualification; never reread RGB."""
from pathlib import Path
from functools import lru_cache
from collections import Counter, defaultdict
import argparse
import json

import numpy as np

import prepare_native_multicamera as prepared
from rinkspace.path_policy import guard_project_write
from rinkspace.session.material_tracks import source_line_reference_support


def dependency_path(value, packet):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (packet.parent/path).resolve()


def revalidate_edge(edge, first, last, qualified, frame):
    """Filter original source_nodes indices, retaining only original unambiguous edges."""
    if edge.get('unambiguous') is not True:
        return None, 'original_edge_not_unambiguous'
    if first not in qualified or last not in qualified:
        return None, 'qualified_fragment_absent'
    a, b = frame(first[:2]), frame(last[:2])
    pairs = np.asarray(edge['pairs'])
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu':
        raise ValueError('Invalid cached source_nodes pair indices')
    if np.any(pairs < 0) or np.any(pairs[:, 0] >= len(a['ids'])) or np.any(pairs[:, 1] >= len(b['ids'])):
        raise ValueError('Cached source_nodes pair outside original node order')
    candidates_a, candidates_b = a['ids'][pairs[:, 0]], b['ids'][pairs[:, 1]]
    if (not np.isin(candidates_a, a['fragments'][first[2]]['candidate_ids']).all()
            or not np.isin(candidates_b, b['fragments'][last[2]]['candidate_ids']).all()):
        raise ValueError('Cached pairs do not belong to their original fragments')
    keep = np.isin(candidates_a, list(qualified[first])) & np.isin(candidates_b, list(qualified[last]))
    pairs = pairs[keep]
    source_ids, target_ids = np.unique(candidates_a[keep]), np.unique(candidates_b[keep])
    if min(len(source_ids), len(target_ids)) < 3:
        return None, 'fewer_than_three_distinct_qualified_candidates'
    span = float(np.linalg.norm(np.ptp(b['uv'][target_ids], axis=0)))
    if span < 56.:
        return None, 'qualified_target_span_below_56px'
    return dict(edge, from_fragment=list(first), to_fragment=list(last), pairs=pairs.tolist(),
                support=len(target_ids), previous_support=len(source_ids),
                qualified_target_span_px=span), None


def seed_support(rows, reference_uv, candidate_query):
    """Match distinct old pixels to finite, continuously queried measured segments."""
    uv = np.array([row['uv'] for row in rows], float).reshape(-1, 2)
    ids = np.array([row['candidate'] for row in rows], int)
    reference = np.empty((0, 2)) if reference_uv is None else reference_uv
    return source_line_reference_support(reference, uv, np.asarray(candidate_query)[ids])


def reviewed_anchor_seeds(document, rows_by_node, anchor_path):
    """Use only explicit source-image/topology reviews supplied by the caller."""
    if not isinstance(document.get('anchors'), list):
        raise ValueError('Source anchors need an explicit anchors list')
    seeds, checks, images = set(), [], {}
    for number, anchor in enumerate(document['anchors']):
        reason = anchor.get('reason')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('Source anchor needs a nonempty source-based reason')
        pts, candidates = anchor.get('pts'), anchor.get('candidate_ids')
        if isinstance(pts, bool) or not isinstance(pts, int):
            raise ValueError('Source anchor needs integer source PTS')
        if (not isinstance(candidates, list) or not candidates
                or any(isinstance(cid, bool) or not isinstance(cid, int) or cid < 0 for cid in candidates)
                or len(set(candidates)) != len(candidates)):
            raise ValueError('Source anchor needs distinct original candidate_ids')
        for name in ('shot', 'identity', 'path_id', 'evidence_image'):
            if not isinstance(anchor.get(name), str) or not anchor[name]:
                raise ValueError('Source anchor is missing '+name)
        image = dependency_path(anchor['evidence_image'], Path(anchor_path))
        if not image.is_file():
            raise ValueError('Source anchor evidence_image does not exist')
        if str(image) not in images:
            images[str(image)] = prepared.digest(image)
        matches = {}
        for node, rows in rows_by_node.items():
            if node[:2] != (anchor['shot'], pts):
                continue
            selected = {cid: row for cid, row in rows.items()
                        if row['identity'] == anchor['identity'] and row['path_id'] == anchor['path_id']}
            if selected:
                matches[node] = selected
        available = {cid for rows in matches.values() for cid in rows}
        missing = sorted(set(candidates)-available)
        if missing:
            raise ValueError('Source anchor candidates not in qualified accepted at same PTS/identity/path: '+str(missing))
        hits = []
        for node, rows in matches.items():
            ids = sorted(set(candidates)&rows.keys())
            uv = np.array([rows[cid]['uv'] for cid in ids], float).reshape(-1, 2)
            span = float(np.linalg.norm(np.ptp(uv, axis=0))) if len(uv) else 0.
            retained = len(ids) >= 3 and np.isfinite(span) and span >= 56.
            if retained:
                seeds.add(node)
            hits.append(dict(fragment=list(node), candidate_ids=ids,
                             qualified_source_uv_span_px=span, retained=bool(retained)))
        checks.append(dict(anchor_index=number, shot=anchor['shot'], pts=pts,
            identity=anchor['identity'], path_id=anchor['path_id'], candidate_ids=candidates,
            reason=reason, evidence_image=str(image), evidence_image_sha256=images[str(image)],
            original_fragment_hits=hits,
            identity_authority='caller_supplied_independent_source_image_and_known_view_topology_review'))
    return seeds, checks


def anchored_nodes(nodes, edges, seeds):
    parent = {node: node for node in nodes}
    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    for edge in edges:
        first, last = tuple(edge['from_fragment']), tuple(edge['to_fragment'])
        parent[find(last)] = find(first)
    roots = {find(seed) for seed in seeds}
    return {node for node in nodes if find(node) in roots}


def revalidate(args):
    output = guard_project_write(args.output)
    if output.exists():
        raise FileExistsError('Preserve existing output: '+str(output))
    if output.is_relative_to(prepared.ROOT.resolve()):
        raise ValueError('Source observation outputs belong outside OneDrive')
    source, linked, qualified_root, seed_root = [Path(p).resolve() for p in
        (args.source, args.linked_source, args.qualified_source, args.seed_root)]
    binding_path = source/'binding.json'
    binding = prepared.read(binding_path)
    linked_binding = prepared.read(linked/'binding.json')
    if linked_binding.get('parent_binding_sha256') != prepared.digest(binding_path):
        raise ValueError('Linked source belongs to another original source binding')
    seed_manifest = prepared.read(seed_root/'joint_manifest.json')
    input_path = seed_root/'point_observations.npz'
    input_hash = prepared.digest(input_path)
    if input_hash != binding['input_sha256'] or seed_manifest['point_input_sha256'] != input_hash:
        raise ValueError('Seed prepared input differs from original capture binding')
    seed = prepared.npz(input_path)
    seed_mapping = prepared.read(seed_root/'mapping.json')
    path_lookup = {path: index for index, path in enumerate(seed_mapping['path_ids'])}
    seed_order = np.lexsort((seed['path_indices'], seed['pts']))
    ordered_keys = np.c_[seed['pts'][seed_order], seed['path_indices'][seed_order]]
    changes = np.r_[0, np.flatnonzero(np.any(np.diff(ordered_keys, axis=0), axis=1))+1, len(seed_order)]
    seed_groups = {(int(ordered_keys[start, 0]), int(ordered_keys[start, 1])): seed_order[start:stop]
                   for start, stop in zip(changes[:-1], changes[1:])}
    qualified, rows_by_node, rows_by_shot = defaultdict(set), defaultdict(dict), {}
    frame_sources, packets, dependencies = {}, {}, {}
    for packet in sorted(qualified_root.glob('*/source_verified.json')):
        data = prepared.read(packet)
        sid = packet.parent.name
        if data.get('shot') != sid or data.get('fit_eligible') is not True:
            raise ValueError('Qualified packet needs its explicit shot and fit_eligible')
        evidence = {}
        for name in ('source_packet', 'source_links'):
            path = dependency_path(data[name], packet)
            digest = prepared.digest(path)
            if digest != data[name+'_sha256']:
                raise ValueError('Qualified source dependency differs: '+str(path))
            evidence[name] = dict(path=str(path), sha256=digest)
        packets[sid] = dict(path=str(packet), sha256=prepared.digest(packet), dependencies=evidence)
        rows_by_shot[sid] = data['accepted']
        for row in data['accepted']:
            key = (sid, int(row['pts']), int(row['fragment']))
            cid = int(row['candidate'])
            raw = dependency_path(row['source_reference'], packet)
            frame_key = key[:2]
            if frame_key in frame_sources and frame_sources[frame_key] != raw:
                raise ValueError('One original frame has multiple candidate/node inventories')
            frame_sources[frame_key] = raw
            if cid in rows_by_node[key] and rows_by_node[key][cid] != row:
                raise ValueError('Conflicting qualified records for one original candidate')
            qualified[key].add(cid)
            rows_by_node[key][cid] = row
    if not packets:
        raise ValueError('No qualified source_verified packets')

    @lru_cache(maxsize=24)
    def frame(key):
        raw = frame_sources[key]
        parts = prepared.read(raw.with_suffix('.json'))['fragments']
        ids = np.unique([cid for part in parts for cid in part['candidate_ids']]).astype(int)
        with np.load(raw, allow_pickle=False) as measured:
            uv = measured['candidate_uv']
            query = measured['candidate_query']
        for node, rows in rows_by_node.items():
            if node[:2] != key:
                continue
            members = set(parts[node[2]]['candidate_ids'])
            for cid, row in rows.items():
                if cid not in members or not np.array_equal(uv[cid], row['uv']):
                    raise ValueError('Qualified candidate differs from its original fragment/UV')
        return dict(ids=ids, uv=uv, query=query, fragments=parts)

    old_seed_nodes, cached = [], []
    for sid in packets:
        path = source/sid/'source_links.json'
        data = prepared.read(path)
        dependencies[str(path)] = prepared.digest(path)
        old_seed_nodes.extend((sid, int(pt), int(fragment)) for pt, fragment in data['seed_fragments'])
        cached.extend((edge, (sid, int(edge['from_pts']), int(edge['a'])),
                       (sid, int(edge['to_pts']), int(edge['b'])), str(path), index)
                      for index, edge in enumerate(data['edges']))
    cross_path = linked/'connections.jsonl'
    dependencies[str(cross_path)] = prepared.digest(cross_path)
    with cross_path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            edge = json.loads(line)
            cached.append((edge, tuple(edge['from_fragment']), tuple(edge['to_fragment']), str(cross_path), line_number))
    kept, rejected = [], Counter()
    for edge, first, last, path, index in cached:
        revised, reason = revalidate_edge(edge, first, last, qualified, frame)
        if revised is None:
            rejected[reason] += 1
        else:
            kept.append(dict(revised, cached_edge_origin=dict(path=path, index=index)))
    seed_checks, seeds = [], set()
    historical_seeds = set(old_seed_nodes)
    references = {}
    # Any qualified source fragment can establish same-line overlap. The old
    # point-sampling phase is not an eligibility gate or the universe of seeds.
    for node in sorted(rows_by_node):
        rows = list(rows_by_node[node].values())
        current = frame(node[:2])
        paths = {row['path_id'] for row in rows}
        if len(paths) != 1:
            raise ValueError('One original fragment names multiple physical paths')
        path = paths.pop()
        key = (node[1], path_lookup.get(path, -1))
        if key not in references:
            old = seed_groups.get(key)
            references[key] = seed['uv'][old] if old is not None else None
        support = seed_support(rows, references[key], current['query'])
        if support['qualified']:
            seeds.add(node)
        seed_checks.append(dict(fragment=list(node), historical_seed=node in historical_seeds,
            distinct_old_uv_on_qualified_source_segments=support['count'],
            matched_old_uv_span_px=support['span_px'], retained=support['qualified']))
    line_seeds = set(seeds)
    anchor_path = getattr(args, 'source_anchors', None)
    anchor_reference = None
    if anchor_path is not None:
        anchor_path = Path(anchor_path).resolve()
        anchor_hash = prepared.digest(anchor_path)
        reviewed_seeds, anchor_checks = reviewed_anchor_seeds(
            prepared.read(anchor_path), rows_by_node, anchor_path)
        for node in reviewed_seeds:
            frame(node[:2])
        seeds.update(reviewed_seeds)
        anchor_reference = dict(path=str(anchor_path), sha256=anchor_hash,
            anchors=anchor_checks, reviewed_seed_fragments=len(reviewed_seeds),
            newly_seeded_fragments=len(reviewed_seeds-line_seeds),
            source_anchors_generated_by_adapter=False, evidence_images_decoded=False)
    connected = anchored_nodes(qualified, kept, seeds)
    # Authenticate all pending rows too; they retain evidence but gain no identity admission.
    for key in frame_sources:
        frame(key)
    output.mkdir(parents=True)
    edge_path = output/'connections.jsonl'
    with edge_path.open('w', encoding='utf-8') as stream:
        for edge in kept:
            stream.write(json.dumps(edge, ensure_ascii=False)+'\n')
    counts = {}
    for sid, rows in rows_by_shot.items():
        accepted = [row for row in rows if (sid, int(row['pts']), int(row['fragment'])) in connected]
        pending = [row for row in rows if (sid, int(row['pts']), int(row['fragment'])) not in connected]
        folder = output/sid
        folder.mkdir()
        prepared.write(folder/'source_pixels.json', dict(shot=sid, accepted=accepted, pending=pending,
            source_UV_changed=False, identity_status='existing_source_linked_hypothesis', user_visual_acceptance=False))
        counts[sid] = dict(qualified_rows=len(rows), accepted_rows=len(accepted), pending_rows=len(pending),
            retained_seed_fragments=sum(node[0] == sid for node in seeds),
            requalified_historical_seed_fragments=sum(node[0] == sid for node in line_seeds & historical_seeds),
            newly_discovered_source_line_seed_fragments=sum(node[0] == sid for node in line_seeds-historical_seeds))
    receipt_path = output/'graph_receipt.json'
    receipt = dict(schema='rinkspace.revalidated_centerline_identity_graph.v1',
        original_source=str(source), source_binding_sha256=prepared.digest(binding_path),
        linked_source=str(linked), linked_binding_sha256=prepared.digest(linked/'binding.json'),
        qualified_source=str(qualified_root), qualified_packets=packets, cached_edge_sources_sha256=dependencies,
        seed_root=str(seed_root), seed_input_sha256=input_hash,
        old_edges=len(cached), retained_edges=len(kept), rejected_edges_by_reason=dict(rejected),
        old_seed_fragments=len(set(old_seed_nodes)), retained_seed_fragments=len(seeds), seed_checks=seed_checks,
        retained_old_seed_fragments=len(line_seeds & historical_seeds),
        newly_discovered_source_line_seed_fragments=len(line_seeds-historical_seeds),
        source_line_seed_fragments=len(line_seeds),
        historical_seed_fragments_absent_from_qualified=len(historical_seeds-set(qualified)),
        source_anchor_review=anchor_reference,
        qualified_fragment_nodes=len(qualified), anchored_fragment_nodes=len(connected), shots=counts,
        connections=str(edge_path), connections_sha256=prepared.digest(edge_path),
        original_ambiguous_edges_never_promoted=True, required_distinct_candidates_per_end=3,
        required_target_bbox_span_norm_px=56., seed_matching_radius_px=3.,
        seed_matching_rule='Distinct old UV within 3px of finite qualified source segments joining consecutive queries only',
        required_distinct_old_seed_uv=3, required_matched_old_seed_uv_span_px=56.,
        cached_pairs_index_space='Original source_nodes: np.unique(all original fragment candidate_ids)',
        source_RGB_reread=False, source_UV_changed=False, model_projection_used_for_pixels=False,
        fit_executed=False, identity_status='existing_source_linked_hypothesis', user_visual_acceptance=False)
    prepared.write(receipt_path, receipt)
    prepared.write(output/'binding.json', dict(binding, graph_revalidation_receipt=str(receipt_path),
        graph_revalidation_receipt_sha256=prepared.digest(receipt_path)))
    for sid in packets:
        rows = rows_by_shot[sid]
        accepted = [row for row in rows if (sid, int(row['pts']), int(row['fragment'])) in connected]
        prepared.write(output/sid/'source_verified.json', dict(shot=sid, accepted=accepted,
            source_packet=packets[sid]['path'], source_packet_sha256=packets[sid]['sha256'],
            source_links=str(receipt_path), source_links_sha256=prepared.digest(receipt_path),
            fit_eligible=True, source_UV_synthesized=False,
            identity_status='existing_source_linked_hypothesis', user_visual_acceptance=False))
    print(json.dumps(dict(output=str(output), retained_edges=len(kept), rejected_edges=dict(rejected),
                         retained_seed_fragments=len(seeds), shots=counts)), flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('qualified-source', 'linked-source', 'source', 'seed-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-anchors', type=Path,
        help='Explicit reviewed source-image/known-view-topology anchors; never generated from model residuals')
    revalidate(parser.parse_args())
