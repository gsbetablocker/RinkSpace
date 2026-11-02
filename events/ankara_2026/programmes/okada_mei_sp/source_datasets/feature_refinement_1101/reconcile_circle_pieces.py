"""Partition stored circle readouts into visible arcs and rebuild their source graph.

No image/video decoding, new measurement, geometry fitting or old-seed inheritance.
"""
from collections import Counter, defaultdict
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

PROJECT = Path('D:/OneDrive/codex_projs/RinkSpace')
sys.path.insert(0, str(PROJECT/'src'))
from rinkspace.session.material_tracks import split_visible_source_arcs, qualify_fragment_links
BASE = Path('D:/CodexData/RinkSpace_Data/runs/native_programme_20261031/feature_refinement_1101')
SOURCE = BASE/'a_circles'
OUTPUT = BASE/'circles_separated'
BATCHES = {'wide_full_1':['s01','s02','s03'], 'wide_full_2':['s04','s05','s06'], 'wide_full_3':['s07','s08','s09']}
HASHES = {}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in HASHES:
        with path.open('rb') as stream:
            HASHES[key] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return HASHES[key]


def ref(path):
    return dict(path=str(Path(path).resolve(strict=True)), sha256=sha(path))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def span(uv):
    uv = np.asarray(uv, float).reshape(-1, 2)
    return float(np.linalg.norm(np.ptp(uv, axis=0))) if len(uv) else 0.


def main(dry_run=False):
    if OUTPUT.exists():
        raise FileExistsError('Preserve existing graph output: '+str(OUTPUT))
    inventory = read(SOURCE/'all9_inventory.json')
    visual_path = BASE/'a_circle_visual_review.json'
    withheld_path = BASE/'circle_reviews.json'
    a4_path = BASE/'a4_partial_arc_source_review.json'
    visual, reviews, a4 = map(read, (visual_path, withheld_path, a4_path))
    withheld = {(r['shot'], int(r['pts']), r['path_id'], *map(float, r['uv'])): r for r in reviews['withheld_source_uv']}
    assert len(withheld) == 3
    visual_anchors = {(r['representative_shot'], int(r['representative_pts']), r['claimed_identity']): r
        for r in visual['reviews'] if r['result'] == 'support'}
    assert set(visual_anchors) == {('s02',628992,'A1:centreline'), ('s07',1573632,'A1:centreline')}
    assert a4['shot'] == 's07' and a4['pts'] == 1441024 and a4['candidate_ids'] == [3,4,5,7]
    assert a4['path_id'] == 'demo_initial/A4:outer'
    function_path = Path(split_visible_source_arcs.__code__.co_filename).resolve(strict=True)
    all_counts = Counter()
    shot_receipts = {}
    anchor_checks = []
    if not dry_run:
        OUTPUT.mkdir()
    for batch, sids in BATCHES.items():
        binding = read(SOURCE/batch/'binding.json')
        assert binding['input_sha256'] == inventory['input_sha256']
        identities = [p['curve_id'] for p in binding['inventory']]
        out_batch = OUTPUT/batch
        if not dry_run:
            out_batch.mkdir()
            new_binding = copy.deepcopy(binding)
            new_binding.update(reused_actual_source=str((SOURCE/batch).resolve()),
                reused_capture_binding_sha256=sha(SOURCE/batch/'binding.json'),
                source_graph_reconciliation=dict(parent_binding=ref(SOURCE/batch/'binding.json'),
                    implementation=ref(function_path), builder=ref(Path(__file__)),
                    source_identity_anchors=[ref(visual_path),ref(a4_path)], withheld_source_UV=ref(withheld_path),
                    original_seeds_inherited=False, raw_readouts_reused_without_change=True,
                    no_new_readouts=True, source_images_opened=False, prepare_executed=False, fit_executed=False))
            write(out_batch/'binding.json', new_binding)
        for sid in sids:
            original_folder = SOURCE/batch/sid
            folder = out_batch/sid
            pixels_path, links_path = original_folder/'source_pixels.json', original_folder/'source_links.json'
            assert sha(pixels_path) == inventory['shots'][sid]['source_pixels']['sha256']
            assert sha(links_path) == inventory['shots'][sid]['source_links']['sha256']
            pixels, old_links = read(pixels_path), read(links_path)
            old_rows = {}
            for collection in ('accepted','pending'):
                for row in pixels[collection]:
                    key = (int(row['pts']),int(row['fragment']),int(row['candidate']))
                    assert key not in old_rows
                    old_rows[key] = row
            frame_nodes, nodes, candidate_to_new, candidate_to_old = {}, {}, {}, {}
            fragment_map, counts = [], Counter()
            if not dry_run:
                folder.mkdir()
            frame_files = sorted((p for p in original_folder.glob('*.json') if p.stem.isdigit()), key=lambda p:int(p.stem))
            expected = list(range(inventory['shots'][sid]['first_pts'], inventory['shots'][sid]['last_pts']+1,256))
            assert [int(p.stem) for p in frame_files] == expected
            for metadata_path in frame_files:
                pt = int(metadata_path.stem)
                frame = read(metadata_path)
                with np.load(original_folder/f'{pt}.npz', allow_pickle=False) as z:
                    measurement = {key:z[key] for key in ('candidate_query','query_uv','candidate_uv')}
                original_ids = np.unique([i for f in frame['fragments'] for i in f['candidate_ids']]).astype(int)
                original_frag_by_id, new_frag_by_id, new_fragments, local_map = {}, {}, [], []
                for old_index, fragment in enumerate(frame['fragments']):
                    pieces = split_visible_source_arcs(measurement, fragment)
                    assert sorted(i for p in pieces for i in p['candidate_ids']) == sorted(fragment['candidate_ids'])
                    counts['original_fragment_nodes'] += 1
                    counts['split_original_fragments'] += len(pieces)>1
                    new_indices = []
                    for piece in pieces:
                        index = len(new_fragments)
                        piece = dict(piece, original_fragment=old_index,
                            source_fragment_partition='split_visible_source_arcs: unchanged original candidate membership only')
                        new_fragments.append(piece)
                        new_indices.append(index)
                        new_rows = []
                        for cid in piece['candidate_ids']:
                            assert cid not in original_frag_by_id
                            original_frag_by_id[cid] = old_index
                            new_frag_by_id[cid] = index
                            old = old_rows[(pt,old_index,cid)]
                            assert old['identity'] == identities[fragment['curve']]
                            assert np.array_equal(np.asarray(old['uv']), measurement['candidate_uv'][cid])
                            new = dict(old, fragment=index, original_fragment=old_index)
                            assert {k:v for k,v in new.items() if k not in ('fragment','original_fragment')} == {k:v for k,v in old.items() if k!='fragment'}
                            new_rows.append(new)
                        nodes[(pt,index)] = new_rows
                    item = dict(pts=pt, original_fragment=old_index, new_fragments=new_indices,
                        original_candidate_ids=list(fragment['candidate_ids']),
                        new_candidate_ids=[list(new_fragments[j]['candidate_ids']) for j in new_indices],
                        original_frame_metadata=ref(metadata_path))
                    fragment_map.append(item)
                    local_map.append(item)
                new_ids = np.unique([i for f in new_fragments for i in f['candidate_ids']]).astype(int)
                assert np.array_equal(new_ids, original_ids)
                if len(original_ids):
                    frame_nodes[pt] = dict(ids=original_ids, uv=measurement['candidate_uv'][original_ids].copy(), fragments=new_fragments)
                    candidate_to_new[pt], candidate_to_old[pt] = new_frag_by_id, original_frag_by_id
                if not dry_run:
                    raw = original_folder/f'{pt}.npz'
                    os.link(raw, folder/raw.name)
                    assert raw.samefile(folder/raw.name)
                    new_frame = dict(frame, fragments=new_fragments, parent_frame_metadata=ref(metadata_path),
                        old_to_new_fragment_mapping=local_map, source_UV_modified=False, original_candidates_preserved=True)
                    write(folder/metadata_path.name, new_frame)
                counts['native_source_frames'] += 1
                counts['NPZ_hard_links'] += not dry_run
            assert len(old_rows) == sum(map(len,nodes.values()))
            grouped_edges, removed_edges = defaultdict(list), []
            old_link_ref = ref(links_path)
            for edge_index, edge in enumerate(old_links['edges']):
                counts['original_edges'] += 1
                previous, current = int(edge['from_pts']), int(edge['to_pts'])
                a_frame, b_frame = frame_nodes[previous], frame_nodes[current]
                pair_groups = defaultdict(list)
                for old_node, now_node in edge['pairs']:
                    ca, cb = int(a_frame['ids'][old_node]), int(b_frame['ids'][now_node])
                    assert candidate_to_old[previous][ca] == edge['a']
                    assert candidate_to_old[current][cb] == edge['b']
                    pair_groups[(candidate_to_new[previous][ca], candidate_to_new[current][cb])].append([old_node,now_node])
                assert sum(map(len,pair_groups.values())) == len(edge['pairs'])
                counts['original_edge_pairs'] += len(edge['pairs'])
                counts['original_edges_split_by_membership'] += len(pair_groups)>1
                for (a,b), pairs in sorted(pair_groups.items()):
                    src = sorted({x for x,_ in pairs}); dst = sorted({y for _,y in pairs})
                    actual_span = span(b_frame['uv'][dst])
                    record = dict(from_pts=previous, to_pts=current, a=a, b=b,
                        crosses_missing_readout=edge.get('crosses_missing_readout',False),
                        support=len(dst), previous_support=len(src), pairs=pairs,
                        current_actual_UV_span_px=actual_span, source_only=True,
                        identity_assumed_from_curve_name=False,
                        original_edge_reference=dict(source=old_link_ref,json_pointer='/edges/'+str(edge_index),index=edge_index),
                        original_edge_unambiguous=edge.get('unambiguous') is True,
                        original_RGB_eligibility_preserved=True, RGB_recomputed=False,
                        original_RGB_statistics={k:v for k,v in edge.items() if k.endswith('_median_px') or k=='correlation_median'},
                        statistics_scope='RGB medians belong only to the referenced unsplit original edge; no subedge medians are asserted')
                    counts['regrouped_edge_pairs'] += len(pairs)
                    counts['membership_subedges'] += 1
                    reason = 'fewer_than_three_distinct_nodes' if min(len(src),len(dst))<3 else 'current_actual_UV_span_below_56' if actual_span<56. else None
                    if reason:
                        record['removed_reason'] = reason
                        removed_edges.append(record)
                        counts['removed_subedges_'+reason] += 1
                    else:
                        grouped_edges[(previous,current)].append(record)
            edges = []
            for frame_pair, records in sorted(grouped_edges.items()):
                qualify_fragment_links(records)
                for edge in records:
                    qualified_now = edge['unambiguous']
                    edge['subedge_competition_unambiguous'] = qualified_now
                    same_identity = nodes[(edge['from_pts'],edge['a'])][0]['identity'] == nodes[(edge['to_pts'],edge['b'])][0]['identity']
                    edge['same_recorded_identity'] = same_identity
                    edge['unambiguous'] = qualified_now and edge['original_edge_unambiguous'] and same_identity
                    counts['qualified_subedges'] += 1
                    counts['unambiguous_subedges'] += edge['unambiguous']
                    counts['originally_ambiguous_subedges_withheld'] += not edge['original_edge_unambiguous']
                    counts['subedge_competition_ambiguous'] += not qualified_now
                    edges.append(edge)
            assert counts['original_edge_pairs'] == counts['regrouped_edge_pairs']
            parent = {n:n for n in nodes}
            def find(node):
                while parent[node] != node:
                    parent[node] = parent[parent[node]]
                    node = parent[node]
                return node
            for edge in edges:
                if edge['unambiguous']:
                    parent[find((edge['from_pts'],edge['a']))] = find((edge['to_pts'],edge['b']))
            anchors = []
            for (as_shot,pt,identity), review in visual_anchors.items():
                if as_shot != sid:
                    continue
                assert review['source_BGR_sha256_match'] is True
                rows = [r for n,rs in nodes.items() if n[0]==pt for r in rs if r['identity']==identity]
                assert rows and all(r['source_BGR_sha256']==review['source_BGR_sha256_expected']==review['source_BGR_sha256_computed'] for r in rows)
                approved = {r['candidate'] for r in rows if (sid,pt,r['path_id'],*map(float,r['uv'])) not in withheld}
                uncertain = {r['candidate'] for r in rows if (sid,pt,r['path_id'],*map(float,r['uv'])) in withheld}
                anchors.append(dict(pts=pt, identity=identity, support_candidates=approved, allowed_withheld_candidates=uncertain,
                    evidence=ref(visual_path), component_id=review['component_id']))
            if sid == 's07':
                pt = a4['pts']
                rows = [r for n,rs in nodes.items() if n[0]==pt for r in rs if r['path_id']==a4['path_id']]
                bycid = {r['candidate']:r for r in rows}
                assert [bycid[c]['uv'] for c in a4['candidate_ids']] == a4['uv']
                assert all(bycid[c]['source_BGR_sha256']==a4['source_BGR_sha256'] for c in a4['candidate_ids'])
                anchors.append(dict(pts=pt,identity='A4:centreline',support_candidates=set(a4['candidate_ids']),allowed_withheld_candidates=set(),evidence=ref(a4_path)))
            seeds = set()
            local_anchor_checks = []
            for anchor in anchors:
                for node, rows in sorted(nodes.items()):
                    if node[0]!=anchor['pts'] or rows[0]['identity']!=anchor['identity']:
                        continue
                    ids = {r['candidate'] for r in rows}
                    support = [r for r in rows if r['candidate'] in anchor['support_candidates']]
                    unknown = ids-anchor['support_candidates']-anchor['allowed_withheld_candidates']
                    actual_span = span([r['uv'] for r in support])
                    seed = not unknown and len(support)>=3 and actual_span>=56.
                    if seed:
                        seeds.add(node)
                    report = dict(shot=sid,pts=node[0],fragment=node[1],identity=anchor['identity'],
                        all_candidate_ids=sorted(ids),support_candidate_ids=[r['candidate']for r in support],
                        support_UV=[r['uv']for r in support],support_actual_UV_span_px=actual_span,
                        allowed_withheld_candidate_ids=sorted(ids & anchor['allowed_withheld_candidates']),
                        unsupported_candidate_ids=sorted(unknown),is_explicit_seed=seed,evidence=anchor['evidence'],
                        rule='At least three explicitly supported candidates spanning 56px; other members may only be the three explicitly withheld uncertain UVs, which do not support the seed and never qualify')
                    local_anchor_checks.append(report)
                    anchor_checks.append(report)
            roots = {find(n) for n in seeds}
            accepted, pending = [], []
            for node, rows in sorted(nodes.items()):
                (accepted if find(node) in roots else pending).extend(rows)
            rebuilt = {(r['pts'],r['original_fragment'],r['candidate']):r for r in accepted+pending}
            assert rebuilt.keys() == old_rows.keys()
            for key, row in rebuilt.items():
                old = old_rows[key]
                assert row['original_fragment']==old['fragment']
                assert {k:v for k,v in row.items() if k not in ('fragment','original_fragment')} == {k:v for k,v in old.items() if k!='fragment'}
            a4_check = None
            if sid == 's07':
                pt = a4['pts'];good = {candidate_to_new[pt][c] for c in [3,4,5,7]};bad = {candidate_to_new[pt][c] for c in [25,27,29,31,32]}
                assert good.isdisjoint(bad)
                good_roots, bad_roots = {find((pt,f))for f in good}, {find((pt,f))for f in bad}
                a4_check = dict(shot=sid,pts=pt,good_candidate_ids=[3,4,5,7],bad_candidate_ids=[25,27,29,31,32],
                    good_fragments=sorted(good),bad_fragments=sorted(bad),different_fragments=True,
                    graph_components_separated=good_roots.isdisjoint(bad_roots),
                    good_fragments_explicit_seed=all((pt,f)in seeds for f in good),
                    bad_fragment_seed_connected=any(find((pt,f))in roots for f in bad))
                assert a4_check['graph_components_separated'] and a4_check['good_fragments_explicit_seed'] and not a4_check['bad_fragment_seed_connected'], a4_check
            counts.update(new_fragment_nodes=len(nodes),all_source_rows=len(rebuilt),accepted_rows=len(accepted),pending_rows=len(pending),
                explicit_seed_fragments=len(seeds),original_seed_fragments_ignored=len(old_links['seed_fragments']))
            if not dry_run:
                partition_path = folder/'fragment_partition.json'
                write(partition_path, dict(shot=sid,source_pixels=ref(pixels_path),source_links=old_link_ref,
                    implementation=ref(function_path),mapping=fragment_map,candidate_node_order='np.unique(all original fragment candidate_ids); unchanged after split',
                    all_candidate_ids_preserved=True,UV_modified=False))
                write(folder/'source_pixels.json',dict(shot=sid,accepted=accepted,pending=pending,source_UV_changed=False,
                    source_graph_reconciliation=dict(parent_source_pixels=ref(pixels_path),fragment_partition=ref(partition_path),
                        old_seed_inheritance=False,explicit_source_review_seeds_only=True,
                        original_fragment_field_preserved=True,all_other_original_row_fields_exactly_preserved=True)))
                write(folder/'source_links.json',dict(edges=edges,seed_fragments=[list(n)for n in sorted(seeds)],
                    source_identity_is_assistant_inference=True,
                    seed_matching_rule='Only explicit source-reviewed anchors, qualified using supported exact original candidates; no inherited old seeds',
                    required_distinct_old_seed_uv=3,required_matched_old_seed_uv_span_px=56.,
                    parent_source_links=old_link_ref,fragment_partition=ref(partition_path),graph_nodes=[list(n)for n in sorted(nodes)],
                    original_seed_fragments=old_links['seed_fragments'],retained_original_seed_fragments=[],
                    original_seeds_ignored_reason='This A-circle source family has known identity confusion',
                    explicit_seed_evidence=local_anchor_checks,removed_subedges=removed_edges,
                    pair_indices_are_unchanged_global_source_node_indices=True,
                    only_original_unambiguous_edges_may_propagate=True,new_subedge_competition_rechecked=True))
                coverage = {p['curve_id']:dict(read_frames=0,source_rows=0,seed_frames=0)for p in binding['inventory']}
                for node, rows in nodes.items():
                    row = coverage[rows[0]['identity']]
                    row['read_frames'] += 1;row['source_rows'] += len(rows);row['seed_frames'] += node in seeds
                write(folder/'coverage.json',dict(by_path=coverage,accepted_rows=len(accepted),pending_rows=len(pending),
                    identity_propagation_graph_nodes=len(nodes),source_record_fragment_nodes=len(nodes),explicit_seed_fragments=len(seeds)))
                write(folder/'source_verified.json',dict(shot=sid,accepted=accepted,source_packet=str(folder/'source_pixels.json'),
                    source_packet_sha256=sha(folder/'source_pixels.json'),source_links=str(folder/'source_links.json'),source_links_sha256=sha(folder/'source_links.json'),
                    fit_eligible=False,review_state='explicit_identity_seed_source_graph_not_pixel_accuracy_acceptance',
                    user_visual_acceptance=False,source_UV_changed=False))
            shot_receipts[sid] = dict(batch=batch,counts=dict(counts),original_source_pixels=ref(pixels_path),original_source_links=old_link_ref,
                seed_fragments=[list(n)for n in sorted(seeds)],anchors=local_anchor_checks,A4_separation=a4_check)
            all_counts.update(counts)
            print(json.dumps(dict(shot=sid,counts=dict(counts),A4_separation=a4_check),ensure_ascii=True),flush=True)
    assert all_counts['all_source_rows'] == inventory['total_linked_rows']+inventory['total_unlinked_rows']
    receipt = dict(schema='rinkspace.separated_circle_source_graph.v1',source_root=str(SOURCE),output=str(OUTPUT),
        input_sha256=inventory['input_sha256'],state_sha256=inventory['state_sha256'],
        parent_inventory=ref(SOURCE/'all9_inventory.json'),builder=ref(Path(__file__)),implementation=ref(function_path),
        source_reviews=[ref(visual_path),ref(a4_path)],withheld_source_UV_evidence=ref(withheld_path),counts=dict(all_counts),shots=shot_receipts,
        original_seed_inheritance=False,source_node_order_unchanged=True,original_edge_pairs_all_accounted_for=True,
        all_original_source_readout_rows_preserved=True,all_original_row_fields_except_fragment_exactly_preserved=True,
        fragment_renumbering_preserves_original_fragment_field=True,raw_NPZ_reused_as_hard_links=not dry_run,
        source_images_opened=False,source_video_opened=False,source_UV_modified=False,no_new_readouts=True,
        prepared_modified=False,prepare_executed=False,fit_executed=False,user_visual_acceptance=False,pixel_accuracy_accepted=False,
        uncertain_withheld_identity_policy='The S07 A1 fragment identity may seed from its 12 supported candidates; only its three exact explicitly uncertain withheld UV members may remain, never as seed support and never qualified')
    if not dry_run:
        write(OUTPUT/'reconciliation_receipt.json',receipt)
    print('SPLIT_SUMMARY '+json.dumps(dict(dry_run=dry_run,counts=dict(all_counts),A4_separation=shot_receipts['s07']['A4_separation'],
        seed_fragments={sid:r['seed_fragments']for sid,r in shot_receipts.items()},output=str(OUTPUT)),ensure_ascii=True),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    main(parser.parse_args().dry_run)
