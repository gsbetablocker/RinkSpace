"""Reconcile frozen source identities against same-exposure source evidence only."""
from collections import Counter,defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

PROJECT=Path('D:/OneDrive/codex_projs/RinkSpace')
sys.path.insert(0,str(PROJECT/'src'))
from rinkspace.session.material_tracks import source_line_reference_conflict
BASE=Path('D:/CodexData/RinkSpace_Data/runs/native_programme_20261031/feature_refinement_1101')
SOURCE=BASE/'material_linked';OUTPUT=BASE/'material_short_guard'
PREPARED=BASE.parent/'prepared_source_recovered'
BATCHES={'wide_full_1':['s01','s02','s03'],'wide_full_2':['s04','s05','s06'],'wide_full_3':['s07','s08','s09']}
HASHES={}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    p=Path(p).resolve(strict=True);st=p.stat();key=(str(p),st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    if key not in HASHES:
        with p.open('rb') as f:HASHES[key]=hashlib.file_digest(f,'sha256').hexdigest()
    return HASHES[key]
def ref(p):return dict(path=str(Path(p).resolve(strict=True)),sha256=sha(p))
def verify_ref(e):assert sha(e['path'])==e['sha256'];return e
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def compact(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
def nodekey(row):return int(row['pts']),int(row['fragment'])
def rowkey(row):return int(row['pts']),int(row['fragment']),int(row['candidate'])
def main():
    if OUTPUT.exists():raise FileExistsError('Preserve existing graph output: '+str(OUTPUT))
    inventory=read(SOURCE/'all9_inventory.json');manifest=read(PREPARED/'joint_manifest.json');mapping=read(PREPARED/'mapping.json')
    assert inventory['input_sha256']==manifest['point_input_sha256']==mapping['point_input_sha256']
    assert sha(PREPARED/'point_observations.npz')==inventory['input_sha256']
    review_receipt_path=BASE/'c_review_mapping_receipt.json';review_receipt=read(review_receipt_path)
    line_reviews_path=BASE/'line_reviews.json';line_reviews=read(line_reviews_path)
    assert sha(line_reviews_path)==review_receipt['line_reviews']['sha256']
    source_review=verify_ref(review_receipt['source_review']);known_review_path=BASE/'new_line_source_review/manifest.json';known_review=read(known_review_path)
    known_refs=[ref(known_review_path)]
    for r in known_review:
        known_refs.extend([ref(r['source_image']),ref(r['actual_image'])])
    hotspot_review_path=BASE/'preflight_identity/source_review/manifest.json';hotspot_review=read(hotspot_review_path)
    assert hotspot_review['complete'] and hotspot_review['new_over_20_rows_rendered']==66
    root_reviews=[]
    for pt,path,conclusion in ((1235456,'d78_material_145/path0','The three C-labelled new source points are on red small target-circle material, not the same-exposure clearly visible C straight stripe.'),
            (1007360,'d78_material_145/path0','The three C-labelled new source points are on red small target-circle material, not the same-exposure clearly visible C straight stripe.'),
            (1011456,'d78_material_143/path0','The three B2-labelled new source points are on another lower long line and are not collinear with the clearly visible true B2 in this exposure.')):
        frame=next(f for f in hotspot_review['frames'] if f['shot']=='s05' and f['pts']==pt)
        points=[p for p in frame['source_points'] if p['path_id']==path and p['new_over_20']]
        assert len(points)==3;verify_ref(frame['source_image']);verify_ref(frame['actual_image'])
        root_reviews.append(dict(shot='s05',pts=pt,path_id=path,reviewer='main_session',reviewed_surface='actual image containing only current source UV',
            source_BGR_sha256=frame['source_BGR_sha256'],source_image=frame['source_image'],actual_image=frame['actual_image'],points=points,
            source_identity_conclusion=conclusion,review_provenance='Explicit main-session instruction after actually viewing these images',
            user_visual_acceptance=False,pixel_accuracy_accepted=False,scope='Only these source identities in the stated exposures; no decision on the other hotspot groups.'))
    protected_A406=ref(BASE/'separated_circle_packets/validation.json')
    function_path=Path(source_line_reference_conflict.__code__.co_filename).resolve(strict=True)
    bindings={b:read(SOURCE/b/'binding.json') for b in BATCHES}
    assert all(p['feature_id'] in ('B1','B2','C') and p['world_curve_role']=='segment0' and p['closed'] is False
        for binding in bindings.values() for p in binding['inventory'])
    selected_paths={p['native_path_id'] for b in bindings.values() for p in b['inventory']}
    path_index={p:mapping['path_ids'].index(p) for p in selected_paths}
    # One load of the three small old-source fields, never the U96 ID array.
    with np.load(PREPARED/'point_observations.npz',allow_pickle=False) as z:
        old_pts=z['pts'];old_uv=z['uv'];old_pi=z['path_indices']
    reference_groups={}
    for shot in manifest['shots']:
        sid=shot['id'];first,last=shot['row_begin'],shot['row_stop']
        available=np.flatnonzero(np.isin(old_pi[first:last],list(path_index.values())))+first
        order=available[np.lexsort((old_pi[available],old_pts[available]))]
        if len(order):
            keys=np.c_[old_pts[order],old_pi[order]];cuts=np.r_[0,np.flatnonzero(np.any(np.diff(keys,axis=0),axis=1))+1,len(order)]
            for begin,end in zip(cuts[:-1],cuts[1:]):
                pt,pi=map(int,keys[begin]);reference_groups[sid,pt,pi]=old_uv[order[begin:end]].copy()
    OUTPUT.mkdir()
    evidence_path=OUTPUT/'source_evidence_manifest.json'
    write(evidence_path,dict(schema='rinkspace.line_source_conflict_review_evidence.v1',previous_source_evidence=known_refs,
        current_source_review_manifest=ref(hotspot_review_path),main_session_source_identity_reviews=root_reviews,
        original_C_anchor_review=source_review,original_C_anchor_mapping=ref(review_receipt_path),original_line_reviews=ref(line_reviews_path),
        user_visual_acceptance=False,pixel_accuracy_accepted=False,source_images_visually_reviewed_by_this_data_task=False,
        source_UV_modified=False,fit_executed=False))
    known_refs.extend([ref(evidence_path),ref(hotspot_review_path)])
    checks_path=OUTPUT/'source_conflict_checks.jsonl';conflicts=[];payload={};counts=Counter();review_anchor_nodes=defaultdict(set);anchor_checks=[]
    for anchor in review_receipt['anchor_mappings']:
        assert anchor['supported_rows']>0 and anchor['path_id']=='d78_material_145/path0'
        assert anchor['already_seed_connected'] or all(c in line_reviews['approved_components'] for c in anchor['final_pending_components'])
        verify_ref(anchor['source_image']);verify_ref(anchor['actual_image'])
        review_anchor_nodes[anchor['shot']].add((anchor['pts'],anchor['current_fragment']))
    with checks_path.open('x',encoding='utf8') as checks_out:
        for batch,sids in BATCHES.items():
            binding=bindings[batch];assert binding['input_sha256']==inventory['input_sha256'];paths=[p['native_path_id'] for p in binding['inventory']]
            for sid in sids:
                pixels_path=SOURCE/batch/sid/'source_pixels.json';links_path=SOURCE/batch/sid/'source_links.json'
                assert sha(pixels_path)==inventory['shots'][sid]['source_pixels']['sha256']
                assert sha(links_path)==inventory['shots'][sid]['source_links']['sha256']
                pixels=read(pixels_path);links=read(links_path);nodes=defaultdict(list);original_collections={};all_rows={}
                for collection in ('accepted','pending'):
                    for row in pixels[collection]:
                        k=rowkey(row);assert k not in all_rows;all_rows[k]=row
                        n=nodekey(row);nodes[n].append(row)
                        if n in original_collections:assert original_collections[n]==collection
                        original_collections[n]=collection
                excluded=set();node_identities={};node_paths={};node_reports={}
                for n,rows in sorted(nodes.items()):
                    rows.sort(key=lambda r:r['candidate']);identity=rows[0]['identity'];path=rows[0]['path_id']
                    assert all(r['identity']==identity and r['path_id']==path for r in rows)
                    node_identities[n]=identity;node_paths[n]=path
                    reference=reference_groups.get((sid,n[0],path_index[path]),np.empty((0,2)))
                    measured=np.asarray([r['uv'] for r in rows],float);quantiles=np.asarray([r['actual_source_quantiles'] for r in rows],float)
                    result=source_line_reference_conflict(reference,measured,quantiles,known_straight_extension=True)
                    report=dict(shot=sid,batch=batch,pts=n[0],fragment=n[1],identity=identity,native_path_id=path,source_BGR_sha256=rows[0]['source_BGR_sha256'],
                        original_collection=original_collections[n],source_pixels=inventory['shots'][sid]['source_pixels'],source_rows=len(rows),reference_rows=len(reference),
                        candidate_ids=[r['candidate'] for r in rows],compared_candidate_ids=[rows[i]['candidate'] for i in result['compared_candidates']],
                        conflicting_candidate_ids=[rows[i]['candidate'] for i in result['conflicting_candidates']],result=result)
                    checks_out.write(compact(report));counts['fragments_checked']+=1
                    if not len(reference):counts['fragments_without_old_same_exposure_source']+=1
                    elif not result['reference_supported']:counts['fragments_without_supported_old_straight_reference']+=1
                    else:counts['fragments_with_supported_old_straight_reference']+=1
                    node_reports[n]=report
                    if result['contradicted']:
                        excluded.add(n);counts['contradicted_fragments']+=1;counts['contradicted_rows']+=len(rows)
                        if original_collections[n]=='accepted':counts['originally_accepted_contradicted_rows']+=len(rows)
                        conflicts.append(dict(report,old_same_exposure_source_UV=reference.tolist(),actual_measured_UV=measured.tolist(),actual_measured_quantiles=quantiles.tolist(),
                            raw_reference=rows[0]['source_reference'],raw_reference_sha256=sha(rows[0]['source_reference']),frame_metadata=ref(SOURCE/batch/sid/f'{n[0]}.json'),
                            disposition='Source identity contradiction; raw and metadata preserved, this fragment has no identity propagation edges or seed.'))
                for anchor in [a for a in review_receipt['anchor_mappings'] if a['shot']==sid]:
                    n=(anchor['pts'],anchor['current_fragment']);assert n in nodes
                    for hit in anchor['candidate_mapping']:
                        original=pixels[hit['current_collection']][hit['current_row_index']]
                        assert original['pts']==anchor['pts'] and original['fragment']==anchor['current_fragment'] and original['candidate']==hit['current_candidate_id']
                        assert original['uv']==hit['uv'] and original['source_BGR_sha256']==anchor['source_BGR_sha256'] and original['path_id']==anchor['path_id']
                    anchor_checks.append(dict(shot=sid,pts=n[0],fragment=n[1],review_index=anchor['review_index'],excluded_by_source_conflict=n in excluded,
                        source_image=anchor['source_image'],actual_image=anchor['actual_image'],source_BGR_sha256=anchor['source_BGR_sha256']))
                payload[sid]=dict(batch=batch,pixels=pixels,links=links,nodes=nodes,all_rows=all_rows,excluded=excluded,identities=node_identities,paths=node_paths,original_collections=original_collections,reports=node_reports)
    targets=[]
    for sid,pt,path in (('s05',951552,'d78_material_142/path0'),('s06',1391104,'d78_material_142/path0'),
            ('s05',965888,'d78_material_142/path0'),('s05',1234944,'d78_material_145/path0'),
            ('s05',1235456,'d78_material_145/path0'),('s05',1007360,'d78_material_145/path0'),('s05',1011456,'d78_material_143/path0')):
        hits=[r for r in conflicts if r['shot']==sid and r['pts']==pt and r['native_path_id']==path]
        targets.append(dict(shot=sid,pts=pt,native_path_id=path,contradiction_hit=bool(hits),fragments=[r['fragment'] for r in hits],rows=sum(r['source_rows'] for r in hits)))
    previous_path=BASE/'material_identity/source_conflicts.json';previous=read(previous_path)
    previous_nodes={(r['shot'],r['pts'],r['fragment']) for r in previous['conflicts']}
    current_nodes={(r['shot'],r['pts'],r['fragment']) for r in conflicts}
    assert previous_nodes<=current_nodes
    additional=[r for r in conflicts if (r['shot'],r['pts'],r['fragment']) not in previous_nodes]
    additional_counts=dict(previous_conflicted_fragments=len(previous_nodes),additional_conflicted_fragments=len(additional),
        additional_conflicted_rows=sum(r['source_rows'] for r in additional),
        additional_originally_accepted_fragments=sum(r['original_collection']=='accepted' for r in additional),
        additional_originally_accepted_rows=sum(r['source_rows'] for r in additional if r['original_collection']=='accepted'),
        previous_originally_accepted_conflicted_rows=previous['counts']['originally_accepted_contradicted_rows'],
        current_originally_accepted_conflicted_rows=counts['originally_accepted_contradicted_rows'])
    conflict_path=OUTPUT/'source_conflicts.json'
    conflict_doc=dict(schema='rinkspace.same_exposure_source_line_conflicts.v1',input_sha256=inventory['input_sha256'],state_sha256=inventory['state_sha256'],
        parent_source_inventory=ref(SOURCE/'all9_inventory.json'),implementation=ref(function_path),function='source_line_reference_conflict',function_defaults=dict(tolerance_px=3.,known_straight_extension=True),
        evidence_policy='Only same-exposure old source UV and frozen new actual boundary quantiles; explicitly verified native B1/B2/C straight continuation uses propagated endpoint uncertainty outside the old tangent interval. No UV is generated. Missing, curved or inconsistent old support does not negate a fragment.',
        comparison_to_previous=dict(previous_report=ref(previous_path),counts=additional_counts,
            additional_conflicted_nodes=[dict(shot=r['shot'],pts=r['pts'],fragment=r['fragment'],identity=r['identity'],rows=r['source_rows'],original_collection=r['original_collection']) for r in additional]),
        source_review_evidence=known_refs,checks=ref(checks_path),counts=dict(counts),known_error_targets=targets,reviewed_C_anchor_checks=anchor_checks,conflicts=conflicts,
        source_UV_modified=False,source_identity_fields_modified=False,prepare_executed=False,fit_executed=False)
    write(conflict_path,conflict_doc)
    print('SCAN_SUMMARY '+compact(dict(counts=dict(counts),additional_vs_previous=additional_counts,known_error_targets=targets,reviewed_C_anchor_conflicts=[a for a in anchor_checks if a['excluded_by_source_conflict']],report=ref(conflict_path))),flush=True)
    batch_receipts={};shot_receipts={};total_linked=0;total_pending=0;hard_link_count=0
    for batch,sids in BATCHES.items():
        out_batch=OUTPUT/batch;out_batch.mkdir()
        original_binding=bindings[batch];new_binding=copy.deepcopy(original_binding)
        new_binding.update(reused_actual_source=str((SOURCE/batch).resolve()),reused_capture_binding_sha256=sha(SOURCE/batch/'binding.json'),
            source_graph_reconciliation=dict(parent_binding=ref(SOURCE/batch/'binding.json'),source_conflicts=ref(conflict_path),implementation=ref(function_path),
                reviewed_C_anchors=ref(review_receipt_path),source_reviews=ref(line_reviews_path),old_source_prepared=str(PREPARED),input_sha256=inventory['input_sha256'],
                raw_readouts_reused_without_change=True,no_new_readouts=True,source_images_opened=False,fit_executed=False))
        write(out_batch/'binding.json',new_binding);batch_receipts[batch]=ref(out_batch/'binding.json')
        for sid in sids:
            data=payload[sid];folder=out_batch/sid;folder.mkdir();original_folder=SOURCE/batch/sid
            for f in original_folder.iterdir():
                if f.stem.isdigit() and f.suffix.lower() in ('.json','.npz'):
                    target=folder/f.name;os.link(f,target);assert f.samefile(target);hard_link_count+=1
            excluded=data['excluded'];nodes=data['nodes'];identities=data['identities'];old_links=data['links']
            allowed=set(nodes)-excluded;parent={n:n for n in allowed}
            def find(n):
                while parent[n]!=n:parent[n]=parent[parent[n]];n=parent[n]
                return n
            edges=[];removed_edges=[];ignored=Counter()
            for edge in old_links['edges']:
                a=(edge['from_pts'],edge['a']);b=(edge['to_pts'],edge['b']);assert a in nodes and b in nodes
                reason=None
                if edge.get('unambiguous') is not True:reason='original_edge_not_unambiguous'
                elif identities[a]!=identities[b]:reason='different_recorded_identity'
                elif a in excluded or b in excluded:reason='incident_to_source_contradicted_fragment'
                if reason:removed_edges.append(dict(edge=edge,reason=reason));ignored[reason]+=1;continue
                assert edge.get('source_only') is True and edge.get('identity_assumed_from_curve_name') is False
                parent[find(a)]=find(b);edges.append(edge)
            old_seeds={tuple(n) for n in old_links['seed_fragments']};assert old_seeds<=set(nodes)
            kept_old_seeds=old_seeds-excluded;review_seeds=review_anchor_nodes[sid]-excluded
            seeds=kept_old_seeds|review_seeds;assert seeds<=allowed
            roots={find(n) for n in seeds};accepted=[];pending=[]
            for n in sorted(nodes):
                target=accepted if n in allowed and find(n) in roots else pending
                target.extend(nodes[n])
            rebuilt={rowkey(r):r for rows in (accepted,pending) for r in rows};assert rebuilt.keys()==data['all_rows'].keys()
            data['new_accepted_keys']={rowkey(r) for r in accepted}
            assert all(rebuilt[k]==data['all_rows'][k] for k in rebuilt)
            pixels=dict(shot=sid,accepted=accepted,pending=pending,source_UV_changed=False,
                source_graph_reconciliation=dict(source_conflicts=ref(conflict_path),contradicted_fragment_nodes=[list(n) for n in sorted(excluded)],
                    contradicted_rows_preserved_in_pending=True,raw_record_fields_unchanged=True,source_identity_pending_not_confirmed_error=True))
            source_path=folder/'source_pixels.json';write(source_path,pixels)
            links=dict(edges=edges,seed_fragments=[list(n) for n in sorted(seeds)],source_identity_is_assistant_inference=True,
                seed_matching_rule='Retained original source-overlap seeds plus exact reviewed C exposure/fragment anchors; conflicting fragments cannot seed or propagate.',
                required_distinct_old_seed_uv=old_links['required_distinct_old_seed_uv'],required_matched_old_seed_uv_span_px=old_links['required_matched_old_seed_uv_span_px'],
                parent_source_links=ref(original_folder/'source_links.json'),source_conflicts=ref(conflict_path),excluded_conflicting_fragments=[list(n) for n in sorted(excluded)],
                graph_nodes=[list(n) for n in sorted(allowed)],original_seed_fragments=[list(n) for n in sorted(old_seeds)],
                retained_original_seed_fragments=[list(n) for n in sorted(kept_old_seeds)],reviewed_C_seed_fragments=[list(n) for n in sorted(review_seeds)],
                reviewed_C_seed_evidence=ref(review_receipt_path),withheld_source_UV_evidence=ref(line_reviews_path),removed_edges=removed_edges,
                edges_only_original_same_identity_unambiguous=True,contradicted_fragments_retained_only_as_isolated_pending_records=True)
            link_path=folder/'source_links.json';write(link_path,links)
            coverage={p['curve_id']:dict(read_frames=0,source_rows=0,seed_frames=0) for p in original_binding['inventory']}
            for n,rs in nodes.items():
                row=coverage[identities[n]];row['read_frames']+=1;row['source_rows']+=len(rs);row['seed_frames']+=int(n in seeds)
            write(folder/'coverage.json',dict(by_path=coverage,accepted_rows=len(accepted),pending_rows=len(pending),
                contradicted_fragments=len(excluded),contradicted_rows=sum(len(nodes[n]) for n in excluded),
                identity_propagation_graph_nodes=len(allowed),source_record_fragment_nodes=len(nodes),reviewed_C_seed_fragments=len(review_seeds),
                excluded_fragments_still_count_as_stored_source_rows=True))
            write(folder/'source_verified.json',dict(shot=sid,accepted=accepted,source_packet=str(source_path),source_packet_sha256=sha(source_path),
                source_links=str(link_path),source_links_sha256=sha(link_path),fit_eligible=False,review_state='awaiting_source_conflict_list_and_representative_review',
                user_visual_acceptance=False,source_conflicts=ref(conflict_path),contradicted_fragments_withheld=True,source_UV_changed=False))
            shot_receipts[sid]=dict(batch=batch,original_source_pixels=ref(original_folder/'source_pixels.json'),source_pixels=ref(source_path),source_links=ref(link_path),
                source_rows=len(rebuilt),old_accepted_rows=len(data['pixels']['accepted']),accepted_rows=len(accepted),old_pending_rows=len(data['pixels']['pending']),pending_rows=len(pending),
                contradicted_fragments=len(excluded),contradicted_rows=sum(len(nodes[n]) for n in excluded),identity_propagation_graph_nodes=len(allowed),
                retained_edges=len(edges),removed_edges=dict(ignored),original_seeds=len(old_seeds),removed_original_seeds=len(old_seeds&excluded),
                retained_original_seeds=len(kept_old_seeds),reviewed_C_seeds=len(review_seeds),new_reviewed_C_seeds=len(review_seeds-kept_old_seeds),
                raw_record_fields_exactly_preserved=True)
            total_linked+=len(accepted);total_pending+=len(pending)
            print('RECONCILED_SHOT '+compact(dict(shot=sid,**{k:v for k,v in shot_receipts[sid].items() if k in ('accepted_rows','pending_rows','contradicted_fragments','contradicted_rows','removed_edges','reviewed_C_seeds')})),flush=True)
    assert total_linked+total_pending==inventory['total_linked_rows']+inventory['total_unlinked_rows']
    hotspot_points=[]
    for frame in hotspot_review['frames']:
        sid=frame['shot'];data=payload[sid]
        for point in [p for p in frame['source_points'] if p['new_over_20']]:
            matches=[(node,row) for node,rs in data['nodes'].items() if node[0]==frame['pts'] for row in rs
                if row['path_id']==point['path_id'] and row['uv']==point['uv'] and row['source_BGR_sha256']==frame['source_BGR_sha256']]
            assert len(matches)==1,(sid,frame['pts'],point['prepared_row'],matches)
            node,row=matches[0];direct=node in data['excluded'];accepted=rowkey(row) in data['new_accepted_keys']
            state='isolated_by_same_exposure_source_conflict' if direct else 'retained_seed_connected' if accepted else 'pending_without_seed_after_graph_reconciliation'
            hotspot_points.append(dict(prepared_identity_row=point['prepared_row'],shot=sid,pts=frame['pts'],path_id=row['path_id'],uv=row['uv'],
                raw_fragment=node[1],raw_candidate_id=row['candidate'],original_collection=data['original_collections'][node],
                source_BGR_sha256=row['source_BGR_sha256'],source_conflict=direct,final_collection='accepted' if accepted else 'pending',
                disposition=state,source_reference=row['source_reference'],reference_rows=data['reports'][node]['reference_rows'],
                reference_supported=data['reports'][node]['result']['reference_supported']))
    assert len(hotspot_points)==66 and len({p['prepared_identity_row'] for p in hotspot_points})==66
    hotspot_check_path=OUTPUT/'hotspot66_source_conflict_check.json'
    write(hotspot_check_path,dict(schema='rinkspace.hotspot_source_conflict_postcheck.v1',source_review_manifest=ref(hotspot_review_path),
        source_evidence_manifest=ref(evidence_path),source_conflicts=ref(conflict_path),reported_points=66,
        disposition_counts=dict(Counter(p['disposition'] for p in hotspot_points)),points=hotspot_points,
        hotspot_membership_used_to_decide_source_conflict=False,source_conflict_inputs='Same-exposure old source UV and actual new paired-boundary quantiles only',
        source_UV_modified=False,prepare_executed=False,fit_executed=False))
    verify_ref(protected_A406)
    receipt=dict(schema='rinkspace.reconciled_source_line_graph.v1',source_root=str(SOURCE),output=str(OUTPUT),input_sha256=inventory['input_sha256'],state_sha256=inventory['state_sha256'],
        source_conflicts=ref(conflict_path),source_evidence_manifest=ref(evidence_path),hotspot66_source_conflict_check=ref(hotspot_check_path),
        protected_A406_packets_validation=protected_A406,reviewed_C_anchor_mapping=ref(review_receipt_path),line_reviews=ref(line_reviews_path),implementation=ref(function_path),builder=ref(Path(__file__)),
        counts=dict(counts),comparison_to_previous=additional_counts,known_error_targets=targets,batches=batch_receipts,shots=shot_receipts,all_source_rows=total_linked+total_pending,
        accepted_rows=total_linked,pending_rows=total_pending,raw_and_frame_metadata_hard_links=hard_link_count,
        parent_small_point_field_loads=1,parent_fields_loaded=['pts','uv','path_indices'],parent_observation_ids_loaded=False,
        raw_record_fields_exactly_preserved=True,source_images_opened=False,source_UV_modified=False,source_identity_fields_modified=False,
        source_root_modified=False,prepare_executed=False,fit_executed=False,qualifier_executed=False,packet_builder_executed=False,
        limitation='No conflict means no contradiction established by this comparison; it is not source-identity confirmation. Excluded nodes remain raw, isolated pending history without propagation edges or seeds.')
    write(OUTPUT/'reconciliation_receipt.json',receipt)
    print('RECONCILIATION_COMPLETE '+compact(dict(receipt=ref(OUTPUT/'reconciliation_receipt.json'),accepted_rows=total_linked,pending_rows=total_pending,hard_links=hard_link_count,known_error_targets=targets)),flush=True)
if __name__=='__main__':main()
