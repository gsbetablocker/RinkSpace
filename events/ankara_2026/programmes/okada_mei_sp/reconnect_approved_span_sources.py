"""Verify disconnected finite-span source groups against the approved source RGB.

Reuses immutable actual RGB readouts and all public thresholds. A source H only
initializes bidirectional RGB profile matching; it never supplies measurement UV.
"""
from pathlib import Path
import argparse,json,hashlib,shutil,time
from collections import Counter
import cv2,numpy as np
from rinkspace.session.material_tracks import source_nodes,link_fragments,propagate_readout_identities
from capture_approved_span_pilot import digest,SID,IDENTITY,REFERENCE_PTS
from capture_independent_ice import load_navigation_graph,navigation_between,ice_source_mask,capture_frames
from capture_independent_materials import native_ice_inventory,source_exclusion,spaced_indices,write,plain
from capture_independent_badge import exact_frames
from build_independent_shot_packets import build


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in('root','capture-run','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();run=a.capture_run.resolve();out=a.output.resolve();cv2.setNumThreads(2);start=time.monotonic()
    if out.exists():raise FileExistsError(out)
    index=json.loads((root/'index.json').read_text('utf8'));old=json.loads((run/'report.json').read_text('utf8'));source=Path(index['source']['resolved_path'])
    if digest(source)!=index['source']['actual_sha256']or old['source_video_sha256']!=index['source']['actual_sha256']:raise ValueError('Original source differs')
    capture=run/'capture'/SID;old_labels=json.loads((capture/'source_labels.json').read_text('utf8'));old_links=json.loads((capture/'source_links.json').read_text('utf8'))
    records=[json.loads(l)for l in(capture/'source_readout_reports.jsonl').open(encoding='utf8')];lookup={r['pts']:i for i,r in enumerate(records)};refi=lookup[REFERENCE_PTS]
    groups=old_labels['whole_source_groups'];anchors=old_labels['anchors'];counts=np.diff(old_labels['frame_offsets']);components=np.array(old_labels['candidate_component']);offsets=np.array(old_labels['frame_offsets'])
    labelmap={l['component']:l for l in old_labels['labels']}
    targets=sorted({g['frame']for g in groups if not any(labelmap.get(int(components[offsets[g['frame']]+ci]),{}).get('status')=='anchored'for ci in g['candidate_ids'])})
    shot=next(s for s in index['shots']if s['shot_id']==SID);graph,gn=load_navigation_graph(root,shot)
    world=dict(np.load(root/'native_world.npz',allow_pickle=False));cuts=json.loads((root/'native_world.json').read_text('utf8'))['cuts'];paths=[p for p in native_ice_inventory(cuts,world['xyz'])if p['native_path_id']==IDENTITY]
    paths[0]=dict(paths[0],material_signal='red')
    ref=records[refi];reference_image=exact_frames(source,[REFERENCE_PTS])[REFERENCE_PTS];reference_rgb=cv2.cvtColor(reference_image,cv2.COLOR_BGR2RGB)
    if hashlib.sha256(reference_image.tobytes()).hexdigest()!=ref['source_BGR_sha256']:raise ValueError('Anchor original RGB changed')
    reference_mask,_=source_exclusion(root,SID,REFERENCE_PTS);reference_mask,_=ice_source_mask(reference_rgb,reference_mask)
    reference_measurement=dict(np.load(ref['actual_source_measurements'],allow_pickle=False));reference_nodes=source_nodes(reference_rgb,reference_measurement,ref['whole_source_fragments'],paths)
    links=list(old_links['readout_links']);attempts=[];extra=[]
    for _,(pts,image)in capture_frames(index,shot,[records[i]['pts']for i in targets]):
        fi=lookup[pts];record=records[fi];rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);mask,mp=source_exclusion(root,SID,pts);mask,_=ice_source_mask(rgb,mask)
        if hashlib.sha256(image.tobytes()).hexdigest()!=record['source_BGR_sha256']or hashlib.sha256(mask.tobytes()).hexdigest()!=record['applied_mask_sha256']:
            raise ValueError('Actual source RGB or fine exclusion changed')
        raw=Path(record['actual_source_measurements'])
        if digest(raw)!=record['actual_source_measurements_sha256']:raise ValueError('Original RGB readout changed')
        measured=dict(np.load(raw,allow_pickle=False));nodes=source_nodes(rgb,measured,record['whole_source_fragments'],paths)
        h,reason=navigation_between(graph,REFERENCE_PTS,pts);edges=[]if h is None else link_fragments(reference_rgb,rgb,reference_nodes,nodes,reference_mask,mask,h)
        attempts.append(dict(reference_pts=REFERENCE_PTS,current_pts=pts,navigation_status=reason,
            navigation_H=None if h is None else h.tolist(),actual_RGB_fragment_edges=edges,source_BGR_sha256=record['source_BGR_sha256'],
            actual_UV_unchanged=True,H_or_profile_flow_UV_is_measurement=False))
        for e in edges:
            link=dict(previous=refi,current=fi,pairs=[[int(reference_nodes['ids'][aa]),int(nodes['ids'][bb])]for aa,bb in e['pairs']],
                unambiguous=e['unambiguous'],source_only=True,fragment_edge=dict(e),node_index_domain='original actual_RGB candidate IDs',
                connection='direct actual_RGB reference check across missing source links')
            links.append(link);extra.append(link)
    offsets,components,labels=propagate_readout_identities(counts,links,anchors,qualified_groups=groups);labelmap={l['component']:l for l in labels}
    out.mkdir(parents=True);qualified=out/'qualified';folder=qualified/SID;folder.mkdir(parents=True)
    write(out/'source_links.json',dict(original_adjacent_links=old_links['adjacent_links'],reference_check_attempts=attempts,readout_links=links,
        original_source_links=str(capture/'source_links.json'),original_source_links_sha256=digest(capture/'source_links.json'),source_only=True))
    write(out/'source_labels.json',dict(anchors=anchors,whole_source_groups=groups,labels=labels,candidate_component=components,frame_offsets=offsets,identity_from_H_or_curve_names=False))
    proof=dict(source_links=str(out/'source_links.json'),source_links_sha256=digest(out/'source_links.json'),source_labels=str(out/'source_labels.json'),source_labels_sha256=digest(out/'source_labels.json'),
        source_graph=gn,anchor_approval=old['anchor_approval'],source_span_reference=old['source_span_reference'],source_span_reference_sha256=old['source_span_reference_sha256'])
    summaries=[];selected_frames=[];qualified_records={};reasons=Counter()
    with (folder/'source_rows.jsonl').open('x',encoding='utf8')as selected,(out/'all_candidate_rows.jsonl').open('x',encoding='utf8')as candidate_out:
        for fi,line in enumerate((capture/'all_candidate_rows.jsonl').open(encoding='utf8')):
            source_row=json.loads(line);record=records[fi];all_rows=[];eligible=[]
            ids={ci for g in groups if g['frame']==fi for ci in g['candidate_ids']}
            for ci,row in enumerate(source_row['all_actual_RGB_candidates']):
                why=[w for w in row['pending_reasons']if w!='no_unique_approved_anchor_RGB_link_identity'];label=labelmap.get(int(components[offsets[fi]+ci]))
                if not label or label['status']!='anchored'or label['identities']!=[IDENTITY]:why.append('no_unique_approved_anchor_RGB_link_identity')
                row=dict(row,pending_reasons=why,machine_source_qualified=not why,source_identity_basis='approved finite original-source span + verified actual bidirectional RGB links + unchanged public whole-source paired-line organization')
                all_rows.append(row);reasons.update(why)
                if not why:eligible.append(ci)
            rows=[all_rows[eligible[i]]for i in spaced_indices([all_rows[k]['uv']for k in eligible],spacing=48.)]
            candidate_out.write(json.dumps(plain(dict(source_row,all_actual_RGB_candidates=all_rows,fit_eligible=False)),ensure_ascii=False)+'\n')
            new_record=dict(record,rows=rows,fit_eligible=bool(rows),source_identity_proof=proof,
                review_state='approved_finite_span_and_actual_RGB_links_machine_qualified'if rows else'pending_no_supported_source_span',
                qualification=dict(source_capture=str(capture/'candidate_source_rows.jsonl'),source_capture_sha256=digest(capture/'candidate_source_rows.jsonl'),
                    actual_UV_unchanged=True,native_residual_used=False,public_thresholds_changed=False,human_accuracy_claim=False))
            selected.write(json.dumps(plain(new_record),ensure_ascii=False)+'\n');summaries.append(dict(pts=record['pts'],rows=len(rows),actual_RGB_candidates=len(all_rows),source_navigation=record['navigation_status'],pending_reasons=record['pending_reasons']))
            if rows:selected_frames.append(fi);qualified_records[fi]=new_record
    shutil.copyfile(run/'qualified'/SID/'poses.npz',folder/'poses.npz')
    binding=json.loads((run/'qualified/binding.json').read_text('utf8'));binding.update(source_identity_proof=proof,source_capture=str(capture.parent),
        reference_reconnection_implementation=str(Path(__file__).resolve()),reference_reconnection_implementation_sha256=digest(__file__),
        reference_RGB_check_thresholds_changed=False,original_source_capture_binding_sha256=digest(capture.parent/'binding.json'))
    write(qualified/'binding.json',binding)
    first,last=selected_frames[0],selected_frames[-1];images=exact_frames(source,[records[first]['pts'],records[last]['pts']]);new_proofs=[]
    for fi in sorted({first,last}):
        record=qualified_records[fi];image=images[record['pts']].copy();guide=record['finite_source_guide'];cv2.polylines(image,[np.rint(guide).astype('int32')],False,(0,220,220),1)
        for row in record['rows']:
            cv2.circle(image,tuple(np.rint(row['uv']).astype(int)),4,(0,255,0),1)
            for side,c in zip(np.array(row['actual_source_quantiles']),[(255,100,0),(255,0,255)]):cv2.line(image,tuple(np.rint(side[0]).astype(int)),tuple(np.rint(side[2]).astype(int)),c,1)
        cv2.putText(image,f'PTS {record["pts"]} | finite source span | actual RGB identity verified | NO FIT',(20,35),0,.65,(0,220,255),2)
        p=out/f'{record["pts"]}_qualified_endpoint_proof.jpg';cv2.imwrite(str(p),image);new_proofs.append(str(p))
    packet=build(root,qualified,out/'source_packets',shot_ids=[SID]);missing=[s['pts']for s in summaries if not s['rows']];maxrun=0;runlength=0
    for s in summaries:
        runlength=runlength+1 if not s['rows']else 0;maxrun=max(maxrun,runlength)
    report=dict(source_video_sha256=index['source']['actual_sha256'],native_frames=len(records),qualified_frames=len(selected_frames),qualified_rows=sum(s['rows']for s in summaries),
        source_span_extended=False,actual_UV_unchanged=True,source_graph_connected_frames=old['source_graph_connected_frames'],
        public_thresholds_changed=False,actual_RGB_candidates=int(counts.sum()),original_capture_run=str(run),original_capture_report_sha256=digest(run/'report.json'),
        direct_reference_RGB_check_frames=len(attempts),direct_reference_RGB_fragment_edges=len(extra),new_unambiguous_reference_RGB_edges=sum(e['unambiguous']for e in extra),
        source_identity_proof=proof,all_candidate_pending_reasons=dict(reasons),pending_native_PTS=missing,longest_no_qualified_run_frames=maxrun,longest_no_qualified_run_seconds=maxrun*.02,
        source_rows=str(folder/'source_rows.jsonl'),source_packets=str(out/'source_packets'/SID),packet_manifest=str(out/'source_packets/packets_manifest.json'),
        old_same_shot_PTS_path_group_upsert_replaces_not_appends=True,group_key=[SID,'native PTS',IDENTITY],source_UV_or_fit_state_modified=False,
        representative_proof_images=old['representative_proof_images']+new_proofs,frames=summaries,implementation_sha256={str(Path(__file__).resolve()):digest(__file__)},complete=True,seconds=time.monotonic()-start)
    write(out/'report.json',report);print(json.dumps({k:report[k]for k in['native_frames','qualified_frames','qualified_rows','direct_reference_RGB_check_frames','new_unambiguous_reference_RGB_edges','source_packets','seconds']}),flush=True)


if __name__=='__main__':main()