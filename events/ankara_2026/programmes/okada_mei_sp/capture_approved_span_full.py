"""Stream S08's approved finite goal-line span into source-qualified packets.

Only original RGB q50s are observations. The four observed source vertices are
never extended. Supported source H guides a search; actual paired boundaries,
whole-source line organization and bidirectional RGB links qualify identity.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,time
from collections import Counter
import cv2
import numpy as np
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader,_batched_actual_profiles
from rinkspace.vision_source_shapes import _MaterialProfileFrame
from rinkspace.session.material_tracks import (qualify_visible_line_fragments,source_nodes,
    source_line_reference_support,link_fragments,propagate_readout_identities)
from capture_approved_span_pilot import finite_source_span,digest,SID,IDENTITY,REFERENCE_PTS
from capture_independent_badge import exact_frames
from capture_independent_ice import approval_sources,anchor_records,load_navigation_graph,navigation_between,ice_source_mask
from capture_independent_materials import native_ice_inventory,source_exclusion,spaced_indices,plain,write
from prepare_independent_navigation import frames
from build_independent_shot_packets import build


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    cv2.setNumThreads(2);started=time.monotonic();index=json.loads((root/'index.json').read_text('utf8'))
    source=Path(index['source']['resolved_path']);source_sha=digest(source)
    if source_sha!=index['source']['actual_sha256']:raise ValueError('Original video hash differs')
    declarations,approval=approval_sources(root/'source_anchor_bank',root/'source_anchor_bank/approval.json',[SID],source_sha)
    references=[g for g in anchor_records(declarations,source_sha)if g['pts']==REFERENCE_PTS and g['path_id']==IDENTITY]
    if len(references)!=1:raise ValueError('Unique approved finite source reference required')
    reference=references[0];shot=next(s for s in index['shots']if s['shot_id']==SID);graph,graph_note=load_navigation_graph(root,shot)
    world=dict(np.load(root/'native_world.npz',allow_pickle=False));cuts=json.loads((root/'native_world.json').read_text('utf8'))['cuts']
    paths=[q for q in native_ice_inventory(cuts,world['xyz'])if q['native_path_id']==IDENTITY]
    if len(paths)!=1:raise ValueError('Unique current goal-line metadata required')
    paths[0]=dict(paths[0],material_signal=reference['material_signal']or'red')
    reader=IceProgrammeSourceReader(None,source_sha256=source_sha,time_base='1/12800',native_paths=paths,
        sample_spacing_px=24.,search_radius_px=40.,separate_bands=True,prepared_source_context=dict(source_sha256=source_sha,
            time_base='1/12800',authority=[dict(path=reference['source_rows_file'],sha256=reference['source_rows_file_sha256'])]))
    out.mkdir(parents=True);capture=out/'capture';qualified=out/'qualified'
    for base in(capture,qualified):(base/SID).mkdir(parents=True)
    folder=capture/SID;proofs=folder/'source_profiles';proofs.mkdir()
    project_root=Path(__file__).resolve().parents[4]
    implementation=[Path(__file__),Path(finite_source_span.__code__.co_filename),Path(reader.read.__code__.co_filename),
        Path(qualify_visible_line_fragments.__code__.co_filename),Path(approval_sources.__code__.co_filename),
        Path(build.__code__.co_filename),Path(source_exclusion.__code__.co_filename)]
    binding=dict(schema='rinkspace.finite_source_span_full.v1',source_video_sha256=source_sha,source_sha256=source_sha,
        source_video=str(source),source_index_sha256=digest(root/'index.json'),source_graph=graph_note,
        independent_source_capture=True,legacy_new_shot_inputs_read=False,native_state_loaded=False,selected_shot_ids=[SID],
        selected_path_ids=[IDENTITY],source_span_reference=reference['source_rows_file'],source_span_reference_sha256=reference['source_rows_file_sha256'],
        source_span_reference_pts=REFERENCE_PTS,source_span_UV=reference['uv'].tolist(),source_span_extended=False,
        anchor_approval=approval,primary_pilot_review='Primary reviewed 1806080 and 1805568 finite-span source crops and authorized the same finite-span RGB algorithm over all S08 exposures.',
        query_spacing_px=24.,fit_spacing_px=48.,short_feature_minimum_samples=4,search_radius_px=40.,
        H_is_navigation_only=True,source_RGB_supplies_all_UV=True,source_UV_synthesized=False,
        public_qualification_thresholds_changed=False,source_reference_incoming_navigation_is_not_same_frame_validity=True,
        implementation_sha256={str(f.resolve()):digest(f)for f in implementation},fit_eligible=False,solver_started=False)
    write(capture/'binding.json',binding);data=[];whole_groups=[];anchors=[];source_links=[];link_records=[];previous=None
    for fi,(pts,image)in enumerate(frames(index,shot)):
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);bgr_sha=hashlib.sha256(image.tobytes()).hexdigest()
        mask,mp=source_exclusion(root,SID,pts);mask,floor=ice_source_mask(rgb,mask);h,reason=navigation_between(graph,REFERENCE_PTS,pts)
        if pts==REFERENCE_PTS and bgr_sha!=reference['source_BGR_sha256']:raise ValueError('Same-frame approved anchor RGB differs')
        note=dict(pts=pts,source_BGR_sha256=bgr_sha,source_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            source_video_sha256=source_sha,mask_path=str(mp),mask_sha256=digest(mp),ice_floor_receipt=floor,
            applied_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),navigation_status=reason,
            incoming_navigation_supported=None if graph is None else bool(graph['supported'][graph['index'][pts]]),
            reference_pts=REFERENCE_PTS,fit_eligible=False,whole_source_fragments=[],source_qualification=None,
            finite_source_guide=None,query_count=0,actual_RGB_candidate_count=0,actual_source_measurements=None,
            pending_reasons=['unsupported_source_graph_interval']if h is None else [])
        measured=None;nodes=None
        if h is not None:
            guide=finite_source_span(reference['uv'],h)
            measured=reader.read(rgb,pts,None,mask,shot_id=SID,native_projections={IDENTITY:dict(uv=guide,valid=np.ones(len(guide),bool))})
            fragments,qualification=qualify_visible_line_fragments(measured,[0],mask);nodes=source_nodes(rgb,measured,fragments,paths)
            arrays={k:v for k,v in measured.items()if isinstance(v,np.ndarray)}
            if len(measured['query_uv']):
                pp=_batched_actual_profiles(_MaterialProfileFrame(np.ascontiguousarray(rgb),dict(source_sha256=source_sha,source_pts=pts,
                    time_base='1/12800',shot_id=SID)),measured['query_uv'],measured['query_normal'],measured['query_interval'],np.full(len(measured['query_uv']),'red'))
                arrays.update(profile_offsets=pp[0],actual_luma_profiles=pp[1],smoothed_luma_profiles=pp[2],profile_valid=pp[3])
                endpoints=measured['query_uv'][:,None]+measured['query_normal'][:,None]*measured['query_interval'][:,:,None]
                lo=np.maximum(np.floor(endpoints.reshape(-1,2).min(0)-3).astype(int),[0,0]);hi=np.minimum(np.ceil(endpoints.reshape(-1,2).max(0)+4).astype(int),[1920,1080])
                arrays.update(source_RGB_roi=rgb[lo[1]:hi[1],lo[0]:hi[0]],applied_exclusion_roi=mask[lo[1]:hi[1],lo[0]:hi[0]],proof_roi_xyxy=np.r_[lo,hi])
            arrays.update(finite_source_guide=guide,H_current_from_reference=h)
            raw=proofs/f'{pts}.npz';np.savez_compressed(raw,**arrays)
            note.update(source_navigation_H=h.tolist(),finite_source_guide=guide.tolist(),search_guide_span_px=float(np.linalg.norm(np.diff(guide,axis=0),axis=1).sum()),
                query_count=len(measured['query_uv']),actual_RGB_candidate_count=len(measured['candidate_uv']),whole_source_fragments=fragments,
                source_qualification=qualification,actual_source_measurements=str(raw),actual_source_measurements_sha256=digest(raw),
                pending_reasons=[]if fragments else['no_whole_qualified_paired_source_span'])
            for report in qualification['whole_line_reports']:
                if report.get('source_line_support_qualified')is True:
                    whole_groups.append(dict(frame=fi,candidate_ids=report['selected_candidate_ids'],source_line_support_qualified=True,
                        proof=str(folder/'source_readout_reports.jsonl'),qualification='unchanged public paired-boundary whole-source-line organizer'))
            if pts==REFERENCE_PTS:
                note['anchor_matches']=[]
                for j,fragment in enumerate(fragments):
                    ids=np.asarray(fragment['candidate_ids'],int);support=source_line_reference_support(reference['uv'],measured['candidate_uv'][ids],measured['candidate_query'][ids])
                    note['anchor_matches'].append(dict(fragment=j,support=support))
                    if support['qualified']:
                        for ci in ids:anchors.append(dict(frame=fi,candidate=int(ci),identity=IDENTITY,source_line_support=support,
                            approval_group=reference['approval_group'],approval_sha256=reference['approval_sha256'],source_line_sha256=reference['source_line_sha256']))
        if previous is not None:
            hh,nav_reason=navigation_between(graph,previous['pts'],pts);edges=[]
            if hh is not None and previous['nodes']is not None and nodes is not None:
                edges=link_fragments(previous['rgb'],rgb,previous['nodes'],nodes,previous['mask'],mask,hh)
                for e in edges:
                    source_links.append(dict(previous=fi-1,current=fi,pairs=[[int(previous['nodes']['ids'][aa]),int(nodes['ids'][bb])]for aa,bb in e['pairs']],
                        unambiguous=e['unambiguous'],source_only=True,fragment_edge=dict(e),node_index_domain='original actual_RGB candidate IDs'))
            link_records.append(dict(previous_pts=previous['pts'],current_pts=pts,navigation_status=nav_reason,
                H_current_from_previous=None if hh is None else hh.tolist(),actual_RGB_fragment_edges=edges,H_or_flow_UV_is_measurement=False))
        data.append(dict(note,measurement=measured));previous=dict(pts=pts,rgb=rgb,mask=mask,nodes=nodes)
        if fi%50==0:print(json.dumps(dict(stage='RGB_read',frame=fi+1,pts=pts,candidates=note['actual_RGB_candidate_count'],seconds=time.monotonic()-started)),flush=True)
    counts=[0 if f['measurement']is None else len(f['measurement']['candidate_uv'])for f in data]
    offsets,components,labels=propagate_readout_identities(counts,source_links,anchors,qualified_groups=whole_groups);by_component={l['component']:l for l in labels}
    write(folder/'source_links.json',dict(adjacent_links=link_records,readout_links=source_links,source_only=True))
    write(folder/'source_labels.json',dict(anchors=anchors,whole_source_groups=whole_groups,labels=labels,candidate_component=components,frame_offsets=offsets,
        identity_from_H_or_curve_names=False))
    identity_proof=dict(source_links=str(folder/'source_links.json'),source_links_sha256=digest(folder/'source_links.json'),
        source_labels=str(folder/'source_labels.json'),source_labels_sha256=digest(folder/'source_labels.json'),anchor_approval=approval,source_graph=graph_note)
    selected_frames=[];summaries=[];all_reasons=Counter()
    with (folder/'candidate_source_rows.jsonl').open('x',encoding='utf8')as rawout,(folder/'all_candidate_rows.jsonl').open('x',encoding='utf8')as allout, (folder/'source_readout_reports.jsonl').open('x',encoding='utf8')as reports, (qualified/SID/'source_rows.jsonl').open('x',encoding='utf8')as selected:
        for fi,frame in enumerate(data):
            measured=frame['measurement'];all_rows=[];eligible=[];rows=[]
            if measured is not None:
                ids={ci for g in whole_groups if g['frame']==fi for ci in g['candidate_ids']}
                for ci,uv in enumerate(measured['candidate_uv']):
                    qi=int(measured['candidate_query'][ci]);label=by_component.get(int(components[offsets[fi]+ci]));reasons=[]
                    if int(measured['candidate_flags'][ci])&1:reasons.append('public_actual_source_profile_truncated_or_reversed')
                    if int(measured['candidate_flags'][ci])&2:reasons.append('public_source_exclusion_intersection')
                    if ci not in ids:reasons.append('not_selected_by_public_whole_paired_source_line')
                    if not label or label['status']!='anchored' or label['identities']!=[IDENTITY]:reasons.append('no_unique_approved_anchor_RGB_link_identity')
                    row=dict(path_id=IDENTITY,entity=paths[0]['curve_id'],curve=0,uv=uv.tolist(),normal=measured['query_normal'][qi].tolist(),material_signal='red',
                        actual_source_sides=measured['candidate_sides'][ci].tolist(),actual_source_quantiles=measured['candidate_side_quantiles_uv'][ci].tolist(),
                        measurement_operator='paired_material_boundary_midline',model_UV_synthesized=False,source_candidate_id=ci,source_query_id=qi,
                        actual_candidate_flags=int(measured['candidate_flags'][ci]),source_profile_proof=frame['actual_source_measurements'],
                        pending_reasons=reasons,machine_source_qualified=not reasons,
                        source_identity_basis='approved finite original-source span + unique actual bidirectional RGB links + unchanged public whole-source paired-line organization')
                    all_rows.append(row);all_reasons.update(reasons)
                    if not reasons:eligible.append(ci)
                for at in spaced_indices(measured['candidate_uv'][eligible],spacing=48.):rows.append(all_rows[eligible[at]])
            public={k:v for k,v in frame.items()if k!='measurement'}
            reports.write(json.dumps(plain(public),ensure_ascii=False)+'\n')
            allout.write(json.dumps(plain(dict(pts=frame['pts'],source_BGR_sha256=frame['source_BGR_sha256'],all_actual_RGB_candidates=all_rows,fit_eligible=False)),ensure_ascii=False)+'\n')
            rawout.write(json.dumps(plain(dict(public,rows=all_rows,fit_eligible=False,review_state='all_actual_source_candidates_preserved')),ensure_ascii=False)+'\n')
            record=dict(public,rows=rows,fit_eligible=bool(rows),source_identity_proof=identity_proof,
                review_state='approved_finite_span_and_actual_RGB_links_machine_qualified'if rows else'pending_no_supported_source_span',
                qualification=dict(source_capture=str(folder/'candidate_source_rows.jsonl'),actual_UV_unchanged=True,native_residual_used=False,
                    source_image_identity='source-hashed approved finite span and actual RGB links',public_thresholds_changed=False,human_accuracy_claim=False))
            selected.write(json.dumps(plain(record),ensure_ascii=False)+'\n')
            if rows:selected_frames.append(fi)
            summary=dict(pts=frame['pts'],rows=len(rows),actual_RGB_candidates=len(all_rows),whole_source_selected=len(eligible),
                source_navigation=frame['navigation_status'],pending_reasons=frame['pending_reasons']);summaries.append(summary)
            data[fi]=dict(pts=frame['pts'],rows=rows,guide=frame['finite_source_guide'])
    pose_source=root/SID/'navigation/poses_reanchored.npz';shutil.copyfile(pose_source,qualified/SID/'poses.npz')
    qualified_binding=dict(binding,fit_eligible=True,source_capture=str(capture),source_capture_binding_sha256=digest(capture/'binding.json'),
        qualification_method='unchanged finite-span original RGB paired-boundary whole-line and unique actual RGB node links to approved contour',
        source_identity_proof=identity_proof,pose_initialization_source=str(pose_source),pose_initialization_source_sha256=digest(pose_source))
    write(qualified/'binding.json',qualified_binding)
    proof_indices=set(range(0,len(data),50))|{next(i for i,f in enumerate(data)if f['pts']==REFERENCE_PTS)}
    if selected_frames:proof_indices|={selected_frames[0],selected_frames[-1]}
    proof_images=exact_frames(source,[data[i]['pts']for i in sorted(proof_indices)])
    proof_paths=[]
    for fi in sorted(proof_indices):
        f=data[fi];proof=proof_images[f['pts']].copy()
        if f['guide']is not None:cv2.polylines(proof,[np.rint(f['guide']).astype('int32')],False,(0,220,220),1,cv2.LINE_AA)
        for row in f['rows']:
            for side,c in zip(np.array(row['actual_source_quantiles']),[(255,100,0),(255,0,255)]):
                cv2.line(proof,tuple(np.rint(side[0]).astype(int)),tuple(np.rint(side[2]).astype(int)),c,1,cv2.LINE_AA)
            cv2.circle(proof,tuple(np.rint(row['uv']).astype(int)),4,(0,255,0),1)
        cv2.putText(proof,f'PTS {f["pts"]} | finite approved span | {len(f["rows"])} actual source-linked points | NO FIT',(20,35),0,.65,(0,220,255),2)
        path=folder/f'{f["pts"]}_source_proof.jpg';cv2.imwrite(str(path),proof);proof_paths.append(str(path))
    packet=build(root,qualified,out/'source_packets',shot_ids=[SID])
    missing=[];run=0;maximum=0
    for s in summaries:
        if not s['rows']:missing.append(s['pts']);run+=1;maximum=max(maximum,run)
        else:run=0
    report=dict(binding,complete=True,native_frames=len(summaries),expected_native_frames=shot['processing']['native_frame_count'],
        source_graph_connected_frames=sum(s['source_navigation']!='unsupported_source_graph_interval'for s in summaries),
        qualified_frames=len(selected_frames),qualified_rows=sum(s['rows']for s in summaries),actual_RGB_candidates=sum(counts),
        whole_source_qualified_groups=len(whole_groups),actual_RGB_link_edges=len(source_links),anchor_nodes=len(anchors),
        pending_native_PTS=missing,longest_no_qualified_run_frames=maximum,longest_no_qualified_run_seconds=maximum*.02,
        all_candidate_pending_reasons=dict(all_reasons),source_rows=str(qualified/SID/'source_rows.jsonl'),
        source_packets=str(out/'source_packets'/SID),packet_manifest=str(out/'source_packets/packets_manifest.json'),
        old_same_shot_PTS_path_group_upsert_replaces_not_appends=True,group_key=[SID,'native PTS',IDENTITY],
        actual_source_UV_unchanged=True,original_points_or_fit_state_modified=False,source_identity_proof=identity_proof,
        representative_proof_images=proof_paths,proof_frame_indices=sorted(proof_indices),frames=summaries,seconds=time.monotonic()-started)
    write(out/'report.json',report)
    print(json.dumps(dict(output=str(out),native_frames=report['native_frames'],qualified_frames=report['qualified_frames'],qualified_rows=report['qualified_rows'],
        source_packet=report['source_packets'],source_graph_connected_frames=report['source_graph_connected_frames'],seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()