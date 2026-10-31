"""Five-frame finite approved-source-span ice prototype; no fit or native state.

H transports only the observed reference polyline. Public RGB readers supply
all midlines and both material boundaries. Whole-source qualification and real
bidirectional RGB profile links establish machine identity; pilot stays pending.
"""
from pathlib import Path
import argparse,hashlib,json,time
import cv2
import numpy as np

from rinkspace.geometry import project
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader, _batched_actual_profiles
from rinkspace.vision_source_shapes import _MaterialProfileFrame
from rinkspace.session.material_tracks import (qualify_visible_line_fragments,source_nodes,
    source_line_reference_support,link_fragments,propagate_readout_identities)
from capture_independent_badge import exact_frames
from capture_independent_ice import (approval_sources,anchor_records,load_navigation_graph,
    navigation_between,ice_source_mask)
from capture_independent_materials import native_ice_inventory,source_exclusion,spaced_indices,plain,write

PTS = [1805568,1805824,1806080,1806336,1806592]
SID = 's08'
IDENTITY = 'demo_initial/goal_line_left:0'
REFERENCE_PTS = 1806080


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def finite_source_span(uv, h):
    """Transfer exact ordered observed vertices; do not fit or extend endpoints."""
    uv=np.asarray(uv,float)
    if uv.ndim!=2 or uv.shape[1]!=2 or len(uv)<2 or not np.isfinite(uv).all():
        raise ValueError('Finite original source polyline vertices required')
    mapped=project(h,uv)
    if not np.isfinite(mapped).all():raise ValueError('Invalid source navigation of observed span')
    return mapped


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();output=a.output.resolve()
    if output.exists():raise FileExistsError(output)
    cv2.setNumThreads(2);started=time.monotonic()
    index=json.loads((root/'index.json').read_text('utf8'));source_sha=index['source']['actual_sha256']
    source_video=Path(index['source']['resolved_path'])
    if digest(source_video)!=source_sha:raise ValueError('Original source video hash differs')
    anchor_root=root/'source_anchor_bank';approval_file=anchor_root/'approval.json'
    declarations,approval=approval_sources(anchor_root,approval_file,[SID],source_sha)
    references=[r for r in anchor_records(declarations,source_sha)
        if r['pts']==REFERENCE_PTS and r['path_id']==IDENTITY]
    if len(references)!=1:raise ValueError('Exactly one explicitly approved finite goal-line reference is required')
    reference=references[0]
    shot=next(s for s in index['shots']if s['shot_id']==SID);graph,graph_note=load_navigation_graph(root,shot)
    world=dict(np.load(root/'native_world.npz',allow_pickle=False));cuts=json.loads((root/'native_world.json').read_text('utf8'))['cuts']
    paths=[q for q in native_ice_inventory(cuts,world['xyz'])if q['native_path_id']==IDENTITY]
    if len(paths)!=1:raise ValueError('Unique current native goal-line path metadata required')
    paths[0]=dict(paths[0],material_signal=reference['material_signal'] or 'red')
    reader=IceProgrammeSourceReader(None,source_sha256=source_sha,time_base='1/12800',native_paths=paths,
        sample_spacing_px=24.,search_radius_px=40.,separate_bands=True,
        prepared_source_context=dict(source_sha256=source_sha,time_base='1/12800',
            authority=[dict(path=reference['source_rows_file'],sha256=reference['source_rows_file_sha256']),
                       dict(path=str(root/'native_world.json'),sha256=digest(root/'native_world.json'))]))
    images=exact_frames(source_video,PTS)
    if hashlib.sha256(images[REFERENCE_PTS].tobytes()).hexdigest()!=reference['source_BGR_sha256']:
        raise ValueError('Approved same-frame anchor original RGB differs')
    output.mkdir(parents=True);folder=output/SID;folder.mkdir()
    implementation=[Path(__file__),Path(reader.read.__code__.co_filename),
        Path(qualify_visible_line_fragments.__code__.co_filename),Path(approval_sources.__code__.co_filename)]
    hashes={str(q.resolve()):digest(q)for q in implementation}
    binding=dict(schema='rinkspace.finite_approved_source_span_pilot.v1',source_video=str(source_video),
        source_video_sha256=source_sha,source_index_sha256=digest(root/'index.json'),
        independent_source_capture=True,legacy_new_shot_inputs_read=False,native_state_loaded=False,
        source_span_reference=reference['source_rows_file'],source_span_reference_sha256=reference['source_rows_file_sha256'],
        anchor_approval=approval,source_graph=graph_note,reference_pts=REFERENCE_PTS,path_id=IDENTITY,
        reference_UV=reference['uv'].tolist(),source_span_extended=False,
        H_is_navigation_only=True,source_RGB_supplies_all_UV=True,source_query_spacing_px=24.,
        final_source_spacing_px=48.,short_feature_minimum_samples=4,search_radius_px=40.,
        source_reference_incoming_navigation_is_not_same_frame_validity=True,
        implementation_sha256=hashes,fit_eligible=False,pilot_requires_primary_source_review=True,
        public_qualification_thresholds_changed=False,solver_started=False)
    write(output/'binding.json',binding);frame_data=[];whole_groups=[];anchors=[];source_links=[];links_records=[]
    for fi,pts in enumerate(PTS):
        image=images[pts];rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask,mask_path=source_exclusion(root,SID,pts);mask,floor=ice_source_mask(rgb,mask)
        h,reason=navigation_between(graph,REFERENCE_PTS,pts)
        incoming=None if graph is None else bool(graph['supported'][graph['index'][pts]])
        stem=folder/str(pts);original=folder/f'{pts}_source.png';cv2.imwrite(str(original),image)
        cv2.imwrite(str(folder/f'{pts}_applied_exclusion.png'),mask.astype('uint8')*255)
        note=dict(pts=pts,navigation_status=reason,incoming_navigation_supported=incoming,
            source_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),source_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            original_source=str(original),original_file_sha256=digest(original),mask_path=str(mask_path),
            mask_sha256=digest(mask_path),ice_floor_receipt=floor,applied_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
            fit_eligible=False)
        if h is None:
            frame_data.append(dict(note,rgb=rgb,mask=mask,fragments=[],measurement=None,nodes=None,
                guide=None,pending_reasons=['unsupported_source_graph_interval'],qualification=None))
            continue
        guide=finite_source_span(reference['uv'],h)
        measured=reader.read(rgb,pts,None,mask,shot_id=SID,native_projections={IDENTITY:dict(uv=guide,valid=np.ones(len(guide),bool))})
        fragments,qualification=qualify_visible_line_fragments(measured,[0],mask)
        nodes=source_nodes(rgb,measured,fragments,paths)
        profiles=_batched_actual_profiles(_MaterialProfileFrame(np.ascontiguousarray(rgb),dict(source_sha256=source_sha,
            source_pts=pts,time_base='1/12800',shot_id=SID)),measured['query_uv'],measured['query_normal'],
            measured['query_interval'],np.full(len(measured['query_uv']),'red'))
        arrays={k:v for k,v in measured.items()if isinstance(v,np.ndarray)}
        arrays.update(profile_offsets=profiles[0],actual_luma_profiles=profiles[1],smoothed_luma_profiles=profiles[2],profile_valid=profiles[3],
            finite_source_guide=guide,H_current_from_reference=h,applied_exclusion=mask)
        np.savez_compressed(stem.with_suffix('.npz'),**arrays);np.savez_compressed(folder/f'{pts}_nodes.npz',**nodes)
        seed_matches=[]
        for report in qualification['whole_line_reports']:
            if report.get('source_line_support_qualified')is True:
                whole_groups.append(dict(frame=fi,candidate_ids=report['selected_candidate_ids'],source_line_support_qualified=True,
                    proof=str(stem.with_suffix('.json')),qualification='unchanged public paired-boundary whole-source-line organizer'))
        if pts==REFERENCE_PTS:
            # Same RGB anchor is valid independently of its incoming H status.
            for j,fragment in enumerate(fragments):
                ids=np.asarray(fragment['candidate_ids'],int)
                support=source_line_reference_support(reference['uv'],measured['candidate_uv'][ids],measured['candidate_query'][ids])
                seed_matches.append(dict(fragment=j,support=support))
                if support['qualified']:
                    for candidate in ids:anchors.append(dict(frame=fi,candidate=int(candidate),identity=IDENTITY,
                        source_line_support=support,approval_group=reference['approval_group'],
                        approval_sha256=reference['approval_sha256'],source_line_sha256=reference['source_line_sha256']))
        note.update(source_navigation_H=h.tolist(),finite_source_guide=guide.tolist(),
            search_guide_span_px=float(np.linalg.norm(np.diff(guide,axis=0),axis=1).sum()),
            query_count=len(measured['query_uv']),actual_RGB_candidate_count=len(measured['candidate_uv']),
            whole_source_fragments=fragments,source_qualification=qualification,anchor_matches=seed_matches,
            actual_source_measurements=str(stem.with_suffix('.npz')),actual_source_measurements_sha256=digest(stem.with_suffix('.npz')))
        write(stem.with_suffix('.json'),note)
        frame_data.append(dict(note,rgb=rgb,mask=mask,fragments=fragments,measurement=measured,nodes=nodes,guide=guide,
            pending_reasons=[] if fragments else ['no_whole_qualified_paired_source_span'],qualification=qualification))
    # Actual RGB links validate material identity. H and normal flow positions
    # remain search matches; all propagated node IDs index immutable RGB q50s.
    for fi in range(len(frame_data)-1):
        previous,current=frame_data[fi:fi+2];h,reason=navigation_between(graph,previous['pts'],current['pts'])
        edges=[]
        if h is not None and previous['nodes']is not None and current['nodes']is not None:
            edges=link_fragments(previous['rgb'],current['rgb'],previous['nodes'],current['nodes'],previous['mask'],current['mask'],h)
            for e in edges:
                pairs=[[int(previous['nodes']['ids'][first]),int(current['nodes']['ids'][last])]for first,last in e['pairs']]
                source_links.append(dict(previous=fi,current=fi+1,pairs=pairs,unambiguous=e['unambiguous'],source_only=True,
                    fragment_edge=dict(e),node_index_domain='original actual_RGB candidate IDs'))
        links_records.append(dict(previous_pts=previous['pts'],current_pts=current['pts'],navigation_status=reason,
            H_current_from_previous=None if h is None else h.tolist(),actual_RGB_fragment_edges=edges,
            H_or_profile_flow_UV_is_measurement=False))
    counts=[0 if f['measurement']is None else len(f['measurement']['candidate_uv'])for f in frame_data]
    offsets,components,labels=propagate_readout_identities(counts,source_links,anchors,qualified_groups=whole_groups)
    label_by_component={l['component']:l for l in labels}
    write(folder/'source_links.json',dict(adjacent_links=links_records,readout_links=source_links,source_only=True))
    write(folder/'source_labels.json',dict(anchors=anchors,whole_source_groups=whole_groups,labels=labels,
        candidate_component=components,frame_offsets=offsets,identity_from_H_or_curve_names=False))
    summaries=[]
    with (folder/'source_rows.jsonl').open('x',encoding='utf8')as rows_out,(folder/'all_candidate_rows.jsonl').open('x',encoding='utf8')as candidates_out:
        for fi,frame in enumerate(frame_data):
            measured=frame['measurement'];rows=[];all_rows=[];eligible=[]
            if measured is not None:
                ids=set(i for g in whole_groups if g['frame']==fi for i in g['candidate_ids'])
                for ci,uv in enumerate(measured['candidate_uv']):
                    query=int(measured['candidate_query'][ci]);label=label_by_component.get(int(components[offsets[fi]+ci]))
                    reasons=[]
                    if int(measured['candidate_flags'][ci])&1:reasons.append('public_actual_source_profile_truncated_or_reversed')
                    if int(measured['candidate_flags'][ci])&2:reasons.append('public_source_exclusion_intersection')
                    if ci not in ids:reasons.append('not_selected_by_public_whole_paired_source_line')
                    if not label or label['status']!='anchored' or label['identities']!=[IDENTITY]:reasons.append('no_unique_approved_anchor_RGB_link_identity')
                    row=dict(path_id=IDENTITY,entity=paths[0]['curve_id'],curve=0,uv=uv.tolist(),
                        material_signal='red',normal=measured['query_normal'][query].tolist(),
                        actual_source_sides=measured['candidate_sides'][ci].tolist(),
                        actual_source_quantiles=measured['candidate_side_quantiles_uv'][ci].tolist(),
                        measurement_operator='paired_material_boundary_midline',model_UV_synthesized=False,
                        source_candidate_id=ci,source_query_id=query,source_profile_proof=frame.get('actual_source_measurements'),
                        actual_candidate_flags=int(measured['candidate_flags'][ci]),pending_reasons=reasons,
                        machine_source_qualified=not reasons,fit_eligible=False,
                        source_identity_basis='approved same-frame finite source contour plus unambiguous bidirectional actual_RGB profile links and unchanged whole-source paired-boundary organization')
                    all_rows.append(row)
                    if not reasons:eligible.append(ci)
                # Public common 48px thinning preserves its existing minimum4
                # short-feature policy; it never interpolates a UV.
                for i in spaced_indices(measured['candidate_uv'][eligible],spacing=48.):
                    rows.append(all_rows[eligible[i]])
            public_record={k:v for k,v in frame.items()if k not in('rgb','mask','fragments','measurement','nodes','guide','qualification')}
            public_record.update(source_video_sha256=source_sha,rows=rows,fit_eligible=False,
                review_state='pending_primary_source_review_of_finite_span_pilot',
                machine_source_qualified_rows=len(rows),actual_UV_unchanged=True,H_UV_used_as_observation=False,
                source_identity_links=str(folder/'source_links.json'),source_identity_labels=str(folder/'source_labels.json'))
            rows_out.write(json.dumps(plain(public_record),ensure_ascii=False)+'\n')
            candidates_out.write(json.dumps(plain(dict(pts=frame['pts'],source_BGR_sha256=frame['source_BGR_sha256'],
                actual_RGB_candidates=all_rows,fit_eligible=False)),ensure_ascii=False)+'\n')
            proof=images[frame['pts']].copy()
            if frame['guide']is not None:cv2.polylines(proof,[np.rint(frame['guide']).astype('int32')],False,(0,220,220),1,cv2.LINE_AA)
            for row in all_rows:
                p=tuple(np.rint(row['uv']).astype(int));colour=(0,180,0)if row['machine_source_qualified']else(0,100,255)
                cv2.circle(proof,p,2,colour,-1)
                qq=np.array(row['actual_source_quantiles'])
                for side,c in zip(qq,[(255,100,0),(255,0,255)]):
                    if np.isfinite(side).all():cv2.line(proof,tuple(np.rint(side[0]).astype(int)),tuple(np.rint(side[2]).astype(int)),c,1,cv2.LINE_AA)
            for row in rows:cv2.circle(proof,tuple(np.rint(row['uv']).astype(int)),4,(0,255,0),1)
            cv2.putText(proof,f'PTS {frame["pts"]}: finite observed span only; {len(rows)} actual_RGB source-linked; PENDING; NO FIT',(20,35),0,.65,(0,220,255),2,cv2.LINE_AA)
            proof_path=folder/f'{frame["pts"]}_finite_span_proof.jpg';cv2.imwrite(str(proof_path),proof)
            if frame['guide']is not None:
                lo=np.maximum(np.floor(frame['guide'].min(0)-60).astype(int),[0,0]);hi=np.minimum(np.ceil(frame['guide'].max(0)+60).astype(int),[1920,1080])
                cv2.imwrite(str(folder/f'{frame["pts"]}_finite_span_crop.png'),proof[lo[1]:hi[1],lo[0]:hi[0]])
            summary=dict(pts=frame['pts'],native_source_time_seconds=frame['pts']/12800.,query_count=frame.get('query_count',0),
                actual_RGB_candidates=len(all_rows),machine_source_qualified_before_thinning=len(eligible),
                final_pending_rows=len(rows),incoming_navigation_supported=frame['incoming_navigation_supported'],
                navigation_status=frame['navigation_status'],source_profile_proof=frame.get('actual_source_measurements'),
                proof_image=str(proof_path),pending_reasons=frame['pending_reasons'])
            summaries.append(summary);print(json.dumps(plain(summary),ensure_ascii=True),flush=True)
    write(output/'report.json',dict(binding,frames=summaries,source_links=str(folder/'source_links.json'),
        source_labels=str(folder/'source_labels.json'),anchor_count=len(anchors),
        actual_RGB_link_edges=len(source_links),whole_source_qualified_groups=len(whole_groups),
        total_actual_RGB_candidates=sum(counts),total_pending_source_rows=sum(s['final_pending_rows']for s in summaries),
        seconds=time.monotonic()-started,shared_capture_file_modified=False,complete=True))


if __name__=='__main__':main()