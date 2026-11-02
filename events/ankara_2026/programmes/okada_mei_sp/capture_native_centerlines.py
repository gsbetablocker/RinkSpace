"""Read native painted midlines from source RGB and SAM2 masks.

The native scene supplies identities/search geometry. Paired source boundaries,
whole visible line support and source-image temporal links supply measured UV.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import os
import cv2
import numpy as np
from scipy.spatial import cKDTree
import run_native_programme as run
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.review.scene import sample_native_review_scene
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader, straight_ice_material_signal
from rinkspace.session.material_tracks import (qualify_line_candidates, source_line_fragments,
    source_nodes, link_fragments, qualify_visible_line_fragments, source_line_reference_support,
    assemble_fragments, source_line_reference_conflict, split_visible_source_arcs)
from rinkspace.session.native_shape_guidance import point_to_polyline


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def source_ice_exclusion(rgb, masks):
    if masks.get('ice') is not None:
        return ~masks['ice'], dict(method='existing_same_exposure_source_surface',
            model_projection_used_for_mask=False)
    if masks.get('unoccluded_source') is None:
        raise ValueError('Source ice search needs an actual same-exposure exclusion mask')
    from capture_independent_ice import ice_source_mask
    invalid, receipt = ice_source_mask(rgb, ~masks['unoccluded_source'])
    return invalid, dict(method='existing_original_RGB_ice_source_mask_fallback',
        implementation=str(Path(ice_source_mask.__code__.co_filename).resolve()),
        implementation_sha256=run.digest(ice_source_mask.__code__.co_filename),
        original_RGB_floor_guard=receipt, model_projection_used_for_mask=False,
        mask_is_search_navigation_not_observation=True)


def qualify_painted_paths(measurement, paths, invalid):
    """Preserve partial circle arcs using their known shape, never generated UV."""
    lines=[i for i,p in enumerate(paths) if not p['closed']]
    fragments,reports=(qualify_visible_line_fragments(measurement,lines,invalid) if lines else
        ([],dict(whole_line_reports=[],source_uv_unchanged=True)))
    # Disconnected material also occurs along a straight search: a house arc
    # can occupy one span while the real line occupies another. Matching the
    # real span must not label the other span through shared fragment identity.
    fragments=[piece for fragment in fragments
               for piece in split_visible_source_arcs(measurement,fragment)]
    closed=[i for i,p in enumerate(paths) if p['closed']]
    if not closed:return fragments,reports
    from rinkspace.session.source_shape_consensus import known_shape_profile_hypotheses
    from rinkspace.session.source_shape_context import measured_band_unmasked
    eligible=measured_band_unmasked(measurement['candidate_side_quantiles_uv'],invalid)
    checked=dict(measurement,candidate_flags=measurement['candidate_flags'].copy())
    checked['candidate_flags'][~eligible] |= 2
    proposals=assemble_fragments(checked,paths)
    query=measurement['candidate_query'];uv=measurement['candidate_uv']
    shape_reports=[]
    for ci in closed:
        hypotheses=known_shape_profile_hypotheses(checked,[ci],proposals,midline_tolerance_px=3.)
        plausible=[h for h in hypotheses if h['score']>=.85*hypotheses[0]['score']] if hypotheses else []
        common=set.intersection(*(set(h['candidate_ids'])for h in plausible)) if plausible else set()
        supported=set.union(*(set(h['candidate_ids'])for h in plausible)) if plausible else set()
        shape_reports.append(dict(curve=ci,hypotheses=hypotheses,plausible_alternatives=len(plausible),
            consensus_candidate_ids=sorted(common),alternative_candidate_ids=sorted(supported),
            identity_from_query_name=False,source_UV_unchanged=True))
        # Several real material branches can explain a partial circle. Taking
        # their global intersection used to erase almost every visible readout.
        # Keep the disjoint RGB-organized branches as alternatives. Recognition
        # and temporal correspondence must still establish their identity;
        # neither membership in this union nor the query name activates a row.
        for proposal in proposals:
            if proposal['curve']!=ci:continue
            ids=sorted(set(proposal['candidate_ids'])&supported,key=lambda i:int(query[i]))
            if not ids:continue
            pieces=[]
            for cid in ids:
                if not pieces or query[cid]-query[pieces[-1][-1]]!=1:pieces.append([])
                pieces[-1].append(cid)
            span=sum(float(np.linalg.norm(np.diff(uv[p],axis=0),axis=1).sum()) for p in pieces if len(p)>1)
            circle_fragment=dict(curve=ci,candidate_ids=ids,source_visible_candidate_pieces=pieces,
                span_px=span,query_first=int(query[ids[0]]),query_last=int(query[ids[-1]]),
                physical_identity=None,organization='separate_known_shape_source_arc_alternatives',
                shape_alternatives=[j for j,h in enumerate(plausible) if set(ids)&set(h['candidate_ids'])],
                unseen_source_pixels_created=False)
            fragments.extend(split_visible_source_arcs(measurement,circle_fragment))
    return fragments,dict(reports,closed_paint_shape_reports=shape_reports)


def capture(args):
    out=args.output
    if out.exists(): raise FileExistsError(out)
    cached=getattr(args,'reuse_source',None)
    reorganize=bool(getattr(args,'reorganize_source',False))
    if reorganize and cached is None:raise ValueError('Source reorganization requires frozen source readouts')
    requested_path_ids=getattr(args,'path_ids',None)
    profile_backend=getattr(args,'profile_backend','cpu')
    identity_reference=run.read(args.source_identity_reference) if getattr(args,'source_identity_reference',None) else None
    identity_curves={} if identity_reference is None else {r['native_path_id']:r for r in identity_reference['curves']}
    from native_source_mask_selection import load_selections, selected_frame
    mask_selections,mask_selection_ref=load_selections(
        getattr(args,'source_mask_selections',None),getattr(args,'bootstrap',None))
    m=run.read(args.run_root/'joint_manifest.json'); v=dict(np.load(args.state))
    mapping=run.read(args.run_root/'mapping.json'); old=dict(np.load(args.run_root/'point_observations.npz'))
    scene_path=Path(m['scene_path']); scene=run.read(scene_path)
    model=SceneParameterModel(scene,scene_path,optimize_scene=True)
    native_library=Path(getattr(args,'native_library',None) or m['native_library']).resolve(strict=True)
    core=NativeCore(native_library); state=ParameterState(model.initial_state.layout,v['scene_values'])
    samples=sample_native_review_scene(scene,core,scene_path,parameter_state=state,observed_path_ids=set())
    paths=[]
    for cut in samples['cuts']:
        if cut['layer']!='ice': continue
        if requested_path_ids and cut['id'] not in requested_path_ids: continue
        name=cut['label'].removeprefix('ice.'); feature,_,role=name.partition(':')
        if args.features and feature not in args.features: continue
        xyz=samples['xyz'][cut['begin']:cut['end']]
        # The actual native primitive determines whether a whole-line reader
        # applies. Named-role lists silently omitted newly modeled short edges.
        # Check the full world path, before image clipping or projection.
        if len(xyz)<2 or not np.isfinite(xyz).all(): continue
        straight=np.linalg.matrix_rank(xyz-xyz[0])==1
        painted_circle=(feature in ('A1','A2','A3','A4','C0') and bool(cut.get('closed')))
        if not straight and not painted_circle: continue
        signal=(args.blue_signal if feature in ('B1','B2') else
            (getattr(args,'circle_signal','red') if painted_circle else straight_ice_material_signal(feature)))
        paths.append(dict(curve_id=name,feature_id=feature,world_curve_role=role,
            native_path_id=cut['id'],xyz=xyz,
            operator='paint_center',material_signal=signal,
            material_transition=None,query_sampling='continuous_arc_length',closed=painted_circle))
    if not paths: raise ValueError('No requested native centerline paths')
    if requested_path_ids:
        missing=set(requested_path_ids)-{p['native_path_id'] for p in paths}
        if missing: raise ValueError('Requested native straight paths unavailable: '+', '.join(sorted(missing)))
    timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time'])
    out.mkdir(parents=True); cv2.setNumThreads(2); started=time.monotonic()
    inventory=[{k:x for k,x in p.items() if k!='xyz'} for p in paths]
    if cached:
        parent_binding=run.read(cached/'binding.json')
        if [(p['curve_id'],p['native_path_id']) for p in parent_binding['inventory']] != [(p['curve_id'],p['native_path_id']) for p in inventory]:
            raise ValueError('Frozen source query inventory differs')
    run.write_json(out/'binding.json',dict(state=str(args.state),state_sha256=run.digest(args.state),
        input_sha256=m['point_input_sha256'],native_library=str(native_library),
        native_library_sha256=run.digest(native_library),
        parent_native_library_sha256=m['native_library_sha256'],
        features=args.features,requested_path_ids=requested_path_ids,shots=args.shots,pilot_pts=args.pts,inventory=inventory,
        query_spacing_px=24.,search_radius_px=args.search_radius,blue_signal=args.blue_signal,
        circle_signal=getattr(args,'circle_signal','red'),
        reused_actual_source=None if cached is None else str(cached.resolve()),
        reused_capture_binding_sha256=None if cached is None else run.digest(cached/'binding.json'),
        reorganized_frozen_readouts=reorganize,
        profile_backend=profile_backend,source_UV_synthesized=False,
        explicit_source_mask_selections=mask_selection_ref,
        missing_ice_surface_fallback='capture_independent_ice.ice_source_mask(original_RGB, source_exclusion)',
        fit_eligible=False,fit_executed=False,
        source_identity_reference=None if identity_reference is None else dict(path=str(args.source_identity_reference),sha256=run.digest(args.source_identity_reference),user_identity_confirmed=identity_reference.get('user_identity_confirmed'),guide_is_not_measured_UV=True),adapter_sha256=run.digest(__file__)))
    print('SOURCE LOG: '+str(out/'capture.jsonl'),flush=True)
    with (out/'capture.jsonl').open('w',encoding='utf-8',buffering=1) as log:
      for sid in args.shots:
        shot=next(s for s in m['shots'] if s['id']==sid)
        if sid in ('s07','s08','s09'):
            from programme_source_frames import iter_registration_source_frames
            bootstrap=getattr(args,'bootstrap',None)
            if bootstrap is None:raise ValueError('S07-S09 require the authenticated source bootstrap')
            source=run.read(bootstrap/'index.json')['source']
            vis=dict(source_video_sha256=source['sha256'])
            frames=iter_registration_source_frames(sid,bootstrap_root=bootstrap,selected_pts=args.pts)
            scope=range(shot['first_pts'],shot['last_pts']+1,256)
            total_frames=sum(args.pts is None or pt in args.pts for pt in scope)
        else:
            vis,frames=run.source_frames(sid,selected_pts=args.pts);total_frames=len(frames)
        folder=out/sid; folder.mkdir(); parents={}; payload={}; seeds=set(); links=[]; previous=None;last_supported={}
        source_conflicts=[]
        old_trees={}
        for ci,path in enumerate(paths):
            index=mapping['path_ids'].index(path['native_path_id'])
            rows=np.flatnonzero((old['path_indices']==index)&(old['pts']>=shot['first_pts'])&(old['pts']<=shot['last_pts']))
            for pt in np.unique(old['pts'][rows]): old_trees[(ci,int(pt))]=cKDTree(old['uv'][rows[old['pts'][rows]==pt]])
        reader=IceProgrammeSourceReader(None,source_sha256=vis['source_video_sha256'],time_base='1/12800',
            native_paths=paths,sample_spacing_px=24.,search_radius_px=args.search_radius,separate_bands=True,
            profile_backend=profile_backend,
            prepared_source_context=dict(source_sha256=vis['source_video_sha256'],time_base='1/12800',
                authority=[dict(path=m['scene_path'],sha256=m['scene_file_sha256'])]))
        def find(key):
            while parents[key]!=key: parents[key]=parents[parents[key]];key=parents[key]
            return key
        counts={p['curve_id']:dict(read_frames=0,source_rows=0,seed_frames=0) for p in paths}
        for fi,frame in enumerate(frames):
            frame=selected_frame(frame,mask_selections.get((sid,frame['pts'])))
            pts=frame['pts']; clock=(pts-shot['reference_pts'])/12800+timeline.offset(shot)
            projections={}; curves=[];native_navigation_masks=[]
            for path in paths:
                result=core.project_motion(v['rig'][shot['camera_index']],v['controls'],timeline.first,timeline.dt,1080,clock,path['xyz'])
                query_uv=result['uv'];query_valid=result['status']==0
                if identity_reference is not None and identity_reference['guide_scope'][0]<=pts<=identity_reference['guide_scope'][1] and path['native_path_id'] in identity_curves:
                    guide=np.asarray(identity_curves[path['native_path_id']]['source_search_polyline'],float);distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(guide,axis=0),axis=1))];along=np.linspace(0.,distance[-1],len(path['xyz']));query_uv=np.c_[np.interp(along,distance,guide[:,0]),np.interp(along,distance,guide[:,1])];query_valid=np.isfinite(query_uv).all(1)
                projections[path['native_path_id']]=dict(uv=query_uv,valid=query_valid)
                curves.append(result['uv']);native_navigation_masks.append(result['status']==0)
            unified=isinstance(frame['image'],np.ndarray)
            image=frame['image'] if unified else cv2.imread(frame['image'])
            if hashlib.sha256(image.tobytes()).hexdigest()!=frame['source_BGR_sha256']: raise ValueError('Source pixels changed')
            rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            if unified:
                masks=dict(unoccluded_source=~frame['source_exclusion_mask'],ice=None)
                mask_sources=frame['mask_sources']
                frame={k:x for k,x in frame.items()if k not in ('image','source_exclusion_mask')}
                frame.update(image=frame.get('image_path'),source_video=source['resolved_path'])
            else:masks,mask_sources=run.surfaces(frame,image)
            invalid,mask_receipt=source_ice_exclusion(rgb,masks)
            mask_receipt.update(source_dependencies=mask_sources,
                applied_exclusion_sha256=hashlib.sha256(np.asarray(invalid,bool).tobytes()).hexdigest())
            if frame.get('source_mask_selection') is not None:
                mask_receipt['explicit_source_mask_selection']=frame['source_mask_selection']
            if cached:
                raw=cached/sid/f'{pts}.npz';z=dict(np.load(raw));record=run.read(cached/sid/f'{pts}.json')
                fragments,reports=record['fragments'],record['reports']
                if record['source_ice_search_mask']['applied_exclusion_sha256']!=mask_receipt['applied_exclusion_sha256']:
                    raise ValueError('Reused source readout mask differs at the same exposure')
                if reorganize:fragments,reports=qualify_painted_paths(z,paths,invalid)
            else:
                z=reader.read(rgb,pts,None,invalid,shot_id=sid,native_projections=projections)
                fragments,reports=qualify_painted_paths(z,paths,invalid)
            stem=folder/f'{pts}'
            if not cached:
                raw=stem.with_suffix('.npz')
                np.savez_compressed(raw,**{k:x for k,x in z.items() if isinstance(x,np.ndarray)})
                run.write_json(stem.with_suffix('.json'),plain(dict(reports=reports,fragments=fragments,
                    source_ice_search_mask=mask_receipt)))
            else:
                # Frozen readouts are shared, not regenerated or copied into
                # new pixel values. Both archive names bind the same bytes.
                os.link(raw,stem.with_suffix('.npz'))
                if reorganize:
                    run.write_json(stem.with_suffix('.json'),plain(dict(reports=reports,fragments=fragments,
                        source_ice_search_mask=mask_receipt,reorganized_source_UV_unchanged=True,
                        parent_frame_metadata=str(raw.with_suffix('.json')),
                        parent_frame_metadata_sha256=run.digest(raw.with_suffix('.json')))))
                else:os.link(raw.with_suffix('.json'),stem.with_suffix('.json'))
                raw=stem.with_suffix('.npz')
            conflicted=set();frame_seed_curves=set()
            for j,fragment in enumerate(fragments):
                ci=fragment['curve']; ids=np.asarray(fragment['candidate_ids'],int); key=(pts,j); parents[key]=key
                tree=old_trees.get((ci,pts))
                if not paths[ci]['closed'] and tree is not None:
                    conflict=source_line_reference_conflict(tree.data,z['candidate_uv'][ids],
                        z['candidate_side_quantiles_uv'][ids],known_straight_extension=True)
                    if conflict['contradicted']:
                        conflicted.add(j)
                        source_conflicts.append(dict(pts=pts,fragment=j,curve=ci,**conflict))
                support=source_line_reference_support(np.empty((0,2)) if tree is None else tree.data,
                    z['candidate_uv'][ids],z['candidate_query'][ids])
                if support['qualified'] and j not in conflicted: seeds.add(key);frame_seed_curves.add(ci)
                if identity_reference is not None and pts==identity_reference['reference_pts'] and paths[ci]['native_path_id'] in identity_curves:
                    if identity_reference.get('user_identity_confirmed') is not True or frame['source_BGR_sha256']!=identity_reference['reference_native_BGR_sha256'] or vis['source_video_sha256']!=identity_reference['source_video_sha256']:raise ValueError('User identity reference source binding changed')
                    guide=np.asarray(identity_curves[paths[ci]['native_path_id']]['source_search_polyline'],float);q=z['candidate_uv'][ids];delta=guide[-1]-guide[0];fraction=np.clip(((q-guide[0])@delta)/(delta@delta),0.,1.);distance=np.linalg.norm(q-(guide[0]+fraction[:,None]*delta),axis=1);inside=distance<=identity_reference['identity_seed_band_px']
                    if inside.all() and len(ids) and j not in conflicted:seeds.add(key);frame_seed_curves.add(ci)
                payload[key]=(frame,ids,raw,ci)
                counts[paths[ci]['curve_id']]['source_rows']+=len(ids)
            for ci in {f['curve']for f in fragments}:counts[paths[ci]['curve_id']]['read_frames']+=1
            for ci in frame_seed_curves:counts[paths[ci]['curve_id']]['seed_frames']+=1
            nodes=source_nodes(rgb,z,fragments,paths)
            world_projection=np.vstack(curves)
            navigation_valid=np.concatenate(native_navigation_masks)
            navigation_valid &= np.isfinite(world_projection).all(1)
            navigation_valid &= (world_projection>[-1920,-1080]).all(1)&(world_projection<[3840,2160]).all(1)
            references={}
            if previous is not None:references[previous['pts']]=previous
            for ci in {f['curve']for f in fragments}:
                if ci in last_supported:references[last_supported[ci]['pts']]=last_supported[ci]
            for reference in references.values():
              if len(nodes['uv']) and len(reference['nodes']['uv']):
                common=navigation_valid&reference['navigation_valid']
                h=None
                if common.sum()>=4:h,_=cv2.findHomography(reference['projection'][common],world_projection[common],0)
                if h is not None and np.isfinite(h).all() and abs(np.linalg.det(h))>1e-12:
                    for edge in link_fragments(reference['rgb'],rgb,reference['nodes'],nodes,reference['mask'],invalid,h):
                        if fragments[edge['b']]['curve']!=reference['fragments'][edge['a']]['curve']: continue
                        if edge['b'] in conflicted or edge['a'] in reference['conflicted']:
                            edge=dict(edge,unambiguous=False,source_identity_conflict=True)
                        a=(reference['pts'],edge['a']);b=(pts,edge['b'])
                        if edge['unambiguous']: parents[find(a)]=find(b)
                        links.append(dict(from_pts=reference['pts'],to_pts=pts,
                            crosses_missing_readout=pts-reference['pts']>256,**edge))
            previous=dict(pts=pts,rgb=rgb,nodes=nodes,mask=invalid,projection=world_projection,fragments=fragments,navigation_valid=navigation_valid,conflicted=conflicted)
            for ci in {f['curve']for f in fragments}:last_supported[ci]=previous
            if args.pts or fi in (0,total_frames//2,total_frames-1):
                proof=image.copy()
                for fragment in fragments:
                    ci=fragment['curve']; color=(255,80,0) if paths[ci]['feature_id'] in ('B1','B2') else (0,220,50)
                    for cid in fragment['candidate_ids']: cv2.circle(proof,tuple(np.rint(z['candidate_uv'][cid]).astype(int)),3,color,-1)
                    at=np.rint(z['candidate_uv'][fragment['candidate_ids'][len(fragment['candidate_ids'])//2]]).astype(int)
                    cv2.putText(proof,paths[ci]['curve_id'],tuple(at+[0,-12]),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1,cv2.LINE_AA)
                cv2.imwrite(str(folder/f'{pts}_source.jpg'),proof)
            if fi%25==0 or fi==total_frames-1:
                row=dict(shot=sid,frames=fi+1,total=total_frames,fragments=len(parents),seed_fragments=len(seeds),source_links=len(links),elapsed_s=time.monotonic()-started)
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
        connected={find(k) for k in seeds}; accepted=[]; pending=[]
        for key,(frame,ids,raw,ci) in payload.items():
            z=np.load(raw); target=accepted if find(key) in connected else pending
            for cid in ids:
                target.append(dict(pts=key[0],identity=paths[ci]['curve_id'],path_id=paths[ci]['native_path_id'],
                    candidate=int(cid),uv=z['candidate_uv'][cid].tolist(),actual_source_sides=z['candidate_sides'][cid].tolist(),
                    actual_source_quantiles=z['candidate_side_quantiles_uv'][cid].tolist(),fragment=key[1],
                    evidence_image=frame['image'],source_BGR_sha256=frame['source_BGR_sha256'],source_reference=str(raw),
                    source_video=frame.get('source_video'),source_video_sha256=vis['source_video_sha256'],
                    mask_path=frame['mask_path'],mask_sha256=frame['mask_sha256'],model_UV_synthesized=False,
                    measurement_operator='paired_material_boundary_midline'))
        source=folder/'source_pixels.json';linkfile=folder/'source_links.json'
        run.write_json(source,dict(shot=sid,accepted=accepted,pending=pending,source_UV_changed=False))
        run.write_json(linkfile,dict(edges=plain(links),seed_fragments=sorted(seeds),source_identity_is_assistant_inference=identity_reference is None,
            user_identity_reference=None if identity_reference is None else dict(path=str(args.source_identity_reference),sha256=run.digest(args.source_identity_reference),user_identity_confirmed=True,guide_UV_not_used_as_measurements=True),
            same_exposure_source_identity_conflicts=source_conflicts,
            seed_matching_rule='Distinct old UV within 3px of finite source segments joining consecutive queries only',
            required_distinct_old_seed_uv=3,required_matched_old_seed_uv_span_px=56.))
        run.write_json(folder/'source_verified.json',dict(shot=sid,accepted=accepted,source_packet=str(source),
            source_packet_sha256=run.digest(source),source_links=str(linkfile),source_links_sha256=run.digest(linkfile),
            fit_eligible=False,review_state='awaiting_source_identity_and_pixel_review',user_visual_acceptance=False))
        run.write_json(folder/'coverage.json',dict(by_path=counts,accepted_rows=len(accepted),pending_rows=len(pending)))
        print(json.dumps(dict(shot=sid,by_path=counts,accepted_rows=len(accepted),pending_rows=len(pending))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('run-root','state','output'): parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--shots',nargs='+',required=True)
    parser.add_argument('--features',nargs='+',help='Optional feature filter; default reads straight ice paths and A1-A4/C0 painted circles')
    parser.add_argument('--path-ids',nargs='+',help='Optional exact native path IDs; default keeps every requested feature path')
    parser.add_argument('--pts',nargs='+',type=int)
    parser.add_argument('--search-radius',type=float,default=48.)
    parser.add_argument('--blue-signal',choices=['blue','dark'],default='blue')
    parser.add_argument('--circle-signal',choices=['red','dark'],default='red')
    parser.add_argument('--profile-backend',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--reuse-source',type=Path)
    parser.add_argument('--reorganize-source',action='store_true',
        help='Rebuild source fragments from reused exact RGB readouts; preserve raw NPZ and original metadata')
    parser.add_argument('--bootstrap',type=Path,help='Authenticated original video and SAM2 masks for S07-S09')
    parser.add_argument('--native-library',type=Path,help='Explicit library for search projection; parent state provenance stays unchanged')
    parser.add_argument('--source-mask-selections',type=Path,
        help='Explicit approved same-RGB search mask selections; all other exposures keep the default mask')
    parser.add_argument('--source-identity-reference',type=Path,help='User-confirmed feature identities with source-image search guides; guides never become measured UV')
    capture(parser.parse_args())
