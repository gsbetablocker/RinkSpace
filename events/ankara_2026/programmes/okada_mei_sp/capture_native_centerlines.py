"""Read complete native straight-mark families from source RGB and SAM2 masks.

The native scene supplies identities/search geometry. Paired source boundaries,
whole visible line support and source-image temporal links supply measured UV.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import cv2
import numpy as np
from scipy.spatial import cKDTree
import run_native_programme as run
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.review.scene import sample_native_review_scene
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
from rinkspace.session.material_tracks import (qualify_line_candidates, source_line_fragments,
    source_nodes, link_fragments, qualify_visible_line_fragments, source_line_reference_support)
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


def capture(args):
    out=args.output
    if out.exists(): raise FileExistsError(out)
    cached=getattr(args,'reuse_source',None)
    requested_path_ids=getattr(args,'path_ids',None)
    profile_backend=getattr(args,'profile_backend','cpu')
    m=run.read(args.run_root/'joint_manifest.json'); v=dict(np.load(args.state))
    mapping=run.read(args.run_root/'mapping.json'); old=dict(np.load(args.run_root/'point_observations.npz'))
    scene_path=Path(m['scene_path']); scene=run.read(scene_path)
    model=SceneParameterModel(scene,scene_path,optimize_scene=True)
    core=NativeCore(m['native_library']); state=ParameterState(model.initial_state.layout,v['scene_values'])
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
        if np.linalg.matrix_rank(xyz-xyz[0])!=1: continue
        paths.append(dict(curve_id=name,feature_id=feature,world_curve_role=role,
            native_path_id=cut['id'],xyz=xyz,
            operator='paint_center',material_signal=args.blue_signal if feature in ('B1','B2') else 'dark',
            material_transition=None,query_sampling='continuous_arc_length',closed=False))
    if not paths: raise ValueError('No requested native centerline paths')
    if requested_path_ids:
        missing=set(requested_path_ids)-{p['native_path_id'] for p in paths}
        if missing: raise ValueError('Requested native straight paths unavailable: '+', '.join(sorted(missing)))
    timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time'])
    out.mkdir(parents=True); cv2.setNumThreads(2); started=time.monotonic()
    inventory=[{k:x for k,x in p.items() if k!='xyz'} for p in paths]
    run.write_json(out/'binding.json',dict(state=str(args.state),state_sha256=run.digest(args.state),
        input_sha256=m['point_input_sha256'],native_library_sha256=m['native_library_sha256'],
        features=args.features,requested_path_ids=requested_path_ids,shots=args.shots,pilot_pts=args.pts,inventory=inventory,
        query_spacing_px=24.,search_radius_px=args.search_radius,blue_signal=args.blue_signal,
        profile_backend=profile_backend,source_UV_synthesized=False,
        missing_ice_surface_fallback='capture_independent_ice.ice_source_mask(original_RGB, source_exclusion)',
        fit_eligible=False,fit_executed=False,
        adapter_sha256=run.digest(__file__)))
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
        folder=out/sid; folder.mkdir(); parents={}; payload={}; seeds=set(); links=[]; previous=None
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
            pts=frame['pts']; clock=(pts-shot['reference_pts'])/12800+timeline.offset(shot)
            projections={}; curves=[]
            for path in paths:
                result=core.project_motion(v['rig'][shot['camera_index']],v['controls'],timeline.first,timeline.dt,1080,clock,path['xyz'])
                projections[path['native_path_id']]=dict(uv=result['uv'],valid=result['status']==0)
                curves.append(result['uv'])
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
            if cached:
                raw=cached/sid/f'{pts}.npz';z=dict(np.load(raw));record=run.read(cached/sid/f'{pts}.json')
                fragments,reports=record['fragments'],record['reports']
            else:
                z=reader.read(rgb,pts,None,invalid,shot_id=sid,native_projections=projections)
                fragments,reports=qualify_visible_line_fragments(z,list(range(len(paths))),invalid)
            stem=folder/f'{pts}'
            if not cached:
                raw=stem.with_suffix('.npz')
                np.savez_compressed(raw,**{k:x for k,x in z.items() if isinstance(x,np.ndarray)})
                run.write_json(stem.with_suffix('.json'),plain(dict(reports=reports,fragments=fragments,
                    source_ice_search_mask=mask_receipt)))
            for j,fragment in enumerate(fragments):
                ci=fragment['curve']; ids=np.asarray(fragment['candidate_ids'],int); key=(pts,j); parents[key]=key
                tree=old_trees.get((ci,pts))
                support=source_line_reference_support(np.empty((0,2)) if tree is None else tree.data,
                    z['candidate_uv'][ids],z['candidate_query'][ids])
                if support['qualified']: seeds.add(key); counts[paths[ci]['curve_id']]['seed_frames']+=1
                payload[key]=(frame,ids,raw,ci)
                counts[paths[ci]['curve_id']]['read_frames']+=1; counts[paths[ci]['curve_id']]['source_rows']+=len(ids)
            nodes=source_nodes(rgb,z,fragments,paths)
            world_projection=np.vstack(curves)
            navigation_valid=np.concatenate([projections[p['native_path_id']]['valid'] for p in paths])
            navigation_valid &= np.isfinite(world_projection).all(1)
            navigation_valid &= (world_projection>[-1920,-1080]).all(1)&(world_projection<[3840,2160]).all(1)
            if previous is not None and len(nodes['uv']) and len(previous['nodes']['uv']):
                common=navigation_valid&previous['navigation_valid']
                h=None
                if common.sum()>=4:h,_=cv2.findHomography(previous['projection'][common],world_projection[common],0)
                if h is not None and np.isfinite(h).all() and abs(np.linalg.det(h))>1e-12:
                    for edge in link_fragments(previous['rgb'],rgb,previous['nodes'],nodes,previous['mask'],invalid,h):
                        if fragments[edge['b']]['curve']!=previous['fragments'][edge['a']]['curve']: continue
                        a=(previous['pts'],edge['a']);b=(pts,edge['b'])
                        if edge['unambiguous']: parents[find(a)]=find(b)
                        links.append(dict(from_pts=previous['pts'],to_pts=pts,**edge))
            previous=dict(pts=pts,rgb=rgb,nodes=nodes,mask=invalid,projection=world_projection,fragments=fragments,navigation_valid=navigation_valid)
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
        run.write_json(linkfile,dict(edges=plain(links),seed_fragments=sorted(seeds),source_identity_is_assistant_inference=True,
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
    parser.add_argument('--features',nargs='+',help='Optional feature filter; default reads all native straight ice paths')
    parser.add_argument('--path-ids',nargs='+',help='Optional exact native path IDs; default keeps every requested feature path')
    parser.add_argument('--pts',nargs='+',type=int)
    parser.add_argument('--search-radius',type=float,default=48.)
    parser.add_argument('--blue-signal',choices=['blue','dark'],default='blue')
    parser.add_argument('--profile-backend',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--reuse-source',type=Path)
    parser.add_argument('--bootstrap',type=Path,help='Authenticated original video and SAM2 masks for S07-S09')
    capture(parser.parse_args())
