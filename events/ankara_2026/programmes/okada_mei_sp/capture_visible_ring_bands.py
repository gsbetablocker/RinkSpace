"""Read visible annulus arcs once and connect source fragments across a shot.

Whole rings establish anchors; an occluded/cropped exposure need not contain
an entire ellipse. All exported UV are original measured material interfaces.
"""
from pathlib import Path
import argparse, collections, hashlib, json, time
import cv2
import numpy as np
from scipy.spatial import cKDTree
import run_native_programme as run
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.compat.d78_entities import sample_entities
from rinkspace.session.material_ring_bands import read_coloured_ring_band
from rinkspace.session.material_tracks import source_nodes, link_fragments


def capture(args):
    root=args.run_root;out=args.output;out.mkdir(parents=True,exist_ok=True)
    m=run.read(root/'joint_manifest.json');v=dict(np.load(args.state));old=dict(np.load(root/'point_observations.npz'))
    mapping=run.read(root/'mapping.json');scene=run.read(m['scene_path'])
    model=SceneParameterModel(scene,Path(m['scene_path']),optimize_scene=True)
    core=NativeCore(m['native_library']);samples=sample_entities(ParameterState(model.initial_state.layout,v['scene_values']).apply_scene(scene),core)
    timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time'])
    specification=run.read(args.identities);cv2.setNumThreads(2)
    from rinkspace.session import material_rings,material_ring_bands,material_tracks,source_shape_context,native_shape_guidance
    from rinkspace import vision_programme_ice_review
    binding=dict(state=str(args.state),state_sha256=run.digest(args.state),identities=str(args.identities),
        identities_sha256=run.digest(args.identities),input_sha256=run.digest(root/'point_observations.npz'),
        module_sha256=run.digest(__file__),algorithm_sha256={str(module.__file__):run.digest(module.__file__) for module in
            (material_rings,material_ring_bands,material_tracks,vision_programme_ice_review,source_shape_context,native_shape_guidance)},spacing_px=24.,source_UV_from_RGB_only=True,
        selected_pts=getattr(args,'pts',None),candidate_only=bool(getattr(args,'candidate_only',False)),
        source_cache=str(args.source_cache)if args.source_cache else None,
        source_cache_identity_sha256=run.digest(args.source_cache/'identity.json')if args.source_cache else None,
        user_visual_acceptance=False,model_UV_is_search_only=True)
    if (out/'identity.json').exists() and run.read(out/'identity.json')!=binding:
        raise ValueError('Source run binding changed')
    run.write_json(out/'identity.json',binding)
    cache_spec=None
    if args.source_cache:
        cache_identity=run.read(args.source_cache/'identity.json')
        if cache_identity['state_sha256']!=binding['state_sha256'] or cache_identity['algorithm_sha256']!=binding['algorithm_sha256']:
            raise ValueError('Reusable source cache has different geometry or RGB reader')
        cache_spec=run.read(cache_identity['identities'])
    xx,yy=np.meshgrid(np.linspace(-.52,.52,41),np.linspace(-.27,.27,23));world=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
    started=time.monotonic();log=(out/'capture.jsonl').open('a',encoding='utf-8',buffering=1)
    print('MAIN LOG: '+str(out/'capture.jsonl'),flush=True)
    for shot_id,definitions in specification['shots'].items():
        vis,frames=run.source_frames(shot_id);shot=next(s for s in m['shots'] if s['id']==shot_id)
        adjacent={(a['pts'],b['pts'])for a,b in zip(frames,frames[1:])}
        if getattr(args,'pts',None):frames=[f for f in frames if f['pts'] in args.pts]
        folder=out/shot_id;folder.mkdir(exist_ok=True);parent={};seeded=set();payload={};edges=[];previous=None
        def find(k):
            while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
            return k
        def join(a,b):parent[find(a)]=find(b)
        cuts={};guides={};centers={};old_tree={};context_curves={};context_uv={}
        for definition in definitions:
            house=definition['house']
            cuts[house]={c['label'].split(':')[-1]:c for c in samples['cuts'] if c['label'].startswith('ice.'+house+':')}
            a,b=(samples['xyz'][cuts[house][role]['begin']:cuts[house][role]['end']] for role in ('middle','outer'))
            guides[house]=.5*(a+b);centers[house]=guides[house][:-1].mean(0)
            for path_id in definition.get('shape_context_path_ids',[]):
                cut=next(c for c in samples['cuts'] if c['id']==path_id)
                context_curves[path_id]=samples['xyz'][cut['begin']:cut['end']]
                pi=mapping['path_ids'].index(path_id)
                use=np.flatnonzero((old['path_indices']==pi)&(old['pts']>=shot['first_pts'])&(old['pts']<=shot['last_pts']))
                for pt in np.unique(old['pts'][use]):
                    context_uv[(path_id,int(pt))]=old['uv'][use[old['pts'][use]==pt]]
            for side,role in enumerate(('middle','outer')):
                pi=mapping['path_ids'].index(cuts[house][role]['id'])
                use=np.flatnonzero((old['path_indices']==pi)&(old['pts']>=shot['first_pts'])&(old['pts']<=shot['last_pts']))
                for pt in np.unique(old['pts'][use]):
                    rows=use[old['pts'][use]==pt];old_tree[(house,int(pt),side)]=cKDTree(old['uv'][rows])
        context=dict(source_sha256=vis['source_video_sha256'],time_base='1/12800',
            authority=[dict(path=str(args.identities),sha256=binding['identities_sha256'])],human_scope_ids=[])
        for fi,frame in enumerate(frames):
            pts=frame['pts'];image=cv2.imread(frame['image'])
            if hashlib.sha256(image.tobytes()).hexdigest()!=frame['source_BGR_sha256']:raise ValueError('Source pixels changed')
            rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);masks,_=run.surfaces(frame,image);invalid=~masks['ice']
            clock=(pts-shot['reference_pts'])/12800+timeline.offset(shot)
            def project(x):return core.project_motion(v['rig'][shot['camera_index']],v['controls'],timeline.first,timeline.dt,1080,clock,x)
            grid=project(world);now={};frame_receipt={};proof=image.copy()
            for definition in definitions:
                house=definition['house'];stem=folder/f'{pts}_{house}';p=project(guides[house]);p['valid']=p['status']==0
                pc=project(centers[house][None])
                search_center_available=bool(pc['status'][0]==0 and np.isfinite(pc['uv'][0]).all())
                if not search_center_available:
                    # A failed native navigation origin cannot define a radial
                    # read. An entirely disabled projection produces zero RGB
                    # queries; the finite placeholder never becomes a sample.
                    p=dict(p,valid=np.zeros(len(p['valid']),bool))
                search_center=pc['uv'][0]if search_center_available else np.zeros(2)
                if cache_spec:
                    old_definition=next((d for d in cache_spec['shots'].get(shot_id,[]) if d['house']==house),None)
                    if old_definition and all(old_definition.get(k)==definition.get(k) for k in ('house','signal','color','profile_direction','search_radius_px','query_sampling')):
                        cached=args.source_cache/shot_id/f'{pts}_{house}'
                        if cached.with_suffix('.npz').exists() and cached.with_suffix('.json').exists():stem=cached
                if stem.with_suffix('.npz').exists() and stem.with_suffix('.json').exists():
                    measured=dict(np.load(stem.with_suffix('.npz')));result=run.read(stem.with_suffix('.json'))
                else:
                    measured,result=read_coloured_ring_band(rgb,invalid,p,source_sha256=vis['source_video_sha256'],pts=pts,
                        center_uv=search_center,material_signal=definition['signal'],
                        spacing_px=24.,band_color=definition['color'],prepared_source_context=context,
                        profile_direction=definition.get('profile_direction','outward_radial'),
                        query_sampling=definition.get('query_sampling'),
                        search_radius_px=definition.get('search_radius_px',160.))
                    if not search_center_available:
                        assert not len(measured['query_uv'])
                        result['native_search_origin_unavailable']=True
                    np.savez_compressed(stem.with_suffix('.npz'),**{k:x for k,x in measured.items() if isinstance(x,np.ndarray)})
                    run.write_json(stem.with_suffix('.json'),result)
                selected=set(result['selected']);source_seeds=[]
                # A complete observed-band consensus can span several short
                # source fragments. Keep its actual readouts together; do not
                # require each local piece to independently determine an oval.
                fragments=([dict(curve=0,candidate_ids=sorted(selected),source_complete_band_consensus=True)] if len(selected)>=3 else [])
                for fragment in result['fragments']:
                    remainder=sorted(set(fragment['candidate_ids'])-selected)
                    if len(remainder)>=4:fragments.append(dict(fragment,candidate_ids=remainder))
                if definition.get('shape_context_path_ids'):
                    local_context={key:context_uv[(key,pts)]for key in definition['shape_context_path_ids']if (key,pts)in context_uv}
                    native_curves={key:project(context_curves[key])['uv']for key in local_context}
                    for role in ('middle','outer'):
                        cut=cuts[house][role]
                        native_curves[role]=project(samples['xyz'][cut['begin']:cut['end']])['uv']
                    qualified=[];shape_reports=[]
                    for fragment in fragments:
                        if not local_context or any(not np.isfinite(p).all()for p in native_curves.values()):
                            shape_reports.append(dict(source_candidates=len(fragment['candidate_ids']),supported=0,status='no_finite_same_frame_source_context'))
                            continue
                        checked=source_shape_context.assess_native_band_shape(measured,fragment['candidate_ids'],native_curves,
                            roles=('middle','outer'),context_observations=local_context,exclusion=invalid)
                        kept=checked['candidate_ids'][checked['source_shape_supported']].tolist()
                        shape_reports.append(dict(source_candidates=checked['candidate_ids'].tolist(),supported_candidates=kept,
                            actual_transition_support=checked['interface_inside_actual_transition'].tolist(),
                            actual_band_unmasked=checked['entire_measured_band_unmasked'].tolist(),
                            same_frame_context_rows=checked['context_rows'],same_frame_context_paths=list(local_context),
                            shared_affine=checked['shared_affine'].tolist(),source_UV_unchanged=checked['source_UV_unchanged'],
                            physical_identity_confirmed=False))
                        if len(kept)>=3:qualified.append(dict(fragment,candidate_ids=kept))
                    fragments=qualified
                    run.write_json(folder/f'{pts}_{house}_shape_context.json',dict(fragments=shape_reports,
                        geometry_is_source_check_not_camera_update=True,fit_eligible=False))
                for j,fragment in enumerate(fragments):
                    key=(pts,house,j);parent[key]=key;ids=np.asarray(fragment['candidate_ids'],int)
                    prior_support=set()
                    for side in (0,1):
                        tree=old_tree.get((house,pts,side))
                        if tree is None:continue
                        distance,near=tree.query(measured['candidate_side_quantiles_uv'][ids,side,1])
                        prior_support.update(map(int,ids[distance<=3.]))
                    anchor=pts in definition.get('reviewed_pts',[]) and len(selected.intersection(ids))>=3
                    if len(prior_support)>=3 or anchor:
                        seeded.add(key);source_seeds.append(dict(fragment=j,basis='existing_qualified_original_source_pixels' if len(prior_support)>=3 else 'reviewed_source_pattern_and_complete_band_consensus'))
                    payload[key]=dict(frame=frame,ids=ids,measurement_file=str(stem.with_suffix('.npz')),
                        paths=[cuts[house][r]['id'] for r in ('middle','outer')])
                    if pts in definition.get('reviewed_pts',[]):
                        for cid in ids:
                            for side in (0,1):
                                uv=measured['candidate_side_quantiles_uv'][cid,side,1]
                                cv2.circle(proof,tuple(np.rint(uv).astype(int)),3,(0,255,0) if side==0 else (0,255,255),-1)
                if fragments:
                    inventory=[dict(operator='paint_center',material_signal=definition['signal'],material_transition=None)]
                    now[house]=source_nodes(rgb,measured,fragments,inventory)
                frame_receipt[house]=dict(queries=len(measured['query_uv']),candidates=len(measured['candidate_uv']),
                    fragments=len(fragments),complete_selected=len(selected),seeds=source_seeds)
            if previous is not None and (previous[0],pts)in adjacent:
                previous_pts,previous_rgb,previous_nodes,previous_mask,pa=previous
                valid=(pa['status']==0)&(grid['status']==0)&np.isfinite(pa['uv']).all(1)&np.isfinite(grid['uv']).all(1)
                for house in set(now)&set(previous_nodes):
                    local=valid.copy()
                    for projected,nodes in ((pa,previous_nodes[house]),(grid,now[house])):
                        low=nodes['uv'].min(0)-120;high=nodes['uv'].max(0)+120
                        local&=(projected['uv']>=low).all(1)&(projected['uv']<=high).all(1)
                    if local.sum()<8:continue
                    h,_=cv2.findHomography(pa['uv'][local],grid['uv'][local],0)
                    if h is None:continue
                    linked=link_fragments(previous_rgb,rgb,previous_nodes[house],now[house],previous_mask,invalid,h)
                    for edge in linked:
                        if edge['unambiguous']:
                            join((previous_pts,house,edge['a']),(pts,house,edge['b']))
                        edges.append(dict(from_pts=previous_pts,to_pts=pts,house=house,**edge))
            previous=(pts,rgb,now,invalid,grid)
            run.write_json(folder/f'{pts}_coverage.json',frame_receipt)
            if any(pts in d.get('reviewed_pts',[]) for d in definitions):cv2.imwrite(str(folder/f'{pts}_candidate_proof.jpg'),proof)
            if fi%50==0 or fi==len(frames)-1:
                entry=dict(shot=shot_id,exposures=fi+1,total=len(frames),fragments=len(parent),seed_fragments=len(seeded),links=len(edges),elapsed_s=time.monotonic()-started)
                log.write(json.dumps(entry)+'\n');print(json.dumps(entry),flush=True)
        connected={find(k) for k in seeded};admitted=[k for k in parent if find(k) in connected]
        # Different actual bands at the same source query remain competing.
        claims=collections.defaultdict(set)
        for key in admitted:
            item=payload[key];z=np.load(item['measurement_file'])
            for cid in item['ids']:claims[(key[0],key[1],int(z['candidate_query'][cid]))].add(int(cid))
        accepted=[];by_frame=collections.defaultdict(list);seen=set()
        for key in admitted:
            item=payload[key];z=np.load(item['measurement_file']);frame=item['frame'];pts,house,fragment=key
            for cid in item['ids']:
                if len(claims[(pts,house,int(z['candidate_query'][cid]))])!=1:continue
                for side,role in enumerate(('middle','outer')):
                    mark=(pts,house,int(cid),side)
                    if mark in seen:continue
                    seen.add(mark);quantiles=z['candidate_side_quantiles_uv'][cid,side]
                    row=dict(pts=pts,identity=f'{house}:{role}',path_id=item['paths'][side],candidate=int(cid)*2+side,
                        uv=quantiles[1].tolist(),actual_source_quantiles=quantiles.tolist(),fragment=fragment,
                        evidence_image=frame['image'],source_BGR_sha256=frame['source_BGR_sha256'],
                        source_reference=str(args.identities),measurement_file=item['measurement_file'],
                        source_query=int(z['candidate_query'][cid]),measured_band_pair=True,model_UV_synthesized=False)
                    accepted.append(row);by_frame[pts].append(row)
        packet=folder/'source_candidates_admitted.json';graph=folder/'source_links.json'
        run.write_json(packet,dict(shot=shot_id,accepted=accepted,source_identity_inference=specification['identity_scope'],
            user_visual_acceptance=False,actual_source_only=True))
        run.write_json(graph,dict(edges=edges,seeds=[list(k) for k in sorted(seeded)],
            connected_fragments=len(admitted),total_fragments=len(parent),ambiguous_queries=sum(len(x)>1 for x in claims.values()),
            qualification='source_pattern_anchors_and_unique_bidirectional_RGB_fragment_graph'))
        result=dict(shot=shot_id,accepted=accepted,source_packet=str(packet),source_packet_sha256=run.digest(packet),
            source_links=str(graph),source_links_sha256=run.digest(graph),source_UV_synthesized=False,
            user_visual_acceptance=False,qualification='original_paired_material_boundaries_on_source_connected_visible_arcs',
            supported_exposures=len(by_frame),total_exposures=len(frames),fit_eligible=not bool(getattr(args,'candidate_only',False)),
            review_state='awaiting_source_identity_and_pixel_review'if getattr(args,'candidate_only',False)else 'source_graph_qualified')
        run.write_json(folder/'source_verified.json',result)
        for pts in sorted({p for d in definitions for p in d.get('reviewed_pts',[])}):
            frame=next(f for f in frames if f['pts']==pts);proof=cv2.imread(frame['image'])
            for row in by_frame.get(pts,[]):
                cv2.circle(proof,tuple(np.rint(row['uv']).astype(int)),3,
                    (0,255,0) if row['identity'].endswith(':middle') else (0,255,255),-1)
            cv2.imwrite(str(folder/f'{pts}_accepted_proof.jpg'),proof)
        print(json.dumps(dict(shot=shot_id,accepted_rows=len(accepted),supported_exposures=len({r['pts'] for r in accepted}))),flush=True)
    log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('run-root','state','identities','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--source-cache',type=Path)
    parser.add_argument('--pts',type=int,nargs='+',help='Optional actual source exposures; temporal links never bridge omitted exposures')
    parser.add_argument('--candidate-only',action='store_true',help='Keep generated source packets ineligible until reviewed')
    capture(parser.parse_args())
