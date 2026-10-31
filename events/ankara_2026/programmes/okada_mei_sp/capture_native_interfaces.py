"""Read native annulus roles independently and link actual visible RGB arcs.

Geometry organizes source searches and competing layer assignments. Original
signed-interface pixels, masks and bidirectional RGB links provide the data.
"""
from pathlib import Path
import argparse, collections, hashlib, json, time, os
import cv2
import numpy as np
from scipy.spatial import cKDTree
import run_native_programme as run
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.compat.d78_entities import sample_entities
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
from rinkspace.session.native_material_association import associate_annulus_interfaces, source_interface_material_support
from rinkspace.session.native_shape_guidance import register_native_shapes
from rinkspace.session.material_tracks import source_nodes, link_fragments, assemble_fragments, source_line_reference_support


def capture(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    m=run.read(args.run_root/'joint_manifest.json');v=dict(np.load(args.state))
    old=dict(np.load(args.run_root/'point_observations.npz'));mapping=run.read(args.run_root/'mapping.json')
    scene=run.read(m['scene_path']);model=SceneParameterModel(scene,Path(m['scene_path']),optimize_scene=True)
    core=NativeCore(m['native_library']);samples=sample_entities(ParameterState(model.initial_state.layout,v['scene_values']).apply_scene(scene),core)
    timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time']);spec=run.read(args.identities)
    from rinkspace.session import native_material_association,native_shape_guidance,material_tracks
    from rinkspace import vision_programme_ice_review
    binding=dict(state=str(args.state),state_sha256=run.digest(args.state),identities=str(args.identities),
        identity_sha256=run.digest(args.identities),point_input_sha256=m['point_input_sha256'],
        native_library_sha256=m['native_library_sha256'],module_sha256=run.digest(__file__),
        algorithm_sha256={str(x.__file__):run.digest(x.__file__) for x in (native_material_association,native_shape_guidance,material_tracks,vision_programme_ice_review)},
        source_uv_changed=False,geometry_only_organizes_source_search=True,sample_spacing_px=24.,
        requested_shots=args.shots,requested_pts=getattr(args,'pts',None),source_material_support=args.material_support,
        source_reader_options=dict(sample_spacing_px=24.,search_radius_px=48.,separate_bands=True))
    if (out/'binding.json').exists() and run.read(out/'binding.json')!=binding:raise ValueError('Capture binding changed')
    if args.source_cache:
        cached=run.read(args.source_cache/'binding.json')
        for key in ('state_sha256','identity_sha256','native_library_sha256'):
            if cached[key]!=binding[key]:raise ValueError('Raw source cache mismatch: '+key)
        if cached.get('source_reader_options')!=binding['source_reader_options']:
            raise ValueError('Raw source cache has different interface extraction options')
        key=str(vision_programme_ice_review.__file__)
        if cached['algorithm_sha256'][key]!=binding['algorithm_sha256'][key]:raise ValueError('Raw RGB reader changed')
    run.write_json(out/'binding.json',binding);started=time.monotonic();cv2.setNumThreads(2)
    with (out/'capture.jsonl').open('a',encoding='utf-8',buffering=1) as log:
      for sid,definitions in spec['shots'].items():
        if args.shots and sid not in args.shots:continue
        vis,frames=run.source_frames(sid,selected_pts=getattr(args,'pts',None));shot=next(s for s in m['shots'] if s['id']==sid)
        folder=out/sid;folder.mkdir(exist_ok=True);prepared={};old_trees={};parent={};payload={};seeds=set();seed_evidence=[];edges=[];previous={};source_count=0
        def find(k):
            while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
            return k
        for d in definitions:
            house=d['house'];paths=[]
            for role in ('middle','outer'):
                c=next(c for c in samples['cuts'] if c['label']==f'ice.{house}:{role}');xyz=samples['xyz'][c['begin']:c['end']]
                paths.append(dict(curve_id=f'{house}:{role}',feature_id=house,world_curve_role=role,native_path_id=c['id'],xyz=xyz,
                    center_xyz=xyz[:-1].mean(0),operator='material_interface',material_signal='dark',
                    material_transition='light_to_dark' if role=='middle' else 'dark_to_light'))
                pi=mapping['path_ids'].index(c['id']);rows=np.flatnonzero((old['path_indices']==pi)&(old['pts']>=shot['first_pts'])&(old['pts']<=shot['last_pts']))
                for pt in np.unique(old['pts'][rows]):old_trees[(house,role,int(pt))]=cKDTree(old['uv'][rows[old['pts'][rows]==pt]])
            reader=IceProgrammeSourceReader(None,source_sha256=vis['source_video_sha256'],time_base='1/12800',native_paths=paths,
                sample_spacing_px=24.,search_radius_px=48.,separate_bands=True,prepared_source_context=dict(source_sha256=vis['source_video_sha256'],time_base='1/12800',
                    authority=[dict(path=str(args.identities),sha256=binding['identity_sha256'])]))
            prepared[house]=(paths,reader)
        for fi,frame in enumerate(frames):
            pts=frame['pts'];clock=(pts-shot['reference_pts'])/12800+timeline.offset(shot);visible={}
            def project(x):return core.project_motion(v['rig'][shot['camera_index']],v['controls'],timeline.first,timeline.dt,1080,clock,x)
            for d in definitions:
                house=d['house'];paths,_=prepared[house];projections={};curves=[]
                for path in paths:
                    p=project(path['xyz']);center=project(path['center_xyz'][None])['uv'][0]
                    projections[path['native_path_id']]=dict(uv=p['uv'],valid=p['status']==0,centre_uv=center);curves.append(p['uv'])
                finite=all(p['valid'].all() and np.isfinite(p['uv']).all() for p in projections.values())
                if finite and any(((uv>[-48,-48])&(uv<[1968,1128])).all(1).any() for uv in curves):visible[house]=(projections,curves)
            if visible:
                im=cv2.imread(frame['image'])
                if hashlib.sha256(im.tobytes()).hexdigest()!=frame['source_BGR_sha256']:raise ValueError('Original pixels changed')
                rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);masks,_=run.surfaces(frame,im);invalid=~masks['ice'];source_count+=1
                for d in definitions:
                    house=d['house']
                    if house not in visible:continue
                    paths,reader=prepared[house];projections,curves=visible[house];stem=folder/f'{pts}_{house}'
                    if stem.with_suffix('.npz').exists() and stem.with_suffix('.json').exists():
                        z=dict(np.load(stem.with_suffix('.npz')));result=run.read(stem.with_suffix('.json'))
                    else:
                        cached_raw=None if args.source_cache is None else args.source_cache/sid/f'{pts}_{house}.npz'
                        if cached_raw is not None and cached_raw.exists():z=dict(np.load(cached_raw))
                        else:z=reader.read(rgb,pts,None,invalid,shot_id=sid,native_projections=projections)
                        eligible=np.ones(len(z['candidate_uv']),bool)
                        if d.get('color')=='neutral_dark' and len(eligible):
                            q=z['candidate_query'];normal=z['query_normal'][q];quant=z['candidate_side_quantiles_uv'][:,0]
                            xy=np.stack([quant[:,0]-normal*5.,quant[:,2]+normal*5.],1).astype('f')
                            pixels=cv2.remap(rgb.astype('f'),xy[...,0],xy[...,1],cv2.INTER_LINEAR)
                            ci=z['query_curve'][q];pigment=np.where((ci==0)[:,None],pixels[:,1],pixels[:,0])
                            eligible &= pigment[:,0]-.5*(pigment[:,1]+pigment[:,2])<=1.5
                        selected,fragments,receipt=associate_annulus_interfaces(z,paths,curves,eligible_candidates=eligible)
                        if len(selected)>=3:
                            ci=z['query_curve'][z['candidate_query'][selected]]
                            names=np.asarray([str(i) for i in ci])
                            try:
                                registration=register_native_shapes(z['candidate_uv'][selected],names,
                                    {str(j):c for j,c in enumerate(curves)},rms_change_tolerance_px=1e-3)
                                affine=registration.affine
                                aligned=[c@affine[:2,:2].T+affine[:2,2] for c in curves]
                                selected,fragments,aligned_receipt=associate_annulus_interfaces(z,paths,aligned,eligible_candidates=eligible)
                                receipt=dict(initial=receipt,aligned=aligned_receipt,shared_affine=affine.tolist(),
                                    source_alignment_rms_px=registration.rms_distance_px,
                                    native_parameters_changed=False,source_uv_changed=False)
                            except ValueError as error:receipt['shared_alignment_unresolved']=str(error)
                        if args.material_support:
                            support,material_receipt=source_interface_material_support(rgb,z,paths,selected,invalid)
                            retained=np.zeros(len(z['candidate_uv']),bool);retained[selected]=True;retained &= support
                            fragments=assemble_fragments(z,paths,eligible_candidates=retained)
                            selected=sorted({int(i) for f in fragments for i in f['candidate_ids']})
                            receipt=dict(shape_association=receipt,actual_material_support=material_receipt,
                                weak_transition_disposition='unresolved_source_material_not_observed_boundary')
                        result=dict(selected_ids=list(map(int,selected)),fragments=fragments,receipt=receipt)
                        if cached_raw is not None and cached_raw.exists():os.link(cached_raw,stem.with_suffix('.npz'))
                        else:np.savez_compressed(stem.with_suffix('.npz'),**{k:x for k,x in z.items() if isinstance(x,np.ndarray)})
                        run.write_json(stem.with_suffix('.json'),result)
                    fragments=result['fragments']
                    for j,fragment in enumerate(fragments):
                        key=(pts,house,j);parent[key]=key;ids=np.asarray(fragment['candidate_ids'],int)
                        role=paths[fragment['curve']]['world_curve_role'];tree=old_trees.get((house,role,pts))
                        # Both sets are measured source pixels. Overlap their
                        # finite observed arcs, without requiring arbitrary
                        # tangent sample positions to coincide point by point.
                        support=source_line_reference_support(
                            np.empty((0,2)) if tree is None else tree.data,
                            z['candidate_uv'][ids],z['candidate_query'][ids])
                        reviewed=pts in d.get('reviewed_pts',[])
                        if support['qualified'] or reviewed:
                            seeds.add(key)
                            seed_evidence.append(dict(pts=pts,house=house,fragment=j,role=role,
                                basis='existing_measured_source_arc_overlap' if support['qualified'] else 'reviewed_source_image_identity',
                                source_overlap=support,model_UV_synthesized=False))
                        payload[key]=(frame,ids,stem.with_suffix('.npz'),fragment['curve'],paths[fragment['curve']])
                    if fragments:
                        nodes=source_nodes(rgb,z,fragments,paths);all_curve=np.vstack(curves)
                        if house in previous:
                            prev=previous[house];h,_=cv2.findHomography(prev['curve'],all_curve,0)
                            if h is not None:
                                links=link_fragments(prev['rgb'],rgb,prev['nodes'],nodes,prev['mask'],invalid,h)
                                for edge in links:
                                    a=(prev['pts'],house,edge['a']);b=(pts,house,edge['b'])
                                    if edge['unambiguous']:parent[find(a)]=find(b)
                                    edges.append(dict(from_pts=prev['pts'],to_pts=pts,house=house,**edge))
                        previous[house]=dict(pts=pts,rgb=rgb,nodes=nodes,mask=invalid,curve=all_curve)
                    if pts in d.get('reviewed_pts',[]) or (sid=='s06' and pts in (1403392,1416192,1418752,1421312)):
                        proof=im.copy()
                        for fragment in fragments:
                            for cid in fragment['candidate_ids']:cv2.circle(proof,tuple(np.rint(z['candidate_uv'][cid]).astype(int)),3,(0,255,0) if fragment['curve']==0 else (0,255,255),-1)
                        cv2.imwrite(str(folder/f'{pts}_{house}_proof.jpg'),proof)
            if fi%50==0 or fi==len(frames)-1:
                r=dict(shot=sid,exposures=fi+1,source_exposures=source_count,total=len(frames),fragments=len(parent),seeds=len(seeds),links=len(edges),elapsed_s=time.monotonic()-started)
                log.write(json.dumps(r)+'\n');print(json.dumps(r),flush=True)
        connected={find(k) for k in seeds};accepted=[];seen=set()
        for key,(frame,ids,raw,curve,path) in payload.items():
            if find(key) not in connected:continue
            z=np.load(raw)
            for cid in ids:
                tag=(key[0],path['native_path_id'],int(cid))
                if tag in seen:continue
                seen.add(tag);quant=z['candidate_side_quantiles_uv'][cid,0]
                accepted.append(dict(pts=key[0],identity=path['curve_id'],path_id=path['native_path_id'],candidate=int(cid),
                    uv=quant[1].tolist(),actual_source_quantiles=quant.tolist(),fragment=key[2],
                    evidence_image=frame['image'],source_BGR_sha256=frame['source_BGR_sha256'],
                    source_reference=str(raw),mask_path=frame['mask_path'],mask_sha256=frame['mask_sha256'],
                    model_UV_synthesized=False,measurement_operator='signed_material_interface'))
        source=folder/'source_pixels.json';links=folder/'source_links.json'
        run.write_json(source,dict(accepted=accepted,shot=sid,identity_basis=spec['identity_scope'],source_UV_changed=False))
        run.write_json(links,dict(edges=edges,seeds=[list(k) for k in sorted(seeds)],seed_evidence=seed_evidence,
            source_identity_is_assistant_inference=True,whole_ring_required=False,unconnected_fragments=len(parent)-sum(find(k) in connected for k in parent)))
        run.write_json(folder/'source_verified.json',dict(shot=sid,accepted=accepted,source_packet=str(source),source_packet_sha256=run.digest(source),
            source_links=str(links),source_links_sha256=run.digest(links),fit_eligible=True,user_visual_acceptance=False))
        print(json.dumps(dict(shot=sid,accepted=len(accepted),supported_exposures=len({r['pts'] for r in accepted}))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('run-root','state','identities','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source-cache',type=Path)
    p.add_argument('--shots',nargs='+')
    p.add_argument('--pts',nargs='+',type=int)
    p.add_argument('--material-support',action='store_true')
    capture(p.parse_args())
