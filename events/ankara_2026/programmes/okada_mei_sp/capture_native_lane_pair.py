"""Read adjacent native lane lines without transferring identity through old points."""
from pathlib import Path
import argparse,hashlib,json,time
import cv2,numpy as np
from programme_source_frames import iter_registration_source_frames
from capture_independent_materials import spaced_indices,plain
from capture_native_centerlines import qualify_painted_paths,source_ice_exclusion
from capture_source_board_pilot import curve_distances
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.session.gpu_geometry import NativeSceneGPU
from rinkspace.session.native_shape_guidance import point_to_polyline
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader

def capture(args):
    if args.output.exists():raise FileExistsError(args.output)
    m=json.loads((args.run_root/'joint_manifest.json').read_text('utf8'))
    state=np.load(args.state);core=NativeCore(m['native_library']);scene=Path(m['scene_path'])
    raw=json.loads(scene.read_text('utf8'));model=SceneParameterModel(raw,scene,optimize_scene=True)
    native=NativeSceneGPU(scene,core,parameter_state=ParameterState(model.initial_state.layout,state['scene_values']))
    paths=[]
    for identity in args.path_ids:
        p=native.paths[identity];xyz=np.asarray(core.sample_path(native.scene['board'],p['kind'],p['parameters'],p['vertices'],np.linspace(0,1,129)))
        if np.linalg.matrix_rank(xyz-xyz[0])!=1:raise ValueError('Expected native straight lane path')
        name=p['label'].removeprefix('ice.');feature,_,role=name.partition(':')
        paths.append(dict(curve_id=name,feature_id=feature,world_curve_role=role,native_path_id=identity,xyz=xyz,
            operator='paint_center',material_signal='dark',material_transition=None,closed=False))
    competing=[]
    for p in native.paths.values():
        if p['layer']=='ice' and p['label'].startswith(('ice.R3:','ice.F3:','ice.C0:')):
            xyz=np.asarray(core.sample_path(native.scene['board'],p['kind'],p['parameters'],p['vertices'],np.linspace(0,1,257)))
            competing.append((p['id'],xyz))
    source=json.loads((args.bootstrap/'index.json').read_text('utf8'))['source'];source_sha=source['actual_sha256']
    timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time']);shot=next(s for s in m['shots']if s['id']==args.shot)
    reader=IceProgrammeSourceReader(None,source_sha256=source_sha,time_base='1/12800',native_paths=paths,
        sample_spacing_px=24.,search_radius_px=40.,separate_bands=True,profile_backend='cuda',
        prepared_source_context=dict(source_sha256=source_sha,time_base='1/12800',authority=[dict(path=str(scene),sha256=m['scene_file_sha256'])]))
    out=args.output;out.mkdir(parents=True);cv2.setNumThreads(2);start=time.monotonic();counts={p['native_path_id']:0 for p in paths}
    binding=dict(state=str(args.state),state_sha256=hashlib.sha256(args.state.read_bytes()).hexdigest(),
        native_library=m['native_library'],native_library_sha256=m['native_library_sha256'],path_ids=args.path_ids,
        profile_backend='cuda',source_UV_synthesized=False,old_observations_loaded=False,source_points_spacing_px=48.,
        source_identity_basis='Known main-camera side and ordered native parallel lane edges; actual paired dark-paint RGB boundaries and whole-line source shape. Adjacent identities searched separately; projection supplies no measured UV.',fit_eligible=False)
    (out/'binding.json').write_text(json.dumps(binding,indent=2),encoding='utf8')
    proofs=set(args.proof_pts)|{args.pts[0],args.pts[-1]}
    with (out/'pending_source_rows.jsonl').open('w',encoding='utf8')as stream:
        for fi,f in enumerate(iter_registration_source_frames(args.shot,bootstrap_root=args.bootstrap,selected_pts=args.pts)):
            pts=f['pts'];t=(pts-shot['reference_pts'])/12800+timeline.offset(shot);projections={}
            for p in paths:
                projected=core.project_motion(state['rig'][shot['camera_index']],state['controls'],timeline.first,timeline.dt,1080,t,p['xyz'])
                projections[p['native_path_id']]=dict(uv=projected['uv'],valid=projected['status']==0)
            rgb=cv2.cvtColor(f['image'],cv2.COLOR_BGR2RGB);invalid,mask_note=source_ice_exclusion(rgb,dict(unoccluded_source=~f['source_exclusion_mask'],ice=None))
            z=(dict(np.load(args.reuse_source/f'{pts}_readout.npz')) if args.reuse_source else
                reader.read(rgb,pts,None,invalid,shot_id=args.shot,native_projections=projections))
            rawfile=out/f'{pts}_readout.npz'
            np.savez_compressed(rawfile,**{k:v for k,v in z.items()if isinstance(v,np.ndarray)})
            # Wide RGB profiles preserve complete paired material boundaries;
            # identity search stays inside each separately ordered native strip.
            # Do not let the wider material context copy an adjacent line label.
            distance=np.array([curve_distances(z['candidate_uv'],projections[p['native_path_id']]['uv'])for p in paths])
            owner=z['query_curve'][z['candidate_query']]
            identity=np.zeros(len(owner),bool)
            for ci in range(len(paths)):
                other=np.min(distance[np.arange(len(paths))!=ci],axis=0)
                identity|=(owner==ci)&(distance[ci]<12.)&(distance[ci]+4.<other)
            own=distance[owner,np.arange(len(owner))]
            arc_distance=np.full(len(owner),np.inf)
            for _,xyz in competing:
                pr=core.project_motion(state['rig'][shot['camera_index']],state['controls'],timeline.first,timeline.dt,1080,t,xyz)
                curve=pr['uv'].copy();curve[pr['status']!=0]=np.nan
                arc_distance=np.minimum(arc_distance,curve_distances(z['candidate_uv'],curve))
            identity &= arc_distance>own+3.
            checked=dict(z,candidate_flags=z['candidate_flags'].copy())
            checked['candidate_flags'][~identity]|=2
            fragments,report=qualify_painted_paths(checked,paths,invalid)
            report['native_identity_neighborhood_only']=dict(material_profile_radius_px=40.,identity_search_radius_px=12.,
                neighbor_identity_margin_px=4.,competing_curve_ids=[i for i,_ in competing],
                source_UV_replaced_by_projection=False,eligible_candidates=int(identity.sum()))
            rows=[];picture=f['image'].copy()
            for ci,p in enumerate(paths):
                ids=sorted({i for frag in fragments if frag['curve']==ci for i in frag['candidate_ids']},key=lambda i:int(z['candidate_query'][i]))
                if not ids:continue
                chosen=[ids[j]for j in spaced_indices(z['candidate_uv'][ids])]
                for i in chosen:
                    row=dict(path_id=p['native_path_id'],entity=p['curve_id'],uv=z['candidate_uv'][i].tolist(),
                        actual_source_sides=z['candidate_sides'][i].tolist(),actual_source_quantiles=z['candidate_side_quantiles_uv'][i].tolist(),
                        measurement_operator='paired_material_boundary_midline',raw_readout=str(rawfile),raw_candidate_id=int(i),
                        model_UV_synthesized=False,fit_eligible=False,source_identity_basis=binding['source_identity_basis'])
                    rows.append(row);cv2.circle(picture,tuple(np.rint(row['uv']).astype(int)),3,[(0,200,0),(0,200,255)][ci%2],1,cv2.LINE_AA)
                counts[p['native_path_id']]+=len(chosen)
            rec=dict(pts=pts,shot_id=args.shot,source_video_sha256=source_sha,source_BGR_sha256=f['source_BGR_sha256'],
                mask_path=str(f['mask_path']),mask_sha256=f['mask_sha256'],rows=rows,fit_eligible=False,
                review_state='pending_root_source_identity_review',source_UV_synthesized=False)
            stream.write(json.dumps(plain(rec),ensure_ascii=False)+'\n');stream.flush()
            (out/f'{pts}_shape.json').write_text(json.dumps(plain(dict(fragments=fragments,report=report,mask=mask_note)),ensure_ascii=False),encoding='utf8')
            if pts in proofs:cv2.imwrite(str(out/f'{pts}_source_points.png'),picture)
            if (fi+1)%25==0 or pts==args.pts[-1]:print(json.dumps(dict(frames=fi+1,pts=pts,rows=counts,elapsed_s=time.monotonic()-start)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('run-root','state','bootstrap','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--shot',required=True);p.add_argument('--pts',type=int,nargs='+',required=True)
    p.add_argument('--path-ids',nargs='+',required=True);p.add_argument('--proof-pts',type=int,nargs='*',default=[])
    p.add_argument('--reuse-source',type=Path)
    capture(p.parse_args())
