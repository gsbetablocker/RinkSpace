"""Independently reacquire the visible centre red stripe from original frames.

Known native camera orientation supplies an orientation/ice search, not UV.
A complete source-colour branch locates the paired actual RGB interfaces.
Captured points remain pending identity propagation from source-read anchors.
"""
from pathlib import Path
import argparse,hashlib,json,time
import cv2,numpy as np
from capture_independent_materials import native_ice_inventory,source_exclusion,spaced_indices,plain,write
from prepare_independent_navigation import frames
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.compat.d78_entities import sample_entities
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
from rinkspace.session.material_tracks import qualify_visible_line_fragments
from rinkspace.session.chromatic_line_navigation import source_chromatic_line_navigation,source_board_floor_limit


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(args):
    if args.output.exists():raise FileExistsError(args.output)
    root=args.root;index=json.loads((root/'index.json').read_text('utf8'));m=json.loads((args.run_root/'joint_manifest.json').read_text('utf8'))
    state=dict(np.load(args.state,allow_pickle=False));library=args.library
    if digest(library)!=str(state['native_library_provenance_sha256']):raise ValueError('State/library provenance differs')
    scene_path=Path(m['scene_path']);raw=json.loads(scene_path.read_text('utf8'));model=SceneParameterModel(raw,scene_path,optimize_scene=True)
    parameters=ParameterState(model.initial_state.layout,state['scene_values']);core=NativeCore(library);sample=sample_entities(parameters.apply_scene(raw),core)
    paths=native_ice_inventory(sample['cuts'],sample['xyz']);path=next(p for p in paths if p['feature_id']=='C')
    # Far-side floor search domain only; its pixels never become measurements.
    floor=sample['xyz'][np.isclose(sample['xyz'][:,2],0.,atol=1e-12)]
    floor=floor[::max(1,len(floor)//3000)]
    timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time'])
    reader=IceProgrammeSourceReader(None,source_sha256=index['source']['actual_sha256'],time_base='1/12800',native_paths=[path],
        sample_spacing_px=24.,search_radius_px=128.,separate_bands=True,
        prepared_source_context=dict(source_sha256=index['source']['actual_sha256'],time_base='1/12800',authority=[dict(path=str(scene_path),sha256=m['scene_file_sha256'])]))
    args.output.mkdir(parents=True);cv2.setNumThreads(2);started=time.monotonic()
    write(args.output/'binding.json',dict(schema='rinkspace.independent_global_centerline.v1',source_video_sha256=index['source']['actual_sha256'],
        state=str(args.state),state_sha256=digest(args.state),native_library=str(library),native_library_sha256=digest(library),
        independent_source_capture=True,legacy_new_shot_inputs_read=False,source_UV_synthesized=False,fit_eligible=False,native_inventory=reader.inventory,adapter_sha256=digest(__file__),navigation_sha256=digest(source_chromatic_line_navigation.__code__.co_filename)))
    for sid in args.shots:
        source=next(s for s in index['shots']if s['shot_id']==sid);shot=next(s for s in m['shots']if s['id']==sid)
        if shot['camera_index']!=0:raise ValueError('This source identity initialization is for the known main rig')
        folder=args.output/sid;folder.mkdir();total=0;valid_frames=0;previous=None;components=[];component=-1;current=[]
        representatives={f['actual_pts']for f in source['frames']}
        with (folder/'source_rows.jsonl').open('w',encoding='utf8',buffering=1)as stream:
            for fi,(pts,image)in enumerate(frames(index,source)):
                if args.pts and pts not in args.pts:continue
                mask,mask_path=source_exclusion(root,sid,pts);rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                clock=(pts-shot['reference_pts'])/12800+timeline.offset(shot)
                def project(xyz):return core.project_motion(state['rig'][0],state['controls'],timeline.first,timeline.dt,1080,clock,xyz)
                guide=project(path['xyz']);finite=guide['uv'][guide['status']==0]
                rows=[];fragments=[];nav=None;receipt=dict(status='native_orientation_unavailable');report={}
                if len(finite)>1 and abs(finite[-1,1]-finite[0,1])>.5*np.linalg.norm(finite[-1]-finite[0]):
                    floor_result=project(floor);uv=floor_result['uv'];visible=(floor_result['status']==0)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<1920)&(uv[:,1]>=0)&(uv[:,1]<1080)
                    ymin=float(np.min(uv[visible,1])) if visible.any() else 0.
                    source_y,source_floor=source_board_floor_limit(rgb,mask)
                    if source_y is not None:ymin=max(ymin,source_y)
                    nav,receipt=source_chromatic_line_navigation(rgb,mask,finite[-1]-finite[0],minimum_y=ymin,radius_px=128.,allow_competing_search=True)
                    receipt['source_floor_search']=source_floor
                if nav is not None:
                    z=reader.read(rgb,pts,None,mask,shot_id=sid,native_projections={path['native_path_id']:dict(uv=nav,valid=np.ones(len(nav),bool))})
                    fragments,report=qualify_visible_line_fragments(z,[0],mask)
                    for fragment in fragments:
                        ids=np.asarray(fragment['candidate_ids'],int)
                        for k in spaced_indices(z['candidate_uv'][ids]):
                            i=int(ids[k]);rows.append(dict(path_id=path['native_path_id'],entity=path['curve_id'],uv=z['candidate_uv'][i].tolist(),
                                actual_source_sides=z['candidate_sides'][i].tolist(),actual_source_quantiles=z['candidate_side_quantiles_uv'][i].tolist(),
                                measurement_operator='paired_material_boundary_midline',model_UV_synthesized=False,candidate=i,
                                source_identity_basis='unique source chromatic stripe; pending known-rig/topology anchor connection'))
                # These links only organize candidate review. No fit eligibility
                # follows from proximity and no absent source point is filled.
                connected=False
                if rows:
                    uv=np.array([r['uv']for r in rows]);center=uv.mean(0);_,_,axes=np.linalg.svd(uv-center,full_matrices=False);direction=axes[0];normal=np.array([-direction[1],direction[0]])
                    if previous and pts-previous['pts']==256:
                        offset=abs(float((center-previous['center'])@previous['normal']));angle=abs(float(direction@previous['direction']))
                        connected=offset<120 and angle>np.cos(np.deg2rad(10.))
                    if not connected:
                        if current:components.append(current)
                        current=[];component+=1
                    current.append(pts);previous=dict(pts=pts,center=center,normal=normal,direction=direction);valid_frames+=1
                else:
                    if current:components.append(current);current=[]
                    previous=None
                record=dict(pts=pts,source_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),source_video_sha256=index['source']['actual_sha256'],
                    mask_path=str(mask_path),mask_sha256=digest(mask_path),fit_eligible=False,review_state='pending_independent_centerline_identity',rows=rows,
                    source_search=receipt,source_component=component if rows else None,source_link_to_previous=connected)
                stream.write(json.dumps(plain(record),ensure_ascii=False)+'\n')
                if pts in representatives or args.pts:
                    proof=image.copy()
                    for row in rows:cv2.circle(proof,tuple(np.rint(row['uv']).astype(int)),3,(0,220,0),-1)
                    cv2.imwrite(str(folder/f'{pts}_source_candidates.jpg'),proof)
                    write(folder/f'{pts}_diagnostic.json',dict(search=receipt,reports=report,fragments=fragments))
                total+=len(rows)
                if fi%25==0 or pts==source['processing']['last_pts']:print(json.dumps(dict(shot=sid,frames=fi+1,pts=pts,rows=total,valid_frames=valid_frames,elapsed_s=time.monotonic()-started)),flush=True)
        if current:components.append(current)
        write(folder/'coverage.json',dict(frames=fi+1,rows=total,valid_frames=valid_frames,components=components,fit_eligible=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','run-root','state','output','library'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--shots',nargs='+',default=['s07','s09']);p.add_argument('--pts',nargs='+',type=int)
    capture(p.parse_args())
if __name__=='__main__':main()
