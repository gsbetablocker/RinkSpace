"""Source-only original low FIT candidates and same-query depth diagnostics.

Uses frozen data-only NPZ ledgers. No programme graph, solver, J or video.
The counterfactual finite-board values never qualify a material role.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
sys.dont_write_bytecode=True
import av, cv2, numpy as np
from scipy.spatial import cKDTree
cv2.setNumThreads(1)
DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
HISTORY=DATA/'history/work_sessions'
ADMISSION=HISTORY/'20261002_01a0fa4f-6592-7ab2-bc33-42ff983aff59/z190_low_fixed_mount_foreground_rf_v1/original_motion_foreground_admission.npz'
SELECTED=HISTORY/'20261002_01a0fa4f-6592-7ab2-bc33-42ff983aff59/z190_low_source_authority_rf_v1/low5_retained_original_source_rows.npz'
RAW=DATA/'programmes/okada_mei_sp/history/tracking_runs/motion/motion_shot_05.npz'
RETIREMENT=DATA/'calibration/runs/spin_source_long_baseline_20261005/actual50_jitter_source_diagnostic_20261005/spin_micro_broadcast_overlay_original_motion_role_retirement_SOURCE_ONLY_v1.json'
ICE=HISTORY/'20261002_01a0fd26-33ba-7873-9aa9-bbd9d2ca1328/low5_repair/core_ice_motion_source_v2/source_admission_reviewed.json'
TRACKS=DATA/'programmes/okada_mei_sp/results/aligned_complete_20261007_478afaf77615/continuous_adjacent_native_board_source_v2/adjacent_real_board_tracks.npz'
PEOPLE=Path('D:/CodexData/RinkSpace_program_tracking_tmp/people/boxes.json')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def arrays(p):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def stats(x):
    x=np.asarray(x,float)
    return dict(count=len(x),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(x.max()))if len(x)else dict(count=0)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--state',type=Path,action='append',required=True)
    ap.add_argument('--auxiliary',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--reuse-native-RGB',type=Path,help='Prior same-query report; authenticate and reuse its six exact source RGBs')
    ap.add_argument('--target-pts',type=int,default=229961216)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    from calibration.solver_runtime import install
    install()
    from rinkspace.vision_camera_initialization import current_complete_source_geometry
    from structured_programme_source_candidate.native_board_track_source import current_far_board_points
    from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown
    from z190_motion_native import native_motion
    retirement=read(RETIREMENT)
    assert sha(ADMISSION)==retirement['original_foreground_admission_npz']['sha256']
    assert sha(SELECTED)==retirement['original_selected_rows_npz']['sha256']
    assert sha(RAW)==retirement['original_motion_npz']['sha256']
    admission,selected,raw,tracks=map(arrays,(ADMISSION,SELECTED,RAW,TRACKS))
    state0=arrays(a.state[0]);lengths=state0['aux__block_lengths']//4;names=list(state0['aux__block_names']);bi=names.index('low')
    lo=int(lengths[:bi].sum());stamps=state0['aux__canonical_pts'][lo:lo+int(lengths[bi])]
    assert len(stamps)==1851 and bi==3
    active=admission['active_original_global_index'];remap=np.full(len(admission['source_uv_a']),-1,int);remap[active]=np.arange(len(active))
    ids=selected['original_global_row_indices'];full=31987+np.arange(len(ids))
    assert len(active)==434991 and len(ids)==319568
    assert np.array_equal(admission['source_uv_a'][full],raw['xy0'][ids])and np.array_equal(admission['source_uv_b'][full],raw['xy1'][ids])
    aliases=arrays(retirement['source_actual_exposure_alias_authority']['path'])
    alias=dict(zip(map(int,aliases['pts']),map(int,aliases['exposure_representative_pts'])))
    actual_a=stamps[admission['source_endpoint_index_a'][full]];actual_b=stamps[admission['source_endpoint_index_b'][full]]
    rawpair=np.repeat(np.arange(len(raw['pair_pts'])),np.diff(raw['offsets']))[ids]
    assert np.array_equal(actual_a,[alias[int(raw['pair_pts'][j,0])]for j in rawpair])
    assert np.array_equal(actual_b,[alias[int(raw['pair_pts'][j,1])]for j in rawpair])
    known_ice={int(r['original_global_index'])for r in read(ICE)['qualified_rows']if r['split']=='fit'}
    retired={int(r['current_bundle_motion_row'])for r in retirement['records']}
    people={int(r['pts']):r['boxes']for r in read(PEOPLE)['frames']}
    tp=list(zip(map(int,tracks['source_pts']),map(int,tracks['target_pts'])))
    lookup={pair:i for i,pair in enumerate(tp)}
    def clear(p,stamp):
        if stamp not in people:return np.zeros(len(p),bool)
        keep=np.ones(len(p),bool)
        for box in people.get(stamp,[]):
            x0,y0,x1,y1=np.asarray(box[:4])*2;dx=max(48.,.4*(x1-x0));dy=48.
            keep&=~((p[:,0]>=x0-dx)&(p[:,0]<=x1+dx)&(p[:,1]>=y0-dy)&(p[:,1]<=y1+dy))
        return keep
    records=[];groups=[]
    for j,pair in enumerate(raw['pair_pts']):
        pa,pb=map(lambda p:alias[int(p)],pair)
        if abs(pa-a.target_pts)>15360 or pa==pb:continue
        rows=np.flatnonzero(rawpair==j);rows=rows[remap[full[rows]]>=0]
        if not len(rows):continue
        reverse=False;ti=lookup.get((pa,pb))
        if ti is None:ti=lookup.get((pb,pa));reverse=True
        if ti is None:continue
        tl,th=map(int,tracks['offsets'][ti:ti+2]);ta,tb=tracks['uv0'][tl:th],tracks['uv1'][tl:th]
        if reverse:ta,tb=tb,ta
        if not len(ta):continue
        p,q=raw['xy0'][ids[rows]],raw['xy1'][ids[rows]]
        _,near=cKDTree(np.c_[ta,tb]).query(np.c_[p,q]);ea=np.linalg.norm(p-ta[near],axis=1);eb=np.linalg.norm(q-tb[near],axis=1)
        keep=(ea<=2.)&(eb<=2.)&clear(p,pa)&clear(q,pb)&(p[:,1]<800)&(q[:,1]<800)
        accepted=[]
        for ri,k in zip(rows[keep],np.flatnonzero(keep)):
            i=int(remap[full[ri]]);g=int(ids[ri])
            if g in known_ice or i in retired:continue
            r=dict(original_raw_cut5_point_index=g,original_raw_pair_index=j,original_raw_pair_local_index=g-int(raw['offsets'][j]),
                original_selected_cut5_row=int(ri),original_full_low_bundle_row=int(full[ri]),current_original_prefix_bundle_index=i,
                source_pts=pa,target_pts=pb,source_native_pair_pts=list(map(int,pair)),source_uv=p[k].tolist(),target_uv=q[k].tolist(),
                source_sigma_px=admission['original_sigma_xy'][full[ri]].tolist(),deep_LK_endpoint_difference_px=[float(ea[k]),float(eb[k])],
                independent_track_index=int(tl+near[k]),independent_deep_LK_FB_native_px=float(tracks['forward_backward_error_px'][tl+near[k]]),
                role_status='unknown_depth_original_FIT; RGB material inspection pending',
                material_role_not_inferred_from_model_residual=True)
            records.append(r);accepted.append(i)
        if accepted:groups.append(dict(original_raw_pair_index=j,source_pts=pa,target_pts=pb,candidate_count=len(accepted)))
    if not records:raise ValueError('No source-only original-query overlap candidates near requested PTS')
    targets=[a.target_pts-7680,a.target_pts,a.target_pts+7680]
    picked=[]
    for target in targets:
        group=min((g for g in groups if g not in picked),key=lambda g:abs(g['source_pts']-target));picked.append(group)
    selected_records=[r for r in records if r['original_raw_pair_index']in {g['original_raw_pair_index']for g in picked}]
    all_points=[]
    for path in a.state:
        z=arrays(path);identity=read(path.parent/'state_identity.json');assert sha(identity['camera_path'])==identity['camera_sha256']
        cameras=arrays(identity['camera_path']);camera_lookup={int(p):i for i,p in enumerate(cameras['exposure_representative_pts'])}
        provider,world,proof=current_complete_source_geometry(path,a.auxiliary)
        course,cq,_=provider.current.print_course_binding(world);bottom,top,*_=course.sample_perimeter(cq,with_jacobian=False)
        bottom=np.asarray(bottom);bottom=bottom[:-1]if np.array_equal(bottom[0],bottom[-1])else bottom
        xy=bottom[:,:2];edges=np.roll(xy,-1,axis=0)-xy;ll=np.einsum('ni,ni->n',edges,edges)
        weights=z['profile__weights_3'];assert weights.shape==(435185,2)
        values=[]
        for group in picked:
            rr=[r for r in selected_records if r['original_raw_pair_index']==group['original_raw_pair_index']]
            p,q=np.asarray([r['source_uv']for r in rr]),np.asarray([r['target_uv']for r in rr]);idx=[r['current_original_prefix_bundle_index']for r in rr]
            ca,cb=[cameras['camera25'][camera_lookup[g]]for g in (group['source_pts'],group['target_pts'])]
            points=current_far_board_points(ca,p,bottom);t=np.clip(np.einsum('nsi,si->ns',points[:,None,:2]-xy[None,:,:],edges)/ll,0.,1.)
            dist=np.linalg.norm(points[:,None,:2]-(xy[None,:,:]+t[:,:,None]*edges),axis=2);facet=np.argmin(dist,axis=1)
            normals=np.c_[edges[facet,1],-edges[facet,0],np.zeros(len(p))]/np.sqrt(ll[facet,None]);planes=np.c_[normals,-np.einsum('ni,ni->n',normals[:,:2],xy[facet])]
            common=dict(a=p,b=q,fa=np.exp(ca[6]),fb=np.exp(cb[6]),ra=cv2.Rodrigues(ca[3:6])[0],rb=cv2.Rodrigues(cb[3:6])[0],
                ca=ca[:3],cb=cb[:3],pp=ca[8:10],k1=ca[11],sigma=np.asarray([r['source_sigma_px']for r in rr]),profile_weight=weights[idx],
                jacobian=False,da=None,db=None,dca_shared=None,dcb_shared=None,dca_view=None,dcb_view=None,
                minimum_source_depth_scene_units=1./provider.scene_unit_m,schur=True,return_trace=True)
            old,_,trace=native_motion(**common,planes=np.zeros((len(p),4)))
            typed,_,typed_trace=native_motion(**common,planes=planes)
            pred,depth,valid=project_central_valid_full_brown(cb,points);assert np.all(valid)
            assert np.max(abs(typed.reshape(-1,2)-(pred-q)))<1e-6
            axial=np.divide(1.-trace[:,1],trace[:,1],out=np.full(len(trace),np.inf),where=trace[:,1]!=0)*provider.scene_unit_m
            typed_axial=np.divide(1.-typed_trace[:,1],typed_trace[:,1])*provider.scene_unit_m
            for k,r in enumerate(rr):values.append(dict(current_original_prefix_bundle_index=r['current_original_prefix_bundle_index'],
                source_pts=r['source_pts'],target_pts=r['target_pts'],untyped_residual_px=old.reshape(-1,2)[k].tolist(),
                counterfactual_current_far_board_residual_px=typed.reshape(-1,2)[k].tolist(),
                untyped_axial_depth_working_m=float(axial[k])if np.isfinite(axial[k])else None,
                unknown_infinite_depth_endpoint=bool(not np.isfinite(axial[k])),
                unknown_profile_at_engineering_1m_floor=bool(np.isclose(axial[k],1.,atol=1e-9,rtol=0.)),
                counterfactual_current_far_board_axial_depth_working_m=float(typed_axial[k]),current_geometry_facet_index=int(facet[k]),
                counterfactual_world_point_working_m=(points[k]*provider.scene_unit_m).tolist()))
        all_points.append(dict(state=str(path),container_sha256=sha(path),q_sha256=identity['q_sha256'],world_proof=proof,
            point_values=values,untyped_norm_px=stats([np.linalg.norm(r['untyped_residual_px'])for r in values]),
            counterfactual_current_far_board_norm_px=stats([np.linalg.norm(r['counterfactual_current_far_board_residual_px'])for r in values])))
    receipt=[]
    if a.reuse_native_RGB:
        previous=read(a.reuse_native_RGB)
        assert previous['selected_three_timepair_records']==selected_records
        receipt=previous['representative_source_RGB']
        for group in receipt:
            for frame in group['source_RGB']:assert sha(frame['path'])==frame['sha256']
        assert [g['original_raw_pair_index']for g in receipt]==[g['original_raw_pair_index']for g in picked]
    with av.open(retirement['source_video'])as container:
        stream=container.streams.video[0];stream.thread_count=1;assert str(stream.time_base)=='1/15360'
        for group in ([]if a.reuse_native_RGB else picked):
            rr=[r for r in selected_records if r['original_raw_pair_index']==group['original_raw_pair_index']]
            wanted={group['source_pts'],group['target_pts']};frames={};container.seek(min(wanted),stream=stream,backward=True,any_frame=False)
            for frame in container.decode(stream):
                if frame.pts is not None and frame.pts>max(wanted):break
                if frame.pts in wanted:frames[frame.pts]=frame.to_ndarray(format='bgr24')
                if set(frames)==wanted:break
            if set(frames)!=wanted:raise ValueError('Exact source RGB PTS unavailable')
            files=[];views=[]
            for pts,key in ((group['source_pts'],'source_uv'),(group['target_pts'],'target_uv')):
                image=frames[pts];out=a.output/f'raw_native_{pts}.png';assert cv2.imwrite(str(out),image)
                files.append(dict(actual_pts=pts,path=str(out),sha256=sha(out),decoded_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
                view=image.copy()
                for r in rr:
                    at=tuple(np.rint(r[key]).astype(int));cv2.circle(view,at,6,(40,255,40),2)
                    cv2.putText(view,str(r['current_original_prefix_bundle_index']),(at[0]+6,at[1]-7),cv2.FONT_HERSHEY_SIMPLEX,.4,(40,255,40),1)
                views.append(cv2.resize(view,(960,540)))
            overview=a.output/f'pair{group["original_raw_pair_index"]}_source_only_candidates.png';assert cv2.imwrite(str(overview),np.hstack(views))
            receipt.append(dict(**group,source_RGB=files,overview=str(overview),visual_material_inspection_status='pending'))
    report=dict(schema='rinkspace.original-low-FIT-near-Ad4-source-only-material-and-depth-audit.v1',target_native_pts=a.target_pts,time_base='1/15360',
        source_video_sha256=retirement['source_video_sha256'],source_ledger=dict(path=str(ADMISSION),sha256=sha(ADMISSION)),
        raw_motion=dict(path=str(RAW),sha256=sha(RAW)),selected_original_rows=dict(path=str(SELECTED),sha256=sha(SELECTED)),
        aliases_authority=retirement['source_actual_exposure_alias_authority'],independent_v2_tracks=dict(path=str(TRACKS),sha256=sha(TRACKS)),
        source_only_gate='exact original both endpoints within 2px of independent v2 tracks; existing people boxes with conservative padding; native y<800 only candidate scope',
        all_nearby_original_query_candidates=len(records),selected_three_timepair_records=selected_records,representative_source_RGB=receipt,
        state_diagnostics=all_points,original_active_prefix_UV_PTS_sigma_exact=True,source_roles_not_modified=True,source_scalar_rows_added=0,
        whole_original435185_not_reclassified=True,all_candidates_pending_actual_RGB_material_inspection=True,
        full_programme_graph_or_J_or_solver_or_video_not_started=True,accuracy_accepted=False)
    out=a.output/'original_low_Ad4_nearby_source_only_depth_audit_v1.json';out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(path=str(out),sha256=sha(out),nearby_candidates=len(records),selected_queries=len(selected_records),groups=picked,
        state_values=[{k:s[k]for k in ('q_sha256','untyped_norm_px','counterfactual_current_far_board_norm_px')}for s in all_points])),flush=True)
if __name__=='__main__':main()
