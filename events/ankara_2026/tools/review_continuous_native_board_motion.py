"""Continuous same-cut, adjacent-real-exposure RGB board motion diagnostic.

Extraction fixes only source pixel evidence. Review reuses the production
finite-board ray transfer on each immutable scene/camera, without a full graph.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
if 'review' in sys.argv:
    from calibration.solver_runtime import install
    install()
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import av
import cv2
import numpy as np
from calibration.program_tracking.render_result import PersonMasks,DEFAULT_MASKS,DEFAULT_PEOPLE
spec=importlib.util.spec_from_file_location('existing_nine_cut_motion',ROOT/'events/ankara_2026/tools/review_nine_shot_source_motion.py')
existing=importlib.util.module_from_spec(spec);spec.loader.exec_module(existing)
cv2.setNumThreads(1)


def adjacent_tracks(ga,gb,mask,*,flow_pyramid_levels=5):
    """Actual source FB tracks with a declared coarse-to-fine LK chart.

    The former level-2 chart produced mutually reversible wrong-letter matches
    during fast pan. Level 5, the existing independent probe's 31-pixel window
    and convergence settings are verified on those original source pairs.
    No camera prediction, world geometry or model residual selects source UV.
    """
    levels=int(flow_pyramid_levels)
    if levels<0 or levels>8:raise ValueError('LK pyramid levels must be between 0 and 8')
    receipt=dict(detected_corner_count=0,forward_status_rejected=0,backward_status_rejected=0,
        source_FB_gate_rejected=0,source_native_domain_rejected=0,source_navigation_rejected=0,
        kept_source_tracks=0,flow_pyramid_levels=levels,flow_window_native_px=31,
        flow_max_iterations=40,flow_epsilon=.005,source_FB_gate_native_px=.8,
        source_selected_by_model=False)
    ys,xs=np.nonzero(mask)
    if not len(xs):return np.empty((0,2)),np.empty((0,2)),np.empty(0),'source_board_navigation_missing',receipt
    lo,hi=int(ys.min()),int(ys.max())+1
    p=cv2.goodFeaturesToTrack(ga[lo:hi],maxCorners=80,qualityLevel=.025,minDistance=13,mask=mask[lo:hi],blockSize=7)
    if p is None:return np.empty((0,2)),np.empty((0,2)),np.empty(0),'no_actual_board_texture',receipt
    receipt['detected_corner_count']=len(p)
    p[:,:,1]+=lo
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.005)
    q,s,_=cv2.calcOpticalFlowPyrLK(ga,gb,p,None,winSize=(31,31),maxLevel=levels,criteria=criteria)
    z,t,_=cv2.calcOpticalFlowPyrLK(gb,ga,q,None,winSize=(31,31),maxLevel=levels,criteria=criteria)
    p=p.reshape(-1,2);q=q.reshape(-1,2);fb=np.linalg.norm(p-z.reshape(-1,2),axis=1)
    native=np.isfinite(q).all(1)&(q[:,0]>1)&(q[:,0]<1918)&(q[:,1]>1)&(q[:,1]<1078)
    valid=(s.ravel()!=0)&(t.ravel()!=0)&np.isfinite(fb)&(fb<.8)&native
    qi=np.rint(np.nan_to_num(q)).astype(int);qi[:,0]=qi[:,0].clip(0,1919);qi[:,1]=qi[:,1].clip(0,1079)
    navigation=mask[qi[:,1],qi[:,0]]!=0
    receipt.update(forward_status_rejected=int((s.ravel()==0).sum()),
        backward_status_rejected=int((t.ravel()==0).sum()),
        source_FB_gate_rejected=int((~np.isfinite(fb)|(fb>=.8)).sum()),
        source_native_domain_rejected=int((~native).sum()),
        source_navigation_rejected=int((~navigation).sum()),
        rejection_reason_counts_can_overlap=True)
    valid&=navigation
    receipt['kept_source_tracks']=int(valid.sum())
    kind='source_static_board_RGB_FB'if valid.any()else'actual_board_texture_all_source_FB_or_navigation_rejected'
    return p[valid],q[valid],fb[valid],kind,receipt


def extract(a):
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'adjacent_real_board_tracks.npz').exists():raise FileExistsError(out)
    camera=np.load(a.camera_archive,allow_pickle=False)
    actual=np.unique(camera['exposure_representative_pts'])
    shots=json.loads((ROOT/'calibration/program_tracking/source_shots.json').read_text(encoding='utf8'))['shots'][1:10]
    mapping={int(p):i for i,s in enumerate(shots,1) for p in actual if int(s['start_pts'])<=p<int(s['end_pts'])}
    wanted=set(mapping);people=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE)
    pa=[];pb=[];cut=[];offset=[0];pall=[];qall=[];fall=[];statuses=[];source_reads=[];missing_masks=0
    previous=None;received=[];started=time.perf_counter()
    with av.open(str(a.source)) as container:
        stream=container.streams.video[0];stream.thread_count=2
        container.seek(min(wanted)-15360,stream=stream,backward=True,any_frame=False)
        for f in container.decode(stream):
            if f.pts is not None and f.pts>max(wanted):break
            if f.pts not in wanted:continue
            if str(f.time_base)!='1/15360':raise ValueError('Unexpected source clock')
            rgb=f.to_ndarray(format='bgr24');gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
            mask,status=people.get(f.pts,require_sam2=False)
            if mask is None:missing_masks+=1;mask=np.zeros(gray.shape,bool)
            received.append(f.pts)
            if previous is not None and mapping[previous[0]]==mapping[f.pts]:
                old,oldrgb,oldgray,oldmask=previous
                nav,receipt=existing.make_mask(oldrgb,rgb,oldmask,mask,'board')
                p,q,fb,kind,source_read=adjacent_tracks(oldgray,gray,nav,flow_pyramid_levels=a.flow_pyramid_levels)
                pa.append(old);pb.append(f.pts);cut.append(mapping[f.pts]);pall.append(p);qall.append(q);fall.append(fb)
                offset.append(offset[-1]+len(p));statuses.append(kind);source_reads.append(source_read)
            previous=(f.pts,rgb,gray,mask)
            if len(received)%500==0:print(json.dumps(dict(exposures=len(received),total=len(wanted),seconds=time.perf_counter()-started)),flush=True)
    if set(received)!=wanted:raise ValueError(('Missing actual source exposure',sorted(wanted-set(received))))
    reason_keys=('detected_corner_count','forward_status_rejected','backward_status_rejected',
        'source_FB_gate_rejected','source_native_domain_rejected','source_navigation_rejected')
    reason_arrays={f'source_{key}':np.asarray([r[key]for r in source_reads],np.int16)for key in reason_keys}
    np.savez_compressed(out/'adjacent_real_board_tracks.npz',source_pts=np.asarray(pa,np.int64),target_pts=np.asarray(pb,np.int64),
        shot_index=np.asarray(cut,np.int16),offsets=np.asarray(offset,np.int64),uv0=np.concatenate(pall).astype(np.float32),
        uv1=np.concatenate(qall).astype(np.float32),forward_backward_error_px=np.concatenate(fall).astype(np.float32),**reason_arrays)
    counts=np.diff(offset);percut=[]
    for number in range(1,10):
        indices=np.flatnonzero(np.asarray(cut)==number)
        percut.append(dict(shot_index=number,pairs=len(indices),supported_six_or_more=int((counts[indices]>=6).sum()),
            source_tracks=int(counts[indices].sum()),missing_or_sparse_pairs=int((counts[indices]<6).sum())))
    report=dict(schema='rinkspace.continuous-source-only-adjacent-native-board-RGB.v2',source_video=str(a.source),
        source_sha256_from_existing_verified_ledger='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891',
        time_base='1/15360',native_hw=[1080,1920],actual_exposures=len(received),adjacent_pairs=len(pa),
        true_exposure_clock_archive=dict(path=str(a.camera_archive),sha256=existing.probe.sha(a.camera_archive)),
        source_UV_selected_by_model=False,source_only_GFTT_maximum=80,LK_pyramid_levels=a.flow_pyramid_levels,
        LK_window_native_px=31,LK_max_iterations=40,LK_epsilon=.005,FB_gate_native_px=.8,
        previous_level2_false_letter_matches_preserved=True,
        extraction_script=dict(path=str(Path(__file__).resolve()),sha256=existing.probe.sha(Path(__file__))),
        no_cross_cut_pairs=True,people_mask_union_dilation_px=41,source_board_navigation='existing_native_RGB_cap_and_yellow_foot_strip',
        decoded_native_pts_sha256=hashlib.sha256(np.asarray(received,np.int64).tobytes()).hexdigest(),
        missing_people_masks=missing_masks,pair_status=statuses,per_shot=percut,
        source_rejection_counts_per_pair_saved_in_point_archive=True,
        source_rejection_reason_totals={key:int(reason_arrays[f'source_{key}'].sum())for key in reason_keys},
        rejection_reason_counts_can_overlap=True,
        source_only_diagnostic_not_new_solver_FIT=True,accuracy_accepted=False,wall_seconds=time.perf_counter()-started)
    if a.parent_source_receipt is not None:
        report['preserved_parent_v1_source_receipt']=dict(path=str(a.parent_source_receipt.resolve()),sha256=existing.probe.sha(a.parent_source_receipt))
    (out/'source_extraction_receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(complete=True,output=str(out),seconds=report['wall_seconds'],per_shot=percut)),flush=True)


def review(a):
    from scripts.diagnostics.verify_current_native_primitives_20261007 import state_camera_adapter
    from rinkspace.vision_camera_initialization import current_complete_source_geometry
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate.native_board_track_source import current_far_board_points
    from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown
    # Each NPZ member is decompressed on access. Read the immutable arrays once,
    # rather than reopening/decompressing the full programme for every pair.
    with np.load(a.tracks/'adjacent_real_board_tracks.npz',allow_pickle=False) as archive:
        source={key:archive[key] for key in ('source_pts','target_pts','shot_index','offsets','uv0','uv1')}
    boxes=json.loads(Path(DEFAULT_PEOPLE).read_text(encoding='utf8'))
    people_boxes={int(r['pts']):r['boxes']for r in boxes['frames']}
    provider,world,proof=current_complete_source_geometry(a.state,a.auxiliary)
    course,cq,_=provider.current.print_course_binding(world);border=np.asarray(course.sample_perimeter(cq)[0],float)
    identity=json.loads((a.state.parent/'state_identity.json').read_text(encoding='utf8'))
    camera_path=Path(identity['camera_path'])
    with np.load(camera_path,allow_pickle=False) as archive:
        cameras={key:archive[key] for key in ('pts','camera25')}
    if existing.probe.sha(camera_path)!=identity['camera_sha256']:raise ValueError('Immutable camera changed')
    ix={int(v):i for i,v in enumerate(cameras['pts'])};rows=[];started=time.perf_counter()
    for i,(pa,pb) in enumerate(zip(source['source_pts'],source['target_pts'])):
        lo,hi=map(int,source['offsets'][i:i+2]);row=dict(source_pts=int(pa),target_pts=int(pb),shot_index=int(source['shot_index'][i]),source_tracks=hi-lo)
        if hi==lo:row['missing_source_support']=True;rows.append(row);continue
        kept=np.ones(hi-lo,bool)
        if a.conservative_people_boxes:
            for stamp,key in [(pa,'uv0'),(pb,'uv1')]:
                uv=source[key][lo:hi]
                for box in people_boxes.get(int(stamp),[]):
                    x0,y0,x1,y1=np.asarray(box[:4])*2
                    pad_x=max(48.,.4*(x1-x0));pad_y=48.
                    kept&=~((uv[:,0]>=x0-pad_x)&(uv[:,0]<=x1+pad_x)&(uv[:,1]>=y0-pad_y)&(uv[:,1]<=y1+pad_y))
        row.update(original_source_tracks=hi-lo,source_tracks=int(kept.sum()),source_people_box_excluded=int((~kept).sum()))
        if not kept.any():row['missing_source_support']=True;rows.append(row);continue
        ca,cb=[cameras['camera25'][ix[int(v)]]for v in (pa,pb)]
        try:
            xyz=current_far_board_points(ca,source['uv0'][lo:hi][kept],border)
            predicted,_,valid=project_central_valid_full_brown(cb,xyz)
            delta=predicted-source['uv1'][lo:hi][kept];norm=np.linalg.norm(delta,axis=1)
            observed=source['uv1'][lo:hi][kept]-source['uv0'][lo:hi][kept]
            motion_energy=float(np.sum(observed*observed))
            dt=(int(pb)-int(pa))/15360.
            row.update(valid_forward_points=int(valid.sum()),error_native_px=existing.probe.stats(norm),
                signed_median_xy_px=np.median(delta,axis=0).tolist(),anchor_Z_scene_units_minmax=[float(xyz[:,2].min()),float(xyz[:,2].max())],
                source_dt_seconds=dt,
                actual_motion_median_xy_px=np.median(observed,axis=0).tolist(),
                actual_motion_median_speed_px_per_second=float(np.median(np.linalg.norm(observed,axis=1))/dt),
                residual_dot_actual_motion_px2=float(np.sum(delta*observed)),
                actual_motion_energy_px2=motion_energy,
                relative_excess_along_actual_motion=float(np.sum(delta*observed)/motion_energy)if motion_energy else None,
                source_median_uv=np.median(source['uv0'][lo:hi][kept],axis=0).tolist(),
                source_target_focal_px=np.exp([ca[6],cb[6]]).tolist())
        except ValueError as e:row['current_model_domain_failure']=str(e)
        rows.append(row)
    summaries=[]
    for number in range(1,10):
        relevant=[r for r in rows if r['shot_index']==number]
        supported=[r for r in relevant if r['source_tracks']>=6 and 'error_native_px'in r]
        worst=max(supported,key=lambda r:r['error_native_px']['median'])if supported else None
        summaries.append(dict(shot_index=number,all_adjacent_pairs=len(relevant),supported_pairs=len(supported),
            source_tracks=sum(r['source_tracks']for r in supported),missing_or_sparse_pairs=len(relevant)-len(supported),worst=worst,
            pair_median_native_px=existing.probe.stats([r['error_native_px']['median']for r in supported])))
    report=dict(schema='rinkspace.continuous-native-board-current-finite-transfer-review.v1',state=str(a.state),state_identity=identity,
        geometry_proof=proof,source_tracks_path=str(a.tracks/'adjacent_real_board_tracks.npz'),source_tracks_sha256=existing.probe.sha(a.tracks/'adjacent_real_board_tracks.npz'),
        current_finite_board_and_actual_lever=True,source_UV_and_navigation_not_changed=True,source_only_diagnostic_not_FIT=True,
        source_people_box_qualification=a.conservative_people_boxes,
        source_people_box_padding_native_px=dict(horizontal='max(48, 0.4*native bbox width)',vertical=48),
        limitations=['Adjacent relative motion does not prove absolute geometry.', 'Correlated tracks and motion blur are not calibrated measurement uncertainty.',
            'Sparse or absent board navigation remains missing evidence; model-domain failures are reported.'],
        accuracy_accepted=False,per_shot=summaries,pairs=rows,wall_seconds=time.perf_counter()-started)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(output=str(a.output),seconds=report['wall_seconds'],per_shot=summaries)),flush=True)


def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='mode',required=True)
    e=sp.add_parser('extract');e.add_argument('--source',required=True,type=Path);e.add_argument('--camera-archive',required=True,type=Path);e.add_argument('--output',required=True,type=Path)
    e.add_argument('--flow-pyramid-levels',type=int,default=5,help='Actual source-only LK pyramid depth; verified deep-LK default is 5')
    e.add_argument('--parent-source-receipt',type=Path,help='Preserved earlier diagnostic receipt; provenance only, never input UV')
    r=sp.add_parser('review');r.add_argument('--tracks',required=True,type=Path);r.add_argument('--state',required=True,type=Path);r.add_argument('--auxiliary',required=True,type=Path);r.add_argument('--output',required=True,type=Path)
    r.add_argument('--conservative-people-boxes',action='store_true',help='Source-only conservative box exclusion for known SAM2 fingers/arms omissions')
    a=p.parse_args();extract(a)if a.mode=='extract'else review(a)


if __name__=='__main__':main()
