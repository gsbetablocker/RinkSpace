"""Freeze fixed source RGB board tracks with the original train/HELD split."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
from calibration.program_tracking.render_result import decode_selected
import cv2
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def spatial_subset(points,indices,maximum):
    if len(indices)<=maximum:return indices
    p=points[indices].astype(float)
    span=np.maximum(np.ptp(p,axis=0),1.)
    z=(p-p.min(0))/span
    # Deterministic farthest-point source-image coverage, no model input.
    chosen=[int(np.lexsort((z[:,1],z[:,0]))[0])]
    distance=np.sum((z-z[chosen[0]])**2,axis=1)
    for _ in range(maximum-1):
        j=int(np.argmax(distance));chosen.append(j)
        distance=np.minimum(distance,np.sum((z-z[j])**2,axis=1));distance[chosen]=-1.
    return indices[np.asarray(chosen)]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--only-pair',type=int,nargs=2)
    parser.add_argument('--source-review',type=Path,help='Explicit source-only material review dispositions keyed by source,target PTS')
    parser.add_argument('--prepare-source-review-only',action='store_true',help='Decode candidate references without admitting any unseen material')
    args=parser.parse_args()
    audit=json.loads(args.audit.read_text(encoding='utf8'));tracks_path=args.audit.parent/'actual_RGB_tracks.npz'
    tracks=np.load(tracks_path,allow_pickle=False)
    ledger={}if not args.source_review else json.loads(args.source_review.read_text(encoding='utf8'))
    selected=[];excluded=[]
    for index,pair in enumerate(audit['pairs']):
        pa,pb=pair['source_pts'],pair['target_pts']
        if args.only_pair and [pa,pb]!=args.only_pair:continue
        row=next((r for r in audit['experiments']if r['material']=='board' and r['source_pts']==pa and r['target_pts']==pb),None)
        key=f'pair{index:02d}_board'
        if row is None or key+'_source'not in tracks:
            excluded.append(dict(source_pts=pa,target_pts=pb,reason=None if row is None else row.get('failure','No preserved board source tracks')));continue
        p,q=tracks[key+'_source'],tracks[key+'_target'];test=tracks[key+'_held'];keep=tracks[key+'_source_static_keep'];fb=tracks[key+'_forward_backward_error']
        valid=keep&(fb<.8)&np.isfinite(p).all(1)&np.isfinite(q).all(1)
        train=spatial_subset(p,np.flatnonzero(valid&~test),30)
        held=np.flatnonzero(valid&test)
        source_review=ledger.get(f'{pa},{pb}',{})
        if [pa,pb]==[229694208,229701888]:source_review=dict(status='actual_native_pair_independently_viewed_static_Ad1_print_material',actual_pixels_viewed=True,source_material='fixed_board_Ad1_ALL_SKATING_ONE_APP_ink_texture',printed_material_not_overlay_or_reflection=True)
        if len(train)<6:
            excluded.append(dict(source_pts=pa,target_pts=pb,reason='Fewer than six independent RGB-static FIT tracks',fit_count=len(train),held_count=len(held)));continue
        if not source_review.get('actual_pixels_viewed') and not args.prepare_source_review_only:
            excluded.append(dict(source_pts=pa,target_pts=pb,reason='Source material review pending; no candidate admitted merely by RGB navigation',fit_count=len(train),held_count=len(held)));continue
        ids=np.sort(np.r_[train,held])
        selected.append(dict(id=f'native_board_motion_cut{pair["shot_index"]}_{pa}_{pb}',source_pts=pa,target_pts=pb,shot_index=pair['shot_index'],shot_id=pair['shot_id'],camera_id=pair['camera_id'],
            source_video_sha256=audit['source_sha256_from_verified_ledger'],time_base='1/15360',native_hw=[1080,1920],image_domain='native_brown_pixels',
            uv0=p[ids].astype(float).tolist(),uv1=q[ids].astype(float).tolist(),split=['HELD'if test[i]else 'FIT'for i in ids],
            original_source_track_indices=ids.tolist(),forward_backward_error_px=fb[ids].astype(float).tolist(),fit_point_count=len(train),held_point_count=len(held),
            original_every_third_HELD_not_promoted=True,source_space_subsampling=dict(maximum_fit_points=30,method='deterministic_source_only_farthest_point_coverage'),
            source_material_review=source_review,source_tracking_method=row['source_tracker'],source_navigation=row['source_navigation'],
            independent_held_image_H_error_px=row['independent_good_held_error_px'],
            original_tracking_evidence=dict(path=str(tracks_path),sha256=sha(tracks_path),keys=[key+'_source',key+'_target',key+'_held',key+'_source_static_keep',key+'_forward_backward_error']),
            source_identity_not_selected_by_model_or_projection=True,physical_board_plane_owner='current_actual_finite_board_course_qualified_source_ray',
            fixed_raw_source_UV=True))
    native_dir=args.output.parent/'native_board_motion_source_frames';native_dir.mkdir(parents=True,exist_ok=True)
    pts=sorted({p[k]for p in selected for k in ('source_pts','target_pts')})
    if pts:
        for stamp,image in decode_selected(Path(audit['source_video']),pts):
            expected=next(r['native_bgr_sha256']for r in audit['source_receipts']if r['pts']==stamp)
            if hashlib.sha256(image.tobytes()).hexdigest()!=expected:raise ValueError(('Original RGB evidence changed',stamp))
            existing=args.audit.parent/f'maximum_candidate_native_{stamp}.png'
            target=existing if existing.exists() else native_dir/f'native_{stamp}.png'
            if not target.exists():
                if not cv2.imwrite(str(target),image):raise OSError(target)
            for pair in selected:
                for key,ref in [('source_pts','source_rawref'),('target_pts','target_rawref')]:
                    if pair[key]==stamp:pair[ref]=dict(path=str(target),png_sha256=sha(target),native_bgr_sha256=expected,source_pts=stamp,time_base='1/15360')
    report=dict(schema='rinkspace.source-fixed-native-board-motion-source-review-only.v1' if args.prepare_source_review_only else 'rinkspace.source-fixed-native-board-motion-pairs.v1',source_video_sha256=audit['source_sha256_from_verified_ledger'],time_base='1/15360',pairs=selected,
        source_audit=dict(path=str(args.audit),sha256=sha(args.audit)),source_review_ledger=None if not args.source_review else str(args.source_review),
        source_trace_HELD_rows_unchanged=True,UV_sigma_or_existing_source_not_changed=True,candidate_only=True,solver_admitted=False,model_parameter_or_camera_free_degrees_added=False,
        excluded_pairs=excluded,fit_points=sum(p['fit_point_count']for p in selected),held_points=sum(p['held_point_count']for p in selected),source_review_preparation_only=args.prepare_source_review_only,accuracy_accepted=False)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(output=str(args.output),pairs=len(selected),fit=report['fit_points'],held=report['held_points'],excluded=len(excluded)),ensure_ascii=False))

if __name__=='__main__':main()
