"""Source-only reinspection of three continuous-board diagnostic outliers.

Six immutable raw images are reused. Independently viewed static print ROIs,
existing deeper LK and SIFT-initialized LK/H helpers cross-check the shallow
diagnostic tracks and the preserved original solver motion-source archive.
No solver source, weight, camera, configuration or video is changed.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'calibration/program_tracking')]
sys.dont_write_bytecode = True
from calibration.solver_runtime import install
install()
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np

spec = importlib.util.spec_from_file_location('existing_nine_motion', ROOT/'events/ankara_2026/tools/review_nine_shot_source_motion.py')
existing = importlib.util.module_from_spec(spec); spec.loader.exec_module(existing)
probe = existing.probe
cv2.setNumThreads(1)

CASES = ((4, 229904384, 229904640, [10, 480, 740, 785]),
         (9, 231126784, 231127040, [5, 446, 480, 645]),
         (6, 230509824, 230510080, [1060, 340, 1880, 575]))


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k:serial(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [serial(v) for v in value]
    return value


def subset(uv, roi):
    x0,y0,x1,y1 = roi
    return (uv[:,0]>=x0)&(uv[:,0]<x1)&(uv[:,1]>=y0)&(uv[:,1]<y1)


def print_summary(p, q):
    return dict(points=len(p), displacement_native_px=probe.stats(np.linalg.norm(q-p,axis=1)),
                median_displacement_xy_native_px=np.median(q-p,axis=0).tolist() if len(p) else None)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--tracks', required=True, type=Path)
    cli.add_argument('--state', required=True, type=Path)
    cli.add_argument('--auxiliary', required=True, type=Path)
    cli.add_argument('--output', required=True, type=Path)
    opt = cli.parse_args()
    out = opt.output
    out.mkdir(parents=True, exist_ok=True)
    report_path = out/'source_and_model_three_pairs_audit_v1.json'
    if report_path.exists(): raise FileExistsError(report_path)
    source = np.load(opt.tracks/'adjacent_real_board_tracks.npz', allow_pickle=False)
    boxes = json.loads(Path(existing.DEFAULT_PEOPLE).read_text(encoding='utf8'))
    people = {int(r['pts']):r['boxes'] for r in boxes['frames']}
    rows = []; experiment_arrays = {}
    for cut,pa,pb,roi in CASES:
        a,b = [cv2.imread(str(out/f'native_{p}.png')) for p in (pa,pb)]
        if a is None or b is None or a.shape != (1080,1920,3) or b.shape != a.shape:
            raise ValueError('The six decoded raw native sources are required')
        ix = np.flatnonzero((source['source_pts']==pa)&(source['target_pts']==pb))
        if len(ix)!=1: raise ValueError('Exact preserved diagnostic pair required')
        lo,hi = map(int,source['offsets'][int(ix[0]):int(ix[0])+2])
        p,q = source['uv0'][lo:hi],source['uv1'][lo:hi]
        kept = np.ones(len(p),bool)
        for stamp,uv in ((pa,p),(pb,q)):
            for box in people.get(stamp,[]):
                x0,y0,x1,y1=np.asarray(box[:4])*2
                px=max(48.,.4*(x1-x0));py=48.
                kept &= ~((uv[:,0]>=x0-px)&(uv[:,0]<=x1+px)&(uv[:,1]>=y0-py)&(uv[:,1]<=y1+py))
        ids=np.flatnonzero(kept);p,q=p[kept],q[kept]
        mask=np.zeros(a.shape[:2],np.uint8);x0,y0,x1,y1=roi;mask[y0:y1,x0:x1]=255
        row=dict(shot_index=cut,source_pts=pa,target_pts=pb,time_base='1/15360',
            source_time_seconds=[pa/15360,pb/15360],source_pixels_original_sha256=[hashlib.sha256(im.tobytes()).hexdigest()for im in (a,b)],
            manual_static_print_ROI_native_xyxy=roi,
            ROI_selection='Actual original RGB viewed first; static printed board clear of foreground and scoreboard; no model or residual used',
            preserved_shallow_LK=print_summary(p,q),preserved_shallow_point_ids=ids.tolist(),independent_source=[])
        experiment_arrays[f'c{cut}_shallow_p']=p;experiment_arrays[f'c{cut}_shallow_q']=q
        Hs=[]
        for name,helper in (('existing_deeper_LK_maxLevel5',probe.track_pair),
                            ('existing_mutual_SIFT_initialized_LK',existing.sift_initialized_source_tracks)):
            try:
                pp,qq,fb,H,test,inlier,residual=helper(a,b,mask)
                trusted=test&inlier
                row['independent_source'].append(dict(method=name,
                    summary=print_summary(pp[inlier],qq[inlier]),H=H.tolist(),
                    H_HELD_native_error=probe.stats(residual[test]),
                    H_HELD_inlier_native_error=probe.stats(residual[trusted]),
                    source_FB_error=probe.stats(fb[inlier]),
                    H_against_preserved_shallow_LK_native_error=probe.stats(np.linalg.norm(probe.hom_apply(H,p)-q,axis=1))))
                Hs.append((name,H))
                for label,value in dict(p=pp,q=qq,fb=fb,H=H,test=test,inlier=inlier,residual=residual).items():
                    experiment_arrays[f'c{cut}_{name}_{label}']=value
            except RuntimeError as error:
                row['independent_source'].append(dict(method=name,failure=str(error)))
        old_path=Path(f'D:/CodexData/RinkSpace_program_tracking_tmp/motion/motion_shot_{cut:02d}.npz')
        old=np.load(old_path,allow_pickle=False);pp=old['pair_pts'];indices=np.flatnonzero((pp[:,0]==pa)&(pp[:,1]==pb))
        old_info=dict(path=str(old_path),sha256=probe.sha(old_path),exact_native_pair_indices=indices.tolist(),
            source_arrays_and_solver_constraints_not_modified=True)
        if len(indices):
            o0,o1=map(int,old['offsets'][int(indices[0]):int(indices[0])+2])
            op,oq=old['xy0'][o0:o1],old['xy1'][o0:o1]
            good=subset(op,roi)&subset(oq,roi)
            old_info.update(all_native_source_points=o1-o0,
                original_static_print_ROI_motion=print_summary(op[good],oq[good]),
                source_homography_inlier_count_in_ROI=int(np.sum(old['homography_inlier'][o0:o1][good])),
                source_FB_native_error=probe.stats(old['fb_error_px'][o0:o1][good]),
                independent_H_errors=[dict(method=name,error=probe.stats(np.linalg.norm(probe.hom_apply(H,op[good])-oq[good],axis=1)))for name,H in Hs])
            experiment_arrays[f'c{cut}_original_motion_ROI_p']=op[good]
            experiment_arrays[f'c{cut}_original_motion_ROI_q']=oq[good]
        row['original_solver_motion_source_archive']=old_info
        # A wide non-repeated glyph group provides an additional source-only
        # template check under the actual motion blur, without a camera prior.
        template_boxes={4:[120,555,655,705],9:[8,490,465,595],6:[1250,370,1790,480]}
        tx0,ty0,tx1,ty1=template_boxes[cut]
        pattern=cv2.cvtColor(a[ty0:ty1,tx0:tx1],cv2.COLOR_BGR2GRAY)
        sx0,sx1=max(0,tx0-180),min(1920,tx1+180)
        sy0,sy1=max(0,ty0-35),min(1080,ty1+35)
        search=cv2.cvtColor(b[sy0:sy1,sx0:sx1],cv2.COLOR_BGR2GRAY)
        correlation=cv2.matchTemplate(search,pattern,cv2.TM_CCOEFF_NORMED)
        _,peak,_,location=cv2.minMaxLoc(correlation)
        row['full_glyph_group_RGB_template']=dict(source_template_native_xyxy=template_boxes[cut],
            source_selected_target_search_native_xyxy=[sx0,sy0,sx1,sy1],
            translation_xy_native_px=[location[0]+sx0-tx0,location[1]+sy0-ty0],peak_normalized_correlation=float(peak),
            local_translation_diagnostic_not_world_or_exact_blur_GT=True)
        panels=[]
        crop=np.array([min(p[:,0].min(),q[:,0].min())-30,min(p[:,1].min(),q[:,1].min())-35,
                       max(p[:,0].max(),q[:,0].max())+30,max(p[:,1].max(),q[:,1].max())+35],int)
        crop[[0,2]]=crop[[0,2]].clip(0,1920);crop[[1,3]]=crop[[1,3]].clip(0,1080)
        cx0,cy0,cx1,cy1=crop
        for raw,uv in ((a,p),(b,q)):
            display=raw.copy()
            for identity,point in zip(ids,uv):
                at=tuple(np.rint(point).astype(int));cv2.circle(display,at,3,(0,255,255),1)
                cv2.putText(display,str(identity),(at[0]+4,at[1]-4),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),2)
                cv2.putText(display,str(identity),(at[0]+4,at[1]-4),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,255),1)
            panels.append(display[cy0:cy1,cx0:cx1])
        image_path=out/f'c{cut}_preserved_shallow_LK_points_source_only.png'
        cv2.imwrite(str(image_path),np.hstack(panels));row['source_only_points_image']=str(image_path)
        row['source_only_points_native_crop_xyxy']=crop.tolist();rows.append(row)
    # Model comparison occurs only after all independent source measurements
    # have been fixed above, against the same immutable actual camera archive.
    from rinkspace.vision_camera_initialization import current_complete_source_geometry
    provider,world,proof=current_complete_source_geometry(opt.state,opt.auxiliary)
    from structured_programme_source_candidate.native_board_track_source import current_far_board_points
    from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown
    course,cq,_=provider.current.print_course_binding(world);border=np.asarray(course.sample_perimeter(cq)[0],float)
    identity=json.loads((opt.state.parent/'state_identity.json').read_text(encoding='utf8'))
    cameras=np.load(identity['camera_path'],allow_pickle=False);indices={int(p):i for i,p in enumerate(cameras['pts'])}
    for row in rows:
        cut=row['shot_index'];pa,pb=row['source_pts'],row['target_pts']
        ca,cb=[cameras['camera25'][indices[p]]for p in (pa,pb)]
        row['actual_camera_exposure_aliases']=[int(cameras['exposure_representative_pts'][indices[p]])for p in (pa,pb)]
        row['view_state4_delta']=(cameras['view_state4'][indices[pb]]-cameras['view_state4'][indices[pa]]).tolist()
        comparisons=[]
        for method in ('shallow','existing_deeper_LK_maxLevel5','existing_mutual_SIFT_initialized_LK','original_motion_ROI'):
            pk=f'c{cut}_{method}_p';qk=f'c{cut}_{method}_q'
            if pk not in experiment_arrays:continue
            p,q=experiment_arrays[pk],experiment_arrays[qk]
            good=experiment_arrays.get(f'c{cut}_{method}_inlier',np.ones(len(p),bool))
            p,q=p[good],q[good]
            if not len(p):continue
            try:
                xyz=current_far_board_points(ca,p,border)
                predicted,depth,valid=project_central_valid_full_brown(cb,xyz)
                comparisons.append(dict(method=method,points=len(p),model_valid_forward=int(valid.sum()),
                    actual_observed=print_summary(p,q),actual_model_prediction=print_summary(p,predicted),
                    model_against_actual_source_native_error=probe.stats(np.linalg.norm(predicted-q,axis=1)),
                    signed_median_error_xy_px=np.median(predicted-q,axis=0).tolist()))
            except ValueError as error:comparisons.append(dict(method=method,model_domain_failure=str(error)))
        row['immutable_696f_model_comparisons']=comparisons
    clock=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/history/tracking_runs/source_cadence_comparison/O6j8L8TmfKo/full_programme_alignment/native60_to_official50.jsonl')
    wanted={p for _,pa,pb,_ in CASES for p in (pa,pb)};clock_rows=[]
    for line in clock.open(encoding='utf8'):
        record=json.loads(line)
        if int(record['full_pts'])in wanted:clock_rows.append(record)
    report=dict(schema='rinkspace.three-continuous-motion-outliers-independent-source-reinspection.v1',
        immutable_state=str(opt.state),geometry_proof=proof,
        preserved_diagnostic_source_archive=dict(path=str(opt.tracks/'adjacent_real_board_tracks.npz'),sha256=probe.sha(opt.tracks/'adjacent_real_board_tracks.npz')),
        original_source_clock_authority=dict(path=str(clock),sha256=probe.sha(clock)),clock_rows=clock_rows,pairs=rows,
        source_FIT_weights_roles_camera_q_changed=False,full_source_reextraction_or_video_run=False,
        agent_visual_source_reinspection_completed=True,accuracy_accepted=False)
    np.savez_compressed(out/'small_pair_source_comparisons.npz',**experiment_arrays)
    report_path.write_text(json.dumps(serial(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(serial(dict(report_path=str(report_path),sha256=probe.sha(report_path),pairs=rows)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
