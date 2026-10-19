"""Independent RGB motion audit of existing camera candidates across nine cuts.

Reuses the reviewed forward/backward tracker and physical transfer diagnostic.
Source masks use only native RGB, saved human masks and fixed UI rectangles.
No camera, projection, old UV or point residual selects source texture.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
from calibration.program_tracking.render_result import PersonMasks,DEFAULT_MASKS,DEFAULT_PEOPLE
import cv2
import numpy as np

spec=importlib.util.spec_from_file_location('original_rgb_motion',ROOT/'scripts/diagnostics/external_image_motion_probe_20261007.py')
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

def source_material_bounds(bgr,person):
    """Conservative global vertical bounds of directly visible foot strip/cap."""
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    usable=~person.copy()
    usable[:255]=False
    usable[935:]=False
    usable[24:260,20:575]=False
    usable[920:1020,1650:1900]=False
    # User-confirmed yellow strip is board material. Its lower side bounds ice.
    yellow=(hsv[:,:,0]>=8)&(hsv[:,:,0]<=45)&(hsv[:,:,1]>=30)&usable
    yellow&=(bgr[:,:,0].astype(int)+15<np.minimum(bgr[:,:,1],bgr[:,:,2]))
    ycount=yellow.sum(1)
    ypeak=int(ycount.argmax())
    if ycount[ypeak]<150:return dict(status='RGB_foot_strip_not_distributed',ice_ymin=800,board_range=None)
    band=np.zeros_like(yellow);band[max(255,ypeak-60):min(935,ypeak+61)]=yellow[max(255,ypeak-60):min(935,ypeak+61)]
    y,x=np.nonzero(band)
    if len(np.unique(x))<600:return dict(status='RGB_foot_strip_not_distributed',ice_ymin=800,board_range=None)
    bottom=int(np.percentile(y,99))+35
    bottom=min(1010,max(500,bottom))
    low=int(np.percentile(y,1))-20
    blue=(hsv[:,:,0]>=90)&(hsv[:,:,0]<=135)&(hsv[:,:,1]>=60)&usable
    blue[low:]=False
    bluecount=blue.sum(1)
    bp=int(bluecount.argmax())
    if bluecount[bp]<300:board=None
    else:
        yy,xx=np.nonzero(blue[max(255,bp-25):min(low,bp+26)])
        top=int(np.percentile(yy,99))+max(255,bp-25)+20
        board=[top,low] if low-top>=30 else None
    return dict(status='source_RGB_foot_strip_and_cap_conservative_bounds',ice_ymin=bottom,
                board_range=board,foot_strip_row_peak=ypeak,foot_strip_x_support=len(np.unique(x)),
                source_role_is_conservative_pixel_navigation_not_label=True)

def make_mask(a,b,ma,mb,material):
    person=cv2.dilate((ma|mb).astype(np.uint8),np.ones((41,41),np.uint8)).astype(bool)
    ba,bb=source_material_bounds(a,person),source_material_bounds(b,person)
    mask=np.zeros((1080,1920),np.uint8)
    if material=='ice':
        lower=max(ba['ice_ymin'],bb['ice_ymin'])
        mask[lower:1030,25:1895]=255
    elif ba['board_range'] and bb['board_range']:
        lo=max(ba['board_range'][0],bb['board_range'][0]);hi=min(ba['board_range'][1],bb['board_range'][1])
        if hi>lo:mask[lo:hi,25:1895]=255
    mask[person]=0
    mask[:265,:585]=0
    mask[930:1020,1660:1895]=0
    return mask,dict(first=ba,second=bb,mask_pixels=int(np.count_nonzero(mask)),human_mask_union_dilation_native_px=41)

def sift_initialized_source_tracks(a,b,mask):
    """Source-only mutual descriptors initialize the same RGB FB check."""
    ga=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY);gb=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
    detector=cv2.SIFT_create(nfeatures=3500,contrastThreshold=.018)
    ka,da=detector.detectAndCompute(ga,mask);kb,db=detector.detectAndCompute(gb,mask)
    if da is None or db is None:raise RuntimeError('Missing mutual source SIFT descriptors')
    matcher=cv2.BFMatcher()
    f=matcher.knnMatch(da,db,k=2);r=matcher.knnMatch(db,da,k=2)
    rev={m.queryIdx:m.trainIdx for group in r if len(group)==2 for m,n in [group] if m.distance<.72*n.distance}
    matches=[m for group in f if len(group)==2 for m,n in [group] if m.distance<.72*n.distance and rev.get(m.trainIdx)==m.queryIdx]
    if len(matches)<12:raise RuntimeError(f'Only {len(matches)} mutual source SIFT descriptors')
    p=np.asarray([ka[m.queryIdx].pt for m in matches],np.float32).reshape(-1,1,2)
    initial=np.asarray([kb[m.trainIdx].pt for m in matches],np.float32).reshape(-1,1,2)
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.005)
    q,s,e=cv2.calcOpticalFlowPyrLK(ga,gb,p,initial.copy(),winSize=(31,31),maxLevel=2,criteria=criteria,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    z,t,_=cv2.calcOpticalFlowPyrLK(gb,ga,q,p.copy(),winSize=(31,31),maxLevel=2,criteria=criteria,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    p=p.reshape(-1,2);q=q.reshape(-1,2);fb=np.linalg.norm(p-z.reshape(-1,2),axis=1)
    qi=np.rint(q).astype(int);qi[:,0]=qi[:,0].clip(0,1919);qi[:,1]=qi[:,1].clip(0,1079)
    valid=(s.ravel()!=0)&(t.ravel()!=0)&(fb<.8)&(np.linalg.norm(q-initial.reshape(-1,2),axis=1)<12)&(mask[qi[:,1],qi[:,0]]!=0)
    p=p[valid];q=q[valid];fb=fb[valid]
    if len(p)<12:raise RuntimeError(f'Only {len(p)} SIFT-initialized source FB-valid tracks')
    test=np.arange(len(p))%3==0
    H,inliers=cv2.findHomography(p[~test],q[~test],cv2.RANSAC,1.5,maxIters=5000,confidence=.999)
    if H is None:raise RuntimeError('No independent RGB image homography')
    residual=np.linalg.norm(probe.hom_apply(H,p)-q,axis=1)
    return p,q,fb,H,test,residual<2,residual

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    shots=json.loads((ROOT/'calibration/program_tracking/source_shots.json').read_text(encoding='utf8'))['shots'][1:10]
    base=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003')
    paths={
        'old917':base/'okada_user_priorities_shared_actual_fit_fast_20261005/complete_physical_camera_model_UNACCEPTED_917c204dafde4027.npz',
        'clear0eb5':base/'source_corrected_clear_actual_complete_joint_20261007/complete_physical_camera_model_UNACCEPTED_46a4f24374875bb3.npz',
        'q478':base/'okada_lane23_Ad7_fast_complete_20261007/complete_physical_camera_model_UNACCEPTED_366aede4cdc70b50.npz'}
    models={name:np.load(path,allow_pickle=False) for name,path in paths.items()}
    indices={name:{int(p):i for i,p in enumerate(d['pts'])} for name,d in models.items()}
    pairs=[]
    for number,shot in enumerate(shots,1):
        start=max(229114112,int(shot['start_pts']));end=int(shot['end_pts'])
        for pa in sorted(set(int(round(p/256))*256 for p in np.linspace(start,end-7936,5))):
            pairs.append(dict(shot_index=number,shot_id=shot['id'],camera_id=shot['camera_id'],source_pts=pa,target_pts=pa+7680))
    for pa in (230043904,230254336,230967552):
        for number,shot in enumerate(shots,1):
            if int(shot['start_pts'])<=pa and pa+7680<int(shot['end_pts']):
                pairs.append(dict(shot_index=number,shot_id=shot['id'],camera_id=shot['camera_id'],source_pts=pa,target_pts=pa+7680,user_reported_context=True))
                break
    source=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/media/originals/oxXu1avhrvs.mp4')
    frames,receipts=probe.native_frames(source,sorted({p[k] for p in pairs for k in ('source_pts','target_pts')}))
    people=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE)
    report=dict(schema='rinkspace.nine-cuts-independent-RGB-camera-motion-regression.v1',
        source_video=str(source),source_sha256_from_verified_ledger='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891',
        source_clock='exact native PTS, same-cut pairs only',pairs=pairs,source_receipts=receipts,
        models={name:dict(path=str(path),sha256=probe.sha(path)) for name,path in paths.items()},
        source_points_selected_by_camera_or_old_UV=False,source_camera_or_observation_mutated=False,
        split='2/3 RGB-homography train, every third source track HELD diagnostic',experiments=[],
        limitations=['Motion transfer does not establish absolute world alignment.',
                    'Board uses rotation-only transfer without board-depth or optical-centre lever-arm parallax.',
                    'Ice uses Z0; reflections can move independently of physical ice.',
                    'Source RGB cap/foot-strip bounds are conservative navigation, not exact boundary measurements.',
                    'Failures and sparse support remain in the report; no interpolation or model-selected source pixels.'])
    tracks={}
    for pairnumber,pair in enumerate(pairs):
        pa,pb=pair['source_pts'],pair['target_pts'];a,b=frames[pa],frames[pb]
        ma,sa=people.get(pa,require_sam2=False);mb,sb=people.get(pb,require_sam2=False)
        for material in ('ice','board'):
            mask,mask_receipt=make_mask(a,b,ma,mb,material)
            entry=dict(**pair,material=material,source_navigation=mask_receipt,human_mask_status=[sa,sb])
            try:
                p,q,fb,H,test,keep,res=probe.track_pair(a,b,mask)
                entry['source_tracker']='existing_5level_RGB_LK_forward_backward'
            except RuntimeError as error:
                entry['initial_RGB_LK_failure']=str(error)
                try:
                    p,q,fb,H,test,keep,res=sift_initialized_source_tracks(a,b,mask)
                    entry['source_tracker']='mutual_source_SIFT_initialized_RGB_LK_forward_backward'
                except RuntimeError as fallback_error:
                    entry['failure']=str(fallback_error);report['experiments'].append(entry);continue
            clean=test&keep
            entry.update(total_tracks=len(p),independent_good_held_count=int(clean.sum()),
                         independent_all_held_error_px=probe.stats(res[test]),independent_good_held_error_px=probe.stats(res[clean]),
                         forward_backward_good_held_error_px=probe.stats(fb[clean]),
                         actual_good_held_motion_px=probe.stats(np.linalg.norm(q[clean]-p[clean],axis=1)),
                         independent_image_H=H.tolist(),models={})
            key=f'pair{pairnumber:02d}_{material}'
            tracks.update({key+'_source':p,key+'_target':q,key+'_held':test,key+'_source_static_keep':keep,key+'_forward_backward_error':fb})
            for name,d in models.items():
                ia,ib=indices[name][pa],indices[name][pb]
                ca,cb=d['camera25'][ia],d['camera25'][ib]
                predicted,valid=probe.model_transfer(ca,cb,p,0. if material=='ice' else None)
                used=clean&valid;error=predicted-q
                values=np.linalg.norm(error[used],axis=1)
                entry['models'][name]=dict(valid_good_held_count=int(used.sum()),
                    actual_source_track_error_px=probe.stats(values),
                    median_signed_error_xy_px=np.median(error[used],axis=0).tolist() if used.any() else None,
                    inverse_Brown_roundtrip_max_px=probe.inverse_roundtrip(ca,p[used]) if used.any() else None,
                    actual_exposure_alias_pts=[int(d['exposure_representative_pts'][ia]),int(d['exposure_representative_pts'][ib])],
                    focal_change_percent=float(np.expm1(cb[6]-ca[6])*100),
                    same_held_source_baseline_error_px=probe.stats(res[used]))
            report['experiments'].append(entry)
        print(json.dumps(dict(progress=pairnumber+1,total=len(pairs),shot=pair['shot_index'],pts=[pa,pb])),flush=True)
    np.savez_compressed(out/'actual_RGB_tracks.npz',**tracks)
    summaries=[]
    for number in range(1,10):
        shot=shots[number-1];rows=[r for r in report['experiments'] if r['shot_index']==number]
        summary=dict(shot_index=number,shot_id=shot['id'],camera_id=shot['camera_id'],source_pts=[shot['start_pts'],shot['end_pts']],materials={})
        for material in ('ice','board'):
            ok=[r for r in rows if r['material']==material and r.get('independent_good_held_count',0)>=6]
            m=dict(successful_pairs=len(ok),failures=[dict(pts=[r['source_pts'],r['target_pts']],reason=r.get('failure','fewer than 6 independent held tracks')) for r in rows if r['material']==material and r not in ok],models={})
            for name in models:
                supported=[r for r in ok if r['models'][name]['valid_good_held_count']>=6]
                worst=max(supported,key=lambda r:r['models'][name]['actual_source_track_error_px']['median']) if supported else None
                m['models'][name]=dict(pairs=len(supported),held_points=sum(r['models'][name]['valid_good_held_count'] for r in supported),
                    pair_median_error_px=probe.stats([r['models'][name]['actual_source_track_error_px']['median'] for r in supported]),
                    worst=None if worst is None else dict(pts=[worst['source_pts'],worst['target_pts']],**worst['models'][name],image_baseline=worst['independent_good_held_error_px']))
            summary['materials'][material]=m
        summaries.append(summary)
    report.update(per_shot=summaries,wall_seconds=time.perf_counter()-t0)
    (out/'nine_shot_source_motion_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'nine_shot_source_motion_summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(completed=True,pairs=len(pairs),seconds=report['wall_seconds'],output=str(out)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
