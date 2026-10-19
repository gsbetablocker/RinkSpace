"""Readonly pan transfer using existing source-only deep LK / mutual SIFT.

Measures real printed-board motion on overlapping intermediate baselines.
The two user endpoints have no common print: no fictitious end-to-end match
is made. New diagnostic UV never enters the solver observation inventory.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
sys.dont_write_bytecode=True
from calibration.solver_runtime import install
install()
import cv2, numpy as np
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from structured_programme_source_candidate.native_board_track_source import current_far_board_points
from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown
spec=importlib.util.spec_from_file_location('source_only_nine_shots',ROOT/'events/ankara_2026/tools/review_nine_shot_source_motion.py')
source_helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(source_helpers)
probe=source_helpers.probe
from calibration.program_tracking.render_result import PersonMasks, DEFAULT_MASKS, DEFAULT_PEOPLE
cv2.setNumThreads(1)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def metric(delta, observed):
    length=np.linalg.norm(observed,axis=1);den=float(np.sum(length*length))
    return dict(native_residual=probe.stats(np.linalg.norm(delta,axis=1)),
        signed_median_xy_px=np.median(delta,axis=0).tolist(),
        signed_mean_xy_px=np.mean(delta,axis=0).tolist(),
        relative_excess_along_actual_motion=float(np.sum(delta*observed)/den)if den else None)


def main():
    ap=argparse.ArgumentParser()
    for key in ('state','camera','auxiliary','tracks','output'):ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--pair',action='append',help='Explicit native source:target PTS for an overlapping readonly baseline')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    provider,world,proof=current_complete_source_geometry(a.state,a.auxiliary)
    course,cq,_=provider.current.print_course_binding(world);border=course.sample_perimeter(cq)[0]
    cameras=dict(np.load(a.camera,allow_pickle=False));ix={int(p):i for i,p in enumerate(cameras['pts'])}
    aliases=dict(zip(map(int,cameras['pts']),map(int,cameras['exposure_representative_pts'])))
    def cam(p):return cameras['camera25'][ix[int(p)]].copy()
    pairs=([tuple(map(int,value.split(':')))for value in a.pair]if a.pair else
           [(231101184,231108864),(231113984,231121664),(231129856,231137792)])
    pairs=[(aliases[pa],aliases[pb])for pa,pb in pairs]
    source=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/media/originals/oxXu1avhrvs.mp4')
    frames,receipts=probe.native_frames(source,sorted({p for pair in pairs for p in pair}))
    people=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE)
    for p,bgr in frames.items():cv2.imwrite(str(a.output/f'native_{p}.png'),bgr)
    metadata=json.loads(str(cameras['metadata_json']))
    installation=str(cameras['installation_id'][ix[pairs[0][0]]])
    cal=metadata['calibrations'][installation]
    pivot=np.asarray(cal['pivot_xyz']);lever=np.asarray(cal['lever_xyz'])
    def transfer(ca,cb,p):
        xyz=current_far_board_points(ca,p,border)
        projected,depth,valid=project_central_valid_full_brown(cb,xyz)
        return projected,valid
    tests=[]
    for pa,pb in pairs:
        first,second=frames[pa],frames[pb]
        ma,_=people.get(pa,require_sam2=False);mb,_=people.get(pb,require_sam2=False)
        mask,nav=source_helpers.make_mask(first,second,ma,mb,'board')
        entry=dict(source_pts=pa,target_pts=pb,source_dt_seconds=(pb-pa)/15360.,material='actual_far_print_board',
                   source_navigation=nav,source_UV_not_selected_by_model=True)
        try:
            # Large source pan requires mutual descriptors to disambiguate
            # different ANKARA letters; LK is initialized from actual RGB only.
            p,q,fb,H,test,keep,herror=source_helpers.sift_initialized_source_tracks(first,second,mask)
        except RuntimeError as error:
            entry['source_match_failure']=str(error);tests.append(entry);print(json.dumps(entry),flush=True);continue
        clean=test&keep
        np.savez_compressed(a.output/f'source_only_{pa}_{pb}.npz',uv0=p,uv1=q,forward_backward_error_px=fb,
                            source_H=H,source_H_train_mask=~test,source_H_development_mask=test,source_H_consistent=keep)
        ca,cb=cam(pa),cam(pb);prediction,valid=transfer(ca,cb,p);selected=clean&valid
        if selected.sum()<4:raise RuntimeError('Insufficient source-only independent H check points')
        observed=q-p
        entry.update(source_points=len(p),source_H_train_points=int((~test).sum()),
            source_H_development_points=int(test.sum()),source_H_consistent_development_points=int(clean.sum()),
            source_H_all_development_error_px=probe.stats(herror[test]),source_H_consistent_development_error_px=probe.stats(herror[clean]),
            source_FB_development_px=probe.stats(fb[clean]),actual_observed_motion_px=probe.stats(np.linalg.norm(observed[selected],axis=1)),
            original_camera_focal_px=[float(np.exp(ca[6])),float(np.exp(cb[6]))],original_camera_PPx_PPy=ca[8:10].tolist(),
            actual_current_board_transfer=metric((prediction-q)[selected],observed[selected]),hypotheses=[])
        entry['clock_shift_hypotheses']=[]
        for shift in (-512,-256,0,256,512):
            if pa+shift in ix and pb+shift in ix:
                shifted,v=transfer(cam(pa+shift),cam(pb+shift),p)
                if v[selected].all():entry['clock_shift_hypotheses'].append(dict(native_pts_shift=shift,
                    actual_camera_exposures=[aliases[pa+shift],aliases[pb+shift]],
                    source_UV_and_native_frame_identity_unchanged=True,
                    comparison=metric((shifted-q)[selected],observed[selected])))
        hypotheses=[('PPx_plus100px',[100.,0.,0.],1.,1.),('PPx_minus100px',[-100.,0.,0.],1.,1.),
            ('PPy_plus100px',[0.,100.,0.],1.,1.),('PPy_minus100px',[0.,-100.,0.],1.,1.),
            ('K1_plus0.1',[0.,0.,.1],1.,1.),('K1_minus0.1',[0.,0.,-.1],1.,1.),
            ('shared_focal_times0.99',[0.,0.,0.],.99,1.),('shared_focal_times1.01',[0.,0.,0.],1.01,1.),
            ('relative_rotation_times0.99',[0.,0.,0.],1.,.99),('relative_rotation_times1.01',[0.,0.,0.],1.,1.01)]
        for name,optical,focal,rotation_scale in hypotheses:
            da,db=ca.copy(),cb.copy()
            for c in (da,db):c[8:10]+=optical[:2];c[11]+=optical[2];c[6:8]+=np.log(focal)
            if rotation_scale!=1.:
                Ra=cv2.Rodrigues(ca[3:6])[0];Rb=cv2.Rodrigues(cb[3:6])[0]
                rv=cv2.Rodrigues(Rb@Ra.T)[0].ravel();newR=cv2.Rodrigues(rotation_scale*rv)[0]@Ra
                db[3:6]=cv2.Rodrigues(newR)[0].ravel();db[:3]=pivot+newR.T@lever
            pp,v=transfer(da,db,p)
            assert v[selected].all()
            entry['hypotheses'].append(dict(name=name,source_points_identical_to_nominal=True,
                pivot_lever_hardware_unchanged=True,comparison=metric((pp-q)[selected],observed[selected])))
        for material,pts in (('source',p),('target',q)):
            im=frames[pa if material=='source'else pb].copy()
            for ordinal,j in enumerate(np.flatnonzero(clean)):
                xy=tuple(np.rint(pts[j]).astype(int));cv2.circle(im,xy,5,(40,255,70),2)
                cv2.putText(im,str(ordinal),xy,cv2.FONT_HERSHEY_SIMPLEX,.5,(0,30,255),1)
            cv2.imwrite(str(a.output/f'{material}_{pa}_{pb}_actual_print_development_points.png'),im)
        tests.append(entry);print(json.dumps(dict(pair=[pa,pb],points=len(p),held=int(clean.sum()),actual=entry['actual_current_board_transfer'])),flush=True)
    # Existing v2 source archive supplies actual adjacent-frame relative drift.
    # Signed medians integrate a common image velocity, not a claimed same-point
    # endpoint track. Missing support stays missing and is explicitly counted.
    tracks=dict(np.load(a.tracks,allow_pickle=False));adjacent=[]
    for i in np.flatnonzero((tracks['shot_index']==9)&(tracks['source_pts']>=231101184)&(tracks['target_pts']<=231137792)):
        pa,pb=int(tracks['source_pts'][i]),int(tracks['target_pts'][i]);lo,hi=map(int,tracks['offsets'][i:i+2])
        item=dict(source_pts=pa,target_pts=pb,source_points=hi-lo)
        if hi-lo>=6:
            p,q=tracks['uv0'][lo:hi],tracks['uv1'][lo:hi]
            predicted,valid=transfer(cam(pa),cam(pb),p)
            if valid.all():item['comparison']=metric(predicted-q,q-p)
        adjacent.append(item)
    good=[r for r in adjacent if 'comparison'in r]
    integrated=np.sum([r['comparison']['signed_median_xy_px']for r in good],axis=0)
    report=dict(schema='rinkspace.readonly-main9-native-source-pan-transfer-audit.v1',state=str(a.state),state_sha256=sha(a.state),
        camera=str(a.camera),camera_sha256=sha(a.camera),source_video=str(source),source_sha256_verified_ledger='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891',
        original_native_frame_receipts=receipts,existing_adjacent_source_archive=str(a.tracks),existing_adjacent_source_sha256=sha(a.tracks),
        two_user_endpoints_have_no_common_print_match=True,intermediate_overlapping_baselines=tests,
        actual_existing_adjacent_pairs=adjacent,adjacent_supported_pairs=len(good),adjacent_missing_or_sparse=len(adjacent)-len(good),
        sum_signed_adjacent_median_error_xy_px=integrated.tolist(),
        accumulated_signed_image_velocity_is_not_a_same_point_end_to_end_track=True,
        immutable_geometry_proof=proof,source_H_is_image_baseline_not_world_measurement=True,
        source_only_UV_not_admitted_or_fitted=True,optical_and_rotation_hypotheses_not_adopted=True,
        fullgraph_or_fullsolver_not_started=True,accuracy_accepted=False,seconds=time.perf_counter()-start)
    dest=a.output/'actual_main9_pan_source_transfer_report.json'
    dest.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(dest),sha256=sha(dest),pairs=len(good),integrated_error=integrated.tolist(),seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
