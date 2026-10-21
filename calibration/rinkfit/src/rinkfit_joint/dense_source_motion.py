"""Independent two-exposure 2D source-motion gate on the physical board.

Each original pixel pair is retained. A ray/continuous-board intersection is a
numerical latent for this check, never a known world point or new observation.
No material blocks, fit factors, source UV tables, or image tracks are generated.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import argparse
import gzip
import hashlib
import json
import time
import numpy as np
from scipy.interpolate import PchipInterpolator
from .io_utils import read_json,read_jsonl,write_json,file_sha256
from .camera_graph import CameraGraph
from .joint_model import graph_from_joint,as_event_view
from .wide_optics_candidate import _camera_geometry
from rinkfit_start.native_scene import NativeScene

VERSION='dense_static_board_bidirectional_source_motion_gate_v1'
STATUS={0:'valid_source_to_source_2D_motion',1:'inverse_Brown_or_ray_domain',2:'no_positive_continuous_board_intersection',3:'outside_physical_board_height',4:'source_RS_inverse_roundtrip_failed',5:'target_RS_unbracketed_or_uncertified',6:'target_board_backface_or_self_occlusion',7:'nonfinite_target_prediction'}


def _ref(path):
    p=Path(path).resolve();return dict(file=str(p),sha256=file_sha256(p))


def _summary(value,denominator=None):
    v=np.asarray(value,float);v=v[np.isfinite(v)]
    result=dict(denominator=int(len(value)if denominator is None else denominator),finite=int(v.size),unknown=int((len(value)if denominator is None else denominator)-v.size))
    if v.size:result.update(rms=float(np.sqrt(np.mean(v*v))),median=float(np.median(v)),p90=float(np.quantile(v,.90)),p95=float(np.quantile(v,.95)),p99=float(np.quantile(v,.99)),maximum=float(np.max(v)),minimum=float(np.min(v)))
    return result


def _inverse_brown_rows(target,coefficients):
    xy=np.array(target,float,copy=True);valid=np.isfinite(xy).all(1);converged=np.zeros(len(xy),bool);previous=np.full(len(xy),np.inf)
    while True:
        ii=np.flatnonzero(valid&~converged)
        if not len(ii):break
        x,y=xy[ii].T;k1,k2,p1,p2=coefficients[ii].T;r2=x*x+y*y
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            radial=1+k1*r2+k2*r2*r2;pred=np.c_[x*radial+2*p1*x*y+p2*(r2+2*x*x),y*radial+p1*(r2+2*y*y)+2*p2*x*y];res=pred-target[ii];error=np.max(np.abs(res),axis=1)
            slope=k1+2*k2*r2;a=radial+2*x*x*slope+2*p1*y+6*p2*x;b=2*x*y*slope+2*p1*x+2*p2*y;d=radial+2*y*y*slope+6*p1*y+2*p2*x;det=a*d-b*b
        done=np.isfinite(error)&(error<2e-13)&(a>0)&(det>0);converged[ii[done]]=True
        failed=(~np.isfinite(error))|((error>=previous[ii])&~done)|(a<=0)|(det<=0);valid[ii[failed]]=False
        move=~(done|failed);jj=ii[move]
        if len(jj):
            delta=np.c_[(d[move]*res[move,0]-b[move]*res[move,1])/det[move],(a[move]*res[move,1]-b[move]*res[move,0])/det[move]]
            xy[jj]-=delta;previous[jj]=error[move]
    return xy,valid&converged


def _rays(camera,frame_t,uv):
    t=np.asarray(frame_t,float)+camera.tau*(uv[:,1]/1079.-.5)
    R,C,lf,z=_camera_geometry(camera,t)
    principal=np.asarray(camera.optics['principal_point'])+z[:,None]*np.asarray(camera.optics['principal_linear'])
    fxy=np.c_[np.exp(lf+camera.optics['eta']/2),np.exp(lf-camera.optics['eta']/2)]
    coefficients=np.c_[np.polynomial.polynomial.polyval(z,camera.optics['k1']),np.polynomial.polynomial.polyval(z,camera.optics['k2']),np.full(len(z),camera.optics['p1']),np.full(len(z),camera.optics['p2'])]
    coefficients+=np.maximum(camera.wide_boundary-lf,0.)[:,None]**2*camera.wide_coefficients
    xy,valid=_inverse_brown_rows((uv-principal)/fxy,coefficients)
    valid&=np.linalg.norm(xy,axis=1)<=float(camera.optics['radius_max'])
    ray=np.einsum('nji,nj->ni',R,np.c_[xy,np.ones(len(xy))])
    return C,ray,valid,t


def _continuous_board_exit(scene,C,ray):
    # The convex XY containment derivative is monotone on a source ray. Locate
    # its minimum and then the far/front inner-board intersection by bisection.
    # Termination is floating-interval stagnation, not a fitted error threshold.
    n=len(C);lo=np.zeros(n);hi=np.full(n,np.inf);valid=np.isfinite(C).all(1)&np.isfinite(ray).all(1)
    for axis,bound in ((0,.5),(1,scene.width/2)):
        a=ray[:,axis];c=C[:,axis];moving=np.abs(a)>np.finfo(float).tiny
        valid&=moving|(np.abs(c)<=bound)
        with np.errstate(divide='ignore',invalid='ignore'):
            x=(-bound-c)/a;y=(bound-c)/a
        lower=np.where(moving,np.minimum(x,y),-np.inf);upper=np.where(moving,np.maximum(x,y),np.inf)
        lo=np.maximum(lo,lower);hi=np.minimum(hi,upper)
    valid&=np.isfinite(hi)&(hi>lo)
    safe_hi=np.where(valid,hi,lo+1.);safe_lo=np.where(valid,lo,0.)
    def f(lam):return scene.containment(C[:,:2]+lam[:,None]*ray[:,:2])
    def df(lam):
        xy=C[:,:2]+lam[:,None]*ray[:,:2]
        u=np.maximum(np.abs(xy[:,0])-(.5-scene.rx),0.)/scene.rx;v=np.maximum(np.abs(xy[:,1])-(scene.width/2-scene.ry),0.)/scene.ry
        return scene.p*(np.sign(xy[:,0])*u**(scene.p-1)*ray[:,0]/scene.rx+np.sign(xy[:,1])*v**(scene.p-1)*ray[:,1]/scene.ry)
    a=safe_lo.copy();b=safe_hi.copy()
    while True:
        mid=(a+b)/2;moving=(mid>a)&(mid<b)
        if not np.any(moving):break
        slope=df(mid);left=(slope<0)&moving
        a=np.where(left,mid,a);b=np.where(~left&moving,mid,b)
    minimum=(a+b)/2;fm=f(minimum);fh=f(safe_hi)
    valid&=np.isfinite(fm)&np.isfinite(fh)&(fm<=2e-12)&(fh>=-2e-12)
    a=minimum.copy();b=safe_hi.copy()
    while True:
        mid=(a+b)/2;moving=(mid>a)&(mid<b)
        if not np.any(moving):break
        inside=f(mid)<0
        a=np.where(inside&moving,mid,a);b=np.where(~inside&moving,mid,b)
    lam=(a+b)/2;X=C+lam[:,None]*ray
    valid&=(lam>0)&np.isfinite(X).all(1)&(np.abs(scene.containment(X[:,:2]))<2e-10)
    return X,valid,lam


def _inward_front(scene,X,C):
    u=np.maximum((np.abs(X[:,0])-(.5-scene.rx))/scene.rx,0.);v=np.maximum((np.abs(X[:,1])-(scene.width/2-scene.ry))/scene.ry,0.)
    outward=np.c_[np.sign(X[:,0])*u**(scene.p-1)/scene.rx,np.sign(X[:,1])*v**(scene.p-1)/scene.ry]
    return np.sum(outward*(C[:,:2]-X[:,:2]),axis=1)<0.


def _station_height(scene,X):
    # Recover the actual common perimeter coordinate without equating arbitrary
    # source curve station or angle across different physical entities.
    xx=X[:,0];yy=X[:,1];a0=.5-scene.rx;b0=scene.width/2-scene.ry
    lengths=np.array([scene.long,scene.arc,scene.short,scene.arc,scene.long,scene.arc,scene.short,scene.arc]);edges=np.r_[0,np.cumsum(lengths)]
    bottom=np.abs(yy+scene.width/2)<1e-10;right=np.abs(xx-.5)<1e-10;top=np.abs(yy-scene.width/2)<1e-10;left=np.abs(xx+.5)<1e-10
    angle=np.arctan2(np.maximum(np.abs(yy)-b0,0.)/scene.ry,np.maximum(np.abs(xx)-a0,0.)/scene.rx)
    arclength=scene.arc_inverse.x;angles=scene.arc_inverse(arclength);forward=PchipInterpolator(angles,arclength,extrapolate=False);arc=np.asarray(forward(np.clip(angle,0,np.pi/2)))
    s=np.full(len(X),np.nan)
    s[bottom]=xx[bottom]+a0;s[right]=edges[2]+yy[right]+b0;s[top]=edges[4]+a0-xx[top];s[left]=edges[6]+b0-yy[left]
    for seg,hit in ((1,(xx>a0)&(yy<-b0)),(3,(xx>a0)&(yy>b0)),(5,(xx<-a0)&(yy>b0)),(7,(xx<-a0)&(yy<-b0))):
        s[hit]=edges[seg]+(scene.arc-arc[hit]if seg in(1,5)else arc[hit])
    return np.mod(s,scene.perimeter),X[:,2]


def _direction(scene,camera,t0,t1,uv0,uv1):
    n=len(uv0);status=np.zeros(n,np.uint8);error=np.full(n,np.nan);delta=np.full((n,2),np.nan);inverse_error=np.full(n,np.nan)
    C,ray,brown,time0=_rays(camera,t0,uv0);status[~brown]=1
    X,root,lam=_continuous_board_exit(scene,C,ray);status[(status==0)&~root]=2
    s,h=_station_height(scene,X);chart=scene.board(np.nan_to_num(s),np.nan_to_num(h));chart_error=np.linalg.norm(chart-X,axis=1)
    good=(status==0)&np.isfinite(s)&np.isfinite(h)&(h>=0)&(h<=scene.height)&_inward_front(scene,X,C)
    status[(status==0)&~good]=3
    source=camera.at(X,time0);inverse_error=np.linalg.norm(source['uv']-uv0,axis=1)
    good&=source['valid']&(inverse_error<=1e-5)&(chart_error<=1e-9)
    status[(status==0)&~good]=4
    ii=np.flatnonzero(good)
    if len(ii):
        projected=camera.project_rs(X[ii],np.asarray(t1)[ii]);pv=projected['valid'];status[ii[~pv]]=5
        target_time=np.asarray(t1)[ii]+camera.tau*(projected['row']/1079.-.5)
        _,target_C,_,_=_camera_geometry(camera,target_time);front=_inward_front(scene,X[ii],target_C);status[ii[pv&~front]]=6
        actual=projected['uv']-uv1[ii];finite=np.isfinite(actual).all(1);status[ii[pv&front&~finite]]=7
        jj=ii[pv&front&finite];delta[jj]=actual[pv&front&finite];error[jj]=np.linalg.norm(delta[jj],axis=1)
    return dict(status=status,error=error,delta=delta,inverse_error=inverse_error,world_chart_roundtrip_L=chart_error)


def _segments(frames,key):
    result=[]
    for sid in sorted({r['shot_id']for r in frames}):
        records=[r for r in frames if r['shot_id']==sid];values=np.asarray([r[key]if r[key]is not None else np.nan for r in records]);finite=np.isfinite(values);threshold=float(np.quantile(values[finite],.95))if finite.any()else None
        for mode in ('empirical_upper_5pct_frame_RMS','any_invalid_original_pair'):
            selected=[i for i,r in enumerate(records)if (finite[i]and values[i]>=threshold)if mode=='empirical_upper_5pct_frame_RMS']if mode=='empirical_upper_5pct_frame_RMS'else[i for i,r in enumerate(records)if r['forward_invalid']or r['reverse_invalid']]
            runs=[];run=[]
            for i in selected:
                if run and records[i]['native_pts']!=records[run[-1]]['native_pts']+256:runs.append(run);run=[]
                run.append(i)
            if run:runs.append(run)
            for run in runs:
                rr=[records[i]for i in run];vv=[r[key]for r in rr if r[key]is not None]
                result.append(dict(shot_id=sid,kind=mode,first_previous_pts=rr[0]['previous_native_pts'],last_native_pts=rr[-1]['native_pts'],source_start_seconds=rr[0]['previous_native_pts']/12800.,source_end_seconds=rr[-1]['native_pts']/12800.,native_adjacent_pairs=len(rr),source_correspondence_pairs=sum(r['source_pairs']for r in rr),forward_invalid=sum(r['forward_invalid']for r in rr),reverse_invalid=sum(r['reverse_invalid']for r in rr),frame_RMS_max=max(vv)if vv else None,descriptive_tail_threshold_px=threshold if mode=='empirical_upper_5pct_frame_RMS'else None,threshold_is_quality_acceptance=False))
    return result


def evaluate_dense_source_motion(model_file,pair_file_index,out,*,baseline=None):
    started=time.perf_counter();out=Path(out).resolve()
    if out.exists():raise ValueError('New immutable source motion audit output required')
    records=list(read_jsonl(pair_file_index));model=read_json(model_file);scene=NativeScene(as_event_view(model,'s01'));graph=CameraGraph(graph_from_joint(model));checked={};frame_stats=[];global_forward=[];global_reverse=[];global_status_f=[];global_status_r=[];byshot=[];proof=hashlib.sha256()
    def pin(ref):
        if not ref:return
        path=ref['file'];sha=ref['sha256']
        if path in checked:
            if checked[path]!=sha:raise ValueError('Conflicting source RGB/mask/material ref')
        else:
            if file_sha256(path)!=sha:raise ValueError('Changed source mask/body/motion evidence: '+path)
            checked[path]=sha
    for sid in sorted({r['shot_id']for r in records}):
        group=sorted([r for r in records if r['shot_id']==sid],key=lambda r:r['native_pts']);aa=[];bb=[];pp=[];qq=[];counts=[]
        for r in group:
            pin(r);m=read_json(r['file']);n=r['actual_2D_correspondence_pairs']
            if m['shot_id']!=sid or m['native_pts']!=r['native_pts']or m['previous_native_pts']!=r['previous_native_pts']or m['native_pts']-m['previous_native_pts']!=256 or m['cross_cut']is not False or m['source_world_projection_used']is not False:raise ValueError('Source pair crosses a cut or changed clock')
            if m['previous_source_RGB_sha256']!=r['previous_RGB_sha256']or m['current_source_RGB_sha256']!=r['current_RGB_sha256']:raise ValueError('Changed actual exposure RGB identity')
            if m['complete_21x21_patch_required_in_both_real_exposure_domains']is not True or m['source_or_current_UV_not_moved']is not True:raise ValueError('Actual complete two-exposure material patch not supported')
            for k in ('source_masks','surface_domains'):
                for ref in r[k]:pin(ref)
            a=np.asarray(m['source_uv'],float);b=np.asarray(m['current_uv'],float)
            if a.shape!=(n,2)or b.shape!=a.shape or not(np.isfinite(a).all()and np.isfinite(b).all()):raise ValueError('Actual full 2D source denominator changed')
            aa.append(a);bb.append(b);pp.append(np.full(n,r['previous_native_pts'],np.int64));qq.append(np.full(n,r['native_pts'],np.int64));counts.append(n)
            proof.update(json.dumps([r['file'],r['sha256'],n],separators=(',',':')).encode()+b'\n')
        a=np.concatenate(aa);b=np.concatenate(bb);p=np.concatenate(pp);q=np.concatenate(qq);camera=graph.native_camera(sid);t0=graph.local_seconds(sid,p);t1=graph.local_seconds(sid,q)
        f=_direction(scene,camera,t0,t1,a,b);rev=_direction(scene,camera,t1,t0,b,a)
        global_forward.append(f['error']);global_reverse.append(rev['error']);global_status_f.append(f['status']);global_status_r.append(rev['status']);cursor=0
        for r,n in zip(group,counts):
            sl=slice(cursor,cursor+n);fe=f['error'][sl];re=rev['error'][sl];ff=_summary(fe);rr=_summary(re);fd=f['delta'][sl];rd=rev['delta'][sl]
            frame_stats.append(dict(shot_id=sid,previous_native_pts=r['previous_native_pts'],native_pts=r['native_pts'],source_pairs=n,forward_valid=ff['finite'],forward_invalid=ff['unknown'],reverse_valid=rr['finite'],reverse_invalid=rr['unknown'],forward_RMS=ff.get('rms'),reverse_RMS=rr.get('rms'),forward_p95=ff.get('p95'),reverse_p95=rr.get('p95'),forward_max=ff.get('maximum'),reverse_max=rr.get('maximum'),forward_signed_mean_xy=np.nanmean(fd,axis=0).tolist()if np.isfinite(fe).any()else None,reverse_signed_mean_xy=np.nanmean(rd,axis=0).tolist()if np.isfinite(re).any()else None,source_motion_ref=dict(file=r['file'],sha256=r['sha256']),forward_invalid_reasons={STATUS[int(k)]:int(v)for k,v in zip(*np.unique(f['status'][sl],return_counts=True))if k},reverse_invalid_reasons={STATUS[int(k)]:int(v)for k,v in zip(*np.unique(rev['status'][sl],return_counts=True))if k}));cursor+=n
        byshot.append(dict(shot_id=sid,source_correspondence_pairs=len(a),adjacent_exposure_pairs=len(group),forward_2D_pixel_error=_summary(f['error']),reverse_2D_pixel_error=_summary(rev['error']),source_inverse_numerical_roundtrip_px=_summary(f['inverse_error']),board_chart_numerical_roundtrip_L=_summary(f['world_chart_roundtrip_L']),forward_invalid_reasons={STATUS[int(k)]:int(v)for k,v in zip(*np.unique(f['status'],return_counts=True))if k},reverse_invalid_reasons={STATUS[int(k)]:int(v)for k,v in zip(*np.unique(rev['status'],return_counts=True))if k}))
        print(json.dumps(dict(shot=sid,pairs=len(a),forward=byshot[-1]['forward_2D_pixel_error'],reverse=byshot[-1]['reverse_2D_pixel_error'])),flush=True)
    out.mkdir(parents=True);framefile=out/'frame_motion.jsonl.gz'
    with gzip.open(framefile,'wt',encoding='utf8')as f:
        for r in frame_stats:f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
    all_f=np.concatenate(global_forward);all_r=np.concatenate(global_reverse)
    report=dict(schema='rinkfit.independent_dense_static_board_source_motion_gate.v1',version=VERSION,model=_ref(model_file),pair_file_index=_ref(pair_file_index),actual_source_pair_set_sha256=proof.hexdigest(),full_source_correspondence_pairs=len(all_f),adjacent_native_exposure_pairs=len(records),forward_2D_pixel_error=_summary(all_f),reverse_2D_pixel_error=_summary(all_r),by_shot=byshot,continuous_tail_and_invalid_segments=_segments(frame_stats,'forward_RMS'),frame_metrics=_ref(framefile),checked_actual_evidence_files=len(checked),source_pixel_arrays_and_UV_tables_copied=False,new_source_pixels_decoded=0,actual_mask_body_and_motion_refs_SHA256_checked=True,source_RGB_identity_compared_to_frozen_qualification=True,source_RGB_pixels_decoded_again=False,source_masks_redecoded_again=False,source_occlusion_policy='Actual original complete 21x21 patches must be qualified in both source exposure domains; a large model residual never changes that qualification.',source_inverse_is_accuracy_measure=False,source_station_and_height_are_numeric_latents_only=True,source_world_coordinates_issued=False,new_model_blocks_or_materials=0,whole_board_occlusion_checked_by_inward_face_and_height=True,source_movement_residual_is_full_2D_UV=True,curve_normal_surrogate_used=False,motion_factors_installed_in_Ceres=False,gate_kind='independent_frozen_source_observation_validation',quality_accepted=False,reader=_ref(__file__),wall_seconds=time.perf_counter()-started)
    if baseline:
        old=read_json(baseline)
        if old['actual_source_pair_set_sha256']!=report['actual_source_pair_set_sha256']:raise ValueError('Motion gate baseline uses different source pairs')
        oldframes={(r['shot_id'],r['native_pts']):r for r in read_jsonl(old['frame_metrics']['file'])};bad=[];improved=[]
        for r in frame_stats:
            oldr=oldframes[(r['shot_id'],r['native_pts'])]
            if oldr['source_pairs']!=r['source_pairs']:raise ValueError('Motion gate comparison drops source denominator')
            for direction in ('forward','reverse'):
                x=r[direction+'_RMS'];y=oldr[direction+'_RMS'];iv=r[direction+'_invalid']-oldr[direction+'_invalid']
                if iv>0 or (x is not None and y is not None and x>y+1e-7):bad.append(dict(shot_id=r['shot_id'],native_pts=r['native_pts'],direction=direction,old_RMS=y,new_RMS=x,invalid_increase=iv))
                elif x is not None and y is not None and x<y-1e-7:improved.append((r['shot_id'],r['native_pts'],direction))
        report['same_source_baseline_comparison']=dict(baseline=_ref(baseline),frame_directions_with_larger_RMS_or_new_invalid=len(bad),frame_directions_with_lower_RMS=len(improved),regressions=bad,no_single_global_cost_override=True,comparison_numerical_padding_px=1e-7,acceptance_decision_requires_model_and_original_source_review=True)
    write_json(out/'report.json',report);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--pairs',required=True);p.add_argument('--out',required=True);p.add_argument('--baseline');a=p.parse_args();r=evaluate_dense_source_motion(a.model,a.pairs,a.out,baseline=a.baseline);print(json.dumps(dict(report=_ref(Path(a.out)/'report.json'),source_pairs=r['full_source_correspondence_pairs'],forward=r['forward_2D_pixel_error'],reverse=r['reverse_2D_pixel_error'],wall_seconds=r['wall_seconds'])))

if __name__=='__main__':main()
