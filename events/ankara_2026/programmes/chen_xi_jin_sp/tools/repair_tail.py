"""Reinitialize the weak ending shots from verified board identities."""
from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

from common import HEIGHT, JOINT, OUTPUT, STEP, WIDTH, frozen_scene, read, save, scope, source_frames, source_pts, write
from measure import DT, FIRST, fold_map, group_foreground, initialize_controls, localize
from navigate import Pose, sequential

INTAKE = OUTPUT.parent/"intake_1115"
AD2_BOTTOM="print_instance/Ad2::Ad2.bottom_edge"
AD2_RIGHT="print_instance/Ad2::Ad2.right_vertical"
FOOT="board/continuous/bottom"


def fit_angles(pose, rig, reference, focal):
    logf=np.log(focal)
    aim=reference[2]
    pan_starts=np.arange(-2.9,3.0,.33)
    def fun(a):
        projection=pose.camera(rig,[*a,logf])[4:13].reshape(3,3)
        return projection[2]-aim
    options=[(np.linalg.norm(fun([pan,0.])),pan) for pan in pan_starts]
    best=min(options)[1]
    q=least_squares(fun,[best,0.],x_scale=[.4,.3],gtol=1.e-11).x
    return np.array([*q,logf])


def s18_seed(native, shot, rig):
    with np.load(OUTPUT/"navigation/s18_flow.npz",allow_pickle=False) as flow:
        pts=flow["pts"]; matrices=flow["adjacent_h"]; valid=flow["counts"]>0
    sample=read(INTAKE/"board_camera_comparison.json")
    eligible=[x for x in sample if x["shot"]=="s18" and x["points"]>=24
              and x["by_rig"]["secondary"]["p50"]<=2.8 and x["by_rig"]["secondary"]["p90"]<6.3
              and x["by_rig"]["secondary"]["p50"]<=x["by_rig"]["main"]["p50"]+.4]
    best={}
    pose=Pose(native.core)
    for item in eligible:
        i=int(np.searchsorted(pts,round(item["time"]*50)*STEP))
        v=item["by_rig"]["secondary"]
        score=v["n4"]/(1.+v["p90"])
        theta=fit_angles(pose,rig,np.asarray(v["R"]),v["f"])
        option=(score,theta,v["p50"],v["p90"],v["n4"],round(item["match_time"]*50)*STEP)
        if i not in best or best[i][0]<score:
            best[i]=option
    if len(best)<3:
        raise ValueError("Ad1/Ad2 cross-installation geometry could not initialize S18")
    nav,report=sequential(pts,best,matrices,valid,pose,rig)
    assert np.array_equal(pts,source_pts(shot))
    return initialize_controls(pts,nav),report


def robust_line(points, slope_limit=.17):
    if len(points)<5:
        return np.empty((0,2))
    points=np.asarray(points)
    x,y=points.T
    strong=[]
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            span=x[j]-x[i]
            if abs(span)<50:
                continue
            slope=(y[j]-y[i])/span
            if abs(slope)>slope_limit:
                continue
            keep=np.abs(y-(y[i]+slope*(x-x[i])))<=2.9
            if np.sum(keep)>len(strong):
                strong=np.flatnonzero(keep)
    if len(strong)<5 or np.ptp(x[strong])<80.:
        return np.empty((0,2))
    return points[strong]


def measured_fall(value, mask, low, high, minimum, max_after=None, min_before=None, after_span=0):
    value=np.asarray(value,float)
    if high<=low+10:
        return None
    smooth=cv2.GaussianBlur(value.reshape(-1,1),(1,11),1.8)[:,0]
    g=-np.gradient(smooth)
    ids=np.arange(max(8,low),min(len(g)-8,high))
    contrast=smooth[ids-7]-smooth[ids+7]
    good=(contrast>minimum)&(smooth[ids-7]>(min_before or .70*minimum))&(smooth[ids+7]<(max_after or .42*minimum))
    good&=~(mask[ids-6]|mask[ids+6]|mask[ids])
    if after_span:
        for j in np.flatnonzero(good):
            k=ids[j]
            visits=slice(k+17,min(len(smooth),k+17+after_span))
            available=smooth[visits][~mask[visits]]
            if len(available)<max(20,after_span//4) or np.mean(available>12.)>.16:
                good[j]=False
    if not good.any():
        return None
    score=g[ids]+(.006*ids if after_span else 0.)
    best=int(ids[np.argmax(np.where(good,score,-np.inf))])
    a,b,c=g[best-1:best+2]
    den=a-2*b+c
    frac=.5*(a-c)/den if den<0. else 0.
    return best+float(np.clip(frac,-.5,.5))


def measure_s20_panel(bgr, mask):
    b,g,r=np.moveaxis(bgr.astype(np.float32),2,0)
    purple=b-g+.48*(r-g)
    yellow=.60*r+.40*g-b
    right=[]
    for y in range(46,350,12):
        val=np.median(purple[y-4:y+5],axis=0)
        blocked=mask[y-4:y+5].any(0)
        x=measured_fall(val,blocked,470,1710,21.,after_span=220)
        if x is not None:
            right.append((y,x))
    right=robust_line(right)
    if not len(right):
        return {}
    maximum_x=int(np.min(right[:,1]))-44
    bottom=[]
    for x in range(136,maximum_x,36):
        val=np.median(purple[:,x-5:x+6],axis=1)
        blocked=mask[:,x-5:x+6].any(1)
        y=measured_fall(val,blocked,130,680,22.,after_span=57)
        if y is not None:
            bottom.append((x,y))
    bottom=robust_line(bottom,.18)
    if not len(bottom):
        return {}
    foot=[]
    y0=int(np.median(bottom[:,1]))
    for x in range(48,WIDTH-30,48):
        val=np.median(yellow[:,x-7:x+8],axis=1)
        blocked=mask[:,x-7:x+8].any(1)
        y=measured_fall(val,blocked,max(110,y0+29),min(HEIGHT-18,y0+185),15.0,max_after=20.,min_before=28.)
        if y is not None:
            foot.append((x,y))
    foot=robust_line(foot)
    ret={AD2_BOTTOM:bottom,AD2_RIGHT:right[:,::-1].copy()}
    if len(foot):
        ret[FOOT]=foot
    return ret


def nearest_residual(sample, segments):
    good=np.isfinite(segments).all(1)
    segments=segments.copy()
    segments[~good]=np.nan
    a=segments[:-1];d=np.diff(segments,axis=0)
    length=np.sum(d*d,axis=1)
    valid=(length>1.e-10)&np.isfinite(length)
    if not valid.any():
        return np.full(sample.shape,1.e3)
    delta=sample[:,None]-a[valid]
    mix=np.clip(np.einsum("iqa,qa->iq",delta,d[valid])/length[valid],0,1.)
    e=delta-mix[...,None]*d[valid]
    select=np.argmin(np.sum(e*e,axis=2),axis=1)
    return e[np.arange(len(e)),select]


def s20_seed(native, record, shot, rig):
    sys.path.insert(0,str(JOINT))
    from joint_inputs import native_paths

    pts=source_pts(shot)
    roles=fold_map(shot,len(pts))
    with np.load(OUTPUT/"navigation/s20.npz",allow_pickle=False) as a:
        initial=a["pose"]
    with np.load(OUTPUT/"foreground/s20_sam.npz",allow_pickle=False) as a:
        np.testing.assert_array_equal(a["pts"],pts)
        masks=a["packed"]
    world=native_paths(native,[AD2_BOTTOM,AD2_RIGHT,FOOT],count=151)
    theta={}
    quality=[]
    data={}
    for i,(p,frame) in enumerate(source_frames(record["source"]["registered_path"],pts)):
        raw=frame.to_ndarray(format="bgr24")
        mask=group_foreground(np.empty((0,5)), np.array([0,0]), 0)
        mask[382:1072,650:1270]=False
        mask|=cv2.dilate(np.unpackbits(masks[i],axis=1)[:,:WIDTH],np.ones((17,17),np.uint8)).astype(bool)
        groups=measure_s20_panel(raw,mask)
        if groups:
            data[i]=groups
        if roles[i]!=0 or len(groups)!=3:
            continue
        seed=initial[i]
        names=list(groups)
        xyz=np.vstack([world[k][0] for k in names])
        starts=np.r_[0,np.cumsum([len(world[k][0]) for k in names])]

        def residual(q):
            uv=native.core.project_motion(rig,np.tile(q,(6,1)),-2.,1.,HEIGHT,0.,xyz)["uv"]
            return np.concatenate([nearest_residual(groups[k],uv[a:b]).ravel()/np.sqrt(len(groups[k]))
                    for k,a,b in zip(names,starts[:-1],starts[1:],strict=True)])
        fit=least_squares(residual,seed,x_scale=[.017,.013,.14],loss="soft_l1",f_scale=1.8,
                          bounds=(seed-[.12,.10,.7],seed+[.12,.10,.7]),max_nfev=38)
        if np.linalg.norm(fit.fun)<19.:
            theta[i]=fit.x
            quality.append(dict(pts=p,n={k:len(g) for k,g in groups.items()},
                                paired_feature_rms=float(np.linalg.norm(fit.fun)/np.sqrt(3))))
    if len(theta)<4:
        raise ValueError(f"S20: only {len(theta)} usable direct edge frames. Inspect source profiles.")
    idx=list(theta)
    delta=np.array([theta[i]-initial[i] for i in idx])
    initial+=np.column_stack([np.interp(np.arange(len(pts)),idx,x) for x in delta.T])
    controls=initialize_controls(pts,initial)
    translated=dict(pts=[],path=[],uv=[],station=[],frame=[])
    for i,groups in data.items():
        t=(pts[i]-pts[0])/12800.
        for key,uv in groups.items():
            xy,s=world[key]
            projected=native.core.project_motion(rig,controls,FIRST,DT,HEIGHT,t,xy)["uv"]
            at,e=localize(projected,s,uv)
            ok=(e<20.)&np.isfinite(at)
            translated["pts"].extend([pts[i]]*int(ok.sum()))
            translated["frame"].extend([i]*int(ok.sum()))
            translated["path"].extend([key]*int(ok.sum()))
            translated["uv"].extend(uv[ok])
            translated["station"].extend(at[ok])
    return controls,quality,translated


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--shots",nargs="+",default=["s18","s20"],choices=("s18","s20"))
    a=p.parse_args()
    record=scope()
    native=frozen_scene(record)
    cv2.setNumThreads(2)
    directory=OUTPUT/a.out
    directory.mkdir(exist_ok=True)
    with np.load(OUTPUT/a.base/"camera.npz",allow_pickle=False) as p:
        fit={k:p[k] for k in p.files}
    report={}
    by={x["id"]:x for x in record["shots"]}
    if "s18" in a.shots:
        fit["s18_controls"],report["s18"]=s18_seed(native,by["s18"],fit["s18_rig"])
    observations={"pts":[]}
    if "s20" in a.shots:
        fit["s20_controls"],report["s20"],observations=s20_seed(native,record,by["s20"],fit["s20_rig"])
        save(directory/"s20_panel_source.npz",**{k:np.array(v) for k,v in observations.items()})
    save(directory/"camera.npz",**fit)
    write(directory/"repair.json",report)
    print("Direct board initialization", {k:len(v) for k,v in report.items()},
          "S20 measured UV",len(observations["pts"]),flush=True)


if __name__=="__main__":
    main()
