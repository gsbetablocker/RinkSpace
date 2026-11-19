"""Measure the opening closeup's visible Ad2 left vertical directly."""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

from common import HEIGHT, JOINT, OUTPUT, WIDTH, frozen_scene, save, scope, source_frames, write
from measure import DT, FIRST, clock, fold_map, group_foreground, initialize_controls, load_camera, localize
from repair_tail import measured_fall, robust_line
from rinkspace.domain.motion_chart import cubic_samples

LEFT = "print_instance/Ad2::Ad2.left_vertical"


def read_left_edge(raw, mask, projection):
    if not np.isfinite(projection).all():
        return np.empty((0,2))
    top, bottom = projection[:,1].min(), projection[:,1].max()
    rows=[]
    for y in range(max(20,int(top)+34),min(HEIGHT-28,int(bottom)-28),21):
        band=slice(y-4,y+5)
        b,g,r=np.median(raw[band,:,:],axis=0).T
        purple=b-g+.48*(r-g)
        covered=(mask[band].mean(axis=0)>.20)
        # The low-to-high purple transition needs a neutral board stretch on its left.
        x=measured_fall(purple[::-1],covered[::-1],WIDTH-1-1000,WIDTH-1-270,22.,after_span=90)
        if x is not None:
            rows.append((y,WIDTH-1-x))
    readings=robust_line(rows,.14)
    return readings[:,::-1].copy() if len(readings)>=8 and np.ptp(readings[:,0])>=140. else np.empty((0,2))


def repair(baseline, folder, use_review_frames):
    record=scope()
    shot=record["shots"][0]
    native=frozen_scene(record)
    sys.path.insert(0,str(JOINT))
    from joint_inputs import native_paths

    world,station_values=native_paths(native,[LEFT],count=151)[LEFT]
    pts,controls,rig=load_camera(shot,baseline)
    time=clock(pts)
    roles=fold_map(shot,len(pts))
    ii,ww=cubic_samples(len(controls),FIRST,DT,time)
    pose=np.einsum("ij,ijk->ik",ww,controls[ii])
    with np.load(OUTPUT/"foreground/s01.npz",allow_pickle=False) as masks:
        np.testing.assert_array_equal(masks["pts"],pts)
        boxes,offsets=masks["boxes"],masks["offsets"]
    source={}
    corrections={}
    diagnostics=[]
    for i,(p,frame) in enumerate(source_frames(record["source"]["registered_path"],pts)):
        before=native.core.project_motion(rig,controls,FIRST,DT,HEIGHT,time[i],world)["uv"]
        uv=read_left_edge(frame.to_ndarray(format="bgr24"),group_foreground(boxes,offsets,i),before)
        if len(uv)<8:
            continue
        source[i]=uv
        if roles[i]!=0 and not use_review_frames:
            continue

        def error(delta):
            q=pose[i]+np.r_[delta,0.,0.]
            z=native.core.project_motion(rig,np.tile(q,(6,1)),-2.,1.,HEIGHT,0.,world)["uv"]
            order=np.argsort(z[:,1])
            return np.interp(uv[:,1],z[order,1],z[order,0])-uv[:,0]

        fit=least_squares(error,[0.],loss="soft_l1",f_scale=1.4,bounds=(-.055,.055),x_scale=.01)
        if np.quantile(abs(fit.fun),.90)<8.:
            corrections[i]=float(fit.x[0])
            diagnostics.append(dict(pts=p,before_p50_px=float(np.median(abs(error([0.])))),
                                    corrected_p90_px=float(np.quantile(abs(fit.fun),.90))))
    if len(corrections)<20:
        raise ValueError("Too few visible opening border profiles")
    anchors=np.array(list(corrections))
    observed_pan=pose[anchors,0]+np.array(list(corrections.values()))
    offset=np.interp(np.arange(len(pts)),anchors,observed_pan)-pose[:,0]
    corrected=controls+initialize_controls(pts,np.c_[offset,np.zeros((len(pts),2))])
    rows={key:[] for key in ("pts","path","uv","station")}
    for i,xy in source.items():
        uv=native.core.project_motion(rig,corrected,FIRST,DT,HEIGHT,time[i],world)["uv"]
        st,error=localize(uv,station_values,xy)
        use=(error<19.)&np.isfinite(st)
        rows["pts"].extend([pts[i]]*int(use.sum()))
        rows["path"].extend([LEFT]*int(use.sum()))
        rows["uv"].extend(xy[use])
        rows["station"].extend(st[use])
    with np.load(baseline/"camera.npz",allow_pickle=False) as z:
        payload={k:z[k] for k in z.files}
    payload["s01_controls"]=corrected
    folder.mkdir(exist_ok=False)
    save(folder/"camera.npz",**payload)
    save(folder/"s01_panel_source.npz",**{k:np.asarray(value) for k,value in rows.items()},
         reused_review_intervals=np.asarray(use_review_frames))
    write(folder/"repair.json",dict(shot="s01",source_original=record["source"]["id"],
             method="signed purple material boundary with neutral board outside; interpolate absolute pan between verified reads",
             reused_review_intervals=use_review_frames,
             rows=len(rows["pts"]),training_anchor_count=len(corrections),profiles=diagnostics))
    print("S01 source profile count",len(source),"training anchors",len(corrections),"direct UV",len(rows["pts"]),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--use-review-frames",action="store_true")
    a=p.parse_args()
    cv2.setNumThreads(2)
    repair(OUTPUT/a.baseline,OUTPUT/a.out,a.use_review_frames)
