"""Recover source rail measurements in gaps with visible paired board edges."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

from common import HEIGHT, JOINT, OUTPUT, WIDTH, frozen_scene, read, save, scope, source_frames, write
from measure import (BOTTOM, DT, FIRST, TOP, clock, fold_map, group_foreground,
                     initialize_controls, load_camera, localize)
from repair_tail import robust_line
from rinkspace.domain.motion_chart import cubic_samples

PATHS = (TOP, BOTTOM)
COLUMNS = np.arange(66, WIDTH-28, 60)


def profile_edges(image, mask, guide, lower):
    smoothed = cv2.GaussianBlur(image.reshape(-1, 1), (1, 11), 1.7)[:, 0]
    gradient = np.gradient(smoothed)
    k = np.arange(max(13, int(guide)-145), min(HEIGHT-14, int(guide)+146))
    before = (smoothed[k-4] + smoothed[k-6] + smoothed[k-8]) / 3
    after = (smoothed[k+4] + smoothed[k+6] + smoothed[k+8]) / 3
    support = (before < -17.) & (after < 17.) if lower else (before < 14.) & (after > 11.)
    valid = support & (after-before > (10. if lower else 8.))
    valid &= ~(mask[k-7] | mask[k] | mask[k+7])
    valid &= (gradient[k] >= gradient[k-1]) & (gradient[k] > gradient[k+1])
    selected = k[valid]
    strength = (after-before)[valid]
    score = strength - .048*abs(selected-guide)
    use = np.argsort(-score)[:7]
    j = selected[use]
    a, g, b = gradient[j-1], gradient[j], gradient[j+1]
    frac = .5*(a-b) / np.minimum(-1.e-8, a-2*g+b)
    return j+np.clip(frac,-.5,.5), strength[use]


def y_at_x(poly, x):
    a, d = poly[:-1], np.diff(poly, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = (np.asarray(x)[:,None]-a[:,0])/d[:,0]
    ys = np.where((tt>=0.)&(tt<=1.),a[:,1]+tt*d[:,1],np.inf)
    return ys.min(axis=1)


def read_edges(bgr, mask, poly):
    guide = np.array([y_at_x(p,COLUMNS) for p in poly]).T
    readings = []
    for x, (top, bottom) in zip(COLUMNS, guide, strict=True):
        if not np.isfinite([top,bottom]).all() or not (50. < bottom-top < 510.):
            continue
        span = slice(x-6,x+7)
        b, g, r = np.median(bgr[:,span],axis=1).T
        occluded = np.mean(mask[:,span],axis=1)>.15
        ty, ts = profile_edges(b-.57*g-.43*r, occluded, top, False)
        by, bs = profile_edges(b-.60*r-.40*g, occluded, bottom, True)
        if not (len(ty) and len(by)):
            continue
        separation = by[None,:]-ty[:,None]
        prior = bottom-top
        score = ts[:,None]+bs[None,:]-.09*abs(separation-prior)
        score -= .048*(abs(ty-top)[:,None]+abs(by-bottom)[None,:])
        valid = (separation>.60*prior)&(separation<1.45*prior)
        if valid.any():
            i,j = np.unravel_index(np.argmax(np.where(valid,score,-np.inf)),score.shape)
            readings.append((x,ty[i],by[j],ts[i],bs[j],top,bottom))
    if len(readings)<8:
        return None
    rows=np.asarray(readings)
    for observed, expected in ((1,5),(2,6)):
        line=robust_line(np.c_[rows[:,0],rows[:,observed]-rows[:,expected]],.20)
        rows=rows[np.isin(rows[:,0],line[:,0])]
    if len(rows)<8 or np.ptp(rows[:,0])<480.:
        return None
    return rows


class Board:
    def __init__(self, record):
        sys.path.insert(0,str(JOINT))
        from joint_inputs import native_paths

        self.native = frozen_scene(record)
        self.core = self.native.core
        self.paths = native_paths(self.native, PATHS, count=401)
        self.world = np.vstack([self.paths[k][0] for k in PATHS])
        self.course = self.paths[TOP][0][:,:2]
        delta = np.gradient(self.course,axis=0)
        self.inward = np.c_[delta[:,1],-delta[:,0]]
        self.inward *= np.where((self.inward*(-self.course)).sum(1)<0,-1.,1.)[:,None]

    def project(self,rig,controls,first,dt,t,centre):
        face = np.sum((self.course-centre[:2])*self.inward,axis=1)<=0.
        z = self.core.project_motion(rig,controls,first,dt,HEIGHT,t,self.world)
        z["uv"][(z["status"]!=0)|~np.tile(face,2)] = np.nan
        return np.split(z["uv"],2)

    def read_anchor(self,bgr,mask,rig,controls,t,index,local,centre):
        base = self.project(rig,controls,FIRST,DT,t,centre)
        raw = read_edges(bgr,mask,base)
        if raw is None:
            return None
        x=raw[:,0]

        def error(q):
            new = np.tile(local + np.r_[0.,q], (6,1))
            poly = self.project(rig,new,-2.,1.,0.,centre)
            dy = np.concatenate([y_at_x(a,x)-raw[:,j] for j,a in enumerate(poly,1)])
            return np.where(np.isfinite(dy),dy,500.)

        trial = least_squares(error,[0.,0.],loss="soft_l1",f_scale=2.,
                              bounds=([-.095,-.45],[.095,.45]),x_scale=[.025,.10],max_nfev=27)
        if np.quantile(abs(trial.fun),.90)>6.:
            return None
        initial = np.concatenate([y_at_x(a,x)-raw[:,j] for j,a in enumerate(base,1)])
        return dict(index=index,raw=raw,correction=np.r_[0.,trial.x],
                    before=float(np.median(abs(initial))),paired_edge_p90=float(np.quantile(abs(trial.fun),.90)))


def recover(record, board, shot, camera, coverage, current):
    sid = shot["id"]
    p,c,r = load_camera(shot,camera)
    t=clock(p)
    roles=fold_map(shot,len(p))
    candidate_gaps=[]
    for gap in coverage["direct_coverage_gaps"]:
        i,j = np.searchsorted(p,[gap["first_pts"],gap["last_pts"]])
        ii = np.arange(i,j+1)
        if gap["frames"]>=21 and np.count_nonzero(roles[ii]==0)>=5:
            candidate_gaps.append(ii)
    if not candidate_gaps:
        return c, current, []
    ids=np.concatenate(candidate_gaps)
    ids=ids[roles[ids]==0]
    groups=np.searchsorted([g[-1] for g in candidate_gaps],ids)
    with np.load(OUTPUT/f"foreground/{sid}.npz",allow_pickle=False) as fg:
        np.testing.assert_array_equal(fg["pts"],p)
        assert fg["source_sha256"].item() == record["source"]["sha256"]
        boxes,offsets=fg["boxes"],fg["offsets"]
    centres=board.core.evaluate_motion_camera(r,c,FIRST,DT,t)["centre"]
    knots,weights=cubic_samples(len(c),FIRST,DT,t)
    instantaneous=np.einsum("ij,ijk->ik",weights,c[knots])
    primary=defaultdict(list)
    digests=dict(zip(current["source_frame_pts"],current["source_frame_BGR_sha256"],strict=True))
    additions=defaultdict(list)
    for (pt, frame), i, group in zip(source_frames(record["source"]["registered_path"],p[ids]),ids,groups,strict=True):
        pixels=frame.to_ndarray(format="bgr24")
        a=board.read_anchor(pixels,group_foreground(boxes,offsets,i),r,c,t[i],i,instantaneous[i],centres[i])
        if a:
            a["sha"]=hashlib.sha256(pixels.tobytes()).hexdigest()
            primary[group].append(a)

    offset=np.zeros_like(instantaneous)
    reports=[]
    for k,ii in enumerate(candidate_gaps):
        anchors=primary[k]
        row=dict(first_pts=int(p[ii[0]]),last_pts=int(p[ii[-1]]),fit_frames=int(np.sum(roles[ii]==0)),
                  rail_anchors=len(anchors),active_added_rows=0,active_added_frames=0)
        reports.append(row)
        if len(anchors)<3:
            continue
        jj=[a["index"] for a in anchors]
        low, high=max(0,jj[0]-6), min(len(p)-1,jj[-1]+6)
        range_=np.arange(low,high+1)
        corr=np.vstack([np.zeros(3),[a["correction"] for a in anchors],np.zeros(3)])
        offset[range_]=np.column_stack([np.interp(range_,[low,*jj,high],v) for v in corr.T])
        row["paired_per_frame_seed_before_p50"]=float(np.median([a["before"] for a in anchors]))
        row["paired_per_frame_seed_fit_p90_median"]=float(np.median([a["paired_edge_p90"] for a in anchors]))
    control=c+initialize_controls(p,offset)
    for k, r0 in enumerate(reports):
        if r0["rail_anchors"]<3:
            continue
        for a in primary[k]:
            i,rows=a["index"],a["raw"]
            poly=board.project(r,control,FIRST,DT,t[i],centres[i])
            used=False
            for j, (path, family) in enumerate(zip(PATHS,("board.top","board.bottom"),strict=True),1):
                uv=rows[:,[0,j]]
                station,d=localize(poly[j-1],board.paths[path][1],uv)
                hit=(d<13.)&np.isfinite(station)
                n=int(hit.sum())
                if n<6:
                    continue
                used=True
                values=dict(path=np.repeat(path,n),family=np.repeat(family,n),pts=np.repeat(p[i],n),
                    idx=np.repeat(i,n),fold=np.zeros(n,np.int8),uv=uv[hit],station=station[hit],
                    material=np.repeat("dense_paired_raw_board_rails",n),normal=np.tile([0.,1.],(n,1)),
                    contrast=rows[hit,j+2],guide_error=d[hit],eligible=np.ones(n,bool),
                    eligible_before_quality_review=np.ones(n,bool),excluded_reason=np.repeat("",n))
                for field, data in values.items():
                    additions[field].append(data)
                r0["active_added_rows"]+=n
            if used:
                digests[p[i]]=a["sha"]
                r0["active_added_frames"]+=1
    if additions:
        for key, elements in additions.items():
            if key not in current:
                fill=np.ones(len(current["pts"]),bool) if key=="eligible_before_quality_review" else np.repeat("",len(current["pts"]))
                current[key]=fill
            current[key]=np.concatenate([current[key],*elements])
    current["source_frame_pts"]=np.array(sorted(digests),np.int64)
    current["source_frame_BGR_sha256"]=np.array([digests[p] for p in sorted(digests)])
    return control,current,reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",required=True)
    p.add_argument("--observations",required=True)
    p.add_argument("--seed-out",required=True)
    p.add_argument("--reads-out",required=True)
    a=p.parse_args()
    cv2.setNumThreads(1)
    record=scope()
    board=Board(record)
    baseline=OUTPUT/a.candidate
    output=OUTPUT/a.reads_out
    seed=OUTPUT/a.seed_out
    output.mkdir(exist_ok=False)
    seed.mkdir(exist_ok=False)
    coverage=read(baseline/"source_coverage.json")
    with np.load(baseline/"camera.npz",allow_pickle=False) as z:
        fit={k:z[k] for k in z.files}
    report={}
    for shot in record["shots"]:
        sid=shot["id"]
        with np.load(OUTPUT/a.observations/f"{sid}.npz",allow_pickle=False) as z:
            current={k:z[k] for k in z.files}
        c,data,result=recover(record,board,shot,baseline,coverage[sid],current)
        if any(row["active_added_frames"] for row in result):
            fit[sid+"_controls"]=c
        save(output/f"{sid}.npz",**data)
        report[sid]=result
        if result:
            print(sid,result,flush=True)
    save(seed/"camera.npz",**fit)
    write(output/"paired_material_recovery.json",dict(source=str(OUTPUT/a.observations),candidate=str(baseline),shots=report))


if __name__=="__main__":
    main()
