"""Infer fixed working exposure steps from coherent raw camera motion changes."""
from __future__ import annotations

import argparse

import numpy as np
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import lsqr

from common import OUTPUT, STEP, save, scope, write
from measure import DT, FIRST, clock, load_camera, motion_clock
from rinkspace.domain.motion_chart import cubic_samples
from rinkspace.domain.multi_shot_motion import exposure_offset_seconds
from stability_audit import load_tracks
from stability_fit import motion_space


def flow_field(pixels, displacement):
    x=np.c_[np.ones(len(pixels)),(pixels-[960.,540.])/[960.,540.]]
    centre=np.median(displacement,axis=0)
    start=np.linalg.norm(displacement-centre,axis=1)
    keep=start<=np.quantile(start,.80)
    for _ in range(3):
        if np.count_nonzero(keep)<4:
            return None
        mat=np.vstack((x[keep],np.diag([0.,.13,.13])))
        values=np.vstack((displacement[keep],np.zeros((3,2))))
        coeff=np.linalg.lstsq(mat,values,rcond=None)[0]
        d=np.linalg.norm(displacement-x@coeff,axis=1)
        keep=d<max(1.5,min(3.5,3.*np.median(d)+.30))
    return coeff,keep


def exposure_steps(track):
    from itertools import groupby

    pick=(track["lag"]==1)&(track["family"]==0)&track["usable"]&(track["quality"]<15.)
    if "role" in track:
        pick&=track["role"]==0
    indices=track["index"]
    image=track["left_uv"]
    delta=track["uv"]-image
    fields={}
    samples={}
    rows=np.flatnonzero(pick)
    rows=rows[np.argsort(indices[rows],kind="stable")]
    for i,g in groupby(rows,key=lambda q:int(indices[q])):
        k=np.array(list(g),np.int64)
        if len(k)<4 or np.max(np.ptp(image[k],axis=0))<122.:
            continue
        found=flow_field(image[k],delta[k])
        if found is None:
            continue
        coeff,keep=found
        fields[i]=coeff
        samples[i]=k[keep]
    steps=[]
    for i in sorted(fields):
        others=[j for j in range(i-3,i+4) if j!=i and j in fields]
        before=[j for j in others if j<i]
        after=[j for j in others if j>i]
        terminal=i==len(track["pts"])-1 and len(before)==3
        if len(before)<2 or (len(after)<2 and not terminal):
            continue
        k=samples[i]
        if len(k)<8:
            continue
        x=np.c_[np.ones(len(k)),(image[k]-[960.,540.])/[960.,540.]]
        past=np.stack([x@fields[j] for j in others if j<i])
        lo=np.median(past,axis=0)
        if terminal:
            lo=past[-1]
            hi=lo+.5*(past[-1]-past[-2])
        else:
            hi=np.median([x@fields[j] for j in after],axis=0)
        pred=.5*(lo+hi)
        mag=np.linalg.norm(pred,axis=1)
        if float(np.median(mag))<6.:
            continue
        # Reject actual acceleration transitions where both sides of the
        # suspected source step do not share a coherent motion direction.
        if np.median(np.linalg.norm(hi-lo,axis=1))>.47*np.median(mag)+1.:
            continue
        power=np.einsum("ij,ij->i",pred,pred)
        observed=delta[k]
        ratio=float(np.median(np.einsum("ij,ij->i",pred,observed)/np.maximum(power,1.)))
        n=round(ratio)
        if n not in (2,3) or abs(ratio-n)>.32:
            continue
        residual=np.median(np.linalg.norm(observed-n*pred,axis=1))
        if residual>max(2.8,.165*np.median(mag)*n):
            continue
        steps.append(dict(first_pts=int(track["pts"][i]),offset_pts=(n-1)*STEP,
              fit_region_patches=len(k),repeat_factor=ratio,expected_nominal_px=float(np.median(mag)),
              measured_px=float(np.median(np.linalg.norm(observed,axis=1))),model_mismatch_px=float(residual)))
    return steps


def initialize(old,source_times,target_times,stride):
    indices,w=cubic_samples(len(old),FIRST,DT,source_times)
    values=np.einsum("ij,ijk->ik",w,old[indices])
    size=int(round(target_times[-1]/DT))+8
    space,design,_,_=motion_space(np.zeros((size,3)),target_times,stride)
    bend=np.diff(np.eye(space.shape[1]),n=2,axis=0)
    matrix=vstack((design,csr_matrix(.055*bend)),format="csr")
    values=np.vstack((values,np.zeros((len(bend),3))))
    coefficients=np.stack([lsqr(matrix,z,atol=2.e-12,btol=2.e-12,iter_lim=900)[0]
                           for z in values.T],axis=1)
    return space@coefficients


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",default="steady03")
    p.add_argument("--tracks",default="steady_tracks_full")
    p.add_argument("--out",default="steady_clock")
    a=p.parse_args()
    r=scope()
    source=OUTPUT/a.source
    tracks=OUTPUT/a.tracks
    dest=OUTPUT/a.out
    dest.mkdir(exist_ok=False)
    record={}
    with np.load(source/"camera.npz",allow_pickle=False) as f:
        output={k:f[k] for k in f.files}
    for shot in r["shots"]:
        sid=shot["id"]
        data=load_tracks(tracks/f"{sid}.npz",r,shot)
        steps=exposure_steps(data)
        if not steps:
            continue
        pts,controls,_=load_camera(shot,source)
        old=motion_clock(shot,source,pts)
        scoped=shot["processing"]|dict(exposure_pts_offsets=steps,time_base_num=1,time_base_den=12800)
        t=clock(pts)+exposure_offset_seconds(scoped,pts)
        c=initialize(controls,old,t,3)
        output[sid+"_exposure_times"]=t
        output[sid+"_controls"]=c
        record[sid]=steps
        print(sid,[round(z["first_pts"]/12800.,3) for z in steps],flush=True)
    write(dest/"working_exposure_clock.json",dict(schema="rinkspace.source_exposure_clock_working.v1",
          source_step_pts=STEP,time_base_num=1,time_base_den=12800,source_PTS_preserved=True,
          track_source=str(tracks),shots=record,
          interpretation="Fixed working exposure offsets inferred from coherent same-shot source motion. Capture/delivery cause undetermined."))
    save(dest/"camera.npz",**output)


if __name__=="__main__":
    main()
