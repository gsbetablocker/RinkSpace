"""Initialize native camera motion using measured per-frame image displacement."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from time import monotonic

for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"

import numpy as np
from scipy.sparse import csr_matrix, diags, eye, kron, vstack
from scipy.sparse.linalg import spsolve

from common import HEIGHT, OUTPUT, frozen_scene, read, save, scope, write
from measure import DT, FIRST, load_camera, motion_clock
from rinkspace.domain.motion_chart import cubic_samples
from stability_audit import load_tracks
from stability_fit import motion_space, pairs_from_source


def exposures(c,t):
    i,w=cubic_samples(len(c),FIRST,DT,t)
    return np.einsum("ij,ijk->ik",w,c[i])


def regularization(n,scale,counts,gauge_weight):
    # The ridge fixes the absolute gauge while relative image measurements
    # determine actual changes. The stronger row gaps constrain only weak support.
    gauge=eye(3*n,format="csr")*gauge_weight
    step=np.repeat(np.where(counts[1:]>=2,.07,.74),3)
    first=diags((-1.,1.),(0,1),shape=(n-1,n),format="csr")
    # Position coordinates are converted to native sensor-pixel scale.
    differences=kron(first,eye(3),format="csr")
    bend=kron(diags((1.,-2.,1.),(0,1,2),shape=(n-2,n),format="csr"),eye(3),format="csr")
    unscale=diags(1./scale.ravel(),format="csr")
    support=kron(eye(n-2),diags([.045,.065,.16]),format="csr")
    weak=kron(eye(n-1),diags([1.,1.,.5]),format="csr")
    # Step constraints apply to the correction. For unmapped exposures the
    # measured coarse trajectory supplies the underlying motion.
    return vstack((gauge,diags(step)@weak@differences,
                   support@bend@unscale*float(np.median(scale[:,0]))),format="csr")


def paired_project(core,r,c,t,p,xyz):
    z=core.project_motion(r,c,FIRST,DT,HEIGHT,t[p].ravel(),xyz)
    return z["uv"].reshape(-1,2,2), (z["status"].reshape(-1,2)==0).all(axis=1)


def solve_transport(core,rig,c,t,tr,gauge_weight,view_radius,free_frames):
    n=len(t)
    xyz=np.repeat(tr["world"],2,axis=0)
    frames=tr["frames"]
    obs=tr["uv"][:,1]-tr["uv"][:,0]
    mass=tr["weight"]
    ref=exposures(c,t)
    space,design,_,_=motion_space(c,t,1)
    zt=core.evaluate_motion_camera(rig,c,FIRST,DT,t)
    scale=np.c_[zt["fx"],zt["fy"],np.full(n,650.)]
    counts=np.bincount(frames[tr["lag"]==1,1],minlength=n)
    pen=regularization(n,scale,counts,gauge_weight)
    free=np.repeat(free_frames,3)
    # At fixed exposure centres the sparse cardinal basis can exactly recover
    # a per-exposure correction. The small ridge selects its boundary halo.
    matrix=design.T@design+eye(design.shape[1],format="csr")*2.e-9
    coords=(3*frames[:,:,None]+np.arange(3)).reshape(-1,6)
    row=np.broadcast_to(np.arange(len(frames)*2).reshape(-1,2,1),(len(frames),2,6))
    initial_error=None
    summaries=[]
    for _ in range(6):
        uv,ok=paired_project(core,rig,c,t,frames,xyz)
        error=obs-(uv[:,1]-uv[:,0])
        if initial_error is None:
            initial_error=np.linalg.norm(error[ok],axis=1)
        jac=[]
        for k,eps in enumerate((4.e-7,4.e-7,5.e-6)):
            moved=c.copy()
            moved[:,k]+=eps
            target,valid=paired_project(core,rig,moved,t,frames,xyz)
            ok&=valid
            jac.append((target-uv)/eps)
        if np.count_nonzero(ok)<6:
            raise ValueError("Too little valid source texture to constrain motion transport")
        j=np.stack(jac,axis=3)/scale[frames][:,:,None,:]
        val=np.concatenate((-j[:,0],j[:,1]),axis=-1)
        e=np.linalg.norm(error[ok],axis=1)
        robust=np.sqrt(np.minimum(1.,3.0/np.maximum(e,1.e-5)))*mass[ok]
        row_keep=row[ok]
        col=np.broadcast_to(coords[:,None,:],row.shape)[ok]
        A=csr_matrix(((val[ok]*robust[:,None,None]).ravel(),(row_keep.ravel(),col.ravel())),
                     shape=(2*len(frames),n*3))
        b=np.zeros((len(frames),2))
        b[ok]=error[ok]*robust[:,None]
        q=exposures(c,t)
        drift=(q-ref)*scale
        small_a=A[:,free]
        small_p=pen[:,free]
        correction=np.zeros(3*n)
        correction[free]=spsolve(small_a.T@small_a+small_p.T@small_p,
               small_a.T@b.ravel()-small_p.T@(pen@drift.ravel()))
        dx=correction.reshape(n,3)
        spatial=float(np.quantile(np.linalg.norm(dx,axis=1),.995))
        alpha=min(1.,.65/max(1.e-9,float(np.max(abs(dx[:,2]))/650.)/.035),
                  80./max(1.e-9,float(np.max(abs(dx[:,:2])))))
        dq=dx*alpha/scale
        coefficients=np.column_stack([spsolve(matrix,design.T@value) for value in dq.T])
        direction=space@coefficients
        accepted=False
        loss=lambda v:float(np.sum(mass[ok]**2*np.where(v<=3.,.5*v**2,3.*(v-1.5))))
        old=loss(e)
        for step in (1.,.5,.25,.125):
            shifted=drift+step*dq*scale
            if view_radius is not None and np.any(np.linalg.norm(shifted,axis=1)>view_radius):
                continue
            proposed=c+step*direction
            trial,valid=paired_project(core,rig,proposed,t,frames,xyz)
            if not (valid|~ok).all():
                continue
            updated=np.linalg.norm((trial[:,1]-trial[:,0]-obs)[ok],axis=1)
            next_=loss(updated)
            old_prior=.5*np.linalg.norm(pen@drift.ravel())**2
            next_prior=.5*np.linalg.norm(pen@shifted.ravel())**2
            if next_+next_prior<old+old_prior:
                c=proposed
                accepted=True
                summaries.append(dict(before=old,after=next_,step=step,frame_correction_p995_px=spatial))
                break
        if not accepted or spatial<.05:
            break
    uv,ok=paired_project(core,rig,c,t,frames,xyz)
    errors=np.linalg.norm(uv[ok,1]-uv[ok,0]-obs[ok],axis=1)
    before=np.quantile(initial_error,[.50,.95,.995,1.])
    after=np.quantile(errors,[.50,.95,.995,1.])
    return c,dict(pairs=len(frames),before_px=before.tolist(),after_px=after.tolist(),
                  linear_trials=summaries,used_for_initialization=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",default="steady03")
    p.add_argument("--tracks",default="steady_tracks_full")
    p.add_argument("--out",default="steady_transport")
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--gauge",type=float,default=.28)
    p.add_argument("--view-radius",type=float,default=25.)
    p.add_argument("--windows",help="Optional source PTS windows. Only listed shots and frames receive a correction.")
    a=p.parse_args()
    started=monotonic()
    record=scope()
    native=frozen_scene(record,verify=True)
    stage=OUTPUT/a.source
    dest=OUTPUT/a.out
    dest.mkdir(exist_ok=False)
    with np.load(stage/"camera.npz",allow_pickle=False) as z:
        payload={k:z[k] for k in z.files}

    def one(shot):
        sid=shot["id"]
        pts,c,rig=load_camera(shot,stage)
        t=motion_clock(shot,stage,pts)
        data=load_tracks(OUTPUT/a.tracks/f"{sid}.npz",record,shot,windows.get(sid,[]) if windows else ())
        track=pairs_from_source(data,1.)
        free_frames=np.ones(len(pts),bool) if windows is None else np.zeros(len(pts),bool)
        if windows is not None:
            for lo,hi in windows[sid]:
                free_frames|=(pts>=lo)&(pts<=hi)
        if not free_frames.any():
            raise ValueError("Motion correction window is outside "+sid)
        control,rep=solve_transport(native.core,rig,c,t,track,a.gauge,a.view_radius,free_frames)
        return sid,control,rep

    reports={}
    windows=read(a.windows) if a.windows else None
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        future=[pool.submit(one,q) for q in record["shots"] if windows is None or q["id"] in windows]
        for job in as_completed(future):
            sid,control,report=job.result()
            payload[sid+"_controls"]=control
            reports[sid]=report
            print(sid,"relative source pixels",np.round(report["before_px"],2),
                  np.round(report["after_px"],2),flush=True)
    save(dest/"camera.npz",**payload)
    write(dest/"transport.json",dict(source=str(stage),source_tracks=str(OUTPUT/a.tracks),
          reason="Initialize exposure-centre native controls from measured source motion. Absolute positions are still fitted by the native joint solver.",
          sensor_scaled_ridge=a.gauge,view_radius_px=a.view_radius,local_source_windows=windows,
          seconds=monotonic()-started,shots=reports))


if __name__=="__main__":
    main()
