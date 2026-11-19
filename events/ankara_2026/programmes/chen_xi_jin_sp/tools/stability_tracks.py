"""Read stationary board and ice tracklets on the Chen original PTS."""
from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass
import sys
from time import monotonic

import cv2
import numpy as np
from scipy.spatial import cKDTree

from common import (HEIGHT, JOINT, OUTPUT, WIDTH, frozen_scene, observation_packet, save, scope,
                    sha, source_frames, source_pts, visibility_packet, write)
from measure import DT, FIRST, group_foreground, load_camera, motion_clock

sys.path.insert(0,str(JOINT))
sys.path.insert(1,str(JOINT/"stability_1113"))
import joint_inputs as ji
from track_source import bind, board_mask, lk, pick, rail_points, safe_mask

SIZE = (WIDTH//2, HEIGHT//2)
SEED_STEP = 3
TRACK_AGE = 10
BLOCK = 28


@dataclass
class Tracklet:
    start: int
    ids: np.ndarray
    family: np.ndarray
    world: np.ndarray
    anchor: np.ndarray
    current: np.ndarray
    previous: np.ndarray


def current_ice(input_path):
    with np.load(input_path,allow_pickle=False) as data:
        mask=data["eligible"]&~np.char.startswith(data["family"],"Ad")&~np.char.startswith(data["family"],"board.")
        uv=data["uv"][mask]
        idx=data["idx"][mask]
    order=np.argsort(idx,kind="stable")
    return {int(idx[g[0]]):uv[g] for g in np.split(order,np.flatnonzero(np.diff(idx[order]))+1) if len(g)}


def masks(rail_uv, status, nrail, ice_uv, occlusion, gray, cropped_board=None):
    board=board_mask(rail_uv,status,nrail) if cropped_board is None else cropped_board
    ice=np.zeros(gray.shape,np.uint8)
    for u in ice_uv:
        cv2.circle(ice,tuple(np.rint((u-.5)*.5).astype(int)),9,255,-1)
    ice[(board!=0)|(gray>235)]=0
    invalid=cv2.dilate(occlusion.astype(np.uint8),np.ones((7,7),np.uint8))!=0
    board[invalid]=0
    ice[invalid]=0
    return board,ice,invalid


def between(point,mask):
    x,y=np.rint(point).astype(int).T
    return ((x>=8)&(x<SIZE[0]-8)&(y>=8)&(y<SIZE[1]-8)&
            ~mask[y.clip(0,SIZE[1]-1),x.clip(0,SIZE[0]-1)])


def measured_pair(pieces, i, g, older, lag, quality, fold):
    amount=len(g.ids)
    triple=g.previous if lag==1 else np.full_like(g.current,np.nan)
    pieces.append(dict(index=np.full(amount,i,np.int32),lag=np.full(amount,lag,np.int8),
                       start=np.full(amount,g.start,np.int32),fold=np.full(amount,fold,np.int8),
                       world=g.world,uv=g.current*2.+.5,left_uv=older*2.+.5,prev2_uv=triple*2.+.5,
                       anchor_uv=g.anchor*2.+.5,track_id=g.ids,family=g.family,quality=quality))


def spatial_roles(world,family,foot):
    # All observations of one track use the same guarded surface strip.
    board=family==0
    along=cKDTree(foot[:,:2]).query(world[board,:2])[1]/(len(foot)-1)
    phase=np.empty(len(family))
    phase[board]=(42.*along+.13)%1.
    phase[~board]=(24.*world[~board,0]+11.*world[~board,1]+.13)%1.
    reserved=(phase>=.18)&(phase<=.34)
    guard=(phase>=.09)&(phase<=.43)
    return np.where(reserved,3,np.where(guard,2,0)).astype(np.int8)


def ice_corners(gray, points):
    """Require source texture with support across two axes in an ice patch."""
    keep=[]
    for x, y in points:
        patch=cv2.getRectSubPix(gray,(13,13),(float(x),float(y))).astype(np.float32)
        gx=(patch[1:-1,2:]-patch[1:-1,:-2])*.5
        gy=(patch[2:,1:-1]-patch[:-2,1:-1])*.5
        tensor=np.array([[np.mean(gx*gx),np.mean(gx*gy)],
                         [np.mean(gx*gy),np.mean(gy*gy)]])
        minor,major=np.linalg.eigvalsh(tensor)
        keep.append(minor >= max(.22,.09*major))
    return points[np.asarray(keep,bool)]


def collect(record, shot, native, raw_world, foot, baseline, input_dir, output_dir, continuous,
            silhouettes=None, two_axis_ice=False, actual_graphics=False):
    from types import SimpleNamespace
    from review import restore_source_graphics

    started=monotonic()
    sid=shot["id"]
    pts,control,rig=load_camera(shot,baseline)
    np.testing.assert_array_equal(pts,source_pts(shot))
    times=motion_clock(shot,baseline,pts)
    source=record["source"]
    data=OUTPUT/"foreground"
    with np.load(data/f"{sid}.npz",allow_pickle=False) as fg:
        np.testing.assert_array_equal(fg["pts"],pts)
        assert fg["source_sha256"].item()==source["sha256"]
        boxes,offsets=fg["boxes"],fg["offsets"]
    silhouette=feet=valid_silhouette=None
    if silhouettes is not None or sid=="s20":
        mask=visibility_packet(silhouettes,sid) if silhouettes is not None else data/"s20_sam.npz"
        with np.load(mask,allow_pickle=False) as fg:
            np.testing.assert_array_equal(fg["pts"],pts)
            assert fg["source_sha256"].item()==source["sha256"]
            silhouette=fg["packed"]
            valid_silhouette=fg["valid"] if "valid" in fg else np.ones(len(pts),bool)
    if sid=="s20":
        with np.load(OUTPUT/"tail20_edges/s20_panel_source.npz",allow_pickle=False) as original:
            feet={int(p):original["uv"][(original["pts"]==p)&(original["path"]=="board/continuous/bottom")]
                  for p in np.unique(original["pts"])}
    if actual_graphics and (valid_silhouette is None or not valid_silhouette.all()):
        raise ValueError("Per-exposure graphics require complete foreground masks")
    patch_margin=np.ones((31,31),np.uint8) if actual_graphics else None
    sample=current_ice(observation_packet(input_dir,sid))
    d=SimpleNamespace(times=times,first=FIRST,dt=DT)
    height=float(native.scene["board"][5])
    fno=int(sid[1:])
    output=output_dir/f"{sid}.npz"
    pieces=[]
    previous=None
    live=[]
    raw=source_frames(source["registered_path"],pts)
    rows=np.repeat(rig[None,:],len(pts),axis=0)
    for left in range(0,len(pts),48):
        right=min(left+48,len(pts))
        projected=ji.project_groups(native,rows,control,FIRST,DT,
            [(i,"rail",times[i]) for i in range(left,right)],{"rail":raw_world})
        for i,(rail,st) in enumerate(projected,left):
            p,frame=next(raw)
            assert p==pts[i]
            gray=cv2.resize(frame.to_ndarray(format="gray"),SIZE,interpolation=cv2.INTER_AREA)
            invalid_full=group_foreground(boxes,offsets,i)
            cropped=None
            if silhouette is not None and valid_silhouette[i]:
                from measure import broadcast_overlay
                person=np.unpackbits(silhouette[i],axis=1,count=WIDTH)
                if actual_graphics:
                    raw_bgr=frame.to_ndarray(format="bgr24")
                    invalid_full=cv2.dilate(restore_source_graphics(raw_bgr,p,person),patch_margin)
                else:
                    invalid_full=broadcast_overlay()|person
            if feet is not None:
                bottom=feet.get(p,np.empty((0,2)))
                if len(bottom)>=6 and np.ptp(bottom[:,0])>=350.:
                    lower=np.polyval(np.polyfit(bottom[:,0],bottom[:,1],1),np.arange(SIZE[0])*2.)
                    y=np.arange(SIZE[1])[:,None]*2.
                    cropped=((y>=20.)&(y<lower[None,:]-30.)).astype(np.uint8)*255
            occ=cv2.resize(invalid_full.astype("u1"),SIZE,interpolation=cv2.INTER_NEAREST)!=0
            # Ice points only select new patches at their measured exposure.
            uv=sample.get(i,np.empty((0,2)))
            board,ice,invalid=masks(rail,st,len(foot),uv,occ,gray,cropped)
            block=i//BLOCK
            fold=(block+fno)%4
            if i%BLOCK==0 and not continuous:
                live=[]
            if live:
                sizes=np.array([len(g.ids) for g in live])
                tracked=lk(gray,previous,np.vstack([g.current for g in live]))
                upcoming=[]
                if tracked is not None:
                    for g,ix in zip(live,np.split(np.arange(sizes.sum()),np.cumsum(sizes)[:-1]),strict=True):
                        xy,known,quality=(a[ix] for a in tracked)
                        ok=known&between(xy,invalid)&((g.family==1)|safe_mask(board,xy))
                        if not ok.any():
                            continue
                        current=Tracklet(g.start,g.ids[ok],g.family[ok],g.world[ok],g.anchor[ok],xy[ok],g.current[ok])
                        age=i-g.start
                        if age<=SEED_STEP+1:
                            triplet=g.previous[ok] if age>=2 else np.full_like(current.current,np.nan)
                            # Record the same patch on every exposure used for the second difference.
                            adjacent=Tracklet(current.start,current.ids,current.family,current.world,
                                              current.anchor,current.current,triplet)
                            measured_pair(pieces,i,adjacent,g.current[ok],1,quality[ok],fold)
                        if age==8:
                            measured_pair(pieces,i,current,current.anchor,8,quality[ok],fold)
                        if age<TRACK_AGE:
                            upcoming.append(current)
                live=upcoming
            if i%SEED_STEP==0 or i%BLOCK==0:
                board_p=pick(gray,board,72)
                ice_p=pick(gray,ice,35)
                if two_axis_ice:
                    ice_p=ice_corners(gray,ice_p)
                q=np.vstack((board_p,ice_p))
                families=np.r_[np.zeros(len(board_p),np.uint8),np.ones(len(ice_p),np.uint8)]
                if len(q):
                    use,xyz=bind(native.core,d,control,rig,i,q,families,foot,height)
                    if len(use):
                        live.append(Tracklet(i,(i*256+use).astype(np.int32),families[use],xyz,q[use],q[use],q[use]))
            previous=gray
            if i%300==299:
                print(sid,"source",i+1,"/",len(pts),"in",round(monotonic()-started,1),flush=True)
    try:
        next(raw)
        raise ValueError("Source yielded more than the full shot")
    except StopIteration:
        pass
    if not pieces:
        raise ValueError("No stationary tracklets in "+sid)
    arrays={k:np.concatenate([v[k] for v in pieces]) for k in pieces[0]}
    if continuous:
        arrays["role"]=spatial_roles(arrays["world"],arrays["family"],foot)
    save(output,pts=pts,source_sha256=source["sha256"],geometry_sha256=record["geometry_binding"]["value_sha256"],
         graphics_mask_per_exposure=actual_graphics,**arrays)
    receipt=dict(shot=sid,frames=len(pts),rows=len(arrays["index"]),
        by_family={("board" if k==0 else "ice"):int(v) for k,v in Counter(arrays["family"]).items()},
        by_lag={str(k):int(v) for k,v in Counter(arrays["lag"]).items()},
        adjacent_frames=len(np.unique(arrays["index"][arrays["lag"]==1])),
        block_frames=BLOCK,across_whole_shot=continuous,
        guarded_spatial_holdout=continuous,two_axis_ice_seeds=two_axis_ice,graphics_mask_per_exposure=actual_graphics,
        silhouettes=str(silhouettes) if silhouettes is not None else None,seconds=monotonic()-started)
    write(output.with_suffix(".json"),receipt)
    print(sid,receipt,flush=True)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--baseline",default="fit07")
    p.add_argument("--observations",default="read07")
    p.add_argument("--out",default="steady_tracks")
    p.add_argument("--workers",type=int,default=5)
    p.add_argument("--shots",nargs="+")
    p.add_argument("--continuous",action="store_true")
    p.add_argument("--silhouettes",help="Use the source-bound instance mask at each exposure")
    p.add_argument("--two-axis-ice",action="store_true",help="Require a two-dimensional texture patch on ice")
    p.add_argument("--actual-graphics",action="store_true",help="Check broadcast graphics on each original frame")
    p.add_argument("--rebind-from",help="Relift existing unchanged source tracklets onto the current frozen surface")
    a=p.parse_args()
    cv2.setNumThreads(1)
    record=scope()
    if sha(record["source"]["registered_path"])!=record["source"]["sha256"]:
        raise ValueError("Original source hash differs")
    ji.enable_historical_source_readers()
    native=frozen_scene(record,verify=True)
    world,foot=rail_points(native)
    out=OUTPUT/a.out
    out.mkdir(exist_ok=True)
    jobs=[s for s in record["shots"] if (a.shots is None or s["id"] in a.shots) and not (out/(s["id"]+".json")).exists()]
    jobs.sort(key=lambda s:s["id"] not in {"s04","s08","s10","s14","s17","s19","s20"})
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        if a.rebind_from:
            active=[pool.submit(rebind,record,shot,native,foot,OUTPUT/a.baseline,OUTPUT/a.rebind_from,out)
                    for shot in jobs]
        else:
            masks=OUTPUT/a.silhouettes if a.silhouettes else None
            active=[pool.submit(collect,record,shot,native,world,foot,OUTPUT/a.baseline,OUTPUT/a.observations,
                                out,a.continuous,masks,a.two_axis_ice,a.actual_graphics)
                    for shot in jobs]
        for f in as_completed(active):
            f.result()


def rebind(record,shot,native,foot,camera,origin,dest):
    from types import SimpleNamespace

    sid=shot["id"]
    pts,c,rig=load_camera(shot,camera)
    d=SimpleNamespace(times=motion_clock(shot,camera,pts),first=FIRST,dt=DT)
    with np.load(origin/f"{sid}.npz",allow_pickle=False) as z:
        a={k:z[k] for k in z.files}
    np.testing.assert_array_equal(a["pts"],pts)
    assert a["source_sha256"].item()==record["source"]["sha256"]
    assert a["geometry_sha256"].item()==record["geometry_binding"]["value_sha256"]
    _,found,inverse=np.unique(a["track_id"],return_index=True,return_inverse=True)
    start=a["start"][found]
    worlds=a["world"][found].copy()
    used=np.zeros(len(found),bool)
    height=float(native.scene["board"][5])
    for group in np.split(np.arange(len(found)),np.flatnonzero(np.diff(start))+1):
        q=found[group]
        i=int(start[group[0]])
        uv=(a["anchor_uv"][q]-.5)*.5
        ix,value=bind(native.core,d,c,rig,i,uv,a["family"][q],foot,height)
        worlds[group[ix]]=value
        used[group[ix]]=True
    a["world"]=worlds[inverse]
    a["current_depth_rebound"]=used[inverse]
    save(dest/f"{sid}.npz",**a)
    receipt=dict(shot=sid,tracks=len(found),rebound=int(used.sum()),original_pixels_preserved=True,
                 spatial_roles_preserved=True,source=str(origin/f"{sid}.npz"),camera=str(camera))
    write(dest/f"{sid}.json",receipt)
    print(sid,receipt["rebound"],"/",receipt["tracks"],"trace depths rebound",flush=True)


if __name__=="__main__":
    main()
