"""Read signed paint, rail, and print measurements from the Chen source."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from itertools import islice
import os
from pathlib import Path
import sys
from time import monotonic, sleep

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from common import HEIGHT, JOINT, OUTPUT, STEP, WIDTH, frozen_scene, read, save, scope, source_frames, source_pts, visibility_packet, write
from rinkspace.domain.motion_chart import UniformMotionChart, cubic_samples
from rinkspace.domain.multi_shot_motion import continue_exposure_halos

DT, FIRST = .02, -.06
TOP, BOTTOM = "board/continuous/top", "board/continuous/bottom"


def clock(pts):
    return (pts-pts[0])/12800.


def motion_clock(shot, stage, pts):
    if stage is not None:
        key=shot["id"]+"_exposure_times"
        with np.load(stage/"camera.npz",allow_pickle=False) as state:
            if key in state:
                native=state[key]
                if native.shape!=pts.shape or not np.isfinite(native).all() or native[0]!=0. or np.any(np.diff(native)<=0.):
                    raise ValueError("Invalid working exposure clock "+shot["id"])
                return native
    return clock(pts)


def fold_map(shot, n):
    """Hold fixed short source intervals and their immediate neighbors."""
    t = np.arange(n)
    out = np.zeros(n, dtype=np.int8)
    if n < 135:
        offset = n//3
        out[max(0,offset-3):offset+10] = 3
        out[offset:offset+7] = 2
        offset = (n*3)//4
        out[offset-3:min(n,offset+8)] = 3
        out[offset:min(n,offset+5)] = 1
    else:
        phase = (t+(int(shot["id"][1:])*21)%121)%169
        out[(phase>=48)&(phase<65)] = 3
        out[(phase>=52)&(phase<61)] = 1
        out[(phase>=113)&(phase<132)] = 3
        out[(phase>=117)&(phase<128)] = 2
    return out


def initialize_controls(pts, poses):
    t = clock(pts)
    y = gaussian_filter1d(poses, .8, axis=0, mode="nearest")
    n = len(poses)+7
    mat = UniformMotionChart.create(n,FIRST,DT,4).refinement
    mat = continue_exposure_halos(np.repeat(mat[:,None],3,axis=1),FIRST,DT,t[[0,-1]])[:,0]
    i,w = cubic_samples(n,FIRST,DT,t)
    A = csr_matrix(np.einsum("ij,ijk->ik",w,mat[i]))
    c = np.column_stack([lsqr(A,x,atol=1.e-10,btol=1.e-10)[0] for x in y.T])
    return mat @ c


def broadcast_overlay():
    mask = np.zeros((HEIGHT,WIDTH),dtype=np.uint8)
    # Actual broadcast overlays. The boundary search has no right to read
    # programme timer or technical marks as material.
    mask[:141,:561] = 1
    mask[134:279,:308] = 1
    mask[941:1024,1645:1895] = 1
    return mask


def group_foreground(boxes, offsets, i):
    v = boxes[offsets[i]:offsets[i+1]]
    mask = broadcast_overlay()
    for x0,y0,x1,y1,_ in v:
        m = 9 + .024 * min(x1-x0,y1-y0)
        a,b = max(0,int(x0-m)), min(WIDTH,int(np.ceil(x1+m)))
        c,d = max(0,int(y0-m)), min(HEIGHT,int(np.ceil(y1+m)))
        mask[c:d,a:b] = 1
    if not len(v):
        mask[382:1072,650:1270] = 1
    return mask.astype(bool)


def localize(poly, station, pixels):
    """Find station guesses on the projected path without changing source UV."""
    starts, delta = poly[:-1], np.diff(poly,axis=0)
    lengths = np.sum(delta*delta,axis=1)
    valid = np.isfinite(starts).all(1) & np.isfinite(delta).all(1) & (lengths>1.e-10)
    if not valid.any():
        return np.full(len(pixels),np.nan), np.full(len(pixels),np.inf)
    diff=pixels[:,None]-starts[valid]
    tt=np.clip(np.einsum("nki,ki->nk",diff,delta[valid])/lengths[valid],0.,1.)
    d=np.linalg.norm(diff-tt[...,None]*delta[valid],axis=2)
    j=np.argmin(d,axis=1)
    sel=np.flatnonzero(valid)[j]
    return station[sel]+tt[np.arange(len(pixels)),j]*(station[sel+1]-station[sel]), d[np.arange(len(pixels)),j]


def spread_selection(uv, count):
    if len(uv)<=count:
        return np.arange(len(uv))
    picked=[int(np.argmin(uv[:,0]+.15*uv[:,1]))]
    nearest=np.linalg.norm(uv-uv[picked[0]],axis=1)
    for _ in range(1,count):
        j=int(np.argmax(nearest))
        if nearest[j] < 4.:
            break
        picked.append(j)
        nearest=np.minimum(nearest,np.linalg.norm(uv-uv[j],axis=1))
    return np.asarray(picked)


class Reader:
    def __init__(self, record):
        sys.path.insert(0,str(JOINT))
        import joint_inputs as ji
        ji.enable_historical_source_readers()
        from current_sources import scan
        import read_current_material as material
        from read_ring_secondary import annulus

        self.ji, self.scan, self.material, self.annulus = ji, scan, material, annulus
        self.native = frozen_scene(record)
        self.core = self.native.core
        self.ice=scan.ice_paths(self.native,self.native.scene)
        self.ads=scan.advertisement_paths(self.native)
        self.extra=material.print_configs(self.native,tuple(range(1,27)))
        # Tiny QR elements are never source-fit paths.
        for catalog in (self.ads,self.extra):
            for k in list(catalog):
                if "qr_" in k.lower() and "frame" not in k.lower():
                    catalog.pop(k)
        keys=list(dict.fromkeys([*self.ice,*self.ads,*self.extra,TOP,BOTTOM]))
        self.ids=keys
        self.family={k:ji.family(k,self.native) for k in keys}
        array=ji.native_paths(self.native,keys,count=125)
        self.station={k:v[1] for k,v in array.items()}
        offset=np.r_[0,np.cumsum([len(array[k][0]) for k in keys])]
        self.slices={k:slice(int(a),int(b)) for k,a,b in zip(keys,offset[:-1],offset[1:],strict=True)}
        self.xyz=np.vstack([array[k][0] for k in keys])
        self.annuli=[tuple(k for role in ("outer","middle") for k in self.ice
                      if self.ice[k][1]==f"ice.{house}:{role}")
                      for house in [f"{side}{n}" for side in "FR" for n in range(1,5)]]

    def read(self,bgr,mask,z, max_error, use_supplement=False):
        scan, material = self.scan,self.material
        uv=z["uv"].copy()
        uv[z["status"]!=0] = np.nan
        foot=scan.visible_foot(uv[self.slices[BOTTOM]])
        rgb=np.ascontiguousarray(bgr[:,:,::-1])
        ink=scan.ice_frame(rgb,mask,z,self.ice,self.slices,foot)
        groups=defaultdict(list)
        for k, rows in ink.items():
            mode, label, _ = self.ice[k]
            if mode=="house":
                # Pair outer/middle transitions below. A small or occluded
                # circle with only one ambiguous flank stays unavailable.
                continue
            for r in rows:
                contrast=float(r["contrast"])
                if contrast < (4.7 if mode=="dark" else 2.8):
                    continue
                if ("goal" in label or "trapezoid" in label) and contrast < 5.0:
                    continue
                groups[k].append((r,"two_paint_edges_midpoint"))
        for pair in self.annuli:
            if len(pair)!=2:
                continue
            a,b=pair
            projected_a=material.subproj(z,self.slices[a])
            projected_b=material.subproj(z,self.slices[b])
            both=np.isfinite(uv[self.slices[a]]).all(1)
            coords=uv[self.slices[a]][both]
            if len(coords)<16 or np.ptp(coords,axis=0).max()<70.:
                continue
            ring=self.annulus(bgr,mask,projected_a,projected_b)
            if len(ring)<5 or (self.ice[a][1].startswith("ice.R") and any(r["tone"]!="red" for r in ring)):
                continue
            for k, role in zip(pair,("outer","middle"),strict=True):
                for r in ring:
                    groups[k].append((dict(r,uv=r[role],source_normal=r["normal"]),"paired_annulus_"+role))

        prints=scan.ad_frame(bgr,mask,z,self.ads,self.slices)
        for k, rows in prints.items():
            for r in rows:
                kind=("actual_blue_band_upper_edge" if k==TOP else "actual_yellow_kickplate_lower_edge" if k==BOTTOM
                      else r.get("material","signed_print_edge"))
                groups[k].append((r,kind))
        if use_supplement:
            available={k:v for k,v in self.extra.items() if k not in prints}
            for part in material.read_prints(bgr,mask,z,self.slices,available):
                for r in part["detail"]:
                    groups[part["path"]].append((r,part["material"]))
        found=[]
        for k,rs in groups.items():
            pts=np.asarray([r["uv"] for r,_ in rs],dtype=np.float64)
            kind=np.asarray([m for _,m in rs])
            at, error = localize(uv[self.slices[k]],self.station[k],pts)
            ok=(np.isfinite(at)&(error<=max_error)&(pts>2).all(1)&(pts<[WIDTH-2,HEIGHT-2]).all(1))
            xy=np.rint(pts[ok]).astype(int)
            candidates=np.flatnonzero(ok)[~mask[xy[:,1],xy[:,0]]]
            if len(candidates)<3:
                continue
            # Reject disjoint/duplicated jumps even when one edge is bright.
            selected=candidates[spread_selection(pts[candidates],13 if k in (TOP,BOTTOM) else 9)]
            normals=np.asarray([rs[j][0].get("source_normal",[0.,0.]) for j in selected],float)
            contrast=np.asarray([rs[j][0].get("source_edge_contrast",rs[j][0].get("contrast",0.)) for j in selected])
            found.append((k,pts[selected],at[selected],kind[selected],normals,contrast,error[selected]))
        return found


def load_camera(shot, previous):
    sid=shot["id"]
    p=OUTPUT/"navigation"/f"{sid}.npz"
    with np.load(p,allow_pickle=False) as seed:
        pts=seed["pts"]
        if previous is None:
            return pts,initialize_controls(pts,seed["pose"]),seed["rig"]
    with np.load(previous/"camera.npz",allow_pickle=False) as result:
        assert np.array_equal(result[sid+"_pts"],pts)
        return pts,result[sid+"_controls"],result[sid+"_rig"]


READER=None
RECORD=None


def worker_init(record):
    global READER, RECORD
    cv2.setNumThreads(1)
    RECORD=record
    READER=Reader(record)


def task(shot, folder, stride, previous, windows=None, silhouette_path=None, guide_distance=None):
    start=monotonic()
    sid=shot["id"]
    pts, c, rig=load_camera(shot,previous)
    np.testing.assert_array_equal(pts,source_pts(shot))
    fg_path=OUTPUT/"foreground"/f"{sid}.npz"
    with np.load(fg_path,allow_pickle=False) as fg:
        assert fg["source_sha256"].item()==RECORD["source"]["sha256"]
        np.testing.assert_array_equal(fg["pts"],pts)
        boxes,offsets=fg["boxes"],fg["offsets"]
    precise=None
    silhouette_valid=None
    direct=None
    panel_source=previous/f"{sid}_panel_source.npz" if previous is not None else None
    if panel_source is not None and panel_source.is_file():
        if sid=="s20":
            with np.load(OUTPUT/"foreground/s20_sam.npz",allow_pickle=False) as a:
                np.testing.assert_array_equal(a["pts"],pts)
                assert a["source_sha256"].item()==RECORD["source"]["sha256"]
                precise=a["packed"]
        with np.load(panel_source,allow_pickle=False) as a:
            direct={k:a[k] for k in a.files}
    if silhouette_path is not None:
        with np.load(visibility_packet(silhouette_path,sid),allow_pickle=False) as a:
            np.testing.assert_array_equal(a["pts"],pts)
            assert a["source_sha256"].item()==RECORD["source"]["sha256"]
            precise=a["packed"]
            silhouette_valid=a["valid"]
    n=len(pts)
    roles=fold_map(shot,n)
    reused_review_intervals=(direct is not None and bool(direct.get("reused_review_intervals",False)))
    if windows is not None:
        selected=np.zeros(n,bool)
        for lo,hi in windows:
            selected|=(pts>=lo)&(pts<=hi)
        if not selected.any():
            raise ValueError("No source frames in remeasurement windows for "+sid)
        indices=np.flatnonzero(selected)[::stride]
        roles[selected]=0
    elif reused_review_intervals:
        roles.fill(0)
    if windows is None:
        indices=np.unique(np.r_[np.arange(0,n,stride),n-1,
                                np.flatnonzero((roles==1)|(roles==2))[::2]])
        indices=indices[roles[indices]!=3]
    times=motion_clock(shot,previous,pts)
    source=source_frames(RECORD["source"]["registered_path"],pts[indices])
    raw_parts=defaultdict(list)
    count=Counter()
    identities=defaultdict(set)
    hashes=[]
    max_error=guide_distance if guide_distance is not None else (29. if previous is None else 21.)
    for batch in iter(lambda:list(islice(source,10)),[]):
        for p, frame in batch:
            i=int(np.searchsorted(pts,p))
            raw=frame.to_ndarray(format="bgr24")
            z=READER.core.project_motion(rig,c,FIRST,DT,HEIGHT,times[i],READER.xyz)
            mask=group_foreground(boxes,offsets,i)
            if precise is not None and (silhouette_valid is None or silhouette_valid[i]):
                person=np.unpackbits(precise[i],axis=1)[:,:WIDTH]
                mask=(broadcast_overlay()|cv2.dilate(person,np.ones((17,17),np.uint8))).astype(bool)
            groups=READER.read(raw,mask,z,max_error,
                    use_supplement=(sid in ("s01","s19","s20") or previous is not None and i%(5*stride)==0))
            if direct is not None:
                hit=direct["pts"]==p
                replacement=[]
                for key in np.unique(direct["path"][hit]):
                    q=hit&(direct["path"]==key)
                    uv=direct["uv"][q]
                    st=direct["station"][q]
                    chosen=spread_selection(uv,15)
                    uv,st=uv[chosen],st[chosen]
                    if len(uv)<4:
                        continue
                    label=("direct_yellow_lower_flank" if key==BOTTOM else "direct_purple_panel_boundary")
                    normal=np.tile([1.,0.] if key.endswith("_vertical") else [0.,1.],(len(uv),1))
                    replacement.append((key,uv,st,np.repeat(label,len(uv)),
                         normal,np.zeros(len(uv)),np.zeros(len(uv))))
                keep_paths={a[0] for a in replacement}
                groups=[g for g in groups if g[0] not in keep_paths]+replacement
            hashes.append((p,hashlib.sha256(raw.tobytes()).hexdigest()))
            for k, uv, st, kind, normal, contrast, error in groups:
                amount=len(uv)
                raw_parts["path"].append(np.repeat(k,amount))
                raw_parts["family"].append(np.repeat(READER.family[k],amount))
                raw_parts["pts"].append(np.repeat(p,amount))
                raw_parts["idx"].append(np.repeat(i,amount))
                raw_parts["fold"].append(np.repeat(roles[i],amount))
                raw_parts["uv"].append(uv)
                raw_parts["station"].append(st)
                raw_parts["material"].append(kind)
                raw_parts["normal"].append(normal)
                raw_parts["contrast"].append(contrast)
                raw_parts["guide_error"].append(error)
                identities[(READER.family[k],int(roles[i]))].add(i)
                count[READER.family[k]]+=amount
        if len(hashes)%50==0:
            print(sid,folder.name,"read",len(hashes),"of",len(indices),"UV",sum(count.values()),
                  f"{monotonic()-start:.1f}s",flush=True)
    if raw_parts:
        arrays={k:np.concatenate(v) for k,v in raw_parts.items()}
    else:
        arrays=dict(path=np.array([],dtype="<U1"),family=np.array([],dtype="<U1"),
            pts=np.empty(0,np.int64),idx=np.empty(0,np.int64),fold=np.empty(0,np.int8),
            uv=np.empty((0,2)),station=np.empty(0),material=np.array([],dtype="<U1"),
            normal=np.empty((0,2)),contrast=np.empty(0),guide_error=np.empty(0))
    # Corroboration is a repeated actual-material read for the same identity.
    # Rows never gain confidence merely by landing close to a projection.
    repeat=np.zeros(len(arrays["pts"]),bool)
    for (family,role),ii in identities.items():
        frames=np.array(sorted(ii))
        if len(frames)<2:
            continue
        delta=np.diff(frames)
        supported=frames[np.r_[False,delta<=max(14,4*stride)]|np.r_[delta<=max(14,4*stride),False]]
        repeat|=(arrays["family"]==family)&(arrays["fold"]==role)&np.isin(arrays["idx"],supported)
    arrays["eligible"]=repeat
    arrays["source_sha256"]=RECORD["source"]["sha256"]
    arrays["geometry_sha256"]=RECORD["geometry_binding"]["value_sha256"]
    arrays["source_frame_pts"]=np.array([p for p,_ in hashes],dtype=np.int64)
    arrays["source_frame_BGR_sha256"]=np.array([h for _,h in hashes])
    save(folder/f"{sid}.npz",**arrays)
    receipt=dict(shot=sid,read_frames=len(hashes),source_rows=len(repeat),
        eligible_rows=int(repeat.sum()),training_rows=int(np.count_nonzero(repeat&(arrays["fold"]==0))),
        eligible_frame_count=int(len(np.unique(arrays["idx"][repeat]))),families=dict(count),
        reused_review_intervals=reused_review_intervals or windows is not None,
        remeasurement_windows_source_pts=windows,seconds=monotonic()-start)
    write(folder/f"{sid}.json",receipt)
    print(sid,folder.name,"done",receipt,flush=True)
    return receipt


def collect(folder, stride, previous=None, shots=None, watch=False, workers=3, windows=None,
            silhouette_path=None, guide_distance=None):
    record=scope()
    folder.mkdir(parents=True,exist_ok=True)
    queue={s["id"]:s for s in record["shots"] if shots is None or s["id"] in shots}
    if windows is not None:
        queue={sid:s for sid,s in queue.items() if sid in windows}
    queue={s:r for s,r in queue.items() if not (folder/f"{s}.json").exists()}
    with ProcessPoolExecutor(max_workers=workers,initializer=worker_init,initargs=(record,)) as pool:
        active={}
        while queue or active:
            for sid in list(queue):
                if ((OUTPUT/"navigation"/f"{sid}.json").exists() and
                        (OUTPUT/"foreground"/f"{sid}.npz").exists() and
                        (silhouette_path is None or visibility_packet(silhouette_path,sid).is_file())):
                    bands=windows[sid] if windows is not None else None
                    active[pool.submit(task,queue.pop(sid),folder,stride,previous,bands,
                                       silhouette_path,guide_distance)]=sid
            ready=[f for f in active if f.done()]
            for f in ready:
                active.pop(f)
                f.result()
            if not active and queue and not watch:
                raise ValueError("Missing navigation or foreground evidence for "+", ".join(queue))
            if queue or active:
                sleep(.4)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True)
    p.add_argument("--stride",type=int,required=True)
    p.add_argument("--previous")
    p.add_argument("--shots",nargs="+")
    p.add_argument("--watch-navigation",action="store_true")
    p.add_argument("--workers",type=int,default=3)
    p.add_argument("--silhouettes")
    p.add_argument("--max-guide-distance",type=float)
    p.add_argument("--windows",help="JSON object mapping shot IDs to source PTS intervals, inclusive")
    a=p.parse_args()
    collect(OUTPUT/a.out,a.stride,
            OUTPUT/a.previous if a.previous else None,a.shots,a.watch_navigation,a.workers,
            read(a.windows) if a.windows else None,
            OUTPUT/a.silhouettes if a.silhouettes else None,a.max_guide_distance)


if __name__=="__main__":
    main()
