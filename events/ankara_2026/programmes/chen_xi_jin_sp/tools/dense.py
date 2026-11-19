"""Fill source coverage between qualified observations with signed raw profiles."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
from time import monotonic

for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"

import cv2
import numpy as np

from common import HEIGHT, OUTPUT, WIDTH, frozen_scene, save, scope, source_frames, write
from measure import (BOTTOM, DT, FIRST, TOP, Reader, clock, fold_map, group_foreground,
                     load_camera, localize, spread_selection)

DATA=None
RECORD=None


class DenseReader:
    def __init__(self,record):
        r=Reader(record)
        self.native=r.native
        self.core=r.core
        self.rails=r.scan.board_read
        self.scan=r.scan
        self.ice={k:v for k,v in r.ice.items() if r.family[k] in {"B1","B2","C"} and not v[2]}
        self.family={k:r.family[k] for k in [TOP,BOTTOM,*self.ice]}
        self.paths=[TOP,BOTTOM,*self.ice]
        self.array=r.ji.native_paths(self.native,self.paths,count=151)
        offset=np.r_[0,np.cumsum([len(self.array[k][0]) for k in self.paths])]
        self.slices={k:slice(a,b) for k,a,b in zip(self.paths,offset[:-1],offset[1:],strict=True)}
        self.xyz=np.vstack([self.array[k][0] for k in self.paths])
        board=self.array[TOP][0][:,:2]
        along=np.gradient(board,axis=0)
        inward=np.c_[along[:,1],-along[:,0]]
        inward*=np.where((inward*(-board)).sum(1)<0,-1.,1.)[:,None]
        self.board_course=board
        self.inward=inward

    def peel(self,raw,mask,z,centre,needed):
        uv=z["uv"]
        status=z["status"]
        faces=np.sum((self.board_course-centre[:2])*self.inward,axis=1)<=0.
        groups=[]
        rail=self.rails
        use_rails=set((TOP,BOTTOM)) & needed
        if use_rails:
            modes={rail.TOP_MODE,rail.FOOT_MODE}
            images=rail.material_images(raw,modes)
            for k in use_rails:
                sl=self.slices[k]
                check=status[sl].copy()
                check[~faces]=1
                readings=rail.measure_rails(k,uv[sl],check,mask,images)
                if len(readings)<4:
                    continue
                pts=np.asarray([r["uv"] for r in readings])
                x=np.rint(pts[:,0]).astype(int).clip(6,WIDTH-7)
                y=np.rint(pts[:,1]).astype(int).clip(10,HEIGHT-11)
                above=raw[y-5,x].astype(float)
                below=raw[y+5,x].astype(float)
                # Physical polarity across the raw far board strip.
                if k==TOP:
                    chin=(below[:,0]-.55*below[:,1]-.45*below[:,2])
                    upstream=(above[:,0]-.55*above[:,1]-.45*above[:,2])
                    ok=(chin>7.)&((chin-upstream)>6.)
                else:
                    ch0=.6*above[:,2]+.4*above[:,1]-above[:,0]
                    ch1=.6*below[:,2]+.4*below[:,1]-below[:,0]
                    ok=(ch0>20.)&((ch0-ch1)>7.)
                readings=[r for r,v in zip(readings,ok,strict=True) if v]
                label="dense_blue_band_upper_flank" if k==TOP else "dense_yellow_strip_lower_flank"
                if len(readings)>=4:
                    groups.append((k,label,readings,6.0))

        selected={k:v for k,v in self.ice.items() if k in needed}
        if selected:
            poly=uv[self.slices[BOTTOM]]
            foot=self.scan.visible_foot(poly[status[self.slices[BOTTOM]]==0])
            rgb=np.ascontiguousarray(raw[:,:,::-1])
            for k,rows in self.scan.ice_frame(rgb,mask,z,selected,self.slices,foot).items():
                rows=[r for r in rows if r["contrast"]>=4.5 and 3.4<=r["width_px"]<=29.]
                if len(rows)>=4:
                    groups.append((k,"dense_two_paint_edges_midpoint",rows,7.0))
        return groups


def init(record):
    global DATA, RECORD
    RECORD=record
    cv2.setNumThreads(1)
    DATA=DenseReader(record)


def densify(args):
    shot,source,candidate,output=args
    start=monotonic()
    sid=shot["id"]
    with np.load(source/f"{sid}.npz",allow_pickle=False) as z:
        current={k:z[k] for k in z.files}
    if sid=="s20":
        save(output/f"{sid}.npz",**current)
        write(output/f"{sid}.json",dict(shot=sid,dense_rows=0,dense_source_frames=0,
              note="Closing shot already has direct boundary measurements on the native frame grid."))
        return
    pts,ctrl,rig=load_camera(shot,candidate)
    t=clock(pts)
    mid=DATA.core.evaluate_motion_camera(rig,ctrl,FIRST,DT,t)
    if np.any(mid["status"]):
        raise ValueError("Missing instantaneous camera for dense reads: "+sid)
    roles=fold_map(shot,len(pts))
    initial=current["eligible"]&(current["fold"]==0)
    found=defaultdict(set)
    for i,k in zip(current["idx"][initial],current["path"][initial]):
        found[int(i)].add(k)
    lookup=set(DATA.paths)
    need={i:lookup-found[i] for i in np.flatnonzero(roles==0)}
    ii=np.array(sorted(i for i,v in need.items() if v),np.int64)
    with np.load(OUTPUT/"foreground"/f"{sid}.npz",allow_pickle=False) as a:
        np.testing.assert_array_equal(a["pts"],pts)
        boxes,offset=a["boxes"],a["offsets"]
    raw_rows=defaultdict(list)
    digests=dict(zip(current["source_frame_pts"],current["source_frame_BGR_sha256"],strict=True))
    frames=0
    for p, frame in source_frames(RECORD["source"]["registered_path"],pts[ii]):
        i=int(np.searchsorted(pts,p))
        raw=frame.to_ndarray(format="bgr24")
        z=DATA.core.project_motion(rig,ctrl,FIRST,DT,HEIGHT,t[i],DATA.xyz)
        mask=group_foreground(boxes,offset,i)
        groups=DATA.peel(raw,mask,z,mid["centre"][i],need[i])
        for k,label,rows,maximum in groups:
            pixels=np.asarray([r["uv"] for r in rows])
            coords=z["uv"][DATA.slices[k]].copy()
            coords[z["status"][DATA.slices[k]]!=0]=np.nan
            station,e=localize(coords,DATA.array[k][1],pixels)
            valid=np.flatnonzero((e<maximum)&np.isfinite(station))
            if len(valid)<4 or np.linalg.norm(np.ptp(pixels[valid],axis=0))<100.:
                continue
            j=valid[spread_selection(pixels[valid],12 if k in (TOP,BOTTOM) else 8)]
            amount=len(j)
            add={
                "path":np.repeat(k,amount),"family":np.repeat(DATA.family[k],amount),"pts":np.repeat(p,amount),
                "idx":np.repeat(i,amount),"fold":np.zeros(amount,np.int8),"uv":pixels[j],
                "station":station[j],"material":np.repeat(label,amount),
                "normal":np.asarray([rows[a]["source_normal"] for a in j]),
                "contrast":np.asarray([rows[a].get("contrast",rows[a].get("source_edge_contrast",0.)) for a in j]),
                "guide_error":e[j],
            }
            for key,values in add.items():
                raw_rows[key].append(values)
        if groups and p not in digests:
            digests[p]=hashlib.sha256(raw.tobytes()).hexdigest()
        frames+=1
        if frames%150==0:
            print(sid,"source",frames,"of",len(ii),f"{monotonic()-start:.1f}s",flush=True)
    if raw_rows:
        added={k:np.concatenate(v) for k,v in raw_rows.items()}
        n=len(added["pts"])
        viable=np.zeros(n,bool)
        for k in np.unique(added["path"]):
            same=added["path"]==k
            same_old=(current["path"]==k)&initial
            combined=np.unique(np.r_[current["idx"][same_old],added["idx"][same]])
            stable=combined[np.r_[False,np.diff(combined)<=4]|np.r_[np.diff(combined)<=4,False]]
            viable|=same & np.isin(added["idx"],stable)
        added.update(eligible=viable,eligible_before_quality_review=np.ones(n,bool),
                     excluded_reason=np.where(viable,"","no_neighboring_source_read_for_dense_feature"))
        for k,v in added.items():
            current[k]=np.concatenate([current[k],v])
        extra_count=int(viable.sum())
    else:
        extra_count=0
    current["source_frame_pts"]=np.array(sorted(digests),np.int64)
    current["source_frame_BGR_sha256"]=np.array([digests[p] for p in sorted(digests)])
    save(output/f"{sid}.npz",**current)
    r=dict(shot=sid,dense_rows=extra_count,dense_source_frames=frames,
           before=int(initial.sum()),after=int(np.sum(current["eligible"]&(current["fold"]==0))),
           seconds=monotonic()-start)
    write(output/f"{sid}.json",r)
    print(r,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True)
    p.add_argument("--candidate",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--workers",type=int,default=6)
    a=p.parse_args()
    record=scope()
    output=OUTPUT/a.out
    output.mkdir(exist_ok=True)
    tasks=[(s,OUTPUT/a.source,OUTPUT/a.candidate,output) for s in record["shots"]
          if not (output/f"{s['id']}.json").exists()]
    with ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(record,)) as pool:
        list(pool.map(densify,tasks))


if __name__=="__main__":
    main()
