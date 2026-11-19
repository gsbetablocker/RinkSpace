"""Add source rereads and initialize stations on their unchanged feature paths."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from time import monotonic

import numpy as np

from common import HEIGHT, OUTPUT, STEP, frozen_scene, read, save, scope, write
from measure import DT, FIRST, load_camera, localize, motion_clock
from stability_tracks import ji

ROW_FIELDS=("path","family","pts","idx","fold","uv","station","material",
            "normal","contrast","guide_error","eligible","eligible_before_quality_review","excluded_reason")
GPU=threading.Lock()


def read_arrays(path):
    with np.load(path,allow_pickle=False) as z:
        state={k:z[k] for k in z.files}
    state.setdefault("eligible_before_quality_review",state["eligible"].copy())
    state.setdefault("excluded_reason",np.repeat("",len(state["pts"])))
    return state


def groups(a):
    return set(zip(a["pts"][a["eligible"]],a["path"][a["eligible"]],strict=True))


def merge(a,b,intervals):
    if b is None:
        a["reread"]=np.zeros(len(a["pts"]),bool)
        return a
    keys=("source_sha256","geometry_sha256")
    for k in keys:
        np.testing.assert_array_equal(a[k],b[k])
    new=groups(b)
    keep=np.array([(t,k) not in new for t,k in zip(a["pts"],a["path"],strict=True)],bool)
    # Previously reserved frames adjacent to any new training window are
    # preserved as guard evidence and excluded from final source-pixel checks.
    near=np.zeros(len(a["pts"]),bool)
    for lo,hi in intervals:
        near|=(a["pts"]>=lo-5*STEP)&(a["pts"]<=hi+5*STEP)
    a["fold"]=np.where(near&(a["fold"]>0),3,a["fold"]).astype(np.int8)
    merged={k:np.concatenate((a[k][keep],b[k])) for k in ROW_FIELDS}
    merged["reread"]=np.r_[np.zeros(keep.sum(),bool),np.ones(len(b["pts"]),bool)]
    sequence=np.argsort(merged["pts"],kind="stable")
    merged={k:v[sequence] for k,v in merged.items()}
    sha_by_pts={}
    for item in (a,b):
        for t,digest in zip(item["source_frame_pts"],item["source_frame_BGR_sha256"],strict=True):
            if t in sha_by_pts and sha_by_pts[t]!=digest:
                raise ValueError("Decoded source differs at a reused PTS")
            sha_by_pts[t]=digest
    ids=np.array(sorted(sha_by_pts))
    merged["source_frame_pts"]=ids
    merged["source_frame_BGR_sha256"]=np.array([sha_by_pts[p] for p in ids])
    return merged|{k:a[k] for k in keys}


def recenter(shot, native, camera, rows):
    pts,controls,rig=load_camera(shot,camera)
    t=motion_clock(shot,camera,pts)
    rate=np.broadcast_to(rig,(len(pts),len(rig)))
    np.testing.assert_array_equal(rows["pts"],pts[rows["idx"]])
    selected=np.flatnonzero(rows["eligible"]&(rows["fold"]==0))
    selected=selected[np.lexsort((rows["path"][selected],rows["idx"][selected]))]
    paths=rows["path"][selected]
    indices=rows["idx"][selected]
    groups=np.split(selected,np.flatnonzero((indices[1:]!=indices[:-1])|(paths[1:]!=paths[:-1]))+1)
    names=np.unique(paths)
    with GPU:
        paths_world=ji.native_paths(native,names,count=137)
    coords={k:x for k,(x,_) in paths_world.items()}
    distance=np.full(len(rows["pts"]),np.nan)
    initial=rows["station"].copy()
    for lo in range(0,len(groups),224):
        active=groups[lo:lo+224]
        query=[(int(rows["idx"][g[0]]),str(rows["path"][g[0]]),t[rows["idx"][g[0]]]) for g in active]
        with GPU:
            projected=ji.project_groups(native,rate,controls,FIRST,DT,query,coords)
        for g,(z,code),(i,path,when) in zip(active,projected,query,strict=True):
            poly=z.copy()
            poly[code!=0]=np.nan
            if np.count_nonzero(code==0)<2:
                q=native.core.project_motion(rig,controls,FIRST,DT,HEIGHT,when,coords[path])
                poly=q["uv"]
                poly[q["status"]!=0]=np.nan
            s,e=localize(poly,paths_world[path][1],rows["uv"][g])
            ok=np.isfinite(s)&np.isfinite(e)
            # An initialization on a fixed curve changes neither the original
            # measurement, nor its physical identity, nor a qualification role.
            rows["station"][g[ok]]=s[ok]
            distance[g[ok]]=e[ok]
    return initial,distance


def process(record, shot, native, baseline, rereads, camera, output, windows, identity_review):
    sid=shot["id"]
    start=monotonic()
    a=read_arrays(baseline/f"{sid}.npz")
    b=read_arrays(rereads/f"{sid}.npz") if sid in windows else None
    rows=merge(a,b,windows.get(sid,[]))
    source=record["source"]
    assert rows["source_sha256"].item()==source["sha256"]
    assert rows["geometry_sha256"].item()==record["geometry_binding"]["value_sha256"]
    previous,distance=recenter(shot,native,camera,rows)
    reviewed=np.zeros(len(distance),bool)
    if sid in identity_review:
        x=identity_review[sid]
        checked=(rows["pts"]>=x["first_pts"])&(rows["pts"]<=x["last_pts"])
        old_white=(rows["family"]==x["family"])&(rows["material"]=="white_letter")&~rows["reread"]
        reviewed=checked&old_white&rows["eligible"]&((distance>30.)|~np.isfinite(distance))
        rows["eligible"][reviewed]=False
        rows["excluded_reason"]=np.where(reviewed,x["reason"],rows["excluded_reason"])
    rows["station_before_stability"]=previous
    rows["current_station_guide_distance_px"]=distance
    done=np.isfinite(distance)&~reviewed
    errors=distance[done]
    worst=np.argsort(-np.where(done,distance,-np.inf))[:12]
    receipt=dict(rows=int(done.sum()),newly_read=int(np.sum(done&rows["reread"])),
                 requalified_as_wrong_identity=int(reviewed.sum()),
                 projected_curve_distance_px=np.quantile(errors,[.5,.95,.99,1.]).tolist(),
                 largest=[dict(pts=int(rows["pts"][i]),family=str(rows["family"][i]),px=float(distance[i]))
                          for i in worst if done[i]],seconds=monotonic()-start)
    save(output/f"{sid}.npz",**rows)
    write(output/f"{sid}.json",receipt)
    print(sid,receipt["rows"],"fresh",receipt["newly_read"], "p50 p95 p99 max",np.round(receipt["projected_curve_distance_px"],2),
          "seconds",round(receipt["seconds"],1),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",default="read07")
    p.add_argument("--rereads",default="steady_reread_qualified")
    p.add_argument("--camera",default="steady_transport_clock")
    p.add_argument("--out",default="steady_observations")
    p.add_argument("--windows",required=True)
    p.add_argument("--identity-review")
    p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    record=scope()
    native=frozen_scene(record,verify=True)
    output=OUTPUT/a.out
    output.mkdir(exist_ok=False)
    windows=read(a.windows)
    identities=read(a.identity_review) if a.identity_review else {}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        work=[pool.submit(process,record,shot,native,OUTPUT/a.baseline,OUTPUT/a.rereads,
                     OUTPUT/a.camera,output,windows,identities) for shot in record["shots"]]
        for job in as_completed(work):
            job.result()
    write(output/"inputs.json",dict(baseline=str(OUTPUT/a.baseline),fresh=str(OUTPUT/a.rereads),
         station_initialization=str(OUTPUT/a.camera),windows=windows,identity_review=identities,
         native_source_presentation_unchanged=True))


if __name__=="__main__":
    main()
