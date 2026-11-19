"""Version actual material readings and preserve reasons for replaced UV rows."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import OUTPUT, WIDTH, save, scope, visibility_packet, write
from stability_observations import ROW_FIELDS, read_arrays


def qualified_profiles(a):
    good=a["eligible"].copy()
    reason=a["excluded_reason"].astype("<U72",copy=True)
    for name in np.unique(a["path"]):
        items=np.flatnonzero(good&(a["path"]==name))
        if not len(items):
            continue
        grouped=np.split(items,np.flatnonzero(np.diff(a["idx"][items]))+1)
        idx=np.array([int(a["idx"][g[0]]) for g in grouped])
        shift=np.array([np.median(a["source_projection_signed_offset_px"][g]) for g in grouped])
        spans=np.array([np.linalg.norm(np.ptp(a["uv"][g],axis=0)) for g in grouped])
        widths=np.array([np.median(np.linalg.norm(a["source_bracket"][g,1]-a["source_bracket"][g,0],axis=1))
                         for g in grouped])
        paired="source_two_flanks_paint_centre"==a["material"][items[0]]
        for k,g in enumerate(grouped):
            adjacent=(abs(idx-idx[k])<=8)&(idx!=idx[k])
            similar=adjacent&(abs(shift-shift[k])<(9. if paired else 6.5))
            if paired:
                similar&=(widths>=.46*widths[k])&(widths<=2.20*widths[k])
            minimum=1 if abs(shift[k])<11. else 2
            if np.count_nonzero(similar)<minimum:
                good[g]=False
                reason[g]="signed_boundary_or_stroke_unconfirmed_in_nearby_exposures"
            elif not paired and abs(shift[k])>20. and spans[k]<340.:
                good[g]=False
                reason[g]="large_boundary_excursion_without_width_support"
    return good,reason


def active_under_silhouette(folder,sid,rows):
    dest=visibility_packet(folder,sid) if folder else None
    if dest is None or not dest.is_file():
        return np.zeros(len(rows["pts"]),bool)
    with np.load(dest,allow_pickle=False) as z:
        assert z["source_sha256"].item()==rows["source_sha256"].item()
        np.testing.assert_array_equal(z["pts"][rows["idx"]],rows["pts"])
        packed=z["packed"]
        valid=z["valid"]
    u,v=np.rint(rows["uv"]).astype(np.int64).T
    bad=np.zeros(len(u),bool)
    for x,y in ((0,0),(-3,0),(3,0),(0,-3),(0,3)):
        b=u+x
        c=v+y
        choose=(b>=0)&(b<WIDTH)&(c>=0)&(c<packed.shape[1])&valid[rows["idx"]]
        q=np.flatnonzero(choose)
        bad[q]|=((packed[rows["idx"][q],c[q],b[q]//8]>>(7-b[q]%8))&1)!=0
    return bad


def merge(shot,baseline,new,output,silhouettes):
    sid=shot["id"]
    old=read_arrays(baseline/f"{sid}.npz")
    now=read_arrays(new/f"{sid}.npz")
    for k in ("source_sha256","geometry_sha256"):
        np.testing.assert_array_equal(old[k],now[k])
    valid,reasons=qualified_profiles(now)
    now["eligible"]=valid
    now["excluded_reason"]=reasons
    pairs=set(zip(now["path"][valid],now["pts"][valid],strict=True))
    replaced=np.array([a in pairs for a in zip(old["path"],old["pts"],strict=True)])
    masked=active_under_silhouette(silhouettes,sid,old)
    old["eligible_before_quality_review"]|=old["eligible"]
    pending=old["eligible"].copy()
    old["eligible"] &= ~(replaced|masked)
    old["excluded_reason"]=np.where(masked&pending,"source_geometry_hidden_by_foreground_silhouette",
                         np.where(replaced&pending,"same_exposure_measured_again_from_signed_material",
                                  old["excluded_reason"]))
    # Preserve original rows and reasons. No source coordinate or material
    # identity is replaced by a projection.
    arrays={k:np.concatenate([old[k],now[k]]) for k in ROW_FIELDS}
    order=np.argsort(arrays["pts"],kind="stable")
    arrays={k:v[order] for k,v in arrays.items()}
    arrays["direct_native_profile"]=np.r_[np.zeros(len(pending),bool),np.ones(len(valid),bool)][order]
    hashes={}
    for a in (old,now):
        for t,h in zip(a["source_frame_pts"],a["source_frame_BGR_sha256"],strict=True):
            if t in hashes and hashes[t]!=h:
                raise ValueError(f"Source image changed at {t}")
            hashes[t]=h
    keys=np.array(sorted(hashes),np.int64)
    arrays.update(source_sha256=old["source_sha256"],geometry_sha256=old["geometry_sha256"],
                  source_frame_pts=keys,source_frame_BGR_sha256=np.array([hashes[i] for i in keys]))
    save(output/f"{sid}.npz",**arrays)
    training=arrays["eligible"]&(arrays["fold"]==0)
    result=dict(shot=sid,active_native_profiles=int(np.sum(valid&(now["fold"]==0))),
                selection_native_profiles=int(np.sum(valid&(now["fold"]==1))),
                reserved_native_profiles=int(np.sum(valid&(now["fold"]==2))),
                original_replaced=int(np.sum(replaced&pending)),
                original_hidden=int(np.sum(masked&pending)),
                new_rejected=int(np.sum(~valid)),active=int(np.sum(training)),
                active_frames=len(np.unique(arrays["pts"][training])),
                material=dict(Counter(arrays["material"][training])))
    print(sid,"physical",result["active_native_profiles"],"replaced",result["original_replaced"],
          "masked",result["original_hidden"],"active",result["active"],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",default="steady_observations_final")
    parser.add_argument("--closeup-reread",help="source-based replacement reads for S01 and S20")
    parser.add_argument("--readings",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--silhouettes")
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    baseline=OUTPUT/args.baseline
    new=OUTPUT/args.readings
    output=OUTPUT/args.out
    silhouettes=OUTPUT/args.silhouettes if args.silhouettes else None
    output.mkdir(exist_ok=False)
    record=scope()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(merge,s,OUTPUT/args.closeup_reread
                         if args.closeup_reread and s["id"] in ("s01","s20") else baseline,
                         new,output,silhouettes) for s in record["shots"]]
        results=[job.result() for job in jobs]
    write(output/"inputs.json",dict(previous=str(baseline),native_material_profiles=str(new),
                                   closeup_reread=str(OUTPUT/args.closeup_reread) if args.closeup_reread else None,
                                   silhouettes=str(silhouettes) if silhouettes else None,shots=results))


if __name__=="__main__":
    main()
