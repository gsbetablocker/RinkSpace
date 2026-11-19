"""Compare source holdouts to the same frozen curves and audit temporal coverage."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np

from common import HEIGHT, JOINT, OUTPUT, STEP, frozen_scene, observation_packet, read, scope, write
from fit import row_layer, statistics
from measure import DT, FIRST, load_camera, localize, motion_clock


def curve_errors(native, shot, data, previous):
    sys.path.insert(0,str(JOINT))
    from joint_inputs import native_paths

    pts, control, rig = load_camera(shot,previous)
    samples = native_paths(native,np.unique(data["path"]),count=221)
    times = motion_clock(shot,previous,pts)
    uv = data["uv"]
    error = np.full(len(uv),np.inf)
    for j in np.unique(data["idx"]):
        selected = np.flatnonzero(data["idx"] == j)
        ids = data["path"][selected]
        unique = np.unique(ids)
        arr = [samples[k][0] for k in unique]
        offsets = np.r_[0,np.cumsum(list(map(len,arr)))]
        result = native.core.project_motion(rig,control,FIRST,DT,HEIGHT,times[j],np.vstack(arr))
        proj = result["uv"]
        proj[result["status"]!=0] = np.nan
        for k, a, b in zip(unique, offsets[:-1], offsets[1:], strict=True):
            rr = selected[ids==k]
            _,error[rr] = localize(proj[a:b],samples[k][1],uv[rr])
    return error


def audit(folder, candidate, baseline, holdouts, fold, excluded=None):
    folder.mkdir(exist_ok=True)
    record = scope()
    native = frozen_scene(record)
    report = {}
    totals = defaultdict(list)
    for shot in record["shots"]:
        sid = shot["id"]
        with np.load(observation_packet(holdouts,sid),allow_pickle=False) as a:
            keep = a["eligible"]&(a["fold"]==fold)
            if excluded is not None:
                for lo,hi in excluded.get(sid,[]):
                    keep&=(a["pts"]<lo-5*STEP)|(a["pts"]>hi+5*STEP)
            arrays = {k:a[k][keep] for k in ("path","idx","pts","uv","family","normal","station")}
        e0,e1 = (curve_errors(native,shot,arrays,stage) if len(arrays["idx"]) else np.array([])
                 for stage in (baseline,candidate))
        layer = row_layer(arrays["family"])
        distinct = np.unique(arrays["pts"])
        q = [dict(pts=int(p),rows=int((arrays["pts"]==p).sum()),
             before=statistics(e0[arrays["pts"]==p]),after=statistics(e1[arrays["pts"]==p])) for p in distinct]
        worst = sorted(q,key=lambda row:-row["after"].get("p95",float("inf")))[:5]
        report[sid] = dict(before=statistics(e0),after=statistics(e1),frame_count=len(distinct),worst=worst,
              by_layer={name:dict(before=statistics(e0[layer==name]),after=statistics(e1[layer==name]))
                         for name in ("ice","board","ad")})
        for fld, mask in (("all",np.ones(len(e0),bool)),*((x,layer==x) for x in ("ice","board","ad"))):
            totals[fld+"_before"].append(e0[mask])
            totals[fld+"_after"].append(e1[mask])
        print(sid,report[sid]["before"],"->",report[sid]["after"],flush=True)
    item = dict(candidate=str(candidate),baseline=str(baseline) if baseline else "original_photo_navigation",
         source_data=str(holdouts),fold=fold,
         excluded_repair_windows=excluded or {},
         assessment="matched-material distance to the native curve; same programme pose navigation used these images",
         all={name:statistics(np.concatenate(data)) for name,data in totals.items()},shots=report)
    write(folder/"comparison.json",item)
    return item


def gaps(expected, present):
    missing = ~np.isin(expected,present)
    continuations = np.flatnonzero(np.diff(np.r_[False,missing,False])).reshape(-1,2)
    return [dict(first_pts=int(expected[a]),last_pts=int(expected[b-1]),frames=int(b-a))
            for a,b in continuations]


def coverage(candidate):
    record = scope()
    result = {}
    with np.load(candidate/"active_rows.npz",allow_pickle=False) as data:
        raw_pts = data["pts"]
        shot_id = data["shot"]
        layer = row_layer(data["family"])
        for shot in record["shots"]:
            sid=shot["id"]
            keep=shot_id==sid
            expected=np.arange(shot["processing"]["first_pts"],shot["processing"]["stop_pts_exclusive"],STEP)
            found=np.unique(raw_pts[keep])
            result[sid]=dict(expected_frames=len(expected),active_rows=int(keep.sum()),frames_with_rows=len(found),
               by_layer={name:int(np.sum(keep&(layer==name))) for name in ("ice","board","ad")},
               direct_coverage_gaps=gaps(expected,found))
    write(candidate/"source_coverage.json",result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",required=True)
    p.add_argument("--baseline")
    p.add_argument("--holdouts",default="read01")
    p.add_argument("--fold",type=int,choices=(1,2),default=1)
    p.add_argument("--out",required=True)
    p.add_argument("--exclude-windows")
    a=p.parse_args()
    fit=OUTPUT/a.candidate
    coverage(fit)
    audit(OUTPUT/a.out,fit,OUTPUT/a.baseline if a.baseline else None,OUTPUT/a.holdouts,a.fold,
          read(a.exclude_windows) if a.exclude_windows else None)


if __name__=="__main__":
    main()
