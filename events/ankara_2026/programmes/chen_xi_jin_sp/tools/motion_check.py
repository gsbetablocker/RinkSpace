"""Audit adjacent-exposure camera motion against original bidirectional feature flow."""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from common import HEIGHT, OUTPUT, frozen_scene, scope, write
from fit import statistics
from measure import DT, FIRST, clock, load_camera


def compare(core, shot, stage):
    pts,c,rig=load_camera(shot,stage)
    t=clock(pts)
    with np.load(OUTPUT/"navigation"/f"{shot['id']}_flow.npz",allow_pickle=False) as z:
        np.testing.assert_array_equal(z["pts"][:len(pts)],pts)
        matrix=z["adjacent_h"][:len(pts)-1]
        stable=(z["counts"][:len(pts)-1]>=24)&(z["error_p90"][:len(pts)-1]<=2.1)
    state=core.evaluate_motion_camera(rig,c,FIRST,DT,t)
    if np.any(state["status"]) or np.min(state["fx"])<=0 or np.min(state["fy"])<=0:
        raise ValueError("Missing or invalid camera on required original PTS "+shot["id"])
    sensor=np.array([[450.,340.],[1460.,340.],[960.,550.],[525.,830.],[1410.,820.]])
    fx=state["values"][:,:]
    q=(sensor[None,:,:]-fx[:,None,18:20])/fx[:,None,16:18]
    cases=state["centre"][:,None,:]+.55*np.einsum("mni,mij->mnj",
               np.concatenate((q,np.ones((*q.shape[:2],1))),axis=2),state["rotation"])
    tested=[]
    nodes=[]
    for j in np.flatnonzero(stable):
        p=core.project_motion(rig,c,FIRST,DT,HEIGHT,t[j],cases[j])
        y=core.project_motion(rig,c,FIRST,DT,HEIGHT,t[j+1],cases[j])
        if np.any(p["status"]) or np.any(y["status"]):
            continue
        source=cv2.perspectiveTransform(p["uv"][None],matrix[j])[0]
        tested.append(float(np.median(np.linalg.norm(y["uv"]-source,axis=1))))
        nodes.append(int(pts[j+1]))
    a=np.asarray(tested)
    top=np.argsort(a)[-7:][::-1]
    return dict(shot=shot["id"],covered_camera_frames=len(t),stable_flow_edges=int(stable.sum()),
          checked_edges=len(tested),median_motion_error_px=statistics(a),
          strongest=[dict(pts=nodes[i],median_disagreement_px=float(a[i])) for i in top],
          focal_range_px=[float(np.min(state["fx"])),float(np.max(state["fx"]))],
          thickness_readout_ms=float(rig[37]*1000)),a


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",required=True)
    p.add_argument("--base")
    a=p.parse_args()
    record=scope()
    native=frozen_scene(record)
    reports={}
    for name in (a.candidate,a.base):
        if name is None:
            continue
        packed=[compare(native.core,s,OUTPUT/name) for s in record["shots"]]
        reports[name]=dict(shots=[v for v,_ in packed],
            aggregate=statistics(np.concatenate([n for _,n in packed])))
        print(name,reports[name]["aggregate"],flush=True)
    write(OUTPUT/a.candidate/"motion_check.json",dict(
            interpretation="Relative native camera projection compared with static original image flow. "
                           "The source flow also contributed to initialization.",
            trials=reports))


if __name__=="__main__":
    main()
