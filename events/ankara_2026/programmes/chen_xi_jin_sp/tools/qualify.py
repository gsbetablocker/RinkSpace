"""Verify source material against recurring ambiguities seen in the first fit."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path

import cv2
import numpy as np

from common import HEIGHT, OUTPUT, WIDTH, save, scope, source_frames, write


def color_at(bgr, uv):
    blurred=cv2.GaussianBlur(bgr,(5,5),.9)
    x=np.clip(np.rint(uv[:,0]).astype(int),0,WIDTH-1)
    y=np.clip(np.rint(uv[:,1]).astype(int),0,HEIGHT-1)
    return blurred[y,x].astype(np.float32)


def nearest_rail(uv, frame_rails):
    if not len(frame_rails):
        return np.zeros(len(uv),bool)
    delta=uv[:,None,:]-frame_rails
    found=(np.abs(delta[:,:,0])<72.)&(np.abs(delta[:,:,1])<22.)
    return found.any(1)


def audit_frame(a, indices, bgr, shot):
    families=a["family"][indices]
    material=a["material"][indices]
    points=a["uv"][indices]
    normals=a["normal"][indices]
    status=np.full(len(indices),"",dtype="<U64")
    p = int(a["pts"][indices[0]])
    sec = p/12800.
    if shot=="s10" and 92.0<=sec<=95.30:
        status[families=="R3"]="wrong_or_ambiguous_layer_in_visible_house"
    if shot=="s13" and 123.44<=sec<=126.48:
        status[families=="F2"]="overlapping_ice_rings_unresolved_radial_layer"

    need = np.char.startswith(families,"goal_")|np.char.startswith(families,"trapezoid_")
    if need.any():
        rr=np.flatnonzero(need)
        col=color_at(bgr,points[rr])
        blue,green,red=col.T
        is_rail=nearest_rail(points[rr],points[families=="board.bottom"])
        warm=(green-blue>8.)&(red-blue>22.)&(red-green<1.24*(green-blue)+2.)
        status[rr[warm|is_rail]]="yellow_kickplate_or_rink_boundary_not_red_paint"

    for name in ("meiwa_blue","meiwa_green"):
        rr=np.flatnonzero(material==name)
        if not len(rr):
            continue
        q,n=points[rr],normals[rr]
        sides=np.concatenate([q+i*n for i in (-7.,-3.,0.,3.,7.)])
        blue,green,red=color_at(bgr,sides).reshape(5,len(rr),3).transpose(2,0,1)
        if name=="meiwa_blue":
            expected=(blue-red>20.)&(blue-green>10.)&(green-red>6.)
        else:
            expected=(green-blue>11.)&(green-red>5.)
        status[rr[~expected.any(axis=0)]]="adjacent_dark_glyph_not_coloured_meiwa_bar"
    return status


def qualify(src, dst, shots=None):
    doc=scope()
    dst.mkdir(exist_ok=True)
    for shot in doc["shots"]:
        sid=shot["id"]
        if shots is not None and sid not in shots:
            continue
        if not (src/f"{sid}.npz").exists():
            raise ValueError("Incomplete source read "+sid)
        with np.load(src/f"{sid}.npz",allow_pickle=False) as data:
            arrays={k:data[k] for k in data.files}
        # Preserve all source readings and their original quality decisions.
        eligible=arrays["eligible"].copy()
        reason=np.full(len(eligible),"",dtype="<U64")
        f=arrays["family"]
        audit=eligible & (
            np.char.startswith(f,"goal_")|np.char.startswith(f,"trapezoid_")|
            np.char.startswith(arrays["material"],"meiwa_")|
            ((f=="R3")&(sid=="s10"))|((f=="F2")&(sid=="s13")))
        fps=np.unique(arrays["pts"][audit])
        if len(fps):
            hash_map=dict(zip(arrays["source_frame_pts"],arrays["source_frame_BGR_sha256"],strict=True))
            for p,frame in source_frames(doc["source"]["registered_path"],fps):
                bgr=frame.to_ndarray(format="bgr24")
                if hashlib.sha256(bgr.tobytes()).hexdigest()!=hash_map[p]:
                    raise ValueError(f"{sid} source decode changed at PTS {p}")
                # Include current actual rail readings to distinguish near-rail
                # reddish shadows from the measured centre of an ice stripe.
                x=np.flatnonzero((arrays["pts"]==p)&eligible)
                labels=audit_frame(arrays,x,bgr,sid)
                reason[x]=labels
        arrays["eligible_before_quality_review"]=eligible
        arrays["excluded_reason"]=reason
        arrays["eligible"]=eligible&(reason=="")
        save(dst/f"{sid}.npz",**arrays)
        r=dict(shot=sid,rows=len(eligible),eligible=int(arrays["eligible"].sum()),
               active=int(np.count_nonzero(arrays["eligible"]&(arrays["fold"]==0))),
               decisions=dict(Counter(reason[reason!=""])),native_frames_reviewed=len(fps))
        write(dst/f"{sid}.json",r)
        print(r,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--shots",nargs="+")
    a=p.parse_args()
    cv2.setNumThreads(2)
    qualify(OUTPUT/a.source,OUTPUT/a.out,a.shots)
