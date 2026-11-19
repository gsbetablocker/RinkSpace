"""Read the exposed right-hand glyphs of the real closing Ad2 board."""
from __future__ import annotations

import argparse
from collections import defaultdict
import sys

import cv2
import numpy as np
from scipy.signal import find_peaks

from common import HEIGHT, JOINT, OUTPUT, WIDTH, frozen_scene, save, scope, source_frames, visibility_packet, write
from measure import DT, FIRST, fold_map, load_camera, motion_clock
from physical_reads import subpixel
from review import restore_source_graphics

PREFIX="print_instance/Ad2::Ad2.print."


def right_intersection(polygon,y,stations=None):
    a,b=polygon[:-1],polygon[1:]
    use=((a[:,1]<y)&(b[:,1]>=y))|((b[:,1]<y)&(a[:,1]>=y))
    if not use.any():
        return None
    a,b=a[use],b[use]
    u=(y-a[:,1])/(b[:,1]-a[:,1])
    q=a[:,0]+(b[:,0]-a[:,0])*u
    j=int(np.argmax(q))
    if stations is None:
        return float(q[j])
    i=np.flatnonzero(use)[j]
    return float(q[j]), float(stations[i]+u[j]*(stations[i+1]-stations[i]))


def read_glyph(raw,mask,polygon,stations,hint,radius):
    visible=np.isfinite(polygon).all(1)
    if np.count_nonzero(visible)<5:
        return []
    y0=max(13,int(np.nanmin(polygon[:,1]))+5)
    y1=min(314,int(np.nanmax(polygon[:,1]))-7)
    hits=[]
    for y in range(y0,y1,8):
        row=right_intersection(polygon,y,stations)
        if row is None:
            continue
        x,station=row
        if not 18.<x+hint<WIDTH-20:
            continue
        near=int(round(x+hint))
        low=max(3,near-radius-10)
        high=min(WIDTH-3,near+radius+10)
        strip=np.median(raw[y-2:y+3,low:high].astype(np.float32),axis=0)
        if len(strip)<18:
            continue
        g=strip[:,1]
        if np.ptp(g)<27.:
            continue
        g=cv2.GaussianBlur(g[None],(7,1),1.15)[0]
        signed=-np.gradient(g)
        peaks=find_peaks(signed,height=2.95,distance=4)[0]
        available=[]
        for j in peaks:
            delta=low+j-x
            if abs(delta-hint)>radius or j<8 or j>=len(g)-9:
                continue
            if mask[y-3:y+4,low+j-7:low+j+8].any():
                continue
            on=strip[j-8:j-3]
            off=strip[j+4:j+9]
            # The outside of the last stroke is the purple panel. The
            # inside is white ink, not another colour change or the HUD.
            white=np.median(on.mean(1))>138 and np.median(on.max(1)-on.min(1))<29
            purple=np.median(off[:,0]+.58*off[:,2]-1.58*off[:,1])>17
            contrast=float(np.median(g[j-8:j-3])-np.median(g[j+4:j+9]))
            if not white or not purple or contrast<33:
                continue
            px=low+j+subpixel(signed,j)
            rank=(contrast*np.sqrt(signed[j]))/(1+abs(delta-hint)/33)
            available.append((rank,px,y,contrast,px-x))
        if available:
            _,xx,yy,q,delta=max(available)
            hits.append((xx,yy,q,delta,station))
    if len(hits)<4:
        return []
    a=np.asarray(hits)
    typical=np.median(a[:,3])
    a=a[abs(a[:,3]-typical)<7.5]
    if len(a)<4 or np.ptp(a[:,1])<24.:
        return []
    return a


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--guide",required=True)
    p.add_argument("--silhouettes",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    cv2.setNumThreads(2)
    record=scope()
    shot=record["shots"][-1]
    native=frozen_scene(record,verify=True)
    sys.path.insert(0,str(JOINT))
    from joint_inputs import native_paths

    keys=[PREFIX+"COM_"+k for k in ("M","O","C")]
    path=native_paths(native,keys,count=241)
    positions=np.vstack([path[k][0] for k in keys])
    stops=np.r_[0,np.cumsum([len(path[k][0]) for k in keys])]
    guide=OUTPUT/a.guide
    pts,c,rig=load_camera(shot,guide)
    t=motion_clock(shot,guide,pts)
    roles=fold_map(shot,len(pts))
    with np.load(visibility_packet(OUTPUT/a.silhouettes,"s20"),allow_pickle=False) as silhouettes:
        np.testing.assert_array_equal(pts,silhouettes["pts"])
        packed=silhouettes["packed"]
    folder=OUTPUT/a.out
    folder.mkdir(exist_ok=True)
    entries=defaultdict(list)
    report=[]
    for i,(ps,frame) in enumerate(source_frames(record["source"]["registered_path"],pts)):
        raw=frame.to_ndarray(format="bgr24")
        mask=np.unpackbits(packed[i],axis=1,count=WIDTH)
        mask=restore_source_graphics(raw,ps,cv2.dilate(mask,np.ones((7,7),np.uint8)))
        proj=native.core.project_motion(rig,c,FIRST,DT,HEIGHT,t[i],positions)
        uv=proj["uv"]
        uv[proj["status"]!=0]=np.nan
        hint=0.
        diagnostics={}
        for j,k in enumerate(keys):
            piece=uv[stops[j]:stops[j+1]]
            # The three unoccluded letters have distinct supports. Seed the
            # outer-stroke search using their spacing in the original crop.
            shift=hint*(1.65 if j==1 else 2.10)
            sample=read_glyph(raw,mask,piece,path[k][1],shift if j else 0.,58 if j==0 else 25)
            if not len(sample):
                if j==0:
                    break
                continue
            if j==0:
                hint=float(np.median(sample[:,3]))
            points=sample[:,:2]
            station=sample[:,4]
            err=abs(sample[:,3])
            valid=np.isfinite(station)&(err<150.)
            n=int(valid.sum())
            if n<4:
                continue
            diagnostics[k.rsplit("_",1)[-1]]=dict(rows=n,median_right_edge_offset_px=float(np.median(sample[valid,3])))
            entry=dict(path=np.repeat(k,n),family=np.repeat("Ad2",n),pts=np.repeat(ps,n),
                       idx=np.repeat(i,n),fold=np.repeat(roles[i],n),uv=points[valid],
                       station=station[valid],material=np.repeat("source_ad2_named_right_stroke",n),
                       normal=np.tile([1.,0.],(n,1)),contrast=sample[valid,2],guide_error=err[valid],
                       eligible=np.ones(n,bool),eligible_before_quality_review=np.ones(n,bool),
                       excluded_reason=np.repeat("",n))
            for field,value in entry.items():
                entries[field].append(value)
        if diagnostics:
            report.append(dict(pts=int(ps),letters=diagnostics))
    if not entries:
        raise ValueError("No unoccluded .COM print edges were identified")
    rows={k:np.concatenate(v) for k,v in entries.items()}
    for k in keys:
        q=rows["path"]==k
        observed=np.unique(rows["idx"][q])
        close=np.r_[False,np.diff(observed)<=4]|np.r_[np.diff(observed)<=4,False] if len(observed) else np.empty(0,bool)
        rows["eligible"][q]=np.isin(rows["idx"][q],observed[close])
    if not rows["eligible"].any():
        raise ValueError("No named .COM material could be located across exposures")
    save(folder/"s20_glyphs.npz",**rows,source_sha256=record["source"]["sha256"],
         geometry_sha256=record["geometry_binding"]["value_sha256"])
    result=dict(shot="s20",guide=str(guide),physical_Ad2_rightward_suffix=True,source_frames=len(report),
                rows=int(rows["eligible"].sum()),time_profiles=report)
    write(folder/"glyphs.json",result)
    print(result["source_frames"],"frames,",result["rows"],"UV,",report[:3],flush=True)


if __name__=="__main__":
    main()
