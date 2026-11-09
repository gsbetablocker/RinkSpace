"""Prioritize original-source shoe crops outside the repaired action, without fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from calibration.program_tracking.body_tracking.rebuild_prepare import load, digest
from rinkspace.viewer3d.source import SourceVideo


def review_other_shots(prepared, result, source_video, output):
    p=load(prepared/'input.npz');obs=load(prepared/'observations.npz')
    pts=p['pts'];shots=p['shot_ids'];tb=tuple(map(int,p['time_base']))
    np.testing.assert_array_equal(pts,obs['pts'])
    with np.load(result,allow_pickle=False) as saved:
        np.testing.assert_array_equal(pts,saved['pts'])
        model=saved['equipment_uv'][:,:,:14]
    uv=obs['uv'];use=obs['use']
    score=np.asarray(obs.get('source_score',use),float)
    if score.shape!=use.shape:score=use.astype(float)
    options={};scan=[]
    for shot in dict.fromkeys(shots):
        if shot=='s09':continue
        for f in (0,1):
            k,a=13+f,15+f
            shin=np.linalg.norm(uv[:,k]-uv[:,a],axis=1)
            gap=np.linalg.norm(uv[:,15]-uv[:,16],axis=1)
            eligible=(shots==shot)&use[:,k]&use[:,a]&np.isfinite(shin)&(shin>20.)
            quality=np.where(eligible,np.minimum(score[:,k],score[:,a])*np.sqrt(np.clip(shin,0.,250.))*
                             (.6+.4*np.minimum(gap/np.maximum(shin,1.),1.)),-1.)
            picked=[]
            for j in np.argsort(-quality):
                if not eligible[j] or len(picked)>=4:break
                if any(abs(j-q)<18 for q in picked):continue
                picked.append(j)
                options.setdefault(j,[]).append((shot,f,float(quality[j]),float(shin[j])))
            scan.append(dict(shot=str(shot),foot=f,native_exposures=int(np.sum(shots==shot)),
                             eligible_saved_detector=int(eligible.sum()),original_frames_sampled=len(picked)))
    output.mkdir(parents=True)
    reader=SourceVideo(source_video,tb);candidates={}
    try:
        for j,feet in sorted(options.items()):
            stamp=int(pts[j]);rgb=reader.read(stamp)
            for shot,f,quality,shin in feet:
                centre=uv[j,15+f];half=np.clip(.66*shin,27.,120.)
                # A source-centred ROI always survives even if the model
                # shoe is grossly wrong. Include nearby model shoe as context.
                window=np.asarray([centre-half,centre+half])
                nearby=np.clip(model[j,f],centre-1.4*half,centre+1.4*half)
                lo=np.floor(np.maximum(np.minimum(window[0],nearby.min(0)),0)).astype(int)
                hi=np.ceil(np.minimum(np.maximum(window[1],nearby.max(0)),[rgb.shape[1],rgb.shape[0]])).astype(int)
                cut=rgb[lo[1]:hi[1],lo[0]:hi[0]]
                if min(cut.shape[:2])<18:continue
                gray=cut.astype(float).mean(-1)
                edge=float(np.percentile(np.abs(np.diff(gray,axis=0)),90)+
                           np.percentile(np.abs(np.diff(gray,axis=1)),90))
                item=dict(shot=str(shot),foot=f,pts=stamp,seconds=stamp*tb[0]/tb[1],
                    detector_priority=quality,edge_contrast_proxy=edge,bounds_xyxy=np.r_[lo,hi].tolist(),
                    source_crop_sha256=hashlib.sha256(cut.tobytes()).hexdigest(),
                    state='candidate crop for visual review; not an admitted contour or measured shoe identity')
                candidates.setdefault((shot,f),[]).append((edge*quality,item,cut))
    finally:reader.close()
    chosen=[max(rows,key=lambda r:r[0]) for rows in candidates.values()]
    panel=Image.new('RGB',(4*292,((len(chosen)+3)//4)*224),(21,25,32));draw=ImageDraw.Draw(panel)
    for i,(_,row,cut) in enumerate(chosen):
        x,y=(i%4)*292,(i//4)*224
        draw.text((x+10,y+6),f'{row["shot"].upper()}  {row["seconds"]:.2f}s  foot hypothesis {row["foot"]}',fill=(236,241,247))
        crop=Image.fromarray(cut);crop.thumbnail((276,191))
        panel.paste(crop,(x+10+(276-crop.width)//2,y+24))
    panel.save(output/'shoe_crop_review.jpg',quality=92)
    with (output/'other_shots.json').open('x',encoding='utf8') as f:
        json.dump(dict(time_base=tb,source_video_sha256=digest(source_video),native_exposures=len(pts),
            original_frames_decoded=len(options),source_shot_counts=scan,prioritized=[r[1] for r in chosen],
            no_new_weighted_rows=True),f,ensure_ascii=False,indent=2)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    for k in ('prepared','result','source-video','output'):a.add_argument('--'+k,type=Path,required=True)
    p=a.parse_args()
    review_other_shots(p.prepared,p.result,p.source_video,p.output)
