"""Source-only temporal shoe identity/material diagnostic, never fit observations."""
from pathlib import Path
import argparse,json
import av,cv2,numpy as np


def review(root,config_path,output):
    root,output=Path(root),Path(output);cfg=json.loads(Path(config_path).read_text());output.mkdir(parents=True,exist_ok=True);p=dict(np.load(root/'input.npz'));first=round(cfg['first_seconds']*12800);last=round(cfg['last_seconds']*12800);anchor=cfg['anchor']['pts'];images={};grays={}
    with av.open(str(root/'offline_review/source_s01.mp4')) as c:
        s=c.streams.video[0];s.codec_context.thread_count=2
        for f in c.decode(s):
            if f.pts>last:break
            if f.pts>=first:images[f.pts]=f.to_ndarray(format='bgr24');grays[f.pts]=cv2.cvtColor(images[f.pts],cv2.COLOR_BGR2GRAY)
    mask=np.zeros_like(grays[anchor]);x0,y0,x1,y1=cfg['anchor']['feature_box'];mask[y0:y1,x0:x1]=255;corners=cv2.goodFeaturesToTrack(grays[anchor],maxCorners=100,qualityLevel=.003,minDistance=3,mask=mask)
    initial=np.concatenate((np.asarray([cfg['anchor']['heel'],cfg['anchor']['toe']],np.float32)[:,None],corners),0);pts=sorted(images);ai=pts.index(anchor);tracks={anchor:dict(points=initial[:,0].copy(),alive=np.ones(len(initial),bool),fb=np.zeros(len(initial)),photo=np.zeros(len(initial)))}
    for direction in [-1,1]:
        old=initial.copy();alive=np.ones(len(initial),bool);prev=anchor
        for ix in range(ai+direction,len(pts) if direction==1 else -1,direction):
            key=pts[ix];new,status,photo=cv2.calcOpticalFlowPyrLK(grays[prev],grays[key],old,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=1e-4);back,sback,_=cv2.calcOpticalFlowPyrLK(grays[key],grays[prev],new,None,winSize=(21,21),maxLevel=3,criteria=(3,30,.01),minEigThreshold=1e-4);fb=np.linalg.norm(back[:,0]-old[:,0],axis=1);alive&=status[:,0].astype(bool)&sback[:,0].astype(bool)&(fb<cfg['tracking']['forward_backward_gate_px'])&(photo[:,0]<cfg['tracking']['photometric_gate_gray']);tracks[key]=dict(points=new[:,0].copy(),alive=alive.copy(),fb=fb,photo=photo[:,0]);old=new;prev=key
    tiles=[];rows=[]
    for key in pts:
        tr=tracks[key];rows.append(dict(pts=key,source_seconds=key/12800,heel_uv=tr['points'][0].tolist(),toe_uv=tr['points'][1].tolist(),heel_supported=bool(tr['alive'][0]),toe_supported=bool(tr['alive'][1]),heel_fb=float(tr['fb'][0]),toe_fb=float(tr['fb'][1]),material_features_alive=int(tr['alive'][2:].sum())))
    wb=dict(np.load(root/'next_round_20261029/wholebody/wholebody_coco23.npz'))
    for t in cfg['sample_seconds']:
        key=round(t*12800);frame=images[key];tr=tracks[key];i=int(np.searchsorted(wb['pts'],key));a=wb['kp2d_coco23'][i,16,:2];x0,y0=np.maximum(np.floor(a-[110,90]),0).astype(int);x1,y1=np.minimum(np.ceil(a+[110,160]),[1920,1080]).astype(int);crop=frame[y0:y1,x0:x1].copy();offset=np.array([x0,y0])
        colors=[(230,210,40),(200,50,240)]
        for j,label in enumerate(['H','T']):
            if tr['alive'][j]:
                xy=tuple(np.rint(tr['points'][j]-offset).astype(int));cv2.circle(crop,xy,3,colors[j],1,cv2.LINE_AA);cv2.putText(crop,label,xy,cv2.FONT_HERSHEY_SIMPLEX,.4,colors[j],1)
        if tr['alive'][:2].all():cv2.arrowedLine(crop,tuple(np.rint(tr['points'][0]-offset).astype(int)),tuple(np.rint(tr['points'][1]-offset).astype(int)),(60,225,100),1,cv2.LINE_AA,tipLength=.15)
        canvas=np.full((390,330,3),24,np.uint8);scale=min(310/crop.shape[1],300/crop.shape[0]);im=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST);canvas[65:65+im.shape[0],10:10+im.shape[1]]=im;cv2.putText(canvas,f'SOURCE {t:.3f}s',(10,26),cv2.FONT_HERSHEY_SIMPLEX,.6,(245,245,245),1,cv2.LINE_AA);cv2.putText(canvas,f'LK H/T support {int(tr["alive"][0])}/{int(tr["alive"][1])}; features {tr["alive"][2:].sum()}',(10,51),cv2.FONT_HERSHEY_SIMPLEX,.43,(245,245,245),1,cv2.LINE_AA);tiles.append(canvas)
    cv2.imwrite(str(output/'continuous_shoe_source.png'),np.vstack((np.hstack(tiles[:5]),np.hstack(tiles[5:]))));summary=dict(schema='rinkspace.source-only-temporal-shoe-audit.v1',source_exposures=len(pts),anchor=cfg['anchor'],thresholds_are_engineering_diagnostics=True,original_pixels_or_semantics_edited=False,used_as_fitting_observations=False,tracks=rows)
    (output/'source_tracks.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(dict(frames=len(pts),anchor_features=len(initial)-2,critical=[r for r in rows if r['pts'] in [325120,332800,336128]])),flush=True);return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();review(a.root,a.config,a.output)
