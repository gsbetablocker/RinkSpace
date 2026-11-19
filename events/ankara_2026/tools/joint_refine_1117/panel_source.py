"""Read the actual Ad16 dark-panel boundary over additional source poses."""
from argparse import ArgumentParser
from hashlib import sha256
import cv2
import numpy as np
from shared import OUT,arrays,read,save,write
from review import camera_state,native_for,source_frames,ji


def boundary(raw,point,normal):
    s=np.arange(-40.,41.);tangent=np.array([-normal[1],normal[0]])
    at=point[None,None,:]+s[None,:,None]*normal+np.array([-3.,0.,3.])[:,None,None]*tangent
    if ((at[...,0]<1)|(at[...,0]>1918)|(at[...,1]<1)|(at[...,1]>1078)).any():return None
    values=cv2.remap(raw.astype(np.float32),at.astype(np.float32),None,cv2.INTER_LINEAR)
    luma=values@np.array([.114,.587,.299],np.float32)
    v=np.median(luma,axis=0);smooth=cv2.GaussianBlur(v[None].astype('f4'),(7,1),1.1)[0]
    rise=np.gradient(smooth);choices=[]
    for j in range(16,len(s)-16):
        if rise[j]<1. or rise[j]<max(rise[j-1],rise[j+1]):continue
        inside=float(np.median(v[j-13:j-6]));outside=float(np.median(v[j+7:j+14]))
        if outside-inside<16.:continue
        half=(inside+outside)*.5
        crossings=np.flatnonzero((v[j-5:j+5]<=half)&(v[j-4:j+6]>=half))
        if not len(crossings):continue
        k=j-5+crossings[np.argmin(abs(crossings-5))]
        at_s=s[k]+(half-v[k])/max(v[k+1]-v[k],1.e-8)
        # The interior must be the dark panel, not a bright letter or white rail.
        contrast=outside-inside
        choices.append((float(rise[j])*min(contrast,90.),at_s,inside,outside,contrast))
    if not choices:return None
    score,offset,inside,outside,contrast=max(choices)
    return dict(uv=point+offset*normal,offset=offset,contrast=contrast,
        source_flanks=np.array([point+(offset-10)*normal,point+(offset+10)*normal]),score=score)


def run(candidate):
    setup=read(OUT/'prepared/setup.json');base=camera_state(candidate)
    folder=OUT/'ad16_extended_source';folder.mkdir(exist_ok=True)
    output=[];cards=[]
    requests={'nayeon_s06':np.arange(104.3,111.91,.4),'jin_s14':np.arange(121.1,124.01,.2)}
    for name,seconds in requests.items():
        shot=setup['shots'][name];native=native_for(candidate,shot['program'])
        keys=[k for k in native.paths if k.startswith('print_instance/Ad16::') and any(
            k.endswith(x) for x in ('bottom_edge','top_edge','left_vertical','right_vertical'))]
        table=ji.native_paths(native,keys,count=101)
        pts=base[name+'_pts'];requested=np.array([round(v*50)*256 for v in seconds],np.int64)
        requested=requested[(requested>=pts[0])&(requested<=pts[-1])]
        for pt,raw in source_frames(shot['source'],requested):
            idx=int(np.searchsorted(pts,pt));rig=base[name+'_rigs'][idx];c=base[name+'_controls'];time=base[name+'_times'][idx]
            projected={k:native.core.project_motion(rig,c,shot['first'],shot['dt'],1080,time,w)['uv'] for k,(w,_) in table.items()}
            alluv=np.vstack(list(projected.values()));centre=np.mean(alluv,axis=0)
            if not np.isfinite(alluv).all():continue
            lo=np.floor(alluv.min(0)-55).astype(int);hi=np.ceil(alluv.max(0)+55).astype(int)
            if hi[0]-lo[0]<70 or hi[1]-lo[1]<55:continue
            marked=raw.copy();new=[]
            for k,uv in projected.items():
                tangent=np.gradient(uv,axis=0)
                for j in (18,34,50,66,82):
                    normal=np.array([-tangent[j,1],tangent[j,0]])
                    normal/=np.linalg.norm(normal)
                    if normal@(uv[j]-centre)<0:normal=-normal
                    hit=boundary(raw,uv[j],normal)
                    if hit is None:continue
                    row=dict(shot=name,pts=int(pt),idx=idx,path=k,station=float(table[k][1][j]),
                        uv=hit['uv'],guide_uv=uv[j],source_normal=normal,source_flanks=hit['source_flanks'],
                        source_contrast=hit['contrast'],source_score=hit['score'],
                        source_frame_BGR_sha256=sha256(raw.tobytes()).hexdigest(),source_sha256=shot['source_sha256'])
                    output.append(row);new.append(row)
                    cv2.drawMarker(marked,tuple(np.rint(hit['uv']).astype(int)),(0,0,255),cv2.MARKER_CROSS,7,1)
            if not new:continue
            lo=np.maximum(lo,[0,0]);hi=np.minimum(hi,[1920,1080])
            if (hi<=lo).any():continue
            a=raw[lo[1]:hi[1],lo[0]:hi[0]];b=marked[lo[1]:hi[1],lo[0]:hi[0]]
            card=np.hstack((a,b));scale=min(2.,1500/card.shape[1])
            card=cv2.resize(card,None,fx=scale,fy=scale)
            dest=folder/(name+'_'+str(pt)+'.jpg');cv2.imwrite(str(dest),card,[cv2.IMWRITE_JPEG_QUALITY,97])
            cards.append(dict(shot=name,pts=int(pt),seconds=pt/12800.,path=str(dest),rows=len(new),
                source_crop=[*lo.tolist(),*hi.tolist()],display_scale=scale))
            print(name,pt/12800.,len(new),flush=True)
    if output:save(folder/'source_proposals.npz',**{k:np.asarray([v[k] for v in output]) for k in output[0]})
    write(folder/'manifest.json',dict(candidate=candidate,cards=cards,rows=len(output),
        admission='pending original-image identity audit; not yet a solver input',
        method='dark panel to bright exterior half-luma crossing; native projection supplies search orientation only'))


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__);ap.add_argument('--candidate',required=True);run(ap.parse_args().candidate)
