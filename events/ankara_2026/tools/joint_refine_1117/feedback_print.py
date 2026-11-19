"""Register the same physical board artwork across sharp and motion-blurred exposures."""
from collections import defaultdict

import cv2
import numpy as np

from feedback_source import DEST,PARENT,contour_samples,read_contour,selected_paths,signals
from lane_end_evidence import foreground
from source_paint import matte
from review import camera_state,ji,native_for,source_frames
from shared import PREPARED,read,save
from shared import arrays

REF_NAME='nayeon_s04'
REF_PTS=813568
PRINTS={'Ad1':'QR_white_frame','Ad24':'outer_ring.outer','Ad25':'shield_print_component001',
        'Ad26':'outer_circle_print_stroke'}
WINDOWS={'jin_s04':[(42.18,('Ad1','Ad25','Ad26'))],
         'nayeon_s03':[(42.34,('Ad1',))],
         'nayeon_s04':[(62.56,tuple(PRINTS)),(63.56,tuple(PRINTS))]}


def homography_points(x,H):
    return cv2.perspectiveTransform(np.ascontiguousarray(x,dtype='f8')[None],H)[0]


def match(ref,img,mask,ref_model,projected,ref_shift,family):
    select=np.isfinite(ref_model).all(1)&np.isfinite(projected).all(1)
    if select.sum()<8:return None
    H,_=cv2.findHomography(ref_model[select],projected[select],method=0)
    if H is None:return None
    test=homography_points(ref_model[select],H)-projected[select]
    if np.quantile(np.linalg.norm(test,axis=1),.9)>2:return None
    H=H@np.array([[1,0,-ref_shift[0]],[0,1,-ref_shift[1]],[0,0,1.]])
    lo,hi=projected[select].min(0),projected[select].max(0)
    w,h=hi-lo
    if min(w,h)<12 or max(w,h)>530:return None
    before=np.maximum([4,4],[w*.15,h*.18])
    beyond=np.maximum([5,5],[w*(1.1 if family in ('Ad25','Ad26') else .13),h*.18])
    # The crop must contain mostly the actual printed region after clipping.
    a=np.maximum([0,0],np.floor(lo-before)).astype(int);b=np.minimum([1920,1080],np.ceil(hi+beyond)).astype(int)
    if min(b-a)<14 or np.prod(b-a)<.65*w*h:return None
    margin=np.array([66,28])
    l=np.maximum([0,0],a-margin);u=np.minimum([1920,1080],b+margin)
    width,height=b-a
    T=np.array([[1.,0,-a[0]],[0,1.,-a[1]],[0,0,1.]])
    source=cv2.warpPerspective(ref,T@H,(width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    target=img[l[1]:u[1],l[0]:u[0]]
    if np.std(source)<6 or np.std(target)<5:return None
    best=None
    for blur in (1,5,11,19,31,47,65):
        pat=cv2.blur(source,(blur,1)) if blur>1 else source
        corr=cv2.matchTemplate(target,pat,cv2.TM_CCOEFF_NORMED)
        _,quality,_,(x,y)=cv2.minMaxLoc(corr)
        if best is None or quality>best[0]:
            best=(quality,x,y,corr,blur)
    quality,x,y,corr,blur=best
    if quality<.59 or x in (0,corr.shape[1]-1) and a[0]>margin[0] and b[0]<1920-margin[0]:return None
    xy=np.array([x,y])+l
    blocked=mask[xy[1]:xy[1]+height,xy[0]:xy[0]+width]
    if np.mean(blocked!=0)>.09:return None
    # Reject repeated letters or a nearby panel that would match just as well.
    rr,cc=np.indices(corr.shape)
    other=corr[((rr-y)/7)**2+((cc-x)/max(8.,.36*blur))**2>1.]
    gap=quality-float(np.max(other)) if len(other) else 1.
    if gap<.021:return None
    delta=np.zeros(2)
    for axis,(j,k) in enumerate([(x,y),(y,x)]):
        v=corr[k] if axis==0 else corr[:,k]
        if 0<j<len(v)-1:
            delta[axis]=np.clip((v[j+1]-v[j-1])/(2*max(1e-6,2*v[j]-v[j-1]-v[j+1])),-.6,.6)
    return H,xy-a+delta,quality,gap,blur


def run():
    cv2.setNumThreads(1)
    b=camera_state(PARENT);n=native_for(PARENT,'nayeon');s0=read(PREPARED/'setup.json')['shots']
    paths=selected_paths(n);refs={f:next(k for k in paths[f] if k.endswith(pat)) for f,pat in PRINTS.items()}
    tab=ji.native_paths(n,list({k for f in refs for k in paths[f]}),count=193);world={k:v[0] for k,v in tab.items()}
    rs=s0[REF_NAME];ref_i=int(np.searchsorted(b[REF_NAME+'_pts'],REF_PTS))
    _,rgb=next(source_frames(rs['source'],[REF_PTS]));ref_gray=signals(rgb)
    ref_mask=foreground(rs,matte(rs,b[REF_NAME+'_pts']),ref_i,REF_PTS)
    anchors={}
    for f,k in refs.items():
        curves={}
        for key in paths[f]:
            p=n.core.project_motion(b[REF_NAME+'_rigs'][ref_i],b[REF_NAME+'_controls'],rs['first'],rs['dt'],
                                    1080,b[REF_NAME+'_times'][ref_i],world[key])
            curves[key]=p['uv'],p['status']
        poly=curves[k][0];parts=contour_samples(curves,tab,paths[f],f)
        hits=[h for h in read_contour(ref_gray,ref_mask,parts,f) if h[0]==k]
        if len(hits)<8:raise ValueError(('Insufficient sharp reference outline',f))
        raw=np.array([h[2] for h in hits]);st=np.array([h[1] for h in hits]);shift=hits[0][6]
        anchors[f]=(poly,raw,st,shift)
    save(DEST/'print_reference/anchor.npz',pts=REF_PTS,source_sha256=rs['source_sha256'],
         **{f+'_'+key:v for f,z in anchors.items() for key,v in zip(('model','uv','station','shift'),z)})
    for name,windows in WINDOWS.items():
        s=s0[name];pts=b[name+'_pts'];m=matte(s,pts);schedule=defaultdict(set)
        for sec,fs in windows:
            i=int(np.searchsorted(pts,round(sec*12800)))
            for j in range(max(0,i-50),min(len(pts),i+53),2):schedule[j].update(fs)
            schedule[i].update(fs)
        data=defaultdict(list);records=[]
        ind=sorted(schedule)
        jobs=[(i,refs[f],b[name+'_times'][i]) for i in ind for f in sorted(schedule[i])]
        proj=ji.project_groups(n,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,world)
        lookup={(i,k):v for (i,k,_),v in zip(jobs,proj)}
        for i,(pt,img) in zip(ind,source_frames(s['source'],pts[ind]),strict=True):
            mask=foreground(s,m,i,pt)
            if mask is None:continue
            gray=signals(img)
            for f in sorted(schedule[i]):
                k=refs[f];uv,status=lookup[i,k];uv[status!=0]=np.nan
                rp,raw,station,shift=anchors[f]
                if not np.isfinite(uv).all():continue
                reg=match(ref_gray,gray,mask,rp,uv,shift,f)
                if reg is None:continue
                H,dx,quality,gap,blur=reg
                measured=homography_points(raw,H)+dx
                guide=np.c_[np.interp(station,tab[k][1],uv[:,0]),np.interp(station,tab[k][1],uv[:,1])]
                valid=(measured>4).all(1)&(measured<[1915,1075]).all(1)
                for j in np.flatnonzero(valid)[::2]:
                    x,y=np.rint(measured[j]).astype(int)
                    if mask[y-2:y+3,x-2:x+3].any():continue
                    for key,v in dict(pts=pt,idx=i,path=k,family=f,uv=measured[j],station=station[j],
                        guide=guide[j],normal=[0.,0.],shift=dx,reference_uv=raw[j],correlation=quality,
                        peak_gap=gap,blur_width=blur,material='registered_same_print').items():data[key].append(v)
                records.append([pt,int(f[2:]),*dx,quality,gap,blur])
        save(DEST/'print_reference'/(name+'.npz'),**{k:np.asarray(v) for k,v in data.items()},
             registrations=np.array(records),reference_pts=REF_PTS,reference_sha256=rs['source_sha256'],
             source_sha256=s['source_sha256'])
        print(name,len(records),len(data['pts']),flush=True)


if __name__=='__main__':run()


def left_end():
    """Compare grazing-angle ink with the same five physical prints seen face-on."""
    b=camera_state(PARENT);n=native_for(PARENT,'nayeon');ss=read(PREPARED/'setup.json')['shots']
    allpaths=selected_paths(n);groups={f:allpaths[f] for f in ('Ad8','Ad9','Ad10','Ad11','Ad12')}
    tab=ji.native_paths(n,[k for paths in groups.values() for k in paths],count=193)
    world={k:v[0] for k,v in tab.items()};candidates=defaultdict(list)
    for name,s in ss.items():
        if name=='nayeon_s03':continue
        a=arrays(DEST/'reads'/(name+'.npz'))
        if 'family' not in a:continue
        for f in groups:
            rr=np.flatnonzero(a['family']==f)
            for i in np.unique(a['idx'][rr]):
                ids=rr[a['idx'][rr]==i];points=a['uv'][ids];width,height=np.ptp(points,axis=0)
                if len(ids)<25 or width<23 or height<12 or width>850 or (points.min(0)<12).any() or (points.max(0)>[1907,1067]).any():continue
                score=min(width,470)*min(height,135)*np.median(a['contrast'][ids])
                candidates[f].append((score,name,int(i)))
    name='nayeon_s03';s=ss[name];anchor=650240
    pts=b[name+'_pts'];idx0=int(np.searchsorted(pts,anchor));inds=np.arange(idx0-54,idx0+55,2)
    m=matte(s,pts);refs={};records=[]
    def project(name,i,k):
        s=ss[name]
        return n.core.project_motion(b[name+'_rigs'][i],b[name+'_controls'],s['first'],s['dt'],1080,b[name+'_times'][i],world[k])
    for f,options in candidates.items():
        for _,prior,i in sorted(options,reverse=True):
            source=ss[prior];pt=int(b[prior+'_pts'][i]);_,img=next(source_frames(source['source'],[pt]))
            mask=foreground(source,matte(source,b[prior+'_pts']),i,pt)
            if mask is None:continue
            curves={}
            for k in groups[f]:
                pred=project(prior,i,k);curves[k]=pred['uv'],pred['status']
            poly=np.concatenate([v[0] for v in curves.values()])
            if not np.isfinite(poly).all() or (poly.min(0)<6).any() or (poly.max(0)>[1913,1073]).any():continue
            a,c=np.rint(poly.min(0)).astype(int),np.rint(poly.max(0)).astype(int)
            if (mask[a[1]:c[1]+1,a[0]:c[0]+1]!=0).mean()>.04:continue
            parts=contour_samples(curves,tab,groups[f],f);gray=signals(img);hits=read_contour(gray,mask,parts,f)
            if len(hits)<25:continue
            refs[f]=(prior,pt,gray,poly,hits[0][6]);break
    for i,(pt,img) in zip(inds,source_frames(s['source'],pts[inds]),strict=True):
        gray=signals(img);mask=foreground(s,m,i,pt)
        if mask is None:continue
        for f,(prior,ref_pt,ref,ref_poly,ref_shift) in refs.items():
            poly=np.concatenate([project(name,i,k)['uv'] for k in groups[f]])
            if not np.isfinite(poly).all():continue
            hit=match(ref,gray,mask,ref_poly,poly,ref_shift,f)
            if hit:
                H,delta,conf,gap,blur=hit
                records.append([pt,int(f[2:]),*delta,conf,gap,blur])
    save(DEST/'print_reference/left_end_transfer.npz',registrations=np.array(records),
         families=list(refs),reference_shot=[z[0] for z in refs.values()],reference_pts=[z[1] for z in refs.values()],
         reference_shift=[z[4] for z in refs.values()],source_sha256=s['source_sha256'])
    print(refs.keys(),len(records),[v for v in records if v[0]==anchor],flush=True)
