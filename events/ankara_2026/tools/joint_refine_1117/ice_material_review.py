"""Resolve weak ice observations against paired pigment boundaries on the source."""
import shutil
from collections import defaultdict

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from lane_end_evidence import BASE, WORK, foreground, ENDS
from review import ji, native_for, camera_state, source_frames
from shared import OUT, PREPARED, arrays, prepared_packet_path, read, save, write
from source_paint import matte


SHOTS={'nayeon_s04':('F2','F4','R1','R4'),'nayeon_s08':('R1','R2','R3','R4'),
       'okada_s03':('R4',),'chen_s07':('R2','R3'),'jin_s18':('F2','F4')}


def crossing(poly,station,x,upper):
    start=poly[:-1];end=poly[1:];dx=end[:,0]-start[:,0]
    hit=(np.minimum(start[:,0],end[:,0])<=x)&(np.maximum(start[:,0],end[:,0])>x)&(abs(dx)>1e-6)
    ix=np.flatnonzero(hit)
    if len(ix)!=2:return None
    w=(x-start[ix,0])/dx[ix];y=start[ix,1]+w*(end[ix,1]-start[ix,1])
    j=np.argmin(y) if upper else np.argmax(y);k=ix[j]
    return y[j],station[k]+w[j]*(station[k+1]-station[k])


def read_annulus(raw,mask,outer,middle):
    a,st=outer;b,sm=middle
    if not np.isfinite(a).all() or not np.isfinite(b).all():return []
    xmin=max(8.,np.min(b[:,0])+10);xmax=min(1911.,np.max(b[:,0])-10)
    if xmax-xmin<85 or np.ptp(a[:,1])<16:return []
    columns=np.arange(np.ceil(xmin/22)*22,xmax,22,dtype=int)
    hits=[]
    for upper in (True,False):
        track=[]
        for x in columns:
            p=crossing(a,st,x,upper);q=crossing(b,sm,x,upper)
            if p is None or q is None:continue
            y0,s0=p;y1,s1=q;sign=1 if upper else -1;width=(y1-y0)*sign
            if not 5.<width<135. or not 0.<y0<1080.:continue
            lo=max(0,int(min(y0,y1)-min(52,12+.85*width)-8))
            hi=min(1080,int(max(y0,y1)+min(52,12+.85*width)+9))
            if hi-lo<15:continue
            y=np.arange(lo,hi)[::sign];rgb=raw[y,x-2:x+3].astype(float).mean(1)
            c=cv2.GaussianBlur(rgb.reshape(-1,1,3),(1,7),1.45).reshape(-1,3)
            brightness=c@np.array([.114,.587,.299]);blue=c[:,0]-.5*(c[:,1]+c[:,2]);red=c[:,2]-.5*(c[:,0]+c[:,1])
            darkness=-brightness;g=np.gradient(darkness);peaks=find_peaks(g,height=.28)[0];ends=find_peaks(-g,height=.28)[0]
            blocked=mask[y,x-2:x+3].any(1);best=None
            for l in peaks[(abs(y[peaks]-y0)<min(49,12+.80*width))&(peaks>=6)]:
                for r in ends[(ends-l>max(3,.53*width))&(ends-l<1.57*width+3)&(ends<len(y)-7)]:
                    if blocked[l-5:r+6].any() or abs(y[r]-y1)>min(48,11+.85*width):continue
                    pad=max(1,(r-l)//4);mid=slice(l+pad,r-pad+1)
                    lum=np.median(brightness[mid]);side0=np.median(brightness[l-6:l-2]);side1=np.median(brightness[r+3:r+7])
                    contrast=min(side0,side1)-lum
                    if contrast<5. or min(side0,side1)<122. or abs(side0-side1)>max(12,.85*contrast):continue
                    chroma=max(np.median(h[mid])-max(np.median(h[l-6:l-2]),np.median(h[r+3:r+7])) for h in (blue,red))
                    if chroma<1.0 or max(g[l],-g[r])>5*min(g[l],-g[r]):continue
                    score=contrast*min(6.,chroma)/(1+.24*abs((r-l)/width-1.))
                    if best is None or score>best[0]:
                        def offset(signal,i):
                            den=2*(2*signal[i]-signal[i-1]-signal[i+1])
                            return np.clip((signal[i+1]-signal[i-1])/max(den,1e-6),-.7,.7)
                        yl=y[l]+sign*offset(g,l);yr=y[r]+sign*offset(-g,r)
                        best=(score,(x,yl,yr,s0,s1,yl-y0,yr-y1,contrast,chroma))
            if best:track.append(best[1])
        if len(track)<5:continue
        h=np.asarray(track)
        # A raw material band must persist on a connected part of the arc.
        # Reject crossings and discontinuous edges even when one column is sharp.
        for segment in np.split(h,np.flatnonzero(np.diff(h[:,0])>70)+1):
            if len(segment)<5 or np.ptp(segment[:,0])<85:continue
            x=segment[:,0]-segment[:,0].mean();x/=max(1,np.ptp(x))
            design=np.c_[np.ones(len(x)),x,x*x]
            keep=np.ones(len(x),bool)
            for _ in range(2):
                if keep.sum()<5:break
                delta=segment[:,5:7]-design@np.linalg.lstsq(design[keep],segment[keep,5:7],rcond=None)[0]
                keep=np.max(abs(delta),axis=1)<3.
            if keep.sum()>=5 and np.ptp(segment[keep,0])>=85:
                hits.extend(segment[keep].tolist())
    return hits


def old_support(raw,mask,xy,norm,tone):
    off=np.arange(-80.,81.);cut=(xy[:,None,:]+off[None,:,None]*norm[:,None,:]).astype('f4')
    px=cv2.remap(raw,cut,None,cv2.INTER_LINEAR).astype(float)
    hidden=cv2.remap(mask,cut,None,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=255)!=0
    lum=px@np.array([.114,.587,.299]);chroma=px[:,:,2]-.5*(px[:,:,0]+px[:,:,1])
    response=chroma if tone=='red' else -lum
    valid=np.zeros(len(xy),bool)
    for i,v in enumerate(gaussian_filter1d(response,1.5,axis=1)):
        g=np.gradient(v);ls=find_peaks(g,height=.12)[0];rs=find_peaks(-g,height=.12)[0]
        for l in ls[(ls>6)&(ls<80)]:
            for r in rs[(rs>80)&(rs<len(off)-7)&(rs-l>=3)]:
                w=r-l;mid=slice(l+w//4,r-w//4+1)
                if abs(.5*(r+l)-80)>max(.18*w,2.5) or hidden[i,l-5:r+6].any():continue
                strength=min(np.median(v[mid])-np.median(v[l-6:l-1]),np.median(v[mid])-np.median(v[r+2:r+7]))
                if strength<(1.7 if tone=='red' else 2.2):continue
                if tone=='dark' and np.ptp(px[i,mid].mean(0))>24:continue
                valid[i]=True;break
            if valid[i]:break
    return valid


def run():
    cv2.setNumThreads(1);folder=WORK/'verified';folder.mkdir(exist_ok=True)
    setup=read(PREPARED/'setup.json');b=camera_state(BASE);n=native_for(BASE,'jin')
    audit={};shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz')
    for name,families in SHOTS.items():
        s=setup['shots'][name];src=prepared_packet_path(name);a=arrays(src)
        with np.load(OUT/BASE/'full_qa/curve_errors.npz') as err:
            e=err[name+'_error'];st0=err[name+'_station']
        base_count=len(e)
        weak=np.flatnonzero((a['fold'][:base_count]<5)&~a['station_fixed'][:base_count]&(e>10.)&
            np.isin(a['family'][:base_count],['C0','C','lane3','lane4','F2','F4','R1','R4'])&
            ~np.isin(a['path'][:base_count],ENDS))
        paths={f:{side:next(k for k,v in n.paths.items() if v['label']=='ice.'+f+':'+side)
            for side in ('outer','middle')} for f in families}
        keys=sorted(set(a['path'][weak])|{p for z in paths.values() for p in z.values()})
        tab=ji.native_paths(n,keys,count=193);world={k:x[0] for k,x in tab.items()}
        pts=b[name+'_pts'];m=matte(s,pts);rows=defaultdict(list);decisions=[]
        foot_rows=np.flatnonzero((a['path']=='board/continuous/bottom')&(a['fold']<5))
        foot_idx=a['idx'][foot_rows]
        for start in range(0,len(pts),45):
            ix=np.arange(start,min(len(pts),start+45))
            jobs=[(i,k,b[name+'_times'][i]) for i in ix for k in keys]
            out=ji.project_groups(n,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,world)
            cache={}
            for (i,k,_),(uv,status) in zip(jobs,out):
                uv[status!=0]=np.nan;cache[i,k]=uv
            if start==0:frames=iter(source_frames(s['source'],pts))
            for i in ix:
                pt,raw=next(frames);assert pt==pts[i]
                mask=foreground(s,m,i,pt)
                if mask is None:continue
                mask=cv2.dilate(mask,np.ones((7,7),np.uint8));mask[:158,:590]=255;mask[:292,:451]=255;mask[928:,1618:]=255
                # Direct yellow kickplate evidence prevents advertising, kickplate
                # and spectators from being fitted as material on the ice.
                foot=a['uv'][foot_rows[foot_idx==i]]
                foot=foot[np.argsort(foot[:,0])] if len(foot) else foot
                for p0,p1 in zip(foot[:-1],foot[1:]):
                    x0,y0=p0;x1,y1=p1
                    if not 2<x1-x0<180 or abs(y1-y0)>.35*(x1-x0)+4:continue
                    x=int((x0+x1)/2);y=int((y0+y1)/2)
                    if not (0<x<1920 and 10<y<1070):continue
                    color=np.median(raw[y-8:y-2,x].astype(float),axis=0)
                    if min(color[1:])-color[0]<10:continue
                    polygon=np.array([[x0,0],[x1,0],[x1,y1-1],[x0,y0-1]],np.int32)
                    cv2.fillConvexPoly(mask,polygon,255)
                for f,lookup in paths.items():
                    ko,km=lookup['outer'],lookup['middle']
                    for x,yo,ym,so,sm,_,_,contrast,chroma in read_annulus(raw,mask,
                          (cache[i,ko],tab[ko][1]),(cache[i,km],tab[km][1])):
                        for key,y,st in [(ko,yo,so),(km,ym,sm)]:
                            for k,v in dict(pts=pt,idx=i,uv=[x,y],path=key,station=st,family=f,
                                paired_edges=[[x,yo],[x,ym]],contrast=contrast,chroma=chroma).items():rows[k].append(v)
                near=weak[a['idx'][weak]==i]
                for path in np.unique(a['path'][near]):
                    rr=near[a['path'][near]==path];uv=a['uv'][rr];poly=cache[i,path]
                    st=tab[path][1];k=np.searchsorted(st,st0[rr]).clip(1,len(st)-1)
                    d=poly[k]-poly[k-1];norm=np.c_[-d[:,1],d[:,0]]/np.maximum(np.linalg.norm(d,axis=1)[:,None],1e-12)
                    xy=np.rint(uv).astype(int);inside=(xy>=[0,0]).all(1)&(xy<[1920,1080]).all(1)
                    safe=np.zeros(len(rr),bool);safe[inside]=mask[xy[inside,1],xy[inside,0]]==0
                    family=a['family'][rr[0]]
                    if family in ('C','C0','lane3','lane4'):
                        safe&=old_support(raw,mask,uv,norm,'red' if family in ('C','C0') else 'dark')
                    for row,ok in zip(rr,safe):
                        if not ok:decisions.append(int(row))
        rawpath=folder/(name+'_annuli.npz')
        save(rawpath,**{k:np.array(v) for k,v in rows.items()},source_sha256=s['source_sha256'])
        # Neighboring exposures must independently contain the same material arc.
        added=np.arange(len(rows['pts']),dtype=int)
        if len(added):
            path=np.array(rows['path']);idx=np.array(rows['idx']);keep=np.zeros(len(idx),bool)
            for k in np.unique(path):
                ids=np.flatnonzero(path==k);frames=np.unique(idx[ids]);ok=frames[np.isin(frames-1,frames)|np.isin(frames+1,frames)]
                keep[ids]=np.isin(idx[ids],ok)
            added=added[keep]
        a['fold'][decisions]=9
        for k in ('admission_reason','restored_admission_reason'):
            if k in a:a[k][decisions]='ambiguous_source_material_or_occluded'
        code=max(map(int,s['source_packets']))+1
        if len(added):
            ext={k:np.asarray(rows[k])[added] for k in ('pts','idx','uv','path','station','family')}
            ext.update(fold=0,source_fold=0,source_packet=code,source_row=added,origin_row=-10000000-added,
                station_fixed=False,baseline_display=False,material='real_paired_annulus_edges',
                admission_reason='paired_pigment_edges',restored_admission_reason='paired_pigment_edges')
            for k,v in a.items():
                extra=np.broadcast_to(ext[k],(len(added),)+v.shape[1:])
                a[k]=np.concatenate((v,extra.astype(v.dtype if v.dtype.kind!='U' else str)))
            s['source_packets'][str(code)]=dict(path=str(rawpath),source_row='row of two original annulus boundaries')
        save(folder/(name+'.npz'),**a);s['observations']=str(folder/(name+'.npz'))
        audit[name]=dict(tested_old=len(weak),excluded_old=decisions,old_packet=str(src),added=len(added),
            candidates=len(rows['pts']),old_reasons='source foreground / yellow-board separation, or no identifiable paired centre at the original UV',
            new_families=families,frames=len(pts))
        write(folder/'source_audit.json',audit);print(name,len(added),len(decisions),flush=True)
    setup['parent']=str(PREPARED);write(folder/'setup.json',setup)


def contrast_at_observation(image,point,tangent,kind):
    """Test native contrast through the recorded UV and nearby points on the same stroke."""
    normal=np.c_[-tangent[:,1],tangent[:,0]]
    offsets=np.arange(-32,33,dtype=float)
    along=np.array([-7.,-3.,0.,3.,7.])
    grid=(point[:,None,None,:]+along[None,:,None,None]*tangent[:,None,None,:]+
          offsets[None,None,:,None]*normal[:,None,None,:]).astype('f4')
    rgb=cv2.remap(image,grid.reshape(-1,len(offsets),2),None,cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REPLICATE).reshape(*grid.shape[:-1],3).astype(float)
    k=len(offsets)//2;score=np.zeros((len(point),len(along)))
    if kind=='arc':
        for step in (2,4,6):
            for shift in (-2,0,2):
                difference=rgb[:,:,k+shift+step]-rgb[:,:,k+shift-step]
                score=np.maximum(score,np.max(abs(difference),axis=-1))
    else:
        response=rgb[:,:,:,2]-.5*(rgb[:,:,:,0]+rgb[:,:,:,1]) if kind=='red' else -(rgb@np.array([.114,.587,.299]))
        for step in (2,3,5,8,12,18,25,30):
            for shift in (-1,0,1):
                mid=response[:,:,k+shift]
                score=np.maximum(score,np.minimum(mid-response[:,:,k+shift-step],mid-response[:,:,k+shift+step]))
    return np.median(score,axis=1),score[:,len(along)//2]


def audit_tail(programme):
    from lane_review import native as native_cached
    from source_bundle import relative
    folder=WORK/'material_tail';setup=read(PREPARED/'setup.json')['shots']
    n=native_cached(BASE);b=camera_state(BASE);cv2.setNumThreads(1)
    for name,s in setup.items():
        if s['program']!=programme:continue
        target=folder/(name+'.npz')
        if target.exists():continue
        a=arrays(prepared_packet_path(name));qa=arrays(WORK/'evaluation/base_verified'/(name+'.npz'))
        rings=np.isin(a['family'],[f'{prefix}{i}' for prefix in 'FR' for i in range(1,5)])
        stripes=np.isin(a['family'],['C','C0','lane1','lane2','lane3','lane4'])&~np.isin(a['path'],ENDS)
        original=(a['origin_row']>=0)&~a['station_fixed']&(a['fold']<5)
        inspect=original&(rings|stripes)&(qa['error']>10.)&np.isfinite(qa['station'])
        ids=np.flatnonzero(inspect);pts=b[name+'_pts']
        keep=np.ones(len(ids),bool);reason=np.full(len(ids),'supported_or_unresolved',dtype='U32')
        strength=np.full((len(ids),2),np.nan)
        if len(ids):
            t=ji.native_paths(n,np.unique(a['path'][ids]),count=257);m=matte(s,pts)
            order=np.argsort(a['pts'][ids],kind='stable');dates=a['pts'][ids][order]
            groups=np.split(order,np.flatnonzero(np.diff(dates))+1)
            for loc,(pt,img) in zip(groups,source_frames(s['source'],np.unique(dates)),strict=True):
                rr=ids[loc];frame=a['idx'][rr[0]];assert (a['pts'][rr]==pt).all()
                mask=foreground(s,m,frame,pt)
                for path in np.unique(a['path'][rr]):
                    loc2=loc[a['path'][rr]==path];rows=ids[loc2];world,station=t[path]
                    p=n.core.project_motion(b[name+'_rigs'][frame],b[name+'_controls'],s['first'],s['dt'],
                        1080,np.repeat(b[name+'_times'][frame],len(world)),world)
                    j=np.searchsorted(station,qa['station'][rows]).clip(1,len(station)-1)
                    tangent=p['uv'][j]-p['uv'][j-1];norm=np.linalg.norm(tangent,axis=1)
                    good=(p['status'][j]==0)&(p['status'][j-1]==0)&(norm>.001)&np.isfinite(norm)
                    loc2=loc2[good];rows=rows[good]
                    if not len(rows):continue
                    uv=a['uv'][rows]
                    family=a['family'][rows[0]]
                    kind='arc' if family[0] in 'FR' else 'red' if family in ('C','C0') else 'dark'
                    broad,centre=contrast_at_observation(img,uv,tangent[good]/norm[good,None],kind)
                    strength[loc2]=np.c_[broad,centre]
                    weak=(broad<2.7)&(centre<3.4) if kind=='arc' else (broad<1.6)&(centre<2.)
                    keep[loc2[weak]]=False;reason[loc2[weak]]='no_native_material_contrast'
                    xy=np.rint(uv).astype(int);xy[:,0]=xy[:,0].clip(0,1919);xy[:,1]=xy[:,1].clip(0,1079)
                    if mask is not None:
                        hidden=mask[xy[:,1],xy[:,0]]!=0
                        keep[loc2[hidden]]=False;reason[loc2[hidden]]='source_foreground'
        save(target,row=ids,keep=keep,reason=reason,contrast=strength,pts=a['pts'][ids],
             source_sha256=s['source_sha256'],old_material=a['material'][ids])
        print(name,len(ids),int((~keep).sum()),flush=True)


def confirmed():
    folder=WORK/'confirmed';setup=read(PREPARED/'setup.json');audit={}
    folder.mkdir(exist_ok=True);shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz')
    for name,s in setup['shots'].items():
        old=prepared_packet_path(name);d=arrays(WORK/'material_tail'/(name+'.npz'))
        visual_rows=[]
        if name=='nayeon_s04':
            original=arrays(old);tested=d['row']
            times=np.rint(12800*np.array([58.10,58.14,58.22,58.26,58.30,58.46])).astype(int)
            ix=np.flatnonzero((original['family'][tested]=='F4')&np.isin(original['pts'][tested],times))
            d['keep'][ix]=False;d['reason'][ix]='unidentified_arc_board_or_ice'
            visual_rows=tested[ix].tolist()
        rows=d['row'][~d['keep']];s['observations']=str(old)
        if len(rows):
            a=arrays(old);assert (a['fold'][rows]<5).all() and not a['station_fixed'][rows].any()
            a['fold'][rows]=9
            if 'restored_admission_reason' in a:
                col=a['restored_admission_reason'].astype('U64');col[rows]=d['reason'][~d['keep']]
                a['restored_admission_reason']=col
            save(folder/(name+'.npz'),**a);s['observations']=str(folder/(name+'.npz'))
        audit[name]=dict(examined=len(d['row']),excluded=len(rows),parent=str(old),
            decisions=str(WORK/'material_tail'/(name+'.npz')),visually_reviewed_F4_rows=visual_rows,
            visual_evidence=str(WORK/'cards/F4_neighbours_0.jpg') if visual_rows else None)
    setup['parent']=str(PREPARED);write(folder/'source_audit.json',audit);write(folder/'setup.json',setup)


if __name__=='__main__':
    from argparse import ArgumentParser
    ap=ArgumentParser(description=__doc__);ap.add_argument('--audit-tail');ap.add_argument('--confirmed',action='store_true')
    opts=ap.parse_args()
    if opts.audit_tail:audit_tail(opts.audit_tail)
    elif opts.confirmed:confirmed()
    else:run()
