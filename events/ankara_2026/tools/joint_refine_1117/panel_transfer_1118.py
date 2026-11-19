"""Read complete coloured prints and the blue-to-white rail in moving source shots."""
from collections import defaultdict

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks

from jin_board_1118 import WINDOWS
from lane_side_1118 import BASE, WORK
from lane_end_evidence import foreground, matte
from review import camera_state, ji, native_for, source_frames
from shared import PREPARED, arrays, read, save
from side_repair import REPORTED


# The far, solid red lozenge was identified visually in eight unannotated
# exposures. These x coordinates only choose the neighbourhood, never the UV.
BADGE = np.array([[122.70,1860],[122.90,1829],[122.98,1779],[123.04,1727],
                  [123.06,1703],[123.12,1639],[123.24,1483],[123.40,1236]])


def rail(raw,mask,uv,sts):
    # Only the visible far rail in these four views is measured here.
    # Require the complete blue strip and its immediate white face below.
    out=[]
    for j in range(1,len(uv)-1):
        x,y=uv[j]
        if not np.isfinite(x+y) or not (6<x<1913 and 105<y<798):continue
        tangent=uv[j+1]-uv[j-1]
        if abs(tangent[1])>.42*abs(tangent[0]) or abs(tangent[0])>58:continue
        xx,yy=int(round(x)),int(round(y)); lo=max(4,yy-42);hi=min(1076,yy+42)
        if hi-lo<30:continue
        strip=np.mean(raw[lo:hi,xx-3:xx+4].astype(float),axis=1)
        blue=strip[:,0]-.55*strip[:,2]-.45*strip[:,1]
        lum=strip@np.array([.114,.587,.299])
        # Blue material followed by the bright, essentially neutral board face.
        # The back edge of a letter or a broad filled ad does not form this band.
        band=(blue>3.5)&(lum<np.percentile(lum,86)-6.)
        spans=np.flatnonzero(np.diff(np.r_[False,band,False]))
        options=[]
        for a,b in spans.reshape(-1,2):
            if a<3 or b+4>=len(blue) or not 2<=b-a<=26:continue
            if mask[lo+a-2:lo+b+3,xx-3:xx+4].any():continue
            strength=np.mean(lum[b+1:b+4])-np.median(lum[a:b])
            if strength<12 or min(lum[b+1:b+4])<126:continue
            color=np.mean(blue[a:b])
            if color<5 or np.mean(blue[b+1:b+4])>max(8.,color*.46):continue
            # Upper edge of the thin navy face. Fit its half-height on the
            # colour step so a broad blue band is not mistaken for its centre.
            above=np.mean(blue[a-3:a-1]);inside=np.percentile(blue[a:b],70)
            if inside-above<3.:continue
            threshold=(above+inside)/2.;k=np.arange(max(0,a-2),min(b,a+6))
            ix=k[(blue[k]<=threshold)&(blue[k+1]>threshold)]
            if not len(ix):continue
            k=ix[np.argmin(abs(ix-a))]
            edge=lo+k+(threshold-blue[k])/(blue[k+1]-blue[k])
            score=(min(strength,90.)*.35+min(color,35.))/(1+abs(edge-y)*.026)
            options.append((score,edge,strength))
        if options:
            _,edge,strength=max(options);out.append((j,x,edge,strength,edge-y))
    if not out:return []
    a=np.asarray(out);final=[]
    for j,x,y,st,delta in out:
        nn=(abs(a[:,1]-x)<77)&(abs(a[:,0]-j)<=22)
        if nn.sum()<3 or abs(delta-np.median(a[nn,4]))>2.1:continue
        final.append((sts[j],[x,y],[0.,1.],st))
    return final


def physical_plate(raw,mask,curves,table):
    # Measure the outer coloured/white transition on each side. Its identity is
    # checked by the full printed strip's width and by the ink on both faces.
    vertical=[k for k in curves if 'vertical' in k]
    if len(vertical)!=2:return []
    edges=sorted(vertical,key=lambda k:np.mean(curves[k][:,0]))
    endpoints=[curves[k] for k in edges]
    if any(not np.isfinite(e).all() for e in endpoints):return []
    x=np.array([np.mean(e[:,0]) for e in endpoints])
    ys=np.array([np.median(e[:,1]) for e in endpoints]);w=np.diff(x)[0]
    heights=np.array([np.ptp(e[:,1]) for e in endpoints])
    if w<20 or w>2300 or np.min(heights)<9 or np.max(abs(ys-500))>580:return []
    outputs=[]
    for side,key in enumerate(edges):
        edge=endpoints[side];orient=1 if side==0 else -1
        j0=np.arange(20,48,4);p=edge[j0];off=np.arange(-54,55,dtype=float)
        xx=(p[:,0,None]+off*orient).round().astype(int)
        best=None
        for dy in np.arange(-12,13,3):
            yy=np.broadcast_to((p[:,1,None]+dy).round().astype(int),xx.shape)
            legal=(xx>=3)&(xx<1917)&(yy>=3)&(yy<1077)
            xx0=xx.clip(0,1919);yy0=yy.clip(0,1079)
            c=raw[yy0,xx0].astype(float);v=255.-c.mean(2)+1.25*np.ptp(c,axis=2)
            valid=legal&(mask[yy0,xx0]==0)
            v=gaussian_filter1d(v,1.5,axis=1);grad=np.mean(np.gradient(v,axis=1),axis=0)
            for k in find_peaks(grad,height=.65)[0]:
                span=int(np.clip(w*.09,8,20))
                if k-span<0 or k+span>=len(off) or valid[:,k-span:k+span+1].mean()<.86:continue
                a=v[:,k-span:k-span+3].mean(1);b=v[:,k+span-2:k+span+1].mean(1)
                excess=b-a
                if np.median(excess)<11 or np.sum(excess>6)<5:continue
                value=min(105.,np.median(excess))/(1+abs(dy)*.036+abs(off[k])*.003)
                if best is None or value>best[0]:best=(value,dy,k,a,b,v.copy(),valid)
        if best is None:continue
        _,dy,k,a,b,v,valid=best
        offsets=[]
        for ii,j in enumerate(j0):
            th=(a[ii]+b[ii])/2.;ks=np.arange(max(0,k-14),min(len(off)-1,k+15))
            hits=ks[(v[ii,ks]<=th)&(v[ii,ks+1]>th)&valid[ii,ks]&valid[ii,ks+1]]
            if not len(hits):continue
            u=hits[np.argmin(abs(hits-k))]
            offsets.append((j,off[u]+(th-v[ii,u])/(v[ii,u+1]-v[ii,u])))
        if len(offsets)<4:continue
        median=np.median([d for _,d in offsets])
        for j,d in offsets:
            if abs(d-median)<4:
                outputs.append((key,table[key][1][j],edge[j]+[orient*d,dy],[1.,0.],best[0]))
    if len(outputs)>0 and len({v[0] for v in outputs})==2:
        # Blur shifts both outer half-height positions together. Do not admit
        # a putative pair with the width of an internal printed motif.
        ds=[np.median([h[2][0]-np.interp(h[1],table[k][1],curves[k][:,0]) for h in outputs if h[0]==k]) for k in edges]
        if abs(ds[1]-ds[0])>max(7.,.055*w):return []
    return outputs


def badge(raw,mask,uv,sts,x_seed):
    # Select the red printed Can lozenge on the distant board. Its left/right
    # extent is measured from the integrated red ink in a patch of original RGB.
    lo=np.floor(uv.min(0));hi=np.ceil(uv.max(0));yy=float((lo[1]+hi[1])/2)
    if not 18<x_seed<1901 or not 10<yy<985:return []
    x0,x1=max(0,int(x_seed-38)),min(1920,int(x_seed+40));y0,y1=int(yy-22),int(yy+23)
    c=raw[y0:y1,x0:x1].astype(float)
    if not c.size or mask[y0:y1,x0:x1].mean()>9:return []
    red=c[:,:,2]-.48*c[:,:,0]-.52*c[:,:,1]
    red-=np.percentile(red,42,axis=1,keepdims=True)
    dist=np.arange(x0,x1)-x_seed
    ypower=np.mean(np.sort(red[:,abs(dist)<19],axis=1)[:,-9:],axis=1)
    cy=np.argmax(gaussian_filter1d(ypower,2))
    if ypower[cy]<4 or cy<3 or cy>len(red)-4:return []
    yslice=red[cy-2:cy+3].mean(0);yslice=gaussian_filter1d(yslice,1.65)
    peak=np.argmax(yslice*np.exp(-.5*(dist/19)**2));height=yslice[peak]
    if height<4:return []
    low=np.median(yslice[abs(dist)>26]);threshold=(height+low)*.5
    xx=np.arange(len(yslice)-1)
    left=xx[(xx<peak)&(yslice[:-1]<=threshold)&(yslice[1:]>threshold)]
    right=xx[(xx>=peak)&(yslice[:-1]>=threshold)&(yslice[1:]<threshold)]
    if not len(left) or not len(right):return []
    edges=[left[-1],right[0]]
    positions=[x0+k+(threshold-yslice[k])/(yslice[k+1]-yslice[k]) for k in edges]
    if not .5*np.ptp(uv[:,0])<positions[1]-positions[0]<1.37*np.ptp(uv[:,0]):return []
    if abs(np.mean(positions)-x_seed)>16:return []
    # Symmetric motion blur broadens both flanks; fixing the group translation
    # from both measured extrema protects against using the leading streak alone.
    return [(sts[i],[v,y0+cy],[1.,0.],height) for i,v in zip([np.argmin(uv[:,0]),np.argmax(uv[:,0])],positions)]


def measure(name):
    cv2.setNumThreads(1)
    n=native_for(BASE,'jin');b=camera_state(BASE);s=read(PREPARED/'setup.json')['shots'][name]
    pts=b[name+'_pts'];t=pts/12800.;lo,hi=WINDOWS[name]
    ins=np.unique(np.r_[np.arange(0,len(t),9),np.flatnonzero((t>=lo)&(t<=hi))[::2],np.searchsorted(pts,np.rint(np.array(REPORTED[name])*12800).astype(int)),len(t)-1])
    if name=='jin_s14':ins=np.union1d(ins,np.flatnonzero((t>=122.68)&(t<=123.43)))
    fams=['Ad1','Ad2','Ad3','Ad4','Ad14'];groups=defaultdict(list)
    for k,v in n.paths.items():
        if v['label'] in fams and 'vertical' in k:groups[v['label']].append(k)
    rail_key='board/continuous/top';mark=next(k for k,v in n.paths.items() if v['label']=='Ad22' and 'red_field' in k)
    table=ji.native_paths(n,[k for v in groups.values() for k in v],count=65)
    table.update(ji.native_paths(n,[rail_key],count=2049));table.update(ji.native_paths(n,[mark],count=257))
    world={k:v[0] for k,v in table.items()};data=defaultdict(list);m=matte(s,pts);frames=source_frames(s['source'],pts[ins])
    for lo in range(0,len(ins),12):
        ii=ins[lo:lo+12];jobs=[(i,k,b[name+'_times'][i]) for i in ii for k in table]
        pred=ji.project_groups(n,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,world)
        by={(i,k):np.where((status==0)[:,None],uv,np.nan) for (i,k,_),(uv,status) in zip(jobs,pred)}
        for i in ii:
            pt,raw=next(frames);mask=foreground(s,m,i,pt,exclude_uncertain_region=False)
            if mask is None:continue
            mask=cv2.dilate(mask,np.ones((9,9),np.uint8));mask[:302,:500]=255;mask[928:,1650:]=255
            def add(f,material,k,st,xy,norm,c):
                for key,v in dict(idx=i,pts=pt,path=k,station=st,uv=xy,normal=norm,contrast=c,material=material,family=f).items():data[key].append(v)
            for family,ks in groups.items():
                for row in physical_plate(raw,mask,{k:by[i,k] for k in ks},table):
                    add(family,'whole_panel_side',*row)
            for row in rail(raw,mask,by[i,rail_key],table[rail_key][1]):add('board.top','blue_rail_pair',rail_key,*row)
            if name=='jin_s14' and BADGE[0,0]<=t[i]<=BADGE[-1,0] and np.isfinite(by[i,mark]).all():
                for row in badge(raw,mask,by[i,mark],table[mark][1],np.interp(t[i],BADGE[:,0],BADGE[:,1])):
                    add('Ad22','measured_badge_tip',mark,*row)
    save(WORK/'panel_source'/(name+'.npz'),**{k:np.asarray(v) for k,v in data.items()},source_sha256=s['source_sha256'])
    print(name,len(data['idx']),flush=True)


def printed_icons(name):
    from feedback_print import PRINTS, REF_NAME, REF_PTS, homography_points, match
    from feedback_source import selected_paths, signals
    from side_repair import PREVIOUS
    cv2.setNumThreads(1)
    anchor=arrays(PREVIOUS/'print_reference/anchor.npz');shots=read(PREPARED/'setup.json')['shots'];s=shots[name]
    n=native_for(BASE,'jin');b=camera_state(BASE);pts=b[name+'_pts'];t=pts/12800.;lo,hi=WINDOWS[name]
    ins=np.unique(np.r_[np.arange(0,len(t),11),np.flatnonzero((t>=lo)&(t<=hi))[::2],
        np.searchsorted(pts,np.rint(np.asarray(REPORTED[name])*12800).astype(int)),len(t)-1])
    _,raw=next(source_frames(shots[REF_NAME]['source'],[REF_PTS]));ref=signals(raw)
    signs=selected_paths(n);keys={f:next(k for k in signs[f] if k.endswith(pat)) for f,pat in PRINTS.items()}
    table=ji.native_paths(n,list(keys.values()),count=193);world={k:v[0] for k,v in table.items()}
    mats=matte(s,pts);frames=source_frames(s['source'],pts[ins]);data=defaultdict(list)
    for start in range(0,len(ins),12):
        ii=ins[start:start+12];jobs=[(i,k,b[name+'_times'][i]) for i in ii for k in table]
        pred=ji.project_groups(n,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,world)
        by={(i,k):np.where((status==0)[:,None],uv,np.nan) for (i,k,_),(uv,status) in zip(jobs,pred)}
        for i in ii:
            pt,raw=next(frames);mask=foreground(s,mats,i,pt,exclude_uncertain_region=False)
            if mask is None:continue
            mask=cv2.dilate(mask,np.ones((13,13),np.uint8));mask[:302,:500]=255;mask[928:,1650:]=255
            gray=signals(raw)
            for f,k in keys.items():
                u=by[i,k]
                if not np.isfinite(u).all() or not (u.min(0)<[1920,1080]).all() or not (u.max(0)>[0,0]).all():continue
                reg=match(ref,gray,mask,anchor[f+'_model'],u,anchor[f+'_shift'],f)
                if reg is None:continue
                H,dx,qual,gap,blur=reg
                if qual<.74 or gap<.031:continue
                obs=homography_points(anchor[f+'_uv'],H)+dx;sts=anchor[f+'_station']
                for j in range(0,len(obs),4):
                    x,y=np.rint(obs[j]).astype(int)
                    if not (4<x<1914 and 4<y<1074) or mask[y,x]:continue
                    for key,val in dict(pts=pt,idx=i,path=k,station=sts[j],uv=obs[j],family=f,
                        material='same_source_icon',normal=[0.,0.],contrast=qual,correlation=qual,peak_gap=gap,blur=blur).items():data[key].append(val)
    if data:
        # The image match supplies one patch translation. Its warped reference
        # contour does not provide a fresh observation of each ink pixel.
        save(WORK/'icon_source'/(name+'.npz'),**{k:np.asarray(v) for k,v in data.items()},
             fold=np.full(len(data['pts']),9,dtype='i1'),
             measurement_definition='reference outline transported by a source patch registration',
             source_sha256=s['source_sha256'])
    print(name,len(data['pts']),flush=True)


def horizontal_rail(name):
    from nayeon_side_1118 import rail_sections
    from feedback_source import contour_samples
    key='board/continuous/top'
    n=native_for(BASE,'jin');b=camera_state(BASE);s=read(PREPARED/'setup.json')['shots'][name]
    pts=b[name+'_pts'];m=matte(s,pts);lo,hi=WINDOWS[name]
    sel=np.unique(np.r_[np.arange(0,len(pts),8),np.flatnonzero((pts/12800.>=lo)&(pts/12800.<=hi))[::2],
        np.searchsorted(pts,np.rint(np.asarray(REPORTED[name])*12800).astype(int)),len(pts)-1])
    from side_repair import native_table
    tab=native_table(n,[key],8193);world={key:tab[key][0]}
    fr=source_frames(s['source'],pts[sel]);data=defaultdict(list)
    for j in range(0,len(sel),10):
        ii=sel[j:j+10];jobs=[(i,key,b[name+'_times'][i]) for i in ii]
        vv=ji.project_groups(n,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,world)
        for i,v in zip(ii,vv):
            pt,raw=next(fr);mask=foreground(s,m,i,pt,exclude_uncertain_region=False)
            if mask is None:continue
            mask=cv2.dilate(mask,np.ones((11,11),np.uint8));mask[:104,:550]=255;mask[102:258,:395]=255;mask[941:,1680:]=255
            parts=contour_samples({key:v},tab,{key:0},'board')
            hits,_=rail_sections(raw,mask,parts,False)
            if name=='jin_s15':
                visible=np.where((v[1]==0)[:,None],v[0],np.nan)
                hits=[('board.top','blue_rail_pair',key,st,pos,pos,no,strength)
                      for st,pos,no,strength in rail(raw,mask,visible,tab[key][1])]
            for fam,kind,path,st,pos,p,no,contrast in hits:
                if abs(no[1])<.94:continue
                for k,value in dict(idx=i,pts=pt,path=path,family=fam,station=st,uv=pos,normal=no,contrast=contrast,
                        material='front_band_top').items():data[k].append(value)
    save(WORK/'rail_source'/(name+'.npz'),**{k:np.asarray(v) for k,v in data.items()},source_sha256=s['source_sha256'])
