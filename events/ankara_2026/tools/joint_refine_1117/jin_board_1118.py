"""Measure print borders, the kickplate, top rail and hockey paint at native PTS."""
from collections import defaultdict

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from lane_side_1118 import BASE, WORK, project
from feedback_source import contour_samples, read_rail, read_stripe, sample_image, selected_paths
from lane_end_evidence import foreground, matte
from review import camera_state, ji, native_for, source_frames
from shared import PREPARED, read, save


CASES={'jin_s08':[69.36,70.84,71.58], 'jin_s14':[122.98,123.06],
       'jin_s15':[126.30], 'jin_s22':[155.06]}
WINDOWS={'jin_s08':(68.25,72.32),'jin_s14':(121.9,123.98),
         'jin_s15':(124.10,127.60),'jin_s22':(153.55,155.08)}


def paths(native):
    result=defaultdict(dict)
    for k,v in native.paths.items():
        name=v['label']
        if name.startswith('Ad') and any(s in k for s in ('.left_vertical','.right_vertical','.top_edge','.bottom_edge')):
            result[name][k]=0
    return dict(result)


def panel_measurements(raw,mask,curves,table):
    """Fit separate transitions of the physical plate, including a blurred midpoint."""
    outline=np.vstack([v[0] for v in curves.values()])
    if not np.isfinite(outline).all() or np.max(abs(outline))>17000:return []
    if (outline.max(0)<[3,3]).any() or (outline.min(0)>[1916,1076]).any():return []
    centre=(outline.min(0)+outline.max(0))/2
    width,height=np.ptp(outline,axis=0)
    if width<14 or height<7:return []
    out=[]
    for key,(uv,status) in curves.items():
        vert='vertical' in key
        idx=np.linspace(9,len(uv)-10,9 if vert else 18).astype(int)
        pts=uv[idx];t=uv[idx+1]-uv[idx-1];no=np.c_[-t[:,1],t[:,0]]
        no/=np.maximum(1.e-9,np.linalg.norm(no,axis=1))[:,None]
        no*=np.sign(np.sum((centre-pts)*no,axis=1))[:,None]
        off=np.arange(-79,80) if vert else np.arange(-32,33)
        grid=pts[:,None]+no[:,None]*off[None,:,None]
        cells=sample_image(raw,grid).astype('f4')
        vis=(grid>1).all(2)&(grid<[1918,1078]).all(2)&(sample_image(mask,grid,cv2.INTER_NEAREST)==0)
        lum=cells@np.array([.114,.587,.299],dtype='f4')
        chroma=np.ptp(cells,axis=2);ink=255.-lum+.75*chroma
        smooth=gaussian_filter1d(ink,1.0,axis=1);deriv=np.gradient(smooth,axis=1)
        candidates=[]
        for j,g in enumerate(deriv):
            if status[idx[j]]!=0:continue
            opts=[]
            for h in find_peaks(g,height=1.0,prominence=.85)[0]:
                span=int(np.clip((width if vert else height)*.18,3,15 if vert else 8))
                if h-span-1<0 or h+span+2>=len(off) or not vis[j,h-span-1:h+span+2].all():continue
                if not vert and abs(off[h])>min(28,height*.85):continue
                left=ink[j,h-span-1:h-span+2].mean();right=ink[j,h+span-1:h+span+2].mean()
                if right-left<14 or chroma[j,h-span-1:h-span+2].mean()>32:continue
                if lum[j,h-span-1:h-span+2].mean()<98:continue
                score=min(float(right-left),85)*min(1.,float(g[h])/5.)
                # The half-height of a finite blurred ink/white step tracks the
                # exposure centre even when its maximum gradient is asymmetric.
                a=ink[j,h-span:h+span+1];target=(right+left)/2
                crosses=np.flatnonzero((a[:-1]<=target)&(a[1:]>target))
                if not len(crosses):continue
                k=crosses[np.argmin(abs(crosses-span))];delta=off[h-span+k]+(target-a[k])/(a[k+1]-a[k])
                opts.append((float(delta),score))
            if opts:candidates.append((j,opts))
        if len(candidates)<3:continue
        offsets=np.arange(off[1],off[-1]+1.)
        votes=sum(np.max([w*np.exp(-.5*((offsets-d)/2.5)**2) for d,w in opts],axis=0) for _,opts in candidates)
        mode=offsets[np.argmax(votes)];part=[]
        for j,opts in candidates:
            good=[(w,d) for d,w in opts if abs(d-mode)<4.]
            if not good:continue
            w,d=max(good);part.append((key,table[key][1][idx[j]],pts[j]+d*no[j],no[j],w))
        if len(part)>=3:out.extend(part)
    return out


def measure(name):
    cv2.setNumThreads(1);b=camera_state(BASE);n=native_for(BASE,'jin')
    s=read(PREPARED/'setup.json')['shots'][name]
    pts=b[name+'_pts'];t=pts/12800.;low,high=WINDOWS[name]
    case=CASES[name];exact=np.array([round(v*12800) for v in case])
    ind=np.unique(np.r_[np.arange(0,len(pts),11),np.flatnonzero((t>=low)&(t<=high))[::2],np.searchsorted(pts,exact),len(pts)-1])
    if name=='jin_s14':
        ind=np.union1d(ind,np.flatnonzero((t>122.65)&(t<123.63)))
    groups=paths(n);extra=selected_paths(n);ice={'jin_s08':['C0'],'jin_s14':['F4','goal_crease_left'],'jin_s15':['B1'],'jin_s22':['B1']}[name]
    ink={k for v in groups.values() for k in v}
    table=ji.native_paths(n,ink,count=65)
    for f in ['board',*ice]:
        table.update(ji.native_paths(n,extra[f],count=2049 if f=='board' else 481))
    world={k:v[0] for k,v in table.items()}
    # Batch GPU curve projection. The source pixels themselves come from the
    # original image at the same integer PTS.
    mats=matte(s,pts);data=defaultdict(list)
    fr=source_frames(s['source'],pts[ind])
    for lo in range(0,len(ind),10):
        indices=ind[lo:lo+10];jobs=[(i,k,b[name+'_times'][i]) for i in indices for k in table]
        uv=ji.project_groups(n,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,world)
        by={(i,k):v for (i,k,_),v in zip(jobs,uv)}
        for i in indices:
            pt,img=next(fr);person=foreground(s,mats,i,pt)
            if person is None:continue
            mask=cv2.dilate(person,np.ones((9,9),np.uint8));mask[:303,:500]=255;mask[930:,1650:]=255
            def append(f,kind,h):
                key,st,pos,no,contrast=h
                for k,v in dict(pts=int(pt),idx=i,path=key,family=f,material=kind,uv=pos,
                        station=float(st),normal=no,contrast=contrast).items():data[k].append(v)
            for fam,keys in groups.items():
                curves={k:by[i,k] for k in keys}
                if len(curves)!=4:continue
                for hit in panel_measurements(img,mask,curves,table):
                    append(fam,'plate_transition',hit)
            for fam in ['board',*ice]:
                curves={k:by[i,k] for k in extra[fam]}
                parts=contour_samples(curves,table,extra[fam],fam)
                if fam=='board':
                    hits=read_rail(img,mask,parts)
                elif fam in ('C0','B1'):
                    hits=read_stripe(img,mask,parts,fam)
                else:continue
                for key,st,pos,guide,no,strength,_ in hits:
                    family=('board.top' if key.endswith('top') else 'board.bottom') if fam=='board' else fam
                    append(family,'review_rail_edge' if fam=='board' else 'review_stripe_pair',(key,st,pos,no,strength))
    save(WORK/'jin_source'/(name+'.npz'),**{k:np.asarray(v) for k,v in data.items()},
        source_sha256=s['source_sha256'])
    print(name,len(data['pts']),flush=True)
