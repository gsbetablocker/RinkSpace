"""Read the November visual-review features on their original, unannotated exposures."""
from collections import defaultdict
from pathlib import Path
import shutil

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from shared import OUT, PREPARED, arrays, prepared_packet_path, read, save, write
from review import camera_state, ji, native_for, source_frames
from source_paint import matte
from lane_end_evidence import foreground

PARENT = 'lane_motion_1118/checked'
DEST = OUT / 'review_fixes_1118'
CASES = {
    'okada_s03': [(57.38, ('R4',))],
    'okada_s05': [(100.26, ('F4', 'goal_crease_left'))],
    'okada_s08': [(139.70, ('F4', 'goal_crease_left'))],
    'jin_s03': [(39.12, ('board', 'Ad7'))],
    'jin_s04': [(42.18, ('Ad1', 'Ad25', 'Ad26')), (47.04, ('F4', 'goal_crease_left'))],
    'jin_s08': [(70.58, ('C0',))],
    'jin_s10': [(83.90, ('B1',))],
    'nayeon_s03': [(42.34, ('Ad1',)), (50.80, ('board','Ad8','Ad9','Ad10','Ad11','Ad12'))],
    'nayeon_s04': [(62.56, ('Ad1','Ad25','Ad26')), (63.56, ('Ad24','Ad25','Ad26')),
                   (78.82, ('Ad4',))],
}
# Each dark-on-light silhouette is traced at its actual ink edge. For the
# recessed QR panel the lighter material is inside. Do not use QR modules.
PRINT = {'Ad1': [('QR_white_frame', -1), ('left_vertical', 0), ('right_vertical', 0)],
         'Ad4': [('native440.boundary_0', -1)],
         'Ad7': [('left_vertical/path0',0), ('right_vertical/path0',0),
                 ('bottom_edge/path0',0), ('top_edge/path0',0)],
         'Ad8': [('blue_tile.boundary.0000',1)],
         'Ad9': [('red_field.boundary.0000',1)],
         'Ad10': [('glyph_1_S.boundary.0000',1), ('glyph_5_T.boundary.0000',1),
                  ('glyph_6_O.boundary.0000',1), ('glyph_8_O.boundary.0000',1)],
         'Ad11': [('outer_ring.outer',1), ('red_shield',1)],
         'Ad12': [('shield_print_component001',1)],
         'Ad24': [('outer_ring.outer',1), ('red_shield',1)],
         'Ad25': [('shield_print_component001',1)],
         'Ad26': [('outer_circle_print_stroke',1)]}


def selected_paths(native):
    groups = defaultdict(dict)
    for key, spec in native.paths.items():
        label = spec['label']
        if label in ('board.top','board.bottom'):
            groups['board'][key] = 0
        if label.startswith('ice.'):
            family = label[4:].split(':')[0]
            if family in ('R4','F4') and label.endswith((':outer', ':middle')):
                groups[family][key] = 1 if label.endswith('outer') else -1
            if family in ('goal_crease_left','C0','B1'):
                groups[family][key] = 1
        if label in PRINT:
            for pattern, sign in PRINT[label]:
                if key.endswith(pattern):
                    groups[label][key] = sign
    return dict(groups)


def sample_image(img, coord, method=cv2.INTER_LINEAR):
    grid = np.ascontiguousarray(coord, dtype=np.float32)
    return cv2.remap(img, grid, None, method, borderMode=cv2.BORDER_REPLICATE)


def orientations(uv, closed):
    tangent = np.gradient(uv,axis=0)
    if closed:
        tangent[[0,-1]] = uv[1] - uv[-2]
    length = np.linalg.norm(tangent,axis=1)
    normal = np.c_[tangent[:,1],-tangent[:,0]] / np.maximum(1.e-9,length[:,None])
    good = np.isfinite(uv).all(1) & np.isfinite(normal).all(1) & (length>.002)
    if closed and np.isfinite(uv).all():
        area = np.sum(uv[:-1,0]*uv[1:,1]-uv[1:,0]*uv[:-1,1])
        if area < 0: normal *= -1
    return normal,good


def contour_samples(curves, table, signs, family):
    result = []
    for key, (uv, status) in curves.items():
        uv=uv.copy()
        uv[status!=0]=np.nan
        closed = (family in ('R4','F4','C0') or
                  (family in PRINT and signs[key] and np.linalg.norm(uv[0]-uv[-1])<2.))
        normal,valid = orientations(uv,closed)
        if family=='goal_crease_left' and np.isfinite(uv).all():
            # The projected half-ellipse is convex. Its chord midpoint is inside.
            centre = (uv[0]+uv[-1])/2
            if np.nanmedian(np.sum((uv-centre)*normal,axis=1))<0: normal *= -1
        valid &= (uv>[-35,-16]).all(1) & (uv<[1955,1096]).all(1)
        if family=='board':
            # The native ring can cross the projection horizon.
            valid &= np.linalg.norm(np.gradient(uv,axis=0),axis=1)<130.
        ix=np.flatnonzero(valid)
        if not len(ix):continue
        # Keep both rounded ends of flat ellipses; equal pixel arc length
        # otherwise gives the two long horizontal branches almost all the mass.
        cells=np.rint(uv[ix]/2).astype(int)
        _,first=np.unique(cells,axis=0,return_index=True)
        ix=ix[np.sort(first)]
        if len(ix)>145 and family!='board':
            ix=ix[np.linspace(0,len(ix)-1,145).round().astype(int)]
        result.append((key,uv[ix],normal[ix],table[key][1][ix],float(signs[key])))
    return result


def signals(raw):
    gray=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY).astype('f4')
    return gray


def read_contour(gray,mask,parts,family,search_limit=None):
    """Register a bounded search region, then measure each independent source edge."""
    pts=np.concatenate([v[1] for v in parts])
    norm=np.concatenate([v[2] for v in parts])
    signed=np.concatenate([np.full(len(v[1]),v[4]) for v in parts])
    if len(pts)<6:return []
    lo=np.maximum([0,0],np.floor(pts.min(0))-[190,85]).astype(int)
    hi=np.minimum([1920,1080],np.ceil(pts.max(0))+[190,85]).astype(int)
    crop=gray[lo[1]:hi[1],lo[0]:hi[0]]
    if min(crop.shape)<8:return []
    blocked=mask[lo[1]:hi[1],lo[0]:hi[0]]
    sig=2.1 if family in ('F4','R4','goal_crease_left') else 1.5
    g=cv2.GaussianBlur(crop,(0,0),sig)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3,scale=1/8)
    gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3,scale=1/8)
    q=pts-lo
    # Equal influence for visible near-horizontal and near-vertical normals.
    # On a low-end view the horizontal extent is particularly informative.
    tip=np.abs(norm[:,0])
    weight=.5+np.minimum(12.,tip/(.015+tip.mean()))*1.7
    cap=6. if family in ('F4','R4','goal_crease_left') else 16.
    def search(shifts):
        scores=[]
        for block in np.array_split(shifts,max(1,int(np.ceil(len(shifts)/220)))):
            z=q[None]+block[:,None,:]
            good=(z>2).all(2)&(z<[crop.shape[1]-3,crop.shape[0]-3]).all(2)
            good&=sample_image(blocked,z,cv2.INTER_NEAREST)==0
            slope=sample_image(gx,z)*norm[:,0]+sample_image(gy,z)*norm[:,1]
            aligned=np.where(signed!=0,slope*signed,abs(slope))
            w=good*weight
            den=w.sum(1)
            score=np.sum(w*np.clip(aligned,-cap/2,cap),axis=1)/np.maximum(den,1.)
            score[(good.sum(1)<6)|(den<.35*weight.sum())]=-1000
            scores.extend(score)
        j=int(np.argmax(scores))
        return shifts[j],scores[j]
    extensive=family in ('F4','R4','goal_crease_left')
    xspan=min(144,max(48,int(np.ptp(pts[:,0])*.30))) if extensive else 52
    yspan=32 if extensive else 24
    if search_limit is not None:
        xspan=min(xspan,int(search_limit[0]))
        yspan=min(yspan,int(search_limit[1]))
    def grid(xs,ys):
        dx,dy=np.meshgrid(xs,ys)
        return np.column_stack((dx.ravel(),dy.ravel()))
    step=5 if extensive else 4
    shift,score=search(grid(np.arange(-xspan,xspan+1,step),np.arange(-yspan,yspan+1,3)))
    shift,score=search(shift+grid(np.arange(-step,step+1),np.arange(-3,4)))
    if score<(.20 if extensive else .38):return []
    offset=np.arange(-10.,11.)
    guide=q+shift
    z=guide[:,None]+norm[:,None]*offset[None,:,None]
    good=(z>1).all(2)&(z<[crop.shape[1]-2,crop.shape[0]-2]).all(2)
    visible=good.all(1)&~(sample_image(blocked,z,cv2.INTER_NEAREST)!=0).any(1)
    val=sample_image(g,z)
    grad=np.gradient(val,axis=1)
    derivative=np.where(signed[:,None]!=0,grad*signed[:,None],abs(grad))
    k=3+np.argmax(derivative[:,3:-3],axis=1)
    ii=np.arange(len(k));peak=derivative[ii,k]
    before,after=derivative[ii,k-1],derivative[ii,k+1]
    sub=np.clip((after-before)/(2*np.maximum(.001,2*peak-before-after)),-.6,.6)
    u=offset[k]+sub
    contrast=np.abs(val[ii,k+3]-val[ii,k-3])
    visible&=(peak>(.65 if extensive else 1.))&(contrast>(3.3 if extensive else 5.))&(abs(u)<7.2)
    if extensive:
        visible&=(val[ii,k-3]>100)&(val[ii,k+3]>100)
    actual=guide+lo+norm*u[:,None]
    result=[];pos=0
    for path,p,no,station,sign in parts:
        sl=slice(pos,pos+len(p));ok=visible[sl].copy()
        # Keep connected runs, excluding isolated reflections and skater edges.
        run=np.convolve(ok.astype(int),np.ones(5,int),'same') if len(p)>=5 else np.zeros(len(p))
        ok&=run>=3
        ids=np.flatnonzero(ok)
        for j in ids:
            a=pos+j
            result.append((path,float(station[j]),actual[a],p[j],no[j],float(contrast[a]),shift))
        pos+=len(p)
    return result


def read_stripe(raw,mask,parts,family,with_brackets=False):
    """Bisect the measured flanks of the blue line or centre circle."""
    result=[]
    for path,p,no,station,_ in parts:
        off=np.arange(-54.,55.)
        grid=p[:,None]+no[:,None]*off[None,:,None]
        im=sample_image(raw,grid).astype('f4')
        valid=(grid>3).all(2)&(grid<[1916,1076]).all(2)&(sample_image(mask,grid,cv2.INTER_NEAREST)==0)
        v=-.75*im.mean(2)+.25*(im[:,:,0]-im[:,:,2])
        v=gaussian_filter1d(v,1.7,axis=1)
        grad=np.gradient(v,axis=1)
        hits=[]
        for j,g in enumerate(grad):
            left=np.flatnonzero((g[1:-1]>=g[:-2])&(g[1:-1]>g[2:])&(g[1:-1]>.20))+1
            right=np.flatnonzero((g[1:-1]<=g[:-2])&(g[1:-1]<g[2:])&(g[1:-1]<-.20))+1
            best=None
            for l in left[left>5]:
                for r in right[(right>l+2)&(right<l+(45 if family=='B1' else 33))&(right<len(off)-6)]:
                    if not valid[j,l-4:r+5].all():continue
                    if np.min(im[j,l-4:r+5].mean(1))<105:continue
                    a,b=v[j,l-4:l-1].mean(),v[j,r+2:r+5].mean()
                    peak=v[j,(3*l+r)//4:(3*r+l)//4+1].mean()
                    strength=peak-max(a,b)
                    if strength<2.8 or abs(a-b)>max(7.,strength):continue
                    mid=(off[l]+off[r])/2
                    if abs(mid)>40:continue
                    score=strength/(1+abs(mid)*.013)
                    if best is None or score>best[0]:
                        best=(score,j,mid,strength,l,r)
            if best:hits.append(best)
        # A wrong stroke crossing seldom continues along the predicted material.
        for _,j,mid,strength,l,r in hits:
            close=[h[2] for h in hits if 0<abs(h[1]-j)<=4]
            if len(close)<2 or abs(mid-np.median(close))>3.5:continue
            hit=(path,float(station[j]),p[j]+mid*no[j],p[j],no[j],float(strength),np.zeros(2))
            result.append(hit+(grid[j,[l,r]],) if with_brackets else hit)
    return result


def read_rail(raw,mask,parts):
    result=[]
    for path,p,no,station,_ in parts:
        off=np.arange(-60.,61.)
        z=p[:,None]+no[:,None]*off[None,:,None]
        img=sample_image(raw,z).astype('f4')
        vis=(z>2).all(2)&(z<[1917,1077]).all(2)&(sample_image(mask,z,cv2.INTER_NEAREST)==0)
        # The kickplate is warmer than ice, whereas the blue top is darker and
        # bluer than the background above it.
        bot=path.endswith('bottom')
        signal=(.5*(img[:,:,2]+img[:,:,1])-img[:,:,0]) if bot else (img[:,:,0]-.48*img[:,:,1]-.52*img[:,:,2])-.22*img.mean(2)
        signal=gaussian_filter1d(signal,1.2,axis=1)
        for j in range(len(p)):
            if abs(no[j,1])<.12:continue
            orient=1 if no[j,1]>0 else -1
            if not bot:orient=-orient
            val=signal[j,::orient];valid=vis[j,::orient]
            slope=-np.gradient(val)
            peaks=np.flatnonzero((slope[1:-1]>slope[:-2])&(slope[1:-1]>=slope[2:])&(slope[1:-1]>.32))+1
            best=None
            for k in peaks[(peaks>=6)&(peaks<len(val)-6)]:
                colin=img[j,::orient][k-5:k-1].mean(0)
                strength=val[k-5:k-1].mean()-val[k+2:k+6].mean()
                if not valid[k-5:k+6].all() or strength<(3.2 if bot else 7.):continue
                if (bot and min(colin[1:])-colin[0]<3.5) or (not bot and colin[0]-colin[2]<7):continue
                score=strength/(1+abs(off[k])*.024)
                if best is None or score>best[0]:best=(score,k,strength)
            if best:
                _,k,strength=best
                delta=off[k]*orient
                result.append((path,float(station[j]),p[j]+no[j]*delta,p[j],no[j],float(strength),np.zeros(2)))
    return result


def scan(names, exact=False):
    cv2.setNumThreads(1)
    setup=read(PREPARED/'setup.json')['shots'];state=camera_state(PARENT);native=native_for(PARENT,'jin')
    paths=selected_paths(native)
    table=ji.native_paths(native,[k for fam in paths.values() for k in fam],count=257)
    world={k:z[0] for k,z in table.items()}
    folder=DEST/('pilot' if exact else 'reads')
    for name,s in setup.items():
        if names and name not in names:continue
        output=folder/(name+'.npz')
        if output.exists() and not exact:continue
        pts=state[name+'_pts'];t=state[name+'_times'];m=matte(s,pts)
        windows=CASES.get(name,[])
        schedule=defaultdict(set)
        for sec,families in windows:
            anchor=int(np.flatnonzero(pts==round(sec*12800))[0])
            for i in ([anchor] if exact else np.unique(np.r_[anchor,np.arange(max(0,anchor-55),min(len(pts),anchor+56),5)])):
                schedule[int(i)].update(families)
        if not exact:
            for i in np.unique(np.r_[0,np.arange(0,len(pts),14 if s['camera_index']==2 else 40),len(pts)-1]):
                schedule[int(i)].update(paths)
        frames=sorted(schedule)
        if not frames:continue
        data=defaultdict(list);counts=defaultdict(int)
        iterator=iter(source_frames(s['source'],pts[frames]))
        for start in range(0,len(frames),16):
            slab=frames[start:start+16]
            jobs=[(i,k,t[i]) for i in slab for fam in sorted(schedule[i]) for k in paths[fam]]
            projected=ji.project_groups(native,state[name+'_rigs'],state[name+'_controls'],s['first'],s['dt'],jobs,world)
            lookup={(i,k):z for (i,k,_),z in zip(jobs,projected)}
            for i in slab:
                pt,img=next(iterator);assert pt==pts[i]
                mask=foreground(s,m,i,pt)
                if mask is None:continue
                mask=cv2.dilate(mask,np.ones((7,7),np.uint8));mask[:270,:492]=255;mask[936:,1670:]=255
                gray=signals(img)
                for fam in sorted(schedule[i]):
                    parts=contour_samples({k:lookup[i,k] for k in paths[fam]},table,paths[fam],fam)
                    if not parts:continue
                    if fam=='board':
                        hits=read_rail(img,mask,parts)
                    elif fam in ('B1','C0'):
                        hits=read_stripe(img,mask,parts,fam)
                    else:
                        hits=read_contour(gray,mask,parts,fam)
                    for path,st,actual,guide,norm,contrast,shift in hits:
                        data['path'].append(path);data['station'].append(st);data['uv'].append(actual)
                        data['guide'].append(guide);data['normal'].append(norm);data['contrast'].append(contrast)
                        data['shift'].append(shift);data['family'].append(fam);data['idx'].append(i);data['pts'].append(pt)
                    counts[fam]+=len(hits)
        save(output,**{k:np.asarray(v) for k,v in data.items()},sampled=pts[frames],source_sha256=s['source_sha256'])
        print(name,dict(counts),flush=True)


if __name__=='__main__':
    from argparse import ArgumentParser
    ap=ArgumentParser(description=__doc__);ap.add_argument('--shots',nargs='+')
    ap.add_argument('--exact',action='store_true');ap.add_argument('--programme')
    args=ap.parse_args()
    names=args.shots
    if args.programme:
        names=[k for k,s in read(PREPARED/'setup.json')['shots'].items() if s['program']==args.programme]
    scan(names,args.exact)
