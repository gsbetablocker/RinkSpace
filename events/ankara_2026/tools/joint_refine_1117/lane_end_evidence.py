"""Measure the four right-side Lane closing stripes on original exposures."""
from argparse import ArgumentParser
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import gzip
import json

import cv2
import numpy as np

from shared import DATA, EVENT, OUT, PREPARED, read, save, write
from review import camera_state, ji, native_for, source_frames
from source_paint import matte


BASE = 'repair/dense_delivery_1118'
WORK = OUT / 'lane_motion_1118'
ENDS = [f'demo_initial/lane{i}:1' for i in range(1, 5)]
ST = np.linspace(.045, .955, 37)
# Distances below are in the provisional L=1 chart. Both measured pixels are
# retained, so the strip search does not impose a width in the fitted scene.
OFF = np.arange(-110, 111, dtype=np.float32)
SPACING = .000095


@lru_cache(2)
def mask_index(programme):
    if programme == 'okada':
        path = DATA / 'runs/native_programme_20261031/source_person_mask_audit/all9_frame_bindings.jsonl.gz'
        with gzip.open(path, 'rt', encoding='utf8') as stream:
            return {entry['pts']: entry for entry in map(json.loads, stream)}
    path = DATA / 'runs/jin_sp_20261103/source_bootstrap/person_masks/frames.jsonl'
    with path.open(encoding='utf8') as stream:
        return {entry['source_binding']['source_pts']: entry for entry in map(json.loads, stream)}


def foreground(shot, packed, index, pt, *, exclude_uncertain_region=True):
    programme = shot['program']
    if packed is not None:
        if not packed['valid'][index]:
            return None
        return np.unpackbits(packed['packed'][index], axis=1, count=1920)
    entry = mask_index(programme).get(int(pt))
    if not entry:
        return None
    if programme == 'jin' and entry['source_binding']['source_sha256'] != shot['source_sha256']:
        raise ValueError('Mask and source identities differ')
    mask = cv2.imread(entry['mask_path'], cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (1080, 1920) or not mask.any():
        return None
    if exclude_uncertain_region and entry.get('diagnostic', {}).get('unresolved_support'):
        for box in entry.get('current_source_person_boxes', []):
            x,y,u,v = box['box_xyxy']
            dx,dy=.16*(u-x),.16*(v-y)
            x,y,u,v=np.clip([x-dx,y-dy,u+dx,v+dy],0,[1920,1080,1920,1080]).astype(int)
            mask[y:v,x:u]=255
    return mask


def display_foreground(shot,index,pt,raw):
    from hashlib import sha256
    entry=mask_index(shot['program']).get(int(pt))
    if not entry or entry.get('rectangle_fallback_applied'):return None
    if shot['program']=='okada':
        if entry['shot_id']!=shot['shot']:return None
        bgr=entry.get('source_BGR_sha256')
        if bgr:
            if sha256(raw.data).hexdigest()!=bgr:return None
        elif sha256(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB).data).hexdigest()!=entry.get('decoded_native_RGB_sha256'):
            return None
    elif entry['source_binding']['shot_id']!=shot['shot']:return None
    return foreground(shot,None,index,pt,exclude_uncertain_region=False)


def guide(native, shot, state, name):
    ends = np.array([native.paths[k]['parameters'] for k in ENDS]).reshape(4, 2, 3)
    world = ends[:, :1] + ST[None, :, None] * np.diff(ends, axis=1)
    probes = np.stack((world, world + [.001, 0., 0.])).reshape(-1, 3)
    t = state[name+'_times']
    uv = []
    for lo in range(0,len(t),96):
        jobs=[(i,'lane',t[i]) for i in range(lo,min(len(t),lo+96))]
        projected=ji.project_groups(native,state[name+'_rigs'],state[name+'_controls'],
            shot['first'],shot['dt'],jobs,{'lane':probes})
        for q,status in projected:
            q[status!=0]=np.nan
            uv.append(q.reshape(2,4,len(ST),2))
    uv=np.asarray(uv)
    xy=uv[:,0]
    tangent=np.gradient(xy,axis=2)
    tangent/=np.maximum(1.e-9,np.linalg.norm(tangent,axis=-1,keepdims=True))
    normal=np.stack((-tangent[...,1],tangent[...,0]),axis=-1)
    scale=np.sum((uv[:,1]-xy)*normal,axis=-1)/.001
    normal*=np.sign(scale)[...,None]
    visible=np.all((xy>=[9.,12.])&(xy<=[1910.,1067.]),axis=-1)&(abs(scale)>850.)
    return xy,normal,np.abs(scale),visible


def measure(raw, mask, xy, normal, scale, visible):
    ids=np.flatnonzero(visible)
    if len(ids)<7:
        return None
    points=xy[ids]
    if np.linalg.norm(points[-1]-points[0])<42.:
        return None
    norm=normal[ids]
    px=scale[ids]*SPACING
    tang=np.column_stack((-norm[:,1],norm[:,0]))
    cut=points[:,None,:]+norm[:,None,:]*(px[:,None]*OFF)[...,None]
    maps=(cut[:,:,None,:]+tang[:,None,None,:]*np.array([-2.5,0.,2.5])[None,None,:,None]).astype(np.float32)
    maps=maps.reshape(len(ids),-1,2)
    colors=cv2.remap(raw,maps,None,cv2.INTER_LINEAR).reshape(len(ids),len(OFF),3,3)
    blocked=cv2.remap(mask,maps,None,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,
        borderValue=255).reshape(len(ids),len(OFF),3).any(axis=2)
    color=np.median(colors,axis=2).astype(np.float32)
    blue,green,red=np.moveaxis(color,-1,0)
    blocked|=(np.max(color,axis=2)-np.min(color,axis=2)>20)|(green<116)
    luminance=.114*blue+.587*green+.299*red
    smooth=cv2.GaussianBlur(luminance,(7,1),1.5,borderType=cv2.BORDER_REPLICATE)
    grad=np.gradient(smooth,axis=1)
    # Locate opposite extrema in the source, with similar neutral ice outside
    # each flank. Vote across the entire visible segment, allowing line tilt.
    choices=[]
    for i,v in enumerate(smooth):
        g=grad[i]
        low=np.flatnonzero((g[1:-1]<=g[:-2])&(g[1:-1]<g[2:])&(g[1:-1]<-.08))+1
        high=np.flatnonzero((g[1:-1]>=g[:-2])&(g[1:-1]>g[2:])&(g[1:-1]>.08))+1
        lo,hi=np.meshgrid(low,high,indexing='ij')
        lo=lo.ravel();hi=hi.ravel()
        width=hi-lo
        good=(width>=8)&(width<=55)&(lo>=10)&(hi<len(v)-10)&(abs((lo+hi)/2-110)<78)
        lo=lo[good];hi=hi[good]
        if not len(lo):
            choices.append(np.empty((0,5)));continue
        width=hi-lo
        acc=np.r_[0,np.cumsum(v,dtype=np.float64)]
        invalid=np.r_[0,np.cumsum(blocked[i],dtype=int)]
        mid=(acc[hi-width//4+1]-acc[lo+width//4])/(hi-width//4+1-lo-width//4)
        left=(acc[lo-2]-acc[lo-9])/7-mid
        right=(acc[hi+10]-acc[hi+3])/7-mid
        contrast=np.minimum(left,right)
        good=(invalid[hi+10]==invalid[lo-9])&(contrast>2.0)&(np.maximum(left,right)<2.5*contrast)&(np.maximum(left,right)<30)
        good&=(np.minimum(-g[lo],g[hi])>.19*np.maximum(-g[lo],g[hi]))
        choices.append(np.c_[(lo+hi)/2-110,width,lo-110,hi-110,contrast][good])
    if sum(bool(len(x)) for x in choices)<7:
        return None
    # A stripe should provide a consistent width and linear offset along the
    # finite segment. Refine the consensus with the actual edge measurements.
    all_hits=np.vstack(choices)
    row=np.repeat(np.arange(len(ids)),[len(x) for x in choices])
    u=ST[ids]-.5
    centres=np.arange(-77.,78.,2.)
    best=0.;winner=None
    for slope in (-42.,-21.,0.,21.,42.):
        for width in (12.,20.,30.,42.):
            votes=np.zeros((len(ids),len(centres)),np.float32)
            strength=np.minimum(all_hits[:,4],12.)*np.maximum(0.,1.-abs(all_hits[:,1]-width)/max(10.,.60*width))
            batch=strength[:,None]*np.maximum(0.,1.-abs(all_hits[:,0,None]-slope*u[row,None]-centres)/8.)
            np.maximum.at(votes,row,batch)
            sums=votes.sum(0);j=int(np.argmax(sums))
            if sums[j]>best:
                best=sums[j];winner=(centres[j],slope,width)
    if winner is None:return None
    intercept,slope,width=winner
    for _ in range(2):
        ix=[];hits=[]
        for i,c in enumerate(choices):
            distance=abs(c[:,0]-intercept-slope*u[i])
            good=(distance<8)&(abs(c[:,1]-width)<max(8.,.47*width))
            if good.any():
                score=np.minimum(c[:,4],12.)/(1.+.18*distance)
                j=np.argmax(np.where(good,score,-1.))
                ix.append(i);hits.append(c[j])
        ix=np.asarray(ix,int);hits=np.asarray(hits)
        if len(ix)<7 or np.ptp(ST[ids[ix]])<.20 or np.linalg.norm(points[ix[-1]]-points[ix[0]])<38.:return None
        intercept,slope=np.linalg.lstsq(np.c_[np.ones(len(ix)),u[ix]],hits[:,0],rcond=None)[0]
        width=np.median(hits[:,1])
    if len(ix)<.46*len(ids) or np.median(hits[:,4])<3.1:return None
    selected=ids[ix]
    # The search columns are only local chart samples. Return source brackets
    # in native pixels, and never promote their along-line guide to an anchor.
    bracket=xy[selected,None,:]+norm[ix,None,:]*(px[ix,None]*hits[:,2:4])[...,None]
    return selected,bracket,hits[:,4],np.float64(intercept*SPACING),np.float64(slope*SPACING)


def scan(programme):
    cv2.setNumThreads(1)
    folder=WORK/'reads';folder.mkdir(parents=True,exist_ok=True)
    setup=read(PREPARED/'setup.json')['shots']
    state=camera_state(BASE);native=native_for(BASE,programme)
    for name,shot in setup.items():
        if shot['program']!=programme:continue
        output=folder/(name+'.npz')
        if output.exists():continue
        pts=state[name+'_pts']
        xy,normal,scale,visible=guide(native,shot,state,name)
        screen=visible.sum(axis=2)>=7
        indices=np.flatnonzero(screen.any(axis=1))
        data=defaultdict(list)
        packed=matte(shot,pts) if len(indices) else None
        for pt,raw in source_frames(shot['source'],pts[indices]) if len(indices) else ():
            i=np.searchsorted(pts,pt)
            mask=foreground(shot,packed,i,pt)
            if mask is None:continue
            mask=cv2.dilate(mask,np.ones((9,9),np.uint8))
            mask[:158,:590]=255;mask[:292,:451]=255;mask[928:,1618:]=255
            for lane in np.flatnonzero(screen[i]):
                m=measure(raw,mask,xy[i,lane],normal[i,lane],scale[i,lane],visible[i,lane])
                if m is None:continue
                js,bracket,contrast,offset,slope=m
                for j,b,c in zip(js,bracket,contrast):
                    for key,value in dict(pts=pt,index=i,lane=lane,station=ST[j],bracket=b,
                        uv=b.mean(0),contrast=c,guide=xy[i,lane,j],norm=normal[i,lane,j],
                        unit=scale[i,lane,j],frame_offset=offset,frame_slope=slope).items():
                        data[key].append(value)
        save(output,**{k:np.asarray(v) for k,v in data.items()},source_sha256=shot['source_sha256'],
            exposure_pts=pts,visible=screen)
        print(name,len(pts),len(indices),'visible / measured',len(data['pts']),flush=True)


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('programme',choices=['jin','okada','chen','nayeon'])
    scan(ap.parse_args().programme)
