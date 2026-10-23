"""Follow a visible painted band using only native pixels and finite source seeds."""
import numpy as np
from .source_worker import _channels,_remeasure

def trace_band(rgb,mask,seed_sides,seed_normal,step=6.):
    channels=_channels(rgb)[:1];h,w=rgb.shape[:2];sides=np.asarray(seed_sides,float);normal=np.asarray(seed_normal,float);normal/=np.linalg.norm(normal);start=.5*(sides[0]+sides[1]);initial=_remeasure(start,normal,channels,'paint_center',sides=sides)
    if initial['current_uv'] is None:return []
    visited=set();branches=[]
    for direction in (-1,1):
        current=initial;normal0=normal.copy();tangent=direction*np.array([normal[1],-normal[0]]);history=[np.array(current['current_uv'])];branch=[]
        while True:
            q=np.array(current['current_uv']);edges=np.array(current['flank_uv']);xy=np.rint(np.vstack([q,edges])).astype(int)
            if np.any(xy[:,0]<4)or np.any(xy[:,0]>=w-4)or np.any(xy[:,1]<4)or np.any(xy[:,1]>=h-4)or np.any(mask[xy[:,1],xy[:,0]]):break
            cell=tuple(np.floor(q/3).astype(int))
            if cell in visited:break
            visited.add(cell);branch.append(current)
            if len(history)>=5:
                recent=np.array(history[-9:]);ds=np.r_[0,np.cumsum(np.linalg.norm(np.diff(recent,axis=0),axis=1))];ds-=ds[-1];coef=np.polyfit(ds,recent,2);v=coef[1];v/=np.linalg.norm(v)
                if np.dot(v,tangent)<.8:break
                tangent=v;normal0=np.array([-tangent[1],tangent[0]])
                if np.dot(normal0,normal)<0:normal0=-normal0
            predict=q+step*tangent;oldwidth=current['width_px'];seed=np.array([predict-normal0*oldwidth/2,predict+normal0*oldwidth/2]);nxt=_remeasure(predict,normal0,channels,'paint_center',sides=seed)
            if nxt['current_uv'] is None or not .5*oldwidth<nxt['width_px']<2*oldwidth:break
            # A narrow band must keep opposite flank polarity; a broad shadow is not a painted line.
            if nxt['flank_profiles'][0]['polarity']*nxt['flank_profiles'][1]['polarity']!=-1:break
            current=nxt;history.append(np.array(current['current_uv']))
        branches.append(branch)
        visited.discard(tuple(np.floor(start/3).astype(int)))
    return list(reversed(branches[0]))+branches[1][1:]
