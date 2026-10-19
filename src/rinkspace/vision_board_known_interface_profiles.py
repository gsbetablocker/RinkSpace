"""Current RGB quantiles for a confirmed printed-panel/neutral-board interface.

A single broad transition has its own measured blur interval. Its gradient
maximum is not required to equal its midpoint. Multiple level crossings or a
material reversal stay missing. Search guides never become output pixels.
"""
from __future__ import annotations
import numpy as np
from .vision_programme_board_review import _sample


def read_known_board_interface_profiles(image, exclusion, guesses, normals, contrast, *, extent=24.):
    guesses=np.asarray(guesses,float);normals=np.asarray(normals,float);contrast=np.asarray(contrast,float)
    if guesses.shape!=normals.shape or contrast.shape!=(len(guesses),3):
        raise ValueError('Current signed board interface navigation arrays differ')
    offsets=np.arange(-extent,extent+.001,.5)
    tangents=np.c_[-normals[:,1],normals[:,0]]
    xy=(guesses[:,None,None]+offsets[None,:,None,None]*normals[:,None,None]
        +np.array([-2.,0.,2.])[None,None,:,None]*tangents[:,None,None])
    colors=np.median(_sample(image,xy.reshape(-1,3,2)).reshape(len(guesses),len(offsets),3,3),axis=2)
    readings=[]
    for i,(guess,normal,direction) in enumerate(zip(guesses,normals,contrast)):
        pixels=np.rint(xy[i]).astype(int)
        if ((pixels<0).any() or (pixels>=[1920,1080]).any()
                or exclusion[pixels[...,1],pixels[...,0]].any()):continue
        current=colors[i];first=current[:5];last=current[-5:]
        # The exact physical board membership comes from the caller. At least
        # one current measured side must actually be stable neutral board;
        # these criteria match the existing outer-print interface reader.
        def board(side):
            mean=side.mean(0)
            return mean.min()>=80. and side.std(0).max()<=5. and np.ptp(mean)<=18.
        if not (board(first) or board(last)):continue
        values=current@direction;low=float(np.median(values[:5]));high=float(np.median(values[-5:]))
        if high-low<24.:continue
        quantiles=[];indices=[]
        for fraction in (.25,.5,.75):
            level=low+fraction*(high-low)
            crossings=np.flatnonzero((values[:-1]<=level)&(values[1:]>level))
            if len(crossings)!=1:break
            k=int(crossings[0]);indices.append(k)
            quantiles.append(float(offsets[k]+.5*(level-values[k])/(values[k+1]-values[k])))
        if len(quantiles)!=3 or not quantiles[0]<quantiles[1]<quantiles[2]:continue
        # Reject a second material excursion within the actually measured
        # transition. Native BGR quantization bounds a subpixel interpolation
        # reversal; it is not a location tolerance against the guide/model.
        margin=float(np.abs(direction).sum())
        central=values[indices[0]:indices[2]+2]
        if np.any(np.diff(central)<-margin):continue
        strength=float((values[4:]-values[:-4]).max())
        readings.append((i,guess+quantiles[1]*normal,normal,strength,
            np.asarray([quantiles[0]-quantiles[1],quantiles[2]-quantiles[1]])))
    return readings
