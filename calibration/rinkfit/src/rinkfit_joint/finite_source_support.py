"""Finite source-image support: never join gaps or extrapolate a named entity.

This is a necessary source-navigation gate, not a grant of physical identity.
An image-plane homography transports only already observed finite segments.
"""
from __future__ import annotations
import numpy as np


def source_segments(annotation, curve_id):
    spans=[]
    for feature in annotation.get('features',[]):
        key=feature.get('world_curve',feature.get('target_id','').removeprefix('ice.'))
        if key!=curve_id: continue
        for segment in feature.get('segments',[]):
            points=np.asarray(segment.get('source_native_readout_uv',segment.get('uv',[])),float)
            if points.ndim==2 and points.shape[1]==2 and len(points)>=2 and np.isfinite(points).all():
                spans.append(points)
    return spans


def finite_support(points, spans, transform=None, *, normal_tolerance_px):
    """Test finite, disconnected spans, rejecting projective poles.

    Endpoints have no longitudinal expansion. A normal search tolerance cannot
    authorize an extension, a missing arc, or a chord across separate spans.
    """
    points=np.asarray(points,float).reshape(-1,2)
    result=np.zeros(len(points),bool)
    transform=np.eye(3) if transform is None else np.asarray(transform,float)
    if transform.shape!=(3,3) or not np.isfinite(transform).all(): return result
    for span in spans:
        raw=np.asarray(span,float)
        homogeneous=np.c_[raw,np.ones(len(raw))]@transform.T
        denom=homogeneous[:,2]
        safe=np.isfinite(homogeneous).all(1)&(np.abs(denom)>1e-9)
        mapped=homogeneous[:,:2]/np.where(safe,denom,1.)[:,None]
        for j in range(len(mapped)-1):
            if not(safe[j] and safe[j+1]) or denom[j]*denom[j+1]<=0: continue
            first,last=mapped[j:j+2];direction=last-first;square=float(direction@direction)
            if square<1e-12: continue
            delta=points-first;t=delta@direction/square
            normal=delta-t[:,None]*direction
            result|=(t>=-1e-9)&(t<=1+1e-9)&(np.sum(normal*normal,1)<=normal_tolerance_px**2)
    return result
