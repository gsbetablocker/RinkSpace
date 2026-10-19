"""Conservative source-only identity closure for exact observed point indices.

Derived indices retain their source-link evidence role. This operation cannot
turn automatic source continuity into independent whole-shot acceptance.
"""
from __future__ import annotations
import numpy as np

def source_point_identity_closure(observed_uv, colour_intervals, colour_peaks,
                                  candidate_offsets, accepted_point_links,
                                  reviewed_labels, expand_measured):
    uv=np.asarray(observed_uv,np.float64)
    bands=np.asarray(colour_intervals,np.float64)
    peaks=np.asarray(colour_peaks,np.float64)
    offsets=np.asarray(candidate_offsets,np.int64)
    labels=np.asarray(reviewed_labels,np.uint8).copy()
    if bands.shape!=uv.shape or peaks.shape!=(len(uv),) or labels.shape!=(len(uv),):
        raise ValueError('Exact source candidate arrays disagree')
    if offsets[0]!=0 or offsets[-1]!=len(uv) or np.any(np.diff(offsets)<0):
        raise ValueError('Exact candidate point partition differs')
    if np.any(labels>2):raise ValueError('Unknown reviewed identity value')
    parent=np.arange(len(uv),dtype=np.int64)
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    for a,b in np.asarray(accepted_point_links,np.int64):
        if not 0<=a<len(uv) or not -1<=b<len(uv):raise ValueError('Source point link index is outside frozen candidates')
        if b>=0:
            ra,rb=find(int(a)),find(int(b))
            if ra!=rb:parent[rb]=ra
    roots=np.asarray([find(i)for i in range(len(uv))],np.int64)
    candidates=[dict(component_id=i,observed_uv=uv[a:b],vertical_transition_interval_px=bands[a:b],color_peak_score=peaks[a:b])for i,(a,b)in enumerate(zip(offsets[:-1],offsets[1:]))]
    steps=0
    while True:
        before=labels.copy();root_labels=np.zeros(len(uv),np.uint8)
        for gi in np.flatnonzero(labels):
            root=int(roots[gi]);value=int(labels[gi])
            if root_labels[root]and root_labels[root]!=value:raise ValueError('Actual source point chain has conflicting reviewed identities')
            root_labels[root]=value
        inherited=root_labels[roots]
        if np.any((labels!=0)&(inherited!=0)&(labels!=inherited)):raise ValueError('Source identity conflict')
        labels=np.maximum(labels,inherited)
        for ci,candidate in enumerate(candidates):
            lo,hi=map(int,offsets[ci:ci+2])
            for value in (1,2):
                selected=np.flatnonzero(labels[lo:hi]==value)
                if not len(selected):continue
                for piece in expand_measured([candidate],[(0,int(i))for i in selected]):
                    indices=lo+np.asarray(piece['observed_uv_indices'],np.int64)
                    if np.any((labels[indices]!=0)&(labels[indices]!=value)):raise ValueError('Another reviewed material is inside a measured source span')
                    labels[indices]=value
        steps+=1
        if np.array_equal(before,labels):return labels,steps
