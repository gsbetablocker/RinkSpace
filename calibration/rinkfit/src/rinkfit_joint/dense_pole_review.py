"""Trace finite source-navigation topology without converting it to UV truth.

A material profile can read real pixels on a line segment that was accidentally
created by joining opposite sides of a homography pole. Such a pixel does not
inherit the named physical curve. This consumer retains the pixel and withholds
that identity; it never chooses a substitute material from model residuals.
"""
from __future__ import annotations
import numpy as np


def trace_source_queries(path,projected,homogeneous_denominator,native_hw,spacing,radius):
    """Mirror the source query sampler and retain each query's finite segment."""
    h,w=native_hw;uv=np.asarray(projected,float);den=np.asarray(homogeneous_denominator,float)
    if den.shape!=(len(uv),):raise ValueError("One actual homogeneous denominator per sample is required")
    good=np.isfinite(uv).all(1)
    output=[];normals=[];segments=[];pole=[]
    for index in range(len(uv)-1):
        if not good[index:index+2].all():continue
        a,b=uv[index:index+2];vector=b-a;length=float(np.linalg.norm(vector))
        if length<=1e-9:continue
        lo,hi=0.,1.
        for axis,limit in ((0,w-1),(1,h-1)):
            if abs(vector[axis])<1e-12:
                if not 0<=a[axis]<=limit:hi=-1.;break
            else:
                ends=sorted((-a[axis]/vector[axis],(limit-a[axis])/vector[axis]))
                lo,hi=max(lo,ends[0]),min(hi,ends[1])
        if hi<lo:continue
        start=a+lo*vector;stop=a+hi*vector
        visible_length=float(np.linalg.norm(stop-start));unit=vector/length;normal=np.array([-unit[1],unit[0]])
        distances=np.arange(0.,visible_length+1e-6,spacing)
        if not len(distances) or visible_length-distances[-1]>spacing*.5:
            distances=np.r_[distances,visible_length]
        crosses=(not np.isfinite(den[index:index+2]).all()) or den[index]*den[index+1]<=0.
        for distance in distances:
            point=start+distance*unit
            if output and np.linalg.norm(point-output[-1])<spacing*.5:continue
            n=-normal if path.get("_projected_center") is not None and (point-path["_projected_center"])@normal<0 else normal
            output.append(point);normals.append(n);segments.append(index);pole.append(bool(crosses))
    points=np.asarray(output,float).reshape(-1,2);normals=np.asarray(normals,float).reshape(-1,2)
    intervals=np.tile([-radius,radius],(len(points),1)).astype(float)
    for axis,limit in ((0,w-1),(1,h-1)):
        movement=np.abs(normals[:,axis])>1e-12
        first=np.divide(4-points[:,axis],normals[:,axis],out=np.full(len(points),-np.inf),where=movement)
        second=np.divide(limit-4-points[:,axis],normals[:,axis],out=np.full(len(points),np.inf),where=movement)
        intervals[:,0]=np.maximum(intervals[:,0],np.minimum(first,second))
        intervals[:,1]=np.minimum(intervals[:,1],np.maximum(first,second))
    enough=intervals[:,1]-intervals[:,0]>=20.
    return dict(uv=points[enough],normal=normals[enough],interval=intervals[enough],
        sample_segment_index=np.asarray(segments,dtype=np.int32)[enough],
        homography_pole_chord=np.asarray(pole,dtype=bool)[enough])


def validate_original_source_query(subject,query,*,max_normal_shift_px=32.):
    """Check exact original query index/material and actual original UV support."""
    uv=np.asarray(subject["uv"],float);q=np.asarray(query["uv"],float);normal=np.asarray(query["normal"],float)
    displacement=uv-q;normal_shift=float(displacement@normal)
    tangential=float(displacement@np.array([-normal[1],normal[0]]))
    if abs(tangential)>1e-6 or abs(normal_shift)>max_normal_shift_px+1e-6:
        raise ValueError("Selected source reading is not on its original reader's declared profile")
    return dict(original_source_UV_unchanged=True,tangential_profile_identity_error_px=tangential,
        original_profile_normal_shift_px=normal_shift,navigation_sample_segment_index=int(query["sample_segment_index"]),
        homography_pole_chord=bool(query["homography_pole_chord"]),
        physical_instance_qualification="unknown" if query["homography_pole_chord"] else "unchanged_unreviewed",
        model_residual_used_for_qualification=False,replacement_UV_or_entity_issued=False)
