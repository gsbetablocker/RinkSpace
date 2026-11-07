"""Explicit source clock shared by body, shoe, physics and export consumers."""
from fractions import Fraction
import numpy as np

def time_base(packet):
    values=[np.asarray(packet[k]) for k in ('time_base','source_time_base') if k in packet]
    if not values:return (1,12800)  # preserved historical schema
    result=[]
    for value in values:
        if value.shape!=(2,)or value.dtype.kind not in 'iu'or np.any(value<=0):raise ValueError('Invalid declared body source time base')
        q=Fraction(int(value[0]),int(value[1]));result.append((q.numerator,q.denominator))
    if len(set(result))!=1:raise ValueError('Conflicting body source clocks')
    return result[0]

def source_seconds(packet,pts=None):
    a,b=time_base(packet);return np.asarray(packet['pts']if pts is None else pts)*a/b

def sample_period(packet):
    pts=np.asarray(packet['pts']);a,b=time_base(packet)
    if pts.ndim!=1 or len(pts)<2 or pts.dtype.kind not in 'iu'or np.any(np.diff(pts)<=0):raise ValueError('Ordered native source exposures required')
    return float(np.median(np.diff(pts)))*a/b

def source_nanoseconds(packet,pts):
    a,b=time_base(packet);return np.rint(np.asarray(pts,dtype=float)*a*1e9/b).astype(np.int64)
