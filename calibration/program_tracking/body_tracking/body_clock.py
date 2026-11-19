"""Explicit source clock shared by body, shoe, physics and export consumers."""
from fractions import Fraction
from math import ceil, floor
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
    a,b=time_base(packet)
    values=np.asarray(packet['pts']if pts is None else pts)
    if values.dtype.kind in 'iu' and values.size:
        bounds=np.iinfo(values.dtype)
        if (a>bounds.max or int(values.min())*a<bounds.min
                or int(values.max())*a>bounds.max):
            # Preserve the integer product before conversion to seconds.
            # Fixed-width multiplication must not wrap a valid source clock.
            return np.asarray(values.astype(object)*a/b,dtype=float)
    return values*a/b

def source_interval_mask(pts, tb, start_seconds, end_seconds):
    """Inclusive selection on the integer source clock, before fit-time mapping."""
    tick=Fraction(*map(int,tb))
    first=ceil(Fraction(str(start_seconds))/tick)
    last=floor(Fraction(str(end_seconds))/tick)
    pts=np.asarray(pts)
    if pts.dtype.kind not in 'iu':
        raise ValueError('Source interval requires integer PTS')
    return (pts>=first)&(pts<=last)

def sample_period(packet):
    pts=np.asarray(packet['pts']);a,b=time_base(packet)
    if pts.ndim!=1 or len(pts)<2 or pts.dtype.kind not in 'iu'or np.any(pts[1:]<=pts[:-1]):raise ValueError('Ordered native source exposures required')
    # Compare before subtracting (unsigned clocks can otherwise wrap), and
    # use wide differences only when a signed storage span can overflow.
    wide=int(pts[-1])-int(pts[0])>np.iinfo(pts.dtype).max
    differences=np.diff(pts.astype(object) if wide else pts)
    return float(np.median(differences))*a/b

def source_nanoseconds(packet,pts):
    a,b=time_base(packet);values=np.asarray(pts)
    bounds=np.iinfo(np.int64)
    if values.dtype.kind in 'iu':
        tick=Fraction(a*1_000_000_000,b)
        result=np.empty(values.size,dtype=np.int64)
        for i,value in enumerate(values.flat):
            quotient,remainder=divmod(int(value)*tick.numerator,tick.denominator)
            # Exact nearest-integer, ties-to-even, including negative PTS.
            if 2*remainder>tick.denominator or (2*remainder==tick.denominator and quotient%2):
                quotient+=1
            if not bounds.min<=quotient<=bounds.max:
                raise OverflowError('Source nanoseconds exceed signed 64-bit storage')
            result[i]=quotient
        result=result.reshape(values.shape)
        return result[()] if values.ndim==0 else result
    # Preserve the existing floating query convention, but never publish the
    # int64 minimum sentinel produced by a nonfinite/out-of-range float cast.
    rounded=np.rint(np.asarray(pts,dtype=float)*a*1e9/b)
    if not np.isfinite(rounded).all():
        raise ValueError('Source nanoseconds must be finite')
    if any(not bounds.min<=int(value)<=bounds.max for value in np.asarray(rounded).flat):
        raise OverflowError('Source nanoseconds exceed signed 64-bit storage')
    return rounded.astype(np.int64)
