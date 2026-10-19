"""Value-only stationary candidates of the original frozen Brown line cost.

No production profiling rule, source observation or domain is changed.  All
constants are the original float64 line chart.  Decimal arithmetic is used
only to distinguish a near-stationary float64 stopping flag from pixel error.
"""
from decimal import Decimal, localcontext
import math
import numpy as np


def _add(a, b):
    n=max(len(a),len(b));out=[Decimal(0)]*n
    for i,v in enumerate(a):out[i]+=v
    for i,v in enumerate(b):out[i]+=v
    return out


def _mul(a, b):
    out=[Decimal(0)]*(len(a)+len(b)-1)
    for i,v in enumerate(a):
        for j,w in enumerate(b):out[i+j]+=v*w
    return out


def _scale(a, s):return [v*s for v in a]
def _derivative(a):return [Decimal(i)*a[i] for i in range(1,len(a))]
def _value(a, x):
    result=Decimal(0)
    for v in reversed(a):result=result*x+v
    return result


def _roots_in_interval(coeff, lower, upper):
    """Derivative partitioning covers every real root on a bounded interval."""
    a=list(coeff)
    while len(a)>1 and a[-1]==0:a.pop()
    if len(a)==1:return []
    if len(a)==2:
        x=-a[0]/a[1]
        return [x] if lower<=x<=upper else []
    critical=_roots_in_interval(_derivative(a),lower,upper)
    points=sorted(set([lower,upper]+critical));roots=[]
    # Detect repeated roots at high-precision derivative zeros.  This is a
    # numerical comparison, not an independent physical-domain certificate.
    for x in points:
        error=sum(abs(v)*(max(Decimal(1),abs(x))**i) for i,v in enumerate(a))
        if abs(_value(a,x))<=Decimal('1e-65')*max(error,Decimal('1e-100')):roots.append(x)
    for left,right in zip(points[:-1],points[1:]):
        fl,fr=_value(a,left),_value(a,right)
        if fl*fr>=0:continue
        for _ in range(300):
            mid=(left+right)/2
            if mid==left or mid==right:break
            fm=_value(a,mid)
            if fm==0:left=right=mid;break
            if fl*fm<0:right=mid
            else:left=mid;fl=fm
        roots.append((left+right)/2)
    return sorted(set(roots))


def compare_failed_point(*,nq,u,fb,pp,k1,b,W,lam,lam_upper):
    """Return original-lambda versus minimum original-interval cost candidates.

    W is the original frozen profile divided by original source sigma squared;
    returned costs are its quadratic nuisance metric, not full Huber source
    costs.  No observed source UV or weights are changed.
    """
    if not all(np.isfinite(np.asarray(v,float)).all() for v in (nq,u,fb,pp,k1,b,W,lam)):
        raise ValueError('Finite original line constants required')
    if lam<0 or lam_upper<=0 or lam>lam_upper or np.any(np.asarray(W)<0):
        raise ValueError('Original nonnegative nuisance interval and weights required')
    D=lambda x:Decimal.from_float(float(x))
    with localcontext() as decimal_context:
        decimal_context.prec=80
        bounded=math.isfinite(lam_upper)
        scale=D(lam_upper if bounded else max(abs(lam),1e-6))
        n=[ [D(nq[i]),scale*D(u[i])] for i in range(2) ]
        radial=_add([Decimal(1)],_scale(_add(_mul(n[0],n[0]),_mul(n[1],n[1])),D(k1)))
        r=[_add(_scale(_mul(n[i],radial),D(fb)),[D(pp[i])-D(b[i])]) for i in range(2)]
        cost=_scale(_add(_scale(_mul(r[0],r[0]),D(W[0])),_scale(_mul(r[1],r[1]),D(W[1]))),Decimal('.5'))
        stationary=_derivative(cost)
        while len(stationary)>1 and stationary[-1]==0:stationary.pop()
        # For an unbounded original interval, Cauchy's root bound includes
        # all finite real stationary candidates; the polynomial tends upward.
        high=Decimal(1) if bounded else Decimal(1)+max((abs(v/stationary[-1]) for v in stationary[:-1]),default=Decimal(0))
        roots=_roots_in_interval(stationary,Decimal(0),high)
        candidates=[Decimal(0)]+roots+([Decimal(1)] if bounded else [])
        current=D(lam)/scale
        best=min(candidates,key=lambda x:_value(cost,x))
        old_r=np.array([float(_value(v,current)) for v in r])
        new_r=np.array([float(_value(v,best)) for v in r])
        gradient=_value(stationary,current)/scale
        second=_value(_derivative(stationary),current)/(scale*scale)
        return dict(original_lambda=float(lam),best_lambda=float(best*scale),original_lambda_upper=float(lam_upper),
            finite_original_interval=bounded,stationary_candidates=len(roots),
            candidate_lambdas=[float(x*scale) for x in candidates],
            original_native_residual=old_r,stationary_native_residual=new_r,
            native_pixel_change=new_r-old_r,max_native_pixel_change=float(np.max(abs(new_r-old_r))),
            frozen_profile_quadratic_cost_before=float(_value(cost,current)),
            frozen_profile_quadratic_cost_after=float(_value(cost,best)),
            frozen_profile_quadratic_cost_change=float(_value(cost,best)-_value(cost,current)),
            nominal_Newton_lambda_change=float(-gradient/second) if second>0 else None,
            original_W_unchanged=True,precision_decimal_digits=80,
            candidate_is_diagnostic_not_source_or_fit_update=True)
