"""Independent local profile root certificates using interval automatic derivatives.

The three unknowns are finite material station, the ORIGINAL source query
scalar, and native rolling-shutter row.  Every Jacobian entry encloses the
complete specified box.  No finite differences or derivative samples enter
the certificate.  Binary64 basic operations are rounded out by nextafter;
sin/cos/exp/log use interval polynomials with explicit Taylor remainder bounds.
An interval crossing a cubic knot or a finite support boundary is unresolved.

Only ice_segment profiles are implemented.  A passed Krawczyk box proves one
root in that local box and nonsingularity throughout it; it does not prove
global uniqueness on the complete source query interval.  Whole-readout RS
uniqueness remains the separate NativeCamera certificate.

The already reconstructed NativeCamera R0/n/b/controls/head/optical fields are
fixed binary64 constants in this proof.  The floating camera_from_event
normalization/rotation reconstruction is not itself an interval computation of
all solver parameters.  Cross-evaluator floating discrepancies are a separate
replay check; this certificate concerns the frozen independent camera model.
"""
from __future__ import annotations

import math
import numpy as np

VERSION = "native_profile_interval_ad_krawczyk_v1"
_NINF, _PINF = -np.inf, np.inf


def _down(x):
    return np.nextafter(np.asarray(x, dtype=np.float64), _NINF)


def _up(x):
    return np.nextafter(np.asarray(x, dtype=np.float64), _PINF)


class Interval:
    """Closed vectorized binary64 intervals, with outward basic arithmetic."""
    __array_priority__ = 1000

    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = self.lo if hi is None else np.asarray(hi, dtype=np.float64)

    @staticmethod
    def of(value):
        return value if isinstance(value, Interval) else Interval(value)

    def __add__(self, other):
        other = self.of(other)
        return Interval(_down(self.lo + other.lo), _up(self.hi + other.hi))

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -self.of(other)

    def __rsub__(self, other):
        return self.of(other) + -self

    def __mul__(self, other):
        other = self.of(other)
        products = np.stack(np.broadcast_arrays(self.lo * other.lo,
            self.lo * other.hi, self.hi * other.lo, self.hi * other.hi))
        return Interval(_down(np.min(products, axis=0)), _up(np.max(products, axis=0)))

    __rmul__ = __mul__

    def reciprocal(self):
        good = (self.lo > 0) | (self.hi < 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lo, hi = _down(1 / self.hi), _up(1 / self.lo)
        return Interval(np.where(good, lo, _NINF), np.where(good, hi, _PINF))

    def __truediv__(self, other):
        return self * self.of(other).reciprocal()

    def __rtruediv__(self, other):
        return self.of(other) * self.reciprocal()

    def __pow__(self, power):
        if not isinstance(power, int) or power < 0:
            raise ValueError("Only nonnegative integer interval powers")
        if power == 0:
            return Interval(np.ones_like(self.lo))
        if power == 1:
            return self
        if power == 2:
            lo = np.where(self.lo >= 0, self.lo*self.lo,
                 np.where(self.hi <= 0, self.hi*self.hi, 0.))
            hi = np.maximum(self.lo*self.lo, self.hi*self.hi)
            return Interval(np.maximum(0., _down(lo)), _up(hi))
        result, value, exponent = Interval(1.), self, power
        while exponent:
            if exponent & 1:
                result = result * value
            exponent //= 2
            if exponent:
                value = value ** 2
        return result

    def absolute_upper(self):
        return np.maximum(abs(self.lo), abs(self.hi))

    def finite(self):
        return np.isfinite(self.lo) & np.isfinite(self.hi) & (self.lo <= self.hi)


def _coefficient(numerator, denominator):
    """The exact rational coefficient is between the adjacent floats."""
    value = float(numerator / denominator)
    return Interval(_down(value), _up(value))


def _sincos(x):
    # No libm trigonometric evaluation is used in the proof.  The camera
    # chart is restricted to |angle|<=4; wider charts fail explicitly.
    x = Interval.of(x)
    squared = x ** 2
    sine = _coefficient((-1)**19, math.factorial(39))
    cosine = _coefficient((-1)**20, math.factorial(40))
    for k in range(18, -1, -1):
        sine = sine*squared + _coefficient((-1)**k, math.factorial(2*k+1))
    for k in range(19, -1, -1):
        cosine = cosine*squared + _coefficient((-1)**k, math.factorial(2*k))
    radius = Interval(x.absolute_upper())
    sin_remainder = (radius ** 41 * _coefficient(1, math.factorial(41))).hi
    cos_remainder = (radius ** 42 * _coefficient(1, math.factorial(42))).hi
    return (sine*x + Interval(-sin_remainder, sin_remainder),
            cosine + Interval(-cos_remainder, cos_remainder))


def _exp(x):
    x = Interval.of(x)
    radius = x.absolute_upper()
    safe_radius = np.where(np.isfinite(radius), radius, 64.)
    exponent = np.maximum(0, np.ceil(np.log2(np.maximum(safe_radius, .125)/.125))).astype(int)
    exponent = np.minimum(exponent, 16)
    divisor = np.ldexp(np.ones_like(safe_radius), exponent)
    reduced = x / divisor
    value = _coefficient(1, math.factorial(18))
    for k in range(17, -1, -1):
        value = value*reduced + _coefficient(1, math.factorial(k))
    # exp(|reduced|)<2 for |reduced|<=.125.  Taylor Lagrange remainder.
    remainder = (Interval(reduced.absolute_upper())**19 * _coefficient(2, math.factorial(19))).hi
    value = value + Interval(-remainder, remainder)
    for k in range(int(np.max(exponent, initial=0))):
        square = value**2
        value = Interval(np.where(exponent > k, square.lo, value.lo),
                         np.where(exponent > k, square.hi, value.hi))
    return value


def _log_positive_point(value):
    """Real log of a positive stored binary64 constant, with series remainder."""
    value = np.asarray(value, float)
    if np.any(value <= 0):
        raise ValueError("positive logarithm constant required")
    mantissa, exponent = np.frexp(value)

    def log_reduced(v):
        z = (Interval(v)-1)/(Interval(v)+1)
        zz = z**2
        p = _coefficient(1, 61)
        for k in range(29, -1, -1):
            p = p*zz + _coefficient(1, 2*k+1)
        r = Interval(z.absolute_upper())
        rem = (2*(r**63)/(63*(1-r**2))).hi
        return 2*z*p + Interval(-rem, rem)

    return log_reduced(mantissa) + exponent*log_reduced(2.)


def _sqrt(x):
    x = Interval.of(x)
    with np.errstate(invalid="ignore"):
        lo = np.maximum(0., _down(np.sqrt(np.maximum(x.lo, 0.))))
        # One outward radicand ulp also covers subnormal squared norms, where
        # moving the sqrt by a few ulps may not change its rounded square.
        hi = _up(np.sqrt(np.maximum(_up(x.hi), 0.)))
    # Verify the direction of each square-root enclosure using outward
    # multiplication; extra ulps are only numerical enclosure, not clamping.
    for _ in range(4):
        lo = np.where((Interval(lo)**2).hi > x.lo, np.maximum(0., _down(lo)), lo)
        hi = np.where((Interval(hi)**2).lo < x.hi, _up(hi), hi)
    verified_lower=(lo==0.)|((Interval(lo)**2).hi<=x.lo)
    verified_upper=(Interval(hi)**2).lo>=x.hi
    verified=(x.lo>=0)&verified_lower&verified_upper
    return Interval(np.where(verified,lo,np.nan),np.where(verified,hi,np.nan))


class _AD:
    """Interval value and three exact analytic derivative enclosures."""
    __array_priority__ = 2000

    def __init__(self, value, derivatives=None):
        self.v = Interval.of(value)
        self.d = [Interval(0.) for _ in range(3)] if derivatives is None else derivatives

    @staticmethod
    def of(value):
        return value if isinstance(value, _AD) else _AD(value)

    @classmethod
    def variable(cls, lo, hi, index):
        derivatives = [Interval(0.) for _ in range(3)]
        derivatives[index] = Interval(1.)
        return cls(Interval(lo, hi), derivatives)

    def __add__(self, other):
        b = self.of(other)
        return _AD(self.v+b.v, [a+c for a,c in zip(self.d,b.d)])

    __radd__ = __add__

    def __neg__(self):
        return _AD(-self.v, [-d for d in self.d])

    def __sub__(self, other):
        return self + -self.of(other)

    def __rsub__(self, other):
        return self.of(other) + -self

    def __mul__(self, other):
        b = self.of(other)
        return _AD(self.v*b.v, [a*b.v+self.v*c for a,c in zip(self.d,b.d)])

    __rmul__ = __mul__

    def reciprocal(self):
        inverse = self.v.reciprocal()
        return _AD(inverse, [-d*inverse**2 for d in self.d])

    def __truediv__(self, other):
        return self * self.of(other).reciprocal()

    def __rtruediv__(self, other):
        return self.of(other) * self.reciprocal()

    def square(self):
        return _AD(self.v**2, [2*self.v*d for d in self.d])

    def sincos(self):
        sine, cosine = _sincos(self.v)
        return _AD(sine,[cosine*d for d in self.d]), _AD(cosine,[-sine*d for d in self.d])

    def exp(self):
        value = _exp(self.v)
        return _AD(value,[value*d for d in self.d])

    def sqrt(self):
        value = _sqrt(self.v)
        return _AD(value,[d/(2*value) for d in self.d])


def _dot(a,b):
    return sum((x*y for x,y in zip(a,b)), _AD(0.))


def _cross(a,b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def _matrix_multiply(a,b):
    return [[_dot(row,[b[k][j] for k in range(3)]) for j in range(3)] for row in a]


def _rotate(axis,angle):
    x,y,z=axis
    K=[[0.,-z,y],[z,0.,-x],[-y,x,0.]]
    # K2 constants themselves must include the rounding of their products:
    # construct exact interval dot products rather than accepting np.dot.
    K2=[[_dot(K[i],[K[k][j] for k in range(3)]) for j in range(3)] for i in range(3)]
    sine,cosine=angle.sincos()
    return [[float(i==j)+sine*K[i][j]+(1-cosine)*K2[i][j] for j in range(3)] for i in range(3)]


def _polynomial(coeff,x):
    value=_AD(coeff[-1])
    for c in coeff[-2::-1]:
        value=value*x+c
    return value


def _profile_system(scene,camera,material,branch,frame_time,curves,lower,upper):
    """Enclose F and J(F) over one whole box per original profile row."""
    n=len(lower);s=_AD.variable(lower[:,0],upper[:,0],0)
    query_alpha=_AD.variable(lower[:,1],upper[:,1],1)
    row=_AD.variable(lower[:,2],upper[:,2],2)
    finite=np.ones(n,bool);q=np.zeros((n,4));reason=np.full(n,"",dtype="U80")
    for j,(mi,pi) in enumerate(zip(material,branch)):
        path=scene.path(mi,pi)
        if path['kind'] not in (0,'ice_segment'):
            finite[j]=False;reason[j]='unsupported_material_primitive';continue
        q[j]=scene.values[int(path['geometry_block'])]
    world=[q[:,0]*(1-s)+q[:,2]*s, q[:,1]*(1-s)+q[:,3]*s, _AD(np.zeros(n))]
    time=frame_time+camera.tau*(row/1079-.5)
    x=(time-camera.first)/camera.dt
    center=(x.v.lo+x.v.hi)/2
    cell=np.floor(center).astype(int)
    cell=np.where(cell==len(camera.controls)-2,cell-1,cell)
    support=(cell>=1)&(cell<=len(camera.controls)-3)
    # A single cubic expression is used only if it covers the entire time box.
    same_cell=support&(x.v.lo>=cell)&(x.v.hi<=cell+1)
    reason[~same_cell]='cubic_knot_or_allocated_time_domain_crossing'
    finite &= same_cell
    safe=np.clip(cell,1,len(camera.controls)-3)
    u=x-safe
    weights=[(1-u)*(1-u)*(1-u)/6,
             (3*u*u*u-6*u*u+4)/6,
             (-3*u*u*u+3*u*u+3*u+1)/6,
             u*u*u/6]
    abf=[sum((weights[k]*camera.controls[safe+k-1,j] for k in range(4)),_AD(0.)) for j in range(3)]
    alpha,beta,logf=abf
    trig_ok=(alpha.v.absolute_upper()<=4)&(beta.v.absolute_upper()<=4)
    exp_ok=logf.v.absolute_upper()<=64
    reason[~trig_ok]='trigonometric_chart_exceeds_certified_Taylor_domain'
    reason[~exp_ok]='exponential_chart_exceeds_certified_Taylor_domain'
    finite &= trig_ok&exp_ok
    R=_matrix_multiply(_matrix_multiply(_rotate(camera.b,beta),camera.R0.tolist()),_rotate(camera.n,alpha))
    wb=[_dot([R[k][j] for k in range(3)],camera.b) for j in range(3)]
    h=_cross(camera.n,wb);hn=_dot(h,h).sqrt();h=[v/hn for v in h]
    z=logf-_AD(_log_positive_point(float(camera.optics['f_ref'])))
    ell=camera.head.get('pupil_linear',0.)*z+camera.head.get('pupil_quadratic',0.)*z*z
    D=[_AD(camera.d[0]),_AD(camera.d[1]),camera.d[2]+ell]
    C=[camera.P[j]+_dot([R[k][j] for k in range(3)],D)+camera.a*h[j] for j in range(3)]
    Y=[_dot(R[j],[world[k]-C[k] for k in range(3)]) for j in range(3)]
    xx,yy=Y[0]/Y[2],Y[1]/Y[2];r2=xx.square()+yy.square()
    k1=_polynomial(camera.optics['k1'],z);k2=_polynomial(camera.optics['k2'],z)
    radial=1+k1*r2+k2*r2*r2
    p1,p2=float(camera.optics['p1']),float(camera.optics['p2'])
    xd=xx*radial+2*p1*xx*yy+p2*(r2+2*xx.square())
    yd=yy*radial+p1*(r2+2*yy.square())+2*p2*xx*yy
    eta=float(camera.optics.get('eta',0.));pp=camera.optics['principal_point'];pl=camera.optics.get('principal_linear',[0.,0.])
    U=pp[0]+z*pl[0]+(logf+eta/2).exp()*xd
    V=pp[1]+z*pl[1]+(logf-eta/2).exp()*yd
    numerator_u=_polynomial(np.asarray(curves['numerator_u']).T,query_alpha)
    numerator_v=_polynomial(np.asarray(curves['numerator_v']).T,query_alpha)
    denominator=_polynomial(np.asarray(curves['denominator']).T,query_alpha)
    F=[U-numerator_u/denominator,V-numerator_v/denominator,row-V]
    denominator_ok=(denominator.v.lo>0)|(denominator.v.hi<0)
    radius_squared_lower=(Interval(float(camera.optics['radius_max']))**2).lo
    domain=(Y[2].v.lo>0)&(hn.v.lo>1e-10)&(r2.v.hi<=radius_squared_lower)&denominator_ok
    reason[~domain]='depth_optical_radius_query_denominator_or_head_domain'
    # Brown Jacobian on every ray of the complete (s,alpha,v) root box.
    # The radial symmetric eigenvalues are L and 1+3*k1*r²+5*k2*r⁴;
    # the tangential symmetric perturbation has norm <=6*||p||*r.
    # Restrict a squared-norm enclosure to [0,+inf), an analytic property,
    # before sqrt; no measured ray or physical parameter is clamped.
    radius_box=_sqrt(Interval(np.maximum(0.,r2.v.lo),r2.v.hi))
    p_squared=Interval(p1)**2+Interval(p2)**2
    p_norm=_sqrt(Interval(np.maximum(0.,p_squared.lo),p_squared.hi))
    tangential_norm=6*p_norm*radius_box
    radial_tangent=1+k1.v*r2.v+k2.v*(r2.v**2)
    radial_normal=1+3*k1.v*r2.v+5*k2.v*(r2.v**2)
    brown_lower=np.minimum((radial_tangent-tangential_norm).lo,
                           (radial_normal-tangential_norm).lo)
    brown_ok=np.isfinite(brown_lower)&(brown_lower>0)
    reason[~brown_ok]='Brown_no_fold_entire_root_box_uncertified'
    finite &= domain&brown_ok
    f_lo=np.stack([f.v.lo+np.zeros(n) for f in F],axis=1)
    f_hi=np.stack([f.v.hi+np.zeros(n) for f in F],axis=1)
    j_lo=np.stack([np.stack([d.lo+np.zeros(n) for d in f.d],axis=1) for f in F],axis=1)
    j_hi=np.stack([np.stack([d.hi+np.zeros(n) for d in f.d],axis=1) for f in F],axis=1)
    arithmetic=np.isfinite(f_lo).all(axis=1)&np.isfinite(f_hi).all(axis=1)&np.isfinite(j_lo).all(axis=(1,2))&np.isfinite(j_hi).all(axis=(1,2))
    reason[~arithmetic]='nonfinite_interval_arithmetic'
    finite &= arithmetic
    physical_bounds={'brown_no_fold_lower':brown_lower,
        'depth_lower':Y[2].v.lo,'ray_radius_upper':radius_box.hi,
        'head_cross_norm_lower':hn.v.lo,
        'query_denominator_abs_lower':np.where(denominator.v.lo>0,denominator.v.lo,-denominator.v.hi)}
    return f_lo,f_hi,j_lo,j_hi,finite,reason,physical_bounds


def certify_profile_roots(scene,camera,material,branch,station,frame_time,curves,
                          alpha,root_rows,*,numerical_valid=None,
                          radii=((1e-7,1e-5,1e-3),(1e-6,1e-4,1e-2),(1e-5,1e-3,.1))):
    """Independently certify local roots; return complete boxes and failure ledger.

    Inputs use the same row order as frozen ``curves['source_rows']``.  Root
    stations/alpha/rows are only centers.  Evidence is recomputed from physical
    blocks and the original rational query.  All three finite domains and the
    complete interval Jacobian are enforced.  No state is mutated.
    """
    material=np.asarray(material);branch=np.asarray(branch);frame_time=np.asarray(frame_time,float)
    center=np.c_[station,alpha,root_rows].astype(float);n=len(center)
    if center.shape!=(n,3) or np.asarray(curves['bounds']).shape!=(n,2):
        raise ValueError('full three-variable profile centers and frozen alpha bounds required')
    eligible=np.isfinite(center).all(axis=1)
    if numerical_valid is not None:eligible &= np.asarray(numerical_valid,bool)
    certified=np.zeros(n,bool);reason=np.where(eligible,'krawczyk_inclusion_unresolved','no_qualified_numerical_root').astype('U80')
    boxes_lo=np.full((n,3),np.nan);boxes_hi=boxes_lo.copy();Klo=boxes_lo.copy();Khi=boxes_lo.copy()
    contractions=np.full(n,np.inf);margins=np.full(n,-np.inf);attempts=np.zeros(n,int)
    jaclo=np.full((n,3,3),np.nan);jachi=jaclo.copy()
    physical_boxes={k:np.full(n,np.nan) for k in ['brown_no_fold_lower','depth_lower',
        'ray_radius_upper','head_cross_norm_lower','query_denominator_abs_lower']}
    for radius in radii:
        active=np.flatnonzero(eligible&~certified)
        if not len(active):break
        reason[active]='krawczyk_inclusion_unresolved'
        c=center[active];rad=np.broadcast_to(np.asarray(radius,float),(len(active),3));lo=_down(c-rad);hi=_up(c+rad)
        bounds=np.asarray(curves['bounds'])[active]
        finite_support=(lo[:,0]>=0)&(hi[:,0]<=1)&(lo[:,1]>=bounds[:,0])&(hi[:,1]<=bounds[:,1])&(lo[:,2]>=0)&(hi[:,2]<=1079)
        reason[active[~finite_support]]='local_box_crosses_finite_station_alpha_or_row_boundary'
        subcurves={k:np.asarray(curves[k])[active] for k in ['numerator_u','numerator_v','denominator','bounds']}
        with np.errstate(all='ignore'):
            _,_,jl,jh,good,why,physical=_profile_system(scene,camera,material[active],branch[active],frame_time[active],subcurves,lo,hi)
            fl,fh,_,_,good0,why0,_=_profile_system(scene,camera,material[active],branch[active],frame_time[active],subcurves,c,c)
        good &= good0&finite_support
        reason[active[finite_support&~good]]=np.where(good0,why,why0)[finite_support&~good]
        middle=jl/2+jh/2;C=np.zeros_like(middle)
        for j in np.flatnonzero(good):
            try:C[j]=np.linalg.inv(middle[j])
            except np.linalg.LinAlgError:good[j]=False;reason[active[j]]='interval_midpoint_jacobian_singular'
        # K(X)=x0-C F(x0)+(I-C J(X))(X-x0), all algebra outward.
        B=[[Interval(float(i==j)) for j in range(3)] for i in range(3)]
        base=[Interval(c[:,i]) for i in range(3)]
        for i in range(3):
            for k in range(3):base[i]=base[i]-Interval(C[:,i,k])*Interval(fl[:,k],fh[:,k])
            for j in range(3):
                for k in range(3):B[i][j]=B[i][j]-Interval(C[:,i,k])*Interval(jl[:,k,j],jh[:,k,j])
        K=[]
        for i in range(3):
            value=base[i]
            for j in range(3):value=value+B[i][j]*(Interval(lo[:,j],hi[:,j])-c[:,j])
            K.append(value)
        kl=np.stack([x.lo for x in K],axis=1);kh=np.stack([x.hi for x in K],axis=1)
        row_bounds=[]
        for b_row in B:
            total=Interval(np.zeros(len(active)))
            for b in b_row:total=total+Interval(b.absolute_upper())
            row_bounds.append(total.hi)
        contraction=np.max(np.stack(row_bounds,axis=1),axis=1)
        margin=np.min(np.minimum(kl-lo,hi-kh)/rad,axis=1)
        passed=good&(contraction<1)&np.isfinite(kl).all(axis=1)&np.isfinite(kh).all(axis=1)&(kl>lo).all(axis=1)&(kh<hi).all(axis=1)
        attempts[active]+=1;boxes_lo[active]=lo;boxes_hi[active]=hi;Klo[active]=kl;Khi[active]=kh;contractions[active]=contraction;margins[active]=margin;jaclo[active]=jl;jachi[active]=jh
        for k,value in physical.items():physical_boxes[k][active]=value
        certified[active[passed]]=True;reason[active[passed]]='certified_local_existence_unique_in_box_nonsingular_jacobian'
    return {'version':VERSION,'certified':certified,'reason':reason,'center':center,'box_lower':boxes_lo,'box_upper':boxes_hi,
        'krawczyk_lower':Klo,'krawczyk_upper':Khi,'jacobian_lower':jaclo,'jacobian_upper':jachi,
        'preconditioned_jacobian_contraction_upper':contractions,'relative_strict_inclusion_margin':margins,'attempts':attempts,
        **{'box_'+k:v for k,v in physical_boxes.items()},
        'scope':'same finite branch local box; no global query-interval uniqueness claim',
        'full_box_analytic_interval_jacobian':True,'sampled_or_finite_difference_jacobian_used':False,
        'whole_readout_RS_certificate_is_separate':True,
        'Brown_no_fold_gate_scope':'every certified row requires a strictly positive outward Jacobian eigenvalue lower bound on its entire local root box; whole frozen disk certification remains separate',
        'fixed_camera_constants_scope':'NativeCamera reconstructed fields are fixed binary64 constants; camera_from_event parameter-to-camera reconstruction is not interval-certified; cross-evaluator discrepancies are checked separately',
        'floating_arithmetic':'IEEE754 binary64 basic operations, nextafter outward; elementary series with explicit remainders'}
