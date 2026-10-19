"""Profile an unknown SO3 target orientation through its exact lever sphere.

The source bbox is untouched. It is a declared engineering approximation to
optical-centre support inside a visible housing, never a pivot observation.
All current Brown14 terms, including active sensor tilt, enter projection,
inverse rays, global interval bounds and the full private-camera Jacobian.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq,math
import cv2,numpy as np
from scipy.optimize import brentq
from projection import to_state,from_state
from types import SimpleNamespace
from z190_continuous_source_bundle import SCENE_UNIT_M
from rinkspace.calibration_companion.wide_angle_camera import _brown_map_jacobian,brown_central_inverse

SCHEMA='rinkspace.unknown-target-SO3-lever-sphere-visible-housing-support.v1'


class ProfileDomainError(ValueError):pass


class I:
    """Outward-rounded double closed interval for global value certificates."""
    __slots__=('lo','hi')
    def __init__(self,lo,hi=None):self.lo=float(lo);self.hi=float(lo if hi is None else hi)
    @staticmethod
    def bound(lo,hi):return I(math.nextafter(float(lo),-math.inf),math.nextafter(float(hi),math.inf))
    def __add__(self,other):
        o=other if isinstance(other,I)else I(other)
        return I.bound(self.lo+o.lo,self.hi+o.hi)
    __radd__=__add__
    def __neg__(self):return I(-self.hi,-self.lo)
    def __sub__(self,other):return self+-_interval(other)
    def __rsub__(self,other):return _interval(other)+-self
    def __mul__(self,other):
        o=_interval(other);v=(self.lo*o.lo,self.lo*o.hi,self.hi*o.lo,self.hi*o.hi)
        return I.bound(min(v),max(v))
    __rmul__=__mul__
    def __truediv__(self,other):
        o=_interval(other)
        if o.lo<=0<=o.hi:raise ProfileDomainError('Interval Brown denominator crosses zero')
        return self*I.bound(1./o.hi,1./o.lo)
    def square(self):
        lo=0. if self.lo<=0<=self.hi else min(self.lo*self.lo,self.hi*self.hi)
        return I.bound(lo,max(self.lo*self.lo,self.hi*self.hi))
    def maximum_abs(self):return max(abs(self.lo),abs(self.hi))


def _interval(v):return v if isinstance(v,I)else I(v)


class D:
    __slots__=('value','gradient')
    def __init__(self,value,gradient):self.value=_interval(value);self.gradient=tuple(map(_interval,gradient))
    def other(self,o):return o if isinstance(o,D)else D(o,[0.]*len(self.gradient))
    def __add__(self,o):
        o=self.other(o);return D(self.value+o.value,[a+b for a,b in zip(self.gradient,o.gradient)])
    __radd__=__add__
    def __neg__(self):return D(-self.value,[-a for a in self.gradient])
    def __sub__(self,o):return self+-self.other(o)
    def __rsub__(self,o):return self.other(o)+-self
    def __mul__(self,o):
        o=self.other(o);return D(self.value*o.value,[a*o.value+self.value*b for a,b in zip(self.gradient,o.gradient)])
    __rmul__=__mul__
    def __truediv__(self,o):
        o=self.other(o);return D(self.value/o.value,[(a*o.value-self.value*b)/(o.value*o.value)for a,b in zip(self.gradient,o.gradient)])
    def square(self):return D(self.value.square(),[2.*self.value*g for g in self.gradient])


def _trig(a,b,cosine=False):
    if b-a>=2*math.pi:return I(-1.,1.)
    f=math.cos if cosine else math.sin;values=[f(a),f(b)]
    shift=0. if cosine else math.pi/2
    for k in range(math.ceil((a-shift)/math.pi),math.floor((b-shift)/math.pi)+1):values.append(1. if k%2==0 else -1.)
    return I.bound(min(values),max(values))


def _brown(x,y,d):
    k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4=d
    xx=x.square();yy=y.square();r2=xx+yy;r4=r2.square();r6=r4*r2
    numerator=1.+k1*r2+k2*r4+k3*r6;denominator=1.+k4*r2+k5*r4+k6*r6
    radial=numerator/denominator;xy=x*y
    return (x*radial+2*p1*xy+p2*(r2+2*xx)+s1*r2+s2*r4,
        y*radial+p1*(r2+2*yy)+2*p2*xy+s3*r2+s4*r4,denominator)


def tilt_matrix(tau_x,tau_y):
    """OpenCV's complete sensor-tilt homography; independent cv2 parity tested."""
    cx,sx=math.cos(tau_x),math.sin(tau_x);cy,sy=math.cos(tau_y),math.sin(tau_y)
    rx=np.array([[1.,0.,0.],[0.,cx,sx],[0.,-sx,cx]])
    ry=np.array([[cy,0.,-sy],[0.,1.,0.],[sy,0.,cy]])
    rotation=ry@rx
    project=np.array([[rotation[2,2],0.,-rotation[0,2]],
        [0.,rotation[2,2],-rotation[1,2]],[0.,0.,1.]])
    return project@rotation


def _tilt(x,y,matrix):
    h=matrix[2,0]*x+matrix[2,1]*y+matrix[2,2]
    return ((matrix[0,0]*x+matrix[0,1]*y+matrix[0,2])/h,
        (matrix[1,0]*x+matrix[1,1]*y+matrix[1,2])/h,h)


def camera25_from_state25(q,template):
    q=np.asarray(q,float)
    if q.shape!=(25,)or not np.isfinite(q).all():raise ProfileDomainError('Conditioned11 plus full raw Brown14 required')
    c=from_state(q[:11],template);c[11:]=q[11:];return c


def current_full_Brown14_world_rays(camera,pixels):
    """Undo actual sensor tilt before original axis-connected Brown inverse."""
    c=np.asarray(camera,float);pixels=np.asarray(pixels,float).reshape(-1,2)
    fx,fy=np.exp(c[6:8]);yn=(pixels[:,1]-c[9])/fy
    xn=(pixels[:,0]-c[8]-c[10]*yn)/fx
    matrix=tilt_matrix(*c[23:25])
    if abs(np.linalg.det(matrix))<1e-12:raise ProfileDomainError('Actual sensor tilt homography is singular')
    h=np.c_[xn,yn,np.ones(len(xn))]@np.linalg.inv(matrix).T
    valid=np.isfinite(h).all(axis=1)&(h[:,2]>1e-12)
    dist=h[:,:2]/h[:,2,None]
    inverse=brown_central_inverse(SimpleNamespace(K_px=np.eye(3),distortion_opencv=c[11:23]),dist,tolerance_px=1e-12)
    rays=np.c_[inverse['normalized_xy'],np.ones(len(pixels))]@cv2.Rodrigues(c[3:6])[0]
    return rays,valid&np.asarray(inverse['valid'],bool)


def _raw_box_r(uv,box):return np.asarray(uv)-np.clip(uv,box[:2],box[2:])
def _loss(r,sigma):
    a=np.abs(np.asarray(r)/sigma);return float(np.where(a<=2.,.5*a*a,2.*a-2.).sum())
def _interval_raw(v,left,right):
    def r(x):return x-left if x<left else x-right if x>right else 0.
    return I.bound(r(v.lo),r(v.hi))
def _psi_interval(raw,sigma):
    # Both endpoints must be clipped on both sides: an interval wholly in
    # the exterior linear Huber branch has a constant, ordered derivative.
    return I.bound(min(2.,max(-2.,raw.lo/sigma))/sigma,
        min(2.,max(-2.,raw.hi/sigma))/sigma)


def _interval_loss(raw,sigma):
    total=I(0.)
    for r in raw:
        normalized=r/sigma
        lo=0. if normalized.lo<=0<=normalized.hi else min(abs(normalized.lo),abs(normalized.hi))
        hi=max(abs(normalized.lo),abs(normalized.hi))
        def endpoint(a):
            value=I(a)
            return .5*value.square()if a<=2. else 2.*value-2.
        total=total+I(endpoint(lo).lo,endpoint(hi).hi)
    return total


def _axis_domain_certificate(camera,nx,ny):
    """Certify the current Brown/tilt chart on a whole axis-to-target region."""
    matrix=tilt_matrix(*camera[23:25])
    pending=[(0.,1.)];tested=0;min_margin=math.inf;min_den=math.inf
    while pending:
        lo,hi=pending.pop();scale=I(lo,hi)
        x=D(nx*scale,(1.,0.));y=D(ny*scale,(0.,1.))
        try:
            xx,yy,den=_brown(x,y,camera[11:23])
            _,_,tiltden=_tilt(xx,yy,matrix)
        except ProfileDomainError:
            if hi-lo<=1e-8:raise ProfileDomainError('Cannot certify current axis-connected Brown denominator')
            mid=.5*(lo+hi);pending.extend(((lo,mid),(mid,hi)));continue
        off=(xx.gradient[1]+yy.gradient[0])*.5
        diagonal_x,diagonal_y=xx.gradient[0],yy.gradient[1]
        determinant=diagonal_x.lo*diagonal_y.lo-off.maximum_abs()**2
        margin=determinant/(diagonal_x.hi+diagonal_y.hi)if diagonal_x.lo>0 and diagonal_y.lo>0 else -math.inf
        tested+=1
        if den.value.lo>1e-8 and margin>1e-8 and tiltden.value.lo>1e-8:
            min_margin=min(min_margin,margin);min_den=min(min_den,den.value.lo);continue
        if hi-lo<=1e-8:
            raise ProfileDomainError('Cannot certify current Brown central chart and positive sensor-tilt denominator')
        mid=.5*(lo+hi);pending.extend(((lo,mid),(mid,hi)))
    return dict(axis_connected_continuation_certified=True,interval_cells=tested,
        minimum_symmetric_J_eigenvalue_lower_bound=min_margin,minimum_denominator_lower_bound=min_den,
        all14_coefficients_retained=True,actual_sensor_tilt_active=True,
        actual_tilt_homography_interval_denominator_certified=True)


class SphereImage:
    """Exact normalized perspective silhouette, then the original Brown map."""
    def __init__(self,camera,pivot_m,lever_m,*,certify_domain=True):
        c=np.asarray(camera,float);p=np.asarray(pivot_m,float);l=np.asarray(lever_m,float)
        if c.shape!=(25,)or p.shape!=(3,)or l.shape!=(3,)or not all(np.isfinite(a).all()for a in(c,p,l)):
            raise ProfileDomainError('Finite full camera25 and metre pivot/lever3 required')
        self.camera=c.copy();self.pivot_m=p.copy();self.lever_m=l.copy()
        self.tilt=tilt_matrix(*c[23:25])
        if abs(np.linalg.det(self.tilt))<1e-12:raise ProfileDomainError('Actual tilt is singular, not a fixed parameter')
        self.radius_m=float(np.linalg.norm(l));rho=self.radius_m/SCENE_UNIT_M
        if self.radius_m<=1e-12:raise ProfileDomainError('Original installation requires finite nonzero lever')
        self.rotation=cv2.Rodrigues(c[3:6])[0];self.v=self.rotation@(p/SCENE_UNIT_M-c[:3])
        vz=self.v[2];s=float(self.v@self.v-rho*rho)
        if vz-rho<=.05 or s<=0:raise ProfileDomainError('Reachable optical-centre sphere must remain on original positive-depth branch')
        vz2=vz*vz-rho*rho
        a=s*np.eye(2)-np.outer(self.v[:2],self.v[:2])
        self.centre=vz*self.v[:2]/vz2
        k=rho*rho*s/vz2
        self.ellipse=math.sqrt(k)*np.linalg.inv(np.linalg.cholesky(a).T)
        self.depth_minimum_scene=float(vz-rho)
        self.domain_certificate=self._certify_domain()if certify_domain else None

    def _certify_domain(self):
        extent=np.linalg.norm(self.ellipse,axis=1)
        nx=I.bound(self.centre[0]-extent[0],self.centre[0]+extent[0])
        ny=I.bound(self.centre[1]-extent[1],self.centre[1]+extent[1])
        # Certify the full cone disk and its axis-connected continuation,
        # stronger than the original nine radial-station source check.
        return dict(_axis_domain_certificate(self.camera,nx,ny),full_cone_disk_and_axis_path_certified=True,
            stronger_than_original_nine_radial_stations=True,
            original_depth_minimum_scene=self.depth_minimum_scene)

    def point(self,theta):
        t=np.asarray(theta);xy=self.centre+np.stack((np.cos(t),np.sin(t)),axis=-1)@self.ellipse.T
        flat=xy.reshape(-1,2)
        fx,fy=np.exp(self.camera[6:8]);k=np.array([[fx,0.,self.camera[8]],[0.,fy,self.camera[9]],[0.,0.,1.]])
        uv,raw=cv2.projectPoints(np.c_[flat,np.ones(len(flat))],np.zeros(3),np.zeros(3),k,self.camera[11:25])
        uv=uv.reshape(-1,2);uv[:,0]+=self.camera[10]/fy*(uv[:,1]-self.camera[9])
        jxy=raw[:,3:5].reshape(len(flat),2,2);jxy[:,0]+=self.camera[10]/fy*jxy[:,1]
        dt=np.stack((-np.sin(t),np.cos(t)),axis=-1)@self.ellipse.T
        dt=dt.reshape(-1,2)
        uv_dt=np.einsum('nij,nj->ni',jxy,dt)
        return uv.reshape(t.shape+(2,)),uv_dt.reshape(t.shape+(2,))

    def interval(self,lo,hi):
        cs,sn=_trig(lo,hi,True),_trig(lo,hi)
        x=D(self.centre[0]+self.ellipse[0,0]*cs+self.ellipse[0,1]*sn,
            (-self.ellipse[0,0]*sn+self.ellipse[0,1]*cs,))
        y=D(self.centre[1]+self.ellipse[1,0]*cs+self.ellipse[1,1]*sn,
            (-self.ellipse[1,0]*sn+self.ellipse[1,1]*cs,))
        bx,by,_=_brown(x,y,self.camera[11:23]);bx,by,_=_tilt(bx,by,self.tilt);fx,fy=np.exp(self.camera[6:8])
        return fx*bx+self.camera[10]*by+self.camera[8],fy*by+self.camera[9]

    def sphere_world_point(self,theta):
        n=self.centre+self.ellipse@np.array([math.cos(theta),math.sin(theta)])
        ray=np.r_[n,1.];lam=float(self.v@ray/(ray@ray))
        point=self.camera[:3]+self.rotation.T@(lam*ray)
        return point*SCENE_UNIT_M


def _interior_witness(image,box):
    centre=.5*(box[:2]+box[2:]);pixels=np.vstack((centre,box[:2],box[2:],
        [box[0],box[3]],[box[2],box[1]],[(box[0]+box[2])/2,box[1]],
        [(box[0]+box[2])/2,box[3]],[box[0],(box[1]+box[3])/2],[box[2],(box[1]+box[3])/2]))
    rays,valid=current_full_Brown14_world_rays(image.camera,pixels)
    for uv,ray,good in zip(pixels,rays,valid):
        if not good:continue
        ray=ray/np.linalg.norm(ray);v=image.pivot_m/SCENE_UNIT_M-image.camera[:3]
        t=float(v@ray);d2=float(v@v-t*t);rho=image.radius_m/SCENE_UNIT_M
        if t>0 and d2<rho*rho:
            hit=image.camera[:3]+(t-math.sqrt(max(0.,rho*rho-d2)))*ray
            cam=image.rotation@(hit-image.camera[:3]);n=cam[:2]/cam[2]
            mapped=_brown_map_jacobian(n[None,:],image.camera[11:23])[0][0]
            h=image.tilt@np.r_[mapped,1.];mapped=h[:2]/h[2]
            fx,fy=np.exp(image.camera[6:8]);actual=np.array([fx*mapped[0]+image.camera[10]*mapped[1]+image.camera[8],fy*mapped[1]+image.camera[9]])
            if np.all(actual>=box[:2])and np.all(actual<=box[2:]):
                try:domain=_axis_domain_certificate(image.camera,I(n[0]),I(n[1]))
                except ProfileDomainError:continue
                return dict(projected_optical_centre_uv=actual.tolist(),optical_centre_world_m=(hit*SCENE_UNIT_M).tolist(),
                    original_source_ray_uv=uv.tolist(),ray_distance_to_pivot_m=math.sqrt(max(0.,d2))*SCENE_UNIT_M,
                    exact_source_bbox_satisfied=True,current_full_Brown14_witness_axis_domain_certificate=domain,
                    actual_reachable_lever_radius_error_m=float(abs(np.linalg.norm(hit*SCENE_UNIT_M-image.pivot_m)-image.radius_m)))
    return None


def _profile_boundary(image,box,sigma):
    def value(theta):
        uv,dt=image.point(theta);r=_raw_box_r(uv,box);psi=np.clip(r/sigma,-2.,2.)/sigma
        return _loss(r,sigma),float(dt@psi),uv,r
    # Seeding only supplies upper bounds. Correctness comes from the complete
    # outward-rounded interval covering and global lower-bound certificate.
    grid=np.linspace(0.,2*math.pi,65);uv,dt=image.point(grid)
    raw=uv-np.clip(uv,box[:2],box[2:]);costs=np.array([_loss(a,sigma)for a in raw])
    slopes=np.einsum('ij,ij->i',dt,np.clip(raw/sigma,-2,2)/sigma)
    index=int(np.argmin(costs));best_theta=float(grid[index]);best=value(best_theta)[0]
    def improve(lo,hi):
        nonlocal best,best_theta
        mid=.5*(lo+hi)
        for t in (lo,mid,hi):
            val=value(t)[0]
            if val<best:best,best_theta=val,t
        fl,fh=value(lo)[1],value(hi)[1]
        if fl<0<fh:
            t=brentq(lambda x:value(x)[1],lo,hi,xtol=5e-324,rtol=4*np.finfo(float).eps)
            val=value(t)[0]
            if val<best:best,best_theta=val,t
    for lo,hi,fl,fh in zip(grid[:-1],grid[1:],slopes[:-1],slopes[1:]):
        if fl<=0<=fh:improve(float(lo),float(hi))
    if best==0.:
        return best_theta,dict(global_cost_lower_bound=0.,global_cost_upper_bound=0.,
            global_bound_gap=0.,certificate='exact feasible silhouette point in original bbox',interval_cells=0)
    def lower(lo,hi):
        try:x,y=image.interval(lo,hi)
        except ProfileDomainError:return 0.
        rr=(_interval_raw(x.value,box[0],box[2]),_interval_raw(y.value,box[1],box[3]))
        derivative=x.gradient[0]*_psi_interval(rr[0],sigma)+y.gradient[0]*_psi_interval(rr[1],sigma)
        mid=.5*(lo+hi);px,py=image.interval(mid,mid)
        point_cost=_interval_loss((_interval_raw(px.value,box[0],box[2]),
            _interval_raw(py.value,box[1],box[3])),sigma)
        # Mean-value bound preserves cancellation at a stationary point;
        # independent UV boxes alone would converge only linearly there. Both
        # the point objective and derivative/distance bound are outward-rounded
        # using this same complete Brown expression, not a cv2 error estimate.
        radius=max((I(mid)-I(lo)).hi,(I(hi)-I(mid)).hi)
        bound=point_cost-I(0.,derivative.maximum_abs())*radius
        return max(0.,bound.lo)
    def point_upper(theta):
        px,py=image.interval(theta,theta)
        return _interval_loss((_interval_raw(px.value,box[0],box[2]),
            _interval_raw(py.value,box[1],box[3])),sigma).hi
    heap=[];cells=0;serial=0
    for lo,hi in zip(grid[:-1],grid[1:]):
        heapq.heappush(heap,(lower(float(lo),float(hi)),serial,float(lo),float(hi)));serial+=1
    best_upper=point_upper(best_theta)
    tolerance=1e-10*max(1.,best_upper)
    while heap and best_upper-heap[0][0]>tolerance:
        lb,_,lo,hi=heapq.heappop(heap)
        if lb>best_upper:continue
        mid=.5*(lo+hi)
        if mid in(lo,hi):raise ProfileDomainError('Global optical-sphere profile exhausted double angle precision')
        improve(lo,hi)
        best_upper=point_upper(best_theta)
        for a,b in ((lo,mid),(mid,hi)):
            bound=lower(a,b);cells+=1
            if bound<=best_upper:
                heapq.heappush(heap,(bound,serial,a,b));serial+=1
        if best==0.:break
    lower_bound=0. if best==0. else min(best_upper,heap[0][0]if heap else best_upper)
    return best_theta,dict(global_cost_lower_bound=lower_bound,global_cost_upper_bound=best_upper,
        global_bound_gap=best_upper-lower_bound,absolute_cost_certificate_tolerance=tolerance,
        certificate='full periodic silhouette interval branch-and-bound with mean-value derivative lower bounds',
        interval_cells=cells,all_angle_intervals_covered=True,seeding_does_not_define_global_minimum=True,
        midpoint_and_candidate_cost_use_same_outward_interval_Brown14=True,
        heuristic_floating_point_safety_margin_used=False)


@dataclass
class LeverSupportResult:
    residual_native:np.ndarray
    pivot_xyz_m_jacobian:np.ndarray|None
    lever_xyz_m_jacobian:np.ndarray|None
    camera_state25_jacobian:np.ndarray|None
    report:dict
    @property
    def camera_state11_jacobian(self):
        return None if self.camera_state25_jacobian is None else self.camera_state25_jacobian[:,:11]


def evaluate_secondary_entity_lever_support(camera25,pivot_xyz_m,lever_xyz_m,support,*,sigma_px=4.,jacobian=True):
    """Raw px2/J wrt metre P3, metre lever3, conditioned11+rawBrown14."""
    sigma=float(sigma_px);box=np.asarray(support['native_support_box_ltrb'],float)
    if sigma!=4. or box.shape!=(4,)or np.any(box[:2]>=box[2:]):
        raise ValueError('Untouched source bbox and explicitly declared original engineering sigma4px required')
    image=SphereImage(camera25,pivot_xyz_m,lever_xyz_m,certify_domain=False)
    witness=_interior_witness(image,box)
    report=dict(schema=SCHEMA,source_bbox_native_ltrb=box.tolist(),engineering_sigma_px=sigma,
        source_bbox_and_original_user_identity_unchanged=True,
        target_orientation='unknown SO3 at this original wide source exposure; not frozen to a programme exposure',
        exact_optical_centre_or_pivot_observed=False,
        working_approximation='actual reachable optical centre may lie within identified visible-housing support',
        lever_radius_m=image.radius_m,current_lever_xyz_m=image.lever_m.tolist(),
        current_pivot_xyz_m=image.pivot_m.tolist(),central_domain_certificate=image.domain_certificate,
        full_current_Brown14_axis_connected_domain_certified=False,
        Brown_tauX_tauY_are_active_not_fixed=True,
        unknown_rotation_profile_adds_no_measurement_or_physical_parameter=True,
        two_dimensional_bbox_expansion_used=False)
    if witness is not None:
        report.update(branch='reachable_interior_zero',witness=witness,global_profile_cost=0.,
            exact_zero_cost_feasibility_certified=True,orientation_nonunique=True,
            witness_current_full_Brown14_central_inverse_valid=True,
            Jacobian_policy='zero residual/cost on open feasible region; no unique orientation asserted')
        zero=np.zeros((2,3))if jacobian else None
        return LeverSupportResult(np.zeros(2),zero,zero.copy()if jacobian else None,np.zeros((2,25))if jacobian else None,report)
    image.domain_certificate=image._certify_domain()
    report['central_domain_certificate']=image.domain_certificate
    report['full_current_Brown14_axis_connected_domain_certified']=True
    theta,certificate=_profile_boundary(image,box,sigma)
    uv,uv_t=image.point(theta);r=_raw_box_r(uv,box)
    report.update(branch='profiled_silhouette',profile_angle_rad=float(theta),projected_optical_centre_uv=uv.tolist(),
        optical_centre_world_m=image.sphere_world_point(theta).tolist(),global_profile_cost=_loss(r,sigma),
        global_certificate=certificate,raw_residual_native_px=r.tolist(),orientation_nonunique=bool(np.all(r==0.)))
    if not jacobian:return LeverSupportResult(r,None,None,None,report)
    if np.all(r==0.):return LeverSupportResult(r,np.zeros((2,3)),np.zeros((2,3)),np.zeros((2,25)),report)
    active=(r!=0.).astype(float);b=active*uv_t;psi=np.clip(r/sigma,-2.,2.)/sigma
    k=active*((np.abs(r)<2*sigma).astype(float))/(sigma*sigma)
    def slope(t):
        u,dt=image.point(t);rr=_raw_box_r(u,box);return float(dt@(np.clip(rr/sigma,-2.,2.)/sigma))
    hphi=1e-5;den=(slope(theta+hphi)-slope(theta-hphi))/(2*hphi)
    if not np.isfinite(den)or den<=1e-10:
        raise ProfileDomainError('Nonunique or flat exterior sphere profile has no unique raw Jacobian')
    qcamera=np.r_[to_state(image.camera),image.camera[11:25]];fixed=[];tgradient=[]
    for kind,count,h in (('pivot',3,1e-6),('lever',3,1e-6),('camera',25,1e-7)):
        for column in range(count):
            pp,pm=image.pivot_m.copy(),image.pivot_m.copy();lp,lm=image.lever_m.copy(),image.lever_m.copy();cp,cm=qcamera.copy(),qcamera.copy()
            if kind=='pivot':pp[column]+=h;pm[column]-=h
            elif kind=='lever':lp[column]+=h;lm[column]-=h
            else:cp[column]+=h;cm[column]-=h
            plus=SphereImage(camera25_from_state25(cp,image.camera),pp,lp,certify_domain=False)
            minus=SphereImage(camera25_from_state25(cm,image.camera),pm,lm,certify_domain=False)
            up,bp=plus.point(theta);um,bm=minus.point(theta)
            a=active*(up-um)/(2*h);bq=active*(bp-bm)/(2*h)
            dt=-(float(bq@psi)+float(b@(k*a)))/den
            fixed.append(a+b*dt);tgradient.append(dt)
    matrix=np.asarray(fixed).T
    report.update(profile_stationary_value=slope(theta),profile_stationary_derivative=den,
        raw_residual_J_policy='fixed-angle exact sphere geometry/camera FD plus implicit actual Huber stationary-angle derivative',
        profiled_angle_parameter_jacobian=tgradient,
        exact_support_boundary_nondifferentiable=bool(np.any((uv==box[:2])|(uv==box[2:]))),
        exact_Huber_boundary_nondifferentiable=bool(np.any(np.abs(r)==2*sigma)))
    return LeverSupportResult(r,matrix[:,:3],matrix[:,3:6],matrix[:,6:],report)
