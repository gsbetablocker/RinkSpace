"""Independent NumPy fixed-head/cubic/Brown evaluator and conservative RS bounds.

This module does not import the Ceres forward evaluator or legacy owner.  Units
are seconds, radians, native pixels, and L=1.  Missing/uncertified domains stay
explicit failures.  All velocity certificates concern the entire row interval,
not a sampled list of roots.
"""
from __future__ import annotations
import numpy as np

VERSION = "native_numpy_fixed_head_v1"


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    angle = np.asarray(angle, dtype=float)
    x,y,z = axis
    K = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return (np.eye(3) + np.sin(angle)[...,None,None]*K
            + (1.-np.cos(angle))[...,None,None]*(K@K))


def expanded_controls(camera):
    b = camera["camera_controls"]
    if "controls" in b:
        full = np.asarray(b["controls"], dtype=float).copy()
    else:
        n = int(b["basis"]["n_controls"])
        dep = int(b["angle_dependent_index"])
        full = np.empty((n,3))
        inds = np.delete(np.arange(n),dep)
        full[inds,0] = b["alpha_free_controls"]
        full[inds,1] = b["beta_free_controls"]
        full[:,2] = b["log_f_controls"]
        first = b["basis"]["first_control_time"]; dt=b["basis"]["dt"]
        x=(b.get("reference_time",0.)-first)/dt; i=int(np.floor(x)); u=x-i
        w=np.array([(1-u)**3,3*u**3-6*u*u+4,-3*u**3+3*u*u+3*u+1,u**3])/6
        ref=np.zeros(n);ref[i-1:i+3]=w
        full[dep,:2]=-(ref[inds,None]*full[inds,:2]).sum(axis=0)/ref[dep]
    if full.ndim!=2 or full.shape[1]!=3 or not np.isfinite(full).all():
        raise ValueError("invalid complete camera controls")
    return full


class NativeCamera:
    def __init__(self, camera):
        self.source=camera
        self.head=camera["shared_head"]; self.optics=camera["optics"]
        self.controls=expanded_controls(camera)
        basis=camera["camera_controls"]["basis"]
        self.first=float(basis["first_control_time"]); self.dt=float(basis["dt"])
        self.domain=(self.first+self.dt,self.first+(len(self.controls)-2)*self.dt)
        self.R0=np.asarray(self.head["R0"],float);self.n=np.asarray(self.head["n"],float);self.b=np.asarray(self.head["b"],float)
        self.P=np.asarray(self.head["P"],float);self.d=np.asarray(self.head["d"],float);self.a=float(self.head["a"])
        self.tau=float(camera["rolling_shutter"]["signed_readout_seconds"])
        self.gauge_error=max(abs(np.linalg.norm(self.n)-1),abs(np.linalg.norm(self.b)-1),
          abs(self.b@self.R0@self.n),float(np.max(np.abs(self.R0@self.R0.T-np.eye(3)))),abs(np.linalg.det(self.R0)-1))
        if self.dt<=0 or self.gauge_error>1e-8: raise ValueError("fixed-head SO3/orthogonal-axis domain")
        self.reference_error=float(np.max(np.abs(self.spline(np.array([0.]))[0,:2])))
        if self.reference_error>1e-8: raise ValueError("angle function-value gauge failed")

    def spline(self,time,derivative=0):
        t=np.asarray(time,float); x=(t-self.first)/self.dt
        if np.any(t<self.domain[0]-1e-12) or np.any(t>self.domain[1]+1e-12):
            raise ValueError("time outside allocated cubic domain; no clamp")
        i=np.floor(x).astype(int);i=np.where(i==len(self.controls)-2,i-1,i);u=x-i
        if np.any(i<1) or np.any(i>len(self.controls)-3): raise ValueError("illegal cubic support")
        if derivative==0:
            w=np.stack(((1-u)**3,3*u**3-6*u*u+4,-3*u**3+3*u*u+3*u+1,u**3),axis=-1)/6
        elif derivative==1:
            w=np.stack((-3*(1-u)**2,9*u*u-12*u,-9*u*u+6*u+3,3*u*u),axis=-1)/(6*self.dt)
        else: raise ValueError("derivative must be 0 or 1")
        return np.einsum("...j,...jk->...k",w,self.controls[i[...,None]+np.arange(-1,3)])

    def at(self,world,time):
        X=np.asarray(world,float);t=np.asarray(time,float)
        shape=np.broadcast_shapes(X.shape[:-1],t.shape)
        X=np.broadcast_to(X,shape+(3,));t=np.broadcast_to(t,shape)
        abf=self.spline(t);alpha,beta,logf=np.moveaxis(abf,-1,0)
        z=logf-np.log(float(self.optics["f_ref"]))
        R=rotation(self.b,beta)@self.R0@rotation(self.n,alpha)
        w_b=np.einsum("...ji,j->...i",R,self.b)
        h=np.cross(self.n,w_b);hn=np.linalg.norm(h,axis=-1)
        h=h/hn[...,None]
        ell=float(self.head.get("pupil_linear",0.))*z+float(self.head.get("pupil_quadratic",0.))*z*z
        D=np.broadcast_to(self.d,X.shape).copy();D[...,2]+=ell
        C=self.P+np.einsum("...ji,...j->...i",R,D)+self.a*h
        Y=np.einsum("...ij,...j->...i",R,X-C);depth=Y[...,2]
        valid=(depth>0)&np.isfinite(Y).all(axis=-1)&(hn>1e-10)
        with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
            xy=Y[...,:2]/depth[...,None];xx,yy=np.moveaxis(xy,-1,0);r2=xx*xx+yy*yy
            k1=np.polynomial.polynomial.polyval(z,self.optics["k1"])
            k2=np.polynomial.polynomial.polyval(z,self.optics["k2"])
            p1=float(self.optics["p1"]);p2=float(self.optics["p2"])
            radial=1+k1*r2+k2*r2*r2
            xd=xx*radial+2*p1*xx*yy+p2*(r2+2*xx*xx)
            yd=yy*radial+p1*(r2+2*yy*yy)+2*p2*xx*yy
            f=np.exp(logf);eta=float(self.optics.get("eta",0.))
            pp=np.asarray(self.optics["principal_point"])+z[...,None]*np.asarray(self.optics.get("principal_linear",[0.,0.]))
            uv=pp+np.stack((f*np.exp(eta/2)*xd,f*np.exp(-eta/2)*yd),axis=-1)
        radius_max=float(self.optics.get("radius_max",np.inf))
        valid &= np.isfinite(uv).all(axis=-1)&(r2<=radius_max*radius_max)
        return {"uv":uv,"valid":valid,"depth":depth,"camera_xyz":Y,"logf":logf,"z":z,"ray_radius":np.sqrt(r2)}

    def optical_domain_certificate(self):
        """Sufficient no-fold bound for the entire frozen disk and cubic zoom hull.

        Cubic value weights are nonnegative and sum to one. Linear-in-zoom
        Brown coefficients attain the radial Jacobian minima at a hull end;
        the disk radial quadratic is minimized at endpoints or its vertex.
        Tangential distortion subtracts its symmetric Jacobian norm bound.
        Floating arithmetic carries an explicit conservative padding.
        """
        low,high=float(self.controls[:,2].min()),float(self.controls[:,2].max())
        z=np.array([low,high])-np.log(float(self.optics['f_ref']))
        k1=np.asarray(self.optics['k1']);k2=np.asarray(self.optics['k2'])
        if len(k1)>2 or len(k2)>2:return {'passed':False,'reason':'higher_degree_zoom_polynomial_not_enclosed'}
        radius=float(self.optics['radius_max']);q=radius*radius
        if not (np.isfinite(radius) and radius>0):return {'passed':False,'reason':'unbounded_or_invalid_Brown_disk'}
        a=np.polynomial.polynomial.polyval(z,k1);b=np.polynomial.polynomial.polyval(z,k2)
        def lower(c1,c2):
            out=np.minimum(1.,1+c1*q+c2*q*q)
            vertex=np.divide(-c1,2*c2,out=np.zeros_like(c1),where=c2!=0)
            return np.minimum(out,np.where((c2>0)&(vertex>0)&(vertex<q),1+c1*vertex+c2*vertex*vertex,np.inf))
        tangential=6*np.hypot(float(self.optics['p1']),float(self.optics['p2']))*radius
        scale=1+np.max(abs(a)*q+abs(b)*q*q)+tangential
        bound=float(min(lower(a,b).min(),lower(3*a,5*b).min())-tangential-1e-10*scale)
        with np.errstate(over='ignore'):focal=np.exp(np.array([low,high]))
        return {'passed':bool(bound>1e-10 and np.isfinite(focal).all()),
            'Brown_Jacobian_eigenvalue_lower':bound,'frozen_ray_radius':radius,
            'full_cubic_log_f_hull':[low,high],'focal_hull_native_px':focal.tolist(),
            'covers_entire_allocated_native_cubic_time_domain':True,
            'covers_entire_frozen_ray_disk':True,'floating_relative_padding':1e-10,
            'scope':'sufficient analytic convex-hull/quadratic bound; not a camera hardware specification'}

    def derivative_bounds(self,frame_time):
        """Cubic derivative convex-hull bound on each entire row-time interval."""
        t=np.asarray(frame_time,float);half=abs(self.tau)/2
        lo=t-half;hi=t+half
        if np.any(lo<self.domain[0]) or np.any(hi>self.domain[1]): raise ValueError("readout extends beyond cubic domain")
        begin=np.floor((lo-self.first)/self.dt).astype(int)-2
        end=np.floor((hi-self.first)/self.dt).astype(int)+2
        d=np.abs(np.diff(self.controls,axis=0)/self.dt)
        result=np.zeros(t.shape+(3,))
        for offset in range(int(np.max(end-begin))+1):
            j=begin+offset;legal=(j<=end)&(j>=0)&(j<len(d))
            result=np.maximum(result,np.where(legal[...,None],d[np.clip(j,0,len(d)-1)],0.))
        return np.nextafter(result*(1+1e-12)+1e-12,np.inf)

    def rs_certificate(self,world,frame_time):
        """Analytic conservative speed/no-fold bound over the full readout.

        Cubic derivative hulls, rigid-rotation norm bounds, and polynomial Brown
        Jacobian bounds provide a sufficient contraction certificate.  Failure
        is uncertified, never silently valid.  Floating padding is explicit.
        """
        X=np.asarray(world,float);t=np.asarray(frame_time,float);mid=self.at(X,t)
        d=self.derivative_bounds(t);omega=d[...,0]+d[...,1];zd=d[...,2]
        half=abs(self.tau)/2;zabs=np.abs(mid["z"])+zd*half
        pl=abs(float(self.head.get("pupil_linear",0.)));pq=abs(float(self.head.get("pupil_quadratic",0.)))
        dy=omega*(np.linalg.norm(X-self.P,axis=-1)+abs(self.a))+(pl+2*pq*zabs)*zd
        span=dy*half;Yabs=np.abs(mid["camera_xyz"])+span[...,None]
        # Already invalid depths stay invalid; use a harmless denominator only
        # inside the bound arithmetic, never in the actual projection.
        zmin=mid["depth"]-span;safe=np.where(zmin>0,zmin,1.)
        x=Yabs[...,0]/safe;y=Yabs[...,1]/safe;r2=x*x+y*y;r=np.sqrt(r2)
        xyprime=dy[...,None]*(safe[...,None]+Yabs[...,:2])/safe[...,None]**2
        k1a=np.abs(self.optics["k1"][0])+sum(abs(v)*zabs**j for j,v in enumerate(self.optics["k1"][1:],1))
        k2a=np.abs(self.optics["k2"][0])+sum(abs(v)*zabs**j for j,v in enumerate(self.optics["k2"][1:],1))
        k1p=sum(j*abs(v)*zabs**(j-1) for j,v in enumerate(self.optics["k1"][1:],1))
        k2p=sum(j*abs(v)*zabs**(j-1) for j,v in enumerate(self.optics["k2"][1:],1))
        p1=abs(float(self.optics["p1"]));p2=abs(float(self.optics["p2"]))
        L=1+k1a*r2+k2a*r2*r2
        Jyx=2*x*y*(k1a+2*k2a*r2)+2*p1*x+2*p2*y
        Jyy=L+2*y*y*(k1a+2*k2a*r2)+6*p1*y+2*p2*x
        ybrown=y*L+p1*(r2+2*y*y)+2*p2*x*y
        direct=y*(k1p*r2+k2p*r2*r2)*zd
        fymax=np.exp(mid["logf"]+zd*half-float(self.optics.get("eta",0.))/2)
        ppprime=abs(self.optics.get("principal_linear",[0.,0.])[1])*zd
        speed=fymax*(ybrown*zd+Jyx*xyprime[...,0]+Jyy*xyprime[...,1]+direct)+ppprime
        speed=np.nextafter(speed*(1+1e-10)+1e-8,np.inf)
        contraction=abs(self.tau)*speed/1079.
        # Conservative symmetric coefficient uncertainty around mid zoom.
        k1mid=np.polynomial.polynomial.polyval(mid["z"],self.optics["k1"])
        k2mid=np.polynomial.polynomial.polyval(mid["z"],self.optics["k2"])
        k1lo=k1mid-k1p*zd*half;k2lo=k2mid-k2p*zd*half
        # If negative quadratic: its minimum is at an interval endpoint;
        # positive quadratic also checks its interior vertex.
        def polymin(c1,c2):
            value=np.minimum(1.,1+c1*r2+c2*r2*r2)
            vertex=np.divide(-c1,2*c2,out=np.zeros_like(c1),where=c2!=0)
            return np.minimum(value,np.where((c2>0)&(vertex>0)&(vertex<r2),1+c1*vertex+c2*vertex*vertex,np.inf))
        fold=np.minimum(polymin(k1lo,k2lo),polymin(3*k1lo,5*k2lo))-6*np.hypot(p1,p2)*r
        domain=(zmin>0)&(r<=float(self.optics.get("radius_max",np.inf)))&(fold>1e-10)
        certified=domain&(contraction<1.-1e-9)&np.isfinite(contraction)
        return {"certified":certified,"contraction_bound":contraction,"depth_lower":zmin,
                "brown_no_fold_lower":fold,"speed_y_bound_px_s":speed,"full_interval":True}

    def project_rs(self,world,frame_time,*,tolerance_px=1e-8,max_steps=48):
        X=np.asarray(world,float);t=np.asarray(frame_time,float)
        shape=np.broadcast_shapes(X.shape[:-1],t.shape)
        X=np.broadcast_to(X,shape+(3,));t=np.broadcast_to(t,shape)
        cert=self.rs_certificate(X,t)
        left=np.zeros(shape);right=np.full(shape,1079.)
        pl=self.at(X,t+self.tau*(left/1079-.5));pr=self.at(X,t+self.tau*(right/1079-.5))
        fl=left-pl["uv"][...,1];fr=right-pr["uv"][...,1]
        bracket=(fl<=0)&(fr>=0)&pl["valid"]&pr["valid"]
        # Bisection guarantees the returned root is independent of optimizer witnesses.
        for _ in range(max_steps):
            v=(left+right)/2;p=self.at(X,t+self.tau*(v/1079-.5));f=v-p["uv"][...,1]
            right=np.where(f>=0,v,right);left=np.where(f<0,v,left)
            if np.max(right-left,initial=0.)<=tolerance_px: break
        v=(left+right)/2;out=self.at(X,t+self.tau*(v/1079-.5));eq=v-out["uv"][...,1]
        valid=out["valid"]&bracket&cert["certified"]&(np.abs(eq)<=tolerance_px*2)
        out.update({"valid":valid,"row":v,"root_equality_px":eq,"bracketed":bracket,"certificate":cert})
        return out
