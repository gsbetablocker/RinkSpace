"""Native Brown-pixel static motion with moving optical centres.

Every original pixel row survives. Reviewed planes supply finite depth;
unknown rows have ONE positive inverse-depth nuisance, profiled per row with
frozen component weights. Returned Jacobians differentiate that profiled
residual implicitly, including residual curvature. They are not the old
fixed-centre ray Jacobians. No scene/source classification occurs here.

Unknown interior depths leave one independent two-view constraint. Two
native residual components remain visible and correlated; they cannot be
treated as two independent ray constraints. For an exact Gauss-Newton Schur
normal matrix, use the raw unprofiled Jacobians and rho tangent in state.
"""
from __future__ import annotations
import numpy as np


class MovingDepthDomainError(ValueError):
    def __init__(self, reason, indices):
        self.reason = reason
        self.indices = np.asarray(indices, int)
        super().__init__(f'{reason}: original batch rows {self.indices[:20].tolist()}')


def _inverse_brown(uv, focal, principal, k1):
    distorted=(uv-principal)/focal[:,None]
    n=distorted.copy()
    for _ in range(12):
        r2=np.sum(n*n,axis=1);s=1.+k1*r2
        a=s+2*k1*n[:,0]**2;b=2*k1*n[:,0]*n[:,1];c=s+2*k1*n[:,1]**2
        det=a*c-b*b
        if np.any(det<=1e-12):
            raise MovingDepthDomainError('noninvertible_source_Brown',np.flatnonzero(det<=1e-12))
        residual=n*s[:,None]-distorted
        step=np.c_[(c*residual[:,0]-b*residual[:,1])/det,
                   (a*residual[:,1]-b*residual[:,0])/det]
        n-=step
        if np.max(np.abs(step),initial=0.)<2e-13:break
    if np.max(np.abs(n*(1.+k1*np.sum(n*n,axis=1))[:,None]-distorted),initial=0.)>1e-9:
        raise MovingDepthDomainError('source_Brown_root_unconverged',np.arange(len(n)))
    r2=np.sum(n*n,axis=1);s=1.+k1*r2
    a=s+2*k1*n[:,0]**2;b=2*k1*n[:,0]*n[:,1];c=s+2*k1*n[:,1]**2;det=a*c-b*b
    # Explicit observed distorted normalized coordinates change by -d logf.
    dnlogf=-np.c_[(c*distorted[:,0]-b*distorted[:,1])/det,
                 (a*distorted[:,1]-b*distorted[:,0])/det]
    return n,dnlogf


def _project(Y, focal, principal, k1, tangent=None):
    z=Y[:,2]
    if np.any(z<=0) or not np.isfinite(Y).all():
        raise MovingDepthDomainError('target_cheirality',np.flatnonzero((z<=0)|~np.isfinite(Y).all(axis=1)))
    n=Y[:,:2]/z[:,None];r2=np.sum(n*n,axis=1);s=1.+k1*r2
    pixel=focal[:,None]*n*s[:,None]+principal
    D=np.zeros((len(Y),2,2));D[:,0,0]=s+2*k1*n[:,0]**2;D[:,1,1]=s+2*k1*n[:,1]**2
    D[:,0,1]=D[:,1,0]=2*k1*n[:,0]*n[:,1]
    A=np.zeros((len(Y),2,3));A[:,0,0]=1.;A[:,1,1]=1.;A[:,:,2]=-n
    A/=z[:,None,None]
    FY=focal[:,None,None]*np.einsum('nij,njk->nik',D,A)
    if tangent is None:return pixel,FY,n,D
    nt=(tangent[:,:2]-n*tangent[:,2,None])/z[:,None]
    ntt=-2.*tangent[:,2,None]*nt/z[:,None]
    second=(np.einsum('nij,nj->ni',D,ntt)+
            2*k1*(2.*np.sum(n*nt,axis=1)[:,None]*nt+np.sum(nt*nt,axis=1)[:,None]*n))
    jrho=np.einsum('nij,nj->ni',FY,tangent)
    return pixel,FY,n,D,jrho,focal[:,None]*second


def _mixed_rho(Y, n, D, tangent, Ytheta, ttheta, focal, k1):
    z=Y[:,2]
    nh=(Ytheta[:,:2,:]-n[:,:,None]*Ytheta[:,2,None,:])/z[:,None,None]
    nt=(tangent[:,:2]-n*tangent[:,2,None])/z[:,None]
    nht=-(Ytheta[:,2,None,:]*nt[:,:,None]+tangent[:,2,None,None]*nh)/z[:,None,None]
    ntparam=(ttheta[:,:2,:]-n[:,:,None]*ttheta[:,2,None,:])/z[:,None,None]
    nht+=ntparam
    second=2*k1*(np.sum(n[:,:,None]*nh,axis=1)[:,None,:]*nt[:,:,None]
                    +np.sum(n*nt,axis=1)[:,None,None]*nh
                    +np.sum(nh*nt[:,:,None],axis=1)[:,None,:]*n[:,:,None])
    return focal[:,None,None]*(np.einsum('nij,njk->nik',D,nht)+second)


def moving_center_motion_value_jacobians(
        a,b,fa,fb,Ra,Rb,dRa,dRb,Ca,Cb,dCa_s,dCb_s,dCa_v,dCb_v,*,
        principal_px,k1,sigma=None,profile_weight=None,finite_normals=None,
        finite_d=None,finite_mask=None,jac=True,return_state=False,
        profile_max_iterations=16,profile_tolerance=1e-8,
        minimum_source_depth_scene_units=0.):
    """Return native flat2N residual, Ja/Jb/Js (2N,4), optionally state.

    Rotations are world-to-camera; dR is (N,3parameter,3,3). Centre derivative
    arrays are (N,3,4); shared slots are caller-owned Pz/100 and lever/100.
    Plane equation is n dot X = finite_d. Native focal is square, principal
    point and k1 constant, other Brown terms zero. sigma is an original native
    observation scale, used only to express the frozen profile metric;
    returned residual/J are NOT sigma-normalized. profile_weight is frozen
    scalar-Huber IRLS component weights, never a source-admission predicate.
    jac=False returns residual; return_state adds state as final tuple member.
    """
    minimum_source_depth_scene_units=float(minimum_source_depth_scene_units)
    if minimum_source_depth_scene_units<0 or not np.isfinite(minimum_source_depth_scene_units):raise ValueError('Finite nonnegative conditional minimum source depth required')
    a=np.asarray(a,float);b=np.asarray(b,float);N=len(a)
    if a.shape!=(N,2) or b.shape!=(N,2):raise ValueError('Original a/b must be Nx2')
    if not N:
        result=(np.empty(0),)*(1 if not jac else 1)+((np.empty((0,4)),)*3 if jac else ())
        return (*result,{})if return_state else(result if jac else result[0])
    fa=np.broadcast_to(np.asarray(fa,float),(N,));fb=np.broadcast_to(np.asarray(fb,float),(N,))
    Ra=np.broadcast_to(np.asarray(Ra,float),(N,3,3));Rb=np.broadcast_to(np.asarray(Rb,float),(N,3,3))
    Ca=np.broadcast_to(np.asarray(Ca,float),(N,3));Cb=np.broadcast_to(np.asarray(Cb,float),(N,3))
    pp=np.asarray(principal_px,float);k1=float(k1)
    if pp.shape!=(2,) or np.any(fa<=0) or np.any(fb<=0):raise ValueError('Explicit fixed PP and positive focal required')
    sig=np.ones((N,2))if sigma is None else np.broadcast_to(np.asarray(sigma,float).reshape(-1,1)if np.asarray(sigma).ndim==1 else np.asarray(sigma,float),(N,2))
    weight=np.ones((N,2))if profile_weight is None else np.asarray(profile_weight,float).reshape(N,2)
    if np.any(sig<=0) or np.any(weight<0) or not np.isfinite(weight).all():raise ValueError('Positive original sigma/frozen nonnegative weights required')
    W=weight/sig**2
    na,dnaf=_inverse_brown(a,fa,pp,k1);nb,_=_inverse_brown(b,fb,pp,k1)
    local=np.c_[na,np.ones(N)]
    da=np.einsum('ni,nij->nj',local,Ra)
    q=np.einsum('nij,nj->ni',Rb,da)
    delta=Ca-Cb;t=np.einsum('nij,nj->ni',Rb,delta)
    known=np.zeros(N,bool)if finite_mask is None else np.asarray(finite_mask,bool)
    if known.shape!=(N,):raise ValueError('Finite mask must retain original batch row shape')
    v=q[:,:2]-nb*q[:,2,None];w=t[:,:2]-nb*t[:,2,None]
    denom=np.sum(w*w,axis=1)
    rho=-np.sum(v*w,axis=1)/np.maximum(denom,1e-30)
    # A compact coordinate includes both one-sided positive-depth limits.
    # Homogeneous Y=(1-tau)q+tau*t equals (q+rho*t)/(1+rho).
    # tau=0 is infinite source depth; tau=1 is zero source depth.
    tangent=t-q
    low=np.zeros(N);upper=np.ones(N)
    positive=tangent[:,2]>0;negative=tangent[:,2]<0
    low[positive]=np.maximum(0.,(1e-12-q[positive,2])/tangent[positive,2])
    upper[negative]=np.minimum(1.,(q[negative,2]-1e-12)/(-tangent[negative,2]))
    invalid=(upper<=low)|((q[:,2]<=0)&(t[:,2]<=0))
    if np.any(invalid):
        raise MovingDepthDomainError('no_positive_target_depth_interval',np.flatnonzero(invalid))
    rho=np.maximum(rho,0.)
    tau=rho/(1.+rho)
    tau=np.minimum(np.maximum(tau,low),upper)
    plane_n=np.zeros((N,3));plane_d=np.zeros(N);plane_distance=np.ones(N)
    if np.any(known):
        if finite_normals is None or finite_d is None:raise ValueError('Reviewed finite rows need plane n,d')
        plane_n=np.broadcast_to(np.asarray(finite_normals,float),(N,3));plane_d=np.broadcast_to(np.asarray(finite_d,float),(N,))
        plane_distance=plane_d-np.sum(plane_n*Ca,axis=1)
        rho[known]=np.sum(plane_n[known]*da[known],axis=1)/plane_distance[known]
        invalid=known&((rho<=0)|(q[:,2]+rho*t[:,2]<=0)|~np.isfinite(rho))
        if np.any(invalid):raise MovingDepthDomainError('reviewed_plane_positive_depth_invalid',np.flatnonzero(invalid))
        tau[known]=rho[known]/(1.+rho[known])
    unknown=~known
    if np.any(q[:,2]<=0):
        raise MovingDepthDomainError('source_ray_infinity_behind_target_line_chart',np.flatnonzero(q[:,2]<=0))
    nq=q[:,:2]/q[:,2,None]
    line_h=t[:,:2]-nq*t[:,2,None]
    line_length=np.linalg.norm(line_h,axis=1)
    identifiable=line_length>1e-14
    line_unit=line_h/np.maximum(line_length,1e-30)[:,None]
    lam=np.maximum(0.,np.sum((nb-nq)*line_unit,axis=1))
    lam_upper=np.full(N,np.inf)
    positive_tz=t[:,2]>0
    lam_upper[positive_tz]=line_length[positive_tz]/t[positive_tz,2]
    # Optional finite positive-depth closure, independent of material/plane.
    # Its line-chart bound contracts continuously to zero with the baseline.
    # It excludes the ill-conditioned rho->infinity near-centre branch;
    # callers must record the engineering bound, never claim measured depth.
    depth_floor_bound=np.zeros(N,bool)
    rho_max=None
    if minimum_source_depth_scene_units>0:
        rho_max=1./minimum_source_depth_scene_units
        floor_den=q[:,2]+rho_max*t[:,2]
        floor_lam=np.divide(rho_max*line_length,floor_den,out=np.full(N,np.inf),where=floor_den>0)
        depth_floor_bound=unknown&(floor_lam<=lam_upper)
        lam_upper=np.minimum(lam_upper,floor_lam)
    lam=np.minimum(lam,lam_upper)
    lam[~identifiable]=0.
    line_tangent=np.c_[line_unit,np.zeros(N)]
    iterations=0
    row_iterations=np.zeros(N,int);line_search_trials=np.zeros(N,int)
    line_search_failed=np.zeros(N,bool)
    for iterations in range(profile_max_iterations):
        Y=np.c_[nq+lam[:,None]*line_unit,np.ones(N)]
        pixel,FY,n,D,jrho,jrr=_project(Y,fb,pp,k1,line_tangent)
        residual=pixel-b;g=np.sum(W*residual*jrho,axis=1)
        GN=np.sum(W*jrho**2,axis=1);H=GN+np.sum(W*residual*jrr,axis=1)
        at_low=lam<=1e-14;at_upper=lam>=lam_upper-1e-14
        active=unknown&identifiable&(GN>1e-22)&~((at_low&(g>=0))|(at_upper&(g<=0)))
        stationary=np.abs(g)<=profile_tolerance*np.maximum(np.sqrt(GN),1e-12)
        active&=~stationary
        if not np.any(active):break
        row_iterations[active]+=1
        step=-g/np.maximum(np.where(H>0,H,GN),1e-30)
        candidate=np.minimum(np.maximum(lam+step,0.),lam_upper)
        basecost=np.sum(W*residual**2,axis=1)
        changed=np.zeros(N,bool)
        for _ in range(12):
            line_search_trials[active]+=1
            candidate[~active]=lam[~active]
            cp=_project(np.c_[nq+candidate[:,None]*line_unit,np.ones(N)],fb,pp,k1)[0]
            cost=np.sum(W*(cp-b)**2,axis=1)
            accept=active&(cost<=basecost+np.finfo(float).eps*8.*np.maximum(1.,basecost))
            lam[accept]=candidate[accept];changed|=accept
            active&=~accept
            if not np.any(active):break
            candidate[active]=(candidate[active]+lam[active])*.5
        line_search_failed|=active
        if not np.any(changed):break
    # In the line chart native Brown pixels are cubic in lambda, avoiding
    # nearly unobservable rho/tau scales when the camera baseline is tiny.
    denom_tau=line_length+lam*(q[:,2]-t[:,2])
    tau[unknown]=np.divide(lam*q[:,2],denom_tau,out=np.zeros(N),where=denom_tau>0)[unknown]
    tau=np.clip(tau,0.,1.)
    Y=np.c_[nq+lam[:,None]*line_unit,np.ones(N)]
    pixel,FY,n,D,jrho,jrr=_project(Y,fb,pp,k1,line_tangent)
    floor_endpoint=depth_floor_bound&(lam>=lam_upper-1e-14)
    if np.any(floor_endpoint):
        # Evaluate the finite endpoint directly. This avoids subtracting
        # nearly equal absolute C values then dividing by its line length.
        pixel[floor_endpoint]=_project(q[floor_endpoint]+rho_max*t[floor_endpoint],fb[floor_endpoint],pp,k1)[0]
    if np.any(known):
        finite_y=q[known]+tau[known,None]*tangent[known]
        pixel[known]=_project(finite_y,fb[known],pp,k1)[0]
    residual=pixel-b
    g=np.sum(W*residual*jrho,axis=1);GN=np.sum(W*jrho**2,axis=1)
    H=GN+np.sum(W*residual*jrr,axis=1)
    at_low=lam<=1e-14;at_upper=lam>=lam_upper-1e-14
    boundary=unknown&((at_low&(g>=0))|(at_upper&(g<=0)))
    stationary=np.abs(g)<=profile_tolerance*np.maximum(np.sqrt(GN),1e-12)
    unidentified=unknown&(~identifiable|(GN<=1e-22))
    converged=known|boundary|unidentified|(unknown&stationary&(H>1e-22))
    interior=unknown&~boundary&~unidentified&stationary&(H>1e-22)
    implicit_valid=unknown&~boundary&~unidentified&(H>1e-22)
    status=np.full(N,'unknown_positive_profile_interior',dtype='<U40')
    status[known]='reviewed_finite_plane'
    status[boundary&at_low]='unknown_limit_infinite_source_depth'
    status[boundary&at_upper]='unknown_limit_zero_source_depth'
    status[boundary&at_upper&depth_floor_bound]='unknown_limit_minimum_source_depth'
    status[unidentified]='unknown_depth_locally_unidentifiable'
    status[unknown&~converged]='unknown_profile_unconverged'
    status[unknown&~boundary&~unidentified&(H<=1e-22)]='unknown_profile_nonpositive_H'
    raw_blocks=[];blocks=[]
    if jac:
        dRa=np.broadcast_to(np.asarray(dRa,float),(N,3,3,3));dRb=np.broadcast_to(np.asarray(dRb,float),(N,3,3,3))
        dCa_s=np.broadcast_to(np.asarray(dCa_s,float),(N,3,4));dCb_s=np.broadcast_to(np.asarray(dCb_s,float),(N,3,4))
        dCa_v=np.broadcast_to(np.asarray(dCa_v,float),(N,3,4));dCb_v=np.broadcast_to(np.asarray(dCb_v,float),(N,3,4))
        dDa=np.zeros((N,3,4))
        dDa[:,:,:3]=np.einsum('ni,npij->njp',local,dRa)
        dDa[:,:,3]=np.einsum('ni,nij->nj',np.c_[dnaf,np.zeros(N)],Ra)
        dqa=np.einsum('nij,njp->nip',Rb,dDa)
        dqb=np.zeros((N,3,4));dqb[:,:,:3]=np.einsum('npij,nj->nip',dRb,da)
        dta=np.einsum('nij,njp->nip',Rb,dCa_v)
        dtb=-np.einsum('nij,njp->nip',Rb,dCb_v)
        dtb[:,:,:3]+=np.einsum('npij,nj->nip',dRb,delta)
        dts=np.einsum('nij,njp->nip',Rb,dCa_s-dCb_s)
        zeros=np.zeros((N,3,4))
        for which,(dq,dt)in enumerate(((dqa,dta),(dqb,dtb),(zeros,dts))):
            dnq=(dq[:,:2,:]-nq[:,:,None]*dq[:,2,None,:])/q[:,2,None,None]
            dh=dt[:,:2,:]-dnq*t[:,2,None,None]-nq[:,:,None]*dt[:,2,None,:]
            dl=np.einsum('ni,nip->np',line_unit,dh)
            du=(dh-line_unit[:,:,None]*dl[:,None,:])/np.maximum(line_length,1e-30)[:,None,None]
            dn=dnq+lam[:,None,None]*du
            raw=fb[:,None,None]*np.einsum('nij,njp->nip',D,dn)
            brown_cross=2*k1*(np.einsum('ni,nip->np',n,dn)[:,None,:]*line_unit[:,:,None]+
                np.sum(n*line_unit,axis=1)[:,None,None]*dn+
                np.einsum('ni,nip->np',line_unit,dn)[:,None,:]*n[:,:,None])
            mixed=fb[:,None,None]*(np.einsum('nij,njp->nip',D,du)+brown_cross)
            if which==1:
                raw[:,:,3]+=pixel-pp
                mixed[:,:,3]+=jrho
            drho=np.zeros((N,4))
            drho[implicit_valid]=-(np.einsum('ni,nip->np',W*jrho,raw)+
                np.einsum('ni,nip->np',W*residual,mixed))[implicit_valid]/H[implicit_valid,None]
            if np.any(known)and which!=1:
                dcentre=dCa_v if which==0 else dCa_s
                ddirection=dDa if which==0 else zeros
                drho[known]=(((np.einsum('ni,nip->np',plane_n[known],ddirection[known])+rho[known,None]*
                    np.einsum('ni,nip->np',plane_n[known],dcentre[known]))/plane_distance[known,None])*
                    (1.-tau[known,None])**2)
            block=raw+jrho[:,:,None]*drho[:,None,:]
            # Positive source depth -> 0 is a one-sided physical endpoint.
            # The line bound L/t_z depends on the camera parameters.
            zero_limit=boundary&at_upper&positive_tz&~depth_floor_bound
            if np.any(zero_limit):
                dbound=(dl*t[:,2,None]-line_length[:,None]*dt[:,2,:])/np.maximum(t[:,2,None]**2,1e-30)
                block[zero_limit]=(raw+jrho[:,:,None]*dbound[:,None,:])[zero_limit]
            finite_depth_limit=boundary&at_upper&depth_floor_bound
            if np.any(finite_depth_limit):
                fy=_project(q[finite_depth_limit]+rho_max*t[finite_depth_limit],fb[finite_depth_limit],pp,k1)[1]
                stable=np.einsum('nij,njp->nip',fy,(dq+rho_max*dt)[finite_depth_limit])
                if which==1:stable[:,:,3]+=(pixel-pp)[finite_depth_limit]
                block[finite_depth_limit]=stable
            if np.any(unidentified):
                block[unidentified]=(fb[:,None,None]*np.einsum('nij,njp->nip',D,dnq))[unidentified]
                if which==1:block[unidentified,:,3]+=(pixel-pp)[unidentified]
            if np.any(known):
                # Typed finite depth uses the original homogeneous model and
                # exact derivative of its plane intersection, no line profile.
                fy=_project(q[known]+tau[known,None]*tangent[known],fb[known],pp,k1)[1]
                yh=((1.-tau[:,None,None])*dq+tau[:,None,None]*dt)
                fraw=np.einsum('nij,njp->nip',fy,yh[known])
                ftan=np.einsum('nij,nj->ni',fy,tangent[known])
                if which==1:fraw[:,:,3]+=pixel[known]-pp
                block[known]=fraw+ftan[:,:,None]*drho[known,None,:]
                raw[known]=fraw
            raw_blocks.append(raw);blocks.append(block)
    state=None
    if return_state:
        with np.errstate(divide='ignore',invalid='ignore'):
            final_rho=tau/(1.-tau);depth=(1.-tau)/tau
        homogeneous_z=q[:,2]+tau*tangent[:,2]
        dtau=line_length*q[:,2]/np.maximum(homogeneous_z**2,1e-300)
        state=dict(inverse_depth_scene_units=final_rho,positive_source_depth_scene_units=depth,
            profile_status=status,profile_iterations=iterations+1,
            profile_iterations_per_row=row_iterations,profile_line_search_trials_per_row=line_search_trials,
            profile_line_search_failed_rows=line_search_failed,profile_converged_rows=converged,
            profile_nonpositive_hessian_rows=unknown&~boundary&~unidentified&(H<=1e-22),
            compact_inverse_depth=tau,profile_line_coordinate=lam,
            profile_nuisance_coordinate='lambda along unit target-normalized epipolar line; tau=rho/(1+rho) trace only',
            frozen_component_weights=weight.copy(),
            profile_stationarity_native_weighted=g,profile_GN_depth_information=GN,
            profile_exact_depth_hessian=H,profile_depth_information_coordinate='lambda',
            profile_nuisance_tangent_native=jrho,
            profile_rho_tangent_native=jrho*(dtau*(1.-tau)**2)[:,None],
            profile_tau_tangent_native=jrho*dtau[:,None],
            finite_rows=known,unknown_interior_rows=interior,unknown_boundary_rows=boundary,
            profile_residual_components_are_correlated=True,
            weight_semantics='native residual; per-component frozen IRLS weights divided by original sigma squared',
            derivatives='exact implicit frozen-weight profile residual J including residual curvature when converged/Hpositive; unconverged rows explicitly flagged; finite plane exact depth derivative',
            raw_unprofiled_Ja=None if not jac else raw_blocks[0],
            raw_unprofiled_Jb=None if not jac else raw_blocks[1],
            raw_unprofiled_Jshared=None if not jac else raw_blocks[2])
        state['minimum_source_depth_scene_units']=minimum_source_depth_scene_units
        # At an unidentifiable (including confirmed same-exposure) baseline,
        # the line interval has zero length: its upper and lower endpoints
        # coincide for every depth. It is not evidence for a floor depth.
        state['conditional_depth_floor_boundary_rows']=boundary&at_upper&depth_floor_bound&~unidentified
    flat=residual.reshape(-1)
    if not jac:return(flat,state)if return_state else flat
    result=(flat,*(x.reshape(-1,4)for x in blocks))
    return(*result,state)if return_state else result
