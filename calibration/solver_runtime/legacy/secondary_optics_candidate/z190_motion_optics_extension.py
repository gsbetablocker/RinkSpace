"""Opt-in three shared physical optics columns for the verified motion kernel.

The installed strict-double kernel and its four returned arrays are untouched.
Unknown converged interior rows use fixed-lambda raw columns followed by the
original frozen-weight depth Schur projection. Other rows use the original
physical profile residual central differences, including finite/endpoint rules.

Columns default to solver coordinates [PPx native px, PPy native px, k1/0.001].
Residuals and columns are native pixels, NOT divided by original source sigma.
Every central difference retains the same frozen weights and complete batch,
including the batch-wide Brown inversion stop rule. No source state is mutated.
"""
from __future__ import annotations
import os
import numpy as np

from z190_motion_native import native_motion, _inverse_brown_native

DEFAULT_STEPS=(1e-4,1e-4,1e-6)
DEFAULT_COLUMN_SCALES=(1.,1.,.001)
_UNSET=object()


def _array(value,shape):
    return np.ascontiguousarray(np.broadcast_to(np.asarray(value,float),shape))


def _weights(k,n):
    sigma=np.asarray(k['sigma'],float)
    if sigma.ndim==1:sigma=sigma.reshape(-1,1)
    sigma=_array(sigma,(n,2))
    weight=np.ones((n,2))if k.get('profile_weight')is None else _array(k['profile_weight'],(n,2))
    return weight/sigma**2


def _line_geometry(k,lam):
    """Replay exactly the native wrapper's Brown ray and einsum geometry."""
    n=len(k['a']);a=_array(k['a'],(n,2));fa=_array(k['fa'],(n,))
    fb=_array(k['fb'],(n,));pp=_array(k['pp'],(2,));k1=float(k['k1'])
    ra=_array(k['ra'],(n,3,3));rb=_array(k['rb'],(n,3,3))
    ca=_array(k['ca'],(n,3));cb=_array(k['cb'],(n,3))
    threads=int(os.environ.get('RINKSPACE_MOTION_THREADS',os.cpu_count()or 1))
    na,_=_inverse_brown_native(a,fa,pp,k1,threads)
    local=np.c_[na,np.ones(n)]
    source_direction=np.einsum('ni,nij->nj',local,ra)
    q=np.einsum('nij,nj->ni',rb,source_direction)
    t=np.einsum('nij,nj->ni',rb,ca-cb)
    nq=q[:,:2]/q[:,2,None]
    h=t[:,:2]-nq*t[:,2,None]
    # The C row sum has two terms, with this exact arithmetic order.
    length=np.sqrt(h[:,0]*h[:,0]+h[:,1]*h[:,1])
    u=h/np.maximum(length,1e-30)[:,None]
    point=nq+np.asarray(lam,float)[:,None]*u
    radius2=point[:,0]*point[:,0]+point[:,1]*point[:,1]
    radial=1.+k1*radius2
    D=np.empty((n,2,2));D[:,0,0]=radial+2.*k1*point[:,0]*point[:,0]
    D[:,0,1]=D[:,1,0]=2.*k1*point[:,0]*point[:,1]
    D[:,1,1]=radial+2.*k1*point[:,1]*point[:,1]
    tangent=np.empty((n,2))
    tangent[:,0]=(fb*D[:,0,0])*u[:,0]+(fb*D[:,0,1])*u[:,1]
    tangent[:,1]=(fb*D[:,1,0])*u[:,0]+(fb*D[:,1,1])*u[:,1]
    return dict(na=na,fa=fa,fb=fb,pp=pp,k1=k1,ra=ra,rb=rb,q=q,t=t,nq=nq,
                length=length,u=u,point=point,radius2=radius2,radial=radial,
                D=D,tangent=tangent)


def fixed_lambda_residual(native_inputs,lam):
    """Original line-chart residual with lambda fixed; useful for raw FD QA."""
    geometry=_line_geometry(native_inputs,lam)
    return (geometry['fb'][:,None]*geometry['point']*geometry['radial'][:,None]+
            geometry['pp']-np.asarray(native_inputs['b'],float)).reshape(-1)


def _raw_optics(k,lam,geometry):
    """Analytic physical d(PPx px,PPy px,k1), at fixed line lambda."""
    na=geometry['na'];k1=geometry['k1'];fa=geometry['fa'];n=len(na)
    r2=na[:,0]*na[:,0]+na[:,1]*na[:,1];s=1.+k1*r2
    a=s+2.*k1*na[:,0]*na[:,0];b=2.*k1*na[:,0]*na[:,1]
    c=s+2.*k1*na[:,1]*na[:,1];det=a*c-b*b
    # D_source dn = -e_axis/fa or -na*||na||^2. PP and k1 affect
    # both the frame-a inverse ray and frame-b forward Brown projection.
    dna=np.empty((n,2,3))
    dna[:,0,0]=-c/(det*fa);dna[:,1,0]=b/(det*fa)
    dna[:,0,1]=b/(det*fa);dna[:,1,1]=-a/(det*fa)
    dna[:,0,2]=-(c*na[:,0]-b*na[:,1])*r2/det
    dna[:,1,2]=-(-b*na[:,0]+a*na[:,1])*r2/det
    local=np.zeros((n,3,3));local[:,:2]=dna
    direction=np.einsum('nik,nij->njk',local,geometry['ra'])
    dq=np.einsum('nij,njp->nip',geometry['rb'],direction)
    q=geometry['q'];nq=geometry['nq'];u=geometry['u']
    dnq=(dq[:,:2]-nq[:,:,None]*dq[:,2,None,:])/q[:,2,None,None]
    dh=-dnq*geometry['t'][:,2,None,None]
    dl=np.einsum('ni,nip->np',u,dh)
    du=(dh-u[:,:,None]*dl[:,None,:])/np.maximum(geometry['length'],1e-30)[:,None,None]
    dn=dnq+np.asarray(lam)[:,None,None]*du
    raw=geometry['fb'][:,None,None]*np.einsum('nij,njp->nip',geometry['D'],dn)
    raw[:,0,0]+=1.;raw[:,1,1]+=1.
    raw[:,:,2]+=geometry['fb'][:,None]*geometry['point']*geometry['radius2'][:,None]
    return raw


def schur_optics_projection(raw,tangent,weights,denominator,active):
    """Same asymmetric W-orthogonal projector used by the native kernel."""
    raw=np.asarray(raw,float);tangent=np.asarray(tangent,float)
    factor=np.einsum('ni,nip->np',weights*tangent,raw)/np.maximum(denominator,1e-30)[:,None]
    projected=raw-tangent[:,:,None]*factor[:,None,:]
    result=raw.copy();result[active]=projected[active]
    return result


def native_motion_with_optics(*,steps=DEFAULT_STEPS,physical_column_scales=DEFAULT_COLUMN_SCALES,
                             minimum_source_depth_scene_units=0.,schur=True,**native_inputs):
    """Return ((r,Ja,Jb,Js,Joptics), checks, state), or ((r,Joptics),...).

    steps are physical central-FD increments [native px,native px,k1].
    physical_column_scales multiply physical derivatives once; k1/0.001
    has derivative 0.001 times the derivative with respect to physical k1.
    The original baseline arrays are returned by identity, never recomputed
    or changed after the call. The state is an optional audit workspace.
    """
    steps=np.asarray(steps,float);scales=np.asarray(physical_column_scales,float)
    if steps.shape!=(3,)or scales.shape!=(3,)or not np.isfinite(steps).all()or not np.isfinite(scales).all()or np.any(steps<=0):
        raise ValueError('Three finite positive physical FD steps and three finite column scales required')
    k=dict(native_inputs);k.pop('return_trace',None)
    baseline,checks,trace=native_motion(**k,minimum_source_depth_scene_units=minimum_source_depth_scene_units,
                                      schur=schur,return_trace=True)
    n=len(k['a']);geometry=_line_geometry(k,trace[:,0]);weights=_weights(k,n)
    tangent=geometry['tangent'];raw=_raw_optics(k,trace[:,0],geometry)
    active=(trace[:,9]!=0)&(trace[:,3]>1e-22)if schur else np.zeros(n,bool)
    optics=schur_optics_projection(raw,tangent,weights,trace[:,3],active)
    branch_changes=[];fallback_fd=None
    if np.any(~active):
        fallback_fd=np.empty((n,2,3))
        for column,step in enumerate(steps):
            plus=dict(k);minus=dict(k);plus['jacobian']=minus['jacobian']=False
            if column<2:
                pp=np.asarray(k['pp'],float).copy();pm=pp.copy();pp[column]+=step;pm[column]-=step
                plus['pp']=pp;minus['pp']=pm
            else:plus['k1']=float(k['k1'])+step;minus['k1']=float(k['k1'])-step
            rp,_,tp=native_motion(**plus,minimum_source_depth_scene_units=minimum_source_depth_scene_units,
                                 schur=schur,return_trace=True)
            rm,_,tm=native_motion(**minus,minimum_source_depth_scene_units=minimum_source_depth_scene_units,
                                 schur=schur,return_trace=True)
            fallback_fd[:,:,column]=((rp-rm)/(2.*step)).reshape(n,2)
            changed=np.any(tp[:,9:13]!=trace[:,9:13],axis=1)|np.any(tm[:,9:13]!=trace[:,9:13],axis=1)
            branch_changes.append(dict(column=column,rows=int(changed.sum()),inactive_rows=int(np.sum(changed&~active))))
        optics[~active]=fallback_fd[~active]
    joptics=(optics*scales[None,None,:]).reshape(2*n,3)
    audit=dict(checks,shared_optics_active_rows=int(active.sum()),shared_optics_inactive_rows=int((~active).sum()),
        maximum_optics_frozen_tangent_projection_native=float(np.max(np.abs(np.einsum('ni,nip->np',
            weights[active]*tangent[active],optics[active])),initial=0.)),
        optics_physical_fd_steps=steps.tolist(),optics_physical_column_scales=scales.tolist(),
        optics_fd_branch_changes=branch_changes,old_kernel_outputs_reused=True,
        depth_projection_weights='original frozen component weights / original sigma squared',
        optics_returned_sigma_normalized=False)
    state=dict(native_trace=trace,profile_nuisance_tangent_native=tangent,weights_native=weights,
        schur_interior_mask=active,raw_fixed_lambda_optics_physical=raw,
        full_profile_optics_fd_physical=fallback_fd,geometry=geometry,
        physical_column_scales=scales.copy(),minimum_source_depth_scene_units=float(minimum_source_depth_scene_units))
    result=(*baseline,joptics)if k['jacobian']else(baseline,joptics)
    return result,audit,state


def iter_motion_optics_inputs(bundle,local_q,current_calibration,profile_weights=_UNSET,*,chunk_points=65536):
    """Original bundle motion order/chunking, no prepared graph or mutation."""
    if chunk_points<=0:raise ValueError('Positive original motion chunk required')
    vv,rr,dr,cc,ds,dv=bundle.cache(local_q,False)
    focal=np.exp(vv[:,3]);cal=current_calibration
    pp=np.array([1920.*cal.principal_x_fraction[0],1080.*cal.principal_y_fraction[0]])
    frozen=bundle.profile_weights if profile_weights is _UNSET else profile_weights
    if frozen is not None and np.shape(frozen)!=(len(bundle.a),2):raise ValueError('Complete original frozen profile weights required')
    for lo in range(0,len(bundle.a),chunk_points):
        hi=min(lo+chunk_points,len(bundle.a));ia,ib=bundle.ia[lo:hi],bundle.ib[lo:hi]
        yield lo,hi,dict(a=bundle.a[lo:hi],b=bundle.b[lo:hi],fa=focal[ia],fb=focal[ib],
            ra=rr[ia],rb=rr[ib],da=None,db=None,ca=cc[ia],cb=cc[ib],
            dca_shared=None,dcb_shared=None,dca_view=None,dcb_view=None,
            pp=pp,k1=cal.k1[0],planes=bundle.finite_planes[lo:hi],jacobian=False,
            sigma=bundle.motion_sigma[2*lo:2*hi].reshape(-1,2),
            profile_weight=None if frozen is None else frozen[lo:hi])


def motion_optics_columns(bundle,local_q,current_calibration,profile_weights=_UNSET,steps=DEFAULT_STEPS,
                          physical_column_scales=DEFAULT_COLUMN_SCALES):
    """Provider returning complete native-pixel motion rows x three columns."""
    out=np.empty((bundle.motion_rows,3));kernel=bundle.motion_kernel
    for lo,hi,k in iter_motion_optics_inputs(bundle,local_q,current_calibration,profile_weights):
        result,_,_=native_motion_with_optics(**k,steps=steps,physical_column_scales=physical_column_scales,
            minimum_source_depth_scene_units=kernel.minimum_source_depth_scene_units,schur=kernel.schur)
        out[2*lo:2*hi]=result[-1]
    return out
