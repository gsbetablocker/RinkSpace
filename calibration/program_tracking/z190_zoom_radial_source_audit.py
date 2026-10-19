"""Audit existing Z190 k1(logf) against the entire unchanged source graph.

The latest Cxyz, PP and k1 at q_reference stay fixed. Only its already modeled
linear k1 zoom response and actual source-exposure R/f are free. No held pixels,
source subsets, arbitrary warp, new rig offsets or old camera seed enter.
"""
from __future__ import annotations
import argparse,json,time
from dataclasses import replace
from pathlib import Path
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from scipy.sparse import csr_matrix,vstack
from z190_source_objective_conflict import load_source_graph,render_focus,sha,IDENTIFIER,component_rows
from z190_continuous_source_bundle import StandingSourceBundle,radial_inverse,standing_relative_kernel
from z190_physical_camera import PhysicalCameraModel,CanonicalView,ViewState
from z190_source_camera_diagnostic import absolute_value_jac15,solve_with_crew_priors,source_errors,jsonable
from continuous_central_brown4_optin import enable_projection_backend
from main7_source_bundle_candidate import huber_cost


def radial_inverse_varying(distorted,k1):
    """Same exact central cubic branch as radial_inverse, vectorized by k1."""
    distorted,k1=np.asarray(distorted,float),np.asarray(k1,float)
    if not np.isfinite(distorted).all() or not np.isfinite(k1).all():
        raise ValueError('Nonfinite source radial ray')
    radius=np.linalg.norm(distorted,axis=1)
    positive=k1>0.;negative=k1<0.
    s=np.sqrt(3.*k1[positive])
    radius[positive]=2./s*np.sinh(np.arcsinh(1.5*s*radius[positive])/3.)
    s=np.sqrt(-3.*k1[negative]);argument=1.5*s*radius[negative]
    if np.any(argument>=1.):raise ValueError('No central radial inverse at this zoom')
    radius[negative]=2./s*np.sin(np.arcsin(argument)/3.)
    factor=1.+k1*radius**2
    local=distorted/factor[:,None]
    r2=np.einsum('ni,ni->n',local,local);eigen=1.+3.*k1*r2
    if np.any(factor<=1e-7) or np.any(eigen<=1e-7):
        raise ValueError('Zoom-dependent source inverse leaves central domain')
    inverse=np.broadcast_to(np.eye(2),(len(local),2,2)).copy()/factor[:,None,None]
    inverse-=(2.*k1/(factor*eigen))[:,None,None]*local[:,:,None]*local[:,None,:]
    return local,inverse


def zoom_radial_kernel(fa,fb,ra,rb,da,db,a,b,ppa,ppb,k1a,k1b,xa,xb,k_slope,pp_slope,jacobian=True):
    local_xy,inverse=radial_inverse_varying((a-ppa)/fa[:,None],k1a)
    local=np.c_[local_xy,np.ones(len(a))]
    world=np.einsum('ni,nij->nj',local,ra)
    target=np.einsum('ni,nji->nj',world,rb)
    if not np.isfinite(target).all() or np.any(target[:,2]<=.05):
        raise ValueError('Source ray leaves target forward domain')
    z=target[:,2];xy=target[:,:2]/z[:,None]
    r2=np.einsum('ni,ni->n',xy,xy);factor=1.+k1b*r2
    if np.any(factor<=1e-7) or np.any(1.+3.*k1b*r2<=1e-7):
        raise ValueError('Zoom-dependent target leaves central radial domain')
    pred=fb[:,None]*xy*factor[:,None]+ppb
    residual=(pred-b).ravel()
    if not jacobian:return residual
    forward=factor[:,None,None]*np.eye(2)+2.*k1b[:,None,None]*xy[:,:,None]*xy[:,None,:]
    normal=np.zeros((len(a),2,3));normal[:,0,0]=normal[:,1,1]=1./z
    normal[:,:,2]=-xy/z[:,None]
    duv=fb[:,None,None]*np.einsum('nij,njk->nik',forward,normal)
    def chain(dlocal):
        dw=np.einsum('ni,nij->nj',dlocal,ra)
        dt=np.einsum('ni,nji->nj',dw,rb)
        return np.einsum('nij,nj->ni',duv,dt)
    source=np.zeros((len(a),2,4));target_j=np.zeros_like(source)
    for axis in range(3):
        dw=np.einsum('ni,nij->nj',local,da[:,axis])
        dt=np.einsum('ni,nji->nj',dw,rb)
        source[:,:,axis]=np.einsum('nij,nj->ni',duv,dt)
        dt=np.einsum('ni,nji->nj',world,db[:,axis])
        target_j[:,:,axis]=np.einsum('nij,nj->ni',duv,dt)
    distorted=(a-ppa)/fa[:,None]
    dxy=-np.einsum('nij,nj->ni',inverse,distorted)
    source[:,:,3]=chain(np.c_[dxy,np.zeros(len(a))])
    target_j[:,:,3]=pred-ppb
    source_r2=np.einsum('ni,ni->n',local_xy,local_xy)
    dk=-np.einsum('nij,nj->ni',inverse,local_xy*source_r2[:,None])
    source_k=chain(np.c_[dk,np.zeros(len(a))])
    target_k=fb[:,None]*xy*r2[:,None]
    source[:,:,3]+=k_slope*source_k
    target_j[:,:,3]+=k_slope*target_k
    source_pp=[]
    for axis in range(2):
        dx=-inverse[:,:,axis]/fa[:,None]
        source_pp.append(chain(np.c_[dx,np.zeros(len(a))]))
    source[:,:,3]+=source_pp[0]*pp_slope[0]+source_pp[1]*pp_slope[1]
    target_j[:,:,3]+=np.asarray(pp_slope)
    shared_pp=np.stack(source_pp,axis=2)*xa[:,None,None]+np.eye(2)[None]*xb[:,None,None]
    shared_pp*=np.asarray([1920.,1080.])[None,None,:]
    shared_k=source_k*xa[:,None]+target_k*xb[:,None]
    shared=np.concatenate((shared_pp,shared_k[:,:,None]),axis=2)
    return residual,source.reshape(-1,4),target_j.reshape(-1,4),shared.reshape(-1,3)


class ZoomRadialBundle(StandingSourceBundle):
    def __init__(self,base,variant='k1_slope'):
        # A completed installation may have its own calibrated height. Preserve
        # it when wrapping the constant-optics graph; do not reset it to 1.6 m.
        super().__init__(base,base.calibration,prior_wide_focal_px=base.prior_wide_focal_px,
            fixed_height=True,fixed_height_scene_units=base.calibration.pivot_xyz[2])
        self.fixed_calibration=base.calibration
        self.variant=variant
        self.optical_template=replace(base.calibration,k1=(base.calibration.k1[0],0.),
            principal_x_fraction=(base.calibration.principal_x_fraction[0],0.),
            principal_y_fraction=(base.calibration.principal_y_fraction[0],0.))
        self.slope_columns=[self.optical_template.shared_state_names().index(k)for k in
            ('principal_x_fraction_degree_1','principal_y_fraction_degree_1','k1_degree_1')]
        self.active=np.asarray([False,False,True]if variant=='k1_slope'else[True,True,False])
    def initial_shared(self):
        return np.r_[super().initial_shared()[:3],np.zeros(3)]
    def current_calibration(self,q):
        slope=q[3:6]*self.active
        return replace(self.optical_template,k1=(self.fixed_calibration.k1[0],float(slope[2])),
            principal_x_fraction=(self.fixed_calibration.principal_x_fraction[0],float(slope[0])),
            principal_y_fraction=(self.fixed_calibration.principal_y_fraction[0],float(slope[1])))
    def priors(self,q,rotations=None):
        old=q.copy();old[:6]=np.r_[np.asarray(self.fixed_calibration.pivot_xyz)/100.,
            self.fixed_calibration.principal_x_fraction[0],self.fixed_calibration.principal_y_fraction[0],
            self.fixed_calibration.k1[0]]
        values,j=super().priors(old,rotations)
        j.data[j.indices<6]=0.
        return values,j
    def evaluate(self,q,jacobian=True,include_priors=True):
        views=q[6:].reshape(-1,4)
        if np.any((views[:,3]<np.log(100.))|(views[:,3]>np.log(50000.))):
            raise ValueError('Current effective focal numerical domain')
        rotations=[cv2.Rodrigues(v[:3])for v in views]
        rot=np.array([r[0]for r in rotations]);drot=np.array([r[1].reshape(3,3,3)for r in rotations])if jacobian else None
        f=np.exp(views[:,3]);x=views[:,3]-np.log(1920.)-self.fixed_calibration.q_reference
        slopes=q[3:6]*self.active
        k=self.fixed_calibration.k1[0]+slopes[2]*x
        pp0=np.asarray([1920.*self.fixed_calibration.principal_x_fraction[0],1080.*self.fixed_calibration.principal_y_fraction[0]])
        pp_slope=slopes[:2]*[1920.,1080.]
        pp=pp0+x[:,None]*pp_slope
        value=[];data=[]
        for lo in range(0,len(self.a),65536):
            hi=min(lo+65536,len(self.a));ia,ib=self.ia[lo:hi],self.ib[lo:hi]
            result=zoom_radial_kernel(f[ia],f[ib],rot[ia],rot[ib],None if drot is None else drot[ia],
                None if drot is None else drot[ib],self.a[lo:hi],self.b[lo:hi],pp[ia],pp[ib],k[ia],k[ib],x[ia],x[ib],
                slopes[2],pp_slope,jacobian)
            sigma=self.motion_sigma[2*lo:2*hi]
            if jacobian:
                residual,ja,jb,js=result;end=np.c_[ja,jb];swap=self.swap_rows[2*lo:2*hi]
                end[swap]=np.c_[jb[swap],ja[swap]]
                data.append((np.c_[js*self.active[None,:],end]/sigma[:,None]).ravel())
            else:residual=result
            value.append(residual/sigma)
        calibration=self.current_calibration(q)
        for p,rows in self.abs_groups:
            i=self.index[p];view=ViewState(tuple(views[i,:3]),float(views[i,3]-np.log(1920.)))
            camera,shared,view_j=calibration.camera25_and_jacobians(view)
            result=absolute_value_jac15(camera,rows,jacobian)
            if jacobian:
                residual,raw=result;scale=np.array([.01]*3+[1.]*5+[.001]*3+[1.]*4)
                shared_j=raw@(scale[:,None]*shared[:15,self.slope_columns])*self.active[None,:]
                local_j=raw@(scale[:,None]*view_j[:15])
                data.append(np.c_[np.zeros((len(residual),3)),shared_j,local_j].ravel())
            else:residual=result
            value.append(residual)
        value=np.concatenate(value)
        prior,pj=self.priors(q,rotations)if include_priors else(np.empty(0),None)
        if not jacobian:return np.r_[value,prior]
        matrix=csr_matrix((np.concatenate(data),self.indices,self.indptr),shape=(len(value),len(q)))
        return (np.r_[value,prior],vstack((matrix,pj),format='csr'))if len(prior)else(value,matrix)


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);enable_projection_backend()
    base,original,packet,rays,_,native,aliases,_=load_source_graph(args.physical,args.rows,args.rays)
    bundle=ZoomRadialBundle(base,args.variant);q=np.r_[bundle.initial_shared(),original[6:]]
    # Equality and J checks on the actual complete source graph before fitting.
    baseline=base.evaluate(original,False);value,j=bundle.evaluate(q)
    equal=float(np.max(np.abs(value-baseline)))
    if equal>1e-6:raise ValueError('Zero zoom slope changed the original source objective')
    checks=[]
    for col in tuple(np.flatnonzero(bundle.active)+3)+(6,7,8,9,6+4*bundle.index[230415104]+3):
        high,low=q.copy(),q.copy();high[col]+=1e-7;low[col]-=1e-7
        numeric=(bundle.evaluate(high,False)-bundle.evaluate(low,False))/2e-7
        actual=j[:,col].toarray().ravel()
        relative=float(np.linalg.norm(numeric-actual)/max(np.linalg.norm(numeric),1e-12))
        checks.append(dict(column=col,max_error=float(np.max(np.abs(numeric-actual))),relative_error=relative))
    if max(c['relative_error']for c in checks)>2e-4:raise ValueError('Actual full source zoom J check failed')
    solved,history,seconds=solve_with_crew_priors(bundle,q,args.output)
    if not np.array_equal(solved[:3],q[:3]):raise ValueError('Fixed Cxyz changed')
    calibration=bundle.current_calibration(solved)
    model=PhysicalCameraModel({IDENTIFIER:calibration})
    states={p:CanonicalView(IDENTIFIER,ViewState(tuple(solved[6+4*i:9+4*i]),
        float(solved[9+4*i]-np.log(1920.))))for i,p in enumerate(bundle.stamps)}
    model.save_npz(args.output/'physical_dense_zoom_radial_unaccepted.npz',native,aliases,states,
        source_sha256='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891',time_base='1/15360')
    np.savez_compressed(args.output/'diagnostic_state6rf.npz',q=solved)
    errors=[]
    for split in ('fit_rows','held_rows'):
        for raw in packet[split]:
            row=dict(raw)
            for k in ('world','uv','sigma','normal','normal_interval'):
                if k in row:row[k]=np.asarray(row[k],float)
            p=int(row['pts']);after=model.camera25(states[aliases[p]])
            errors.append(dict(pts=p,id=row['id'],split=row['split'],fit_role=split,
                before=source_errors(base.camera(original,aliases[p]),[row])[0],after=source_errors(after,[row])[0],
                effective_f_px=float(np.exp(after[6])),evaluated_k1=float(after[11])))
    residual=bundle.evaluate(solved,False)
    parts={name:dict(rows=len(residual[sl]),cost=huber_cost(residual[sl],True)if name in('absolute','motion')else
        .5*float(residual[sl]@residual[sl]))for name,sl in component_rows(bundle).items()}
    info_residual,info_j=bundle.evaluate(solved,include_priors=False)
    w=np.sqrt(np.minimum(1.,2./np.maximum(np.abs(info_residual),1e-12)))
    # One slope column conditional on the final source-connected R/f. This
    # raw norm is not marginal hardware information or lens identification.
    raw_norm=float(np.linalg.norm(info_j[:,5].toarray().ravel()*w))
    report=dict(schema='rinkspace.current-z190-full-source-linear-k1-zoom-audit.v1',accuracy_accepted=False,
        frozen_Cxyz=calibration.pivot_xyz,frozen_PP_px=[1920.*calibration.principal_x_fraction[0],1080.*calibration.principal_y_fraction[0]],
        variant=args.variant,k1_coefficients=calibration.k1,
        principal_x_fraction_coefficients=calibration.principal_x_fraction,
        principal_y_fraction_coefficients=calibration.principal_y_fraction,
        q_reference=calibration.q_reference,source_roles_UV_and_sigmas_unchanged=True,
        original_all_motion_edges=len(rays['pair_pts']),original_all_motion_points=len(base.a),
        development_keys=len(base.abs_groups),held_source_frames=len({r['pts']for r in packet['held_rows']}),
        fixed_skew_k2_tangential_lever_axial_zero=True,shared_optical_active_columns=np.asarray(
            ['principal_x_fraction_degree_1','principal_y_fraction_degree_1','k1_degree_1'])[bundle.active].tolist(),
        original_roll_prior_retained=True,zero_slope_objective_equality_max_sigma=equal,derivative_checks=checks,
        source_connected_cost_components=parts,source_errors=errors,history=history,seconds=seconds,
        k1_slope_data_column_norm_conditional_on_views=raw_norm,
        source_C_reviews=render_focus(args,packet,model,aliases,states),
        bindings=[dict(path=str(p),sha256=sha(p))for p in(args.physical,args.rows,args.rays,args.scene)])
    (args.output/'report.json').write_text(json.dumps(jsonable(report),indent=2),encoding='utf8')
    print(json.dumps(jsonable(dict(stage='full_source_existing_k1_zoom_audit_completed',seconds=seconds,k1=calibration.k1,
        costs=parts,C=[r for r in errors if r['id']=='C']))),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('physical','rows','rays','scene','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--variant',choices=['k1_slope','pp_slope'],default='k1_slope')
    run(p.parse_args())
