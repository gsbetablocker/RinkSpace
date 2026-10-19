"""Full source graph for a fixed pivot and shared nonzero camera-local lever.

Original unknown-depth pairs retain their observed UV and a positive inverse-
depth nuisance; their profile belongs to the moving-centre kernel. Explicit
native-material rows use their source-supported finite plane. No untyped row
is silently assigned ice/board depth. The pivot XY and PP/k1 remain fixed.
"""
from __future__ import annotations
from dataclasses import replace
import json,time
from pathlib import Path
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np
from scipy.sparse import csr_matrix,diags
from scipy.sparse.linalg import spsolve
from z190_physical_camera import ViewState,Z190StandingTripodHypothesis
from z190_source_camera_diagnostic import absolute_value_jac15
from z190_continuous_source_bundle import SCENE_UNIT_M,radial_inverse
from main7_source_bundle_candidate import huber_cost


def motion_source_csr_layout(ia,ib,index,abs_groups):
    """The unchanged 12/8-column source layout, filled without full int64 grids."""
    motion_rows=2*len(ia)
    absolute_rows=sum(2*len(r['uv'])for _,rows in abs_groups for r in rows)
    indices=np.empty(12*motion_rows+8*absolute_rows,np.int32)
    pointer=np.empty(motion_rows+absolute_rows+1,np.int32)
    pointer[:motion_rows+1]=12*np.arange(motion_rows+1,dtype=np.int32)
    swap_rows=np.repeat(np.asarray(ia)>np.asarray(ib),2)
    for lo in range(0,len(ia),65536):
        hi=min(lo+65536,len(ia))
        a,b=np.repeat(ia[lo:hi],2),np.repeat(ib[lo:hi],2)
        target=indices[24*lo:24*hi].reshape(2*(hi-lo),12)
        target[:,:4]=np.arange(4)
        target[:,4:8]=4+4*np.minimum(a,b)[:,None]+np.arange(4)
        target[:,8:12]=4+4*np.maximum(a,b)[:,None]+np.arange(4)
    row_cursor=motion_rows;data_cursor=12*motion_rows
    for p,rows in abs_groups:
        count=2*sum(len(r['uv'])for r in rows)
        target=indices[data_cursor:data_cursor+8*count].reshape(count,8)
        target[:,:4]=np.arange(4);target[:,4:]=4+4*index[p]+np.arange(4)
        pointer[row_cursor+1:row_cursor+count+1]=data_cursor+8*np.arange(1,count+1,dtype=np.int32)
        row_cursor+=count;data_cursor+=8*count
    return indices,pointer,swap_rows


class MovingPivotSourceBundle:
    """Shared [Pz/100,lx/100,ly/100,lz/100], then all actual R3/logf.

    Camera-local axes: x right, y down, z forward. Optical centre is
    P + R.T @ lever. The shared installation lever is constant, not zoom
    dependent. Parameters and height priors are independent per installation.
    """
    def __init__(self,base,calibration,motion_kernel,*,reference_pts,initial_reference_optical_height_scene,
                 finite_planes=None,finite_records=None,scene=None,absolute_evaluator=None):
        self.stamps,self.index=base.stamps,base.index
        self.a,self.b,self.ia,self.ib=base.a,base.b,base.ia,base.ib
        self.motion_sigma,self.motion_rows=base.motion_sigma,2*len(base.a)
        self.abs_groups=base.abs_groups
        self.native_residual_sigma=base.native_residual_sigma
        self.source_residual_rows=len(self.native_residual_sigma)
        self.view_offset=4;self.crew_fixed_height=False
        self.calibration=calibration;self.motion_kernel=motion_kernel
        self.crew_hypothesis=replace(Z190StandingTripodHypothesis.from_calibration(calibration),scene_unit_m=SCENE_UNIT_M)
        self.reference_pts=int(reference_pts);self.reference_index=self.index[self.reference_pts]
        self.initial_reference_optical_height_scene=float(initial_reference_optical_height_scene)
        self.optimized_local_columns=np.arange(4+4*len(self.stamps))
        self.lever_bounds_m=np.array([[-.05,.05],[-.30,-.10],[-.05,.25]])
        self.shared_lower=np.r_[-np.inf,self.lever_bounds_m[:,0]/(100.*SCENE_UNIT_M)]
        self.shared_upper=np.r_[np.inf,self.lever_bounds_m[:,1]/(100.*SCENE_UNIT_M)]
        self.finite_planes=np.zeros((len(self.a),4))if finite_planes is None else np.asarray(finite_planes,float)
        if self.finite_planes.shape!=(len(self.a),4):raise ValueError('Finite source row inventory changed')
        self.finite_mask=np.linalg.norm(self.finite_planes[:,:3],axis=1)>0.
        self.finite_records=[]if finite_records is None else finite_records
        self.scene=scene
        # A typed source adapter can constrain the confirmed physical side
        # with its original support intervals. All existing bundles retain
        # the original absolute evaluator by default.
        self.absolute_evaluator=absolute_value_jac15 if absolute_evaluator is None else absolute_evaluator
        if not callable(self.absolute_evaluator):raise TypeError('Source absolute evaluator must be callable')
        self.robust_phase=False;self.profile_weights=None
        self.indices,self.pointer,self.swap_rows=motion_source_csr_layout(self.ia,self.ib,self.index,self.abs_groups)
        if len(self.pointer)-1!=self.source_residual_rows:raise ValueError('Original residual rows changed')
        if any(calibration.axial)or len(calibration.k1)!=1 or any(calibration.k2)or calibration.allow_effective_pp_offset:
            raise ValueError('First moving-pivot branch has only constant lever and fixed PP/k1')
        self.mapping_names=[2,3,4,5]

    def initial_shared(self):return np.r_[self.calibration.pivot_xyz[2]/100.,np.asarray(self.calibration.lever_xyz)/100.]
    def shared_camera_jacobian(self,shared):
        """Map the physical calibration to this bundle's four shared slots."""
        return shared[:15,self.mapping_names]
    def current_calibration(self,q):
        return replace(self.calibration,pivot_xyz=(*self.calibration.pivot_xyz[:2],100.*q[0]),lever_xyz=tuple(100.*q[1:4]))
    def view(self,q,p):
        start=4+4*self.index[p]
        return ViewState(tuple(q[start:start+3]),float(q[start+3]-np.log(1920.)))
    def camera(self,q,p):return self.current_calibration(q).camera25(self.view(q,p))
    def project_shared_bounds(self,q):
        q=q.copy();q[:4]=np.clip(q[:4],self.shared_lower,self.shared_upper);return q
    def cache(self,q,jacobian):
        if np.any(q[:4]<self.shared_lower-1e-14)or np.any(q[:4]>self.shared_upper+1e-14):raise ValueError('User shared lever bounds')
        vv=q[4:].reshape(-1,4);raw=[cv2.Rodrigues(v[:3])for v in vv]
        rr=np.asarray([r[0]for r in raw]);dr=np.asarray([r[1].reshape(3,3,3)for r in raw])
        lever=100.*q[1:4];p=np.r_[self.calibration.pivot_xyz[:2],100.*q[0]]
        centres=p+np.einsum('vij,i->vj',rr,lever)
        dcshared=np.zeros((len(vv),3,4));dcshared[:,2,0]=100.;dcshared[:,:,1:]=100.*rr.transpose(0,2,1)
        dcview=np.zeros((len(vv),3,4));dcview[:,:,:3]=np.einsum('vaij,i->vja',dr,lever)
        return vv,rr,dr,centres,dcshared,dcview

    def priors(self,q,cache):
        vv,rr,dr,cc,ds,dv=cache
        # Latest hardware facts define a REFERENCE lens height, not pivot
        # height or a mean repeated across changing tilt/native aliases.
        iref=self.reference_index;sigma_m=self.crew_hypothesis.height_sigma_m
        h=(float(cc[iref,2])*SCENE_UNIT_M-self.crew_hypothesis.height_target_m)/sigma_m
        values=[h];columns=list(range(4));data=(ds[iref,2,:]*SCENE_UNIT_M/sigma_m).tolist();counts=[4]
        columns.extend((4+4*iref+np.arange(3)).tolist())
        data.extend((dv[iref,2,:3]*SCENE_UNIT_M/sigma_m).tolist());counts[0]+=3
        roll_sigma=np.deg2rad(.5)
        for i,(r,d)in enumerate(zip(rr,dr)):
            x,y=r[0,2],-r[1,2];norm=x*x+y*y
            if norm<1e-8:raise ValueError('Physical roll undefined')
            values.append(np.arctan2(x,y)/roll_sigma)
            data.extend(((y*d[:,0,2]+x*d[:,1,2])/norm/roll_sigma).tolist());columns.extend((4+4*i+np.arange(3)).tolist());counts.append(3)
        pointer=np.r_[0,np.cumsum(counts)]
        return np.asarray(values),csr_matrix((data,columns,pointer),shape=(len(values),len(q)))

    def validate_finite_extents(self,cache):
        if not self.finite_records:return
        vv,rr,dr,cc,ds,dv=cache
        pp=np.array([1920*self.calibration.principal_x_fraction[0],1080*self.calibration.principal_y_fraction[0]])
        poly=np.asarray(self.scene['ice_boundary']['points_xyz'],float)[:,:2].astype(np.float32)
        top=float(np.max(np.asarray(self.scene['board_surface']['top_edge_xyz'],float)[:,2]))
        for rec in self.finite_records:
            j=rec['global_bundle_index'];i=self.ia[j];f=np.exp(vv[i,3])
            xy,_=radial_inverse((self.a[j:j+1]-pp)/f,self.calibration.k1[0]);ray=rr[i].T@np.r_[xy[0],1.]
            n,d=self.finite_planes[j,:3],self.finite_planes[j,3]
            den=n@ray
            if abs(den)<1e-10:raise ValueError('Qualified finite source tangent to plane')
            distance=-(n@cc[i]+d)/den;world=cc[i]+distance*ray
            if distance<=0:raise ValueError('Qualified finite source negative depth')
            if rec['surface_kind']=='ice_plane':
                if cv2.pointPolygonTest(poly,tuple(world[:2]),False)<0:raise ValueError('Qualified ice point outside hard ice boundary')
            else:
                if not -1e-6<=world[2]<=top+1e-6:raise ValueError('Qualified finite wall point outside actual board height')
                start,end=np.asarray(rec['extent_start_xy']),np.asarray(rec['extent_end_xy']);delta=end-start
                station=(world[:2]-start)@delta/(delta@delta)
                if not -1e-6<=station<=1.+1e-6:raise ValueError('Qualified finite wall point outside source-supported named side')

    def evaluate(self,q,jacobian=True,include_priors=True):
        cache=self.cache(q,jacobian);vv,rr,dr,cc,ds,dv=cache
        self.validate_finite_extents(cache)
        focal=np.exp(vv[:,3]);pp=np.array([1920*self.calibration.principal_x_fraction[0],1080*self.calibration.principal_y_fraction[0]])
        # Allocate the final CSR values once. Keeping every chunk in a list
        # and then concatenating it held two complete motion Jacobians at
        # once; appending priors with vstack created a third copy.
        prior,pj=self.priors(q,cache)if include_priors else(np.empty(0),None)
        source_nnz=int(self.pointer[-1]);prior_nnz=0 if pj is None else pj.nnz
        value=np.empty(self.source_residual_rows+len(prior),float)
        data=np.empty(source_nnz+prior_nnz,float)if jacobian else None
        row_cursor=0;data_cursor=0
        for lo in range(0,len(self.a),65536):
            hi=min(lo+65536,len(self.a));a,b=self.ia[lo:hi],self.ib[lo:hi]
            result=self.motion_kernel(fa=focal[a],fb=focal[b],ra=rr[a],rb=rr[b],
                da=None if not jacobian else dr[a],db=None if not jacobian else dr[b],
                ca=cc[a],cb=cc[b],dca_shared=None if not jacobian else ds[a],dcb_shared=None if not jacobian else ds[b],
                dca_view=None if not jacobian else dv[a],dcb_view=None if not jacobian else dv[b],
                a=self.a[lo:hi],b=self.b[lo:hi],pp=pp,k1=self.calibration.k1[0],
                planes=self.finite_planes[lo:hi],jacobian=jacobian,
                sigma=self.motion_sigma[2*lo:2*hi].reshape(-1,2),
                profile_weight=None if self.profile_weights is None else self.profile_weights[lo:hi])
            sigma=self.motion_sigma[2*lo:2*hi]
            if jacobian:
                residual,ja,jb,shared=result[:4]
                count=2*(hi-lo);target=data[data_cursor:data_cursor+12*count].reshape(count,12)
                np.divide(shared,sigma[:,None],out=target[:,:4])
                np.divide(ja,sigma[:,None],out=target[:,4:8])
                np.divide(jb,sigma[:,None],out=target[:,8:12])
                swap=self.swap_rows[2*lo:2*hi]
                if np.any(swap):
                    target[swap,4:8]=jb[swap]/sigma[swap,None]
                    target[swap,8:12]=ja[swap]/sigma[swap,None]
                data_cursor+=12*count
                del ja,jb,shared,target
            else:residual=result[0]if isinstance(result,tuple)else result
            count=len(residual)
            np.divide(residual,sigma,out=value[row_cursor:row_cursor+count])
            row_cursor+=count
            del residual,result
        cal=self.current_calibration(q);scale=np.array([.01]*3+[1.]*5+[.001]*3+[1.]*4)
        for p,rows in self.abs_groups:
            if jacobian:
                cam,sj,vj=cal.camera25_and_jacobians(self.view(q,p));residual,raw=self.absolute_evaluator(cam,rows,True)
                mapping=np.c_[self.shared_camera_jacobian(sj),vj[:15]]*scale[:,None]
                mapped=(raw@mapping).ravel();data[data_cursor:data_cursor+len(mapped)]=mapped
                data_cursor+=len(mapped)
            else:residual=self.absolute_evaluator(self.camera(q,p),rows,False)
            value[row_cursor:row_cursor+len(residual)]=residual;row_cursor+=len(residual)
        if row_cursor!=self.source_residual_rows or (jacobian and data_cursor!=source_nnz):
            raise ValueError('Original source residual/CSR order changed')
        value[row_cursor:]=prior
        if not jacobian:return value
        if len(prior):
            indices=np.empty(source_nnz+prior_nnz,self.indices.dtype)
            indices[:source_nnz]=self.indices;indices[source_nnz:]=pj.indices
            pointer=np.empty(self.source_residual_rows+len(prior)+1,self.pointer.dtype)
            pointer[:self.source_residual_rows+1]=self.pointer
            pointer[self.source_residual_rows+1:]=source_nnz+pj.indptr[1:]
            data[source_nnz:]=pj.data
        else:indices,pointer=self.indices,self.pointer
        return value,csr_matrix((data,indices,pointer),shape=(len(value),len(q)))


def solve_moving_bundle(bundle,q0,output,*,phases=(False,True),initial_history=None,initial_damping=1e-3):
    """Sparse LM, original linear/Huber2 source phases and quadratic priors.

    The user lever interval is enforced by projecting the four shared slots;
    no residual-based row pruning or bounded exposure interpolation occurs.
    """
    q=q0.copy();history=[]if initial_history is None else list(initial_history)
    elapsed_offset=history[-1]['elapsed_seconds']if history else 0.
    started=time.perf_counter()-elapsed_offset;damping=float(initial_damping);n=bundle.source_residual_rows
    def cost(v,robust):return huber_cost(v[:n],robust)+.5*float(v[n:]@v[n:])
    for robust in phases:
        bundle.robust_phase=robust
        for iteration in range(45):
            if robust:
                preliminary=bundle.evaluate(q,False,include_priors=False)
                bundle.profile_weights=np.minimum(1.,2./np.maximum(np.abs(preliminary[:bundle.motion_rows]),1e-12)).reshape(-1,2)
            r,j=bundle.evaluate(q);before=cost(r,robust);weight=np.ones(len(r))
            if robust:weight[:n]=np.minimum(1.,2./np.maximum(np.abs(r[:n]),1e-12))
            jw=j.multiply(np.sqrt(weight)[:,None]);normal=(jw.T@jw).tocsc();gradient=j.T@(weight*r)
            diagonal=np.maximum(normal.diagonal(),1.);accepted=False;reason='';after=before;step=np.zeros_like(q)
            for trial in range(12):
                rawstep=spsolve(normal+diags(damping*diagonal),-gradient,permc_spec='MMD_AT_PLUS_A')
                nextq=bundle.project_shared_bounds(q+rawstep);step=nextq-q
                try:
                    if not np.isfinite(nextq).all()or np.any((nextq[7::4]<np.log(100.))|(nextq[7::4]>np.log(50000.))):raise ValueError('Numerical focal domain')
                    after=cost(bundle.evaluate(nextq,False),robust)
                    if after<before:q=nextq;accepted=True;damping=max(damping*.3,1e-12);break
                    reason='objective_not_lower'
                except(ValueError,RuntimeError)as exc:reason=str(exc)[:180]
                damping*=10.
            source_step=float(np.max(np.abs((j@step)[:n]*bundle.native_residual_sigma)))
            prior_step=float(np.max(np.abs((j@step)[n:])))
            rec=dict(phase='huber_source'if robust else'linear_source',iteration=iteration,cost=before,next_cost=after if accepted else before,
                accepted=accepted,trials=trial+1,damping=damping,failure_reason=None if accepted else reason,
                maximum_linearized_source_step_px=source_step,maximum_linearized_prior_step_sigma=prior_step,
                pivot_height_m=float(q[0]*100*SCENE_UNIT_M),lever_xyz_m=(q[1:4]*100*SCENE_UNIT_M).tolist(),elapsed_seconds=time.perf_counter()-started)
            history.append(rec);(output/'bundle_iterations.json').write_text(json.dumps(history,indent=2),encoding='utf8')
            checkpoint=output/'latest_accepted_state.tmp.npz'
            np.savez_compressed(checkpoint,q=q,canonical_pts=bundle.stamps)
            checkpoint.replace(output/'latest_accepted_state.npz')
            print(json.dumps(dict(stage='moving_C_iteration',**rec)),flush=True)
            if not accepted or(source_step<.05 and prior_step<.001):break
    return q,history,time.perf_counter()-started
