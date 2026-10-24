"""The sole new physical forward operator used by fitting and rendering.

All points, centers and rows retain their identities. A failed row has a status;
no clipping, hidden observed-row substitution, or batch-wide preparation failure.
The analytic camera mathematics is shared with this task's independently written
converter, not imported from the retired solver.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import BSpline, PPoly

from .d78_import import D78Initialization
from .model import ModelBundle, TARGET_MOUNT, REPLAY_MOUNT, from_d78

VERSION = "rinksolve.physical_projector.v1"
CERTIFICATE_VERSION = "rinksolve.closed_readout_four_cell_certificate.v1"
ROW_STATUS = {
    0: "ok", 1: "native_pts_out_of_scope", 2: "readout_outside_trajectory",
    3: "camera_projection_domain", 4: "no_sensor_bracket",
    5: "endpoint_rounding_indeterminate_use_explicit_row_equality",
    6: "analytic_uniqueness_not_certified", 7: "root_not_converged",
    8: "nonfinite_world_point", 9: "row_auxiliary_outside_sensor",
}


def skew(vector):
    v = np.asarray(vector,dtype=np.float64)
    out=np.zeros(v.shape[:-1]+(3,3))
    out[...,0,1],out[...,0,2]=-v[...,2],v[...,1]
    out[...,1,0],out[...,1,2]=v[...,2],-v[...,0]
    out[...,2,0],out[...,2,1]=-v[...,1],v[...,0]
    return out


def so3_left_jacobian(rotvec):
    v=np.asarray(rotvec,dtype=np.float64);theta=float(np.linalg.norm(v));K=skew(v)
    if theta<1e-5:
        return np.eye(3)+(0.5-theta*theta/24)*K+(1/6-theta*theta/120)*(K@K)
    return np.eye(3)+(1-np.cos(theta))/theta**2*K+(theta-np.sin(theta))/theta**3*(K@K)


def map_reference_gauge_jacobians(control_indices,jacobian_controls):
    """Map local cubic derivatives to the explicit reference-angle tangent views.

    Control 2 contributes log_f only. Angle derivatives chain to controls 1,3.
    The returned union has at most five nonzero native control views per row.
    Padded view ID -1 must not create a native edge; control 2 uses column 2 only.
    """
    indices=np.asarray(control_indices,dtype=np.int64);J=np.asarray(jacobian_controls,dtype=np.float64)
    if indices.ndim!=2 or indices.shape[1]!=4 or J.shape[:2]!=indices.shape or J.shape[-1]!=3:
        raise ValueError("expected [N,4] support and [N,4,residual_dim,3] derivatives")
    has_ref=np.any(indices==2,axis=1)
    candidates=np.concatenate((indices,np.where(has_ref,1,-1)[:,None],np.where(has_ref,3,-1)[:,None]),axis=1)
    candidates=np.sort(candidates,axis=1)
    keep=(candidates>=0)&np.concatenate((np.ones((len(indices),1),dtype=bool),np.diff(candidates,axis=1)!=0),axis=1)
    outidx=np.full((len(indices),5),-1,dtype=np.int64);out=np.zeros((len(indices),5,J.shape[2],3))
    slot=np.cumsum(keep,axis=1)-1
    rows,cols=np.nonzero(keep)
    if np.any(slot[rows,cols]>=5):
        raise ValueError("reference gauge support exceeds expected five views")
    outidx[rows,slot[rows,cols]]=candidates[rows,cols]
    for k in range(4):
        for component in range(3):
            for target,coefficient in ((indices[:,k],1.),):
                mask=np.ones(len(indices),dtype=bool)
                if component<2:
                    mask=indices[:,k]!=2
                matching=(outidx==target[:,None])&mask[:,None]
                rr,cc=np.nonzero(matching);out[rr,cc,:,component]+=coefficient*J[rr,k,:,component]
            if component<2:
                for target in (1,3):
                    matching=(outidx==target)&(indices[:,k,None]==2)
                    rr,cc=np.nonzero(matching);out[rr,cc,:,component]-=.25*J[rr,k,:,component]
    return outidx,out


class ProjectionOperator(D78Initialization):
    """Immutable model-state view; the only caller-facing physical projection.

    pose_at/project_at/RS certificates share the fresh converter mathematics.
    The operational state comes solely from the new typed model. Choosing the
    constant-lever family zeroes no hidden optimized term: it is a separately
    identified model whose non-equivalence is measured by compare_mount_families.
    """
    def __init__(self,model: ModelBundle,shot_id="S01"):
        self.model=model
        self.shot=next((s for s in model.shots if s.id==shot_id),None)
        if self.shot is None:
            raise KeyError(f"shot not in current model: {shot_id}")
        self.rig=next(r for r in model.rigs if r.id==self.shot.rig_id)
        self.camera_definition=self.rig.definition(model.state)
        self.controls=model.state.get(self.shot.control_block_id).reshape(self.shot.control_count,3)
        n=len(self.controls)
        knots=self.shot.first_control_time_seconds+(np.arange(n+4)-2)*self.shot.control_dt_seconds
        self.trajectory=BSpline(knots,self.controls,3,axis=0,extrapolate=False)
        self.derivative_polynomials=[PPoly.from_spline((knots,self.controls[:,j],3)).derivative() for j in range(3)]
        self.domain=(float(knots[3]),float(knots[n]))
        self.native_pts=self.shot.native_pts
        self.board=model.scene.board(model.state)
        self._rate_cache={}

    @property
    def identity(self):
        return {"implementation_id":VERSION,"certificate_policy_id":CERTIFICATE_VERSION,"model_id":self.model.id,"model_revision":self.model.revision,
            "parameter_state_id":self.model.state.id,"parameter_revision":self.model.state.revision,
            "parameter_sha256":self.model.state.digest(),"shot_id":self.shot.id,"rig_id":self.rig.id,
            "coordinate_system_id":self.model.scene.coordinate_system_id,"mount_family":self.rig.mount_family,
            "unit":"scene_unit","metric_verified":self.model.scene.metric_verified}

    def frame_time(self,native_pts):
        return self.shot.frame_time(native_pts)

    def pose_at(self,time):
        return D78Initialization.pose_at(self,time)

    def trajectory_gauge_equalities(self):
        """Representation gauge, not an initial-value physical constraint."""
        v=self.trajectory(0.)[:2]
        weights=BSpline.design_matrix(np.array([0.]),self.trajectory.t,3,extrapolate=False).toarray()[0]
        return {"value":v,"control_indices":np.flatnonzero(weights),"weights":weights[weights!=0],
            "meaning":"alpha(reference)=beta(reference)=0 defines reference R0; do not freeze arbitrary initial angles"}

    def project_at(self,xyz,time):
        xyz=np.asarray(xyz,dtype=np.float64);time=np.asarray(time,dtype=np.float64)
        if xyz.shape[-1:]!=(3,):
            raise ValueError("world points require last dimension 3")
        shape=np.broadcast_shapes(xyz.shape[:-1],time.shape)
        xyz=np.broadcast_to(xyz,shape+(3,));time=np.broadcast_to(time,shape)
        # Internal mathematical calls have all valid times. Public calls preserve
        # invalid rows without making one row reject the rest of a current batch.
        valid_time=np.isfinite(time)&(time>=self.domain[0])&(time<=self.domain[1])&np.isfinite(xyz).all(axis=-1)
        if np.all(valid_time):
            out=D78Initialization.project_at(self,xyz,time)
            out["time_domain_valid"]=valid_time
            return out
        if not np.any(valid_time):
            return self._empty_projection(shape)
        flat_xyz,flat_time=xyz.reshape(-1,3),time.reshape(-1)
        mask=valid_time.reshape(-1)
        valid=D78Initialization.project_at(self,flat_xyz[mask],flat_time[mask])
        result={}
        for k,v in valid.items():
            a=np.asarray(v);tail=a.shape[1:]
            target=np.full((mask.size,)+tail,False if a.dtype.kind=="b" else np.nan,dtype=a.dtype)
            target[mask]=a
            result[k]=target.reshape(shape+tail)
        result["time_domain_valid"]=valid_time
        return result

    @staticmethod
    def _empty_projection(shape):
        return {"uv":np.full(shape+(2,),np.nan),"valid":np.zeros(shape,dtype=bool),
            "depth":np.full(shape,np.nan),"camera_xyz":np.full(shape+(3,),np.nan),
            "ray_radius":np.full(shape,np.nan),"R_cw":np.full(shape+(3,3),np.nan),
            "C_w":np.full(shape+(3,),np.nan),"K":np.full(shape+(3,3),np.nan),
            "brown_k1_k2_p1_p2":np.full(shape+(4,),np.nan),"log_f":np.full(shape,np.nan),
            "zoom_offset":np.full(shape,np.nan),"time_domain_valid":np.zeros(shape,dtype=bool)}

    def derivative_envelope(self,frame_times):
        data=np.asarray(frame_times,dtype=np.float64)
        unique,inverse=np.unique(data.reshape(-1),return_inverse=True)
        missing=[float(t) for t in unique if float(t) not in self._rate_cache]
        if missing:
            bounds=D78Initialization.derivative_envelope(self,np.array(missing))
            self._rate_cache.update(zip(missing,bounds))
        return np.array([self._rate_cache[float(t)] for t in unique])[inverse].reshape(data.shape+(3,))

    def world_path(self,curve_id,station):
        return self.model.scene.world_path(curve_id,station,self.model.state,board=self.board)

    def world_rows(self,curve_ids,stations):
        return self.model.scene.world_rows(curve_ids,stations,self.model.state)

    def project_at_jacobian(self,xyz,time):
        """Analytic d(pixel)/d(world XYZ) and d(pixel)/d(source second).

        This differentiates the same pose, zoom/Brown and nonzero local lever.
        It is also the local kernel needed by row-equation or implicit-RS factors.
        """
        xyz=np.asarray(xyz,dtype=np.float64);time=np.asarray(time,dtype=np.float64)
        shape=np.broadcast_shapes(xyz.shape[:-1],time.shape)
        xyz=np.broadcast_to(xyz,shape+(3,));time=np.broadcast_to(time,shape)
        out=self.project_at(xyz,time)
        rates=self.trajectory.derivative()(time)
        alpha_dot,beta_dot,logf_dot=np.moveaxis(rates,-1,0)
        c=self.camera_definition;R=out["R_cw"]
        Rdot=skew(c["tilt_axis_camera"])@R*beta_dot[...,None,None]+R@skew(c["pan_axis_world"])*alpha_dot[...,None,None]
        tilt_world=np.einsum("...ji,j->...i",R,np.asarray(c["tilt_axis_camera"]))
        tilt_world_dot=np.einsum("...ji,j->...i",Rdot,np.asarray(c["tilt_axis_camera"]))
        cross=np.cross(c["pan_axis_world"],tilt_world)
        cross_dot=np.cross(c["pan_axis_world"],tilt_world_dot)
        norm=np.linalg.norm(cross,axis=-1)
        axis=cross/norm[...,None]
        axis_dot=(cross_dot-axis*np.einsum("...i,...i->...",axis,cross_dot)[...,None])/norm[...,None]
        lever_dot=np.zeros(shape+(3,))
        lever_dot[...,2]=(c["pupil_polynomial"][0]+2*c["pupil_polynomial"][1]*out["zoom_offset"])*logf_dot
        ray_dot=np.einsum("...ij,...j->...i",Rdot,xyz-np.asarray(c["pivot_world"]))-lever_dot
        ray_dot-=c["axis_spacing"]*(np.einsum("...ij,...j->...i",Rdot,axis)+np.einsum("...ij,...j->...i",R,axis_dot))
        q=out["camera_xyz"]
        with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
            x,y=np.moveaxis(q[...,:2]/q[...,2,None],-1,0)
            normalized_dot=(ray_dot[...,:2]-np.stack((x,y),axis=-1)*ray_dot[...,2,None])/q[...,2,None]
            r2=x*x+y*y
            k1,k2,p1,p2=np.moveaxis(out["brown_k1_k2_p1_p2"],-1,0)
            radial=1+k1*r2+k2*r2*r2;g=k1+2*k2*r2
            J=np.empty(shape+(2,2))
            J[...,0,0]=radial+2*x*x*g+2*p1*y+6*p2*x
            J[...,0,1]=2*x*y*g+2*p1*x+2*p2*y
            J[...,1,0]=J[...,0,1]
            J[...,1,1]=radial+2*y*y*g+6*p1*y+2*p2*x
            distorted=np.stack((x*radial+2*p1*x*y+p2*(r2+2*x*x),y*radial+p1*(r2+2*y*y)+2*p2*x*y),axis=-1)
            direct=np.stack((x,y),axis=-1)*(c["k1_coefficients"][1]*r2+c["k2_coefficients"][1]*r2*r2)[...,None]*logf_dot[...,None]
            distorted_dot=np.einsum("...ij,...j->...i",J,normalized_dot)+direct
            focal=np.stack((out["K"][...,0,0],out["K"][...,1,1]),axis=-1)
            uv_dot=focal*(distorted_dot+distorted*logf_dot[...,None])+np.asarray(c["principal_zoom_slope"])*logf_dot[...,None]
            perspective=np.zeros(shape+(2,3))
            perspective[...,0,0],perspective[...,1,1]=1/q[...,2],1/q[...,2]
            perspective[...,0,2],perspective[...,1,2]=-x/q[...,2],-y/q[...,2]
            jac_xyz=focal[...,None]*np.einsum("...ij,...jk,...kl->...il",J,perspective,R)
        out.update(jacobian_world=jac_xyz,jacobian_time=uv_dot,rotation_time_derivative=Rdot,
                   camera_ray_time_derivative=ray_dot)
        return out

    def project_with_row_auxiliary(self,xyz,native_pts,row_auxiliary):
        """Formal RS equality; row is a numerical auxiliary, not a point station.

        Predicted UV and F=row-project_y are evaluated without moving the static
        physical point. Rows outside the closed sensor interval remain failures.
        This path can prepare exact boundary observations without pretending an
        ulp-ambiguous bracket proves an interior row root.
        """
        pts=np.asarray(native_pts)
        if pts.dtype.kind not in "iu":
            raise TypeError("native PTS require integer array; no rounding")
        xyz=np.asarray(xyz,dtype=np.float64);row=np.asarray(row_auxiliary,dtype=np.float64)
        shape=np.broadcast_shapes(xyz.shape[:-1],pts.shape,row.shape)
        xyz=np.broadcast_to(xyz,shape+(3,));pts=np.broadcast_to(pts,shape);row=np.broadcast_to(row,shape)
        frame_time=(pts-self.shot.first_native_pts)*self.shot.time_base_num/self.shot.time_base_den
        tau=self.camera_definition["signed_readout_seconds"];h=self.rig.image_height-1
        time=frame_time+tau*(row/h-.5)
        out=self.project_at_jacobian(xyz,time)
        in_pts=(pts>=self.shot.first_native_pts)&(pts<=self.shot.last_native_pts)&((pts-self.shot.first_native_pts)%self.shot.step_native_pts==0)
        in_sensor=np.isfinite(row)&(row>=0)&(row<=h)
        status=np.zeros(shape,dtype=np.uint8)
        status=np.where(~out["valid"],3,status)
        status=np.where(~out["time_domain_valid"],2,status)
        status=np.where(~in_sensor,9,status)
        status=np.where(~in_pts,1,status)
        equation=row-out["uv"][...,1]
        out.update(row=row,row_equality_px=equation,row_is_auxiliary=np.ones(shape,dtype=bool),
            row_equality_jacobian_world=-out["jacobian_world"][...,1,:],
            row_equality_jacobian_row=1-tau/h*out["jacobian_time"][...,1],
            uv_jacobian_row=tau/h*out["jacobian_time"],
            valid=out["valid"]&in_pts&in_sensor,status_code=status,
            formal_equality_satisfied=np.abs(equation)<=1e-7)
        return out

    def _partitioned_rs_certificate(self,xyz,frame_time,*,partitions=4):
        """A sufficient full-sensor proof from a union of closed time cells.

        The physical projection, original readout and row equation are unchanged.
        A shorter readout in the bound-only view selects each cell's exact cubic
        derivative envelope. Multiplying its contraction by the cell count
        restores the ORIGINAL tau/height in d(project_y(row))/d(row). Every cell
        must satisfy depth, Brown/ray domain and contraction; a failure abstains.
        This is a stronger sufficient certificate, not a sampled uniqueness test.
        """
        from copy import copy
        if type(partitions) is not int or partitions<1:
            raise ValueError("certificate partitions require a positive integer")
        xyz=np.asarray(xyz,dtype=np.float64);frame_time=np.asarray(frame_time,dtype=np.float64)
        shape=np.broadcast_shapes(xyz.shape[:-1],frame_time.shape)
        xyz=np.broadcast_to(xyz,shape+(3,));frame_time=np.broadcast_to(frame_time,shape)
        tau=self.camera_definition["signed_readout_seconds"]
        # Outward overlap covers floating-point rounding at every cell seam
        # and at the original endpoints. This enlarges only proof windows.
        # Multiplication by partitions remains conservative for original tau.
        padding=8*np.finfo(float).eps*max(1.,float(np.max(np.abs(frame_time))) if frame_time.size else 0.,abs(tau))
        bound_tau=np.copysign(abs(tau)/partitions+2*padding,tau) if tau else 0.
        window_ok=(frame_time-abs(tau)/2-2*padding>=self.domain[0])&(frame_time+abs(tau)/2+2*padding<=self.domain[1]) if tau else np.ones(shape,dtype=bool)
        if not np.all(window_ok):
            # A proof must not extrapolate beyond the allocated trajectory.
            # Abstain at a tight boundary; retain the original numerical root.
            result={key:np.full(shape,False if key=="certified_unique" else np.nan,dtype=bool if key=="certified_unique" else float)
                for key in ("certified_unique","contraction_upper","depth_lower","ray_radius_upper","brown_fold_lower")}
            result.update(full_sensor_readout=True,sensor_partition_count=np.full(shape,partitions,dtype=np.int32))
            if np.any(window_ok):
                inside=self._partitioned_rs_certificate(xyz[window_ok],frame_time[window_ok],partitions=partitions)
                for key in ("certified_unique","contraction_upper","depth_lower","ray_radius_upper","brown_fold_lower"):
                    result[key][window_ok]=inside[key]
            return result
        bound_view=copy(self)
        bound_view.camera_definition=dict(self.camera_definition,signed_readout_seconds=bound_tau)
        # The derivative envelope depends on the readout window, so it cannot
        # reuse this operator's coarse-window cache. pose_at is tau-independent.
        bound_view._rate_cache={}
        contraction=np.zeros(shape);depth=np.full(shape,np.inf)
        radius=np.zeros(shape);fold=np.full(shape,np.inf)
        for cell in range(partitions):
            center=frame_time+tau*((cell+.5)/partitions-.5)
            part=D78Initialization.rs_certificate(bound_view,xyz,center)
            contraction=np.maximum(contraction,part["contraction_upper"]*partitions)
            depth=np.minimum(depth,part["depth_lower"])
            radius=np.maximum(radius,part["ray_radius_upper"])
            fold=np.minimum(fold,part["brown_fold_lower"])
        certified=(depth>0)&(radius<=self.camera_definition["radius_max"])&(fold>1e-10)&np.isfinite(contraction)&(contraction<1-1e-9)
        return dict(certified_unique=certified,contraction_upper=contraction,
            depth_lower=depth,ray_radius_upper=radius,brown_fold_lower=fold,
            full_sensor_readout=True,sensor_partition_count=np.full(shape,partitions,dtype=np.int32))

    def project_rs(self,xyz,native_pts,*,root_iterations=16,root_tolerance_px=1e-8):
        xyz=np.asarray(xyz,dtype=np.float64);pts=np.asarray(native_pts)
        if xyz.shape[-1:]!=(3,) or pts.dtype.kind not in "iu":
            raise ValueError("world XYZ and integer native PTS required")
        shape=np.broadcast_shapes(xyz.shape[:-1],pts.shape)
        xyz=np.broadcast_to(xyz,shape+(3,));pts=np.broadcast_to(pts,shape)
        points=xyz.reshape(-1,3);p=pts.reshape(-1);n=len(p)
        frame_time=(p-self.shot.first_native_pts)*self.shot.time_base_num/self.shot.time_base_den
        tau=self.camera_definition["signed_readout_seconds"];h=self.rig.image_height-1
        half=abs(tau)/2
        scope=(p>=self.shot.first_native_pts)&(p<=self.shot.last_native_pts)&((p-self.shot.first_native_pts)%self.shot.step_native_pts==0)
        time_ok=(frame_time-half>=self.domain[0])&(frame_time+half<=self.domain[1])
        finite=np.isfinite(points).all(axis=-1)
        selected=scope&time_ok&finite
        status=np.zeros(n,dtype=np.uint8)
        status[~finite]=8;status[~time_ok]=2;status[~scope]=1
        result=self._empty_projection((n,))
        row_all=np.full(n,np.nan);eq_all=np.full(n,np.nan)
        bracket_all=np.zeros(n,dtype=bool);num_all=np.zeros(n,dtype=bool);aux_all=np.zeros(n,dtype=bool)
        cert_all={k:np.full(n,False if k=="certified_unique" else np.nan,dtype=bool if k=="certified_unique" else np.float64)
            for k in ("certified_unique","contraction_upper","depth_lower","ray_radius_upper","brown_fold_lower")}
        cert_all["coarse_contraction_upper"]=np.full(n,np.nan)
        cert_all["sensor_partition_count"]=np.ones(n,dtype=np.int32)
        if np.any(selected):
            X,t=points[selected],frame_time[selected]
            cert=self.rs_certificate(X,t)
            cert["coarse_contraction_upper"]=cert["contraction_upper"].copy()
            cert["sensor_partition_count"]=np.ones(len(t),dtype=np.int32)
            lower=np.zeros(len(t));upper=np.full(len(t),float(h))
            a=self.project_at(X,t-tau/2);b=self.project_at(X,t+tau/2)
            Fa=-a["uv"][:,1];Fb=h-b["uv"][:,1]
            domain=a["valid"]&b["valid"]
            bracket=domain&(Fa<=0)&(Fb>=0)
            roundtol=8*np.finfo(float).eps*max(1,h)
            borderline=domain&~bracket&(((Fa>0)&(Fa<=roundtol))|((Fb<0)&(Fb>=-roundtol)))
            # Initial row uses the current model at frame-center time only.
            middle=self.project_at(X,t)["uv"][:,1]
            row=np.where((middle>0)&(middle<h),middle,h/2)
            # An in-bracket endpoint is a valid numerical candidate when its
            # directly evaluated equation is already within the requested tolerance.
            # This is root evaluation, not clipping an out-of-domain solution.
            row=np.where(bracket&(np.abs(Fa)<=root_tolerance_px),0.,row)
            row=np.where(bracket&(np.abs(Fb)<=root_tolerance_px),float(h),row)
            converged=np.zeros(len(t),dtype=bool)
            for _ in range(root_iterations):
                at=self.project_at_jacobian(X,t+tau*(row/h-.5))
                F=row-at["uv"][:,1]
                converged|=(np.abs(F)<=root_tolerance_px)&bracket
                lower=np.where((F<0)&~converged,row,lower)
                upper=np.where((F>=0)&~converged,row,upper)
                denominator=1-tau/h*at["jacobian_time"][:,1]
                candidate=row-np.divide(F,denominator,out=np.zeros_like(F),where=denominator!=0)
                inside=(candidate>lower)&(candidate<upper)&np.isfinite(candidate)
                candidate=np.where(inside,candidate,(lower+upper)/2)
                row=np.where(converged,row,candidate)
                if np.all(converged|~bracket):
                    break
            # An explicit endpoint auxiliary is retained for the few boundary
            # rounding cases. It is NOT labelled a valid eliminated trueRS root.
            endpoint=np.where(np.abs(Fa)<=np.abs(Fb),0.,float(h))
            row=np.where(borderline,endpoint,row)
            at=self.project_at(X,t+tau*(row/h-.5))
            equality=row-at["uv"][:,1]
            numerical=at["valid"]&bracket&(np.abs(equality)<=root_tolerance_px)
            # A single global norm envelope may be too broad even when the
            # same physical readout has a unique root. Prove all four cells;
            # never promote a failed proof merely because a root was found.
            refine=np.flatnonzero(numerical&~cert["certified_unique"])
            if len(refine):
                refined=self._partitioned_rs_certificate(X[refine],t[refine])
                proven=refine[refined["certified_unique"]]
                for key in ("certified_unique","contraction_upper","depth_lower","ray_radius_upper","brown_fold_lower","sensor_partition_count"):
                    cert[key][proven]=refined[key][refined["certified_unique"]]
            codes=np.zeros(len(t),dtype=np.uint8)
            codes[~domain]=3
            codes[domain&~bracket]=4
            codes[borderline]=5
            codes[bracket&~numerical]=7
            codes[numerical&~cert["certified_unique"]]=6
            valid=numerical&cert["certified_unique"]
            codes[valid]=0
            status[selected]=codes
            for k,v in at.items():
                result[k][selected]=v
            result["valid"][selected]=valid
            row_all[selected]=row;eq_all[selected]=equality;bracket_all[selected]=bracket
            num_all[selected]=numerical;aux_all[selected]=borderline
            for k in cert_all:
                cert_all[k][selected]=cert[k]
        output={k:v.reshape(shape+v.shape[1:]) for k,v in result.items()}
        output.update(row=row_all.reshape(shape),root_equality_px=eq_all.reshape(shape),bracketed=bracket_all.reshape(shape),
            numerical_root_valid=num_all.reshape(shape),row_is_auxiliary=aux_all.reshape(shape),
            boundary_auxiliary_available=aux_all.reshape(shape),status_code=status.reshape(shape),
            certificate={k:v.reshape(shape) for k,v in cert_all.items()},status_meanings=ROW_STATUS)
        return output

    def project_rs_jacobian(self,xyz,native_pts,**kwargs):
        out=self.project_rs(xyz,native_pts,**kwargs)
        row=out["row"]
        pts=np.asarray(native_pts)
        time=(pts-self.shot.first_native_pts)*self.shot.time_base_num/self.shot.time_base_den
        tau=self.camera_definition["signed_readout_seconds"];h=self.rig.image_height-1
        local=self.project_at_jacobian(xyz,time+tau*(row/h-.5))
        den=1-tau/h*local["jacobian_time"][...,1]
        with np.errstate(divide="ignore",invalid="ignore"):
            drdx=local["jacobian_world"][...,1,:]/den[...,None]
            jac=local["jacobian_world"]+local["jacobian_time"][...,None]*tau/h*drdx[...,None,:]
        out.update(jacobian_world=jac,row_jacobian_world=drdx,row_implicit_denominator=den)
        return out

    def project_rows(self,curve_ids,stations,native_pts,**kwargs):
        return self.project_rs(self.world_rows(curve_ids,stations),native_pts,**kwargs)


    def project_motion_jacobian(self,xyz,time):
        """Analytic pixel derivatives of local alpha,beta,log_f spline values."""
        xyz=np.asarray(xyz,dtype=np.float64);time=np.asarray(time,dtype=np.float64)
        shape=np.broadcast_shapes(xyz.shape[:-1],time.shape)
        xyz=np.broadcast_to(xyz,shape+(3,));time=np.broadcast_to(time,shape)
        out=self.project_at_jacobian(xyz,time)
        c=self.camera_definition;R=out["R_cw"]
        dR=np.stack((R@skew(c["pan_axis_world"]),skew(c["tilt_axis_camera"])@R,np.zeros_like(R)),axis=-3)
        world_tilt=np.einsum("...ji,j->...i",R,np.asarray(c["tilt_axis_camera"]))
        world_tilt_d=np.einsum("...cji,j->...ci",dR,np.asarray(c["tilt_axis_camera"]))
        raw=np.cross(c["pan_axis_world"],world_tilt);raw_d=np.cross(c["pan_axis_world"],world_tilt_d)
        length=np.linalg.norm(raw,axis=-1);unit=raw/length[...,None]
        unit_d=(raw_d-unit[...,None,:]*np.einsum("...i,...ci->...c",unit,raw_d)[...,None])/length[...,None,None]
        ray_d=np.einsum("...cij,...j->...ci",dR,xyz-np.asarray(c["pivot_world"]))
        ray_d[...,2,2]-=c["pupil_polynomial"][0]+2*c["pupil_polynomial"][1]*out["zoom_offset"]
        ray_d-=c["axis_spacing"]*(np.einsum("...cij,...j->...ci",dR,unit)+np.einsum("...ij,...cj->...ci",R,unit_d))
        q=out["camera_xyz"]
        with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
            x,y=np.moveaxis(q[...,:2]/q[...,2,None],-1,0)
            normalized_d=(ray_d[...,:2]-np.stack((x,y),axis=-1)[...,None,:]*ray_d[...,2,None])/q[...,2,None,None]
            r2=x*x+y*y;k1,k2,p1,p2=np.moveaxis(out["brown_k1_k2_p1_p2"],-1,0)
            radial=1+k1*r2+k2*r2*r2;g=k1+2*k2*r2
            J=np.empty(shape+(2,2));J[...,0,0]=radial+2*x*x*g+2*p1*y+6*p2*x
            J[...,0,1]=2*x*y*g+2*p1*x+2*p2*y;J[...,1,0]=J[...,0,1]
            J[...,1,1]=radial+2*y*y*g+6*p1*y+2*p2*x
            distorted=np.stack((x*radial+2*p1*x*y+p2*(r2+2*x*x),y*radial+p1*(r2+2*y*y)+2*p2*x*y),axis=-1)
            distorted_d=np.einsum("...ij,...cj->...ic",J,normalized_d)
            distorted_d[...,2]+=np.stack((x,y),axis=-1)*(c["k1_coefficients"][1]*r2+c["k2_coefficients"][1]*r2*r2)[...,None]
            focal=np.stack((out["K"][...,0,0],out["K"][...,1,1]),axis=-1)
            jac=focal[...,None]*distorted_d
            jac[...,2]+=focal*distorted+np.asarray(c["principal_zoom_slope"])
        out["jacobian_motion_values"]=jac
        return out

    def project_at_rig_jacobian(self,xyz,time):
        """Shared pivot, local lever and Brown/principal optics derivatives.

        Derivatives of the engineering two-axis frame manifold are intentionally
        not silently replaced by per-frame camera-center or orientation variables.
        """
        out=self.project_at_jacobian(xyz,time)
        q=out["camera_xyz"];R=out["R_cw"]
        ray_jac=out["jacobian_world"]@np.swapaxes(R,-1,-2)
        with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
            x,y=np.moveaxis(q[...,:2]/q[...,2,None],-1,0)
            r2=x*x+y*y;z=out["zoom_offset"]
            fx,fy=out["K"][...,0,0],out["K"][...,1,1]
            focal=np.stack((fx,fy),axis=-1)
            principal=out["K"][...,:2,2]
            distorted_pixels=out["uv"]-principal
            J=np.zeros(x.shape+(2,12))
            J[...,0,0]=.5*distorted_pixels[...,0];J[...,1,0]=-.5*distorted_pixels[...,1]
            J[...,0,1]=1.;J[...,1,2]=1.;J[...,0,3]=z;J[...,1,4]=z
            radial1=focal*np.stack((x,y),axis=-1)*r2[...,None]
            radial2=radial1*r2[...,None]
            J[...,5]=radial1;J[...,6]=radial1*z[...,None]
            J[...,7]=radial2;J[...,8]=radial2*z[...,None]
            J[...,9]=focal*np.stack((2*x*y,r2+2*y*y),axis=-1)
            J[...,10]=focal*np.stack((r2+2*x*x,2*x*y),axis=-1)
        lever=np.broadcast_to(np.asarray(self.camera_definition["lever_camera"]),q.shape).copy()
        z=out["zoom_offset"];pupil=self.camera_definition["pupil_polynomial"]
        lever[...,2]+=pupil[0]*z+pupil[1]*z*z
        left=so3_left_jacobian(self.model.state.get(self.rig.key("reference_rotation_vector")))
        head=ray_jac@(-skew(q+lever))@left
        out.update(jacobian_pivot=-out["jacobian_world"],jacobian_lever=-ray_jac,jacobian_optics_at_fixed_time=J,
            jacobian_reference_rotation=head,
            head_frame_parameterization="R0=Exp(v)Rseed,b=Exp(v)bseed,n=nseed; base axis structure is a staged estimate, not measured truth")
        return out


    def project_axis_chart_jacobian(self,xyz,time,*,step=1e-6):
        """Only the 3 axis-chart DOF use central differences during bring-up.

        Both sides construct exact unit/perpendicular camera frames. This is a
        reported numerical derivative of the same operator, not a second model.
        """
        if self.rig.axis_parameterization!='coupled_s2_perpendicular.v1':
            raise ValueError('minimal axis chart not active')
        if step<=0:
            raise ValueError('positive axis derivative step required')
        result={}
        for key in (self.rig.key('pan_axis_chart'),self.rig.key('tilt_axis_chart')):
            q=self.model.state.get(key);columns=[]
            for k in range(len(q)):
                plus=q.copy();minus=q.copy();plus[k]+=step;minus[k]-=step
                op=self.model.with_state(self.model.state.updated({key:plus})).operator(self.shot.id)
                om=self.model.with_state(self.model.state.updated({key:minus})).operator(self.shot.id)
                columns.append((op.project_at(xyz,time)['uv']-om.project_at(xyz,time)['uv'])/(2*step))
            result[key]=np.stack(columns,axis=-1)
        return {'jacobians_by_block':result,'method':'reported_central_difference_only_3_axis_chart_dof','step_radians':step}

    def control_support(self,time):
        shape=np.asarray(time).shape
        flat=np.asarray(time,dtype=np.float64).reshape(-1)
        valid=np.isfinite(flat)&(flat>=self.domain[0])&(flat<=self.domain[1])
        indices=np.full((len(flat),4),-1,dtype=np.int64);weights=np.full((len(flat),4),np.nan)
        if np.any(valid):
            design=BSpline.design_matrix(flat[valid],self.trajectory.t,3,extrapolate=False)
            if np.any(np.diff(design.indptr)!=4):
                raise ValueError("cubic trajectory requires exactly four explicit support entries")
            indices[valid]=design.indices.reshape(-1,4);weights[valid]=design.data.reshape(-1,4)
        return indices.reshape(shape+(4,)),weights.reshape(shape+(4,))

    def project_control_jacobian(self,xyz,native_pts,*,row_auxiliary=None):
        """Sparse four-control Jacobians; arrays [point,4,pixel,3].

        The semantic block is shared shot/S01/controls; each native optimizer
        view is offset 3*control_index, size 3 with the same shot owner.
        """
        if row_auxiliary is None:
            out=self.project_rs(xyz,native_pts)
            row=out["row"]
        else:
            out=self.project_with_row_auxiliary(xyz,native_pts,row_auxiliary)
            row=np.asarray(row_auxiliary)
        pts=np.asarray(native_pts);tau=self.camera_definition["signed_readout_seconds"];h=self.rig.image_height-1
        t=(pts-self.shot.first_native_pts)*self.shot.time_base_num/self.shot.time_base_den+tau*(row/h-.5)
        local=self.project_motion_jacobian(xyz,t)
        indices,weights=self.control_support(t)
        J=local["jacobian_motion_values"]
        den=1-tau/h*local["jacobian_time"][...,1]
        if row_auxiliary is None:
            J=J+local["jacobian_time"][...,None]*tau/h*(J[...,1,:]/den[...,None])[...,None,:]
        out.update(control_indices=indices,control_weights=weights,
            control_semantic_block_id=self.shot.control_block_id,
            control_component_offsets=3*indices,control_component_size=3,
            jacobian_controls=J[...,None,:,:]*weights[..., :,None,None])
        if row_auxiliary is not None:
            out["row_equality_jacobian_controls"]=-local["jacobian_motion_values"][...,None,1,:]*weights[..., :,None]
        return out


def compare_mount_families(model,*,probe_xyz=None,output=None):
    """Measured target simplification; retains the original replay candidate.

    With historical R/K/optics unchanged, find one P and one camera-local d by
    linear least squares to all native exposure centers. This is an initializer
    for the approved simpler mount, never a claim of equivalence or source fit.
    """
    replay=model.operator("S01")
    if replay.rig.mount_family!=REPLAY_MOUNT:
        raise ValueError("comparison requires explicitly identified extended replay candidate")
    t=replay.frame_time(replay.native_pts)
    original=replay.pose_at(t)
    rt=np.swapaxes(original["R_cw"],-1,-2)
    design=np.concatenate((np.broadcast_to(np.eye(3),(len(t),3,3)),rt),axis=-1).reshape(-1,6)
    rhs=original["C_w"].reshape(-1)
    solution,_,rank,singular=np.linalg.lstsq(design,rhs,rcond=None)
    changes={replay.rig.key("pivot"):solution[:3],replay.rig.key("lever"):solution[3:]}
    newstate=model.state.updated(changes,new_id="constant_lever_ls_initial@1")
    rigs=tuple(replace(r,mount_family=TARGET_MOUNT,source_status="explicit_simplification_candidate") if r.id==replay.rig.id else r for r in model.rigs)
    target=replace(model,id=model.id+"_constant_lever_candidate",revision=model.revision+1,state=newstate,rigs=rigs)
    simple=target.operator("S01")
    center_error=np.linalg.norm(simple.pose_at(t)["C_w"]-original["C_w"],axis=-1)
    report={"schema":"rinksolve.mount_family_comparison.v1","source_model":replay.identity,"candidate_model":simple.identity,
        "source_family":REPLAY_MOUNT,"target_family":TARGET_MOUNT,"frames":len(t),"linear_design_rank":int(rank),
        "linear_design_singular_values":singular.tolist(),"candidate_pivot_world":solution[:3].tolist(),"candidate_lever_camera":solution[3:].tolist(),
        "center_delta_scene_units":{"rms":float(np.sqrt(np.mean(center_error**2))),"maximum":float(center_error.max())},
        "equivalent":False,"same_measured_source_fit_claim":False,"old_optimizer_state_reused":False,
        "notes":"Only one shared P,d estimated. Pupil/axis-spacing extensions omitted explicitly; source replay state preserved. Values are unverified scene units, not metres."}
    if probe_xyz is not None:
        xyz,pts=probe_xyz
        original_uv=replay.project_rs(xyz,pts)
        simple_uv=simple.project_rs(xyz,pts)
        mask=original_uv["valid"]&simple_uv["valid"]
        delta=np.linalg.norm(simple_uv["uv"]-original_uv["uv"],axis=-1)
        report["true_rs_pixel_delta"]={"requested":int(delta.size),"both_certified":int(mask.sum()),
            "rms":float(np.sqrt(np.mean(delta[mask]**2))) if np.any(mask) else None,
            "maximum":float(delta[mask].max()) if np.any(mask) else None,
            "median":float(np.median(delta[mask])) if np.any(mask) else None}
    if output is not None:
        path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return target,report
