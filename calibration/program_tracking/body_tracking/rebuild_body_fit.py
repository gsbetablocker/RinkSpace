"""Fixed-native-camera articulation/contact ablation using shared root splines.

A CPU diagnostic extension of the existing full-programme body framework.
This does not add factors to the scene solver or update any camera parameter.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import cv2
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix,csr_matrix,diags,vstack
from scipy.sparse.linalg import lsqr
from scipy.spatial.transform import Rotation
from rinkspace.core.native import NativeCore
from .rebuild_prepare import load,digest
from .rebuild_body_geometry import CompactBody,POSE_DOFS,DOF_NAMES
from .fullprogramme_body_sequence import _basis,_second_derivative_basis,_difference_matrix,SOURCE_INDICES,SOURCE_SIGMA


class FixedNativeProjector:
    def __init__(self,packet,library):
        self.p=packet;self.core=NativeCore(library);self.scale=float(packet['metres_per_scene_unit'])
    def project(self,world_m,indices):
        points=np.asarray(world_m,float);frames=np.asarray(indices,int)
        times=np.repeat(self.p['motion_times'][frames],points.shape[1])
        r=self.core.project_motion(self.p['rig'],self.p['controls'],float(self.p['first_control_time']),
            float(self.p['control_dt']),1080,times,points.reshape(-1,3)/self.scale)
        if np.any(r['status']):raise ValueError('Native projection domain failure: '+str(np.unique(r['status'],return_counts=True)))
        return r['uv'].reshape(points.shape[0],points.shape[1],2)
    def with_jacobian(self,world,indices):
        uv=self.project(world,indices);j=np.empty((*uv.shape,3));step=1e-5
        for axis in range(3):
            a=world.copy();b=world.copy();a[:,:,axis]+=step;b[:,:,axis]-=step
            j[:,:,:,axis]=(self.project(a,indices)-self.project(b,indices))/(2*step)
        return uv,j


class ArticulatedSplineProblem:
    def __init__(self,root,mode,*,spacing=.16):
        self.root=Path(root);self.p=load(self.root/'input.npz');self.meta=json.loads((self.root/'input.json').read_text())
        self.mode=mode;self.articulate=mode!='fixed';self.contact=mode=='pose_contact';self.dof=len(POSE_DOFS)
        self.body=CompactBody(self.root/'compact_smplx.npz',self.p['shared_betas'],float(self.p['body_scale']))
        self.projection=FixedNativeProjector(self.p,self.meta['native_library'])
        self.evidence=json.loads((self.root/'contact_evidence.json').read_text(encoding='utf8'))
        # Native even exposures are the numerical sample; all outputs use 50 Hz.
        # The two diagnostic windows and their neighbours do not enter source fitting.
        self.hold=np.zeros(len(self.p['pts']),bool)
        for c in self.evidence['records']:
            if c['role']=='diagnostic_holdout':self.hold|=abs(self.p['pts']-c['pts'])<=1280
        eligible=np.flatnonzero(~self.hold)
        contacts=[int(np.searchsorted(self.p['pts'],c['pts'])) for c in self.evidence['records'] if c['role']=='fit']
        self.frames=np.unique(np.r_[eligible[::2],contacts]).astype(int);self.n=len(self.frames)
        self.contacts=[(int(np.flatnonzero(self.frames==int(np.searchsorted(self.p['pts'],c['pts'])))[0]),c)
            for c in self.evidence['records'] if c['role']=='fit']
        self.Bfull,self.knots=_basis(self.p['fit_time_seconds'],spacing);self.B=self.Bfull[self.frames];self.m=self.B.shape[1]
        self.bc=self.B.indices.reshape(self.n,4);self.bv=self.B.data.reshape(self.n,4)
        self.pose=self.p['pose_axis_angle'][self.frames];self.qprior=self.p['raw_root_prior_Q_world'][self.frames]
        self.local0,self.native0,self.blade0=self.body.forward(self.pose,np.zeros((self.n,self.dof)))
        self.indices=SOURCE_INDICES.copy();self.sig=SOURCE_SIGMA.copy()
        self.uv=self.p['source_uv'][self.frames][:,self.indices];self.qualified=self.p['source_qualified'][self.frames][:,self.indices]
        self.weight=self.qualified.astype(float)/self.sig
        self.uv=np.where(self.qualified[:,:,None],self.uv,0.)
        self.dt=.02;fulltimes=self.p['fit_time_seconds'];weights=np.r_[.01,np.full(len(fulltimes)-2,.02),.01]
        self.acc=_second_derivative_basis(fulltimes,self.knots).multiply(np.sqrt(weights/.02)[:,None]).tocsr()
        self.curvature=_difference_matrix(self.m)
        initial=np.empty((self.n,3))
        for i,frame in enumerate(self.frames):
            ids=self.indices[self.qualified[i]]
            K=np.array([[self.p['fx'][frame],0.,self.p['cx'][frame]],[0.,self.p['fy'][frame],self.p['cy'][frame]],[0.,0.,1.]])
            try:
                ok,rv,tv=cv2.solvePnP(self.local0[i,ids],self.p['source_uv'][frame,ids],K,None,flags=cv2.SOLVEPNP_SQPNP)
            except cv2.error:ok=False
            if not ok:initial[i]=np.nan;continue
            initial[i]=self.p['camera_centre'][frame]*self.projection.scale+self.p['camera_rotation'][frame].T@tv.ravel()
        valid=np.isfinite(initial).all(1)
        if not valid.any():raise ValueError('No root initializer')
        for axis in range(3):initial[:,axis]=np.interp(self.p['fit_time_seconds'][self.frames],self.p['fit_time_seconds'][self.frames[valid]],initial[valid,axis])
        matrix=vstack([self.B,self.curvature*.01]).tocsr()
        controls=np.stack([lsqr(matrix,np.r_[initial[:,a],np.zeros(self.curvature.shape[0])],atol=1e-10,btol=1e-10)[0] for a in range(3)],axis=1)
        self.size=self.m*(6+(self.dof if self.articulate else 0));self.x0=np.r_[controls.ravel(),np.zeros(self.size-3*self.m)]
        self.cache_x=None;self.cache=None;self.evaluations=0;self.jac_evaluations=0;self.start=time.perf_counter();self.log=None
        self.priors={'pixel_sigma_face_shoulders_hips_knees_px':12.,'pixel_sigma_ankles_px':24.,'root_acceleration_scale_m_s2':[15.,15.,25.],
            'root_rotation_control_prior_deg':30.,'root_rotation_control_curvature_deg':15.,'local_rotation_control_prior_deg':10.,
            'local_rotation_correction_acceleration_scale_deg_s2':400.,'contact_height_interval_m':self.evidence['height_interval_m'],
            'contact_height_scale_m':self.evidence['height_residual_scale_m'],'contact_region_scale_px':self.evidence['region_residual_scale_px'],
            'nonpenetration_allowance_m':.02,'nonpenetration_scale_m':.03,
            'meaning':'explicit engineering residual scales; not calibrated noise or measured uncertainty; identical common terms in all ablations'}

    def state(self,x,frames=None):
        frames=self.frames if frames is None else np.asarray(frames,int)
        B=self.B if np.array_equal(frames,self.frames) else self.Bfull[frames]
        P=B@x[:3*self.m].reshape(self.m,3);dr=B@x[3*self.m:6*self.m].reshape(self.m,3)
        Q=Rotation.from_rotvec(dr).as_matrix()@self.p['raw_root_prior_Q_world'][frames]
        delta=B@x[6*self.m:].reshape(self.m,self.dof) if self.articulate else np.zeros((len(frames),self.dof))
        coco,native,blade=self.body.forward(self.p['pose_axis_angle'][frames],delta)
        local=np.concatenate((coco[:,self.indices],blade.reshape(len(frames),10,3)),axis=1)
        world=np.einsum('nij,nkj->nki',Q,local)+P[:,None]
        return dict(P=P,Q=Q,delta=delta,local=local,world=world,coco=coco,native=native,blade=blade,dr=dr)

    def evaluate(self,x,*,jacobian=False):
        if self.cache_x is not None and np.array_equal(x,self.cache_x) and (not jacobian or self.cache['jac'] is not None):return self.cache
        tick=time.perf_counter();s=self.state(x);npts=s['world'].shape[1]
        if jacobian:uv,jpixel=self.projection.with_jacobian(s['world'],self.frames)
        else:uv=self.projection.project(s['world'],self.frames)
        residual=[];rows=[];cols=[];vals=[];cursor=0;parts={}
        if jacobian:
            # Exact local body model; finite difference only the nine physical rotations.
            localjac=np.zeros((self.n,npts,3,self.dof))
            if self.articulate:
                eps=1e-5
                for d in range(self.dof):
                    a=s['delta'].copy();b=s['delta'].copy();a[:,d]+=eps;b[:,d]-=eps
                    ga=self.body.forward(self.pose,a);gb=self.body.forward(self.pose,b)
                    la=np.concatenate((ga[0][:,self.indices],ga[2].reshape(self.n,10,3)),1)
                    lb=np.concatenate((gb[0][:,self.indices],gb[2].reshape(self.n,10,3)),1)
                    localjac[:,:,:,d]=np.einsum('nij,nkj->nki',s['Q'],(la-lb)/(2*eps))
            jq=np.empty((self.n,npts,3,3))
            for i in range(self.n):
                _,drv=cv2.Rodrigues(s['dr'][i]);dQ=np.einsum('rab,bc->rac',drv.reshape(3,3,3),self.qprior[i])
                jq[i]=np.einsum('rab,kb->kar',dQ,s['local'][i])
            jrot=np.einsum('nkpv,nkvd->nkpd',jpixel,jq)
            jpose=np.einsum('nkpv,nkvd->nkpd',jpixel,localjac)

        def add(res,name,frame_ids=None,dp=None,dq=None,da=None):
            nonlocal cursor
            flat=np.asarray(res,float).ravel();residual.append(flat);parts[name]=float(flat@flat/2)
            if jacobian and frame_ids is not None:
                f=np.asarray(frame_ids,int);count=len(flat);rr=np.repeat(np.arange(cursor,cursor+count),4)
                for deriv,offset,dimension in [(dp,0,3),(dq,3*self.m,3),(da,6*self.m,self.dof)]:
                    if deriv is None:continue
                    for k in range(dimension):
                        rows.append(rr);cols.append((self.bc[f]*dimension+k+offset).ravel())
                        vals.append((np.asarray(deriv)[:,k,None]*self.bv[f]).ravel())
            cursor+=len(flat)

        w=self.weight;err=(uv[:,:len(self.indices)]-self.uv)*w[:,:,None]
        frames_source=np.repeat(np.arange(self.n),len(self.indices)*2)
        add(err,'source_pixels',frames_source,
            (jpixel[:,:len(self.indices)]*w[:,:,None,None]).reshape(-1,3) if jacobian else None,
            (jrot[:,:len(self.indices)]*w[:,:,None,None]).reshape(-1,3) if jacobian else None,
            (jpose[:,:len(self.indices)]*w[:,:,None,None]).reshape(-1,self.dof) if jacobian and self.articulate else None)
        blades=s['world'][:,len(self.indices):].reshape(self.n,2,5,3)
        self.last_blade_heights=blades[:,:,:,2].min(2)
        if self.contact:
            # Unknown contact remains unknown. Nonpenetration alone does not label support.
            fi,foot=np.indices((self.n,2));fi,foot=fi.ravel(),foot.ravel();best=np.argmin(blades[:,:,:,2],axis=2).ravel()
            point=len(self.indices)+foot*5+best;height=blades[fi,foot,best,2]
            active=(height<-.02)&(~self.p['lower_body_inferred'][self.frames[fi]])
            gain=active.astype(float)/.03
            add(np.minimum(height+.02,0)*gain,'ice_nonpenetration',fi,
                np.c_[np.zeros((len(fi),2)),gain] if jacobian else None,
                jq[fi,point,2]*gain[:,None] if jacobian else None,
                localjac[fi,point,2]*gain[:,None] if jacobian else None)
            for ci,c in self.contacts:
                ft=c['foot'];k=int(np.argmin(blades[ci,ft,:,2]));point=len(self.indices)+ft*5+k;z=blades[ci,ft,k,2]
                lo,hi=self.evidence['height_interval_m'];hres=min(z-lo,0)+max(z-hi,0)
                g=float(z<lo or z>hi)/self.evidence['height_residual_scale_m']
                add([hres/self.evidence['height_residual_scale_m']],f'contact_height_{c["pts"]}',[ci],
                    np.array([[0.,0.,g]]) if jacobian else None,
                    jq[ci,point,2][None]*g if jacobian else None,
                    localjac[ci,point,2][None]*g if jacobian else None)
                region=np.array(c['region']);value=uv[ci,point];diff=np.minimum(value-region[:2],0)+np.maximum(value-region[2:],0)
                g=((value<region[:2])|(value>region[2:])).astype(float)/self.evidence['region_residual_scale_px']
                add(diff/self.evidence['region_residual_scale_px'],f'contact_region_{c["pts"]}',[ci,ci],
                    jpixel[ci,point]*g[:,None] if jacobian else None,
                    jrot[ci,point]*g[:,None] if jacobian else None,
                    jpose[ci,point]*g[:,None] if jacobian else None)
        def prior(matrix,controls,scales,offset,name):
            nonlocal cursor
            scales=np.broadcast_to(np.asarray(scales,float),(controls.shape[1],));r=(matrix@controls/scales).ravel()
            residual.append(r);parts[name]=float(r@r/2)
            if jacobian:
                co=matrix.tocoo();dim=controls.shape[1]
                rows.append((co.row[:,None]*dim+np.arange(dim)+cursor).ravel());cols.append((co.col[:,None]*dim+np.arange(dim)+offset).ravel());vals.append((co.data[:,None]/scales).ravel())
            cursor+=len(r)
        prior(self.acc,x[:3*self.m].reshape(self.m,3),(15.,15.,25.),0,'root_acceleration')
        rc=x[3*self.m:6*self.m].reshape(self.m,3)
        prior(self.curvature,rc,np.deg2rad(15.),3*self.m,'root_rotation_curvature')
        prior(diags(np.ones(self.m)),rc,np.deg2rad(30.),3*self.m,'root_rotation_prior')
        if self.articulate:
            ac=x[6*self.m:].reshape(self.m,self.dof)
            prior(diags(np.ones(self.m)),ac,np.deg2rad(10.),6*self.m,'local_pose_prior')
            prior(self.acc,ac,np.deg2rad(400.),6*self.m,'local_pose_correction_acceleration')
        r=np.concatenate(residual)
        J=coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(len(r),self.size)).tocsr() if jacobian else None
        self.cache_x=np.array(x,copy=True);self.cache=dict(residual=r,jac=J,state=s,uv=uv,parts=parts,seconds=time.perf_counter()-tick)
        if jacobian:self.jac_evaluations+=1
        else:
            self.evaluations+=1
            if self.log:self.log.write(json.dumps(dict(event='evaluation',mode=self.mode,evaluation=self.evaluations,objective=float(r@r/2),parts=parts,CPU_seconds=time.perf_counter()-self.start,eval_seconds=self.cache['seconds'],domain_failures=0))+'\n')
        return self.cache
    def fun(self,x):return self.evaluate(x)['residual']
    def jac(self,x):return self.evaluate(x,jacobian=True)['jac']


def export_exposure_state(problem,x):
    """Versioned simultaneous-world and per-row image markers; old fields survive."""
    frames=np.arange(len(problem.p['pts']));state=problem.state(x,frames);P=state['P'];G=P.copy();G[:,2]=0
    coco=np.einsum('nij,nkj->nki',state['Q'],state['coco'])+P[:,None]
    fields=dict(schema=np.array('rinkspace.body-exposure-result.v2'),P_world_mid=P,G_world_mid=G,COCO_world_mid=coco)
    if not hasattr(problem,'output_root_projection'):return fields
    root=problem.output_root_projection(x,frames);body=problem.output_projections(x,frames)
    source_seconds=problem.p['pts']/12800.;tau=float(problem.p['rig'][37])
    def row_time(uv):return source_seconds[:,None]+tau*(np.clip(uv[...,1],0,1079)/1079-.5)
    fields.update(P_uv_dynamic=root['uv'][:,0],P_world_at_row=root['world'][:,0],P_row_time=row_time(root['uv'])[:,0],
        P_inside_exposure_rows=root['inside_exposure_rows'][:,0],COCO_world_at_row=body['world'][:,:17],COCO_row_time=row_time(body['uv'][:,:17]),
        COCO_inside_exposure_rows=body['inside_exposure_rows'][:,:17],blade_world_at_row=body['world'][:,17:].reshape(len(frames),2,5,3),
        blade_row_time=row_time(body['uv'][:,17:]).reshape(len(frames),2,5),blade_inside_exposure_rows=body['inside_exposure_rows'][:,17:].reshape(len(frames),2,5))
    return fields


def run(root,mode,*,warm=None,spacing=.16,body_motion='exposure',exposure_nodes=3,output=None):
    root=Path(root);out=Path(output) if output else root/mode
    if (out/'result.npz').exists() or (out/'fit_progress.log').exists():raise FileExistsError('Preserve existing experiment '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    if body_motion=='exposure':
        from .rebuild_exposure_body import ExposureArticulatedSplineProblem
        problem=ExposureArticulatedSplineProblem(root,mode,spacing=spacing,exposure_nodes=exposure_nodes)
    else:problem=ArticulatedSplineProblem(root,mode,spacing=spacing)
    if warm:
        z=load(warm)
        if not np.array_equal(z['position_knots'],problem.knots):raise ValueError('Warm spline basis differs')
        problem.x0[:6*problem.m]=z['x'][:6*problem.m]
        if problem.articulate and len(z['x'])==problem.size:problem.x0=z['x'].copy()
    config=dict(mode=mode,CPU_only=True,frames=len(problem.p['pts']),numerical_frames=problem.n,spline_spacing_s=spacing,
        variables=problem.size,local_dofs=list(DOF_NAMES) if problem.articulate else [],priors=problem.priors,
        exposure_nodes=exposure_nodes if body_motion=='exposure' else None,export_schema='rinkspace.body-exposure-result.v2',
        warm_start=None if warm is None else dict(path=str(Path(warm).resolve()),sha256=digest(warm)),
        export_time_semantics=dict(P_world_mid='common source reference time',G_world_mid='vertical projection at common reference time',P_uv='legacy: mid-world pelvis held fixed during camera readout',P_uv_dynamic='moving pelvis at its predicted image row',P_row_time='absolute source seconds; rows outside image flagged and bounded',COCO_world_mid='simultaneous model at source reference',blade_world_at_row='each blade point at its own predicted row; not simultaneous'),
        input_sha256=digest(root/'input.npz'),contact_evidence_sha256=digest(root/'contact_evidence.json'),
        body_model_sha256=digest(root/'compact_smplx.npz'),projection=getattr(problem,'body_motion_model','same native complete camera; body frozen inside exposure'),
        fit_diagnostic_windows_excluded_source_seconds=[[31.96,32.16],[39.90,40.10]],camera_and_scene_frozen=True,
        tolerance=dict(ftol=1e-7,xtol=1e-7,gtol=1e-5),final_independent_3d_truth_available=False)
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    with (out/'fit_progress.log').open('w',encoding='utf8',buffering=1) as log:
        problem.log=log;log.write(json.dumps(dict(event='start',**config))+'\n')
        initial=problem.evaluate(problem.x0);initial_cost=float(initial['residual']@initial['residual']/2)
        initial_export=export_exposure_state(problem,problem.x0)
        np.savez_compressed(out/'initial_projection_only.npz',x=problem.x0,position_knots=problem.knots,pts=problem.p['pts'],**initial_export)
        (out/'initial_projection_only.json').write_text(json.dumps(dict(cost=initial_cost,parts=initial['parts'],parameters_refitted=False,exposure_nodes=exposure_nodes),indent=2),encoding='utf8')
        def accepted(x):
            v=problem.evaluate(x);log.write(json.dumps(dict(event='accepted_iteration',objective=float(v['residual']@v['residual']/2),parts=v['parts'],elapsed_seconds=time.perf_counter()-problem.start))+'\n')
            np.savez_compressed(out/'last_accepted.pending.npz',x=x,position_knots=problem.knots)
            (out/'last_accepted.pending.npz').replace(out/'last_accepted.npz')
        fit=least_squares(problem.fun,problem.x0,jac=problem.jac,method='trf',tr_solver='lsmr',x_scale='jac',
            ftol=1e-7,xtol=1e-7,gtol=1e-5,callback=accepted,verbose=0,tr_options={'atol':1e-6,'btol':1e-6})
        problem.log=None
        final=problem.evaluate(fit.x);allframes=np.arange(len(problem.p['pts']));state=problem.state(fit.x,allframes)
        coco_world=np.einsum('nij,nkj->nki',state['Q'],state['coco'])+state['P'][:,None]
        native_world=np.einsum('nij,nkj->nki',state['Q'],state['native'])+state['P'][:,None]
        blade_world=np.einsum('nij,nfkj->nfki',state['Q'],state['blade'])+state['P'][:,None,None]
        if hasattr(problem,'output_projections'):
            exposure=problem.output_projections(fit.x,allframes)
            body_uv=exposure['uv'][:,:17];blade_uv=exposure['uv'][:,17:].reshape(len(allframes),2,5,2)
            blade_world=exposure['world'][:,17:].reshape(len(allframes),2,5,3)
        else:
            body_uv=problem.projection.project(coco_world,allframes);blade_uv=problem.projection.project(blade_world.reshape(len(allframes),10,3),allframes).reshape(len(allframes),2,5,2)
        G=state['P'].copy();G[:,2]=0.;pg=problem.projection.project(np.stack((state['P'],G),axis=1),allframes)
        err=np.linalg.norm(body_uv-problem.p['source_uv'],axis=2);mask=problem.p['source_qualified']
        def stats(a):return dict(n=len(a),median=float(np.median(a)),p95=float(np.percentile(a,95)),maximum=float(np.max(a))) if len(a) else dict(n=0)
        report=dict(mode=mode,success=bool(fit.success),status=int(fit.status),message=fit.message,cost=float(fit.cost),
            optimality=float(fit.optimality),numerically_converged=bool(fit.optimality<=1e-5),initial_cost_at_same_warm_parameters=initial_cost,export_schema='rinkspace.body-exposure-result.v2',nfev=fit.nfev,njev=fit.njev,elapsed_seconds=time.perf_counter()-problem.start,
            parts=final['parts'],all_source_detector_reprojection_px=stats(err[mask]),
            heldout_source_detector_reprojection_px=stats(err[mask&problem.hold[:,None]]),
            local_correction_absolute_deg=stats(np.abs(np.rad2deg(state['delta'])).ravel()),
            visible_body_pelvis_height_conditional_m=stats(state['P'][~problem.p['lower_body_inferred'],2]),
            contact_state_unknown_frames=len(allframes)-len(problem.evidence['records']),
            source_frames=len(allframes),camera_parameters_changed=False,scene_parameters_changed=False,
            source_3d_accuracy_claim=False,rolling_shutter_body_motion=getattr(problem,'body_motion_model','frozen approximation; separate postfit diagnostic required'))
        new_fields=export_exposure_state(problem,fit.x)
        np.savez_compressed(out/'result.npz',**new_fields,x=fit.x,position_knots=problem.knots,pts=problem.p['pts'],P_world_m=state['P'],G_world_m=G,Q_world=state['Q'],
            delta_local_rad=state['delta'],COCO_world_m=coco_world,native22_world_m=native_world,blade_world_m=blade_world,
            COCO_uv=body_uv,blade_uv=blade_uv,P_uv=pg[:,0],G_uv=pg[:,1],source_fit_frames=problem.frames,
            source_qualified=mask,heldout_frames=problem.hold,metres_per_scene_unit=problem.p['metres_per_scene_unit'],
            inferred=problem.p['lower_body_inferred']|problem.p['pose_interpolated'])
        (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');log.write(json.dumps(dict(event='complete',**report))+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--mode',choices=['fixed','pose','pose_contact'],required=True);ap.add_argument('--warm',type=Path);ap.add_argument('--spacing',type=float,default=.16);ap.add_argument('--body-motion',choices=['exposure','frozen'],default='exposure');ap.add_argument('--exposure-nodes',type=int,choices=[3,5],default=3);ap.add_argument('--output',type=Path);a=ap.parse_args();run(a.root,a.mode,warm=a.warm,spacing=a.spacing,body_motion=a.body_motion,exposure_nodes=a.exposure_nodes,output=a.output)

if __name__=='__main__':main()
