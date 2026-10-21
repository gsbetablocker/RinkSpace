"""Offline multi-exposure, source-qualified Brown-hinge initialization.

Only the existing shared radial/tangential hinge coefficients change. Pose, signed RS,
base optics, scene, source pixels and all production parameter blocks stay fixed.
Each finite line is an eliminated straightness nuisance, never an installed
private scene or an observed world station. Two real flank pixels are inverted
at their own source rows; the existing straight-line +/- local-halfwidth model
is then profiled with independent station per side and width per source section.
"""
from __future__ import annotations
import argparse,copy,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,minimize,OptimizeResult
from .io_utils import read_json,read_jsonl,write_json,file_sha256
from .joint_model import graph_from_joint,as_event_view
from .camera_graph import CameraGraph
from .wide_optics_candidate import _camera_geometry
from rinkfit_start.native_scene import NativeScene


def _brown(xy,c):
    x,y=xy.T;k1,k2,p1,p2=c.T;r=x*x+y*y;L=1+k1*r+k2*r*r
    return np.c_[x*L+2*p1*x*y+p2*(r+2*x*x),y*L+p1*(r+2*y*y)+2*p2*x*y]


def _lower(c,radius):
    k1,k2,p1,p2=c.T;q=radius*radius
    def minimum(a,b):
        t=np.divide(-a,2*b,out=np.zeros_like(a),where=b!=0)
        return np.minimum(np.minimum(1.,1+a*q+b*q*q),np.where((b>0)&(t>0)&(t<q),1+a*t+b*t*t,np.inf))
    return np.minimum(minimum(k1,k2),minimum(3*k1,5*k2))-6*np.hypot(p1,p2)*radius


def _inverse(target,c,fxy,radius):
    """Same positive whole-disk branch and decreasing-step logic as native ray."""
    lower=_lower(c,radius);valid=np.isfinite(target).all(1)&(lower>0)
    reason=np.full(len(target),'unique_in_certified_disk',object);reason[~valid]='uncertified_whole_Brown_disk'
    xy=target.copy();rr=np.linalg.norm(xy,axis=1);outside=rr>radius;xy[outside]*=(radius[outside]/(2*rr[outside]))[:,None]
    eps=32*np.finfo(float).eps*np.maximum(1.,fxy.max(1))*(1+np.linalg.norm(target,axis=1))
    while True:
        err=np.linalg.norm((_brown(xy,c)-target)*fxy,axis=1);active=valid&(err>eps)
        if not active.any():break
        ids=np.flatnonzero(active);x,y=xy[ids].T;k1,k2,p1,p2=c[ids].T;r=x*x+y*y;L=1+k1*r+k2*r*r;A=k1+2*k2*r
        a=L+2*x*x*A+2*p1*y+6*p2*x;b=2*x*y*A+2*p1*x+2*p2*y;d=L+2*y*y*A+6*p1*y+2*p2*x;det=a*d-b*b
        good=np.isfinite(det)&(a>0)&(det>0);bad=ids[~good];valid[bad]=False;reason[bad]='inverse_local_Jacobian_not_positive';ids=ids[good]
        if not len(ids):continue
        a,b,d,det=a[good],b[good],d[good],det[good];r0=_brown(xy[ids],c[ids])-target[ids]
        step=np.c_[(d*r0[:,0]-b*r0[:,1])/det,(a*r0[:,1]-b*r0[:,0])/det]
        pending=np.arange(len(ids));fraction=np.ones(len(ids))
        while len(pending):
            jj=ids[pending];q=xy[jj]-fraction[pending,None]*step[pending];changed=np.any(q!=xy[jj],axis=1)
            stalled=pending[~changed];valid[ids[stalled]]=False;reason[ids[stalled]]='inverse_no_representable_decreasing_in_disk_step'
            errq=np.linalg.norm((_brown(q,c[jj])-target[jj])*fxy[jj],axis=1)
            ok=changed&(np.linalg.norm(q,axis=1)<=radius[jj])&(errq<err[jj]);xy[jj[ok]]=q[ok]
            pending=pending[changed&~ok];fraction[pending]*=.5
    valid&=np.linalg.norm(xy,axis=1)<=radius
    return xy,valid,reason,lower


def _stats(values,denominator=None):
    x=np.asarray(values,float);good=x[np.isfinite(x)];n=x.size if denominator is None else int(denominator)
    return dict(denominator=n,finite=int(good.size),failed_or_unavailable=int(n-good.size),rms=float(np.sqrt(np.mean(good*good)))if len(good)else None,median=float(np.median(good))if len(good)else None,p95=float(np.quantile(good,.95))if len(good)else None,max=float(good.max())if len(good)else None)


class MultiSourceWideProfile:
    """Small fixed-pose grouped evaluator; no packet, cached solver or RGB IO."""
    def __init__(self,model,sections):
        self.model=model;self.graph=CameraGraph(graph_from_joint(model));self.cameras={s:self.graph.native_camera(s)for s in {r['shot_id']for r in sections}}
        if any(model['shots'][s]['rig_id']!='main'for s in self.cameras):raise ValueError('Only the same verified main optical rig may share these coefficients')
        self.sections=sections;self.uv=np.asarray([v for r in sections for v in r['actual_flank_uv']],float);self.groups=defaultdict(list)
        for i,r in enumerate(sections):self.groups[(r['shot_id'],int(r['native_pts']),r['target_id'])].append(i)
        self.group_keys=list(self.groups);self.group_indices=[np.asarray(self.groups[k],int)for k in self.group_keys]
        self.family_exposures=Counter((s,t)for s,p,t in self.group_keys);self.family_count=len(self.family_exposures)
        n=len(self.uv);self.R=np.empty((n,3,3));self.C=np.empty((n,3));self.logf=np.empty(n);self.base=np.empty((n,4));self.fxy=np.empty((n,2));self.pp=np.empty((n,2));self.times=np.empty(n);self.frame=np.empty(n);self.radius=np.empty(n)
        for sid,cam in self.cameras.items():
            ids=np.array([2*i+j for i,r in enumerate(sections)if r['shot_id']==sid for j in (0,1)],int);uv=self.uv[ids];pts=np.array([sections[i//2]['native_pts']for i in ids],np.int64);frame=(pts-model['shots'][sid]['time_origin_pts'])/12800.;times=frame+cam.tau*(uv[:,1]/1079.-.5)
            R,C,lf,z=_camera_geometry(cam,times);self.R[ids]=R;self.C[ids]=C;self.logf[ids]=lf;self.frame[ids]=frame;self.times[ids]=times;o=cam.optics
            self.base[ids]=np.c_[np.polynomial.polynomial.polyval(z,o['k1']),np.polynomial.polynomial.polyval(z,o['k2']),np.full(len(ids),o['p1']),np.full(len(ids),o['p2'])]
            self.fxy[ids]=np.exp(lf[:,None])*np.exp(np.array([o.get('eta',0),-o.get('eta',0)])/2);self.pp[ids]=np.asarray(o['principal_point'])+z[:,None]*np.asarray(o.get('principal_linear',[0.,0.]));self.radius[ids]=o['radius_max']
        self.boundary=float(next(iter(self.cameras.values())).wide_boundary);self.basis=np.maximum(self.boundary-self.logf,0.)**2;self.target=(self.uv-self.pp)/self.fxy;self.initial=np.array(next(iter(self.cameras.values())).wide_coefficients);self.scene=NativeScene(as_event_view(model,'s01'))
        self.weight=np.zeros(len(sections))
        for (sid,pts,target),ids in zip(self.group_keys,self.group_indices):self.weight[ids]=1/np.sqrt(self.family_count*self.family_exposures[(sid,target)]*len(ids))

    def project(self,X,coeff):
        Y=np.einsum('nij,nj->ni',self.R,X-self.C);xy=Y[:,:2]/Y[:,2:3];uv=self.pp+self.fxy*_brown(xy,coeff)
        good=(Y[:,2]>0)&np.isfinite(uv).all(1)&(np.linalg.norm(xy,axis=1)<=self.radius)
        return uv,good

    def evaluate(self,extension,details=False):
        coeff=self.base+self.basis[:,None]*np.asarray(extension);xy,valid,reason,lower=_inverse(self.target,coeff,self.fxy,self.radius)
        direction=np.einsum('nji,nj->ni',self.R,np.c_[xy,np.ones(len(xy))]);depth=-self.C[:,2]/direction[:,2];valid&=np.isfinite(depth)&(depth>0)&(self.C[:,2]>0)
        X=self.C+depth[:,None]*direction;X[:,2]=0;raw=X.reshape(-1,2,3);trial=X.copy();result=[];section_valid=valid.reshape(-1,2).all(1);widths=np.full(len(self.sections),np.nan);sides=np.zeros(len(self.sections),int);station=np.full((len(self.sections),2),np.nan)
        for key,ids in zip(self.group_keys,self.group_indices):
            if not section_valid[ids].all():
                result.append(dict(key=list(key),sections=len(ids),shape_constraint_dof=max(0,len(ids)-2),valid=False,reason='one_or_more_original_flank_inverse_or_forward_ice_rays_failed'));continue
            points=raw[ids,:,:2].mean(1)
            if len(ids)==1:
                normal=raw[ids[0],1,:2]-raw[ids[0],0,:2];norm=np.linalg.norm(normal)
                if norm==0:normal=np.array([1.,0.])
                else:normal/=norm
            else:
                _,_,Vt=np.linalg.svd(points-points.mean(0),full_matrices=False);normal=Vt[-1]
            tangent=np.array([-normal[1],normal[0]]);c=float(points.mean(0)@normal);shift=points@normal-c
            delta=(raw[ids,1,:2]-raw[ids,0,:2])@normal;w=np.abs(delta)/2;widths[ids]=w;sides[ids]=np.where(delta>=0,1,-1)
            # Each real boundary has its own tangent station. Only the local
            # halfwidth is shared within that source-confirmed section.
            flat=(2*ids[:,None]+np.arange(2)).reshape(-1);trial[flat,:2]-=np.repeat(shift,2)[:,None]*normal;station[ids]=np.einsum('nsi,i->ns',trial.reshape(-1,2,3)[ids,:,:2],tangent)
            width_ok=np.isfinite(w)&(w>=1e-12)&(w<=1.);section_valid[ids]&=width_ok
            result.append(dict(key=list(key),sections=len(ids),shape_constraint_dof=max(0,len(ids)-2),valid=bool(width_ok.all()),normal_xy=normal.tolist(),offset_L1=c,finite_station_interval_L1=[float(station[ids].min()),float(station[ids].max())],minimum_halfwidth_L1=float(w.min()),maximum_halfwidth_L1=float(w.max()),width_failed=int((~width_ok).sum()),world_normal_shift_L1=shift.tolist()))
        pred,pvalid=self.project(trial,coeff);section_valid&=pvalid.reshape(-1,2).all(1);residual=(pred-self.uv).reshape(-1,2,2);residual[~section_valid]=1e6
        weighted=residual*self.weight[:,None,None]
        if not details:return weighted.ravel()
        inside=np.asarray(self.scene.containment(trial[:,:2]))<=0
        return dict(coefficients=np.asarray(extension).tolist(),groups=result,residual_observed_source_row_px=residual,predicted_observed_row_uv=pred,weighted_residual=weighted,raw_inverse_world=raw,profiled_interface_world=trial,section_valid=section_valid,inverse_valid=valid,inverse_reason=reason,Brown_disk_lower=lower,inside_finite_ice=inside,halfwidth=widths,side_assignment=sides,tangent_station=station)

    def summarize(self,extension):
        d=self.evaluate(extension,True);rsuv=np.full_like(self.uv,np.nan);rsvalid=np.zeros(len(self.uv),bool);rsbracket=np.zeros(len(self.uv),bool);rscert=np.zeros(len(self.uv),bool)
        for sid,old in self.cameras.items():
            cam=copy.deepcopy(old);cam.wide_coefficients=np.asarray(extension);ids=np.array([2*i+j for i,r in enumerate(self.sections)if r['shot_id']==sid for j in (0,1)],int)
            r=cam.project_rs(d['profiled_interface_world'][ids],self.frame[ids]);rsuv[ids]=r['uv'];rsvalid[ids]=r['valid'];rsbracket[ids]=r['bracketed'];rscert[ids]=r['certificate']['certified']
        good=d['section_valid'].repeat(2)&rsvalid;error=np.linalg.norm(rsuv-self.uv,axis=1);error[~good]=np.nan;observed=np.linalg.norm(d['residual_observed_source_row_px'].reshape(-1,2),axis=1);observed[~d['section_valid'].repeat(2)]=np.nan
        rows=[]
        for i,r in enumerate(self.sections):
            ids=np.array([2*i,2*i+1]);rows.append(dict(source_section_id=r['source_section_id'],source_row_ids=r['source_row_ids'],shot_id=r['shot_id'],native_pts=r['native_pts'],target_id=r['target_id'],source_families=r['source_families'],original_flank_uv=r['actual_flank_uv'],source_row_times_seconds=self.times[ids].tolist(),source_row_focal_native_px=np.exp(self.logf[ids]).tolist(),source_hinge_basis=self.basis[ids].tolist(),inverse_valid=d['inverse_valid'][ids].tolist(),inverse_reason=d['inverse_reason'][ids].tolist(),profile_valid=bool(d['section_valid'][i]),true_RS_valid=rsvalid[ids].tolist(),RS_bracketed=rsbracket[ids].tolist(),RS_whole_row_certified=rscert[ids].tolist(),observed_row_residual_px=[None if not np.isfinite(v)else float(v)for v in observed[ids]],true_RS_residual_px=[None if not np.isfinite(v)else float(v)for v in error[ids]],inside_finite_ice=d['inside_finite_ice'][ids].tolist(),profiled_local_halfwidth_L1=None if not np.isfinite(d['halfwidth'][i])else float(d['halfwidth'][i]),allowed_source_sides=[int(-d['side_assignment'][i]),int(d['side_assignment'][i])]))
        bygroup=[]
        for g,ids in zip(d['groups'],self.group_indices):
            fi=(2*ids[:,None]+np.arange(2)).reshape(-1);q=copy.deepcopy(g);q.update(original_flank_denominator=len(fi),source_focal_range_native_px=np.exp([self.logf[fi].min(),self.logf[fi].max()]).tolist(),hinge_active_flanks=int(np.count_nonzero(self.basis[fi])),true_RS_error_px=_stats(error[fi]),observed_row_error_px=_stats(observed[fi]),outside_finite_ice_flanks=int((~d['inside_finite_ice'][fi]).sum()),RS_failed_flanks=int((~good[fi]).sum()));bygroup.append(q)
        byentity=[]
        for sid,target in self.family_exposures:
            fi=np.array([2*i+j for i,r in enumerate(self.sections)if r['shot_id']==sid and r['target_id']==target for j in (0,1)],int);byentity.append(dict(shot_id=sid,target_id=target,exposures=self.family_exposures[(sid,target)],original_flank_denominator=len(fi),focal_native_px_range=np.exp([self.logf[fi].min(),self.logf[fi].max()]).tolist(),true_RS_error_px=_stats(error[fi]),observed_row_error_px=_stats(observed[fi]),outside_finite_ice_flanks=int((~d['inside_finite_ice'][fi]).sum())))
        bands=[]
        for name,mask in [('below_original_1734_fit',np.exp(self.logf)<1733.7172177279342),('original_1734_fit_band',(np.exp(self.logf)>=1733.7172177279342)&(np.exp(self.logf)<=1734.0056315754648)),('above_original_fit_below_hinge',(np.exp(self.logf)>1734.0056315754648)&(self.logf<self.boundary)),('hinge_zero',self.logf>=self.boundary)]:
            bands.append(dict(cohort=name,original_flank_denominator=int(mask.sum()),true_RS_error_px=_stats(error[mask]),observed_row_error_px=_stats(observed[mask]),outside_finite_ice_flanks=int((~d['inside_finite_ice'][mask]).sum())))
        return dict(coefficients=list(map(float,extension)),source_sections=len(self.sections),original_flank_denominator=len(self.uv),invalid_inverse_or_positive_plane_flanks=int((~d['inverse_valid']).sum()),invalid_profile_sections=int((~d['section_valid']).sum()),true_RS_failed_flanks=int((~good).sum()),outside_finite_ice_flanks=int((~d['inside_finite_ice']).sum()),whole_source_Brown_lower_min=float(d['Brown_disk_lower'].min()),criterion_half_square=float(np.sum(d['weighted_residual']**2)/2),true_RS_error_px=_stats(error),observed_row_error_px=_stats(observed),per_entity=byentity,per_focal_band=bands,per_exposure_entity=bygroup,source_rows=rows),d


def _read_sections(path):
    result={};raw_count=0
    for r in read_jsonl(path):
        raw_count+=1
        if not r.get('source_geometry_qualified')or r.get('physical_surface')!='ice':raise ValueError('Unqualified physical source in optical input')
        uv=np.asarray(r['actual_flank_uv'],float)
        if uv.shape!=(2,2)or not np.isfinite(uv).all()or np.any(uv<0)or np.any(uv>np.array([1919.,1079.])):raise ValueError('Original actual source flank UV missing or outside native sensor')
        key=r.get('paired_section_id',r['source_row_id']);entry=result.setdefault(key,dict(source_section_id=key,source_row_ids=[],shot_id=r['shot_id'],native_pts=r['native_pts'],target_id=r['target_id'],actual_flank_uv=uv.tolist(),source_families=[]))
        if entry['actual_flank_uv']!=uv.tolist()or any(entry[k]!=r[k]for k in ('shot_id','native_pts','target_id')):raise ValueError('Source section identity/UV changed')
        entry['source_row_ids'].append(r['source_row_id']);fam=r['qualification_bundle']
        if fam not in entry['source_families']:entry['source_families'].append(fam)
    return list(result.values()),raw_count


def build_multisource_wide_candidate(model_file,source_index_file,output,*,inventory_file,qualification_QA_file,fit_tangential=False,fit_method='least_squares'):
    out=Path(output)
    if out.exists()and any(out.iterdir()):raise FileExistsError('New immutable optical candidate directory required')
    out.mkdir(parents=True,exist_ok=True);model=read_json(model_file);inventory=read_json(inventory_file);qa=read_json(qualification_QA_file)
    if inventory['source_index']['sha256']!=file_sha256(source_index_file)or qa['inventory']['sha256']!=file_sha256(inventory_file):raise ValueError('Source index/qualification inventory pin mismatch')
    if not qa.get('source_images_not_read_or_decoded')or not qa.get('source_UV_unchanged'):raise ValueError('No frozen existing-source receipt QA')
    sections,nraw=_read_sections(source_index_file);profile=MultiSourceWideProfile(model,sections);before,bdata=profile.summarize(profile.initial)
    initial=profile.initial.copy() if fit_tangential else profile.initial[:2].copy();trajectory=[]
    def residual(p):
        ext=np.asarray(p) if fit_tangential else np.r_[p,profile.initial[2:]];r=profile.evaluate(ext);trajectory.append(dict(fitted_coefficients=p.tolist(),half_square=float(r@r/2)));return r
    if fit_method=='least_squares':
        fit=least_squares(residual,initial,ftol=1e-10,xtol=1e-10,gtol=1e-7)
    elif fit_method=='physical_slsqp':
        all_main={s:copy.deepcopy(profile.graph.native_camera(s))for s in model['shots']if model['shots'][s]['rig_id']=='main'}
        cached={}
        def state(p):
            key=np.asarray(p,float).tobytes()
            if cached.get('key')!=key:
                ext=np.asarray(p)if fit_tangential else np.r_[p,profile.initial[2:]];d=profile.evaluate(ext,True);raw=(d['predicted_observed_row_uv']-profile.uv).reshape(-1,2,2)*profile.weight[:,None,None];cached.update(key=key,extension=ext,detail=d,raw_residual=raw)
            return cached
        def physical_constraints(p):
            c=state(p);d=c['detail'];X=d['profiled_interface_world'];Y=np.einsum('nij,nj->ni',profile.R,X-profile.C);r2=np.sum((Y[:,:2]/Y[:,2:3])**2,axis=1);g=[]
            for cam in all_main.values():
                cam.wide_coefficients=np.asarray(c['extension']);g.append(float(cam.optical_domain_certificate()['Brown_Jacobian_eigenvalue_lower']))
            g.extend([float(np.min(profile.radius**2-r2)),float(np.min(Y[:,2])),float(np.nanmin(d['halfwidth'])-1e-12),float(1.-np.nanmax(d['halfwidth'])),float(np.min(-profile.scene.containment(X[:,:2])))])
            return np.asarray(g)
        def objective(p):
            c=state(p);d=c['detail'];r=c['raw_residual'];value=float(np.sum(r*r)/2)
            if not d['inverse_valid'].all()or not np.isfinite(value):value=1e12
            trajectory.append(dict(fitted_coefficients=np.asarray(p).tolist(),half_square=value,physical_constraints=physical_constraints(p).tolist(),inverse_failures=int((~d['inverse_valid']).sum())))
            return value
        opt=minimize(objective,initial,method='SLSQP',constraints=[dict(type='ineq',fun=physical_constraints)],options=dict(ftol=1e-10))
        step=np.cbrt(np.finfo(float).eps)*(1+np.abs(opt.x));J=[]
        for k in range(len(opt.x)):
            a=opt.x.copy();b=opt.x.copy();a[k]+=step[k];b[k]-=step[k];J.append((state(a)['raw_residual'].ravel()-state(b)['raw_residual'].ravel())/(2*step[k]))
        J=np.asarray(J).T;r=state(opt.x)['raw_residual'].ravel();fit=OptimizeResult(x=opt.x,jac=J,success=opt.success,message=opt.message,nfev=opt.nfev,njev=opt.njev,optimality=float(np.max(np.abs(J.T@r))),physical_constraints=physical_constraints(opt.x).tolist(),jacobian_scope='central FD of continuous original-flank profile mapping, no invalid-state sentinel; active physical constraints may remove locally feasible directions')
    else:raise ValueError('Unknown offline optical initialization method')
    extension=np.asarray(fit.x) if fit_tangential else np.r_[fit.x,profile.initial[2:]];after,adata=profile.summarize(extension)
    candidate=copy.deepcopy(model);meta=candidate['rigs']['main']['camera']['brown_wide_extension'];old=copy.deepcopy(meta);meta['coefficients']=extension.tolist();meta['calibration_status']='candidate_fixed_pose_multiexposure_'+('radial_tangential'if fit_tangential else'radial')+'_plumbline_profile';meta['hardware_calibration_claimed']=False;meta['tangential_extension_identified']=False;meta['outside_observed_wide_neighborhood']='unverified extrapolation; candidate only, requires complete shared-source audit';meta['source_support']=dict(source_index=dict(file=str(Path(source_index_file).resolve()),sha256=file_sha256(source_index_file)),inventory=dict(file=str(Path(inventory_file).resolve()),sha256=file_sha256(inventory_file)),source_receipt_QA=dict(file=str(Path(qualification_QA_file).resolve()),sha256=file_sha256(qualification_QA_file)))
    if candidate['blocks']!=model['blocks']or candidate['shots']!=model['shots']or candidate['materials']!=model['materials']:raise AssertionError('Physical parameter blocks, shots or geometry changed')
    first=profile.graph.native_camera('s01');zero=float(first.controls[:,2].min())>=profile.boundary
    if not zero:raise ValueError('Existing hinge is active on allocated S01 cubic hull; S01 cannot be protected by this candidate')
    singular=np.linalg.svd(fit.jac,compute_uv=False);reasons=[]
    if not fit.success:reasons.append('optimizer_not_converged')
    if after['invalid_profile_sections']or after['true_RS_failed_flanks']:reasons.append('one_or_more_complete_original_source_sections_or_true_RS_fail')
    if after['outside_finite_ice_flanks']:reasons.append('one_or_more_derived_source_interface_points_outside_finite_current_rink')
    if not after['criterion_half_square']<before['criterion_half_square']:reasons.append('same_source_profile_criterion_not_improved')
    cert={}
    graph=CameraGraph(graph_from_joint(candidate))
    for sid in model['shots']:
        if model['shots'][sid]['rig_id']=='main':
            cert[sid]=graph.native_camera(sid).optical_domain_certificate()
            if not cert[sid]['passed']:reasons.append(sid+'_whole_allocated_Brown_domain_failed')
    write_json(out/'model.json',candidate);write_json(out/'before.json',before);write_json(out/'after.json',after);write_json(out/'optimization_trace.json',trajectory)
    np.savez_compressed(out/'profiled_nuisance.npz',source_flank_uv=profile.uv,source_native_pts=np.repeat(np.array([r['native_pts']for r in sections],np.int64),2),source_row_time_seconds=profile.times,before_interface_world=bdata['profiled_interface_world'],after_interface_world=adata['profiled_interface_world'],before_local_halfwidth=bdata['halfwidth'],after_local_halfwidth=adata['halfwidth'],before_side_assignment=bdata['side_assignment'],after_side_assignment=adata['side_assignment'],before_tangent_station=bdata['tangent_station'],after_tangent_station=adata['tangent_station'])
    report=dict(schema='rinkfit.fixed_pose_multisource_wide_hinge_candidate.v1',input_model=dict(file=str(Path(model_file).resolve()),sha256=file_sha256(model_file)),candidate_model=dict(file=str((out/'model.json').resolve()),sha256=file_sha256(out/'model.json')),source_index=dict(file=str(Path(source_index_file).resolve()),sha256=file_sha256(source_index_file)),source_inventory=dict(file=str(Path(inventory_file).resolve()),sha256=file_sha256(inventory_file)),source_qualification_QA=dict(file=str(Path(qualification_QA_file).resolve()),sha256=file_sha256(qualification_QA_file)),source_index_rows=nraw,source_sections=len(sections),original_flank_denominator=2*len(sections),same_exposure_entity_groups=len(profile.groups),groups_with_fewer_than_three_sections=sum(len(x)<3 for x in profile.groups.values()),groups_with_fewer_than_three_are_retained_but_do_not_identify_curvature=True,before=dict(file=str(out/'before.json'),sha256=file_sha256(out/'before.json')),after=dict(file=str(out/'after.json'),sha256=file_sha256(out/'after.json')),initial_coefficients=profile.initial.tolist(),candidate_coefficients=extension.tolist(),fitted_existing_components=['k1','k2','p1','p2']if fit_tangential else['k1','k2'],fit_method=fit_method,physical_constraint_values=fit.get('physical_constraints'),optimizer=dict(success=bool(fit.success),message=str(fit.message),nfev=int(fit.nfev),njev=int(fit.njev),optimality=float(fit.optimality),conditional_J_singular_values=singular.tolist()),S01=dict(all_physical_blocks_and_shot_metadata_equal=True,all_base_optics_and_signed_RS_equal=True,activation_boundary_equal=old['activation_log_f_native']==meta['activation_log_f_native'],allocated_control_log_f_min=float(first.controls[:,2].min()),activation_log_f_native=profile.boundary,hinge_exact_zero_entire_allocated_cubic_control_hull=zero,entire_native_scope_and_allowed_RS_halo_covered_by_existing_allocation=True),main_whole_allocated_optical_domain=cert,rejection_reasons=sorted(set(reasons)),candidate_source_accuracy_accepted=False,production_installation_performed=False,Solve_started=False,scope=['Only the selected existing shared C1 hinge coefficients vary; base Brown, all pose/scene/geometry/station data and all current parameter blocks remain fixed. Tangential extension is fitted only when explicitly selected, and remains an uncertified conditional optical parameter.','Every original material flank has its own exact source-row RS time. Each profiled finite straight line has two geometric nuisance degrees per exposure/entity. The two boundary stations are independent; one positive local halfwidth is shared only within each source-confirmed section, using the existing +/- width-normal line primitive.','Original image-space midpoints are never inverted as an exact zero-width geometric observation. World midpoints used by TLS are derived trial arithmetic for eliminating two +/- offset equations, not new source XYZ or paired station correspondences.','Line TLS is an eliminated straightness initializer; its normal-distance solution is not claimed to be the globally optimal pixel/true-RS nuisance profile. Actual original-flank observed-row and certified true-RS reprojection errors are reported separately.','Criterion averages squared 2D original-flank residuals per physical shot/entity, then per exposure, then per section. These are explicit initialization weights, not calibrated W/covariance or independent-row precision. Dense related source readings and temporal descendants remain identified by source family.','Current pose and signed readout are provisional fixed conditioning assumptions. Scene/pose/RS and radial/base-optics tradeoffs, absolute optical calibration, tangential extension and out-of-support extrapolation remain unidentified.','No global template/world station correspondence, hardware calibration, new scene or production packet is issued. All source failures and underdetermined groups retain their full denominator.'])
    write_json(out/'report.json',report);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--source-index',required=True);p.add_argument('--inventory',required=True);p.add_argument('--qualification-QA',required=True);p.add_argument('--out',required=True);p.add_argument('--fit-tangential',action='store_true');p.add_argument('--fit-method',choices=['least_squares','physical_slsqp'],default='least_squares');a=p.parse_args();r=build_multisource_wide_candidate(a.model,a.source_index,a.out,inventory_file=a.inventory,qualification_QA_file=a.qualification_QA,fit_tangential=a.fit_tangential,fit_method=a.fit_method);print(json.dumps(dict(candidate_model=r['candidate_model'],coefficients=r['candidate_coefficients'],optimizer=r['optimizer'],rejection_reasons=r['rejection_reasons']),ensure_ascii=False))


if __name__=='__main__':main()
