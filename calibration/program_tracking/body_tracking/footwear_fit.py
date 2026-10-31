"""Fixed-pelvis two-boot calibration against source silhouettes and blade curves.

CPU development diagnostic. All camera, scene, pelvis, beta and bone lengths
are frozen. Shared equipment geometry and optional articulated leg corrections
are fitted to explicitly admitted source pixels; no contact-height factor.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import cv2
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix,kron,eye,diags
from .rebuild_prepare import load,digest
from .rebuild_exposure_body import ExposureArticulatedSplineProblem,moving_projection
from .fullprogramme_body_sequence import _basis,_second_derivative_basis
from .footwear_geometry import NAMES,INITIAL,LOWER,UPPER,LEG_DOFS,decode,extra_rotations,equipment_world,lowest_blade_world


def hull(points):
    points=np.asarray(points,float);indices=cv2.convexHull(points.astype(np.float32),returnPoints=False).ravel()
    return points[indices]


def sample_polyline(points,count=48,closed=False):
    points=np.asarray(points,float);path=np.vstack((points,points[0])) if closed else points
    length=np.linalg.norm(np.diff(path,axis=0),axis=1);cumulative=np.r_[0,np.cumsum(length)]
    targets=np.linspace(0,cumulative[-1],count,endpoint=not closed);out=np.stack([np.interp(targets,cumulative,path[:,a]) for a in range(2)],axis=1)
    return out


def point_polyline_distance(points,line,closed=False):
    line=np.asarray(line,float);a=line;b=np.roll(line,-1,axis=0) if closed else line[1:]
    if not closed:a=a[:-1]
    vector=b-a;difference=np.asarray(points)[:,None]-a[None]
    denominator=np.sum(vector*vector,axis=1)
    alpha=np.clip(np.sum(difference*vector[None],axis=2)/np.maximum(denominator,1e-16)[None],0,1)
    closest=a[None]+alpha[:,:,None]*vector[None]
    return np.sqrt(np.maximum(np.sum((np.asarray(points)[:,None]-closest)**2,axis=2).min(1),0))


def bilinear(image,points):
    points=np.asarray(points);x=points[:,0];y=points[:,1];h,w=image.shape
    # Outside-crop distance cannot disappear. Sampling clamps only the
    # distance field, then adds the actual distance beyond its support.
    xc=np.clip(x,0,w-1.0000001);yc=np.clip(y,0,h-1.0000001)
    ix=np.floor(xc).astype(int);iy=np.floor(yc).astype(int);ax=xc-ix;ay=yc-iy
    result=(1-ax)*(1-ay)*image[iy,ix]+ax*(1-ay)*image[iy,ix+1]+(1-ax)*ay*image[iy+1,ix]+ax*ay*image[iy+1,ix+1]
    return result+np.hypot(x-xc,y-yc)


class FootwearProblem:
    def __init__(self,root,review_path,mode,spacing=.4,dataset=None,exposure_nodes=3,base_body=None,additional_frames=(),source_window=None):
        self.root=Path(root);self.out=Path(dataset) if dataset else self.root/'footwear_v2';self.mode=mode;self.articulate=mode=='articulated';self.dof=len(LEG_DOFS)
        self.p=load(self.root/'input.npz');self.base_path=Path(base_body) if base_body else self.root/'pose/result.npz';self.base=load(self.base_path);self.meta=json.loads((self.root/'input.json').read_text(encoding='utf8'))
        self.review=json.loads(Path(review_path).read_text(encoding='utf8'));self.review_path=Path(review_path)
        self.index=json.loads((self.out/'source_index.json').read_text(encoding='utf8'));self.admission=json.loads((self.out/'admission.json').read_text(encoding='utf8'))
        self.rows=[r for r in self.admission['records'] if r['status']=='admitted_visible_boundary']
        if source_window is not None:
            inside=lambda r:source_window[0]<=r['pts']<=source_window[1]
            self.rows=[r for r in self.rows if inside(r)]
            self.review=dict(self.review,blade_curves=[r for r in self.review.get('blade_curves',[]) if inside(r)],boot_landmarks=[r for r in self.review.get('boot_landmarks',[]) if inside(r)])
        self.fast_projection=all(not r.get('reverse_boundary',True) for r in self.rows)
        pts=sorted(set(r['pts'] for r in self.rows)|set(r['pts'] for r in self.review.get('blade_curves',[]))|set(r['pts'] for r in self.review.get('boot_landmarks',[]))|set(self.p['pts'][np.asarray(additional_frames,int)].tolist()))
        self.frames=np.searchsorted(self.p['pts'],pts)
        if np.any(self.frames>=len(self.p['pts'])) or not np.array_equal(self.p['pts'][self.frames],pts):raise ValueError('Footwear source PTS must exactly match admitted body exposures')
        self.frame_lookup={p:i for i,p in enumerate(pts)};self.n=len(pts)
        self.body_problem=ExposureArticulatedSplineProblem(root,'pose',exposure_nodes=exposure_nodes);self.body=self.body_problem.body;self.projector=self.body_problem.projection
        self.exposure_nodes=self.body_problem.exposure_nodes;self.exposure_offsets=np.linspace(-.5,.5,self.exposure_nodes)
        self.basestate=self.body_problem.kinematics(self.base['x'],self.frames,True);self.nodes=self.basestate['inputs']
        self.Bfull,self.knots=_basis(self.p['fit_time_seconds'],spacing);self.m=self.Bfull.shape[1];self.spacing=spacing
        times=(self.p['fit_time_seconds'][self.frames,None]+self.exposure_offsets[None]*float(self.p['rig'][37])).ravel()
        self.Bnodes=BSpline.design_matrix(times,self.knots,3,extrapolate=True).tocsr();self.bc=self.Bnodes.indices.reshape(self.n,self.exposure_nodes,4)
        self.visible_rows=[];bypts={r['pts']:r for r in self.index['rows']}
        for row in self.rows:
            source=bypts[row['pts']];packet=load(row['mask_file']);mask=packet['masks'][row['foot']]
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);contour=max(contours,key=cv2.contourArea)
            boundary=np.ones(mask.shape,np.uint8);cv2.drawContours(boundary,[contour],-1,0,1)
            field=cv2.distanceTransform(boundary,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            other=cv2.dilate(packet['masks'][1-row['foot']],np.ones((5,5),np.uint8)).astype(bool)
            if row['partial']:field[other]=0
            self.visible_rows.append(dict(**row,frame=self.frame_lookup[row['pts']],offset=np.asarray(source['crop_xyxy'][:2]),field=field,
                observed=np.asarray(row['source_boundary_native_uv']),other=other))
        self.blade_rows=[dict(**r,frame=self.frame_lookup[r['pts']]) for r in self.review.get('blade_curves',[])]
        self.landmark_rows=[dict(**r,frame=self.frame_lookup[r['pts']]) for r in self.review.get('boot_landmarks',[])]
        weights=np.r_[.01,np.full(len(self.p['pts'])-2,.02),.01]
        self.acc=_second_derivative_basis(self.p['fit_time_seconds'],self.knots).multiply(np.sqrt(weights/.02)[:,None]).tocsr()
        self.size=len(INITIAL)+(self.m*self.dof if self.articulate else 0)
        self.x0=np.r_[INITIAL,np.zeros(self.size-len(INITIAL))];self.bounds=(np.r_[LOWER,np.full(self.size-len(INITIAL),-np.deg2rad(45))],np.r_[UPPER,np.full(self.size-len(INITIAL),np.deg2rad(45))])
        self.shape_sigma=np.r_[np.full(10,.35),np.deg2rad([12,12,12,8])]
        self.knee_uv=self.p['source_uv'][self.frames][:,[13,14]];self.knee_mask=self.p['source_qualified'][self.frames][:,[13,14]]
        self.cache_x=None;self.cache=None;self.log=None;self.evaluations=0;self.start=time.perf_counter()
        self.sparsity=self.evaluate(self.x0,pattern=True)['pattern']
    def state(self,x,frames=None,fast=False,root_controls=None,project=True):
        frames=self.frames if frames is None else np.asarray(frames,int)
        if np.array_equal(frames,self.frames):base=self.basestate;B=self.Bnodes
        else:
            base=self.body_problem.kinematics(self.base['x'],frames,True)
            times=(self.p['fit_time_seconds'][frames,None]+self.exposure_offsets[None]*float(self.p['rig'][37])).ravel()
            B=BSpline.design_matrix(times,self.knots,3,extrapolate=True).tocsr()
        if root_controls is not None:base=dict(base,P=base['P']+B@np.asarray(root_controls).reshape(self.m,3))
        correction=B@x[len(INITIAL):].reshape(self.m,self.dof) if self.articulate else np.zeros((self.exposure_nodes*len(frames),self.dof))
        coco,joints,oldblade,rotations=self.body.forward(base['inputs']['pose'],base['delta'],additional_rotations=extra_rotations(correction),with_frames=True)
        gear,parts,bootR,bootO=equipment_world(base['P'],base['Q'],joints,rotations,x[:len(INITIAL)])
        bodyworld=base['P'][:,None]+np.einsum('nij,nkj->nki',base['Q'],coco)
        allworld=np.concatenate((gear.reshape(self.exposure_nodes*len(frames),-1,3),bodyworld),axis=1).reshape(len(frames),self.exposure_nodes,-1,3)
        if not project:return dict(world_nodes=allworld,base=base,correction=correction,parts=parts,boot_rotation=bootR,boot_origin=bootO)
        if fast:
            selected=np.r_[np.arange(14),np.arange(48,62),[109,110]]
            first=moving_projection(self.projector,allworld[:,:,selected],frames)
            uv_all=np.full((len(frames),113,2),np.nan);world_all=np.full((len(frames),113,3),np.nan)
            uv_all[:,selected]=first['uv'];world_all[:,selected]=first['world']
            needed=np.unique([r['frame'] for r in self.blade_rows]).astype(int);blade_ids=np.r_[np.arange(14,48),np.arange(62,96)]
            blade_iterations=0
            if len(needed):
                blade_projection=moving_projection(self.projector,allworld[needed][:,:,blade_ids],np.asarray(frames)[needed]);blade_iterations=blade_projection['row_iterations']
                for local,fi in enumerate(needed):
                    uv_all[fi,blade_ids]=blade_projection['uv'][local];world_all[fi,blade_ids]=blade_projection['world'][local]
            projected=dict(uv=uv_all,world=world_all,row_iterations=max(first['row_iterations'],blade_iterations))
        else:projected=moving_projection(self.projector,allworld,frames)
        minimum=lowest_blade_world(bootR,bootO,x[:len(INITIAL)])
        return dict(uv=projected['uv'][:,:96].reshape(len(frames),2,48,2),body_uv=projected['uv'][:,96:],
            world=projected['world'][:,:96].reshape(len(frames),2,48,3),parts=parts,correction=correction,
            nominal_blade_minimum=minimum.reshape(len(frames),self.exposure_nodes,2,3)[:,self.exposure_nodes//2],base=base,row_iterations=projected['row_iterations'])
    def evaluate(self,x,pattern=False):
        if not pattern and self.cache_x is not None and np.array_equal(x,self.cache_x):return self.cache
        tick=time.perf_counter();state=self.state(x,fast=self.fast_projection);res=[];parts={};patterns=[];offset=0
        def add(value,name,frame=None,explicit=None):
            nonlocal offset
            value=np.asarray(value,float).ravel();res.append(value);parts[name]=parts.get(name,0)+float(value@value/2)
            if pattern:
                if explicit is not None:patterns.append((offset,explicit))
                elif frame is not None:
                    columns=np.arange(len(INITIAL))
                    if self.articulate:columns=np.r_[columns,(np.unique(self.bc[frame])[:,None]*self.dof+np.arange(self.dof)+len(INITIAL)).ravel()]
                    patterns.append((offset,(len(value),np.asarray(columns,int))))
            offset+=len(value)
        for row in self.visible_rows:
            fi,foot=row['frame'],row['foot'];indices=state['parts']['shell'] if row['observed_kind']=='boot_external_contour' else np.arange(48)
            polygon=hull(state['uv'][fi,foot,indices]);source_distance=point_polyline_distance(row['observed'],polygon,True)
            if row.get('reverse_boundary',True):
                predicted=sample_polyline(polygon,48,True);source_distance=np.r_[source_distance,bilinear(row['field'],predicted-row['offset'])]
            add(source_distance/row['sigma_px']/np.sqrt(48),'boot_boundary',fi)
        for row in self.blade_rows:
            fi,foot=row['frame'],row['foot'];curve=state['uv'][fi,foot,state['parts']['blade_left']]
            observed=sample_polyline(row['uv'],24);distance=point_polyline_distance(observed,curve)
            if not row['partial']:distance=np.r_[distance,point_polyline_distance(sample_polyline(curve,24),row['uv'])]
            add(distance/row['sigma_px']/np.sqrt(24),'blade_lower_edge',fi)
        for row in self.landmark_rows:
            fi,foot=row['frame'],row['foot']
            for key,pair in [('heel',[0,7]),('toe',[1,8])]:
                edge=state['uv'][fi,foot,pair];vector=edge[1]-edge[0];source=np.asarray(row[key],float)
                alpha=np.clip(np.dot(source-edge[0],vector)/max(np.dot(vector,vector),1e-16),0,1)
                residual=(edge[0]+alpha*vector-source)/row['sigma_px']
                add(residual,'boot_heel_toe_semantics',fi)
        for fi in range(self.n):
            difference=(state['body_uv'][fi,[13,14]]-self.knee_uv[fi])*self.knee_mask[fi,:,None]/12
            add(difference,'source_knees',fi)
        # Residual units and engineering scales are explicit in config. These
        # priors are not learned noise, measured uncertainty, or scene weights.
        add((x[:len(INITIAL)]-INITIAL)/self.shape_sigma,'shared_shape_prior',explicit=(len(INITIAL),np.arange(len(INITIAL))))
        if self.articulate:
            controls=x[len(INITIAL):].reshape(self.m,self.dof)
            add(controls/np.deg2rad(15),'leg_rotation_prior',explicit=eye(controls.size,format='coo'))
            add(self.acc@controls/np.deg2rad(400),'leg_correction_acceleration',explicit=kron(self.acc,eye(self.dof),format='coo'))
        vector=np.concatenate(res);plan=None
        if pattern:
            rr=[];cc=[]
            for start,pat in patterns:
                if isinstance(pat,tuple):
                    count,columns=pat;rr.extend(np.repeat(np.arange(start,start+count),len(columns)));cc.extend(np.tile(columns,count))
                else:
                    co=pat.tocoo();rr.extend(co.row+start);cc.extend(co.col+len(INITIAL))
            plan=coo_matrix((np.ones(len(rr),bool),(rr,cc)),shape=(len(vector),self.size)).tocsr()
        out=dict(residual=vector,parts=parts,state=state,pattern=plan,seconds=time.perf_counter()-tick)
        if not pattern:
            self.cache_x=np.array(x,copy=True);self.cache=out;self.evaluations+=1
            if self.log:self.log.write(json.dumps(dict(event='evaluation',evaluation=self.evaluations,cost=float(vector@vector/2),parts=parts,elapsed_seconds=time.perf_counter()-self.start,eval_seconds=out['seconds'],domain_failures=0))+'\n')
        return out
    def fun(self,x):return self.evaluate(x)['residual']


def run(root,review_path,mode,warm=None,dataset=None,jac_scheme='3-point',exposure_nodes=3):
    problem=FootwearProblem(root,review_path,mode,dataset=dataset,exposure_nodes=exposure_nodes);out=problem.out/mode
    if out.exists():raise FileExistsError('Preserve existing run: '+str(out))
    out.mkdir();(out/'source_review_at_run.json').write_bytes(Path(review_path).read_bytes());config=dict(mode=mode,device='cpu',variables=problem.size,shared_parameter_names=NAMES,
        camera_scene_pelvis_beta_bone_lengths_frozen=True,contact_height_factor=False,local_dofs=LEG_DOFS if problem.articulate else [],
        exposure_nodes=problem.exposure_nodes,correction_spline_spacing_s=problem.spacing,shape_prior_log_sigma=.35,boot_alignment_prior_deg=12,leg_control_prior_deg=15,
        leg_correction_acceleration_scale_deg_s2=400,mask_residual='source-to-projected visible contour; reverse direction only for explicitly complete source boundaries; normalized by sqrt(48) per object',
        source_boundary_scales_px=sorted(set(float(r['sigma_px']) for r in problem.rows)),heel_toe_factor_count=2*len(problem.landmark_rows),
        knee_residual_scale_px=12,priors_are_engineering_scales=True,admission_sha256=digest(problem.out/'admission.json'),
        review_sha256=digest(review_path),base_body_sha256=digest(problem.root/'pose/result.npz'),
        input_sha256=digest(problem.root/'input.npz'),metric_scale_status='conditional',rolling_shutter='same native complete camera with moving articulated body',jacobian_scheme=jac_scheme,visible_source_contours_only=problem.fast_projection)
    if warm:
        z=load(warm);problem.x0[:len(INITIAL)]=z['x'][:len(INITIAL)]
        if len(z['x'])==problem.size and np.array_equal(z.get('correction_knots'),problem.knots):problem.x0=z['x'].copy()
    config['warm_start']=None if not warm else dict(path=str(Path(warm).resolve()),sha256=digest(warm))
    np.savez_compressed(out/'initial_state.npz',x=problem.x0,correction_knots=problem.knots)
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    with (out/'fit_progress.jsonl').open('w',encoding='utf8',buffering=1) as log:
        problem.log=log;log.write(json.dumps(dict(event='start',**config))+'\n')
        def accepted(x):
            v=problem.evaluate(x);log.write(json.dumps(dict(event='accepted',cost=float(v['residual']@v['residual']/2),parts=v['parts'],elapsed_seconds=time.perf_counter()-problem.start))+'\n')
            np.savez_compressed(out/'accepted.pending.npz',x=x,correction_knots=problem.knots);(out/'accepted.pending.npz').replace(out/'last_accepted.npz')
        fit=least_squares(problem.fun,problem.x0,jac=jac_scheme,jac_sparsity=problem.sparsity,bounds=problem.bounds,diff_step=1e-4,
            method='trf',tr_solver='lsmr',x_scale='jac',ftol=1e-8,xtol=1e-8,gtol=1e-4,callback=accepted,tr_options={'atol':1e-7,'btol':1e-7})
        v=problem.evaluate(fit.x);problem.log=None;frames=np.arange(len(problem.p['pts']));allstate=problem.state(fit.x,frames)
        # Root and global orientation are exactly the selected B state.
        np.savez_compressed(out/'result.npz',x=fit.x,correction_knots=problem.knots,pts=problem.p['pts'],
            equipment_uv=allstate['uv'],equipment_world_m=allstate['world'],COCO_uv=allstate['body_uv'],
            nominal_blade_minimum_world_m=allstate['nominal_blade_minimum'],P_world_m=problem.base['P_world_m'],Q_world=problem.base['Q_world'],
            local_correction_nodes_rad=allstate['correction'],fit_frames=problem.frames)
        report=dict(mode=mode,optimizer_success=bool(fit.success),numerically_converged=bool(fit.optimality<=1e-4),
            termination_message=fit.message,cost=float(fit.cost),optimality=float(fit.optimality),nfev=fit.nfev,njev=fit.njev,
            actual_function_evaluations=problem.evaluations,elapsed_seconds=time.perf_counter()-problem.start,parts=v['parts'],parameters=decode(fit.x[:len(INITIAL)]),
            camera_and_scene_changed=False,pelvis_changed=False,bone_lengths_changed=False,user_visually_accepted=False,independent_3d_truth=False)
        (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');log.write(json.dumps(dict(event='complete',**report))+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--review',type=Path,required=True);ap.add_argument('--mode',choices=['shared','articulated'],required=True);ap.add_argument('--warm',type=Path);ap.add_argument('--dataset',type=Path);ap.add_argument('--jac-scheme',choices=['2-point','3-point'],default='3-point');ap.add_argument('--exposure-nodes',type=int,choices=[3,5],default=3);a=ap.parse_args();run(a.root,a.review,a.mode,a.warm,a.dataset,a.jac_scheme,a.exposure_nodes)
