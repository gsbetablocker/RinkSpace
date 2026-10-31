"""CPU research activity set for a shared root/leg/equipment explanation.

The camera, beta, Q and baseline nine pose corrections stay frozen. J0 varies
legs, J1 additionally varies a continuous root displacement, and J2 additionally
varies explicitly selected shared dimensions. This is not a second production
scene solver and has no contact or camera feedback.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix,csr_matrix,kron,eye,hstack
from .footwear_fit import FootwearProblem
from .footwear_geometry import INITIAL,LOWER,UPPER,NAMES,extra_rotations,equipment_world,lowest_blade_world
from .footwear_pixel_factors import pixel_factors
from .rebuild_exposure_body import moving_projection
from .rebuild_prepare import load,digest


class JointFootwearProblem:
    def __init__(self,root,review,dataset,base_body,mode='J0',window=None):
        self.root=Path(root);self.mode=mode;self.free_root=mode in ['J1','J2'];self.free_geometry=mode=='J2';self.geometry_ids=np.array([1,2],int) if self.free_geometry else np.array([],int)
        self.review_path=Path(review);self.review=json.loads(self.review_path.read_text(encoding='utf8'));self.base_body=Path(base_body)
        # Use the same baseline exposure grid and body-source exclusion windows.
        from .rebuild_exposure_body import ExposureArticulatedSplineProblem
        bp=ExposureArticulatedSplineProblem(root,'pose',exposure_nodes=5);extra=bp.frames
        if window is not None:extra=extra[(bp.p['pts'][extra]>=window[0])&(bp.p['pts'][extra]<=window[1])]
        self.p=FootwearProblem(root,review,'articulated',dataset=dataset,exposure_nodes=5,base_body=base_body,additional_frames=extra,source_window=window)
        self.g=load(self.review['frozen_equipment_source'])['x'][:14].copy();self.installation_reference=self.g[10:13].copy();self.m=self.p.m;self.dof=self.p.dof;self.nleg=self.m*self.dof;self.nroot=self.m*3 if self.free_root else 0;self.size=self.nleg+self.nroot+len(self.geometry_ids)
        self.x0=np.r_[load(self.review['frozen_equipment_source'])['x'][14:],np.zeros(self.nroot),self.g[self.geometry_ids]]
        self.bounds=(np.r_[np.full(self.nleg,-np.pi),np.full(self.nroot,-np.inf),LOWER[self.geometry_ids]],np.r_[np.full(self.nleg,np.pi),np.full(self.nroot,np.inf),UPPER[self.geometry_ids]])
        self.fixed_root=np.zeros((self.m,3));self.frames=self.p.frames;self.n=self.p.n;self.mid=2
        self.Bnodes=self.p.Bnodes;self.bc=self.p.bc;self.bv=self.Bnodes.data.reshape(self.n,5,4)
        ids=[];fi=[]
        for frame in range(self.n):
            points=list(96+self.p.body_problem.indices)
            if any(r['frame']==frame for r in self.p.visible_rows) or any(r['frame']==frame for r in self.p.landmark_rows):points.extend(np.r_[np.arange(14),np.arange(48,62)].tolist())
            if any(r['frame']==frame for r in self.p.blade_rows):points.extend(np.r_[np.arange(14,48),np.arange(62,96)].tolist())
            points=sorted(set(points));ids.extend(points);fi.extend([frame]*len(points))
        self.fi=np.asarray(fi,int);self.pi=np.asarray(ids,int);self.uv_rows=(self.fi*113+self.pi)[:,None]*2+np.arange(2)
        self.base_acceleration=self.p.body_problem.acc@self.p.base['x'][:3*self.p.body_problem.m].reshape(self.p.body_problem.m,3)
        self.cache_x=None;self.cache=None;self.log=None;self.start=time.perf_counter();self.evaluations=0;self.domain_failures=0
        self.initial=self.evaluate(self.x0,jacobian=True)
    def unpack(self,x):
        g=self.g.copy();g[self.geometry_ids]=x[self.nleg+self.nroot:];leg=np.asarray(x[:self.nleg]).reshape(self.m,self.dof);root=np.asarray(x[self.nleg:self.nleg+self.nroot]).reshape(self.m,3) if self.free_root else self.fixed_root
        return g,leg,root
    def raw(self,x,frames=None):
        g,leg,root=self.unpack(x);return self.p.state(np.r_[g,leg.ravel()],frames=frames,root_controls=root,project=False)
    def evaluate(self,x,jacobian=False):
        if self.cache_x is not None and np.array_equal(self.cache_x,x) and (not jacobian or self.cache['jac'] is not None):return self.cache
        tick=time.perf_counter();g,leg,root=self.unpack(x);raw=self.raw(x);world=raw['world_nodes'];nodes=world[self.fi,:,self.pi][:,:,None]
        projection=moving_projection(self.p.projector,nodes,self.frames[self.fi],jacobian=jacobian);uv=np.full((self.n,113,2),np.nan);uv[self.fi,self.pi]=projection['uv'][:,0]
        data,A,parts,choices=pixel_factors(uv,self.p,body=True,blade_side_candidates=True);res=[data];matrices=[]
        if jacobian:
            jp=projection['pixel_jac_nodes'][:,0];leg_derivatives=[];base=raw['base'];correction=raw['correction'];h=1e-5
            for d in range(self.dof):
                worlds=[]
                for sign in [1,-1]:
                    c=correction.copy();c[:,d]+=sign*h;coco,joints,_,rotations=self.p.body.forward(base['inputs']['pose'],base['delta'],additional_rotations=extra_rotations(c),with_frames=True)
                    gear,*_=equipment_world(base['P'],base['Q'],joints,rotations,g);body=base['P'][:,None]+np.einsum('nij,nkj->nki',base['Q'],coco);W=np.concatenate((gear.reshape(self.n*5,96,3),body),1).reshape(self.n,5,113,3);worlds.append(W[self.fi,:,self.pi])
                derivative=(worlds[0]-worlds[1])/(2*h);leg_derivatives.append(np.einsum('mplc,mlc->mpl',jp,derivative))
            local=np.stack(leg_derivatives,-1);basis=self.bv[self.fi];columns=self.bc[self.fi]*self.dof
            values=(local[:,:,:,None,:]*basis[:,None,:,:,None]).ravel();rows=np.broadcast_to(self.uv_rows[:,:,None,None,None],(*local.shape[:3],4,self.dof)).ravel();cols=np.broadcast_to(columns[:,None,:,:,None]+np.arange(self.dof),(*local.shape[:3],4,self.dof)).ravel()
            rr=[rows];cc=[cols];vv=[values]
            if self.free_root:
                values=(jp[:,:,:,None,:]*basis[:,None,:,:,None]).ravel();rows=np.broadcast_to(self.uv_rows[:,:,None,None,None],(*jp.shape[:3],4,3)).ravel();cols=np.broadcast_to(self.nleg+self.bc[self.fi,None,:,:,None]*3+np.arange(3),(*jp.shape[:3],4,3)).ravel();rr.append(rows);cc.append(cols);vv.append(values)
            for q,j in enumerate(self.geometry_ids):
                sides=[]
                for sign in [1,-1]:
                    changed=g.copy();changed[j]+=sign*h;coco,joints,_,rotations=self.p.body.forward(base['inputs']['pose'],base['delta'],additional_rotations=extra_rotations(correction),with_frames=True);gear,*_=equipment_world(base['P'],base['Q'],joints,rotations,changed);W=gear.reshape(self.n,5,96,3);derivative=np.zeros((len(self.fi),5,3));mask=self.pi<96;derivative[mask]=W[self.fi[mask],:,self.pi[mask]];sides.append(derivative)
                change=np.einsum('mplc,mlc->mp',jp,(sides[0]-sides[1])/(2*h));rr.append(self.uv_rows.ravel());cc.append(np.full(self.uv_rows.size,self.nleg+self.nroot+q));vv.append(change.ravel())
            U=coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(uv.size,self.size)).tocsr();U.eliminate_zeros();matrices.append(A@U)
        def prior(value,name,matrix):
            value=np.asarray(value).ravel();res.append(value);parts[name]=float(value@value/2)
            if jacobian:matrices.append(matrix)
        def embed(matrix,offset):return hstack((csr_matrix((matrix.shape[0],offset)),matrix,csr_matrix((matrix.shape[0],self.size-offset-matrix.shape[1]))),format='csr')
        prior(leg/np.deg2rad(15),'leg_prior',embed(eye(self.nleg,format='csr')/np.deg2rad(15),0) if jacobian else None)
        prior(self.p.acc@leg/np.deg2rad(400),'leg_acceleration',embed(kron(self.p.acc,eye(self.dof),format='csr')/np.deg2rad(400),0) if jacobian else None)
        scales=np.array([15.,15.,25.]);value=(self.base_acceleration+self.p.acc@root)/scales
        matrix=embed(kron(self.p.acc,csr_matrix(np.diag(1/scales)),format='csr'),self.nleg) if jacobian and self.free_root else csr_matrix((value.size,self.size)) if jacobian else None;prior(value,'root_acceleration',matrix)
        prior((g[:10]-INITIAL[:10])/.35,'conditional_shape_prior',coo_matrix((np.full(len(self.geometry_ids),1/.35),(self.geometry_ids,np.arange(self.nleg+self.nroot,self.size))),shape=(10,self.size)).tocsr() if jacobian else None)
        from scipy.sparse import vstack
        vector=np.concatenate(res);J=vstack(matrices,format='csr') if jacobian else None
        self.cache_x=np.array(x,copy=True);self.cache=dict(residual=vector,jac=J,parts=parts,uv=uv,choices=choices,raw=raw,seconds=time.perf_counter()-tick,data_rows=len(data));self.evaluations+=1
        if self.log:self.log.write(json.dumps(dict(event='evaluation',evaluation=self.evaluations,cost=float(vector@vector/2),parts=parts,gradient_computed=jacobian,seconds=time.perf_counter()-self.start,eval_seconds=self.cache['seconds'],domain_failures=self.domain_failures))+'\n')
        return self.cache
    def fun(self,x):return self.evaluate(x)['residual']
    def jac(self,x):return self.evaluate(x,jacobian=True)['jac']
    def export(self,x):
        frames=np.arange(len(self.p.p['pts']));g,leg,root=self.unpack(x);full=self.p.state(np.r_[g,leg.ravel()],frames,root_controls=root);raw=self.raw(x,frames);P=raw['base']['P'].reshape(len(frames),5,3);mid=P[:,2];G=mid.copy();G[:,2]=0;projected=moving_projection(self.p.projector,P[:,:,None],frames);source=self.p.p['pts']/12800.;tau=float(self.p.p['rig'][37]);world=raw['world_nodes']
        return dict(x=x,pts=self.p.p['pts'],correction_knots=self.p.knots,shared_geometry=g,leg_controls=leg,root_delta_controls=root,P_world_mid=mid,G_world_mid=G,Q_world=self.p.base['Q_world'],P_uv_dynamic=projected['uv'][:,0],P_world_at_row=projected['world'][:,0],P_row_time=source+tau*(np.clip(projected['uv'][:,0,1],0,1079)/1079-.5),COCO_world_mid=world[:,2,96:],COCO_uv=full['body_uv'],equipment_uv=full['uv'],blade_world_at_row=full['world'][:,:,14:],nominal_blade_minimum_world_m=full['nominal_blade_minimum'],source_lower_body_inferred=self.p.p['lower_body_inferred'],contact_allowed_for_root=np.zeros(len(frames),bool),exposure_nodes=np.asarray(5))


def run(root,review,dataset,base_body,mode,output,warm=None,initial=None,initial_file=None):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);problem=JointFootwearProblem(root,review,dataset,base_body,mode)
    if warm:
        saved=load(warm);problem.x0[:problem.nleg]=saved['leg_controls'].ravel()
        if problem.free_root:problem.x0[problem.nleg:problem.nleg+problem.nroot]=saved['root_delta_controls'].ravel()
        if problem.free_geometry:problem.x0[-len(problem.geometry_ids):]=saved['shared_geometry'][problem.geometry_ids]
    if initial_file:initial=load(initial_file)['x']
    if initial is not None:
        if np.asarray(initial).shape!=(problem.size,):raise ValueError('Initial active variables differ')
        problem.x0=np.asarray(initial).copy()
    config=dict(schema='rinkspace.fixed-camera-root-leg-footwear.v1',mode=mode,base_body=str(Path(base_body).resolve()),base_body_sha256=digest(base_body),review_sha256=digest(review),admission_sha256=digest(Path(dataset)/'admission.json'),device='CPU',exposure_nodes=5,warm_start=None if not warm else dict(path=str(Path(warm).resolve()),sha256=digest(warm)),initial_file=None if not initial_file else dict(path=str(Path(initial_file).resolve()),sha256=digest(initial_file)),root_Q_and_baseline_nine_pose_frozen=True,root_P_variable=problem.free_root,shared_geometry_active=[NAMES[i] for i in problem.geometry_ids],installation_gauge='three boot_align values frozen at declared conditional equipment reference',installation_angles_rad=problem.installation_reference.tolist(),local_chart_domain='rotation vector components +/- pi; numerical domain, not measured joint limits',contact_height_factor=False,camera_updated=False,body_source_pixels_preserved=True,spline_spacing_s=problem.p.spacing,source_frames=problem.n,gtol=1e-4,ftol=1e-7,xtol=1e-7)
    (output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8');(output/'source_review.json').write_bytes(Path(review).read_bytes());np.savez_compressed(output/'initial.npz',x=problem.x0)
    with (output/'fit_progress.jsonl').open('w',encoding='utf8',buffering=1) as log:
        problem.log=log;log.write(json.dumps(dict(event='start',**config))+'\n')
        def accepted(x):
            v=problem.evaluate(x,jacobian=True);log.write(json.dumps(dict(event='accepted',cost=float(v['residual']@v['residual']/2),gradient_inf=float(np.max(abs(v['jac'].T@v['residual']))),parts=v['parts'],seconds=time.perf_counter()-problem.start))+'\n');np.savez_compressed(output/'accepted.pending.npz',x=x);(output/'accepted.pending.npz').replace(output/'last_accepted.npz')
        fit=least_squares(problem.fun,problem.x0,jac=problem.jac,bounds=problem.bounds,method='trf',tr_solver='lsmr',x_scale='jac',ftol=1e-7,xtol=1e-7,gtol=1e-4,callback=accepted,tr_options={'atol':1e-7,'btol':1e-7})
        v=problem.evaluate(fit.x,jacobian=True);problem.log=None;packet=problem.export(fit.x);np.savez_compressed(output/'result.npz',**packet)
        report=dict(mode=mode,optimizer_success=bool(fit.success),termination=fit.message,optimality=float(fit.optimality),gradient_inf=float(np.max(abs(v['jac'].T@v['residual']))),numerically_converged=bool(fit.optimality<=1e-4),cost=float(fit.cost),parts=v['parts'],nfev=fit.nfev,njev=fit.njev,seconds=time.perf_counter()-problem.start,camera_updated=False,root_updated=problem.free_root,contact_feedback=False,footwear_installation_gauge=config['installation_gauge'],blade_candidates=v['choices'],user_accepted=False)
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');log.write(json.dumps(dict(event='complete',**report))+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--review',required=True);p.add_argument('--dataset',required=True);p.add_argument('--base-body',required=True);p.add_argument('--mode',choices=['J0','J1','J2'],required=True);p.add_argument('--output',required=True);p.add_argument('--warm');p.add_argument('--initial');a=p.parse_args();run(a.root,a.review,a.dataset,a.base_body,a.mode,a.output,a.warm,initial_file=a.initial)
