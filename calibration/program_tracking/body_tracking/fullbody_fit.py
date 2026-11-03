"""Controlled sparse full-body experiment on a frozen native camera.

This is the existing CPU research fitting path, not a replacement scene solver.
Local derivatives are chained through exposure nodes and compact spline support;
there is no finite differencing over the complete programme parameter vector.
"""
from pathlib import Path
from types import SimpleNamespace
import json,time,os
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix,kron,eye,vstack
from scipy.optimize import least_squares
from .rebuild_prepare import load,digest
from .rebuild_exposure_body import moving_projection
from .footwear_pixel_factors import pixel_factors
from .footwear_layout import OBSERVATION_LAYOUT
from .fullbody_state import WindowState
from .body_execution import get_body_execution


def footwear_rows(root,frames,pts):
    root=Path(root);review_path=root/'next_round_20261029/J1_s2/source_review.json'
    review=json.loads(review_path.read_text(encoding='utf8'))
    target=json.loads((review_path.parent/'config.json').read_text(encoding='utf8'))['admission_sha256']
    paths=[root/'footwear_v2/admission.json',*sorted((root/'footwear_v2').glob('*/admission.json'))]
    path=next(p for p in paths if digest(p)==target)
    admission=json.loads(path.read_text(encoding='utf8'));lookup={int(pts[f]):i for i,f in enumerate(frames)}
    visible=[dict(r,frame=lookup[r['pts']],observed=np.asarray(r['source_boundary_native_uv']))
             for r in admission['records'] if r['status']=='admitted_visible_boundary' and r['pts'] in lookup]
    blades=[dict(r,frame=lookup[r['pts']]) for r in review.get('blade_curves',[]) if r['pts'] in lookup]
    landmarks=[dict(r,frame=lookup[r['pts']]) for r in review.get('boot_landmarks',[]) if r['pts'] in lookup]
    return SimpleNamespace(visible_rows=visible,blade_rows=blades,landmark_rows=landmarks),dict(
        review_path=str(review_path),review_sha256=digest(review_path),admission_path=str(path),admission_sha256=target)


def skew(v):
    v=np.asarray(v);out=np.zeros((*v.shape[:-1],3,3))
    out[...,0,1]=-v[...,2];out[...,0,2]=v[...,1];out[...,1,0]=v[...,2]
    out[...,1,2]=-v[...,0];out[...,2,0]=-v[...,1];out[...,2,1]=v[...,0]
    return out


def right_jacobian(v):
    angle=np.linalg.norm(v,axis=-1);safe=np.maximum(angle,1e-7);K=skew(v)
    a=np.where(angle<1e-4,.5-angle**2/24,(1-np.cos(safe))/safe**2)
    b=np.where(angle<1e-4,1/6-angle**2/120,(safe-np.sin(safe))/safe**3)
    return np.eye(3)-a[...,None,None]*K+b[...,None,None]*(K@K)


class FullBodyProblem:
    def __init__(self,reference,projector,ledger,seconds,mode,log=None,*,state_factory=WindowState):
        self.reference=reference;self.projector=projector;self.obs=load(ledger);self.log=log
        self.layout=getattr(reference,'layout',OBSERVATION_LAYOUT)
        self.execution=get_body_execution(reference)
        # Native camera calls are separate from parallel body-world evaluations.
        self.projector.body_execution=self.execution
        if hasattr(self.projector,'core'):
            native_threads=int(os.environ.get('RINKSPACE_BODY_NATIVE_THREADS',str(self.execution.config.workers)))
            if native_threads<1:raise ValueError('RINKSPACE_BODY_NATIVE_THREADS must be positive')
            self.projector.core.motion_projection_threads=native_threads
        self.ledger_path=Path(ledger)
        np.testing.assert_array_equal(self.obs['pts'],reference.p['pts'])
        self.state=state_factory(reference,seconds,mode)
        self.frames=np.flatnonzero((reference.p['pts']/12800>=seconds[0]-1e-9)&(reference.p['pts']/12800<=seconds[1]+1e-9))
        self.n=len(self.frames);self.size=self.state.size;self.start=time.perf_counter();self.evaluations=0
        self.tau=float(projector.p['rig'][37]);self.times=(reference.times[self.frames,None]+np.linspace(-.5,.5,5)*self.tau).ravel()
        if abs(self.tau)/2>np.min(np.diff(reference.times))+1e-12:raise ValueError('Readout crosses an additional source-pose knot')
        self.ref=reference.at(self.times);self.B=self.state.basis(self.times).reshape(self.n,5,self.state.m)
        self.eval_basis=self.B.reshape(-1,self.state.m)
        self.gear,self.gear_sources=footwear_rows(reference.root,self.frames,reference.p['pts'])
        fi=[];pi=[]
        for f in range(self.n):
            ids=self.layout.body_landmarks.tolist()
            if any(r['frame']==f for r in self.gear.visible_rows+self.gear.landmark_rows):
                for side in self.layout.sides:ids+=self.layout.footwear_landmarks(side,'shell').tolist()
            if any(r['frame']==f for r in self.gear.blade_rows):
                for side in self.layout.sides:ids+=self.layout.footwear_landmarks(side,'blade').tolist()
            ids=sorted(set(ids));fi.extend([f]*len(ids));pi.extend(ids)
        self.fi=np.asarray(fi);self.pi=np.asarray(pi);self.uv_rows=(self.fi*self.layout.point_count+self.pi)[:,None]*2+np.arange(2)
        self.of,self.oj=np.nonzero(self.obs['use'][self.frames])
        # This scatter structure depends only on the frozen window and basis.
        # Assemble values each linearization, never rediscover support in it.
        self.spline_scatter=[]
        for node in range(5):
            obs,knot=np.nonzero(self.B[self.fi,node])
            shape=(len(obs),2,self.state.d)
            rr=np.broadcast_to(self.uv_rows[obs,:,None],shape).ravel().copy()
            cc=np.broadcast_to(knot[:,None,None]*self.state.d+np.arange(self.state.d),shape).ravel().copy()
            weight=self.B[self.fi[obs],node,knot,None,None]
            self.spline_scatter.append((obs,rr,cc,weight))
        self.spline_rows=np.concatenate([v[1] for v in self.spline_scatter])
        self.spline_cols=np.concatenate([v[2] for v in self.spline_scatter])
        # Sample actual delta/velocity/acceleration, with the taper included.
        self.prior_matrices=[]
        active=self.state.active;angular=active>=3
        for order,linear_scale,angular_degrees,name in [(0,.35,25,'reference_deviation'),(1,3.,400,'correction_velocity'),(2,15.,800,'correction_acceleration')]:
            scale=np.where(angular,np.deg2rad(angular_degrees),linear_scale)
            B=csr_matrix(self.state.basis(reference.times[self.frames],order))
            # Integral quadrature at native cadence; 25 Hz-equivalent old scale.
            matrix=kron(B,csr_matrix(np.diag(1/scale)),format='csr')/np.sqrt(2)
            self.prior_matrices.append((name,matrix))
        self.cache=None;self.cache_x=None

    def body_factors(self,uv):
        fi,j=self.of,self.oj;sig=self.obs['sigma_px'][self.frames[fi],j]
        point_ids=self.layout.body_landmarks[j]
        error=(uv[fi,point_ids]-self.obs['uv'][self.frames[fi],j])/sig[:,None]
        # Isotropic pseudo-Huber on each 2D group, not independent coordinates.
        c=2.;s=np.sum(error**2,axis=1);u=np.sqrt(1+s/c**2);k=np.sqrt(2/(u+1))
        dk=-k/(4*c*c*u*(u+1));D=(k[:,None,None]*np.eye(2)+2*dk[:,None,None]*error[:,:,None]*error[:,None,:])/sig[:,None,None]
        residual=(error*k[:,None]).ravel()
        cols=(fi*self.layout.point_count+point_ids)[:,None,None]*2+np.arange(2)[None,None,:]
        rows=np.arange(len(residual)).reshape(-1,2,1)
        A=coo_matrix((D.ravel(),(np.broadcast_to(rows,D.shape).ravel(),np.broadcast_to(cols,D.shape).ravel())),shape=(len(residual),uv.size)).tocsr()
        return residual,A,dict(body_pixels=float(residual@residual/2)),error

    def evaluate(self,x,jacobian=False):
        if self.cache_x is not None and np.array_equal(x,self.cache_x) and (not jacobian or self.cache['jac'] is not None):return self.cache
        tick=time.perf_counter();phase={}
        reuse=self.cache is not None and self.cache_x is not None and np.array_equal(x,self.cache_x)
        if reuse:
            forward=self.cache['_forward']
            delta,s,W,nodes,proj,uv,br,BA,bparts,errors,gr,GA,gparts,choices=forward
            start=time.perf_counter()
            proj=moving_projection(self.projector,nodes,self.frames[self.fi],jacobian=jacobian,projection=proj)
            phase.update(world_seconds=0.,projection_seconds=time.perf_counter()-start,factors_seconds=0.)
        else:
            start=time.perf_counter();delta=np.zeros((len(self.times),69))
            delta[:,self.state.active]=self.eval_basis@np.asarray(x).reshape(self.state.m,self.state.d)
            s=self.reference.world(self.ref,delta);W=s['world'].reshape(self.n,5,self.layout.point_count,3)
            nodes=W[self.fi,:,self.pi][:,:,None];phase['world_seconds']=time.perf_counter()-start
            start=time.perf_counter()
            proj=moving_projection(self.projector,nodes,self.frames[self.fi],jacobian=jacobian)
            phase['projection_seconds']=time.perf_counter()-start
            start=time.perf_counter();uv=np.full((self.n,self.layout.point_count,2),np.nan);uv[self.fi,self.pi]=proj['uv'][:,0]
            br,BA,bparts,errors=self.body_factors(uv)
            if self.gear.visible_rows or self.gear.blade_rows or self.gear.landmark_rows:
                edges=self.visible_boot_edges(W[:,2]) if hasattr(self,'visible_boot_edges') else None
                gr,GA,gparts,choices=pixel_factors(uv,self.gear,body=False,blade_side_candidates=True,boot_edges=edges,layout=self.layout)
            else:gr=np.empty(0);GA=csr_matrix((0,uv.size));gparts={};choices=[]
            phase['factors_seconds']=time.perf_counter()-start
        forward=(delta,s,W,nodes,proj,uv,br,BA,bparts,errors,gr,GA,gparts,choices)
        residual=[br,gr];parts={**bparts,**gparts};data_rows=len(br)+len(gr);matrices=[]
        if jacobian:
            start=time.perf_counter();jp=proj['pixel_jac_nodes'][:,0];derivatives=[]
            Jr=right_jacobian(delta[:,3:6]);rot_axes=s['Q']@Jr
            radial=s['world']-s['P'][:,None]
            columns=self.state.active[self.state.active>=6]
            physical=dict(self.execution.world_derivatives(self.reference,self.ref,delta,columns,step=1e-5))
            for active in self.state.active:
                if active<3:
                    derivative=np.zeros_like(W);derivative[...,active]=1
                elif active<6:
                    derivative=np.cross(rot_axes[:,:,active-3][:,None],radial).reshape(self.n,5,self.layout.point_count,3)
                else:
                    derivative=physical[int(active)].reshape(self.n,5,self.layout.point_count,3)
                derivatives.append(np.einsum('mplc,mlc->mpl',jp,derivative[self.fi,:,self.pi]))
            local=np.stack(derivatives,axis=-1)
            phase['local_derivatives_seconds']=time.perf_counter()-start
            start=time.perf_counter()
            # Only nonzero exposure-node basis supports are assembled.
            vv=[]
            for node in range(5):
                obs,_,_,weight=self.spline_scatter[node]
                values=local[obs,:,node,:]*weight
                vv.append(values.ravel())
            U=coo_matrix((np.concatenate(vv),(self.spline_rows,self.spline_cols)),shape=(uv.size,self.size)).tocsr()
            U.eliminate_zeros();matrices=[BA@U,GA@U]
            phase['sparse_chain_seconds']=time.perf_counter()-start
        for name,matrix in self.prior_matrices:
            r=matrix@x;residual.append(r);parts[name]=float(r@r/2)
            if jacobian:matrices.append(matrix)
        r=np.concatenate(residual);J=vstack(matrices,format='csr') if jacobian else None
        self.cache_x=np.array(x,copy=True);self.evaluations+=1
        self.cache=dict(residual=r,jac=J,parts=parts,uv=uv,world=s,data_rows=data_rows,choices=choices,
                        eval_seconds=time.perf_counter()-tick,normalized_body_error=errors,
                        phases=phase,forward_reused=bool(reuse),_forward=forward)
        if self.log:
            self.log.write(json.dumps(dict(event='evaluation',mode=self.state.mode,window=self.state.seconds,
                evaluation=self.evaluations,cost=float(r@r/2),parts=parts,with_jacobian=jacobian,
                seconds=time.perf_counter()-self.start,evaluation_seconds=self.cache['eval_seconds'],
                variables=self.size,source_frames=self.n,body_points=len(self.of),domain_failures=0,
                jacobian_nnz=None if J is None else J.nnz,phases=phase,forward_reused=bool(reuse)))+'\n')
        return self.cache

    def fun(self,x):return self.evaluate(x)['residual']
    def jac(self,x):return self.evaluate(x,True)['jac']

    def solve(self,output,*,initial_x=None):
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        if (output/'result.npz').exists():raise FileExistsError(output)
        x0=np.zeros(self.size) if initial_x is None else np.asarray(initial_x,float).copy()
        if x0.shape!=(self.size,):raise ValueError('Initial controls have the wrong state layout')
        initial=self.evaluate(x0,True)
        config=dict(schema='rinkspace.fullbody-window.v1',mode=self.state.mode,window_seconds=self.state.seconds,
            source_pts=self.reference.p['pts'][self.frames].tolist(),time_base=[1,12800],active_state_columns=self.state.active.tolist(),
            state_layout='P[0:3], right root Q[3:6], right local native1..21[6:69]',
            source_frames=self.n,body_points=len(self.of),knots=self.state.knots.tolist(),taper_seconds=self.state.taper,
            boundary='C2 quintic envelope returns to unchanged reference; outside window unchanged',
            frozen=['camera','scene','beta','body scale','bone lengths','shared equipment','boot installation','wrists','neck','collars'],
            wrist_orientation_observable=False,physical_contact_factor=False,reference_hashes=self.reference.hashes,
            priors='engineering reference/velocity/acceleration, no contact or gait assumptions',
            prior_scales=dict(root_displacement_m=.35,angular_displacement_deg=25,root_velocity_m_s=3,angular_velocity_deg_s=400,root_acceleration_m_s2=15,angular_acceleration_deg_s2=800),
            robust_body='2D isotropic pseudo-Huber, c=2 after engineering sigma',equipment_sources=self.gear_sources,
            observation_array_sha256=digest(self.ledger_path),
            backend='scipy sparse TRF/LSMR; native camera with existing moving-body implicit-row chain',
            execution=self.execution.record(),
            derivative='analytic root P/Q; local physical rotation finite differences at exposure nodes; sparse spline chain',
            ftol=1e-7,xtol=1e-7,gtol=1e-4,no_iteration_or_time_budget=True,
            initial_cost=float(initial['residual']@initial['residual']/2),initial_parts=initial['parts'])
        (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
        accepted_count=0
        def accepted(x):
            nonlocal accepted_count
            accepted_count+=1;v=self.evaluate(x,True)
            np.savez_compressed(output/'accepted.pending.npz',x=x,knots=self.state.knots,active=self.state.active)
            (output/'accepted.pending.npz').replace(output/'last_accepted.npz')
            if self.log:self.log.write(json.dumps(dict(event='accepted',mode=self.state.mode,window=self.state.seconds,
                iteration=accepted_count,cost=float(v['residual']@v['residual']/2),parts=v['parts'],
                gradient_inf=float(np.max(abs(v['jac'].T@v['residual']))),seconds=time.perf_counter()-self.start))+'\n')
        lower=np.broadcast_to(np.where(self.state.active<3,-np.inf,-np.pi),(self.state.m,self.state.d)).ravel()
        fit=least_squares(self.fun,x0,jac=self.jac,bounds=(lower,-lower),method='trf',tr_solver='lsmr',x_scale='jac',
            ftol=1e-7,xtol=1e-7,gtol=1e-4,callback=accepted,tr_options=dict(atol=1e-7,btol=1e-7))
        final=self.evaluate(fit.x,True)
        np.savez_compressed(output/'result.npz',x=fit.x,knots=self.state.knots,active=self.state.active,
            seconds=np.asarray(self.state.seconds),taper=np.asarray(self.state.taper),frames=self.frames,pts=self.reference.p['pts'][self.frames],
            COCO_uv=final['uv'][:,self.layout.body_landmarks],
            equipment_uv=final['uv'][:,:self.layout.equipment_count].reshape(self.n,len(self.layout.sides),self.layout.per_foot,2))
        report=dict(mode=self.state.mode,window=self.state.seconds,nfev=fit.nfev,njev=fit.njev,
            accepted_iterations=accepted_count,stopping_reason=str(fit.message),status=int(fit.status),
            optimality=float(fit.optimality),numerically_converged=bool(fit.optimality<=1e-4),
            gradient_inf=float(np.max(abs(final['jac'].T@final['residual']))),
            initial_cost=config['initial_cost'],final_cost=float(fit.cost),initial_parts=initial['parts'],final_parts=final['parts'],
            elapsed_seconds=time.perf_counter()-self.start,body_points=len(self.of),variables=self.size,
            blade_choices=final['choices'],user_accepted=False,independent_3d_truth=False)
        (output/'result.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        if self.log:self.log.write(json.dumps(dict(event='window_complete',**report))+'\n')
        return fit.x,report


def verify_problem(problem,output):
    """New chain checks on nonzero real poses, including row-time derivatives."""
    output=Path(output);x=np.zeros(problem.size);x.reshape(problem.state.m,problem.state.d)[2]=.007
    value=problem.evaluate(x,True);rng=np.random.default_rng(1102);checks=[]
    for name,select in [('root_P',problem.state.active<3),('root_Q',(problem.state.active>=3)&(problem.state.active<6)),
                        ('upper_limb',np.isin(problem.state.active,[6+(j-1)*3+a for j in (16,17,18,19) for a in range(3)])),
                        ('torso',np.isin(problem.state.active,[6+(j-1)*3+a for j in (3,6,9) for a in range(3)])),
                        ('wrist',np.isin(problem.state.active,[6+(j-1)*3+a for j in (20,21) for a in range(3)])),
                        ('legs',np.isin(problem.state.active,[6+(j-1)*3+a for j in (1,2,4,5,7,8) for a in range(3)]))]:
        if not select.any():continue
        direction=np.zeros((problem.state.m,problem.state.d));direction[:,select]=rng.normal(size=(problem.state.m,int(select.sum())))
        direction=direction.ravel();direction/=np.linalg.norm(direction);h=1e-5
        exact=value['jac']@direction;numeric=(problem.fun(x+h*direction)-problem.fun(x-h*direction))/(2*h)
        diff=float(np.max(abs(exact-numeric)));relative=float(np.linalg.norm(exact-numeric)/max(np.linalg.norm(numeric),1e-12))
        checks.append(dict(block=name,maximum_absolute=diff,relative_norm=relative))
        if relative>2e-4:raise ValueError(f'Derivative mismatch {name}: {relative}')
    output.write_text(json.dumps(checks,indent=2),encoding='utf8')
    return checks
