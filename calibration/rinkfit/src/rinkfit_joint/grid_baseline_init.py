"""Initialize the current physical camera family from paired 2-D ice grids.

Old grid UVs are numerical navigation targets, never new source observations.
This module does not consume historical R/C/K or import historical calibration.
The current shared scene, S01 camera function, parameter bounds and allocations
remain unchanged.  A produced candidate still requires native-source acceptance.
"""
from __future__ import annotations
import argparse
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from .io_utils import read_json,write_json,file_sha256,object_sha256
from .joint_model import as_event_view
from .role_guard import FrameRoleGuard,DENY_TRAIN
from .native_baseline_init import _physical,_free,_preserved_payload,_basis
from rinkfit_start.native_scene import NativeScene,camera_from_event
from rinkfit_start.native_camera import rotation


class GridTargets:
    """Ragged per-frame pairs, optionally backed by independent memmap arrays."""
    required=('frame_native_pts','frame_shot_number','frame_navigation_kind','frame_offsets','target_uv','current_world','material_index','path_index','station','curve_crosswalk_index')
    def __init__(self,manifest_file):
        self.path=Path(manifest_file);self.manifest=read_json(self.path);self.arrays={}
        refs=self.manifest.get('arrays',self.manifest.get('fields'))
        if isinstance(refs,dict) and 'crosswalk_index'in refs and 'curve_crosswalk_index'not in refs:refs={**refs,'curve_crosswalk_index':refs['crosswalk_index']}
        if isinstance(refs,dict):
            for name in self.required:
                ref=refs[name];path=Path(ref.get('file',ref.get('path')))
                if ref.get('sha256') and file_sha256(path)!=ref['sha256']:raise ValueError('Grid array changed: '+name)
                self.arrays[name]=np.load(path,mmap_mode='r',allow_pickle=False) if path.suffix=='.npy' else np.memmap(path,dtype=ref['dtype'],mode='r',shape=tuple(ref['shape']))
        else:
            ref=self.manifest['output'];path=Path(ref.get('file',ref.get('path')))
            if file_sha256(path)!=ref['sha256']:raise ValueError('Paired-grid archive changed')
            archive=np.load(path,allow_pickle=False)
            self.arrays={name:archive[name] for name in self.required}
        a=self.arrays;self.pts=np.asarray(a['frame_native_pts']);self.shot=np.asarray(a['frame_shot_number']);self.kind=np.asarray(a['frame_navigation_kind']);self.offset=np.asarray(a['frame_offsets'])
        F=len(self.pts);M=len(a['target_uv'])
        if self.pts.dtype.kind not in 'iu' or self.offset.dtype.kind not in 'iu' or self.offset.shape!=(F+1,) or self.offset[0]!=0 or self.offset[-1]!=M or np.any(np.diff(self.offset)<0):raise ValueError('Invalid original native ragged grid layout')
        if a['target_uv'].shape!=(M,2) or a['current_world'].shape!=(M,3):raise ValueError('Invalid paired grid coordinates')
        if not np.isfinite(a['target_uv']).all() or not np.isfinite(a['current_world']).all() or not np.isfinite(a['station']).all():raise ValueError('Unresolved grid references cannot masquerade as finite targets')
        if len(set(zip(self.pts.tolist(),self.shot.tolist())))!=F:raise ValueError('Duplicate or conflicting native frame identity')

    def rows(self,frame_mask):
        return np.concatenate([np.arange(self.offset[i],self.offset[i+1],dtype=np.int64) for i in np.flatnonzero(frame_mask)])


class GridMotion:
    """Original cubic map assembled as arrays, without per-sample Python tuples."""
    def __init__(self,model,sids,numbers,basis_ticks):
        self.model=model;self.N=len(basis_ticks);self.refs=[];self.x=[];self.lo=[];self.hi=[];self.scale=[];self.axis=[]
        r=[[],[],[]];c=[[],[],[]];v=[[],[],[]];fixed=np.zeros((self.N,3))
        for sid in sids:
            shot=model['shots'][sid];rows=np.flatnonzero(numbers==int(sid[1:]));t=(basis_ticks[rows]-shot['time_origin_pts'])/12800.;B=_basis(t,shot['camera'])
            if shot.get('control_coordinates'):raise ValueError('Non-S01 packed chart needs its existing explicit adapter')
            values=np.asarray([model['blocks'][bi]['values']for bi in shot['control_blocks']]);ref=model['blocks'][shot['reference_angles_block']];cam=shot['camera'];dep=cam['dependent_control'];gauge=np.zeros(len(values));gauge[cam['reference_indices']]=cam['reference_weights'];pivot=gauge[dep]
            for axis in range(3):
                A=B if axis==2 else B-B[:,dep]@sparse.csr_matrix((gauge/pivot)[None,:]);fixed[rows,axis]=A@values[:,axis]+(ref['values'][axis]if axis<2 else 0.)
                free=[];columns=[]
                for j,bi in enumerate(shot['control_blocks']):
                    block=model['blocks'][bi]
                    if (axis<2 and j==dep)or not _free(block,axis):continue
                    free.append(j);columns.append(len(self.x));self.refs.append((bi,axis));self.axis.append(axis);self.x.append(block['values'][axis]);self.lo.append(block['lower'][axis]);self.hi.append(block['upper'][axis]);self.scale.append(block['scales'][axis])
                co=A[:,free].tocoo();r[axis].append(rows[co.row]);c[axis].append(np.asarray(columns,dtype=np.int32)[co.col]);v[axis].append(co.data)
                if axis<2 and _free(ref,axis):
                    pi=len(self.x);self.refs.append((shot['reference_angles_block'],axis));self.axis.append(axis);self.x.append(ref['values'][axis]);self.lo.append(ref['lower'][axis]);self.hi.append(ref['upper'][axis]);self.scale.append(ref['scales'][axis]);r[axis].append(rows);c[axis].append(np.full(len(rows),pi,dtype=np.int32));v[axis].append(np.ones(len(rows)))
        self.x=np.asarray(self.x);self.lo=np.asarray(self.lo);self.hi=np.asarray(self.hi);self.scale=np.asarray(self.scale);self.D=[]
        for axis in range(3):self.D.append(sparse.csr_matrix((np.concatenate(v[axis]),(np.concatenate(r[axis]),np.concatenate(c[axis]))),shape=(self.N,len(self.x))))
        self.constant=fixed-np.column_stack([D@self.x for D in self.D]);self.coo=[D.tocoo(copy=False)for D in self.D]
    def values(self,x):return self.constant+np.column_stack([D@x for D in self.D])
    def store(self,x):
        for (bi,j),value in zip(self.refs,x):self.model['blocks'][bi]['values'][j]=float(value)


def _inverse_grid(uv,logf,o,meta):
    z=logf-np.log(meta['f_ref']);f=np.exp(logf);pp=o[:2]+z[:,None]*o[3:5];wanted=(uv-pp)/np.c_[f*np.exp(o[2]/2),f*np.exp(-o[2]/2)];xy=wanted.copy();k1=o[5]+o[6]*z;k2=o[7]+o[8]*z
    for _ in range(30):
        x,y=xy.T;r=x*x+y*y;L=1+k1*r+k2*r*r;A=k1+2*k2*r;e=np.c_[x*L+2*o[9]*x*y+o[10]*(r+2*x*x),y*L+o[9]*(r+2*y*y)+2*o[10]*x*y]-wanted
        a=L+2*x*x*A+2*o[9]*y+6*o[10]*x;b=2*x*y*A+2*o[9]*x+2*o[10]*y;d=L+2*y*y*A+6*o[9]*y+2*o[10]*x;det=a*d-b*b;step=np.c_[d*e[:,0]-b*e[:,1],a*e[:,1]-b*e[:,0]]/np.where(abs(det)>1e-12,det,1e-12)[:,None];step*=np.minimum(1.,.05/np.maximum(np.linalg.norm(step,axis=1),1e-30))[:,None];xy-=step
    x,y=xy.T;r=x*x+y*y;L=1+k1*r+k2*r*r;replay=np.c_[x*L+2*o[9]*x*y+o[10]*(r+2*x*x),y*L+o[9]*(r+2*y*y)+2*o[10]*x*y]
    error=np.linalg.norm((replay-wanted)*f[:,None],axis=1);return xy,error,np.sqrt(r)


def _reduced_angles(motion):
    import copy
    reduced=copy.copy(motion);take=np.asarray(motion.axis)!=2;reduced.refs=[p for p,k in zip(motion.refs,take)if k];reduced.axis=np.asarray(motion.axis)[take]
    reduced.x=motion.x[take];reduced.lo=motion.lo[take];reduced.hi=motion.hi[take];reduced.scale=motion.scale[take];reduced.D=[D[:,take]for D in motion.D];reduced.constant=motion.values(motion.x)-np.column_stack([D@reduced.x for D in reduced.D]);reduced.coo=[D.tocoo(copy=False)for D in reduced.D];return reduced


def _disk_lower(o,meta,lo,hi):
    z=np.array([lo,hi])-np.log(meta['f_ref']);k1=o[5]+o[6]*z;k2=o[7]+o[8]*z;R=meta['radius_max'];s=R*R
    def low(a,b):
        vertex=np.divide(-a,2*b,out=np.zeros_like(a),where=b!=0);return np.minimum(np.minimum(1.,1+a*s+b*s*s),np.where((b>0)&(vertex>0)&(vertex<s),1+a*vertex+b*vertex*vertex,np.inf))
    return float(min(low(k1,k2).min(),low(3*k1,5*k2).min())-6*np.hypot(o[9],o[10])*R)


class FiniteCurveQuery:
    """Eliminate legal 1-D stations numerically; no station optimization state."""
    def __init__(self,model,material,path,station,world):
        N=len(material);self.circle=np.zeros(N,bool);self.origin=np.zeros((N,3));self.vector=np.zeros((N,3));self.radius=np.zeros(N);self.start=np.zeros(N);self.span=np.zeros(N);self.seed=np.asarray(station).copy();self.last=None;self.model=model
        for mi,pi in np.unique(np.c_[material,path],axis=0):
            ids=(material==mi)&(path==pi);pa=model['materials'][int(mi)]['paths'][int(pi)];g=np.asarray(model['blocks'][pa['geometry_block']]['values'])
            if pa.get('allowed_station_intervals',[[0.,1.]])!=[[0.,1.]]:raise ValueError('Curve layout requires explicit original interval-union nearest point adapter')
            if pa['kind']==0:self.origin[ids,:2]=g[:2];self.vector[ids,:2]=g[2:4]-g[:2]
            elif pa['kind']==1:
                if pa.get('allowed_station_intervals',[[0.,1.]])!=[[0.,1.]]:raise ValueError('Curve layout needs explicit interval-union nearest-point handling')
                self.circle[ids]=True;self.origin[ids,:2]=g[:2];self.radius[ids]=g[pa['radius_index']];self.start[ids]=pa['start_angle'];self.span[ids]=pa['end_angle']-pa['start_angle']
            else:raise ValueError('Unsupported paired ice path for finite layout query')
        self.closed=self.circle&(abs(abs(self.span)-2*np.pi)<1e-10)
    def world(self,station):
        result=self.origin+station[:,None]*self.vector;angle=self.start+self.span*station;m=self.circle
        result[m,0]=self.origin[m,0]+self.radius[m]*np.cos(angle[m]);result[m,1]=self.origin[m,1]+self.radius[m]*np.sin(angle[m]);return result
    def constrain(self,s):return np.where(self.closed,np.mod(s,1.),np.clip(s,0.,1.))
    def residual(self,q,h,o,meta,target,*,ray=False):
        R,C=_physical(q,h,o,meta);z=q[:,2]-np.log(meta['f_ref']);f=np.exp(q[:,2]);pp=o[:2]+z[:,None]*o[3:5];k1=o[5]+o[6]*z;k2=o[7]+o[8]*z
        if ray:
            # Analytic plane line/conic normal navigation. This has the same
            # complete carrier zero set, without per-target mutable stations.
            # Finite support, actual RS and visibility remain posterior checks.
            v=np.einsum('nji,nj->ni',R,np.c_[target,np.ones(len(q))]);D=v[:,2];dx=C[:,0]-self.origin[:,0];dy=C[:,1]-self.origin[:,1];X=dx*D-C[:,2]*v[:,0];Y=dy*D-C[:,2]*v[:,1]
            dD=R[:,:2,2];dX=dx[:,None]*dD-C[:,2,None]*R[:,:2,0];dY=dy[:,None]*dD-C[:,2,None]*R[:,:2,1]
            n=np.c_[-self.vector[:,1],self.vector[:,0]];n/=np.maximum(np.linalg.norm(n,axis=1),1e-30)[:,None];g=np.sum(n*np.c_[X,Y],axis=1);gradient=n[:,0,None]*dX+n[:,1,None]*dY
            circle=self.circle;r2=self.radius*self.radius;g[circle]=(X*X+Y*Y-r2*D*D)[circle];gradient[circle]=(2*X[:,None]*dX+2*Y[:,None]*dY-2*(r2*D)[:,None]*dD)[circle]
            result=g[:,None]*gradient/np.maximum(np.sum(gradient*gradient,axis=1),1e-20)[:,None]*f[:,None]
            self.diagnostics=dict(preroll_method='analytic projective-plane line/conic Sampson normal',normal_RMS_native=float(np.sqrt(np.mean(np.sum(result*result,axis=1)))),finite_coverage_certified=False,actual_RS_certified=False,source_observation=False)
            return result
        def project(station):
            X=self.world(station);Y=np.einsum('nij,nj->ni',R,X-C);depth=Y[:,2];safe=np.where(abs(depth)>1e-9,depth,np.where(depth>=0,1e-9,-1e-9));xy=Y[:,:2]/safe[:,None]
            if ray:return xy,depth,np.linalg.norm(xy,axis=1)
            xx,yy=xy.T;r=xx*xx+yy*yy;L=1+k1*r+k2*r*r;uv=pp+np.c_[f*np.exp(o[2]/2)*(xx*L+2*o[9]*xx*yy+o[10]*(r+2*xx*xx)),f*np.exp(-o[2]/2)*(yy*L+o[9]*(r+2*yy*yy)+2*o[10]*xx*yy)]
            return uv,depth,np.sqrt(r)
        same=self.last is not None and self.last[0]==ray and np.array_equal(q,self.last[1])and np.array_equal(h,self.last[2])and np.array_equal(o,self.last[3])
        if same:return self.last_residual.copy()
        else:
            station=self.seed.copy();best=np.sum((project(station)[0]-target)**2,axis=1)
            # Complete finite-domain coarse navigation, followed by local 1-D
            # refinement. Only these transient double arrays are retained.
            for value in np.linspace(0.,1.,17):
                proposal=np.full(len(q),value);p,d,r=project(proposal);cost=np.sum((p-target)**2,axis=1);better=cost<best;station=np.where(better,proposal,station);best=np.minimum(best,cost)
        eps=2e-5
        for _ in range(12):
            p,d,r=project(station);plus=project(self.constrain(station+eps))[0];minus=project(self.constrain(station-eps))[0];den=np.where(self.closed,2*eps,np.minimum(1.,station+eps)-np.maximum(0.,station-eps));tangent=(plus-minus)/np.maximum(den,1e-15)[:,None];e=p-target;step=np.sum(e*tangent,axis=1)/np.maximum(np.sum(tangent*tangent,axis=1),1e-20);step=np.clip(step,-.15,.15);proposal=self.constrain(station-step);trial=project(proposal)[0];worse=np.sum((trial-target)**2,axis=1)>np.sum(e*e,axis=1)
            for _ in range(4):
                if not worse.any():break
                step[worse]*=.5;proposal=self.constrain(station-step);trial=project(proposal)[0];worse=np.sum((trial-target)**2,axis=1)>np.sum(e*e,axis=1)
            station=np.where(worse,station,proposal)
            if np.max(abs(step),initial=0)<1e-9:break
        p,depth,radius=project(station);plus=project(self.constrain(station+eps))[0];minus=project(self.constrain(station-eps))[0];tangent=plus-minus;length=np.linalg.norm(tangent,axis=1);normal=np.c_[-tangent[:,1],tangent[:,0]]/np.maximum(length,1e-20)[:,None];error=p-target;normal_residual=normal*np.sum(normal*error,axis=1)[:,None]
        # At a finite endpoint, tangential gap remains a coverage failure and
        # cannot rotate a camera to absorb a different physical line extent.
        endpoint=(~self.closed)&((station<1e-7)|(station>1-1e-7));tangential=np.abs(np.sum(tangent/np.maximum(length,1e-20)[:,None]*error,axis=1));self.diagnostics=dict(full_2D_RMS=float(np.sqrt(np.mean(np.sum(error*error,axis=1)))),normal_RMS=float(np.sqrt(np.mean(np.sum(normal_residual*normal_residual,axis=1)))),endpoint_count=int(endpoint.sum()),endpoint_gap_over_1px=int((endpoint&(tangential>1.)).sum()),endpoint_gap_max=float(np.max(np.where(endpoint,tangential,0.),initial=0)),negative_depth=int((depth<=0).sum()),outside_domain=int((radius>meta['radius_max']).sum()))
        self.last=(ray,q.copy(),h.copy(),o.copy(),station.copy());self.last_normal=normal.copy();self.last_depth=depth;self.last_radius=radius
        if ray:
            self.last_residual=normal_residual*f[:,None];return self.last_residual.copy()
        # A valid current disk cannot authorize an actual folded outer ray.
        rr=radius*radius;radial=1+3*k1*rr+5*k2*rr*rr-6*np.hypot(o[9],o[10])*radius
        if np.any(depth<=0)or np.any(radial<=1e-10):self.last_residual=np.full((len(q),2),1e12)
        else:self.last_residual=normal_residual.copy()
        return self.last_residual.copy()


def _fit_grid(motion,h,o,meta,X,target,static,max_nfev,*,ray_target=None,curve_query=None,checkpoint_dir=None,stage_name=None):
    """Vectorized per-point derivatives chained to the original sparse cubic map."""
    nh=len(static);blocks=motion.model['blocks'];rig=motion.rig
    def block(which):return blocks[rig['head_block' if which=='head' else 'optics_block']]
    x0=np.r_[[block(which)['values'][j] for which,j in static],motion.x]
    lo=np.r_[[block(which)['lower'][j] for which,j in static],motion.lo];hi=np.r_[[block(which)['upper'][j] for which,j in static],motion.hi]
    scale=np.r_[[block(which)['scales'][j] for which,j in static],motion.scale]
    def unpack(x):
        head=h.copy();optic=o.copy()
        for value,(which,j) in zip(x[:nh],static):(head if which=='head' else optic)[j]=value
        return motion.values(x[nh:]),head,optic
    original_logf=np.asarray([motion.model['blocks'][bi]['values'][2]for sid,shot in motion.model['shots'].items()if shot['rig_id']==next(k for k,v in motion.model['rigs'].items()if v is rig) for bi in shot['control_blocks']])
    def residual(q,head,optic):
        if curve_query is not None:
            return curve_query.residual(q,head,optic,meta,ray_target if ray_target is not None else target,ray=ray_target is not None)
        if ray_target is not None:
            R,C=_physical(q,head,optic,meta);Y=np.einsum('nij,nj->ni',R,X-C);safe=np.where(Y[:,2]>1e-9,Y[:,2],1e-9)
            return (Y[:,:2]/safe[:,None]-ray_target)*np.exp(q[:,2,None])
        if _disk_lower(optic,meta,min(original_logf.min(),q[:,2].min()),max(original_logf.max(),q[:,2].max()))<=1e-10:return np.full((len(X),2),1e12)
        uv,depth,radius,R,C=_physical(q,head,optic,meta,X[:,None,:]);Y=np.einsum('nij,nj->ni',R,X-C);safe=np.where(Y[:,2]>1e-9,Y[:,2],1e-9);xy=Y[:,:2]/safe[:,None];xx,yy=xy.T;z=q[:,2]-np.log(meta['f_ref']);k1=optic[5]+optic[6]*z;k2=optic[7]+optic[8]*z;r=xx*xx+yy*yy;L=1+k1*r+k2*r*r;A=k1+2*k2*r;a=L+2*xx*xx*A+2*optic[9]*yy+6*optic[10]*xx;b=2*xx*yy*A+2*optic[9]*xx+2*optic[10]*yy;d=L+2*yy*yy*A+6*optic[9]*yy+2*optic[10]*xx
        if np.any(depth<=0)or np.any(a*d-b*b<=1e-10)or np.any(a+d<=0):return np.full((len(X),2),1e12)
        return np.nan_to_num(uv[:,0]-target,nan=1e12,posinf=1e12,neginf=-1e12)
    def fun(x):
        q,head,optic=unpack(x)
        candidate_logf=x[nh:][np.asarray(motion.axis)==2]
        if ray_target is None and len(candidate_logf) and _disk_lower(optic,meta,min(original_logf.min(),candidate_logf.min()),max(original_logf.max(),candidate_logf.max()))<=1e-10:return np.full(2*len(X),1e12)
        return residual(q,head,optic).ravel()
    # One reusable COO value buffer; no list of dense static columns.
    nstatic=2*len(X)*nh;nnz=nstatic+2*sum(D.nnz for D in motion.D)
    jrow=np.empty(nnz,np.int32);jcol=np.empty(nnz,np.int32);jvalue=np.empty(nnz,np.float64)
    if nh:
        jrow[:nstatic].reshape(2*len(X),nh)[:]=np.arange(2*len(X),dtype=np.int32)[:,None]
        jcol[:nstatic].reshape(2*len(X),nh)[:]=np.arange(nh,dtype=np.int32)
    offset=nstatic
    for D in motion.coo:
        n=2*len(D.data);jrow[offset:offset+n].reshape(-1,2)[:]=2*D.row[:,None]+np.arange(2,dtype=np.int32);jcol[offset:offset+n].reshape(-1,2)[:]=nh+D.col[:,None];offset+=n
    def jac(x):
        q,head,optic=unpack(x);base=residual(q,head,optic);eps=2e-7
        for col,(which,j) in enumerate(static):
            h1=head.copy();o1=optic.copy();(h1 if which=='head' else o1)[j]+=eps
            jvalue[:nstatic].reshape(2*len(X),nh)[:,col]=((residual(q,h1,o1)-base)/eps).ravel()
        offset=nstatic
        for axis,D in enumerate(motion.coo):
            q1=q.copy();q1[:,axis]+=eps;derivative=(residual(q1,head,optic)-base)/eps;n=2*len(D.data)
            jvalue[offset:offset+n].reshape(-1,2)[:]=D.data[:,None]*derivative[D.row];offset+=n
        return sparse.csr_matrix((jvalue,(jrow,jcol)),shape=(2*len(X),len(x)),copy=True)
    def callback(intermediate_result):
        if checkpoint_dir is None:return
        import copy
        q,head,optic=unpack(intermediate_result.x);candidate=copy.deepcopy(motion.model)
        for (bi,j),value in zip(motion.refs,intermediate_result.x[nh:]):candidate['blocks'][bi]['values'][j]=float(value)
        for value,(which,j) in zip(intermediate_result.x[:nh],static):candidate['blocks'][rig['head_block'if which=='head'else 'optics_block']]['values'][j]=float(value)
        state=dict(stage=stage_name,nfev=int(intermediate_result.nfev),cost=float(intermediate_result.cost),accepted_numerical_navigation_step=True,source_acceptance=False,complete_joint_working_state=True)
        candidate['grid_navigation_work_state']=state;directory=Path(checkpoint_dir);directory.mkdir(parents=True,exist_ok=True);temp=directory/'model.tmp.json';write_json(temp,candidate);temp.replace(directory/'model.json');ref=dict(**state,model=dict(path=str(directory/'model.json'),sha256=file_sha256(directory/'model.json')))
        write_json(directory/'receipt.json',ref)
        with (directory/'accepted_steps.jsonl').open('a',encoding='utf8')as f:f.write(json.dumps(ref,separators=(',',':'))+'\n')
    start=time.perf_counter();before=fun(x0)
    result=least_squares(fun,x0,jac=jac,bounds=(lo,hi),x_scale=scale,tr_solver='lsmr',tr_options=dict(atol=1e-9,btol=1e-9),ftol=1e-10,xtol=1e-10,gtol=1e-5,max_nfev=max_nfev,verbose=2,callback=callback)
    q,head,optic=unpack(result.x);motion.store(result.x[nh:])
    report=dict(seconds=time.perf_counter()-start,points=len(X),residuals=2*len(X),parameters=len(x0),sparse_nnz=nnz,COO_work_bytes=jrow.nbytes+jcol.nbytes+jvalue.nbytes,static_coordinates=[list(p) for p in static],cost_before=float(before@before/2),cost_after=float(result.fun@result.fun/2),nfev=result.nfev,njev=result.njev,status=int(result.status),message=result.message,finite_curve_layout=curve_query.diagnostics if curve_query is not None else None,objective='same-entity finite-curve normal layout; transient 1D station elimination, endpoint tangential gaps excluded and reported; no source W')
    print(json.dumps(report),flush=True);return head,optic,report


def _memory():
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(n,ctypes.c_size_t)for n in ('peak_ws','ws','peak_paged_pool','paged_pool','peak_nonpaged_pool','nonpaged_pool','commit','peak_commit','private_commit')]
    m=Counters();m.cb=ctypes.sizeof(m)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(m),m.cb):return None
    return dict(working_set_bytes=m.ws,peak_working_set_bytes=m.peak_ws,private_commit_bytes=m.private_commit,peak_commit_bytes=m.peak_commit)


def _posterior(model,grid,out):
    """Every paired point remains in the posterior, including protected frames."""
    from .joint_replay import render_visibility
    a=grid.arrays;M=len(a['target_uv']);metrics=np.memmap(out/'all_grid_posterior.f64',dtype='<f8',mode='w+',shape=(M,7));reports={};scene=NativeScene(as_event_view(model,'s01'))
    # Columns: target-row pixel error, radius, depth, actual-RS pixel error,
    # actual-RS valid, actual RS equality, finite-scene visibility.
    for sid,s in model['shots'].items():
        if sid=='s01':continue
        mask=grid.shot==int(sid[1:]);ids=grid.rows(mask);count=np.diff(grid.offset)[mask];pts=np.repeat(grid.pts[mask],count);local=(pts-s['time_origin_pts'])/12800.;cam=camera_from_event(as_event_view(model,sid));ref=np.asarray(model['blocks'][s['reference_angles_block']]['values']);cam.R0=rotation(cam.b,ref[1])@cam.R0@rotation(cam.n,ref[0]);totals=dict(points=len(ids),RS_valid=0,RS_invalid=0,negative_depth=0,outside_ray_domain=0,hidden=0);sum_error=0.;max_error=0.
        for begin in range(0,len(ids),32768):
            ii=ids[begin:begin+32768];t=local[begin:begin+32768];X=np.asarray(a['current_world'][ii]);target=np.asarray(a['target_uv'][ii]);query=FiniteCurveQuery(model,np.asarray(a['material_index'][ii]),np.asarray(a['path_index'][ii]),np.asarray(a['station'][ii]),X);conditional=cam.spline(t+cam.tau*(target[:,1]/1079-.5));conditional[:,:2]+=ref;rig=model['rigs'][s['rig_id']];hh=np.asarray(model['blocks'][rig['head_block']]['values']);oo=np.asarray(model['blocks'][rig['optics_block']]['values']);query.residual(conditional,hh,oo,rig['camera'],target);X=query.world(query.last[4]);p=cam.at(X,t+cam.tau*(target[:,1]/1079-.5));rs=cam.project_rs(X,t);visible=render_visibility(cam,scene,X,[dict(label='ice.grid',begin=0,end=len(ii),board_material_surface=False)],t,rs['uv']);error=np.linalg.norm(rs['uv']-target,axis=1);metrics[ii]=np.c_[np.linalg.norm(p['uv']-target,axis=1),p['ray_radius'],p['depth'],error,rs['valid'].astype(float),rs['root_equality_px'],visible.astype(float)];totals['RS_valid']+=int(rs['valid'].sum());totals['RS_invalid']+=int((~rs['valid']).sum());totals['negative_depth']+=int((p['depth']<=0).sum());totals['outside_ray_domain']+=int((p['ray_radius']>cam.optics['radius_max']).sum());totals['hidden']+=int((~visible).sum());sum_error+=float(np.sum(error*error));max_error=max(max_error,float(error.max(initial=0.)))
        totals.update(all_point_actual_RS_RMS_px=float(np.sqrt(sum_error/len(ids)))if len(ids)else None,maximum_actual_RS_error_px=max_error,full_original_control_hull_optical_certificate=cam.optical_domain_certificate(),empty_native_frames=int((count==0).sum()),all_native_frames=int(mask.sum()));reports[sid]=totals;print(json.dumps({'posterior':{sid:totals}}),flush=True)
    metrics.flush();return reports


def initialize_grid(model_file,target_manifest,roles_file,outdir,*,rig_stages=None,max_nfev=100):
    """Fit paired old UV/current world targets; preserve the complete S01 state.

    rig_stages maps secondary/end to lists of coordinate selections, e.g.
    {'secondary':[{'head':list(range(15)),'optics':list(range(11))}]}.
    Every rig first receives a motion-only stage. Main static coordinates are
    always preserved here. Signed readout is held at the input value during
    navigation, then remains its original production degree of freedom.
    """
    start=time.perf_counter();out=Path(outdir)
    if out.exists():raise ValueError('Use a new immutable paired-grid initializer generation')
    model=read_json(model_file);grid=GridTargets(target_manifest);a=grid.arrays;guard=FrameRoleGuard(roles_file)
    if model['source_clock']['time_base']!=[1,12800] or model['source_clock']['frame_step_pts']!=256:raise ValueError('Current official native source clock required')
    training=np.zeros(len(grid.pts),bool)
    for i,(p,sn) in enumerate(zip(grid.pts,grid.shot)):
        p=int(p);sid=f's{int(sn):02d}'
        if sid not in model['shots'] or p not in guard.roles or guard.roles[p]['shot_id']!=sid:raise ValueError('Unknown native grid frame or cross-cut identity')
        scope=model['shots'][sid]['scope']
        if not scope['first_pts']<=p<=scope['last_pts'] or (p-scope['first_pts'])%256:raise ValueError('Non-native grid exposure')
        if sid!='s01' and guard.roles[p]['role'] not in DENY_TRAIN and int(grid.kind[i])!=0:guard.assert_training_pixels([p],shot_id=sid);training[i]=grid.offset[i+1]>grid.offset[i]
    scene=NativeScene(as_event_view(model,'s01'));world_error=0.
    pairs=np.stack((a['material_index'],a['path_index']),axis=1)
    for mi,pi in np.unique(pairs,axis=0):
        mi=int(mi);pi=int(pi)
        if not model['materials'][mi]['label'].startswith('ice.'):raise ValueError('Paired-grid initializer permits explicitly paired ice entities only')
        ids=np.flatnonzero((pairs[:,0]==mi)&(pairs[:,1]==pi));s=np.asarray(a['station'][ids])
        if np.any(s<0) or np.any(s>1):raise ValueError('Paired station outside original finite domain')
        replay=scene.world(mi,pi,s);world_error=max(world_error,float(np.max(np.abs(replay-a['current_world'][ids]))))
    if world_error>2e-12:raise ValueError('Target current-world values do not belong to the unchanged current scene')
    changed={model['rigs'][r][key] for r in ('secondary','end') for key in ('head_block','optics_block')}
    for sid,s in model['shots'].items():
        if sid!='s01':changed.update(s['control_blocks']);changed.add(s['reference_angles_block'])
    frozen=object_sha256(_preserved_payload(model,changed));structure=object_sha256(dict(blocks=[{k:v for k,v in b.items() if k!='values'} for b in model['blocks']],shots=model['shots'],rigs=model['rigs']))
    if rig_stages is None:rig_stages={rid:[dict(head=list(range(15)),optics=list(range(11)))] for rid in ('secondary','end')}
    reports={};metrics=[]
    for rid in ('main','end','secondary'):
        sids=[s for s,v in model['shots'].items() if s!='s01' and v['rig_id']==rid];selection=training&np.isin(grid.shot,[int(s[1:])for s in sids]);frames=np.flatnonzero(selection)
        if not len(frames):raise ValueError('No role-allowed paired grid frames for '+rid)
        ids=grid.rows(selection);counts=np.diff(grid.offset)[selection];pts=np.repeat(grid.pts[selection],counts);numbers=np.repeat(grid.shot[selection],counts)
        X=np.asarray(a['current_world'][ids]);target=np.asarray(a['target_uv'][ids]);rig=model['rigs'][rid];hb=model['blocks'][rig['head_block']];ob=model['blocks'][rig['optics_block']];h=np.asarray(hb['values']);o=np.asarray(ob['values']);original_tau=float(o[11])
        # These floating ticks are only a basis evaluation coordinate. Native
        # source identity above remains its original integer PTS, never remapped.
        basis_ticks=pts.astype(float)+original_tau*(target[:,1]/1079-.5)*12800.
        motion=GridMotion(model,sids,numbers,basis_ticks);motion.rig=rig;initialq=motion.values(motion.x)
        before=_physical(initialq,h,o,rig['camera'],X[:,None,:])[0][:,0]-target
        curve_query=FiniteCurveQuery(model,np.asarray(a['material_index'][ids]),np.asarray(a['path_index'][ids]),np.asarray(a['station'][ids]),X)
        stages=[]
        # First rotate rays toward inverses of the actual paired grid UVs. This
        # never evaluates the physical Brown polynomial on a folded world ray.
        # A uniform existing shot-logf seed offset is used only when target UVs
        # have no legal inverse at the input zoom; no camera calibration is copied.
        for sn in np.unique(numbers):
            mask=numbers==sn;sid=f's{int(sn):02d}';shot=model['shots'][sid]
            for attempt in range(20):
                print(json.dumps({'phase':'inverse_grid_seed','rig':rid,'shot':sid,'attempt':attempt,'points':int(mask.sum())}),flush=True)
                current=motion.values(motion.x);inverse,inv_error,rr=_inverse_grid(target[mask],current[mask,2],o,rig['camera'])
                if np.max(inv_error)<1e-6 and np.max(rr)<.9*rig['camera']['radius_max']:break
                for bi in shot['control_blocks']:
                    block=model['blocks'][bi]
                    if not _free(block,2)or block['values'][2]+.1>block['upper'][2]:raise ValueError('No original-bound logf seed with legal target inverse')
                    block['values'][2]+=.1
                motion=GridMotion(model,sids,numbers,basis_ticks);motion.rig=rig
            else:raise ValueError('Paired UVs have no certified target-bearing seed')
        current=motion.values(motion.x);inverse,inv_error,rr=_inverse_grid(target,current[:,2],o,rig['camera']);angular=_reduced_angles(motion);angular.rig=rig
        print(json.dumps({'phase':'finite_curve_preroll','rig':rid,'points':len(X)}),flush=True)
        h,o,preroll=_fit_grid(angular,h,o,rig['camera'],X,target,[],max_nfev,ray_target=inverse,curve_query=curve_query,checkpoint_dir=out.parent/(out.name+'_partial'),stage_name=rid+'/preroll');stages.append(preroll)
        motion=GridMotion(model,sids,numbers,basis_ticks);motion.rig=rig
        for stage_index,stage in enumerate([{}]+([] if rid=='main' else rig_stages.get(rid,[]))):
            static=[]
            for which,size,block in (('head',15,hb),('optics',11,ob)):
                for j in stage.get(which,[]):
                    if not isinstance(j,int) or not 0<=j<size:raise ValueError('Invalid existing static navigation coordinate')
                    if _free(block,j):static.append((which,j))
            h,o,report=_fit_grid(motion,h,o,rig['camera'],X,target,static,max_nfev,curve_query=curve_query,checkpoint_dir=out.parent/(out.name+'_partial'),stage_name=rid+'/pixel/'+str(stage_index));hb['values']=h.tolist();ob['values']=o.tolist();motion=GridMotion(model,sids,numbers,basis_ticks);motion.rig=rig;stages.append(report)
        q=motion.values(motion.x);pred,depth,radius,_,_=_physical(q,h,o,rig['camera'],X[:,None,:]);error=np.linalg.norm(pred[:,0]-target,axis=1)
        if float(o[11])!=original_tau:raise ValueError('Navigation changed held readout')
        curve_query.residual(q,h,o,rig['camera'],target)
        reports[rid]=dict(finite_curve_layout=curve_query.diagnostics,memory=_memory(),motion_rows=motion.N,motion_nnz=[D.nnz for D in motion.D],motion_CPU_bytes=sum(D.data.nbytes+D.indices.nbytes+D.indptr.nbytes for D in motion.D)+sum(C.row.nbytes for C in motion.coo),shots=sids,fit_frames=len(frames),fit_points=len(ids),stages=stages,before_fraction_pair_difference_diagnostic_only_RMS_px=float(np.sqrt(np.mean(np.sum(before*before,axis=1)))),after_fraction_pair_difference_diagnostic_only_RMS_px=float(np.sqrt(np.mean(error*error))),maximum_fraction_pair_difference_diagnostic_only_px=float(error.max()),negative_depth_points=int((depth[:,0]<=0).sum()),outside_current_ray_domain=int((radius[:,0]>rig['camera']['radius_max']).sum()))
        metrics.append((ids,error,radius[:,0],depth[:,0]));print(json.dumps({rid:reports[rid]}),flush=True)
    if frozen!=object_sha256(_preserved_payload(model,changed)):raise ValueError('S01/shared scene/main physical state changed')
    if structure!=object_sha256(dict(blocks=[{k:v for k,v in b.items() if k!='values'} for b in model['blocks']],shots=model['shots'],rigs=model['rigs'])):raise ValueError('Bounds, free flags, allocation or static chart changed')
    for b in model['blocks']:
        v=np.asarray(b['values'])
        if not np.isfinite(v).all() or np.any(v<b['lower']) or np.any(v>b['upper']):raise ValueError('Candidate outside original parameter domain')
    model['paired_grid_initializer']=dict(schema='rinkfit.current_family_paired_ice_grid_initializer.v1',input_model=dict(path=str(Path(model_file).resolve()),sha256=file_sha256(model_file)),paired_grid=dict(path=str(Path(target_manifest).resolve()),sha256=file_sha256(target_manifest)),roles=dict(path=str(Path(roles_file).resolve()),sha256=file_sha256(roles_file)),targets_are_numerical_navigation_not_observations=True,historical_calibration_imported=False,historical_R_C_K_loss_used=False,fractional_station_is_not_a_physical_point_correspondence=True,camera_loss='same-entity finite-curve normal distance; endpoint tangential gap report only',S01_shared_scene_main_static_preserved_sha256=frozen,UV_Query_W_qualification_modified=False,original_structure_sha256=structure,requires_native_source_auxiliary_rebuild_and_whole_shot_validation=True,accepted=False)
    out.mkdir(parents=True);write_json(out/'model.json',model)
    posterior=_posterior(model,grid,out)
    np.savez(out/'fit_grid_metrics.npz',target_row_index=np.concatenate([m[0]for m in metrics]),pixel_error=np.concatenate([m[1]for m in metrics]),ray_radius=np.concatenate([m[2]for m in metrics]),depth=np.concatenate([m[3]for m in metrics]))
    report=dict(schema='rinkfit.current_family_paired_ice_grid_report.v1',seconds=time.perf_counter()-start,model=dict(path=str((out/'model.json').resolve()),sha256=file_sha256(out/'model.json')),frames_all=len(grid.pts),points_all=len(a['target_uv']),frames_used=int(training.sum()),frame_navigation_kinds=dict(Counter(str(int(x))for x in grid.kind)),current_world_replay_max_abs=world_error,rigs=reports,all_point_posterior=posterior,memory=_memory(),frames_without_paired_points=int((np.diff(grid.offset)==0).sum()),targets_not_in_W=True,source_observations_changed=False,production_fit_started=False,source_accuracy_accepted=False)
    write_json(out/'report.json',report);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--grid-manifest',required=True);p.add_argument('--roles',required=True);p.add_argument('--out',required=True);p.add_argument('--rig-stages');p.add_argument('--max-nfev',type=int,default=100);a=p.parse_args()
    stages=read_json(a.rig_stages) if a.rig_stages else None
    print(json.dumps(initialize_grid(a.model,a.grid_manifest,a.roles,a.out,rig_stages=stages,max_nfev=a.max_nfev)),flush=True)


if __name__=='__main__':main()
