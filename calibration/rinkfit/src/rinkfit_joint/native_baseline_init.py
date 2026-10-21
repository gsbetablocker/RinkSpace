"""Bounded transfer of complete historical native navigation into the current family.

This produces an unaccepted joint initializer, never a source likelihood or a
production checkpoint.  Source roles, all spline allocations/bounds/gauges and
S01's coupled physical state remain intact.  Dense old states are approximated;
no exact-anchor claim is made.  Interpolation remains numerical navigation.
"""
from __future__ import annotations
import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares, lsq_linear
from scipy.spatial.transform import Rotation

from .io_utils import read_json, write_json, file_sha256, object_sha256
from .joint_model import as_event_view
from .role_guard import FrameRoleGuard, DENY_TRAIN
from rinkfit_start.native_camera import rotation
from rinkfit_start.native_scene import NativeScene, camera_from_event


def _head(h, meta):
    v=h[:3]; a=np.linalg.norm(v)
    R0=(rotation(v/a,a) if a else np.eye(3))@np.asarray(meta['R0_seed']).reshape(3,3)
    n=np.asarray(meta['n_seed'])+h[3]*np.asarray(meta['n_tangent1'])+h[4]*np.asarray(meta['n_tangent2'])
    n=n/np.linalg.norm(n); q=R0@n; seed=np.asarray(meta['b_seed'])
    e=seed-q*(seed@q); e=e/np.linalg.norm(e)
    b=e*np.cos(h[5])+np.cross(q,e)*np.sin(h[5])
    return R0,n,b


def _physical(q,h,o,meta,X=None):
    R0,n,b=_head(h,meta)
    R=rotation(b,q[:,1])@R0@rotation(n,q[:,0])
    z=q[:,2]-np.log(meta['f_ref'])
    normal=np.cross(n,np.einsum('nji,j->ni',R,b)); normal/=np.linalg.norm(normal,axis=1)[:,None]
    d=np.broadcast_to(h[9:12],(len(q),3)).copy(); d[:,2]+=h[13]*z+h[14]*z*z
    C=h[6:9]+np.einsum('nji,nj->ni',R,d)+h[12]*normal
    if X is None:return R,C
    Y=np.einsum('nij,nkj->nki',R,X-C[:,None,:]); depth=Y[...,2]
    safe=np.where(np.abs(depth)>1e-9,depth,np.where(depth>=0,1e-9,-1e-9))
    xx=Y[...,0]/safe; yy=Y[...,1]/safe; r2=xx*xx+yy*yy
    k1=o[5]+o[6]*z; k2=o[7]+o[8]*z
    extension=meta.get('brown_wide_extension')
    if extension:
        basis=np.maximum(float(extension['activation_log_f_native'])-q[:,2],0.)**2
        ec=np.asarray(extension['coefficients'])
        k1=k1+ec[0]*basis;k2=k2+ec[1]*basis
        p1=o[9]+ec[2]*basis;p2=o[10]+ec[3]*basis
    else:p1=np.full(len(q),o[9]);p2=np.full(len(q),o[10])
    radial=1+k1[:,None]*r2+k2[:,None]*r2*r2
    xd=xx*radial+2*p1[:,None]*xx*yy+p2[:,None]*(r2+2*xx*xx)
    yd=yy*radial+p1[:,None]*(r2+2*yy*yy)+2*p2[:,None]*xx*yy
    f=np.exp(q[:,2]); pp=o[:2]+z[:,None]*o[3:5]
    uv=pp[:,None,:]+np.stack((f[:,None]*np.exp(o[2]/2)*xd,f[:,None]*np.exp(-o[2]/2)*yd),axis=-1)
    return uv,depth,np.sqrt(r2),R,C


def _basis(t,camera):
    n=int(camera['n_controls']); x=(np.asarray(t)-camera['first_control_time'])/camera['dt']
    cell=np.floor(x).astype(int); cell=np.where(cell==n-2,cell-1,cell); u=x-cell
    if np.any(cell<1) or np.any(cell>n-3):raise ValueError('Native time outside original allocated cubic support')
    weights=np.stack(((1-u)**3,3*u**3-6*u*u+4,-3*u**3+3*u*u+3*u+1,u**3),axis=1)/6
    return sparse.csr_matrix((weights.ravel(),(np.repeat(np.arange(len(t)),4),(cell[:,None]+np.arange(-1,3)).ravel())),shape=(len(t),n))


def _free(block,j):
    return not block.get('constant',False) and j not in block.get('fixed_indices',[]) and block['lower'][j]<block['upper'][j]


class Motion:
    """The original independent raw coordinates, with the original gauge eliminated."""
    def __init__(self,model,sids,shot_numbers,pts):
        self.model=model; self.sids=list(sids); self.N=len(pts); self.refs=[]; self.axis=[]
        self.x=[];self.lo=[];self.hi=[];self.scale=[]; entries=[[],[],[]]; fixed=np.zeros((self.N,3))
        for sid in sids:
            s=model['shots'][sid]; mask=shot_numbers==int(sid[1:]); rows=np.flatnonzero(mask)
            t=(pts[mask]-s['time_origin_pts'])/12800.; B=_basis(t,s['camera']); n=B.shape[1]
            if s.get('control_coordinates'):raise ValueError('Non-S01 packed chart needs explicit existing manifold adapter')
            controls=np.asarray([model['blocks'][i]['values'] for i in s['control_blocks']])
            ref=model['blocks'][s['reference_angles_block']]; cam=s['camera']; dep=cam['dependent_control']
            gauge=np.zeros(n);gauge[cam['reference_indices']]=cam['reference_weights']; pivot=gauge[dep]
            if not pivot:raise ValueError('Invalid immutable dependent-control gauge')
            for axis in range(3):
                A=B if axis==2 else B-B[:,dep]@sparse.csr_matrix((gauge/pivot)[None,:])
                if axis<2:A=A.tolil();A[:,dep]=0;A=A.tocsr()
                fixed[rows,axis]=np.asarray(A@controls[:,axis]).ravel()+(ref['values'][axis] if axis<2 else 0.)
                for j,bi in enumerate(s['control_blocks']):
                    block=model['blocks'][bi]
                    if (axis<2 and j==dep) or not _free(block,axis):continue
                    col=A[:,j].tocoo(); pi=len(self.x); self.refs.append((bi,axis));self.axis.append(axis)
                    for rr,val in zip(col.row,col.data):entries[axis].append((int(rows[rr]),pi,float(val)))
                    self.x.append(block['values'][axis]);self.lo.append(block['lower'][axis]);self.hi.append(block['upper'][axis]);self.scale.append(block['scales'][axis])
                if axis<2 and _free(ref,axis):
                    pi=len(self.x);self.refs.append((s['reference_angles_block'],axis));self.axis.append(axis)
                    entries[axis].extend((int(rr),pi,1.) for rr in rows)
                    self.x.append(ref['values'][axis]);self.lo.append(ref['lower'][axis]);self.hi.append(ref['upper'][axis]);self.scale.append(ref['scales'][axis])
        self.x=np.asarray(self.x);self.lo=np.asarray(self.lo);self.hi=np.asarray(self.hi);self.scale=np.asarray(self.scale);self.axis=np.asarray(self.axis)
        self.D=[]
        for e in entries:
            rr,cc,vv=map(np.asarray,zip(*e));self.D.append(sparse.csr_matrix((vv,(rr.astype(int),cc.astype(int))),shape=(self.N,len(self.x))))
        self.constant=fixed-np.column_stack([D@self.x for D in self.D]);self.coo=[D.tocoo() for D in self.D]

    def values(self,x):return self.constant+np.column_stack([D@x for D in self.D])

    def seed(self,target):
        x=self.x.copy()
        for axis,D in enumerate(self.D):
            ids=np.flatnonzero(self.axis==axis); A=D[:,ids]
            result=lsq_linear(A,target[:,axis]-self.constant[:,axis],bounds=(self.lo[ids],self.hi[ids]),tol=1e-11,lsmr_tol=1e-12,max_iter=100)
            x[ids]=result.x
        return x

    def store(self,x):
        for (bi,j),value in zip(self.refs,x):self.model['blocks'][bi]['values'][j]=float(value)


def _angles(R,h,meta):
    R0,n,b=_head(h,meta); u=b@R0; v=np.einsum('j,njk->nk',b,R)
    alpha=np.arctan2(-np.einsum('j,nj->n',n,np.cross(u,v)),v@u)
    M=R@rotation(n,-alpha)@R0.T
    beta=np.einsum('nj,j->n',Rotation.from_matrix(M).as_rotvec(),b)
    return np.column_stack((alpha,beta))


def _shared_import(model,manifest,nav):
    proof={}; rigs=manifest['rig_order']
    for rid in ('secondary','end'):
        k=rigs.index(rid); rig=model['rigs'][rid]; hb=model['blocks'][rig['head_block']]; ob=model['blocks'][rig['optics_block']]
        c=manifest['original_shared_calibrations'][k]
        if any(c['axial_coefficients']) or c['aspect_fx_over_fy_in_domain']!=1 or c['allow_effective_pp_offset']:raise ValueError('Historical calibration is outside the explicitly supported exact import')
        if len(c['principal_x_fraction_coefficients'])!=1 or len(c['principal_y_fraction_coefficients'])!=1 or len(c['k1_coefficients'])!=1 or any(c['k2_coefficients']) or any(c['tangential_p1_p2']):raise ValueError('Full old optical import needs a wider explicit adapter')
        hv=np.asarray(hb['values']).copy();hv[6:9]=nav['pivot_normalized'][k];hv[9:12]=nav['lever_normalized'][k];hv[12:15]=0.
        ov=np.asarray(ob['values']).copy();ov[:11]=[c['principal_x_fraction_coefficients'][0]*1920,c['principal_y_fraction_coefficients'][0]*1080,0.,0.,0.,c['k1_coefficients'][0],0.,0.,0.,0.,0.]
        for block,values in ((hb,hv),(ob,ov)):
            for j,value in enumerate(values):
                if value!=block['values'][j] and not _free(block,j):raise ValueError('Exact historical import would change an existing fixed coordinate')
            if np.any(values<block['lower']) or np.any(values>block['upper']):raise ValueError('Exact historical calibration exceeds original parameter bounds')
            block['values']=values.tolist()
        proof[rid]=dict(P=hv[6:9].tolist(),d=hv[9:12].tolist(),a=0.,pupil=[0.,0.],PP=ov[:2].tolist(),k1=ov[5],readout_retained_unknown_historical_value=ov[11],original_bounds_unchanged=True)
    return proof


def _witnesses(scene,R,C,logf,calib):
    pixels=np.asarray([(u,v) for v in (240.,420.,600.,780.,960.) for u in (240.,480.,720.,960.,1200.,1440.,1680.)])
    pp=np.array([calib['principal_x_fraction_coefficients'][0]*1920,calib['principal_y_fraction_coefficients'][0]*1080])
    f=np.exp(logf); distorted=(pixels[None,:,:]-pp)/f[:,None,None]
    dist=np.array([calib['k1_coefficients'][0],0.,0.,0.,0.])
    xy=cv2.undistortPoints(distorted.reshape(-1,1,2),np.eye(3),dist).reshape(len(R),len(pixels),2)
    rays=np.einsum('nji,nkj->nki',R,np.concatenate((xy,np.ones((*xy.shape[:2],1))),axis=2))
    lam=-C[:,None,2]/rays[...,2]; ice=C[:,None,:]+lam[...,None]*rays
    valid=(lam>0)&np.isfinite(ice).all(2)&(scene.containment(ice[...,:2])<-.001)
    board=scene.board(np.linspace(0,scene.perimeter,96,endpoint=False),scene.height*.5)
    board=np.broadcast_to(board,(len(R),len(board),3)); X=np.concatenate((ice,board),axis=1)
    Y=np.einsum('nij,nkj->nki',R,X-C[:,None,:]); xx=Y[...,0]/Y[...,2]; yy=Y[...,1]/Y[...,2]; rr=xx*xx+yy*yy; radial=1+dist[0]*rr
    uv=pp+f[:,None,None]*np.stack((xx*radial,yy*radial),axis=2)
    valid=np.concatenate((valid,np.ones(board.shape[:2],bool)),axis=1)&(Y[...,2]>0)&np.isfinite(uv).all(2)&(uv[...,0]>60)&(uv[...,0]<1860)&(uv[...,1]>60)&(uv[...,1]<1020)
    K=12; out=np.empty((len(R),K,3)); target=np.empty((len(R),K,2)); mask=np.zeros((len(R),K),bool)
    for i in range(len(R)):
        ids=np.flatnonzero(valid[i]); take=ids[np.unique(np.linspace(0,len(ids)-1,min(K,len(ids))).astype(int))] if len(ids) else np.array([],int)
        if not len(take):raise ValueError('No finite original navigation witnesses')
        out[i]=X[i,take[0]];target[i]=uv[i,take[0]];out[i,:len(take)]=X[i,take];target[i,:len(take)]=uv[i,take];mask[i,:len(take)]=True
    return out,target,mask


def _fit(motion,h,o,meta,Rtarget,logftarget,X,target,mask,static_indices,x0,mode,max_nfev):
    nh=len(static_indices); initial=np.r_[h[static_indices],x0]
    hb=motion.model['blocks'][motion.head_block]
    lower=np.r_[np.asarray(hb['lower'])[static_indices],motion.lo];upper=np.r_[np.asarray(hb['upper'])[static_indices],motion.hi]
    scale=np.r_[np.asarray(hb['scales'])[static_indices],motion.scale]
    oldf=np.exp(logftarget); rows_per=10 if mode=='rotation' else mask.shape[1]*2
    def residual(q,head):
        if mode=='rotation':
            R,_=_physical(q,head,o,meta)
            return np.c_[((R-Rtarget)*oldf[:,None,None]).reshape(len(q),9),(q[:,2]-logftarget)*oldf]
        uv,*_=_physical(q,head,o,meta,X)
        return np.where(mask[...,None],uv-target,0.).reshape(len(q),-1)
    def unpack(x):
        head=h.copy();head[static_indices]=x[:nh]
        return motion.values(x[nh:]),head
    def fun(x):
        q,head=unpack(x); return residual(q,head).ravel()
    def jac(x):
        q,head=unpack(x); base=residual(q,head); parts=[]
        eps=2e-7
        if nh:
            cols=[]
            for j in static_indices:
                trial=head.copy();trial[j]+=eps;cols.append(((residual(q,trial)-base)/eps).ravel())
            parts.append(sparse.csr_matrix(np.column_stack(cols)))
        r=[];c=[];v=[]
        for axis,D in enumerate(motion.coo):
            trial=q.copy();trial[:,axis]+=eps; derivative=(residual(trial,head)-base)/eps
            r.append((D.row[:,None]*rows_per+np.arange(rows_per)).ravel());c.append(np.broadcast_to(D.col[:,None],(len(D.data),rows_per)).ravel());v.append((D.data[:,None]*derivative[D.row]).ravel())
        parts.append(sparse.csr_matrix((np.concatenate(v),(np.concatenate(r),np.concatenate(c))),shape=(len(q)*rows_per,len(motion.x))))
        return sparse.hstack(parts,format='csr')
    start=time.perf_counter(); before=fun(initial)
    result=least_squares(fun,initial,jac=jac,bounds=(lower,upper),x_scale=scale,tr_solver='lsmr',tr_options=dict(atol=1e-9,btol=1e-9),ftol=1e-10,xtol=1e-10,gtol=1e-5,max_nfev=max_nfev)
    q,head=unpack(result.x)
    report=dict(stage=mode,seconds=time.perf_counter()-start,nfev=result.nfev,njev=result.njev,cost_before=float(before@before/2),cost_after=float(result.fun@result.fun/2),status=int(result.status),message=result.message,parameters=len(initial),residuals=len(before),original_bounds=True)
    print(json.dumps(report),flush=True)
    return head,result.x[nh:],report


def _preserved_payload(model,changed_blocks):
    return dict(blocks=[(i,b) for i,b in enumerate(model['blocks']) if i not in changed_blocks],materials=model['materials'],main=model['rigs']['main'],S01=model['shots']['s01'],linear_control_manifolds=model.get('linear_control_manifolds'),source_clock=model['source_clock'],coordinate_gauge=model['coordinate_gauge'])


def initialize(model_file,navigation_manifest,roles_file,outdir,*,max_nfev=100):
    start=time.perf_counter(); out=Path(outdir)
    if out.exists():raise ValueError('An initializer generation must be new and immutable')
    model=read_json(model_file);manifest=read_json(navigation_manifest);navfile=Path(manifest['output']['file'])
    if file_sha256(navfile)!=manifest['output']['sha256']:raise ValueError('Native navigation archive hash mismatch')
    nav=np.load(navfile,allow_pickle=False);pts=nav['native_pts'];numbers=nav['shot_number'];kind=nav['navigation_kind'];R=nav['navigation_R_world_to_camera'];C=nav['navigation_C_normalized'];logf=nav['navigation_log_f_native_pixels']
    if model['source_clock']['time_base']!=[1,12800] or model['source_clock']['frame_step_pts']!=256 or model['source_clock']['source_video_sha256']!=manifest['official_source']['sha256']:raise ValueError('Official source clock/identity mismatch')
    guard=FrameRoleGuard(roles_file);training=np.zeros(len(pts),bool)
    for i,(p,sn) in enumerate(zip(pts,numbers)):
        p=int(p);sid=f's{int(sn):02d}'
        if p not in guard.roles or guard.roles[p]['shot_id']!=sid:raise ValueError('Missing native frame role or cross-cut navigation')
        if guard.roles[p]['role'] not in DENY_TRAIN:guard.assert_training_pixels([p],shot_id=sid);training[i]=True
    finite=np.isfinite(R).all((1,2))&np.isfinite(C).all(1)&np.isfinite(logf)
    fitmask=training&finite
    changed={model['rigs'][r][key] for r in ('secondary','end') for key in ('head_block','optics_block')}
    for sid,s in model['shots'].items():
        if sid!='s01':changed.update(s['control_blocks']);changed.add(s['reference_angles_block'])
    frozen=object_sha256(_preserved_payload(model,changed))
    original_structure=object_sha256(dict(blocks=[{k:v for k,v in b.items() if k!='values'} for b in model['blocks']],shots=model['shots'],rigs=model['rigs']))
    shared=_shared_import(model,manifest,nav)
    scene=NativeScene(as_event_view(model,'s01'));reports={};before_all=np.full(len(pts),np.nan);after_all=before_all.copy();angle_all=before_all.copy();bad_all=np.zeros(len(pts),int);fullrows=[]
    out.mkdir(parents=True)
    for rid in ('secondary','end','main'):
        sids=[sid for sid,s in model['shots'].items() if sid!='s01' and s['rig_id']==rid]
        select=np.isin(numbers,[int(s[1:]) for s in sids])&finite;fitselect=select&fitmask;idx=np.flatnonzero(fitselect);allidx=np.flatnonzero(select)
        rig=model['rigs'][rid];hb=model['blocks'][rig['head_block']];h=np.asarray(hb['values']).copy();o=np.asarray(model['blocks'][rig['optics_block']]['values']);meta=rig['camera']
        if not len(idx):raise ValueError('No role-allowed finite navigation for rig '+rid)
        X,uv,wmask=_witnesses(scene,R[allidx],C[allidx],logf[allidx],manifest['original_shared_calibrations'][manifest['rig_order'].index(rid)])
        loc=np.searchsorted(allidx,idx);motion=Motion(model,sids,numbers[idx],pts[idx]);motion.head_block=rig['head_block']
        fullmotion=Motion(model,sids,numbers[allidx],pts[allidx]);qbefore=fullmotion.values(fullmotion.x)
        uvbefore,*_=_physical(qbefore,h,o,meta,X);before_all[allidx]=np.sqrt(np.sum(np.where(wmask[...,None],(uvbefore-uv)**2,0.),axis=(1,2))/wmask.sum(1))
        if rid!='main':
            ref=rig['reference_shot'];refidx=idx[numbers[idx]==int(ref[1:])][0];want=R[refidx]
            value=Rotation.from_matrix(want@np.asarray(meta['R0_seed']).reshape(3,3).T).as_rotvec()
            if all(_free(hb,j) and hb['lower'][j]<=value[j]<=hb['upper'][j] for j in range(3)):h[:3]=value
        targetq=np.c_[_angles(R[idx],h,meta),logf[idx]]
        for sn in np.unique(numbers[idx]):
            m=numbers[idx]==sn;targetq[m,:2]=np.unwrap(targetq[m,:2],axis=0)
        x0=motion.seed(targetq);static=np.array([j for j in range(6) if rid!='main' and _free(hb,j)],int)
        h,x1,p1=_fit(motion,h,o,meta,R[idx],logf[idx],X[loc],uv[loc],wmask[loc],static,x0,'rotation',max_nfev)
        h,x2,p2=_fit(motion,h,o,meta,R[idx],logf[idx],X[loc],uv[loc],wmask[loc],static,x1,'pixels',max_nfev)
        hb['values']=h.tolist();motion.store(x2)
        fullmotion=Motion(model,sids,numbers[allidx],pts[allidx]);qafter=fullmotion.values(fullmotion.x)
        pred,depth,radius,Rnow,Cnow=_physical(qafter,h,o,meta,X)
        after_all[allidx]=np.sqrt(np.sum(np.where(wmask[...,None],(pred-uv)**2,0.),axis=(1,2))/wmask.sum(1));angle_all[allidx]=Rotation.from_matrix(Rnow@np.transpose(R[allidx],(0,2,1))).magnitude();bad_all[allidx]=np.sum(wmask&((depth<=0)|(radius>meta['radius_max'])),axis=1)
        verification=0.
        for sid in sids:
            ss=model['shots'][sid];m=numbers[allidx]==int(sid[1:]);cam=camera_from_event(as_event_view(model,sid));ref=np.asarray(model['blocks'][ss['reference_angles_block']]['values']);cam.R0=rotation(cam.b,ref[1])@cam.R0@rotation(cam.n,ref[0]);t=(pts[allidx[m]]-ss['time_origin_pts'])/12800.
            native=cam.at(X[m],t[:,None]);verification=max(verification,float(np.max(np.abs(native['uv']-pred[m]))))
        if verification>2e-6:raise ValueError('Native production replay disagrees with initializer evaluator')
        reports[rid]=dict(shots=sids,fit_frames=len(idx),all_finite_frames=len(allidx),stages=[p1,p2],independent_native_forward_max_abs_px=verification,current_center_vs_old_max_L=float(np.max(np.linalg.norm(Cnow-C[allidx],axis=1))))
        write_json(out/f'{rid}_progress.json',reports[rid])
    if frozen!=object_sha256(_preserved_payload(model,changed)):raise ValueError('S01/shared scene/main physical state changed during bootstrap')
    if original_structure!=object_sha256(dict(blocks=[{k:v for k,v in b.items() if k!='values'} for b in model['blocks']],shots=model['shots'],rigs=model['rigs'])):raise ValueError('Original allocation, bounds, free flags or static chart changed')
    for b in model['blocks']:
        v=np.asarray(b['values'])
        if np.any(v<b['lower']) or np.any(v>b['upper']) or not np.isfinite(v).all():raise ValueError('Initializer produced an invalid original parameter domain')
    for sid in sorted(model['shots']):
        if sid=='s01':continue
        m=numbers==int(sid[1:]);finite_sid=m&finite;fit_sid=m&fitmask
        def stats(values,mask):
            a=values[mask];return dict(count=len(a),rms=float(np.sqrt(np.mean(a*a))) if len(a) else None,max=float(np.max(a)) if len(a) else None,p95=float(np.percentile(a,95)) if len(a) else None,max_native_pts=int(pts[mask][np.argmax(a)]) if len(a) else None)
        fullrows.append(dict(shot=sid,total=int(m.sum()),finite=int(finite_sid.sum()),conflict_no_value=int((m&~finite).sum()),fit=int(fit_sid.sum()),protected=int((m&~training).sum()),navigation_kind_counts=dict(Counter(str(int(x)) for x in kind[m])),native_witness_before_px=stats(before_all,finite_sid),native_witness_after_px=stats(after_all,finite_sid),fit_witness_after_px=stats(after_all,fit_sid),protected_witness_after_px=stats(after_all,finite_sid&~training),rotation_error_rad=stats(angle_all,finite_sid),invalid_current_witnesses=int(bad_all[m].sum())))
    model['native_baseline_initializer']=dict(schema='rinkfit.bounded_dense_native_baseline_initializer.v1',navigation_manifest=dict(path=str(Path(navigation_manifest).resolve()),sha256=file_sha256(navigation_manifest)),input_model=dict(path=str(Path(model_file).resolve()),sha256=file_sha256(model_file)),roles=dict(path=str(Path(roles_file).resolve()),sha256=file_sha256(roles_file)),historical_values_are_numerical_navigation_only=True,source_likelihood_modified=False,UV_W_Query_admission_modified=False,static_family_and_original_motion_allocation_unchanged=True,S01_shared_scene_main_head_optics_preserved_sha256=frozen,exact_calibration_import=shared,RS_support_unchanged=True,requires_native_auxiliary_reinitialization=True,requires_original_native_whole_shot_visual_validation=True,accepted=False)
    write_json(out/'model.json',model)
    np.savez(out/'native_transfer_metrics.npz',native_pts=pts,shot_number=numbers,navigation_kind=kind,fit_allowed=fitmask,protected=~training,before_witness_rms_px=before_all,after_witness_rms_px=after_all,rotation_error_rad=angle_all,invalid_witness_count=bad_all)
    report=dict(schema='rinkfit.bounded_dense_native_baseline_report.v1',seconds=time.perf_counter()-start,model=dict(path=str((out/'model.json').resolve()),sha256=file_sha256(out/'model.json')),roles_protected=int((~training).sum()),finite_navigation=int(finite.sum()),navigation_fit_frames=int(fitmask.sum()),all_native_denominator=len(pts),shots=fullrows,rigs=reports,S01_exact_coupled_payload_preserved=True,dense_navigation_transfer_is_approximate=True,measurements_created=0,original_structure_preserved_sha256=original_structure,production_fit_started=False,source_accuracy_accepted=False)
    write_json(out/'report.json',report);print(json.dumps(report),flush=True);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--navigation-manifest',required=True);p.add_argument('--roles',required=True);p.add_argument('--out',required=True);p.add_argument('--max-nfev',type=int,default=100)
    a=p.parse_args();initialize(a.model,a.navigation_manifest,a.roles,a.out,max_nfev=a.max_nfev)


if __name__=='__main__':main()
