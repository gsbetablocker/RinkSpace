"""Refit only the reviewed S09 pose and depth controls against original evidence."""
from __future__ import annotations
from copy import copy
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import bsr_matrix, csr_matrix, diags, vstack
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve, greville_times
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference, chain_exposure, curve_from_packet
from calibration.program_tracking.body_tracking.compact_body_derivatives import CompactBodyLinearization
from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from calibration.program_tracking.body_tracking.fullbody_fit import body_joint_assignment, right_jacobian, skew
from calibration.program_tracking.body_tracking.programme_com import com_local_jacobians
from calibration.program_tracking.body_tracking.programme_joint_limits import JOINTS, ProgrammeJointLimits
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.rebuild_prepare import digest, load
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState, shoe_curve_from_packet
from calibration.program_tracking.body_tracking.source_contact_image import NativeSupportContactFactors
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference

ALL_LEGS=(1,2,4,5,7,8,10,11)
CASES={
    'step':dict(controls=(150.40,151.54),observed=(150.55,151.47),joints=ALL_LEGS,xy=.14,z=.10,pose=1.33),
    'spin_entry':dict(controls=(156.78,157.72),observed=(156.93,157.67),joints=ALL_LEGS,xy=.11,z=.09,pose=1.05),
    'depth':dict(controls=(159.25,164.27),observed=(159.74,163.95),joints=(1,4,7,10),xy=1.40,z=.32,pose=.55),
}

def log(**value):
    print(json.dumps(value,ensure_ascii=False),flush=True)


def robust_pairs(r,j,threshold):
    h=np.sqrt(1+np.sum(r*r,axis=1)/(threshold*threshold))
    w=np.sqrt(2/(1+h))
    dr=w[:,None,None]*np.eye(2)-(w/(2*threshold*threshold*h*(1+h)))[:,None,None]*r[:,:,None]*r[:,None,:]
    D=bsr_matrix((dr,np.arange(len(r)),np.arange(len(r)+1)),shape=(2*len(r),2*len(r)))
    return (r*w[:,None]).ravel(),D@j


def fresh_edges(curve,active,joints):
    edges=np.unique(np.r_[active-1,active])
    relative=curve.rotation_controls[edges].swapaxes(-1,-2)@curve.rotation_controls[edges+1]
    q=Rotation.from_matrix(relative.reshape(-1,3,3)).as_quat().reshape(len(edges),22,4)
    old=curve.edge_reference_quaternion.copy()
    sign=np.where(np.sum(q*old[edges],axis=-1,keepdims=True)<0,-1.,1.)
    old[np.ix_(edges,joints)]=(q*sign)[:,joints]
    refreshed=AbsoluteMotionCurve(curve.knots,curve.position_controls,curve.rotation_controls,old)
    np.testing.assert_array_equal(curve._check_log_chart(),refreshed._check_log_chart())
    return refreshed


def save_curve(saved,actual,active):
    old=curve_from_packet(saved,seed=True)
    n=len(old.position_controls)
    x=saved['x'].copy()
    q=x[17:17+n*69].reshape(n,69)
    q[active,:3]=actual.position_controls[active]-old.position_controls[active]
    v=Rotation.from_matrix((old.rotation_controls[active].swapaxes(-1,-2)@actual.rotation_controls[active]).reshape(-1,3,3)).as_rotvec()
    r=2*np.pi*v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-14)
    branch=np.stack([v,v+r,v-r])
    best=np.argmin(np.linalg.norm(branch-q[active,3:].reshape(-1,3),axis=-1),axis=0)
    q[active,3:]=branch[best,np.arange(len(v))].reshape(len(active),66)
    rebuild=old.retracted(q[:,:3],q[:,3:].reshape(n,22,3))
    np.testing.assert_allclose(rebuild.rotation_controls,actual.rotation_controls,atol=1.1e-11,rtol=0)
    np.testing.assert_allclose(rebuild.position_controls,actual.position_controls,atol=3e-13,rtol=0)
    return dict(saved,x=x,absolute_position_controls=actual.position_controls.copy(),
                absolute_rotation_controls=actual.rotation_controls.copy(),
                absolute_rotation_edge_reference_quaternion=actual.edge_reference_quaternion.copy())


def fit_case(name,settings,saved,shared,camera,doc,prepared,observations,stored,marks,ice,axis):
    started=time.perf_counter()
    base=curve_from_packet(saved)
    pts=shared.p['pts']
    seconds=pts/12800.
    ctime=greville_times(base.knots)+seconds[0]
    lo,hi=settings['controls']
    active=np.flatnonzero((ctime>=lo)&(ctime<=hi))
    limbs=np.array(settings['joints'],int)
    base=fresh_edges(base,active,limbs)
    intervals=np.clip(np.searchsorted(base.knots,shared.times,side='right')-4,0,len(ctime)-4)
    use=np.isin(intervals[:,None]+np.arange(4),active).any(1)
    native=np.flatnonzero(use)
    n=len(native)
    sec=seconds[native]
    ts=shared.times[native]
    tau=camera.readout_times(native)
    assert len(set(shared.p['shot_ids'][native]))==1 and np.all(tau==tau[0])
    nodes=(ts[:,None]+tau[:,None]*np.linspace(-.5,.5,5)).ravel()
    first=int(intervals[native].min()-4)
    last=int(intervals[native].max()+7)
    knots=base.knots[first:last+5].copy()
    knots[:4]=knots[3];knots[-4:]=knots[-4]
    small=AbsoluteMotionCurve(knots,base.position_controls[first:last+1],
                    base.rotation_controls[first:last+1],base.edge_reference_quaternion[first:last])
    global_ref=absolute_reference(shared,base,root=prepared)
    ref=absolute_reference(shared,small,root=prepared)
    for rr in (ref,global_ref):
        rr.relative_skate_curve=shoe_curve_from_packet(saved)
        rr.binding=copy(shared.binding)
        rr.binding.reference=rr
        rr.binding.x=saved['x'][:17].copy()
        rr.binding._parameter_key=rr.binding._packet=None
    ratio=ref.binding.ratio
    state=SequenceState(ref,(shared.times[0],shared.times[-1]))
    whole=SequenceState(global_ref,(shared.times[0],shared.times[-1]))
    node_ref=ref.at(nodes);mid_ref=ref.at(ts)
    for k,v in global_ref.at(nodes).items():
        np.testing.assert_allclose(node_ref[k],v,rtol=0,atol=2e-12)

    lcols=(6+3*(limbs-1)[:,None]+np.arange(3)).ravel()
    columns=np.r_[np.arange(3),lcols]
    width=len(columns)
    cols=((active-first)[:,None]*69+columns).ravel()
    global_cols=(active[:,None]*69+columns).ravel()
    vreg=np.r_[np.full(3,2.5*ratio if name=='depth' else 5.*ratio),
               np.full(len(lcols),2.1)]
    prior=diags(np.tile(vreg,len(active)),format='csr')
    nearby=np.flatnonzero(np.diff(active)==1)
    ii=(nearby[:,None]*width+np.arange(width)).ravel()
    gg=np.tile(np.r_[np.full(3,1.3*ratio),np.full(len(lcols),2.)],len(nearby))
    diff=csr_matrix((np.r_[-gg,gg],(np.tile(np.arange(len(ii)),2),np.r_[ii,ii+width])),
                    shape=(len(ii),len(cols)))
    reg=vstack([prior,diff],format='csr')

    assignment=body_joint_assignment(observations)[native]
    f,slot=np.nonzero(observations['use'][native])
    j=assignment[f,slot]
    raw=observations['uv'][native][f,slot]
    std=np.maximum(observations['sigma_px'][native][f,slot],3.)
    raw_w=np.where(j>=11,.62,.29)/std
    raw_w[j<5]*=.28
    inside=(sec>=settings['observed'][0])&(sec<=settings['observed'][1])
    if name!='depth':
        raw_w[inside[f]&(j>=13)]*=.045
    else:
        raw_w[(j==13)|(j==15)]*=.47
    baseline_uv=stored['COCO_uv'][native]
    refs,upper=np.nonzero(np.ones((n,17),bool)&(np.arange(17)[None,:]>=5)&(np.arange(17)[None,:]<=12))
    keep=upper!=7
    refs,upper=refs[keep],upper[keep]
    baseline_pixels=baseline_uv[refs,upper]
    baseline_w=np.full(len(upper),.036 if name=='depth' else .058)
    # Outside each reviewed core the fitted image is a finite continuity prior.
    outside,nk=np.nonzero((~inside[:,None])&(np.arange(17)[None,:]>=13))
    old_leg=baseline_uv[outside,nk]
    hold_w=np.full(len(outside),.055 if name!='depth' else .032)

    manual=[r for r in marks['manual_leg_landmarks'] if r['group']==name and r['role']=='fit']
    manual_check=[r for r in marks['manual_leg_landmarks'] if r['group']==name and r['role']=='check']
    mk=np.searchsorted(pts[native],[r['pts'] for r in manual]).astype(int)
    mj=np.array([r['joint'] for r in manual],int)
    mu=np.array([r['uv'] for r in manual],float).reshape(-1,2)
    entry_detail=settings.get('entry_detail',False)
    mw=(7.8 if entry_detail else 6.4)/np.array([r['radius_px'] for r in manual],float)
    np.testing.assert_array_equal(pts[native][mk],[r['pts'] for r in manual])
    mc=np.searchsorted(pts[native],[r['pts'] for r in manual_check]).astype(int)
    mjc=np.array([r['joint'] for r in manual_check],int)
    muc=np.array([r['uv'] for r in manual_check],float).reshape(-1,2)

    factor=None
    entry_factor=None
    if entry_detail and marks.get('entry_runner_bands'):
        bands=[dict(b,objective_weight=1.0) for b in marks['entry_runner_bands'] if b['role']=='fit']
        entry_factor=NativeSupportContactFactors(SimpleNamespace(reference=ref,binding=ref.binding,camera=camera,
            body=SimpleNamespace(state=state,tau=tau[0]),size=state.size+17),native_bands=bands)
    if name=='depth':
        bands=[dict(b,objective_weight=1.0) for b in marks['left_runner_ice_interface_bands'] if b['role']=='fit']
        position=[r for r in ice if r['role']=='fit']
        np.testing.assert_array_equal([b['pts'] for b in bands],[r['pts'] for r in position])
        # The plane rows only check provenance; their pixels already enter as native bands.
        factor=NativeSupportContactFactors(SimpleNamespace(reference=ref,binding=ref.binding,camera=camera,
            body=SimpleNamespace(state=state,tau=tau[0]),size=state.size+17),native_bands=bands)
        mass=ContinuousMassGeometry(ref)
        s=np.arange(6,n-6,2)
        A=csr_matrix((np.tile([1.,-2.,1.],len(s)),(np.repeat(np.arange(len(s)),3),
                           np.c_[s-6,s,s+6].ravel())),shape=(len(s),n))*(.22/.12**2)
    policy=json.loads(Path(doc['programme_structure']['policy']).read_text('utf8'))
    limits=ProgrammeJointLimits(SimpleNamespace(reference=global_ref,body=SimpleNamespace(state=whole)),
                                policy,np.zeros(whole.size),anchor=load(doc['programme_structure']['chart_anchor']))
    limit_ids=np.flatnonzero(np.isin(JOINTS,limbs))

    cache=None
    last_cost=np.inf
    evaluations=0
    accepted=0
    def evaluate(x):
        nonlocal cache,last_cost,evaluations,accepted
        if cache is not None and np.array_equal(cache[0],x):
            return cache[1:]
        dx=np.zeros(state.size)
        dx[cols]=x
        dn=state.increments(dx,nodes)
        dm=state.increments(dx,ts)
        cp=CompactBodyLinearization(ref.body,node_ref['local_R'],dn[:,6:].reshape(-1,21,3),
                                    hand_local_R=node_ref.get('hand_local_R'))
        Q=node_ref['Q']@Rotation.from_rotvec(dn[:,3:6]).as_matrix()
        Y=np.einsum('nij,nkj->nki',Q,cp.coco)+(node_ref['P']+dn[:,:3])[:,None]
        pr=moving_projection(camera,Y.reshape(n,5,17,3),native,jacobian=True)
        uv=pr['uv'];pixels=pr['pixel_jac_nodes']
        Jlocal=np.zeros((n,17,2,5,69))
        Jlocal[..., :3]=pixels
        for c in lcols:
            slope=np.einsum('nij,nkj->nki',Q,cp.derivative(c,kinematics=False)['coco']).reshape(n,5,17,3)
            Jlocal[...,c]=np.einsum('nkjhc,nhkc->nkjh',pixels,slope)
        Juv=chain_exposure(Jlocal.reshape(n,34,5,69),dn)[:,cols]
        del Jlocal

        parts={}
        values=[];blocks=[]
        def add(label,r,J):
            r=np.asarray(r).ravel()
            parts[label]=float(r@r/2)
            values.append(r)
            blocks.append(J)
        def image(label,f,j,target,weight,robust=None):
            error=(uv[f,j]-target)*weight[:,None]
            row=(2*(17*f+j)[:,None]+np.arange(2)).ravel()
            J=diags(np.repeat(weight,2))@Juv[row]
            if robust is not None:
                error,J=robust_pairs(error,J,robust)
            add(label,error,J)

        image('admitted_original_pixels',f,j,raw,raw_w,3.1)
        image('upper_body_continuity_pixels',refs,upper,baseline_pixels,baseline_w,2.5)
        image('outside_reviewed_time_leg_prior',outside,nk,old_leg,hold_w,4.)
        if len(mk):
            image('manually_reviewed_visible_leg_pixels',mk,mj,mu,mw,4.6)

        if entry_factor is not None:
            e=entry_factor.evaluate(dx,True)
            add('entry_support_skate_image_bands',5.5*e['residual'],5.5*e['jac'][:,17+cols])
        if factor is not None:
            F=factor.evaluate(dx,True)
            add('native_visible_runner_bands',3.3*F['residual'],3.3*F['jac'][:,17+cols])
            C,Jc,_=com_local_jacobians(mass,mid_ref,dm)
            accel=A@(C[:,:2]@axis)
            mid_j=dm.chain(np.einsum('i,nik->nk',axis,Jc[:,:2])[:,None])[:,cols]
            add('finite_COM_depth_curvature',accel,A@mid_j)

        V=limits.values(dm,native)[0][:,limit_ids]
        penalty=V
        deriv=np.zeros((n,3*len(limit_ids),69))
        eps=1e-6
        for k,joint_ix in enumerate(limit_ids):
            idx=6+3*(JOINTS[joint_ix]-1)
            for a in range(3):
                u,v=np.asarray(dm).copy(),np.asarray(dm).copy()
                u[:,idx+a]+=eps;v[:,idx+a]-=eps
                d=(limits.values(u,native)[0][:,joint_ix]-limits.values(v,native)[0][:,joint_ix])/(2*eps)
                deriv[:,3*k:3*k+3,idx+a]=d
        add('finite_anatomical_domain',70*penalty,70*dm.chain(deriv)[:,cols])
        add('local_control_prior',reg@x,reg)
        residual=np.concatenate(values)
        J=vstack(blocks,format='csr')
        cost=float(residual@residual/2)
        evaluations+=1
        if cost<last_cost:
            last_cost=cost
            accepted+=1
        cache=(x.copy(),residual,J,uv,parts,V)
        if evaluations==1 or evaluations%4==0:
            log(event='evaluation',case=name,evaluation=evaluations,improvements=accepted,cost=cost,
                block_count=len(parts),parts=parts,rows=len(residual),columns=J.shape[1],
                cpu_elapsed_s=round(time.perf_counter()-started,3),source_frames=n)
        return cache[1:]

    x0=np.zeros(len(cols))
    baseline=evaluate(x0)
    rng=np.random.default_rng(1109)
    direction=rng.normal(size=len(cols))
    direction/=np.linalg.norm(direction)
    h=1e-6
    derivative=(evaluate(h*direction)[0]-evaluate(-h*direction)[0])/(2*h)
    analytic=baseline[1]@direction
    err=float(np.linalg.norm(analytic-derivative)/max(np.linalg.norm(derivative),1e-8))
    log(event='checked_derivative',case=name,relative_error=err,reference='same native frames and factors')
    if not err<.008:
        raise ValueError(f'Analytic derivative mismatch for {name}: {err}')
    box=np.r_[np.array([settings['xy'],settings['xy'],settings['z']])/ratio,
              np.full(len(lcols),settings['pose'])]
    fit=least_squares(lambda x:evaluate(x)[0],x0,jac=lambda x:evaluate(x)[1],
                      bounds=(-np.tile(box,len(active)),np.tile(box,len(active))),
                      tr_solver='lsmr',method='trf',x_scale='jac',ftol=1.0e-4,xtol=1.0e-4,gtol=3e-4)
    x=np.zeros(whole.size)
    x[global_cols]=fit.x
    actual=whole.current_curve(x)
    fixed=np.setdiff1d(np.arange(len(ctime)),active)
    np.testing.assert_array_equal(actual.position_controls[fixed],base.position_controls[fixed])
    np.testing.assert_array_equal(actual.rotation_controls[fixed],base.rotation_controls[fixed])
    np.testing.assert_array_equal(actual.rotation_controls[:,0],base.rotation_controls[:,0])
    now=evaluate(fit.x)
    def error(frame,j,target,uv):
        if not len(frame):
            return None
        e=np.linalg.norm(uv[frame,j]-target,axis=1)
        return dict(median_px=float(np.median(e)),p95_px=float(np.percentile(e,95)),max_px=float(e.max()))
    report=dict(case=name,active_controls=active.tolist(),source_exposures=n,
                before_parts=baseline[3],after_parts=now[3],derivative_error=err,
                fit_manual_before=error(mk,mj,mu,baseline[2]),fit_manual_after=error(mk,mj,mu,now[2]),
                reserved_manual_before=error(mc,mjc,muc,baseline[2]),reserved_manual_after=error(mc,mjc,muc,now[2]),
                before_domain_peak=baseline[4].max(axis=(0,2)).tolist(),after_domain_peak=now[4].max(axis=(0,2)).tolist(),
                native_first_last_seconds=sec[[0,-1]].tolist(),nfev=fit.nfev,solver_message=fit.message,
                seconds=time.perf_counter()-started)
    log(event='case_saved',**report)
    return save_curve(saved,actual,active),report


def main():
    parser=argparse.ArgumentParser()
    for key in ('project','prepared','source','parameter','result','annotations','ice','before-audit','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--anatomy-metrics',type=Path,help='Deprecated archived input; no programme peak can grant a local joint allowance')
    parser.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
    parser.add_argument('--entry-detail',action='store_true',
                        help='Use the second-pass bounds and support-runner evidence for spin_entry')
    args=parser.parse_args()
    if args.entry_detail and args.cases != ['spin_entry']:
        parser.error('--entry-detail requires --cases spin_entry')
    obs=load(args.prepared/'observations.npz')
    shared,camera,info,doc,_=load_reference(args.project,args.prepared,args.source)
    marks=json.loads(args.annotations.read_text('utf8'))
    ice=json.loads(args.ice.read_text('utf8'))
    axis=load(args.before_audit)['camera_planar_axes'][:,1]
    start=load(args.parameter)
    p=start
    with np.load(args.result,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['pts'],shared.p['pts'])
        stored=dict(COCO_uv=z['COCO_uv'])
    result=[]
    for name in args.cases:
        settings=CASES[name]
        if args.entry_detail:
            settings=dict(settings,xy=.06,z=.06,pose=.66,entry_detail=True)
        p,q=fit_case(name,settings,p,shared,camera,doc,args.prepared,obs,stored,marks,ice,axis)
        result.append(q)
        interim=args.output.with_name(args.output.stem+'_'+name+'_checkpoint.npz')
        with interim.open('xb') as f:np.savez_compressed(f,**p)
    for key in ('relative_skate_rotation_controls','prepared_input_sha256','camera_binding_json',
                'sequence_contact_modes','sequence_contact_status'):
        np.testing.assert_array_equal(start[key],p[key])
    with args.output.open('xb') as f:np.savez_compressed(f,**p)
    receipt=dict(parent_sha256=digest(args.parameter),candidate_sha256=digest(args.output),
                 annotations_sha256=digest(args.annotations),ice_footprints_sha256=digest(args.ice),
                 camera_sha256=info['state_sha256'],cases=result,accuracy_accepted=False,
                 acceptance='proposal_only', full_objective_replayed=False,
                 scope='Local proposal on the exact whole curve; requires separate full replay and structure review.')
    with args.output.with_suffix('.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
    log(event='complete',output=str(args.output),sha256=receipt['candidate_sha256'])


if __name__=='__main__':
    main()
