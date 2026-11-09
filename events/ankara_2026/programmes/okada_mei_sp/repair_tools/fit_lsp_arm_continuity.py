"""Local controls of one complete saved programme curve, exact native exposure fit."""
from pathlib import Path
from types import SimpleNamespace
from copy import copy
import argparse,json,time
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix,diags,vstack
from scipy.spatial.transform import Rotation
from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve,RotationChartDomainError,greville_times
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference,chain_exposure,curve_from_packet
from calibration.program_tracking.body_tracking.compact_body_derivatives import CompactBodyLinearization
from calibration.program_tracking.body_tracking.fullbody_fit import body_joint_assignment,right_jacobian,skew
from calibration.program_tracking.body_tracking.programme_joint_limits import JOINTS,NAMES,ProgrammeJointLimits
from calibration.program_tracking.body_tracking.programme_surface import CollisionSurface,ProgrammeSurfaceFactor
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.rebuild_prepare import digest,load
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState,shoe_curve_from_packet
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from calibration.program_tracking.body_tracking.skate_template import profile_height
from calibration.program_tracking.body_tracking.spin_grasp_factors import palm_local
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference

parser=argparse.ArgumentParser()
for s in ('project','prepared','source','seed','parent','result','output'):
    parser.add_argument(s,type=Path)
parser.add_argument('--source-landmarks',type=Path,required=True)
parser.add_argument('--source-runner',type=Path,required=True)
parser.add_argument('--surface-scan',type=Path,required=True)
args=parser.parse_args()
out=args.output
started=time.perf_counter()
seed=load(args.seed);parent=load(args.parent)
shared,camera,info,doc,_=load_reference(args.project,args.prepared,args.source)
base=curve_from_packet(seed)
origin=shared.p['pts'][0]/12800.
meta=json.loads((args.seed.parent/'source_turn_seed.json').read_text())
active=np.asarray(meta['changed_controls'],int)
times=shared.times
ratio=shared.binding.ratio
old=np.load(args.result,allow_pickle=False)
np.testing.assert_array_equal(old['pts'],shared.p['pts'])
support=np.isin(np.clip(np.searchsorted(base.knots,times,side='right')-4,0,len(base.position_controls)-4)[:,None]+np.arange(4),active).any(axis=1)
native=np.flatnonzero(support)
sec=shared.p['pts'][native]/12800
mid_times=times[native]
readout=camera.readout_times(native)
nodes=(mid_times[:,None]+readout[:,None]*np.linspace(-.5,.5,5)).ravel()
starts=np.clip(np.searchsorted(base.knots,np.r_[nodes,mid_times],side='right')-4,0,len(base.position_controls)-4)
first=int(starts.min()-3);last=int(starts.max()+6)
knots=base.knots[first:last+5].copy();knots[:4]=knots[3];knots[-4:]=knots[-4]
compact_curve=AbsoluteMotionCurve(knots,base.position_controls[first:last+1],
    base.rotation_controls[first:last+1],base.edge_reference_quaternion[first:last])
whole_ref=absolute_reference(shared,base,root=args.prepared)
ref=absolute_reference(shared,compact_curve,root=args.prepared)
for rr in (whole_ref,ref):
    rr.relative_skate_curve=shoe_curve_from_packet(seed)
    rr.binding=copy(shared.binding)
    rr.binding.reference=rr
    rr.binding.x=seed['x'][:17].copy()
    rr.binding._packet=rr.binding._parameter_key=None
whole=SequenceState(whole_ref,(times[0],times[-1]));state=SequenceState(ref,whole.seconds)
node_ref=ref.at(nodes);mid_ref=ref.at(mid_times)
for t in (nodes,mid_times):
    actual=whole_ref.at(t);reduced=ref.at(t)
    for k in ('P','Q','local_R','skate_relative_R'):np.testing.assert_allclose(actual[k],reduced[k],rtol=0,atol=2e-12)
n=len(native)
columns=np.arange(69)
cols=((active-first)[:,None]*69+columns).ravel()
global_cols=(active[:,None]*69+columns).ravel()
weight_control=np.full(69,4.8)
weight_control[:3]=4*ratio
weight_control[3:6]=5.
weight_control[6+3*(np.array([7,8,10,11,20,21])-1)[:,None]+np.arange(3)]=2.8
regularizer=diags(np.tile(weight_control,len(active)),format='csr')
timepairs=np.flatnonzero(np.diff(native)==1)
active_clock=greville_times(base.knots)[active]+origin
same_window=np.flatnonzero(np.diff(active)==1)
i0=np.ravel_multi_index((same_window[:,None]+np.zeros((len(same_window),69),int),np.arange(69)[None,:]),(len(active),69)).ravel()
j0=i0+69
drscale=np.tile(np.r_[np.full(3,1.4*ratio),np.full(3,1.3),np.full(63,2.2)],(len(same_window),1))
catch_edges=(active_clock[same_window]>=162.26)&(active_clock[same_window]<=163.20)
held_columns=(6+3*(np.array([14,17,19,21])-1)[:,None]+np.arange(3)).ravel()
drscale[np.ix_(catch_edges,held_columns)]=11.
drscale=drscale.ravel()
dreg=csr_matrix((np.r_[-drscale,drscale],(np.tile(np.arange(len(i0)),2),np.r_[i0,j0])),shape=(len(i0),len(cols)))
regularizer=vstack([regularizer,dreg],format='csr')
obs=load(args.prepared/'observations.npz')
assign=body_joint_assignment(obs)[native]
f,slots=np.nonzero(obs['use'][native])
j=assign[f,slots]
observed=obs['uv'][native][f,slots]
sigma=obs['sigma_px'][native][f,slots]
image_weights=np.full(17,.68)
image_weights[[5,6,11,12]]=1.
image_weights[[13,14,15,16]]=1.05
image_weights[:5]=.28
weight=.26*image_weights[j]/np.maximum(sigma,2.)
manual=json.loads(args.source_landmarks.read_text('utf8'))['landmarks']
manf=np.searchsorted(shared.p['pts'][native],[p['pts'] for p in manual])
np.testing.assert_array_equal(shared.p['pts'][native][manf],[p['pts'] for p in manual])
mani=np.array([p['joint'] for p in manual])
manuv=np.asarray([p['uv'] for p in manual])
manweight=2.8/np.asarray([p['radius_px'] for p in manual])
marked=set(zip(manf,mani))
weight*=np.array([.4 if (row,joint) in marked else 1. for row,joint in zip(f,j)])
nobs=len(f)
outrows=(2*(17*f+slots)[:,None]+np.arange(2)).ravel()
old_uv=old['COCO_uv'][native]
old_pixels=np.linalg.norm(old_uv[f,j]-observed,axis=-1)
old_left=old['skate_reference_world_mid'][native,0]/ratio
old_R=old['skate_R_world_mid'][native,0];old_O=old['skate_origin_world_mid'][native,0]/ratio
material=np.mean(np.einsum('nji,nj->ni',old_R,old_left-old_O),axis=0)
np.testing.assert_allclose(old_O+np.einsum('nij,j->ni',old_R,material),old_left,atol=5e-10)
params=ref.binding.packet()['parameters']
runner_x=np.linspace(-params.runner_rear_extension_m,params.sole_length_m-params.runner_front_inset_m,35)
steel=np.c_[runner_x,np.zeros(len(runner_x)),profile_height(runner_x,params)]
cfg=json.loads(Path(doc['programme_structure']['policy']).read_text('utf8'))
limits=ProgrammeJointLimits(SimpleNamespace(reference=whole_ref,body=SimpleNamespace(state=whole)),
    cfg,np.zeros(whole.size),anchor=load(doc['programme_structure']['chart_anchor']))
parent_stats=json.loads((args.parent.parent.parent/'final_metrics.json').read_text())['physical_diagnostics']['anatomical_joint_domain']['joints']
cap=np.array([x['peak_dimensionless_violation'] for x in parent_stats])
guard_joints=6+3*(JOINTS-1)
grasp=(sec>=162.34)&(sec<=163.40)
hold_ids=np.flatnonzero(grasp)
patch=ref.grasp_geometry['palm_patches']['right']
blobs=json.loads(args.source_runner.read_text('utf8'))['segments']
blades=np.searchsorted(shared.p['pts'],[a['pts'] for a in blobs])
np.testing.assert_array_equal(shared.p['pts'][blades],[a['pts'] for a in blobs])
blnodes=(times[blades,None]+camera.readout_times(blades)[:,None]*np.linspace(-.5,.5,5)).ravel()
blref=ref.at(blnodes)
blline=np.array([a['edge_uv'] for a in blobs])
bld=blline[:,1]-blline[:,0];blt=bld/np.linalg.norm(bld,axis=-1)[:,None]
blnormal=blt[:,[1,0]]*np.array([-1.,1.])
blc=blline.mean(axis=1)
blsigma=np.array([a['band_px'] for a in blobs])
blweight=np.stack([4.0/blsigma,4.0/blsigma,1.0/(2*blsigma)],axis=-1)
blselectors=np.stack([blnormal,blnormal,blt],axis=1)*blweight[:,:,None]
blx=runner_x[0]+np.ptp(runner_x)*np.array([.16,.83])
blsteel=np.c_[blx,np.zeros(2),profile_height(blx,params)]
def runner_points(delta):
    pp=fk_state(ref,blref,delta)
    return pp['O'][:,1,None]+np.einsum('nij,kj->nki',pp['R'][:,1],blsteel)
body=np.zeros(state.size)
seed_pose=fk_state(ref,mid_ref,state.increments(body,mid_times))
def headings(A):
    a=A[:,:2,0]
    return np.arctan2(a[:,1],a[:,0])
def angle_delta(a,b):
    return np.arctan2(np.sin(a-b),np.cos(a-b))
def geom(dm):
    p=fk_state(ref,mid_ref,dm)
    hip=headings(p['Q'])
    torso=headings(p['Q']@p['frames'][:,9])
    shoe=headings(p['R'][:,0])
    F=p['O'][:,0]+np.einsum('nij,j->ni',p['R'][:,0],material)
    # Semantic palm patch, from the same posed body as the continuous runner.
    h={k:p[k][hold_ids] for k in ('native','frames','hand_local_R') if k in p}
    H=p['P'][hold_ids]+np.einsum('nij,nj->ni',p['Q'][hold_ids],palm_local(ref,h,patch))
    v=p['O'][hold_ids,1,None]+np.einsum('nij,kj->nki',p['R'][hold_ids,1],steel)
    u=np.linalg.norm(H[:,None]-v,axis=-1).min(axis=-1)
    grip=np.zeros(n);grip[hold_ids]=u
    return np.column_stack([hip,torso,shoe,F,grip])
seed_geo=geom(state.increments(body,mid_times))
seed_rates=angle_delta(seed_geo[timepairs+1,:3],seed_geo[timepairs,:3])/.02
grip_allow=np.maximum(np.minimum(seed_geo[hold_ids,6]+.002/ratio,.061/ratio),.043/ratio)
gain_grip=2.4*ratio/.013
support_gain=2.2*ratio/.045
limit_gain=95.
contiguous=np.flatnonzero(np.diff(timepairs)==1)
scan=load(args.surface_scan)
scan_report=json.loads(args.surface_scan.with_suffix('.json').read_text('utf8'))
if scan_report['parameter_sha256']!=digest(args.seed):
    raise ValueError('Unmatched surface scan')
faces=CollisionSurface(ref,doc).parts
ix=scan['sample'];st=scan['times'][ix]+origin
a,b=faces[scan['face_a']],faces[scan['face_b']]
names=scan['parts'];aa,bb=names[a],names[b]
focus=['head','left_shank','right_palm','right_forearm','right_upper_arm','right_shank','left_palm']
important=np.isin(aa,focus)|np.isin(bb,focus)
inside=((st>=157.04)&(st<=158.85))|((st>=162.10)&(st<=163.68))
surf_rows=inside&(ix%4==0)&(scan['initial_gap']<-.004)&important&(a!=b)
witness_keys=('sample','vertex','face_vertices','bary','normal','initial_gap','face_a','face_b')
witness={k:scan[k][surf_rows] for k in witness_keys}
witness.update(times=scan['times'],surface_response_policy=scan['surface_response_policy'])
sf=ProgrammeSurfaceFactor(SimpleNamespace(reference=ref,body=SimpleNamespace(state=state)),doc,witness,scale_m=.020)
sf_gain=52.
print(json.dumps(dict(event='verified_local_surface_active_set',rows=int(surf_rows.sum()),samples=int(np.unique(ix[surf_rows]).size))),flush=True)
is_core=((sec[timepairs]>=157.42)&(sec[timepairs+1]<=158.20))|((sec[timepairs]>=162.54)&(sec[timepairs+1]<=163.06))
phase_rows=timepairs[is_core]
cached=None
evals=0
def quality(g):
    rate=angle_delta(g[timepairs+1,:3],g[timepairs,:3])/.02
    return dict(core_minimum_rates_radps=rate[is_core].min(axis=0).tolist(),
                core_reverse_rows=(rate[is_core]<0).sum(axis=0).tolist(),
                core_max_20ms_rate_change_radps=np.max(np.abs(np.diff(rate,axis=0))[np.diff(timepairs)==1],axis=0).tolist(),
                support_planar_offset_model_m_p95=float(np.percentile(np.linalg.norm((g[:,3:5]-old_left[:,:2])*ratio,axis=-1),95)),
                grip_palm_proxy_p95_model_m=float(np.percentile(g[hold_ids,6]*ratio,95)))

def evaluate(x):
    global cached,evals,parts_size
    if cached is not None and np.array_equal(x,cached[0]):return cached[1:]
    state_x=np.zeros(state.size);state_x[cols]=x
    dn=state.increments(state_x,nodes);dm=state.increments(state_x,mid_times)
    A=CompactBodyLinearization(ref.body,node_ref['local_R'],dn[:,6:].reshape(-1,21,3),hand_local_R=node_ref.get('hand_local_R'))
    Q=node_ref['Q']@Rotation.from_rotvec(dn[:,3:6]).as_matrix()
    offs=np.einsum('nij,nkj->nki',Q,A.coco)
    pos=offs+(node_ref['P']+dn[:,:3])[:,None]
    projection=moving_projection(camera,pos.reshape(n,5,17,3),native,jacobian=True)
    uv=projection['uv'];err=uv[f,j]-observed
    r=err*weight[:,None]
    hh=np.sqrt(1.+np.sum(r*r,axis=-1)/9.)
    w=np.sqrt(2./(1.+hh))
    transform=w[:,None,None]*np.eye(2)-(w/(18.*hh*(hh+1.)))[:,None,None]*r[:,:,None]*r[:,None,:]
    pix=projection['pixel_jac_nodes'][f,j]
    tangent=Q@right_jacobian(dn[:,3:6])
    mpix=projection['pixel_jac_nodes'][manf,mani]
    def project_columns(pix,world):
        return np.einsum('nphc,nhcd->nphd',pix,world)
    mimg=np.zeros((len(manual),2,5,69))
    mimg[:,:,:,:3]=mpix*manweight[:,None,None,None]
    local_img=np.zeros((n,17,2,5,69))
    local_img[f,slots,:,:,:3]=np.einsum('nuv,nvpc->nupc',transform,pix)*weight[:,None,None,None]
    root_world=(-skew(offs)@tangent[:,None]).reshape(n,5,17,3,3)[f,:,j]
    mworld=(-skew(offs)@tangent[:,None]).reshape(n,5,17,3,3)[manf,:,mani]
    mimg[:,:,:,3:6]=project_columns(mpix,mworld)*manweight[:,None,None,None]
    dr=np.einsum('nphk,nhkj->nphj',pix,root_world)
    local_img[f,slots,:,:,3:6]=np.einsum('nuv,nvhc->nuhc',transform,dr)*weight[:,None,None,None]
    for c in range(6,69):
        k=np.einsum('nij,nkj->nki',Q,A.derivative(c,kinematics=False)['coco']).reshape(n,5,17,3)
        mimg[:,:,:,c]=np.einsum('nphc,nhc->nph',mpix,k[manf,:,mani])*manweight[:,None,None]
        deriv=np.einsum('nphc,nhc->nph',pix,k[f,:,j])
        local_img[f,slots,:,:,c]=np.einsum('nab,nbh->nah',transform,deriv)*weight[:,None,None]
    Ji=chain_exposure(local_img.reshape(n,34,5,69),dn)[outrows][:,cols]
    # Keep original assignments intact and add independently reviewed landmarks.
    full_manual=np.zeros((n,34,5,69))
    manrows=2*mani[:,None]+np.arange(2)
    full_manual[manf[:,None],manrows]=mimg
    Jmanual=chain_exposure(full_manual,dn)[(34*manf[:,None]+manrows).ravel()][:,cols]
    rmanual=(uv[manf,mani]-manuv)*manweight[:,None]

    bd=state.increments(state_x,blnodes)
    bp=runner_points(bd)
    proj=moving_projection(camera,bp.reshape(len(blades),5,2,3),blades,jacobian=True)
    buv=proj['uv']
    v=buv-blc[:,None]
    ru=np.einsum('fip,fip->fi',v,blselectors[:,:2])
    rm=np.einsum('fp,fp->f',v.mean(axis=1),blselectors[:,2])
    direction=buv[:,1]-buv[:,0]
    length=np.sqrt(np.sum(direction**2,axis=-1)+4.)
    transverse=np.sum(direction*blnormal,axis=1)
    angular_gain=42.
    ra=angular_gain*transverse/length
    ag=angular_gain*(blnormal/length[:,None]-(transverse/length**3)[:,None]*direction)
    length_min=np.maximum(np.linalg.norm(bld,axis=-1)*.55,8.)
    rl=np.maximum(length_min-length,0.)*.5
    lg=-.5*(rl>0)[:,None]*direction/length[:,None]
    rblade=np.column_stack([ru,rm,ra,rl])
    jb=np.zeros((len(blades),2,2,5,69))
    pixb=proj['pixel_jac_nodes']
    for c in np.r_[np.arange(6),np.concatenate([np.arange(6+3*(k-1),9+3*(k-1)) for k in (2,5,8,11)])]:
        p,m=bd.copy(),bd.copy();p[:,c]+=1e-6;m[:,c]-=1e-6
        db=((runner_points(p)-runner_points(m))/2e-6).reshape(len(blades),5,2,3).transpose(0,2,1,3)
        jb[...,c]=np.einsum('fkphc,fkhc->fkph',pixb,db)
    jnormal=np.einsum('fkp,fkphc->fkhc',blselectors[:,:2],jb)
    jcentroid=np.einsum('fp,fphc->fhc',blselectors[:,2],jb.mean(axis=1))
    dv=jb[:,1]-jb[:,0]
    jangle=np.einsum('fp,fphc->fhc',ag,dv)
    jlength=np.einsum('fp,fphc->fhc',lg,dv)
    Jblade=chain_exposure(np.concatenate([jnormal,jcentroid[:,None],jangle[:,None],jlength[:,None]],axis=1),bd)[:,cols]

    g=geom(dm)
    Jg=np.zeros((n,7,69));eps=1e-6
    kin=np.r_[np.arange(6),np.concatenate([np.arange(6+3*(k-1),9+3*(k-1)) for k in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21)])]
    for c in kin:
        plus,minus=dm.copy(),dm.copy();plus[:,c]+=eps;minus[:,c]-=eps
        gp,gm=geom(plus),geom(minus)
        Jg[:,:3,c]=angle_delta(gp[:,:3],gm[:,:3])/(2*eps)
        Jg[:,3:,c]=(gp[:,3:]-gm[:,3:])/(2*eps)
    Jangle=dm.chain(Jg[:,:3])[:,cols]
    delta_phase=angle_delta(g[:,:3],seed_geo[:,:3])
    root_weight=np.array([4.1,.30,.35])
    rp=delta_phase*root_weight
    jp=diags(np.tile(root_weight,n))@Jangle
    idx=(3*timepairs[:,None]+np.arange(3)).ravel()
    S=(Jangle[idx+3]-Jangle[idx])/.02
    rates=angle_delta(g[timepairs+1,:3],g[timepairs,:3])/.02
    # Root speed can follow the source interval cadence; the distal and
    # torso headings have slack to change their own tilt and stride.
    rates_weight=np.array([.19,.10,.12])
    rd=(rates-seed_rates)*rates_weight
    jd=diags(np.tile(rates_weight,len(timepairs)))@S
    accel_weight=np.array([1.35,.72,.90])
    ar=(rates[contiguous+1]-rates[contiguous])*accel_weight
    ti=(3*contiguous[:,None]+np.arange(3)).ravel()
    ja=diags(np.tile(accel_weight,len(contiguous)))@(S[ti+3]-S[ti])
    # Permit relative rotation as the skater changes pose, with a broad deadband.
    together=is_core
    relative_rate=rates[together,1:]-rates[together,:1]
    relative_gain=np.array([1.10,1.25])
    carry=np.maximum(abs(relative_rate)-2.8,0.)*relative_gain
    jd3=S[np.repeat(together,3)].tocsr()
    base_row=3*np.arange(int(together.sum()))
    rr=(base_row[:,None]+np.array([1,2])).ravel()
    root_row=np.repeat(base_row,2)
    jcarry=diags(((carry>0)*np.sign(relative_rate)*relative_gain).ravel())@(jd3[rr]-jd3[root_row])
    minrates=np.array([6.,4.,3.])
    reverse=np.maximum(minrates-rates[is_core],0.)
    reverse_weight=np.array([1.6,1.05,1.25])
    row_mask=np.repeat(is_core,3)
    jr=diags((-(reverse>0).astype(float)*reverse_weight).ravel())@S[row_mask]
    fp=(g[:,3:5]-old_left[:,:2])*support_gain
    jf=dm.chain(Jg[:,3:5]*support_gain)[:,cols]
    grip=np.maximum(g[hold_ids,6]-grip_allow,0.)
    jh=dm.chain(Jg[hold_ids,6:7]*(grip>0)[:,None,None]*gain_grip,sample_indices=hold_ids)[:,cols]
    score=limits.values(dm,native)[0]
    over=np.maximum(score-np.maximum(cap-.013,0.)[None,:,None],0.)
    L=np.zeros((n,3*len(cap),69))
    for k,c in enumerate(guard_joints):
        for a in range(3):
            p,m=np.asarray(dm).copy(),np.asarray(dm).copy();p[:,c+a]+=eps;m[:,c+a]-=eps
            slope=(limits.values(p,native)[0][:,k]-limits.values(m,native)[0][:,k])/(2*eps)
            L[:,3*k:3*k+3,c+a]=limit_gain*slope*(over[:,k]>0)
    jl=dm.chain(L)[:,cols]
    values=[(r*w[:,None]).ravel(),rmanual.ravel(),rblade.ravel(),rp.ravel(),rd.ravel(),ar.ravel(),carry.ravel(),(reverse*reverse_weight).ravel(),fp.ravel(),grip*gain_grip,limit_gain*over.ravel(),regularizer@x]
    ss=sf.evaluate(state_x,True)
    values.append(ss['residual']*sf_gain)
    parts_size=np.cumsum([len(v) for v in values])[:-1]
    residual=np.concatenate(values)
    J=vstack([Ji,Jmanual,Jblade,jp,jd,ja,jcarry,jr,jf,jh,jl,regularizer,ss['jac'][:,17+cols]*sf_gain],format='csr')
    costs=dict(zip(('raw_native_body','manual_visible_joints','visible_runner_band','phase','rate','accel','relative_speed','direction','left_material','grip_proxy','anatomical','control_prior','actual_surface'),(float(q@q/2) for q in values)))
    cached=x.copy(),residual,J,g,uv,score,costs
    evals+=1
    if evals==1 or evals%4==0:
        print(json.dumps(dict(event='local_native_objective',eval=evals,cost=float(residual@residual/2),parts=costs,
             source_frames=n,nnz=J.nnz,elapsed_s=time.perf_counter()-started,**quality(g))),flush=True)
    return cached[1:]

def summary(r):
    e=np.linalg.norm(r[3][f,j]-observed,axis=-1)
    return dict(raw_image_rms_px=float(np.sqrt(np.mean(e*e))),raw_image_p95_px=float(np.percentile(e,95)),
                manual_p95_px=float(np.percentile(np.linalg.norm(r[3][manf,mani]-manuv,axis=-1),95)),
        pose_domain_peak=r[4].max(axis=(0,2)).tolist(),parts=r[5],**quality(r[2]))
x0=np.zeros(len(cols))
first_value=evaluate(x0)
rng=np.random.default_rng(1109)
direction=rng.normal(size=len(x0));direction/=np.linalg.norm(direction)
h=7e-6
der=(evaluate(h*direction)[0]-evaluate(-h*direction)[0])/(2*h)
analytic=first_value[1]@direction
jac_error=float(np.linalg.norm(der-analytic)/max(np.linalg.norm(der),1e-9))
print(json.dumps(dict(event='local_jacobian_checked',relative_error=jac_error,by_block={k:[float(np.linalg.norm(a-b)),float(np.linalg.norm(a))] for k,a,b in zip(first_value[5],np.split(der,parts_size),np.split(analytic,parts_size))})),flush=True)
if not jac_error<.008:raise ValueError('Local exact-direction check failed')
box=np.r_[np.array([.070,.070,.045])/ratio,np.full(3,.14),np.full(63,.45)]
fit=least_squares(lambda x:evaluate(x)[0],x0,jac=lambda x:evaluate(x)[1],
    bounds=(-np.tile(box,len(active)),np.tile(box,len(active))),method='trf',tr_solver='lsmr',
    x_scale='jac',ftol=6e-5,xtol=8e-5,gtol=2e-4)
full=np.zeros(whole.size);full[global_cols]=fit.x
actual=whole.current_curve(full)
prior=curve_from_packet(seed,seed=True)
x=seed['x'].copy();xx=x[17:17+whole.body_size].reshape(-1,69)
xx[active,:3]=actual.position_controls[active]-prior.position_controls[active]
a=Rotation.from_matrix((prior.rotation_controls[active].swapaxes(-1,-2)@actual.rotation_controls[active]).reshape(-1,3,3)).as_rotvec()
nr=np.linalg.norm(a,axis=1,keepdims=True)
lift=2*np.pi*a/np.maximum(nr,1e-12)
old_x=xx[active,3:].reshape(-1,3)
cand=np.stack([a,a+lift,a-lift])
dist=np.linalg.norm(cand-old_x,axis=-1)
xx[active,3:]=cand[dist.argmin(0),np.arange(len(a))].reshape(len(active),66)
test=prior.retracted(xx[:,:3],xx[:,3:].reshape(-1,22,3))
np.testing.assert_allclose(test.rotation_controls,actual.rotation_controls,rtol=0,atol=1.1e-11)
take=np.setdiff1d(np.arange(len(actual.position_controls)),active)
np.testing.assert_array_equal(actual.rotation_controls[take],parent['absolute_rotation_controls'][take])
np.testing.assert_array_equal(actual.position_controls[take],parent['absolute_position_controls'][take])
saved=dict(seed,x=x,absolute_position_controls=actual.position_controls.copy(),
           absolute_rotation_controls=actual.rotation_controls.copy())
with out.open('xb') as file:np.savez_compressed(file,**saved)
after=evaluate(fit.x)
report=dict(seed=summary(first_value),candidate=summary(after),parent_source_rms_px=float(np.sqrt(np.mean(old_pixels**2))),
    parent_source_p95_px=float(np.percentile(old_pixels,95)),elapsed_seconds=time.perf_counter()-started,
    parent_sha256=digest(args.parent),candidate_sha256=digest(out),affected_exposures=n,affected_intervals=[
        [float(sec[a[0]]),float(sec[a[-1]])] for a in np.split(np.arange(n),np.flatnonzero(np.diff(native)>1)+1)],
    actual_jacobian_direction_error=jac_error,solver_message=fit.message,nfev=fit.nfev,
    whole_continuous_controls=active.tolist(),unchanged_outside_support=True,source_assignments_unchanged=True,accuracy_accepted=False,manual_landmarks_sha256=digest(args.source_landmarks),
    visible_runner_bands_sha256=digest(args.source_runner))
with out.with_suffix('.json').open('x',encoding='utf8') as fobj:json.dump(report,fobj,indent=2)
print(json.dumps(dict(event='local_trial_saved',**report)),flush=True)
