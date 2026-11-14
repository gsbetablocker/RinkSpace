from pathlib import Path
from types import SimpleNamespace
import json
import os
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.sparse import diags, vstack, block_diag
from scipy.sparse.linalg import lsmr
from scipy.spatial.transform import Rotation
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference, curve_from_packet
from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve, _exp, _log
from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.programme_repair_factors import component_ice
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from experiments.skate_video_v1.programme_audit_repair import fit_rotation_track
from experiments.skate_video_v1.programme_motion_followup import bind
from experiments.skate_video_v1.motion_review.scoped_fit import geometry,save_candidate

run=Path(os.environ['RINKSPACE_AUDIT_RUN'])
w=run.parent/'audit_af48_1113'
prep=run/'prepared_refined'
cfile=run/'opening_direction_1113/refit/solve/candidate.npz'
bfile=run/'joint_repair_detail/solve/candidate.npz'
r,cam,info,doc,C=bind(prep,w/'revised_source.json',cfile)
B=load(bfile)
b,c=curve_from_packet(B),curve_from_packet(C)
np.testing.assert_array_equal(b.knots,c.knots)
clock=np.asarray(r.times)
source=np.asarray(r.p['pts'])/12800
obs=load(prep/'observations.npz')
mass=ContinuousMassGeometry(r)
def interval(first,last):
    return np.flatnonzero((source>=first-1e-9)&(source<=last+1e-9))
def time_at(source_seconds):
    return np.interp(source_seconds,source,clock)
def active(first,last):
    a,z=time_at([first,last])
    return np.flatnonzero((c.knots[:-4]<z)&(c.knots[4:]>a))
def ref_for(curve):
    trial=absolute_reference(r,curve)
    trial.relative_skate_curve=r.relative_skate_curve
    return trial
def state(curve,t):
    trial=ref_for(curve)
    return fk_state(trial,trial.at(t),np.zeros((len(t),69)))
def pixels(curve,frames,jac=False):
    t=clock[frames,None]+cam.readout_times(frames)[:,None]*np.linspace(-.5,.5,5)
    s=state(curve,t.ravel())
    pp=r.body.forward_rotations(s['local_R'],hand_local_R=s.get('hand_local_R'))[0]
    world=s['P'][:,None]+np.einsum('nij,nkj->nki',s['Q'],pp)
    out=moving_projection(cam,world.reshape(len(frames),5,17,3),frames,jacobian=jac)
    return out,t
def pixel_report(curve,frames):
    uv=pixels(curve,frames)[0]['uv']
    error=np.linalg.norm(uv-obs['uv'][frames],axis=-1)
    use=obs['use'][frames]
    return [float(np.median(error[use])),float(np.percentile(error[use],90)),
            float(np.percentile(error[:,[5,6,11,12]][use[:,[5,6,11,12]]],90))]
def opening(curve,weight):
    ids=active(24.44,25.04)
    frames=interval(24.24,25.18)
    # Use the same camera, model and actual admitted rows for the opening.
    # A sparse native-curve solve updates translation with analytic B''.
    t=np.unique(np.r_[np.arange(time_at(24.18),time_at(25.24),.005),c.knots])
    t=t[(t>=time_at(24.18))&(t<=time_at(25.24))]
    ns=np.interp(t,clock,source)
    D=c.basis_sparse(t,2)
    M=c.basis_sparse(t)
    wh=np.where((ns>=24.54)&(ns<=24.98),1.,.65)
    wm=np.where((ns<=24.52)|(ns>=25.00),1.2,.4)
    p=curve.position_controls.copy()
    before=curve.position_controls.copy()
    target=b.position_derivative(t,2)
    for _ in range(3):
        fit=geometry(c,p,curve.rotation_controls)
        out,readout=pixels(fit,frames,jac=True)
        ff,jj=np.nonzero(obs['use'][frames])
        residual=obs['uv'][frames[ff],jj]-out['uv'][ff,jj]
        sigma=np.maximum(obs['sigma_px'][frames[ff],jj],2.)
        if sigma.ndim==1:sigma=sigma[:,None]
        wt=(1.+(residual/(2.5*sigma))**2)**(-.25)/sigma
        camera=np.asarray(out['pixel_jac_nodes'])[ff,jj]
        time_b=c.basis_sparse(readout.ravel())[:,ids].toarray().reshape(len(frames),5,len(ids))[ff]
        JP=np.einsum('nulk,nli->nuki',camera,time_b)
        JP=(JP*wt[:,:,None,None]).reshape(len(ff)*2,-1)
        A=(D[:,ids].multiply((weight*wh)[:,None])).toarray()
        P=(M[:,ids].multiply((12*wm)[:,None])).toarray()
        matrix=np.vstack([JP,block_diag([A]*3).toarray(),block_diag([P]*3).toarray()])
        rhs=np.r_[(residual*wt).ravel(),((target-D@p)*(weight*wh)[:,None]).T.ravel(),
                     ((M@(before-p))*(12*wm)[:,None]).T.ravel()]
        dp=lsmr(matrix,rhs,atol=1e-10,btol=1e-10,maxiter=600)[0].reshape(3,-1).T
        p[ids]+=dp
    return geometry(c,p,curve.rotation_controls)
def legs(curve,scale):
    R=curve.rotation_controls.copy()
    for first,last,tracks in [(57.02,57.20,[2,5,8]),(69.32,69.60,[1,2,4,5,7,8])]:
        ix=active(first,last)
        delta=_log(R[ix[:,None],tracks].swapaxes(-1,-2)@b.rotation_controls[ix[:,None],tracks])
        R[ix[:,None],tracks]=R[ix[:,None],tracks]@_exp(scale*delta)
    return geometry(c,curve.position_controls,R)
def spin(curve,scale):
    frames=interval(81.30,92.86)
    t=clock[frames]
    s=state(curve,t)
    raw=np.unwrap(np.arctan2(s['R'][:,1,1,0],s['R'][:,1,0,0]))
    # Interpolate across the bad branch using confirmed progression on both
    # sides. End-on frames do not supply a precise image-plane angle.
    anchors=[84.82,84.92,85.04,85.16,85.26]
    t0=time_at(anchors)
    ph=np.interp(t0,t,raw)
    ph[2]=.61*ph[1]+.39*ph[3]
    if not np.all(np.diff(ph)>0):
        raise RuntimeError(f'No monotone donor at boundaries {np.rad2deg(ph)}')
    target=PchipInterpolator(t0,ph)
    change=np.zeros(len(t))
    m=(t>=t0[0])&(t<=t0[-1])
    change[m]=scale*(target(t[m])-raw[m])
    R=curve.rotation_controls.copy()
    # In world space this changes the whole assembly, transferred through
    # the fixed ankle binding into the anatomical local_R coordinates.
    parent=s['Q']@s['frames'][:,5]
    q=Rotation.from_euler('z',change).as_matrix()
    wanted=parent.swapaxes(-1,-2)@q@parent@s['local_R'][:,7]
    sub=interval(84.54,85.54)
    use=np.searchsorted(frames,sub)
    tt=t[use]
    Bmat=c.basis_sparse(tt)
    weights=np.where((tt>=t0[0])&(tt<=t0[-1]),1.,12.)
    updated=fit_rotation_track(c.knots,tt,Bmat,weights,curve.position_controls,
                               R[:,8],curve.edge_reference_quaternion[:,8],wanted[use])
    ix=active(84.82,85.26)
    R[ix,8]=updated[ix]
    return geometry(c,curve.position_controls,R)
def inspect(curve):
    a=interval(24.60,24.90);t=clock[a]
    sub=np.unique(np.r_[np.linspace(t[0],t[-1],301),c.knots[(c.knots>=t[0])&(c.knots<=t[-1])]])
    peak=float(np.linalg.norm(curve.position_derivative(sub,2)*r.binding.ratio,axis=1).max())
    co=mass.com(mass.from_pose(state(curve,t))[0])
    com=float(np.linalg.norm(np.diff(co,n=2,axis=0)/(.02**2),axis=1).max())
    v={}
    for t0,foot in [(57.12,1),(69.44,0)]:
        ii=interval(t0-.12,t0+.12);times=np.sort(np.r_[clock[ii],(clock[ii][1:]+clock[ii][:-1])/2])
        n,h=component_ice(r.binding,state(curve,times))
        k=int(np.argmin(abs(times-time_at(t0))))
        v[str(t0)]=dict(at=dict(zip(n,h[k,foot].tolist())),interval_min=dict(zip(n,h[:,foot].min(0).tolist())))
    u=state(curve,clock[interval(84.64,85.26)])
    yaw=np.rad2deg(np.unwrap(np.arctan2(u['R'][:,1,1,0],u['R'][:,1,0,0])))
    return dict(opening_root_peak_mps2=peak,opening_COM_peak_mps2=com,
        opening_pixels=pixel_report(curve,interval(22.12,25.18)), ice=v,
        spin_net_8494_8500=yaw[18]-yaw[15],spin_net_8464_8476=yaw[6]-yaw[0],
        spin_max_step_deg=np.max(abs(np.diff(yaw))),
        step_8498_8500_deg=float(np.rad2deg(np.linalg.norm(_log(u['local_R'][17,7].T@u['local_R'][18,7])))))

from calibration.program_tracking.body_tracking.skate_physics import NORMALIZATION
out=w/'proposals'
selected=load(w/'solve_visible_ice/solve/candidate.npz')
chosen=curve_from_packet(selected)
trial=curve_from_packet(load(out/'spin_hip_motion.npz'))
frame=interval(84.58,85.51)
ts=clock[frame]
def envelope(p):
    fit=state(p,ts)
    v=r.binding.foot_metrics(fit['native'],fit['frames'],hand_local_R=fit.get('hand_local_R'),
                             skate_relative_R=fit['skate_relative_R'])
    return np.stack([np.max(v0,axis=(1,2)) for v0 in v['violations']],axis=1)*r.binding.ratio
before=envelope(chosen)
bad=envelope(trial)
def score(p): return ((envelope(p)/NORMALIZATION)**2*.02*float(selected['penalty'])).sum(axis=0)
refscore=score(chosen);failed=score(trial)
delta=failed-refscore
print('cavity native changed cost',refscore,failed,delta,flush=True)
controls=active(84.90,85.08)
controls=controls[(c.knots[controls]>=time_at(84.78)-1.e-10)&(c.knots[controls+4]<=time_at(85.18)+1.e-10)]
# Toes remain inside the fixed boot, with the same ankle and immutable mount.
# Test small left foot-joint and ankle updates using the actual shaped surface.
# Full fit replays both feet, all source rows and each physical guard.
for name,tracks in [('spin_hip_cavity_toes',[10]),('spin_hip_cavity_leg',[7,10])]:
    grads=np.zeros((len(controls),len(tracks),3))
    h=8.e-5
    for i,k in enumerate(controls):
        for j,joint in enumerate(tracks):
            for axis in range(3):
                v=np.eye(3)[axis]*h
                Ra=trial.rotation_controls.copy();Rb=Ra.copy()
                Ra[k,joint]=Ra[k,joint]@_exp(v)
                Rb[k,joint]=Rb[k,joint]@_exp(-v)
                p=geometry(c,trial.position_controls,Ra)
                q=geometry(c,trial.position_controls,Rb)
                grads[i,j,axis]=(score(p)[0]-score(q)[0])/(2*h)
    print(name,'grad',np.linalg.norm(grads),flush=True)
    target=max(3.0*delta[0],.016)
    for frac in (1.,2.):
        d=-grads*target*frac/np.sum(grads**2)
        maxstep=np.linalg.norm(d,axis=-1).max()
        print('max step',np.rad2deg(maxstep),flush=True)
        if maxstep>np.deg2rad(2.0):
            continue
        R=trial.rotation_controls.copy()
        R[controls[:,None],tracks]=R[controls[:,None],tracks]@_exp(d)
        current=geometry(c,trial.position_controls,R)
        actual=score(current)
        print(name,frac,'left right cost',actual-refscore,'previous',delta,flush=True)
        if np.any(actual-refscore>1.e-4):
            continue
        save_candidate(C,current,out/(name+'.npz'),parent=cfile)
        break
