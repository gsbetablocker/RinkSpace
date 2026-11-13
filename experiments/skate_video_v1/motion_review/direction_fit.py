"""Fit the orientation of the attached skate from within-shot source anchors."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.absolute_motion_curve import _exp, greville_times, RotationChartDomainError
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference
from calibration.program_tracking.body_tracking.programme_opening import standing_contact_rows
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.skate_physics import batch_features, fk_state
from ..body_skate_fit import Assembly
from ..inputs import PROGRAMME, atomic_json, digest, read
from ..programme_motion_followup import bind
from .scoped_fit import Fit, cpu, save_candidate, smooth, ti, tn
from .source_audit import projection_blocks, static_assembly

LIMITS='Partial, end-on and occluded views were used for review only. Endpoints are direct native-image estimates with blur and perspective uncertainty.'

def norms(u):
    return np.linalg.norm(u,axis=-1)
def xyangle(u,v):
    return np.degrees(np.arctan2(abs(u[...,0]*v[...,1]-u[...,1]*v[...,0]),(u*v).sum(-1)))
def uv_error(uv,target):
    diff=uv-target
    v,w=np.diff(uv,axis=1)[:,0],np.diff(target,axis=1)[:,0]
    angle=xyangle(v,w)
    return dict(uv_px=norms(diff).mean(axis=1).tolist(),signed_deg=angle.tolist(),
        displacement=norms(np.diff(diff,axis=0)).mean(axis=1).tolist())
def summary_error(row):
    return np.mean(row['uv_px']),np.mean(row['signed_deg'])
def mid(r,curve,t):
    q=absolute_reference(r,curve)
    s=q.at(t)
    return fk_state(q,s,np.zeros((len(t),69)))
def native_eval(r,c,curve,frames,f,columns,mat):
    r=absolute_reference(r,curve)
    a=Assembly(r.binding)
    uv,_=projection_blocks(r,c,frames,a,mat)
    fk=fk_state(r,r.at(r.times[frames]),np.zeros((len(frames),69)))
    g=batch_features(r.binding,fk['R'],fk['O'])
    h=standing_contact_rows(g,np.arange(len(frames)))['height_by_foot']
    W=fk['O'][:,:,None]+np.einsum('nfij,mj->nfmi',fk['R'],mat)
    return uv[:,f,columns],h,W[:,:,columns],fk
def initial_heading(fit,curve,seq,positions,local,foot,targets,ix,columns):
    # Solve an image-space orientation seed in the native shutter centre.
    # This only initializes the anatomical retraction; the actual solve uses
    # the full five moving exposure nodes, both supports, and unchanged camera.
    from scipy.optimize import least_squares
    s=mid(fit.ref,curve,fit.time[ix])
    O,R=s['O'][:,foot],s['R'][:,foot]
    ankle=s['native_world'][:,7+foot]
    local_ankle=np.einsum('nji,nj->ni',R,ankle-O)
    offsets=fit.material[columns]-local_ankle[:,None]
    base=s['Q']@s['frames'][:,4+foot]
    max_angle=np.deg2rad(117.)
    curves=[]
    for i,xy in enumerate(targets):
        w=R[i]@offsets[i].T
        n=w[:,1]-w[:,0]
        amp=np.linalg.norm(xy[1]-xy[0])
        if amp<9:continue
        def residual(d):
            u=ankle[i]+(R[i]@_exp(d)@offsets[i].T).T
            proj=fit.camera.project(u[None],fit.frames[ix[i:i+1]])[0]
            a=np.diff(proj,axis=0)[0]
            return np.r_[(a-(xy[1]-xy[0]))/max(7.,amp*.24),d/2.1]
        candidates=[least_squares(residual,g, max_nfev=25,bounds=(-max_angle,max_angle)) for g in
            ([0.,0.,0.],[0.,1.3,0.],[0.,-1.3,0.],[0.,0.,1.3],[0.,0.,-1.3])]
        best=min(candidates,key=lambda x:x.cost)
        rotvec=(R[i]@best.x)
        curves.append((fit.time[ix[i]],rotvec))
    if not curves:return
    knots,rots=map(np.asarray,zip(*curves))
    delta=np.zeros((len(fit.ctrl),3))
    gt=greville_times(curve.knots)[fit.ctrl]
    for j in range(3):
        w=np.interp(gt,knots,rots[:,j])
        damp=smooth(knots[0]-.20,knots[0]-.035,gt)*(1-smooth(knots[-1]+.035,knots[-1]+.22,gt))
        delta[:,j]=w*damp
    C=mid(fit.ref,curve,gt)
    # Preserve the measured foot positions on the initial retraction. Hip
    # and knee still participate when the solve distributes a large yaw.
    reference_axes=C['Q']@C['frames'][:,7+foot]
    translation=fit.parameter.data
    limb=np.where(fit.active==7+foot)[0][0]+int(fit.fit_position)
    translation[:,limb]=tn(np.einsum('nji,nj->ni',reference_axes,delta)*.58)


def anatomical_swing(local, basis):
    relative=basis.transpose(-1,-2)@local@basis
    # The swing angle is the angle between the rest twist axis and
    # its image under the local rotation; twist leaves that axis fixed.
    axial=relative[...,0,0].clamp(-1.,1.)
    cross=torch.linalg.vector_norm(relative[...,1:,0],dim=-1)
    return torch.atan2(cross,axial)


def authored_limits(fit, frames, foot, controls):
    from calibration.program_tracking.body_tracking.anatomical_chart import JointChart, JointEnvelope
    from calibration.program_tracking.body_tracking.programme_joint_limits import angles_joint
    policy=read(PROGRAMME/'structure_camera_1108_policy.json')
    ids=np.array([4+foot,foot,6+foot])
    names=('hip','knee','ankle')
    chart=JointChart(controls['anatomical_rest_R'][ids],controls['anatomical_basis'][ids],
                     controls['anatomical_reference_twist'][:,ids])
    inverse=tn(chart.basis.swapaxes(-1,-2)@chart.rest.swapaxes(-1,-2))
    basis=tn(chart.basis.copy())
    anchor=tn(chart.reference_twist[frames])
    envelopes=[]
    torch_rows=[]
    for name in names:
        cfg=policy['envelopes'][name]
        low,high=np.deg2rad(cfg['flex_degrees'])
        grid=np.deg2rad(cfg['flex_grid_degrees'])
        axes=np.deg2rad(cfg['swing_axes_degrees'])
        flex_scale,swing_scale=np.deg2rad(cfg['scale_degrees'])
        envelopes.append(JointEnvelope(low,high,grid,axes,flex_scale,swing_scale,policy['provenance']))
        torch_rows.append((low,high,tn(grid),tn(axes),flex_scale,swing_scale))
    def exact(local):
        angle=chart.evaluate(local,sample_indices=frames)
        return np.stack([e.residual(angles_joint(angle,j)) for j,e in enumerate(envelopes)],axis=1)
    def trial(local):
        delta=inverse@local@basis
        u=torch.atan2(delta[...,2,1]-delta[...,1,2],delta[...,1,1]+delta[...,2,2])
        twist=anchor+torch.remainder(u-anchor+torch.pi,2*torch.pi)-torch.pi
        off=torch.stack([-delta[...,2,0],delta[...,1,0]],dim=-1)
        norm=torch.linalg.vector_norm(off,dim=-1)
        swing=torch.atan2(norm,delta[...,0,0])
        vector=off*(swing/norm.clamp_min(1.e-7))[...,None]
        output=[]
        for j,(low,high,grid,axes,flex_scale,swing_scale) in enumerate(torch_rows):
            x=twist[:,j].clamp(grid[0],grid[-1])
            i=torch.searchsorted(grid,x).clamp(1,len(grid)-1)
            frac=(x-grid[i-1])/(grid[i]-grid[i-1])
            radius=axes[i-1]+frac[:,None]*(axes[i]-axes[i-1])
            aspect=torch.linalg.vector_norm(vector[:,j]/radius,dim=-1)
            excess=(swing[:,j]-swing[:,j]/aspect.clamp_min(1.e-6))/swing_scale
            output.append(torch.stack([(low-twist[:,j])/flex_scale,
                                       (twist[:,j]-high)/flex_scale,excess],dim=-1))
        return torch.relu(torch.stack(output,dim=1))
    return trial,exact


def refine_window(r,c,curve,window,full_obs,mat,controls,*,guard_ankle_swing=False):
    measured=window['rows'];foot=window['foot'];other=1-foot
    all_pts=r.p['pts']; shot=r.p['shot_ids']
    idx=np.searchsorted(all_pts,[x['pts'] for x in measured])
    measured_uv=np.asarray([x['uv'] for x in measured])
    fit_mask=np.array([x['role']=='fit' for x in measured])
    check=~fit_mask
    cols=np.array([0,1] if window['kind']=='sole' else [2,3])
    if not fit_mask.any():raise ValueError('No actual fit exposures')
    np.testing.assert_array_equal(all_pts[idx],[x['pts'] for x in measured])
    i0,i1=idx[[0,-1]];sid=shot[i0]
    if np.any(shot[i0:i1+1]!=sid):raise ValueError('Source window crosses shot cut')
    same=np.flatnonzero(shot==sid)
    g=greville_times(curve.knots);t=r.times
    # Free controls covering the original-image episode plus its entry/exit.
    # A spline control influences a continuous interval, not a single image.
    ctrl=np.flatnonzero((g>=t[i0]-.26)&(g<=t[i1]+.26)&
                        (curve.knots[:-4]>=t[same[0]]-1.e-8)&
                        (curve.knots[4:]<=t[same[-1]]+1.e-8))
    if window.get('broader_motion_context'):
        ctrl=np.flatnonzero((g>=t[i0]-.36)&(g<=t[i1]+.36)&
                            (curve.knots[:-4]>=t[same[0]]-1.e-8)&
                            (curve.knots[4:]<=t[same[-1]]+1.e-8))
    if not len(ctrl):
        return curve,dict(window=window['name'],state='source review at cut',shot=str(sid))
    lo=t[0] if not len(ctrl) else curve.knots[ctrl[0]]
    hi=curve.knots[ctrl[-1]+4]
    frames=np.flatnonzero((t>=lo-.03)&(t<=hi+.03)&(shot==sid))
    foot_track=[1+foot,4+foot,7+foot,10+foot]
    fit=Fit(r,c,curve,frames,foot_track,ctrl,fit_position=False)
    # Exact raw-clock holdouts never enter the fitting or direction initializer.
    tr=np.searchsorted(frames,idx[fit_mask]);uv_fit=measured_uv[fit_mask]
    ix_t=ti(tr);tgt=tn(uv_fit)
    dt=np.diff(r.times[idx[fit_mask]])
    pair=np.flatnonzero((dt<=.18)&(dt>=.019999))
    pair_t=ti(pair)
    step_dt=tn(dt[pair])[:,None,None]
    width=np.maximum(norms(uv_fit[:,1]-uv_fit[:,0]),14.)
    radius=np.array([q.get('uncertainty_px',7.) for q in measured])[fit_mask]
    norm=tn(np.maximum(radius,.15*width))
    mv=np.diff(uv_fit,axis=1)[:,0]
    tdir=tn(mv/np.maximum(norms(mv)[:,None],1.e-8))
    quiet=ti(np.flatnonzero((full_obs['use'][frames] &
        (full_obs['model_joint_ids'][frames]==13+foot)).any(axis=1)&
        ~np.isin(frames,idx[check])))
    uk,jk=np.where((full_obs['use'][frames]&(full_obs['model_joint_ids'][frames]==13+foot))&
        (~np.isin(frames,idx[check]))[:,None])
    knee_uv=tn(full_obs['uv'][frames[uk],jk])
    # Body detector's ankle can lock to the wrong ski; camera-visible knee
    # and neighbouring validated shoe evidence keep the limb anchored.
    before,old_h,old_W,fk=native_eval(r,c,curve,frames,foot,cols,mat)
    p0=tn(fk['native_world'][:,[4+foot,7+foot]])
    q0=tn(fk['local_R'])
    oldR=tn(fk['R'][:,foot])
    oldZ=tn(old_h)
    old_point=tn(before)
    old_mat=tn(old_W[:,foot])
    sec=t[frames]
    interior=smooth(t[i0]-.28,t[i0]-.07,sec)*(1-smooth(t[i1]+.07,t[i1]+.28,sec))
    border=tn(1.-interior)
    do_contact=window.get('support') is not None
    expected=tn(smooth(t[i0]-.10,t[i0],sec)*(1-smooth(t[i1],t[i1]+.10,sec)))
    anatomical=window.get('anatomical_repair',False)
    if anatomical:
        from calibration.program_tracking.body_tracking.programme_joint_limits import authored_bases
        model_basis=tn(authored_bases(r.body.J)[[4+foot,foot,6+foot]])
        legs=[foot_track[j]-1 for j in (0,1,2)]
        swing0=anatomical_swing(q0[:,legs],model_basis)
        # Swing is checked in the same shaped-joint axes as the main solver.
        # Existing excess is permitted to improve without raising its local peak.
        swing_cap=torch.maximum(swing0.max(dim=0).values,tn(np.deg2rad([67.,24.,47.])))
    exact_joint_bounds=window.get('authored_joint_envelope',False)
    if exact_joint_bounds:
        chart_loss,chart_check=authored_limits(fit,frames,foot,controls)
        source_envelope=chart_check(fk['local_R'][:,legs])
        old_envelope=tn(source_envelope)
        np.testing.assert_allclose(cpu(chart_loss(q0[:,legs])),source_envelope,atol=2.e-4,rtol=0.)
    endpoint_fit=fit.material_world
    def objective(s):
        w=endpoint_fit(s)[:,foot,cols]
        uv=fit.project(w)
        proposed=uv[ix_t]
        error=(proposed-tgt)/norm[:,None,None]
        ends=(torch.sqrt(1+error.square())-1).mean()*5.
        d=proposed[:,1]-proposed[:,0]
        d=d/torch.linalg.vector_norm(d,dim=1,keepdim=True).clamp_min(4.)
        polarity=((d-tdir).square().sum(1)).mean()*5.
        vv=uv[ix_t]
        if len(pair):
            change=(vv[pair_t+1]-vv[pair_t]-tgt[pair_t+1]+tgt[pair_t])/step_dt/140.
            step=(torch.sqrt(1+change.square())-1).mean()*3.0
        else:step=ends*0.
        # Maintain temporal smoothness in the exact material endpoints and
        # around the replacement window. Natural fast turns remain permitted.
        world=w[2::5]
        dw=torch.diff(world,dim=0,n=2)
        dq=torch.diff(s[6][2::5,foot],dim=0,n=2)
        curve_penalty=((dw/.012).square().mean()+(dq/.09).square().mean())*.65
        if anatomical:
            # Second differences regularize changes of velocity, preserving
            # real glide and turns. Source deltas constrain the observed pairs.
            curve_penalty*=2.5
            reduced_pose=s[2][2::5][:,legs]
            swing=anatomical_swing(reduced_pose,model_basis)
            excess=torch.relu(swing-swing_cap[None,:])
            body_domain=(excess/tn(np.deg2rad([6.,4.,7.]))).square().mean()*24.
            body_domain+=(swing[:,1]/np.deg2rad(30.)).square().mean()*1.6
        else:
            body_domain=ends*0.

        if exact_joint_bounds:
            actual=chart_loss(s[2][2::5][:,legs])
            local_excess=torch.relu(actual-old_envelope-.10)
            flex_violation=actual[:,:,:2]
            body_domain+=42.*actual.square().mean()+94.*local_excess.square().mean()
            body_domain+=160.*torch.relu(flex_violation-old_envelope[:,:,:2]).square().sum(dim=1).amax()
            if guard_ankle_swing:
                body_domain+=120.*torch.relu(actual[:,2,2]-old_envelope[:,2,2]).square().amax()
        boundary=(((uv-old_point)/11.).square().mean(dim=(1,2))*border).mean()*2.5
        ground=fit.contact_heights(s)
        lift=(torch.relu(-ground[:,foot]-.006)/.014).square().mean()*3.6
        support=((ground[:,foot]/.014).square()*expected).mean()*2.4 if do_contact else (
            ((torch.relu((ground[:,foot]-oldZ[:,foot]).abs()-.035)/.052).square()*expected).mean()*.9)
        pose=s[2][2::5];j=s[5][2::5]
        kd=((j[:,[4+foot,7+foot]]-p0)/.065).square().mean()
        axis_deviation=(pose[:,foot_track[1]-1,0,1].square()+pose[:,foot_track[1]-1,0,2].square()).mean()
        local=(q0[:,foot_track[2]-1].transpose(1,2)@pose[:,foot_track[2]-1]-fit.eye)
        # Excess ankle twist must be shared with a plausible shank/hip
        # motion and is penalized; no independent boot rotation is used.
        twist=((pose[:,foot_track[2]-1,0,2]-q0[:,foot_track[2]-1,0,2]).square()).mean()
        preservation=kd*.9+axis_deviation*3.7+local.square().mean()*1.25+twist*1.7
        kn=fit.project(s[5][:,4+foot:5+foot])[:,0]
        if len(uk):
            source_knee=(torch.sqrt(1.+((kn[ti(uk)]-knee_uv)/15.).square())-1.).mean()*.9
        else:source_knee=ends*0.
        return dict(signed_material_endpoints=ends,heel_to_toe_direction=polarity,
            signed_endpoint_displacement=step,material_acceleration=curve_penalty,
            window_boundary=boundary,steel_clearance=lift,visible_support_or_depth_prior=support,
            limb_anatomy_and_pose=preservation,joint_swing=body_domain,source_knee=source_knee)
    old_test,old_height,world0,before_pose=native_eval(r,c,curve,idx,foot,cols,mat)
    old_ang=uv_error(old_test,measured_uv)
    if np.max(np.asarray(old_ang['signed_deg'])[fit_mask])>55. and not window.get('keep_branch_seed'):
        initial_heading(fit,curve,window,0,0,foot,uv_fit,tr,cols)
    trial=fit.run(objective,55 if exact_joint_bounds else (40 if anatomical else 25),window['name'])
    new_test,new_height,world1,after_pose=native_eval(r,c,trial,idx,foot,cols,mat)
    new_ang=uv_error(new_test,measured_uv)
    before_uv,before_ang=summary_error(uv_error(old_test[check],measured_uv[check])) if check.any() else summary_error(old_ang)
    after_uv,after_ang=summary_error(uv_error(new_test[check],measured_uv[check])) if check.any() else summary_error(new_ang)
    old_nat=native_eval(r,c,curve,frames,foot,cols,mat)
    new_nat=native_eval(r,c,trial,frames,foot,cols,mat)
    d0,d1=old_nat[3],new_nat[3]
    arch=tuple((f,float(np.degrees(np.max(Rotation.from_matrix(
      (d0['local_R'][:,7+f-1].swapaxes(-1,-2)@d1['local_R'][:,7+f-1])).magnitude())))) for f in (0,1))
    move0=np.max(norms(np.diff(old_nat[2][:,foot],axis=0,n=2)))
    move1=np.max(norms(np.diff(new_nat[2][:,foot],axis=0,n=2)))
    old_z, new_z=old_nat[1][:,foot],new_nat[1][:,foot]
    train_improved=np.mean(new_ang['signed_deg'])+2.5<np.mean(old_ang['signed_deg']) or np.mean(new_ang['uv_px'])+2.<np.mean(old_ang['uv_px'])
    # Admission uses untrained raw exposures plus anatomy, unchanged opposite
    # skate, and a continuity check over the entire changed support. No gain
    # is claimed when evidence is nearly end-on.
    score_ok=((after_ang+4<before_ang and after_uv<before_uv+4) or
        (after_uv+2.5<before_uv and after_ang<before_ang+5))
    stable=move1<=max(move0*1.6,.052)
    penetration=min(new_z)>=min(-.026,min(old_z)-.010)
    turns=np.degrees(Rotation.from_matrix(
       (d1['R'][:-1,foot].swapaxes(-1,-2)@d1['R'][1:,foot])).magnitude())
    old_turns=np.degrees(Rotation.from_matrix(
       (d0['R'][:-1,foot].swapaxes(-1,-2)@d0['R'][1:,foot])).magnitude())
    continuity=float(max(turns))<=max(32.,float(max(old_turns))+7.)
    other_change=float(np.max(norms(new_nat[2][:,other]-old_nat[2][:,other])))
    anatomy_ok=arch[foot][1]<102. and other_change<1.e-8
    swing_peaks=None
    if anatomical:
        swing_peaks=[np.rad2deg(cpu(anatomical_swing(tn(p['local_R'][:,legs]),model_basis).max(dim=0).values))
                     for p in (d0,d1)]
        swing_allowance=np.maximum(swing_peaks[0],np.array([67.,24.,47.]))+np.array([5.,4.,6.])
        anatomy_ok=anatomy_ok and np.all(swing_peaks[1]<=swing_allowance)
    chart_peaks=None
    if exact_joint_bounds:
        final_limits=chart_check(d1['local_R'][:,legs])
        chart_peaks=np.stack([source_envelope.max(0),final_limits.max(0)])
        anatomy_ok=(other_change<1.e-8 and np.all(swing_peaks[1]<=swing_allowance)
                    and np.all(chart_peaks[1]<=chart_peaks[0]+.16))
    accepted=bool(score_ok and train_improved and stable and penetration and continuity and anatomy_ok)
    print(json.dumps(dict(window=window['name'],retained=accepted,
       held_before=[before_uv,before_ang],held_after=[after_uv,after_ang],
       gates=dict(score=bool(score_ok),training=bool(train_improved),continuity=bool(continuity),
       anatomical=bool(anatomy_ok),accel=bool(stable),penetration=bool(penetration)))),flush=True)
    report=dict(window=window['name'],state='retained' if accepted else 'unresolved',shot=str(sid),foot=foot,
        kind=window['kind'],before=old_ang,trial=new_ang,
        held_before_px=before_uv,held_trial_px=after_uv,held_before_deg=before_ang,held_trial_deg=after_ang,
        measured_pts=all_pts[idx].tolist(),fit_pts=all_pts[idx[fit_mask]].tolist(),check_pts=all_pts[idx[check]].tolist(),
        source_window_pts=[int(all_pts[frames[0]]),int(all_pts[frames[-1]])],
        max_ankle_change_deg=arch[foot][1],opposite_shoe_max_change_m=other_change,
        max_material_second_difference_m=[float(move0),float(move1)],
        max_native_rotation_step_deg=[float(max(old_turns)),float(max(turns))],
        min_steel_height_m=[float(min(old_z)),float(min(new_z))],
        authored_envelope_violation_peaks=None if chart_peaks is None else chart_peaks.tolist(),
        model_swing_peak_deg=None if swing_peaks is None else [x.tolist() for x in swing_peaks],
        gates=dict(score=bool(score_ok),training=bool(train_improved),continuity=bool(continuity),
          anatomical=bool(anatomy_ok),accel=bool(stable),penetration=bool(penetration)),
        measured_uv=measured_uv.tolist(),baseline_uv=old_test.tolist(),trial_uv=new_test.tolist())
    return (trial if accepted else curve),report

def main(args):
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    opening, out, source_path = args.parameter, args.output, args.source
    out.mkdir(parents=True, exist_ok=True)
    r, c, info, _, z = bind(args.prepared, args.evidence, opening)
    original = load(args.parent)
    _, mat = static_assembly(r)
    obs = load(args.prepared/'observations.npz')
    cases=read(source_path)
    rows=[];curve=r.absolute_curve
    for window in cases['windows']:
        try:
            curve,report=refine_window(r,c,curve,window,obs,mat,z,guard_ankle_swing=args.guard_ankle_swing)
        except (RotationChartDomainError, ArithmeticError) as error:
            report=dict(window=window['name'], foot=window['foot'], state='numerical failure',
                        measured_pts=[row['pts'] for row in window['rows']], error=str(error))
            print(json.dumps(report),flush=True)
        rows.append(report)
        atomic_json(out/'direction_progress.json',dict(rows=rows))
    save_candidate(original,curve,out/'candidate.npz',parent=args.parent)
    report=dict(scope='all 8329 source exposures and both attached skates screened, no across-cut image displacement',
        direct_source_path=str(source_path),notes=LIMITS,windows=rows,
        status_counts=dict(Counter(x['state'] for x in rows)),
        parent_parameter_sha256=digest(args.parent),result_sha256=digest(out/'candidate.npz'),
        camera_sha256=info['state_sha256'],opening_parameter_sha256=digest(opening))
    atomic_json(out/'assessment.json',report)
    print(json.dumps(dict(status='complete',candidate=str(out/'candidate.npz'),decision=report['status_counts'])),flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("prepared", "evidence", "parameter", "parent", "source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--guard-ankle-swing", action="store_true")
    main(parser.parse_args())
