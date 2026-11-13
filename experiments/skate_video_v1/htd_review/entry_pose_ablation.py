"""Paired native SO(3) refits of the flying-entry upper body, with and without PVA."""
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .baseline_audit import metric
from ..motion_review.scoped_fit import Fit, ti, tn
from calibration.program_tracking.body_tracking.absolute_motion_curve import _exp, _log
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference
from calibration.program_tracking.body_tracking.anatomical_chart import JointChart, JointEnvelope
from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from calibration.program_tracking.body_tracking.programme_joint_limits import angles_joint
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.skate_physics import fk_state, batch_features
from experiments.skate_video_v1.body_skate_fit import Assembly
from experiments.skate_video_v1.deliver_body_skate_fit import bounds, display_mesh, panel_crop, projection_batch, put, tint
from experiments.skate_video_v1.inputs import PROGRAMME, atomic_json, atomic_npz, digest, emit, read
from experiments.skate_video_v1.programme_motion_followup import bind
from rinkspace.viewer3d.source import SourceVideo
from calibration.program_tracking.body_tracking.centroidal_physics_diagnostics import flight_position_operator

# The controls influence only collar, shoulder and elbow articulation. The
# left and right wrist positions are downstream. Pelvis and legs are upstream.
COCO=np.arange(5,11)
FRAME_RANGE=(81.26,81.56)
MAIN_REGION=(81.36,81.48)

# Approximate native centres read from raw source frames, in the order
# L shoulder / R shoulder / L elbow / R elbow / L wrist / R wrist.
# Wrist is the sleeve-to-hand joint, not the fingertips.
WITNESSES={
    81.36: [[1020,424],[1073,372],[976,472],[1081,319],[923,516],[1090,258]],
    81.38: [[991,434],[1058,378],[930,482],[1092,320],[831,517],[1125,257]],
    81.40: [[964,434],[1053,375],[877,466],[1106,318],[774,487],[1160,251]],
    81.42: [[950,430],[1047,374],[859,446],[1115,319],[750,462],[1172,264]],
    81.44: [[932,428],[1030,379],[826,439],[1100,329],[713,443],[1180,273]],
    81.46: [[919,427],[1015,386],[824,427],[1101,337],[731,427],[1180,288]],
    81.48: [[913,422],[992,393],[829,422],[1086,350],[762,414],[1170,305]],
    81.50: [[905,424],[968,400],[851,410],[1061,367],[809,406],[1150,322]],
}
# These rows are reserved for comparison, including a frame omitted by the
# 30 Hz sampling. They are not injected as extra fit points or used to pick weights.
WITNESS_TIMES=np.array(list(WITNESSES))
FIT_HOLD=np.array([81.40,81.42,81.46,81.50])


def robust(x):
    return torch.sqrt(1.+x.square())-1.


def pose_world(ref,frames,clock):
    s=ref.at(clock.ravel())
    coco=ref.body.forward_rotations(s["local_R"],hand_local_R=s.get("hand_local_R"))[0]
    world=s["P"][:,None]+np.einsum("nij,nkj->nki",s["Q"],coco)
    return world.reshape(len(frames),5,17,3)


def joint_guard(controls, frames, baseline):
    ids=np.array([2,3])
    rest=controls["anatomical_rest_R"][ids]
    basis=controls["anatomical_basis"][ids]
    chart=JointChart(rest,basis,controls["anatomical_reference_twist"][:,ids])
    import json
    cfg=json.loads(str(controls["anatomical_policy_json"]))
    pars=cfg["envelopes"]["elbow"]
    envelope=JointEnvelope(*np.deg2rad(pars["flex_degrees"]),np.deg2rad(pars["flex_grid_degrees"]),
        np.deg2rad(pars["swing_axes_degrees"]),*np.deg2rad(pars["scale_degrees"]),cfg["provenance"])
    base=chart.evaluate(baseline[:,17:19],sample_indices=frames)
    tolerance=np.stack([envelope.residual(angles_joint(base,i)) for i in range(2)],axis=1)
    B=tn(basis)
    left=B.transpose(-1,-2)@tn(rest).transpose(-1,-2)
    anchor=tn(chart.reference_twist[frames])
    lo,hi=np.deg2rad(pars["flex_degrees"])
    grid=tn(np.deg2rad(pars["flex_grid_degrees"]))
    axes=tn(np.deg2rad(pars["swing_axes_degrees"]))
    scale_flex,scale_swing=np.deg2rad(pars["scale_degrees"])
    allowed=tn(tolerance)
    def loss(local):
        M=left@local[:,17:19]@B
        twist=torch.atan2(M[...,2,1]-M[...,1,2],M[...,1,1]+M[...,2,2])
        twist=anchor+torch.remainder(twist-anchor+torch.pi,2*torch.pi)-torch.pi
        cross=torch.stack([-M[...,2,0],M[...,1,0]],dim=-1)
        dist=torch.linalg.vector_norm(cross,dim=-1)
        swing=torch.atan2(dist,M[...,0,0])
        vector=cross*(swing/dist.clamp_min(1.e-7))[...,None]
        x=twist.clamp(grid[0],grid[-1]).contiguous()
        k=torch.searchsorted(grid,x).clamp(1,len(grid)-1)
        f=(x-grid[k-1])/(grid[k]-grid[k-1])
        radii=axes[k-1]+f[...,None]*(axes[k]-axes[k-1])
        n=torch.linalg.vector_norm(vector/radii,dim=-1).clamp_min(1.e-6)
        resid=torch.relu(torch.stack(((lo-twist)/scale_flex,(twist-hi)/scale_flex,
                                 (swing-swing/n)/scale_swing),dim=-1))
        return .8*resid.square().mean()+2.5*torch.relu(resid-allowed).square().mean()
    def audit(local):
        angle=chart.evaluate(local[:,17:19],sample_indices=frames)
        e=np.stack([envelope.residual(angles_joint(angle,i)) for i in range(2)],axis=1)
        return dict(twist_degrees=np.rad2deg(angle.twist).tolist(),max_violation=e.max(axis=-1).tolist(),
                    outside_per_arm=np.sum(e.max(axis=-1)>0.,axis=0).tolist())
    return loss,audit


def get_targets(prepared, frames, baseline_uv, pva, witness_pts, global_pts):
    with np.load(prepared/"observations.npz",allow_pickle=False) as f:
        np.testing.assert_array_equal(f["pts"],global_pts)
        body=f["uv"][frames]
        weights=f["sigma_px"][frames]
        model=f["model_joint_ids"][frames]
        mask=f["use"][frames]&np.isin(model,COCO)&~np.isin(global_pts[frames],witness_pts)[:,None]
    rows, k=np.where(mask)
    j=model[rows,k]
    image=body[rows,k]
    sigma=np.maximum(weights[rows,k],8.)
    frame_pts=global_pts[frames]
    tau=float(1/12800)
    s=pva["sample_pts"]
    selected=np.where((s[:-1]>=frame_pts[0])&(s[1:]<=frame_pts[-1]))[0]
    ii=np.searchsorted(frame_pts,s[selected])
    jj=np.searchsorted(frame_pts,s[selected+1])
    np.testing.assert_array_equal(frame_pts[ii],s[selected])
    np.testing.assert_array_equal(frame_pts[jj],s[selected+1])
    uv=pva["source_image_keypoints"][:,COCO]
    flags=pva["source_keypoints_valid"][:,COCO]
    conf=np.clip(pva["heatmap_confidence"][:,COCO],0.,1.)
    v=pva["velocity_world_mps"][selected][:,COCO]
    # Source clock differs by ±6.67 ms from the network input grid. Use the
    # nominal 30 Hz derivative on the same sampled frames for both terms,
    # keeping the actual programme derivative for all physical diagnostics.
    valid=flags[selected]&flags[selected+1]&np.isfinite(v).all(-1)
    valid&=(conf[selected]>.20)&(conf[selected+1]>.20)
    # Identity compatibility is checked at the shoulder with a broad image
    # radius. Do not discard the target just because the failed wrists differ.
    align=np.linalg.norm(uv[selected,:2]-baseline_uv[ii][:,5:7],axis=-1)<110.
    align_end=np.linalg.norm(uv[selected+1,:2]-baseline_uv[jj][:,5:7],axis=-1)<110.
    same=align&align_end
    valid&=np.tile(same,(1,3))
    valid&=~pva["box_inferred"][selected,None]&~pva["box_inferred"][selected+1,None]
    a,b=np.where(valid)
    return (rows,j,image,sigma),(ii[a],jj[a],b,v[a,b]),dict(
        velocity_row_times=(s[selected]*tau).tolist(),admitted_arm_components=int(valid.sum()),
        possible_arm_components=int(valid.size),body_fit_points=len(j),joint_ids=COCO.tolist())


def persist(controls,curve,path,parent,report):
    saved=dict(controls)
    saved["absolute_position_controls"]=curve.position_controls.copy()
    saved["absolute_rotation_controls"]=curve.rotation_controls.copy()
    saved["absolute_rotation_edge_reference_quaternion"]=curve.edge_reference_quaternion.copy()
    saved["x"]=controls["x"].copy()
    n=len(curve.position_controls)
    x=saved["x"][17:17+n*69].reshape(n,69)
    changed=np.any(curve.rotation_controls!=controls["absolute_rotation_controls"],axis=(1,2,3))
    rotations=_log(controls["absolute_seed_rotation_controls"][changed].swapaxes(-1,-2)@curve.rotation_controls[changed])
    x[changed,3:]=rotations.reshape(-1,66)
    np.testing.assert_allclose(controls["absolute_seed_rotation_controls"]@_exp(x[:,3:].reshape(n,22,3)),
                               curve.rotation_controls,atol=1e-8,rtol=0.)
    saved.update(scoped_repair_parent_sha256=np.array(digest(parent)),scoped_repair_report=np.array(str(report)),
                 historical_solver_diagnostics_apply_to_parent=np.array(True))
    atomic_npz(path,**saved)


def com_torch(s,mass,rows):
    i=2+5*rows
    joints=s[5][i][:,mass.ids]*mass.binding.ratio
    R=s[1][i,None]@s[3][i][:,mass.ids]
    centre=joints+torch.einsum("nkij,kj->nki",R,tn(mass.offsets[:mass.count]))
    boot=s[7][i]*mass.binding.ratio+torch.einsum("nkij,kj->nki",s[6][i],tn(mass.offsets[mass.count:]))
    return torch.einsum("k,nkc->nc",tn(mass.mass),torch.cat((centre,boot),dim=1))


def review_images(args,ref,cam,variants,frames,witness_idx,titles=None):
    vertices,faces=display_mesh(ref.binding.surface)
    assemblies=[Assembly(r.binding) for r in variants]
    clock=ref.times[:,None]+cam.readout_times(np.arange(len(ref.times)))[:,None]*np.linspace(-.5,.5,5)
    problem=SimpleNamespace(times=clock,camera=cam)
    frames=frames[witness_idx]
    uv=[projection_batch(problem,r,frames,a,vertices) for r,a in zip(variants,assemblies,strict=True)]
    video=read(PROGRAMME/"rebuild_scope.json")["source"]
    source=SourceVideo(video["registered_path"],(1,12800))
    data=[]
    for i,f in enumerate(frames):
        stamp=ref.p["pts"][f]
        image=cv2.cvtColor(source.read(stamp),cv2.COLOR_RGB2BGR)
        rr=[image]
        for (b,s),a in zip(uv,assemblies,strict=True):
            img=image.copy()
            tint(img,b[i],faces,(130,225,242),.20)
            for j,color in enumerate(((96,234,126),(239,174,94))):
                tint(img,s[i,j],a.faces,color,.25)
            rr.append(img)
        crop=bounds([b[i] for b,s in uv]+[s[i,j] for b,s in uv for j in range(2)],17,image.shape)
        row=np.full((292,len(rr)*362,3),20,np.uint8)
        for j,(img,title) in enumerate(zip(rr,(f"Source {stamp/12800:.3f}s",*titles),strict=True)):
            p=row[:,j*362:(j+1)*362]
            panel_crop(p[26:],img,crop)
            put(p,title,10,20,.57)
        data.append(row)
    source.close()
    for i in range(0,len(data),4):
        cv2.imwrite(str(args.output/f"entry_reprojection_{i//4+1}.jpg"),np.vstack(data[i:i+4]),[cv2.IMWRITE_JPEG_QUALITY,95])


def main(args):
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    ref,cam,info,doc,controls=bind(args.prepared,args.evidence,args.parameter)
    p=ref.p["pts"]
    secs=p/12800.
    assert info["source_time_base"]==[1,12800] or tuple(info["source_time_base"])==(1,12800)
    frames=np.flatnonzero((secs>=FRAME_RANGE[0]-.00001)&(secs<=FRAME_RANGE[1]+.00001))
    assert (ref.p["shot_ids"][frames]=="s05").all()
    curve=ref.absolute_curve
    centre=(ref.times[frames[0]],ref.times[frames[-1]])
    active=np.flatnonzero((curve.knots[:-4]>=centre[0]-.035)&(curve.knots[4:]<=centre[1]+.035))
    if len(active)<2:
        raise ValueError("No eligible spline control support")
    tracks=(16,17,18,19) if args.kinematic else (13,14,16,17,18,19)
    solver=Fit(ref,cam,curve,frames,tracks,active,fit_position=False)
    query=solver.clock.reshape(len(frames),5)
    pva={k:v for k,v in np.load(args.pva,allow_pickle=False).items()}
    uv0=moving_projection(cam,pose_world(ref,frames,query),frames)["uv"]
    witnesses=np.asarray(list(WITNESSES.values()),float)
    wpts=np.rint(WITNESS_TIMES*12800).astype(int)
    withhold=np.rint(FIT_HOLD*12800).astype(int)
    held=np.isin(wpts,withhold)
    witness_rows=np.searchsorted(p[frames],wpts)
    np.testing.assert_array_equal(p[frames][witness_rows],wpts)
    (fi,ji,images,sigma),(left,right,joint,target),meta=get_targets(args.prepared,frames,uv0,pva,withhold,p)
    if not len(left):
        raise ValueError("No PVA velocities available at flying entry")
    if args.variant_grid:
        s=pva["sample_pts"]
        valid_interval=(s>=p[frames][0])&(s<=p[frames][-1])
        sam=np.flatnonzero(valid_interval)
        native_row=np.searchsorted(p[frames],s[sam])
        np.testing.assert_array_equal(p[frames][native_row],s[sam])
        pva_xy=pva["source_image_keypoints"][sam][:,COCO]
        pva_conf=np.clip(pva["heatmap_confidence"][sam][:,COCO],0.,1.)
        source_valid=pva["source_keypoints_valid"][sam][:,COCO]&(~pva["box_inferred"][sam,None])
        source_valid&=pva_conf>.28
        identity=np.linalg.norm(pva_xy[:,:2]-uv0[native_row,5:7],axis=-1)<115.
        source_valid&=np.tile(identity,(1,3))
        yr,yj=np.where(source_valid)
        pva_uv_mask=(ti(native_row[yr]),ti(COCO[yj]))
        pva_uv_target=tn(pva_xy[yr,yj])
        pva_uv_sigma=tn(np.maximum(11.,16./np.sqrt(pva_conf[yr,yj])))
        # Torso translation cancels in relative targets. Ankles and ice
        # evidence are not modified in this upper-body experiment.
        ts=np.flatnonzero(valid_interval[:-1]&valid_interval[1:])
        pr=pva["velocity_world_mps"]
        re=pr[ts][:,COCO]-pr[ts][:,[11,12]].mean(1)[:,None]
        keep=source_valid[:-1]&source_valid[1:]
        vi,vj=np.where(keep)
        vr=ti(native_row[vi+1]);vl=ti(native_row[vi]);vj_t=ti(COCO[vj]);vt=tn(re[vi,vj])
        ts=np.flatnonzero(valid_interval[:-2]&valid_interval[1:-1]&valid_interval[2:])
        ar=pva["acceleration_world_mps2"]
        arel=ar[ts][:,COCO]-ar[ts][:,[11,12]].mean(1)[:,None]
        keepa=source_valid[:-2]&source_valid[1:-1]&source_valid[2:]
        ai,aj=np.where(keepa)
        al,am,ah,aj_t=map(ti,(native_row[ai],native_row[ai+1],native_row[ai+2],COCO[aj]))
        at=tn(arel[ai,aj])
        meta.update(pva_image_points=len(yr),pva_relative_velocity_rows=len(vi),pva_relative_acceleration_rows=len(ai))
    images=tn(images)
    sigma=tn(sigma)
    ij=(ti(fi),ti(ji))
    l,r,j=map(ti,(left,right,joint))
    v=tn(target)
    ratio=ref.binding.ratio
    state0=ref.at(ref.times[frames])
    safe,audit_anatomy=joint_guard(controls,frames,state0["local_R"])
    boundary=np.isin(p[frames],p[frames[[0,1,-2,-1]]])
    uv_anchor=tn(uv0[boundary][:,COCO])
    boundary_idx=ti(np.flatnonzero(boundary))
    axes=ti(np.asarray(tracks)-1)
    params=tn(state0["local_R"][:,np.asarray(tracks)-1])
    init=solver.parameter.detach().clone()
    histories={}
    mass=ContinuousMassGeometry(ref)
    flight=np.flatnonzero((secs[frames]>=MAIN_REGION[0]-1e-7)&(secs[frames]<=MAIN_REGION[1]+1e-7))
    flight_idx=ti(flight)
    gt=np.array([0.,0.,-9.81])
    op=flight_position_operator(ref.times[frames][flight],gt)
    ballistic=tn(op["operator"].toarray() if hasattr(op["operator"],"toarray") else op["operator"])
    flight_gravity_position=tn(op["gravity_position_m"])
    # The rigid-skate and shaped body masses match the saved replay exactly.
    with torch.no_grad():
        reference_com=com_torch(solver.state(),mass,flight_idx).cpu().numpy()
    pose0=fk_state(ref,ref.at(ref.times[frames][flight]),np.zeros((len(flight),69)))
    np.testing.assert_allclose(reference_com,mass.com(mass.from_pose(pose0)[0]),atol=3.e-6,rtol=0.)
    emit("entry_refit_begin",source_pts=p[frames[[0,-1]]].tolist(),controls=len(active),output=str(args.output),
        source_hz=50,pva_target=str(args.pva),pva_sha256=digest(args.pva),flight_term_enabled=args.flight,**meta)
    def objective(s, variant):
        world=solver.coco(s)
        mid=world[2::5]
        uv=solver.project(world)
        j_arm=s[2][2::5][:,axes]
        term=dict(image=10.*robust((uv[ij]-images)/sigma[:,None]).mean(),
                  boundary=1.7*robust((uv[boundary_idx,5:11]-uv_anchor)/9.).mean(),
                  pose=.20*(j_arm-params).square().mean()+.16*solver.parameter.square().mean(),
                  angular_accel=.45*robust(torch.diff(j_arm,n=2,dim=0)/.25).mean(),
                  elbow_envelope=safe(s[2][2::5])*(3.5 if args.kinematic else 1.))
        if args.kinematic:
            prev=j_arm[:-1].transpose(-1,-2)@j_arm[1:]
            # Smooth the changes relative to the already moving source fit.
            # A signed centre component can follow the real sweep. The
            # robust penalty discourages sudden large local cancellations.
            cos=((prev.diagonal(dim1=-2,dim2=-1).sum(-1)-1.)/2).clamp(-.999999,.999999)
            angle=torch.acos(cos)
            term["joint_step_guard"]=1.45*robust(torch.relu(angle-.39)/.22).mean()
        if args.flight:
            com=com_torch(s,mass,flight_idx)
            path=ballistic@(com-flight_gravity_position)
            acceleration=torch.diff(com,n=2,dim=0)/.02**2-tn(gt)
            term["flight_com_ballistic"]=robust(path/.015).mean()*1.6+robust(acceleration/40.).mean()*.85
        if args.variant_grid:
            if variant!="pixels_only":
                term["pva_image"]=3.8*robust((uv[pva_uv_mask]-pva_uv_target)/pva_uv_sigma[:,None]).mean()
            centred=mid-mid[:,[11,12]].mean(dim=1,keepdim=True)
            if variant in ("pva_relative_v","pva_relative_va"):
                rel=(centred[vr,vj_t]-centred[vl,vj_t])*ratio*30.
                term["pva_relative_velocity"]=.90*robust((rel-vt)/2.).mean()
            if variant=="pva_relative_va":
                relative_a=(centred[ah,aj_t]-2*centred[am,aj_t]+centred[al,aj_t])*ratio*30.**2
                term["pva_relative_acceleration"]=.35*robust((relative_a-at)/30.).mean()
        elif variant=="pixels_pva":
            speed=(mid[r,5+j]-mid[l,5+j])*ratio*30.
            term["pva_velocity"]=.90*robust((speed-v)/2.0).mean()
        return term
    keys=("pixels_only","pva_2d","pva_relative_v","pva_relative_va") if args.variant_grid else ("pixels_only","pixels_pva")
    if args.kinematic:
        keys=("pva_2d","pva_relative_va")
    for variant in keys:
        solver.parameter.data.copy_(init)
        current=solver.run(lambda state:objective(state,variant),args.steps,variant)
        persist(controls,current,args.output/f"{variant}.npz",args.parameter,args.output/"entry_assessment.json")
        histories[variant]=current

    r_ref=[ref]
    stats={}
    outputs=dict(source_pts=p[frames],witness_pts=wpts,witness_uv=witnesses,witness_held=held)
    steel0=None
    for label,now in (("current",curve),*histories.items()):
        full=absolute_reference(ref,now)
        if label!="current":
            r_ref.append(full)
        W=pose_world(full,frames,query)
        uv=moving_projection(cam,W,frames)["uv"]
        s=full.at(ref.times[frames])
        pose=fk_state(full,s,np.zeros((len(frames),69)))
        centres,rotations=mass.from_pose(pose)
        com=mass.com(centres)
        e=np.linalg.norm(uv[witness_rows][:,COCO]-witnesses,axis=-1)
        both=(secs[frames]>=MAIN_REGION[0]-.00001)&(secs[frames]<=MAIN_REGION[1]+.00001)
        steel=batch_features(full.binding,pose["R"],pose["O"])
        height=steel["height"][...,steel["eligible"]].min(-1)
        if steel0 is None:
            steel0=height.copy()
        else:
            np.testing.assert_allclose(height,steel0,rtol=0.,atol=1.e-9)
        ca=(com[:-2]-2*com[1:-1]+com[2:])/.02**2
        acc=ca[both[:-2]&both[1:-1]&both[2:]]
        world=W[:,2]*ratio
        vshape=(world[right,5+joint]-world[left,5+joint])*30.
        wrist_rotation=s["local_R"][:,np.asarray(tracks)-1]
        step=wrist_rotation[:-1].swapaxes(-1,-2)@wrist_rotation[1:]
        step=np.rad2deg(Rotation.from_matrix(step.reshape(-1,3,3)).magnitude()).reshape(len(frames)-1,-1)
        gcomp=com[flight]-op["gravity_position_m"]
        ballistic_error=op["operator"]@gcomp
        momentum,_,_=mass.H(ref.times[frames][flight],centres[flight],rotations[flight])
        row=dict(held_raw_native_px=metric(e[held]),all_source_witness_px=metric(e),
                 ballistic_distance_mm=metric(np.linalg.norm(ballistic_error,axis=-1)*1000),
                 flight_angular_impulse_m2ps=metric(np.linalg.norm(np.diff(momentum,axis=0),axis=-1)),
                 held_per_frame={str(WITNESS_TIMES[k]):metric(e[k]) for k in np.flatnonzero(held)},
                 flight_com_gravity_residual_mps2=metric(np.linalg.norm(acc-[0,0,-9.81],axis=-1)),
                 angular_step_degrees=metric(step),elbow=audit_anatomy(s["local_R"]),
                 pva_velocity_residual_mps=metric(np.linalg.norm(vshape-target,axis=-1)),
                 shoulder_world_shift_m=metric(np.linalg.norm((W[:,2,5:7]-pose_world(ref,frames,query)[:,2,5:7])*ratio,axis=-1)))
        stats[label]=row
        outputs.update({label+"_uv":uv,label+"_com":com,label+"_held_errors":e[held]})
    atomic_npz(args.output/"entry_values.npz",**outputs)
    atomic_json(args.output/"entry_assessment.json",dict(
        source_seconds=[float(secs[frames[0]]),float(secs[frames[-1]])],camera_sha256=info["state_sha256"],
        baseline_sha256=digest(args.parameter),tracks=tracks,stats=stats,flight_term_enabled=args.flight,
        changed_control_source_seconds=[float(curve.knots[active[0]]+secs[0]-ref.times[0]),float(curve.knots[active[-1]+4]+secs[0]-ref.times[0])],
        manual_witness="native raw-source joint-center estimates by Codex, uncertainty 10-25 px, held rows excluded from 50 Hz pixel objective; same source video and PVA can see them",
        fields_preserved="root, legs, skates, skin shape, camera and native PTS",
        **meta))
    key_titles={"pixels_only":"Pixels only","pixels_pva":"Pixels + PVA velocity","pva_2d":"PVA 2D",
                "pva_relative_v":"PVA 2D + rel. V","pva_relative_va":"PVA 2D + rel. V/A"}
    names=("Current fit",*(key_titles[key] for key in histories))
    review_images(args,ref,cam,r_ref,frames,witness_rows,names)
    emit("entry_refit_complete",metrics={k:{f:v[f] for f in ("held_raw_native_px","flight_com_gravity_residual_mps2","pva_velocity_residual_mps")} for k,v in stats.items()})


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__)
    for name in ("prepared","evidence","parameter","pva","output"):
        cli.add_argument("--"+name,type=Path,required=True)
    cli.add_argument("--steps",type=int,default=20)
    cli.add_argument("--flight",action="store_true")
    cli.add_argument("--variant-grid",action="store_true")
    cli.add_argument("--kinematic",action="store_true")
    args=cli.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    main(args)
