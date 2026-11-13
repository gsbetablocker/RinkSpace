"""Score camera-7 image-derived motion against the original FS-Jump3D trial."""
from argparse import ArgumentParser
from fractions import Fraction
from pathlib import Path

import numpy as np

from .baseline_audit import metric
from calibration.program_tracking.body_tracking.fs_jump3d_validation import BODY_PAIRS, camera, project
from experiments.skate_video_v1.inputs import atomic_json, atomic_npz, digest, emit, read


def derivatives(points, time):
    dt=np.diff(time)
    v=np.diff(points,axis=0)/dt[:,None,None]
    a=np.diff(v,axis=0)/(0.5*(dt[:-1]+dt[1:]))[:,None,None]
    return v,a


def report_v(predict,truth,valid):
    dv=predict-truth
    magnitude=np.linalg.norm(truth,axis=-1)
    pred_mag=np.linalg.norm(predict,axis=-1)
    speed=(magnitude>=.7)&(pred_mag>=.1)&valid
    cos=(predict*truth).sum(-1)/np.maximum(pred_mag*magnitude,1e-10)
    return dict(error=metric(np.linalg.norm(dv,axis=-1)[valid]),native_speed=metric(magnitude[valid]),
                predicted_speed=metric(pred_mag[valid]),direction_error_degrees=metric(np.rad2deg(np.arccos(np.clip(cos[speed],-1.,1.)))),
                samples=int(valid.sum()))


def score(outputs,valid_v,valid_a,rotation):
    base,gt=outputs
    errors={}
    for tag,i,valid in (("velocity",0,valid_v),("acceleration",1,valid_a)):
        bv,gv=base[i],gt[i]
        dv=bv-gv
        rel=lambda x:x-x[:,[6,7]].mean(1)[:,None]
        joint_valid=valid&valid[:,[6,7]].all(1)[:,None]
        err=dict(absolute_vector=metric(np.linalg.norm(dv,axis=-1)[valid]),
                 mid_hip_relative_vector=metric(np.linalg.norm(rel(bv)-rel(gv),axis=-1)[joint_valid]),
                 optical_axis_absolute=metric(np.abs(np.einsum("i,nki->nk",rotation[2],dv))[valid]),
                 measured_joint_vector_magnitude=metric(np.linalg.norm(gv,axis=-1)[valid]))
        if i==0:
            err.update(report_v(bv,gv,valid))
        errors[tag]=err
    return errors


def score_main(args):
    root=args.trial
    refpath=root/"reference_original142.npz"
    basepath=root/"input_cam7/all_camera_joints.npz"
    calpath=root/"official_cam_param.json"
    cam=camera(read(calpath)["fs_camera_7"])
    with np.load(refpath) as ref,np.load(basepath) as base,np.load(args.images) as images:
        idx=np.searchsorted(base["source_pts"],images["sample_pts"])
        np.testing.assert_array_equal(base["source_pts"][idx],images["sample_pts"])
        np.testing.assert_array_equal(base["source_time_base"],images["source_time_base"])
        np.testing.assert_allclose(images["R_w2c"],np.broadcast_to(cam["R_w2c"],images["R_w2c"].shape),atol=1e-8)
        np.testing.assert_array_equal(ref["source_frame"]-1,np.arange(len(base["source_pts"])))
        time=images["sample_pts"]*float(Fraction(*images["source_time_base"]))
        np.testing.assert_allclose(time,ref["reference_source_pts"][idx]*float(Fraction(*ref["reference_time_base"])),rtol=0.,atol=1e-9)
        oi=[ref["names"].tolist().index(b) for _,b in BODY_PAIRS]
        ji=[base["joint_names"].tolist().index(a) for a,_ in BODY_PAIRS]
        true=ref["world_points_mm"][idx][:,oi]/1000.
        valid=ref["valid_mask"][idx][:,oi]&np.isfinite(true).all(-1)
        world=base["hand_nofoot_joints_camera_m"][idx][:,ji]@np.linalg.inv(cam["R_w2c"]).T+cam["C_world_m"]
        valid &= np.isfinite(world).all(-1)
        sample_pts=images["sample_pts"].copy()
    gt=derivatives(true,time)
    fit=derivatives(world,time)
    valid_v=valid[:-1]&valid[1:]
    valid_a=valid[:-2]&valid[1:-1]&valid[2:]
    fields=dict(source_pts=sample_pts,source_seconds=time,reference_world_m=true,baseline_world_m=world,
                reference_velocity=gt[0],reference_acceleration=gt[1],baseline_velocity=fit[0],
                baseline_acceleration=fit[1],velocity_valid=valid_v,acceleration_valid=valid_a,
                joints=np.array([a for a,_ in BODY_PAIRS]))
    comparisons=dict(exmr_hand_nofoot=score((fit,gt),valid_v,valid_a,cam["R_w2c"]))
    has_pva=False
    native_uv,front=project(true,cam)
    shape=(1920,1088)
    front&=(native_uv>=0).all(-1)&(native_uv<shape).all(-1)&valid
    if args.pva:
        with np.load(args.pva) as pva:
            np.testing.assert_array_equal(sample_pts,pva["sample_pts"])
            if str(pva["input_sha256"])!=digest(args.images):
                raise ValueError("PVA prediction was not made from this exact source input")
            pv=pva["velocity_world_mps"][:,5:17]
            pa=pva["acceleration_world_mps2"][:,5:17]
            flags=pva["source_keypoints_valid"][:,5:17]
            pix=pva["source_image_keypoints"][:,5:17]
            q=pva["heatmap_confidence"][:,5:17]
        matched_v=valid_v&np.isfinite(pv).all(-1)
        matched_a=valid_a&np.isfinite(pa).all(-1)
        err=np.linalg.norm(native_uv-pix,axis=-1)
        comparisons["pva"]=score(((pv,pa),gt),matched_v,matched_a,cam["R_w2c"])
        comparisons["pva"]["source_native_2d_px"]=metric(err[front&flags])
        comparisons["pva"]["visible_theia_pixels"]=int(front.sum())
        comparisons["pva"]["matched_theia_pixels"]=int((front&flags).sum())
        comparisons["exmr_on_same_rows"]=score((fit,gt),matched_v,matched_a,cam["R_w2c"])
        fields.update(pva_velocity=pv,pva_acceleration=pa,pva_2d=pix,pva_confidence=q)
        has_pva=True
    result=dict(dataset="FS-Jump3D Skater_A Axel_1",pva_used=has_pva,input_camera=7,
                source_exposures=len(time),sampling_hz=30,shared_joint_proxies=BODY_PAIRS,
                reference_sha256=digest(refpath),baseline_sha256=digest(basepath),camera_sha256=digest(calpath),
                images_sha256=digest(args.images),sample_pts=[int(sample_pts[0]),int(sample_pts[-1])],
                raw_source_exposures=300,raw_sampling_hz=60,source_seconds=time[[0,-1]].tolist(),
                interpretation="Theia multi-view virtual joint proxies, no registration, scale or time fit",
                timing="released synchronized 300 samples, take actual 30 Hz image PTS and matching Theia samples",
                comparisons=comparisons)
    atomic_npz(args.output.with_suffix(".npz"),**fields)
    atomic_json(args.output,result)
    emit("external_reference_audit",**result)


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__)
    for k in ("trial","images","output"):
        cli.add_argument("--"+k,type=Path,required=True)
    cli.add_argument("--pva",type=Path)
    score_main(cli.parse_args())
