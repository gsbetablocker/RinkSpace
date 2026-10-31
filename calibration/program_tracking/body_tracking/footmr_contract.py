#!/usr/bin/env python3
"""Weight-free RinkSpace S01 <-> FootMR candidate adapter.

This module never loads a neural checkpoint, never runs inference, and never
changes source observations, the scene camera, or pelvis state.  NumPy/SciPy
are sufficient. External model results remain UNREVIEWED candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

COCO23 = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle", "left_big_toe",
    "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel",
)
POSE_DOFS = ((1, 0), (2, 0), (4, 0), (5, 0), (7, 0), (8, 0), (3, 0), (6, 0), (9, 0))
KNEES = (4, 5)
ANKLES = (7, 8)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def so3_check(r, name):
    r = np.asarray(r)
    if r.shape[-2:] != (3, 3) or not np.isfinite(r).all():
        raise ValueError(f"{name}: finite (...,3,3) rotations required")
    if np.max(np.abs(r @ np.swapaxes(r, -1, -2) - np.eye(3))) > 2e-5:
        raise ValueError(f"{name}: non-orthogonal rotations")
    if np.max(np.abs(np.linalg.det(r) - 1)) > 2e-5:
        raise ValueError(f"{name}: determinant must be +1")


def to6(r):
    """PyTorch3D convention: first TWO ROWS, not first two columns."""
    r = np.asarray(r)
    return r[..., :2, :].reshape(*r.shape[:-2], 6)


def from6(x):
    x = np.asarray(x, dtype=float)
    if x.shape[-1] != 6 or not np.isfinite(x).all():
        raise ValueError("rotation6d: finite last dimension 6 required")
    a, b = x[..., :3], x[..., 3:]
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(n < 1e-8):
        raise ValueError("rotation6d: degenerate first row")
    a = a / n
    b = b - np.sum(a*b, axis=-1, keepdims=True)*a
    n = np.linalg.norm(b, axis=-1, keepdims=True)
    if np.any(n < 1e-8):
        raise ValueError("rotation6d: collinear rows")
    b = b/n
    return np.stack((a, b, np.cross(a, b)), axis=-2)


def interp_rot(times, rotations, query):
    """SO(3) interpolation, preserving every trailing joint axis."""
    r = np.asarray(rotations)
    shape = r.shape[1:-2]
    flat = r.reshape(len(times), -1, 3, 3)
    pieces = [Slerp(times, Rotation.from_matrix(flat[:, j]))(query).as_matrix()
              for j in range(flat.shape[1])]
    return np.stack(pieces, axis=1).reshape(len(query), *shape, 3, 3)


def linear(times, x, query):
    a = np.asarray(x)
    b = a.reshape(len(times), -1)
    return np.stack([np.interp(query, times, b[:, j]) for j in range(b.shape[1])], -1).reshape(len(query), *a.shape[1:])


def brackets(times, query):
    hi = np.searchsorted(times, query, side="left").clip(0, len(times)-1)
    lo = (hi-1).clip(0, len(times)-1)
    exact = np.isclose(times[hi], query, rtol=0, atol=1e-10)
    lo[exact] = hi[exact]
    den = times[hi]-times[lo]
    alpha = np.divide(query-times[lo], den, out=np.zeros_like(query), where=den>0)
    return lo, hi, alpha


def body_local_rotations(pose, correction):
    n = len(pose)
    local = np.broadcast_to(np.eye(3), (n,22,3,3)).copy()
    local[:,1:] = Rotation.from_rotvec(np.asarray(pose).reshape(-1,3)).as_matrix().reshape(n,21,3,3)
    for i,(j,axis) in enumerate(POSE_DOFS):
        v = np.zeros((n,3)); v[:,axis] = correction[:,i]
        local[:,j] = local[:,j] @ Rotation.from_rotvec(v).as_matrix()
    return local


def fk_rotation(local, parents):
    out = local.copy()
    for j in range(1,22):
        if not (0 <= parents[j] < j):
            raise ValueError("SMPL-X parents must be ordered before their children")
        out[:,j] = out[:,parents[j]] @ local[:,j]
    return out


def prepare(package, outdir, wholebody=None, fps=30.0, *, source_root=None, body_result=None, leg_result=None):
    outdir = Path(outdir)
    if source_root is None:
        package=Path(package);inp_path=package/"data/body/input.npz";params_path=package/"programme/body_s01_parameters_20261029.npz";compact_path=package/"data/body/compact_smplx.npz"
        a,b,compact=map(load_npz,(inp_path,params_path,compact_path))
    else:
        source_root=Path(source_root);inp_path=source_root/"input.npz";params_path=Path(body_result);compact_path=source_root/"compact_smplx.npz"
        a,state,compact=map(load_npz,(inp_path,params_path,compact_path));b=dict(pts=state["pts"],source_time_base=np.array([1,12800]),pose__delta_local_rad=state["delta_local_rad"],pose__Q_world=state["Q_world"])
    pts = np.asarray(a["pts"], np.int64)
    if not np.array_equal(pts,b["pts"]): raise ValueError("B state PTS mismatch")
    time_base = np.asarray(b["source_time_base"],np.int64)
    if tuple(time_base) != (1,12800): raise ValueError("This S01 adapter expects 1/12800 source time base")
    if len(pts)<2 or np.any(np.diff(pts)<=0): raise ValueError("PTS must strictly increase")
    if not np.isfinite(fps) or fps <= 0: raise ValueError("positive model sample rate required")
    t = (pts-pts[0])*float(time_base[0])/float(time_base[1])
    model_t = np.arange(int(np.floor(t[-1]*fps+1e-9))+1,dtype=float)/fps
    lo,hi,alpha = brackets(t,model_t)
    kp = np.zeros((len(t),23,3),float)
    kp[:,:17,:2] = a["source_uv"]
    kp[:,:17,2] = a["source_score"]*a["source_qualified"]
    source_valid = np.asarray(a["source_qualified"],bool) & np.isfinite(kp[:,:17]).all(-1)
    kp[:,:17] = np.where(source_valid[...,None],kp[:,:17],0.)
    wholebody_sha = None
    if wholebody is not None:
        w = load_npz(wholebody)
        if not np.array_equal(w["pts"],pts) or not np.array_equal(w["source_time_base"],time_base):
            raise ValueError("Wholebody must identify every source exposure by exact original PTS")
        if tuple(w["keypoint_names"].tolist()) != COCO23:
            raise ValueError("Wholebody keypoint order/semantics must be explicit COCO23")
        if str(w["pixel_space"].item()) != "original_full_image":
            raise ValueError("Wholebody pixels must be mapped back to original full image")
        kp = np.asarray(w["kp2d_coco23"],float)
        if kp.shape != (len(t),23,3): raise ValueError("Wholebody expected (N,23,3)")
        wholebody_sha = digest(wholebody)
    if not np.isfinite(kp).all() or np.any(kp[...,2]<0):
        raise ValueError("Keypoint coordinates and nonnegative scores must be finite; missing=0")
    box = np.asarray(a["source_bbox"],float)
    box_valid = np.isfinite(box).all(-1) & np.all(box[:,2:]>box[:,:2],axis=-1)
    if box_valid.sum()<2: raise ValueError("At least two usable source bboxes required")
    box = box.copy()
    box[~box_valid] = linear(t[box_valid],box[box_valid],t[~box_valid])
    extent = box[:,2:]-box[:,:2]
    if np.any(extent <= 0): raise ValueError("bbox must be original xyxy with positive extents")
    center = (box[:,:2]+box[:,2:])/2
    size = np.maximum(extent[:,1],extent[:,0]/0.75)*1.2
    bbox_xys = np.column_stack((center,size))
    K = np.zeros((len(t),3,3)); K[:,2,2]=1
    K[:,0,0]=a["fx"]; K[:,1,1]=a["fy"]; K[:,0,2]=a["cx"]; K[:,1,2]=a["cy"]
    if np.any(K[:,[0,1],[0,1]]<=0): raise ValueError("positive focal lengths required")
    local = body_local_rotations(a["pose_axis_angle"],b["pose__delta_local_rad"])
    leg_sha=None
    if leg_result is not None:
        from scipy.interpolate import BSpline
        from .footwear_geometry import extra_rotations
        leg=load_npz(leg_result)
        if not np.array_equal(leg["pts"],pts):raise ValueError("Joint leg candidate PTS mismatch")
        extra=BSpline.design_matrix(a["fit_time_seconds"],leg["correction_knots"],3,extrapolate=True)@leg["leg_controls"]
        for joint,values in extra_rotations(extra).items():local[:,joint]=local[:,joint]@Rotation.from_rotvec(values).as_matrix()
        leg_sha=digest(leg_result)
    fk = fk_rotation(local,compact["parents"])
    Rwc = np.asarray(a["camera_rotation"],float)
    Qworld = np.asarray(b["pose__Q_world"],float)
    so3_check(Rwc,"world_to_camera");so3_check(Qworld,"body_to_world")
    Rcb = Rwc @ Qworld
    global_cam = Rcb[:,None] @ fk[:,[4,5,7,8]]
    global_model = interp_rot(t,global_cam,model_t)
    # Derive the baseline relative ankle from the SAME interpolated global frames.
    base_local_model = np.swapaxes(global_model[:,:2],-1,-2) @ global_model[:,2:]
    kp_model = linear(t,kp,model_t)
    # Interpolation is a model input, never a new observed source point. A missing
    # endpoint remains missing: do not interpolate confident hallucinated feet.
    kp_model[...,2] = np.minimum(kp[lo,:,2],kp[hi,:,2])
    bb_model = linear(t,bbox_xys,model_t);K_model=linear(t,K,model_t)
    xy = 2*(kp_model[...,:2]-bb_model[:,None,:2])/bb_model[:,None,2:3]
    inside = np.all(np.abs(xy)<=1,axis=-1)
    obs = np.concatenate((xy,(kp_model[...,2]*inside)[...,None]),-1)[:,15:23]
    cliff = np.column_stack((bb_model[:,0]-K_model[:,0,2],bb_model[:,1]-K_model[:,1,2],bb_model[:,2]))/K_model[:,0,0,None]
    outdir.mkdir(parents=True,exist_ok=True)
    path=outdir/"footmr_input_contract.npz"
    np.savez_compressed(path, schema=np.array("rinkspace.footmr-contract.v1"),
        pts=pts,source_time_base=time_base,source_relative_seconds=t,
        model_relative_seconds=model_t,model_fps=np.array(fps),
        model_source_lower_index=lo,model_source_upper_index=hi,model_source_alpha=alpha,
        model_is_interpolated=lo!=hi,source_lower_body_inferred=a["lower_body_inferred"],
        source_bbox_inferred=~box_valid,model_bbox_inferred=(~box_valid[lo])|(~box_valid[hi]),
        foot_keypoint_names=np.array(COCO23[15:23]),global_rotation_names=np.array(["left_knee","right_knee","left_ankle","right_ankle"]),
        source_base_leg_result_sha256=np.array(leg_sha or ""),source_foot_keypoint_scores=kp[:,17:23,2],foot_obs_model=obs,f_cliffcam_model=cliff,global_rot6d_model=to6(global_model).reshape(len(model_t),24),
        global_knee_cam_model=global_model[:,:2],base_ankle_local_model=base_local_model,
        base_ankle_local_source=local[:,[7,8]],K_pinhole_model=K_model,
        candidate_only=np.array(True),contact_allowed_for_root=np.array(False))
    report=dict(schema="rinkspace.footmr-contract-report.v1",model_executed=False,
        original_frames=len(t),source_fps=float(1/np.median(np.diff(t))),model_frames=len(model_t),model_fps=fps,
        interpolation="uniform real-time grid; SO(3) rotations; 2D linear with min endpoint confidence; provenance retained",
        missing_new_six_foot_points=wholebody is None,
        source_missing_bbox_filled_as_model_input_only=int((~box_valid).sum()),
        confidence_semantics="raw nonnegative detector scores, may exceed 1; upstream visibility threshold 0.5 is not calibrated probability",
        foot_observations_confident_per_joint=(obs[...,2]>.5).sum(0).tolist(),
        candidate_export_does_not_extrapolate_tail_seconds=float(t[-1]-model_t[-1]),
        ready_for_model_inference=bool(wholebody is not None and np.any(obs[:,2:,2]>.5)),
        unresolved=["Validate native-to-model pinhole pixel mapping; direct original pixels use a candidate-only approximation",
            "Confirm physical left/right and anatomical heel/toes; confidence is not human approval",
            "Load FootMR weights and execute separate model runner; no model run performed here"]+(["Supply COCO23 wholebody observations"] if wholebody is None else []),
        source_base_leg_result_sha256=leg_sha,inputs={"body_input_sha256":digest(inp_path),"B_parameters_sha256":digest(params_path),"compact_sha256":digest(compact_path),"wholebody_sha256":wholebody_sha},
        output=path.name,output_sha256=digest(path),
        rotation_convention="active column-vector matrices; Rcam_joint=Rworld_to_cam @ Qbody_to_world @ Rbody_joint; 6D stores first two matrix rows",
        effect="No source, root, camera, geometry, or pose file overwritten")
    (outdir/"footmr_input_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
    return report


def check_input(path):
    z=load_npz(path);n=len(z["pts"]);m=len(z["model_relative_seconds"])
    for key,shape in [("foot_obs_model",(m,8,3)),("f_cliffcam_model",(m,3)),("global_rot6d_model",(m,24)),("global_knee_cam_model",(m,2,3,3)),("base_ankle_local_model",(m,2,3,3)),("base_ankle_local_source",(n,2,3,3))]:
        if z[key].shape!=shape or not np.isfinite(z[key]).all(): raise ValueError(f"{key}: expected finite {shape}")
    if tuple(z["foot_keypoint_names"].tolist())!=COCO23[15:23]:raise ValueError("foot keypoint order mismatch")
    if np.any(np.diff(z["pts"])<=0) or np.any(np.diff(z["model_relative_seconds"])<=0):raise ValueError("timestamps not strictly increasing")
    if not np.allclose(np.diff(z["model_relative_seconds"]),1/float(z["model_fps"]),rtol=0,atol=1e-9):raise ValueError("model input is not uniformly sampled in real time")
    for key in ["global_knee_cam_model","base_ankle_local_model","base_ankle_local_source"]:so3_check(z[key],key)
    so3_check(from6(z["global_rot6d_model"].reshape(m,4,6)),"6D input")
    return dict(contract_valid=True,model_executed=False,frames_source=n,frames_model=m,
        any_new_toe_heel_evidence=bool(np.any(z["foot_obs_model"][:,2:,2]>.5)),
        validation_scope="array shape, finite values, SO(3), semantics order, and time-grid only; no accuracy approval")


def export_candidate(input_path,prediction_path,output_path):
    check_input(input_path);z=load_npz(input_path);p=load_npz(prediction_path)
    t=z["model_relative_seconds"];m=len(t)
    if not np.array_equal(p["model_relative_seconds"],t):raise ValueError("prediction model time grid mismatch")
    if "input_contract_sha256" not in p or str(p["input_contract_sha256"].item())!=digest(input_path):raise ValueError("prediction must bind exact input contract SHA256")
    if "model_executed" not in p or not bool(p["model_executed"].item()):raise ValueError("prediction must explicitly identify an actual model run")
    pred=p["global_ankle_rot6d"]
    if pred.shape!=(m,2,6):raise ValueError("prediction expected (model_frames,2,6), physical left then right")
    Rpred=from6(pred)
    Rlocal=np.swapaxes(z["global_knee_cam_model"],-1,-2) @ Rpred
    # Existing CompactBody composes additional_rotations on the RIGHT, after B.
    delta=np.swapaxes(z["base_ankle_local_model"],-1,-2) @ Rlocal
    ts=z["source_relative_seconds"];valid=(ts>=t[0]-1e-10)&(ts<=t[-1]+1e-10)
    dsource=np.broadcast_to(np.eye(3),(len(ts),2,3,3)).copy()
    dsource[valid]=interp_rot(t,delta,ts[valid])
    aa=Rotation.from_matrix(dsource.reshape(-1,3,3)).as_rotvec().reshape(len(ts),2,3)
    np.savez_compressed(output_path,pts=z["pts"],source_time_base=z["source_time_base"],
        joint_ids=np.array([7,8]),additional_right_rotation_axis_angle=aa,
        candidate_supported_time=valid,source_lower_body_inferred=z["source_lower_body_inferred"],
        source_base_leg_result_sha256=z.get("source_base_leg_result_sha256",np.array("")),candidate_only=np.array(True),human_reviewed=np.zeros(len(ts),bool),
        source_foot_observation_supported=np.stack([(z["source_foot_keypoint_scores"][:,:3]>.5).any(1),(z["source_foot_keypoint_scores"][:,3:]>.5).any(1)],1) if "source_foot_keypoint_scores" in z else np.zeros((len(ts),2),bool),
        contact_allowed_for_root=np.zeros(len(ts),bool),input_contract_sha256=np.array(digest(input_path)),
        prediction_sha256=np.array(digest(prediction_path)))
    return dict(candidate_saved=str(output_path),original_pts_preserved=True,root_or_camera_changed=False,
                human_approval=False,contact_feedback=False,model_time_supported_frames=int(valid.sum()))


def selftest():
    # Noncommuting rotations catch row/column, transpose and composition mistakes.
    rng=np.random.default_rng(729)
    r=Rotation.random(100,random_state=rng).as_matrix()
    assert np.max(np.abs(from6(to6(r))-r))<1e-12
    Rwc,Q,Gk,La=Rotation.random(4,random_state=rng).as_matrix()
    knee=Rwc@Q@Gk;ankle=knee@La
    reconstructed=knee.T@from6(to6(ankle))
    assert np.max(np.abs(reconstructed-La))<1e-12
    correction=Rotation.from_rotvec([.13,-.21,.07]).as_matrix()
    pred=knee@La@correction
    recovered=La.T@knee.T@pred
    assert np.max(np.abs(recovered-correction))<1e-12
    rejected=False
    try:from6(np.zeros((1,6)))
    except ValueError:rejected=True
    assert rejected
    t=np.arange(1143)/50;tm=np.arange(int(np.floor(t[-1]*30))+1)/30
    lo,hi,a=brackets(t,tm)
    assert np.max(np.abs((1-a)*t[lo]+a*t[hi]-tm))<1e-12
    return dict(passed=True,model_executed=False,checks=["noncommuting rotation roundtrip","right-side correction composition","degenerate 6D rejection","50Hz-to-30Hz real-time mapping"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    s=sub.add_parser("prepare");s.add_argument("--package",required=True);s.add_argument("--out",required=True);s.add_argument("--wholebody");s.add_argument("--model-fps",type=float,default=30)
    s=sub.add_parser("check");s.add_argument("input")
    s=sub.add_parser("export-candidate");s.add_argument("--input",required=True);s.add_argument("--prediction",required=True);s.add_argument("--out",required=True)
    s=sub.add_parser("prepare-live");s.add_argument("--root",required=True);s.add_argument("--body-result",required=True);s.add_argument("--leg-result");s.add_argument("--out",required=True);s.add_argument("--wholebody");s.add_argument("--model-fps",type=float,default=30)
    sub.add_parser("selftest")
    a=p.parse_args()
    if a.action=="prepare-live":r=prepare(None,a.out,a.wholebody,a.model_fps,source_root=a.root,body_result=a.body_result,leg_result=a.leg_result)
    elif a.action=="prepare":r=prepare(a.package,a.out,a.wholebody,a.model_fps)
    elif a.action=="check":r=check_input(a.input)
    elif a.action=="export-candidate":r=export_candidate(a.input,a.prediction,a.out)
    else:r=selftest()
    print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
