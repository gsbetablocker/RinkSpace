"""Prepare the existing G1 residual problem for correlation-aware G2.

This module compiles identities/weights/activity. Native owns all numerical
linearization and reduction. It does not infer new geometry or measurements.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .dataset import validate_current_noise_groups

def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def prepare_g2(g1_dir, *, motion_activity="level"):
    if motion_activity not in ("level","affine"):
        raise ValueError("unsupported motion activity")
    root=Path(g1_dir);problem=read(root/"problem.json")
    z=dict(np.load(root/problem["arrays_ref"],allow_pickle=False))
    camera=read(root/problem["camera_ref"])
    scene=read(root/problem["scene_ref"])
    controls=np.load(root/camera["controls_ref"],allow_pickle=False)
    source_groups=read(root/"noise_groups.json")
    # G1 already validated these current references; only W is newly consumed.
    ref=source_groups["weights_source"];wp=Path(problem["prepared_path"])/ref["path"]
    if hashlib.sha256(wp.read_bytes()).hexdigest()!=ref["sha256"]:raise ValueError("original W changed")
    original_weights=np.load(wp,allow_pickle=False)
    row_map={int(raw):i for i,raw in enumerate(z["rows"])}
    offsets=[0];weights_offsets=[0];members=[];weights=[];ranks=[];records=[]
    indices=np.full(len(z["rows"]),-1,np.int32)
    current_groups=[]
    for record in source_groups["groups"]:
        old=record["group"];rr=record["original_member_rows"]
        if rr!=record["selected_member_rows"] or not all(int(r) in row_map for r in rr):
            raise ValueError("partial correlation group is not eligible for G2: "+old["id"])
        mapped=np.array([row_map[int(r)] for r in rr],dtype=np.int64)
        w=original_weights[old["weight_start"]:old["weight_start"]+old["rank"]*2*len(rr)].copy()
        if hashlib.sha256(w.astype("<f8").tobytes()).hexdigest()!=old["matrix_sha256"]:
            raise ValueError("W group identity mismatch")
        group={**old,"member_start":len(members),"weight_start":len(weights)}
        current_groups.append(group)
        indices[mapped]=len(current_groups)-1
        members.extend(mapped.tolist());weights.extend(w.tolist());ranks.append(old["rank"])
        offsets.append(len(members));weights_offsets.append(len(weights))
        records.append({"id":old["id"],"origin":"original_correlated_group","source":record,
                        "rows":mapped.tolist(),"rank":old["rank"],"matrix_sha256":old["matrix_sha256"]})
    # Reuse the current group binding validator before adding the true independent rows.
    validate_current_noise_groups(current_groups,np.asarray(members,dtype="<i8"),
        np.asarray(weights,dtype="<f8"),{"pts":z["pts"],"split":z["split"],"noise_group_index":indices})
    for raw in source_groups["independent_rows"]:
        row=row_map[int(raw)]
        if indices[row]!=-1:raise ValueError("independent row overlaps a correlated group")
        xx,xy,yy=z["covariance"][row]
        covariance=np.array([[xx,xy],[xy,yy]])
        # Inverse Cholesky is a square-root precision with A C A^T = I.
        w=np.linalg.solve(np.linalg.cholesky(covariance),np.eye(2)).astype("<f8")
        members.append(row);weights.extend(w.ravel().tolist());ranks.append(2)
        offsets.append(len(members));weights_offsets.append(len(weights))
        records.append({"id":"independent/"+z["observation_ids"][row].decode(),"origin":"original_independent_covariance",
                        "rows":[row],"rank":2,"covariance":covariance.tolist(),
                        "matrix_sha256":hashlib.sha256(w.tobytes()).hexdigest()})
    if sorted(members)!=list(range(len(z["rows"]))):raise ValueError("noise partition is not complete and disjoint")
    # Keep original local-variable identity; fixed observations do not create station.
    local_ids=np.full(len(z["rows"]),-1,np.int32)
    old_locals=np.unique(z["local_index"][z["kind"]==1])
    if np.any(old_locals<0) or np.any(z["local_index"][z["kind"]==0]!=-1):
        raise ValueError("invalid existing station/fixed-point binding")
    remap={int(v):i for i,v in enumerate(old_locals)}
    for row in np.flatnonzero(z["kind"]==1):local_ids[row]=remap[int(z["local_index"][row])]
    # Six slow window directions in the existing control grid: level and slope
    # of pan/tilt/logf. This retains all 105 old coefficients and knot times.
    lo,hi=float(z["times"].min()),float(z["times"].max())
    centre=(lo+hi)/2;half=(hi-lo)/2
    if not half>0:raise ValueError("affine window activity requires multiple source times")
    knot=camera["first_control_time"]+np.arange(len(controls))*camera["control_dt"]
    directions=np.zeros((len(controls),3,6),np.float64)
    names=("pan","tilt","log_f")
    parameters=[]
    for ch,name in enumerate(names):
        directions[:,ch,ch]=.001
        directions[:,ch,ch+3]=.001*(knot-centre)/half
        parameters.append({"column":ch,"id":"motion/"+name+"/level","physical_scale":.001,
            "unit":"radian" if ch<2 else "log_ratio","control_basis":"constant","initial":0.})
    for ch,name in enumerate(names):
        parameters.append({"column":ch+3,"id":"motion/"+name+"/slope","physical_scale":.001,
            "unit":"radian" if ch<2 else "log_ratio","control_basis":"(control_time-window_centre)/window_half_duration","initial":0.})
    tested_parameters=parameters
    if motion_activity=="level":
        directions=directions[:,:,:3].copy();parameters=parameters[:3]
    return dict(problem=problem,arrays=z,scene=scene,camera=camera,controls=controls,
        directions=directions,local_ids=local_ids,local_scales=np.full(len(old_locals),.001),
        group_offsets=np.array(offsets,np.int32),members=np.array(members,np.int32),
        ranks=np.array(ranks,np.int32),weight_offsets=np.array(weights_offsets,np.int32),
        weights=np.array(weights,np.float64),group_records=records,
        activity={"parameters":parameters,"original_local_indices":old_locals.tolist(),
            "local_scale":.001,"local_coordinate":"original finite station in [0,1]",
            "world_gauge":"current native L=1; original axes/origin; geometry held fixed",
            "camera_gauge":"reference pose/axes/lever/optics/readout frozen; original reference control lies outside this window",
            "original_control_indices":camera["snapshot_origin"]["control_indices"],
            "window_centre_relative_seconds":centre,"window_half_duration_seconds":half,
            "activity_scope":motion_activity+" window directions on existing controls; no new knots or per-exposure free poses",
            "frozen_motion_directions":tested_parameters[3:] if motion_activity=="level" else [],
            "frozen":["motion slopes in level activity","all shared geometry","all advertisements","rig installation","Brown optics coefficients","readout time"],
            "loss":{"family":"rank_normalized_huber","delta":3.0,"threshold_unit":"whitened group RMS",
                    "reason":"explicit G2 linearization setting; not a fitted tolerance or geometry acceptance"},
            "motion_prior":{"enabled":False,"reason":"G2 audits data terms only; no uncalibrated prior weight introduced"},
            "parameter_update_executed":False})

def load_linearization_problem(g1_dir, g2_dir):
    """Consume G2's prepared numeric plan; no historical producer/weight rebuild."""
    root=Path(g1_dir);linear=Path(g2_dir)
    problem=read(root/"problem.json");camera=read(root/problem["camera_ref"])
    p=dict(problem=problem,camera=camera,scene=read(root/problem["scene_ref"]),
        arrays=dict(np.load(root/problem["arrays_ref"],allow_pickle=False)),
        controls=np.load(root/camera["controls_ref"],allow_pickle=False),
        activity=read(linear/"activity.json"),group_records=read(linear/"noise_groups.json"))
    p.update(dict(np.load(linear/"plan.npz",allow_pickle=False)))
    nl=len(p["local_scales"]);lo=np.full(nl,-np.inf);hi=np.full(nl,np.inf)
    for row,local in enumerate(p["local_ids"]):
        if local<0:continue
        support=p["arrays"]["support"][row]
        lo[local]=max(lo[local],support[0]);hi[local]=min(hi[local],support[1])
    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(lo>hi):
        raise ValueError("inconsistent original finite station supports")
    p["lower"]=lo;p["upper"]=hi
    return p
