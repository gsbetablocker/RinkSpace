"""Screen both final trajectories on all original exposure times."""
from pathlib import Path
from argparse import ArgumentParser
import json

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.skate_physics import batch_features, fk_state
from ..programme_motion_followup import bind
from .source_audit import projection_blocks, silhouette_score, static_assembly
from calibration.program_tracking.body_tracking.anatomical_chart import JointChart, JointEnvelope
from calibration.program_tracking.body_tracking.programme_joint_limits import JOINTS, NAMES, angles_joint
from calibration.program_tracking.body_tracking.programme_opening import standing_contact_rows
from experiments.skate_video_v1.inputs import atomic_json, atomic_npz, digest, read, emit



def stats(a):
    x=np.asarray(a).ravel()
    x=x[np.isfinite(x)]
    return dict(n=len(x),median=float(np.median(x)),p90=float(np.percentile(x,90)),
                p99=float(np.percentile(x,99)),maximum=float(np.max(x))) if len(x) else dict(n=0)


def rotation_steps(rot):
    adjacent=rot[:-1].swapaxes(-1,-2)@rot[1:]
    return np.degrees(Rotation.from_matrix(adjacent.reshape(-1,3,3)).magnitude()).reshape(-1,2)


def vec_length(x):
    return np.linalg.norm(x,axis=-1)


def evaluate(param, output, job, baseline_path, prepared_audit, prepared, evidence, opening_assessment):
    cv2.setNumThreads(1)
    base, cam, info, _, z0 = bind(prepared, evidence, baseline_path)
    new, _, new_info, _, z1 = bind(prepared, evidence, param)
    assert new_info["state_sha256"] == info["state_sha256"]
    src = load(prepared_audit)
    clock, shots = new.p["pts"], new.p["shot_ids"]
    np.testing.assert_array_equal(clock, src["pts"])
    np.testing.assert_array_equal(shots, src["shots"])
    source_seconds = clock / 12800.
    same = shots[1:] == shots[:-1]
    same_three = same[1:] & same[:-1]
    static, mats = static_assembly(new)
    ratio = new.binding.ratio
    source_contour=src["contour"].copy()
    source_contour[clock==2012928,1]=np.nan
    new_contour = np.full(source_contour.shape,np.nan)
    new_uv = np.empty(src["foot_uv"].shape)
    states = [fk_state(r, r.at(r.times), np.zeros((len(clock),69))) for r in (base,new)]
    world = [s["O"][:,:,None]+np.einsum("nfij,kj->nfki",s["R"],mats) for s in states]
    np.testing.assert_allclose(states[0]["native_world"] * ratio, src["world_joints"], rtol=0, atol=2.e-8)
    np.testing.assert_allclose(z0["anatomical_basis"], z1["anatomical_basis"], rtol=0, atol=1.e-12)

    for a in range(0,len(clock),64):
        k=np.arange(a,min(a+64,len(clock)))
        uv, vertices = projection_blocks(new, cam, k, static, mats, full_uv=True)
        new_uv[k]=uv
        for j,f in np.argwhere(np.isfinite(source_contour[k])):
            new_contour[k[j],f]=silhouette_score(static, vertices[j,f], src["source_region"][k[j],f])
        if a%2048==0:
            emit("full_source_projection", checked=a, exposures=len(clock))

    so3=np.stack([rotation_steps(s["R"]) for s in states])
    matstep=np.stack([1000*ratio*vec_length(np.diff(w[:,:,2:4],axis=0)).max(axis=2) for w in world])
    matdelta=np.stack([1000*ratio*vec_length(np.diff(w[:,:,2:4],axis=0,n=2)).max(axis=2) for w in world])
    height=np.stack([standing_contact_rows(batch_features(r.binding,s["R"],s["O"]),np.arange(len(clock)))["height_by_foot"]
                     for r,s in zip((base,new),states)])
    # Adjacent native source PTS are exactly 20 ms. Across-cut geometry is
    # evaluated too, but 2D displacement is only meaningful within one view.
    np.testing.assert_array_equal(np.diff(clock),np.full(len(clock)-1,256))
    motion=dict(same_shot_pairs=int(same.sum()), cut_pairs=int((~same).sum()),
                same_shot=[dict(rot_deg=stats(so3[k,same]), steel_endpoint_step_mm=stats(matstep[k,same]),
                                steel_endpoint_second_difference_mm=stats(matdelta[k,same_three]))
                           for k in range(2)],
                cut=[dict(source_pts=int(clock[j+1]),source_seconds=float(source_seconds[j+1]),
                          from_shot=str(shots[j]),to_shot=str(shots[j+1]),
                          rotation_deg=so3[:,j].tolist(),endpoint_step_mm=matstep[:,j].tolist())
                     for j in np.flatnonzero(~same)])
    improvement = new_contour - source_contour
    direction=np.diff(new_uv[:,:,:2],axis=2)[:,:,0]
    img_heading_change=np.degrees(np.arctan2(
        np.abs(direction[:-1,:,0]*direction[1:,:,1]-direction[:-1,:,1]*direction[1:,:,0]),(direction[:-1]*direction[1:]).sum(-1)))
    charts=JointChart(z1["anatomical_rest_R"], z1["anatomical_basis"], z1["anatomical_reference_twist"])
    anatomy=[charts.evaluate(s["local_R"][:,JOINTS-1]) for s in states]
    policy=json.loads(str(z1["anatomical_policy_json"]))
    envelope = {}
    for j, name in enumerate(NAMES):
        c=policy["envelopes"][name.rsplit("_",1)[1]]
        e=JointEnvelope(*np.deg2rad(c["flex_degrees"]),np.deg2rad(c["flex_grid_degrees"]),
                        np.deg2rad(c["swing_axes_degrees"]),*np.deg2rad(c["scale_degrees"]),policy["provenance"])
        residual=np.stack([e.residual(angles_joint(a,j)) for a in anatomy])
        envelope[name]=dict(swing_deg=[stats(np.rad2deg(a.swing[:,j])) for a in anatomy],
                            domain_peak=residual.max(axis=1).tolist(),
                            domain_rms=np.sqrt((residual**2).mean(axis=1)).tolist(),
                            worst_new_pts=int(clock[np.argmax(residual[1].max(axis=1))]))
    contours={str(shot):[dict(before=stats(source_contour[shots==shot,f]),
                             after=stats(new_contour[shots==shot,f]),
                             changed=stats(improvement[shots==shot,f]))
                        for f in range(2)] for shot in np.unique(shots)}
    event_measurements={}
    held={}
    selected_refits=read(job/"anatomy_repair/assessment.json")["windows"]
    replacement={row["window"]:row for row in selected_refits}
    ankle=read(job/"final_ankle/assessment.json")["windows"][0]
    replacement[ankle["window"]]=ankle
    ledgers={}
    for sub in ('shoe_direction','opening_shoes','checked_direction','additional_direction','free_foot_final'):
        for row in read(job/sub/"assessment.json")["windows"]:
            name=row["window"]
            if sub=="checked_direction" and name.startswith("172.80"):
                ledgers.pop("172.8–173.6 右自由脚",None)
            if sub=="free_foot_final":
                ledgers.pop("142.94 右脚伸展",None)
            ledgers[name]=replacement.get(name,row)
    for row in ledgers.values():
        label=row["window"]
        if row["state"]!="retained" or label.startswith("101.54"):
            continue
        foot=row["foot"]
        pts=np.array(row["measured_pts"])
        keys=np.searchsorted(clock,pts)
        col=slice(0,2) if row["kind"]=="sole" else slice(2,4)
        xy=np.asarray(row["measured_uv"])
        both=np.stack([src["foot_uv"][keys,foot,col],new_uv[keys,foot,col]])
        d=np.diff(both,axis=2)[:,:,0]
        t=np.diff(xy,axis=1)[:,0]
        z=vec_length(both-xy).mean(axis=2)
        cross=d[...,0]*t[...,1]-d[...,1]*t[...,0]
        err=np.degrees(np.arctan2(np.abs(cross),(d*t).sum(-1)))
        which=np.isin(pts,row["check_pts"])
        for p,flag,delta,angle in zip(pts,which,z.T,err.T):
            if flag:
                held[int(p),foot,row["kind"]]=dict(label=label,pts=int(p),foot=foot,
                    before_px=float(delta[0]),final_px=float(delta[1]),
                    before_deg=float(angle[0]),final_deg=float(angle[1]))
        event_measurements[label]=dict(measured_pts=pts.tolist(),foot=foot,
            endpoint_px=z.tolist(),direction_deg=err.tolist(),holdout=which.tolist())
    verify=list(held.values())
    opening=read(opening_assessment)
    lower={k:float(np.median(opening[k]["direct_held_legs_px"])) for k in ("old","new")}
    quiet=(source_seconds>=23.22)&(source_seconds<=24.74)
    incident=[]
    for f in range(2):
        for shot in np.unique(shots):
            members=np.flatnonzero(same & (shots[1:]==shot))
            for ids,key,a in ((members,"orientation_step",so3[:,:,f]),(members,"steel_endpoint_step",matstep[:,:,f])):
                top=ids[np.argmax(a[1,ids])]
                incident.append(dict(kind=key,foot=f,shot=str(shot),pts=int(clock[top+1]),
                    source_seconds=float(source_seconds[top+1]),before=float(a[0,top]),after=float(a[1,top])))
    result=dict(exposures=len(clock),skate_exposures=2*len(clock),source_pts=clock[[0,-1]].tolist(),
                source_time_base=[1,12800],camera_sha256=info["state_sha256"],parameter_sha256=digest(param),
                comparison_parameter_sha256=digest(baseline_path),continuity=motion,
                source_outline_stats_by_shot=contours,
                source_outline_all=[stats(source_contour),stats(new_contour)],
                source_outline_warning="existing tracker outlines can contain misidentifications; ranking only",
                opening=dict(maximum_min_steel_mm=(1000*height[:,quiet].min(2).max(1)).tolist(),
                             direct_development_legs_median_px=lower,
                             check_assessment=str(opening_assessment)),
                directions=dict(roles="within-episode development checks; neighboring exposures are correlated",
                                endpoint_px=[stats([x[key] for x in verify]) for key in ("before_px","final_px")],
                                angle_deg=[stats([x[key] for x in verify]) for key in ("before_deg","final_deg")],
                                rows=verify,episodes=event_measurements),
                anatomy=envelope,high_motion_events=incident,
                flying_entry=read(job/"spin_entry/assessment.json"),
                cut_centre_fit=read(job/"cut_position/assessment.json"),
                source_exclusion=dict(source_pts=[2012928], reason="157.26 right-skate outline is on the other shoe"),
                accuracy_accepted=False)
    atomic_npz(output/"source_review_values.npz",pts=clock,shots=shots,source_outline_use=np.isfinite(source_contour),
        old_foot_uv=src["foot_uv"],new_foot_uv=new_uv,old_contour_px=source_contour,new_contour_px=new_contour,
        rotation_step_deg=so3,steel_endpoint_step_mm=matstep,steel_endpoint_second_difference_mm=matdelta,
        image_direction_step_deg=np.where(same[:,None],img_heading_change,np.nan),steel_height_m=height)
    atomic_json(output/"review_audit.json",result)
    emit("full_motion_review_complete",exposures=len(clock),opening=result["opening"],
         source_angle_deg=result["directions"]["angle_deg"],same_shot=motion["same_shot"])


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("parameter", "output", "job", "baseline", "source-audit", "prepared", "evidence", "opening-assessment"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evaluate(args.parameter, args.output, args.job, args.baseline, args.source_audit,
             args.prepared, args.evidence, args.opening_assessment)
