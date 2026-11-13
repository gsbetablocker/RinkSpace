"""Compare a joint native programme candidate with the previous display."""
from argparse import ArgumentParser
from copy import copy
from pathlib import Path
import json
import os
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from scipy.spatial import cKDTree

from .body_skate_fit import Assembly, Problem, projection
from .deliver_body_skate_fit import archived_edge_check, open_collar_clearance, render_video, saved_ref
from .inputs import PROJECT, atomic_json, atomic_npz, digest, distribute, emit, read
from .programme_exmr_fit import CASES, statistics
from calibration.program_tracking.body_tracking.absolute_motion_curve import _log
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference, curve_from_packet
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.sequence_skate_state import shoe_curve_from_packet
from calibration.program_tracking.body_tracking.skate_checkpoint import validate_checkpoint_identity
from calibration.program_tracking.body_tracking.skate_physics import batch_features, fk_state
from calibration.program_tracking.body_tracking.skate_template import SkateParameters
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference


NAME = "programme_exmr_skates_1112"
AUDIT_NAME = "programme_skates_audit_cd1_1112"
CLIPS = {
    "first_spin_direction": (84.48, 85.16),
    "first_spin_support": (86.74, 92.78),
    "shank_and_collar": (109.94, 110.42),
    "s09_raised_right_skate": (150.72, 151.32),
}


def load_candidate(prepared, parameter):
    ref, _, _, _, _ = load_reference(PROJECT, prepared)
    controls = load(parameter)
    validate_checkpoint_identity(controls, ref.fixed_checkpoint_identity)
    ref = absolute_reference(ref, curve_from_packet(controls), root=prepared)
    ref.relative_skate_curve = shoe_curve_from_packet(controls)
    ref.binding.x = controls["x"][:17].copy()
    ref.binding._parameter_key = ref.binding._packet = None
    ref.binding.reference = ref
    return ref


def outline(problem, ref, use):
    result = np.full(use.shape, np.nan)
    assembly = Assembly(ref.binding)
    for foot in (0, 1):
        frames = np.flatnonzero(use[:, foot])
        for start in range(0, len(frames), 48):
            group = frames[start:start + 48]
            query = problem.times[group].ravel()
            state = fk_state(ref, ref.at(query), np.zeros((len(query),69)))
            w = assembly.points(state, foot, transforms=(state["R"],state["O"]))
            uv = projection(problem.camera, w, group)["uv"]
            for i, pixels in zip(group,uv,strict=True):
                try:
                    boundary = np.concatenate(assembly.raster(pixels))
                except ValueError:
                    # Record a finite distance to the projected model even for a grossly misplaced trial.
                    boundary = pixels
                source = distribute(problem.region[i,foot],96,closed=True)
                result[i,foot] = .5*(cKDTree(boundary).query(source)[0].mean()
                                    +cKDTree(source).query(boundary)[0].mean())
    return result


def geometry_profile(problem, ref, previous, job, output):
    """Evaluate both archived shape families at the same body and static mount."""
    archive = previous/"audit.json"
    dimensions = read(archive)["geometry"]
    roles = load(job/"whole_skate_roles.npz")
    use = roles["historical_development"] & problem.valid
    output.mkdir(exist_ok=True)
    rows = []
    for label, fraction in (("current",0.),("midpoint",.5),("earlier",1.)):
        trial = absolute_reference(ref,ref.absolute_curve)
        binding = trial.binding = copy(ref.binding)
        binding.reference = trial
        geometry = {k:(1-fraction)*v + fraction*dimensions["before"][k]
                    for k,v in dimensions["after"].items()}
        height = geometry.pop("upper_height_scale")
        binding.base_parameters = SkateParameters(**geometry)
        binding.x = binding.x.copy()
        binding.x[12:] = 0.
        binding.fixed_upper_height_log = float(np.log(height))
        binding._parameter_key = binding._packet = binding._collar_factor_spec = None
        R,t,_ = binding.parameters()
        fixed_R,fixed_t,_ = ref.binding.parameters()
        np.testing.assert_array_equal(R,fixed_R)
        np.testing.assert_array_equal(t,fixed_t)
        error = outline(problem,trial,use)
        rows.append(dict(name=label,parameters=dict(geometry,upper_height_scale=height),
                         silhouette_px=statistics(error[use]),
                         by_shot_and_foot={f"{shot}:{foot}":statistics(error[problem.i.shots==shot,foot])
                             for shot in np.unique(problem.i.shots) for foot in (0,1)},
                         archived_runner_px=archived_edge_check(problem,trial)[0],
                         cavity=open_collar_clearance(problem,trial)))
        emit("shared_geometry_source_profile",**{k:rows[-1][k] for k in ("name","silhouette_px","cavity")})
    atomic_json(output/"geometry_profile.json",dict(same_body_and_mount=True,
                published_parameter_source_sha256=digest(archive),role="development",
                state="fitted" if (output/"solve/candidate.npz").is_file() else "initialization",
                profile=rows,absolute_dimensions_identified=False))


def kinematics(ref, result):
    s = fk_state(ref, ref.at(ref.times), np.zeros((len(ref.times), 69)))
    replay = {}
    for field, key, scale in (("native22_world_mid","native_world",ref.binding.ratio),
                               ("skate_R_world_mid","R",1.),("skate_origin_world_mid","O",ref.binding.ratio)):
        replay[field] = float(np.max(np.abs(result[field]-s[key]*scale)))
        if replay[field] > 1.e-7:
            raise ValueError(f"Replay differs in {field}")
    geo = batch_features(ref.binding,s["R"],s["O"])
    runner = np.array([k=="runner" for k in geo["semantics"]])
    toe = np.array([k=="toe_pick" for k in geo["semantics"]])
    angles = np.rad2deg(np.linalg.norm(_log(s["R"][:-1].swapaxes(-1,-2)@s["R"][1:]),axis=-1))
    local_step = np.rad2deg(np.linalg.norm(_log(s["local_R"][:-1].swapaxes(-1,-2)@s["local_R"][1:]),axis=-1))
    heading = np.unwrap(np.arctan2(s["R"][:,:,1,0],s["R"][:,:,0,0]),axis=0)
    pelvis = np.unwrap(np.arctan2(s["Q"][:,1,0],s["Q"][:,0,0]))
    com = result["com_world_mid"]
    com_acc = np.full(len(com), np.nan)
    dt = np.diff(ref.times)
    v = np.diff(com,axis=0)/dt[:,None]
    com_acc[1:-1] = np.linalg.norm(np.diff(v,axis=0)/((dt[:-1]+dt[1:])*.5)[:,None],axis=-1)
    return dict(replay=replay, runner_mm=1000*geo["height"][...,runner].min(axis=-1),
                toe_mm=1000*geo["height"][...,toe].min(axis=-1), heading_deg=np.rad2deg(heading),
                root_heading_deg=np.rad2deg(pelvis), shoe_step_deg=np.vstack((np.full((1,2),np.nan),angles)),
                local_step_deg=np.vstack((np.full((1,21),np.nan),local_step)), com_acc_mps2=com_acc)


def collar_at_cases(ref, indices):
    times=ref.times[indices]
    state=fk_state(ref,ref.at(times),np.zeros((len(times),69)))
    collar=ref.binding.collar_metrics(state["native"],state["frames"],
                                     skate_relative_R=state["skate_relative_R"])
    return [dict(collar_section_gap_mm=[None if unknown else float(gap*1000*ref.binding.ratio)
                     for gap,unknown in zip(row,flag,strict=True)], collar_section_status=reason.tolist())
            for row,flag,reason in zip(collar["minimum_sample_gap_solver"],collar["unknown"],
                                       collar["unknown_reason"],strict=True)]


def episode_review(pts, poses, errors, role, first, last, foot, turn_sign=None):
    window = (pts >= round(first*12800)) & (pts <= round(last*12800))
    ids = np.flatnonzero(window)
    rows = []
    for pose, error in zip(poses, errors, strict=True):
        step = pose["shoe_step_deg"][ids[1:], foot]
        peak = int(np.argmax(step))
        row = dict(contour_px=statistics(error[window, foot]),
                   minimum_runner_mm=float(pose["runner_mm"][window, foot].min()),
                   maximum_runner_mm=float(pose["runner_mm"][window, foot].max()),
                   peak_so3_deg=float(step[peak]), peak_pts=int(pts[ids[peak + 1]]))
        if turn_sign is not None:
            heading_change = turn_sign * np.diff(pose["heading_deg"][ids, foot])
            row["opposite_heading_deg"] = float(-np.minimum(heading_change, 0.).sum())
        rows.append(row)
    return dict(foot=foot, source_pts=pts[ids[[0,-1]]].tolist(), exposures=len(ids),
                outline_covered=int(np.isfinite(errors[0][window,foot]).sum()),
                fit_outline_exposures=int(np.sum(role[window,foot]==1)), before=rows[0], after=rows[1])


def audit(problem, refs, previous, solve, job, prepared):
    pts=problem.i.pts
    requested=("source_pts","native22_world_mid","com_world_mid","com_projection_world_mid",
               "skate_R_world_mid","skate_origin_world_mid","COCO_uv","skate_contact_mode_mid",
               "sequence_contact_modes","sequence_contact_status")
    packets=[]
    for path in (previous/"fitted/result.npz",solve/"fitted/result.npz"):
        with np.load(path,allow_pickle=False) as data:
            packets.append({key:data[key] for key in requested if key in data})
    old,new=packets
    for r in (old,new):
        np.testing.assert_array_equal(r["source_pts"],pts)
    effective=read(solve/"run_summary.json").get("source_evidence",str(prepared/"source_evidence.json"))
    document=read(effective)
    programme=document.get("programme_audit_repair",document.get("programme_exmr",{}))
    roles=load(document["whole_skate_source"]["path"])
    mask=roles["historical_development"] & problem.valid
    special=np.searchsorted(pts,np.rint(np.array(CASES)*12800).astype(int))
    mask[special]|=roles["visible"][special]
    errors=[outline(problem,r,mask) for r in refs]
    poses=[kinematics(r,packet) for r,packet in zip(refs,(old,new),strict=True)]
    collars=[collar_at_cases(r,special) for r in refs]
    obs=load(problem.i.parent["body_observations"]["path"])
    idx=np.arange(len(pts))[:,None]
    body=[np.linalg.norm(r["COCO_uv"][idx,obs["model_joint_ids"]]-obs["uv"],axis=-1) for r in (old,new)]
    use=obs["use"] & np.isfinite(obs["uv"]).all(axis=-1)
    silhouette=roles["historical_development"] & np.isfinite(errors[0]) & np.isfinite(errors[1])
    spin=(pts>=round(81.62*12800))&(pts<=round(92.72*12800))
    spin_direction=document["whole_skate_source"]["support_shoe_phase"]["turn_direction_sign"]
    free=(pts>=round(92.2*12800))&(pts<=round(92.7*12800))
    cases={}
    for case,(sec,i) in enumerate(zip(CASES,special,strict=True)):
        cases[f"{sec:.2f}"]=[dict(
            contour_px=[float(v) if np.isfinite(v) else None for v in e[i]],
            runner_mm=k["runner_mm"][i].tolist(),toe_mm=k["toe_mm"][i].tolist(),
            shoe_step_deg=k["shoe_step_deg"][i].tolist(),
            ankle_step_deg=k["local_step_deg"][i,[6,7]].tolist(),
            knee_step_deg=k["local_step_deg"][i,[3,4]].tolist(),
            com_acc_mps2=float(k["com_acc_mps2"][i]),**c[case])
            for e,k,c in zip(errors,poses,collars,strict=True)]
    pair=errors[1]-errors[0]
    cuts=np.flatnonzero(problem.i.shots[1:]!=problem.i.shots[:-1])+1
    seconds=lambda p:int(round(p*12800))
    turn=np.searchsorted(pts,[seconds(v) for v in (84.64,84.76,84.82)])
    report=dict(source_pts=pts[[0,-1]].tolist(),exposures=len(pts),camera_sha256=problem.i.camera_sha,
                input_visual_sha256=digest(problem.visual/"whole_follow_evidence.npz"),
                old_controls_sha256=digest(previous/"controls.npz"),
                controls_sha256=digest(solve/"solve/candidate.npz"),
                output_sha256=digest(solve/"fitted/result.npz"),all_joint_replay=[k["replay"] for k in poses],
                geometry_preserved=refs[0].binding.packet()["parameters"]==refs[1].binding.packet()["parameters"],
                development_outline=[statistics(e[silhouette]) for e in errors],
                historical_improved=int((pair[silhouette]<0).sum()),historical_regressed=int((pair[silhouette]>0).sum()),
                body_px=[statistics(e[use]) for e in body],spin_body_px=[statistics(e[spin][use[spin]]) for e in body],
                spin_right_outline_px=[statistics(e[spin & silhouette[:,1],1]) for e in errors],
                spin_supported_heading_steps_deg=[np.diff(k["heading_deg"][turn,1]).tolist() for k in poses],
                spin_body_heading_steps_deg=[np.diff(k["root_heading_deg"][turn]).tolist() for k in poses],
                free_left_peak_deg=[float(np.max(k["shoe_step_deg"][free,0])) for k in poses],
                episodes=dict(first_spin_direction=episode_review(pts,poses,errors,roles["role"],84.54,85.00,1,spin_direction),
                              first_spin_right_support=episode_review(pts,poses,errors,roles["role"],81.62,92.72,1,spin_direction),
                              spin_exit_left_free=episode_review(pts,poses,errors,roles["role"],92.20,92.70,0)),
                cut_diagnostics=[dict(source_seconds=float(pts[i]/12800),
                    before_shoe_step_deg=poses[0]["shoe_step_deg"][i].tolist(),
                    after_shoe_step_deg=poses[1]["shoe_step_deg"][i].tolist(),
                    before_com_acc=float(poses[0]["com_acc_mps2"][i]),
                    after_com_acc=float(poses[1]["com_acc_mps2"][i])) for i in cuts],
                cavity=[open_collar_clearance(problem,r) for r in refs],
                collar_measurement="signed gap in a model collar section; grazing sections can include distant shank skin",
                historical_runner=[archived_edge_check(problem,r)[0] for r in refs],
                source_cases=cases,independent_final_check=0,absolute_3d_accuracy_accepted=False,
                source_roles=read(job/"whole_skate_role_receipt.json"))
    if programme.get("representation_revision"):
        report["representation_revision"]=programme["representation_revision"]
    if programme.get("collar_fit_revision"):
        report["collar_fit_revision"]=programme["collar_fit_revision"]
    if "programme_audit_repair" in document:
        report["committed_state_import"]=programme
        fit_metrics=read(solve/"final_metrics.json")
        report["physical_diagnostics"]=fit_metrics["physical_diagnostics"]
        report["primary_guard_result"]=fit_metrics["primary_guards"]
        from calibration.program_tracking.body_tracking.anatomical_chart import JointChart
        axes=load(solve/"anatomical_frames.npz")
        np.testing.assert_array_equal(axes["source_pts"],pts)
        chart=JointChart(axes["rest_R"],axes["anatomical_basis"],axes["reference_twist"])
        old_ref=refs[0]
        old_state=fk_state(old_ref,old_ref.at(old_ref.times[special]),np.zeros((len(special),69)))
        old_angles=chart.evaluate(old_state["local_R"][:,axes["joint_ids"]-1],sample_indices=special)
        for seconds,name in ((66.24,"left_knee"),(92.4,"left_ankle"),(97.06,"right_ankle"),
                             (100.84,"left_ankle"),(159.58,"right_knee")):
            case=CASES.index(seconds)
            j=int(np.flatnonzero(axes["joint_names"]==name)[0])
            for side,twist,swing in zip(cases[f"{seconds:.2f}"],
                    (old_angles.twist[case,j],axes["twist"][special[case],j]),
                    (old_angles.swing[case,j],axes["swing"][special[case],j]),strict=True):
                side["model_joint_chart"]=dict(joint=name,flex_deg=float(np.rad2deg(twist)),
                    off_axis_deg=float(np.rad2deg(swing)),clinical_angle=False)
        visual=load(problem.visual/"whole_follow_evidence.npz")
        report["whole_skate_source_coverage"]=[dict(shot=str(shot),foot=foot,
            exposures=int(use.sum()),visible=int(roles["visible"][use,foot].sum()),
            used_for_contour_fit=int(np.sum(roles["role"][use,foot]==1)),
            missed=int(np.sum(visual["status"][use,foot]!=1)),
            extent_uncertain=int(np.sum((visual["status"][use,foot]==1)&(visual["extent"][use,foot]<=0))),
            identity_unresolved=int(visual["identity_unresolved"][use,foot].sum()),
            partial_or_full_occlusion=int(np.sum(visual["partial_occlusion"][use,foot]|visual["skate_occluded"][use,foot])))
            for shot in np.unique(problem.i.shots) for foot in (0,1)
            for use in (problem.i.shots==shot,)]
        trace_binding=programme.get("reviewed_runner_fit",{})
        trace_path=Path(trace_binding.get("path",job/"support_pixel_review.json"))
        if trace_path.is_file():
            checksum=digest(trace_path)
            if checksum!=trace_binding.get("sha256",checksum):
                raise ValueError("Source runner trace provenance changed")
            trace=read(trace_path)
            if tuple(trace["clock"])!=tuple(problem.i.tb) or trace["source_sha256"]!=problem.i.media_sha:
                raise ValueError("Reviewed runner traces use another source clock or video")
            traces=trace["rows"]
            report["reviewed_runner_traces"]=dict(role="development",path=str(trace_path),sha256=checksum,
                source_pts=[r["pts"] for r in traces],
                before=archived_edge_check(problem,refs[0],traces)[1],
                after=archived_edge_check(problem,refs[1],traces)[1])
    if "repair_role_receipt" in document["whole_skate_source"]:
        report["source_roles"]["regression_repair"]=document["whole_skate_source"]["repair_role_receipt"]
    rows=[]
    for shot in np.unique(problem.i.shots):
        for foot in (0,1):
            m=silhouette[:,foot] & (problem.i.shots==shot)
            row=dict(shot_id=str(shot),foot=foot,before=statistics(errors[0][m,foot]),after=statistics(errors[1][m,foot]))
            if m.any():
                i=np.flatnonzero(m)[int(np.argmax(pair[m,foot]))]
                row.update(worst_regression_pts=int(pts[i]),worst_regression_px=float(pair[i,foot]))
            rows.append(row)
    report["development_by_shot_and_foot"]=rows
    shape_a,shape_b=(r.binding.packet() for r in refs)
    geometry=programme.get("geometry_trial")
    if geometry is None:
        if not report["geometry_preserved"]:
            raise ValueError("First-stage shoe and steel geometry changed")
        for k in ("upper","inner","adapter","edges"):
            np.testing.assert_allclose(shape_a[k],shape_b[k],rtol=0,atol=5.e-12)
    else:
        for key,value in geometry["shape"].items():
            if key=="upper_height_scale":
                actual=np.exp(refs[1].binding.fixed_upper_height_log)
            else:
                actual=getattr(shape_b["parameters"],key)
                if not key.endswith("_fraction"):
                    actual*=refs[1].binding.ratio
            np.testing.assert_allclose(actual,value,rtol=1.e-12,atol=0.)
        report["geometry_trial"]=geometry
    atomic_json(solve/"review_audit.json",report)
    atomic_npz(solve/"review_values.npz",source_pts=pts,time_base=problem.i.tb,shot_ids=problem.i.shots,
               before_outline_px=errors[0],after_outline_px=errors[1],role=roles["role"],
               **{f"{side}_{k}":r[k] for side,r in (("before",old),("after",new))
                  for k in ("native22_world_mid","com_world_mid","com_projection_world_mid",
                            "skate_R_world_mid","skate_origin_world_mid","COCO_uv",
                            "skate_contact_mode_mid","sequence_contact_modes","sequence_contact_status") if k in r},
               **{f"{side}_{k}":v for side,d in zip(("before","after"),poses,strict=True)
                  for k,v in d.items() if isinstance(v,np.ndarray)})
    emit("actual_programme_review",exposures=len(pts),body=report["body_px"],
         outline=report["development_outline"],first_spin=report["spin_supported_heading_steps_deg"],
         free_left_peak=report["free_left_peak_deg"],geometry_preserved=report["geometry_preserved"])
    return report


def register(problem, previous, solve, report):
    config=read(previous/"viewer3d.json")
    prior=config["variants"][0]
    repair="committed_state_import" in report
    name=AUDIT_NAME if repair else NAME
    label="旧版人体与整鞋联合修复" if repair else "全节目 ExMR 与整鞋联合拟合"
    candidate=dict(prior,id=name,candidate_id=name,label=label,
                   result=str(solve/"fitted/result.npz"),result_sha256=report["output_sha256"],
                   parameter=str(solve/"solve/candidate.npz"),parameter_sha256=report["controls_sha256"],
                   source_projection_origin="same continuous native exposure solution",
                   accuracy_accepted=False,user_accepted=False,body_reference_label=label)
    if "collar_fit_revision" in report:
        candidate["label"]="全节目 ExMR 与整鞋联合拟合（实际鞋口网格）"
    event_specs=[(84.54,85.00,"首旋右支撑鞋转向"),(86.80,92.72,"首旋支撑与自由足"),
                 (92.20,92.70,"左自由脚旋转峰值")]
    event_specs.extend((max(float(problem.i.times[0]),s-.22),min(float(problem.i.times[-1]),s+.22),
                        f"源 {s:.2f} 秒核对") for s in CASES)
    regressions=[r for r in report["development_by_shot_and_foot"] if r.get("worst_regression_px",0.)>0.]
    for row in sorted(regressions,key=lambda r:r["worst_regression_px"],reverse=True):
        sec=row["worst_regression_pts"]/12800.
        side=("左","右")[row["foot"]]
        event_specs.append((max(float(problem.i.times[0]),sec-.22),min(float(problem.i.times[-1]),sec+.22),
                            f"{row['shot_id']} {side}鞋最大轮廓退步 {sec:.2f} 秒"))
    nav=read(previous/"navigation.json")
    np.testing.assert_array_equal([r["pts"] for r in nav["cuts"]],
                                  [round(r["source_seconds"]*12800) for r in report["cut_diagnostics"]])
    def stamp(seconds):
        return int(problem.i.pts[np.argmin(abs(problem.i.times-seconds))])
    nav["events"]=[dict(start_pts=stamp(a),end_pts=stamp(b),worst_pts=stamp((a+b)/2),
                        label=label,kind="source_review") for a,b,label in event_specs]
    atomic_json(solve/"navigation.json",nav)
    config.update(variants=[candidate,*config["variants"]],default_variant=name,session_id=name,
                  latest_review_variant=name,comparison_variants=[name,prior["id"]],
                  source_navigation=str(solve/"navigation.json"),trajectory_reference="com",
                  review_status="同一原生状态的人体、整鞋、质心与冰面投影")
    atomic_json(solve/"viewer3d.json",config)
    shutil.copyfile(previous/"START_REFITTED_3D.cmd",solve/"START_REFITTED_3D.cmd")


def package(solve, job, target):
    members=[solve/"review_audit.json",solve/"review_values.npz",solve/"navigation.json",
             solve/"run_summary.json",job/"whole_skate_role_receipt.json",
             solve/"solve/candidate.npz"]
    members.extend(p for p in (job/"exmr/inference.json",job/"initialization.json",
                   job/"displayed_baseline_replay.json",job/"state_import.json",
                   job/"state_import_refined.json") if p.is_file())
    traces=read(solve/"review_audit.json").get("reviewed_runner_traces")
    if traces:
        path=Path(traces["path"])
        if digest(path)!=traces["sha256"]:
            raise ValueError("Reviewed runner traces changed after the comparison")
        members.append(path)
        sources=read(path)
        for key in ("cropped_source_panel","additional_source_panel"):
            if key in sources:
                panel=Path(sources[key]["path"])
                if digest(panel)!=sources[key]["sha256"]:
                    raise ValueError("Source crop panel changed after annotation")
                members.append(panel)
    for name in ("geometry_profile.json","whole_skate_direction.json","viewer_verified.json","video_receipt.json","solver_performance.json",
                 "surface_verified/surface/surface_scan.json","actual_surface/surface_scan.json"):
        path=solve/name
        if path.is_file() and all(other.name!=path.name for other in members):
            members.append(path)
    manifest=[dict(member=p.name,sha256=digest(p),bytes=p.stat().st_size) for p in members]
    with ZipFile(target,"x",compression=ZIP_DEFLATED,compresslevel=5) as archive:
        for p in members:
            archive.write(p,p.name)
        archive.writestr("MANIFEST.json",json.dumps(manifest,ensure_ascii=False,indent=2))
    with ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError("Offline review archive integrity failure")
    emit("offline_review_package",path=str(target),bytes=target.stat().st_size,files=len(members)+1)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("audit","proof","clips","video","package","geometry"))
    parser.add_argument("visual",type=Path)
    parser.add_argument("previous",type=Path)
    parser.add_argument("job",type=Path)
    parser.add_argument("solve",type=Path)
    parser.add_argument("recovery",type=Path)
    parser.add_argument("--package",type=Path)
    parser.add_argument("--prepared",type=Path)
    a=parser.parse_args()
    if a.stage=="package":
        if a.package is None:
            parser.error("package requires --package")
        package(a.solve,a.job,a.package)
        return
    os.environ["RINKSPACE_SKATE_RECOVERY_BINDING"]=str(a.recovery)
    problem=Problem(a.visual,a.solve)
    parameter=a.solve/"solve/candidate.npz"
    prepared=a.prepared or a.job/"prepared"
    if a.stage=="geometry":
        candidate=(load_candidate(prepared,parameter) if parameter.is_file()
                   else load_reference(PROJECT,prepared)[0])
        geometry_profile(problem,candidate,a.previous,a.job,a.solve)
        return
    refs=(saved_ref(problem,a.previous/"controls.npz"),load_candidate(prepared,parameter))
    source=read(a.solve/"run_summary.json").get("source_evidence") if (a.solve/"run_summary.json").is_file() else None
    label=("Audited joint body/skate repair" if source and "programme_audit_repair" in read(source) else None)
    if a.stage=="audit":
        report=audit(problem,refs,a.previous,a.solve,a.job,prepared)
        register(problem,a.previous,a.solve,report)
    elif a.stage=="clips":
        windows=dict(CLIPS,post_spin_ankles=(96.94,101.10)) if label else CLIPS
        for name,window in windows.items():
            dest=a.solve/"clips"/name
            dest.mkdir(parents=True,exist_ok=True)
            render_video(problem,dest,references=refs,source_window=window,parameter=parameter,refit_label=label)
    else:
        render_video(problem,a.solve,proof_only=a.stage=="proof",references=refs,
                     proof_seconds=CASES,parameter=parameter,refit_label=label)


if __name__=="__main__":
    main()
