"""Immutable multi-time print QA using actual existing source operators."""
from pathlib import Path
import argparse, json
import cv2
import numpy as np
from evaluate_user_lane_states_20261007 import read, sha, accepted, DATA, RUNS
from z190_source_camera_diagnostic import absolute_value_jac15
from z190_shared_board_pattern_source import _project
from rinkspace.vision_camera_initialization import frozen_scene_curve
from short_board_global_source_20261007 import OUT


def circle_world(scene, row):
    if row["curve"]["advertisement_id"] == 24:
        return frozen_scene_curve(scene, row["curve"])
    return np.asarray(scene["shared_board_patterns"]["circle_outer_stroke_paths_xyz_scene_units"]["26"], float)


def main():
    p=argparse.ArgumentParser();p.add_argument("--state-dir",action="append")
    args=p.parse_args()
    frozen=read(DATA/"programmes/okada_mei_sp/results/shared_radius_05db_pelvis_priority_full_programme_20261005/FROZEN_INPUTS_UNACCEPTED.json")
    for key in ("scene","camera"):
        if sha(frozen[key]["path"])!=frozen[key]["sha256"]:raise ValueError("Old immutable video changed")
    states=[dict(label="old_video4075_camera917c204",q_sha256=frozen["q_sha256"],scene_path=frozen["scene"]["path"],camera_path=frozen["camera"]["path"]),
        accepted(RUNS/"source_corrected_clear_actual_complete_joint_20261007/accepted_states/0eb5c6d4d2b0a24ab201f65361ea0ed2de409b4bdfdddb1fc68c6b331ae54e63","clear0eb5"),
        accepted(RUNS/"okada_lane23_Ad7_fast_complete_20261007/accepted_states/478afaf776150c398c48e7492da911e1219036cfad3295629a3df309b7d1089c","q478")]
    if args.state_dir:states += [accepted(Path(d)) for d in args.state_dir]
    else:
        dirs=sorted((RUNS/"okada_lane23_Ad7_material_accelerated_20261007/accepted_states").iterdir(),key=lambda d:d.stat().st_mtime)
        states.append(accepted(dirs[-1],"latest_material"))
    candidate=OUT/"short_board_multitime_source_candidate.json";source=read(candidate)
    rows=source["circle_rows"]+source["shield_rows"]
    measures=[];image_panels=[];full_panels={}
    for state in states:
        scene=read(state["scene_path"])
        with np.load(state["camera_path"],allow_pickle=False) as saved:
            pts=saved["pts"].copy();cameras=saved["camera25"].copy();actual=saved["exposure_representative_pts"].copy()
        for row in rows:
            if sha(row["source_raw_frame"])!=row["source_file_sha256"]:raise ValueError("Source changed")
            indices=np.flatnonzero(pts==row["pts"])
            if len(indices)!=1 or int(actual[indices[0]])!=row["actual_pts"]:raise ValueError("Native clock identity failed")
            camera=cameras[indices[0]];uv=np.asarray(row["uv"],float)
            if row.get("point_correspondence"):
                points=scene["shared_board_patterns"]["shield_fixed_material_points_xyz_scene_units"]["25"]
                world=np.array([points[k] for k in row["material_roles"]],float)
                projected=_project(camera,world,False)[0];raw=projected-uv
                normals=None;signed=None
            else:
                world=circle_world(scene,row)
                raw=absolute_value_jac15(camera,[dict(row,uv=uv,world=world,sigma=np.ones(len(uv)))],False).reshape(-1,2)
                projected=uv+raw;normals=np.asarray(row["normal"],float)
                normals/=np.linalg.norm(normals,axis=1)[:,None];signed=np.einsum("ni,ni->n",raw,normals)
            record=dict(state=state["label"],q_sha256=state["q_sha256"],id=row["id"],pts=row["pts"],actual_pts=row["actual_pts"],
                advertisement_id=row["curve"]["advertisement_id"],split=row["split"],source_uv=uv.tolist(),raw_model_minus_source_UV=raw.tolist(),
                mean_model_minus_source_XY_px=raw.mean(axis=0).tolist(),mean_vector_distance_px=float(np.linalg.norm(raw,axis=1).mean()),
                maximum_vector_distance_px=float(np.linalg.norm(raw,axis=1).max()),signed_source_normal_px=None if signed is None else signed.tolist(),
                mean_absolute_normal_px=None if signed is None else float(np.abs(signed).mean()),
                blur_sigma_px_engineering_only=row["sigma_px"],source_generated_from_model=False)
            if not row.get("point_correspondence") and len(uv)>=9:
                source_ellipse=cv2.fitEllipse(uv.astype(np.float32))
                model_curve=_project(camera,world,False)[0]
                model_ellipse=cv2.fitEllipse(model_curve.astype(np.float32))
                record["diagnostic_only_ellipse_fits_not_world_source_factors"]=dict(source_center_xy=source_ellipse[0],
                    source_diameters=source_ellipse[1],model_center_xy=model_ellipse[0],model_diameters=model_ellipse[1],
                    model_minus_source_center_xy=(np.array(model_ellipse[0])-np.array(source_ellipse[0])).tolist(),
                    source_from_real_visible_UV_only=True,diagnostic_fit_not_exact_physical_circle_center=True)
            measures.append(record)
            if state==states[-1]:
                if row["pts"] not in full_panels:
                    raw_image=cv2.imread(row["source_raw_frame"]);full_panels[row["pts"]]=raw_image[230:500].copy()
                full=full_panels[row["pts"]]
                if not row.get("point_correspondence"):
                    curve=_project(camera,world,False)[0]-[0,230]
                    cv2.polylines(full,[np.rint(curve).astype(np.int32)],False,(210,30,210),1,cv2.LINE_AA)
                for a,b in zip(uv,projected):
                    aa=tuple(np.rint(a-[0,230]).astype(int));bb=tuple(np.rint(b-[0,230]).astype(int))
                    cv2.circle(full,aa,3,(25,160,25),-1,cv2.LINE_AA)
                    cv2.drawMarker(full,bb,(210,30,210),cv2.MARKER_CROSS,9,1,cv2.LINE_AA)
                cv2.putText(full,f'Ad{record["advertisement_id"]}',tuple(np.rint(uv.mean(axis=0)-[0,210]).astype(int)),0,.5,(25,160,25),1)
            if state in (states[0],states[-1]):
                raw_image=cv2.imread(row["source_raw_frame"]);lo=np.floor(np.min(uv,axis=0)-60).astype(int);hi=np.ceil(np.max(uv,axis=0)+60).astype(int)
                lo=np.maximum(lo,[0,0]);hi=np.minimum(hi,[1920,1080]);crop=raw_image[lo[1]:hi[1],lo[0]:hi[0]].copy()
                for a,b in zip(uv,projected):
                    a=tuple(np.rint(a-lo).astype(int));b=tuple(np.rint(b-lo).astype(int))
                    cv2.circle(crop,a,3,(25,160,25),-1,cv2.LINE_AA);cv2.drawMarker(crop,b,(210,30,210),cv2.MARKER_CROSS,9,1,cv2.LINE_AA)
                    cv2.line(crop,a,b,(210,30,210),1,cv2.LINE_AA)
                canvas=np.full((crop.shape[0]+44,700,3),245,np.uint8);canvas[44:,:crop.shape[1]]=crop
                cv2.putText(canvas,f'{state["label"]} Ad{record["advertisement_id"]} PTS{row["pts"]}',(5,18),0,.45,(20,20,20),1)
                cv2.putText(canvas,f'SOURCE green dots; model magenta X; mean {record["mean_vector_distance_px"]:.2f}px',(5,35),0,.45,(20,20,20),1)
                image_panels.append(canvas)
    report=dict(schema="rinkspace.immutable-short-board-multitime-source-comparison.v1",source_candidate=dict(path=str(candidate),sha256=sha(candidate)),
        states=states,measurements=measures,source_operator="Actual existing full Brown projection; same frozen_scene_curve and native finite nearest curve for Ad24; original q9 curve for Ad26; named fixed material points for Ad25",
        all_old_source_packets_unchanged=True,whole_video_accuracy_accepted=False,source_points_blur_and_reader_scope_not_exact_pixel_GT=True,
        circle_nearest_curve_XY_is_local_rim_displacement_not_projected_center_error=True)
    path=OUT/"short_board_immutable_state_source_errors.json";path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    cv2.imwrite(str(OUT/"short_board_native_source_model_residuals.png"),np.vstack(image_panels))
    native=[]
    for pts,frame in full_panels.items():
        header=np.full((35,1920,3),245,np.uint8)
        cv2.putText(header,f'{states[-1]["label"]} PTS{pts} | green source; magenta existing print geometry; native y230..500 DIAGNOSTIC',(8,24),0,.6,(20,20,20),1)
        native.append(np.vstack((header,frame)))
    cv2.imwrite(str(OUT/"short_board_latest_native_full_width_residuals.png"),np.vstack(native))
    print(json.dumps(dict(report=str(path),states=[dict(state=s["label"],q=s["q_sha256"],rows=[dict(ad=m["advertisement_id"],pts=m["pts"],xy=m["mean_model_minus_source_XY_px"],distance=m["mean_vector_distance_px"],normal=m["mean_absolute_normal_px"]) for m in measures if m["state"]==s["label"]]) for s in states])))


if __name__=="__main__":main()
