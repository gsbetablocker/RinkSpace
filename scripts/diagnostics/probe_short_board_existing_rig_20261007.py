"""Read-only existing-rig explanatory J at the bad short-board endpoint."""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from evaluate_short_board_states_20261007 import read,accepted,RUNS,OUT,circle_world
from calibration.program_tracking.z190_physical_camera import SharedCalibration,ViewState
from rinkspace.vision_camera_initialization import source_jacobian_in_camera25_units
from z190_source_camera_diagnostic import absolute_value_jac15


def main():
    p=argparse.ArgumentParser();p.add_argument("--state-dir")
    a=p.parse_args()
    if a.state_dir:directory=Path(a.state_dir)
    else:directory=max((RUNS/"okada_lane23_Ad7_material_accelerated_20261007/accepted_states").iterdir(),key=lambda d:d.stat().st_mtime)
    identity=accepted(directory);scene=read(identity["scene_path"])
    with np.load(identity["camera_path"],allow_pickle=False) as file:
        pts=file["pts"].copy();i=int(np.flatnonzero(pts==230254336)[0]);camera=file["camera25"][i].copy()
        view=file["view_state4"][i].copy();installation=str(file["installation_id"][i]);metadata=json.loads(file["metadata_json"].item())
    cal=SharedCalibration.from_manifest(metadata["calibrations"][installation]);actual,shared_map,view_map=cal.camera25_and_jacobians(ViewState.from_state4(view))
    if not np.array_equal(camera,actual):raise ValueError("Existing original full rig not bitexact")
    source=read(OUT/"short_board_multitime_source_candidate.json")
    selected=[r for r in source["circle_rows"]+source["shield_rows"] if r["pts"]==230254336]
    rows=[];intervals=[];offset=0
    for row in selected:
        if row.get("point_correspondence"):
            points=scene["shared_board_patterns"]["shield_fixed_material_points_xyz_scene_units"]["25"]
            world=np.array([points[k] for k in row["material_roles"]],float)
        else:world=circle_world(scene,row)
        rows.append(dict(row,world=world,uv=np.asarray(row["uv"]),sigma=np.ones(len(row["uv"]))))
        intervals.append((offset,offset+2*len(row["uv"])));offset+=2*len(row["uv"])
    baseline,rawJ=absolute_value_jac15(camera,rows,True)
    J=source_jacobian_in_camera25_units(rawJ.toarray() if hasattr(rawJ,"toarray") else rawJ)@view_map[:15]
    R=cv2.Rodrigues(view[:3])[0];h=1e-6;tangents=[]
    for k in range(3):
        axis=np.zeros(3);axis[k]=h
        vp=cv2.Rodrigues(cv2.Rodrigues(axis)[0]@R)[0].reshape(-1)
        vm=cv2.Rodrigues(cv2.Rodrigues(-axis)[0]@R)[0].reshape(-1)
        tangents.append((vp-vm)/(2*h))
    basis=np.zeros((4,4));basis[:3,:3]=np.column_stack(tangents);basis[3,3]=1
    Jlocal=J@basis;weights=np.concatenate([np.repeat(np.asarray(r["sigma_px"]),2) for r in rows])
    names=["local_pitch_camera_X_rad","local_yaw_camera_Y_rad","local_roll_camera_Z_rad","existing_log_focal"]
    def evaluate(v):return absolute_value_jac15(cal.camera25(ViewState.from_state4(v)),rows,False)
    fd_checks=[]
    for k in range(4):
        def move(amount):
            v=view.copy()
            if k==3:v[3]+=amount
            else:
                axis=np.zeros(3);axis[k]=amount;v[:3]=cv2.Rodrigues(cv2.Rodrigues(axis)[0]@R)[0].reshape(-1)
            return v
        fd=(evaluate(move(h))-evaluate(move(-h)))/(2*h)
        fd_checks.append(float(np.linalg.norm(fd-Jlocal[:,k])/max(1,np.linalg.norm(fd))))
    trials=[]
    choices=[([1],"all_three_prints"),([3],"all_three_prints"),([1,3],"all_three_prints"),([0,1,2,3],"all_three_prints"),
        ([1],"Ad25_named_material_points_only"),([0,1,3],"Ad25_named_material_points_only"),([0,1,2,3],"Ad25_named_material_points_only")]
    for indices,family in choices:
        selection=np.ones(len(baseline),bool)
        if family=="Ad25_named_material_points_only":
            selection[:]=False
            for r,(s,e) in zip(rows,intervals):
                if r["curve"]["advertisement_id"]==25:selection[s:e]=True
        block=Jlocal[selection][:,indices]/weights[selection,None];amount=np.linalg.lstsq(block,-baseline[selection]/weights[selection],rcond=None)[0]
        delta=np.zeros(4);delta[indices]=amount
        v=view.copy();v[:3]=cv2.Rodrigues(cv2.Rodrigues(delta[:3])[0]@R)[0].reshape(-1);v[3]+=delta[3]
        residual=evaluate(v);trial=dict(parameter_names=[names[k] for k in indices],linear_diagnostic_source_family=family,diagnostic_deltas=amount.tolist(),
            local_rotation_degrees=(delta[:3]*180/np.pi).tolist(),focal_ratio=float(np.exp(delta[3])),
            nonlinear_sources=[dict(ad=r["curve"]["advertisement_id"],distance_before=float(np.linalg.norm(baseline[s:e].reshape(-1,2),axis=1).mean()),
                distance_after=float(np.linalg.norm(residual[s:e].reshape(-1,2),axis=1).mean())) for r,(s,e) in zip(rows,intervals)],
            no_state_initializer_or_solver_written=True)
        trials.append(trial)
    report=dict(schema="rinkspace.short-board-existing-full-rig-read-only-J.v1",state=identity,pts=230254336,installation_id=installation,
        original_full_fixed_pivot_nonzero_lever_camera25_reproduced_bitexact=True,
        names=names,actual_source_operator_analytic_J_FD_relative_L2=fd_checks,diagnostic_trials=trials,
        actual_rig_expression_preserved=True,shared_world_mount_optics_and_all_other_exposures_not_changed=True,
        exploratory_delta_not_a_full_target_accepted_state_or_verified_whole_shot_fix=True)
    path=OUT/"short_board_endpoint_existing_rig_J.json";path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(report=str(path),state=identity["q_sha256"],FD=fd_checks,trials=trials)))


if __name__=="__main__":main()
