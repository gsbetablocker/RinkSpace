"""Verify actual material FD worker transport after world-dimension expansion."""
from pathlib import Path
import hashlib,json,pickle
import numpy as np
from calibration.solver_runtime import install

BASE=Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/external_expert_20261007/physical_geometry_unlock_20261008")
FAILED=Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_unlocked_geometry_current_696f_strict_blue_roles_20261008")

def main():
    install()
    from prepared_cache_helpers import install_mappingproxy_pickle,install_geometry_unpickle
    install_mappingproxy_pickle();install_geometry_unpickle()
    from calibration.solver_runtime.absolute_geometry_fd_pool_candidate import AbsoluteGeometryGroupPool
    from structured_programme_source_candidate.batched_absolute_geometry_fd import evaluate_geometry_group
    from structured_programme_source_candidate.original_review_source import PACKET,load_original_absolute_evaluators
    from z190_physical_camera import PhysicalCameraModel
    info=json.loads((FAILED/"actual_absolute_geometry_FD_pool_runtime.json").read_text(encoding="utf-8"))
    owner=info["material_owner"]
    geometry=pickle.loads(Path(owner["path"]).read_bytes())
    with np.load(BASE/"strict_half_circle/complete_state_initializer_UNACCEPTED.npz",allow_pickle=False)as a:
        q=a["q"];indices=a["aux__current_shared_world_geometry_parameter_indices"].astype(int);world=q[indices]
    camera_path=next(FAILED.glob("complete_physical_camera_model_UNACCEPTED_*.npz"))
    model,pts,aliases,views=PhysicalCameraModel.load_npz(camera_path)
    calls,authorities=load_original_absolute_evaluators()
    packet=pickle.loads(PACKET.read_bytes())
    chosen=[];kinds=set()
    for item in packet["programme_absolute_rows"]:
        row=item["source_row"]
        key=geometry._key(row)
        if item["collection_role"]!="FIT"or key not in geometry.bindings:continue
        kind=geometry.bindings[key]["kind"]
        if kind in kinds:continue
        kinds.add(kind);chosen.append(item)
        if len(chosen)==3:break
    records=[]
    with_pool=AbsoluteGeometryGroupPool(owner,workers=1,groups_per_job=1,max_inflight=1)
    try:
        for item in chosen:
            row=item["source_row"];stamp=int(item["group_actual_pts"]);camera=model.camera25(views[aliases[stamp]])
            actual=calls[item["block"]]
            direct=evaluate_geometry_group(geometry,world,camera,[row],actual,1e-7)
            result=list(with_pool.ordered_groups([(stamp,[row],camera)],world,actual,1e-7))[0]
            err=float(np.max(np.abs(result["geo"]-direct)))
            if direct.shape!=(2*len(row["uv"]),98)or err!=0.:raise ValueError("Actual dynamic material worker/direct result changed")
            records.append(dict(id=row["id"],actual_pts=stamp,block=item["block"],material_kind=geometry.bindings[geometry._key(row)]["kind"],J_shape=list(direct.shape),worker_vs_parent_max_error=err,worker_vs_parent_bitexact=True,source_group_sha256=result["source_group_sha256"]))
    finally:with_pool.close()
    report=dict(schema="rinkspace.dynamic-absolute-material-FD-worker-transport-check.v1",world_parameter_count=98,material_owner_sha256=owner["sha256"],original_source_groups=records,all_source_UV_sigma_and_FD_step_preserved=True,whole_motion_graph_not_loaded=True,passed=True)
    dest=BASE/"strict_half_circle/actual_dynamic_absolute_FD_pool_check.json";dest.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report),flush=True);print("report_sha256",hashlib.sha256(dest.read_bytes()).hexdigest())

if __name__=="__main__":main()
