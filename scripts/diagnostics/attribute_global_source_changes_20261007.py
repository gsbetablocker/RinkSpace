"""Read-only world/camera attribution, never a coherent candidate or seed."""
from pathlib import Path
import argparse,json
import numpy as np
from evaluate_global_target_state_20261007 import BASE,CONFIG,OUT,read,accepted,load_camera
from rinkspace.vision_camera_initialization import frozen_scene_curve
from z190_shared_board_pattern_source import _project
from wide_anchor_candidate.full_brown_projection import curve_vector_error_full_brown


def world(scene,row):
    curve=row["curve"];ad=curve.get("advertisement_id")
    if row.get("point_correspondence"):
        points=scene["shared_board_patterns"]["shield_fixed_material_points_xyz_scene_units"][str(ad)]
        return np.asarray([points[k] for k in row["material_roles"]])
    if curve["kind"]=="current_goal_crease":
        f=next(f for f in scene["ice_features"] if f["id"]==curve["feature_id"]);xyz=np.asarray(f["polygon_xyz"])
        return xyz if np.array_equal(xyz[0],xyz[-1]) else np.vstack((xyz,xyz[0]))
    return frozen_scene_curve(scene,curve)


def main():
    p=argparse.ArgumentParser();p.add_argument("--state-dir",required=True,type=Path);a=p.parse_args()
    old=accepted(BASE);new=accepted(a.state_dir);options=read(CONFIG)["source_options"]
    scenes=[read(v["scene_path"]) for v in (old,new)];cameras=[load_camera(v["camera_path"]) for v in (old,new)]
    sources=[]
    for option in ("--additional-board-material-point-source","--additional-ice-source"):
        for row in read(options[option])["rows"]:
            if (row["curve"]["kind"]=="current_goal_crease" or row["id"].startswith("short_board_Ad25")):
                sources.append(row)
    records=[]
    for row in sources:
        i=int(np.flatnonzero(cameras[0]["pts"]==row["pts"])[0]);j=int(np.flatnonzero(cameras[1]["pts"]==row["pts"])[0]);uv=np.asarray(row["uv"],float)
        values=[]
        for si,ci in ((0,0),(0,1),(1,0),(1,1)):
            c=cameras[ci]["camera25"][(i,j)[ci]];xyz=world(scenes[si],row)
            if row.get("point_correspondence"):raw=_project(c,xyz,False)[0]-uv
            else:
                raw=curve_vector_error_full_brown(c,xyz,uv,with_jacobian=False)
                if isinstance(raw,tuple):raw=raw[0]
                raw=np.asarray(raw).reshape(-1,2)
            values.append(dict(world="before" if si==0 else "after",camera="before" if ci==0 else "after",
                mean_native_px=float(np.linalg.norm(raw,axis=1).mean()),mean_XY_px=raw.mean(axis=0).tolist(),raw_model_minus_source_UV=raw.tolist()))
        records.append(dict(id=row["id"],pts=row["pts"],split=row["split"],curve=row["curve"],read_only_cross_comparisons=values))
    with np.load(BASE/"complete_state_UNACCEPTED.npz",allow_pickle=False) as aold,np.load(a.state_dir/"complete_state_UNACCEPTED.npz",allow_pickle=False) as anew:
        names=anew["aux__current_named66_parameter_names"].tolist();slots=anew["aux__current_shared_world_geometry_parameter_indices"]
        q0=aold["q"];q1=anew["q"];changes=[dict(name=n,before=float(q0[k]),after=float(q1[k]),delta=float(q1[k]-q0[k]),global_q_index=int(k)) for n,k in zip(names,slots)]
        suffix=[dict(global_q_index=k,before=float(q0[k]) if k<len(q0) else 0.,after=float(q1[k]),delta=float(q1[k]-(q0[k] if k<len(q0) else 0.))) for k in range(34296,len(q1))]
    report=dict(schema="rinkspace.read-only-global-source-world-camera-attribution.v1",before=old,after=new,source_comparisons=records,
        actual_existing_named66_changes=changes,actual_print_suffix_changes=suffix,
        cross_world_camera_pairs_are_diagnostic_only_not_physically_coherent_accepted_states=True,
        no_state_initializer_or_source_modified=True)
    path=OUT/f'{new["q_sha256"][:12]}_world_camera_attribution.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(path=str(path),sources=[dict(id=r["id"],split=r["split"],comparisons=[dict(world=v["world"],camera=v["camera"],mean=v["mean_native_px"],xy=v["mean_XY_px"]) for v in r["read_only_cross_comparisons"]]) for r in records],largest_named66_changes=sorted(changes,key=lambda r:abs(r["delta"]),reverse=True)[:8],suffix=suffix)))


if __name__=="__main__":main()
