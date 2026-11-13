"""Export the isolated HTD candidate through the current native programme viewer."""
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.skate_video_v1 import programme_motion_followup as followup
from experiments.skate_video_v1.inputs import atomic_json, digest, emit, read


def export(args):
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    base=args.baseline/"fitted/result.npz"
    assessment=read(args.candidate.parent/"entry_assessment.json")
    nav=read(args.baseline/"navigation.json")
    name="HTD 局部候选 · S05 跳入旋转"
    low,high=(int(round(v*12800)) for v in assessment["changed_control_source_seconds"])
    nav["events"]=[dict(start_pts=low,end_pts=high,label=name),*nav["events"]]
    atomic_json(out/"navigation.json",nav)
    followup.VARIANT="programme_htd_entry_trial_1113"
    followup.LABEL="HTD 入旋双臂试验"
    followup.export(SimpleNamespace(output=out,previous=args.baseline,parameter=args.candidate,
                  prepared=args.prepared,evidence=args.evidence))
    cfg=read(out/"viewer3d.json")
    cfg.update(initial_source_pts=1042176,initial_view="side",body_window_scope="全节目回放 · S05 入旋局部双臂试验",
        body_scope_label="HTD 入旋双臂试验",display_outside_window_label="该曝光使用 11 月 13 日的人体状态。",
        review_status="局部回投候选，三维物理仍有未决偏差")
    atomic_json(out/"viewer3d.json",cfg)
    file=out/"fitted/result.npz"
    with np.load(file,allow_pickle=False) as candidate,np.load(base,allow_pickle=False) as reference:
        np.testing.assert_array_equal(candidate["source_pts"],reference["source_pts"])
        delta=np.linalg.norm(candidate["COCO_world_mid"]-reference["COCO_world_mid"],axis=-1)
        keep=~candidate["body_refinement_window_mask"]
        np.testing.assert_allclose(delta[keep],0,atol=1.e-9,rtol=0.)
        for k in ("P_world_mid","skate_R_world_mid","skate_origin_world_mid","equipment_uv"):
            np.testing.assert_allclose(candidate[k],reference[k],atol=1.e-8,rtol=0.)
        np.testing.assert_allclose(candidate["com_projection_world_mid"][:,:2],candidate["com_world_mid"][:,:2],atol=1.e-12,rtol=0.)
        np.testing.assert_array_equal(candidate["com_projection_world_mid"][:,2],0.)
        r=dict(candidate_sha256=digest(file),source_exposures=len(delta),changed_source_exposures=int((delta.max(1)>1.e-8).sum()),
               shared_camera=True,rigid_skate_pose_max_difference=0.,com_projection_world_z=0.,
               source_pts=[int(candidate["source_pts"][0]),int(candidate["source_pts"][-1])])
    atomic_json(out/"htd_export_verify.json",r)
    emit("htd_export_verified",**r)


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__)
    for k in ("output","baseline","prepared","evidence","candidate"):
        cli.add_argument("--"+k,type=Path,required=True)
    export(cli.parse_args())
