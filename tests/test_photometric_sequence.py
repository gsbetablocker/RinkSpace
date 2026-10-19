"""Saved workbench evidence obeys masks, real PTS, missing maps and declared cuts."""
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from rinkspace.io import dump_json, load_json
from rinkspace.photometric_sequence import register_photometric_sequence


def samples(directory, poison=False):
    directory.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(741)
    base=cv2.GaussianBlur(rng.integers(170,240,(180,260),dtype=np.uint8),(5,5),1.)
    rows=[]
    for i in range(5):
        image=cv2.warpAffine(base,np.float32([[1,0,i*1.5],[0,1,0]]),(260,180))
        if poison:
            image[65:106,100:141]=rng.integers(0,255,(41,41),dtype=np.uint8)
        path=f"frame_{i:06d}.png"
        cv2.imwrite(str(directory/path),cv2.cvtColor(image,cv2.COLOR_GRAY2BGR))
        rows.append(dict(sample_index=i,source_frame_index=100+i,pts=(100+i)*256,
            time_base="1/12800",source_time_s=(100+i)/50,image=path,
            image_width=260,image_height=180,image_scale_from_source=1))
    dump_json(directory/"samples.json",dict(source_sha256="a"*64,source_filename="synthetic_fixture",
                                           every_n=1,samples=rows))
    return rows


def test_runner_preserves_pts_protection_failure_cut_and_full_local_audit(tmp_path):
    source=tmp_path/"source"; original=samples(source)
    full=[[0,0],[259,0],[259,179],[0,179]]
    corrections={"*":{"include_polygons":[full]},"2":{"exclude_polygons":[full]}}
    exclusions={"*":[{"polygon":[[100,65],[140,65],[140,105],[100,105]],"physical_landmark_id":"whole_group"}]}
    out=tmp_path/"out"
    result=register_photometric_sequence(source,out,corrections=corrections,exclusions=exclusions,shot_breaks=[4])
    rows=result["frames"]
    assert [(r["pts"],r["source_frame_index"]) for r in rows] == [(r["pts"],r["source_frame_index"]) for r in original]
    assert rows[0]["reference_frame_only"] and rows[4]["reference_frame_only"]
    assert rows[2]["status"] == "unavailable" and rows[2]["H_reference_from_current"] is None
    assert "fixed_image" not in rows[2]
    assert rows[3]["recovered_after_missing"] and rows[3]["graph_component_id"] == rows[0]["graph_component_id"]
    assert rows[4]["graph_component_id"] != rows[0]["graph_component_id"]
    assert len(result["canvases"]) == 2
    assert result["formal_acceptance"] is result["independent_validation"] is False
    for row in rows:
        usable=cv2.imread(str(out/row["mask_path"]),0)
        protected=cv2.imread(str(out/row["protected_mask_path"]),0)
        assert not np.any((usable>0)&(protected>0))
        with gzip.open(out/row["photometric_audit_path"],"rt",encoding="utf-8") as f:
            audit=json.load(f)
        assert audit["current_sample"]["pts"] == row["pts"]
        if row["status"] == "candidate" and not row.get("reference_frame_only"):
            assert row["fit_points_full_count"] > len(row["fit_current_xy"])
            assert row["train_inliers"] == 0
            assert "not_independent" in row["fit_points_role"]
    assert len(load_json(out/"footprints/manifest.json")["files"]) == 5


def test_protected_pixel_values_cannot_change_masks_or_maps_in_workbench_pipeline(tmp_path):
    exclusions={"*":[{"polygon":[[100,65],[140,65],[140,105],[100,105]],"physical_landmark_id":"whole_group"}]}
    results=[]
    for poisoned in (False,True):
        source=tmp_path/f"source-{poisoned}"; samples(source,poisoned)
        out=tmp_path/f"out-{poisoned}"
        result=register_photometric_sequence(source,out,exclusions=exclusions,render=False)
        results.append((out,result))
    for a,b in zip(results[0][1]["frames"],results[1][1]["frames"]):
        assert a["status"] == b["status"]
        assert a["H_reference_from_current"] == b["H_reference_from_current"]
        assert (results[0][0]/a["mask_path"]).read_bytes() == (results[1][0]/b["mask_path"]).read_bytes()
        assert (results[0][0]/a["photometric_audit_path"]).read_bytes() == (results[1][0]/b["photometric_audit_path"]).read_bytes()


def test_frozen_backend_rejects_parameter_overrides(tmp_path):
    with pytest.raises(ValueError,match="does not accept parameter overrides"):
        register_photometric_sequence(tmp_path/"missing",tmp_path/"out",config={"motion_model":"homography"})


def test_experimental_camera_backend_records_assumptions_and_real_cuts(tmp_path):
    source=tmp_path/'source'; samples(source)
    result=register_photometric_sequence(source,tmp_path/'out',backend='rotation_zoom_experimental',
        corrections={'*':{'include_polygons':[[[0,0],[259,0],[259,179],[0,179]]]}},shot_breaks=[4])
    assert result['backend']=='rotation_zoom_experimental'
    assert result['config']['photometric']['initial_focal_width_ratio']==4
    assert result['freeze_verification']['verified'] is False
    assert result['frames'][0]['graph_component_id']!=result['frames'][4]['graph_component_id']
    assert all(r['camera_model_independently_verified'] is False for r in result['frames'])
    assert result['formal_acceptance'] is False


def test_camera_display_rejects_horizon_without_altering_geometry(tmp_path):
    from rinkspace.photometric_sequence import render_camera_frames
    horizon=[[1.,0.,0.],[0.,1.,0.],[-.008,0.,1.]]
    rows=[dict(sample_index=0,status='candidate',graph_component_id='a',H_reference_from_current=horizon),
          dict(sample_index=1,status='candidate',graph_component_id='a',H_reference_from_current=np.eye(3).tolist())]
    record={'decoded_dimensions':[260,180],'frames':rows}
    assert render_camera_frames(tmp_path,tmp_path,record,10000)=={}
    assert all(r['status']=='candidate' and 'fixed_path' not in r for r in rows)
    assert rows[0]['H_reference_from_current']==horizon
    assert record['display_rejected_components']==['a']
