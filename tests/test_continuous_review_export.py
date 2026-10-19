"""Exporter checks protect against falsely displaying projective or wrong-time maps."""
from fractions import Fraction
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


spec=importlib.util.spec_from_file_location("continuous_review_export",Path(__file__).parents[1]/"scripts/build_continuous_page.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_projective_maps_are_not_silently_drawn_as_affine():
    h=np.eye(3);h[2,0]=0.001
    with pytest.raises(ValueError,match="Projective homography"):
        module.affine_matrix(h)
    assert np.array_equal(module.affine_matrix(2*np.eye(3)),np.eye(3))


def test_source_preview_keeps_every_actual_vfr_pts(tmp_path):
    rows=[]
    for i,pts in enumerate([1000,1020,1070,1090]):
        name=f"frame_{i:06d}.jpg"
        cv2.imwrite(str(tmp_path/name),np.full((64,96,3),i*60,np.uint8))
        rows.append({"sample_index":i,"source_frame_index":50+i,"image":name,
                     "pts":pts,"time_base":"1/1000","source_time_s":pts/1000})
    result=module.encode_source(tmp_path,rows,tmp_path/"source.mp4",96,64)
    assert result["exact_relative_pts_verified"]
    assert [r["preview_time_s"] for r in result["frames"]]==[0,.02,.07,.09]
    assert [r["pts"] for r in result["frames"]]==[1000,1020,1070,1090]


def test_source_preview_refuses_reordered_pts(tmp_path):
    rows=[{"pts":p,"time_base":"1/1000"} for p in [10,9]]
    with pytest.raises(ValueError,match="strictly increase"):
        module.encode_source(tmp_path,rows,tmp_path/"bad.mp4",96,64)


def test_disconnected_references_cannot_be_made_into_a_single_view(tmp_path):
    reg={"sample_directory":str(tmp_path),"decoded_dimensions":[96,64],"source_sha256":"test",
         "frames":[{"H_reference_from_current":np.eye(3).tolist(),"graph_component_id":c} for c in ["a","b"]]}
    (tmp_path/"registration.json").write_text(json.dumps(reg),encoding="utf-8")
    with pytest.raises(ValueError,match="Disconnected components"):
        module.build(tmp_path,None,None,tmp_path/"out")


def test_projective_polygon_denominator_is_checked_before_division():
    polygon=[[0,0],[100,0],[100,60],[0,60]]
    matrix=np.array([[1,0,4],[0,1,7],[.002,0,1.]])
    actual=module.map_observed_polygon(matrix,polygon)
    assert np.allclose(actual[1],[104/1.2,7/1.2])
    inverse=np.linalg.inv(matrix)
    assert np.allclose(module.map_observed_polygon(inverse,actual.tolist()),polygon)
    crossing=np.array([[1,0,0],[0,1,0],[.02,0,-1.]])
    with pytest.raises(ValueError,match="horizon crosses"):
        module.map_observed_polygon(crossing,polygon)
    touching=np.array([[1,0,0],[0,1,0],[.01,0,0.]])
    with pytest.raises(ValueError,match="horizon crosses"):
        module.map_observed_polygon(touching,polygon)


def test_projective_export_requires_explicit_opt_in_and_keeps_fixed_gauge(tmp_path):
    rows=[]
    for i in range(3):
        image=f"frame_{i:06d}.jpg";mask=f"mask_{i:06d}.png"
        cv2.imwrite(str(tmp_path/image),np.full((64,96,3),50+i*40,np.uint8))
        cv2.imwrite(str(tmp_path/mask),np.full((64,96),255,np.uint8))
        h=np.array([[1,0,3*i],[0,1,i],[.0004*(i+1),0,1.]])
        rows.append({"sample_index":i,"source_frame_index":100+i,"image":image,"mask_path":mask,
                     "pts":1000+20*i,"time_base":"1/1000","source_time_s":1+.02*i,
                     "H_reference_from_current":h.tolist(),"graph_component_id":"same",
                     "status":"candidate","reference_frame_only":i==0})
    reg={"sample_directory":str(tmp_path),"decoded_dimensions":[96,64],"source_sha256":"fixture",
         "source_filename":"fixture.mp4","frames":rows}
    (tmp_path/"registration.json").write_text(json.dumps(reg),encoding="utf-8")
    with pytest.raises(ValueError,match="Projective homography"):
        module.build(tmp_path,None,None,tmp_path/"default")
    result=module.build(tmp_path,None,None,tmp_path/"optin",allow_projective=True)
    assert result["frame_count"]==3
    pts=json.loads((tmp_path/"optin/source_pts_map.json").read_text(encoding="utf-8"))
    assert pts["renderer"]=="projective_webgl_inverse_h"
    assert pts["reference_sample"]==1
    assert np.allclose(np.asarray(pts["fixed_reference_gauge"])@np.asarray(rows[1]["H_reference_from_current"]),np.eye(3))
    assert pts["exact_relative_pts_verified"]
