import hashlib
import json

import cv2
import numpy as np
import pytest

from calibration.program_tracking import native_board_material_support as support


def packet():
    def feature(role, runs):
        return dict(id="board_"+role+".upper_left",physical_surface="board",allowed_for_fit=False,
            segments=[dict(measurement_origin="current_native_joint_smooth_material_curve_fit",
                source_evidence_measured_in_this_frame=True,source_native_readout_uv=uv,
                source_native_transition_iqr_px=[2.]*len(uv)) for uv in runs])
    return dict(features=[feature("top",[[[30,100],[300,110]],[[500,100],[900,110]]]),
                          feature("contact",[[[30,500],[300,510]],[[500,500],[900,510]]])])


def test_source_runs_remain_separate_and_derived_inside():
    regions=support.derive_regions(packet())
    assert len(regions)==2
    assert regions[0]["polygon_xy"]==[[30.,112.],[300.,112.],[300.,498.],[30.,498.]]
    assert regions[1]["polygon_xy"][0][0]==500
    assert all(r["upper"]["readout_count"]==2 for r in regions)


def test_unmeasured_boundary_never_becomes_material_support():
    p=packet()
    p["features"][0]["segments"][0]["source_evidence_measured_in_this_frame"]=False
    assert len(support.derive_regions(p))==1
    p["features"][0]["physical_surface"]="unknown"
    assert support.derive_regions(p)==[]


def test_full_footprint_not_only_center_and_exact_pts():
    regions=support.derive_regions(packet())
    s=support.NativeBoardMaterialSupport(dict(schema=support.SCHEMA,world_projection_used=False,
        fit_observations_added=False,frames=[dict(frame_index_1based=1,native_pts=256256,
            source_video_sha256="a"*64,regions=regions,exclusions=dict(image_border_px=20))]))
    f=s.frames[1]
    s._authenticated[1]=(f,[[200,200,210,210]])
    assert s.require(1,256256,[100,200],17)==regions[0]["region_id"]
    with pytest.raises(ValueError,match="outside"):
        s.require(1,256256,[310,200],17)
    with pytest.raises(ValueError,match="person"):
        s.require(1,256256,[190,190],17)
    with pytest.raises(ValueError,match="PTS"):
        s.require(1,256257,[100,200],17)
    with pytest.raises(ValueError,match="35px"):
        s.require(1,256256,[100,200],8)


def test_tampered_raw_and_region_fail_before_material_admission(tmp_path):
    p=packet()
    pp=tmp_path/"packet.json"
    pp.write_text(json.dumps(p),encoding="utf8")
    ref=support.reference(pp)
    pp.write_text(json.dumps(dict(p,changed=True)),encoding="utf8")
    with pytest.raises(ValueError,match="changed"):
        support._read(ref)


def test_original_packet_region_recomputed_and_all_pixel_bytes_authenticated(tmp_path):
    image=np.zeros((1080,1920),np.uint8)
    mask=image.copy();mask[200:211,200:211]=255
    ip=tmp_path/"original.png";mp=tmp_path/"mask.png"
    assert cv2.imwrite(str(ip),image) and cv2.imwrite(str(mp),mask)
    ir=support.reference(ip);mr=support.reference(mp)
    p=packet();p["source"]=dict(native_pts=256256,programme_frame_index_1based=1,
        source_video_sha256="a"*64,time_base="1/12800",image_path=str(ip),png_sha256=ir["sha256"],
        mask_path=str(mp),mask_sha256=mr["sha256"])
    pp=tmp_path/"packet.json";pp.write_text(json.dumps(p),encoding="utf8")
    f=dict(frame_index_1based=1,native_pts=256256,source_video_sha256="a"*64,
        source_frame=ir,person_mask=mr,source_packet=support.reference(pp),regions=support.derive_regions(p),
        admission_basis="original_rgb_boundary_readouts",exclusions=dict(person_mask_positive_pixels=True,
            person_halo_px=15,broadcast_rectangles_xyxy=[[0,0,599,139],[1660,930,1900,1020]],image_border_px=20))
    evidence=dict(schema=support.SCHEMA,world_projection_used=False,fit_observations_added=False,frames=[f])
    safe=support.NativeBoardMaterialSupport(evidence)
    assert safe.require(1,256256,[100,300],17)
    with pytest.raises(ValueError,match="person"):
        safe.require(1,256256,[180,200],17)
    f["regions"][0]["polygon_xy"][0][0]-=1
    with pytest.raises(ValueError,match="exact original"):
        support.NativeBoardMaterialSupport(evidence).require(1,256256,[100,300],17)
    f["regions"]=support.derive_regions(p)
    ip.write_bytes(b"changed")
    with pytest.raises(ValueError,match="changed"):
        support.NativeBoardMaterialSupport(evidence).require(1,256256,[100,300],17)


def test_explicit_legacy_mode_preserves_raw_profiles_and_visible_gap():
    p=dict(features=[dict(id="board_top",physical_surface="board",source_straight_line_estimate=dict(
        model_projection_used=False,evidence_uv=[[50,100],[100,102],[500,104],[600,103]],
        evidence_localization_sigma_px=[3,3,3,3]),segments=[dict(measurement_origin="same_frame_joint_straight_edge_fit",
        source_evidence_measured_in_this_frame=True,hidden_span_filled=False,uv=[[30,100],[200,102]]),
        dict(measurement_origin="same_frame_joint_straight_edge_fit",source_evidence_measured_in_this_frame=True,
        hidden_span_filled=False,uv=[[450,104],[650,103]])]),dict(id="board_contact",physical_surface="board",
        segments=[dict(pixels_measured_in_this_frame=True,additional_points_measured_in_native_pixels=True,
        uv=[[30,500],[200,500]],endpoint_evidence_transition_iqr_px=[4,4],
        point_bindings=[dict(measurement_origin="native_colour_half_transition")]),
        dict(pixels_measured_in_this_frame=True,additional_points_measured_in_native_pixels=True,
        uv=[[450,500],[650,500]],endpoint_evidence_transition_iqr_px=[4,4],
        point_bindings=[dict(measurement_origin="native_colour_half_transition")])])])
    assert support.derive_regions(p)==[]
    r=support.derive_regions(p,include_legacy=True)
    assert len(r)==2
    assert r[0]["polygon_xy"]==[[50.,105.],[100.,105.],[100.,496.],[50.,496.]]
    assert r[1]["polygon_xy"][0][0]==500


def test_legacy_original_bgr_bytes_checked_before_png_reference_created(tmp_path):
    image=np.full((1080,1920,3),32,np.uint8)
    ip=tmp_path/"original.png";mp=tmp_path/"mask.png"
    assert cv2.imwrite(str(ip),image) and cv2.imwrite(str(mp),np.zeros((1080,1920),np.uint8))
    p=packet();p["source"]=dict(native_pts=256256,programme_frame_index_1based=1,
        source_video_sha256="a"*64,time_base="1/12800",image_path=str(ip),
        decoded_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        mask_path=str(mp),mask_sha256=support.reference(mp)["sha256"])
    pp=tmp_path/"packet.json";pp.write_text(json.dumps(p),encoding="utf8")
    meta=dict(frame=1,native_pts=256256,source_video_sha256="a"*64,image_path=str(ip),mask_path=str(mp),
        mask_sha256=p["source"]["mask_sha256"],source_packet=support.reference(pp))
    with pytest.raises(ValueError,match="incomplete"):
        support.frame_evidence(meta)
    f=support.frame_evidence(meta,include_legacy=True)
    assert f["source_frame"]==support.reference(ip)
    image[0,0,0]+=1
    assert cv2.imwrite(str(ip),image)
    with pytest.raises(ValueError,match="BGR identity"):
        support.frame_evidence(meta,include_legacy=True)
