import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from rinkspace.vision_source_shapes import SourceMaterialShapeReader, read_source_material_transition_profiles


def bound(rgb):
    return dict(source_sha256="a"*64, source_pts=256, time_base="1/15360",
                shot_id="source_material_test", native_hw=list(rgb.shape[:2]),
                decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())


def read(rgb, *, point=(64, 48), interval=(-38, 38), normal=(0, 1), transition="dark_to_light"):
    return read_source_material_transition_profiles(rgb, binding=bound(rgb),
        search_points_uv=[point], normals=[normal], search_intervals_px=[interval],
        transition=transition, keep_profiles=True)


def test_blurred_neutral_boundary_has_luminance_measurement_with_wide_real_support():
    rgb = np.full((96, 128, 3), 140, np.uint8)
    rgb[48:] = 220
    rgb = cv2.GaussianBlur(rgb, (0, 0), 6.)
    # The former blue-minus-red proxy contains no signal at this actual edge.
    assert not np.any(rgb[:, :, 2].astype(int)-rgb[:, :, 0])
    result = read(rgb, point=(64, 42))
    row = result["readings"][0]
    assert row["disposition"] == "measured_material_transition"
    assert abs(result["actual_source_uv"][0, 1]-47.5) < 1
    assert row["observed_normal_interval_px"][1]-row["observed_normal_interval_px"][0] > 12
    assert not row["weak_or_wide_used_as_hard_veto"]
    assert not result["world_camera_projection_input_used"]
    assert result["physical_id"] is None


def test_two_gray_to_white_edges_do_not_choose_the_one_nearest_a_search_seed():
    rgb = np.full((96, 128, 3), 145, np.uint8)
    rgb[29:49] = 220
    rgb[68:] = 220
    rgb = cv2.GaussianBlur(rgb, (0, 0), 1.4)
    result = read(rgb, point=(64, 32), interval=(-18, 54))
    row = result["readings"][0]
    assert row["source_flags"]["competing_material_reversal"]
    assert row["disposition"] == "source_material_role_unknown"
    assert result["actual_source_uv"].shape == (0, 2)
    assert np.isfinite(row["actual_source_transition_candidate_uv"]).all()
    assert len(row["observed_material_intervals"]) >= 2


def test_cutting_a_transition_or_the_native_frame_keeps_uncertain_candidate():
    rgb = np.full((96, 128, 3), 140, np.uint8)
    rgb[48:] = 220
    rgb = cv2.GaussianBlur(rgb, (0, 0), 5.)
    row = read(rgb, interval=(-3, 4))["readings"][0]
    assert row["source_flags"]["search_interval_truncated"]
    assert row["disposition"] == "source_material_role_unknown"
    row = read(rgb, point=(1, 48))["readings"][0]
    assert row["source_flags"]["source_image_border_truncated"]
    assert row["disposition"] == "source_material_role_unknown"


def test_native_rgb_identity_cannot_be_replaced_and_polarity_must_be_explicit():
    rgb = np.full((96, 128, 3), 150, np.uint8)
    binding = bound(rgb)
    rgb[0, 0] = 0
    with pytest.raises(ValueError, match="bound original RGB differ"):
        read_source_material_transition_profiles(rgb, binding=binding,
            search_points_uv=[[64, 48]], normals=[[0, 1]],
            search_intervals_px=[[-20, 20]], transition="dark_to_light")
    with pytest.raises(ValueError, match="explicit signed"):
        read(rgb, transition="blue_minus_red")


def test_existing_shape_reader_material_path_requires_actual_rgb_and_keeps_memory_separate():
    rgb = np.full((96, 128, 3), 145, np.uint8)
    rgb[48:] = 220
    rgb = cv2.GaussianBlur(rgb, (0, 0), 3.)
    reader = SourceMaterialShapeReader()
    packet = dict(source_binding=bound(rgb), RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
    result = reader.read_material_transition_profiles(rgb, packet, search_points_uv=[[64, 48]],
        normals=[[0, 1]], search_intervals_px=[[-30, 30]], transition="dark_to_light")
    assert len(result["actual_source_uv"]) == 1 and reader.previous is None
    assert not result["physical_edge_role_verified"]
    with pytest.raises(ValueError, match="bound source RGB ledger"):
        reader.read_material_transition_profiles(rgb, {"source_binding": bound(rgb)})


_REAL_SOURCE = Path("D:/CodexData/.codex/visualizations/2026/10/02/"
    "private-thread-00008/RinkSpace_tmp/low5_repair/core_ice_source")


@pytest.mark.skipif(not (_REAL_SOURCE/"230365184_raw.png").exists(),
                    reason="Original Ankara native RGB evidence is not installed")
def test_two_real_exposures_recover_gray_disc_interface_without_colour_proxy_or_model():
    cases = [
        # These brackets and validation intervals were independently inspected
        # on original material layers. No world/camera prediction is supplied.
        (230365184, "cdcf7d67a0e9abc7bf9fdad931623d6db87c20e9c031e8b7744f310d0b090bba",
         [[315, 860], [340, 864]], [[-14, 17], [-11, 11]],
         [[848.283552011, 873.049269530], [856.581301257, 873.187077044]],
         [896.507122451, 889.388532504]),
        (230359040, "ee8ee12cf7b37a6aeca00b12361a91159e5948de8416672a6f9e36ea1f746a5a",
         [[850, 862]], [[-14, 14]], [[850, 872]], None),
    ]
    for pts, digest, points, intervals, validated_support, wrong_proxy_y in cases:
        path = _REAL_SOURCE/f"{pts}_raw.png"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        binding = {**bound(rgb), "source_sha256": "f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891",
                   "source_pts": pts, "shot_id": "programme_shot_05"}
        result = SourceMaterialShapeReader().read_material_transition_profiles(rgb,
            dict(source_binding=binding, RGB_sha256=binding["decoded_native_rgb_sha256"]),
            search_points_uv=points, normals=np.tile([0, 1], (len(points), 1)),
            search_intervals_px=intervals, transition="dark_to_light")
        assert len(result["actual_source_uv"]) == len(points)
        assert not result["world_camera_projection_input_used"]
        for row, support in zip(result["readings"], validated_support):
            assert support[0] <= row["actual_source_uv"][1] <= support[1]
        if wrong_proxy_y:
            for row, wrong in zip(result["readings"], wrong_proxy_y):
                y = row["actual_source_uv"][1]
                assert wrong > y+row["observed_normal_interval_px"][1]+6
