"""The active scene keeps the lane uncertainty and confirmed ad topology."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rinkspace.calibration_companion import active_scene as active_scene_module
from rinkspace.calibration_companion.active_scene import active_scene_path, load_active_scene
from rinkspace.calibration_companion.advertisement_topology import validate_user_confirmed_advertisements
from rinkspace.calibration_companion.scene_camera import Scene
from rinkspace.storage import resolve_asset


ROOT = Path(__file__).resolve().parents[1]
LEGACY_V023 = resolve_asset(ROOT, "outputs/rink_model/ankara_2026_shared/v0.23-high-view-ad-corners/scene.json")


def test_active_selector_loads_hashed_refit_release_without_altering_scene(tmp_path, monkeypatch):
    config = json.loads((ROOT / active_scene_module.ACTIVE_CONFIG).read_text(encoding="utf-8"))
    release_path = resolve_asset(ROOT, config["release_path"])
    release = json.loads(release_path.read_text(encoding="utf-8"))
    scene_path = active_scene_path(ROOT)
    scene = load_active_scene(ROOT)
    assert scene_path == resolve_asset(ROOT, config["model_path"])
    assert scene.sha256 == config["model_sha256"] == release["model_sha256"]
    assert hashlib.sha256(release_path.read_bytes()).hexdigest() == config["release_sha256"]
    assert scene.revision == config["revision"] == release["revision"]
    assert scene.data["schema_version"] == release["model_schema_version"]
    assert scene.data["model_id"] == release["model_id"]
    assert "revision" not in scene.data  # revision lives in the release, not the fitted scene
    assert len(scene.features) == 22
    assert len(scene.ads) == 10
    assert all(key.startswith("user_number:") for key in scene.ads)
    with pytest.raises(ValueError, match="unknown advert attachment_s"):
        scene.point({"kind": "advert_anchor", "advert_id": next(iter(scene.ads))})

    bad_config = dict(config, release_sha256="0" * 64)
    selector_path = tmp_path / "tampered_selector.json"
    selector_path.write_text(json.dumps(bad_config), encoding="utf-8")
    monkeypatch.setattr(active_scene_module, "ACTIVE_CONFIG", selector_path)
    with pytest.raises(ValueError, match="active scene release SHA-256 mismatch"):
        active_scene_path(ROOT)


def test_active_scene_is_integral_and_lane_endpoints_are_independent_unknowns():
    scene = Scene(LEGACY_V023)
    assert scene.data["revision"] == "v0.23-high-view-ad-corners"
    assert scene.data["cameras"] == scene.data["frame_cameras"] == []
    goal = scene.features["goal_line_left"]
    goal_x = float(goal["segments_xyz"][0][0][0])
    endpoints = []
    for index in range(1, 5):
        lane = scene.features[f"lane{index}"]
        outer = next(r for r in scene.features[f"F{index}"]["rings"] if r["role"] == "outer")
        f_left = outer["center_xyz"][0] - outer["radius"]
        assert lane["left_endpoint_status"] == "unresolved"
        assert lane["polygon_left_vertices_role"] == "guaranteed_core_display_proxy_not_physical_endpoints"
        sources = lane["left_endpoint_x_interval_bound_sources"]
        assert sources["lower_x"]["feature_id"] == "goal_line_left"
        assert sources["upper_x"]["feature_id"] == f"F{index}"
        assert np.isclose(sources["lower_x"]["x_scene_units"], goal_x)
        assert np.isclose(sources["upper_x"]["x_scene_units"], f_left)
        for edge, vertex in (("upper", 0), ("lower", 3)):
            interval = lane["left_endpoint_x_intervals_scene_units"][edge]
            assert np.allclose(interval, [goal_x, f_left])
            assert interval[0] < interval[1]
            assert np.isclose(lane["polygon_xyz"][vertex][0], interval[1])
            endpoints.append((f"lane{index}", edge))
    assert len(set(endpoints)) == 8
    assert scene.data["hard_geometry_checks"]["all_constraints_satisfied"] is True

    model_dir = scene.path.parent
    checks = json.loads((model_dir / "scene_checks.json").read_text(encoding="utf-8"))
    geometry_hash = hashlib.sha256((model_dir / "scene_geometry.json").read_bytes()).hexdigest()
    assert checks["scene_sha256"] == scene.sha256
    assert scene.data["source"]["geometry_json_sha256"] == geometry_hash
    assert checks["checks"]["mesh_and_grid_checks_passed"] is True
    assert checks["checks"]["physical_accuracy_verified"] is False


def test_user_confirmed_advertisement_ring_is_locked_without_fixing_physical_geometry():
    scene = Scene(LEGACY_V023).data
    contract_path = ROOT / "configs/rink_model/ankara_2026_advertisements_user_confirmed_20260924.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    validate_user_confirmed_advertisements(scene, contract, contract_sha)
    ads = scene["advertisements"]
    assert [ad["user_number"] for ad in ads] == list(range(1, 27))
    assert len({ad["id"] for ad in ads}) == 26
    assert scene["advertisement_topology"]["side_sequences"] == {
        "upper_left_to_right": [7, 6, 5, 4, 3, 2, 1],
        "left_top_to_bottom": [8, 9, 10, 11, 12, 13],
        "lower_left_to_right": [14, 15, 16, 17, 18, 19, 20],
        "right_top_to_bottom": [26, 25, 24, 23, 22, 21],
    }
    assert scene["advertisement_topology"]["schematic_central_symmetry"]["pair_numbers"] == [
        [number, number + 13] for number in range(1, 14)
    ]
    for first in range(1, 14):
        assert ads[first - 1]["visual_content_id"] == ads[first + 12]["visual_content_id"]
        assert ads[first - 1]["id"] != ads[first + 12]["id"]
    assert ads[8]["review_ppt_label"] == "can 红徽"
    assert ads[21]["review_ppt_label"] == "红徽"
    for ad in ads:
        assert all(ad[key] is None for key in (
            "attachment_s", "board_panel_id", "width_along_board", "height", "height_above_ice"
        ))
        assert ad["anchor_xyz"] == [None, None, None]
    for ad, position in zip(ads, contract["positions"]):
        assert ad["schematic_anchor_shape_name"] == position["ppt_anchor_shape_name"]
        assert ad["schematic_anchor_center_emu"] == position["schematic_anchor_center_emu"]

    changed_identity = copy.deepcopy(scene)
    changed_identity["advertisements"][21]["visual_content_id"] = "unrelated_logo"
    with pytest.raises(ValueError, match="visual_content_id"):
        validate_user_confirmed_advertisements(changed_identity, contract, contract_sha)

    changed_layout = copy.deepcopy(scene)
    changed_layout["advertisements"][0]["schematic_anchor_center_emu"][0] += 1000
    with pytest.raises(ValueError, match="schematic_anchor_center_emu"):
        validate_user_confirmed_advertisements(changed_layout, contract, contract_sha)

    future_fit = copy.deepcopy(scene)
    future_fit["advertisements"][0]["attachment_s"] = 12.0
    future_fit["advertisements"][0]["width_along_board"] = 2.0
    validate_user_confirmed_advertisements(future_fit, contract, contract_sha)


def test_high_view_advertisement_corners_are_source_bound_and_not_live_geometry():
    scene = Scene(LEGACY_V023).data
    observation_path = resolve_asset(ROOT, scene["source"]["high_view_ad_corner_observations"])
    record = json.loads(observation_path.read_text(encoding="utf-8"))
    record_sha = hashlib.sha256(observation_path.read_bytes()).hexdigest()
    assert record_sha == scene["source"]["high_view_ad_corner_observations_sha256"]
    assert record["source_frame"]["source_pts"] == 56218880
    assert record["source_frame"]["time_base"] == "1/15360"
    assert [row["user_number"] for row in record["advertisements"]] == list(range(1, 8))
    assert sum(len(row["corner_observations"]) for row in record["advertisements"]) == 26
    assert [row["board_segment"] for row in record["advertisements"]] == [
        "upper_right", "upper_long_side", "upper_long_side", "upper_long_side",
        "upper_long_side", "upper_long_side", "upper_left",
    ]
    for row in record["advertisements"]:
        for point in row["corner_observations"]:
            assert point["source_frame_uv"] == pytest.approx([
                0.96 * value - 0.02 for value in point["annotated_image_uv"]
            ], abs=0.0001)
            assert point["conditional_ray_board_intersection"]["valid_on_finite_board"]
        assert scene["advertisements"][row["user_number"] - 1]["high_view_corner_evidence"]["record_sha256"] == record_sha
    first = record["advertisements"][0]
    fifth = record["advertisements"][4]
    assert first["fit"]["extent_candidate"]["width_along_board_scene_units"] is None
    assert fifth["fit"]["extent_scope"] == "internal_graphic_only"
    assert fifth["observed_feature_scope"] == "internal_graphic_rectangle_not_ad_extent"
    assert all(ad["attachment_s"] is None and ad["width_along_board"] is None and ad["height"] is None
               for ad in scene["advertisements"])
    assert max(row["fit"]["max_corner_reprojection_source_px"] for row in record["advertisements"]) < 2


def test_frozen_v020_scene_is_unchanged():
    frozen = resolve_asset(ROOT, "outputs/rink_model/ankara_2026_shared/v0.20-wide-A-pair-upper-arc/scene.json")
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == (
        "9fa6c4abbbf8665ce6131ce5a8387a9db523e51bf2df8d2a4ce68ec94246a03b"
    )
