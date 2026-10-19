"""Admission and consumption of authenticated official-source observations."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from calibration.program_tracking import official_static_surface_dataset as dataset


@pytest.fixture
def empty_mask():
    return np.zeros((1080, 1920), bool)


def measured_segment(**changes):
    value = dict(uv=[[100.125, 120.5], [120.25, 140.75]],
                 pixels_measured_in_this_frame=True, sigma_px=[0.3, 0.7])
    value.update(changes)
    return value


@pytest.mark.parametrize("feature,descriptor", [
    ({"physical_surface": "stands"}, {"kind": "board_edge"}),
    ({"physical_surface": "judge_table"}, {"kind": "ice_curve"}),
    ({"physical_surface": "building"}, {"kind": "arena_texture"}),
    ({}, {}),
])
def test_outside_or_unknown_surface_never_admitted(feature, descriptor, empty_mask):
    _, decisions, _, _ = dataset.classify_segment(
        feature, measured_segment(), descriptor, empty_mask)
    assert decisions == ["outside_or_unconfirmed_model_scope"] * 2


@pytest.mark.parametrize("feature,segment", [
    ({"projected_uv_used_as_observations": True}, measured_segment()),
    ({}, measured_segment(projected_uv_used_as_observations=True)),
    ({}, measured_segment(hidden_span_filled=True)),
])
def test_projection_and_hidden_span_do_not_become_fit(feature, segment, empty_mask):
    _, decisions, _, _ = dataset.classify_segment(
        feature, segment, {"kind": "board_edge"}, empty_mask)
    assert decisions == ["inferred_or_projected_pixel"] * 2


def test_current_source_evidence_required_despite_scope_and_sigma(empty_mask):
    _, decisions, _, _ = dataset.classify_segment(
        {}, measured_segment(pixels_measured_in_this_frame=False),
        {"kind": "board_edge"}, empty_mask)
    assert decisions == ["no_current_source_measurement_evidence"] * 2


def test_timer_halo_and_person_mask_are_excluded_before_shared_evidence(empty_mask):
    segment = measured_segment(uv=[[100, 120], [121, 140], [130, 160]],
                               sigma_px=[0.3, 0.7, 0.9],
                               representative_points_share_line_estimate=True)
    empty_mask[140, 121] = True
    timer = dict(actual_native_rectangle_xyxy_inclusive=[101, 121, 110, 130],
                 edge_halo_px=1)
    _, decisions, _, _ = dataset.classify_segment(
        {}, segment, {"kind": "board_edge"}, empty_mask, timer=timer)
    assert decisions == ["broadcast_graphic", "foreground_mask_conflict", "shared_line_evidence"]
    # A material review can resolve a conservative person mask, never subtitles.
    _, decisions, _, _ = dataset.classify_segment(
        {}, segment, {"kind": "board_edge"}, empty_mask,
        reviewed_mask_vertices=[0, 1], timer=timer)
    assert decisions == ["broadcast_graphic", "shared_line_evidence", "shared_line_evidence"]


@pytest.mark.parametrize("changes,kind", [
    ({"representative_points_share_line_estimate": True}, "board_edge"),
    ({"inferred_from_current_source_readouts": True}, "board_edge"),
    ({"measurement_origin": "current_source_robust_line_intersection"}, "ad_corner"),
    ({"measurement_origin": "two_source_measured_straight_edge_intersections"}, "ad_corner"),
])
def test_shared_line_and_derived_corner_are_evidence_not_independent_fit(changes, kind, empty_mask):
    _, decisions, sigma, _ = dataset.classify_segment(
        {}, measured_segment(**changes), {"kind": kind}, empty_mask)
    assert decisions == ["shared_line_evidence"] * 2
    np.testing.assert_array_equal(sigma, [0.3, 0.7])


def test_missing_scales_retain_nan_and_endpoint_evidence_only(empty_mask):
    segment = measured_segment(uv=[[100, 120], [110, 130], [120, 140]])
    segment.pop("sigma_px")
    sigma, _ = dataset.source_scales(segment)
    assert np.isnan(sigma).all()
    segment["endpoint_evidence_localization_sigma_px"] = [0.2, 0.6]
    sigma, _ = dataset.source_scales(segment)
    np.testing.assert_array_equal(sigma[[0, 2]], [0.2, 0.6])
    assert np.isnan(sigma[1])
    _, decisions, _, _ = dataset.classify_segment(
        {}, segment, {"kind": "board_edge"}, empty_mask)
    assert decisions == ["fit", "missing_source_scale", "fit"]


def test_per_vertex_scales_preserved_and_invalid_values_excluded(empty_mask):
    segment = measured_segment(uv=[[100, 120], [110, 130], [120, 140], [130, 150]],
                               sigma_px=[0.25, 0.0, np.inf, -2.0])
    _, decisions, sigma, field = dataset.classify_segment(
        {}, segment, {"kind": "board_edge"}, empty_mask)
    assert field == "sigma_px"
    assert decisions == ["fit"] + ["missing_source_scale"] * 3
    np.testing.assert_array_equal(sigma, segment["sigma_px"])
    segment.pop("sigma_px")
    segment["point_bindings"] = [{"localization_sigma_px": 0.4}, {},
                                  {"localization_sigma_px": 0.8}, {}]
    sigma, _ = dataset.source_scales(segment)
    np.testing.assert_array_equal(sigma[[0, 2]], [0.4, 0.8])
    assert np.isnan(sigma[[1, 3]]).all()


def test_original_opaque_point_ids_not_reconstructed_from_address():
    segment = measured_segment(point_ids=["opaque user address: 17", "not/a/vertex/address"])
    assert dataset.source_point_id("board.top", 5, segment, 0) == "opaque user address: 17"
    assert dataset.source_point_id("board.top", 5, segment, 1) == "not/a/vertex/address"
    with pytest.raises(ValueError, match="opaque point identities"):
        dataset.source_point_id("board.top", 5, measured_segment(point_ids=["one"]), 0)


def save_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf8")
    return dataset.reference(path)


@pytest.fixture
def official_fixture(tmp_path):
    """Small authentic-file graph; no real project observations are opened."""
    video = tmp_path / "official.mp4"
    video.write_bytes(b"test official native source identity")
    video_ref = dataset.reference(video)
    source = dict(path=video_ref["path"], sha256=video_ref["sha256"],
                  native_fps="50/1", time_base="1/12800", native_pts_step=256)
    primary = save_json(tmp_path / "primary.json", dict(primary_source=source,
        processing_policy="ISU_official_single_programme_native_50fps_PTS"))
    q = np.asarray([0.125, -0.25], np.float64)
    checkpoint_path = tmp_path / "checkpoint.npz"
    np.savez(checkpoint_path, q=q)
    q_sha = hashlib.sha256(q.tobytes()).hexdigest()
    identity = save_json(tmp_path / "identity.json", dict(q_sha256=q_sha, parameter_count=len(q)))
    scene = save_json(tmp_path / "scene.json", dict(board_surface={
        "top_edge_xyz": [[0, 0, 1], [10, 0, 1], [10, 10, 1], [0, 10, 1]]}))
    descriptor = dict(kind="board_edge", role="top")
    registry = save_json(tmp_path / "registry.json", dict(targets={"board.top": descriptor}))
    templates = save_json(tmp_path / "templates.json", {})
    image = np.zeros((1080, 1920, 3), np.uint8)
    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), image)
    mask = np.zeros((1080, 1920), np.uint8)
    mask[141, 120] = 255
    mask_path = tmp_path / "mask.png"
    assert cv2.imwrite(str(mask_path), mask)
    segments = [
        measured_segment(point_ids=["opaque: a", "opaque: mask"]),
        measured_segment(uv=[[300, 320], [320, 340]], sigma_px=[0.2, 0.8],
                         representative_points_share_line_estimate=True,
                         point_ids=["opaque: line 1", "opaque: line 2"]),
        measured_segment(uv=[[400, 420], [420, 440]], point_ids=["opaque: missing 1", "opaque: missing 2"]),
        measured_segment(uv=[[500, 520], [520, 540]], hidden_span_filled=True,
                         point_ids=["opaque: hidden 1", "opaque: hidden 2"]),
        measured_segment(uv=[[600, 620], [620, 640]], point_ids=["opaque: timer 1", "opaque: timer 2"]),
    ]
    segments[2].pop("sigma_px")
    feature = dict(id="board_top", physical_surface="board", segments=segments)
    native = dict(source_video_sha256=video_ref["sha256"], native_pts=256256,
                  actual_pts=256256, time_base="1/12800", source_time="00:20.020",
                  image_path=str(image_path), decoded_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                  mask_path=str(mask_path), mask_sha256=dataset.reference(mask_path)["sha256"],
                  broadcast_timer_exclusion={"actual_native_rectangle_xyxy_inclusive": [590, 610, 630, 650]})
    packet = save_json(tmp_path / "packet.json", dict(source=native, features=[feature],
        point_count=10, observation_coordinates_from_projection=False, user_accepted=False,
        inferred_segments=[dict(uv=[[700, 720], [720, 740]])]))
    observations = [dict(id=f"observation-{i}", source_feature_id=feature["id"],
        source_segment_index=i, source_vertex_indices=[0, 1], source_point_ids=s["point_ids"],
        target_id="board.top", point_correspondence=False, dependence_group=f"group-{i}")
        for i, s in enumerate(segments)]
    binding = save_json(tmp_path / "binding.json", dict(model_id="test-model", source_packet=packet, source_pts=256256,
        source_time_base="1/12800", observations=observations))
    acceptance = save_json(tmp_path / "acceptance.json", dict(accepted=True, source_time_base="1/12800",
        frames=[dict(index=1, packet_sha256=packet["sha256"], binding_sha256=binding["sha256"])]))
    batch = save_json(tmp_path / "batch.json", dict(frames=[dict(index=1, packet=packet, model_binding=binding)]))
    model = save_json(tmp_path / "model.json", dict(model_id="test-model",
        parent=dict(q_sha256=q_sha, state_identity=identity, complete_state=dataset.reference(checkpoint_path)),
        scene=scene, feature_registry=registry, material_templates=templates,
        accepted_batch_datasets=[dict(**batch, user_acceptance=acceptance)]))
    return dict(model=model, primary=primary, packet=packet, binding=binding,
                selection_path=tmp_path / "selection.json")


def test_frozen_partition_and_real_residual_consumer_preserve_original_evidence(official_fixture):
    f = official_fixture
    original_bytes = {name: Path(f[name]["path"]).read_bytes() for name in ("model", "packet", "binding")}
    reference, selection = dataset.build_dataset(f["model"]["path"], f["primary"]["path"], f["selection_path"])
    assert selection["counts"] == dict(fit=1, foreground_mask_conflict=1, shared_line_evidence=2,
        missing_source_scale=2, inferred_or_projected_pixel=2, broadcast_graphic=2,
        top_level_inferred_vertices_excluded=2)
    assert selection["historical_pixels_imported"] == 0
    assert selection["legacy_objectives_enabled"] is False
    assert selection["unknown_depth_points_admitted"] == 0
    rows = list(dataset.iter_fit_rows(reference))
    assert len(rows) == 1
    np.testing.assert_array_equal(rows[0]["uv"], [[100.125, 120.5]])
    np.testing.assert_array_equal(rows[0]["sigma"], [0.3])
    assert rows[0]["source_point_ids"] == ["opaque: a"]
    assert rows[0]["native_pts"] == rows[0]["actual_pts"] == 256256
    assert rows[0]["time_base"] == "1/12800"
    assert rows[0]["diagnostic_unit_weights"] is False
    shared = list(dataset.iter_observations(reference, role="shared_line_evidence"))
    assert shared[0]["source_point_ids"] == ["opaque: line 1", "opaque: line 2"]
    missing = list(dataset.iter_observations(reference, role="missing_source_scale"))
    assert np.isnan(missing[0]["sigma"]).all()
    for name, value in original_bytes.items():
        assert Path(f[name]["path"]).read_bytes() == value


def test_source_reference_mutation_fails_before_observation_consumption(official_fixture):
    f = official_fixture
    reference, _ = dataset.build_dataset(f["model"]["path"], f["primary"]["path"], f["selection_path"])
    with open(f["packet"]["path"], "a", encoding="utf8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="Source reference changed"):
        list(dataset.iter_observations(reference))


def test_legacy_objective_cannot_be_enabled_in_consumption(official_fixture):
    f = official_fixture
    _, selection = dataset.build_dataset(f["model"]["path"], f["primary"]["path"], f["selection_path"])
    tampered = deepcopy(selection)
    tampered["legacy_objectives_enabled"] = True
    ref = save_json(f["selection_path"].with_name("legacy-enabled.json"), tampered)
    with pytest.raises(ValueError, match="scope-clean official"):
        list(dataset.iter_observations(ref))


def test_unknown_named_target_cannot_fall_back_to_the_whole_board(official_fixture):
    f = official_fixture
    model = dataset.read_reference(f["model"])
    registry = dataset.read_reference(model["feature_registry"])
    registry["targets"]["board.top"]["role"] = "judge_table"
    model["feature_registry"] = save_json(Path(model["feature_registry"]["path"]), registry)
    model_ref = save_json(Path(f["model"]["path"]), model)
    with pytest.raises(ValueError, match="Unknown board edge role"):
        dataset.build_dataset(model_ref["path"], f["primary"]["path"], f["selection_path"])
    assert not f["selection_path"].exists()


def test_relabelled_masked_point_is_rechecked_at_consumption(official_fixture):
    f = official_fixture
    _, selection = dataset.build_dataset(f["model"]["path"], f["primary"]["path"], f["selection_path"])
    tampered = deepcopy(selection)
    tampered["frames"][0]["observations"][0]["selected_vertex_indices"]["fit"] = [1]
    ref = save_json(f["selection_path"].with_name("masked-relabelled.json"), tampered)
    with pytest.raises(ValueError, match="physical/evidence admission"):
        list(dataset.iter_fit_rows(ref))


def test_sixty_fps_replay_cannot_be_reused_as_the_processing_source(official_fixture):
    f = official_fixture
    primary = dataset.read_reference(f["primary"])
    primary["primary_source"].update(native_fps="60/1", time_base="1/15360")
    ref = save_json(Path(f["primary"]["path"]), primary)
    with pytest.raises(ValueError, match="Official native 50fps source required"):
        dataset.build_dataset(f["model"]["path"], ref["path"], f["selection_path"])
    assert not f["selection_path"].exists()


def assert_exact_accepted_frame_replacement(frame, packet_ref, packet):
    """Compare consumed UV addresses, not a manifest's claimed point count."""
    features = {feature["id"]: feature for feature in packet["features"]}
    expected = {(feature["id"], si, vi)
                for feature in packet["features"]
                for si, segment in enumerate(feature["segments"])
                for vi in range(len(segment["uv"]))}
    seen, seen_ids = set(), set()
    for row in frame["rows"]:
        assert row["source_packet"] == packet_ref
        fid, si = row["source_feature_id"], row["source_segment_index"]
        segment = features[fid]["segments"][si]
        assert len(row["uv"]) == len(row["source_point_ids"]) == len(row["original_vertex_indices"])
        for uv, pid, vi in zip(row["uv"], row["source_point_ids"], row["original_vertex_indices"], strict=True):
            address = (fid, si, vi)
            assert address in expected and address not in seen
            assert pid == dataset.source_point_id(fid, si, segment, vi) and pid not in seen_ids
            np.testing.assert_array_equal(uv, segment["uv"][vi])
            seen.add(address)
            seen_ids.add(pid)
    assert seen == expected
    assert len(seen) == frame["source_point_count"] == packet["point_count"]
    assert frame["native_pts"] == packet["source"]["native_pts"]
