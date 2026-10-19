"""Actual correlated FIT consumption and accepted-source replacement guards."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from calibration.program_tracking import complete_official_fit_dataset as complete
from calibration.program_tracking import official_static_surface_dataset as scope
from calibration.program_tracking.framewise_model import evaluate_rows
from calibration.program_tracking.source_evidence_whitening import SourceLineEvidence, line_whitening_operator
from test_official_static_surface_dataset import assert_exact_accepted_frame_replacement


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf8")
    return scope.reference(path)


def source_camera():
    camera = np.zeros(25, np.float64)
    camera[:3] = [0, 0, 10]
    camera[3] = np.pi
    camera[6:8] = np.log(1000)
    camera[8:10] = [960, 540]
    return camera


def evaluated_frame():
    ordinary = dict(id="direct", target={"kind": "board_edge", "role": "top"},
        world=np.array([[-1, 0, 0], [1, 0, 0]], float), uv=np.array([[910, 546], [1010, 548]], float),
        sigma=np.array([0.4, 1.7]), point_correspondence=False,
        source_point_ids=["native a", "native b"])
    shared = dict(id="line", target={"kind": "board_edge", "role": "top"},
        world=np.array([[-1, 0, 0], [1, 0, 0]], float),
        uv=np.array([[890, 551], [940, 553.5], [1010, 557]], float),
        sigma=np.array([1.2, 2.1, 0.75]), point_correspondence=False,
        source_point_ids=["opaque representative a", "opaque representative b", "opaque representative c"])
    line = SourceLineEvidence([0.05, 506.5], [[1e-5, -0.009], [-0.009, 9]], 0,
                              "current-source-line", "measured_source_covariance")
    operator = line_whitening_operator(shared["uv"], shared["source_point_ids"], line)
    assert operator.rank == 2
    return dict(native_pts=256256, rows=[ordinary, shared], source_point_count=5,
                whitening_groups=[dict(operator=operator, point_indices=[2, 3, 4])])


def test_actual_vector_residual_and_jacobian_use_the_same_fixed_whitener():
    frame, camera = evaluated_frame(), source_camera()
    original_uv = [r["uv"].copy() for r in frame["rows"]]
    original_sigma = [r["sigma"].copy() for r in frame["rows"]]
    raw_r, raw_j = evaluate_rows(camera, frame["rows"])
    operator = frame["whitening_groups"][0]["operator"]
    scales = np.repeat(np.concatenate([r["sigma"] for r in frame["rows"]]), 2)
    expected_r = np.r_[operator.matrix @ (raw_r[4:] * scales[4:]), raw_r[:4]]
    expected_j = np.vstack((operator.matrix @ (raw_j[4:] * scales[4:, None]), raw_j[:4]))
    residual, jacobian = complete.evaluate_frame(camera, frame)
    np.testing.assert_allclose(residual, expected_r, rtol=0, atol=1e-12)
    np.testing.assert_allclose(jacobian, expected_j, rtol=0, atol=1e-12)
    np.testing.assert_allclose(complete.evaluate_frame(camera, frame, jacobian=False), residual, atol=1e-12)
    assert residual.shape == (6,) and jacobian.shape == (6, 15)
    assert operator.consumed_point_count == 3
    for row, uv, sigma in zip(frame["rows"], original_uv, original_sigma, strict=True):
        np.testing.assert_array_equal(row["uv"], uv)
        np.testing.assert_array_equal(row["sigma"], sigma)


def test_whitened_jacobian_matches_actual_camera_finite_differences():
    frame, camera = evaluated_frame(), source_camera()
    _, jacobian = complete.evaluate_frame(camera, frame)
    state = complete.camera_fit_state(camera)
    for column in (1, 3, 4, 5, 6, 7, 8, 9, 11, 12):
        step = 2e-6
        before, after = state.copy(), state.copy()
        before[column] -= step
        after[column] += step
        difference = (complete.evaluate_frame(complete.camera_from_fit_state(after, camera), frame, False)
                      - complete.evaluate_frame(complete.camera_from_fit_state(before, camera), frame, False)) / (2 * step)
        np.testing.assert_allclose(jacobian[:, column], difference, rtol=2e-5, atol=2e-5)


def test_multiple_operators_cannot_count_the_same_source_scalar_twice():
    frame = evaluated_frame()
    frame["whitening_groups"].append(frame["whitening_groups"][0])
    with pytest.raises(ValueError, match="multiple FIT operators"):
        complete.evaluate_frame(source_camera(), frame)


@pytest.fixture(scope="module")
def accepted_replacement(tmp_path_factory):
    """200 receipt-bound synthetic frames exercise the real preparation route."""
    tmp_path = tmp_path_factory.mktemp("complete_official_replacement")
    video = tmp_path / "source.mp4"
    video.write_bytes(b"authenticated synthetic official50 source")
    video_ref = scope.reference(video)
    primary = write_json(tmp_path / "primary.json", dict(
        primary_source=dict(**video_ref, native_fps="50/1", time_base="1/12800", native_pts_step=256),
        processing_policy="ISU_official_single_programme_native_50fps_PTS"))
    image = np.zeros((1080, 1920, 3), np.uint8)
    image_path, mask_path = tmp_path / "image.png", tmp_path / "mask.png"
    assert cv2.imwrite(str(image_path), image)
    assert cv2.imwrite(str(mask_path), image[:, :, 0])
    mask_ref = scope.reference(mask_path)
    q = np.asarray([0.125, -0.25], np.float64)
    checkpoint_path = tmp_path / "checkpoint.npz"
    np.savez(checkpoint_path, q=q)
    q_sha = hashlib.sha256(q.tobytes()).hexdigest()
    identity = write_json(tmp_path / "identity.json", dict(q_sha256=q_sha, parameter_count=2))
    scene = write_json(tmp_path / "scene.json", dict(board_surface={
        "top_edge_xyz": [[-1, 0, 0], [1, 0, 0], [1, 1, 0], [-1, 1, 0]]}))
    registry = write_json(tmp_path / "registry.json", dict(targets={
        "board.top": {"kind": "board_edge", "role": "top"}}))
    templates = write_json(tmp_path / "templates.json", {})
    records, receipts, recoveries, packets = [], [], [], []
    for index in range(1, 201):
        pts = 256000 + 256 * index
        source = dict(native_pts=pts, actual_pts=pts, time_base="1/12800", source_time=f"{pts / 12800:.3f}",
            source_video_sha256=video_ref["sha256"], image_path=str(image_path),
            decoded_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            mask_path=str(mask_path), mask_sha256=mask_ref["sha256"])
        features = [
            dict(id="direct_source", physical_surface="board", segments=[dict(
                uv=[[300.125, 320.5], [320.25, 340.75]], sigma_px=[0.4, 0.8],
                pixels_measured_in_this_frame=True, point_ids=["opaque direct a", "opaque direct b"])]),
            dict(id="board_top", physical_surface="board", source_line_fit=dict(
                coefficient_y_equals_ax_plus_b=[0.05, 530], covariance_ab=[[1e-5, -0.009], [-0.009, 9]],
                current_frame_only=True, old_model_projection_used_as_output_points=False),
                segments=[dict(uv=[[400, 550], [600, 560], [1000, 580]],
                    representative_points_share_line_estimate=True, source_evidence_measured_in_this_frame=True,
                    point_ids=["opaque shared a", "opaque shared b", "opaque shared c"])]),
            dict(id="recovered_source", physical_surface="board", segments=[dict(
                uv=[[700.25, 720.125]], pixels_measured_in_this_frame=True, point_ids=["opaque recovered"])]),
        ]
        packet = dict(source=source, features=features, point_count=6,
                      observation_coordinates_from_projection=False, user_accepted=False)
        packet_ref = write_json(tmp_path / f"packet_{index:03d}.json", packet)
        observations = [dict(id=f"observation_{i}", source_feature_id=f["id"], source_segment_index=0,
            source_vertex_indices=list(range(len(f["segments"][0]["uv"]))),
            source_point_ids=f["segments"][0]["point_ids"], target_id="board.top",
            point_correspondence=False, dependence_group=f["id"]) for i, f in enumerate(features)]
        binding_ref = write_json(tmp_path / f"binding_{index:03d}.json", dict(model_id="test-model",
            source_packet=packet_ref, source_pts=pts, source_time_base="1/12800", observations=observations))
        records.append(dict(index=index, packet=packet_ref, model_binding=binding_ref))
        receipts.append(dict(index=index, packet_sha256=packet_ref["sha256"], binding_sha256=binding_ref["sha256"]))
        recoveries.append(dict(native_pts=pts, source_feature_id="recovered_source", source_segment_index=0,
            source_vertex_index=0, source_point_id="opaque recovered", source_packet=packet_ref,
            uv=[700.25, 720.125], sigma_px=1.75))
        packets.append((packet_ref, packet))
    acceptance = write_json(tmp_path / "acceptance.json", dict(accepted=True, source_time_base="1/12800", frames=receipts))
    batch = write_json(tmp_path / "batch.json", dict(frames=records))
    model = write_json(tmp_path / "model.json", dict(model_id="test-model",
        parent=dict(q_sha256=q_sha, state_identity=identity, complete_state=scope.reference(checkpoint_path)),
        scene=scene, feature_registry=registry, material_templates=templates,
        accepted_batch_datasets=[dict(**batch, user_acceptance=acceptance)]))
    selection_ref, _ = scope.build_dataset(model["path"], primary["path"], tmp_path / "selection.json")
    scale_ref = write_json(tmp_path / "scales.json", dict(records=recoveries))
    complete_ref = complete.prepare_complete_dataset(selection_ref, scale_ref, tmp_path / "complete.json")
    return dict(reference=complete_ref, packets=packets, selection=selection_ref, model=model, tmp_path=tmp_path)


def test_real_consumer_replaces_every_accepted_address_and_preserves_recovered_scales(accepted_replacement):
    f = accepted_replacement
    frames = list(complete.iter_fit_frames(f["reference"]))
    assert len(frames) == 200
    assert sum(frame["source_point_count"] for frame in frames) == 1200
    for frame, (packet_ref, packet) in zip(frames, f["packets"], strict=True):
        assert_exact_accepted_frame_replacement(frame, packet_ref, packet)
        assert len(frame["whitening_groups"]) == 1
        assert frame["whitening_groups"][0]["operator"].rank == 2
        assert frame["whitening_groups"][0]["operator"].consumed_point_count == 3
        assert frame["rows"][0]["sigma"].tolist() == [0.4, 0.8]
        assert frame["rows"][2]["sigma"].tolist() == [1.75]
        assert np.isnan(scope.source_scales(packet["features"][2]["segments"][0])[0]).all()


@pytest.mark.parametrize("mutation", ["foreign_packet", "dropped_point", "duplicate_point"])
def test_replacement_frame_rejects_foreign_missing_or_duplicate_active_points(accepted_replacement, mutation):
    f = accepted_replacement
    manifest = scope.read_reference(f["reference"])
    frame = next(complete.iter_fit_frames(f["reference"]))
    rows = deepcopy(frame["rows"])
    if mutation == "foreign_packet":
        rows[0]["source_packet"] = {"path": "old standalone frame200.json", "sha256": "a" * 64}
    elif mutation == "dropped_point":
        rows[0]["source_point_ids"] = rows[0]["source_point_ids"][:-1]
    else:
        rows[0]["source_point_ids"][1] = rows[0]["source_point_ids"][0]
    with pytest.raises(ValueError, match="foreign|lost or duplicated"):
        complete._fit_frame(manifest["frames"][0], rows, frame["scene"], frame["templates"])


@pytest.mark.parametrize("mutation", ["duplicate_frame", "drop_frame", "foreign_frame_packet"])
def test_frame_manifest_cannot_append_drop_or_replace_an_accepted_frame(accepted_replacement, mutation):
    f = accepted_replacement
    manifest = scope.read_reference(f["reference"])
    if mutation == "duplicate_frame":
        manifest["frames"].append(deepcopy(manifest["frames"][0]))
    elif mutation == "drop_frame":
        manifest["frames"].pop()
    else:
        manifest["frames"][0]["source_packet"] = {"path": "old standalone frame200.json", "sha256": "a" * 64}
    ref = write_json(f["tmp_path"] / f"{mutation}.json", manifest)
    with pytest.raises(ValueError):
        list(complete.iter_fit_frames(ref))


def test_negative_group_point_address_cannot_double_consume_a_regular_scalar(accepted_replacement):
    f = accepted_replacement
    manifest = scope.read_reference(f["reference"])
    frame = next(complete.iter_fit_frames(f["reference"]))
    record = deepcopy(manifest["frames"][0])
    group = record["whitening_groups"][0]
    group["point_indices"] = [i - frame["source_point_count"] for i in group["point_indices"]]
    with pytest.raises(ValueError):
        complete._fit_frame(record, frame["rows"], frame["scene"], frame["templates"])
