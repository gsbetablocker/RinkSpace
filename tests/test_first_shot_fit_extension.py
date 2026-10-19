"""The accepted first shot extends FIT without changing its first 200 frames."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from calibration.program_tracking import complete_official_fit_dataset as complete
from calibration.program_tracking import official_static_surface_dataset as scope
from calibration.program_tracking.source_evidence_whitening import (
    line_evidence_from_feature, line_whitening_operator,
)
from test_complete_official_fit_dataset import accepted_replacement, source_camera
from test_official_static_surface_dataset import assert_exact_accepted_frame_replacement


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf8")
    return scope.reference(path)


def source_only_evidence(observations, frame):
    """Explicit synthetic source covariances exercise the production loader."""
    scales, locations, groups = {}, {}, []
    for row in observations:
        for point_id, sigma in zip(row["source_point_ids"], row["sigma"], strict=True):
            locations[point_id] = len(locations)
            scales[point_id] = float(sigma)
        if row["source_segment"].get("representative_points_share_line_estimate"):
            line = line_evidence_from_feature(row["source_feature"],
                group_id=f'{frame["native_pts"]}/{row["source_feature_id"]}')
            operator = line_whitening_operator(row["uv"], row["source_point_ids"], line)
            groups.append(dict(operator=operator.to_record(),
                point_indices=[locations[point_id] for point_id in row["source_point_ids"]]))
            for uv, point_id in zip(row["uv"], row["source_point_ids"], strict=True):
                design = np.array([uv[line.axis], 1.])
                scales[point_id] = float(np.sqrt(design @ line.covariance_ab @ design))
    return dict(source_scales_by_point_id=scales, whitening_groups=groups)


@pytest.fixture(scope="module")
def accepted_first_shot(accepted_replacement, tmp_path_factory):
    """All 1,248 frames are synthetic; never open or edit user observations."""
    base = accepted_replacement
    temporary = tmp_path_factory.mktemp("first_shot_extension")
    old_scope = scope.read_reference(base["selection"])
    old_model = scope.read_reference(base["model"])
    sample = base["packets"][0][1]
    records, receipts, packets = [], [], []
    for index in range(201, 1249):
        native = deepcopy(sample["source"])
        pts = 256000 + 256 * index
        native.update(native_pts=pts, actual_pts=pts, source_time=f"{pts / 12800:.3f}")
        features = deepcopy(sample["features"][:2])
        packet = dict(source=native, features=features, point_count=5,
            observation_coordinates_from_projection=False, user_accepted=False)
        packet_ref = write_json(temporary / f"p{index}.json", packet)
        observations = [dict(id=f"observation_{i}", source_feature_id=feature["id"],
            source_segment_index=0, source_vertex_indices=list(range(len(feature["segments"][0]["uv"]))),
            source_point_ids=feature["segments"][0]["point_ids"], target_id="board.top",
            point_correspondence=False, dependence_group=feature["id"])
            for i, feature in enumerate(features)]
        binding_ref = write_json(temporary / f"b{index}.json", dict(model_id=old_model["model_id"],
            source_packet=packet_ref, source_pts=pts, source_time_base="1/12800", observations=observations))
        records.append(dict(index=index, packet=packet_ref, model_binding=binding_ref))
        receipts.append(dict(index=index, packet_sha256=packet_ref["sha256"], binding_sha256=binding_ref["sha256"]))
        packets.append((packet_ref, packet))
    acceptance = write_json(temporary / "acceptance.json", dict(accepted=True,
        source_time_base="1/12800", frames=receipts))
    batch = write_json(temporary / "batch.json", dict(frames=records))
    model = deepcopy(old_model)
    model["accepted_batch_datasets"] = [dict(**batch, user_acceptance=acceptance)]
    model_ref = write_json(temporary / "model.json", model)
    append_ref, append_scope = scope.build_dataset(model_ref["path"],
        old_scope["primary_source"]["path"], temporary / "selection.json")
    preserved = {key: Path(base[key]["path"]).read_bytes()
                 for key in ("reference", "selection", "model")}
    extension_ref = complete.extend_complete_dataset(base["reference"], append_ref,
        temporary / "complete.json", frame_evidence_builder=source_only_evidence)
    original_frames = list(complete.iter_fit_frames(base["reference"]))
    extended_frames = list(complete.iter_fit_frames(extension_ref))
    return dict(base=base, appended_scope=append_scope, appended_reference=append_ref,
        reference=extension_ref, packets=packets, temporary=temporary, preserved=preserved,
        original_frames=original_frames, extended_frames=extended_frames)


def test_all_accepted_first_shot_addresses_reach_fit_and_the_model(accepted_first_shot):
    fixture = accepted_first_shot
    frames = fixture["extended_frames"]
    assert len(frames) == 1248
    assert sum(frame["source_point_count"] for frame in frames) == 1200 + 1048 * 5
    packets = fixture["base"]["packets"] + fixture["packets"]
    for frame, (packet_ref, packet) in zip(frames, packets, strict=True):
        assert_exact_accepted_frame_replacement(frame, packet_ref, packet)
        assert len(frame["whitening_groups"]) == 1
        operator = frame["whitening_groups"][0]["operator"]
        assert operator.consumed_point_count == 3 and operator.rank == 2
        for row in frame["rows"]:
            assert row["target"] == {"kind": "board_edge", "role": "top"}
            assert np.isfinite(row["world"]).all()
            assert np.isfinite(row["sigma"]).all() and (row["sigma"] > 0).all()
            assert row["native_pts"] == row["actual_pts"]
            assert row["time_base"] == "1/12800"
    for key, original in fixture["preserved"].items():
        assert Path(fixture["base"][key]["path"]).read_bytes() == original


def test_first_200_uv_scales_and_whitening_are_exactly_preserved(accepted_first_shot):
    fixture = accepted_first_shot
    original_frames = fixture["original_frames"]
    extended_frames = fixture["extended_frames"]
    for original, extended in zip(original_frames, extended_frames[:200], strict=True):
        assert original["native_pts"] == extended["native_pts"]
        assert original["source_point_count"] == extended["source_point_count"]
        for before, after in zip(original["rows"], extended["rows"], strict=True):
            assert before["source_point_ids"] == after["source_point_ids"]
            assert before["source_packet"] == after["source_packet"]
            assert before["target"] == after["target"]
            for field in ("uv", "sigma", "world"):
                np.testing.assert_array_equal(before[field], after[field])
        for before, after in zip(original["whitening_groups"], extended["whitening_groups"], strict=True):
            assert before["point_indices"] == after["point_indices"]
            assert before["operator"].to_record() == after["operator"].to_record()
    for index in (0, 199):
        before = complete.evaluate_frame(source_camera(), original_frames[index])
        after = complete.evaluate_frame(source_camera(), extended_frames[index])
        for old, new in zip(before, after, strict=True):
            np.testing.assert_array_equal(old, new)


def test_appended_points_produce_finite_residual_and_the_same_jacobian_operator(accepted_first_shot):
    frame = accepted_first_shot["extended_frames"][-1]
    camera = source_camera()
    residual, jacobian = complete.evaluate_frame(camera, frame)
    assert residual.shape == (6,) and jacobian.shape == (6, 15)
    assert np.isfinite(residual).all() and np.isfinite(jacobian).all()
    state = complete.camera_fit_state(camera)
    step = 2e-6
    before, after = state.copy(), state.copy()
    before[8] -= step
    after[8] += step
    finite = (complete.evaluate_frame(complete.camera_from_fit_state(after, camera), frame, False)
        - complete.evaluate_frame(complete.camera_from_fit_state(before, camera), frame, False)) / (2 * step)
    np.testing.assert_allclose(jacobian[:, 8], finite, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("mutation", ["source_clock", "parent_q", "missing_frame", "duplicate_pts"])
def test_extension_rejects_foreign_state_clock_or_incomplete_exposure_inventory(accepted_first_shot, mutation):
    fixture = accepted_first_shot
    selection = deepcopy(fixture["appended_scope"])
    if mutation == "source_clock":
        selection["source_clock"]["fps"] = "60/1"
    elif mutation == "parent_q":
        selection["parent_q_sha256"] = "0" * 64
    elif mutation == "missing_frame":
        selection["frames"].pop(10)
    else:
        selection["frames"][1]["native_pts"] = selection["frames"][0]["native_pts"]
    ref = write_json(fixture["temporary"] / f"scope_{mutation}.json", selection)
    output = fixture["temporary"] / f"fit_{mutation}.json"
    with pytest.raises(ValueError):
        complete.extend_complete_dataset(fixture["base"]["reference"], ref, output,
            frame_evidence_builder=source_only_evidence)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["missing_scale", "foreign_scale", "duplicate_operator", "foreign_address"])
def test_no_approved_point_can_be_missing_extra_or_double_counted_in_fit(accepted_first_shot, mutation):
    fixture = accepted_first_shot
    def broken_evidence(observations, frame):
        evidence = source_only_evidence(observations, frame)
        if mutation == "missing_scale":
            evidence["source_scales_by_point_id"].pop(next(iter(evidence["source_scales_by_point_id"])))
        elif mutation == "foreign_scale":
            evidence["source_scales_by_point_id"]["old replay point outside accepted packet"] = 1.
        elif mutation == "duplicate_operator":
            evidence["whitening_groups"].append(deepcopy(evidence["whitening_groups"][0]))
        else:
            evidence["whitening_groups"][0]["point_indices"][0] = -1
        return evidence
    output = fixture["temporary"] / f"fit_{mutation}.json"
    with pytest.raises(ValueError):
        ref = complete.extend_complete_dataset(fixture["base"]["reference"],
            fixture["appended_reference"], output, frame_evidence_builder=broken_evidence)
        list(complete.iter_fit_frames(ref))


def test_appended_model_cannot_change_the_original_200_world_targets(accepted_first_shot):
    fixture = accepted_first_shot
    selection = deepcopy(fixture["appended_scope"])
    model = scope.read_reference(selection["model"])
    scene = scope.read_reference(model["scene"])
    scene["board_surface"]["top_edge_xyz"][0][0] += 0.125
    model["scene"] = write_json(fixture["temporary"] / "changed_scene.json", scene)
    selection["model"] = write_json(fixture["temporary"] / "changed_model.json", model)
    ref = write_json(fixture["temporary"] / "changed_model_selection.json", selection)
    output = fixture["temporary"] / "changed_model_fit.json"
    with pytest.raises(ValueError):
        complete.extend_complete_dataset(fixture["base"]["reference"], ref, output,
            frame_evidence_builder=source_only_evidence)
    assert not output.exists()


def test_frozen_prefix_reference_is_authenticated_before_consumption(accepted_first_shot):
    fixture = accepted_first_shot
    manifest = scope.read_reference(fixture["reference"])
    manifest["base_dataset"]["sha256"] = "0" * 64
    ref = write_json(fixture["temporary"] / "bad_base_hash.json", manifest)
    with pytest.raises(ValueError):
        list(complete.iter_fit_frames(ref))


@pytest.mark.parametrize("field,value", [
    ("parent_complete_state", None),
    ("frame_range", [1, 200]),
    ("added_fit_points", -1),
    ("prefix_frame_records_reused_unchanged", False),
    ("prefix_UV_scales_and_operators_changed", True),
    ("source_UV_modified", True),
    ("legacy_objectives_enabled", True),
    ("old_or_unaccepted_points_appended", 1),
])
def test_consumer_rejects_changed_extension_metadata_before_frame_assembly(
        accepted_first_shot, monkeypatch, field, value):
    fixture = accepted_first_shot
    manifest = scope.read_reference(fixture["reference"])
    manifest[field] = value
    ref = write_json(fixture["temporary"] / f"changed_metadata_{field}.json", manifest)

    def forbidden_frame_assembly(*args, **kwargs):
        raise AssertionError("Tampered extension metadata reached frame consumption")

    monkeypatch.setattr(complete, "_fit_frame", forbidden_frame_assembly)
    with pytest.raises(ValueError, match="Extension manifest changed"):
        list(complete.iter_fit_frames(ref))
