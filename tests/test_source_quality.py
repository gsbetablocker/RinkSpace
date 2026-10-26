"""Meaningful acceptance-bound diagnostics; no model or human state mutation."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from rinkspace.session.source_quality import (
    EvaluationUsage, QualityProtocol, ReferenceFrame, ReferenceSegment,
    compare_source_frame, load_accepted_reference,
)


SOURCE_SHA = "a" * 64
RGB_SHA = "b" * 64


def segment(identity="ice.lane3:edge0", uv=((0, 20), (100, 20)), **kwargs):
    return ReferenceSegment(identity, identity, kwargs.pop("segment_index", 0),
        kwargs.pop("layer", "ice"), kwargs.pop("measurement", "centreline"),
        tuple(map(tuple, uv)), False, kwargs.pop("geometry", {}), **kwargs)


def reference(*segments):
    return ReferenceFrame(SOURCE_SHA, 309760, "1/12800", "s01", (1080, 1920),
        RGB_SHA, "c" * 64, "d" * 64, "e" * 64, tuple(segments or [segment()]),
        ("ice", "board_boundary"))


def reading(uv=((0, 20), (100, 20)), identity="ice.lane3:edge0", *, indices=None, runs=True, **kwargs):
    points = [list(p) for p in uv]
    indices = list(range(len(points))) if indices is None else indices
    row = {"physical_id": identity, "expected_id": identity, "kind": "ice_paint",
           "native_measurement": "centreline", "actual_source_uv": points,
           "source_indices": indices,
           "measured_runs": [{"actual_source_uv": points, "source_indices": indices, "segment_id": 0}] if runs else []}
    row.update(kwargs)
    return row


def record(*readings):
    return {"source_binding": {"source_sha256": SOURCE_SHA, "source_pts": 309760,
        "time_base": "1/12800", "shot_id": "s01", "native_hw": [1080, 1920],
        "decoded_native_rgb_sha256": RGB_SHA}, "guided": {"readings": list(readings)}}


def independent_usage(**kwargs):
    values = dict(training_exposure="excluded", tuning_exposure="excluded",
                  human_mask_used=False, same_exposure_reference_used=False,
                  data_dependencies_audited=True)
    values.update(kwargs)
    return EvaluationUsage(**values)


def codes(report):
    return {row["code"] for row in report["failures"]}


def test_real_saved_reference_receipt_scope_and_clock():
    """The repository's actual reviewed dataset is part of this integration test."""
    repo = Path(__file__).resolve().parents[1]
    base = repo / "events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013/first_shot_remaining_50fps/repair_20261014"
    ref = load_accepted_reference(base / "frames/000210.json", base / "bindings/000210.json", base / "user_acceptance.json")
    assert ref.native_pts == 309760
    assert ref.time_base == "1/12800"
    assert len({s.feature_id for s in ref.segments}) == 14
    assert all(s.feature_id.startswith(("ice.", "board_top.", "board_contact.")) for s in ref.segments)
    assert any(s.feature_id == "ice.lane4:edge2" for s in ref.segments)
    with pytest.raises(ValueError, match="scope"):
        load_accepted_reference(base / "frames/000210.json", base / "bindings/000210.json", base / "user_acceptance.json", reviewed_layers=("ads",))


def test_acceptance_cannot_be_reused_after_file_edit(tmp_path):
    packet = {"source": {"source_video_sha256": SOURCE_SHA, "native_pts": 309760,
        "actual_pts": 309760, "time_base": "1/12800", "native_hw": [1080, 1920]},
        "features": [{"id": "ice.a", "segments": [{"uv": [[0, 1], [10, 1]]}]}]}
    packet_path, binding_path, acceptance_path = [tmp_path / name for name in ("packet.json", "binding.json", "acceptance.json")]
    packet_path.write_text(json.dumps(packet))
    packet_sha = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    binding = {"source_packet": {"sha256": packet_sha}, "source_pts": 309760,
        "source_time_base": "1/12800", "observations": [{"source_feature_id": "ice.a", "source_segment_index": 0, "target_id": "ice.a"}]}
    binding_path.write_text(json.dumps(binding))
    acceptance_path.write_text(json.dumps({"accepted": True, "frames": [{"native_pts": 309760,
        "packet_sha256": packet_sha, "binding_sha256": hashlib.sha256(binding_path.read_bytes()).hexdigest()}]}))
    assert load_accepted_reference(packet_path, binding_path, acceptance_path).segments
    packet_path.write_text(json.dumps(packet, indent=2))
    with pytest.raises(ValueError, match="SHA"):
        load_accepted_reference(packet_path, binding_path, acceptance_path)


def test_correct_pixels_do_not_change_human_state_or_prove_training_independence():
    data = record(reading())
    original = json.dumps(data, sort_keys=True)
    result = compare_source_frame(reference(), data)
    assert result["evaluation_role"] == "regression"
    assert "training_exposure" in result["independence"]["unknown_dependencies"]
    assert not result["failures"]
    assert result["features"][0]["support_fraction"] == 1
    assert result["accuracy_accepted"] is False
    assert result["human_state_modified"] is False
    assert json.dumps(data, sort_keys=True) == original


def test_missing_and_localized_errors_are_actionable_source_bound_records():
    result = compare_source_frame(reference(segment(), segment("ice.lane4:edge2")), record(reading(uv=((0, 27), (100, 27)))))
    assert {"missing_support", "localization"} <= codes(result)
    failure = next(f for f in result["failures"] if f["code"] == "localization")
    assert failure["details"]["query_indices"] == [0, 1]
    assert failure["source_pts"] == 309760
    assert all(a["first_pts"] == a["last_pts"] == 309760 for a in result["repair_actions"])
    assert {a["action"] for a in result["repair_actions"]} == {"reread_material", "discover_material"}


def test_unmarked_extensions_and_ads_are_not_false_positives():
    result = compare_source_frame(reference(), record(
        reading(uv=((150, 20), (200, 20))),
        reading(identity="print_instance/Ad5::Ad5.print.x", kind="ad_boundary")))
    assert "localization" not in codes(result)
    assert result["features"][0]["points_outside_reviewed_support"] == 2
    assert result["unscored"][0]["reason"] == "outside_reviewed_scope"


@pytest.mark.parametrize('label,path,feature,kind', [
    ('board.top','board/continuous/top','board_top.upper_right','board_top'),
    ('board.bottom','board/continuous/bottom','board_contact.upper_right','ice_contact')])
def test_continuous_board_label_requires_exact_native_alias_binding(label,path,feature,kind):
    ref=reference(segment(feature,layer='board_boundary',measurement='contour'))
    row=reading(identity=label,kind=kind,native_measurement='contour',native_path_id=path,
        native_geometry={'path_id':path})
    original=json.dumps(row,sort_keys=True)
    report=compare_source_frame(ref,record(row))
    assert report['features'][0]['support_fraction']==1.
    assert report['identity_alias_resolutions'][0]['source_physical_id']==label
    assert json.dumps(row,sort_keys=True)==original
    row['native_geometry']['path_id']='other_boundary'
    report=compare_source_frame(ref,record(row))
    assert report['features'][0]['support_fraction']==0.
    assert not report.get('identity_alias_resolutions')


def test_partial_circle_does_not_score_unlabelled_upper_arc_against_lower_arc():
    geometry = {"kind": "projected_physical_circle_ellipse", "center_uv": [200, 200],
                "axes_px": [100, 100], "angle_degrees": 0}
    ref = reference(segment("ice.R4:inner", uv=((150, 200), (200, 250), (250, 200)),
                            measurement="contour", geometry=geometry))
    result = compare_source_frame(ref, record(reading(identity="ice.R4:inner", uv=((200, 150),), native_measurement="contour")))
    assert "localization" not in codes(result)
    assert result["features"][0]["points_outside_reviewed_support"] == 1
    assert result["features"][0]["support_fraction"] == 0


def test_occlusion_gaps_are_not_filled_by_continuous_looking_endpoints():
    result = compare_source_frame(reference(), record(reading(indices=[0, 7])))
    assert result["features"][0]["support_fraction"] < 0.2
    assert result["features"][0]["reviewed_outline_to_measured_runs"]["max_px"] > 40
    assert "missing_support" in codes(result)
    accepted_gap = reference(segment(uv=((0, 20), (20, 20))), segment(uv=((80, 20), (100, 20)), segment_index=1))
    unmarked = compare_source_frame(accepted_gap, record(reading(uv=((50, 20),))))
    assert "localization" not in codes(unmarked)
    assert unmarked["features"][0]["points_outside_reviewed_support"] == 1


def test_unresolved_identity_cannot_be_named_by_hint_or_nearest_geometry():
    result = compare_source_frame(reference(), record(reading(identity=None,
        expected_id="ice.lane3:edge0", native_path_id="ice.lane3:edge0")))
    assert "unresolved_identity" in codes(result)
    assert result["features"][0]["actual_point_count"] == 0
    assert result["features"][0]["support_fraction"] == 0


def test_explicit_semantic_identity_conflict_is_detected_without_nearest_relabelling():
    ref = reference(segment("board_top.upper_right", layer="board_boundary"))
    result = compare_source_frame(ref, record(reading(identity="board_top.upper_right", kind="ice_contact")))
    assert "wrong_identity" in codes(result)
    assert result["features"][0]["actual_point_count"] == 0
    assert any(a["action"] == "reassociate_entity" for a in result["repair_actions"])


def test_whole_board_identity_is_not_localized_against_every_boundary_segment():
    ref = reference(segment("board_top.upper_right", layer="board_boundary"),
        segment("board_top.upper_long_side", uv=((0, 100), (100, 100)), layer="board_boundary"))
    result = compare_source_frame(ref, record(reading(identity="board/continuous/top", kind="board_top")))
    assert "localization" not in codes(result)
    assert result["features"][0]["support_fraction"] == 1
    assert result["features"][1]["support_fraction"] == 0


@pytest.mark.parametrize("dependency", ["human_mask_used", "same_exposure_reference_used", "training_exposure"])
def test_leakage_overrides_independent_declaration(dependency):
    usage = independent_usage(**{dependency: "included" if dependency == "training_exposure" else True})
    result = compare_source_frame(reference(), record(reading()), usage=usage)
    assert result["evaluation_role"] == "regression"
    assert "leakage" in codes(result)
    assert {a["action"] for a in result["repair_actions"]} == {"repair_evaluation_setup"}


def test_inherited_human_seed_overrides_false_clean_flag():
    data = record(reading(identity_authority={"human_authority": True, "packet": {"sha256": "9" * 64}}))
    data["accepted_reference_used_for_guides"] = False
    result = compare_source_frame(reference(), data, usage=independent_usage())
    assert "human_seed_in_temporal_chain" in result["independence"]["known_dependencies"]
    assert result["evaluation_role"] == "regression"
    clean = compare_source_frame(reference(), record(reading()), usage=independent_usage())
    assert clean["evaluation_role"] == "independent"
    assert clean["accuracy_accepted"] is False


@pytest.mark.parametrize("field,value", [("source_pts", 309761), ("source_sha256", "0" * 64),
                                        ("time_base", "1/50"), ("decoded_native_rgb_sha256", "0" * 64)])
def test_different_exposure_or_pixels_are_not_comparable(field, value):
    data = record(reading())
    data["source_binding"][field] = value
    with pytest.raises(ValueError, match="different"):
        compare_source_frame(reference(), data)


def test_fitted_run_substitution_is_rejected_and_projection_not_scored():
    row = reading()
    row["measured_runs"][0]["actual_source_uv"] = [[0, 20], [100, 19]]
    with pytest.raises(ValueError, match="substitutes"):
        compare_source_frame(reference(), record(row))
    data = record(reading())
    data["hint_pixels_used_as_output"] = True
    result = compare_source_frame(reference(), data)
    assert "invalid_source_evidence" in codes(result)
    assert result["features"][0]["actual_point_count"] == 0


def test_run_connectivity_cannot_override_original_segment_ids():
    row = reading(segment_ids=[0, 1])
    with pytest.raises(ValueError, match="segment IDs"):
        compare_source_frame(reference(), record(row))


def test_protocol_validation_and_empty_scope():
    with pytest.raises(ValueError):
        QualityProtocol(sample_spacing_px=0)
    with pytest.raises(ValueError):
        EvaluationUsage(training_exposure="clean")
    empty = replace(reference(), segments=())
    assert compare_source_frame(empty, record(reading()))["evaluation_role"] == "unscored"
