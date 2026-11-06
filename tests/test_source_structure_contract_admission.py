"""Finite source-contract admission; synthetic fixtures, no model or fit run."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from calibration.program_tracking.body_tracking.source_structure_integration import integration


def _write(path, document):
    path.write_text(json.dumps(document), encoding="utf8")
    return path


@pytest.fixture
def inputs(tmp_path):
    lower = dict(points_uv=[[0, 5], [4, 5], [8, 5]], closed=False, status="ambiguous", pixel_uncertainty_radius_px=4)
    assembly = dict(points_uv=[[0, 0], [2, 0], [4, 0], [4, 2], [4, 4], [6, 4], [8, 4], [10, 4]],
                    closed=True, status="observed", pixel_uncertainty_radius_px=6)
    frame = dict(pts=1024, foot=1, source_crop_sha256="synthetic_crop", source_support={"supported": False},
                 annotation_provenance={"path": "annotations_combined.json", "sha256": "synthetic_annotation"},
                 features={"runner_lower_edge": lower, "blade_assembly_outline": assembly})
    source = dict(source_sha256="synthetic_video", time_base=[1, 12800], native_size=[1920, 1080], frames=[frame],
                  unassigned_identity_observations=[dict(deepcopy(frame), foot=None, source_track_id="synthetic_unknown_track")])
    contracts = []
    for name, feature in frame["features"].items():
        contract = dict(pts=frame["pts"], foot=1, feature=name, source_track_id="synthetic_derived_track",
                        source_track_id_provenance="synthetic existing-foot grouping", identity_provenance="synthetic source sequence",
                        visibility="visible", completeness="complete", role="fit", human_confirmed=False,
                        source_range=dict(index_base=0, original_index_range_inclusive=[0, len(feature["points_uv"])-1],
                                          points_uv_original=deepcopy(feature["points_uv"]),
                                          pixel_uncertainty_radius_px_original=feature["pixel_uncertainty_radius_px"],
                                          original_status=feature["status"], source_closed=feature["closed"]),
                        source_regions={"original_uncertainty_radius_px": feature["pixel_uncertainty_radius_px"],
                                        "silhouette_terminals": {"front_region": {"index": 0, "native_uv": feature["points_uv"][0], "uncertainty_radius_px": 5}}},
                        endpoints={"material_front": None, "material_rear": None}, evidence={"source_crop_sha256": "synthetic_crop"},
                        local_structural_notes=[])
        if name == "blade_assembly_outline":
            contract["local_structural_notes"] = [
                {"index_range_inclusive": [0, 2], "meaning_zh": "与原 boot_sole 鞋底轮廓重合；不能按钢刃外缘匹配。"},
                {"index_range_inclusive": [2, 4], "meaning_zh": "鞋跟块及其与后支架连接候选；不是连续钢片下缘。"},
                {"index_range_inclusive": [5, 7], "meaning_zh": "刀体外部下缘的连续外观范围。"}]
        contracts.append(contract)
    contract_doc = dict(schema="rinkspace.known_foot_source_contour_contract_review.v1",
                        source_video={"sha256": source["source_sha256"]}, source_time_base=source["time_base"],
                        native_size=source["native_size"], rows=contracts)
    return (source, contract_doc, _write(tmp_path / "source.json", source),
            _write(tmp_path / "contracts.json", contract_doc),
            _write(tmp_path / "contacts.json", {"source_sha256": source["source_sha256"], "frames": []}))


def test_no_contract_filename_cannot_authorize_reverse_and_old_call_still_writes(inputs, tmp_path):
    source, _, src, _, contacts = inputs
    output = tmp_path / "new_output.json"
    doc = integration(src, contacts, output)
    assert output.exists() and not doc["complete_runner_extent"]["rows"]
    assert "association_policy" not in doc["full_source_skate_structures"]
    assert "association_policy" not in doc["complete_runner_extent"]
    row = doc["full_source_skate_structures"]["rows"][0]
    assert row["visibility"] == "unknown" and not row["source_contract_matched"]
    assert row["source_annotation"] == source["frames"][0]["features"]["runner_lower_edge"]


def test_explicit_complete_ambiguous_source_preserves_finite_ranges_and_budget(inputs):
    source, contract, src, cp, contacts = inputs
    before = {p: p.read_bytes() for p in (src, cp, contacts)}
    plain = integration(src, contacts)
    doc = integration(src, contacts, contracts=cp)
    reverse, = doc["complete_runner_extent"]["rows"]
    forward = next(r for r in doc["full_source_skate_structures"]["rows"] if r["source_structure_feature"] == "runner_lower_edge")
    baseline = next(r for r in plain["full_source_skate_structures"]["rows"] if r["structure_feature"] == "runner_lower_edge")
    assert forward["source_status"] == "ambiguous"
    assert forward["feature_budget_fraction"] == reverse["feature_budget_fraction"] == baseline["feature_budget_fraction"] / 2
    assert reverse["visibility"] == "visible" and reverse["completeness"] == "complete"
    assert forward["source_track_id"] == "synthetic_derived_track"
    assert forward["source_contract_sha256"] == hashlib.sha256(cp.read_bytes()).hexdigest()
    assert forward["source_range"] == contract["rows"][0]["source_range"]
    assert forward["source_regions"] == contract["rows"][0]["source_regions"]
    assert forward["source_endpoints"] == {"material_front": None, "material_rear": None}
    assert forward["annotation_provenance"] == source["frames"][0]["annotation_provenance"]
    assert doc["full_source_skate_structures"]["association_policy"] == dict(model_visibility="view_visible_exterior", freeze_inner=True, arc_length_measure=True)
    assert doc["complete_runner_extent"]["association_policy"]["model_visibility"] == "near_visible_runner_branch"
    for group in doc["unassigned_source_skate_sets"]:
        for branch in group["branches"]:
            assert "association_policy" not in branch
            assert all(r["source_anatomical_foot"] is None and not r["source_contract_matched"] for r in branch["rows"])
    assert all(p.read_bytes() == b for p, b in before.items())


@pytest.mark.parametrize("visibility, completeness", [("partial", "partial"), ("partial", "complete"), ("visible", "partial"), ("unknown", "unknown"), ("occluded", "complete")])
def test_partial_unknown_or_occluded_contract_keeps_forward_without_reverse(inputs, visibility, completeness):
    _, c, src, cp, contacts = inputs
    c["rows"][0].update(visibility=visibility, completeness=completeness)
    _write(cp, c)
    doc = integration(src, contacts, contracts=cp)
    assert not doc["complete_runner_extent"]["rows"]
    row = next(r for r in doc["full_source_skate_structures"]["rows"] if r["source_structure_feature"] == "runner_lower_edge")
    assert row["visibility"] == visibility
    baseline = next(r for r in integration(src, contacts)["full_source_skate_structures"]["rows"] if r["structure_feature"] == "runner_lower_edge")
    assert row["feature_budget_fraction"] == baseline["feature_budget_fraction"]


def test_explicit_assembly_indices_partition_all_edges_and_share_original_budget(inputs):
    _, _, src, cp, contacts = inputs
    baseline = next(r for r in integration(src, contacts)["full_source_skate_structures"]["rows"] if r["structure_feature"] == "blade_assembly_outline")
    doc = integration(src, contacts, contracts=cp)
    rows = [r for r in doc["full_source_skate_structures"]["rows"] if r["source_structure_feature"] == "blade_assembly_outline"]
    assert {r["structure_feature"] for r in rows} == {"boot_sole", "heel_block", "runner_lower_edge", "blade_assembly_outline"}
    original_path = baseline["source_retained_point_indices"] + [0]
    edges = [(a, b) for r in rows for a, b in zip(r["source_retained_point_indices"], r["source_retained_point_indices"][1:])]
    assert edges == list(zip(original_path, original_path[1:]))
    assert sum(r["feature_budget_fraction"] for r in rows) == pytest.approx(baseline["feature_budget_fraction"])
    # The assembly's lower subarc is not another complete lower-runner record.
    assert len(doc["complete_runner_extent"]["rows"]) == 1


def test_semantic_notes_without_explicit_indices_do_not_split_or_guess(inputs):
    _, c, src, cp, contacts = inputs
    c["rows"][1]["local_structural_notes"] = [{"meaning_zh": "与原 boot_sole 鞋底轮廓重合"}, {"index_range_inclusive": [0, 7], "meaning_zh": "mixed uncertain structure"}, {"index_range_inclusive": [0, 7], "meaning_zh": "鞋跟块不可见，不能确认"}]
    _write(cp, c)
    rows = integration(src, contacts, contracts=cp)["full_source_skate_structures"]["rows"]
    assert len([r for r in rows if r["source_structure_feature"] == "blade_assembly_outline"]) == 1


@pytest.mark.parametrize("mutation, message", [
    (lambda c: c["source_video"].update(sha256="wrong"), "identity"),
    (lambda c: c.update(source_time_base=[1, 1000]), "clock/pixel"),
    (lambda c: c["rows"][0]["source_range"]["points_uv_original"][0].__setitem__(0, 99), "original"),
    (lambda c: c["rows"][0]["source_range"].update(pixel_uncertainty_radius_px_original=1), "original"),
    (lambda c: c["rows"][0]["evidence"].update(source_crop_sha256="wrong"), "crop identity"),
    (lambda c: c["rows"][1]["local_structural_notes"][0].update(index_range_inclusive=[0, 99]), "structural indices"),
    (lambda c: c["rows"].append(deepcopy(c["rows"][0])), "Duplicate"),
])
def test_mismatched_contract_is_rejected_before_writing(inputs, tmp_path, mutation, message):
    _, c, src, cp, contacts = inputs
    mutation(c)
    _write(cp, c)
    output = tmp_path / "must_not_exist.json"
    with pytest.raises(ValueError, match=message):
        integration(src, contacts, output, contracts=cp)
    assert not output.exists()


_LOCAL = Path("D:/CodexData/RinkSpace_Data/runs/body_s01_20261029/body_sequence_1105")
@pytest.mark.skipif(not (_LOCAL / "source_contract_review_1106/known_foot_contracts.json").exists(), reason="requires existing reviewed local source contracts")
def test_current_review_admits_exact_28_complete_lower_rows_without_writing_candidates():
    src = _LOCAL / "full_structure_1106_inputs/structures_fit_v1.json"
    cp = _LOCAL / "source_contract_review_1106/known_foot_contracts.json"
    contacts = _LOCAL / "full_structure_1106_inputs/contacts_fit_v1.json"
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (src, cp, contacts)}
    doc = integration(src, contacts, contracts=cp)
    reverse = doc["complete_runner_extent"]["rows"]
    assert len(reverse) == 28
    assert sum(r["source_status"] == "ambiguous" for r in reverse) == 6
    original = integration(src, contacts)
    before = {r["source_budget_group_id"]: r["feature_budget_fraction"] for r in original["full_source_skate_structures"]["rows"]}
    after = {key: 0. for key in before}
    for r in doc["full_source_skate_structures"]["rows"] + reverse:
        after[r["source_budget_group_id"]] += r["feature_budget_fraction"]
    assert after == pytest.approx(before)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in hashes.items())
