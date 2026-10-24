"""Existing atlas pixels bind to named assets without new charts or weights."""
from copy import deepcopy
import hashlib
import json

from PIL import Image
import pytest

from rinkspace.compat.ad_assets import append_source_atlas, import_mixed_ad_template
from rinkspace.session.ad_review import source_panels


def save_json(path, value):
    path.write_text(json.dumps(value), encoding="utf8")
    return path


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def evidence(tmp_path):
    old_image = tmp_path/"old.png"
    current_image = tmp_path/"current.png"
    atlas_image = tmp_path/"atlas.png"
    for image, color in ((old_image, "red"), (current_image, "blue"), (atlas_image, "green")):
        Image.new("RGB", (8, 8), color).save(image)
    a = {"id": "A", "closed": True, "role": "ink_boundary",
         "source_native_uv": [[1., 1.], [3., 1.], [2., 4.], [1., 1.]],
         "material_uv": [[.1, .1], [.3, .1], [.2, .4], [.1, .1]],
         "source_native_boundary_uncertainty_px": 1.2}
    ribbon = {"id": "ribbon", "closed": False, "role": "visible_blue_ribbon_material_boundary",
              "source_native_uv": [[1., 4.], [2., 3.], [3., 2.]],
              "material_uv": [[.1, .4], [.2, .3], [.3, .2]]}
    mixed = {"schema": "legacy.mixed", "advertisement_number": 2,
             "source": {"source_sha256": "a"*64, "source_pts": 100,
                        "time_base": "1/15360", "source_frame_path": str(old_image),
                        "source_frame_sha256": file_hash(old_image)},
             "contours": {"A": a}, "advertisements": {"2": {
                 "advertisement_number": 2, "source": {"path": str(current_image),
                 "sha256": file_hash(current_image), "native_pts": 256},
                 "contours": {"A": dict(deepcopy(a), advertisement_id=2), "ribbon": ribbon}}}}
    mixed_path = save_json(tmp_path/"mixed.json", mixed)
    source = {"video_sha256": "b"*64, "time_base": "1/12800", "native_hw": [8, 8]}
    assets = import_mixed_ad_template(mixed_path, source)
    atlas = {"schema": "rinkfit.native_complete_print_glyphs.v1", "native_pts": 512,
             "source_image": {"file": str(atlas_image)},
             "source_model_projection_used": False, "threshold_green": 140.,
             "paths": [{"label": "Ad2.print.A", "closed": True,
                        "source_native_uv": [[1., 1.], [2., 1.], [3., 1.],
                                             [2., 4.], [1., 1.]]}]}
    atlas_path = save_json(tmp_path/"atlas.json", atlas)
    return {"assets": assets, "source": source, "atlas_path": atlas_path,
            "atlas": atlas, "frame_sha256": file_hash(atlas_image),
            "mixed_path": mixed_path, "tmp": tmp_path, "current_image": current_image}


def append(evidence, assets=None):
    return append_source_atlas(evidence["assets"] if assets is None else assets,
                               evidence["atlas_path"], evidence["source"],
                               frame_sha256=evidence["frame_sha256"])


def test_same_video_two_exposures_preserve_both_frames_and_clocks(evidence):
    result = append(evidence)
    official = next(s for s in result["sources"].values() if s["sha256"] == "b"*64)
    assert official["time_base"] == "1/12800"
    assert {e["pts"] for e in official["exposures"]} == {256, 512}
    frames = {e["pts"]: e for e in official["exposures"]}
    assert frames[256]["frame_path"] == str(evidence["current_image"])
    assert frames[256]["frame_sha256"] == file_hash(evidence["current_image"])
    assert frames[512]["frame_sha256"] == evidence["frame_sha256"]
    historical = next(s for s in result["sources"].values() if s["sha256"] == "a"*64)
    assert (historical["pts"], historical["time_base"]) == (100, "1/15360")


def test_append_is_pure_and_keeps_original_pixels_with_no_invented_chart(evidence):
    before = deepcopy(evidence["assets"])
    mixed_bytes = evidence["mixed_path"].read_bytes()
    atlas_bytes = evidence["atlas_path"].read_bytes()
    result = append(evidence)
    assert evidence["assets"] == before
    assert evidence["mixed_path"].read_bytes() == mixed_bytes
    assert evidence["atlas_path"].read_bytes() == atlas_bytes
    row = result["observations"][-1]
    assert row["source_native_uv"] == evidence["atlas"]["paths"][0]["source_native_uv"]
    assert row["material_uv"] is None
    assert "covariance" not in row
    assert not row["active_for_fit"] and not row["approval_inferred"]
    assert not row["new_measurement_created"]
    assert [p["material_uv"] for p in result["templates"][0]["paths"]] == [
        p["material_uv"] for p in before["templates"][0]["paths"]]
    row["source_native_uv"][0][0] = 7.
    assert evidence["atlas"]["paths"][0]["source_native_uv"][0][0] == 1.


def test_high_resolution_and_historical_evidence_bind_same_feature(evidence):
    result = append(evidence)
    a = next(p for p in result["templates"][0]["paths"] if p["id"] == "Ad2.print.A")
    rows = [r for r in result["observations"] if r["feature_id"] == a["id"]]
    assert len(rows) == 2
    assert set(a["source_observation_ids"]) == {r["id"] for r in rows}
    assert a["source_observation_id"] == rows[0]["id"]
    assert a["preferred_pixel_evidence_id"] == rows[1]["id"]
    assert a["template_shape_refinement_pending"]
    assert len(result["templates"][0]["paths"]) == 2
    assert len(result["aliases"]) == 1
    assert result["counts"]["unique_contour_observations"] == 3


def test_repeated_atlas_rejects_duplicate_weight_without_editing_input(evidence):
    first = append(evidence)
    before = deepcopy(first)
    with pytest.raises(ValueError, match="already imported|double count"):
        append(evidence, first)
    assert first == before


def test_atlas_unknown_feature_cannot_create_an_identity(evidence):
    atlas = evidence["atlas"]
    atlas["paths"][0]["label"] = "Ad15.print.A"
    save_json(evidence["atlas_path"], atlas)
    with pytest.raises(ValueError, match="no confirmed template identity"):
        append(evidence)


def test_source_model_projection_is_not_source_only_atlas(evidence):
    evidence["atlas"]["source_model_projection_used"] = True
    save_json(evidence["atlas_path"], evidence["atlas"])
    with pytest.raises(ValueError, match="source-only"):
        append(evidence)


def test_atlas_requires_explicit_native_image_hash(evidence):
    with pytest.raises(ValueError, match="native-frame SHA256"):
        append_source_atlas(evidence["assets"], evidence["atlas_path"],
                            evidence["source"], frame_sha256="")


def test_review_rejects_native_image_byte_change(evidence):
    result = append(evidence)
    rows = [r for r in result["observations"] if r["evidence_role"] == "current_programme_source_glyph"]
    Image.new("RGB", (8, 8), "black").save(evidence["atlas"]["source_image"]["file"])
    with pytest.raises(ValueError, match="native source-frame bytes changed"):
        source_panels(rows, "Q", evidence["tmp"], "synthetic regression")


def test_current_source_pts_conflict_does_not_override_atlas(evidence):
    with pytest.raises(ValueError, match="conflicting.*pts"):
        append_source_atlas(evidence["assets"], evidence["atlas_path"],
                            dict(evidence["source"], pts=999), frame_sha256=evidence["frame_sha256"])


def test_atlas_repackaging_existing_source_row_cannot_double_weight(evidence):
    """The same source observation through two adapter paths is still one row."""
    original = next(r for r in evidence["assets"]["observations"] if r["original_id"] == "ribbon")
    atlas = {"schema": "rinkfit.native_complete_print_glyphs.v1", "native_pts": original["pts"],
             "source_image": {"file": original["source_frame_path"]},
             "source_model_projection_used": False,
             "paths": [{"label": original["feature_id"], "closed": original["closed"],
                        "source_native_uv": deepcopy(original["source_native_uv"])}]}
    save_json(evidence["atlas_path"], atlas)
    with pytest.raises(ValueError, match="already imported|double count"):
        append_source_atlas(evidence["assets"], evidence["atlas_path"], evidence["source"],
                            frame_sha256=original["source_frame_sha256"])


def test_numeric_json_encoding_cannot_duplicate_the_same_pixels(evidence):
    """Legacy OpenCV integer contours and float64 JSON can encode one reading."""
    original = next(r for r in evidence["assets"]["observations"] if r["original_id"] == "ribbon")
    atlas = {"schema": "rinkfit.native_complete_print_glyphs.v1", "native_pts": original["pts"],
             "source_image": {"file": original["source_frame_path"]},
             "source_model_projection_used": False,
             "paths": [{"label": original["feature_id"], "closed": original["closed"],
                        "source_native_uv": [[int(v) for v in p] for p in original["source_native_uv"]]}]}
    save_json(evidence["atlas_path"], atlas)
    with pytest.raises(ValueError, match="already imported|double count"):
        append_source_atlas(evidence["assets"], evidence["atlas_path"], evidence["source"],
                            frame_sha256=original["source_frame_sha256"])
