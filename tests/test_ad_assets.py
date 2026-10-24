"""Mixed-source migration regression: no remeasurement or approval inference."""
from copy import deepcopy
import json

import pytest

from rinkspace.compat.ad_assets import import_mixed_ad_template


def contour(key, *, closed=True, parent=None):
    result = {"id": key, "role": "ink_boundary", "closed": closed,
              "source_native_uv": [[1., 2.], [3., 4.]],
              "material_uv": [[.1, .2], [.3, .4]],
              "source_native_boundary_uncertainty_px": 2.0,
              "noise_group_id": "original/correlated", "revision": "original"}
    if parent:
        result["parent_contour_key"] = parent
    return result


@pytest.fixture
def packet(tmp_path):
    original = {f"glyph{i}": contour(f"glyph{i}") for i in range(15)}
    original["A_inner"] = contour("A_inner", parent="glyph0")
    original["O_inner"] = contour("O_inner", parent="glyph1")
    children = {k: dict(deepcopy(v), advertisement_id=2) for k, v in original.items()}
    children.update({f"ribbon{i}": contour(f"ribbon{i}", closed=False) for i in range(15)})
    data = {"schema": "legacy.mixed", "advertisement_number": 2,
            "source": {"source_sha256": "a"*64, "source_pts": 56218880,
                       "time_base": "1/15360", "source_frame_path": "historical.png"},
            "contours": original, "advertisements": {"2": {
                "advertisement_number": 2,
                "source": {"path": "current.png", "sha256": "c"*64, "native_pts": 702976},
                "contours": children}}}
    path = tmp_path/"mixed.json"
    path.write_text(json.dumps(data), encoding="utf8")
    return path, data


CURRENT = {"video_sha256": "b"*64, "time_base": "1/12800"}


def test_copied_glyphs_do_not_add_observations_or_weight(packet):
    path, _ = packet
    result = import_mixed_ad_template(path, CURRENT)
    assert result["counts"] == {"historical_unique_contours": 17, "current_unique_contours": 15,
                                "copied_aliases": 17, "unique_contour_observations": 32}
    observations = {r["id"]: r for r in result["observations"]}
    assert len(observations) == 32
    for alias in result["aliases"]:
        original = observations[alias["canonical_observation_id"]]
        assert original["source_sha256"] == "a"*64
        assert not alias["additional_statistical_weight"]
    assert all(not r["active_for_fit"] for r in observations.values())


def test_mixed_source_clock_and_original_pixels_are_preserved(packet):
    path, data = packet
    result = import_mixed_ad_template(path, CURRENT)
    for row in result["observations"]:
        historic = row["original_id"].startswith("glyph") or row["original_id"].endswith("inner")
        assert (row["source_sha256"], row["pts"], row["time_base"]) == (
            ("a"*64, 56218880, "1/15360") if historic else ("b"*64, 702976, "1/12800"))
        assert row["source_native_uv"] == [[1., 2.], [3., 4.]]
        assert row["source_frame_path"] == ("historical.png" if historic else "current.png")
    assert result["provenance"][0]["raw_record"] == data["contours"]["glyph0"]
    assert result["provenance"][0]["raw_record"]["noise_group_id"] == "original/correlated"


def test_holes_and_open_source_arcs_keep_topology(packet):
    path, _ = packet
    paths = {p["original_id"]: p for p in import_mixed_ad_template(path, CURRENT)["templates"][0]["paths"]}
    assert paths["A_inner"]["parent_contour_key"] == "glyph0"
    assert paths["O_inner"]["parent_contour_key"] == "glyph1"
    assert all(not paths[f"ribbon{i}"]["closed"] for i in range(15))
    assert all(paths[f"ribbon{i}"]["material_uv"] == [[.1, .2], [.3, .4]] for i in range(15))


def test_absent_current_clock_never_falls_back_to_historical_parent(packet):
    path, _ = packet
    with pytest.raises(ValueError, match="video SHA256"):
        import_mixed_ad_template(path, {})
    with pytest.raises(ValueError, match="time_base"):
        import_mixed_ad_template(path, {"video_sha256": "b"*64})


def test_same_id_different_pixels_requires_explicit_revision(packet):
    path, data = packet
    data["advertisements"]["2"]["contours"]["glyph0"]["source_native_uv"][0][0] += .5
    path.write_text(json.dumps(data), encoding="utf8")
    with pytest.raises(ValueError, match="different content"):
        import_mixed_ad_template(path, CURRENT)


def test_repeated_import_has_stable_ids_and_does_not_edit_source(packet):
    path, _ = packet
    before = path.read_bytes()
    first = import_mixed_ad_template(path, CURRENT)
    second = import_mixed_ad_template(path, CURRENT)
    assert first == second
    assert before == path.read_bytes()


def test_explicit_current_pts_conflict_is_rejected(packet):
    path, _ = packet
    with pytest.raises(ValueError, match="conflicting.*pts"):
        import_mixed_ad_template(path, dict(CURRENT, native_pts=1))


def test_alias_metadata_cannot_change_physical_advertisement(packet):
    path, data = packet
    data["advertisements"]["2"]["contours"]["glyph0"]["advertisement_id"] = 15
    path.write_text(json.dumps(data), encoding="utf8")
    with pytest.raises(ValueError, match="different physical advertisement"):
        import_mixed_ad_template(path, CURRENT)
