"""Small synthetic metric comparisons; no fitting or local result writes."""
from copy import deepcopy

import pytest

from calibration.program_tracking.body_tracking.source_guided_metrics import compare_metrics


def row(pts=100, feature="runner_lower_edge", distances=(0., 10.), uv=None, suffix=""):
    uv = uv or [[0., 0.], [10., 0.]]
    return dict(id=f"full_source_{pts}_foot1_{feature}{suffix}", pts=pts, foot=1, role="fit",
                structure_feature=feature, source_component_budget_share=None,
                source_polyline_uv=uv, distance_px=list(distances), pixel_uncertainty_radius_px=2.,
                model_closest_uv=[[p[0], p[1]+d] for p, d in zip(uv, distances)])


def contract(pts=100, feature="runner_lower_edge"):
    return dict(pts=pts, foot=1, feature=feature, visibility="visible", completeness="complete",
                source_order="front_to_rear", source_range={"points_uv_original": [[0., 0.], [10., 0.]], "pixel_uncertainty_radius_px_original": 2.},
                source_regions={"silhouette_terminals": {"front_region": {"native_uv": [0., 0.], "uncertainty_radius_px": 2.},
                                                       "rear_region": {"native_uv": [10., 0.], "uncertainty_radius_px": 3.}}})


def document(rows):
    return dict(visible_runner_contours=rows, complete_runner_extents=[], source_support_locations=[],
                body_pixel_distance={"median": 4., "p95": 20.}, physical={"feasible": False, "scope": "synthetic collar only"})


def test_arc_length_statistics_are_invariant_to_more_clicks_and_keep_terminal_bands():
    before = document([row()])
    after = document([row(distances=(0., 5., 10.), uv=[[0., 0.], [5., 0.], [10., 0.]])])
    saved = deepcopy((before, after))
    result = compare_metrics(before, after, {(100, 1)}, {"rows": [contract()]})
    for phase in ("before", "after"):
        s = result[phase]["groups"]["old22"]["canonical_lower_source_to_model_px"]
        assert (s["median"], s["p95"], s["max"]) == pytest.approx((5., 9.5, 10.))
        assert s["outside_band_px"]["median"] == pytest.approx(3.)
        assert s["arc_coverage_fraction"] == pytest.approx(.2)
        assert result[phase]["groups"]["old22"]["canonical_lower_count"] == 1
    c = result["after"]["contours"][0]
    assert c["appearance_tail"]["outside_band_px"] == 7.  # declared rear region radius=3, not common band=2
    assert c["full_length_source_arc_coverage_fraction"] == pytest.approx(.2)
    assert result["after"]["body_pixel_distance_native_px"] == before["body_pixel_distance"]
    assert result["after"]["physical_diagnostics"] == before["physical"]
    assert not result["accuracy_pass_declared"] and not result["independent_heldout"]["available"]
    assert result["independent_heldout"]["source_exposure_count"] == 0
    assert (before, after) == saved


def test_reverse_uses_only_visible_mask_and_separates_legacy_two_sides():
    before, after = document([row()]), document([row()])
    reverse = dict(id="reverse", pts=100, foot=1, actual_model_runner_uv=[[[0., 0.], [10., 0.]], [[0., 1.], [10., 1.]]],
                   model_to_source_distance_px=[[1000., 1000.], [1., 1.]], pixel_uncertainty_radius_px=2.)
    before["complete_runner_extents"] = [deepcopy(reverse)]
    reverse["model_visible_mask"] = [[False, False], [True, True]]
    after["complete_runner_extents"] = [reverse]
    result = compare_metrics(before, after, {(100, 1)}, {"rows": [contract()]})
    a, b = (result[p]["groups"]["old22"] for p in ("before", "after"))
    assert a["visible_model_to_source_native_px"] is None
    assert a["legacy_unmasked_two_side_diagnostic"]["max"] == 1000.
    assert b["visible_model_to_source_native_px"]["max"] == 1.
    assert b["legacy_unmasked_two_side_diagnostic"] is None
    assert b["visible_model_to_source_native_px"]["arc_coverage_fraction"] == 1.
    reverse["model_visible_mask"] = [[False, False], [False, False]]
    result = compare_metrics(before, after, {(100, 1)}, {"rows": [contract()]})
    assert result["after"]["groups"]["old22"]["visible_model_to_source_native_px"] is None


def test_assembly_lower_components_do_not_duplicate_canonical_lower_sources():
    lower = row()
    assembly = row(feature="blade_assembly_outline", distances=(1., 1.))
    before = document([lower, assembly])
    first = row(feature="blade_assembly_outline", distances=(1., 1.), uv=[[0., 0.], [5., 0.]], suffix="_component0_boot_sole")
    first["structure_feature"] = "boot_sole"
    second = row(feature="blade_assembly_outline", distances=(1., 1.), uv=[[5., 0.], [10., 0.]], suffix="_component1_runner_lower_edge")
    second["structure_feature"] = "runner_lower_edge"
    # Early saved JSON has only the IDs; later JSON supplies the original family.
    for explicit in (False, True):
        if explicit:
            first["source_structure_feature"] = second["source_structure_feature"] = "blade_assembly_outline"
            first["source_component_budget_share"] = second["source_component_budget_share"] = .5
        result = compare_metrics(before, document([lower, first, second]), {(100, 1)}, {"rows": [contract(), contract(feature="blade_assembly_outline")]})
        old = result["after"]["groups"]["old22"]
        assert old["canonical_lower_count"] == 1 and old["original_contour_count"] == 2
        assert old["feature_row_count"] == 3
        assert len(result["contour_changes"]) == 2


def test_old_new_membership_missing_lower_and_per_exposure_regressions():
    before = document([row(distances=(1., 1.)), row(pts=200, distances=(1., 1.))])
    after = document([row(distances=(1., 6.)), row(pts=200, distances=(1., 1.))])
    c = {"rows": [contract(), contract(pts=200), contract(pts=300)]}
    result = compare_metrics(before, after, {(100, 1)}, c)
    assert result["after"]["groups"]["old22"]["source_exposure_count"] == 1
    assert result["after"]["groups"]["new_source"]["expected_source_exposure_count"] == 2
    assert result["after"]["groups"]["new_source"]["missing_canonical_lower_keys"] == [[300, 1]]
    regression, = result["exposure_regressions"]
    assert (regression["pts"], regression["foot"]) == (100, 1)
    assert "appearance_tail_outside_source_band_increased" in regression["source_changes"][0]["regression_flags"]
    assert "polyline_end_outside_source_band_increased" in regression["source_changes"][0]["regression_flags"]


def test_xy_keeps_each_conditional_source_and_uses_no_global_tolerance():
    before, after = document([row()]), document([row()])
    before["source_support_locations"] = [dict(pts=100, foot=1, outside_source_range_xy_conditional_m=[.03, .04], source_range_model="synthetic finite range")]
    after["source_support_locations"] = [dict(pts=100, foot=1, outside_source_range_xy_conditional_m=[.06, .08], source_range_model="synthetic finite range")]
    result = compare_metrics(before, after, {(100, 1)}, {"rows": [contract()]})
    xy = result["after"]["xy"]
    assert xy["outside_distance_conditional_m"]["median"] == pytest.approx(.1)
    assert xy["source_count"] == 1 and not xy["threshold_applied"]
    assert xy["per_source"][0]["source_range_model"] == "synthetic finite range"
    assert result["xy_changes"][0]["delta_conditional_m"] == pytest.approx(.05)


def test_changed_source_band_cannot_masquerade_as_improvement():
    before, after = document([row()]), document([row()])
    after["visible_runner_contours"][0]["pixel_uncertainty_radius_px"] = 100.
    result = compare_metrics(before, after, {(100, 1)}, {"rows": [contract()]})
    change, = result["contour_changes"]
    assert not change["source_range_comparable"] and not change["deltas"]
