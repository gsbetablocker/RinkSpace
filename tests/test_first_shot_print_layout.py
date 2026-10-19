"""One shared print map must not move source points or the outer canvas."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from calibration.solver_runtime import install
install()

from calibration.program_tracking.first_shot_print_layout import (
    LAYOUT_PARAMETER_NAMES, PrintLayoutDomainError, apply_ad2_print_layout,
    ad2_print_world_layout_jacobian, material_world_layout_jacobian,
    validate_layout, warp_normalized_material_uv, apply_print_layout,
    ad_print_world_layout_jacobian, print_layout_template, SUPPORTED_ADVERTISEMENT_IDS,
)
from calibration.program_tracking.framewise_model import feature_world


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "events/ankara_2026/models/framewise_8a28_first_shot_20261014/repair_20261014/ad2_material_template.json"


def test_centered_homography_all_eight_derivatives_match_finite_difference():
    uv = np.array([[.16416931, .27058821], [.5, .5], [.86644928, .68627433]])
    q = np.array([.1, -.03, .04, -.06, .01, -.02, .08, -.05])
    _, jacobian = warp_normalized_material_uv(uv, q)
    assert LAYOUT_PARAMETER_NAMES == ("a", "b", "c", "d", "tx", "ty", "p", "q")
    for column in range(8):
        plus, minus = q.copy(), q.copy(); plus[column] += 1e-6; minus[column] -= 1e-6
        fd = (warp_normalized_material_uv(uv, plus)[0]
              - warp_normalized_material_uv(uv, minus)[0]) / 2e-6
        np.testing.assert_allclose(jacobian[:, :, column], fd, rtol=2e-8, atol=5e-11)


def test_zero_layout_retains_material_bits_and_template_fields_exactly():
    uv = np.array([[.16416931, .27058821], [.86644928, .68627433]])
    warped, _ = warp_normalized_material_uv(uv, np.zeros(8))
    assert warped.tobytes() == uv.tobytes() and not np.shares_memory(warped, uv)
    templates = json.loads(TEMPLATE.read_text(encoding="utf8"))
    before = json.dumps(templates, sort_keys=True, separators=(",", ":"))
    current = apply_ad2_print_layout(templates, np.zeros(8))
    assert json.dumps(current, sort_keys=True, separators=(",", ":")) == before
    assert current is not templates and current["contours"] is not templates["contours"]


@pytest.mark.parametrize("q", [
    [-1, 0, 0, 0, 0, 0, 0, 0],  # degenerate
    [-2, 0, 0, 0, 0, 0, 0, 0],  # reversed orientation
    [0, 0, 0, 0, 0, 0, 2, 0],   # pole at a unit-canvas side
    [0, 0, 0, 0, 0, 0, 2.1, 0], # negative denominator despite sampled print remaining positive
    [0, 0, 0, 0, np.nan, 0, 0, 0],
])
def test_true_homography_domain_failures_are_typed(q):
    with pytest.raises(PrintLayoutDomainError):
        validate_layout(q, [[.45, .45], [.55, .55]])


def test_large_valid_parameters_have_no_arbitrary_cap():
    # This map is valid projectively; physical template containment is a
    # separate real-canvas check, not a made-up bound on tx or scale.
    q = np.array([1e100, 0, 0, 1e100, 1e100, -1e100, 0, 0])
    warped, jacobian = warp_normalized_material_uv([[.2, .3], [.7, .8]], q)
    assert np.isfinite(warped).all() and np.isfinite(jacobian).all()


def test_actual_source_template_only_print_changes_and_prototype_is_preserved():
    templates = json.loads(TEMPLATE.read_text(encoding="utf8")); before = deepcopy(templates)
    q = np.array([-.15, .01, -.02, -.1, .03, -.025, .025, .02])
    current = apply_ad2_print_layout(templates, q)
    assert templates == before and current["advertisements"] == before["advertisements"]
    assert len(current["contours"]) == len(before["contours"]) == 17
    for key, record in current["contours"].items():
        original = before["contours"][key]
        assert record["prototype_material_uv"] == original["material_uv"]
        assert record["source_native_uv"] == original["source_native_uv"]
        assert record["closed"] == original["closed"]
        expected = warp_normalized_material_uv(original["material_uv"], q)[0]
        np.testing.assert_array_equal(record["material_uv"], expected)
        if record["closed"]:
            np.testing.assert_array_equal(record["material_uv"][0], record["material_uv"][-1])
    # Reusing an explicit laid-out copy still maps from original prototype,
    # rather than repeatedly compounding the map.
    assert apply_ad2_print_layout(current, q) == current


def test_actual_canvas_domain_refuses_print_outside_without_clamping():
    templates = json.loads(TEMPLATE.read_text(encoding="utf8"))
    q = np.zeros(8); q[4] = 1
    validate_layout(q)
    with pytest.raises(PrintLayoutDomainError, match="actual outer canvas"):
        apply_ad2_print_layout(templates, q)


class StraightMaterial:
    def __init__(self):
        self.calls = 0

    def material(self, advertisement_id, u, v_up, q):
        assert advertisement_id == 2
        self.calls += 1
        return np.column_stack((2+6*u, np.zeros(len(u)), .5+2*v_up)), None, {}


def scene_and_template(*, explicit_closure=False):
    scene = dict(ice_boundary=dict(points_xyz=[[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]]),
        board_surface=dict(top_edge_xyz=[[0, 0, 3], [10, 0, 3], [10, 10, 3], [0, 10, 3]]),
        advertisements=[dict(user_number=2, extent_scope="external_border",
            corners_xyz_scene_units=dict(left_top=[2, 0, 2.5], left_bottom=[2, 0, .5],
                right_top=[8, 0, 2.5], right_bottom=[8, 0, .5]))])
    uv = [[.2, .3], [.7, .3], [.7, .65], [.2, .65]]
    if explicit_closure:
        uv.append(uv[0].copy())
    template = dict(advertisement_number=2, contours=dict(loop=dict(material_uv=uv, closed=True,
        source_native_uv=[[192, 300]], material_coordinate_domain="Ad2_external_purple_panel_unit_rectangle")))
    return scene, template


@pytest.mark.parametrize("explicit_closure", [True, False])
def test_world_chain_v_direction_and_exact_closure_match_full_layout_fd(explicit_closure):
    scene, templates = scene_and_template(explicit_closure=explicit_closure)
    target = dict(kind="ad_print", advertisement_id=2, contour_key="loop")
    q = np.array([.05, .01, -.02, -.04, .03, .015, .04, -.06])
    geometry = StraightMaterial()
    world, jacobian = ad2_print_world_layout_jacobian(geometry, scene, target,
        templates, q, np.zeros(100))
    assert geometry.calls == 4 and world.shape == (5, 3) and jacobian.shape == (5, 3, 8)
    np.testing.assert_array_equal(world[-1], world[0])
    np.testing.assert_array_equal(jacobian[-1], jacobian[0])
    for column in range(8):
        plus, minus = q.copy(), q.copy(); plus[column] += 1e-6; minus[column] -= 1e-6
        fd = (feature_world(scene, target, apply_ad2_print_layout(templates, plus))
              - feature_world(scene, target, apply_ad2_print_layout(templates, minus))) / 2e-6
        np.testing.assert_allclose(jacobian[:, :, column], fd, rtol=5e-8, atol=8e-10)
    # Positive ty moves down the physical material, never up through a second flip.
    assert np.all(jacobian[:, 2, 5] < 0)
    for role in ("left_top", "left_bottom", "right_top", "right_bottom"):
        corner = dict(kind="ad_corner", advertisement_id=2, role=role)
        np.testing.assert_array_equal(feature_world(scene, corner, templates),
            feature_world(scene, corner, apply_ad2_print_layout(templates, q)))


def test_missing_material_interface_is_not_reclassified_as_physical_domain():
    class Missing:
        def material(self, *args):
            raise KeyError("Missing model interface")
    with pytest.raises(KeyError, match="Missing model interface"):
        material_world_layout_jacobian(Missing(), 2, [[.2, .3], [.7, .8]], np.zeros(8), np.zeros(100))


@pytest.mark.parametrize("number", SUPPORTED_ADVERTISEMENT_IDS)
def test_actual_approved_scene_identity_world_is_bit_exact_for_every_print(number):
    binding_path = ROOT / "events/ankara_2026/programmes/okada_mei_sp/source_datasets/shot1_fit_20261014/fit_input.json"
    binding = json.loads(binding_path.read_text(encoding="utf8"))
    model = json.loads(Path(binding["model"]["path"]).read_text(encoding="utf8"))
    scene = json.loads(Path(model["scene"]["path"]).read_text(encoding="utf8"))
    templates = json.loads(TEMPLATE.read_text(encoding="utf8"))
    identity = apply_print_layout(templates, number, np.zeros(8))
    assert json.dumps(identity, sort_keys=True) == json.dumps(templates, sort_keys=True)
    for key in print_layout_template(templates, number)["contours"]:
        target = dict(kind="ad_print", advertisement_id=number, contour_key=key)
        old_world, new_world = feature_world(scene, target, templates), feature_world(scene, target, identity)
        assert old_world.tobytes() == new_world.tobytes()


@pytest.mark.parametrize("number", SUPPORTED_ADVERTISEMENT_IDS)
def test_each_actual_print_layout_preserves_other_ads_and_all_source_metadata(number):
    templates = json.loads(TEMPLATE.read_text(encoding="utf8")); before = deepcopy(templates)
    q = np.array([-.1, .005, -.01, -.1, .006, -.004, .02, -.01])
    current = apply_print_layout(templates, number, q)
    assert templates == before
    for other in SUPPORTED_ADVERTISEMENT_IDS:
        original = print_layout_template(templates, other)
        changed = print_layout_template(current, other)
        if other != number:
            # The actual Ad2 material is also the compound file's root; its
            # sibling-advertisement subtree may contain the one changed ad.
            assert {k: v for k, v in changed.items() if k != "advertisements"} == {
                k: v for k, v in original.items() if k != "advertisements"}
        else:
            for key, record in changed["contours"].items():
                assert record["prototype_material_uv"] == original["contours"][key]["material_uv"]
                for field in original["contours"][key]:
                    if field != "material_uv":
                        assert record[field] == original["contours"][key][field]
                expected = warp_normalized_material_uv(original["contours"][key]["material_uv"], q)[0]
                np.testing.assert_array_equal(record["material_uv"], expected)
    for key in templates["advertisements"]:
        if int(key) != number:
            assert current["advertisements"][key] == templates["advertisements"][key]


@pytest.mark.parametrize("number", [8, 9, 10, 11, 12, True, 2.0])
def test_existing_shape_interfaces_do_not_get_duplicate_layouts(number):
    templates = json.loads(TEMPLATE.read_text(encoding="utf8"))
    with pytest.raises(ValueError, match="Ad2/3/4/5/6/7 only"):
        apply_print_layout(templates, number, np.zeros(8))


def test_actual_ad4_atlas_and_tail_share_one_layout_and_keep_original_facts():
    templates = json.loads(TEMPLATE.read_text(encoding="utf8")); before = deepcopy(templates)
    original = print_layout_template(templates, 4)
    assert original["schema"] == "rinkspace.same-physical-print-current-source-atlas.v1"
    assert len(original["contours"]) == 19
    assert sum(len(c["material_uv"]) for c in original["contours"].values()) == 1071
    q = np.array([-.1, .005, -.01, -.1, .006, -.004, .02, -.01])
    current = apply_print_layout(templates, 4, q)
    material = print_layout_template(current, 4)
    for field, value in original.items():
        if field != "contours":
            assert material[field] == value
    assert any(key.startswith("current_tail") for key in material["contours"])
    for key, record in material["contours"].items():
        assert record["material_coordinate_domain"] == "ad_rectangle_u_left_to_right_v_top_to_bottom"
        assert record["prototype_material_uv"] == original["contours"][key]["material_uv"]
        np.testing.assert_array_equal(record["material_uv"],
            warp_normalized_material_uv(original["contours"][key]["material_uv"], q)[0])
    assert templates == before


def test_actual_ad5_extended_internal_graphic_domain_is_retained_without_unit_clipping():
    templates = json.loads(TEMPLATE.read_text(encoding="utf8")); before = deepcopy(templates)
    material = print_layout_template(templates, 5)
    prototype = np.concatenate([np.asarray(c["material_uv"], float) for c in material["contours"].values()])
    assert len(prototype) == 824
    assert int(((prototype < 0) | (prototype > 1)).any(axis=1).sum()) == 678
    q = np.array([-.1, .005, -.01, -.1, .006, -.004, .02, -.01])
    current = apply_print_layout(templates, 5, q)
    result = print_layout_template(current, 5)
    for key, record in result["contours"].items():
        assert record["material_coordinate_domain"] == material["contours"][key]["material_coordinate_domain"]
        assert record["material_coordinate_domain"] == "continuous_board_s_h_relative_to_named_internal_graphic"
        np.testing.assert_array_equal(record["material_uv"],
            warp_normalized_material_uv(material["contours"][key]["material_uv"], q)[0])
    warped = np.concatenate([np.asarray(c["material_uv"], float) for c in result["contours"].values()])
    assert (warped[:, 0] < 0).any() and (warped[:, 0] > 1).any()
    assert templates == before


def test_ad5_projective_pole_on_real_prototype_hull_is_rejected_even_if_internal_unit_is_valid():
    templates = json.loads(TEMPLATE.read_text(encoding="utf8"))
    material = print_layout_template(templates, 5)
    prototype = np.concatenate([np.asarray(c["material_uv"], float) for c in material["contours"].values()])
    q = np.zeros(8); q[6] = -1
    validate_layout(q)  # Positive on all four internal-graphic unit corners.
    with pytest.raises(PrintLayoutDomainError, match="nonpositive pole"):
        validate_layout(q, prototype)
    with pytest.raises(PrintLayoutDomainError, match="nonpositive pole"):
        apply_print_layout(templates, 5, q)


def test_missing_named_advertisement_never_falls_back_to_ad2_material():
    _, templates = scene_and_template()
    with pytest.raises(ValueError, match="Missing existing Ad3"):
        apply_print_layout(templates, 3, np.zeros(8))


@pytest.fixture(scope="module")
def actual_state():
    from calibration.program_tracking.first_shot_physical_state import FirstShotPhysicalState
    return FirstShotPhysicalState.from_binding(ROOT / "events/ankara_2026/programmes/okada_mei_sp/source_datasets/shot1_fit_20261014/fit_input.json")


@pytest.mark.parametrize("number", SUPPORTED_ADVERTISEMENT_IDS)
def test_each_actual_public_material_derivative_matches_exact_feature_world(actual_state, number):
    state = actual_state; q = state.initial_q.copy(); scene = state.scene(q)
    material = print_layout_template(state.templates, number)
    # Use a complete real contour per ID, including its physical closure.
    key = next(iter(material["contours"]))
    record = material["contours"][key]; uv = np.asarray(record["material_uv"], float)
    target = dict(kind="ad_print", advertisement_id=number, contour_key=key)
    old_world = feature_world(scene, target, state.templates)
    native_world = state.provider.current.material(number, uv[:, 0], 1-uv[:, 1], q[:100])[0]
    if len(old_world) > len(native_world):
        native_world = np.vstack((native_world, native_world[:1]))
    # The public physical lifting computes equivalent station/height values
    # with roundoff; do not falsely claim it preserves the old bytes.
    np.testing.assert_allclose(native_world, old_world, rtol=0, atol=1e-10)
    exact_world, identity_j = ad_print_world_layout_jacobian(state.provider.current,
        scene, target, state.templates, np.zeros(8), q[:100])
    assert exact_world.tobytes() == old_world.tobytes()
    layout = np.array([-.1, .005, -.01, -.1, .006, -.004, .02, -.01])
    _, jacobian = ad_print_world_layout_jacobian(state.provider.current,
        scene, target, state.templates, layout, q[:100])
    assert np.isfinite(identity_j).all()
    for column in range(8):
        plus, minus = layout.copy(), layout.copy(); plus[column] += 1e-6; minus[column] -= 1e-6
        fd = (feature_world(scene, target, apply_print_layout(state.templates, number, plus))
              - feature_world(scene, target, apply_print_layout(state.templates, number, minus))) / 2e-6
        np.testing.assert_allclose(jacobian[:, :, column], fd, rtol=2e-6, atol=4e-8)
    if record["closed"]:
        np.testing.assert_array_equal(jacobian[0], jacobian[-1])
    for role in ("left_top", "left_bottom", "right_top", "right_bottom"):
        corner = dict(kind="ad_corner", advertisement_id=number, role=role)
        np.testing.assert_array_equal(feature_world(scene, corner, state.templates),
            feature_world(scene, corner, apply_print_layout(state.templates, number, layout)))
