import numpy as np
import pytest

from rinkspace.session.constraint_sampling import sample_projective_constraint_rows


def _sample(uv, normals, *, pts=None, entities=None, ids=None, kind=None, **options):
    n = len(uv)
    return sample_projective_constraint_rows(
        np.asarray(uv, dtype=float).reshape(n, 2),
        np.asarray(normals, dtype=float).reshape(n, 2),
        np.full(n, 256, dtype=np.int64) if pts is None else pts,
        np.full(n, "physical/curve") if entities is None else entities,
        np.array([f"source-{i:04}" for i in range(n)], dtype="U32") if ids is None else ids,
        np.ones(n, dtype=np.uint8) if kind is None else kind,
        image_width=options.pop("image_width", 1920),
        image_height=options.pop("image_height", 1080),
        **options,
    )


def _ellipse(a=620., b=270., count=121):
    theta = np.linspace(-.8, 3.7, count)
    uv = np.c_[960. + a*np.cos(theta), 540. + b*np.sin(theta)]
    normals = np.c_[np.cos(theta)/a, np.sin(theta)/b]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return uv, normals


@pytest.mark.parametrize("slope", [0., .35, -1.1])
def test_a_source_line_has_two_independent_normal_constraints(slope):
    x = np.linspace(100., 1500., 201)
    uv = np.c_[x, 150. + slope*x]
    normal = np.tile([-slope, 1.], (len(uv), 1))
    result = _sample(uv, normal)
    assert result["groups"] == [dict(pts=256, physical_entity_id="physical/curve",
        rank=2, available=201, unique_available=201, selected=2)]
    assert len(result["selected_indices"]) == 2


def test_model_line_positions_keep_rank_two_without_replacing_noisy_source_uv():
    x = np.linspace(100., 1500., 101)
    geometry = np.c_[x, 150. + .35*x]
    geometry[50] = geometry[49]  # Model coincidence must not deduplicate distinct source UV.
    uv = geometry + np.random.default_rng(31).normal(0., 2., geometry.shape)
    normal = np.tile([-.35, 1.], (len(uv), 1))
    originals = [array.copy() for array in (uv, geometry, normal)]
    for array in (uv, geometry, normal):
        array.flags.writeable = False
    assert _sample(uv, normal)["groups"][0]["rank"] > 2
    result = _sample(uv, normal, geometry_uv=geometry)
    assert result["groups"][0]["rank"] == 2
    assert result["groups"][0]["unique_available"] == len(uv)
    selected = result["selected_indices"]
    assert selected.dtype == np.int64 and len(selected) == 2
    np.testing.assert_array_equal(uv[selected], originals[0][selected])
    assert not np.array_equal(uv[selected], geometry[selected])
    for array, original in zip((uv, geometry, normal), originals):
        np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize("a,b", [(310., 310.), (620., 270.)])
def test_circle_arcs_and_ellipses_supply_five_projective_constraints(a, b):
    uv, normals = _ellipse(a, b)
    result = _sample(uv, normals)
    assert result["groups"][0]["rank"] == 5
    assert result["groups"][0]["selected"] == 5
    selected = result["selected_indices"]
    independent = _sample(uv[selected], normals[selected])
    assert independent["groups"][0]["rank"] == 5


def test_general_fixed_points_span_eight_dimensions_and_keep_original_observations():
    uv = np.array([[100., 100.], [1700., 120.], [1650., 950.], [150., 980.],
                   [700., 350.], [1200., 800.], [400., 550.]])
    result = _sample(uv, np.zeros_like(uv), kind=np.zeros(len(uv), dtype=np.uint8))
    assert result["groups"][0]["rank"] == 8
    selected = result["selected_indices"]
    assert 4 <= len(selected) <= 7
    assert _sample(uv[selected], np.zeros((len(selected), 2)),
        kind=np.zeros(len(selected), dtype=np.uint8))["groups"][0]["rank"] == 8


def test_row_permutation_and_duplicate_uv_preserve_selected_geometry():
    uv, normals = _ellipse()
    ids = np.array([f"original-{i:04}" for i in range(len(uv))])
    baseline = _sample(uv, normals, ids=ids)
    expected = set(map(tuple, uv[baseline["selected_indices"]]))
    repeated = np.r_[np.arange(len(uv)), np.repeat(np.arange(10, 100), 4)]
    # Smaller duplicate IDs deliberately replace provenance representatives.
    # They must not reorder the canonical geometry or create additional rank.
    dense_ids = np.r_[ids, [f"a-copy-{i:04}" for i in range(len(repeated)-len(uv))]]
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(len(repeated))
        points = uv[repeated][order]
        result = _sample(points, normals[repeated][order], ids=dense_ids[order])
        assert set(map(tuple, points[result["selected_indices"]])) == expected
        assert result["groups"][0]["rank"] == 5
        assert result["groups"][0]["available"] == len(repeated)
        assert result["groups"][0]["unique_available"] == len(uv)


def test_duplicate_provenance_is_selected_stably_and_fixed_supersedes_curve():
    uv = np.array([[20., 20.], [20., 20.], [20., 20.]])
    result = _sample(uv, np.tile([0., 1.], (3, 1)),
        ids=np.array(["z-fixed", "a-fixed", "0-curve"]), kind=np.array([0, 0, 1]))
    np.testing.assert_array_equal(result["selected_indices"], [1])
    assert result["groups"][0]["rank"] == 2


def test_exposures_and_physical_entities_have_separate_rank_groups():
    uv = np.tile(np.c_[np.linspace(100., 1500., 31), np.full(31, 300.)], (4, 1))
    normals = np.tile([0., 1.], (len(uv), 1))
    times = np.repeat([256, 256, 512, 512], 31)
    entities = np.repeat(["one", "two", "one", "two"], 31)
    result = _sample(uv, normals, pts=times, entities=entities)
    assert [(g["pts"], g["physical_entity_id"], g["rank"], g["selected"])
        for g in result["groups"]] == [(256, "one", 2, 2), (256, "two", 2, 2),
                                     (512, "one", 2, 2), (512, "two", 2, 2)]
    assert len(result["selected_indices"]) == 8


def test_rank_tolerance_is_explicit_and_inputs_are_unchanged():
    uv, normals = _ellipse()
    times = np.full(len(uv), 256, dtype=np.int64)
    entities = np.full(len(uv), "physical/curve")
    ids = np.array([f"id-{i:04}" for i in range(len(uv))])
    kind = np.ones(len(uv), dtype=np.uint8)
    arrays = [uv, normals, times, entities, ids, kind]
    originals = [a.copy() for a in arrays]
    for array in arrays:
        array.flags.writeable = False
    tight = _sample(uv, normals, pts=times, entities=entities, ids=ids, kind=kind,
                    rank_rtol=1e-8)
    loose = _sample(uv, normals, pts=times, entities=entities, ids=ids, kind=kind,
                    rank_rtol=.5)
    assert tight["groups"][0]["rank"] == 5
    assert loose["groups"][0]["rank"] < tight["groups"][0]["rank"]
    for array, original in zip(arrays, originals):
        np.testing.assert_array_equal(array, original)


def test_empty_input_has_no_rank_groups():
    result = _sample([], [])
    assert result["selected_indices"].dtype == np.int64
    assert result["selected_indices"].size == 0
    assert result["groups"] == []


@pytest.mark.parametrize("option,value", [("image_width", 0), ("image_height", np.inf),
    ("rank_rtol", 0), ("rank_rtol", 1), ("rank_rtol", np.nan)])
def test_invalid_image_dimensions_or_rank_tolerance_are_rejected(option, value):
    with pytest.raises(ValueError, match=option):
        _sample([[100., 100.]], [[0., 1.]], **{option: value})


def test_unknown_curve_normal_cannot_create_a_constraint():
    with pytest.raises(ValueError, match="curve normals"):
        _sample([[100., 100.]], [[0., 0.]])


@pytest.mark.parametrize("geometry", [np.array([100., 100.]), np.zeros((2, 2)),
    np.array([[np.nan, 100.]]), np.array([[100., np.inf]]), np.array([["x", "y"]])])
def test_geometry_positions_require_matching_finite_numeric_uv(geometry):
    with pytest.raises(ValueError, match="geometry_uv"):
        _sample([[100., 100.]], [[0., 1.]], geometry_uv=geometry)
