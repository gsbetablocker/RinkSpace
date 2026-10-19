"""A local candidate must recover known motion without seeing protected pixels."""
import cv2
import numpy as np

from rinkspace.masked_photometric import estimate_photometric


def scene():
    rng = np.random.default_rng(17)
    gray = cv2.GaussianBlur(rng.uniform(.2, .7, (180, 260)), (7, 7), 1.3)
    # Broad, smooth structure prevents the tests from being only random speckle.
    y, x = np.indices(gray.shape)
    gray += .08*np.sin(x/17) + .04*np.cos(y/23)
    return gray


def project(h, xy):
    q = np.c_[xy, np.ones(len(xy))] @ np.asarray(h).T
    return q[:, :2]/q[:, 2:]


def case():
    ref = scene()
    h = np.array([[1.001, .0015, -1.7], [-.0012, .999, 1.1], [8e-6, -7e-6, 1.]])
    current = cv2.warpPerspective(ref, np.linalg.inv(h), (260, 180))
    current = 1.12*current + .026
    ref_mask = np.full(ref.shape, 255, np.uint8)
    cur_mask = ref_mask.copy()
    ref_mask[65:105, 100:135] = 0
    cur_mask[62:110, 98:140] = 0
    ref_mask[:8] = ref_mask[-8:] = 0
    cur_mask[:8] = cur_mask[-8:] = 0
    return ref, current, ref_mask, cur_mask, h


def test_recovers_translation_tilt_and_exposure_as_candidate():
    ref, cur, rm, cm, truth = case()
    result = estimate_photometric(ref, cur, rm, cm, np.eye(3))
    assert result["status"] == "candidate", result["reason"]
    probes = np.array([[25, 25], [230, 25], [230, 155], [25, 155], [160, 130]])
    error = np.linalg.norm(project(result["H_reference_from_current"], probes)-project(truth, probes), axis=1)
    assert max(error) < .12
    assert abs(result["gain"]-1.12) < .035
    assert abs(result["bias"]-.026) < .02
    assert result["independent_validation"] is False
    assert len(result["fit_current_xy"]) >= 160
    assert result["final_rmse_0_to_1"] < result["initial_rmse_0_to_1"]


def test_values_inside_both_protected_regions_cannot_change_any_fit_or_trace():
    ref, cur, rm, cm, _ = case()
    baseline = estimate_photometric(ref, cur, rm, cm, np.eye(3))
    # NaN and large illegal values would break a solver that examines excluded data.
    altered_ref, altered_cur = ref.copy(), cur.copy()
    altered_ref[rm == 0] = np.nan
    altered_cur[cm == 0] = 999999.
    changed = estimate_photometric(altered_ref, altered_cur, rm, cm, np.eye(3))
    assert changed["status"] == "candidate"
    assert changed == baseline


def test_every_recorded_evaluation_has_a_complete_safe_sampling_footprint():
    ref, cur, rm, cm, _ = case()
    result = estimate_photometric(ref, cur, rm, cm, np.eye(3))
    pool = np.asarray(result["sampling_pool_current_xy"])
    rs = cv2.erode(rm, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    cs = cv2.erode(cm, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    assert np.all(cs[pool[:, 1], pool[:, 0]] > 0)
    assert len(result["iterations"]) > 1
    for row in result["iterations"]:
        ids = np.array([i for start, end in row["valid_pool_index_ranges"] for i in range(start, end)])
        assert len(ids) == row["valid_count"]
        xy = project(row["H_reference_from_current"], pool[ids])
        ix, iy = np.floor(xy).astype(int).T
        assert np.all(rs[iy, ix]) and np.all(rs[iy, ix+1])
        assert np.all(rs[iy+1, ix]) and np.all(rs[iy+1, ix+1])


def test_flat_image_refused_even_if_identity_has_zero_residual():
    ref = np.full((160, 240), 205, np.uint8)
    mask = np.full(ref.shape, 255, np.uint8)
    result = estimate_photometric(ref, ref, mask, mask, np.eye(3))
    assert result["status"] == "unavailable"
    assert result["H_reference_from_current"] is None
    assert result["reason"] == "insufficient_texture"


def test_unrelated_textured_images_cannot_be_promoted_to_candidate():
    ref = scene()
    current = cv2.GaussianBlur(np.random.default_rng(81).uniform(.2, .7, ref.shape), (7, 7), 1.3)
    mask = np.full(ref.shape, 255, np.uint8)
    result = estimate_photometric(ref, current, mask, mask, np.eye(3))
    assert result["status"] == "unavailable"
    assert result["H_reference_from_current"] is None
    assert result["reason"] == "insufficient_photometric_correlation"


def test_one_directional_texture_does_not_determine_a_homography():
    x = np.arange(260)
    ref = np.tile(.5 + .15*np.sin(x/8), (180, 1))
    mask = np.full(ref.shape, 255, np.uint8)
    result = estimate_photometric(ref, ref, mask, mask, np.eye(3))
    assert result["status"] == "unavailable"
    assert result["reason"] == "photometric_system_rank_deficient"
    assert result["H_reference_from_current"] is None


def test_invalid_initial_horizon_is_refused_without_sampling_across_it():
    ref = scene()
    mask = np.full(ref.shape, 255, np.uint8)
    horizon = np.array([[1., 0, 0], [0, 1., 0], [-.01, 0, 1.]])
    result = estimate_photometric(ref, ref, mask, mask, horizon)
    assert result["status"] == "unavailable"
    assert result["reason"] == "initial_homography_has_a_horizon_or_reflection"
    assert result["iterations"] == []


def test_affine_mode_recovers_affine_exposure_without_perspective_parameters():
    ref, _, rm, cm, truth = case()
    truth[2,:2] = 0
    current = 1.12*cv2.warpPerspective(ref,np.linalg.inv(truth),(260,180)) + .026
    result = estimate_photometric(ref,current,rm,cm,np.eye(3),motion_model="affine")
    assert result["status"] == "candidate"
    assert result["optimized_parameter_count_including_gain_bias"] == 8
    np.testing.assert_array_equal(np.asarray(result["H_reference_from_current"])[2], [0,0,1])
    probes = np.array([[25,25],[230,25],[230,155],[25,155]])
    assert np.max(np.linalg.norm(project(result["H_reference_from_current"],probes)-project(truth,probes),axis=1)) < .12
    assert "approximation" in result["model_assumption"]


def test_explicit_prior_and_affine_modes_preserve_mask_value_invariance():
    ref, cur, rm, cm, _ = case()
    altered_ref, altered_cur = ref.copy(), cur.copy()
    altered_ref[rm==0] = np.nan
    altered_cur[cm==0] = -12345
    for model in ("affine","homography_regularized"):
        baseline = estimate_photometric(ref,cur,rm,cm,np.eye(3),motion_model=model)
        changed = estimate_photometric(altered_ref,altered_cur,rm,cm,np.eye(3),motion_model=model)
        assert baseline["status"] == "candidate"
        assert baseline == changed
        assert baseline["perspective_prior"]["weight"] == (.001 if model=="homography_regularized" else 0)
        assert baseline["perspective_prior"]["final_penalty"] >= 0


def test_similarity_has_four_geometric_dof_and_recovers_motion_and_exposure():
    ref, _, rm, cm, _ = case()
    truth = np.array([[1.004, -.007, -1.7], [.007, 1.004, 1.1], [0, 0, 1.]])
    cur = 1.12 * cv2.warpPerspective(ref, np.linalg.inv(truth), (260, 180)) + .026
    result = estimate_photometric(ref, cur, rm, cm, np.eye(3), motion_model="similarity")
    assert result["status"] == "candidate", result["reason"]
    assert result["geometric_parameter_count"] == 4
    assert result["optimized_parameter_count_including_gain_bias"] == 6
    assert "not_a_physical_camera_or_metric_model" in result["model_assumption"]
    probes = np.array([[25, 25], [230, 25], [230, 155], [25, 155]])
    assert max(np.linalg.norm(project(result["H_reference_from_current"], probes)
                              - project(truth, probes), axis=1)) < .12
    assert abs(result["gain"] - 1.12) < .035
    assert abs(result["bias"] - .026) < .02
    # Constraints hold throughout the actual solve, not just after projection.
    for row in result["iterations"]:
        if row["H_reference_from_current"] is None:
            continue
        h = np.asarray(row["H_reference_from_current"])
        np.testing.assert_allclose(h[0, 0], h[1, 1], atol=1e-13)
        np.testing.assert_allclose(h[0, 1], -h[1, 0], atol=1e-13)
        np.testing.assert_array_equal(h[2], [0, 0, 1])


def test_similarity_preserves_both_mask_value_isolation_and_refuses_one_direction():
    ref, cur, rm, cm, _ = case()
    baseline = estimate_photometric(ref, cur, rm, cm, np.eye(3), motion_model="similarity")
    ref[rm == 0] = np.nan
    cur[cm == 0] = -float('inf')
    assert estimate_photometric(ref, cur, rm, cm, np.eye(3), motion_model="similarity") == baseline
    assert baseline["status"] == "candidate"
    striped = np.tile(.5 + .15 * np.sin(np.arange(260) / 8), (180, 1))
    mask = np.full(striped.shape, 255, np.uint8)
    result = estimate_photometric(striped, striped, mask, mask, np.eye(3), motion_model="similarity")
    assert result["status"] == "unavailable"
    assert result["reason"] == "photometric_system_rank_deficient"
    assert result["H_reference_from_current"] is None
