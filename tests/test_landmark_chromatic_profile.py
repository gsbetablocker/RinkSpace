import numpy as np

from rinkspace.landmark_chromatic_profile import relative_chromatic_excess, measure_chromatic_profiles


def test_neutral_darkening_removes_local_ice_colour_cast():
    reference = np.array([[210., 205., 200.], [220., 215., 210.]])
    dark = reference * np.array([.7, .9])[:, None]
    np.testing.assert_allclose(relative_chromatic_excess(dark, reference), 0, atol=1e-12)


def test_desaturated_blue_band_support_does_not_require_hsv_saturation_twelve():
    image = np.full((160, 320, 3), 220, np.uint8)
    image[70:90] = [207, 203, 202]
    points = [[x, 70] for x in range(40, 281, 12)]
    observation = {'observation_id': 'measured', 'points_source_px': points,
                   'association_status': 'detected_unidentified', 'selected_model_id': 'ignored'}
    profile = measure_chromatic_profiles(image, [observation])[0]
    assert profile['colour_evidence'] == 'blue_ink_supported'
    assert profile['blue_excess_point_fraction'] == 1.
    assert profile['model_camera_or_landmark_identity_used'] is False
    assert observation['points_source_px'] == points


def test_neutral_and_masked_marks_stay_unknown_not_proven_gray():
    image = np.full((160, 320, 3), 220, np.uint8)
    image[70:90] = 200
    row = {'observation_id': 'neutral', 'points_source_px': [[x, 70] for x in range(40, 281, 12)]}
    profile = measure_chromatic_profiles(image, [row])[0]
    assert profile['colour_evidence'] == 'unknown'
    assert profile['absence_of_blue_support_is_not_gray_identity'] is True
    image[70:90] = [207, 203, 202]
    row['association_status'] = 'masked_out'
    profile = measure_chromatic_profiles(image, [row])[0]
    assert profile['colour_evidence'] == 'unknown' and profile['reliable_point_count'] == 0
