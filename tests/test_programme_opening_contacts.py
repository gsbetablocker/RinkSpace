"""Opening contact rows preserve source exposures and selected steel points."""
import numpy as np
import pytest

from calibration.program_tracking.body_tracking import programme_opening


@pytest.fixture
def opening_sections():
    height = np.full((6, 2, 5), -.9)
    # Only 0, 2 and 3 are eligible. Slots 1 and 4 are lower on purpose.
    height[1] = [[.018, -.48, .001, .008, -.30], [.067, -.20, .070, .061, -.70]]
    height[3] = [[.006, -.31, .012, .010, -.50], [.011, -.20, -.002, .004, -.70]]
    height[5] = [[.024, -.81, .021, .018, -.12], [.004, -.66, .008, .010, -.40]]
    points = np.full((6, 2, 5, 3), 99.)
    points[1, 0, 2] = [.17, -.010, -.022]
    points[1, 1, 3] = [.22, .015, .001]
    points[3, 0, 0] = [.09, -.021, -.013]
    points[3, 1, 2] = [.18, .014, -.018]
    points[5, 0, 3] = [.26, -.019, -.006]
    points[5, 1, 0] = [.11, .019, -.025]
    return dict(height=height, local_point=points, eligible=np.array([True, False, True, True, False]))


def test_both_feet_keeps_the_suspended_skate_and_matches_selected_steel_points(opening_sections):
    quiet = np.array([1, 3, 5])
    pts = np.array([270336, 270592, 270848, 271104, 271360, 271616])
    selected = programme_opening.standing_contact_rows(opening_sections, quiet, 'both_feet')

    np.testing.assert_array_equal(selected['rows'], [1, 1, 3, 3, 5, 5])
    np.testing.assert_array_equal(pts[selected['rows']], [270592, 270592, 271104, 271104, 271616, 271616])
    np.testing.assert_array_equal(selected['feet'], [0, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(selected['features'], [2, 3, 0, 2, 3, 0])
    # At the first exposure the left steel is 1 mm from the plane and the
    # right steel is 61 mm above it. Both must reach the standing residual.
    np.testing.assert_array_equal(selected['heights'], [.001, .061, .006, -.002, .018, .004])
    np.testing.assert_array_equal(selected['height_by_foot'], [[.001, .061], [.006, -.002], [.018, .004]])
    np.testing.assert_array_equal(selected['minimum_height'], [.001, -.002, .004])
    np.testing.assert_array_equal(selected['points'], [
        [.17, -.010, -.022], [.22, .015, .001], [.09, -.021, -.013],
        [.18, .014, -.018], [.26, -.019, -.006], [.11, .019, -.025],
    ])
    np.testing.assert_array_equal(quiet, [1, 3, 5])


def test_legacy_anonymous_mode_keeps_the_original_lowest_foot_and_witness(opening_sections):
    selected = programme_opening.standing_contact_rows(opening_sections, np.array([1, 3, 5]), 'anonymous_minimum')

    np.testing.assert_array_equal(selected['rows'], [1, 3, 5])
    np.testing.assert_array_equal(selected['feet'], [0, 1, 1])
    np.testing.assert_array_equal(selected['features'], [2, 2, 0])
    np.testing.assert_array_equal(selected['heights'], [.001, -.002, .004])
    np.testing.assert_array_equal(selected['height_by_foot'], [[.001, .061], [.006, -.002], [.018, .004]])
    np.testing.assert_array_equal(selected['minimum_height'], [.001, -.002, .004])
    np.testing.assert_array_equal(selected['points'], [
        [.17, -.010, -.022], [.18, .014, -.018], [.11, .019, -.025],
    ])


def test_legacy_anonymous_tie_keeps_the_first_eligible_feature_on_the_left():
    features = dict(
        eligible=np.array([False, True, True]),
        height=np.array([[[-.2, .004, .004], [-.3, .004, .02]]]),
        local_point=np.array([[
            [[.03, -.02, 0.], [.12, -.03, -.01], [.18, -.02, -.03]],
            [[.04, .02, 0.], [.14, .03, -.02], [.19, .02, -.03]],
        ]]),
    )
    selected = programme_opening.standing_contact_rows(features, np.array([0]), 'anonymous_minimum')
    np.testing.assert_array_equal(selected['rows'], [0])
    np.testing.assert_array_equal(selected['feet'], [0])
    np.testing.assert_array_equal(selected['features'], [1])
    np.testing.assert_array_equal(selected['points'], [[.12, -.03, -.01]])
