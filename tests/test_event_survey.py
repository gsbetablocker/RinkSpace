from fractions import Fraction
import numpy as np
import pytest
from rinkspace.event_survey import coverage_intervals, cut_distance, visual_features, covers_interval


def test_coverage_preserves_missing_frame_gap():
    intervals = coverage_intervals(np.array([0, 1, 2, 5, 6]), Fraction(1, 25), .04)
    assert np.allclose(intervals, [[0, .12], [.2, .28]])


def test_coverage_rejects_duplicate_pts():
    with pytest.raises(ValueError):
        coverage_intervals(np.array([0, 1, 1]), Fraction(1, 25), .04)


def test_resume_complete_requires_start_and_no_internal_gap():
    assert covers_interval([[0, 600]], 0, 600)
    assert not covers_interval([[.04, 600]], 0, 600)
    assert not covers_interval([[0, 100], [100.04, 600]], 0, 600)
    assert not covers_interval([[0, 599.96]], 0, 600)


def test_cut_score_responds_to_single_frame_change():
    dark = np.zeros((46, 80), np.uint8)
    light = np.full_like(dark, 255)
    assert cut_distance(dark, light) == 1
    assert cut_distance(dark, dark) == 0


def test_rank_features_never_claim_segmentation():
    row = visual_features(np.full((240, 426, 3), 230, np.uint8))
    assert row['ice_candidate_fraction'] == 1
    assert 'ice_mask' not in row
