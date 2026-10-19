"""Observed-keyframe composition and explicit gaps must preserve honest maps."""
import cv2
import numpy as np

from rinkspace.geometry import project
from rinkspace.photometric_registration import PhotometricRegistrar


def image():
    rng = np.random.default_rng(28)
    raw = cv2.GaussianBlur(rng.integers(30, 220, (180, 260), dtype=np.uint8), (5, 5), 1.)
    return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)


def sample(i):
    return dict(sample_index=i, source_frame_index=i, pts=i, time_base="1/50", source_time_s=i/50)


def mask(raw, empty=False):
    usable = np.full(raw.shape[:2], 0 if empty else 255, np.uint8)
    return dict(usable=usable, mask_status="synthetic", ice_fraction=1., usable_fraction=float(not empty))


def test_missing_observation_recovers_to_existing_reference_and_saves_full_audit():
    first = image()
    tracker = PhotometricRegistrar()
    a = tracker.push(first, sample(0), mask(first))
    missing = tracker.push(first, sample(1), mask(first, True))
    assert missing["status"] == "unavailable" and missing["H_reference_from_current"] is None
    moved = cv2.warpAffine(first, np.float32([[1,0,4],[0,1,2]]), (260,180))
    b = tracker.push(moved, sample(2), mask(moved))
    assert b["status"] == "candidate" and b["recovered_after_missing"]
    assert b["graph_component_id"] == a["graph_component_id"]
    assert b["reference_sample_index"] == 0 and not b.get("reference_frame_only")
    np.testing.assert_allclose(project(b["H_reference_from_current"], [[104,82]]), [[100,80]], atol=.12)
    assert len(b["fit_current_xy"]) <= 200 < b["fit_points_full_count"]
    assert b["fit_points_are_subset"] and "not_independent" in b["fit_points_role"]
    chosen = tracker.last_photometric_audit["attempts"][-1]
    assert len(chosen["fit_current_xy"]) == b["fit_points_full_count"]
    np.testing.assert_equal(np.asarray(chosen["fit_current_xy"])[b["fit_points_uniform_indices"]], b["fit_current_xy"])


def test_source_gap_never_silently_starts_an_identity_component():
    raw = image()
    tracker = PhotometricRegistrar()
    first = tracker.push(raw, sample(0), mask(raw))
    gap = tracker.push(raw, sample(40), mask(raw))
    after = tracker.push(raw, sample(41), mask(raw))
    assert gap["status"] == after["status"] == "unavailable"
    assert after["H_reference_from_current"] is None
    declared = tracker.push(raw, sample(42), mask(raw), force_break=True)
    assert declared["reference_frame_only"]
    assert declared["graph_component_id"] != first["graph_component_id"]


def test_fixed_keyframe_edge_is_composed_in_correct_direction():
    raw = image()
    tracker = PhotometricRegistrar()
    tracker.push(raw, sample(0), mask(raw))
    for i in (10,20,21,30,40,41):
        dx = i*.18
        moved = cv2.warpAffine(raw, np.float32([[1,0,dx],[0,1,0]]), (260,180))
        row = tracker.push(moved, sample(i), mask(moved))
        assert row["status"] == "candidate"
    np.testing.assert_allclose(project(row["H_reference_from_current"], [[130+dx,90]]), [[130,90]], atol=.3)
    assert row["reference_sample_index"] == 0
    assert len(tracker.keys) == 3


def test_local_photo_match_does_not_allow_global_horizon_through_fitting_support():
    raw = image()
    tracker = PhotometricRegistrar()
    tracker.push(raw,sample(0),mask(raw))
    # A locally perfect edge cannot repair a corrupt accumulated reference.
    # This mirrors the observed failure of a long unconstrained 8-DOF chain.
    invalid_global = np.array([[1.,0,0],[0,1.,0],[-.008,0,1.]])
    tracker.keys[0]["H_reference_from_keyframe"] = invalid_global
    tracker.keys[0]["points"] = np.empty((0,2),np.float32)
    tracker.last_h = invalid_global.copy()
    row = tracker.push(raw,sample(1),mask(raw))
    assert row["status"] == "unavailable" and row["H_reference_from_current"] is None
    assert row["keyframe_attempts"][0]["reason"] == "composed_reference_horizon_crosses_fitting_support"
    assert tracker.last_photometric_audit["attempts"][0]["status"] == "candidate"
