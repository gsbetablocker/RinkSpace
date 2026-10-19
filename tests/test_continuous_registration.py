"""Meaningful safeguards for observed-keyframe recovery and fitting isolation."""
import cv2
import numpy as np

from rinkspace.continuous_registration import ContinuousRegistrar
from rinkspace.geometry import project


def image(seed=4):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 8, (360, 640)).astype(np.float32)
    raw = np.clip(205 + cv2.GaussianBlur(noise, (3, 3), .6), 0, 255).astype(np.uint8)
    return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)


def sample(i):
    return dict(sample_index=i, source_frame_index=i, pts=i, time_base="1/50", source_time_s=i/50)


def masks(usable):
    return dict(usable=usable, mask_status="synthetic", ice_fraction=1., usable_fraction=float(np.mean(usable>0)))


def test_missing_frame_relocalizes_by_fresh_correspondences_to_same_reference():
    first = image()
    valid = masks(np.full(first.shape[:2],255,np.uint8))
    tracker = ContinuousRegistrar()
    a = tracker.push(first,sample(0),valid)
    missing = tracker.push(np.zeros_like(first),sample(1),masks(np.zeros(first.shape[:2],np.uint8)))
    assert missing["status"] == "unavailable" and missing["H_reference_from_current"] is None
    moved = cv2.warpAffine(first,np.float32([[1,0,5],[0,1,3]]),(640,360))
    b = tracker.push(moved,sample(2),valid)
    assert b["status"] == "candidate" and not b.get("reference_frame_only")
    assert b["graph_component_id"] == a["graph_component_id"]
    assert b["recovered_after_missing"] and b["edge_keyframe_sample_index"] == 0
    assert np.max(np.linalg.norm(project(b["H_reference_from_current"],[[105,103],[405,203]])-[[100,100],[400,200]],axis=1)) < .2


def test_protected_image_values_have_no_effect_on_estimated_map():
    first=image(); usable=np.full(first.shape[:2],255,np.uint8); usable[130:230,260:360]=0
    changed=first.copy(); changed[130:230,260:360]=np.random.default_rng(51).integers(0,255,(100,100,3),dtype=np.uint8)
    maps=[]
    for initial in (first,changed):
        tracker=ContinuousRegistrar()
        tracker.push(initial,sample(0),masks(usable))
        row=tracker.push(first,sample(1),masks(usable))
        assert row["status"] == "candidate"
        assert row["dense_optimization_used"] is False
        maps.append(row["H_reference_from_current"])
    np.testing.assert_array_equal(maps[0],maps[1])


def test_declared_cut_cannot_connect_unrelated_references():
    tracker=ContinuousRegistrar()
    usable=masks(np.full((360,640),255,np.uint8))
    a=tracker.push(image(2),sample(0),usable)
    b=tracker.push(image(6),sample(1),usable,force_break=True)
    assert b["reference_frame_only"]
    assert b["graph_component_id"] != a["graph_component_id"]


def test_long_unobservable_gap_does_not_create_identity_successes():
    tracker=ContinuousRegistrar(); raw=image(); usable=masks(np.full(raw.shape[:2],255,np.uint8))
    tracker.push(raw,sample(0),usable)
    for i in range(1,36):
        row=tracker.push(np.zeros_like(raw),sample(i),masks(np.zeros(raw.shape[:2],np.uint8)))
        assert row["H_reference_from_current"] is None
    row=tracker.push(raw,sample(36),usable)
    assert row["status"] == "unavailable" and row["H_reference_from_current"] is None
    assert row["reason"] == "recovery_gap_exceeded_requires_explicit_new_component"
