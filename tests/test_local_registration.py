import cv2
import json
import numpy as np
import pytest

from rinkspace.geometry import project
from rinkspace.local_registration import LocalRegistrar, ice_mask


def sample(n, pts=None):
    return {"sample_index": n, "source_frame_index": n * 5,
            "pts": n * 10 if pts is None else pts, "time_base": "1/100",
            "source_time_s": n / 10}


def textured_ice(seed=7):
    rng = np.random.default_rng(seed)
    image = np.full((360, 640, 3), 211, np.uint8)
    for xy in rng.integers([20, 30], [620, 340], (1600, 2)):
        shade = int(rng.integers(116, 205))
        cv2.circle(image, tuple(xy), 1, (shade, shade, shade), -1)
    return cv2.GaussianBlur(image, (3, 3), .6)


def test_registration_retains_fixed_reference_and_excludes_independent_region():
    image = textured_ice()
    protected = [{"polygon": [[250, 90], [370, 90], [370, 230], [250, 230]], "landmark_id": "whole-logo"}]
    tracker = LocalRegistrar()
    first = tracker.push(image, sample(0), ice_mask(image, protected_regions=protected))
    assert first["status"] == "candidate"
    assert first["geometry_level"] == "local_projective"
    for n in range(1, 8):
        dx, dy = n * 3, n * 2
        shifted = cv2.warpAffine(image, np.array([[1, 0, dx], [0, 1, dy]], np.float32), (640, 360), borderValue=(211, 211, 211))
        region = [{"polygon": (np.asarray(protected[0]["polygon"]) + [dx, dy]).tolist()}]
        masks = ice_mask(shifted, protected_regions=region)
        row = tracker.push(shifted, sample(n), masks)
        assert row["status"] == "candidate", row
        assert row["graph_component_id"] == first["graph_component_id"]
        returned = project(row["H_reference_from_current"], np.array([[100, 100], [540, 240]]) + [dx, dy])
        assert np.max(np.linalg.norm(returned - [[100, 100], [540, 240]], axis=1)) < .6
        for x, y in row["fit_current_xy"]:
            assert masks["protected"][round(y), round(x)] == 0
        if tracker.last_dense_fit_mask is not None:
            assert not np.any(tracker.last_dense_fit_mask[masks["protected"] > 0])
            original_group = np.zeros((360, 640), np.uint8)
            cv2.fillPoly(original_group, [np.asarray(protected[0]["polygon"], np.int32)], 255)
            projected_group = cv2.warpPerspective(original_group,
                np.linalg.inv(np.asarray(row["H_reference_from_current"])), (640, 360))
            assert not np.any(tracker.last_dense_fit_mask[projected_group > 0])
        assert row["independent_validation"] is False


def test_cut_does_not_reuse_previous_map_and_next_ice_starts_new_component():
    image = textured_ice()
    tracker = LocalRegistrar()
    first = tracker.push(image, sample(0), ice_mask(image))
    dark = np.zeros_like(image)
    failed = tracker.push(dark, sample(1), ice_mask(dark))
    assert failed["status"] == "unavailable"
    assert failed["H_reference_from_current"] is None
    assert failed["graph_component_id"] is None
    recovered = tracker.push(image, sample(2), ice_mask(image))
    assert recovered["status"] == "candidate"
    assert recovered["graph_component_id"] != first["graph_component_id"]


def test_low_texture_cannot_pass_as_identity_success():
    image = np.full((360, 640, 3), 220, np.uint8)
    row = LocalRegistrar().push(image, sample(0), ice_mask(image))
    assert row["status"] == "unavailable"
    assert row["H_reference_from_current"] is None


def test_pts_time_disagreement_rejected():
    image = textured_ice()
    with pytest.raises(ValueError, match="PTS"):
        LocalRegistrar().push(image, sample(0, pts=22), ice_mask(image))


def test_protected_landmark_cannot_be_reintroduced_by_mask_correction():
    image = textured_ice()
    protected = [{"xy": [300, 180], "radius_px": 40}]
    masks = ice_mask(image, {"include_polygons": [[[0, 0], [639, 0], [639, 359], [0, 359]]]}, protected)
    assert not np.any(masks["usable"][masks["protected"] > 0])


def test_observed_yellow_boundary_and_foreground_are_dynamic():
    image = textured_ice()
    image[:70] = (230, 230, 230)
    image[70:78] = (20, 180, 230)
    cv2.rectangle(image, (290, 110), (315, 280), (55, 75, 140), -1)
    masks = ice_mask(image)
    assert masks["boundary_detected"]
    assert not np.any(masks["usable"][:78])
    assert not np.any(masks["usable"][110:280, 290:315])
    assert masks["usable"][310, 450]


def test_sequence_canvas_is_fixed_and_preserves_source_pts_and_unobserved_alpha(tmp_path):
    from rinkspace.local_registration import register_sequence
    source = tmp_path / "samples"
    source.mkdir()
    image = textured_ice()
    rows = []
    for n in range(3):
        translated = cv2.warpAffine(image, np.array([[1, 0, 7*n], [0, 1, 2*n]], np.float32),
                                   (640, 360), borderValue=(211, 211, 211))
        name = f"frame_{n}.jpg"
        cv2.imwrite(str(source / name), translated)
        rows.append({**sample(n), "image": name})
    (source / "samples.json").write_text(json.dumps({"source_sha256": "test-source-hash", "samples": rows}))
    output = tmp_path / "registered"
    result = register_sequence(source, output)
    assert result["summary"]["candidate_count"] == 3
    assert result["summary"]["component_count"] == 1
    assert result["source_sha256"] == "test-source-hash"
    pictures = [cv2.imread(str(output / row["fixed_image"]), cv2.IMREAD_UNCHANGED) for row in result["frames"]]
    assert len({picture.shape for picture in pictures}) == 1
    assert all(picture.shape[2] == 4 and np.any(picture[:, :, 3] == 0) for picture in pictures)
    assert [row["pts"] for row in result["frames"]] == [0, 10, 20]
    assert [row["source_frame_index"] for row in result["frames"]] == [0, 5, 10]
    assert not result["independent_validation"]
