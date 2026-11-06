"""Synthetic interface checks only: no FS/AP experiment or accuracy claim."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json

import numpy as np
import pytest

from calibration.program_tracking.body_tracking.fs_jump3d_trial import (
    SCHEMA, TrialContractError, load_trial, validate_trial, validate_trial_splits,
)


def synthetic_trial():
    """All hashes/IDs below are synthetic, not third-party provenance."""
    names = [f"original_point_{i}" for i in range(18)] + ["LHeel", "LToe", "LForefoot", "LHeel"]
    sources = {k: {"path": f"synthetic_{k}.bin", "sha256": hashlib.sha256(k.encode()).hexdigest(), "kind": kind}
               for k, kind in [("world", "world_points"), ("identity", "identity"), ("licence", "licence"),
                               ("cal_a", "camera"), ("cal_b", "camera"), ("video_a", "video"),
                               ("video_b", "video"), ("sync_a", "sync"), ("sync_b", "sync")]}
    cameras = []
    for index, cid in enumerate(("a", "b")):
        camera = dict(camera_id=cid, role="input" if index == 0 else "holdout",
                      world_frame_id="original_world", metric_unit="m", R_w2c=np.eye(3).tolist(),
                      K=[[1000, 0, 320], [0, 1000, 240], [0, 0, 1]],
                      distortion={"model": "none", "coefficients": []},
                      pixel_transform=np.eye(3).tolist(), image_size=[640, 480],
                      calibration_source_key=f"cal_{cid}",
                      video_sync={"mapping": "explicit_pts", "evidence_kind": "source_metadata",
                                  "source_pts": [900, 921, 943], "time_base": [1, 1000],
                                  "uncertainty_seconds": 0.001, "video_source_key": f"video_{cid}",
                                  "sync_source_key": f"sync_{cid}"})
        camera.update({"t_w2c": [0, 0, 0]} if index == 0 else {"C_world": [1, 0, 0]})
        cameras.append(camera)
    return dict(schema_version=SCHEMA, dataset_id="synthetic_FS", trial_id="synthetic_trial",
                physical_trial_id="synthetic_physical_trial", subject_id="synthetic_subject",
                trial_aliases=[{"dataset_id": "synthetic_FS", "trial_id": "synthetic_trial"}],
                split="test", complete_trial=True, source_frame_range=[100, 102],
                source_frame=[100, 101, 102], source_pts=[1000, 1021, 1043], time_base=[1, 1000],
                world_points=np.broadcast_to([1000., 2000., 10000.], (3, len(names), 3)).tolist(),
                names=names, valid_mask=np.ones((3, len(names)), dtype=bool).tolist(), metric_unit="mm",
                world_frame_id="original_world", point_inventory={"selection": "all_source_points", "original_names": names.copy()},
                world_transform={"method": "declared_once", "declaration_id": "synthetic_fixed_axes",
                                 "source_world_frame": "original_world", "evaluation_world_frame": "evaluation_world",
                                 "R_source_to_evaluation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]], "translation_m": [3, 4, 5]},
                cameras=cameras, input_camera_id="a", source_hashes=sources,
                world_source_key="world", identity_source_key="identity",
                licence_origin={"dataset": "synthetic_FS", "licence": "synthetic test fixture only",
                                "source_uri": "synthetic:test-fixture", "terms_source_key": "licence"})


def test_original_foot_names_invalid_slots_world_units_and_exact_clock_survive():
    payload = synthetic_trial()
    payload["valid_mask"][1][20] = False
    payload["world_points"][1][20] = [float("nan")] * 3
    trial = validate_trial(payload)
    assert trial.names == tuple(payload["names"]) and len(trial.names) == 22
    assert trial.names[-4:] == ("LHeel", "LToe", "LForefoot", "LHeel")
    np.testing.assert_equal(trial.world_points, payload["world_points"])
    assert trial.time_base == Fraction(1, 1000)
    assert trial.metadata["time_base"] == (1, 1000)
    np.testing.assert_array_equal(trial.source_pts, [1000, 1021, 1043])
    np.testing.assert_allclose(trial.evaluation_points_m()[0, 0], [1, 5, 15])
    payload["world_transform"]["translation_m"][0] = 99
    payload["world_points"][0][0][0] = 99
    np.testing.assert_allclose(trial.evaluation_points_m()[0, 0], [1, 5, 15])
    with pytest.raises(ValueError):
        trial.world_points[0, 0, 0] = 0
    with pytest.raises(TypeError):
        trial.metadata["world_transform"]["method"] = "fit"


def test_single_input_view_excludes_3d_and_holdout_sources():
    trial = validate_trial(synthetic_trial())
    view = trial.input_view()
    assert not hasattr(view, "world_points") and not hasattr(view, "metadata")
    assert view.camera["camera_id"] == "a"
    assert set(view.sources) == {"cal_a", "video_a", "sync_a"}
    assert trial.holdout_camera_ids() == ("b",)


def test_reference_projection_respects_camera_units_center_and_native_pixel_domain():
    p = synthetic_trial()
    p["cameras"][1]["pixel_transform"] = [[.5, 0, -10], [0, .5, 7], [0, 0, 1]]
    trial = validate_trial(p)
    uv, mask = trial.project_reference("a")
    np.testing.assert_allclose(uv[0, 0], [420, 440])
    uv, mask = trial.project_reference("b")
    np.testing.assert_allclose(uv[0, 0], [150, 227])
    assert mask.all()
    p["world_points"][0][0][2] = -1
    uv, mask = validate_trial(p).project_reference("a")
    assert not mask[0, 0] and np.isnan(uv[0, 0]).all()


def test_declared_brown_radial_distortion():
    p = synthetic_trial()
    p["cameras"][0]["distortion"] = {"model": "opencv_brown_conrady", "coefficients": [.5, 0, 0, 0]}
    uv, _ = validate_trial(p).project_reference("a")
    # x=.1, y=.2, r^2=.05: radial magnification 1.025.
    np.testing.assert_allclose(uv[0, 0], [422.5, 445])


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p.pop("time_base"), "missing time_base"),
    (lambda p: p.update(time_base=[1, 0]), "positive"),
    (lambda p: p.update(source_pts=[1000., 1021., 1043.]), "integer"),
    (lambda p: p.update(source_pts=[1000, 1000, 1043]), "strictly increase"),
    (lambda p: p.update(source_frame=[100, 102, 103]), "every original frame"),
    (lambda p: p["names"].__setitem__(-1, "blade_tail"), "original_names"),
    (lambda p: p["world_points"][0][0].__setitem__(0, float("nan")), "finite"),
    (lambda p: p["valid_mask"][0].__setitem__(0, 0), "boolean"),
    (lambda p: p["world_transform"].update(method="per_frame_procrustes"), "declared_once"),
    (lambda p: p["world_transform"].update(R_source_to_evaluation=[np.eye(3).tolist()] * 3), "shape"),
    (lambda p: p["cameras"][1].update(role="input"), "exactly one"),
    (lambda p: p["cameras"][0].update(C_world=[0, 0, 0]), "exactly one of"),
    (lambda p: p["cameras"][1].update(world_frame_id="other"), "world frame"),
    (lambda p: p["cameras"][0].update(R_w2c=(np.eye(3)*2).tolist()), "orthonormal"),
    (lambda p: p["cameras"][0].update(pixel_transform=np.zeros((3, 3)).tolist()), "invertible"),
    (lambda p: p["cameras"][0]["distortion"].update(model="unknown"), "distortion"),
    (lambda p: p["cameras"][1]["video_sync"].update(source_pts=[1, 2]), "shape"),
    (lambda p: p["cameras"][1]["video_sync"].update(mapping="fps"), "explicit_pts"),
    (lambda p: p["source_hashes"].pop("sync_b"), "source reference"),
    (lambda p: p["licence_origin"].pop("source_uri"), "source_uri"),
])
def test_incomplete_or_ambiguous_input_is_rejected(mutate, message):
    p = synthetic_trial()
    mutate(p)
    with pytest.raises(TrialContractError, match=message):
        validate_trial(p)


def test_full_trial_split_registry_catches_physical_alias_and_raw_reuse():
    original = synthetic_trial()
    train = deepcopy(original)
    train["split"] = "train"
    with pytest.raises(TrialContractError, match="split leakage"):
        validate_trial_splits([validate_trial(original), validate_trial(train)])
    train["physical_trial_id"] = "different_declared_id"
    train["dataset_id"], train["trial_id"] = "synthetic_AP", "synthetic_copy"
    train["trial_aliases"].append({"dataset_id": train["dataset_id"], "trial_id": train["trial_id"]})
    with pytest.raises(TrialContractError, match="alias"):
        validate_trial_splits([validate_trial(original), validate_trial(train)])
    train["trial_aliases"] = train["trial_aliases"][-1:]
    with pytest.raises(TrialContractError, match="raw"):
        validate_trial_splits([validate_trial(original), validate_trial(train)])
    train["split"] = "test"
    validate_trial_splits([validate_trial(original), validate_trial(train)])


def test_json_npz_hash_and_original_source_verification(tmp_path):
    p = synthetic_trial()
    archive = tmp_path / "arrays.npz"
    keys = ("world_points", "source_frame", "source_pts", "valid_mask", "names")
    arrays = {k: np.asarray(p[k]) for k in keys}
    np.savez(archive, **arrays)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    for key in keys:
        p[key] = {"npz": "arrays.npz", "key": key, "sha256": digest}
    # Only synthetic originals, physically present; never data from the dataset.
    for source in p["source_hashes"].values():
        target = tmp_path / source["path"]
        target.write_bytes(source["kind"].encode())
        source["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = tmp_path / "trial.json"
    manifest.write_text(json.dumps(p), encoding="utf8")
    trial = load_trial(manifest, verify_sources=True)
    np.testing.assert_array_equal(trial.world_points, arrays["world_points"])
    np.testing.assert_array_equal(trial.source_pts, arrays["source_pts"])
    (tmp_path / p["source_hashes"]["video_b"]["path"]).write_bytes(b"changed")
    with pytest.raises(TrialContractError, match="original source SHA256"):
        load_trial(manifest, verify_sources=True)
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(TrialContractError, match="NPZ SHA256"):
        load_trial(manifest)


def test_explicit_repeated_video_exposures_are_preserved_without_inventing_samples():
    p = synthetic_trial()
    p["cameras"][0]["video_sync"]["source_pts"] = [900, 900, 943]
    trial = validate_trial(p)
    assert trial.input_view().camera["video_sync"]["source_pts"] == (900, 900, 943)
    p["cameras"][0]["video_sync"]["source_pts"] = [900, 899, 943]
    with pytest.raises(TrialContractError, match="must not decrease"):
        validate_trial(p)
