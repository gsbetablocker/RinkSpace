"""Localization noise changes precision, never means or source information rank."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from calibration.program_tracking.first_shot_noise_revision import (
    SCHEMA, make_noise_replacement, revise_localization_noise, fixed_mode_transform,
)
from calibration.program_tracking.source_evidence_whitening import latent_whitening_operator


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf8")
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def source(tmp_path):
    # Four native midpoint measurements generate three correlated line
    # representatives. Common-mode readout noise must survive propagation.
    native_station = np.array([-2., -1., 1., 2.])
    design = np.c_[native_station, np.ones(4)]
    differential = np.linalg.pinv(design)
    sensitivity = np.c_[[-1.5, 0., 1.5], np.ones(3)]
    basis = np.zeros((3, 6)); basis[:, 1::2] = np.eye(3)
    support = np.diag([10., 12., 14., 16.])
    noise = np.c_[.4*np.eye(4), .3*np.ones(4)]
    ids = ("a", "b", "c")
    old = latent_whitening_operator(ids, sensitivity,
        explicit_noise_factor=differential @ support, physical_basis=basis,
        group_id="256256/ice.C/source_line", covariance_origin="old profile support")
    rows = [dict(source_point_ids=[pid], uv=np.array([[u, 500.]]), sigma=np.array([12.]),
        target=dict(kind="ice_curve", curve_id="C"), source_feature_id="ice.C",
        original_vertex_indices=[i]) for i, (pid, u) in enumerate(zip(ids, [10., 20., 30.]))]
    group = dict(point_indices=[0, 1, 2], operator=old)
    frame = dict(native_pts=256256, rows=rows, whitening_groups=[group])
    evidence = save(tmp_path/"image_noise.json", dict(method="native patch noise fixture"))
    item = make_noise_replacement(frame, group, physical_basis=basis,
        estimator_output_sensitivity=sensitivity, coefficient_influence=differential,
        original_support_factor=support, localization_noise_factor=noise,
        evidence_references=[evidence], noise_method="correlated native midpoint noise")
    binding = dict(source=dict(sha256="video"), source_clock=dict(time_base="1/12800"),
        complete_fit_input=dict(path="source"), model=dict(path="world"))
    revision = dict(schema=SCHEMA, base_binding=save(tmp_path/"base.json", binding),
        source_video_sha256="video", source_time_base="1/12800",
        source_UV_IDs_targets_and_estimator_D_modified=False, groups=[item])
    def run():
        return revise_localization_noise([frame], save(tmp_path/"revision.json", revision), binding)
    return dict(frame=frame, group=group, item=item, revision=revision, run=run,
        D=differential, S=sensitivity, B=basis, noise=noise, path=tmp_path)


def test_correlated_midpoint_noise_propagates_through_same_estimator(source):
    frames, proof = source["run"]()
    new = frames[0]["whitening_groups"][0]["operator"]
    L = source["B"].T @ source["S"] @ source["D"] @ source["noise"]
    np.testing.assert_allclose(new.matrix @ L @ L.T @ new.matrix.T, np.eye(2), atol=1e-12)
    # Mean-coordinate variance retains .3^2 common error; it is not divided
    # by the number of native samples or by representative count.
    covariance = source["D"] @ source["noise"] @ source["noise"].T @ source["D"].T
    assert covariance[1, 1] == pytest.approx(.4**2/4 + .3**2)
    assert frames[0]["rows"] is source["frame"]["rows"]
    assert new.rank == source["group"]["operator"].rank == 2
    assert proof["source_UV_IDs_targets_and_estimator_D_modified"] is False


def test_existing_raw_jacobian_can_be_rewhitened_exactly(source):
    frames, _ = source["run"]()
    old = source["group"]["operator"]
    new = frames[0]["whitening_groups"][0]["operator"]
    transform, error = fixed_mode_transform(old, new)
    rng = np.random.default_rng(190)
    raw_jacobian = rng.normal(size=(6, 11)); raw_residual = rng.normal(size=6)
    np.testing.assert_allclose(transform @ (old.matrix @ raw_jacobian), new.matrix @ raw_jacobian, atol=1e-12)
    np.testing.assert_allclose(transform @ (old.matrix @ raw_residual), new.matrix @ raw_residual, atol=1e-12)
    assert error < 1e-12


@pytest.mark.parametrize("changed", ["UV", "target", "source_id", "support"])
def test_new_noise_cannot_hide_changed_source_facts(source, changed):
    row = source["frame"]["rows"][0]
    if changed == "UV": row["uv"][0, 1] += .01
    if changed == "target": row["target"]["curve_id"] = "B2"
    if changed == "source_id": row["source_point_ids"][0] = "other"
    if changed == "support": row["sigma"][0] += 1.
    with pytest.raises(ValueError, match="means/IDs/targets/operator"):
        source["run"]()


def test_false_old_covariance_and_forced_new_operator_are_rejected(source):
    original = deepcopy(source["item"])
    source["item"]["original_support_factor"][0][0] *= 2.
    with pytest.raises(ValueError, match="original source covariance"):
        source["run"]()
    source["item"].clear(); source["item"].update(original)
    source["item"]["new_operator"]["whitening_matrix"][0][1] *= 2.
    with pytest.raises(ValueError, match="not the measured noise"):
        source["run"]()


def test_source_image_noise_evidence_must_be_immutable(source):
    (source["path"]/"image_noise.json").write_text("{}", encoding="utf8")
    with pytest.raises(ValueError, match="evidence changed"):
        source["run"]()


def test_source_localization_noise_cannot_add_or_drop_residual_modes(source):
    source["item"]["localization_noise_factor"] = (.3*np.ones((4, 1))).tolist()
    with pytest.raises(ValueError, match="residual rank"):
        source["run"]()
