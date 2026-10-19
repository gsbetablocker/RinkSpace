import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.program_tracking.first_shot_revision import SOURCE_SCHEMA, revise_frames


def save(path, data):
    path.write_text(json.dumps(data), encoding="utf8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def source(tmp_path):
    target = {"kind": "ad_print", "advertisement_id": 5, "contour_key": "component022"}
    row = dict(target=target, _target_key=json.dumps(target, sort_keys=True, separators=(",", ":")),
               uv=np.array([[100., 200.], [101., 210.], [150., 220.], [152., 230.]]),
               sigma=np.array([1., 2., 3., 4.]), source_point_ids=["p0", "p1", "p2", "p3"],
               original_vertex_indices=[0, 1, 2, 3], point_correspondence=False)
    matrix = np.array([[1., 0., .25, 0.], [0., 1., 0., .25]])
    operator = SimpleNamespace(matrix=matrix)
    frame = dict(index=1, native_pts=256256, rows=[row], source_point_count=4,
                 whitening_groups=[dict(point_indices=[0, 1], operator=operator)],
                 source_packet={"path": "original-packet"})
    binding = dict(source={"sha256": "video", "width": 1920, "height": 1080},
                   source_clock={"time_base": "1/12800"}, complete_fit_input={"path": "base"},
                   model={"path": "model"})
    base = save(tmp_path / "base.json", binding)
    evidence = save(tmp_path / "evidence.json", {"native_pts": 256256, "actual_RGB_review": True})
    revision = dict(schema=SOURCE_SCHEMA, source_video_sha256="video", source_time_base="1/12800",
                    base_binding=base, original_source_UV_IDs_and_approvals_modified=False)
    def item(index):
        return dict(native_pts=256256, source_point_id=f"p{index}", original_UV=row["uv"][index].tolist(),
                    original_target=target.copy(), authority="agent_rgb_review", reason="Verified source letter",
                    evidence_references=[evidence])
    def run():
        return revise_frames([frame], save(tmp_path / "revision.json", revision), binding)
    return SimpleNamespace(frame=frame, row=row, revision=revision, item=item, run=run, evidence=evidence)


def test_identity_changes_keep_exact_uv_sigma_order_and_group_operator(source):
    change = source.item(2)
    change["target"] = dict(source.row["target"], contour_key="component018")
    source.revision["identity_corrections"] = [change]
    frames, targets, proof = source.run()
    rows = frames[0]["rows"]
    assert np.array_equal(np.concatenate([r["uv"] for r in rows]), source.row["uv"])
    assert np.array_equal(np.concatenate([r["sigma"] for r in rows]), source.row["sigma"])
    assert [r["source_point_ids"][0] for r in rows] == source.row["source_point_ids"]
    assert rows[2]["target"]["contour_key"] == "component018"
    assert source.row["target"]["contour_key"] == "component022"
    assert frames[0]["whitening_groups"][0]["operator"] is source.frame["whitening_groups"][0]["operator"]
    assert proof["identity_corrections"] == 1
    assert len(targets) == 2


def test_wrong_pixel_and_invented_user_confirmation_are_rejected(source):
    change = source.item(2)
    change["target"] = dict(source.row["target"], contour_key="component018")
    source.revision["identity_corrections"] = [change]
    change["original_UV"][0] += 1
    with pytest.raises(ValueError, match="source UV"):
        source.run()
    change["original_UV"] = source.row["uv"][2].tolist()
    change["identity_confirmed_by_user"] = True
    with pytest.raises(ValueError, match="invent a user"):
        source.run()


def test_correlated_group_cannot_be_silently_partly_removed(source):
    source.revision["quarantined_correspondences"] = [source.item(0)]
    with pytest.raises(ValueError, match="Partial correlated-group"):
        source.run()
    source.revision["quarantined_correspondences"].append(source.item(1))
    frames, _, proof = source.run()
    assert frames[0]["source_point_count"] == 2
    assert not frames[0]["whitening_groups"]
    assert len(frames[0]["inactive_source_correspondences"]) == 2
    assert source.frame["source_point_count"] == 4
    assert proof["quarantined_correspondences"] == 2


def test_new_native_readout_preserves_originals_and_requires_direct_evidence(source):
    added = dict(native_pts=256256, source_point_id="native/left/0", source_feature_id="Ad4.left_vertical",
                 UV=[300., 400.], sigma_px=2., target={"kind": "ad_edge", "advertisement_id": 4,
                 "role": "left_vertical"}, authority="agent_rgb_review", reason="Visible material transition",
                 evidence_references=[source.evidence], measurement_kind="direct_material_transition",
                 derived_from_shared_fit=False)
    source.revision["added_native_readouts"] = [added]
    frames, _, proof = source.run()
    assert frames[0]["source_point_count"] == 5
    assert np.array_equal(np.concatenate([r["uv"] for r in frames[0]["rows"]])[:4], source.row["uv"])
    assert proof["added_native_readouts"] == 1
    added["derived_from_shared_fit"] = True
    with pytest.raises(ValueError, match="independent native"):
        source.run()


def test_partial_group_requires_matching_fresh_source_operator(source):
    from calibration.program_tracking.source_evidence_whitening import SourceWhiteningOperator
    old = SourceWhiteningOperator(np.array([[1., 0., .25, 0.], [0., 1., 0., .25]]),
                                  ("p0", "p1"), "native/edge", "original native profiles")
    new = SourceWhiteningOperator(np.eye(2) * .75, ("p1",), "native/edge", "rebuilt native profiles")
    source.frame["whitening_groups"][0]["operator"] = old
    source.revision["quarantined_correspondences"] = [source.item(0)]
    replacement = dict(native_pts=256256, group_id=old.group_id, original_point_indices=[0, 1],
        original_operator=old.to_record(), new_operator=new.to_record(), authority="agent_rgb_review",
        reason="Reconstruct surviving native profile uncertainty", evidence_references=[source.evidence],
        raw_whitening_row_or_column_slice_used=False)
    source.revision["correlated_operator_replacements"] = [replacement]
    frames, _, proof = source.run()
    group = frames[0]["whitening_groups"][0]
    assert group["point_indices"] == [0]
    np.testing.assert_array_equal(group["operator"].matrix, new.matrix)
    assert proof["rebuilt_correlated_source_groups"] == 1
    replacement["original_point_indices"] = [1, 0]
    with pytest.raises(ValueError, match="exact original"):
        source.run()
