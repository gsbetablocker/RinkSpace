from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.program_tracking import first_shot_ad7_source_paths as paths
from calibration.program_tracking import first_shot_disjoint_print as disjoint
from calibration.program_tracking.first_shot_fit import FrozenWhitening, _target_key


def fixture(monkeypatch):
    material_ref = dict(path="explicit-material", sha256="material-sha")
    target = dict(kind="ad_print_curve", advertisement_id=7, contour_key="component000",
        curve_topology=dict(schema=disjoint.TOPOLOGY, path_ids=list(paths.SOURCE_PATHS),
            offsets=[0, 2, 4, 6, 8], closed=[True]*4, material_reference=material_ref))
    material = dict(contours={"logo_part03":dict(closed=True,
        material_uv=[[0., 0.], [1., 0.], [0., 0.]])})
    monkeypatch.setattr(disjoint, "_CONFIG", dict(manifest=dict(material_override=material_ref),
        templates=dict(advertisements={"7":material})))
    source_id = "Ad7.print.component000/segment/2/vertex/0"
    row = dict(target=target, _target_key=_target_key(target), source_point_ids=[source_id],
        uv=np.array([[300., 596.]]), sigma=np.array([2.]), point_correspondence=False)
    operator = SimpleNamespace(matrix=np.array([[.2, -.4], [.7, .3]]))
    frame = dict(index=1, native_pts=256256, rows=[row], source_point_count=1,
        whitening_groups=[dict(point_indices=[0], operator=operator)])
    record = dict(frame=1, native_pts=256256, source_point_id=source_id,
        source_uv=[300., 596.], true_path_id="logo_part03",
        source_physical_identity="ISU_logo_S_lower_detached_ink_boundary")
    return frame, {(1, source_id):record}


def test_only_true_lower_S_path_can_be_selected(monkeypatch):
    frame, lookup = fixture(monkeypatch)
    before = FrozenWhitening(frame).apply(np.array([3., -5.]))
    output, proof = paths.transition_frames([frame], lookup=lookup,
        mapping_reference=dict(path="mapping", sha256="mapping-sha"))
    old, new = frame["rows"][0], output[0]["rows"][0]
    assert new["target"]["curve_topology"]["path_ids"] == ["logo_part03"]
    assert "logo_part00" not in new["target"]["curve_topology"]["path_ids"]
    assert new["target"]["curve_topology"]["offsets"] == [0, 3]
    assert old["target"]["curve_topology"]["path_ids"] == list(paths.SOURCE_PATHS)
    assert new["uv"] is old["uv"]
    assert new["sigma"] is old["sigma"]
    assert new["source_point_ids"] is old["source_point_ids"]
    assert output[0]["whitening_groups"] is frame["whitening_groups"]
    assert np.array_equal(FrozenWhitening(output[0]).apply(np.array([3., -5.])), before)
    assert proof["source_contract_before"] == proof["source_contract_after"]
    assert new["source_true_path_history"][0]["previous_target"] == old["target"]


def test_wrong_UV_is_rejected(monkeypatch):
    frame, lookup = fixture(monkeypatch)
    next(iter(lookup.values()))["source_uv"][0] += .001
    with pytest.raises(ValueError, match="UV or native clock"):
        paths.transition_frames([frame], lookup=lookup, mapping_reference={})


def test_missing_current_source_ID_is_rejected(monkeypatch):
    frame, lookup = fixture(monkeypatch)
    key = next(iter(lookup))
    lookup[(1, "Ad7.print.component000/segment/99/vertex/0")] = lookup.pop(key)
    with pytest.raises(ValueError, match="absent from the current source"):
        paths.transition_frames([frame], lookup=lookup, mapping_reference={})


def test_unreviewed_rows_retain_original_alias(monkeypatch):
    frame, lookup = fixture(monkeypatch)
    row = dict(frame["rows"][0], source_point_ids=["unreviewed"], uv=np.array([[50., 60.]]))
    frame["rows"].append(row)
    frame["source_point_count"] = 2
    output, _ = paths.transition_frames([frame], lookup=lookup, mapping_reference={})
    assert output[0]["rows"][1] is row


def test_cannot_expand_alias_to_whole_logo(monkeypatch):
    frame, _ = fixture(monkeypatch)
    target = frame["rows"][0]["target"]
    with pytest.raises(ValueError, match="reviewed original"):
        paths.reviewed_target(target, "logo_part00")
    changed = deepcopy(target)
    changed["contour_key"] = "component001"
    with pytest.raises(ValueError, match="reviewed original"):
        paths.reviewed_target(changed, "logo_part03")
