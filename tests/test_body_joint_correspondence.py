"""A source label repair changes correspondence/Jacobian, never source pixels."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from calibration.program_tracking.body_tracking.fullbody_fit import FullBodyProblem,body_joint_assignment
from calibration.program_tracking.body_tracking.actual_observation_ledger import update_actual_observation_ledger
from calibration.program_tracking.body_tracking.body_source_revision import revise_body_observations
from calibration.program_tracking.body_tracking.programme_skate_evidence import programme_observation_path
from calibration.program_tracking.body_tracking.rebuild_prepare import digest
from rinkspace.viewer3d.source import body_overlay_primitives, read_body_overlay


def test_bilateral_correspondence_uses_target_model_columns_and_original_sigma():
    # Construct only the actual pixel factor, independent of model assets/FK.
    p=FullBodyProblem.__new__(FullBodyProblem)
    p.of=np.array([0]);p.oj=np.array([15]);p.model_oj=np.array([16]);p.frames=np.array([0])
    p.layout=SimpleNamespace(body_landmarks=np.arange(17),point_count=17)
    p.obs={'uv':np.zeros((1,17,2)),'sigma_px':np.ones((1,17))*3.}
    uv=np.zeros((1,17,2));uv[0,15]=[100,200];uv[0,16]=[4,-2]
    before=p.obs['uv'].copy();r,A,_,_=p.body_factors(uv)
    assert np.array_equal(p.obs['uv'],before)
    assert set(A.indices)=={32,33}
    d=np.arange(34).reshape(1,17,2)/100.;h=1e-6
    numeric=(p.body_factors(uv+h*d)[0]-p.body_factors(uv-h*d)[0])/(2*h)
    np.testing.assert_allclose(A@d.ravel(),numeric,atol=2e-9)


def test_assignment_cannot_alias_points_change_joint_kind_or_omit_pts_provenance():
    ids=np.arange(17)[None].copy();ids[:,15:17]=[16,15]
    o={'pts':np.array([256]),'model_joint_ids':ids,'body_joint_assignment_provenance_json':np.array(json.dumps(dict(kind='source_observation_correspondence',source_pts=[256],source_measurements_unchanged=True)))}
    np.testing.assert_array_equal(body_joint_assignment(o),ids)
    o['model_joint_ids']=ids.copy();o['model_joint_ids'][0,15]=15
    with pytest.raises(ValueError):body_joint_assignment(o)
    o['model_joint_ids']=ids;o['body_joint_assignment_provenance_json']=np.array('{}')
    with pytest.raises(ValueError):body_joint_assignment(o)


def test_revision_follows_the_selected_ledger_into_scatter_and_export(tmp_path):
    pts = np.arange(256, 5*256, 256, dtype=np.int64)
    uv = np.arange(len(pts)*17*2, dtype=float).reshape(-1, 17, 2)
    parent = dict(pts=pts, uv=uv, sigma_px=np.full((len(pts), 17), 3.), use=np.ones((len(pts), 17), bool))
    parent_path = tmp_path/'observations.npz'
    np.savez(parent_path, **parent)
    legacy = dict(path=str(parent_path), sha256=digest(parent_path), parent_observations_sha256='legacy-parent')
    assert programme_observation_path(tmp_path, dict(programme_body_observation_revision=legacy)) == parent_path
    revised, _ = revise_body_observations(parent,
        [dict(source_pts=int(t), joint_pairs=[[9, 10], [15, 16]]) for t in pts[1:3]],
        [dict(source_pts=int(pts[2]), joint_id=16, reason='duplicated visible boot')], {})
    revised_path = tmp_path/'reviewed.npz'
    np.savez(revised_path, **revised)
    document = dict(programme_body_observation_revision=dict(path=str(revised_path),
        sha256=digest(revised_path), parent_observations_sha256=digest(parent_path)))
    path = programme_observation_path(tmp_path, document)
    assert path == revised_path
    for key in ('uv', 'sigma_px'):
        np.testing.assert_array_equal(revised[key], parent[key])
    body = FullBodyProblem.__new__(FullBodyProblem)
    body.obs, body.frames, body.ledger_path = revised, np.arange(len(pts)), path
    body.of, body.oj = np.nonzero(revised['use'])
    body.model_oj = body_joint_assignment(revised)[body.of, body.oj]
    body.layout = SimpleNamespace(body_landmarks=np.arange(17), point_count=17)
    model_uv = np.take_along_axis(uv, np.argsort(body_joint_assignment(revised), axis=1)[..., None], axis=1)
    residual, jac, _, _ = body.body_factors(model_uv)
    assert np.all(residual == 0)
    sample = np.flatnonzero((body.of == 1) & (body.oj == 9))[0]
    assert set(jac[2*sample:2*sample+2].indices) == {54, 55}
    exported = update_actual_observation_ledger(dict(pts=pts), body)
    np.testing.assert_array_equal(exported['source_observation_objective_use'], revised['use'])
    np.testing.assert_array_equal(exported['source_body_model_joint_ids'][:, 9:11], [[9, 10], [10, 9], [10, 9], [9, 10]])
    np.testing.assert_array_equal(exported['source_uv_native'], uv)
    assert str(exported['source_observation_ledger_sha256']) == digest(revised_path)
    result_path = tmp_path/'result.npz'
    np.savez(result_path, **exported, source_pts=pts, COCO_uv=model_uv)
    overlay = read_body_overlay(result_path, pts)
    items = {p['source_slot']:p for p in body_overlay_primitives(overlay, 2, show_model=False)}
    assert 16 not in items
    assert items[9]['model_joint'] == 10
    np.testing.assert_array_equal(items[9]['points'][0], uv[2,9])
    with pytest.raises(AssertionError):
        read_body_overlay(result_path, pts+1)
    body.model_oj = body.oj
    with pytest.raises(ValueError, match='stale.*mapping'):
        update_actual_observation_ledger(dict(pts=pts), body)
