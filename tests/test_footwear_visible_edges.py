"""Explicit finite boot edges, local pixel derivatives, and source semantics only."""
import json
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from calibration.program_tracking.body_tracking.footwear_geometry import INITIAL,local_vertices
from calibration.program_tracking.body_tracking.footwear_pixel_factors import nearest,nearest_segment,pixel_factors
from calibration.program_tracking.body_tracking.footwear_observations import visible_shell_edges,audit_observation_semantics
from calibration.program_tracking.body_tracking.rebuild_prepare import digest
from rinkspace.viewer3d.open_boot_topology import open_boot_shell,COLLAR_LOOP


def problem(points,frame=0,foot=0):
    return SimpleNamespace(visible_rows=[dict(frame=frame,foot=foot,observed=np.asarray(points),sigma_px=2.,reverse_boundary=False)],
                           blade_rows=[],landmark_rows=[])


def test_disconnected_explicit_edges_do_not_invent_a_connecting_curve():
    vertices=np.array([[0.,0.],[0.,1.],[10.,0.],[10.,1.]])
    distance,which,alpha,unit=nearest_segment([[5.,.5],[-3.,-2.]],vertices,np.array([[0,1],[2,3]]))
    np.testing.assert_allclose(distance,[5.,np.sqrt(13.)])
    np.testing.assert_allclose(alpha,[.5,0.]);assert which.tolist()==[0,0]
    np.testing.assert_allclose(unit[0],[-1,0])


def test_explicit_boot_edge_pixel_jacobian_uses_frame_and_physical_foot_indices(monkeypatch):
    uv=np.zeros((2,113,2));uv[1,48:52]=[[0.,0.],[3.,0.],[5.,0.],[5.,3.]]
    p=problem([[.5,2.],[6.,1.],[8.,5.]],frame=1,foot=1)
    edges={(1,1):dict(edges=np.array([[0,1],[2,3]]))}
    monkeypatch.setattr(cv2,'convexHull',lambda *a,**k:pytest.fail('explicit path called convex hull'))
    residual,jac,parts,_=pixel_factors(uv,p,body=False,boot_edges=edges)
    np.testing.assert_allclose(residual,np.array([2.,1.,np.sqrt(13.)])/(2*np.sqrt(48)))
    rng=np.random.default_rng(2102);direction=rng.normal(size=uv.shape);h=1e-6
    plus=pixel_factors(uv+h*direction,p,body=False,boot_edges=edges)[0]
    minus=pixel_factors(uv-h*direction,p,body=False,boot_edges=edges)[0]
    np.testing.assert_allclose((plus-minus)/(2*h),jac@direction.ravel(),atol=2e-10,rtol=2e-8)
    assert jac.shape==(3,uv.size)
    assert parts['visible_boot_edges']==pytest.approx(float(residual@residual/2))
    assert set(jac.nonzero()[1])<=set(((113+np.arange(48,52))[:,None]*2+np.arange(2)).ravel())


@pytest.mark.parametrize('edges',[{}, {(0,0):np.empty((0,2),dtype=int)}])
def test_missing_explicit_edges_never_fall_back_or_remove_observations(edges,monkeypatch):
    monkeypatch.setattr(cv2,'convexHull',lambda *a,**k:pytest.fail('missing explicit edges called convex hull'))
    with pytest.raises(ValueError,match='no convex-hull fallback'):
        pixel_factors(np.zeros((1,113,2)),problem([[1.,2.]]),body=False,boot_edges=edges)


def test_none_preserves_historical_source_only_hull_and_48_sample_normalization():
    rng=np.random.default_rng(23);uv=rng.normal(size=(1,113,2));p=problem([[.2,1.5],[2.,-.5]])
    ids=cv2.convexHull(uv[0,:14].astype(np.float32),returnPoints=False).ravel()
    expected=nearest(p.visible_rows[0]['observed'],uv[0,ids],closed=True)[0]/(2*np.sqrt(48))
    np.testing.assert_array_equal(pixel_factors(uv,p,body=False)[0],expected)
    explicit=np.c_[ids,np.roll(ids,-1)]
    old=pixel_factors(uv,p,body=False);new=pixel_factors(uv,p,body=False,boot_edges={(0,0):explicit})
    np.testing.assert_array_equal(new[0],old[0]);np.testing.assert_array_equal(new[1].toarray(),old[1].toarray())


def test_no_source_rows_return_no_observations():
    p=SimpleNamespace(visible_rows=[],blade_rows=[],landmark_rows=[])
    residual,jac,parts,choices=pixel_factors(np.zeros((1,113,2)),p,body=False,boot_edges={})
    assert residual.shape==(0,) and jac.shape==(0,226) and parts=={} and choices==[]


def test_directed_open_topology_keeps_visible_collar_and_rejects_occluded_midpoints():
    vertices=local_vertices(INITIAL)[0][:14];before=vertices.copy();faces,_=open_boot_shell(vertices)
    assert faces.shape==(22,3)
    regions=np.array(['surface']*len(faces));regions[-2:]='upper'
    top=visible_shell_edges(vertices,faces,[0.,1.,0.],region_ids=regions)
    bottom=visible_shell_edges(vertices,faces,[0.,-1.,0.],region_ids=regions)
    collar={tuple(sorted((int(a),int(b)))) for a,b in zip(COLLAR_LOOP,np.roll(COLLAR_LOOP,-1))}
    assert collar<=set(map(tuple,top['edges']))
    assert not collar.intersection(set(map(tuple,bottom['edges'])))
    assert collar<=set(map(tuple,bottom['occluded_edges']))
    assert len(top['region_ids'])==len(top['edges'])
    actual={tuple(sorted((int(a),int(b)))) for face in faces for a,b in zip(face,np.roll(face,-1))}
    assert set(map(tuple,top['edges']))<=actual
    assert 'midpoint only' in top['approximation']
    np.testing.assert_array_equal(vertices,before)


def test_visible_edges_are_invariant_under_a_common_rigid_world_transform():
    vertices=local_vertices(INITIAL)[0][:14];faces,_=open_boot_shell(vertices);camera=np.array([1.,.3,.5])
    original=visible_shell_edges(vertices,faces,camera)
    R=Rotation.from_rotvec([.4,-.7,.2]).as_matrix();t=np.array([4.,-2.,8.])
    moved=visible_shell_edges(vertices@R.T+t,faces,R@camera+t)
    np.testing.assert_array_equal(original['edges'],moved['edges'])
    np.testing.assert_array_equal(original['occluded_edges'],moved['occluded_edges'])
    np.testing.assert_array_equal(original['face_front_facing'],moved['face_front_facing'])


def test_inconsistent_winding_is_not_silently_repaired():
    vertices=local_vertices(INITIAL)[0][:14];faces,_=open_boot_shell(vertices)
    faces=faces.copy();faces[0]=faces[0,::-1]
    with pytest.raises(ValueError,match='winding'):
        visible_shell_edges(vertices,faces,[1.,.3,.5])


def test_semantics_audit_keeps_source_values_and_separates_missing_toe_pick(tmp_path):
    root=tmp_path/'run';review_dir=root/'next_round_20261029/J1_s2';review_dir.mkdir(parents=True)
    admission_dir=root/'footwear_v2';admission_dir.mkdir()
    points=np.arange(68,dtype=float).reshape(2,17,2);points[1,16,1]=np.nan
    scores=np.ones((2,17));scores[1,16]=np.nan
    np.savez(root/'input.npz',pts=np.array([256,512]),source_uv=points,source_score=scores,
             source_qualified=np.ones((2,17),bool),lower_body_inferred=np.zeros(2,bool))
    contour=np.c_[np.linspace(0,1,48),np.full(48,3.)].tolist()
    admission=admission_dir/'admission.json'
    admission.write_text(json.dumps(dict(records=[dict(pts=256,foot=0,status='admitted_visible_boundary',
        source_boundary_native_uv=contour,reverse_boundary=False,sigma_px=3.)])),encoding='utf8')
    review=review_dir/'source_review.json'
    review.write_text(json.dumps(dict(source=dict(id='fixture'),
        blade_curves=[dict(pts=256,foot=0,uv=[[0.,4.],[1.,4.]],sigma_px=2.,partial=True)],
        boot_landmarks=[dict(pts=256,foot=0,heel=[0.,3.],toe=[1.,3.],sigma_px=2.)])),encoding='utf8')
    (review_dir/'config.json').write_text(json.dumps(dict(admission_sha256=digest(admission))),encoding='utf8')
    tracked=[root/'input.npz',admission,review,review_dir/'config.json'];before={p:digest(p) for p in tracked}
    with pytest.raises(ValueError,match='cannot overwrite'):
        audit_observation_semantics(root,output=review)
    report=audit_observation_semantics(root,output=tmp_path/'semantic_audit.json')
    assert before=={p:digest(p) for p in tracked}
    assert report['counts']['raw_ankle_detector_proxy_points']==4
    assert report['counts']['outer_boot_contour_source_points']==48
    assert report['counts']['outer_boot_heel_toe_points']==2
    assert report['counts']['toe_pick_source_points']==0
    assert report['counts']['nonfinite_raw_ankle_proxy_points']==1
    missing=next(r for r in report['records'] if r['kind']=='raw_ankle_detector_proxy' and r['pts']==512 and r['foot']==1)
    assert missing['raw_uv'][1] is None and missing['raw_uv_nonfinite']=={'1':'nan'}
    assert missing['raw_score'] is None and missing['raw_score_nonfinite']=='nan'
    json.dumps(report,allow_nan=False)
    assert report['missing']['toe_pick']['blade_front_endpoint_is_toe_pick'] is False
    heel=next(r for r in report['records'] if r['kind']=='outer_boot_heel_edge')
    assert heel['raw_uv']==[0.,3.] and heel['finite_shell_vertex_pair']==[0,7]
    raw_ankle=next(r for r in report['records'] if r['kind']=='raw_ankle_detector_proxy')
    assert raw_ankle['raw_uv']==points[0,15].tolist() and not raw_ankle['admitted_as_exact_bony_ankle']
    shell=next(r for r in report['records'] if r['kind']=='outer_boot_shell_boundary')
    np.testing.assert_array_equal(shell['raw_uv'],contour)
