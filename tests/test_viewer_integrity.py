"""Execute the production viewer data/cache functions on tiny temporary NPZs.

No GUI, SMPL-X weights, camera fitting or pose inference is run. AssetStore and
body-model imports are isolated; the cache baker receives a synthetic Surface.
Set RINKSPACE_AUDIT_SOURCE to the pinned original, patched snapshot or checkout.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

# Preload lazy scientific imports before patch.dict restores sys.modules.
import scipy.interpolate
import scipy.sparse
import scipy.spatial

SOURCE=Path(os.environ.get('RINKSPACE_AUDIT_SOURCE', Path(__file__).resolve().parents[1])).resolve()


def _module(name, **attrs):
    result=types.ModuleType(name)
    result.__dict__.update(attrs)
    return result


def _unused(*args, **kwargs):
    raise AssertionError('A real external asset/model must not run in this test')


def _equipment(P,Q,native,frames,geometry):
    n=len(P)
    return np.zeros((n,2,48,3)),{},np.zeros((n,2,3,3)),np.zeros((n,2,3))


def _minimum(rot,origin,geometry):
    return np.zeros((len(origin),2,3))


def load_modules():
    folder=SOURCE/'src/rinkspace/viewer3d'
    package=_module('_audit_viewer',__path__=[str(folder)])
    stubs={
        'rinkspace':_module('rinkspace',__path__=[]),
        'rinkspace.storage':_module('rinkspace.storage',AssetStore=_unused),
        'rinkspace.metric_scale':_module('rinkspace.metric_scale',estimate_uniform_scale=_unused),
        'calibration':_module('calibration',__path__=[]),
        'calibration.program_tracking':_module('calibration.program_tracking',__path__=[]),
        'calibration.program_tracking.body_tracking':_module('calibration.program_tracking.body_tracking',__path__=[]),
        'calibration.program_tracking.body_tracking.rebuild_body_geometry':_module('body_geometry',CompactBody=_unused),
        'calibration.program_tracking.body_tracking.footwear_geometry':_module('footwear_geometry',extra_rotations=_unused,equipment_world=_equipment,lowest_blade_world=_minimum),
        '_audit_viewer':package,
    }
    with patch.dict(sys.modules,stubs):
        modules=[]
        for leaf in ('data','prepare'):
            name='_audit_viewer.'+leaf
            spec=importlib.util.spec_from_file_location(name,folder/(leaf+'.py'))
            obj=importlib.util.module_from_spec(spec)
            sys.modules[name]=obj
            spec.loader.exec_module(obj)
            modules.append(obj)
    return modules


DATA,PREPARE=load_modules()
FLAG_REL=Path('events/ankara_2026/programmes/okada_mei_sp/body_s01_joint_candidate_20261029.npz')


def write_npz(path,**arrays):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)


def replace_npz(path,**changes):
    current=DATA.load_npz(path);current.update(changes);write_npz(path,**current)


def catalog(tmp_path,kind='pose'):
    c=DATA.ResultCatalog.__new__(DATA.ResultCatalog)
    c.project=tmp_path/'project';c.root=tmp_path/'run';c.cache_root=tmp_path/'cache'
    c.project.mkdir();c.root.mkdir()
    c.pts=np.array([283136,283392,283648],np.int64)
    c.scale=60.;c.legacy_scale=60.;c.unit_ratio=1.
    c._common_fingerprint=None
    c.packet=dict(pts=c.pts,fit_time_seconds=np.array([0.,.02,.04]),pose_axis_angle=np.zeros((3,21,3)),lower_body_inferred=np.array([True,False,False]))
    write_npz(c.root/'input.npz',**c.packet)
    write_npz(c.root/'compact_smplx.npz',placeholder=np.zeros(1))
    c.model_path=c.root/'synthetic_model_identity.npz';write_npz(c.model_path,placeholder=np.ones(1))
    for name in ('rebuild_body_geometry.py','footwear_geometry.py'):
        p=c.project/'calibration/program_tracking/body_tracking'/name
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# synthetic dependency identity\n')
    write_npz(c.project/FLAG_REL,pts=c.pts,source_identity_suspect=np.array([False,True,False]))
    v=dict(id='fixture',kind=kind,result='result.npz')
    result=dict(pts=c.pts,P_world_mid=np.zeros((3,3)),Q_world=np.tile(np.eye(3),(3,1,1)),COCO_world_mid=np.zeros((3,17,3)))
    if kind=='joint':
        v['base']='base.npz'
        write_npz(c.root/v['base'],pts=c.pts,delta_local_rad=np.zeros((3,9)))
        result.update(correction_knots=np.r_[np.zeros(4),np.full(4,.04)],leg_controls=np.zeros((4,14)),shared_geometry=np.zeros(14),nominal_blade_minimum_world_m=np.zeros((3,2,3)))
    else:result['delta_local_rad']=np.zeros((3,9))
    write_npz(c.root/v['result'],**result)
    c.variants=[v]
    return c


class Surface:
    def __init__(self,*,error=0.,vertex_value=0.,joint_value=0.,proxy_value=0.):
        self.shaped=np.zeros((4,3));self.faces=np.array([[0,1,2]],np.int32)
        self.body=types.SimpleNamespace(parents=np.r_[-1,np.zeros(21,dtype=int)])
        self.error=error;self.vertex_value=vertex_value;self.joint_value=joint_value;self.proxy_value=proxy_value
    def forward(self,pose,delta,leg):
        n=len(pose)
        return (np.full((n,4,3),self.vertex_value),np.zeros((n,17,3)),np.full((n,22,3),self.joint_value),np.full((n,2,5,3),self.proxy_value),np.tile(np.eye(3),(n,22,1,1)),self.error)


@pytest.mark.parametrize('kind',['pose','joint'])
def test_valid_bake_matches_existing_values(tmp_path,kind):
    c=catalog(tmp_path,kind);path=PREPARE.prepare_variant(c,'fixture',Surface())
    seq=DATA.CachedSequence(path)
    np.testing.assert_array_equal(seq.state['pts'],c.pts)
    np.testing.assert_array_equal(seq.state['source_identity_suspect'],[False,True,False])
    assert seq.meta['saved_coco_mid_maximum_reconstruction_error_m']==0.
    assert np.isfinite(seq.vertices).all()
    assert PREPARE.prepare_variant(c,'fixture',Surface(error=10.))==path # unchanged valid cache reused


@pytest.mark.parametrize('change',['shift','reorder'])
def test_joint_base_clock_must_match(tmp_path,change):
    c=catalog(tmp_path,'joint')
    wrong=c.pts+256 if change=='shift' else c.pts[::-1]
    replace_npz(c.root/'base.npz',pts=wrong)
    with pytest.raises((ValueError,AssertionError)):
        c.parameters('fixture')


def test_joint_base_matching_clock_keeps_coefficients(tmp_path):
    c=catalog(tmp_path,'joint');delta=np.arange(27,dtype=float).reshape(3,9)/100
    replace_npz(c.root/'base.npz',delta_local_rad=delta)
    np.testing.assert_array_equal(c.parameters('fixture')['delta'],delta)


def test_source_flag_change_invalidates_existing_cache(tmp_path):
    c=catalog(tmp_path);old=PREPARE.prepare_variant(c,'fixture',Surface())
    write_npz(c.project/FLAG_REL,pts=c.pts,source_identity_suspect=np.array([True,False,True]))
    new=PREPARE.prepare_variant(c,'fixture',Surface())
    assert new!=old,'changed warning source reused old cache'
    np.testing.assert_array_equal(DATA.load_npz(new/'state.npz')['source_identity_suspect'],[True,False,True])
    np.testing.assert_array_equal(DATA.load_npz(old/'state.npz')['source_identity_suspect'],[False,True,False])


@pytest.mark.parametrize('field',['COCO_world_mid','nominal_blade_minimum_world_m'])
@pytest.mark.parametrize('invalid',[np.nan,np.inf,-np.inf])
def test_invalid_saved_comparison_cannot_publish(tmp_path,field,invalid):
    c=catalog(tmp_path,'joint');saved=DATA.load_npz(c.root/'result.npz')[field];saved[0,0,0]=invalid
    replace_npz(c.root/'result.npz',**{field:saved})
    with pytest.raises((ValueError,AssertionError)):
        PREPARE.prepare_variant(c,'fixture',Surface())
    assert not list(c.cache_root.glob('fixture_*/metadata.json'))


@pytest.mark.parametrize('invalid',[np.nan,np.inf,-np.inf])
def test_invalid_surface_error_cannot_publish(tmp_path,invalid):
    c=catalog(tmp_path)
    with pytest.raises((ValueError,AssertionError)):
        PREPARE.prepare_variant(c,'fixture',Surface(error=invalid))
    assert not list(c.cache_root.glob('fixture_*/metadata.json'))


@pytest.mark.parametrize('kw',[dict(vertex_value=np.nan),dict(vertex_value=1e40),dict(joint_value=np.nan),dict(proxy_value=np.nan)])
def test_invalid_display_geometry_cannot_publish(tmp_path,kw):
    c=catalog(tmp_path)
    with np.errstate(over='ignore',invalid='ignore'):
        with pytest.raises((ValueError,AssertionError)):
            PREPARE.prepare_variant(c,'fixture',Surface(**kw))
    assert not list(c.cache_root.glob('fixture_*/metadata.json'))


def test_finite_mismatch_is_still_rejected(tmp_path):
    c=catalog(tmp_path);replace_npz(c.root/'result.npz',COCO_world_mid=np.ones((3,17,3))*.001)
    with pytest.raises(ValueError):PREPARE.prepare_variant(c,'fixture',Surface())
    assert not list(c.cache_root.glob('fixture_*/metadata.json'))


def test_source_arrays_not_mutated_by_bake(tmp_path):
    c=catalog(tmp_path,'joint');files=[c.root/'input.npz',c.root/'result.npz',c.root/'base.npz',c.project/FLAG_REL]
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    PREPARE.prepare_variant(c,'fixture',Surface())
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def test_warning_clock_mismatch_rejected_before_publish(tmp_path):
    c=catalog(tmp_path);write_npz(c.project/FLAG_REL,pts=c.pts[::-1],source_identity_suspect=np.array([False,True,False]))
    with pytest.raises((ValueError,AssertionError)):
        PREPARE.prepare_variant(c,'fixture',Surface())
    assert not list(c.cache_root.glob('fixture_*/metadata.json'))


def test_warning_length_mismatch_rejected_before_publish(tmp_path):
    c=catalog(tmp_path);write_npz(c.project/FLAG_REL,pts=c.pts,source_identity_suspect=np.array([True]))
    with pytest.raises((ValueError,AssertionError)):
        PREPARE.prepare_variant(c,'fixture',Surface())
    assert not list(c.cache_root.glob('fixture_*/metadata.json'))


def test_historical_warning_table_without_pts_keeps_format(tmp_path):
    c=catalog(tmp_path);write_npz(c.project/FLAG_REL,source_identity_suspect=np.array([True,False,True]))
    path=PREPARE.prepare_variant(c,'fixture',Surface())
    np.testing.assert_array_equal(DATA.load_npz(path/'state.npz')['source_identity_suspect'],[True,False,True])


def test_broadcasted_saved_pose_is_not_a_matching_pose(tmp_path):
    c=catalog(tmp_path);replace_npz(c.root/'result.npz',COCO_world_mid=np.zeros((3,1,3)))
    with pytest.raises((ValueError,AssertionError)):
        PREPARE.prepare_variant(c,'fixture',Surface())
    assert not list(c.cache_root.glob('fixture_*/metadata.json'))


def test_existing_reconstruction_thresholds_unchanged(tmp_path):
    c=catalog(tmp_path);replace_npz(c.root/'result.npz',COCO_world_mid=np.full((3,17,3),1e-7))
    path=PREPARE.prepare_variant(c,'fixture',Surface(error=2e-5))
    meta=json.loads((path/'metadata.json').read_text())
    assert meta['saved_coco_mid_maximum_reconstruction_error_m']==1e-7
    assert meta['full_vs_compact_coco_maximum_error_m']==2e-5


def test_warning_changes_between_key_and_snapshot_are_rejected(tmp_path):
    c=catalog(tmp_path);original=c.cache_path
    def race(variant):
        path,identity=original(variant)
        write_npz(c.project/FLAG_REL,pts=c.pts,source_identity_suspect=np.array([True,False,True]))
        return path,identity
    c.cache_path=race
    with pytest.raises((ValueError,AssertionError)):
        PREPARE.prepare_variant(c,'fixture',Surface())
    assert not list(c.cache_root.glob('fixture_*/metadata.json'))


def fullbody_catalog(tmp_path):
    c=catalog(tmp_path);c.scale=90.
    c.variants[0]['kind']='fullbody'
    n=len(c.pts);P=np.tile([1.,2.,3.],(n,1));Q=np.tile(np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]),(n,1,1))
    local=np.broadcast_to(np.eye(3),(n,21,3,3)).copy();local[:,5]=Q
    native=P[:,None]+np.einsum('nij,nkj->nki',Q,np.ones((n,22,3))*.5)
    gear=np.arange(n*2*48*3,dtype=float).reshape(n,2,48,3)/100
    G=P.copy();G[:,2]=0
    write_npz(c.root/'result.npz',pts=c.pts,source_pts=c.pts,P_world_mid=P,Q_world=Q,Q_world_mid=Q,
              local_R_mid=local,native22_world_mid=native,COCO_world_mid=np.broadcast_to(P[:,None],(n,17,3)),
              boot_world_mid=gear[:,:,:14],blade_world_mid=gear[:,:,14:],gear_world_mid=gear,G_world_mid=G,
              metres_per_scene_unit=np.array(30.),solver_metres_per_scene_unit=np.array(60.),body_length_ratio=np.array(.5),
              window_mask=np.array([False,True,False]),source_lower_body_inferred=np.array([False,False,True]),
              ledger_id=np.array('evidence:fixture'),reference_hashes_json=np.array('{"old_j1":"synthetic_sha"}'),
              shared_geometry=np.full(14,np.nan)) # provenance must never drive equipment
    return c


class MatrixSurface(Surface):
    def __init__(self):super().__init__(vertex_value=.25,joint_value=1.);self.received=[]
    def forward(self,*args):raise AssertionError('Unified rotations received a second pose-correction pass')
    def forward_rotations(self,local_R):
        self.received.append(local_R.copy())
        return Surface.forward(self,np.zeros((len(local_R),21,3)),None,None)


def test_fullbody_bake_uses_effective_rotations_candidate_units_and_saved_equipment(tmp_path):
    c=fullbody_catalog(tmp_path);surface=MatrixSurface();saved=DATA.load_npz(c.root/'result.npz')
    path=PREPARE.prepare_variant(c,'fixture',surface);seq=DATA.CachedSequence(path)
    np.testing.assert_array_equal(np.concatenate(surface.received),saved['local_R_mid'])
    # Candidate root is in 30 m/scene units; its model lengths are half the old
    # 60-unit model. Reexpress both together in the 90-unit comparison scene.
    expected=(saved['P_world_mid'][:,None]+np.einsum('nij,nvj->nvi',saved['Q_world_mid'],np.full((3,4,3),.125)))*3
    np.testing.assert_allclose(seq.vertices,expected)
    np.testing.assert_allclose(seq.state['P'],saved['P_world_mid']*3)
    np.testing.assert_allclose(seq.state['joints'],saved['native22_world_mid']*3)
    np.testing.assert_allclose(seq.state['gear'],saved['gear_world_mid']*3)
    np.testing.assert_array_equal(seq.state['window_mask'],[False,True,False])
    np.testing.assert_array_equal(seq.state['source_lower_body_inferred'],[False,False,True])
    assert seq.meta['uniform_unit_reexpression_ratio']==3.
    assert seq.meta['body_length_ratio']==.5
    assert seq.meta['optimized_frames']==1
    assert seq.meta['candidate_provenance']['ledger_id']=='evidence:fixture'
    assert not seq.meta['accuracy_accepted']


def test_fullbody_source_pts_alone_is_a_valid_source_clock(tmp_path):
    c=fullbody_catalog(tmp_path);saved=DATA.load_npz(c.root/'result.npz');del saved['pts']
    write_npz(c.root/'result.npz',**saved)
    np.testing.assert_array_equal(c.parameters('fixture')['saved']['source_pts'],c.pts)


@pytest.mark.parametrize('field,value',[
    ('local_R_mid',np.zeros((3,21,3,3))),
    ('window_mask',np.array([0,1,0])),
    ('window_mask',np.array([True])),
    ('source_pts',np.array([1,2,3])),
    ('body_length_ratio',np.array(1.)),
    ('solver_metres_per_scene_unit',np.array(30.)),
    ('native22_world_mid',np.full((3,22,3),np.nan)),
    ('boot_world_mid',np.zeros((3,2,14,3))),
])
def test_fullbody_invalid_contract_cannot_publish(tmp_path,field,value):
    c=fullbody_catalog(tmp_path);replace_npz(c.root/'result.npz',**{field:value})
    with pytest.raises((ValueError,AssertionError)):PREPARE.prepare_variant(c,'fixture',MatrixSurface())
    assert not list(c.cache_root.glob('fixture_*/metadata.json'))


def test_fullbody_native_reconstruction_mismatch_cannot_publish(tmp_path):
    c=fullbody_catalog(tmp_path);saved=DATA.load_npz(c.root/'result.npz')
    replace_npz(c.root/'result.npz',native22_world_mid=saved['native22_world_mid']+.001)
    with pytest.raises(ValueError):PREPARE.prepare_variant(c,'fixture',MatrixSurface())
    assert not list(c.cache_root.glob('fixture_*/metadata.json'))


def test_candidate_scale_override_invalidates_cache_and_must_match_result(tmp_path):
    c=fullbody_catalog(tmp_path);old,_=c.cache_path('fixture')
    c.variants[0]['metres_per_scene_unit']=31.
    new,_=c.cache_path('fixture');assert new!=old
    with pytest.raises((ValueError,AssertionError)):c.parameters('fixture')


@pytest.mark.parametrize('metric_base,absolute',[(None,False),('project',False),('data',False),(None,True),('project',True),('data',True),('invalid',False)])
def test_comparison_scene_does_not_require_old_input_and_new_camera_same_scale(tmp_path,monkeypatch,metric_base,absolute):
    project=tmp_path/'project';data=tmp_path/'data';run=data/'old_run';scene=data/'new_scene'
    model=project/'model.npz';write_npz(model,placeholder=np.zeros(1))
    pts=np.array([100,356],np.int64)
    write_npz(run/'input.npz',pts=pts,metres_per_scene_unit=np.array(60.))
    write_npz(scene/'camera_snapshot.npz',board=np.array([1.,2.,3.]))
    metric_root=data if metric_base=='data' else project
    metric=metric_root/'metric.json';metric.write_text(json.dumps(dict(metres_per_scene_unit=50.)))
    values=dict(source_run='old_run',scene_root=str(scene),metric_binding=str(metric) if absolute else 'metric.json',
        display_metres_per_scene_unit=50.,model_asset='model.npz',shot='s01',variants=[])
    if metric_base is not None:values['metric_binding_base']=metric_base
    config=project/'comparison.json';config.write_text(json.dumps(values))
    class Assets:
        def __init__(self,*args,**kwargs):self.data=data
        def resolve(self,path,**kwargs):return path
    monkeypatch.setattr(DATA,'AssetStore',Assets)
    monkeypatch.setattr(DATA,'estimate_uniform_scale',lambda _:50.)
    if metric_base=='invalid':
        with pytest.raises(ValueError,match='Invalid metric_binding_base'):DATA.ResultCatalog(project,data_root=data,config=config)
        return
    c=DATA.ResultCatalog(project,data_root=data,config=config)
    assert c.root==run and c.scene_root==scene and c.metric_path==metric
    assert c.scale==50. and c.legacy_scale==60. and c.unit_ratio==50./60.


def boot_proxy():
    profile=np.array([[0.,0.],[.30,0.],[.32,.04],[.12,.12],[.10,.28],[-.02,.27],[-.04,.06]])
    side=np.column_stack([profile[:,0],np.full(7,-.04),profile[:,1]])
    other=side.copy();other[:,1]=.04
    return np.r_[side,other]


def test_open_collar_retains_profile_and_only_intended_boundary():
    from collections import Counter
    points=boot_proxy();before=points.copy();faces=DATA.shell_triangles(points)
    np.testing.assert_array_equal(points,before)
    assert faces.shape==(22,3)
    edges=Counter(tuple(sorted((int(a),int(b)))) for tri in faces for a,b in zip(tri,np.roll(tri,-1)))
    assert {edge for edge,count in edges.items() if count==1}=={(4,5),(5,12),(11,12),(4,11)}
    assert set(edges.values())=={1,2}
    moved=points@np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])+[12.,-7.,2.]
    moved_faces=DATA.shell_triangles(moved)
    assert {tuple(sorted(t)) for t in moved_faces}=={tuple(sorted(t)) for t in faces}
    closed=DATA.shell_triangles(points,'closed_hull')
    closed_edges=Counter(tuple(sorted((int(a),int(b)))) for tri in closed for a,b in zip(tri,np.roll(tri,-1)))
    assert set(closed_edges.values())=={2}
    # The concavity remains in the explicit side instead of being convexified.
    side_area=sum(np.linalg.norm(np.cross(points[b]-points[a],points[c]-points[a]))*.5 for a,b,c in faces[:5])
    x,z=points[:7,0],points[:7,2]
    expected=abs(np.sum(x*np.roll(z,-1)-z*np.roll(x,-1)))*.5
    np.testing.assert_allclose(side_area,expected,rtol=0,atol=1e-12)


def test_open_collar_rejects_crossing_profile():
    points=boot_proxy();points[[2,4]]=points[[4,2]];points[[9,11]]=points[[11,9]]
    with pytest.raises(ValueError):DATA.shell_triangles(points)


def add_footwear(c):
    saved=DATA.load_npz(c.root/'result.npz');surface=np.broadcast_to(boot_proxy(),(3,2,14,3)).copy()
    surface[1,:,4,0]+=.003 # finite upper deformation with the same faces
    gear=saved['gear_world_mid'].copy();gear[:,:,:14]=surface
    axes=np.zeros((3,2,9,2,3));axes[:,:,:,1]=np.eye(3)[np.arange(9)%3]*.1
    faces=DATA.shell_triangles(boot_proxy())
    replace_npz(c.root/'result.npz',gear_world_mid=gear,boot_world_mid=surface,boot_surface_world_mid=surface,
                boot_surface_faces=faces,boot_surface_region_ids=np.r_[np.zeros(5,int),np.ones(5,int),np.full(12,2)],
                collar_boundary=np.array([4,5,12,11]),footbed_world_mid=surface[:,:,[0,1,8,7]],
                cavity_world_mid=surface[:,:,[0,1,4,7]],cavity_faces=np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]]),
                anchors_world_mid=surface[:,:,[0,1,7,8]],anchor_labels=np.array(['foot_heel','foot_ball','boot_heel','boot_ball']),
                axes_world_mid=axes,axis_labels=np.array(['shank_x','shank_y','shank_z','boot_x','boot_y','boot_z','blade_x','blade_y','blade_z']),
                upper_deformation_mid=np.full((3,2,3),.02),effective_ankle_relative_rotation_mid=np.tile(np.eye(3),(3,2,1,1)),
                footwear_binding_applied_mask=np.array([True,True,True]),body_refinement_window_mask=np.array([False,True,False]),
                cavity_penetration_mid=np.full((3,2),.001),heel_drift_mid=np.full((3,2),.002),
                forefoot_error_mid=np.full((3,2),.003),blade_tilt_rad_mid=np.full((3,2),.2))
    return DATA.load_npz(c.root/'result.npz')


def test_saved_dynamic_footwear_keeps_faces_and_scales_only_lengths(tmp_path):
    c=fullbody_catalog(tmp_path);saved=add_footwear(c)
    seq=DATA.CachedSequence(PREPARE.prepare_variant(c,'fixture',MatrixSurface()))
    for target,source in [('boot_surface','boot_surface_world_mid'),('footbed','footbed_world_mid'),('cavity','cavity_world_mid'),('anchors','anchors_world_mid'),('axes','axes_world_mid')]:
        np.testing.assert_array_equal(seq.state[target],saved[source]*3)
    np.testing.assert_array_equal(seq.state['boot_faces'],saved['boot_surface_faces'])
    np.testing.assert_array_equal(seq.state['upper_deformation'],saved['upper_deformation_mid'])
    np.testing.assert_array_equal(seq.state['effective_ankle_relative_rotation'],saved['effective_ankle_relative_rotation_mid'])
    np.testing.assert_array_equal(seq.state['blade_tilt_rad'],saved['blade_tilt_rad_mid'])
    np.testing.assert_array_equal(seq.state['heel_drift'],saved['heel_drift_mid']*3)
    np.testing.assert_array_equal(seq.state['footwear_binding_applied_mask'],[True,True,True])
    np.testing.assert_array_equal(seq.state['body_refinement_window_mask'],[False,True,False])
    assert seq.meta['footwear_display']['saved_dynamic_surface']
    assert not seq.meta['footwear_display']['retriangulated']


def test_footwear_metadata_separates_new_body_frames_from_shared_binding(tmp_path):
    c=fullbody_catalog(tmp_path);add_footwear(c)
    replace_npz(c.root/'result.npz',window_mask=np.array([True,True,True]))
    seq=DATA.CachedSequence(PREPARE.prepare_variant(c,'fixture',MatrixSurface()))
    assert seq.meta['optimized_frames']==1
    assert seq.meta['body_refinement_frames']==1
    assert seq.meta['parent_window_frames']==3
    assert seq.meta['footwear_binding_applied_frames']==3
    assert seq.meta['footwear_binding_scope']=='shared across all S01 exposures'
    assert 'fullbody1102' in seq.meta['outside_window']
    assert 'J1' not in seq.meta['outside_window']
    assert 'local dynamic upper' in seq.meta['equipment']


@pytest.mark.parametrize('variant,reference,intervals',[
    ({},'fullbody1102',[[35.2,36.8],[40.,41.5]]),
    ({'body_reference_label':'boot_integration1102','body_refinement_intervals_seconds':[[22.12,25.]]},'boot_integration1102',[[22.12,25.]]),
    ({'body_reference_label':'boot_integration1102','body_refinement_intervals_seconds':[]},'boot_integration1102',[]),
])
def test_body_refinement_metadata_uses_variant_parent_and_intervals_with_existing_defaults(tmp_path,variant,reference,intervals):
    c=fullbody_catalog(tmp_path);add_footwear(c)
    c.config={'body_refinement_intervals_seconds':[[35.2,36.8],[40.,41.5]]}
    c.variants[0].update(variant)
    seq=DATA.CachedSequence(PREPARE.prepare_variant(c,'fixture',MatrixSurface()))
    assert seq.meta['body_reference_label']==reference
    assert seq.meta['body_reference']=='parent '+reference+' trajectory'
    assert seq.meta['outside_window']=='body inherited from parent '+reference+' trajectory; shared footwear binding can still apply'
    assert seq.meta['body_refinement_intervals_seconds']==intervals


@pytest.mark.parametrize('field,value',[
    ('boot_surface_faces',np.array([[0,1,14]])),
    ('boot_surface_faces',np.array([[0.,1.,2.]])),
    ('axes_world_mid',np.full((3,2,9,2,3),np.nan)),
    ('anchor_labels',np.array(['heel'])),
    ('effective_ankle_relative_rotation_mid',np.zeros((3,2,3,3))),
    ('heel_drift_mid',np.zeros((3,1))),
    ('footwear_binding_applied_mask',np.array([1,1,1])),
    ('body_refinement_window_mask',np.array([True])),
])
def test_invalid_saved_footwear_cannot_publish(tmp_path,field,value):
    c=fullbody_catalog(tmp_path);add_footwear(c);replace_npz(c.root/'result.npz',**{field:value})
    with pytest.raises((ValueError,AssertionError)):PREPARE.prepare_variant(c,'fixture',MatrixSurface())
    assert not list(c.cache_root.glob('fixture_*/metadata.json'))


def synthetic_skinning_surface():
    """Small, nontrivial skinning model; no external body asset is loaded."""
    rng=np.random.default_rng(20261102);surface=PREPARE.FullSurface.__new__(PREPARE.FullSurface)
    J=rng.normal(0,.1,(55,3));parents=np.r_[-1,np.zeros(54,dtype=int)]
    surface.body=types.SimpleNamespace(parents=parents,J=J,scale=.91,default=np.tile(np.eye(3),(33,1,1)))
    surface.shaped=rng.normal(0,.2,(8,3)).astype(np.float32)
    surface.posedirs=rng.normal(0,.003,(486,24)).astype(np.float32)
    weights=rng.uniform(0,1,(8,55));surface.weights=(weights/weights.sum(axis=1,keepdims=True)).astype(np.float32)
    return surface


def test_double_precision_skinning_has_stable_pose_differences_and_full_subset_agreement():
    from scipy.spatial.transform import Rotation
    surface=synthetic_skinning_surface();ids=np.array([0,3,7]);J=surface.body.J
    def at(angle,double_precision=True,selected=ids):
        frames=np.tile(np.eye(3),(1,22,1,1));frames[:,7]=Rotation.from_rotvec([angle,.13,-.07]).as_matrix()
        joints=np.broadcast_to((J[:22]-J[0])*surface.body.scale,(1,22,3)).copy()
        return surface.sample_vertices(joints,frames,selected,double_precision=double_precision)
    h=1e-5
    difference_h=(at(.2+h)-at(.2-h))/(2*h)
    difference_2h=(at(.2+2*h)-at(.2-2*h))/(4*h)
    np.testing.assert_allclose(difference_h,difference_2h,rtol=0,atol=2e-9)
    np.testing.assert_allclose(at(.2),at(.2,selected=None)[:,ids],rtol=0,atol=1e-14)
    np.testing.assert_allclose(at(.2),at(.2,double_precision=False),rtol=0,atol=2e-6)


def test_default_skinning_branch_remains_the_same_as_explicit_false():
    surface=synthetic_skinning_surface();frames=np.tile(np.eye(3),(1,22,1,1))
    joints=np.broadcast_to((surface.body.J[:22]-surface.body.J[0])*surface.body.scale,(1,22,3)).copy()
    np.testing.assert_array_equal(surface.sample_vertices(joints,frames),surface.sample_vertices(joints,frames,double_precision=False))
