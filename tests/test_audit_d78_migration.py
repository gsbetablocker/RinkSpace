"""Authenticated source proof has separate tests; these exercise migration packing."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from rinkspace.compat.d78_observations import import_d78_observations
from rinkspace.session.dataset import read_json,write_json

ROOT=Path(__file__).resolve().parents[1]

def fixture(tmp_path,monkeypatch):
    source=tmp_path/'old';source.mkdir()
    for name in ('packet','semantic','query'):(source/name).mkdir()
    def arr(owner,name,value):
        a=np.ascontiguousarray(value);p=source/owner/(name+'.bin');p.write_bytes(a.tobytes())
        return dict(file=p.name,dtype=a.dtype.str,shape=list(a.shape),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    def doc(owner,name,value):
        p=source/owner/name;write_json(p,value)
        return dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    native=dict(native_uv=arr('packet','uv',np.array([[2.,3.],[5.,6.]])),native_pts=arr('packet','pts',np.array([100,110],'<i8')),source_rows=arr('packet','row',np.arange(2,dtype='<u8')),material_index=arr('packet','material',np.zeros(2,'<u4')),point_id_index=arr('packet','point',np.arange(2,dtype='<u4')),engineering_budget_px=arr('packet','budget',np.ones(2)))
    write_json(source/'packet/manifest.json',dict(observations=2,arrays=native,string_tables=doc('packet','tables.json',dict(stable_observation_ids=['a','b'],source_point_ids=['a','b']))))
    sem={k:arr('semantic',k,v) for k,v in dict(kind=np.zeros(2,'u1'),profile_normal=np.array([[1.,0.],[1.,0.]]),query_uv=np.array([[2.,3.],[5.,6.]]),alpha_bounds=np.array([[-1.,1.],[-1.,1.]])).items()}
    weights=arr('semantic','weights',np.eye(4).ravel());members=arr('semantic','members',np.arange(2,dtype='<u8'));ordinary=arr('semantic','ordinary',np.array([],dtype='<u8'));sigma=arr('semantic','sigma',np.ones(2))
    group=source/'semantic/groups.jsonl';group.write_text(json.dumps(dict(group_id='correlated',member_start=0,member_count=2,rank=4,weight_start=0))+'\n')
    write_json(source/'semantic/manifest.json',dict(arrays=dict(sem,weights=weights,group_rows=members,ordinary_rows=ordinary,source_sigma=sigma),resolved_material_index=arr('semantic','mi',np.zeros(2,'<u4')),resolved_variant_index=arr('semantic','vi',np.zeros(2,'<u4')),resolved_variant_ids=['v'],active_source_mask=arr('semantic','active',np.ones(2,'u1')),statistical_active_mask=arr('semantic','statistical',np.ones(2,'u1')),assets={'groups.jsonl':dict(file=group.name,sha256=hashlib.sha256(group.read_bytes()).hexdigest())},statistics=dict(formal_residual_rows=4)))
    write_json(source/'query/manifest.json',dict(arrays=dict(source_rows=arr('query','rows',np.arange(2,dtype='<u8'))),definition={},parameter_definition={},certificates=doc('query','certificates.json',{})))
    write_json(source/'model.json',dict(materials=[dict(label='ice.C',paths=[dict(id='line',variant_id='v',measurement='centreline')])]))
    np.zeros(2,'<u4').tofile(source/'path.u32');np.save(source/'pts.npy',np.array([100,110],'<i8'))
    scope=dict(event_id='test',programme_id='clip',source=dict(id='video',registered_path='unread-video.mp4',sha256='a'*64,width=64,height=64),source_clock=dict(time_base_num=1,time_base_den=1000),required_shot_ids=['s01','s02'],shots=[dict(id='s01',rig_id='main',processing=dict(first_pts=100,stop_pts_exclusive=120)),dict(id='s02',rig_id='side',processing=dict(first_pts=120,stop_pts_exclusive=140))])
    write_json(source/'scope.json',scope)
    shots=[dict(scope['shots'][1],source_id='video'),dict(scope['shots'][0],source_id='video',camera_ref='camera.json',spline_reference_pts=100)]
    write_json(source/'shots.json',dict(schema='rinkspace.shots.v1',shots=shots));write_json(source/'rigs.json',dict(schema='rinkspace.rigs.v1',rigs=[dict(id='main',camera_ref='camera.json')]))
    np.save(source/'controls.npy',np.arange(12,dtype='<f8').reshape(4,3))
    write_json(source/'camera.json',dict(schema='rinkspace.camera.motion.v1',rig=[0.]*38,rig_id='main',shot_id='s01',controls_ref='controls.npy',reference_pts=100,first_control_time=-1.,control_dt=1.,source_binding_verified=True))
    monkeypatch.setattr('rinkspace.compat.d78_observations.resolve_d78_source_binding',lambda *a,**kw:dict(original_reference_pts=100,verified=True))
    return source,dict(input_path=source,model_path=source/'model.json',scope_path=source/'scope.json',pts_index_path=source/'pts.npy',out_dir=tmp_path/'published',path_indices_path=source/'path.u32',scene_path=ROOT/'events/ankara_2026/scene/field/model.json',rigs_path=source/'rigs.json',shots_path=source/'shots.json',source_binding_path=source/'unused-proof.json')

def test_relative_owners_shot_order_and_relocation(tmp_path,monkeypatch):
    source,args=fixture(tmp_path,monkeypatch);import_d78_observations(**args)
    out=args['out_dir'];obs=read_json(out/'observations.json');groups=read_json(out/'noise_groups.json')
    np.testing.assert_array_equal(np.load(out/obs['arrays']['uv']['path']),[[2.,3.],[5.,6.]])
    np.testing.assert_array_equal(np.load(out/obs['arrays']['shot_index']['path']),[1,1])
    np.testing.assert_array_equal(np.load(out/groups['weights_ref']['path']),np.eye(4).ravel())
    moved=tmp_path/'relocated';out.rename(moved);source.rename(tmp_path/'original-no-longer-at-declared-path')
    code="""from pathlib import Path
import sys
from rinkspace.session.dataset import read_json,load_array
from rinkspace.domain.runtime_rigs import _camera_data
from rinkspace.domain.scene_assets import read_scene_assets
r=Path(sys.argv[1]);o=read_json(r/'observations.json')
for d in o['arrays'].values():load_array(r,d,verify=True)
s=read_json(r/'shots.json')['shots'];c,_,_,_,k=_camera_data(r/s[1]['camera_ref'])
assert c['reference_pts']==100 and k.shape==(4,3)
assert read_scene_assets(read_json(r/'scene.json'),r/'scene.json')
assert read_json(r/'input.json')['model_configuration']['available_camera_shot_ids']==['s01']
"""
    p=subprocess.run([sys.executable,'-I','-B','-c',code,str(moved)],cwd=tmp_path,capture_output=True,text=True)
    assert p.returncode==0,p.stderr

@pytest.mark.parametrize('conflict',['mixed_shots','rig','clock_origin','missing_camera'])
def test_conflicts_do_not_publish_partial_migration(tmp_path,monkeypatch,conflict):
    source,args=fixture(tmp_path,monkeypatch)
    if conflict=='mixed_shots':
        np.save(source/'row_shots.npy',np.array(['s01','s02']));args['row_shot_ids_path']=source/'row_shots.npy'
    else:
        shots=read_json(source/'shots.json')
        if conflict=='rig':shots['shots'][1]['rig_id']='side'
        if conflict=='clock_origin':shots['shots'][1]['spline_reference_pts']=110
        if conflict=='missing_camera':shots['shots'][1]['camera_ref']='missing.json'
        write_json(source/'shots.json',shots)
    with pytest.raises((ValueError,FileNotFoundError)):import_d78_observations(**args)
    assert not args['out_dir'].exists()
