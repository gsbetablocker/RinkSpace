from pathlib import Path
from types import SimpleNamespace
import sys
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'events/ankara_2026/programmes/okada_mei_sp')]


def test_actual_reader_cannot_extend_finite_photo_line_into_collinear_material():
    from reread_s04_lines import NativeLineReader
    # The right-hand band represents a different collinear painted object.
    # RGB contrast alone cannot assign it to the finite 400..900 source line.
    image=np.full((1080,1920,3),210,np.uint8);image[595:605,50:1850]=80
    q=np.c_[np.linspace(50,1850,100),np.full(100,600.)]
    path=dict(id='line',label='ice.lane1:edge0',curve_id='lane1:edge0',feature_id='lane1',world_curve_role='edge0',
        operator='paint_center',material_signal='dark',material_transition=None,_circle=False,
        xyz=[[0.,0.,0.],[1.,0.,0.]],_source_query_spacing_px=35,_separate_profile_support=True)
    class Identity:
        def select_guide(self,path,pt,guide,origin):return guide,origin,dict(current_identity_constrained=False)
        def record(self,*args):pass
    photo=SimpleNamespace(at=lambda *args:(dict(pts=256,spans=[np.array([[400.,600.],[900.,600.]])]),np.eye(3)))
    grid=SimpleNamespace(project=lambda pt:{'line':dict(uv=q,valid=np.ones(len(q),bool))})
    counts=[]
    for bounded in [False,True]:
        reader=object.__new__(NativeLineReader);reader.grid=grid;reader.paths=[path];reader.photographic=photo;reader.identity=Identity()
        reader.source_prefix='test';reader.source_cross_line_guides=True;reader.source_parallel_references=True;reader.source_finite_extents=bounded;reader.recover_partial_blue=False
        features,_=reader.read(256,image,np.zeros((1080,1920),bool),hashlib.sha256(image.tobytes()).hexdigest())
        rows=features['ice.lane1:edge0']['rows'];uv=np.array([r['uv']for r in rows]);counts.append(len(rows))
        if bounded:
            assert uv[:,0].min()>=400 and uv[:,0].max()<=900
            assert np.max(abs(uv[:,1]-599.5))<1
        else:assert uv[:,0].max()>1500
    assert counts[0]>counts[1]>2


def test_sparse_current_guide_uses_containing_photograph_only(monkeypatch):
    import points_s03_s05 as driver
    sys.path.insert(0,str(ROOT/'calibration/rinkfit/src'))
    from rinkfit_joint.finite_source_support import finite_support
    nearest=dict(pts=768,spans=[np.array([[500.,600.],[520.,600.]])])
    photo=dict(pts=512,path='actual_source_packet',spans=[np.array([[400.,600.],[900.,600.]])])
    value=object.__new__(driver.CompleteSourceGuides)
    value.paths={'line':{}};value.transforms={512:(np.eye(3),0),768:(np.eye(3),0)}
    value.photographs={'line':[photo]};value.extended=[];value.base=SimpleNamespace(support=finite_support)
    monkeypatch.setattr(driver.common.SourceGuides,'at',lambda *args:(nearest,np.eye(3)))
    assert value.at('line',768)[0]is photo
    # The broader picture must not authorize another parallel stripe or a
    # separate registration component after visibility/motion continuity fails.
    nearest['spans']=[np.array([[500.,650.],[520.,650.]])]
    assert value.at('line',768)[0]is nearest
    nearest['spans']=[np.array([[500.,600.],[520.,600.]])]
    value.transforms[512]=(np.eye(3),1)
    assert value.at('line',768)[0]is nearest


def test_circle_registration_ignores_withdrawn_material_points(tmp_path,monkeypatch):
    import json
    from points_s02 import SourceHouseReader
    from reread_s04_house import NativeHouseReader
    theta=np.linspace(0,2*np.pi,64,endpoint=False)
    source=np.c_[800+150*np.cos(theta),600+150*np.sin(theta)]
    base=np.c_[700+120*np.cos(theta),500+120*np.sin(theta)]
    actual=[dict(uv=q.tolist())for q in source]
    withdrawn=[dict(uv=q.tolist(),historical_identity_hold={'reason':'wrong_source_material'})for q in source+[450,250]]
    (tmp_path/'256.json').write_text(json.dumps(dict(features={'ice.F3:outer':dict(rows=actual+withdrawn)})),encoding='utf8')
    reader=object.__new__(SourceHouseReader);reader.source_pool_dir=tmp_path;reader.registration={};reader.registration_proof={};reader.cache={}
    reader.groups={'F3':{}};reader.native_paths={('F3','outer'):(1,[],dict(label='ice.F3:outer'))}
    monkeypatch.setattr(NativeHouseReader,'projection',lambda *args:(base,np.ones(len(base))))
    reader.prepare_family('F3',256)
    proof=reader.registration_proof[(256,'F3')]
    assert proof['original_actual_material_points']==len(source)
    H=reader.registration[(256,'F3')]
    assert np.allclose((H@np.array([700.,500.,1.]))[:2],[800,600],atol=.01)


def test_circle_registration_checks_support_again_after_ellipse_refit(tmp_path,monkeypatch):
    import json,cv2
    from points_s02 import SourceHouseReader
    from reread_s04_house import NativeHouseReader
    theta=np.linspace(0,2*np.pi,16,endpoint=False)
    uv=np.c_[800+150*np.cos(theta),600+30*np.sin(theta)].astype(np.float32)
    (tmp_path/'256.json').write_text(json.dumps(dict(features={'ice.F3:outer':dict(rows=[dict(uv=p.tolist())for p in uv])})),encoding='utf8')
    reader=object.__new__(SourceHouseReader);reader.source_pool_dir=tmp_path;reader.registration={};reader.registration_proof={};reader.cache={}
    reader.groups={'F3':{}};reader.native_paths={('F3','outer'):(1,[],dict(label='ice.F3:outer'))}
    monkeypatch.setattr(NativeHouseReader,'projection',lambda *args:(uv,np.ones(len(uv))))
    original=cv2.getClosestEllipsePoints;calls=[]
    def support_changes_after_refit(ellipse,points):
        calls.append(1)
        result=original(ellipse,points)
        if len(calls)==2:result[0]+=np.array([0.,12.],np.float32)
        return result
    monkeypatch.setattr(cv2,'getClosestEllipsePoints',support_changes_after_refit)
    reader.prepare_family('F3',256)
    assert len(calls)==2
    assert (256,'F3')not in reader.registration_proof
    assert np.array_equal(reader.registration[(256,'F3')],np.eye(3))


def test_final_support_repair_reads_rgb_instead_of_straightening_old_points():
    from reread_s04_lines import EntryProfiles,reread_final_support
    image=np.full((1080,1920,3),210,np.uint8);image[596:605,50:1850]=80
    path=dict(id='line',label='ice.B1:segment0',curve_id='B1:segment0',feature_id='B1',world_curve_role='segment0',
        operator='paint_center',material_signal='blue',material_transition=None,_circle=False,xyz=[[0.,0.,0.],[1.,0.,0.]])
    q=np.c_[np.linspace(100,1800,40),np.full(40,600.)];q[5,1]=592.
    fit=dict(point_origin=[100.,600.],unit_tangent=[1.,0.],unit_normal=[0.,1.],normal_offset_coefficients=[0.,0.],
        direct_readouts=[dict(uv=p.tolist())for p in q],residual_max_px=8.)
    native=EntryProfiles(image,dict(source_video_sha256=hashlib.sha256(image.tobytes()).hexdigest(),native_pts=256,time_base='1/12800'),np.zeros((1080,1920),bool))
    reread=reread_final_support(path,fit,native)['source_line_fit']
    assert reread['residual_max_px']<1
    assert fit['direct_readouts'][5]['uv'][1]==592.
    for row in reread['direct_readouts']:
        assert np.allclose(row['uv'],np.mean(row['source_evidence']['actual_source_sides_uv'],axis=0),atol=.001)
        assert abs(row['uv'][1]-600.)<1


def test_partial_ad_reference_confirms_full_artwork_by_shared_materials():
    from points_s03_s05 import photographic_path_agreement
    full={'top':[np.array([[100.,100.],[400.,100.]])],
        'bottom':[np.array([[100.,200.],[400.,200.]])],
        'letter_outside_local_crop':[np.array([[700.,120.],[720.,140.]])]}
    local={k:v for k,v in full.items()if k!='letter_outside_local_crop'}
    assert photographic_path_agreement(full,local,32.)['supported']
    # A coloured edge on the other advertisement is not instance evidence.
    wrong={k:[p+[1300.,0.]for p in v]for k,v in local.items()}
    assert not photographic_path_agreement(full,wrong,32.)['supported']
    # A single resolved 2D source mark also fixes the physical instance; do
    # not discard adjacent real letters merely because its crop is partial.
    tile={'tile':[np.array([[100.,100.],[130.,100.],[130.,130.],[100.,130.],[100.,100.]])]}
    assert photographic_path_agreement(tile,tile,32.)['supported']


def test_imperfect_camera_cannot_override_crossline_photographic_direction():
    from reread_s04_lines import NativeLineReader
    image=np.full((1080,1920,3),210,np.uint8);image[595:605,50:1850]=[190,90,50]
    x=np.linspace(50,1850,100);q=np.c_[x,600+.05*(x-50)]
    path=dict(id='line',label='ice.B1:segment0',curve_id='B1:segment0',feature_id='B1',world_curve_role='segment0',
        operator='paint_center',material_signal='blue',material_transition=None,_circle=False,
        xyz=[[0.,0.,0.],[1.,0.,0.]],_source_query_spacing_px=35,_separate_profile_support=True)
    class Identity:
        def select_guide(self,path,pt,guide,origin):return guide,origin,dict(current_identity_constrained=False)
        def record(self,*args):pass
    counts=[]
    for photographic in [False,True]:
        reader=object.__new__(NativeLineReader);reader.grid=SimpleNamespace(project=lambda pt:{'line':dict(uv=q,valid=np.ones(len(q),bool))})
        reader.paths=[path];reader.photographic=SimpleNamespace(at=lambda *args:(dict(pts=256,spans=[np.array([[50.,600.],[1850.,600.]])]),np.eye(3)))
        reader.identity=Identity();reader.source_prefix='test';reader.source_cross_line_guides=True;reader.source_parallel_references=True
        reader.source_finite_extents=True;reader.recover_partial_blue=False;reader.photographic_cross_direction=photographic
        features,_=reader.read(256,image,np.zeros((1080,1920),bool),hashlib.sha256(image.tobytes()).hexdigest())
        rows=features.get(path['label'],{}).get('rows',[]);counts.append(len(rows))
        if photographic:assert max(abs(r['uv'][1]-599.5)for r in rows)<1.
    assert counts[1]>30 and counts[1]>counts[0]


def test_blue_line_cannot_gain_identity_by_reading_other_side_of_red_centre():
    from points_s03_s05 import crossline_source_side,source_line_frame
    boundary=source_line_frame([[100.,600.],[1800.,630.]])
    correct=[dict(uv=[100.,500.],source_row_id='a'),dict(uv=[1700.,520.],source_row_id='b')]
    wrong=[dict(uv=[300.,780.],source_row_id='c'),dict(uv=[1500.,800.],source_row_id='d')]
    assert crossline_source_side(correct+wrong,boundary,-1)==correct
    assert crossline_source_side(correct+wrong,boundary,1)==wrong
    assert wrong[0]['uv']==[300.,780.]


def test_crossline_keeps_temporal_identity_when_another_blue_band_is_stronger():
    from reread_s04_lines import NativeLineReader
    image=np.full((1080,1920,3),210,np.uint8)
    image[595:605,50:1850]=[170,130,110]
    image[645:655,50:1850]=[230,40,20]
    q=np.c_[np.linspace(50,1850,100),np.full(100,600.)]
    path=dict(id='line',label='ice.B1:segment0',curve_id='B1:segment0',feature_id='B1',world_curve_role='segment0',
        operator='paint_center',material_signal='blue',material_transition=None,_circle=False,
        xyz=[[0.,0.,0.],[1.,0.,0.]],_source_query_spacing_px=35,_separate_profile_support=True)
    class Identity:
        def select_guide(self,path,pt,guide,origin):
            return guide,origin,dict(current_identity_constrained=True,identity_source='previous_actual_material',
                reference_frame=0,normal_identity_tolerance_px=9.,search_radius_px=18.,registration_error_px=0.)
        def record(self,*args):pass
    reader=object.__new__(NativeLineReader);reader.grid=SimpleNamespace(project=lambda pt:{'line':dict(uv=q,valid=np.ones(len(q),bool))})
    reader.paths=[path];reader.photographic=SimpleNamespace(at=lambda *args:(dict(pts=256,spans=[q]),np.eye(3)))
    reader.identity=Identity();reader.source_prefix='test';reader.source_cross_line_guides=True;reader.source_parallel_references=True
    reader.source_finite_extents=True;reader.recover_partial_blue=False;reader.photographic_cross_direction=True
    features,_=reader.read(256,image,np.zeros((1080,1920),bool),hashlib.sha256(image.tobytes()).hexdigest())
    rows=features[path['label']]['rows']
    assert len(rows)>30
    assert all(abs(r['uv'][1]-599.5)<1. for r in rows)
    assert all('temporal_material_identity'in r['source_evidence']for r in rows)


def test_family_retry_keeps_the_same_material_identity(monkeypatch):
    from reread_s04_lines import Extractor
    calls=[]
    def retry_once(self,index,features,searches,native):
        calls.append(index)
        if len(calls)==1:
            return [],[dict(target_id='ice.B1:segment0')],dict(slope_vs_y960=[0.,0.])
        return features,[],dict(slope_vs_y960=[0.,0.])
    monkeypatch.setattr(Extractor,'family_audit',retry_once)
    test_crossline_keeps_temporal_identity_when_another_blue_band_is_stronger()
    assert len(calls)==2


def test_blue_crossing_does_not_supply_a_thin_connection_reading():
    from calibration.program_tracking.audit_first_shot_circle_material_identity import centreline_house_conflicts
    blue=[dict(uv=[x,600.],source_row_id=f'b{x}',source_evidence={'actual_source_sides_uv':[[x,594.],[x,606.]]})for x in [450.,500.,550.]]
    thin=[dict(uv=[500.,600.],source_row_id='ambiguous'),dict(uv=[480.,580.],source_row_id='independent'),dict(uv=[800.,600.],source_row_id='outside_observed_band')]
    features={'ice.B1:segment0':dict(native_path_id='blue',rows=blue),'ice.house_connection_F2_R2:segment0':dict(native_path_id='thin',rows=thin)}
    entities={'blue':{'measurement':'centreline'},'thin':{'measurement':'centreline'}}
    assert not centreline_house_conflicts(features,entities)[0]
    rejected,_,_=centreline_house_conflicts(features,entities,include_crosslines=True)
    assert [r['source_row_id']for r in rejected]==['ambiguous']
    assert thin[0]['uv']==[500.,600.]
