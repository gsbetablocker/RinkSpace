from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import cv2,pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'events/ankara_2026/programmes/okada_mei_sp')]
from calibration.program_tracking.extract_first_shot_ice import NativeProfiles
from calibration.program_tracking.native_ice_line_continuity import thin_material_width_limit,straight_guide
from package_s02_repaired import source_trace


def test_other_shot_parallel_references_are_not_limited_to_s01_six_lines():
    from calibration.program_tracking.extract_first_shot_ice import Extractor
    def feature(cid,y,m):
        q=np.array([[100.,y+m*(100-960)],[700.,y+m*(700-960)],[1300.,y+m*(1300-960)],[1900.,y+m*(1900-960)]])
        return dict(target_id='ice.'+cid,world_curve=cid,point_count=4,source_line_fit=dict(
            point_origin=[960.,y],unit_tangent=[1.,m],unit_normal=[-m,1.],normal_offset_coefficients=[0.,0.],
            direct_readouts=[dict(uv=p.tolist())for p in q],actual_native_readout_count=4,residual_max_px=0.))
    features=[feature('lane1:edge0',600.,.2),feature('lane1:edge2',800.,.4)]
    reader=SimpleNamespace(paths=[],source_parallel_reference_ids={'lane1:edge0','lane1:edge2'},
        current_native_parallel_direction=lambda pt:(.0003,.12),line=lambda *a,**kw:(dict(point_count=0),[],None))
    checked,rejected,proof=Extractor.family_audit(reader,1335296,features,{},None)
    assert checked==features and rejected==[]
    assert proof['reference_origin']=='current_shot_photographic_parallel_line_family'
    assert set(proof['reference_target_ids'])=={'ice.lane1:edge0','ice.lane1:edge2'}
    assert np.allclose(proof['slope_vs_y960'],[.001,-.4])


def test_longitudinal_width_does_not_reject_a_blurred_transverse_end():
    native=SimpleNamespace(_thin_paint_width_reference=[3.]*8,_thin_paint_width_reference_axis_xy=np.array([1.,0.]))
    base=dict(feature_id='lane3',operator='paint_center',curve_id='lane3:edge1',xyz=[[0.,0.,0.],[0.,1.,0.]])
    assert thin_material_width_limit(base,native)is None
    assert thin_material_width_limit(dict(base,xyz=[[0.,0.,0.],[1.,0.,0.]]),native)['maximum_width_px']==12.


def test_sparse_circle_support_never_becomes_a_cross_screen_chord():
    raw=[dict(uv=[10.829,847.5165],theta=.2),dict(uv=[1895.1464,984.9356],theta=2.7)]
    before=[r['uv'].copy()for r in raw]
    edges,_=source_trace(raw,np.zeros((1080,1920),bool),'ice_circle',True)
    assert edges==[] and [r['uv']for r in raw]==before


def test_partial_flat_circle_does_not_close_across_unseen_arc():
    raw=[dict(uv=[x,y],theta=t)for x,y,t in [(100.,100.,.3),(125.,96.,.8),(150.,100.,1.3)]]
    edges,_=source_trace(raw,np.zeros((200,200),bool),'ice_circle',True)
    assert len(edges)==2
    assert not any(np.array_equal(e,np.array([[150.,100.],[100.,100.]]))for e in edges)


def test_flat_circle_missing_turn_is_not_a_short_cross_ring_chord():
    raw=[dict(uv=q,theta=t,source_evidence=dict(query_index=i))for q,t,i in
         [([95.,82.],.28,10),([115.,78.],.32,11),([76.,18.],1.10,23),([56.,15.],1.14,24)]]
    edges,_=source_trace(raw,np.zeros((200,200),bool),'ice_circle',False)
    assert len(edges)==2
    assert not any(np.array_equal(edge,np.array([[115.,78.],[76.,18.]]))for edge in edges)


def test_finite_photographic_arc_length_is_not_a_full_circle_angle():
    raw=[dict(uv=[x,y],source_order_arc_px=s)for x,y,s in [(100.,100.,0.),(110.,96.,10.),(120.,100.,20.)]]
    edges,_=source_trace(raw,np.zeros((200,200),bool),'ice_circle',True)
    assert len(edges)==2


def test_actual_source_703744_requires_flanks_beyond_identity_window():
    out=Path('D:/CodexData/RinkSpace_Data/runs/native_points_s02_20261027')
    path=out/'source_frames/703744.png'
    if not path.exists():pytest.skip('Local original source exposure is not public')
    import json
    v=next(r for r in json.loads((out/'evidence/visibility.json').read_text(encoding='utf8'))['frames']if r['pts']==703744)
    image=cv2.imread(str(path));mask=cv2.imread(v['mask_path'],0)>0
    frame=NativeProfiles(image[:,:,::-1],dict(source_video_sha256=v['source_video_sha256'],native_pts=703744,time_base='1/12800'),mask)
    p=dict(feature_id='lane3',curve_id='lane3:edge1',operator='paint_center',material_signal='dark',material_transition=None,_collect_line_candidates=True,_minimum_contrast=1.6)
    guide=straight_guide([[936.96,950.24],[1047.04,1078.24]],[[936.96,950.24],[1047.04,1078.24]])
    old,_=frame.sample(p,guide,radius=18,spacing=35)
    current,_=frame.sample(dict(p,_separate_profile_support=True),guide,radius=18,spacing=35)
    assert not old and len(current)>=3
    for r in current:
        e=r['source_evidence']
        assert abs(e['native_profile_shift_px'])<=14
        assert e['center_search_radius_px']==18 and e['material_profile_support_radius_px']>18
        assert e['actual_source_support_outside_person_and_broadcast_masks']


def test_historical_blue_alias_requires_the_current_physical_contour():
    from rinkspace.vision_native import NativeSourceGeometry
    geometry=object.__new__(NativeSourceGeometry)
    path=dict(id='existing',measurement='contour',role='physical_half_ellipse_boundary')
    geometry.paths={'ice.goal_crease_right':path}
    assert geometry.lookup('ice.goal_crease_right:boundary')is path
    assert geometry.lookup('ice.A1:outer')is None
    path['measurement']='centreline'
    assert geometry.lookup('ice.goal_crease_right:boundary')is None


def test_historical_blue_material_definition_is_not_a_generic_contour_alias():
    from ice_points_s04 import qualify
    row=dict(source_row_id='real_blue',uv=[100.,100.],definition='filled_blue_goal_area_material_boundary',
             pixel_support=dict(material_signal='blue'))
    path=dict(measurement='contour',role='physical_half_ellipse_boundary')
    graphics=np.zeros((1080,1920),bool);border=np.zeros(1920);distance=np.full((1080,1920),100.)
    assert qualify(row,path,{}, {},graphics,border,distance)is None
    assert qualify(row,dict(path,role='target_ring'),{}, {},graphics,border,distance)=='current_native_contour_semantics_not_proven'
    assert qualify(row,path,{}, {'real_blue':{}},graphics,border,distance)=='later_identity_withdrawal'
