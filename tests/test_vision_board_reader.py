"""Physical exclusion must survive the material-specific reread entry."""
from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from rinkspace.vision_board_reader import refine_board_readings


def material_case(kind='board_top', mask=None, guide_updates=None):
    rgb = np.full((140, 200, 3), 160, np.uint8)
    rgb[42:100] = 225
    rgb[30:42] = [45, 65, 130]
    rgb[100:108] = [220, 190, 80]
    binding = dict(source_sha256='a'*64, source_pts=8, time_base='1/15360', native_hw=[140, 200], shot_id='probe')
    uv = np.column_stack((np.arange(15, 185, 5), np.full(34, 32 if kind == 'board_top' else 105))).astype(np.float32)
    guide = dict(id=kind+'.test', kind=kind, source_board_view='front_face', points_uv=uv)
    if mask is not None:
        guide.update(source_exclusion_mask=mask, source_exclusion_binding=binding)
    guide.update(guide_updates or {})
    row = dict(actual_source_uv=torch.empty((0, 2)), metadata={}, expected_id=guide['id'])
    authority = dict(source_sha256='a'*64, reference={'synthetic_test': True}, kind='direct_user_confirmation',
                     board_foot_material='warm_yellow_strip_belongs_to_board')
    return rgb, binding, guide, row, authority


def reread(case):
    rgb, binding, guide, row, authority = case
    result = refine_board_readings(rgb, binding, [guide], dict(readings=[row]), authority=authority,
        feedback=SimpleNamespace(material_authority=lambda a: a), prediction=None, classes=(), device='cpu')
    return row, result


@pytest.mark.parametrize('kind', ['board_top', 'ice_contact'])
def test_masked_material_reread_cannot_restore_rejected_points(kind):
    baseline, _ = reread(material_case(kind))
    assert len(baseline['actual_source_uv']) == 34
    row, packet = reread(material_case(kind, np.ones((140, 200), bool)))
    assert len(row['actual_source_uv']) == len(row['coherent_source_uv']) == 0
    assert row['measured_runs'] == row['coherent_measured_runs'] == []
    assert not row['source_quality_mask'].any()
    assert row['flags']['source_exclusion_conflict'].all()
    assert len(row['source_excluded_material_candidates']['actual_source_uv']) == 34
    assert packet['material_packets'][kind]['source_exclusion_admission'][0]['excluded_count'] == 34


@pytest.mark.parametrize('kind', ['board_top', 'ice_contact'])
def test_partial_mask_breaks_material_runs_and_preserves_visible_measurements(kind):
    baseline, _ = reread(material_case(kind))
    assert len(baseline['actual_source_uv']) == 34
    mask = np.zeros((140, 200), bool)
    mask[:, 80:105] = True
    row, _ = reread(material_case(kind, mask))
    expected = (baseline['actual_source_uv'][:, 0] < 80) | (baseline['actual_source_uv'][:, 0] >= 105)
    torch.testing.assert_close(row['actual_source_uv'], baseline['actual_source_uv'][expected])
    assert [len(r['actual_source_uv']) for r in row['measured_runs']] == [13, 16]
    assert not row['source_quality_mask'][13:18].any()


@pytest.mark.parametrize('kind', ['board_top', 'ice_contact'])
@pytest.mark.parametrize('masked_part', ['query', 'output', 'material_interior'])
def test_navigation_output_and_material_support_are_each_checked(kind, masked_part):
    baseline, _ = reread(material_case(kind))
    assert len(baseline['actual_source_uv']) == 34
    mask = np.zeros((140, 200), bool)
    if masked_part == 'query':
        y = 32 if kind == 'board_top' else 105
    elif masked_part == 'output':
        y = int(np.floor(float(baseline['actual_source_uv'][0, 1])))
    else:
        y = 35 if kind == 'board_top' else 103
    mask[y, :] = True
    row, _ = reread(material_case(kind, mask))
    assert len(row['actual_source_uv']) == 0
    if masked_part == 'material_interior':
        evidence = row['source_excluded_material_candidates']
        assert evidence['material_support_excluded'].all()
        assert not evidence['query_excluded'].any() and not evidence['output_excluded'].any()


def test_mask_remains_specific_to_its_guide_and_flow_support_is_preserved():
    case = material_case(mask=np.ones((140, 200), bool))
    rgb, binding, guide, row, authority = case
    guide['points_uv'] = guide['points_uv'][:17]
    other = dict(id='board_top.visible', kind='board_top', source_board_view='front_face',
        points_uv=material_case()[2]['points_uv'][17:], temporal_flow_supported=np.r_[np.ones(8, bool), np.zeros(9, bool)])
    visible = dict(actual_source_uv=torch.empty((0, 2)), metadata={})
    refine_board_readings(rgb, binding, [guide, other], dict(readings=[row, visible]), authority=authority,
        feedback=SimpleNamespace(material_authority=lambda a: a), prediction=None, classes=(), device='cpu')
    assert len(row['actual_source_uv']) == 0
    assert len(visible['actual_source_uv']) == 8
    assert visible['source_indices'].tolist() == list(range(8))


@pytest.mark.parametrize('verified', [True,False])
def test_current_paired_face_extends_only_verified_runs_and_rereads_rgb(verified):
    import hashlib
    from rinkspace.vision_board_reader import read_connected_board_gaps
    rgb,binding,_,_,authority=material_case()
    points=np.c_[np.arange(15.,185.,5),np.full(34,32.)]
    bottom=points.copy();bottom[:,1]=105.
    scene=dict(path='current_native',sha256='b'*64)
    def lookup(name):
        if 'top' in name:return dict(id='board/continuous/top',layer='board',label='board.top')
        if 'bottom' in name or 'foot' in name:return dict(id='board/continuous/bottom',layer='board',label='board.bottom')
    def bind(items):
        return [dict(g,primitive='free_curve',physical_role='board_top_edge' if g['kind']=='board_top' else 'ice_board_contact',
            native_geometry=dict(path_id=g['id'],scene=scene)) for g in items],dict(unresolved=[])
    geometry=SimpleNamespace(lookup=lookup,bind=bind)
    guides=[];rows=[]
    for name,kind,uv in (('top','board_top',points),('foot','ice_contact',bottom)):
        guides.append(dict(id=name,kind=kind,points_uv=uv[:2].copy(),source_binding=binding,
            source_board_view='front_face',identity_confirmed=verified,
            identity_authority={'kind':'actual_source_identity'} if verified else None))
        rows.append(dict(expected_id=name,physical_id=name if verified else None,
            actual_source_uv=torch.tensor(uv[:2]),source_indices=torch.arange(2)))
    originals=[r['actual_source_uv'].clone() for r in rows]
    excluded=np.zeros(rgb.shape[:2],bool);excluded[:,80:105]=True
    packet=dict(source=dict(binding,decoded_BGR_sha256=hashlib.sha256(rgb[:,:,::-1].copy().tobytes()).hexdigest()),
        boundary_candidates=[dict(candidate_id='paired_run',blue_upper_uv=points,yellow_lower_uv=bottom)])
    face=dict(regions=[dict(candidate_id='paired_run',named_source_boundaries=dict(board_top=['top'],ice_contact=['foot']))])
    result=read_connected_board_gaps(rgb,binding,guides,dict(readings=rows),source_packet=packet,
        face_receipt=face,geometry=geometry,authority=authority,
        feedback=SimpleNamespace(material_authority=lambda a:a),prediction=None,classes=(),device='cpu',exclusion=excluded)
    for old,row in zip(originals,rows):torch.testing.assert_close(old,row['actual_source_uv'])
    if not verified:
        assert result['added_query_count']==0 and len(rows)==2
        return
    assert result['added_query_count']==2 and result['added_actual_point_count']>0
    for row in rows[2:]:
        uv=row['actual_source_uv']
        assert not ((uv[:,0]>=80)&(uv[:,0]<105)).any()
        # Search positions y=32/105 are not copied as measured interfaces.
        assert not torch.allclose(uv[:,1],torch.full_like(uv[:,1],32. if row['kind']=='board_top' else 105.))
        assert row['identity_authority']['human_pixel_acceptance'] is False


@pytest.mark.parametrize('invalid', ['missing_binding', 'different_pts', 'wrong_shape', 'not_boolean'])
def test_material_exclusion_requires_current_native_frame(invalid):
    case = material_case(mask=torch.ones((1, 1, 140, 200), dtype=torch.bool))
    guide = case[2]
    if invalid == 'missing_binding':
        del guide['source_exclusion_binding']
    elif invalid == 'different_pts':
        guide['source_exclusion_binding'] = {**case[1], 'source_pts': 9}
    elif invalid == 'wrong_shape':
        guide['source_exclusion_mask'] = np.ones((70, 100), bool)
    else:
        guide['source_exclusion_mask'] = np.ones((140, 200), np.uint8)
    with pytest.raises(ValueError):
        reread(case)


def test_real_grey_deck_is_excluded_through_advertisement_reread_entry():
    import hashlib
    av = pytest.importorskip('av')
    from rinkspace.vision_review import reread_source_advertisements
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/'configs/rink_vision.json').read_text(encoding='utf-8-sig'))
    ref = cfg['near_board_source_evidence']['gray_history']
    if not Path(ref['path']).is_file() or not Path(cfg['source_video']).is_file():
        pytest.skip('Historical native source evidence not installed')
    history = json.loads(Path(ref['path']).read_text(encoding='utf-8-sig'))
    pts = 230416128
    with av.open(cfg['source_video']) as video:
        stream = video.streams.video[0]
        video.seek(pts, stream=stream, backward=True)
        frame = next(f for f in video.decode(stream) if f.pts >= pts)
        assert frame.pts == pts
        rgb = frame.to_ndarray(format='rgb24')
    evidence = next(r for r in history['frames'] if r['native_pts'] == pts)
    assert hashlib.sha256(np.ascontiguousarray(rgb[..., ::-1]).tobytes()).hexdigest() == evidence['source_pixel_sha256']
    binding = dict(source_sha256=history['provenance']['source_video_sha256'], source_pts=pts,
        time_base=history['provenance']['native_time_base'], shot_id='historical_grey', native_hw=list(rgb.shape[:2]))
    uv = np.array([[149., 900.]], np.float32)
    packet = dict(source_binding=binding, instances=[dict(kind='advertisement_panel',
        region_mask=np.ones(rgb.shape[:2], bool), boundary_readings=[dict(actual_source_uv=uv, source_indices=np.array([0]))])])
    result = reread_source_advertisements(rgb, packet, SimpleNamespace(reread_rgb=True),
        near_board_evidence=cfg["near_board_source_evidence"])
    assert result['near_board_ad_exclusion']['exclusion_pixel_count'] == 112313
    assert len(result['instances'][0]['boundary_readings'][0]['actual_source_uv']) == 0
    assert not result['instances'][0]['region_mask'][900, 149]


def test_no_observed_blue_band_returns_empty_without_reviving_generic_points():
    case = material_case(mask=np.zeros((140, 200), bool))
    case[0][30:42] = 160
    row, _ = reread(case)
    assert len(row['actual_source_uv']) == 0
    assert row['measured_runs'] == []
    assert not row['source_quality_mask'].any()
