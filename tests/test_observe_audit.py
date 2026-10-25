import numpy as np
import pytest
from rinkspace.session.observe import measure, _pixel_requests
from rinkspace.vision_source_shapes import _MaterialProfileFrame


def frame(rgb):
    return _MaterialProfileFrame(rgb, {'source_sha256':'a'*64,'source_pts':17,
        'time_base':'1/12800','native_hw':list(rgb.shape[:2]),'shot_id':'fixture'})


def request():
    return {'id':'stripe','roi_xyxy':[0,10,120,110],
        'measurement':'paint_center','material_signal':'dark'}


CONFIG={'scan_axis':'y','sample_step_px':24,'sampling_inset_px':8}


def test_blank_frame_emits_missing_without_fake_uv():
    result=measure(frame(np.full((120,120,3),220,np.uint8)),[request()],CONFIG)
    assert result[0]['queries']
    assert all(q['status']=='no_pixel_support' and not q['candidates'] for q in result[0]['queries'])


def test_material_center_comes_from_both_measured_interfaces():
    rgb=np.full((120,120,3),220,np.uint8);rgb[46:55]=30
    result=measure(frame(rgb),[request()],CONFIG)
    for query in result[0]['queries']:
        assert query['status']=='candidate_not_reviewed'
        candidate=query['candidates'][0]
        np.testing.assert_allclose(candidate['uv_or_curve'],np.mean(candidate['actual_sides_uv'],axis=0))
        assert abs(candidate['uv_or_curve'][1]-50)<.1
        assert candidate['actual_sides_uv'][0][1]<46 and candidate['actual_sides_uv'][1][1]>54
        assert candidate['active_for_fit'] is False


def test_competing_material_is_not_silently_selected():
    rgb=np.full((120,120,3),220,np.uint8);rgb[35:44]=30;rgb[70:79]=30
    result=measure(frame(rgb),[request()],CONFIG)
    assert all(q['status']=='ambiguous_material' and len(q['candidates'])==2 for q in result[0]['queries'])


def test_pixel_stage_strips_reference_and_identity_metadata():
    original={**request(),'entity_candidates':['known_identity'],'navigation_origin':{'old_uv':[12,34]}}
    pixels=_pixel_requests([original]);assert pixels==[request()]
    with pytest.raises(ValueError,match='reference/identity'):
        measure(frame(np.full((120,120,3),220,np.uint8)),[original],CONFIG)


@pytest.mark.parametrize('roi',[[0,0,120,120],[0,10,120,15],[-1,10,120,110]])
def test_incomplete_source_support_is_rejected(roi):
    req={**request(),'roi_xyxy':roi}
    with pytest.raises(ValueError,match='broad source ROI'):
        measure(frame(np.full((120,120,3),220,np.uint8)),[req],CONFIG)
