import hashlib

import numpy as np

from rinkspace.vision_blue_paint_RGB_vector_scope_delta import (
    _supported_interval, append_blue_paint_scope_RGB_vector_readouts)


def test_scope_interval_keeps_bilinear_tangent_and_smoothing_halo_inside_ice():
    mask=np.zeros((200,200),bool);mask[60:]=True
    interval,proof=_supported_interval(mask,np.array([100.,100.]),np.array([0.,1.]),np.array([-70.,70.]))
    assert interval.tolist()==[-38.,68.]
    assert proof['native_scope_run']==[-40.,70.]
    unavailable,_=_supported_interval(mask,np.array([100.,40.]),np.array([0.,1.]),np.array([-30.,100.]))
    assert unavailable is None


def test_scope_successor_preserves_original_queries_and_never_uses_board_as_blue_flank():
    rgb=np.full((200,200,3),220,np.uint8)
    rgb[40:60]=[130,125,130]
    rgb[90:104]=[90,110,155]
    mask=np.zeros((200,200),bool);mask[60:]=True
    measurement=dict(source_binding=dict(source_sha256='a'*64,time_base='1/15360',
        actual_pts=1,native_pts=1,source_pts=1,shot_id='shot',native_hw=[200,200]),
        decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        query_curve=np.array([0],np.int16),query_uv=np.array([[100.,100.]],np.float64),
        query_normal=np.array([[0.,1.]],np.float64),query_interval=np.array([[-70.,70.]],np.float64),
        query_status=np.array([2],np.uint8),candidate_uv=np.empty((0,2),np.float32),
        candidate_sides=np.empty((0,2,2),np.float32),candidate_query=np.empty(0,np.int32),
        candidate_flags=np.empty(0,np.uint8),candidate_side_quantiles_uv=np.empty((0,2,3,2),np.float64))
    frozen={k:v.tobytes()for k,v in measurement.items()if isinstance(v,np.ndarray)}
    inventory=[dict(curve_id='B1:segment0',operator='paint_center',material_signal='blue')]
    result=append_blue_paint_scope_RGB_vector_readouts(rgb,measurement,np.zeros_like(mask),inventory,
        source_ice_scope_mask=mask,curve_ids=('B1:segment0',),keep_profiles=True)
    current=result['measurement']
    for name in ('curve','uv','normal','interval','status'):
        assert current['query_'+name][:1].tobytes()==frozen['query_'+name]
    for name,values in frozen.items():
        assert measurement[name].tobytes()==values
    assert current['query_interval'][1].tolist()==[-38.,68.]
    assert len(current['candidate_uv'])==1
    sides=current['candidate_side_quantiles_uv'][0,:,1]
    assert np.allclose(np.sort(sides[:,1]),[89.5,103.5],atol=.3)
    assert current['candidate_flags'][0]==8
    assert result['source_blue_RGB_vector_scope']['source_FIT_admitted']is False
