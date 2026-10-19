"""Final native support determines line means after source branch selection."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from calibration.program_tracking import extract_first_shot_ice as extraction


def _path(feature='synthetic'):
    return dict(feature_id=feature,curve_id=feature+':segment0',world_curve_role='segment0',
                operator='paint_center',material_signal='red' if feature=='C' else 'dark',
                material_transition=None,_circle=False)


def _native_rows(values, signals=None):
    values=np.asarray(values,float)
    if values.ndim==1:values=values[:,None]
    signals=np.full(values.shape,10.) if signals is None else np.asarray(signals,float)
    origin=np.array([80.,650.]);rows=[]
    for qi in range(len(values)):
        for branch in range(values.shape[1]):
            center=origin+[50.+100.*qi,values[qi,branch]]
            rows.append(dict(uv=center.tolist(),sigma_px=1.,source_evidence=dict(
                query_index=qi,profile_normal=[0.,1.],
                signed_interface_iqr_contrast=[float(signals[qi,branch])]*2,
                actual_source_sides_uv=[(center-[0,2]).tolist(),(center+[0,2]).tolist()])))
    return rows


def _extract(rows,feature='synthetic',family_slope=None):
    native=SimpleNamespace(mask=np.zeros((1080,1920),bool),
        frame=SimpleNamespace(binding=dict(source_pts=123456)),
        sample=lambda *args,**kwargs:(deepcopy(rows),dict(query_count=len(rows))))
    guide=np.array([[80.,650.],[1080.,650.]])
    return extraction.Extractor.line(None,_path(feature),guide,native,family_slope=family_slope)[0]


def _assert_fixed_support_mean(feature):
    fit=feature['source_line_fit'];rows=fit['direct_readouts']
    origin=np.asarray(fit['point_origin']);v=np.asarray(fit['unit_tangent']);n=np.asarray(fit['unit_normal'])
    points=np.asarray([r['uv'] for r in rows]);t=(points-origin)@v;d=(points-origin)@n
    broad=feature['world_curve'].split(':')[0] in ('B1','B2','C')
    weight=np.asarray([min(r['source_evidence']['signed_interface_iqr_contrast'])**2 for r in rows]) if broad else np.ones(len(rows))
    X=np.c_[t,np.ones(len(t))]
    D=np.linalg.solve(X.T@(weight[:,None]*X),X.T*weight)
    np.testing.assert_allclose(fit['normal_offset_coefficients'],D@d,rtol=1e-11,atol=1e-11)
    expected_residual=d-X@np.asarray(fit['normal_offset_coefficients'])
    assert fit['residual_max_px']==pytest.approx(float(np.max(abs(expected_residual))),abs=1e-12)
    assert fit['estimator_contract']['final_support_refit_unconstrained'] is True
    assert fit['estimator_contract']['derivative_normal_projection_factor_required'] is True
    for segment in feature['segments']:
        representatives=np.asarray(segment['uv']);station=(representatives-origin)@v
        np.testing.assert_allclose((representatives-origin)@n,np.polyval(fit['normal_offset_coefficients'],station),atol=7.2e-5)
        assert segment['representative_points_share_line_estimate'] is True
        assert segment['individual_native_pixel_readouts'] is False


def test_family_direction_selects_support_without_clipping_its_emitted_mean():
    rows=_native_rows(.010*np.arange(50.,1000.,100.));before=deepcopy(rows)
    feature=_extract(rows,family_slope=0.)
    assert feature['point_count']>0
    assert feature['source_line_fit']['parallel_family_direction_used_for_candidate_disambiguation'] is True
    # The old candidate clip was +/-7/1000; the current native support is 0.010.
    assert feature['source_line_fit']['normal_offset_coefficients'][0]==pytest.approx(.010,abs=1e-12)
    _assert_fixed_support_mean(feature)
    assert rows==before


def test_last_branch_selection_is_refit_with_its_final_weighted_support(monkeypatch):
    values=[[-2.8099800894727793,1.955349120071332],[1.0605527754407416,-3.6280251670098758],
            [-5.53737679396559,-1.9387639927579117],[-3.6312864409023904,1.691534382873538],
            [3.359974774635671,4.261589182040196],[9.56959395570048,.1464831620910747],
            [.11471987022136831,.011827903067249679],[8.768205168034234,-3.5551425866623534],
            [1.4420870534751937,6.500764795068041],[5.858502803496624,.7077941697791745]]
    signals=[[11.52833185933878,3.8145229050130216],[5.513119516916976,4.497493976172077],
             [3.080004207830365,1.2076710054634991],[1.7816068615949479,5.24483852516283],
             [3.5460943076696245,1.9790188263927164],[2.4217828076996684,1.5875667049403972],
             [2.4355446303999546,2.086184620616807],[2.173765008537808,4.235260366878411],
             [4.21764075043681,2.947322035320153],[5.433819998016455,5.579924484808623]]
    rows=_native_rows(values,signals);inputs=[];polyfit=np.polyfit
    def observe(*args,**kwargs):
        inputs.append(np.c_[args[0],args[1]].copy())
        return polyfit(*args,**kwargs)
    monkeypatch.setattr(extraction.np,'polyfit',observe)
    feature=_extract(rows,feature='C');fit=feature['source_line_fit']
    assert len(inputs)==3
    current=np.asarray([np.asarray(r['uv'])-[80.,650.] for r in fit['direct_readouts']])
    assert not np.array_equal(inputs[-1],current)
    assert [r['source_evidence']['query_index'] for r in fit['direct_readouts']]==[0,1,2,3,5,6,8,9]
    _assert_fixed_support_mean(feature)
    assert fit['estimator_contract']['weights_policy']=='minimum_two_interface_contrast_squared'


def test_saved_native_1248_support_refit_removes_actual_navigation_bound():
    path=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013/first_shot_remaining_50fps/repair_20261014/ice_candidates/001248.json')
    if not path.is_file():pytest.skip('The saved official native 1248 support is unavailable')
    before=path.read_bytes();packet=json.loads(before);feature=next(f for f in packet['features']if f['target_id']=='ice.lane1:edge0');z=feature['source_line_fit'];rows=z['direct_readouts']
    p=np.asarray([r['uv'] for r in rows]);o=np.asarray(z['point_origin']);v=np.asarray(z['unit_tangent']);n=np.asarray(z['unit_normal']);t=(p-o)@v;d=(p-o)@n
    result=extraction.final_support_line_fit(t,d,np.ones(len(rows)))
    assert z['normal_offset_coefficients'][0]==pytest.approx(-7./2159.082051438298,abs=1e-15)
    assert result[0]==pytest.approx(-.0043146736300728,abs=1e-12)
    representative=np.asarray([q for s in feature['segments']for q in s['uv']]);stations=(representative-o)@v
    assert np.max(abs(np.polyval(result-np.asarray(z['normal_offset_coefficients']),stations)))>1.
    assert path.read_bytes()==before
