"""Saved model diagnostics on exact PTS; no GUI, fitting or source inference."""
import json

import numpy as np
import pytest

from rinkspace.viewer3d.blade_review import BladeReview,diagnostic_lines,overlay_primitives


def review_asset(tmp_path, **extra):
    values=dict(pts=np.array([1000,1256],dtype=np.int64),
        test_blade_uv=np.arange(24,dtype=float).reshape(2,2,3,2),
        test_boot_uv=np.arange(16,dtype=float).reshape(2,2,2,2),
        test_axis_rate_rad_s=np.array([[np.nan,np.nan],[.5,.75]]))
    values.update(extra)
    np.savez_compressed(tmp_path/'projection.npz',**values)
    document=dict(schema='rinkspace.blade-player-review.v1',projection_npz='projection.npz',
        variants={'candidate':'test'},time_base=[1,12800],
        source_visibility=[dict(pts=1256,foot=0,blade_axis_observable=False,
            contact_state='unknown',material_state='unknown',orientation_unknown_reason='接触/材料 unknown')])
    path=tmp_path/'review.json';path.write_text(json.dumps(document),encoding='utf8')
    return BladeReview(path,time_base=(1,12800))


def model_diagnostics():
    return dict(test_pitch_degrees=np.array([[1.,2.],[3.,4.]]),
        test_roll_degrees=np.array([[5.,6.],[7.,8.]]),
        test_runner_min_x_fraction=np.array([[.1,.2],[.3,.4]]),
        test_station_height_m=np.array([[[.01,.02,.03],[.04,.05,.06]],[[.07,.08,.09],[.10,.11,.12]]]),
        test_runner_minus_pick_m=np.array([[-.01,.02],[-.03,.04]]))


def test_old_asset_keeps_original_fields_and_unknown_text(tmp_path):
    review=review_asset(tmp_path);current=review.at('candidate',1256)
    assert set(current['model'])=={'blade_uv','boot_uv','axis_rate_rad_s'}
    lines=diagnostic_lines(current)
    assert len(lines)==4
    assert '原片刀轴：unknown' in lines[1]
    assert '原片刀轴：unknown' in lines[3]
    assert not any('刀纵倾' in line or '原始参考面高度' in line for line in lines)


def test_optional_model_arrays_keep_exact_pts_and_source_unknown(tmp_path):
    values=model_diagnostics();review=review_asset(tmp_path,**values)
    current=review.at('candidate',1256)
    for key,value in values.items():
        np.testing.assert_array_equal(current['model'][key.removeprefix('test_')],value[1])
    missing=review.at('candidate',1255)
    assert missing['model'] is None and missing['source_visibility']==[]
    assert not any('刀纵倾' in line for line in diagnostic_lines(missing))
    source=current['source_visibility'][0]
    assert source['contact_state']==source['material_state']=='unknown'
    lines=diagnostic_lines(current)
    text='\n'.join(lines)
    assert '刀纵倾 3.00°' in text and '刀侧倾 7.00°' in text
    assert 'runner最低点 0.300（0后1前）' in text
    assert 'rear/mid/front 7.00/8.00/9.00 cm' in text
    assert 'runner-pick相对高度 -3.00 cm' in text
    assert '原始诊断，非真实穿冰判据' in text
    assert '原片刀轴：unknown' in text and '接触/材料 unknown' in text
    # New model diagnostics do not alter native projection primitives.
    old=dict(current,model={key:value for key,value in current['model'].items()
                           if key in ('blade_uv','boot_uv','axis_rate_rad_s')})
    before,after=overlay_primitives(old),overlay_primitives(current)
    assert len(before)==len(after)
    for a,b in zip(before,after):
        assert a.keys()==b.keys()
        for key in a:
            if key=='points':np.testing.assert_array_equal(a[key],b[key])
            else:assert a[key]==b[key]


def test_optional_arrays_are_independent(tmp_path):
    review=review_asset(tmp_path,test_roll_degrees=np.full((2,2),12.))
    text='\n'.join(diagnostic_lines(review.at('candidate',1256)))
    assert '刀侧倾 12.00°' in text
    assert '刀纵倾' not in text and '原始参考面高度' not in text


@pytest.mark.parametrize('key',model_diagnostics())
def test_bad_optional_shape_is_rejected(tmp_path,key):
    with pytest.raises(ValueError,match=key):
        review_asset(tmp_path,**{key:np.zeros((2,))})


@pytest.mark.parametrize('value',[np.nan,np.inf])
def test_nonfinite_optional_model_diagnostic_is_rejected(tmp_path,value):
    with pytest.raises(ValueError,match='test_pitch_degrees'):
        review_asset(tmp_path,test_pitch_degrees=np.full((2,2),value))
