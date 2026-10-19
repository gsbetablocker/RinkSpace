from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.program_tracking.strict_model_scope import file_sha
from calibration.program_tracking.strict_source_representatives import (
    coarsen_measured_path, compile_strict_representatives, representative_rows)


def test_dense_straight_path_retains_actual_endpoints_and_mass():
    uv=np.c_[np.arange(101.),np.zeros(101)]
    result=coarsen_measured_path(uv,[list(range(101))],quantization_step_px=1.)
    assert np.array_equal(result['kept_indices'],[0,100])
    assert result['original_mass'].sum()==result['retained_mass'].sum()==100.
    assert result['maximum_chord_deviation_px']<1e-12
    assert np.array_equal(uv[result['kept_indices']],[[0.,0.],[100.,0.]])


def test_turns_protected_anchors_and_visibility_gaps_remain():
    uv=np.array([[0.,0.],[1.,0.],[2.,0.],[2.,1.],[2.,2.],
        [100.,0.],[101.,0.],[102.,0.]])
    result=coarsen_measured_path(uv,[[0,1,2,3,4],[5,6,7]],
        quantization_step_px=1.,protected=[1])
    assert {0,1,2,4,5,7}.issubset(result['kept_indices'])
    assert result['retained_mass'].sum()==6.
    assert set(result['original_to_retained'][:5]).issubset({0,1,2,4})
    assert set(result['original_to_retained'][5:]).issubset({5,7})


def test_curve_shape_is_bounded_by_own_quantization_not_blur():
    angle=np.linspace(0,np.pi,151)
    uv=np.c_[100*np.cos(angle),100*np.sin(angle)]
    result=coarsen_measured_path(uv,[list(range(151))],quantization_step_px=.25)
    assert 2<len(result['kept_indices'])<151
    assert result['maximum_chord_deviation_px']<=.125
    with pytest.raises(ValueError,match='quantization'):
        coarsen_measured_path(uv,[list(range(151))],quantization_step_px=np.nan)
    with pytest.raises(ValueError,match='disjoint'):
        coarsen_measured_path(uv,[[0,1],[1,2]],quantization_step_px=1.)


def make_pack(tmp_path):
    path=tmp_path/'authenticated_source.npz';path.write_bytes(b'immutable source pilot')
    uv=np.c_[np.arange(15.),np.zeros(15)]
    rows=[]
    for actual,split in ((100,'fit'),(200,'fit'),(300,'held')):
        rows.append(dict(actual_pts=actual,shot_id='cut1',split=split,scope='ice',
            physical_material_id='C:segment0',point_count=5,
            final_current_RGB_pixel_precision_checked=True,
            source_query_indices=list(range(5)),
            source_readout_uncertainty=dict(quantization_step_px=1.)))
    sigma=np.ones(15);sigma[7]=np.nan
    pack=SimpleNamespace(rows=rows,values=dict(row_offsets=np.array([0,5,10,15]),
        uv=uv,sigma_px=sigma,source_sides_uv=np.repeat(uv[:,None],2,axis=1),
        raw_candidate_uv=uv.copy(),source_side_quantiles_uv=np.full((15,2,3,2),np.nan),
        original_point_indices=np.arange(15)))
    return pack,dict(path=str(path),sha256=file_sha(path))


def test_actual_frames_held_and_missing_support_are_separate(tmp_path):
    pack,ref=make_pack(tmp_path)
    plan=compile_strict_representatives(pack,source_reference=ref,protected_points={0:[2]})
    assert plan['fit_points_before']==10 and plan['fit_points_after']==7
    assert plan['held_points']==5 and plan['actual_exposures_with_finite_fit_after']==2
    assert np.array_equal(plan['pending_support_global_point_indices'],[7])
    assert not plan['time_frames_subsampled'] and not plan['solver_admission_verified']
    rows=representative_rows(pack,plan)
    assert {row['actual_pts'] for row in rows if row['split']=='fit'}=={100,200}
    assert np.array_equal(rows[-1]['uv'],pack.values['uv'][10:15])
    assert np.isnan(pack.values['sigma_px'][7])
    assert all(np.isclose(row['source_support_integration_weights'].sum(),1.)
        for row in rows if row['split']=='fit')


def test_no_adjacency_does_not_guess_visible_path(tmp_path):
    pack,ref=make_pack(tmp_path)
    pack.rows[0]['final_current_RGB_pixel_precision_checked']=False
    pack.rows[1]['source_readout_uncertainty']={}
    plan=compile_strict_representatives(pack,source_reference=ref)
    assert plan['fit_points_after']==9
    assert plan['records'][1]['source_shape_tolerance_px'] is None


def test_current_readout_quantiles_follow_selected_exact_points(tmp_path):
    pack,ref=make_pack(tmp_path)
    quantiles=np.arange(60.).reshape(5,2,3,2)
    pack.rows[0]['source_readout_uncertainty']['actual_side_gradient_quantiles_uv']=quantiles
    plan=compile_strict_representatives(pack,source_reference=ref)
    rows=representative_rows(pack,plan)
    assert np.array_equal(rows[0]['source_readout_uncertainty']['actual_side_gradient_quantiles_uv'],quantiles[[0,4]])
    assert np.array_equal(pack.rows[0]['source_readout_uncertainty']['actual_side_gradient_quantiles_uv'],quantiles)


def test_tampered_held_weights_or_solver_flags_cannot_pass(tmp_path):
    pack,ref=make_pack(tmp_path)
    plan=compile_strict_representatives(pack,source_reference=ref)
    bad=deepcopy(plan);bad['records'][-1]['kept_indices']=[0,4]
    with pytest.raises(ValueError,match='HELD'):
        representative_rows(pack,bad)
    bad=deepcopy(plan);bad['records'][0]['point_integration_weights']=[1.,1.]
    with pytest.raises(ValueError,match='budget'):
        representative_rows(pack,bad)
    bad=deepcopy(plan);bad['solver_admission_verified']=True
    with pytest.raises(ValueError,match='authorize'):
        representative_rows(pack,bad)
    (tmp_path/'authenticated_source.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):
        representative_rows(pack,plan)
