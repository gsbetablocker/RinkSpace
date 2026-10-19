"""A local derivative chart cannot veto a feasible original curve witness."""
from types import SimpleNamespace
import numpy as np
import pytest
from calibration.solver_runtime import install
install()
from structured_programme_source_candidate.source_step_curvature_tightening import (
    measured_outward_step_tightening,dynamic_source_group_merits)
from calibration.solver_runtime.optional_soc_runtime import original_nonlinear_step_face_feasibility
from calibration.solver_runtime.native_truth_nominal_chart import (
    capture_native_nominal_query_chart,native_nominal_query_carriers)


def response(base,groups,targets):
    base=np.asarray(base,float);groups=np.asarray(groups,int)
    return dict(row_start=0,row_stop=2*len(base),nominal_carrier=base.copy(),
        retained_face_mask=np.ones(2*len(base),bool),dynamic_group_limits=np.asarray(targets,float),
        dynamic_row_group_index=groups.copy(),dynamic_guard_comparison_tolerance_px=1e-8)


def test_saved_F2_witness_switch_is_not_an_outward_guard_violation():
    # Native38 vertex910, saved q276 full direction. The old chart's tangent
    # exceeds its source band although another real curve witness is feasible.
    base=np.array([2.8002537481086724,7.259050246838736])
    frozen=np.array([2.804946849992254,7.277501069339699])
    dynamic=np.array([2.3914218409489685,-.1334395708212763])
    lower=np.array([-4.,-7.277469784851016]);upper=-lower
    bounds=np.c_[lower,upper];before=bounds.copy()
    J=(frozen-base).reshape(2,1);A=np.vstack((J,-J));b=np.r_[lower-base,base-upper]
    metadata={'native':response(base,[0,0,0,0],[0.])}
    historical=measured_outward_step_tightening(A,b,[1.],metadata,{'native':frozen})
    assert historical is not None
    assert 'actual_original_violation_px'in historical['report']['compensated_faces'][0]
    assert np.all(abs(dynamic)<upper)
    actual={'native':np.array([0.])}
    assert measured_outward_step_tightening(A,b,[1.],metadata,{'native':frozen},actual_group_violations=actual)is None
    merit=original_nonlinear_step_face_feasibility(A,b,[1.],metadata,{'native':frozen},actual_group_violations=actual)
    assert merit['maximum_violation_px']==merit['total_violation_px']==0.
    assert merit['merit_basis']=='original_dynamic_group_excess_above_same_parent_priority_target'
    np.testing.assert_array_equal(bounds,before)


def test_actual_violating_group_still_gets_same_chart_curvature_cut_without_cap_growth():
    A=np.array([[1.],[-1.]]);b=np.array([2.8,-100.]);step=np.array([2.8])
    old=b.copy();target=np.array([0.]);base=np.array([0.])
    metadata={'native':response(base,[0,0],target)}
    result=measured_outward_step_tightening(A,b,step,metadata,{'native':step-.1*step**2},
        actual_group_violations={'native':np.array([.9])})
    assert result is not None
    assert result['lower'][0]==pytest.approx(2.8+.784+1e-4)
    assert result['lower'][1]==b[1]
    assert result['report']['original_dynamic_group_guard_gates_local_chart_cuts']
    detail=result['report']['compensated_faces'][0]
    assert detail['local_chart_face_violation_px']==pytest.approx(.784)
    assert detail['original_source_group_index']==0
    assert detail['actual_original_group_violation_px']==pytest.approx(.9)
    assert detail['group_target_px']==0.
    assert detail['actual_original_group_excess_px']==pytest.approx(.9)
    assert 'actual_original_violation_px'not in detail
    np.testing.assert_array_equal(b,old);np.testing.assert_array_equal(target,[0.])


def test_actual_and_predicted_progress_use_groups_not_number_of_FM_faces():
    A=np.ones((4,1));b=np.full(4,.2)
    metadata={'native':response([0.,0.],[0,0,0,0],[0.])}
    merit=dynamic_source_group_merits(A,b,[0.],metadata,{'native':np.array([.3])})
    assert merit['total_violation_px']==pytest.approx(.3)
    assert merit['predicted_total_violation_px']==pytest.approx(.2)
    assert merit['complete_original_source_groups']==1


def test_existing_nonzero_restoration_cap_is_kept_by_actual_gate():
    A=np.array([[1.],[-1.]]);b=np.array([.3,-100.])
    metadata={'native':response([0.],[0,0],[.5])}
    # This group has an original restoration cap. It must not be forced to
    # zero merely because the local frozen chart predicts an excess.
    assert measured_outward_step_tightening(A,b,[0.],metadata,{'native':np.array([-.4])},
        actual_group_violations={'native':np.array([.4])})is None
    np.testing.assert_array_equal(metadata['native']['dynamic_group_limits'],[.5])


def test_invalid_dynamic_group_ownership_is_rejected():
    metadata={'native':response([0.],[0,2],[0.])}
    with pytest.raises(ValueError,match='row ownership'):
        dynamic_source_group_merits(np.ones((2,1)),np.zeros(2),[0.],metadata,{'native':np.array([0.])})


def test_nonFM_replay_uses_same_four_frozen_query_formulas_as_native_J(monkeypatch):
    from structured_programme_source_candidate import human_stroke_truth as query_module
    from structured_programme_source_candidate import native_camera_local as camera_module
    fields=dict(nominal_finite_segments=np.array([3]),nominal_finite_segment_categories=np.array([0]),
        nominal_source_support_faces=np.array([5]),nominal_source_station_branches=np.array([2]))
    record=dict(row={'actual_pts':99,'curve':{'kind':'house_ring'}},uv=np.zeros((1,2)),
        normals=np.array([[1.,0.]]),bounds=np.array([[-1.,1.]]),tangent_bounds=np.array([[-1.,1.]]),
        factor=SimpleNamespace(_world=lambda q,curve:np.array([[0.,0.,1.],[1.,0.,1.]])))
    block=SimpleNamespace(bundle=SimpleNamespace(camera=lambda q,p:q.copy()))
    validations=[]
    truth=SimpleNamespace(parameter_count=1,height_columns=np.empty(0,int),world_indices=np.array([0]),
        full_stroke_records=[record],blocks={99:block},state=SimpleNamespace(
            geometry=SimpleNamespace(validate_parameters=lambda q:validations.append(q.copy())),
            structured_programme_coordinate_chart=SimpleNamespace(core_count=1)),
        queries=lambda q,**kw:[dict(query=dict(fields,signed_normal_px=np.array([.2]),coverage_tangent_px=np.array([.3])))])
    captured=[]
    def query(camera,world,uv,normals,bounds,**kw):
        captured.append(kw)
        return dict(signed_normal_px=np.array([.2+camera[0]]),coverage_tangent_px=np.array([.3]))
    monkeypatch.setattr(query_module,'finite_stroke_queries',query)
    monkeypatch.setattr(camera_module,'native_camera_local',lambda b,q:q)
    chart=capture_native_nominal_query_chart(truth,np.array([0.]))
    values=native_nominal_query_carriers(truth,np.array([.4]),chart)
    np.testing.assert_allclose(values,[.6,.3]);assert len(validations)==1
    for field,frozen in [('nominal_finite_segments','frozen_segments'),('nominal_finite_segment_categories','frozen_segment_categories'),
                         ('nominal_source_support_faces','frozen_support_faces'),('nominal_source_station_branches','frozen_station_branches')]:
        np.testing.assert_array_equal(captured[0][frozen],fields[field])
    np.testing.assert_array_equal(record['bounds'],[[-1.,1.]])
