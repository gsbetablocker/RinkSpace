from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from calibration.solver_runtime import install

install()
from structured_programme_source_candidate.source_step_curvature_tightening import actual_carrier_step_response,measured_outward_step_tightening
from calibration.solver_runtime.optional_soc_runtime import original_nonlinear_step_face_feasibility
from calibration.solver_runtime import finite_rectangle_station_step_constraints as model


def test_sparse_face_map_preserves_upper_sign_and_SOC_full_original_merit():
    base=np.array([2.,3.,5.]);projection=csr_matrix([[1.,0.,0.],[-.25,-.75,0.],[0.,0.,-1.]])
    metadata=dict(nominal_carrier=base,row_carrier_projection=projection,row_start=0,row_stop=3)
    actual=base+np.array([.1,.3,.4])
    np.testing.assert_allclose(actual_carrier_step_response(metadata,actual),[.1,-.25,-.4])
    A=np.array([[1.],[-2.5],[-4.]]);lower=np.array([0.,-.2,-.3])
    report=original_nonlinear_step_face_feasibility(A,lower,np.array([.1]),{'native':metadata},{'native':actual})
    assert abs(report['maximum_violation_px']-.1)<1e-14
    assert abs(report['total_violation_px']-.15)<1e-14
    tightened=measured_outward_step_tightening(A,lower,np.array([.1]),{'native':metadata},{'native':actual})
    assert tightened is not None and len(tightened['report']['compensated_faces'])==2


def fixture():
    record=dict(uv=np.zeros((1,2)),normals=np.array([[1.,0.]]),bounds=np.array([[-1.,1.]]),
        tangent_bounds=np.array([[-1.,1.]]),row={'curve':{'kind':'current_ice_segment'}},factor=object())
    carrier=np.array([1.1,1.1]);J=np.array([[-.2],[-.2]])
    base=dict(matrix=np.vstack((J,-J)),lower=np.array([-2.1,-2.1,0.,0.]),report={},
        step_carrier_response=dict(nominal_carrier=carrier,retained_face_mask=np.ones(4,bool)))
    query=dict(finite_source_association='original_support_rectangle_Linf',nearest_finite_segment_index=np.array([0]),
        nearest_finite_segment_station=np.array([.2]),nominal_finite_segment_categories=np.array([0]),residual=np.array([.1,.1]))
    truth=SimpleNamespace(full_stroke_records=[record],queries=lambda q,**k:[dict(query=query)],
        _carriers=lambda q,**k:carrier+J[:,0]*q[0])
    projected=dict(projected_foot_xy=np.array([[1.1,1.1]]),fixed_foot_jacobian=np.array([[[-1.],[.5]]]),
        station_screen_tangent=np.array([[1.,-1.]]),nominal_physical_station=np.array([.1]),
        physical_endpoints_xy=np.array([[[1.,1.2],[0.,0.]]]),endpoint_jacobian=np.array([[[[-1.],[1.]],[[0.],[0.]]]]),
        endpoint_valid=np.array([[True,False]]))
    return truth,base,projected


def test_finite_model_uses_only_local_fixed_foot_chart_and_exact_group_map(monkeypatch):
    from structured_programme_source_candidate import source_step_curvature_tightening as original
    from calibration.solver_runtime import finite_rectangle_projection_fd as fd
    truth,base,projections=fixture()
    monkeypatch.setattr(original,'constraints_on_declared_free_chart',lambda *a,**k:base)
    monkeypatch.setattr(model,'_current_boundary_clipped_primitive',lambda *a:np.zeros((3,3)))
    monkeypatch.setattr(fd,'projected_station_contexts',lambda *a:projections)
    monkeypatch.setattr(fd,'projected_fixed_foot_carriers',lambda *a:projections['projected_foot_xy'])
    result=model.finite_rectangle_station_constraints(truth,np.zeros(1),np.array([0]),group_limits=np.array([.2]),
        return_step_carrier_response=True)
    report=result['report']['finite_rectangle_station_local_model'];response=result['step_carrier_response']
    assert report['original_local_carrier_faces_replaced']==4
    assert all('physical_'not in p['kind']for p in report['face_provenance'])
    assert np.all(response['row_group_index']==0)and np.all(response['row_limit_expansion']==.2)
    assert response['row_carrier_projection'].shape==(len(result['lower']),4)
    assert not response['row_carrier_projection'][:,:2].nnz
    # At nominal, no arbitrary opposite-pole constraint changes feasibility.
    assert np.all(result['lower']<=1e-15)


def test_ineligible_closed_and_computational_primitive_leave_original_faces(monkeypatch):
    from structured_programme_source_candidate import source_step_curvature_tightening as original
    truth,base,_=fixture()
    monkeypatch.setattr(original,'constraints_on_declared_free_chart',lambda *a,**k:base)
    for kind in ('house_ring','hockey_circle','curling_centreline','current_goal_crease'):
        truth.full_stroke_records[0]['row']['curve']['kind']=kind
        assert model.finite_rectangle_station_constraints(truth,np.zeros(1),np.array([0]))is base
