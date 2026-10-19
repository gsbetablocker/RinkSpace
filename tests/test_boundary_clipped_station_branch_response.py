"""An authenticated whole ice line uses one station chart in internal cells."""
from types import SimpleNamespace
import numpy as np
import pytest
from calibration.solver_runtime import install
install()
from structured_programme_source_candidate.normal_scanline_human_query_acceleration import finite_source_rectangle_projected_queries
from structured_programme_source_candidate.fixed_pivot_positive_domain_fd import positive_pivot_coordinate_derivative
from structured_programme_source_candidate.source_step_curvature_tightening import actual_carrier_step_response
from calibration.solver_runtime import finite_rectangle_station_step_constraints as model
from calibration.solver_runtime import finite_rectangle_projection_fd as projection


def fixture(monkeypatch,*,curved=False):
    v=np.linspace(0.,1.,17);uv=np.zeros((1,2));normal=np.array([[1.,0.]]);bounds=np.array([[-1.,1.]])
    curve=dict(kind='current_ice_segment',feature_id='UNNAMED_REGISTERED_LINE',segment_index=0)
    def world(q,curve):return np.c_[np.full(17,q[0]-q[1]),v,np.zeros(17)]
    def feature(p):return dict(kind='polyline',surface_id='ice_surface',
        structured_material_relation='rinkspace.direct-coupled-physical-hockey-curling-template-latent48.v1',
        boundary_rule='Tilted physical line extended and clipped to the fitted superellipse ice boundary',
        segments_xyz=world(p,curve)[[0,-1]][None,:,:])
    course=SimpleNamespace(sample_perimeter=lambda *a,**k:(np.array([[-10.,0.,0.],[-10.,1.,0.],[10.,1.,0.],[10.,0.,0.],[-10.,0.,0.]]),None))
    geometry=SimpleNamespace(effective_world=lambda p:p,print_course_binding=lambda p:(course,np.zeros(0),None))
    factor=SimpleNamespace(geometry=geometry,world_indices=np.array([0,1]),_world=world,
        _ice_material_view=SimpleNamespace(features=lambda p:{curve['feature_id']:feature(p)}))
    record=dict(uv=uv,normals=normal,bounds=bounds,tangent_bounds=bounds.copy(),factor=factor,
        row=dict(actual_pts=99,curve=curve))
    block=SimpleNamespace(bundle=SimpleNamespace(camera=lambda value,p:value),local=lambda value:value)
    def project(camera,xyz):
        s=xyz[:,1]-.53
        xy=(np.c_[.98+20.*s*s+xyz[:,0]+.001*camera[0]**2,s]if curved else
            np.c_[1.-.02*s+xyz[:,0]+.001*camera[0]**2,1.-.0025+s])
        return xy,np.ones(len(xyz)),np.ones(len(xyz),bool)
    def query(q,**frozen):
        xy,depth,central=project(q,world(q,curve))
        return finite_source_rectangle_projected_queries(xy,depth,central,np.ones(16,bool),uv,normal,bounds,
            tangent_bounds=bounds,native=False,**frozen)
    def carriers(q):
        value=query(q);return np.c_[value['signed_normal_px'],value['coverage_tangent_px']].ravel()
    truth=SimpleNamespace(parameter_count=2,height_columns=np.empty(0,int),world_indices=np.array([0,1]),
        steps={0:1e-6,1:1e-6},full_stroke_records=[record],blocks={99:block},
        state=SimpleNamespace(geometry=SimpleNamespace(validate_parameters=lambda p:True),
            structured_programme_coordinate_chart=SimpleNamespace(core_count=2)),
        queries=lambda q,**k:[dict(query=query(q))],_carriers=lambda q,**k:carriers(q))
    monkeypatch.setattr(projection,'_helpers',lambda:(lambda b,q:b.local(q),positive_pivot_coordinate_derivative,
        lambda t,records:[np.array([0,1])],project))
    nominal=query(np.zeros(2));frozen=dict(frozen_segments=nominal['nominal_finite_segments'],
        frozen_segment_categories=nominal['nominal_finite_segment_categories'],
        frozen_support_faces=nominal['nominal_source_support_faces'],
        frozen_station_branches=nominal['nominal_source_station_branches'])
    def old_carrier(q):
        value=query(q,**frozen);return np.c_[value['signed_normal_px'],value['coverage_tangent_px']].ravel()
    J=np.column_stack([positive_pivot_coordinate_derivative(old_carrier,np.zeros(2),i,1e-6,np.empty(0,int))for i in range(2)])
    carrier=carriers(np.zeros(2));A=np.vstack((J,-J));targets=np.r_[[-1.,-1.]-carrier,carrier-[1.,1.]]
    kept=np.any(A!=0.,axis=1)
    original=dict(matrix=A[kept],lower=targets[kept],report={},
        step_carrier_response=dict(nominal_carrier=carrier,retained_face_mask=kept))
    from structured_programme_source_candidate import source_step_curvature_tightening as original_module
    monkeypatch.setattr(original_module,'constraints_on_declared_free_chart',lambda *a,**k:original)
    return truth,query,old_carrier,J,original


def test_real_upper_double_slab_switch_on_internal_cell_uses_whole_physical_line(monkeypatch):
    truth,query,old_carrier,J,original=fixture(monkeypatch);q=np.zeros(2);step=np.ones(2)
    before=query(q);after=query(step)
    assert before['source_support_branch_signature'][0].tolist()==[8,0,0,2]
    assert after['source_support_branch_signature'][0].tolist()==[8,0,5,2]
    assert np.array_equal(J[0],np.zeros(2))
    assert max(abs(J@step))<1e-8
    assert max(abs(after['residual']))>.0009
    assert max(abs(old_carrier(step)-old_carrier(q)-J@step))>.04
    result=model.finite_rectangle_station_constraints(truth,q,np.array([0,1]),
        group_limits=np.array([0.]),return_step_carrier_response=True)
    response=result['step_carrier_response'];report=result['report']['finite_rectangle_station_local_model']
    assert report['contexts'][0]['whole_boundary_clipped_straight']
    assert not report['contexts'][0]['whole_vertical_print']
    assert report['contexts'][0]['segment']==8
    assert report['original_local_carrier_faces_replaced']==2
    assert not any('physical_'in p['kind']for p in report['face_provenance'])
    assert report['FM_local_face_values_are_not_dynamic_nonlinear_guard_violations']
    assert all(p['physical_station_bounds']==[0.,1.]for p in report['face_provenance'])
    actual=actual_carrier_step_response(response,model.actual_station_model_carriers(truth,step,response))
    slack=actual-result['lower'];worst=int(np.argmin(slack))
    assert np.maximum(-slack,0.).max()==pytest.approx(max(abs(after['residual'])),abs=1e-12)
    assert (result['matrix']@step-actual)[worst]==pytest.approx(.001/1.02,abs=1e-8)
    for h in (1e-5,1e-6):
        plus=actual_carrier_step_response(response,model.actual_station_model_carriers(truth,q+h*step,response))
        minus=actual_carrier_step_response(response,model.actual_station_model_carriers(truth,q-h*step,response))
        np.testing.assert_allclose((plus-minus)/(2*h),result['matrix']@step,atol=1e-8)
    assert truth.parameter_count==2 and np.array_equal(truth.full_stroke_records[0]['bounds'],[[-1.,1.]])
    np.testing.assert_array_equal(query(q)['residual'],before['residual'])


def test_certificate_once_per_record_and_exact_current_sample(monkeypatch):
    truth,*_=fixture(monkeypatch);record=truth.full_stroke_records[0];calls=[]
    original=model._current_boundary_clipped_primitive
    def observed(r,q):calls.append(q.copy());return original(r,q)
    monkeypatch.setattr(model,'_current_boundary_clipped_primitive',observed)
    contexts=[dict(record=record,group=0,source_vertex=0,segment=8,station=.48,
        whole_boundary_clipped_straight=True)]*3
    projection.projected_physical_endpoint_carriers(truth,np.zeros(2),contexts)
    assert len(calls)==1
    projection.projected_physical_endpoint_carriers(truth,np.ones(2),contexts)
    assert len(calls)==2 and np.array_equal(calls[-1],np.ones(2))


def test_no_uncertified_curve_or_sample_cap_extension(monkeypatch):
    truth,*_=fixture(monkeypatch);record=truth.full_stroke_records[0];curve=record['row']['curve']
    assert model._current_boundary_clipped_primitive(record,np.zeros(2)).shape==(17,3)
    view=record['factor']._ice_material_view;original=view.features
    view.features=lambda p:{curve['feature_id']:dict(original(p)[curve['feature_id']],boundary_rule='Unconfirmed finite samples')}
    assert model._current_boundary_clipped_primitive(record,np.zeros(2))is None
    with pytest.raises(ValueError,match='whole boundary-clipped straight physical line certificate'):
        projection.projected_physical_endpoint_carriers(truth,np.zeros(2),[dict(record=record,group=0,
            source_vertex=0,segment=8,station=.48,whole_boundary_clipped_straight=True)])
    view.features=original;old_world=record['factor']._world
    record['factor']._world=lambda q,c:old_world(q,c)[2:-2]
    with pytest.raises(ValueError,match='exact current registered whole primitive'):
        model._current_boundary_clipped_primitive(record,np.zeros(2))


def test_unused_distant_cap_does_not_enter_local_station_chart(monkeypatch):
    truth,*_=fixture(monkeypatch)
    result=model.finite_rectangle_station_constraints(truth,np.zeros(2),np.array([0,1]),
        group_limits=np.array([0.]),return_step_carrier_response=True)
    response=result['step_carrier_response'];assert response['finite_station_contexts'][0]['nominal_physical_endpoint_side']is None
    old=projection._helpers()
    def changed(camera,world):
        pixels,depth,central=old[-1](camera,world)
        if camera[0]>0.:central[-1]=False;pixels[-1]=np.nan
        return pixels,depth,central
    monkeypatch.setattr(projection,'_helpers',lambda:(*old[:-1],changed))
    # A distant physical endpoint is not used as a cap in this interior local
    # witness. The fixed nominal cell still retains all its central checks.
    actual=model.actual_station_model_carriers(truth,np.ones(2),response)
    assert np.isfinite(actual).all()


def test_world_straight_but_curved_screen_nominal_witness_is_not_excluded(monkeypatch):
    truth,query,*_=fixture(monkeypatch,curved=True);q=np.zeros(2)
    nominal=query(q);assert max(abs(nominal['residual']))==0.
    result=model.finite_rectangle_station_constraints(truth,q,np.array([0,1]),
        group_limits=np.array([0.]),return_step_carrier_response=True)
    response=result['step_carrier_response'];contexts=response['finite_station_contexts']
    fixed=projection.projected_station_contexts(truth,q,contexts,np.array([],int))
    # The true line has an interior source witness; neither distant endpoint
    # belongs to that source rectangle after the curved screen projection.
    assert np.all(fixed['physical_endpoints_xy'][0,:,0]>1.)
    assert 0.<fixed['nominal_physical_station'][0]<1.
    assert np.max(result['lower'],initial=-np.inf)<=0.
    assert all('physical_'not in p['kind']for p in result['report']['finite_rectangle_station_local_model']['face_provenance'])
    for h in (1e-5,1e-6):
        d=np.array([.3,-.2]);plus=model.actual_station_model_carriers(truth,q+h*d,response)
        minus=model.actual_station_model_carriers(truth,q-h*d,response)
        np.testing.assert_allclose((actual_carrier_step_response(response,plus)-
            actual_carrier_step_response(response,minus))/(2*h),result['matrix']@d,atol=1e-8)


def test_nominal_consistency_rejects_false_face_even_when_cap_would_hide_it(monkeypatch):
    truth,*_=fixture(monkeypatch);context=dict(record=truth.full_stroke_records[0],group=0,source_vertex=0)
    with pytest.raises(ValueError,match='excludes its original support witness'):
        model._verify_nominal_face(.01,0.,context=context,
            provenance=dict(kind='fake_distant_cap'),operands=[1.,1.])
    bound=model._verify_nominal_face(np.finfo(float).eps,0.,context=context,
        provenance=dict(kind='arithmetic_identity'),operands=[1.,1.])
    assert bound==64.*np.finfo(float).eps*2.


def test_true_endpoint_category_uses_literal_endpoint_but_internal_knot_remains_free(monkeypatch):
    truth,query,*_=fixture(monkeypatch);q=np.zeros(2);value=query(q)
    value['nearest_finite_segment_index']=np.array([0])
    value['nearest_finite_segment_station']=np.array([np.nextafter(0.,1.)])
    value['nominal_finite_segment_categories']=np.array([1])
    context=model.nominal_station_contexts(truth,q,full_queries=[dict(query=value)])[0]
    assert context['nominal_physical_endpoint_side']=='lower'and context['station']==0.
    assert context['original_nominal_segment_station']>0.
    value['nearest_finite_segment_index']=np.array([8])
    context=model.nominal_station_contexts(truth,q,full_queries=[dict(query=value)])[0]
    assert context['nominal_physical_endpoint_side']is None
    assert context['station']==np.nextafter(0.,1.)
    value['nearest_finite_segment_index']=np.array([15]);value['nearest_finite_segment_station']=np.array([np.nextafter(1.,0.)])
    value['nominal_finite_segment_categories']=np.array([2])
    context=model.nominal_station_contexts(truth,q,full_queries=[dict(query=value)])[0]
    assert context['nominal_physical_endpoint_side']=='upper'and context['station']==1.
