"""A real finite rectangle switch must use the same eliminated station chart."""
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


def fixture(monkeypatch):
    v=np.linspace(0.,1.,17);uv=np.zeros((1,2));n=np.array([[1.,0.]]);bounds=np.array([[-1.,1.]])
    geometry=SimpleNamespace(paired_material_protocol_schema='rinkspace.same-size-paired-original-materials-independent-station-zmin.v1',
        _materials={(20,'right_vertical_xyz'):(np.ones(17),v)})
    def world(q,curve):return np.c_[np.full(17,q[0]-q[1]),np.zeros(17),v]
    factor=SimpleNamespace(geometry=geometry,_world=world)
    record=dict(uv=uv,normals=n,bounds=bounds,tangent_bounds=bounds.copy(),factor=factor,
        row=dict(actual_pts=99,curve=dict(kind='advertisement_boundary',advertisement_number=20,
            edge='right_vertical',extent_scope='external_border')))
    block=SimpleNamespace(bundle=SimpleNamespace(camera=lambda value,p:value),local=lambda value:value)
    def project(camera,xyz):
        s=xyz[:,2]-.53 # source slab intersection lies inside a sample cell
        xy=np.c_[-1.-.003*s+xyz[:,0]-.001*camera[0]**2,-1.+.002863+s]
        return xy,np.ones(len(xyz)),np.ones(len(xyz),bool)
    def query(q,**frozen):
        xyz=world(q,record['row']['curve']);xy,depth,central=project(q,xyz)
        return finite_source_rectangle_projected_queries(xy,depth,central,np.ones(16,bool),uv,n,bounds,
            tangent_bounds=record['tangent_bounds'],native=False,**frozen)
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
    c=carriers(np.zeros(2));A=np.vstack((J,-J));targets=np.r_[[-1.,-1.]-c,c-[1.,1.]];kept=np.any(A!=0.,axis=1)
    original=dict(matrix=A[kept],lower=targets[kept],report={},
        step_carrier_response=dict(nominal_carrier=c,retained_face_mask=kept))
    from structured_programme_source_candidate import source_step_curvature_tightening as original_module
    monkeypatch.setattr(original_module,'constraints_on_declared_free_chart',lambda *a,**k:original)
    return truth,query,old_carrier,J,original


def test_double_slab_switch_retains_the_whole_physical_station_and_correct_curvature(monkeypatch):
    truth,query,old_carrier,J,original=fixture(monkeypatch);q=np.zeros(2);step=np.ones(2)
    before=query(q);after=query(step)
    assert before['nominal_finite_segments'][0]not in (0,15)
    assert before['nominal_source_support_faces'][0]==0
    assert after['nominal_source_support_faces'][0]==6
    assert np.array_equal(J[0],np.zeros(2)) # correct exact normal-slab identity
    assert np.max(abs(J@step))<1e-8
    assert abs(after['residual']).max()>.00098
    old_response=actual_carrier_step_response(original['step_carrier_response'],
        np.c_[after['signed_normal_px'],after['coverage_tangent_px']].ravel())
    frozen_error=old_carrier(step)-old_carrier(q)-J@step
    assert max(abs(frozen_error))>.33
    assert max(abs(old_response))<.004 # a switched optimum cannot be used as the old tangent coordinate
    result=model.finite_rectangle_station_constraints(truth,q,np.array([0,1]),
        group_limits=np.array([0.]),return_step_carrier_response=True)
    response=result['step_carrier_response'];report=result['report']['finite_rectangle_station_local_model']
    assert report['contexts'][0]['whole_vertical_print']
    assert report['original_local_carrier_faces_replaced']==2
    assert not any('physical_'in p['kind']for p in report['face_provenance'])
    assert report['FM_local_face_values_are_not_dynamic_nonlinear_guard_violations']
    assert all(p['physical_station_bounds']==[0.,1.]for p in report['face_provenance'])
    actual=actual_carrier_step_response(response,model.actual_station_model_carriers(truth,step,response))
    slack=actual-result['lower'];worst=int(np.argmin(slack))
    assert np.maximum(-slack,0.).max()==pytest.approx(abs(after['residual']).max(),abs=1e-12)
    assert (result['matrix']@step-actual)[worst]==pytest.approx(.001/1.003,abs=1e-8)
    # Public dynamic truth/source support never changes, and no station enters q.
    assert truth.parameter_count==2 and np.array_equal(truth.full_stroke_records[0]['bounds'],[[-1.,1.]])
    assert np.array_equal(before['residual'],query(q)['residual'])
    for h in (1e-5,1e-6):
        plus=actual_carrier_step_response(response,model.actual_station_model_carriers(truth,q+h*step,response))
        minus=actual_carrier_step_response(response,model.actual_station_model_carriers(truth,q-h*step,response))
        np.testing.assert_allclose((plus-minus)/(2*h),result['matrix']@step,atol=1e-8)


def test_only_original_full_vertical_external_material_is_admitted(monkeypatch):
    truth,*_=fixture(monkeypatch);record=truth.full_stroke_records[0];curve=record['row']['curve']
    assert model._current_vertical_print_primitive(record,np.zeros(2)).shape==(17,3)
    for changes in ({'edge':'top_edge'},{'edge':'bottom_edge'},{'extent_scope':'printed_component'},
                    {'kind':'printed_component_contour'}):
        old=curve.copy();curve.update(changes)
        assert model._current_vertical_print_primitive(record,np.zeros(2))is None
        curve.clear();curve.update(old)
    geometry=record['factor'].geometry
    geometry._materials[20,'right_vertical_xyz']=(np.linspace(0.,1.,17),np.linspace(0.,1.,17))
    with pytest.raises(ValueError,match='exact developed station'):
        model._current_vertical_print_primitive(record,np.zeros(2))


def test_current_whole_vertical_certificate_once_per_record_and_exact_sample(monkeypatch):
    truth,*_=fixture(monkeypatch);record=truth.full_stroke_records[0];original=record['factor']._world;calls=[]
    def observed(value,curve):calls.append(value.copy());return original(value,curve)
    record['factor']._world=observed
    contexts=[dict(record=record,group=0,source_vertex=0,segment=8,station=.48,whole_vertical_print=True)]*2
    projection.projected_physical_endpoint_carriers(truth,np.zeros(2),contexts)
    assert len(calls)==1
    projection.projected_physical_endpoint_carriers(truth,np.ones(2),contexts)
    assert len(calls)==2 and np.array_equal(calls[-1],np.ones(2))


def test_actual_curvature_requires_the_original_nominal_foot_cell_central_domain(monkeypatch):
    truth,*_=fixture(monkeypatch)
    result=model.finite_rectangle_station_constraints(truth,np.zeros(2),np.array([0,1]),
        group_limits=np.array([0.]),return_step_carrier_response=True)
    response=result['step_carrier_response']
    context=response['finite_station_contexts'][0]
    assert context['nominal_physical_endpoint_side']is None
    old=projection._helpers()
    def changed(camera,world):
        pixels,depth,central=old[-1](camera,world)
        if camera[0]>0.:
            central[context['segment']]=False;pixels[context['segment']]=np.nan
        return pixels,depth,central
    monkeypatch.setattr(projection,'_helpers',lambda:(*old[:-1],changed))
    with pytest.raises(ValueError,match='nominal segment must retain positive central support'):
        model.actual_station_model_carriers(truth,np.ones(2),response)
