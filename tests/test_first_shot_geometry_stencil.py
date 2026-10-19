"""Physical-domain stencils keep their original step and real errors visible."""
from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.program_tracking import first_shot_fit as fit
from calibration.program_tracking.first_shot_physical_state import GeometryDomainError


class PhysicalDomainState:
    geometry_count = 118
    provider = SimpleNamespace(fd_step=1e-6, current=SimpleNamespace(lane_right_extension_indices={
        'house_connection_F1_R1':94, 'house_connection_F2_R2':95,
        'house_connection_F3_R3':96, 'house_connection_F4_R4':97}))
    def __init__(self, both_sides_invalid=False, broken_world=False):
        self.both_sides_invalid=both_sides_invalid; self.broken_world=broken_world
    def guard_geometry(self, q):
        if q[0]<0:raise GeometryDomainError('actual lower footprint domain')
        if q[1]>0:raise GeometryDomainError('actual upper footprint domain')
        if self.both_sides_invalid and q[2]!=0:raise GeometryDomainError('both physical sides unavailable')
        return {'passed':True}
    def world_paths(self,q,descriptors):
        if self.broken_world and q[3]!=0:raise ValueError('actual world interface failure')
        station=np.linspace(0.,1.,129)
        p=np.zeros((129,3));p[:,0]=station*(1+108/945+q[95]);p[:,1]=q[0]+q[1]
        return (p,)


def bundle(state):
    result=fit.FirstShotSourceBundle.__new__(fit.FirstShotSourceBundle)
    result.state=state
    descriptor={'kind':'ice_curve','curve_id':'house_connection_F2_R2:segment0'}
    key=fit._target_key(descriptor)
    result.descriptors=OrderedDict([(key,descriptor)])
    result.target_descriptors=(descriptor,);result.keys=(key,)
    result._world_cache=OrderedDict();result._source_pool=result._geometry_pool=None
    result.geometry_workers=1;result.timings=[]
    return result


def test_original_step_one_sided_geometry_domain_and_analytic_terminal_extent():
    item=bundle(PhysicalDomainState());q=np.zeros(118)
    before=q.copy();world,derivatives=item.world(q,jacobian=True)
    derivative=derivatives[item.keys[0]]
    np.testing.assert_allclose(derivative[:,1,0],1.)
    np.testing.assert_allclose(derivative[:,1,1],1.)
    np.testing.assert_array_equal(derivative[:,0,95],np.linspace(0.,1.,129))
    np.testing.assert_array_equal(derivative[:,:,96:98],np.zeros((129,3,2)))
    stencils={r['column']:r for r in item.timings[-1]['geometry_derivative_stencils']}
    assert stencils[0]['method']=='forward_original_step'
    assert stencils[1]['method']=='backward_original_step'
    assert stencils[4]['method']=='central_original_step'
    assert stencils[0]['original_step']==stencils[1]['original_step']==1e-6
    assert stencils[95]['method']=='analytic_current_house_connection_terminal_extent'
    assert stencils[95]['source_or_world_column_removed'] is False
    np.testing.assert_array_equal(q,before)


def test_both_original_geometry_domain_sides_fail_explicitly():
    item=bundle(PhysicalDomainState(both_sides_invalid=True))
    with pytest.raises(GeometryDomainError,match='Both original shared-geometry FD sides'):
        item.world(np.zeros(118),jacobian=True)


def test_finite_difference_does_not_swallow_actual_world_interface_error():
    item=bundle(PhysicalDomainState(broken_world=True))
    with pytest.raises(ValueError,match='actual world interface failure') as failure:
        item.world(np.zeros(118),jacobian=True)
    assert not isinstance(failure.value,GeometryDomainError)


def test_process_worker_reports_only_typed_geometry_domain_sides(monkeypatch):
    state=PhysicalDomainState(broken_world=True)
    monkeypatch.setattr(fit,'_geometry_worker',(state,({'kind':'ice_curve'},)))
    col,shift,value,reason=fit._geometry_sample((np.zeros(118),0,-1e-6))
    assert col==0 and shift==-1e-6 and value is None and 'lower footprint' in reason
    with pytest.raises(ValueError,match='actual world interface failure'):
        fit._geometry_sample((np.zeros(118),3,1e-6))


def test_continuous_lines_follow_scene_primitives_and_preserve_width_information():
    scene={'ice_features':[
        {'id':'B1','kind':'polyline','segments_xyz':[[[0.,0.,0.],[0.,2.,0.]]]},
        {'id':'C0','kind':'circle'},
        {'id':'curved','kind':'polyline','segments_xyz':[[[0.,0.,0.],[1.,.1,0.],[2.,0.,0.]]]},
        {'id':'lane1','kind':'polygon','paint_segments_xyz':[
            [[0.,1.,0.],[2.,1.,0.]],[[2.,1.,0.],[2.,-1.,0.]],[[2.,-1.,0.],[0.,-1.,0.]]]},
        *[{'id':f'house_connection_F{i}_R{i}','kind':'polyline',
           'segments_xyz':[[[0.,0.,0.],[2.,0.,0.]]]} for i in range(1,5)]]}
    def metadata(curve):
        return fit._continuous_line_metadata({'kind':'ice_curve','curve_id':curve},scene)
    assert metadata('B1:segment0')=={'_continuous_world_line':True}
    assert metadata('C0:outer')==metadata('curved:segment0')=={}
    assert metadata('lane1:edge0')['_line_end_extent_geometry_columns']==[22]
    assert metadata('lane1:edge2')['_line_start_extent_geometry_columns']==[22]
    assert metadata('lane1:edge1')=={'_continuous_world_line':True}
    assert all(21 not in r.get('_line_end_extent_geometry_columns',[])
               and 21 not in r.get('_line_start_extent_geometry_columns',[])
               for r in [metadata('lane1:edge0'),metadata('lane1:edge2')])
    for i in range(1,5):
        assert metadata(f'house_connection_F{i}_R{i}:segment0')['_line_end_extent_geometry_columns']==[93+i]


def test_real_current_lane_producer_endpoint_formula_and_width_translation():
    from calibration.program_tracking.first_shot_physical_state import FirstShotPhysicalState
    state=FirstShotPhysicalState();q=state.initial_q[:118].copy()
    assert state.provider.parameter_names[21]=='four_Lane_paint_common_width_m'
    assert state.provider.parameter_names[22]=='four_Lane_paint_right_end_from_Rtee_m'
    targets=[{'kind':'ice_curve','curve_id':f'lane{i}:edge{edge}'}
             for i in range(1,5) for edge in (0,2)]
    paths=state.world_paths(q,targets)
    ca=np.array([np.cos(q[14]),np.sin(q[14]),0.])/state.scene_unit_working_m
    cv=np.array([-np.sin(q[14]),np.cos(q[14]),0.])/state.scene_unit_working_m
    for col in (21,22):
        plus=q.copy();minus=q.copy();plus[col]+=1e-6;minus[col]-=1e-6
        forward=state.world_paths(plus,targets);backward=state.world_paths(minus,targets)
        for descriptor,path,a,b in zip(targets,paths,forward,backward,strict=True):
            edge=int(descriptor['curve_id'][-1]);t=np.linspace(0.,1.,len(path))
            expected=((t if edge==0 else 1.-t)[:,None]*ca if col==22 else
                      np.broadcast_to((1. if edge==0 else -1.)*cv/2,path.shape))
            # Subtracting two scene coordinates has this scale-dependent
            # floating-point error after the original 1e-6 FD step.
            bound=8*np.finfo(float).eps*np.maximum(1.,np.abs(path)).max()/1e-6
            np.testing.assert_allclose((a-b)/(2e-6),expected,rtol=0,atol=bound)
    for identity in ('B1','B2','C'):
        assert fit._continuous_line_metadata({'kind':'ice_curve','curve_id':identity+':segment0'},state.approved_scene)=={'_continuous_world_line':True}


def test_shared_right_end_overrides_only_proved_lane_targets():
    class LaneState(PhysicalDomainState):
        scene_unit_working_m=.5
        provider=SimpleNamespace(fd_step=1e-6,
            parameter_names=['']*22+['four_Lane_paint_right_end_from_Rtee_m'],
            current=PhysicalDomainState.provider.current)
        def world_paths(self,q,descriptors):
            ca=np.array([np.cos(q[14]),np.sin(q[14]),0.])/.5
            cv=np.array([-np.sin(q[14]),np.cos(q[14]),0.])/.5
            t=np.linspace(0.,1.,257)
            a=t[:,None]*(2.+q[22])*ca+q[21]*cv/2
            b=(1.-t[:,None])*(2.+q[22])*ca-q[21]*cv/2
            other=np.zeros((2,3));other[:,2]=q[22]
            return a,b,other
    item=bundle(LaneState())
    ds=[{'kind':'ice_curve','curve_id':'lane1:edge0'},
        {'kind':'ice_curve','curve_id':'lane1:edge2'},
        {'kind':'ice_curve','curve_id':'other:segment0'}]
    item.descriptors=OrderedDict((fit._target_key(d),d) for d in ds)
    item.target_descriptors=tuple(ds);item.keys=tuple(item.descriptors)
    q=np.zeros(118);q[14]=.03
    paths,derivatives=item.world(q,jacobian=True)
    t=np.linspace(0.,1.,257);ca=np.array([np.cos(q[14]),np.sin(q[14]),0.])/.5
    np.testing.assert_array_equal(derivatives[item.keys[0]][...,22],t[:,None]*ca)
    np.testing.assert_array_equal(derivatives[item.keys[1]][...,22],(1.-t)[:,None]*ca)
    np.testing.assert_array_equal(derivatives[item.keys[2]][:,2,22],np.ones(2))
    assert np.linalg.norm(derivatives[item.keys[0]][...,21])>1
    stencil=item.timings[-1]['geometry_derivative_stencils'][22]
    assert stencil['exact_target_endpoint_overrides']['width_column21_preserved']


@pytest.mark.parametrize('field', ['model_association_overrides','source_point_ids',
    'source_estimator_B_and_noise_factor_L_preserved','physical_material_residual_basis_only_changed'])
def test_target_identity_change_cannot_reuse_unmatched_source_normal_operator(field):
    ref={'path':'authenticated identity receipt','sha256':'receipt digest'}
    evidence=dict(model_association_overrides=ref,source_point_ids=['acceptedID'],
        source_estimator_B_and_noise_factor_L_preserved=True,physical_material_residual_basis_only_changed=True)
    fit._validate_corrected_association_record({'corrected_evidence':evidence},{'acceptedID'},ref)
    incorrect=deepcopy(evidence);incorrect.pop(field)
    with pytest.raises(ValueError,match='matching corrected source-normal'):
        fit._validate_corrected_association_record({'corrected_evidence':incorrect},{'acceptedID'},ref)


def test_association_override_preserves_exact_native_address_and_UV():
    observation=dict(native_pts=406272,time_base='1/12800',source_packet={'path':'native packet','sha256':'native digest'},
        source_point_ids=['opaque acceptedID'],source_feature_id='ice.C0:outer',source_segment_index=2,
        original_vertex_indices=[1],target_id='ice.C0:outer',uv=np.array([[1490.1259,870.]]))
    item=dict(native_pts=406272,time_base='1/12800',source_packet=observation['source_packet'],
        source_point_id='opaque acceptedID',original_source_feature_id='ice.C0:outer',
        original_source_segment_index=2,original_source_vertex_index=1,original_source_model_target_id='ice.C0:outer',
        approved_source_UV=[1490.1259,870.],model_target_descriptor={'kind':'ice_curve','curve_id':'B1:segment0'})
    assert fit._association_target(item,observation,0)==item['model_target_descriptor']
    item['approved_source_UV'][1]+=.001
    with pytest.raises(ValueError,match='original native source address or UV'):
        fit._association_target(item,observation,0)
