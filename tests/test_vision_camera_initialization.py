import hashlib

import numpy as np
import pytest

from rinkspace.vision_camera_initialization import (authenticated_view_slots, apply_absolute_camera_initializer,
    frozen_scene_curve, forward_connected_perimeter_indices, unit_source_normals, camera_with_pivot_height,
    source_jacobian_in_camera25_units)


def test_view_indices_use_original_component_order_not_sorted_source_time():
    q = np.zeros(24, dtype=np.float64)
    q[8:12] = [1., 2., 3., .4]
    q[16:20] = [4., 5., 6., .7]
    state = dict(q=q, aux__canonical_pts=np.array([1024, 1280, 0, 256]),
        aux__block_lengths=np.array([8, 8]), aux__block_start=np.array([4, 12]))
    camera = dict(pts=np.array([0, 256, 1024, 1280]), exposure_representative_pts=np.array([0, 256, 1024, 1280]),
        view_state4=np.array([[0.,0.,0.,0.],[4.,5.,6.,-6.860080465021827],[0.,0.,0.,0.],[1.,2.,3.,-7.160080465021827]]),
        camera25=np.zeros((4,25)))
    camera['camera25'][1,6:8]=.7; camera['camera25'][3,6:8]=.4
    indices, native_index, actual = authenticated_view_slots(state, camera, 256)
    assert indices.tolist() == [16, 17, 18, 19] and native_index == 1 and actual == 256
    with pytest.raises(ValueError, match="Reference"):
        authenticated_view_slots(state, camera, 1024)
    reference,_,actual=authenticated_view_slots(state,camera,1024,allow_reference=True)
    assert reference.tolist()==[4,5,6,7] and actual==1024
    camera["view_state4"][1, 0] += .001
    with pytest.raises(ValueError, match="exactly"):
        authenticated_view_slots(state, camera, 256)


def packet_fixture(tmp_path):
    parent = np.arange(20, dtype=np.float64)
    indices = np.array([12,13,14,15],dtype=np.int64)
    path = tmp_path / 'delta.npz'
    np.savez_compressed(path, parameter_indices=indices, original_values=parent[indices], candidate_values=parent[indices]+.01)
    packet = dict(schema='rinkspace.source-qualified-camera-pose-initializer.v1', role='initializer_only', accuracy_accepted=False,
        parent_q_sha256=hashlib.sha256(parent.tobytes()).hexdigest(), initializer_is_not_accepted_fit=True,
        shared_world_pivots_levers_and_optics_unchanged=True, fresh_complete_source_profiles_and_J_required=True,
        delta_npz=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        permitted_RF4_indices=indices.tolist(),protected_shared_parameter_indices=list(range(8)))
    return parent, packet, path


def test_initializer_applies_copy_only_and_rejects_stale_parent_or_shared_slots(tmp_path):
    parent, packet, path = packet_fixture(tmp_path)
    untouched = parent.copy()
    result = apply_absolute_camera_initializer(parent, packet)
    assert np.array_equal(parent, untouched)
    assert np.flatnonzero(result != parent).tolist() == [12,13,14,15]
    changed = parent.copy();changed[19] += .001
    with pytest.raises(ValueError, match='parent'):
        apply_absolute_camera_initializer(changed, packet)
    packet['protected_shared_parameter_indices'].append(12)
    with pytest.raises(ValueError, match='shared'):
        apply_absolute_camera_initializer(parent, packet)


def test_changed_typed_delta_and_bare_print_ink_cannot_enter_existing_camera_target(tmp_path):
    parent, packet, path = packet_fixture(tmp_path)
    np.savez_compressed(path,parameter_indices=np.array([12,13,14,15]),original_values=parent[12:16],candidate_values=parent[12:16]+.1)
    with pytest.raises(ValueError,match='delta changed'):
        apply_absolute_camera_initializer(parent,packet)
    with pytest.raises(ValueError,match='Printed ink'):
        frozen_scene_curve(dict(ice_features=[]),dict(kind='advertisement_boundary',extent_scope='printed_pattern_external_contour'))


def test_cyclic_forward_scope_preserves_actual_wrap_edges_and_never_bridges_missing_world():
    assert forward_connected_perimeter_indices(np.array([True,True,False,False,True])).tolist()==[4,0,1]
    with pytest.raises(ValueError,match='Multiple'):
        forward_connected_perimeter_indices(np.array([True,False,True,False,True]))


def test_float32_normal_roundoff_adapter_does_not_change_original_or_accept_scaled_directions():
    source = np.array([[.6, .8]], dtype=np.float32)
    original = source.copy()
    normals, error = unit_source_normals(source, 1)
    assert np.array_equal(source, original)
    assert 0 < error < 1e-6
    assert np.linalg.norm(normals[0]) == 1.
    with pytest.raises(ValueError, match='float32 roundoff'):
        unit_source_normals(source * 2, 1)
    with pytest.raises(ValueError, match='per source UV'):
        unit_source_normals(source, 2)


def test_existing_shared_pivot_height_moves_optical_center_without_changing_lever_or_optics():
    from calibration.program_tracking.z190_physical_camera import SharedCalibration, ViewState
    calibration = SharedCalibration('existing', (10.,-20.,4.), lever_xyz=(.1,-.6,.3), k1=(.2,))
    view = ViewState((.3,.1,-.2),1.2)
    baseline = calibration.camera25(view)
    identical = camera_with_pivot_height(calibration,view,1.2,1.2,.3)[0]
    assert np.array_equal(identical,baseline)
    changed = camera_with_pivot_height(calibration,view,1.2,1.5,.3)[0]
    assert np.array_equal(changed[:2],baseline[:2])
    assert np.array_equal(changed[3:],baseline[3:])
    assert changed[2] == baseline[2]+1.
    assert calibration.pivot_xyz == (10.,-20.,4.) and calibration.lever_xyz == (.1,-.6,.3)
    # The actual camera centre differs from its fixed pivot because of lever.
    assert not np.array_equal(baseline[:3],calibration.pivot_xyz)
    with pytest.raises(ValueError,match='positive'):
        camera_with_pivot_height(calibration,view,1.2,0.,.3)


def test_original_source_J_conditioning_maps_actual_lever_and_height_derivatives_to_value_FD(monkeypatch):
    monkeypatch.setenv('RINKSPACE_PROJECTION_BACKEND','numpy')
    from calibration.program_tracking.z190_physical_camera import SharedCalibration, ViewState
    from calibration.solver_runtime import install
    install()
    from z190_source_camera_diagnostic import absolute_value_jac15
    calibration = SharedCalibration('existing',(10.,-20.,4.),lever_xyz=(.1,-.6,.3),k1=(.2,))
    view = np.array([.3,.1,-.2,1.2])
    row = dict(world=np.array([[11.,-18.,30.],[12.,-21.,34.]]),uv=np.array([[950.,500.],[1050.,600.]]),
        sigma=np.ones(2),point_correspondence=True)
    camera,_,mapping = calibration.camera25_and_jacobians(ViewState.from_state4(view))
    _,legacy = absolute_value_jac15(camera,[row],True)
    physical = source_jacobian_in_camera25_units(legacy)
    expected = physical@mapping[:15]
    actual=[]
    for column in range(4):
        plus,minus=view.copy(),view.copy();plus[column]+=1e-7;minus[column]-=1e-7
        a=absolute_value_jac15(calibration.camera25(ViewState.from_state4(plus)),[row],False)
        b=absolute_value_jac15(calibration.camera25(ViewState.from_state4(minus)),[row],False)
        actual.append((a-b)/2e-7)
    np.testing.assert_allclose(expected,np.stack(actual,axis=1),rtol=1e-7,atol=1e-5)
    a=camera_with_pivot_height(calibration,ViewState.from_state4(view),1.2,1.2+1e-6,.3)[0]
    b=camera_with_pivot_height(calibration,ViewState.from_state4(view),1.2,1.2-1e-6,.3)[0]
    derivative=(absolute_value_jac15(a,[row],False)-absolute_value_jac15(b,[row],False))/2e-6
    np.testing.assert_allclose(physical[:,2]/.3,derivative,rtol=1e-7,atol=1e-5)
    # Direct multiplication of legacy columns really changes the coupled
    # lever derivative; this guards the concrete bug, not just a scale helper.
    assert np.max(np.abs(legacy@mapping[:15]-expected)) > 1.


def test_partial_initializer_uses_actual_projection_and_fixed_existing_prior_once(monkeypatch):
    monkeypatch.setenv('RINKSPACE_PROJECTION_BACKEND','numpy')
    from rinkspace.vision_camera_initialization import solve_source_qualified_installation_views
    from calibration.program_tracking.z190_physical_camera import SharedCalibration, ViewState
    from calibration.solver_runtime import install
    install()
    from z190_source_camera_diagnostic import absolute_value_jac15
    from correlated_roll_prior_candidate.explicit_bias_source import theta_and_gradient
    calibration=SharedCalibration('existing',(10.,-20.,4.),lever_xyz=(.1,-.6,.3),k1=(.2,))
    initial=np.array([[.3,.1,-.2,1.2],[.3,.1,-.2,1.21]])
    original=initial.copy()
    world=np.array([[11.,-18.,30.],[12.,-21.,34.],[9.,-19.,32.],[13.,-17.,36.]])
    callbacks=[]
    for view in initial:
        target=view+np.array([0.,0.,0.,.002])
        camera=calibration.camera25(ViewState.from_state4(target))
        row=dict(world=world.copy(),uv=np.zeros((4,2)),sigma=np.ones(4),point_correspondence=True)
        row['uv']=absolute_value_jac15(camera,[row],False).reshape(-1,2)
        callbacks.append(lambda camera,jac,row=row:absolute_value_jac15(camera,[row],jac))
    bias=float(theta_and_gradient(initial[:,:3])[0].mean())
    scale=1.6/calibration.camera25(ViewState.from_state4(initial[0]))[2]
    result=solve_source_qualified_installation_views(calibration,initial,1.2,scale,callbacks,
        source_quadratic_masks=[np.ones(8,bool),np.ones(8,bool)],roll_bias_radians=bias,
        optical_height_reference_index=0,with_height=False,max_nfev=25)
    assert np.array_equal(initial,original)
    assert result['candidate_pivot_Z']==1.2
    assert result['original_roll_prior_count']==2 and result['original_optical_height_prior_count']==1
    assert result['after_cost'] < result['before_cost']/3
    assert max(c['actual_value_centered_FD_relative_error'] for c in result['initial_actual_value_derivative_checks']) < 1e-6
    assert result['source_quadratic_scalar_count']==16 and result['source_Huber2_scalar_count']==0
    with pytest.raises(ValueError,match='quadratic masks'):
        solve_source_qualified_installation_views(calibration,initial,1.2,scale,callbacks,
            source_quadratic_masks=[np.ones(8),np.ones(8)],roll_bias_radians=bias)
    with pytest.raises(ValueError,match='actual source residuals'):
        solve_source_qualified_installation_views(calibration,initial,1.2,scale,callbacks,
            source_quadratic_masks=[np.ones(6,bool),np.ones(8,bool)],roll_bias_radians=bias)
