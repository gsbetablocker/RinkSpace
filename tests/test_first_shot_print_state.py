"""Shared Ad2 layout appends material freedom without moving camera columns."""
from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from calibration.program_tracking import first_shot_fit as fit
from calibration.program_tracking.first_shot_physical_state import FirstShotPhysicalState, GeometryDomainError, _print_layout_ids
from calibration.program_tracking.official_static_surface_dataset import read_reference, reference


@pytest.fixture(scope='module')
def states():
    old=FirstShotPhysicalState()
    folder=Path('D:/CodexData/RinkSpace_Data/tmp/shot1_print_state_tests')/uuid4().hex
    folder.mkdir(parents=True)
    config=folder/'layout.json'
    config.write_text(json.dumps(dict(schema='rinkspace.first-shot-ad2-print-layout.v1',advertisement_id=2)),encoding='utf8')
    binding=deepcopy(old.binding);binding['print_layout']=reference(config)
    path=folder/'binding.json';path.write_text(json.dumps(binding),encoding='utf8')
    new=FirstShotPhysicalState.from_binding(path)
    return old,new


@pytest.fixture(scope='module')
def generic_state(states):
    old,_=states
    folder=Path('D:/CodexData/RinkSpace_Data/tmp/shot1_print_state_tests')/uuid4().hex;folder.mkdir(parents=True)
    config=folder/'layouts.json'
    config.write_text(json.dumps(dict(schema='rinkspace.first-shot-print-layouts.v1',
        layouts=[dict(advertisement_id=n) for n in (2,3,6,7)])),encoding='utf8')
    binding=deepcopy(old.binding);binding['print_layout']=reference(config)
    path=folder/'binding.json';path.write_text(json.dumps(binding),encoding='utf8')
    return FirstShotPhysicalState.from_binding(path)


@pytest.fixture(scope='module')
def all_layout_state(states):
    old,_=states
    folder=Path('D:/CodexData/RinkSpace_Data/tmp/shot1_print_state_tests')/uuid4().hex;folder.mkdir(parents=True)
    config=folder/'layouts.json'
    config.write_text(json.dumps(dict(schema='rinkspace.first-shot-print-layouts.v1',
        layouts=[dict(advertisement_id=n) for n in (2,3,4,5,6,7)])),encoding='utf8')
    binding=deepcopy(old.binding);binding['print_layout']=reference(config)
    path=folder/'binding.json';path.write_text(json.dumps(binding),encoding='utf8')
    return FirstShotPhysicalState.from_binding(path)


def test_shared_tail_has_explicit_noncontiguous_world_and_schur_indices(states):
    old,new=states
    assert old.count==5119 and new.count==5127
    assert old.geometry_count==new.geometry_count==118
    assert old.shared_count==new.shared_count==127
    assert old.view_stop==new.view_stop==5119
    assert new.world_parameter_count==126 and new.global_shared_count==135
    np.testing.assert_array_equal(new.world_parameter_indices,np.r_[np.arange(118),np.arange(5119,5127)])
    np.testing.assert_array_equal(new.global_shared_indices,np.r_[np.arange(127),np.arange(5119,5127)])
    np.testing.assert_array_equal(new.initial_q[:5119],old.initial_q)
    np.testing.assert_array_equal(new.initial_q[5119:],np.zeros(8))
    np.testing.assert_array_equal(new.parent_projection(new.initial_q),old.parent_q)
    for index in (1,201,1248):
        np.testing.assert_array_equal(new.view_columns(index),old.view_columns(index))
        np.testing.assert_array_equal(new.camera25(new.initial_q,index),old.camera25(old.initial_q,index))
    light=fit._CameraState(new)
    np.testing.assert_array_equal(light.camera25(new.initial_q,1248),new.camera25(new.initial_q,1248))
    np.testing.assert_array_equal(light.world_parameter_indices,new.world_parameter_indices)
    assert light.view_stop==5119


def test_identity_print_layout_preserves_exact_scene_templates_and_all_target_worlds(states):
    old,new=states
    before=deepcopy(old.templates)
    assert new.scene(new.initial_q)==old.scene(old.initial_q)
    assert new.templates_at(new.initial_q)==old.templates==before
    descriptors=list(old.registry['targets'].values())
    for a,b in zip(old.world_paths(old.initial_q,descriptors),new.world_paths(new.initial_q,descriptors),strict=True):
        np.testing.assert_array_equal(a,b)
    assert new.guard(new.initial_q)['passed']


def test_shared_print_layout_changes_ink_and_not_border_or_camera(states):
    old,state=states;q=state.initial_q.copy();q[5119+4]=.002;q[5119+5]=.001
    targets=[{'kind':'ad_print','advertisement_id':2,'contour_key':'ISU_S'},
             {'kind':'ad_edge','advertisement_id':2,'role':'top_edge'},
             {'kind':'ice_curve','curve_id':'B1:segment0'}]
    initial=state.world_paths(state.initial_q,targets);changed=state.world_paths(q,targets)
    assert not np.array_equal(initial[0],changed[0])
    np.testing.assert_array_equal(initial[1],changed[1]);np.testing.assert_array_equal(initial[2],changed[2])
    np.testing.assert_array_equal(state.camera25(q,201),state.camera25(state.initial_q,201))
    assert state.templates==old.templates
    assert state.templates_at(q)['advertisements']==old.templates['advertisements']
    np.testing.assert_array_equal(state.parent_projection(q),state.parent_q)
    scene=state.review_scene(q)
    pattern=next(p for p in scene['source_conditioned_board_prints']['patterns'] if p['advertisement_id']==2)
    assert pattern['retain_outer_canvas_border'] is True
    printed=next(c for c in pattern['external_contours'] if c['id']=='ISU_S')
    np.testing.assert_array_equal(np.asarray(printed['xyz_scene_units']),changed[0])


def test_layout_world_chain_matches_full_world_FD_and_closed_inventory(states):
    _,state=states;q=state.initial_q.copy();q[5119:]=[.001,.002,-.001,.001,.002,.001,.0002,-.0001]
    targets=[{'kind':'ad_print','advertisement_id':2,'contour_key':key}
             for key in ('ISU_S','ISU_I','COM_O_inner')]
    targets.append({'kind':'ad_edge','advertisement_id':2,'role':'top_edge'})
    derivatives=state.print_layout_world_jacobians(q,targets)
    assert not np.any(derivatives[-1])
    for j in range(8):
        plus=q.copy();minus=q.copy();plus[5119+j]+=1e-6;minus[5119+j]-=1e-6
        a=state.world_paths(plus,targets);b=state.world_paths(minus,targets)
        for d,p,m in zip(derivatives[:-1],a[:-1],b[:-1],strict=True):
            np.testing.assert_allclose(d[...,j],(p-m)/(2e-6),rtol=3e-7,atol=3e-8)
            np.testing.assert_array_equal(d[-1],d[0])


def test_layout_guard_is_typed_and_RF4_does_not_read_print_tail(states):
    _,state=states;q=state.initial_q.copy()
    # This very large positive scale is not a camera focal coordinate. It is
    # rejected for actual printed material leaving its normalized canvas.
    q[5119]=40.
    with pytest.raises(GeometryDomainError,match='Ad2 print|homography'):
        state.guard(q)
    # Negative homogeneous denominators are a physical layout domain error.
    q=state.initial_q.copy();q[5119+6]=-3.
    with pytest.raises(GeometryDomainError,match='homography'):
        state.guard_geometry(q)


def test_shared_layout_global_CSR_columns_are_the_tail_without_RF4_overlap(states):
    _,state=states
    target={'kind':'ad_print','advertisement_id':2,'contour_key':'ISU_S'};key=fit._target_key(target)
    world=state.world_paths(state.initial_q,[target])[0]
    camera=state.camera25(state.initial_q,201)
    from calibration.program_tracking.first_shot_source_kernel import _interfaces
    project=_interfaces()[0];uv=project(camera,world,with_jacobian=False)[0]
    row=dict(target=target,_target_key=key,uv=uv,sigma=np.ones(len(uv)),point_correspondence=False)
    frame=dict(index=201,rows=[row],source_point_count=len(uv),whitening_groups=[])
    whitening=fit.FrozenWhitening(frame)
    derivatives=np.zeros(world.shape+(126,));derivatives[...,118:]=state.print_layout_world_jacobians(state.initial_q,[target])[0]
    raw,j=fit._evaluate_frame(state,state.initial_q,frame,whitening,{key:world},{key:derivatives})
    assert j.shape==(len(raw),5127) and j.has_sorted_indices and j.has_canonical_format
    assert j[:,5119:].nnz>0
    assert j[:,127:927].nnz==0
    assert j[:,931:5119].nnz==0
    assert not np.any(j[:,:118].data)


@pytest.mark.parametrize('numbers',[(2,3,1,7),(2,3,8,7),(2,3,12,7),(2,2),(),(True,)])
def test_generic_layout_configuration_cannot_change_other_material_interfaces(numbers):
    cfg=dict(schema='rinkspace.first-shot-print-layouts.v1',layouts=[dict(advertisement_id=n) for n in numbers])
    with pytest.raises(ValueError):_print_layout_ids(cfg)


def test_four_print_layouts_keep_exact_original_model_and_old_view_indices(states,generic_state):
    old,_=states;s=generic_state
    assert s.count==5151 and s.world_parameter_count==150 and s.global_shared_count==159
    assert s.view_stop==5119 and s.print_layout_advertisement_ids==(2,3,6,7)
    np.testing.assert_array_equal(s.world_parameter_indices,np.r_[np.arange(118),np.arange(5119,5151)])
    np.testing.assert_array_equal(s.global_shared_indices,np.r_[np.arange(127),np.arange(5119,5151)])
    np.testing.assert_array_equal(s.initial_q[:5119],old.initial_q)
    np.testing.assert_array_equal(s.parent_projection(s.initial_q),old.parent_q)
    assert s.scene(s.initial_q)==old.scene(old.initial_q)
    assert s.templates_at(s.initial_q)==old.templates
    descriptors=list(old.registry['targets'].values())
    for a,b in zip(old.world_paths(old.initial_q,descriptors),s.world_paths(s.initial_q,descriptors),strict=True):
        np.testing.assert_array_equal(a,b)
    for index in (1,201,1248):
        np.testing.assert_array_equal(s.view_columns(index),old.view_columns(index))
        np.testing.assert_array_equal(s.camera25(s.initial_q,index),old.camera25(old.initial_q,index))
    # Public guard records must remain JSON serializable for branch evidence.
    guarded=s.guard(s.initial_q);json.dumps(guarded,allow_nan=False)
    assert set(guarded['shared_print_layout_domain'])=={'2','3','6','7'}


@pytest.mark.parametrize('number',[2,3,6,7])
def test_each_print_layout_moves_only_its_own_ink_and_matches_world_FD(generic_state,number):
    s=generic_state;q=s.initial_q.copy();cols=s.print_layout_global_indices_by_ad[number]
    q[cols]=[.001,.002,-.001,.001,.002,.001,.0002,-.0001]
    templates=s.templates
    targets=[{'kind':'ad_print','advertisement_id':n,'contour_key':next(iter(
        templates.get('advertisements',{}).get(str(n),templates)['contours']))} for n in (2,3,6,7)]
    targets += [{'kind':'ad_edge','advertisement_id':n,'role':'top_edge'} for n in (2,3,6,7)]
    before=s.world_paths(s.initial_q,targets);after=s.world_paths(q,targets)
    changed=[not np.array_equal(a,b) for a,b in zip(before,after,strict=True)]
    assert changed==[n==number for n in (2,3,6,7)]+[False]*4
    current=s.templates_at(q)
    for n in (3,4,5,6,7,12):
        if n != number:assert current['advertisements'][str(n)]==templates['advertisements'][str(n)]
    if number != 2:assert current['contours']==templates['contours']
    derivative=s.print_layout_world_jacobians(q,targets)
    relative=s.print_layout_tail_slices[number]
    for i,d in enumerate(derivative):
        if i>=4:
            assert not np.any(d)
        else:
            own=s.print_layout_tail_slices[targets[i]['advertisement_id']]
            outside=np.ones(32,bool);outside[own]=False
            assert not np.any(d[...,outside])
            assert np.any(d[...,own])
    slot=(2,3,6,7).index(number)
    for j,col in enumerate(cols):
        a=q.copy();b=q.copy();a[col]+=1e-6;b[col]-=1e-6
        forward=s.world_paths(a,[targets[slot]])[0];backward=s.world_paths(b,[targets[slot]])[0]
        np.testing.assert_allclose(derivative[slot][...,relative.start+j],(forward-backward)/(2e-6),rtol=4e-7,atol=4e-8)
    reviewed=s.review_scene(q)
    assert reviewed['source_conditioned_board_prints']['retain_outer_canvas_border_advertisement_ids']==[2,3,6,7]
    before_patterns=s.scene(q)['source_conditioned_board_prints']['patterns']
    assert reviewed['source_conditioned_board_prints']['patterns'][:len(before_patterns)]==before_patterns
    patterns={p['advertisement_id']:p for p in reviewed['source_conditioned_board_prints']['patterns']
              if p['advertisement_id'] in (2,3,6,7)}
    assert set(patterns)=={2,3,6,7}
    for i,n in enumerate((2,3,6,7)):
        assert patterns[n]['retain_outer_canvas_border']
        path=next(c for c in patterns[n]['external_contours'] if c['id']==targets[i]['contour_key'])
        np.testing.assert_array_equal(np.asarray(path['xyz_scene_units']),after[i])


@pytest.mark.parametrize('number',[2,3,6,7])
def test_four_print_layouts_have_explicit_tail_CSR_and_typed_domain(generic_state,number):
    s=generic_state;target={'kind':'ad_print','advertisement_id':number,'contour_key':next(iter(
        s.templates.get('advertisements',{}).get(str(number),s.templates)['contours']))};key=fit._target_key(target)
    world=s.world_paths(s.initial_q,[target])[0]
    from calibration.program_tracking.first_shot_source_kernel import _interfaces
    uv=_interfaces()[0](s.camera25(s.initial_q,201),world,with_jacobian=False)[0]
    row=dict(target=target,_target_key=key,uv=uv,sigma=np.ones(len(uv)),point_correspondence=False)
    frame=dict(index=201,rows=[row],source_point_count=len(uv),whitening_groups=[])
    d=np.zeros(world.shape+(150,));d[...,118:]=s.print_layout_world_jacobians(s.initial_q,[target])[0]
    _,matrix=fit._evaluate_frame(s,s.initial_q,frame,fit.FrozenWhitening(frame),{key:world},{key:d})
    assert matrix.shape[1]==5151 and matrix.has_sorted_indices and matrix.has_canonical_format
    assert matrix[:,s.print_layout_global_indices_by_ad[number]].nnz>0
    for other in (2,3,6,7):
        if other!=number:assert matrix[:,s.print_layout_global_indices_by_ad[other]].nnz==0
    assert matrix[:,127:927].nnz==matrix[:,931:5119].nnz==0
    invalid=s.initial_q.copy();invalid[s.print_layout_global_indices_by_ad[number][6]]=-3.
    with pytest.raises(GeometryDomainError,match='homography'):
        s.guard_geometry(invalid)


def test_all_six_layout_indices_zero_exact_and_previous_configurations_supported(states,all_layout_state):
    old,_=states;s=all_layout_state
    assert s.count==5167 and s.world_parameter_count==166 and s.global_shared_count==175
    assert s.print_layout_advertisement_ids==(2,3,4,5,6,7)
    np.testing.assert_array_equal(s.world_parameter_indices,np.r_[np.arange(118),np.arange(5119,5167)])
    np.testing.assert_array_equal(s.global_shared_indices,np.r_[np.arange(127),np.arange(5119,5167)])
    assert _print_layout_ids(dict(schema='rinkspace.first-shot-print-layouts.v1',layouts=[dict(advertisement_id=n) for n in (2,3,5,6,7)]))==(2,3,5,6,7)
    np.testing.assert_array_equal(s.initial_q[:5119],old.initial_q)
    np.testing.assert_array_equal(s.parent_projection(s.initial_q),old.parent_q)
    assert s.scene(s.initial_q)==old.scene(old.initial_q)
    targets=list(old.registry['targets'].values())
    for a,b in zip(old.world_paths(old.initial_q,targets),s.world_paths(s.initial_q,targets),strict=True):
        np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(s.camera25(s.initial_q,1248),old.camera25(old.initial_q,1248))
    domains=s.guard(s.initial_q)['shared_print_layout_domain'];json.dumps(domains,allow_nan=False)
    assert set(domains)=={str(n) for n in (2,3,4,5,6,7)}
    ad5=domains['5']['actual_continuous_board_material_domain']
    assert ad5['valid'] and ad5['material_u_range'][0]<0 and ad5['material_u_range'][1]>1
    assert ad5['absolute_height_range_scene_units']==pytest.approx([1.0910871626,2.5590439239])


@pytest.mark.parametrize('number',[4,5])
def test_new_Ad4_Ad5_layout_world_FD_keeps_named_frame_and_has_own_tail_CSR(all_layout_state,number):
    s=all_layout_state;material=s.templates['advertisements'][str(number)];key=next(iter(material['contours']))
    target={'kind':'ad_print','advertisement_id':number,'contour_key':key};frame_target={'kind':'ad_edge','advertisement_id':number,'role':'top_edge'}
    q=s.initial_q.copy();columns=s.print_layout_global_indices_by_ad[number]
    q[columns]=[.001,.002,-.001,.001,.002,.001,.0002,-.0001]
    before=s.world_paths(s.initial_q,[target,frame_target]);after=s.world_paths(q,[target,frame_target])
    assert not np.array_equal(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1])
    np.testing.assert_array_equal(q[54:62],s.initial_q[54:62])
    current=s.templates_at(q)['advertisements'][str(number)]
    for field,value in material.items():
        if field!='contours':assert current[field]==value
    derivative=s.print_layout_world_jacobians(q,[target,frame_target]);relative=s.print_layout_tail_slices[number]
    assert derivative[0].shape==after[0].shape+(48,) and not np.any(derivative[1])
    outside=np.ones(48,bool);outside[relative]=False;assert not np.any(derivative[0][...,outside])
    for j,col in enumerate(columns):
        plus=q.copy();minus=q.copy();plus[col]+=1e-6;minus[col]-=1e-6
        a=s.world_paths(plus,[target])[0];b=s.world_paths(minus,[target])[0]
        np.testing.assert_allclose(derivative[0][...,relative.start+j],(a-b)/(2e-6),rtol=4e-7,atol=4e-8)
    from calibration.program_tracking.first_shot_source_kernel import _interfaces
    uv=_interfaces()[0](s.camera25(q,201),after[0],with_jacobian=False)[0];tk=fit._target_key(target)
    row=dict(target=target,_target_key=tk,uv=uv,sigma=np.ones(len(uv)),point_correspondence=False)
    frame=dict(index=201,rows=[row],source_point_count=len(uv),whitening_groups=[])
    d=np.zeros(after[0].shape+(166,));d[...,118:]=derivative[0]
    _,matrix=fit._evaluate_frame(s,q,frame,fit.FrozenWhitening(frame),{tk:after[0]},{tk:d})
    assert matrix.shape[1]==5167 and matrix.has_sorted_indices and matrix.has_canonical_format
    assert matrix[:,columns].nnz>0
    for other in (2,3,4,5,6,7):
        if other!=number:assert matrix[:,s.print_layout_global_indices_by_ad[other]].nnz==0
    scene=s.review_scene(q);desc=scene['source_conditioned_board_prints']
    assert desc['retain_outer_canvas_border_advertisement_ids']==[2,3,4,5,6,7]
    item=next(p for p in desc['patterns'] if p['advertisement_id']==number)
    assert item['material_frame']==('named_internal_graphic_extended' if number==5 else 'existing_outer_canvas')


def test_Ad5_extended_ink_domain_allows_seam_and_rejects_height_or_multiple_turns(all_layout_state):
    s=all_layout_state;q=s.initial_q.copy();material=s.templates['advertisements']['5']
    uv=np.concatenate([np.asarray(c['material_uv'],float) for c in material['contours'].values()])
    original=s.print_material_domain_status(q,5,uv)
    assert original['valid'] and original['material_u_range'][0]<0 and original['material_u_range'][1]>1
    # A whole-period station shift is a legal gauge; seam location cannot be
    # turned into a false finite-open-chain domain bound.
    from calibration.program_tracking.ad_landmarks import AdGeometry
    geometry=AdGeometry(s.scene(q),5);shift=geometry._perimeter/(geometry._width_sign*geometry._nominal_width)
    crossed=uv.copy();crossed[:,0]+=shift
    assert s.print_material_domain_status(q,5,crossed)['valid']
    assert not s.print_material_domain_status(q,5,np.c_[uv[:,0]*1000,uv[:,1]])['valid']
    assert not s.print_material_domain_status(q,5,np.c_[uv[:,0],uv[:,1]+10])['valid']
    invalid=q.copy();invalid[s.print_layout_global_indices_by_ad[5][5]]=10.
    with pytest.raises(GeometryDomainError,match='actual continuous board material domain'):
        s.guard_geometry(invalid)
    invalid=q.copy();invalid[s.print_layout_global_indices_by_ad[5][0]]=1000.
    with pytest.raises(GeometryDomainError,match='spans an entire perimeter'):
        s.guard_geometry(invalid)
