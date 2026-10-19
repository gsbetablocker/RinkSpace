from copy import deepcopy
import hashlib
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.solver_runtime import install
install()
from calibration.solver_runtime import original_measured_location_mode as mode
from calibration.solver_runtime.original_absolute_input_compaction import slice_observed_points
from projection import project_central_valid, from_state, to_state
from source_edge_intervals import SourceNormalIntervals
from z190_source_camera_diagnostic import absolute_value_jac15, _curve_vector_error15
from main9_typed_source_adapter import typed_absolute_value_jac15
import z190_secondary_original_ring_source_qualification as secondary
from z190_secondary_connected_cap_source import connected_cap_absolute_value_jac15


def camera():
    c=np.zeros(25);c[:3]=[.17,-.09,-10.];c[3:6]=[.012,-.023,.017]
    c[6:8]=np.log([1210.,1190.]);c[8:11]=[960.,540.,.3]
    c[11:15]=[.01,-.001,.0002,-.0003]
    return c


def ordinary_row(c):
    world=np.array([[-1.,0.,0.],[1.,0.,0.]])
    p=project_central_valid(c,world)[0]
    uv=p[0]+np.array([.3,.7])[:,None]*(p[1]-p[0])
    normal=np.array([-(p[1]-p[0])[1],(p[1]-p[0])[0]]);normal/=np.linalg.norm(normal)
    uv=uv+np.array([2.,-3.])[:,None]*normal
    return dict(id='advertisement7.left_outer_boundary',pts=100,split='fit',world=world,
        uv=uv,sigma=np.array([2.,3.]),normal=np.tile(normal,(2,1)),
        normal_interval=np.tile([-10.,10.],(2,1)),point_correspondence=False)


def main9_row(c):
    world=np.array([[-1.,-.7,0.],[1.,.7,0.]])
    p=project_central_valid(c,world)[0]
    uv=p[0]+np.array([.3,.7])[:,None]*(p[1]-p[0]);uv[:,0]+=[3.,-2.]
    return dict(id='C',pts=100,split='fit',world=world,uv=uv,sigma=np.array([2.,3.]),
        normal=np.tile([1.,0.],(2,1)),normal_interval=np.tile([-10.,10.],(2,1)),
        curve_observation_kind='axis_scanline_branch',scanline_fixed_axis='y',scanline_branch='unique_x')


def secondary_row(c,branch='min_y',outside=False):
    a=np.linspace(0,2*np.pi,257);world=np.c_[np.cos(a),.6*np.sin(a),np.zeros(len(a))]
    p=project_central_valid(c,world)[0];lo,hi=np.min(p[:,0]),np.max(p[:,0])
    x=np.array([lo-7.,hi+11.]) if outside else np.linspace(lo+20.,hi-20.,7)
    return dict(id='R3.outer',pts=100,split='development',world=world,
        uv=np.c_[x,np.full(len(x),510. if branch=='min_y' else 575.)],sigma=np.linspace(1.5,2.5,len(x)),
        normal=np.tile([0.,1.],(len(x),1)),normal_interval=np.tile([-150.,150.],(len(x),1)),
        original_outer_source_qualification_operator=secondary.STRICT_KIND,
        curve_observation_kind='axis_scanline_branch',scanline_fixed_axis='x',scanline_branch=branch,
        must_preserve_upper_branch=branch=='min_y')


def measured(row,tmp_path):
    record=dict(source_row_id=row['id'],original_source_row_sha256='qualified-original-row',measurement_bindings=[0])
    path=tmp_path/'authority.json';path.write_text(json.dumps({'records':[record]}),encoding='utf8')
    ref=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return mode.install_measured_location_mode(row,authority_ref=ref,source_row_index=0,qualified_receipt_record=record)


@pytest.mark.parametrize('factory,evaluator',[(ordinary_row,absolute_value_jac15),(main9_row,typed_absolute_value_jac15)])
def test_qualified_locations_restore_exact_existing_raw_values_and_camera_J(tmp_path,factory,evaluator):
    c=camera();old=factory(c);before=deepcopy(old);new=measured(old,tmp_path)
    raw=dict(old);raw.pop('normal_interval')
    expected=evaluator(c,[raw],True);actual=evaluator(c,[new],True)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(evaluator(c,[new],False),expected[0])
    assert np.linalg.norm(actual[0])>.5
    assert np.linalg.norm(evaluator(c,[old],False))<1e-12
    for k in ['uv','sigma','normal','normal_interval','world']:
        np.testing.assert_array_equal(old[k],before[k]);np.testing.assert_array_equal(new[k],before[k])
    # Independent full-Brown physical-camera finite differences, all 15 columns.
    state=np.r_[to_state(c),c[11:15]];numeric=np.empty_like(actual[1]);h=1e-7
    for col in range(15):
        d=np.eye(15)[col]*h
        numeric[:,col]=(evaluator(from_state(state+d,c),[new],False)-evaluator(from_state(state-d,c),[new],False))/(2*h)
    np.testing.assert_allclose(actual[1],numeric,rtol=3e-5,atol=3e-5)


@pytest.mark.parametrize('branch',['min_y','max_y'])
@pytest.mark.parametrize('outside',[False,True])
def test_secondary_keeps_same_branch_and_finite_extent_raw_before_support(tmp_path,branch,outside):
    c=camera();old=secondary_row(c,branch,outside);new=measured(old,tmp_path)
    value,jac=secondary.qualified_original_ring_absolute_value_jac15(c,[new],True)
    if outside:
        p=project_central_valid(c,old['world'])[0]
        selected=secondary._source_branch_extent_indices(p,branch)
        expected=((p[selected]-old['uv'])/old['sigma'][:,None]).ravel()
        np.testing.assert_array_equal(value,expected)
    else:
        raw=dict(old);raw.pop('normal_interval')
        expected,expected_J=connected_cap_absolute_value_jac15(c,[raw],True)
        np.testing.assert_array_equal(value,expected);np.testing.assert_array_equal(jac,expected_J)
    state=np.r_[to_state(c),c[11:15]];numeric=np.empty_like(jac);h=1e-7
    for col in range(15):
        d=np.eye(15)[col]*h
        numeric[:,col]=(secondary.qualified_original_ring_absolute_value_jac15(from_state(state+d,c),[new],False)
            -secondary.qualified_original_ring_absolute_value_jac15(from_state(state-d,c),[new],False))/(2*h)
    np.testing.assert_allclose(jac,numeric,rtol=3e-5,atol=3e-5)


@pytest.mark.parametrize('factory,evaluator',[(ordinary_row,absolute_value_jac15),(main9_row,typed_absolute_value_jac15),
    (secondary_row,secondary.qualified_original_ring_absolute_value_jac15)])
@pytest.mark.parametrize('jacobian',[False,True])
def test_naked_metadata_cannot_bypass_original_support(factory,evaluator,jacobian):
    r=factory(camera());r[mode.MODE_FIELD]=mode.MODE
    r[mode.AUTHORITY_FIELD]={'schema':mode.SCHEMA,'authenticated':True}
    with pytest.raises(ValueError,match='authenticated installer certificate'):evaluator(camera(),[r],jacobian)


def test_legacy_support_values_and_J_are_bit_exact():
    c=camera();row=ordinary_row(c)
    raw_px,raw_J=_curve_vector_error15(c,row['world'],row['uv'])
    support=SourceNormalIntervals(row['normal'],row['normal_interval'],np.ones(len(row['uv']),bool))
    supported,matrices=support.apply(raw_px,[raw_J])
    actual=absolute_value_jac15(c,[row],True)
    np.testing.assert_array_equal(actual[0],supported/np.repeat(row['sigma'],2))
    np.testing.assert_array_equal(actual[1],matrices[0]/np.repeat(row['sigma'],2)[:,None])


def test_certificate_pickle_compaction_world_rebind_and_source_identity(tmp_path):
    from structured_programme_source_candidate.complete_same_state_replay import _source_record_identity
    old=ordinary_row(camera());new=measured(old,tmp_path)
    original_identity=_source_record_identity(new)
    restored=pickle.loads(pickle.dumps(new));restored['world']=restored['world']+1.
    assert mode.uses_original_normal_support(restored) is False
    assert _source_record_identity(restored[mode.AUTHORITY_FIELD])==_source_record_identity(new[mode.AUTHORITY_FIELD])
    compacted=slice_observed_points(restored,np.array([1]))
    assert mode.uses_original_normal_support(compacted) is False
    assert original_identity['authenticated_measured_location']['authenticated_measured_location']['original_point_count']==2
    compacted['uv'][0,0]+=.01
    with pytest.raises(ValueError,match='UV/sigma/normal/support changed'):mode.uses_original_normal_support(compacted)


@pytest.mark.parametrize('field',['uv','sigma','normal','normal_interval','scanline_branch'])
def test_certificate_rejects_changed_original_observed_attributes(tmp_path,field):
    new=measured(main9_row(camera()),tmp_path)
    if field=='scanline_branch':new[field]='nearest'
    else:
        new[field]=np.asarray(new[field]).copy();new[field].flat[0]+=.01
    with pytest.raises(ValueError):mode.uses_original_normal_support(new)


def test_source_qualification_accepts_additive_authority_fields(tmp_path):
    from calibration.solver_runtime.original_source_point_qualification import _verify_runtime_row
    old=ordinary_row(camera());new=measured(old,tmp_path)
    _verify_runtime_row(new,old,{'original_packet_row_index':0},[0],SimpleNamespace())
    assert mode.uses_original_normal_support(new) is False


def test_other_typed_interval_likelihood_is_not_silently_bypassed(tmp_path):
    r=ordinary_row(camera());r['curve_observation_kind']='confirmed_lower_long_side_top_horizontal_support'
    with pytest.raises(ValueError,match='own measured-axis qualification'):measured(r,tmp_path)


def test_installer_requires_exact_authority_bytes_and_existing_interval(tmp_path):
    row=ordinary_row(camera());path=tmp_path/'authority.json';path.write_text('{}')
    with pytest.raises(ValueError,match='authority bytes changed'):
        mode.install_measured_location_mode(row,authority_ref={'path':str(path),'sha256':'wrong'},source_row_index=0,qualified_receipt_record={'q':1})
    row.pop('normal_interval')
    with pytest.raises(ValueError,match='original optical normal interval'):measured(row,tmp_path)


@pytest.mark.parametrize('native_mode',['measured_transition_midpoint','measured_transition_location'])
def test_native_board_modes_reach_original_absolute_without_claiming_original_mode(native_mode,monkeypatch):
    monkeypatch.setenv('RINKSPACE_PROJECTION_BACKEND','numpy')
    from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
    factor=NativeBoardEdgeFactor.__new__(NativeBoardEdgeFactor);factor.parameter_count=1
    factor.rows=[dict(id='native_board_transition',actual_pts=100,curve={'kind':'advertisement_boundary'},
        uv=np.array([[865.,540.]]),sigma=np.array([8.]),normal=np.array([[1.,0.]]),
        normal_interval=np.array([[-10.,10.]]),curve_observation_kind='normal_interval',
        boundary_location_observation=native_mode)]
    factor._world=lambda q,curve:np.array([[-1.,-1.,10.],[-1.,1.,10.]])
    def cameras(q):
        c=np.zeros(25);c[6:8]=np.log(1000.);c[8:10]=[960.+50.*q[0],540.]
        return {100:c}
    factor._cameras=cameras
    # This traverses NativeBoardEdgeFactor._source_rows -> the actual shared
    # absolute evaluator, where the original-only authority must not collide.
    assert mode.uses_original_normal_support(factor.rows[0]) is True
    np.testing.assert_array_equal(factor._residual(np.zeros(1),factor.rows),[-5./8.,0.])
    np.testing.assert_allclose(factor._residual(np.array([.1]),factor.rows),0.,atol=1e-12)
    q=np.zeros(1);association=factor._nominal_associations(q,cameras(q));h=1e-5
    observed=(factor._residual(q+h,factor.rows)-factor._residual(q-h,factor.rows))/(2*h)
    replay=(factor._derivative_residual(q+h,cameras=cameras(q+h),associations=association)
        -factor._derivative_residual(q-h,cameras=cameras(q-h),associations=association))/(2*h)
    np.testing.assert_allclose(replay,observed,atol=1e-9,rtol=0.)
