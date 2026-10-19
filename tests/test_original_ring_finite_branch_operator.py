"""Source-branch domain regression using the actual pinhole/Brown projector."""
from copy import deepcopy
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

PROGRAM=Path(__file__).resolve().parents[1]/'calibration/program_tracking'
if str(PROGRAM)not in sys.path:
    sys.path.insert(0,str(PROGRAM))
api=importlib.import_module('z190_secondary_original_ring_source_qualification')
from projection import from_state,project_central_valid,to_state
from reference_board_pair import BoardIntersectionError
from z190_secondary_connected_cap_source import connected_cap_absolute_value_jac15


def camera():
    result=np.zeros(25)
    result[:3]=[.17,-.09,-10.]
    result[3:6]=[.012,-.023,.017]
    result[6:8]=np.log([1210.,1190.])
    result[8:11]=[960.,540.,.3]
    result[11:15]=[.01,-.001,.0002,-.0003]
    return result


def row(camera,branch='min_y',outside=False):
    angle=np.linspace(0.,2.*np.pi,257)
    world=np.c_[np.cos(angle),.6*np.sin(angle),np.zeros_like(angle)]
    projected=project_central_valid(camera,world)[0]
    low,high=projected[:,0].min(),projected[:,0].max()
    x=np.array([low-7.,high+11.])if outside else np.linspace(low+20.,high-20.,7)
    uv=np.c_[x,np.full(len(x),510. if branch=='min_y'else 575.)]
    return dict(world=world,uv=uv,sigma=np.linspace(1.5,2.5,len(uv)),
        normal=np.tile([0.,1.],(len(uv),1)),normal_interval=np.tile([.15,.7],(len(uv),1)),
        original_outer_source_qualification_operator=api.STRICT_KIND,
        curve_observation_kind='axis_scanline_branch',scanline_fixed_axis='x',
        scanline_branch=branch,must_preserve_upper_branch=branch=='min_y')


def legacy_inside_value_jac(camera,row):
    """Frozen pre-revision contract: base scanline then normalized support."""
    rawrow=dict(row)
    rawrow.pop('normal_interval')
    value,matrix=connected_cap_absolute_value_jac15(camera,[rawrow],True)
    raw=value.reshape(-1,2).copy()
    normals=np.asarray(row['normal'])
    limits=np.asarray(row['normal_interval'])/row['sigma'][:,None]
    signed=np.einsum('ni,ni->n',raw,normals)
    raw-=normals*np.clip(signed,limits[:,0],limits[:,1])[:,None]
    selected=(signed>limits[:,0])&(signed<limits[:,1])
    mapped=matrix.reshape(len(raw),2,15).copy()
    n=normals[selected]
    component=np.einsum('ni,nij->nj',n,mapped[selected])
    mapped[selected]-=n[:,:,None]*component[:,None,:]
    return raw.ravel(),mapped.reshape(matrix.shape)


@pytest.mark.parametrize('branch',['min_y','max_y'])
def test_inside_original_values_and_jacobian_are_exact_and_inputs_immutable(branch):
    c=camera();r=row(c,branch);before=deepcopy(r)
    expected=legacy_inside_value_jac(c,r)
    observed=api.qualified_original_ring_absolute_value_jac15(c,[r],True)
    for old,new in zip(expected,observed):
        np.testing.assert_array_equal(new,old)
    np.testing.assert_array_equal(api.qualified_original_ring_absolute_value_jac15(c,[r],False),expected[0])
    for key in r:
        if isinstance(r[key],np.ndarray):np.testing.assert_array_equal(r[key],before[key])
        else:assert r[key]==before[key]


@pytest.mark.parametrize('branch',['min_y','max_y'])
def test_outside_preserves_horizontal_error_and_all_fifteen_derivatives(branch):
    c=camera();r=row(c,branch,True)
    value,jac=api.qualified_original_ring_absolute_value_jac15(c,[r],True)
    assert np.isfinite(value).all()and np.isfinite(jac).all()
    np.testing.assert_allclose(value[0::2],[7./1.5,-11./2.5],rtol=0.,atol=1e-12)
    state=np.r_[to_state(c),c[11:15]]
    numeric=np.empty_like(jac)
    step=1e-7
    for k in range(15):
        delta=np.zeros(15);delta[k]=step
        plus=api.qualified_original_ring_absolute_value_jac15(from_state(state+delta,c),[r],False)
        minus=api.qualified_original_ring_absolute_value_jac15(from_state(state-delta,c),[r],False)
        numeric[:,k]=(plus-minus)/(2.*step)
    np.testing.assert_allclose(jac,numeric,rtol=2e-5,atol=2e-5)


@pytest.mark.parametrize('branch',['min_y','max_y'])
@pytest.mark.parametrize('side',['left','right'])
def test_same_physical_branch_is_continuous_at_extent(branch,side):
    c=camera();r=row(c,branch,True)
    projected=project_central_valid(c,r['world'])[0]
    x=projected[:,0].min()if side=='left'else projected[:,0].max()
    r['uv']=np.array([[x,530.]])
    r['sigma']=np.ones(1)
    r['normal']=np.array([[0.,1.]])
    r['normal_interval']=np.array([[.15,.7]])
    edge=api.qualified_original_ring_absolute_value_jac15(c,[r],False)
    # Both sides approach the same branch frontier; no source UV is clipped.
    for sign in (-1.,1.):
        errors=[]
        for distance in (1e-5,1e-7,1e-9):
            near=deepcopy(r);near['uv'][0,0]+=sign*distance
            observed=api.qualified_original_ring_absolute_value_jac15(c,[near],False)
            errors.append(np.linalg.norm(observed-edge))
        assert errors[2]<errors[1]<errors[0]
        assert errors[2]<2e-5


def test_vertical_extent_tie_uses_source_branch_not_nearest_source_y():
    c=np.zeros(25);c[2]=-10.;c[6:8]=np.log(1000.);c[8:10]=[960.,540.]
    world=np.array([[-1.,-1.,0.],[-1.,1.,0.],[1.,1.,0.],[1.,-1.,0.],[-1.,-1.,0.]])
    results={}
    for branch in ('min_y','max_y'):
        r=dict(world=world,uv=np.array([[850.,630.]]),sigma=np.ones(1),
            normal=np.array([[0.,1.]]),normal_interval=np.array([[-.1,.1]]),
            original_outer_source_qualification_operator=api.STRICT_KIND,
            curve_observation_kind='axis_scanline_branch',scanline_fixed_axis='x',
            scanline_branch=branch,must_preserve_upper_branch=branch=='min_y')
        results[branch]=api.qualified_original_ring_absolute_value_jac15(c,[r],False)
    np.testing.assert_allclose(results['min_y'],[10.,-189.9])
    np.testing.assert_allclose(results['max_y'],[10.,9.9])


def test_camera_behind_curve_is_still_a_real_domain_failure():
    c=camera();r=row(c,outside=True);c[2]=10.
    for jacobian in (False,True):
        with pytest.raises(BoardIntersectionError,match='qualified_outer_curve_not_central_forward'):
            api.qualified_original_ring_absolute_value_jac15(c,[r],jacobian)
