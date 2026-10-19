from pathlib import Path
from types import SimpleNamespace
import ctypes
import json
import numpy as np
import pytest

from calibration.solver_runtime import runtime
runtime.install()
from calibration.solver_runtime.motion_bounded_schur_terms import (
    native_chunk_terms,prepare_programme_bounded_terms,
)
import z190_motion_native as native
from secondary_optics_candidate.z190_motion_optics_extension import fixed_lambda_residual,_line_geometry


def inputs():
    import cv2
    va=np.array([.02,-.03,.01]);vb=np.array([.025,-.029,.015])
    ra,da=cv2.Rodrigues(va);rb,db=cv2.Rodrigues(vb)
    a=np.array([[700.,450.],[850.,510.],[1010.,550.],[1120.,390.],[660.,650.]])
    n=len(a);ca=np.array([0.,1.,.5]);cb=np.array([.04,1.001,.499])
    dca=np.zeros((3,4));dcb=np.zeros((3,4));dca[:,:3]=np.eye(3)*.015;dcb[:,:3]=np.eye(3)*.014
    sa=np.zeros((3,4));sb=np.zeros((3,4));sa[:,:3]=np.eye(3)*.03;sb[:,:3]=np.eye(3)*.025
    k=dict(a=a,b=np.zeros((n,2)),fa=1100.,fb=1110.,ra=ra,rb=rb,da=da.reshape(3,3,3),db=db.reshape(3,3,3),
        ca=ca,cb=cb,dca_shared=sa,dcb_shared=sb,dca_view=dca,dcb_view=dcb,
        pp=np.array([960.,540.]),k1=.13,planes=np.zeros((n,4)),jacobian=True,
        sigma=np.array([[1.2,1.5],[2.,1.1],[.8,1.8],[1.3,1.3],[1.5,1.1]]),
        profile_weight=np.array([[1.,.7],[.8,.9],[1.,1.],[.6,1.],[1.,.8]]))
    geom=_line_geometry(k,np.zeros(n));rho=np.linspace(.08,.18,n)
    lam=rho*geom['length']/(geom['q'][:,2]+rho*geom['t'][:,2])
    k['b']=fixed_lambda_residual(k,lam).reshape(n,2)
    # A small perpendicular error leaves every point interior and stationary.
    perp=np.c_[-geom['u'][:,1],geom['u'][:,0]]
    k['b']+=.015*perp
    return k,va,vb


def test_optional_native_moments_preserve_original_abi_bitwise(monkeypatch):
    if not native.metadata()['available']:pytest.skip('Matching strict native build unavailable')
    k,_,_=inputs()
    ordinary,_,tr=native.native_motion(**k,minimum_source_depth_scene_units=3.,return_trace=True)
    moment,_,tm,state=native.native_motion(**k,minimum_source_depth_scene_units=3.,return_nuisance=True)
    assert all(np.array_equal(a,b)for a,b in zip(ordinary,moment))
    assert np.array_equal(tr,tm)
    assert np.all(tm[:,9]==1)
    # Compare with the unchanged authenticated pre-extension DLL as well.
    directory=runtime.data_root()/'native_motion'
    manifest=json.loads((directory/'build.json').read_text(encoding='utf8'))
    old=ctypes.CDLL(str(directory/manifest['binary'])).motion_strict
    old.argtypes=native._kernel.argtypes;old.restype=ctypes.c_int
    monkeypatch.setattr(native,'_kernel',old)
    previous,_,tp=native.native_motion(**k,minimum_source_depth_scene_units=3.,return_trace=True)
    assert all(np.array_equal(a,b)for a,b in zip(previous,ordinary))
    assert np.array_equal(tp,tr)
    assert state['fixed_lambda_projection_factors'].shape==(len(k['a']),3,4)


def test_all_actual_local_columns_recover_fixed_lambda_derivatives():
    import cv2
    k,va,vb=inputs();unit=.01
    terms=native_chunk_terms(k,minimum_source_depth_scene_units=3.,optics_column_scales=(1.,1.,.001),scene_unit_m=unit)
    assert np.all(terms['nominal_interior'])
    raw=terms['projected_J']+terms['tangent'][:,:,None]*terms['projection_factors'][:,None,:]
    sigma=k['sigma'];h=1e-6;lam=terms['lambda0']
    for column in range(18):
        p=dict(k);m=dict(k)
        if column<4:
            for name,key in (('ca','dca_shared'),('cb','dcb_shared')):
                p[name]=k[name]+h*k[key][:,column];m[name]=k[name]-h*k[key][:,column]
        elif column<12:
            side='a'if column<8 else'b';j=(column-4)%4;v=va if side=='a'else vb
            if j<3:
                inc=np.eye(3)[j]*h;p['r'+side]=cv2.Rodrigues(v+inc)[0];m['r'+side]=cv2.Rodrigues(v-inc)[0]
            else:p['f'+side]=k['f'+side]*np.exp(h);m['f'+side]=k['f'+side]*np.exp(-h)
            p['c'+side]=k['c'+side]+h*k['dc'+side+'_view'][:,j]
            m['c'+side]=k['c'+side]-h*k['dc'+side+'_view'][:,j]
        elif column<15:
            j=column-12
            if j<2:
                p['pp']=k['pp']+h*np.eye(2)[j];m['pp']=k['pp']-h*np.eye(2)[j]
            else:p['k1']=k['k1']+h*.001;m['k1']=k['k1']-h*.001
        else:
            j=column-15
            for side in ('a','b'):
                dc=k['r'+side].T[:,j]/unit
                p['c'+side]=k['c'+side]+h*dc;m['c'+side]=k['c'+side]-h*dc
        fd=((fixed_lambda_residual(p,lam)-fixed_lambda_residual(m,lam))/(2*h)).reshape(-1,2)/sigma
        np.testing.assert_allclose(raw[:,:,column],fd,atol=8e-6,rtol=3e-6)


@pytest.mark.parametrize('jac,schur',[(False,True),(True,False)])
def test_moment_request_cannot_use_an_uneliminated_or_value_call(jac,schur):
    k,_,_=inputs();k['jacobian']=jac
    with pytest.raises(ValueError,match='Schur Jacobian'):
        native.native_motion(**k,schur=schur,return_nuisance=True)


def test_global_preparation_maps_units_and_keeps_outer_metric_and_retirement(monkeypatch):
    import calibration.solver_runtime.motion_bounded_schur_terms as module
    from calibration.solver_runtime.source_loss import SourceLossWeights
    from z190_continuous_source_bundle import SCENE_UNIT_M
    n=3;P=35
    local=dict(r=np.array([[3.,.5],[9.,-3.],[.2,.1]]),tangent=np.ones((n,2)),
        projected_J=np.arange(n*2*18,dtype=float).reshape(n,2,18)/100,
        projection_factors=np.arange(n*18,dtype=float).reshape(n,18)/10,
        lambda0=np.full(n,.1),inner_gradient=np.zeros(n),inner_GN=np.ones(n),
        nominal_interior=np.array([True,True,False]))
    monkeypatch.setattr(module,'native_chunk_terms',lambda *args,**kw:{k:v.copy()for k,v in local.items()})
    bundle=SimpleNamespace(a=np.zeros((n,2)),b=np.zeros((n,2)),ia=np.array([0,0,1]),ib=np.array([1,1,0]),
        motion_rows=6,motion_sigma=np.ones(6),profile_weights=np.full((n,2),.3),finite_planes=np.zeros((n,4)),
        motion_kernel=SimpleNamespace(minimum_source_depth_scene_units=3.))
    bundle.current_calibration=lambda q:SimpleNamespace(principal_x_fraction=(.5,),principal_y_fraction=(.5,),k1=(.1,))
    bundle.cache=lambda q,j:(np.zeros((2,4)),np.broadcast_to(np.eye(3),(2,3,3)),np.zeros((2,3,3,3)),
        np.zeros((2,3)),np.zeros((2,3,4)),np.zeros((2,3,4)))
    owner=SimpleNamespace(name='low',bundle=bundle,quadratic=np.zeros(6,bool),xyz_slots=np.array([0,1,2]),
        start=3,optics_chart=SimpleNamespace(indices=(12,13,14),mount_indices=(15,16,17)),local=lambda q:q)
    state=SimpleNamespace(blocks=[owner],structured_programme_coordinate_chart=SimpleNamespace(core_count=P))
    base=SimpleNamespace(state=state)
    # Canonical permutation changes every complete-column position.
    perm=np.arange(P)[::-1]
    layout=SimpleNamespace(evaluation_q=lambda q:q[perm],evaluation_to_canonical=perm)
    retirement=SimpleNamespace(global_rows=np.array([2,3]))
    inner=SimpleNamespace(parent=base,
        _payloads={'programme:low':SourceLossWeights(np.array([.5,.25,1.,1.,1.,1.]))},
        static_motion_role_retirement=retirement)
    source=SimpleNamespace(base=inner,state=state,count=P,quadratic=np.zeros(6,bool),parameter_layout=layout)
    chunks=prepare_programme_bounded_terms(source,np.zeros(P),profile_token='frozen')
    assert len(chunks)==1
    c=chunks[0];assert c['projection_factors'].shape==(1,P)
    assert c['source_identity']['active_bundle_point_indices'].tolist()==[0]
    assert c['source_identity']['original_precompaction_global_bundle_indices'].tolist()==[0]
    expected=np.zeros(P);idx=np.r_[np.array([0,1,2]),3+np.arange(4),7+np.arange(4),[12,13,14],[15,16,17]]
    keep=np.r_[np.arange(3),np.arange(4,18)]
    expected[perm[idx]]=local['projection_factors'][0,keep]*np.r_[np.full(3,1/(100*SCENE_UNIT_M)),np.ones(14)]
    np.testing.assert_array_equal(c['projection_factors'].toarray()[0],expected)
    # Original outer Huber metric: 2/3*.5 and 1*.25, not frozen .3 twice.
    assert c['outer_GN'][0]==pytest.approx(2/3*.5+.25)
    assert c['outer_gradient'][0]==pytest.approx(3*(2/3*.5)+.5*.25)
    assert c['inner_GN'][0]==1.
