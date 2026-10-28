from pathlib import Path
import json, hashlib
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from ptzcheck.geometry import *
from ptzcheck.solver import Layout, initial, fit_one, observability, subset
from ptzcheck.synthetic import make_graph
from ptzcheck.features import Tracks, partition_track, graph_components, coverage
from ptzcheck.io import fresh_output, assert_unchanged, sha256, select_frames
from ptzcheck.compare import read_current_camera
from ptzcheck.report import model_diagnostics, homography_diagnostic


def test_brown_roundtrip():
    xy=np.random.default_rng(3).uniform(-.5,.5,(100,2));d=np.array([-.12,.025,.002,-.003])
    out,ok=undistort(distort(xy,d),d)
    assert ok.all();assert np.max(abs(out-xy))<1e-10


def test_brown_batched():
    xy=np.random.default_rng(2).uniform(-.3,.3,(100,2));d=np.tile([.02,.01,0,0],(100,1))
    out,ok=undistort(distort(xy,d),d);assert ok.all();np.testing.assert_allclose(out,xy,atol=1e-10)


def test_rotation_convention():
    g,o,R=make_graph();r,v=symmetric_errors(g['data'],o,R)
    assert v.all();assert np.max(abs(r))<1e-9


def test_radial_rotation_convention():
    g,o,R=make_graph(radial=.08);r,v=symmetric_errors(g['data'],o,R)
    assert v.all();assert np.max(abs(r))<1e-8


def test_so3_projection():
    R=nearest_rotation(np.diag([1.,1.,-1.]));assert np.linalg.det(R)>0.999
    np.testing.assert_allclose(R@R.T,np.eye(3),atol=1e-12)


def test_negative_depth_invalid():
    _,ok=project_rays(np.array([[1.,1.,-1.]]),np.array([100,100,50,50,0,0,0,0.]));assert not ok[0]


def test_track_conflict():
    t=Tracks();assert t.union((0,0),(1,0));assert not t.union((0,1),(1,0))
    assert t.root((0,0))!=t.root((0,1))


def test_whole_track_split_deterministic():
    a=[partition_track(str(i),14) for i in range(100)]
    assert a==[partition_track(str(i),14) for i in range(100)];assert 15<sum(a)<45


def test_components_no_false_bridge():
    assert graph_components(5,[(0,1),(1,2),(3,4)])==[[0,1,2],[3,4]]


def test_ptime_not_fps():
    p=dict(pts=[100,110,135,180,200],time_base='1/100')
    ids=select_frames(p,110,200,.4,20)
    assert ids[0]==1 and ids[-1]==4;assert all(p['pts'][i]>=110 for i in ids)


def test_output_never_overwrites(tmp_path):
    source=tmp_path/'input';source.mkdir()
    with pytest.raises(ValueError):fresh_output(source/'result',[source])
    out=fresh_output(tmp_path/'out',[source])
    with pytest.raises(FileExistsError):fresh_output(out,[source])


def test_hash_guard(tmp_path):
    p=tmp_path/'a';p.write_text('original');d={str(p):sha256(p)};assert_unchanged(d)
    p.write_text('changed')
    with pytest.raises(RuntimeError):assert_unchanged(d)


def test_pure_zoom_aspect_unidentifiable():
    g,o,R=make_graph(pure_zoom=True);o2=o.copy();o2[:,0]*=1.3
    r,v=symmetric_errors(g['data'],o2,R)
    assert np.max(abs(r))<1e-9
    # Absolute focal length is also unobservable under pure zoom.
    o2[:,:2]*=4.;r,v=symmetric_errors(g['data'],o2,R);assert np.max(abs(r))<1e-9


def test_layout_has_no_metric_height():
    l=Layout(5,1280,720,'free_pp_radial')
    assert not any('height' in n or 'pivot' in n or 'lever' in n for n in l.names)


def test_sparse_pattern_exact():
    g,o,R=make_graph(n=4,points=10);l=Layout(4,1280,720,'free_pp_radial');x=initial(l,g['meta'],2.)
    p=l.sparsity(g['data']).toarray();a,_=symmetric_errors(g['data'],*l.unpack(x));a=a.ravel()
    for k in range(l.size):
        b=x.copy();b[k]+=1e-6;r,_=symmetric_errors(g['data'],*l.unpack(b));der=(r.ravel()-a)/1e-6
        assert np.max(np.abs(der[p[:,k]==0]),initial=0)<1e-7


def test_exact_recovery_from_images_only():
    g,o,R=make_graph(points=60);l=Layout(7,1280,720,'free_pp')
    fit=fit_one(l,g['meta'],g['data'],focal_widths=2.,max_nfev=200)
    assert fit['summary']['train']['median_px']<1e-4
    assert abs(fit['summary']['fx_over_fy']-1.18)<1e-4
    np.testing.assert_allclose(fit['optics'][:,:4],o[:,:4],rtol=1e-4,atol=.05)


def test_heldout_never_changes_fit():
    g,_,_=make_graph(n=4,points=40);l=Layout(4,1280,720,'square_center')
    a=fit_one(l,g['meta'],g['data'],max_nfev=25)
    g['data']['b'][g['data']['test']]+=200
    b=fit_one(l,g['meta'],g['data'],max_nfev=25)
    np.testing.assert_array_equal(a['x'],b['x'])
    assert b['summary']['holdout_all']['median_px']>100


def test_exact_pts_adapter_and_layout(tmp_path):
    g,o,R=make_graph(n=4);v=np.zeros((4,24));v[:,4:13]=R.reshape(-1,9);v[:,16:24]=o
    p=tmp_path/'c.npz';np.savez(p,source_pts=np.arange(4)*256,instantaneous_camera=v,camera_status=np.zeros(4,int))
    a,b=read_current_camera(p,g['meta']['frames']);np.testing.assert_array_equal(a,o);np.testing.assert_array_equal(b,R)
    g['meta']['frames'][1]['pts']=257
    with pytest.raises(ValueError):read_current_camera(p,g['meta']['frames'])


def test_zero_motion_not_physical_pass():
    g,o,R=make_graph(pure_zoom=True,points=60)
    r,v=symmetric_errors(g['data'],o,R)
    s=dict(optimizer_success=True,bounds_hit=[],local_condition=1e15,holdout_screened=error_stats(r[g['data']['test']]),physical_acceptance=False)
    f=dict(summary=s,residual=r,valid=v,rotations=R)
    hd,_=homography_diagnostic(g);d=model_diagnostics(g,f,hd)
    assert d['status']=='INCONCLUSIVE';assert 'NEAR_ZERO_ROTATION' in d['reasons'];assert not d['physical_camera_certified']


def test_noncentral_motion_not_covered_by_ray_rotation():
    g,o,R=make_graph(n=4,points=40);rng=np.random.default_rng(22)
    xyz=np.column_stack((rng.uniform(-1,1,40),rng.uniform(-.5,.5,40),rng.uniform(5,20,40)))
    C=np.array([[0,0,0],[.2,0,0],[0,.1,0],[.2,.1,0]])
    uv=np.array([project_rays((xyz-c)@r.T,k)[0] for c,r,k in zip(C,R,o)])
    for m,(i,j,t) in enumerate(zip(g['data']['i'],g['data']['j'],g['data']['track'])):
        g['data']['a'][m]=uv[i,t];g['data']['b'][m]=uv[j,t]
    r,v=symmetric_errors(g['data'],o,R);assert error_stats(r,v)['median_px']>1


def test_disconnected_rotation_gauges():
    g,o,R=make_graph(n=7,points=30)
    mask=((g['data']['i']<3)&(g['data']['j']<3))|((g['data']['i']>=3)&(g['data']['j']>=3))
    d=subset(g['data'],mask)
    g['meta']['edges']=[e for e in g['meta']['edges'] if (e['i']<3)==(e['j']<3)]
    g['meta']['anchors']=[0,3]
    l=Layout(7,1280,720,'free_pp',anchors=[0,3]);x=initial(l,g['meta'],2.)
    assert l.nr==15
    oo,rr=l.unpack(x);np.testing.assert_array_equal(rr[0],np.eye(3));np.testing.assert_array_equal(rr[3],np.eye(3))
    fit=fit_one(l,g['meta'],d,max_nfev=200)
    assert fit['summary']['train']['median_px']<1e-3
    assert abs(fit['summary']['fx_over_fy']-1.18)<1e-3
