import numpy as np
import pytest
from rinkfit_start.native_scene import NativeScene


def scene_model():
    g=np.zeros(32);g[:5]=[.5,.12,.08,1.9051596,.02]
    return {"blocks":[{"name":"global","values":g.tolist()},
      {"name":"install","values":[.3,.1,.001,.018]}],
      "materials":[{"label":"separated","paths":[
        {"kind":2,"geometry_block":-1,"advertisement_block":1,"points":[[0,0],[.2,0]]},
        {"kind":2,"geometry_block":-1,"advertisement_block":1,"points":[[.8,1],[1,1]]}]}]}


def test_continuous_superellipse_perimeter_uses_true_arclength_both_directions():
    scene=NativeScene(scene_model());edges=np.cumsum([0,scene.long,scene.arc,scene.short,scene.arc,scene.long,scene.arc,scene.short,scene.arc])
    x=scene.board(edges,np.zeros(len(edges)))
    np.testing.assert_allclose(x[0],x[-1],atol=1e-12)
    for k in [1,3,5,7]:
        ss=np.linspace(edges[k]+.03*scene.arc,edges[k+1]-.03*scene.arc,15)
        eps=1e-7
        speed=np.linalg.norm((scene.board(ss+eps,0)-scene.board(ss-eps,0))/(2*eps),axis=1)
        np.testing.assert_allclose(speed,1,atol=2e-6)
    assert scene.arclength_numerical_max_error<1e-9


def test_separated_paths_cannot_create_a_connecting_segment():
    scene=NativeScene(scene_model())
    a=scene.world(0,0,[0.,1.]);b=scene.world(0,1,[0.,1.])
    assert np.min(np.linalg.norm(a[:,None,:]-b[None,:,:],axis=-1))>1e-3
    with pytest.raises(IndexError):scene.world(0,2,.5)
    with pytest.raises(ValueError):scene.world(0,0,1.01)


def test_internal_graphic_chart_is_not_an_outer_physical_panel_box():
    model=scene_model();model['blocks'][1]['name']='ad/5'
    model['materials'][0]['paths'][0]['points']=[[1.37,.22],[1.88,1.016]]
    scene=NativeScene(model)
    assert scene.check_physics()['passed']
    model['materials'][0]['paths'][0]['physical_canvas_bounds']=[[0,0],[1,1]]
    model['materials'][0]['paths'][0]['physical_canvas_provenance']='explicit synthetic fixture outer material boundary'
    assert not NativeScene(model).check_physics()['passed']


def test_boundary_segment_station_never_uses_an_excluded_opposite_side():
    model=scene_model();model['materials'][0]['paths']=[{'kind':3,'boundary_segment':4,'fixed_height_fraction':1.}]
    scene=NativeScene(model);p=scene.world(0,0,np.array([0.,.5,1.]))
    np.testing.assert_allclose(p[:,1],scene.width/2,atol=1e-12)
    np.testing.assert_allclose(p[:,2],scene.height,atol=1e-12)
    assert p[0,0]>p[-1,0]
