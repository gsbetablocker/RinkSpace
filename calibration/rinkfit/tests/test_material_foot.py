import numpy as np
from rinkfit_start.material_capture import capture_finite_arrays
from rinkfit_start.material_foot import select_certified_upper_bound


def test_saved_legal_upper_bound_cannot_be_lost_or_replaced_by_invalid_seed():
    old=dict(uv=np.array([[1.,0.],[1.,0.],[9.,0.]]),valid=np.array([True,True,False]))
    new=dict(uv=np.array([[2.,0.],[0.,0.],[8.,0.]]),valid=np.array([True,False,True]))
    s,b,r,take=select_certified_upper_bound(np.array([.1,.2,.3]),np.zeros(3,int),old,
        np.array([.4,.5,.6]),np.ones(3,int),new,np.zeros((3,2)))
    assert take.tolist()==[False,False,True]
    assert np.array_equal(s,[.1,.2,.6]) and b.tolist()==[0,0,1]
    assert r['valid'].tolist()==[True,True,True]


def test_array_capture_reuses_finite_kernel_without_mutating_entity_or_profile():
    class Scene:
        model={'materials':[{'paths':[{'kind':0,'variant_id':'w0'}]}]}
        def world(self,m,b,s):
            s=np.asarray(s);return np.stack((s,np.zeros_like(s),np.ones_like(s)),axis=-1)
        def world_rows(self,m,b,s):return self.world(0,0,s),np.ones(len(s),bool),np.full(len(s),'')
    class Camera:
        tau=0.
        def at(self,w,t):return {'uv':np.asarray(w)[...,:2]*10.,'valid':np.ones(np.shape(w)[:-1],bool)}
    station=np.array([.8,.3,.4]);branch=np.zeros(3,int);frozen=station.copy()
    r=capture_finite_arrays(Scene(),Camera(),np.array([[2.,0.],[0.,0.],[0.,0.]]),np.ones(3,int),
        np.zeros(3,int),np.array([0,1,2]),np.zeros(3,int),['w0'],station,branch,progress=None)
    assert abs(r['stations'][0]-.2)<1e-10
    assert np.array_equal(r['stations'][1:],frozen[1:]) and np.array_equal(station,frozen)
