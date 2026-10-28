import numpy as np

from calibration.program_tracking.native_ice_material_repair import NestedHouseReader


def test_flat_small_circle_keeps_both_turnarounds_as_search_queries():
    reader=object.__new__(NestedHouseReader)
    reader.cameras={0:0}
    reader.project=lambda camera,xyz:(np.array([[960.,540.]]),np.array([1.]))
    reader.projection=lambda group,index,role,theta,parameters=None:(
        np.c_[960.+25.*np.cos(theta),540.+1.4*np.sin(theta)],np.ones(len(theta)))
    group=dict(center=np.zeros(2),radii={'center':1.},outline=False)
    _,uv,_,_=reader.queries(group,0,'center',None,spacing=35)
    assert uv[:,0].min()<935.01
    assert uv[:,0].max()>984.99
    assert uv[:,1].min()<538.601 and uv[:,1].max()>541.399
    assert len(uv)<12  # Four geometric turns need no dense local point cloud.
