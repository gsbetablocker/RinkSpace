import numpy as np
import pytest
from rinkspace.mosaic import ProvenanceMosaic


def test_missing_pixels_transparent():
    m=ProvenanceMosaic(30,30,'a');a=np.full((10,10,3),100,np.uint8)
    m.add(a,np.ones((10,10),np.uint8),np.eye(3),7,.5,'a')
    assert m.rgba()[20,20,3]==0 and m.source_index[20,20]==-1
    assert m.rgba()[4,4,3]==255 and m.source_index[4,4]==7


def test_masked_object_cannot_fill_background():
    m=ProvenanceMosaic(30,30,'a');a=np.full((30,30,3),100,np.uint8)
    mask=np.ones((30,30),np.uint8);mask[8:20,8:20]=0
    m.add(a,mask,np.eye(3),0,.5,'a')
    assert m.source_index[12,12]==-1


def test_source_selection_provenance_changes_together():
    m=ProvenanceMosaic(30,30,'a');a=np.full((30,30,3),100,np.uint8);mask=np.ones((30,30),np.uint8)
    m.add(a,mask,np.eye(3),0,.5,'a');m.add(a*2,mask,np.eye(3),1,.4,'a')
    assert m.source_index[12,12]==0 and m.rgb[12,12,0]==100
    m.add(a*2,mask,np.eye(3),1,.8,'a')
    assert m.source_index[12,12]==1 and m.rgb[12,12,0]==200


def test_ice_epochs_not_blended():
    m=ProvenanceMosaic(20,20,'before')
    with pytest.raises(ValueError,match="epochs"):
        m.add(np.zeros((20,20,3),np.uint8),np.ones((20,20),np.uint8),np.eye(3),0,1,'after')
