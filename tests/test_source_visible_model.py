import numpy as np
from calibration.program_tracking.body_tracking.source_visible_model import unoccluded,monotone_match
from calibration.program_tracking.body_tracking.weighted_blocks import block,assemble,verify,full_columns
from scipy.sparse import csr_matrix


def test_ray_excludes_hidden_edge_but_retains_own_surface():
    triangles=np.array([[[-1,-1,0],[1,-1,0],[1,1,0]],[[-1,-1,0],[1,1,0],[-1,1,0]]],float)
    visible=unoccluded(np.array([0.,0.,2.]),np.array([[0.,0.,0.],[0.,0.,-1.],[2.,0.,-1.]]),triangles)
    np.testing.assert_array_equal(visible,[True,False,True])


def test_ordered_correspondence_cannot_jump_back_and_forth():
    d=np.array([[0.,4.,5.,6.],[4.,0.,2.,3.],[0.,1.,4.,5.],[5.,4.,1.,0.]])
    choice,reverse=monotone_match(d)
    delta=np.diff(choice)
    assert np.all(delta>=0)or np.all(delta<=0)
    assert float(d[np.arange(4),choice].sum())<5.


def test_weighted_registry_reconstructs_actual_scaled_sparse_rows():
    b=block('native_pixels',np.array([2.,3.]),csr_matrix([[1.,4.],[2.,0.]]),dict(pts=123))
    c=full_columns(b,shared=2,scale=np.sqrt(.02))
    r,J=assemble([c],True);report=verify([c],r,J)
    assert report['rows']==2 and report['columns']==4
    np.testing.assert_allclose(r,b.residual*np.sqrt(.02))
    np.testing.assert_allclose(J.toarray()[:,2:],b.jacobian.toarray()*np.sqrt(.02))
    np.testing.assert_array_equal(J.toarray()[:,:2],0.)
