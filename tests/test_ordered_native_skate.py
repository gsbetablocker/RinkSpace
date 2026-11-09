"""Check both sides of the visible contour objective in image space."""
import numpy as np
import pytest

from calibration.program_tracking.body_tracking.ordered_native_skate import EdgeMatch, edge_terms, monotone


@pytest.mark.parametrize('complete',[False,True])
def test_moving_visible_model_edge_gradient(complete):
    source=np.array([[3.,3.],[5.,5.],[10.,5.],[13.,4.]])
    model=np.array([[0.,0.],[6.,1.],[11.,-.5],[15.,0.]])
    active=EdgeMatch((0,1,2,3),(0,0,1,2),(0,0,1,2),(.15,.33,.35,.17))
    v=np.array([[.2,.3],[-.4,.7],[.6,-.9],[.4,-.5]])
    def evaluate(points):
        return edge_terms(points,source,active,1.,.02,complete,[])
    r,J,raw,Jraw=evaluate(model)
    e=1.e-5
    p,m=evaluate(model+e*v),evaluate(model-e*v)
    np.testing.assert_allclose(J@v.ravel(),(p[0]-m[0])/(2*e),atol=2.e-9,rtol=1.e-6)
    np.testing.assert_allclose(Jraw@v.ravel(),((p[2]-m[2])/(2*e)).ravel(),atol=2.e-9,rtol=1.e-6)
    assert len(r)==2*(len(source)+len(model)*int(complete))


def test_ordered_match_cannot_reverse_or_infer_ends():
    cost=np.array([[10,1,0],[2,0,4],[0,1,3]],float)
    path=monotone(cost)
    assert list(path)==sorted(path) and path==(1,1,1)
    match=EdgeMatch((0,1,2),(1,1),(0,0,0),(.2,.5,.3))
    source=np.array([[12.,0.],[14.,0.]])
    model=np.array([[0.,0.],[10.,0.],[25.,0.]])
    # A visible fragment fits an interior part of a long lower edge.
    # No unseen endpoint or out-of-range model extent enters the residual.
    partial=edge_terms(model,source,match,1.,.02,False,[])
    complete=edge_terms(model,source,match,1.,.02,True,[])
    assert not np.any(partial[0])
    assert np.linalg.norm(complete[0])>0
