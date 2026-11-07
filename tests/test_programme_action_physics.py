"""Regressions for multi-event rows, phase priority and collision geometry."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.episode_factor_adapter import EpisodeScopedFactor
from calibration.program_tracking.body_tracking.event_physics import ballistic_position_operator, flight_momentum_residual
from calibration.program_tracking.body_tracking.phase_ledger import phase_weights
from calibration.program_tracking.body_tracking.programme_anatomy import closest_segments_batch, box_separation, hinge_angles
from calibration.program_tracking.body_tracking.weighted_blocks import block, pack_weighted, verify


def test_transition_and_flight_win_over_support_in_either_order():
    from types import SimpleNamespace
    support = dict(episode_id='support',kind='support_candidate',start_pts=10,end_pts_inclusive=40)
    flight = dict(episode_id='flight',kind='flight_visible_candidate',start_pts=20,end_pts_inclusive=30)
    for episodes in ([support,flight],[flight,support]):
        ref=SimpleNamespace(source_document=dict(source_episode_review=dict(episodes=episodes)))
        np.testing.assert_allclose(phase_weights(ref,[0,10,20,30,40,50],stable=.8),
                                   [1,.8,.1,.1,.8,1])


def test_two_flights_keep_distinct_actual_rows_and_weights():
    class Factor:
        def evaluate(self,x,jacobian=False):
            r=np.array([x,2*x],float)
            J=csr_matrix([[1.],[2.]]) if jacobian else None
            name='flight_integrated_com'
            return dict(residual=r,jac=J,parts={name:float(r@r/2)},
                        weighted_blocks=[block(name,r,J)],metrics={'core':True})
    a,b=(EpisodeScopedFactor(Factor(),name) for name in ('first','second'))
    av,bv=a.evaluate(3.,True),b.evaluate(4.,True)
    rows=av['weighted_blocks']+bv['weighted_blocks']
    assert rows[0].name!=rows[1].name
    r,J,weighted=pack_weighted(rows,{rows[0].name:4.,rows[1].name:9.},True)
    np.testing.assert_array_equal(r,[6.,12.,12.,24.])
    np.testing.assert_array_equal(J.toarray().ravel(),[2.,4.,3.,6.])
    assert verify(weighted,r,J)['rows']==4


def test_event_quadrature_eliminates_unknown_ballistic_start_and_nonzero_momentum():
    for n in (13,25):
        t=np.linspace(0,.48,n)
        g=np.array([0,0,-9.81])
        C=np.array([5,-2,1])+t[:,None]*[2,1,2]+.5*t[:,None]**2*g
        A,b=ballistic_position_operator(t,g,.03)
        np.testing.assert_allclose(A@C+b,0.,atol=1e-12)
        H=np.broadcast_to([1.,.3,2.5],(n,3))
        r=flight_momentum_residual(H,t,sigma_specific_H=.2,phase='flight')['residual']
        np.testing.assert_allclose(r,0.,atol=1e-13)


def test_skew_segments_and_box_intersection_have_distinct_distance_signs():
    a=np.array([[-1.,0.,0.],[0.,0.,0.]])
    b=np.array([[1.,0.,0.],[1.,0.,0.]])
    c=np.array([[0.,-1.,.2],[2.,0.,0.]])
    d=np.array([[0.,1.,.2],[3.,0.,0.]])
    delta,s,t=closest_segments_batch(a,b,c,d)
    np.testing.assert_allclose(delta,[[0.,0.,-.2],[-1.,0.,0.]])
    np.testing.assert_allclose(s,[.5,1.])
    np.testing.assert_allclose(t,[.5,0.])
    centres=np.array([[.5,0.,0.],[1.5,0.,0.]])
    R=np.broadcast_to(np.eye(3),(2,3,3))
    np.testing.assert_allclose(box_separation(np.zeros((2,3)),R,[.5]*3,centres,R,[.5]*3),
                               [-.5,.5])


def test_model_hinge_distinguishes_flexion_from_reverse_rotation():
    axis=np.array([[1.,0.,0.]])
    values=np.deg2rad([60.,-60.])
    matrices=Rotation.from_rotvec(values[:,None]*axis).as_matrix()[:,None]
    flexion,swing=hinge_angles(matrices,axis)
    np.testing.assert_allclose(flexion[:,0],values,atol=1e-13)
    np.testing.assert_allclose(swing,0.,atol=1e-7)
