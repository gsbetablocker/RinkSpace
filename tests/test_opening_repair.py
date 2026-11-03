"""Regression cases for cropped-opening inference, independent of raw assets."""
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from calibration.program_tracking.body_tracking.opening_repair import OpeningReference,OpeningWindow,START,STATIONARY_END,END


def reference():
    def parent_at(t):
        t=np.atleast_1d(t);n=len(t)
        return dict(P=np.c_[1+t,.1*t,1+.01*t],Q=Rotation.from_rotvec(np.c_[0*t,0*t,.1*t]).as_matrix(),
                    local_R=np.broadcast_to(np.eye(3),(n,21,3,3)).copy(),times=t)
    r=object.__new__(OpeningReference)
    r.parent=SimpleNamespace(at=parent_at,p=dict(pts=np.array([283136])))
    r.refinements=[];r.preparation=None;r.donor_R=np.broadcast_to(np.eye(3),(21,3,3)).copy()
    r.heading_times=np.array([-1.,4.]);r.heading=np.array([0.,6.]);r.anchor=np.array([2.,3.,1.]);r.origin=START
    return r


def test_first_source_frame_is_free_but_later_state_is_exact_parent():
    r=reference();w=OpeningWindow(r,(START,END),'full')
    np.testing.assert_allclose(w.basis([0.]).sum(1),1.,atol=1e-14)
    x=np.zeros((w.m,w.d));x[:,0]=.10;r.refinements=[(w,x.ravel())]
    np.testing.assert_allclose(r.at([0.])['P'][0],r.anchor+[.1,0,0],atol=1e-14)
    times=np.array([END-START,END-START+.02,4.])
    for key in ('P','Q','local_R'):np.testing.assert_array_equal(r.at(times)[key],r.parent.at(times)[key])


def test_constant_speed_has_a_velocity_cost_even_when_acceleration_is_zero():
    r=reference();w=OpeningWindow(r,(START,END),'full')
    greville=np.array([np.mean(w.knots[i+1:i+4]) for i in range(w.m)])
    times=np.linspace(.1,1.5,9)
    np.testing.assert_allclose(w.basis(times,1)@greville,1.,atol=1e-13)
    np.testing.assert_allclose(w.basis(times,2)@greville,0.,atol=1e-12)
    assert np.linalg.norm((w.basis(times,1)@greville)/.5)>1.


def test_parent_attachment_preserves_value_and_vanishing_correction_derivatives():
    r=reference();w=OpeningWindow(r,(START,END),'full')
    x=np.full(w.size,.015);r.refinements=[(w,x)]
    end=END-START
    for order in (0,1,2):np.testing.assert_allclose(w.basis([end],order),0.,atol=1e-12)
    # The reset reference itself also approaches the parent cubically.
    errors=[]
    for h in (.002,.001):
        current=r.at([end-h]);old=r.parent.at([end-h])
        errors.append(np.linalg.norm(current['P']-old['P'])+np.linalg.norm(current['Q']-old['Q']))
    assert errors[1]<errors[0]/6


def test_preparation_lower_pose_is_released_without_leaking_into_later_frames():
    r=reference();lower=r.donor_R.copy()
    lower[0]=Rotation.from_rotvec([.1,.2,.3]).as_matrix()
    r.preparation=dict(local_R=lower)
    np.testing.assert_allclose(r.at([0.])['local_R'][0,0],lower[0],atol=1e-15)
    np.testing.assert_allclose(r.at([STATIONARY_END-START])['local_R'][0],r.donor_R,atol=1e-15)
    for key in ('P','Q','local_R'):
        np.testing.assert_array_equal(r.at([END-START+.1])[key],r.parent.at([END-START+.1])[key])
