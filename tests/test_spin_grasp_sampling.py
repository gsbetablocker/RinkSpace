from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.program_tracking.body_tracking.skate_contact import trapezoid_time_weights
from calibration.program_tracking.body_tracking.spin_grasp_factors import SpinGraspFactors,source_sample_indices


PTS=np.arange(972800,1216000+256,256,dtype=np.int64)
BOUNDARIES=[82.,84.4,86.,86.8,92.46,92.8,94.8]


def legacy_indices(seconds):
    source=PTS/12800.
    selected=(source>=seconds[0])&(source<=seconds[1])
    boundaries=[round(v*12800) for v in BOUNDARIES]
    return np.unique(np.r_[np.flatnonzero(selected)[::2],np.flatnonzero(np.isin(PTS,boundaries))])


def make_factor(seconds):
    times=(PTS-PTS[0])/12800.
    reference=SimpleNamespace(p={'pts':PTS},times=times,grasp_geometry={'segment_local_solver':[[0.,0.,0.],[1.,0.,0.]]},
        at=lambda t: {'P':np.zeros((len(t),3))})
    state=SimpleNamespace(seconds=seconds,m=1,basis=lambda t:np.ones((len(t),1)))
    problem=SimpleNamespace(reference=reference,body=SimpleNamespace(state=state,tau=.01),binding=SimpleNamespace(ratio=1.))
    relations=dict(factor_parameters={},phase_boundaries_seconds=BOUNDARIES,source_id='source',shot_id='s05',time_base=[1,12800],
        grip_interval=dict(start_pts=1111040,end_pts=1183488,hand='left',foot=0,state='held',role='fit',
            evidence_pts=[1111040],provenance='explicit test interval'),
        support_intervals=[dict(start_pts=1044736,end_pts=1183488,foot=1,weight=1.)],body_regions=[])
    return SpinGraspFactors(problem,relations)


def test_full_context_preserves_original_native_pts_exactly():
    np.testing.assert_array_equal(source_sample_indices(PTS,(76.,95.),BOUNDARIES),legacy_indices((76.,95.)))


@pytest.mark.parametrize('seconds',[(80.8,82.4),(87.2,90.4)])
def test_local_context_does_not_admit_distant_phase_boundaries(seconds):
    samples=PTS[source_sample_indices(PTS,seconds,BOUNDARIES)]
    assert np.all((samples/12800.>=seconds[0])&(samples/12800.<=seconds[1]))
    assert np.all(np.isin(samples,PTS))
    assert samples.dtype==PTS.dtype
    assert np.all(np.diff(samples)>0)


def test_phase_boundaries_use_available_exposures_without_fabricating_pts():
    # 92.46 s is an odd source sample omitted by the ordinary stride; 92.47 s
    # is absent from the native 50 Hz stream and must not become a new row.
    samples=PTS[source_sample_indices(PTS,(92.4,92.5),[92.46,92.47])]
    assert round(92.46*12800) in samples
    assert round(92.47*12800) not in samples
    assert np.all(np.isin(samples,PTS))


@pytest.mark.parametrize('seconds',[(80.8,82.4),(87.2,90.4)])
def test_constructor_support_quadrature_covers_only_selected_context(seconds):
    # No body/FK object is constructed: these stubs expose only clocks/bases
    # needed by factor preparation and exercise the actual support_dt chain.
    factor=make_factor(seconds)
    support_pts=factor.pts[factor.support_mask]
    assert np.all((support_pts/12800.>=seconds[0])&(support_pts/12800.<=seconds[1]))
    expected_times=(support_pts-PTS[0])/12800.
    np.testing.assert_array_equal(factor.support_times,expected_times)
    np.testing.assert_array_equal(factor.support_dt,trapezoid_time_weights(expected_times,singleton_duration_s=.04))
    assert np.max(np.diff(expected_times))<=.04+1e-12
    assert factor.support_dt.sum()==pytest.approx(expected_times[-1]-expected_times[0])
    # Reproduce the previous outside-context leak to show this catches the
    # actual multi-second integration gap, not merely a different row count.
    old_pts=PTS[legacy_indices(seconds)]
    old_support=old_pts[(old_pts>=1044736)&(old_pts<=1183488)]/12800.
    assert np.max(np.diff(old_support))>1.


def test_entry_without_grip_returns_zero_row_jacobian(monkeypatch):
    factor=make_factor((80.8,82.4));state=factor.body.state
    assert not factor.held.any()
    assert factor.grip_weights.shape==(0,)
    state.active=np.array([0,6]);state.d=2;state.size=2;factor.body.size=2
    state.increments=lambda x,t:np.zeros((len(t),69))
    factor.config.update(grip={},local_motion={'scale_m_s2':20.,'weight':.8,'robust_delta':3.})
    n=len(factor.times)
    monkeypatch.setattr(factor,'geometry_at',lambda d:(np.zeros((n,13,3)),np.zeros((n,3)),np.tile(np.eye(3),(n,1,1)),np.zeros((n,3))))
    monkeypatch.setattr(factor,'continuous_support',lambda x,jacobian:(np.empty(0),csr_matrix((0,19)),{}))
    monkeypatch.setattr(factor,'image',lambda x,jacobian:(np.empty(0),csr_matrix((0,19)),[]))
    value=factor.evaluate(np.zeros(2),jacobian=True)
    grip_rows=value['block_slices']['grip']
    assert value['residual'][grip_rows].shape==(0,)
    assert value['jac'][grip_rows].shape==(0,19)
    assert value['jac'].shape==(len(value['residual']),19)
    assert value['parts']['grip']==0.
    assert value['metrics']['grip_samples']==0
    assert np.isfinite(value['jac'].data).all()
