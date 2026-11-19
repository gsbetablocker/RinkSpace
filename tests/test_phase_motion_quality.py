import numpy as np
import pytest

from calibration.program_tracking.body_tracking.weighted_blocks import block


def test_flight_interior_exposes_alternation_even_with_the_same_net_impulse():
    from calibration.program_tracking.body_tracking.phase_motion_quality import flight_acceleration_metrics

    pts = np.arange(7) * 256
    times = pts / 12800
    gravity = np.array([0., 0., -9.81])
    reference = np.broadcast_to(gravity, (5, 3)).copy()
    trial = reference.copy()
    trial[:, 0] = [900., -28., 56., -28., 900.]
    definition = dict(episode_id='flight', start_pts=256, end_pts_inclusive=1280)
    assert trial[1:-1, 0].sum() == 0.
    summary, rows = flight_acceleration_metrics(times, pts, trial, definition, gravity)
    baseline, _ = flight_acceleration_metrics(times, pts, reference, definition, gravity)
    assert rows['source_pts'] == [512, 768, 1024]
    assert summary['rms_gravity_error_mps2'] > 30.
    assert summary['worst_source_pts'] == 768
    assert baseline['rms_gravity_error_mps2'] == 0.


def test_joint_report_keeps_added_and_resolved_pts_even_when_counts_cancel():
    from calibration.program_tracking.body_tracking.phase_motion_quality import joint_domain_changes

    pts = np.arange(3)*256 + 10000
    old = np.zeros((3, 1, 3)); new = old.copy()
    old[0, 0, 1] = .3
    new[2, 0, 2] = .24
    report = joint_domain_changes(pts, ['left_ankle'], old, new)['left_ankle']
    assert report['before_count'] == report['after_count'] == 1
    assert report['new'][0]['pts'] == pts[2]
    assert report['new'][0]['components'] == [0., 0., .24]
    assert report['resolved'][0]['pts'] == pts[0]


def test_zero_update_records_only_declared_objective_family_changes():
    from calibration.program_tracking.body_tracking.scoped_candidate import compare_objective_replay

    old = np.array([1., 2., 3., 4., 5.])
    registry = [dict(name=n, start=a, stop=b) for n, a, b in (
        ('body_pixels', 0, 2), ('static', 2, 3), ('actual_model_com_acceleration', 3, 5))]
    new = [block('body_pixels', [1.]), block('static', [3.]),
           block('actual_model_com_acceleration', [4., 5., 6.]),
           block('episode/flight/flight_dense_com_acceleration', [7., 8.])]
    changed = {'body_pixels', 'actual_model_com_acceleration', 'episode/flight/flight_dense_com_acceleration'}
    receipt = compare_objective_replay(old, registry, new, changed)
    assert receipt['unchanged_rows'] == 1
    assert receipt['changes']['body_pixels']['before_rows'] == 2
    assert receipt['changes']['episode/flight/flight_dense_com_acceleration']['before_rows'] == 0
    new[1] = block('static', [3.1])
    with pytest.raises(ValueError, match='static'):
        compare_objective_replay(old, registry, new, changed)


def test_phase_direction_does_not_consume_reserved_shoe_source():
    from calibration.program_tracking.body_tracking.scoped_phase_quality import ScopedPhaseQuality

    specification = dict(fit_shoe_points=[dict(role='additional_check',pts=1042176,foot=0)])
    with pytest.raises(ValueError,match='Only fitted shoe evidence'):
        ScopedPhaseQuality(None,None,None,specification,[],None,None,None)


def test_median_with_fixed_pivot_uses_a_real_movable_neighbour():
    from types import SimpleNamespace
    from scipy.sparse import csr_matrix
    from calibration.program_tracking.body_tracking.scoped_phase_quality import ScopedBodyQuantiles

    evaluator=object.__new__(ScopedBodyQuantiles)
    evaluator.problem=SimpleNamespace(reference=SimpleNamespace(p={'pts':np.arange(5)*256}))
    evaluator.observed=np.zeros((2,2))
    evaluator.project_pixels=lambda *args:np.array([[1.30,0.],[1.60,0.]])
    evaluator.frames=np.array([3,4]);evaluator.ids=np.array([13,14])
    evaluator.points=evaluator.frames
    evaluator.rows=np.arange(5)
    evaluator.fixed_errors=np.array([.2,.4,1.2,1.3,1.6])
    evaluator.derivative=csr_matrix([[1.],[0.],[2.],[0.]])
    evaluator.local_index=np.array([-1,-1,-1,0,1])
    evaluator.selected=np.array([0])
    evaluator.movable=np.ones(2,bool)
    evaluator.groups=[('median',np.arange(5),50,1.)]

    jac,margin,labels=evaluator.blocks(np.zeros(1))
    assert labels==['median/affected_row_768']
    np.testing.assert_allclose(margin,[-.3])
    np.testing.assert_array_equal(jac.toarray(),[[1.]])


def test_scope_resume_binds_only_surface_from_its_exact_saved_pose(tmp_path):
    import json
    from calibration.program_tracking.body_tracking.programme_skate_replay import ScopedSurfaceGuard
    from calibration.program_tracking.body_tracking.rebuild_prepare import digest
    from calibration.program_tracking.body_tracking.weighted_blocks import signature

    accepted = np.array([.1,.3])
    scan=tmp_path/'scan.npz'
    np.savez_compressed(scan, times=[0.,.02])
    receipt=tmp_path/'retained_surface.json'
    receipt.write_text(json.dumps(dict(parameter_vector_signature=signature(accepted),
                                     surface_sha256=digest(scan),surface_path=str(scan))),encoding='utf8')
    guard=object.__new__(ScopedSurfaceGuard)
    original={'times':np.array([.08])}
    guard.scan=original
    guard.emit=lambda *args,**kwargs:None

    with pytest.raises(ValueError,match='exact resumed geometry'):
        guard.bind_current_scan(np.array([.1,.4]),receipt)
    guard.bind_current_scan(accepted,receipt)
    np.testing.assert_array_equal(guard.current_scan['times'],[0.,.02])
    assert guard.scan is original


def test_landing_step_constraints_preserve_the_foot_and_end_pts():
    from types import SimpleNamespace
    from scipy.sparse import csr_matrix
    from calibration.program_tracking.body_tracking.scoped_phase_quality import ScopedPhaseQuality
    from calibration.program_tracking.body_tracking.programme_source_step import SourceMargins

    evaluator=object.__new__(ScopedPhaseQuality)
    evaluator.cavity=None
    shoe_ball=dict(offset=np.array([[.12,0.,0.],[.26,0.,0.]]),radius=np.array([.24,.24]),
                   jacobian=csr_matrix(np.arange(12).reshape(6,2)),
                   labels=['shoe/1043712/foot_0','shoe/1043712/foot_1'])
    evaluator.blocks=lambda x:(SourceMargins(csr_matrix((0,2)),np.empty(0),[],[shoe_ball]),None,None)
    J,excess,labels=evaluator.cuts(np.zeros(2))
    assert labels==['shoe/1043712/foot_1']
    np.testing.assert_allclose(J.toarray(),[[6,7]])
    np.testing.assert_allclose(excess,[.02])


def test_landing_proposal_can_retain_flight_at_its_exact_current_boundary():
    from scipy.sparse import csr_matrix
    from calibration.program_tracking.body_tracking.scoped_motion_step import reduce_domains
    from calibration.program_tracking.body_tracking.programme_source_step import SourceMargins

    flight=dict(source_id='flight',jacobian=csr_matrix([[1.,0.],[0.,0.]]),offset=np.array([[.3,.4]]),
                radius=np.array([.5]),labels=['__flight_dense__/episode/flight/physics'])
    margins=SourceMargins(csr_matrix((0,2)),np.empty(0),[],[flight])
    *_,balls=reduce_domains(margins,np.arange(2),np.ones(2),-np.ones(2),np.ones(2),
                            reserve_at_flight_boundary=False)
    jac,offset,radius,label=balls[0]
    assert label==flight['labels'][0]
    np.testing.assert_allclose(np.linalg.norm(offset),radius)


def test_rotation_derivative_receipt_allows_a_protected_step_outside_active_support():
    from scipy.sparse import csr_matrix
    from calibration.program_tracking.body_tracking.scoped_motion_step import derivative_receipt

    # These guarded rotations still have derivatives in the programme chart.
    # Their features have no derivative for this later landing-only fit.
    measurements=([],[],[],csr_matrix((0,30)),[],[],[],csr_matrix(([1.],([0],[20])),shape=(3,30)))
    assert derivative_receipt(None,np.zeros(30),measurements,np.array([25]))==[]


def test_rank_prefix_protects_both_interpolation_pivots_after_rank_crossings():
    from calibration.program_tracking.body_tracking.scoped_phase_quality import percentile_prefix

    before=np.array([1.,2.,3.,6.,8.,33.])
    percent=70
    target=np.percentile(before,percent)
    caps=percentile_prefix(before,percent,target)
    np.testing.assert_allclose(caps,[6.,6.,6.,6.,8.])
    # The last row can cross the rank arbitrarily, and the originally small
    # member errors need not preserve their ranking.
    after=np.r_[caps[[0,1,2,3,4]],2.5]
    after[:4]=[5.9,5.8,6.,5.99]
    assert np.percentile(after,percent)<=target
