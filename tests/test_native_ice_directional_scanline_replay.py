"""A new x(y) row must not poison an unrelated legacy one-sided proof."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix


@pytest.fixture
def diagnostic():
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate import native_ice_directional_diagnostic as d
    return d


def mixed_source(d):
    from source_edge_intervals import SourceNormalIntervals
    camera = np.zeros(25)
    camera[6:8] = np.log(1000.)
    camera[8:10] = [960., 540.]
    angle = np.linspace(0., 2*np.pi, 361)
    ring = np.c_[2*np.cos(angle), 1.7*np.sin(angle), np.full(len(angle), 10.)]
    line = np.array([[0., -1., 10.], [0., 1., 10.]])
    support = SourceNormalIntervals(np.array([[1., 0.]]), np.array([[-1., 0.]]), np.array([True]))
    rows = [dict(id='legacy_boundary', split='fit', feature_id='B1', curve=dict(kind='current_ice_segment'),
        pts=100, actual_pts=100, uv=np.array([[960., 540.]]), sigma=np.ones(1),
        source_raw_frame='unchanged_source.png', source_file_sha256='0'*64, _source_optical_support=support),
        dict(id='measured_scanline', split='fit', feature_id='F3', curve=dict(kind='house_ring'),
        pts=101, actual_pts=101, uv=np.array([[800., 580.]]), sigma=np.array([70.]),
        source_raw_frame='unchanged_source.png', source_file_sha256='0'*64, _source_optical_support=None,
        curve_observation_kind='axis_scanline_branch', scanline_fixed_axis='y', scanline_branch='max_x')]
    def cameras(q):
        result = camera.copy(); result[8] += 10.*q[0]
        return {100: result, 101: result}
    def world(q, curve):
        return ring if curve['kind']=='house_ring' else line
    factor = SimpleNamespace(rows=rows, parameter_count=1, row_count=4, scales=np.array([1.,1.,70.,70.]),
        quadratic=np.ones(4, bool), contract={'packet_sha256': '1'*64},
        _qualified_active_fit_view=lambda: None, _cameras=cameras, _world=world)
    def actual(q):
        cams=cameras(q); parts=[]
        for row in rows:
            w=world(q,row['curve'])
            if row.get('curve_observation_kind')=='axis_scanline_branch':
                raw=d._scanline_branch(cams[row['actual_pts']],w,row)['scanline_raw_vector']
            else:
                raw=d.curve_vector_error_full_brown(cams[row['actual_pts']],w,row['uv'],with_jacobian=False)
            parts.append(d._native_error_with_optical_support(row,raw)/np.repeat(row['sigma'],2))
        return np.concatenate(parts)
    return factor, actual


def test_measured_scanline_branch_signature_uses_intersection_not_nearest(diagnostic):
    d=diagnostic; factor, actual=mixed_source(d); row=factor.rows[1]; q=np.zeros(1)
    camera=factor._cameras(q)[101]; world=factor._world(q,row['curve'])
    branch=d._scanline_branch(camera,world,row)
    assert branch['scanline_raw_vector'][1] == 0.
    assert branch['scanline_raw_vector'][0] > 300.
    old=d._branch(camera,world,row['uv'])
    assert abs(old['nearest_uv'][0,0]-row['uv'][0,0]) < 100.
    signature=d.actual_native_ice_row_branch_signature(camera,world,row)
    np.testing.assert_array_equal(signature[0],branch['scanline_segment_index'])
    assert len(signature)==3


@pytest.mark.parametrize('wrong_scanline_J', [False, True])
def test_mixed_record_exact_and_legacy_one_sided_qualification_not_weakened(diagnostic, wrong_scanline_J):
    d=diagnostic; factor, actual=mixed_source(d)
    q=np.zeros(1); direction=np.ones(1); h=1e-6
    base, plus, minus=actual(q),actual(q+h*direction),actual(q-h*direction)
    J=csr_matrix(np.array([[10.], [0.], [10./70.+(10. if wrong_scanline_J else 0.)], [0.]]))
    block=SimpleNamespace(r=base,J=J,scales=factor.scales,quadratic=factor.quadratic)
    # Production recording is explicitly Data-only. Temporary unit outputs
    # are removed at scope exit; no source images or persistent reports.
    data_root=Path('D:/CodexData/RinkSpace_Data')
    with TemporaryDirectory(prefix='native_ice_branch_unit_',dir=data_root) as tmp:
        report=d.record_native_ice_directional_diagnostic(factor,q,q,block,direction,h,
            plus,minus,J@direction,tmp,{'frozen_profile_token':'fixed'})
        assert report['branch_value_replay_max_difference']=={'base':0.,'plus':0.,'minus':0.}
        assert report['row_ledger'][1]['curve_observation_kind']=='axis_scanline_branch'
        assert report['per_row'][1]['measured_axis_scanline_actual_operator_replayed'] is True
        checked,summary=d.check_actual_native_ice_directional_branches(q,block,direction,h,
            plus,minus,J@direction,report)
    assert summary['real_branch_residual_replay_exact'] is True
    assert summary['original_allowed']==summary['checked_allowed']
    assert [x['id'] for x in summary['qualified_UV']]==['legacy_boundary']
    assert summary['qualified_UV'][0]['stable_real_side']=='plus'
    assert summary['passed'] is (not wrong_scanline_J)
    if wrong_scanline_J:
        assert [x['id'] for x in summary['unqualified_UV']]==['measured_scanline']
        assert 'nearest/clamp qualification does not apply' in summary['unqualified_UV'][0]['refusal_reason']
    assert summary['source_UV_sigma_roles_and_support_changed'] is False
    assert summary['mathematical_J_or_trial_value_operator_changed'] is False
