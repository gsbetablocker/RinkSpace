from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.solver_runtime import install
install()
from z190_streaming_source_normal import SourceRowBlock
from z190_joint_source_solver import source_cost
from calibration.solver_runtime.source_loss import SourceLossWeights
from calibration.program_tracking.static_venue_source import StaticCameraPolicy, StaticVenueFactor
from calibration.program_tracking.first_shot_static_venue_stage import StaticVenueStage


class Programme:
    def __init__(self):
        self.start = np.array([.2, .5, 0.])
        self.state = SimpleNamespace(count=3, world_parameter_indices=np.array([0]))
        self.shared, self.fixed = np.array([0]), np.array([2])
        self.scales, self.quadratic = np.array([.7]), np.array([False])
        self.source_producers, self.recovery_references = [], []
        self.guard_calls, self.recover_calls = [], []
    def guard(self, q):
        assert q.shape == (3,) and q[2] == 0.
        self.guard_calls.append(q.copy())
        return True
    def source_blocks(self, q, derivative):
        assert q.shape == (3,)
        yield SourceRowBlock(np.array([q[0]+q[1]-1.]),
            csr_matrix([[1., 1., 0.]]) if derivative else None,
            self.quadratic, self.scales, {'family':'original-native'})
    def raw(self, q, **kwargs):
        assert q.shape == (3,)
        return np.array([abs(q[0]+q[1]-1.)])
    def linear_constraints(self, q, selected):
        assert q.shape == (3,) and np.array_equal(selected, self.shared)
        return dict(matrix=np.array([[1.]]), lower=np.array([.1-q[0]]))
    def recover(self, parent, proposed):
        assert parent.shape == proposed.shape == (3,)
        self.recover_calls.append((parent.copy(), proposed.copy()))
        return dict(q=proposed.copy(), report={'original':True})
    def checkpoint_arrays(self):
        return {'native_pts':np.array([256256])}
    def complete_state_payload(self, q):
        return dict(q_augmented=q.copy(), original=q.copy())
    def checkpoint_identity(self):
        return {'binding':{'path':'original-programme','sha256':'same'}}
    def solver_kwargs(self):
        return dict(global_schur_count=1, global_schur_indices=self.shared,
            fixed_parameter_indices=self.fixed, shared_parameter_bounds={})


def make_stage():
    programme = Programme()
    def scene(q):
        return {'ice_features':[dict(id='C', kind='polyline', segments_xyz=[
            [[-2., float(q[0]), 10.], [2., float(q[0]), 10.]]])]}
    row = dict(id='real-wide-line', target=dict(kind='line_segment',feature_id='C'),
        pixels_uv=[[900., 560.],[1000., 561.]], sigma_px=[.5, 1.5])
    packet = SimpleNamespace(rows=(row,), scales=np.repeat(row['sigma_px'],2),
        quadratic=np.zeros(4,bool), loss_weights=SourceLossWeights(np.full(4,.5)),
        ledger={'family':'independent-wide'}, identity_sha256='explicit-packet',
        frame={'PTS':5,'time_base':'1/25'})
    camera = np.zeros(25); camera[6:8]=np.log(800.); camera[8:10]=[960.,540.]
    factor = StaticVenueFactor(packet, StaticCameraPolicy('brown13',camera),
        programme_count=3, geometry_columns=np.array([0]), scene_at=scene,
        geometry_guard=programme.guard, geometry_fd_step=1e-5, samples=8)
    return programme, StaticVenueStage(programme, factor,
        parent_reference={'path':'accepted-programme','sha256':'accepted'})


def test_append_keeps_native_rows_and_padding_exact_and_uses_current_geometry():
    programme, stage = make_stage()
    blocks = list(stage.source_blocks(stage.start, True))
    assert len(stage.start) == 16 and len(blocks) == 2
    np.testing.assert_array_equal(stage.start[:3].view(np.uint64),programme.start.view(np.uint64))
    original = next(programme.source_blocks(programme.start,True))
    np.testing.assert_array_equal(blocks[0].r,original.r)
    np.testing.assert_array_equal(blocks[0].J[:,:3].toarray(),original.J.toarray())
    assert blocks[0].J[:,3:].nnz == 0 and blocks[1].J.shape == (4,16)
    assert np.linalg.norm(blocks[1].J[:,0].toarray()) > 0.
    changed=stage.start.copy();changed[0]+=.01
    changed_blocks=list(stage.source_blocks(changed,False))
    assert not np.array_equal(changed_blocks[0].r,blocks[0].r)
    assert not np.array_equal(changed_blocks[1].r,blocks[1].r)
    expected=sum(source_cost(b.r,b.quadratic,b.loss_weights)for b in blocks)
    assert stage.cost(stage.start) == expected


def test_new_complete_payload_constraints_and_recovery_preserve_both_charts():
    programme, stage = make_stage()
    proposed=stage.start.copy();proposed[0]+=.01;proposed[-1]+=.001
    result=stage.recover(stage.start,proposed)
    np.testing.assert_array_equal(result['q'],proposed)
    assert programme.recover_calls
    domain=stage.linear_constraints(proposed,stage.shared)
    np.testing.assert_allclose(domain['lower'],[-.11])
    np.testing.assert_array_equal(domain['matrix'],np.r_[1.,np.zeros(13)][None])
    np.testing.assert_array_equal(stage.shared,np.r_[0,np.arange(3,16)])
    assert stage.raw(proposed).shape == (3,)
    assert stage.raw_programme(proposed).shape == (1,)
    payload=stage.complete_state_payload(proposed)
    assert payload['q_augmented'].shape == (16,)
    np.testing.assert_array_equal(payload['programme_q_augmented'],proposed[:3])
    np.testing.assert_array_equal(payload['static_venue_camera_state'],proposed[3:])
    assert payload['static_venue_camera25'].shape == (25,)
    kwargs=stage.solver_kwargs()
    assert kwargs['checkpoint_arrays']['programme_parameter_count'][0] == 3
    assert kwargs['source_block_factory'] == stage.source_blocks
    np.testing.assert_array_equal(kwargs['fixed_parameter_indices'],[2])
    assert kwargs['checkpoint_identity']['persistent_static_venue_stage'][
        'previous_fixed_world_coordinates_released']
    assert kwargs['checkpoint_identity']['initial_exact_complete_current_state_full_J_reuse'] is False
    assert stage.guard(stage.start)


def test_missing_wide_block_cannot_be_silently_admitted_by_acceleration():
    programme, stage = make_stage()
    def missing(factory,q,jacobian):
        from scipy.sparse import hstack
        for b in factory(q[:3],jacobian):
            yield SourceRowBlock(b.r,hstack((b.J,csr_matrix((len(b.r),13))),format='csr')
                if jacobian else None,b.quadratic,b.scales,b.row_ledger)
    stage.factor.append_source_blocks = missing
    with pytest.raises(ValueError,match='absent or duplicated'):
        list(stage.source_blocks(stage.start,True))


def test_whole_group_revision_rebuilds_real_bundle_without_reusing_old_arrays(tmp_path):
    import hashlib
    from calibration.program_tracking.first_shot_fit import FrozenWhitening
    from calibration.program_tracking.first_shot_source_transition import reference
    from calibration.program_tracking.source_evidence_whitening import SourceWhiteningOperator
    from calibration.program_tracking.first_shot_static_venue_stage import apply_complete_group_quarantine
    rows=[dict(_target_key=str(i),target={'kind':'ice_curve','curve_id':str(i)},
        uv=np.array([[i+.2,i+.7]]),sigma=np.array([1.3+i]),source_point_ids=[str(i)])for i in range(5)]
    A=SourceWhiteningOperator(np.array([[1.,.2,-.4,.3],[.1,1.1,.3,-.2],[.2,-.1,.7,.6]]),('0','2'),'kept','real-source')
    B=SourceWhiteningOperator(np.array([[.5,.1,.2,.8],[.1,.9,-.3,.2]]),('1','3'),'uncertain','real-source')
    frame=dict(index=7,native_pts=8128,rows=rows,source_point_count=5,
        whitening_groups=[dict(point_indices=[0,2],operator=A),dict(point_indices=[1,3],operator=B)])
    w=FrozenWhitening(frame)
    manifest=tmp_path/'group.json';manifest.write_text('{"explicit":true}',encoding='utf8')
    spec=dict(native_pts=8128,source_point_ids=['1','3'],operator_matrix_sha256=hashlib.sha256(B.matrix.tobytes()).hexdigest(),
        operator_matrix_shape=list(B.matrix.shape),material_identity_confirmed=False,
        reason='RGB unresolved material family',evidence_references=[reference(manifest)])
    programme=Programme();programme.pool=None
    programme.bundle=SimpleNamespace(frames=[frame],whitening=[w],ledgers=[{'family':'native','source_points':5,'formal_rows':w.count}],
        _source_pool=None,_geometry_pool=None,_world_cache={'old':1},_finite_coverage_cache={'old':1})
    programme.scales=w.scales.copy();programme.quadratic=np.arange(w.count)%2==0
    old_mask=programme.quadratic.copy();programme.transverse_world_cache={'old':1};programme.guard_cache={'old':1}
    proof=apply_complete_group_quarantine(programme,[spec],manifest_reference=reference(manifest))
    assert proof['removed_complete_groups']==1 and proof['source_points']==3
    assert programme.bundle.frames[0]['inactive_source_groups'][0]['original_complete_group']['operator'] is B
    assert programme.bundle.whitening[0].count==5 and len(programme.scales)==5
    np.testing.assert_array_equal(programme.quadratic,old_mask[[0,1,2,5,6]])
    np.testing.assert_array_equal(programme.formal_offsets,[0,5])
    assert programme.bundle.ledgers[0]['source_points']==3
    assert not programme.transverse_world_cache and not programme.guard_cache and not programme.bundle._world_cache
    assert programme.checkpoint_identity()['explicit_complete_group_source_quarantine']['old_source_r_J_reused'] is False
