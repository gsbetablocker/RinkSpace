from copy import deepcopy
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.solver_runtime.original_source_point_qualification import (
    FAMILY, OriginalSourcePointQualification, wrap_original_source_point_qualification,
)
from calibration.solver_runtime.source_point_qualification import SCHEMA, file_sha256, original_uv_sha256


def write_json(path, data):
    path.write_text(json.dumps(data), encoding='utf8')


def fixture(tmp_path, *, held=False, direct_user=False):
    from calibration.solver_runtime import install
    install()
    from z190_streaming_source_normal import SourceRowBlock
    raw = tmp_path / 'native.png'; raw.write_bytes(b'exact original native RGB')
    review = tmp_path / 'source_review.json'; write_json(review, dict(source='actual original RGB'))
    row = dict(id='F2_inner', pts=200, split='holdout' if held else 'fit',
        uv=[[1., 2.], [3., 4.], [5., 6.]], sigma=[2., 3., 4.],
        world=[[1., 2., 0.], [3., 4., 0.]], normal=[[0., 1.]]*3,
        normal_interval=[[-1., 1.]]*3, point_correspondence=False,
        origin='actual old source reader')
    if direct_user: row['source'] = dict(kind='digitized_user_stroke')
    item = dict(block='secondary', collection_role='HELD' if held else 'FIT',
        group_actual_pts=256, group_order=0, row_order=0, source_row=row)
    parent = tmp_path / 'original.pkl'; parent.write_bytes(pickle.dumps(dict(programme_absolute_rows=[item])))
    locator = {k: v for k, v in item.items() if k != 'source_row'}
    record = dict(original_packet_row_index=0, locator=locator, row_id=row['id'], source_pts=row['pts'],
        original_uv_float64_sha256=original_uv_sha256(row), excluded_point_indices=[1],
        review_basis='original_rgb_only', decision_is_not_from_model_residual_or_cost=True,
        reason='Native RGB shows an unresolved competing paint branch',
        review_authority=dict(path=str(review), sha256=file_sha256(review)),
        native_rgb_evidence=[dict(path=str(raw), sha256=file_sha256(raw), source_pts=256,
            source_video_sha256='f'*64, time_base='1/15360')])
    manifest = dict(schema=SCHEMA, source_family=FAMILY,
        source_video_sha256='f'*64, time_base='1/15360',
        original_packet=dict(path=str(parent), sha256=file_sha256(parent), format='original_absolute_compact_pickle'),
        records=[record])
    packet = tmp_path / 'qualification.json'; write_json(packet, manifest)
    state = SimpleNamespace(original=SimpleNamespace(aliases={200: 256}), blocks=[
        SimpleNamespace(name='main1', quadratic=np.zeros(5, bool), groups=[], bundle=SimpleNamespace(motion_rows=5)),
        SimpleNamespace(name='secondary', quadratic=np.zeros(11, bool), groups=[(256, [deepcopy(row)])],
            bundle=SimpleNamespace(motion_rows=4))])
    blocks=[]
    for name, start, stop in [('main1', 0, 5), ('secondary', 5, 16)]:
        r=np.arange(start, stop, dtype=float)+1.
        blocks.append(SourceRowBlock(r, csr_matrix(np.arange(len(r)*3).reshape(len(r), 3)+1.),
            np.zeros(len(r), bool), np.full(len(r), 3.),
            dict(family='programme:'+name, original_start=start, original_stop=stop, source_sha256='f'*64)))
    base=SimpleNamespace(count=3, updater=object(), contract={'schema':'actual-source'},
        scales=np.full(16, 3.), quadratic=np.zeros(16, bool), guard=lambda q: q[0]>=0,
        blocks=lambda q,jacobian=True: iter(blocks))
    return SimpleNamespace(packet=packet, manifest=manifest, state=state, base=base,
        blocks=blocks, row=row, parent=parent, raw=raw, review=review, record=record)


def test_none_is_exact_base_and_mask_preserves_stream_guard_native_layout(tmp_path):
    f=fixture(tmp_path)
    assert wrap_original_source_point_qualification(f.base, f.state, None) is f.base
    source=wrap_original_source_point_qualification(f.base, f.state, f.packet)
    qualified=source.original_point_qualification
    assert qualified.contract['active_runtime_global_scalar_slots']==[11,12]
    assert qualified.contract['fresh_complete_source_profiles_and_J_required'] is True
    before=f.blocks[1]; after=list(source.blocks(np.zeros(3), True))[1]
    keep=np.array([0,1,2,3,4,5,8,9,10])
    assert list(source.blocks(np.zeros(3)))[0] is f.blocks[0]
    assert np.array_equal(after.r[keep],before.r[keep])
    assert np.array_equal(after.J.toarray()[keep],before.J.toarray()[keep])
    assert np.array_equal(after.r[6:8],[0.,0.]) and after.J[6:8].nnz==0
    assert after.scales is before.scales and after.quadratic is before.quadratic
    assert after.r.shape==before.r.shape and after.J.shape==before.J.shape
    assert np.all(before.r>0) and before.J.nnz==33
    assert source.scales is f.base.scales and source.updater is f.base.updater
    assert source.guard is f.base.guard and source.guard([1.])
    assert after.row_ledger['inactive_scalar_slots_excluded_from_source_accuracy_statistics']==[11,12]
    history=qualified.history_review()['rows'][0]
    assert history['accuracy_metric'] is None and history['included_in_active_FIT_accuracy_statistics'] is False
    assert history['original_source_row']['uv']==f.row['uv']
    # Old profiles/J/source inventory cannot masquerade as the new objective.
    from z190_streaming_source_normal import _checked_block
    assert _checked_block(before,3,True)[-1]!=_checked_block(after,3,True)[-1]
    f.blocks[1].row_ledger['original_stop']+=1
    with pytest.raises(ValueError,match='scalar range changed'):
        list(source.blocks(np.zeros(3)))


@pytest.mark.parametrize('failure',['held','direct_user','parent_hash','RGB_hash','UV_hash','locator','clock','cost_based','duplicate_index','world','sigma'])
def test_provenance_or_role_changes_fail_closed(tmp_path,failure):
    f=fixture(tmp_path,held=failure=='held',direct_user=failure=='direct_user')
    if failure=='parent_hash': f.parent.write_bytes(f.parent.read_bytes()+b'changed')
    elif failure=='RGB_hash': f.raw.write_bytes(b'changed pixels')
    elif failure=='UV_hash': f.record['original_uv_float64_sha256']='0'*64
    elif failure=='locator': f.record['locator']['group_order']=3
    elif failure=='clock': f.record['native_rgb_evidence'][0]['source_pts']=200
    elif failure=='cost_based': f.record['review_basis']='residual_threshold'
    elif failure=='duplicate_index': f.record['excluded_point_indices']=[1,1]
    elif failure in ('world','sigma'):
        f.state.blocks[1].groups[0][1][0][failure][0]=[9.,9.,9.] if failure=='world' else 99.
    write_json(f.packet,f.manifest)
    with pytest.raises(ValueError):
        wrap_original_source_point_qualification(f.base,f.state,f.packet)


def test_other_point_authorized_pixel_correction_composes_but_corrected_point_is_protected(tmp_path):
    f=fixture(tmp_path)
    runtime=f.state.blocks[1].groups[0][1][0]
    runtime['uv'][0]=[10.,20.]
    runtime['active_source_pixel_correction']=dict(packet_sha256='a'*64)
    f.state.original_ice_pixel_correction=dict(packet_sha256='a'*64,corrected_rows=[dict(
        original_packet_row_index=0, locator=f.record['locator'],original_source_row=deepcopy(f.row),
        point_indices=[0],corrected_uv=[[10.,20.]],optical_support_updated=False)])
    source=wrap_original_source_point_qualification(f.base,f.state,f.packet)
    assert source.original_point_qualification.contract['active_FIT_points_retired']==1
    f.record['excluded_point_indices']=[0]; write_json(f.packet,f.manifest)
    with pytest.raises(ValueError,match='source-corrected/user pixel'):
        wrap_original_source_point_qualification(f.base,f.state,f.packet)


def test_existing_retirement_and_optional_loss_weight_metadata_is_preserved(tmp_path):
    f=fixture(tmp_path)
    from z190_streaming_source_normal import SourceRowBlock
    from calibration.solver_runtime.source_loss import SourceLossWeights
    before=f.blocks[1]
    weights=SourceLossWeights(np.ones(len(before.r)))
    ledger=dict(before.row_ledger,inactive_scalar_slots_excluded_from_source_accuracy_statistics=[10])
    f.blocks[1]=SourceRowBlock(before.r,before.J,before.quadratic,before.scales,ledger,weights)
    source=wrap_original_source_point_qualification(f.base,f.state,f.packet)
    after=list(source.blocks(np.zeros(3)))[1]
    assert after.loss_weights is weights
    assert after.row_ledger['inactive_scalar_slots_excluded_from_source_accuracy_statistics']==[10,11,12]


def test_full_source_robust_normal_excludes_only_qualified_rows_and_old_replay_rejects(tmp_path):
    f=fixture(tmp_path)
    source=wrap_original_source_point_qualification(f.base,f.state,f.packet)
    from z190_streaming_source_normal import accumulate_source_normal, replay_source_cost
    from z190_joint_source_solver import source_cost
    normal=accumulate_source_normal(lambda:source.blocks(np.zeros(3)),3,
        expected_row_count=16,frozen_profile_token='actual-frozen-profile')
    residual=np.concatenate([b.r for b in f.blocks])
    jacobian=np.concatenate([b.J.toarray() for b in f.blocks])
    keep=np.ones(16,bool);keep[[11,12]]=False
    r=residual[keep];j=jacobian[keep]
    weights=np.minimum(1.,2./np.maximum(abs(r),1e-12))
    assert normal.cost==source_cost(r,np.zeros(len(r),bool))
    assert np.allclose(normal.gradient,j.T@(weights*r))
    assert np.allclose(normal.normal.toarray(),j.T@(weights[:,None]*j))
    assert normal.contract.row_count==16
    with pytest.raises(ValueError,match='row order/ledger/mask/native scales changed'):
        replay_source_cost(lambda:iter(f.blocks),normal.contract,frozen_profile_token='actual-frozen-profile')


def test_mask_revision_changes_production_target_fingerprint_and_rejects_same_q_profile_replay(tmp_path):
    f=fixture(tmp_path)
    previous=wrap_original_source_point_qualification(f.base,f.state,f.packet)
    from structured_programme_source_candidate.truth_source_role_migration import target
    from z190_streaming_source_normal import accumulate_source_normal,replay_source_cost
    def actual_target(source):
        return target(dict(full_parameters=source.count,source_rows=len(source.scales),source_contract=source.contract))
    old_target=actual_target(previous)
    normal=accumulate_source_normal(lambda:previous.blocks(np.zeros(3)),3,
        expected_row_count=16,frozen_profile_token='unchanged-q-and-original-profile')
    # The immutable source UV/sigma/world/row layout is the same; only an
    # additional RGB-certified inactive point changes the actual objective.
    f.record['excluded_point_indices']=[0,1]
    write_json(f.packet,f.manifest)
    current=wrap_original_source_point_qualification(f.base,f.state,f.packet)
    new_target=actual_target(current)
    assert old_target['parameter_count']==new_target['parameter_count']==3
    assert old_target['source_rows']==new_target['source_rows']==16
    assert old_target['source_contract_sha256']!=new_target['source_contract_sha256']
    with pytest.raises(ValueError,match='row order/ledger/mask/native scales changed'):
        replay_source_cost(lambda:current.blocks(np.zeros(3),False),normal.contract,
            frozen_profile_token='unchanged-q-and-original-profile')
