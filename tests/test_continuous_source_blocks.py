"""Parallel source updates must preserve the one continuous motion curve."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
    'events/ankara_2026/programmes/okada_mei_sp/tools/identity_refit'))
from source_continuous_refine import independent_stage_groups, merge_independent_curves
from calibration.program_tracking.body_tracking.absolute_motion_curve import (
    AbsoluteMotionCurve, clamped_cubic_knots, _exp,
)
from calibration.program_tracking.body_tracking.absolute_motion_state import evaluate_curve
from shoe_direction_source import qualify_coupled_flow, transport_existing_flow_regions


def base_curve():
    knots=clamped_cubic_knots(np.linspace(0.,10.,101));m=len(knots)-4
    return AbsoluteMotionCurve(knots,np.zeros((m,3)),np.broadcast_to(np.eye(3),(m,22,3,3)).copy())


def changed(base,indices):
    p=base.position_controls.copy();r=base.rotation_controls.copy()
    p[indices,0]+=.02
    r[indices,7]=r[indices,7]@_exp(np.array([.01,.04,-.07]))
    return AbsoluteMotionCurve(base.knots,p,r)


def test_independent_merge_equals_each_actual_curve_on_all_affected_times():
    base=base_curve();indices=[np.arange(10,18),np.arange(60,68)]
    children=[changed(base,ix) for ix in indices]
    merged=merge_independent_curves(base,children)
    times=np.linspace(0.,10.,2001);value=evaluate_curve(merged,times)
    supports=[np.asarray(base.basis_sparse(times)[:,ix].getnnz(axis=1)>0) for ix in indices]
    assert not np.any(supports[0]&supports[1])
    for child,support in zip(children,supports):
        expected=evaluate_curve(child,times)
        for key in ('P','rotations'):
            np.testing.assert_allclose(value[key][support],expected[key][support],atol=2.e-12,rtol=0.)
    unchanged=~(supports[0]|supports[1]);expected=evaluate_curve(base,times)
    for key in ('P','rotations'):
        np.testing.assert_allclose(value[key][unchanged],expected[key][unchanged],atol=2.e-12,rtol=0.)


def test_merge_rejects_shared_control_or_adjacent_rotation_edge():
    base=base_curve()
    with pytest.raises(ValueError,match='overlap'):
        merge_independent_curves(base,[changed(base,[10]),changed(base,[11])])


def test_colouring_keeps_overlapping_source_halos_in_different_groups():
    base=base_curve()
    specs=[dict(name=str(i),seconds=[1.+2*i,3.4+2*i]) for i in range(4)]
    groups=independent_stage_groups(base,specs,1.)
    assert len(groups)==2
    assert [[s['name'] for s in group] for group in groups]==[['0','2'],['1','3']]


def test_lone_heel_track_on_background_cannot_survive_lost_toe():
    anchors=[dict(id=f'a:{feature}',pts=256,foot=0,feature=feature,role='fit',
                  association_group='visible_boot',source_kind='assistant_native_source_review')
             for feature in ('heel','toe','cuff_center')]
    rows=[]
    for anchor,stamps in zip(anchors,([512,768],[512],[512,768])):
        for pts in stamps:
            rows.append(dict(anchor,id=f"{anchor['id']}:flow:{pts}",pts=pts,
                source_kind='adjacent_source_flow',source_anchor_id=anchor['id'],uv=[10.,20.],weight=.5))
    doc=dict(samples=anchors+rows);result=qualify_coupled_flow(doc)
    assert result['rejected_ids']==['a:heel:flow:768']
    assert all(r['role']=='fit' for r in rows if r['id']!='a:heel:flow:768')
    assert all(r['uv']==[10.,20.] and r['weight']==.5 for r in rows)
    assert all(r['role']=='fit' for r in anchors)


def test_flow_region_follows_fast_turn_without_moving_original_reading():
    anchor=dict(id='toe',pts=256,foot=0,feature='toe',role='fit',
        source_kind='assistant_native_source_review',uv=[1067.,857.],
        pixel_box=[[1058.,848.],[1076.,866.]],weight=9.)
    flow=dict(anchor,id='toe:flow:512',pts=512,source_kind='adjacent_source_flow',
        source_anchor_id='toe',uv=[1045.,863.],weight=1.5)
    doc=dict(samples=[anchor,flow]);result=transport_existing_flow_regions(doc)
    assert result['active_changed_ids']==['toe:flow:512']
    np.testing.assert_allclose(flow['pixel_box'],[[1036.,854.],[1054.,872.]])
    assert anchor['pixel_box']==[[1058.,848.],[1076.,866.]]
    assert anchor['uv']==[1067.,857.] and flow['uv']==[1045.,863.]
    assert flow['weight']==1.5 and flow['role']=='fit'
    assert transport_existing_flow_regions(doc)['changed_ids']==[]
    np.testing.assert_allclose(flow['pixel_box'],[[1036.,854.],[1054.,872.]])


def test_checkpoint_replace_recovers_after_windows_reader_releases_file(tmp_path,monkeypatch):
    from calibration.program_tracking.first_shot_atomic_output import replace_file
    import calibration.program_tracking.first_shot_atomic_output as output
    pending=tmp_path/'candidate.pending.npz';target=tmp_path/'candidate.npz'
    pending.write_bytes(b'complete new state');target.write_bytes(b'previous state')
    replace=type(pending).replace;attempts=[]
    def held_then_released(source,destination):
        attempts.append(1)
        if len(attempts)==1:
            assert target.read_bytes()==b'previous state'
            error=PermissionError('reader holds target');error.winerror=32
            raise error
        return replace(source,destination)
    monkeypatch.setattr(type(pending),'replace',held_then_released)
    monkeypatch.setattr(output.time,'sleep',lambda _:None)
    replace_file(pending,target)
    assert target.read_bytes()==b'complete new state' and not pending.exists()
    assert len(attempts)==2


def test_checkpoint_replace_preserves_pending_bytes_on_unrelated_error(tmp_path,monkeypatch):
    from calibration.program_tracking.first_shot_atomic_output import replace_file
    pending=tmp_path/'candidate.pending.npz';target=tmp_path/'candidate.npz'
    pending.write_bytes(b'complete new state');target.write_bytes(b'previous state')
    def denied(source,destination):
        error=PermissionError('not a sharing violation');error.winerror=13
        raise error
    monkeypatch.setattr(type(pending),'replace',denied)
    with pytest.raises(PermissionError,match='not a sharing violation'):
        replace_file(pending,target)
    assert pending.read_bytes()==b'complete new state'
    assert target.read_bytes()==b'previous state'
