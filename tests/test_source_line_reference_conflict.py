import numpy as np

from rinkspace.session.material_tracks import source_line_reference_conflict, split_visible_source_arcs


def band(points, half_width):
    sides=np.asarray(points)[:,None,None,:]+np.array([[[[0.,-half_width]],[[0.,half_width]]]])
    return np.repeat(sides,3,axis=2)


def test_same_paint_remeasurement_is_allowed_but_other_parallel_band_is_not():
    old=np.c_[np.arange(0.,201.,20.),np.zeros(11)]
    shifted=np.c_[np.arange(0.,201.,25.),np.full(9,10.)]
    assert not source_line_reference_conflict(old,shifted,band(shifted,15.))['contradicted']
    other=shifted+[0.,90.]
    assert source_line_reference_conflict(old,other,band(other,8.))['contradicted']


def test_no_veto_from_extrapolation_or_inconsistent_source_reference():
    old=np.c_[np.arange(0.,201.,20.),np.zeros(11)]
    remote=np.c_[np.arange(300.,501.,25.),np.full(9,100.)]
    assert not source_line_reference_conflict(old,remote,band(remote,8.))['contradicted']
    assert source_line_reference_conflict(old,remote,band(remote,8.),known_straight_extension=True)['contradicted']
    continuation=remote-[0.,100.]
    assert not source_line_reference_conflict(old,continuation,band(continuation,8.),known_straight_extension=True)['contradicted']
    old[:,1]=np.sin(old[:,0]/30.)*30
    assert not source_line_reference_conflict(old,remote-[300.,0.],band(remote-[300.,0.],8.))['reference_supported']


def test_recognized_arc_does_not_lend_identity_to_opposite_disconnected_band():
    query=np.array([0,1,3,4,20,21,22]);uv=np.c_[query*24.,np.zeros(7)]
    m=dict(candidate_query=query,candidate_uv=uv,query_uv=np.c_[np.arange(23)*24.,np.zeros(23)])
    f=dict(curve=2,candidate_ids=list(range(7)))
    pieces=split_visible_source_arcs(m,f)
    assert [p['candidate_ids']for p in pieces]==[[0,1,2,3],[4,5,6]]
    assert pieces[0]['source_visible_candidate_pieces']==[[0,1],[2,3]]
    assert np.array_equal(m['candidate_uv'],uv)


def test_short_wrong_band_cannot_borrow_identity_from_a_later_crossing():
    old=np.c_[np.arange(0.,201.,20.),np.zeros(11)]
    short=np.array([[50.,60.],[74.,60.],[98.,60.]])
    assert source_line_reference_conflict(old,short,band(short,4.))['contradicted']
    assert source_line_reference_conflict(old,short[:1],band(short[:1],4.))['contradicted']
    valid=short-[0.,60.]
    mixed=np.vstack([valid,short[:1]])
    result=source_line_reference_conflict(old,mixed,band(mixed,4.))
    assert not result['contradicted']
    assert result['conflicting_candidates']==[3]
