from copy import deepcopy
import numpy as np
import pytest
from rinkspace.domain.motion_chart import cubic_samples
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline, continue_exposure_halos


def timeline(local=False):
    shots=[]
    for i in range(2):
        first=100000*i
        shot=dict(id=f's{i}',camera_index=i,control_begin=100*i,control_stop=100*(i+1),
            row_begin=2*i,row_stop=2*i+2,reference_pts=first,first_pts=first,last_pts=first+75*256,
            stop_pts_exclusive=first+76*256,time_base_num=1,time_base_den=12800,
            first_control_time=-.2,knot_stride=5,motion_reference_time=0.)
        if local and i==0:shot['local_motion_refinement']=[dict(first_pts=first,last_pts=first+20*256,target_stride=1)]
        shots.append(shot)
    return ShotMotionTimeline(shots,.02)


def values(d,t):
    ids,w=cubic_samples(100,-.2,.02,t)
    return np.einsum('ij,ijck->ick',w,d[ids]).reshape(-1,d.shape[2])


def test_local_spline_can_represent_short_motion_without_changing_old_space_or_other_shot():
    base,old_labels=timeline().directions(continue_endpoint_cubics=True)
    fine,new_labels=timeline(True).directions(continue_endpoint_cubics=True)
    by_label={label:i for i,label in enumerate(new_labels)}
    np.testing.assert_array_equal(fine[:,:,[by_label[x] for x in old_labels]],base)
    added=[i for i,x in enumerate(new_labels) if x not in set(old_labels)]
    assert added and all(new_labels[i][0]=='s0' for i in added)
    np.testing.assert_array_equal(fine[100:,:,added],0)
    # A real 80-ms compact cubic pan pulse exceeds the old 100-ms knot space.
    pulse=np.zeros((100,3));pulse[16,0]=.001
    pulse=continue_exposure_halos(pulse,-.2,.02,[0.,1.5])
    t=np.linspace(0,1.5,601);ids,w=cubic_samples(100,-.2,.02,t)
    target=np.einsum('ij,ijk->ik',w,pulse[ids]).ravel()
    old=[i for i,x in enumerate(old_labels) if x[0]=='s0']
    new=[i for i,x in enumerate(new_labels) if x[0]=='s0']
    A=values(base[:100,:,old],t);B=values(fine[:100,:,new],t)
    assert np.linalg.norm(A@np.linalg.lstsq(A,target,rcond=None)[0]-target)>1e-5
    np.testing.assert_allclose(B@np.linalg.lstsq(B,target,rcond=None)[0],target,atol=2e-15)
    np.testing.assert_allclose(values(fine[:100,:,added],np.linspace(.4,1.5,201)),0.,atol=2e-15)
    # Added coordinates respect the existing physical-camera pan/tilt gauge.
    at_ref=values(fine[:100,:,added],np.array([0.])).reshape(3,-1)
    np.testing.assert_allclose(at_ref[:2],0.,atol=2e-15)
    assert np.linalg.matrix_rank(B)==B.shape[1]


def test_local_sparse_and_dense_directions_match():
    dense,labels=timeline(True).directions(continue_endpoint_cubics=True)
    sparse,other=timeline(True).directions(continue_endpoint_cubics=True,sparse=True)
    assert labels==other
    np.testing.assert_array_equal(sparse.toarray(),dense.reshape(600,-1))


@pytest.mark.parametrize('entry',[
    dict(first_pts=False,last_pts=5120,target_stride=1),
    dict(first_pts=0,last_pts=999999,target_stride=1),
    dict(first_pts=0,last_pts=5120,target_stride=3),
    dict(first_pts=5120,last_pts=5120,target_stride=1),
])
def test_invalid_local_refinement_is_rejected(entry):
    model=timeline(True);model.shots[0]['local_motion_refinement']=[entry]
    with pytest.raises(ValueError):model.directions(continue_endpoint_cubics=True)


def test_pan_only_refinement_does_not_add_zoom_or_tilt_wiggles():
    old,labels=timeline().directions(continue_endpoint_cubics=True)
    model=timeline(True);model.shots[0]['local_motion_refinement'][0]['channels']=['pan']
    new,more=model.directions(continue_endpoint_cubics=True)
    extra=[i for i,x in enumerate(more) if x not in set(labels)]
    assert extra and all(more[i][2]==0 for i in extra)
    np.testing.assert_array_equal(new[:,1:,extra],0.)
    for channels in ([],['roll'],['pan','pan']):
        model.shots[0]['local_motion_refinement'][0]['channels']=channels
        with pytest.raises(ValueError):model.directions(continue_endpoint_cubics=True)
