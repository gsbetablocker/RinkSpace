"""Row-time integration retains the full correlated source and neighbour J."""
from copy import copy, deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from calibration.program_tracking.first_shot_physical_state import FirstShotPhysicalState
from calibration.program_tracking.first_shot_fit import _CameraState, load_parameter_initializer
from calibration.program_tracking.first_shot_source_kernel import evaluate_projected_targets, _interfaces
from calibration.program_tracking.first_shot_row_source import evaluate_row_timed_frame, row_timed_raw
from calibration.program_tracking.official_static_surface_dataset import reference


@pytest.fixture(scope="module")
def row_state(tmp_path_factory):
    old = FirstShotPhysicalState.from_binding()
    folder = tmp_path_factory.mktemp("native_row_state")
    timing = dict(schema="rinkspace.first-shot-shared-row-timing.v1",source_clock=old.source_clock,
        source_video_sha256=old.source["sha256"],readout_parameter_units="signed_milliseconds",
        positive_is_top_to_bottom=True,manufacturer_measured_readout=False,initial_readout_milliseconds=10.46)
    config = folder/"timing.json";config.write_text(json.dumps(timing),encoding="utf8")
    binding = deepcopy(old.binding);binding["row_timing_model"] = reference(config)
    path=folder/"binding.json";path.write_text(json.dumps(binding),encoding="utf8")
    return old,FirstShotPhysicalState.from_binding(path)


def test_row_parameter_is_shared_and_does_not_enter_material_chart(row_state):
    old,state=row_state
    assert state.row_readout_index==old.count
    assert state.count==old.count+1
    np.testing.assert_array_equal(state.world_parameter_indices,old.world_parameter_indices)
    np.testing.assert_array_equal(state.global_shared_indices,np.r_[old.global_shared_indices,old.count])
    np.testing.assert_array_equal(state.initial_q[:-1],old.initial_q)
    np.testing.assert_array_equal(state.camera25(state.initial_q,500),old.camera25(old.initial_q,500))
    light=_CameraState(state)
    assert light.row_readout_index==state.row_readout_index
    np.testing.assert_array_equal(light.pts,state.pts)
    bad=state.initial_q.copy();bad[-1]=20.01
    with pytest.raises(ValueError,match="readout"):state.guard(bad)


def test_old_full_parameters_extend_only_the_declared_row_slot(row_state,tmp_path):
    old,state=row_state
    saved=tmp_path/"old.npz"
    np.savez(saved,q=old.initial_q,aux__native_pts=old.pts,aux__parent_q=old.parent_q,
             aux__parent_shared_slots=old.parent_shared_slots)
    q,proof=load_parameter_initializer(SimpleNamespace(state=state),saved,reference(saved)["sha256"])
    np.testing.assert_array_equal(q[:-1],old.initial_q)
    assert q[-1]==10.46
    assert proof["appended_shared_row_timing"]["column"]==old.count


def test_dense_whitening_keeps_neighbour_and_world_derivatives(row_state):
    _,full=row_state
    state=copy(full);state.world_parameter_indices=np.array([0,1]);state.world_parameter_count=2
    q=state.initial_q.copy();index=500;camera=state.camera25(q,index)
    rotation=cv2.Rodrigues(camera[3:6])[0]
    intrinsic=np.array([[np.exp(camera[6]),camera[10],camera[8]],
                        [0.,np.exp(camera[7]),camera[9]],[0.,0.,1.]])
    def in_view(uv,depth):
        rays=cv2.undistortPoints(np.asarray(uv,float).reshape(-1,1,2),intrinsic,camera[11:25]).reshape(-1,2)
        return camera[:3]+(np.c_[rays,np.ones(len(rays))]*depth)@rotation
    base={"line":in_view([[500.,250.],[1300.,850.]],20.),
          "points":in_view([[700.,350.],[1200.,750.]],18.)}
    derivative={k:np.zeros(v.shape+(2,)) for k,v in base.items()}
    for k in base:
        derivative[k][:,0,0]=1.;derivative[k][:,2,1]=1.
    def worlds(qq):
        return {k:v+np.einsum('ijk,k->ij',derivative[k],qq[:2]-q[:2]) for k,v in base.items()}
    project=_interfaces()[0]
    points=project(camera,base['points'],with_jacobian=False)[0]
    line=project(camera,base['line'],with_jacobian=False)[0]
    observations=.25*line[[0,1]]+.75*line[[1,0]]+np.array([[1.,-2.],[-1.5,2.]])
    frame=dict(index=index,source_point_count=4,rows=[
        dict(_target_key='line',target=dict(kind='ice_curve'),point_correspondence=False,uv=observations),
        dict(_target_key='points',target=dict(kind='ad_corner'),point_correspondence=True,uv=points+[[1.,1.],[-1.,-1.]])])
    matrix=np.array([[1.,.2,.3,0,0,0,.1,0],[0,1,0,.4,0,.1,0,0],
                     [.1,0,1,0,.2,0,0,.2],[0,.2,0,1,0,.1,.3,0],[0,0,.2,.1,1,.3,.1,.4]])
    whitening=SimpleNamespace(apply=lambda a:matrix@a)
    original=[r['uv'].copy() for r in frame['rows']]
    raw=row_timed_raw(state,q,frame,worlds(q))
    r,j=evaluate_row_timed_frame(state,q,frame,whitening,worlds(q),derivative)
    np.testing.assert_allclose(r,matrix@raw,atol=1e-12)
    columns=np.r_[0,1,np.arange(118,127),state.view_columns(index-1),state.view_columns(index),state.view_columns(index+1),state.row_readout_index]
    fd=np.empty((len(r),len(columns)))
    for k,col in enumerate(columns):
        h=1e-6 if col>=127 and col!=state.row_readout_index else 1e-5
        plus=q.copy();minus=q.copy();plus[col]+=h;minus[col]-=h
        fd[:,k]=(matrix@row_timed_raw(state,plus,frame,worlds(plus))-matrix@row_timed_raw(state,minus,frame,worlds(minus)))/(2*h)
    np.testing.assert_allclose(j[:,columns].toarray(),fd,rtol=2e-5,atol=3e-5)
    assert np.linalg.norm(j[:,state.view_columns(index-1)].toarray())>0
    assert np.linalg.norm(j[:,state.view_columns(index+1)].toarray())>0
    zero=q.copy();zero[-1]=0.
    baseline=evaluate_projected_targets(state.camera25(zero,index),frame['rows'],worlds(zero),jacobian=False)[0]
    np.testing.assert_allclose(row_timed_raw(state,zero,frame,worlds(zero)),baseline,atol=2e-10)
    for before,row in zip(original,frame['rows'],strict=True):np.testing.assert_array_equal(before,row['uv'])
