"""Source decoding and camera conventions used by the real full-trial run."""
import json
import numpy as np
import pytest
from calibration.program_tracking.body_tracking.fs_jump3d_validation import qtm_reference,camera,project


def test_qtm_parts_keep_original_frame_slots_and_fourth_channel(tmp_path):
    source=dict(Timebase={'Frequency':60.,'Range':{'Start':1,'End':4}},Markers=[
        dict(Name='L_HEEL',TrajectoryId=17,Parts=[{'Range':{'Start':2,'End':3},'Values':[[1,2,3,0],[4,5,6,-1]]}]),
        dict(Name='L_Toe',TrajectoryId=18,Parts=[{'Range':{'Start':1,'End':1},'Values':[[7,8,9,0]]},{'Range':{'Start':4,'End':4},'Values':[[10,11,12,0]]}])])
    path=tmp_path/'source.json';path.write_text(json.dumps(source));result=qtm_reference(path)
    np.testing.assert_array_equal(result['source_frame'],[1,2,3,4])
    np.testing.assert_array_equal(result['reference_time_base'],[1,60])
    np.testing.assert_array_equal(result['valid_mask'],[[False,True],[True,False],[False,False],[False,True]])
    np.testing.assert_array_equal(result['world_points_mm'][2,0],[4,5,6])
    assert result['original_fourth_channel'][2,0]==-1
    assert np.isnan(result['world_points_mm'][0,0]).all()
    assert result['names'].tolist()==['L_HEEL','L_Toe']


def test_qtm_range_does_not_silently_interpolate_a_short_part(tmp_path):
    source=dict(Timebase={'Frequency':60.,'Range':{'Start':1,'End':3}},Markers=[dict(Name='original',Parts=[{'Range':{'Start':1,'End':3},'Values':[[1,2,3,0]]}])])
    path=tmp_path/'source.json';path.write_text(json.dumps(source))
    with pytest.raises(ValueError,match='part/range'):qtm_reference(path)


def test_released_camera_centre_mm_and_brown_coefficient_order():
    raw=dict(extrinsic_matrix=np.diag([1,-1,-1]).tolist(),xyz=[1000,2000,3000],affine_intrinsics_matrix=[[1000,0,500],[0,800,400],[0,0,1]],distortion=[.1,.2,.3,.01,.02])
    cam=camera(raw);np.testing.assert_array_equal(cam['R_w2c'],np.eye(3));np.testing.assert_array_equal(cam['C_world_m'],[1,2,3])
    np.testing.assert_array_equal(cam['distortion_opencv'],[.1,.2,.01,.02,.3])
    uv,valid=project(np.array([[1.2,2.4,5.],[1,2,2.]]),cam)
    # Normalized (x,y)=(.1,.2): radial=1+.1*.05+.2*.0025+.3*.000125.
    # Tangential offsets are dx=.0018 and dy=.0021 in normalized pixels.
    np.testing.assert_allclose(uv[0],[602.35375,562.566],atol=1e-9)
    np.testing.assert_array_equal(valid,[True,False])
