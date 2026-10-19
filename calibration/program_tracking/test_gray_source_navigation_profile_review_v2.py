import hashlib
import cv2
import numpy as np
from rinkspace.vision_source_shapes import _MaterialProfileFrame
from rinkspace.vision_programme_ice_review import _batched_actual_profiles,_actual_profile_candidates
from calibration.program_tracking.gray_source_navigation_profile_review_v2 import same_producer_actual_gray_profiles


def test_cached_native_operator_is_bitexact_to_original_producer_and_frame_scoped():
    rgb=np.full((100,160,3),235,np.uint8)
    cv2.line(rgb,(5,56),(150,56),(140,141,139),7,cv2.LINE_AA)
    points=np.asarray([[30.,55.],[70.,55.],[110.,55.]])
    normals=np.tile([0.,1.],(3,1));intervals=np.asarray([[-24.,24.],[-18.,21.],[-24.,24.]])
    for rgb in (rgb,rgb//2):
        binding=dict(source_sha256='a'*64,source_pts=100,time_base='1/100',shot_id='fixture',
            actual_pts=100,native_pts=100,decoded_native_BGR_sha256=hashlib.sha256(rgb[:,:,::-1].tobytes()).hexdigest())
        frame=_MaterialProfileFrame(rgb,binding);native=frame.rgb.astype(np.float32)
        result=same_producer_actual_gray_profiles(frame,points,normals,intervals,native)
        for i in range(3):
            offsets,profiles,smooth,valid=_batched_actual_profiles(frame,points[i:i+1],normals[i:i+1],intervals[i:i+1],np.asarray(['dark']))
            original=_actual_profile_candidates(points[i],normals[i],offsets[0,valid[0]],profiles[0,valid[0]],smooth[0,valid[0]],operator='paint_center',signal='dark',transition=None,with_quantiles=True)
            assert len(result[i])==len(original)>0
            for left,right in zip(result[i],original):
                assert left[2]==right[2]
                for k in (0,1,3):assert np.array_equal(left[k],right[k])
        assert np.array_equal(frame.material_signals['gray_review_same_producer_native_luminance_float32'],frame.luminance.astype(np.float32))
