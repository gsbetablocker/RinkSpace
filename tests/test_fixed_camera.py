import cv2
import numpy as np
import pytest

from rinkspace.fixed_camera import fit_rotation_zoom, rotation_zoom_h, camera_from_record
from rinkspace.multisurface_camera import Camera, world_rays, project_camera


def camera():
    return Camera(np.eye(3),np.array([0.,-60.,8.]),900.,np.array([480.,270.]))


def test_cross_programme_large_zoom_and_orientation_preserves_installation():
    ref=camera()
    target=Camera(cv2.Rodrigues(np.array([.12,-.18,.025]))[0],ref.centre.copy(),1550.,ref.principal.copy())
    xy=np.array([[100,100],[300,100],[600,100],[800,100],[100,300],[400,300],[700,300],[800,480]],float)
    rays,_=world_rays(ref,xy);observed,_=project_camera(target,rays,rays=True)
    fit,diag=fit_rotation_zoom(ref,xy,observed,max_nfev=120)
    assert diag['training']['p95_px']<1e-5
    assert np.array_equal(fit.centre,ref.centre)
    assert abs(fit.focal-target.focal)<1e-3
    assert diag['data_jacobian_rank']==4


def test_ice_holdout_cannot_modify_background_pose():
    ref=camera();xy=np.array([[100,100],[300,100],[600,100],[800,100],[100,200],[400,200],[700,200],[800,200],[200,400],[700,430]],float)
    observed=xy.copy();observed[-2:]+=[0,45]
    fit,diag=fit_rotation_zoom(ref,xy,observed,['board']*8+['ice']*2,fit_mask=[True]*8+[False]*2)
    assert diag['training']['p95_px']<1e-5
    assert diag['withheld']['median_px']==pytest.approx(45.,abs=1e-4)
    assert diag['surfaces']['ice']['fit_count']==0
    assert not diag['independent_accuracy_validated']


def test_board_transfer_rejects_different_installations_and_centres():
    ref=camera();xy=np.array([[1,1],[2,1],[2,2],[1,2]],float)
    with pytest.raises(ValueError,match='installations'):
        fit_rotation_zoom(ref,xy,xy,reference_installation_id='main',current_installation_id='end')
    changed=camera();changed.centre=changed.centre+1
    with pytest.raises(ValueError,match='optical centre'):
        rotation_zoom_h(ref,changed)


def test_camera_record_roundtrip_retains_model_to_sample_projection():
    original=camera();restored=camera_from_record(original.to_dict())
    assert np.allclose(original.H,restored.H)
    assert restored.size==(960,540)
